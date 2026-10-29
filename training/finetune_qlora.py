"""QLoRA fine-tuning of Llama-3.2-3B to predict prices (run on a GPU, e.g. Colab A100/T4).

    pip install -e ".[finetune,data]"
    python training/finetune_qlora.py --epochs 1 --push

What happens:
1. The base model is loaded in 4-bit NF4 (QLoRA) so it fits on one consumer GPU.
2. Small LoRA adapters (rank r) are trained on the attention + MLP projections;
   the 3B base weights stay frozen.
3. Loss is computed only on the completion (the price), not the prompt, so every
   gradient step is spent learning to predict the number.
4. The adapter (~100 MB) is pushed to the Hub and served by modal_services/pricer_service.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from price_estimate.config import settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-train", type=int, default=None, help="subset for a quick run")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    owner = settings.my_hf_user or settings.data_hf_user
    data = load_dataset(settings.dataset_name("items_prompts", owner=owner))
    train = data["train"]
    if args.max_train:
        train = train.select(range(min(args.max_train, len(train))))
    val = data["val"].select(range(min(1000, len(data["val"]))))

    run_name = f"price-{datetime.now():%Y-%m-%d_%H.%M.%S}"
    hub_id = f"{settings.my_hf_user}/{run_name}" if settings.my_hf_user else None
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        ),
        device_map="auto",
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    config = SFTConfig(
        output_dir=run_name,
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        weight_decay=0.001,
        max_grad_norm=0.3,
        optim="paged_adamw_32bit",
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        max_length=182,
        completion_only_loss=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=2000,
        save_total_limit=2,
        report_to="none",
        push_to_hub=bool(args.push and hub_id),
        hub_model_id=hub_id,
        hub_private_repo=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train,
        eval_dataset=val,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(run_name)
    if args.push and hub_id:
        trainer.model.push_to_hub(hub_id, private=True)
        print(f"Adapter pushed to {hub_id}; deploy with ADAPTER_REPO={hub_id}")


if __name__ == "__main__":
    main()
