"""Serve the fine-tuned Llama-3.2-3B price model on a Modal GPU.

    modal deploy modal_services/pricer_service.py

Which adapter is served is controlled by environment variables at deploy time:

    ADAPTER_REPO      e.g. "<your-hf-user>/price-2026-11-01_12.00.00"
    ADAPTER_REVISION  a commit hash on that repo (optional, pins the exact checkpoint)

Until you have trained your own adapter (training/finetune_qlora.py), the default
is the public adapter published with the LLM Engineering course, so the rest of
the system can run end to end.

Design notes:
* The model loads once per container (@modal.enter), not once per request.
* Weights are cached on a Modal Volume, so cold starts skip the 6 GB download.
* 4-bit NF4 quantisation lets a 3B model fit easily on a T4.
"""

from __future__ import annotations

import os

import modal

APP_NAME = "pricer-service"
GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER_REPO = os.getenv("ADAPTER_REPO", "ed-donner/price-2025-11-28_18.47.07")
ADAPTER_REVISION = os.getenv("ADAPTER_REVISION", "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd") or None
CACHE_DIR = "/cache"
MIN_CONTAINERS = int(os.getenv("MIN_CONTAINERS", "0"))  # 1 = always warm (costs money)

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft", "huggingface_hub")
    .env(
        {
            "HF_HUB_CACHE": CACHE_DIR,
            "ADAPTER_REPO": ADAPTER_REPO,
            "ADAPTER_REVISION": ADAPTER_REVISION or "",
        }
    )
)
cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface-secret")]


def extract_price(generated: str) -> float:
    import re

    tail = generated.split(PREFIX, 1)[-1].replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", tail)
    return float(match.group()) if match else 0.0


@app.cls(
    image=image,
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: cache},
)
class Pricer:
    @modal.enter()
    def setup(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(
            base, os.environ["ADAPTER_REPO"], revision=os.environ.get("ADAPTER_REVISION") or None
        )
        self.model.eval()

    @modal.method()
    def price(self, description: str) -> float:
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=5, pad_token_id=self.tokenizer.eos_token_id
            )
        return extract_price(self.tokenizer.decode(output[0], skip_special_tokens=True))
