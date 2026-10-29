import pytest

torch = pytest.importorskip("torch")

from price_estimate.models.dnn import DNNConfig, DNNPricer  # noqa: E402
from tests.conftest import make_items  # noqa: E402

SMALL = DNNConfig(n_features=512, num_layers=3, hidden_size=64, batch_size=32)


def test_train_save_load_roundtrip(tmp_path):
    items = make_items(400, seed=2)
    train, val = items[:350], items[350:]
    pricer = DNNPricer(SMALL, device="cpu")
    history = pricer.fit(train, val, epochs=8, verbose=False)
    assert history[-1]["val_mae"] < history[0]["val_mae"]
    assert history[-1]["val_mae"] < 120  # constant guess is ~150 on this data

    path = tmp_path / "dnn.pth"
    pricer.save(path)
    loaded = DNNPricer.load(path, device="cpu")
    text = val[0].text
    assert loaded.price(text) == pytest.approx(pricer.price(text), rel=1e-5)
    assert loaded.y_mean == pytest.approx(pricer.y_mean)


def test_legacy_state_dict(tmp_path):
    pricer = DNNPricer(DNNConfig(n_features=5000, num_layers=2, hidden_size=16), device="cpu")
    path = tmp_path / "legacy.pth"
    torch.save(pricer.model.state_dict(), path)
    with pytest.raises(RuntimeError):  # wrong architecture for a legacy file -> clear error
        DNNPricer.load(path, device="cpu")
