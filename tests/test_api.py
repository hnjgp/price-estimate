import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from price_estimate.app.api import app, get_pricer  # noqa: E402
from price_estimate.models.ensemble import EnsemblePricer  # noqa: E402


class Fixed:
    def __init__(self, value):
        self.value = value

    def price(self, text):
        return self.value


@pytest.fixture
def client():
    pricer = EnsemblePricer({"a": Fixed(100), "b": Fixed(300)}, {"a": 0.5, "b": 0.5})
    app.dependency_overrides[get_pricer] = lambda: pricer
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    body = client.post("/predict", json={"description": "A nice blender"}).json()
    assert body["estimate"] == 200
    assert body["breakdown"] == {"a": 100, "b": 300}


def test_validation(client):
    assert client.post("/predict", json={"description": ""}).status_code == 422
