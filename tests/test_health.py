from fastapi.testclient import TestClient

from therapy import __version__
from therapy.server.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_index_serves_html() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "TheraPy" in response.text
