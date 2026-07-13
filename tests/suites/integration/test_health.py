from therapy import __version__


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_index_serves_html(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "TheraPy" in response.text
