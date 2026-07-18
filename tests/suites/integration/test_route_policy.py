"""Route-policy manifest vs. the live FastAPI application (plan O0.1 item 2).

Every API operation must be classified in the manifest before it can ship;
an unclassified route (or a stale manifest row) fails this suite.
"""

from fastapi.routing import APIRoute

from therapy.observability.model import HTTP_ROUTE_MANIFEST


def _app_operations() -> set[tuple[str, str, str]]:
    from therapy.server.app import app

    operations: set[tuple[str, str, str]] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue  # static mounts and non-API machinery
        methods = route.methods
        assert methods is not None
        for method in methods - {"HEAD", "OPTIONS"}:
            operations.add((method, route.path, route.name))
    return operations


def test_manifest_matches_every_fastapi_operation() -> None:
    manifest = {(r.method, r.path, r.name) for r in HTTP_ROUTE_MANIFEST}
    app_ops = _app_operations()

    unclassified = app_ops - manifest
    stale = manifest - app_ops
    assert not unclassified, f"routes missing from manifest: {sorted(unclassified)}"
    assert not stale, f"manifest rows without live routes: {sorted(stale)}"
