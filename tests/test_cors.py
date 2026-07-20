import os
from fastapi.testclient import TestClient
from app.config import settings

def test_cors_multiple_origins(monkeypatch):
    # Set LINK_COR with multiple comma-separated URLs
    test_origins = "http://localhost:3000, https://erp.igentechsolutions.com, https://staging-erp.igentechsolutions.com"
    monkeypatch.setenv("LINK_COR", test_origins)
    settings.LINK_COR = test_origins

    # Re-import or re-create app to apply new CORS settings
    import importlib
    import app.main
    importlib.reload(app.main)

    client = TestClient(app.main.app)

    # Test request from origin 1
    resp1 = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert resp1.headers.get("access-control-allow-origin") == "http://localhost:3000"

    # Test request from origin 2
    resp2 = client.options(
        "/api/v1/health",
        headers={
            "Origin": "https://erp.igentechsolutions.com",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert resp2.headers.get("access-control-allow-origin") == "https://erp.igentechsolutions.com"

    # Test request from unauthorized origin
    resp_unauthorized = client.options(
        "/api/v1/health",
        headers={
            "Origin": "https://unauthorized-domain.com",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert resp_unauthorized.headers.get("access-control-allow-origin") != "https://unauthorized-domain.com"
