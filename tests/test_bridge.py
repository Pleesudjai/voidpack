"""Gate for T15. Written before the implementation. Do not edit.

The bridge is a thin wrapper around T10. These tests verify the routing and key
safety without hitting the live model.
"""
import json
import os
import threading
import time
import requests

import pytest

from src.types import Item, Box
from src.air.bridge import make_server, serve


ITEMS = [Item(label="egg", axes=(22.0, 22.0, 30.0), mass_g=60.0),
         Item(label="potato", axes=(35.0, 28.0, 25.0), mass_g=170.0),
         Item(label="lime", axes=(24.0, 24.0, 26.0), mass_g=70.0)]
CATALOG = [Box(300.0, 200.0, 150.0)]


def test_server_fails_fast_if_key_is_missing():
    """Fail at startup, not on the first request during a pitch."""
    # Temporarily remove key if present
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            make_server(ITEMS, CATALOG, port=8001)
    finally:
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key


def test_server_binds_to_loopback_only():
    """Never bind to 0.0.0.0 on conference wifi."""
    # Set a dummy key for this test
    os.environ["OPENAI_API_KEY"] = "test_key_for_bridge_test"
    try:
        server = make_server(ITEMS, CATALOG, port=8002)
        # Check that the server is bound to 127.0.0.1
        assert server.server_address[0] == "127.0.0.1"
    finally:
        del os.environ["OPENAI_API_KEY"]


def test_get_root_returns_static_page():
    """The demo page is served from /."""
    os.environ["OPENAI_API_KEY"] = "test_key_for_bridge_test"
    
    # Start server in background
    server = make_server(ITEMS, CATALOG, port=8003)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # Give server time to start
    
    try:
        response = requests.get("http://127.0.0.1:8003/")
        assert response.status_code == 200
        assert "text/html" in response.headers["Content-Type"]
    finally:
        server.shutdown()
        del os.environ["OPENAI_API_KEY"]


def test_post_turn_returns_transcript():
    """The turn endpoint returns the full transcript."""
    os.environ["OPENAI_API_KEY"] = "test_key_for_bridge_test"
    
    # Start server in background
    server = make_server(ITEMS, CATALOG, port=8004)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # Give server time to start
    
    try:
        response = requests.post(
            "http://127.0.0.1:8004/api/turn",
            json={"text": "list the items"},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "transcript" in data
        assert isinstance(data["transcript"], list)
    finally:
        server.shutdown()
        del os.environ["OPENAI_API_KEY"]


def test_key_never_appears_in_response():
    """The key must never leak into any response body."""
    os.environ["OPENAI_API_KEY"] = "secret_test_key_12345"
    
    # Start server in background
    server = make_server(ITEMS, CATALOG, port=8005)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # Give server time to start
    
    try:
        # Test both endpoints
        for endpoint in ["/", "/api/turn"]:
            if endpoint == "/":
                response = requests.get(f"http://127.0.0.1:8005{endpoint}")
            else:
                response = requests.post(
                    f"http://127.0.0.1:8005{endpoint}",
                    json={"text": "list the items"},
                    headers={"Content-Type": "application/json"}
                )
            
            assert response.status_code == 200
            content = response.text
            assert "secret_test_key_12345" not in content
            assert "12345" not in content  # Even partial leaks are forbidden
    finally:
        server.shutdown()
        del os.environ["OPENAI_API_KEY"]


def test_unknown_route_returns_404():
    """Unknown routes return 404, not 200 with an error body."""
    os.environ["OPENAI_API_KEY"] = "test_key_for_bridge_test"
    
    # Start server in background
    server = make_server(ITEMS, CATALOG, port=8006)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # Give server time to start
    
    try:
        response = requests.get("http://127.0.0.1:8006/unknown")
        assert response.status_code == 404
    finally:
        server.shutdown()
        del os.environ["OPENAI_API_KEY"]