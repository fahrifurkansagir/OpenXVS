"""Unit tests for XVSServer: threading, error non-caching, event-based logging, and allowlist.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §2.3, §3 (G1) and Findings C3, C4, C5, C6.
"""

import json
from pathlib import Path
import socket
import threading
import time
import urllib.request
import urllib.error
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest

from xvs.sim.server import XVSServer
from xvs.sim.allowlist import URLClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MockUpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/tile/10/500/"):
            # Return 500 Internal Server Error
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"upstream error")
        elif self.path.startswith("/tile/10/404/"):
            # Return 404 Not Found
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
        elif self.path.startswith("/tile/10/200/"):
            # Return 200 OK
            data = b"MOCK_JPEG_IMAGE_BYTES_12345"
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(400)
            self.end_headers()


def test_server_threading_inheritance_and_daemon():
    """Finding C3: XVSServer must subclass ThreadingHTTPServer with daemon_threads = True."""
    assert issubclass(XVSServer, ThreadingHTTPServer)
    assert XVSServer.daemon_threads is True


def test_upstream_error_non_caching_and_404_negative_cache():
    """Finding C4: Upstream failures (5xx/429/timeout) are NEVER cached; 404 is cached as negative entry."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_path = Path(tmp_str)
        mock_port = get_free_port()
        mock_server = HTTPServer(("127.0.0.1", mock_port), MockUpstreamHandler)
        mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
        mock_thread.start()

        server_port = get_free_port()
        cache_dir = tmp_path / "cache"
        log_dir = tmp_path / "logs"

        server = XVSServer(
            server_address=("127.0.0.1", server_port),
            cesium_build_dir=REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium",
            page_dir=REPO_ROOT / "sim" / "cesium" / "page",
            cache_dir=cache_dir,
            log_dir=log_dir,
            offline=False,
            upstream_base_url=f"http://127.0.0.1:{mock_port}",
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            # 1. Test 500 error response from upstream
            tile_url_500 = f"http://127.0.0.1:{server_port}/tiles/usgs_imagery/10/500/1"
            req = urllib.request.Request(tile_url_500)
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req)
            # Server should return 502 Bad Gateway
            assert exc_info.value.code == HTTPStatus.BAD_GATEWAY

            # Ensure NO tile or meta is written to cache! (Finding C4)
            tile_path_500 = cache_dir / "10/500/1.tile"
            meta_path_500 = cache_dir / "10/500/1.meta.json"
            assert not tile_path_500.exists(), "Upstream 500 error MUST NOT be cached as .tile!"
            assert not meta_path_500.exists(), "Upstream 500 error MUST NOT be cached as .meta.json!"

            # Ensure tile_errors is incremented in stats
            assert server.stats["usgs_imagery"]["tile_errors"] >= 1

            # 2. Test 404 response from upstream
            tile_url_404 = f"http://127.0.0.1:{server_port}/tiles/usgs_imagery/10/404/1"
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(tile_url_404)
            assert exc_info.value.code == HTTPStatus.NOT_FOUND

            # 404 should be cached as negative entry
            tile_path_404 = cache_dir / "10/404/1.tile"
            meta_path_404 = cache_dir / "10/404/1.meta.json"
            assert tile_path_404.exists(), "Upstream 404 should be cached as negative entry"
            assert meta_path_404.exists()
            meta_404 = json.loads(meta_path_404.read_text(encoding="utf-8"))
            assert meta_404["upstream_status"] == 404

            # Subsequent request served from cache as 404
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(tile_url_404)
            assert exc_info.value.code == HTTPStatus.NOT_FOUND
            assert server.stats["usgs_imagery"]["tiles_cache_hits"] >= 1

            # 3. Test 200 OK from upstream
            tile_url_200 = f"http://127.0.0.1:{server_port}/tiles/usgs_imagery/10/200/1"
            with urllib.request.urlopen(tile_url_200) as resp:
                data = resp.read()
                assert resp.status == 200
                assert data == b"MOCK_JPEG_IMAGE_BYTES_12345"

            tile_path_200 = cache_dir / "10/200/1.tile"
            meta_path_200 = cache_dir / "10/200/1.meta.json"
            assert tile_path_200.exists()
            assert meta_path_200.exists()
            meta_200 = json.loads(meta_path_200.read_text(encoding="utf-8"))
            assert meta_200["upstream_status"] == 200

            # Subsequent request is cache hit
            with urllib.request.urlopen(tile_url_200) as resp:
                data = resp.read()
                assert resp.status == 200
                assert data == b"MOCK_JPEG_IMAGE_BYTES_12345"

        finally:
            server.shutdown()
            server.server_close()
            mock_server.shutdown()
            mock_server.server_close()


def test_event_based_data_sources_logging():
    """Finding C5: data_sources.json is built from actual runtime events and reference metadata."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_path = Path(tmp_str)
        server_port = get_free_port()
        cache_dir = tmp_path / "cache"
        log_dir = tmp_path / "logs"

        server = XVSServer(
            server_address=("127.0.0.1", server_port),
            cesium_build_dir=REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium",
            page_dir=REPO_ROOT / "sim" / "cesium" / "page",
            cache_dir=cache_dir,
            log_dir=log_dir,
            offline=True,  # offline mode
        )
        # Initialized in __init__
        assert "usgs_imagery" in server.stats
        assert server.stats["usgs_imagery"]["tiles_served_total"] == 0
        assert server.stats["usgs_imagery"]["tiles_cache_hits"] == 0
        assert server.stats["usgs_imagery"]["tiles_upstream_fetches"] == 0

        # Reference file metadata loaded dynamically
        if (REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv.meta.json").is_file():
            assert "ourairports" in server.stats
            assert server.stats["ourairports"]["bytes"] == 3965119
            assert server.stats["ourairports"]["sha256"] == "97f30f038be33a1c93c190b9a90c46fabbe16e36a86f5decdac62259613411ae"
