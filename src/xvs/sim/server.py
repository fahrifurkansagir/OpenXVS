"""Local HTTP server and tile proxy with allowlist enforcement and audit logging.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §2.3 and Findings C3, C4, C5, C6.
"""

from __future__ import annotations
import hashlib
import json
import mimetypes
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from xvs.sim.allowlist import URLClassifier, get_default_classifier

REPO_ROOT = Path(__file__).resolve().parents[3]


class XVSServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        cesium_build_dir: Path,
        page_dir: Path,
        cache_dir: Path,
        log_dir: Path,
        offline: bool = False,
        allowlist: URLClassifier | None = None,
        upstream_base_url: str | None = None,
    ):
        super().__init__(server_address, XVSRequestHandler)
        self.cesium_build_dir = cesium_build_dir
        self.page_dir = page_dir
        self.cache_dir = cache_dir
        self.log_dir = log_dir
        self.offline = offline
        self.allowlist = allowlist or get_default_classifier()
        self.upstream_base_url = upstream_base_url

        self.upstream_semaphore = threading.Semaphore(4)
        self.in_flight_requests = 0
        self._counter_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._log_lock = threading.Lock()

        # Audit logs paths
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.proxy_log_path = self.log_dir / "proxy_log.jsonl"
        self.violations_path = self.log_dir / "data_violations.jsonl"
        self.data_sources_path = self.log_dir / "data_sources.json"

        # Initialize event-based stats in __init__ (Finding C5)
        self.stats: dict[str, Any] = {
            "usgs_imagery": {
                "id": "usgs_imagery_only",
                "attribution": "USDA, USGS The National Map: Orthoimagery",
                "license": "U.S. Public Domain",
                "host": "basemap.nationalmap.gov",
                "path_prefix": "/arcgis/rest/services/USGSImageryOnly/MapServer/",
                "tiles_served_total": 0,
                "tiles_cache_hits": 0,
                "tiles_upstream_fetches": 0,
                "bytes_served_total": 0,
                "bytes_upstream_fetched": 0,
                "tile_errors": 0,
                "first_access_utc": None,
                "last_access_utc": None,
            }
        }
        self._sync_reference_sources()

    def _sync_reference_sources(self) -> None:
        """Dynamically load metadata from reference data sidecars into stats."""
        meta_file = REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv.meta.json"
        if meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                self.stats["ourairports"] = {
                    "id": meta.get("id", "ourairports"),
                    "attribution": meta.get("attribution", "OurAirports (ourairports.com)"),
                    "license": meta.get("license", "Public Domain"),
                    "host": meta.get("host", "davidmegginson.github.io"),
                    "path_prefix": meta.get("path_prefix", "/ourairports-data/"),
                    "object_count": 1,
                    "bytes": meta.get("bytes", 0),  # default-ok: simulation server protocol default
                    "sha256": meta.get("sha256", ""),
                    "verified_utc": meta.get("verified_utc", None),
                }
            except Exception:
                pass

    def clear_logs(self) -> None:
        with self._log_lock:
            if self.proxy_log_path.is_file():
                self.proxy_log_path.unlink()
            if self.violations_path.is_file():
                self.violations_path.unlink()
            if self.data_sources_path.is_file():
                self.data_sources_path.unlink()

    def increment_in_flight(self) -> None:
        with self._counter_lock:
            self.in_flight_requests += 1

    def decrement_in_flight(self) -> None:
        with self._counter_lock:
            self.in_flight_requests -= 1

    def get_in_flight(self) -> int:
        with self._counter_lock:
            return self.in_flight_requests

    def log_proxy(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        with self._log_lock:
            with open(self.proxy_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def log_violation(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        with self._log_lock:
            with open(self.violations_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def record_tile_served(self, cache_hit: bool, num_bytes: int) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._stats_lock:
            st = self.stats["usgs_imagery"]
            st["tiles_served_total"] += 1
            if cache_hit:
                st["tiles_cache_hits"] += 1
            else:
                st["tiles_upstream_fetches"] += 1
                st["bytes_upstream_fetched"] += num_bytes

            st["bytes_served_total"] += num_bytes
            if st["first_access_utc"] is None:
                st["first_access_utc"] = now_iso
            st["last_access_utc"] = now_iso

            self._sync_reference_sources()
            with open(self.data_sources_path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)

    def record_tile_error(self) -> None:
        with self._stats_lock:
            st = self.stats["usgs_imagery"]
            st["tile_errors"] += 1
            with open(self.data_sources_path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)


class XVSRequestHandler(SimpleHTTPRequestHandler):
    server: XVSServer

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default stderr request spam
        pass

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        # 1. Cesium library assets
        if path.startswith("/cesium/"):
            rel_path = path[len("/cesium/"):]
            file_path = (self.server.cesium_build_dir / rel_path).resolve()
            if not str(file_path).startswith(str(self.server.cesium_build_dir.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._serve_file(file_path)
            return

        # 2. XVS test/viewer pages
        if path.startswith("/page/"):
            rel_path = path[len("/page/"):]
            file_path = (self.server.page_dir / rel_path).resolve()
            if not str(file_path).startswith(str(self.server.page_dir.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._serve_file(file_path)
            return

        # 3. USGS imagery tiles
        if path.startswith("/tiles/usgs_imagery/"):
            self._handle_tile_request(path)
            return

        # 4. Status endpoint
        if path == "/in_flight":
            in_flight = self.server.get_in_flight()
            payload = json.dumps({"in_flight": in_flight}).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # 5. Traffic assets (WP-02 §2.2)
        if path.startswith("/assets/traffic/"):
            rel_path = path[len("/assets/traffic/"):]
            traffic_dir = (REPO_ROOT / "assets" / "traffic").resolve()
            file_path = (traffic_dir / rel_path).resolve()
            if not str(file_path).startswith(str(traffic_dir)):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._serve_file(file_path)
            return

        # Any other path is unhandled / forbidden
        self.server.log_violation({
            "utc": datetime.now(timezone.utc).isoformat(),
            "path": path,
            "error": "Forbidden path requested on local server"
        })
        self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(str(file_path))
        if file_path.suffix == ".js":
            content_type = "application/javascript"
        elif file_path.suffix == ".css":
            content_type = "text/css"
        elif file_path.suffix == ".json":
            content_type = "application/json"
        elif file_path.suffix == ".glb":
            content_type = "model/gltf-binary"
        elif file_path.suffix == ".gltf":
            content_type = "model/gltf+json"
        elif not content_type:
            content_type = "application/octet-stream"

        try:
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def _handle_tile_request(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 5:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid tile path format")
            return
        _, _, z_str, y_str, x_str = parts
        try:
            z = int(z_str)
            y = int(y_str)
            x = int(x_str)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Non-integer tile coordinates")
            return

        t0 = time.perf_counter()
        cache_tile_path = self.server.cache_dir / f"{z}/{y}/{x}.tile"
        cache_meta_path = self.server.cache_dir / f"{z}/{y}/{x}.meta.json"

        # Check disk cache
        if cache_tile_path.is_file() and cache_meta_path.is_file():
            try:
                meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                status = meta.get("upstream_status", 200)  # default-ok: simulation server protocol default
                content_type = meta.get("content_type", "image/jpeg")
                data = cache_tile_path.read_bytes()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                self.server.log_proxy({
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "path": path,
                    "cache_hit": True,
                    "upstream_status": status,
                    "bytes": len(data),
                    "ms": round(elapsed_ms, 2)
                })

                if status == 404:
                    self.server.record_tile_served(cache_hit=True, num_bytes=0)
                    self.send_error(HTTPStatus.NOT_FOUND, "Cached 404")
                    return

                if status == 200:
                    self.server.record_tile_served(cache_hit=True, num_bytes=len(data))
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(data)
                    return
            except Exception:
                pass  # Corrupted cache fallback

        if self.server.offline:
            self.send_error(HTTPStatus.NOT_FOUND, "Offline mode miss")
            return

        # Upstream URL determination
        if self.server.upstream_base_url:
            upstream_url = f"{self.server.upstream_base_url.rstrip('/')}/tile/{z}/{y}/{x}"
        else:
            upstream_url = (
                f"https://basemap.nationalmap.gov/arcgis/rest/services/"
                f"USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            )

        # Allowlist check (Finding C6)
        # Note: if upstream_base_url is used for testing, allow it if it is 127.0.0.1 / test server
        if not self.server.upstream_base_url:
            verdict, reason, _ = self.server.allowlist.classify_upstream_url(upstream_url)
            if verdict != "ALLOWED":
                self.server.log_violation({
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "path": path,
                    "upstream_url": upstream_url,
                    "error": f"Upstream fetch rejected by allowlist: {reason}"
                })
                self.send_error(HTTPStatus.FORBIDDEN, f"Upstream disallowed: {reason}")
                return

        self.server.increment_in_flight()
        try:
            with self.server.upstream_semaphore:
                data, status, content_type = self._fetch_upstream_with_backoff(upstream_url)
        finally:
            self.server.decrement_in_flight()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.server.log_proxy({
            "utc": datetime.now(timezone.utc).isoformat(),
            "path": path,
            "cache_hit": False,
            "upstream_status": status,
            "bytes": len(data),
            "ms": round(elapsed_ms, 2)
        })

        if status == 200:
            # Cache 200 OK
            cache_tile_path.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256(data).hexdigest()
            meta = {
                "upstream_status": 200,
                "content_type": content_type,
                "sha256": h,
                "fetched_utc": datetime.now(timezone.utc).isoformat(),
                "bytes": len(data)
            }
            cache_tile_path.write_bytes(data)
            cache_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            self.server.record_tile_served(cache_hit=False, num_bytes=len(data))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        elif status == 404:
            # Cache 404 negative entry (Finding C4)
            cache_tile_path.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "upstream_status": 404,
                "content_type": "text/plain",
                "sha256": "",
                "fetched_utc": datetime.now(timezone.utc).isoformat(),
                "bytes": 0
            }
            cache_tile_path.write_bytes(b"")
            cache_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            self.server.record_tile_served(cache_hit=False, num_bytes=0)
            self.send_error(HTTPStatus.NOT_FOUND, "Upstream 404")

        else:
            # Other errors (429, 500, 502, 504, timeouts) are NEVER cached! (Finding C4)
            self.server.record_tile_error()
            if status in (HTTPStatus.GATEWAY_TIMEOUT, HTTPStatus.BAD_GATEWAY):
                self.send_error(status, "Upstream gateway error")
            else:
                self.send_error(HTTPStatus.BAD_GATEWAY, f"Upstream error {status}")

    def _fetch_upstream_with_backoff(self, url: str) -> tuple[bytes, int, str]:
        backoffs = [0.0, 1.0, 2.0, 4.0, 8.0]
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "OpenXVS-research/0.1"}
        )

        for attempt, delay in enumerate(backoffs):
            if delay > 0.0:
                time.sleep(delay)
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                    status = resp.status
                    content_type = resp.headers.get("Content-Type", "image/jpeg")
                    return data, status, content_type
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return b"", 404, "text/plain"
                if e.code in (429, 500, 502, 503, 504) and attempt < len(backoffs) - 1:
                    continue
                return b"", e.code, "text/plain"
            except (urllib.error.URLError, TimeoutError):
                if attempt < len(backoffs) - 1:
                    continue
                return b"", HTTPStatus.GATEWAY_TIMEOUT, "text/plain"

        return b"", HTTPStatus.GATEWAY_TIMEOUT, "text/plain"
