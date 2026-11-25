"""CesiumJS viewport controller and frame renderer using Playwright.

Conforms strictly to:
- docs/xvs/CONVENTIONS.md v1.0 §2.4, §2.5, §5, §6
- docs/xvs/WP01_SIM_RIG.md v1.2 §2.4, §2.5, §3 (G2)
- Director review decisions D5, D6, D7 (WP01_G1_director_review.md)
"""

from __future__ import annotations
import base64
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page, Playwright

from xvs.sim.allowlist import URLClassifier, setup_playwright_route_guard
from xvs.sim.server import XVSServer

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs" / "xvs"


@dataclass
class RenderResult:
    """Result of a single camera frame render and capture."""
    image: np.ndarray  # (1024, 1024, 3) uint8 RGB, top-down
    n_renders: int
    stable: bool
    incomplete: bool
    last_max_diff: int
    last_n_diff: int
    readback: dict[str, Any]
    ms_set_camera: float
    ms_render_loop: float
    ms_readback: float
    drawing_buffer: tuple[int, int] = (1024, 1024)
    rgb_sha256: str = ""
    tile_state: dict[str, Any] | None = None  # D26a: globe tiles loaded and pending requests at capture


class XVSRenderer:
    """Headless Chromium renderer controlling CesiumJS viewports."""

    SUPPORTED_PROFILES = (
        "rgb",
        "mask_earth",
        "mask_earth_binary",
        "markers",
        "mask_visible",
        "mask_amodal",
        "mask_calib",
    )
    SUPPORTED_BACKENDS = ("ref", "hw")

    def __init__(
        self,
        port: int,
        backend: str = "ref",
        server: XVSServer | None = None,
        allowlist: URLClassifier | None = None,
        headless: bool = True,
    ):
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Must be one of {self.SUPPORTED_BACKENDS}")

        self.port = port
        self.backend = backend
        self.server = server
        self.allowlist = allowlist or (server.allowlist if server else None)
        self.headless = headless

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._cam_pages: dict[str, Page] = {}
        self._active_profile: str | None = None

        # Load render profiles configuration
        profiles_path = CONFIGS_DIR / "render_profiles.json"
        if profiles_path.exists():
            with open(profiles_path, "r", encoding="utf-8") as f:
                self.profiles_config = json.load(f).get("profiles", {})
        else:
            self.profiles_config = {}

    def get_browser_args(self) -> list[str]:
        """Return Chromium launch arguments according to backend (§2.5).

        Exhaustive list per spec §2.5: no undeclared flags.
        """
        if self.backend == "ref":
            return [
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ]
        elif self.backend == "hw":
            return [
                "--use-angle=d3d11",
                "--enable-gpu",
                "--ignore-gpu-blocklist",
            ]
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def start(self) -> XVSRenderer:
        """Launch Playwright browser, initialize context, and open primary page."""
        if self._playwright is not None:
            return self

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=self.get_browser_args(),
        )

        self._context = self._browser.new_context(
            viewport={"width": 1024, "height": 1024},
            device_scale_factor=1.0,
        )

        if self.allowlist is not None:
            setup_playwright_route_guard(self._context, classifier=self.allowlist, server=self.server)

        self._page = self._context.new_page()
        self._load_cesium_page(self._page)
        return self

    def _load_cesium_page(self, page: Page) -> None:
        """Navigate a page to the Cesium container and wait for initialization."""
        url = f"http://127.0.0.1:{self.port}/page/index.html"
        page.goto(url)
        page.wait_for_function(
            "typeof Cesium !== 'undefined' && typeof window.initXVSViewer !== 'undefined'"
        )

    def new_camera_page(self, cam_id: str, profile: str = "rgb") -> Page:
        """Create a dedicated resident page for a specific camera (spec §2.5)."""
        if self._context is None:
            raise RuntimeError("Renderer must be started before creating pages")

        page = self._context.new_page()
        self._load_cesium_page(page)
        self.set_profile(profile, page=page)
        self._cam_pages[cam_id] = page
        return page

    def get_page(self, cam_id: str | None = None) -> Page:
        """Retrieve the page for a camera ID or default to the primary page."""
        if cam_id is not None and cam_id in self._cam_pages:
            return self._cam_pages[cam_id]
        if self._page is not None:
            return self._page
        raise RuntimeError("Renderer has no active page")

    def set_profile(
        self,
        profile_name: str,
        msaa_override: int | None = None,
        max_level: int = 16,
        page: Page | None = None,
    ) -> dict[str, Any]:
        """Initialize Cesium viewer with the specified profile (§2.4)."""
        if profile_name not in self.SUPPORTED_PROFILES:
            raise ValueError(f"Unknown render profile '{profile_name}'. Must be one of {self.SUPPORTED_PROFILES}")

        p = page or self.get_page()
        profile_cfg = self.profiles_config.get(profile_name, {})
        res = p.evaluate(
            "([name, port, maxLvl, msaa, cfg]) => window.initXVSViewer(name, port, maxLvl, msaa, cfg)",
            [profile_name, self.port, max_level, msaa_override, profile_cfg],
        )
        self._active_profile = profile_name
        self._resolved_settings = res
        return res

    @property
    def resolved_settings(self) -> dict[str, Any] | None:
        """Return the resolved render profile settings applied to the active viewer."""
        return getattr(self, "_resolved_settings", None)

    def set_camera(
        self,
        position: list[float] | np.ndarray,
        direction: list[float] | np.ndarray,
        up: list[float] | np.ndarray,
        fov_rad: float,
        aspect_ratio: float,
        page: Page | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Assign camera vectors directly in ECEF (CONVENTIONS §6, Note N3)."""
        p = page or self.get_page()
        pos_list = position.tolist() if isinstance(position, np.ndarray) else list(position)
        dir_list = direction.tolist() if isinstance(direction, np.ndarray) else list(direction)
        up_list = up.tolist() if isinstance(up, np.ndarray) else list(up)

        t0 = time.perf_counter()
        p.evaluate(
            "([pos, dir, u, fov, asp]) => window.setXVSCamera(pos, dir, u, fov, asp)",
            [pos_list, dir_list, up_list, float(fov_rad), float(aspect_ratio)],
        )
        readback = p.evaluate("window.getXVSReadback()")
        ms_set = (time.perf_counter() - t0) * 1000.0
        return readback, ms_set

    def set_clock(self, iso_utc_string: str, page: Page | None = None) -> None:
        """Set Cesium clock to the specified ISO-8601 UTC time."""
        p = page or self.get_page()
        p.evaluate("utc => window.setXVSClock(utc)", iso_utc_string)

    def set_markers(self, markers: list[dict[str, Any]], page: Page | None = None) -> None:
        """Upload markers to Cesium PointPrimitiveCollection (§3 G2)."""
        p = page or self.get_page()
        p.evaluate("markers => window.setXVSMarkers(markers)", markers)

    def clear_markers(self, page: Page | None = None) -> None:
        """Clear all active markers from the scene."""
        p = page or self.get_page()
        p.evaluate("window.clearXVSMarkers()")

    def get_marker_projections(
        self,
        markers: list[dict[str, Any]],
        page: Page | None = None,
    ) -> list[dict[str, Any] | None]:
        """Compute Cesium worldToWindowCoordinates for each marker (CONVENTIONS §6)."""
        p = page or self.get_page()
        return p.evaluate("markers => window.getXVSMarkerProjections(markers)", markers)

    def add_model(
        self,
        url: str,
        model_matrix: list[float] | np.ndarray | None = None,
        custom_shader: str | None = None,
        page: Page | None = None,
    ) -> dict[str, Any]:
        """Load glTF/GLB model into Cesium scene (WP-02 §2.2)."""
        p = page or self.get_page()
        mat = model_matrix.tolist() if isinstance(model_matrix, np.ndarray) else model_matrix
        return p.evaluate(
            "params => window.addXVSModel(params)",
            {"url": url, "modelMatrix": mat, "customShader": custom_shader},
        )

    def clear_models(self, page: Page | None = None) -> None:
        """Remove and destroy all active models from the scene."""
        p = page or self.get_page()
        p.evaluate("window.clearXVSModels()")

    def set_model_pose(
        self,
        index: int,
        model_matrix: list[float] | np.ndarray,
        page: Page | None = None,
    ) -> None:
        """Update modelMatrix of loaded model by index."""
        p = page or self.get_page()
        mat = model_matrix.tolist() if isinstance(model_matrix, np.ndarray) else model_matrix
        p.evaluate(
            "([idx, m]) => window.setXVSModelPose(idx, m)",
            [index, mat],
        )

    def set_model_show(self, index: int, show: bool, page: Page | None = None) -> None:
        """Toggle visibility of model by index."""
        p = page or self.get_page()
        p.evaluate("([idx, s]) => window.setXVSModelShow(idx, s)", [index, bool(show)])

    def set_depth_test_against_terrain(self, enabled: bool, page: Page | None = None) -> None:
        """Toggle globe.depthTestAgainstTerrain at runtime (WP-02 G0 item 5)."""
        p = page or self.get_page()
        p.evaluate("e => window.setXVSDepthTestAgainstTerrain(e)", bool(enabled))

    def set_roi_frustum(
        self,
        u0: int,
        v0: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        n: float = 0.5,
        w: int = 1024,
        h: int = 1024,
        page: Page | None = None,
    ) -> None:
        """Set PerspectiveOffCenterFrustum for camera ROI rendering (WP-02 §2.4)."""
        p = page or self.get_page()
        p.evaluate(
            "([u0, v0, fx, fy, cx, cy, n, w, h]) => window.setXVSROIFrustum(u0, v0, fx, fy, cx, cy, n, w, h)",
            [int(u0), int(v0), float(fx), float(fy), float(cx), float(cy), float(n), int(w), int(h)],
        )

    def render_frame(
        self,
        page: Page | None = None,
        max_renders: int = 600,
        ms_set_camera: float = 0.0,
        transfer_method: str = "png",
        require_no_pending: bool = False,
    ) -> RenderResult:
        """Execute in-page stable-frame loop and capture final image (§2.5, N11).

        Transfers pixels to Python ONLY ONCE upon stability or cap reached.
        Uses in-page lossless PNG encoding (adopted per N11) for 4.6x speedup.
        """
        p = page or self.get_page()
        p.evaluate("window.resetXVSStability()")

        t_render_start = time.perf_counter()
        n_renders = 0
        last_max_diff = 0
        last_n_diff = 0
        stable = False

        while n_renders < max_renders:
            proxy_idle = (self.server.get_in_flight() == 0) if self.server else True
            res = p.evaluate(
                "([b, idle, np]) => window.stepXVSStability(b, idle, np)",
                [self.backend, proxy_idle, bool(require_no_pending)],
            )
            n_renders = res["renderCount"]
            last_max_diff = res.get("maxDiff", 0)  # default-ok: simulation server protocol default
            last_n_diff = res.get("nDiff", 0)  # default-ok: simulation server protocol default

            if res["stable"]:
                stable = True
                break

            if not res.get("readyForComparison", False):
                if n_renders >= 5:
                    time.sleep(0.01)

        ms_render_loop = (time.perf_counter() - t_render_start) * 1000.0

        # Read back final image pixels exactly once via fastest bit-exact method (N11)
        t_readback_start = time.perf_counter()
        if transfer_method == "png":
            b64_str = p.evaluate("window.readXVSPixelsPNG()")
            raw_png = base64.b64decode(b64_str)
            pil_img = Image.open(io.BytesIO(raw_png)).convert("RGB")
            img_rgb = np.array(pil_img)
        elif transfer_method == "rgb":
            b64_str = p.evaluate("window.readXVSPixelsRGB()")
            raw_bytes = base64.b64decode(b64_str)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(1024, 1024, 3)
            img_rgb = np.flipud(arr).copy()
        else:
            b64_str = p.evaluate("window.readXVSPixelsBase64()")
            raw_bytes = base64.b64decode(b64_str)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(1024, 1024, 4)
            img_rgb = np.flipud(arr[:, :, :3]).copy()

        ms_readback = (time.perf_counter() - t_readback_start) * 1000.0

        readback = p.evaluate("window.getXVSReadback()")
        dw_val = p.evaluate("window.xvsWidget.scene.drawingBufferWidth")
        dh_val = p.evaluate("window.xvsWidget.scene.drawingBufferHeight")
        drawing_buffer = (int(dw_val), int(dh_val))
        rgb_sha256 = hashlib.sha256(np.ascontiguousarray(img_rgb).tobytes()).hexdigest()
        tile_state = p.evaluate("window.getXVSTileState()") if require_no_pending else None

        return RenderResult(
            image=img_rgb,
            n_renders=n_renders,
            stable=stable,
            incomplete=(not stable),
            last_max_diff=last_max_diff,
            last_n_diff=last_n_diff,
            readback=readback,
            ms_set_camera=ms_set_camera,
            ms_render_loop=ms_render_loop,
            ms_readback=ms_readback,
            drawing_buffer=drawing_buffer,
            rgb_sha256=rgb_sha256,
            tile_state=tile_state,
        )

    def save_png(
        self,
        image: np.ndarray,
        output_path: str | Path,
        compress_level: int = 6,
    ) -> tuple[int, float]:
        """Encode and save uint8 RGB image as PNG via Pillow.

        Returns (file_size_bytes, encode_ms).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        pil_img = Image.fromarray(image, mode="RGB")
        pil_img.save(path, format="PNG", compress_level=compress_level)
        ms_encode = (time.perf_counter() - t0) * 1000.0

        size_bytes = path.stat().st_size
        return size_bytes, ms_encode

    def get_webgl_diagnostics(self, page: Page | None = None) -> dict[str, Any]:
        """Query WebGL context vendor, renderer, and sample limits."""
        p = page or self.get_page()
        return p.evaluate("window.getWebGLDiagnostics()")

    def close(self) -> None:
        """Cleanly close all pages, context, browser, and Playwright session."""
        for p in self._cam_pages.values():
            try:
                p.close()
            except Exception:
                pass
        self._cam_pages.clear()

        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None

        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None

        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None

        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def __enter__(self) -> XVSRenderer:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
