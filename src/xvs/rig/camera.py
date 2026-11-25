"""Ideal pinhole camera model, projection, back-projection, and intrinsics.

Conforms strictly to docs/xvs/CONVENTIONS.md v1.0 §5.
"""

from __future__ import annotations
import math
import numpy as np


class PinholeCamera:
    """Ideal pinhole camera with square pixels and zero distortion (WP-01).

    Pixel convention:
      Integer (u, v) are pixel centres. (0, 0) is the centre of top-left pixel.
      Image area spans u in [-0.5, W - 0.5], v in [-0.5, H - 0.5].
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        fov_h_deg: float = 100.0,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
    ):
        self.width = width
        self.height = height
        self.fov_h_deg = fov_h_deg

        if fx is None or fy is None:
            fov_rad = math.radians(fov_h_deg)
            self.fx = (self.width / 2.0) / math.tan(fov_rad / 2.0)
            self.fy = self.fx
        else:
            self.fx = float(fx)
            self.fy = float(fy)

        if cx is None:
            self.cx = (self.width - 1.0) / 2.0
        else:
            self.cx = float(cx)

        if cy is None:
            self.cy = (self.height - 1.0) / 2.0
        else:
            self.cy = float(cy)

        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def project_cam(self, p_cam: np.ndarray) -> tuple[float, float, bool]:
        """Project a 3D point in camera frame to pixel coordinates (u, v).

        Returns (u, v, is_front) where is_front is True iff p_cam[2] > 0.
        """
        x, y, z = p_cam
        if z <= 0.0:
            return float("nan"), float("nan"), False
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return float(u), float(v), True

    def project_ecef(
        self,
        p_ecef: np.ndarray,
        p_ecef_cam: np.ndarray,
        r_ecef_cam: np.ndarray,
    ) -> tuple[float, float, bool]:
        """Project an ECEF point to pixel coordinates given camera ECEF pose."""
        # pc = R_ecef_cam.T @ (P - p_ecef_cam)
        p_cam = r_ecef_cam.T @ (p_ecef - p_ecef_cam)
        return self.project_cam(p_cam)

    def unproject(self, u: float, v: float) -> np.ndarray:
        """Back-project a pixel coordinate (u, v) to a unit ray in camera frame.

        ray_cam = normalize([(u - cx) / fx, (v - cy) / fy, 1.0])
        """
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        ray = np.array([x, y, 1.0], dtype=np.float64)
        norm = np.linalg.norm(ray)
        return ray / norm

    def in_image(self, u: float, v: float) -> bool:
        """Check if pixel coordinate (u, v) falls within image boundaries."""
        return (
            (-0.5 <= u <= self.width - 0.5) and
            (-0.5 <= v <= self.height - 0.5)
        )
