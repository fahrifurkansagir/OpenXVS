"""Traffic camera ring and ROI rendering math (WP02 §2.4, spec/traffic_ring_v1.json).

Manages the 8-camera ring (T0-T7) and region-of-interest (ROI) frustum calculations.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC_PATH = REPO_ROOT / "configs" / "xvs" / "spec" / "traffic_ring_v1.json"

from xvs.geo import wgs84, rotations
from xvs.rig.camera import PinholeCamera


@dataclass
class RingCamera:
    id: str
    azimuth_body_deg: float
    lever_arm_m: np.ndarray  # (3,)
    r_body_cam: np.ndarray   # (3, 3)
    boresight_body: np.ndarray  # (3,)


class TrafficRing:
    """Manages the 8-camera traffic ring geometry and ROI calculations."""

    def __init__(self, spec_path: str | Path = DEFAULT_SPEC_PATH) -> None:
        self.spec_path = Path(spec_path)
        with open(self.spec_path, "r", encoding="utf-8") as f:
            self.spec = json.load(f)

        self.ring_id: str = self.spec["ring_id"]
        img_cfg = self.spec["image"]
        self.width: int = img_cfg["width"]
        self.height: int = img_cfg["height"]

        intrin_cfg = self.spec["intrinsics"]
        self.fx: float = float(intrin_cfg["fx"])
        self.fy: float = float(intrin_cfg["fy"])
        self.cx: float = float(intrin_cfg["cx"])
        self.cy: float = float(intrin_cfg["cy"])
        self.fov_h_deg: float = float(intrin_cfg["fov_h_deg"])
        self.fov_v_deg: float = float(intrin_cfg["fov_v_deg"])

        self.camera_model = PinholeCamera(
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )

        self.cameras: list[RingCamera] = []
        self._cameras_by_id: dict[str, RingCamera] = {}
        for c in self.spec["cameras"]:
            rc = RingCamera(
                id=c["id"],
                azimuth_body_deg=float(c["azimuth_body_deg"]),
                lever_arm_m=np.array(c["lever_arm_m"], dtype=np.float64),
                r_body_cam=np.array(c["R_body_cam"], dtype=np.float64),
                boresight_body=np.array(c["boresight_body"], dtype=np.float64),
            )
            self.cameras.append(rc)
            self._cameras_by_id[rc.id] = rc

    def get_camera(self, cam_id: str) -> RingCamera:
        return self._cameras_by_id[cam_id]

    def compute_roi(self, u: float, v: float) -> tuple[int, int]:
        """Compute ROI top-left (u0, v0) integer clamped to full image (WP02 §2.4)."""
        u0 = int(np.clip(np.round(u - 511.5), 0, self.width - 1024))
        v0 = int(np.clip(np.round(v - 511.5), 0, self.height - 1024))
        return u0, v0

    def compute_roi_frustum(
        self,
        u0: int,
        v0: int,
        near: float = 0.5,
        roi_w: int = 1024,
        roi_h: int = 1024,
    ) -> dict[str, float]:
        """Compute PerspectiveOffCenterFrustum bounds for Cesium (WP02 §2.4)."""
        cx_prime = self.cx - float(u0)
        cy_prime = self.cy - float(v0)
        left = near * (-0.5 - cx_prime) / self.fx
        right = near * ((roi_w - 0.5) - cx_prime) / self.fx
        top = near * (cy_prime + 0.5) / self.fy
        bottom = -near * ((roi_h - 0.5) - cy_prime) / self.fy

        return {
            "near": near,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "cx_prime": cx_prime,
            "cy_prime": cy_prime,
        }

    def compute_poses(
        self,
        lat_deg: float,
        lon_deg: float,
        h_m: float,
        yaw_rad: float,
        pitch_rad: float,
        roll_rad: float,
    ) -> dict[str, dict[str, Any]]:
        """Compute ECEF camera poses for all 8 ring cameras."""
        p_arp_ecef = wgs84.geodetic_to_ecef(lat_deg, lon_deg, h_m)
        r_ecef_ned = wgs84.r_ecef_ned(lat_deg, lon_deg)
        r_ned_body = rotations.euler_to_r(yaw_rad, pitch_rad, roll_rad)
        r_ecef_body = r_ecef_ned @ r_ned_body

        poses = {}
        for c in self.cameras:
            p_cam_ecef = p_arp_ecef + r_ecef_body @ c.lever_arm_m
            r_ecef_cam = r_ecef_body @ c.r_body_cam

            pos = p_cam_ecef.tolist()
            dir_vec = r_ecef_cam[:, 2].tolist()
            up_vec = (-r_ecef_cam[:, 1]).tolist()
            right_vec = r_ecef_cam[:, 0].tolist()

            poses[c.id] = {
                "position": pos,
                "direction": dir_vec,
                "up": up_vec,
                "right": right_vec,
                "r_ecef_cam": r_ecef_cam.tolist(),
                "boresight_body": c.boresight_body.tolist(),
                "lever_arm_m": c.lever_arm_m.tolist(),
            }
        return poses
