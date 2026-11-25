"""Distributed-aperture rig definitions, nominal & true rigs, and pose calculators.

Conforms strictly to docs/xvs/CONVENTIONS.md v1.0 and docs/xvs/spec/rig_nominal.json.
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from xvs.geo import wgs84, rotations
from xvs.rig.camera import PinholeCamera


CAMERA_ORDER = ("F", "R", "B", "L", "U", "D")


@dataclass
class CameraMount:
    id: str
    name: str
    mount: str
    lever_arm_m: np.ndarray  # (3,) in BODY frame
    r_body_cam: np.ndarray   # (3, 3) v_body = R_body_cam @ v_cam
    boresight_body: np.ndarray  # (3,)
    omega_cam_rad: np.ndarray | None = None  # Perturbation if true rig
    delta_lever_arm_m: np.ndarray | None = None  # Perturbation if true rig


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RIG_SPEC = REPO_ROOT / "configs" / "xvs" / "spec" / "rig_nominal.json"


class Rig:
    def __init__(
        self,
        rig_id: str,
        cameras: list[CameraMount],
        is_nominal: bool = True,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        fov_h_deg: float = 100.0,
    ):
        self.rig_id = rig_id
        self.cameras = cameras
        self.is_nominal = is_nominal
        self.seed = seed
        self.width = width
        self.height = height
        self.fov_h_deg = fov_h_deg
        self.fov_rad = math.radians(fov_h_deg)
        self.aspect_ratio = float(width) / float(height)
        self.camera_model = PinholeCamera(width, height, fov_h_deg)
        self.cam_by_id = {cam.id: cam for cam in cameras}
        assert tuple(c.id for c in cameras) == CAMERA_ORDER, f"Camera order must be {CAMERA_ORDER}"

    @classmethod
    def load_nominal(cls, spec_path: str | Path = DEFAULT_RIG_SPEC) -> Rig:
        with open(spec_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cams = []
        for c in data["cameras"]:
            cams.append(
                CameraMount(
                    id=c["id"],
                    name=c["name"],
                    mount=c["mount"],
                    lever_arm_m=np.array(c["lever_arm_m"], dtype=np.float64),
                    r_body_cam=np.array(c["R_body_cam"], dtype=np.float64),
                    boresight_body=np.array(c["boresight_body"], dtype=np.float64),
                )
            )
        width = int(data.get("image", {}).get("width", 1024))  # default-ok: geometry dimension fallback
        height = int(data.get("image", {}).get("height", 1024))  # default-ok: geometry dimension fallback
        fov_h_deg = float(data.get("intrinsics", {}).get("fov_h_deg", 100.0))  # default-ok: geometry dimension fallback
        return cls(
            rig_id=data["rig_id"],
            cameras=cams,
            is_nominal=True,
            width=width,
            height=height,
            fov_h_deg=fov_h_deg,
        )

    def make_true(self, seed: int = 20260922) -> Rig:
        """Create rig_true with seeded installation errors per rig_nominal.json."""
        rng = np.random.default_rng(seed)
        sigma_rot = math.radians(0.20)
        clip_rot = math.radians(0.60)
        sigma_trans = 0.005
        clip_trans = 0.015

        true_cams = []
        for cam_id in CAMERA_ORDER:
            nom_cam = self.cam_by_id[cam_id]
            # Draw order: omega(3) then delta(3)
            omega = rng.normal(0.0, sigma_rot, size=3)
            omega = np.clip(omega, -clip_rot, clip_rot)

            delta = rng.normal(0.0, sigma_trans, size=3)
            delta = np.clip(delta, -clip_trans, clip_trans)

            # R_body_cam_true = R_body_cam_nominal @ Exp(omega)
            r_true = nom_cam.r_body_cam @ rotations.so3_exp(omega)
            lever_arm_true = nom_cam.lever_arm_m + delta
            boresight_true = r_true @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

            true_cams.append(
                CameraMount(
                    id=nom_cam.id,
                    name=nom_cam.name,
                    mount=nom_cam.mount,
                    lever_arm_m=lever_arm_true,
                    r_body_cam=r_true,
                    boresight_body=boresight_true,
                    omega_cam_rad=omega,
                    delta_lever_arm_m=delta,
                )
            )

        return Rig(
            rig_id=f"{self.rig_id}-true-seed{seed}",
            cameras=true_cams,
            is_nominal=False,
            seed=seed,
            width=self.width,
            height=self.height,
            fov_h_deg=self.fov_h_deg,
        )

    def record(self) -> dict[str, Any]:
        """Return the serializable rig descriptor without fallbacks (C16)."""
        return {
            "kind": "nominal" if self.is_nominal else "true",
            "seed": self.seed,
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
        """Compute camera poses in ECEF and Cesium direct vectors for the aircraft state.

        Conforms to CONVENTIONS.md §3, §4, §6.
        """
        r_ecef_ned_mat = wgs84.r_ecef_ned(lat_deg, lon_deg)
        r_ned_body_mat = rotations.euler_to_r(yaw_rad, pitch_rad, roll_rad)
        r_ecef_body_mat = r_ecef_ned_mat @ r_ned_body_mat
        p_ecef_body = wgs84.geodetic_to_ecef(lat_deg, lon_deg, h_m)

        poses = {}
        for cam in self.cameras:
            # R_ecef_cam = R_ecef_body @ R_body_cam
            r_ecef_cam = r_ecef_body_mat @ cam.r_body_cam
            # p_ecef_cam = p_ecef_body + R_ecef_body @ lever_arm_body
            p_ecef_cam = p_ecef_body + r_ecef_body_mat @ cam.lever_arm_m

            # Cesium camera vectors (CONVENTIONS.md §6):
            # position = p_ecef_cam
            # direction = R_ecef_cam[:, 2]
            # up = -R_ecef_cam[:, 1]
            # right = R_ecef_cam[:, 0]
            direction = r_ecef_cam[:, 2]
            up = -r_ecef_cam[:, 1]
            right = r_ecef_cam[:, 0]

            poses[cam.id] = {
                "p_ecef_cam": p_ecef_cam,
                "r_ecef_cam": r_ecef_cam,
                "position": p_ecef_cam,
                "direction": direction,
                "up": up,
                "right": right,
                "fov_rad": self.fov_rad,
                "aspect_ratio": self.aspect_ratio,
                "lever_arm_body": cam.lever_arm_m,
            }
        return poses
