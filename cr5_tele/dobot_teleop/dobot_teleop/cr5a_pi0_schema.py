"""CR5A-specific observation and action conventions for the PI0 pipeline.

This module deliberately describes CR5A data only.  It does not adapt ALOHA
joint actions or reuse any ALOHA naming convention.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


CR5A_IMAGE_KEYS = {
    "d435": "image_primary",
    "d415": "image_secondary",
}
CR5A_STATE_KEYS = {
    "tcp_pose": "state.tcp_pose",
    "gripper_center_pose": "state.gripper_center_pose",
    "joints": "state.joints",
    "gripper": "state.gripper",
}
CR5A_ACTION_DIM = 7
CR5A_ACTION_KEYS = ["dx_mm", "dy_mm", "dz_mm", "dRx_deg", "dRy_deg", "dRz_deg", "gripper"]

ImageLayout = Literal["HWC", "CHW", "auto"]


def _rgb_uint8(image: np.ndarray, *, layout: ImageLayout, name: str) -> np.ndarray:
    """Return an RGB uint8 image in HWC layout, accepting HWC or CHW inputs."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D RGB image, got shape {array.shape}")
    if layout == "auto":
        if array.shape[-1] == 3:
            layout = "HWC"
        elif array.shape[0] == 3:
            layout = "CHW"
        else:
            raise ValueError(f"Cannot infer RGB layout for {name} with shape {array.shape}")
    if layout == "CHW":
        if array.shape[0] != 3:
            raise ValueError(f"{name} declared CHW but has shape {array.shape}")
        array = np.transpose(array, (1, 2, 0))
    elif layout == "HWC":
        if array.shape[-1] != 3:
            raise ValueError(f"{name} declared HWC but has shape {array.shape}")
    else:
        raise ValueError(f"Unsupported image layout {layout!r}")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf")
    if np.issubdtype(array.dtype, np.floating):
        # Camera SDKs commonly expose either [0, 1] or [0, 255] float images.
        scale = 255.0 if float(np.max(array)) <= 1.0 else 1.0
        array = np.rint(array * scale)
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def _vector(value: object, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values, got shape {array.shape}")
    return array


def validate_cr5a_action(action: object) -> np.ndarray:
    """Validate one CR5A Cartesian delta action and return float32 ``(7,)``."""
    return _vector(action, CR5A_ACTION_DIM, "CR5A action")


def build_cr5a_observation(
    image_d435: np.ndarray,
    image_d415: np.ndarray,
    tcp_pose: object,
    joints: object | None = None,
    gripper: float | None = None,
    prompt: str | None = None,
    *,
    image_layout: ImageLayout = "auto",
    gripper_center_pose: object | None = None,
    tcp_pose_raw: object | None = None,
) -> dict:
    """Build the raw CR5A observation schema used by recording and inference.

    Images are RGB ``uint8`` HWC arrays.  OpenPI's stock model later maps
    these friendly names to its fixed image slots and uses a 7D model state
    (Cartesian pose plus gripper); joints remain recorded as CR5A metadata
    because stock Pi0 couples state width to action width.

    Args:
        tcp_pose: Primary Cartesian pose.  When ``use_gripper_center_pose``
            is active, this should be the gripper-center pose.
        gripper_center_pose: If provided, stored separately as
            ``state.gripper_center_pose`` (the training pose).
        tcp_pose_raw: If provided, stored as ``state.tcp_pose_raw``
            (the raw Dobot TCP pose for debug).
    """
    tcp_pose_array = _vector(tcp_pose, 6, "tcp_pose")
    joints_array = None if joints is None else _vector(joints, 6, "joints")
    gripper_value = 0.0 if gripper is None else float(gripper)
    if not np.isfinite(gripper_value):
        raise ValueError("gripper must be finite")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("prompt must be a string or None")

    state: dict = {
        "tcp_pose": tcp_pose_array,
        "joints": joints_array,
        "gripper": gripper_value,
    }
    if gripper_center_pose is not None:
        state["gripper_center_pose"] = _vector(
            gripper_center_pose, 6, "gripper_center_pose"
        )
    if tcp_pose_raw is not None:
        state["tcp_pose_raw"] = _vector(tcp_pose_raw, 6, "tcp_pose_raw")

    return {
        "images": {
            CR5A_IMAGE_KEYS["d435"]: _rgb_uint8(image_d435, layout=image_layout, name="image_d435"),
            CR5A_IMAGE_KEYS["d415"]: _rgb_uint8(image_d415, layout=image_layout, name="image_d415"),
        },
        "state": state,
        "prompt": "" if prompt is None else prompt,
    }
