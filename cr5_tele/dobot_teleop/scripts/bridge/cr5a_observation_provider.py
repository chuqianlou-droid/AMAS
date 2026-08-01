#!/usr/bin/env python3
"""Observation provider for CR5A OpenPI websocket inference.

This module is loaded by ``pi0_cr5a_bridge.py`` with:

    --observation-provider scripts/bridge/cr5a_observation_provider.py:make_observation

It owns the camera pipelines so the bridge can stay focused on safety and
motion execution.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from dobot_teleop.realsense_dual_rgb_provider import DualRealSenseRGBProvider


_CAMERA: DualRealSenseRGBProvider | None = None


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _camera() -> DualRealSenseRGBProvider:
    global _CAMERA
    if _CAMERA is None:
        _CAMERA = DualRealSenseRGBProvider(
            d415_serial=os.environ.get("CR5A_D415_SERIAL", "841612070371"),
            d435_serial=os.environ.get("CR5A_D435_SERIAL", "801312070525"),
            width=_env_int("CR5A_IMAGE_WIDTH", 224),
            height=_env_int("CR5A_IMAGE_HEIGHT", 224),
            fps=_env_int("CR5A_CAMERA_FPS", 15),
        )
        _CAMERA.start()
    return _CAMERA


def _action_from_context(context: dict[str, Any]) -> np.ndarray:
    rtc_prev_actions = context.get("rtc_prev_actions")
    if rtc_prev_actions is None:
        return np.zeros(7, dtype=np.float32)

    actions = np.asarray(rtc_prev_actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"Expected RTC previous actions with shape (horizon, 7), got {actions.shape}")
    return actions


def make_observation(context: dict[str, Any]) -> dict[str, Any]:
    """Return the dict expected by ``LeRobotCR5ADataConfig`` at inference time."""
    d435_rgb, d415_rgb = _camera().get_rgb_images()
    joints = np.asarray(context["cr5_joints_deg"], dtype=np.float32)
    if joints.shape != (6,):
        raise ValueError(f"Expected 6 CR5A joints, got shape {joints.shape}")

    # Training state is [j1..j6, gripper].  If no gripper state feedback is
    # available, keep it open/zero by default and override with env when needed.
    gripper = np.float32(_env_float("CR5A_GRIPPER_STATE", 0.0))
    state = np.concatenate([joints, np.asarray([gripper], dtype=np.float32)])

    action = _action_from_context(context)
    rtc_options = context.get("rtc_options")
    output = {
        # Canonical LeRobot keys consumed by openpi.training.config.LeRobotCR5ADataConfig.
        "observation.images.d435": d435_rgb,
        "observation.images.d415": d415_rgb,
        "observation.state": state,
        "action": action,
        # Legacy flat keys kept for older local experiments that bypass repack_transforms.
        "observation/image": d435_rgb,
        "observation/wrist_image": d415_rgb,
        "observation/state": state,
        "prompt": context.get("instruction", ""),
    }
    if rtc_options is not None:
        output["__rtc"] = dict(rtc_options)
    return output
