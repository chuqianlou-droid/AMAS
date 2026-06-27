#!/usr/bin/env python3
"""
Unified tool-offset / coordinate-frame transforms for the CR5A pipeline.

All functions in this module operate on the Dobot-native pose convention:
    pose6d = [X, Y, Z, Rx, Ry, Rz]
where:
  - XYZ are in mm
  - Rx/Ry/Rz are in degrees (intrinsic XYZ Euler angles, matching scipy's "XYZ")

The core offset ``T_tcp_gripper`` shifts along the TCP local Z axis by
``tool_offset_mm`` mm (default 160.0 mm = 16 cm).  Rotation is identity.

Key formulas:

    T_base_gripper = T_base_tcp @ T_tcp_gripper
    T_base_tcp      = T_base_gripper @ inv(T_tcp_gripper)

where T_tcp_gripper = [[I, [0, 0, tool_offset_mm]], [0, 1]].

Do NOT simply do ``Z += 160`` — the offset is in the TCP local frame, not the
world frame.  A rotated TCP will carry the offset off the world Z axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Euler angle order — must match Dobot's convention used everywhere in this
# project (cr5_tele/mapping, bridge, etc.).
# ---------------------------------------------------------------------------
DOBOT_EULER_SEQ = "XYZ"  # scipy intrinsic rotation order for Rx,Ry,Rz

# Default tool offset (TCP → gripper center) in the TCP local frame.
# Units: mm, rotation identity.
DEFAULT_TOOL_OFFSET_MM = (0.0, 0.0, 160.0)
DEFAULT_TOOL_OFFSET_RPY_DEG = (0.0, 0.0, 0.0)


@dataclass
class ToolOffsetConfig:
    """Configuration for the TCP → gripper-center transform.

    Attributes:
        enabled: Whether the software-side offset is applied at all.
        xyz_mm: Translation offset in TCP local frame (mm).
        rpy_deg: Rotation offset in TCP local frame (degrees, XYZ Euler).
        frame_name: Human-readable label for the target frame.
        controller_tool_offset_already_set: If True, the Dobot controller
            already reports the gripper-center pose via GetPose / ToolVectorActual,
            so the software offset is **skipped** (passthrough).
    """

    enabled: bool = True
    xyz_mm: Tuple[float, float, float] = DEFAULT_TOOL_OFFSET_MM
    rpy_deg: Tuple[float, float, float] = DEFAULT_TOOL_OFFSET_RPY_DEG
    frame_name: str = "gripper_center"
    controller_tool_offset_already_set: bool = False

    def effective(self) -> bool:
        """Return True only when the software offset should be applied."""
        return self.enabled and not self.controller_tool_offset_already_set


# ---------------------------------------------------------------------------
# Low-level conversion helpers
# ---------------------------------------------------------------------------


def pose6d_to_matrix(
    pose: List[float],
    *,
    degrees: bool = True,
    xyz_unit: str = "mm",
) -> np.ndarray:
    """Convert a Dobot-style [X, Y, Z, Rx, Ry, Rz] pose to a 4×4 homogeneous matrix.

    Args:
        pose: 6-element pose vector.
        degrees: If True, Rx/Ry/Rz are interpreted as degrees.
        xyz_unit: The spatial unit (``"mm"`` or ``"m"``).  The resulting
            matrix translation is in the **same** unit.

    Returns:
        A float64 (4, 4) homogeneous matrix.
    """
    if len(pose) != 6:
        raise ValueError(f"pose must contain 6 values, got {len(pose)}")

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(pose[:3], dtype=np.float64)
    T[:3, :3] = R.from_euler(
        DOBOT_EULER_SEQ, pose[3:], degrees=degrees
    ).as_matrix()
    return T


def matrix_to_pose6d(
    T: np.ndarray,
    *,
    degrees: bool = True,
    xyz_unit: str = "mm",
) -> List[float]:
    """Convert a 4×4 homogeneous matrix back to a Dobot-style [X,Y,Z,Rx,Ry,Rz] list.

    Args:
        T: (4, 4) homogeneous matrix.
        degrees: If True, output Rx/Ry/Rz are in degrees.
        xyz_unit: The spatial unit of the translation (``"mm"`` or ``"m"``).

    Returns:
        A 6-element list of floats.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must be (4,4), got {T.shape}")
    euler = R.from_matrix(T[:3, :3]).as_euler(DOBOT_EULER_SEQ, degrees=degrees)
    return [
        float(T[0, 3]),
        float(T[1, 3]),
        float(T[2, 3]),
        float(euler[0]),
        float(euler[1]),
        float(euler[2]),
    ]


def _build_offset_matrix(
    xyz_mm: Tuple[float, float, float],
    rpy_deg: Tuple[float, float, float],
) -> np.ndarray:
    """Build the 4×4 T_tcp_gripper matrix from translation + rotation offsets."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(xyz_mm, dtype=np.float64)
    if any(abs(r) > 1e-12 for r in rpy_deg):
        T[:3, :3] = R.from_euler(DOBOT_EULER_SEQ, rpy_deg, degrees=True).as_matrix()
    return T


# ---------------------------------------------------------------------------
# Public API: forward / inverse tool offset
# ---------------------------------------------------------------------------


def tcp_pose_to_gripper_center_pose(
    tcp_pose: List[float],
    offset: Tuple[float, float, float] = DEFAULT_TOOL_OFFSET_MM,
    offset_rpy_deg: Tuple[float, float, float] = DEFAULT_TOOL_OFFSET_RPY_DEG,
    *,
    xyz_unit: str = "mm",
    degrees: bool = True,
) -> List[float]:
    """Convert a base→TCP pose to a base→gripper-center pose.

    Implements:

        T_base_gripper = T_base_tcp @ T_tcp_gripper

    where ``T_tcp_gripper`` is a fixed transform: translate along the TCP
    local Z axis by ``offset[2]`` mm (plus optional rotation).

    Args:
        tcp_pose: [X, Y, Z, Rx, Ry, Rz] in Dobot convention (mm, deg).
        offset: Translation in TCP local frame (mm).  Default (0, 0, 160).
        offset_rpy_deg: Optional rotation offset in TCP local frame (deg).
            Default identity.
        xyz_unit: Spatial unit of *both input and output*.
        degrees: Angle unit of *both input and output*.

    Returns:
        [gripper_X, gripper_Y, gripper_Z, gripper_Rx, gripper_Ry, gripper_Rz]
        in the same units as the input.
    """
    T_base_tcp = pose6d_to_matrix(tcp_pose, degrees=degrees, xyz_unit=xyz_unit)
    T_tcp_gripper = _build_offset_matrix(
        (float(offset[0]), float(offset[1]), float(offset[2])),
        (float(offset_rpy_deg[0]), float(offset_rpy_deg[1]), float(offset_rpy_deg[2])),
    )
    T_base_gripper = T_base_tcp @ T_tcp_gripper
    return matrix_to_pose6d(T_base_gripper, degrees=degrees, xyz_unit=xyz_unit)


def gripper_center_pose_to_tcp_pose(
    gripper_pose: List[float],
    offset: Tuple[float, float, float] = DEFAULT_TOOL_OFFSET_MM,
    offset_rpy_deg: Tuple[float, float, float] = DEFAULT_TOOL_OFFSET_RPY_DEG,
    *,
    xyz_unit: str = "mm",
    degrees: bool = True,
) -> List[float]:
    """Convert a base→gripper-center pose back to a base→TCP pose.

    Implements:

        T_base_tcp = T_base_gripper @ inv(T_tcp_gripper)

    This is the **inverse** of ``tcp_pose_to_gripper_center_pose`` and is
    used at inference time to convert a PI0-predicted gripper-center target
    into a ServoP-ready TCP target.

    Args:
        gripper_pose: [X, Y, Z, Rx, Ry, Rz] in Dobot convention (mm, deg).
        offset: Translation in TCP local frame (mm).  Default (0, 0, 160).
        offset_rpy_deg: Optional rotation offset in TCP local frame (deg).
            Default identity.
        xyz_unit: Spatial unit of *both input and output*.
        degrees: Angle unit of *both input and output*.

    Returns:
        [tcp_X, tcp_Y, tcp_Z, tcp_Rx, tcp_Ry, tcp_Rz]
        in the same units as the input.
    """
    T_base_gripper = pose6d_to_matrix(
        gripper_pose, degrees=degrees, xyz_unit=xyz_unit
    )
    T_tcp_gripper = _build_offset_matrix(
        (float(offset[0]), float(offset[1]), float(offset[2])),
        (float(offset_rpy_deg[0]), float(offset_rpy_deg[1]), float(offset_rpy_deg[2])),
    )
    T_base_tcp = T_base_gripper @ np.linalg.inv(T_tcp_gripper)
    return matrix_to_pose6d(T_base_tcp, degrees=degrees, xyz_unit=xyz_unit)


# ---------------------------------------------------------------------------
# Convenience: apply / remove offset using a config object
# ---------------------------------------------------------------------------


def apply_tool_offset(
    tcp_pose: List[float],
    config: ToolOffsetConfig,
) -> Tuple[List[float], List[float]]:
    """Return ``(gripper_center_pose, tcp_pose)`` after consulting the config.

    If the offset is disabled or the controller already reports the gripper
    center, the returned ``gripper_center_pose`` is identical to ``tcp_pose``.

    Returns:
        (gripper_center_pose, tcp_pose_raw)
    """
    if not config.effective():
        return list(tcp_pose), list(tcp_pose)
    gripper = tcp_pose_to_gripper_center_pose(
        tcp_pose, offset=config.xyz_mm, offset_rpy_deg=config.rpy_deg
    )
    return gripper, list(tcp_pose)


def remove_tool_offset(
    gripper_pose: List[float],
    config: ToolOffsetConfig,
) -> List[float]:
    """Return the TCP pose corresponding to a gripper-center pose.

    If the offset is disabled, the input is returned unchanged.
    """
    if not config.effective():
        return list(gripper_pose)
    return gripper_center_pose_to_tcp_pose(
        gripper_pose, offset=config.xyz_mm, offset_rpy_deg=config.rpy_deg
    )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def format_tool_offset_config(config: ToolOffsetConfig) -> str:
    """One-line summary of the tool offset configuration for startup logs."""
    effective = config.effective()
    return (
        f"[TOOL_OFFSET] enabled: {config.enabled}\n"
        f"[TOOL_OFFSET] tcp -> {config.frame_name} offset: "
        f"{list(config.xyz_mm)} mm, {list(config.rpy_deg)} deg\n"
        f"[TOOL_OFFSET] controller_tool_offset_already_set: "
        f"{config.controller_tool_offset_already_set}\n"
        f"[TOOL_OFFSET] software offset active: {effective}\n"
        f"[TOOL_OFFSET] dataset Cartesian pose frame: "
        f"{config.frame_name if effective else 'tcp'}"
    )


def format_pose_diff(
    raw_tcp: List[float],
    gripper_center: List[float],
) -> str:
    """One-line debug string showing raw TCP, gripper center, and world delta."""
    dx = gripper_center[0] - raw_tcp[0]
    dy = gripper_center[1] - raw_tcp[1]
    dz = gripper_center[2] - raw_tcp[2]
    return (
        f"raw_tcp: X={raw_tcp[0]:.1f} Y={raw_tcp[1]:.1f} Z={raw_tcp[2]:.1f} "
        f"Rx={raw_tcp[3]:.1f} Ry={raw_tcp[4]:.1f} Rz={raw_tcp[5]:.1f} | "
        f"gripper_center: X={gripper_center[0]:.1f} Y={gripper_center[1]:.1f} "
        f"Z={gripper_center[2]:.1f} Rx={gripper_center[3]:.1f} "
        f"Ry={gripper_center[4]:.1f} Rz={gripper_center[5]:.1f} | "
        f"delta_world: dx={dx:.1f} dy={dy:.1f} dz={dz:.1f}"
    )
