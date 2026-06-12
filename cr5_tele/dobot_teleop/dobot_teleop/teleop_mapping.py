#!/usr/bin/env python3
"""
Quest 3 → Dobot pose mapper.

References lerobot_franka_teleop's OculusRobot._compute_delta_pose() for:
  - Frame-to-frame delta computation (position + orientation)
  - Coordinate frame transformation (Oculus → Robot)
  - Quaternion → rotation matrix → delta rotvec pipeline
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .quest_udp import QuestPose


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def norm3(v: List[float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


@dataclass
class QuestTeleopConfig:
    """Configuration for Quest→Dobot pose mapping.

    Coordinate Systems:
        Oculus (left-hand): X(right), Y(up), Z(towards user / backward when
                            the controller points forward)
        Dobot:             standard robot base frame

    The pos_transform / rot_transform matrices map oculus deltas to robot
    deltas.  Default matches the original mapping:
        robot_x = -oculus_x
        robot_y = -oculus_z
        robot_z =  oculus_y
    """
    # ---- coordinate transforms (each a 3×3 matrix) -------------------------
    pos_transform: List[List[float]] = field(
        default_factory=lambda: [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    rot_transform: List[List[float]] = field(
        default_factory=lambda: [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    # ---- scaling -----------------------------------------------------------
    position_scale: float = 0.20       # oculus-pos → robot-pos multiplier
    rotation_scale: float = 0.50       # rotvec (rad) multiplier
    rotation_mode: str = "frame_delta" # frame_delta or origin_delta

    # ---- per-channel sign flip (applied after transform + scale) -----------
    channel_signs: List[int] = field(
        default_factory=lambda: [1, 1, 1, 1, 1, 1]
    )

    # ---- step limits -------------------------------------------------------
    target_deadband_mm: float = 2.0
    target_deadband_deg: float = 1.0
    max_step_mm: float = 6.0
    max_step_deg: float = 3.0

    # ---- total-accumulation limits (safety fence) --------------------------
    max_total_translation_mm: float = 120.0
    max_total_rotation_deg: float = 90.0

    # ---- workspace ---------------------------------------------------------
    workspace_min_x_mm: float = -700.0
    workspace_max_x_mm: float = 700.0
    workspace_min_y_mm: float = -700.0
    workspace_max_y_mm: float = 350.0
    workspace_min_z_mm: float = 50.0
    workspace_max_z_mm: float = 800.0


class QuestTeleopMapper:
    """Frame-to-frame delta mapper: Quest 3 → Dobot ServoP target.

    Strategy (same as lerobot_franka_teleop's OculusRobot):
      - While RG is pressed: small position + rotation deltas from the
        previous Quest frame are accumulated.
      - When RG is released: the accumulator freezes and the previous-
        frame reference is cleared so the next grip doesn't cause a jump.
    """

    def __init__(self, config: QuestTeleopConfig):
        self.cfg = config
        self._pos_T = np.array(config.pos_transform, dtype=float)
        self._rot_T = np.array(config.rot_transform, dtype=float)
        self._signs = np.array(config.channel_signs[:6], dtype=float)

        # ---- state reset by reset() ----------------------------------------
        self.quest_origin: Optional[QuestPose] = None
        self.robot_origin: Optional[List[float]] = None
        self._prev_quest_T: Optional[np.ndarray] = None       # 4×4
        self._accum_delta_pos: np.ndarray = np.zeros(3)       # accumulated robot pos delta (mm)
        self._accum_delta_rot: np.ndarray = np.zeros(3)       # accumulated robot rot delta (rad)
        self._last_target: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, quest_pose: QuestPose, robot_pose: List[float]) -> None:
        """Set the alignment origin.  Call once on 'E' / auto-enable."""
        self.quest_origin = quest_pose
        self.robot_origin = list(robot_pose)

        q = [quest_pose.qx, quest_pose.qy, quest_pose.qz, quest_pose.qw]
        self._prev_quest_T = self._quat_pos_to_T(
            [quest_pose.x, quest_pose.y, quest_pose.z], q
        )

        self._accum_delta_pos = np.zeros(3)
        self._accum_delta_rot = np.zeros(3)
        self._last_target = list(robot_pose)

    def target_from_quest(
        self, quest_pose: QuestPose, rg_pressed: bool = False
    ) -> Tuple[List[float], Dict[str, float]]:
        """Compute robot ServoP target from current Quest pose.

        Args:
            quest_pose: Latest decoded Quest packet.
            rg_pressed: Right Grip (deadman).  When **False** the
                        accumulator freezes.

        Returns:
            target: [x, y, z, rx, ry, rz] in mm & degrees.
            info:   Debug dict with delta magnitudes.
        """
        if self.quest_origin is None or self.robot_origin is None:
            raise RuntimeError("QuestTeleopMapper.reset() must be called first")

        delta_pos, frame_delta_rot = self._compute_frame_delta(quest_pose, rg_pressed)

        # ---- scale & sign --------------------------------------------------
        scaled_pos = delta_pos * self.cfg.position_scale * 1000.0  # m → mm
        if self.cfg.rotation_mode == "origin_delta":
            raw_rot = self._compute_origin_delta_rot(quest_pose, rg_pressed)
        else:
            raw_rot = frame_delta_rot
        scaled_rot = raw_rot * self.cfg.rotation_scale

        for i in range(3):
            scaled_pos[i] *= self._signs[i]
            scaled_rot[i] *= self._signs[i + 3]

        # ---- total-motion limits (applied to each step) --------------------
        scaled_pos = self._cap_norm(
            scaled_pos, self.cfg.max_total_translation_mm
        )
        scaled_rot = self._cap_norm(
            scaled_rot, math.radians(self.cfg.max_total_rotation_deg)
        )

        # accumulate position. Rotation can be frame-to-frame accumulated or
        # absolute relative to the alignment origin, depending on mode.
        self._accum_delta_pos += scaled_pos
        if self.cfg.rotation_mode == "origin_delta":
            self._accum_delta_rot = scaled_rot
        else:
            self._accum_delta_rot += scaled_rot

        # ---- build target (correct rotation composition) --------------------
        # target_R = delta_R @ origin_R   (matrix multiply, not Euler add)
        origin_R = self._euler_deg_to_R(self.robot_origin[3:])
        delta_R_mat = R.from_rotvec(self._accum_delta_rot).as_matrix()
        target_R = delta_R_mat @ origin_R
        target_euler = R.from_matrix(target_R).as_euler("XYZ", degrees=True)

        raw_target: List[float] = [
            self.robot_origin[0] + self._accum_delta_pos[0],
            self.robot_origin[1] + self._accum_delta_pos[1],
            self.robot_origin[2] + self._accum_delta_pos[2],
            float(target_euler[0]),
            float(target_euler[1]),
            float(target_euler[2]),
        ]
        self._clamp_workspace(raw_target)

        target = self._deadband_and_step_limit(raw_target)

        info: Dict[str, float] = {
            "delta_pos_mm": float(np.linalg.norm(self._accum_delta_pos)),
            "delta_rot_deg": float(math.degrees(np.linalg.norm(self._accum_delta_rot))),
            "sent_step_mm": float(
                math.sqrt(
                    (target[0] - self._last_target[0]) ** 2
                    + (target[1] - self._last_target[1]) ** 2
                    + (target[2] - self._last_target[2]) ** 2
                )
            ),
        }

        self._last_target = target
        return target, info

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_pos_to_T(pos: List[float], quat: List[float]) -> np.ndarray:
        """4×4 rigid transform from [x,y,z] and [qx,qy,qz,qw]."""
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = R.from_quat(quat).as_matrix()
        return T

    # ---- frame-to-frame delta (lerobot style) -----------------------------

    def _compute_frame_delta(
        self, quest_pose: QuestPose, rg_pressed: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Frame-to-frame oculus delta → robot frame.

        When RG is *released* the previous transform is cleared so that the
        first frame after re-grip produces zero delta (no jump).
        """
        if not rg_pressed:
            self._prev_quest_T = None
            return np.zeros(3), np.zeros(3)

        q = [quest_pose.qx, quest_pose.qy, quest_pose.qz, quest_pose.qw]
        current_T = self._quat_pos_to_T(
            [quest_pose.x, quest_pose.y, quest_pose.z], q
        )

        if self._prev_quest_T is None:
            # first frame after press → seed and return zero
            self._prev_quest_T = current_T.copy()
            return np.zeros(3), np.zeros(3)

        # Position delta (oculus frame, metres)
        oculus_dp = current_T[:3, 3] - self._prev_quest_T[:3, 3]
        robot_dp = self._pos_T @ oculus_dp

        # Rotation delta
        #   delta_rot = current @ prev^T   → rotation from prev to current
        delta_oculus = current_T[:3, :3] @ self._prev_quest_T[:3, :3].T
        oculus_rv = R.from_matrix(delta_oculus).as_rotvec()
        robot_rv = self._rot_T @ oculus_rv

        self._prev_quest_T = current_T.copy()
        return robot_dp, robot_rv

    def _compute_origin_delta_rot(
        self, quest_pose: QuestPose, rg_pressed: bool
    ) -> np.ndarray:
        """Quest orientation delta from alignment origin mapped to robot axes."""
        if not rg_pressed or self.quest_origin is None:
            return np.zeros(3)

        origin_q = [
            self.quest_origin.qx,
            self.quest_origin.qy,
            self.quest_origin.qz,
            self.quest_origin.qw,
        ]
        current_q = [quest_pose.qx, quest_pose.qy, quest_pose.qz, quest_pose.qw]
        origin_R = R.from_quat(origin_q).as_matrix()
        current_R = R.from_quat(current_q).as_matrix()
        delta_oculus = current_R @ origin_R.T
        oculus_rv = R.from_matrix(delta_oculus).as_rotvec()
        return self._rot_T @ oculus_rv

    # ---- utilities --------------------------------------------------------

    @staticmethod
    def _euler_deg_to_R(euler_deg: List[float]) -> np.ndarray:
        """Dobot Euler XYZ (degrees) → 3×3 rotation matrix."""
        return R.from_euler("XYZ", euler_deg, degrees=True).as_matrix()

    @staticmethod
    def _cap_norm(v: np.ndarray, limit: float) -> np.ndarray:
        if limit <= 0.0:
            return v
        norm = np.linalg.norm(v)
        if norm <= limit or norm < 1e-9:
            return v
        return v * (limit / norm)

    def _clamp_workspace(self, pose: List[float]) -> None:
        c = self.cfg
        pose[0] = clamp(pose[0], c.workspace_min_x_mm, c.workspace_max_x_mm)
        pose[1] = clamp(pose[1], c.workspace_min_y_mm, c.workspace_max_y_mm)
        pose[2] = clamp(pose[2], c.workspace_min_z_mm, c.workspace_max_z_mm)

    def _deadband_and_step_limit(
        self, raw_target: List[float]
    ) -> List[float]:
        cur = self._last_target
        pos_dist = norm3([raw_target[i] - cur[i] for i in range(3)])
        rot_dist = norm3([raw_target[i] - cur[i] for i in range(3, 6)])

        db_mm = self.cfg.target_deadband_mm
        db_deg = self.cfg.target_deadband_deg
        if pos_dist < db_mm and rot_dist < db_deg:
            return list(cur)

        tgt = list(cur)
        # position step limit
        ps = self.cfg.max_step_mm
        if pos_dist > ps > 0.0:
            r = ps / pos_dist
            for i in range(3):
                tgt[i] += (raw_target[i] - cur[i]) * r
        else:
            for i in range(3):
                tgt[i] = raw_target[i]

        # rotation step limit
        rs = self.cfg.max_step_deg
        if rot_dist > rs > 0.0:
            r = rs / rot_dist
            for i in range(3, 6):
                tgt[i] += (raw_target[i] - cur[i]) * r
        else:
            for i in range(3, 6):
                tgt[i] = raw_target[i]

        return tgt

    def mapping_text(self) -> str:
        return (
            f"pos_scale={self.cfg.position_scale}, rot_scale={self.cfg.rotation_scale}, "
            f"rot_mode={self.cfg.rotation_mode}, "
            f"signs={self.cfg.channel_signs}"
        )

    # ---- gripper helper ---------------------------------------------------

    @staticmethod
    def gripper_from_trigger(quest_pose: QuestPose) -> float:
        """Extract gripper command [0=open … 1=closed] from right trigger.

        Returns 0.0 (open) when no trigger data is present.
        """
        val = quest_pose.buttons.get("rightTrig", 0.0)
        return clamp(float(val), 0.0, 1.0)


# ---- backward-compatible aliases ----------------------------------------
DirectTeleopConfig = QuestTeleopConfig
DirectTeleopMapper = QuestTeleopMapper
