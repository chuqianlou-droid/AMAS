#!/usr/bin/env python3
"""
Quest 3 → Dobot pose mapper with tool-frame orientation control.

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


def angle_diff_deg(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def normalize_angle_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


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
    filter_ratio_pos: float = 0.0      # EMA ratio: 0=no filter, 0.95=heavy
    filter_ratio_rot: float = 0.0      # rotation EMA ratio on SO(3)

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


class ToolFrameQuestTeleopMapper:
    """Quest 3 → Dobot target mapper with tool-frame rotation.

    Position mapping is intentionally unchanged from the original mapper.
    Orientation mapping uses rotation-matrix conjugation for Quest→robot axes
    and composes the final orientation in the tool frame:
        target_R = origin_R @ delta_R
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
        self._filtered_pos: Optional[np.ndarray] = None
        self._filtered_R: Optional[R] = None
        self._accum_delta_pos: np.ndarray = np.zeros(3)       # accumulated robot pos delta (mm)
        self._accum_delta_rot: np.ndarray = np.zeros(3)       # accumulated robot rot delta (rad)
        self._origin_delta_base_rot: np.ndarray = np.zeros(3) # rot offset preserved across RG clutch
        self._last_target: Optional[List[float]] = None
        self._last_rg_pressed = False

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
        self._seed_filter(quest_pose)

        self._accum_delta_pos = np.zeros(3)
        self._accum_delta_rot = np.zeros(3)
        self._origin_delta_base_rot = np.zeros(3)
        origin_R = self._euler_deg_to_R(robot_pose[3:])
        canonical_euler = R.from_matrix(origin_R).as_euler("XYZ", degrees=True)
        self._last_target = [
            robot_pose[0],
            robot_pose[1],
            robot_pose[2],
            float(canonical_euler[0]),
            float(canonical_euler[1]),
            float(canonical_euler[2]),
        ]
        self._last_rg_pressed = False

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

        if not rg_pressed:
            self._seed_filter(quest_pose)
            self._prev_quest_T = None
            self._last_rg_pressed = False
            return list(self._last_target), self._info_for_target(self._last_target)

        if not self._last_rg_pressed:
            self._reset_hand_origin_for_clutch(quest_pose)
        self._last_rg_pressed = True

        delta_pos, frame_delta_rot = self._compute_frame_delta(quest_pose, rg_pressed)

        # ---- scale & sign --------------------------------------------------
        scaled_pos = delta_pos * self.cfg.position_scale * 1000.0  # m → mm
        if self.cfg.rotation_mode == "origin_delta":
            raw_rot = self._compute_origin_delta_rot(rg_pressed)
        else:
            raw_rot = frame_delta_rot
        scaled_rot = raw_rot * self.cfg.rotation_scale

        for i in range(3):
            scaled_pos[i] *= self._signs[i]
            scaled_rot[i] *= self._signs[i + 3]

        # accumulate position. Rotation can be frame-to-frame accumulated or
        # absolute relative to the alignment origin, depending on mode.
        self._accum_delta_pos += scaled_pos
        self._accum_delta_pos = self._cap_norm(
            self._accum_delta_pos, self.cfg.max_total_translation_mm
        )
        if self.cfg.rotation_mode == "origin_delta":
            self._accum_delta_rot = self._origin_delta_base_rot + scaled_rot
        else:
            self._accum_delta_rot += scaled_rot
        self._accum_delta_rot = self._cap_norm(
            self._accum_delta_rot, math.radians(self.cfg.max_total_rotation_deg)
        )

        # ---- build target --------------------------------------------------
        # Tool-frame rotation: right-multiply the local delta onto the origin.
        origin_R = self._euler_deg_to_R(self.robot_origin[3:])
        delta_R_mat = R.from_rotvec(self._accum_delta_rot).as_matrix()
        target_R = origin_R @ delta_R_mat
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
        self._sync_accum_pos_from_target(raw_target)

        target = self._deadband_and_step_limit(raw_target)

        info = self._info_for_target(target)

        self._last_target = target
        return target, info

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _reset_hand_origin_for_clutch(self, quest_pose: QuestPose) -> None:
        """Reset Quest hand origin on RG press without resetting robot origin.

        The accumulated robot delta is intentionally preserved so releasing RG,
        moving the controller back to a comfortable pose, and pressing RG again
        behaves like a clutch.
        """
        self.quest_origin = quest_pose
        self._seed_filter(quest_pose)
        q = [quest_pose.qx, quest_pose.qy, quest_pose.qz, quest_pose.qw]
        self._prev_quest_T = self._quat_pos_to_T(
            [quest_pose.x, quest_pose.y, quest_pose.z], q
        )
        self._origin_delta_base_rot = self._accum_delta_rot.copy()

    def sync_accum_to_pose(self, target_pose: List[float]) -> None:
        """Reset accumulated position AND rotation delta so raw_target matches
        ``target_pose``.

        Call this on clutch (RG press) to eliminate governor lag: the raw
        mapper target snaps to the current robot command pose, and the next
        Quest delta starts from there.

        Also syncs ``_last_target`` to the canonical Euler representation
        used by ``target_from_quest``, preventing spurious angular
        differences caused by Euler-angle non-uniqueness
        (e.g. −172° vs 8° for the same rotation matrix).
        """
        if self.robot_origin is None:
            return
        # Position sync
        self._accum_delta_pos = np.array(
            [
                target_pose[0] - self.robot_origin[0],
                target_pose[1] - self.robot_origin[1],
                target_pose[2] - self.robot_origin[2],
            ],
            dtype=float,
        )
        # Rotation sync: compute the rotvec delta between origin_R and target_R.
        # target_R = origin_R @ delta_R  →  delta_R = origin_R^T @ target_R
        origin_R = self._euler_deg_to_R(self.robot_origin[3:])
        target_R = self._euler_deg_to_R(target_pose[3:])
        delta_R = origin_R.T @ target_R
        self._accum_delta_rot = R.from_matrix(delta_R).as_rotvec()
        self._origin_delta_base_rot = self._accum_delta_rot.copy()
        # Sync _last_target to the canonical Euler representation so that
        # _info_for_target's angle_diff_deg won't see a spurious jump.
        canonical_euler = R.from_matrix(origin_R @ R.from_rotvec(self._accum_delta_rot).as_matrix()).as_euler(
            "XYZ", degrees=True
        )
        self._last_target = [
            target_pose[0],
            target_pose[1],
            target_pose[2],
            float(canonical_euler[0]),
            float(canonical_euler[1]),
            float(canonical_euler[2]),
        ]

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
            self._seed_filter(quest_pose)
            self._prev_quest_T = None
            return np.zeros(3), np.zeros(3)

        current_T = self._filtered_quest_T(quest_pose)

        if self._prev_quest_T is None:
            # first frame after press → seed and return zero
            self._prev_quest_T = current_T.copy()
            return np.zeros(3), np.zeros(3)

        # Position delta (oculus frame, metres)
        oculus_dp = current_T[:3, 3] - self._prev_quest_T[:3, 3]
        robot_dp = self._pos_T @ oculus_dp

        # Rotation delta: current @ prev^T maps previous controller frame to
        # current controller frame. Convert axes with matrix conjugation rather
        # than treating the rotvec as a position vector.
        delta_oculus = current_T[:3, :3] @ self._prev_quest_T[:3, :3].T
        delta_robot = self._map_rotation_matrix(delta_oculus)
        robot_rv = R.from_matrix(delta_robot).as_rotvec()

        self._prev_quest_T = current_T.copy()
        return robot_dp, robot_rv

    def _compute_origin_delta_rot(self, rg_pressed: bool) -> np.ndarray:
        """Tool-frame Quest orientation delta from alignment origin."""
        if not rg_pressed or self.quest_origin is None or self._filtered_R is None:
            return np.zeros(3)

        origin_q = [
            self.quest_origin.qx,
            self.quest_origin.qy,
            self.quest_origin.qz,
            self.quest_origin.qw,
        ]
        origin_R = R.from_quat(origin_q).as_matrix()
        current_R = self._filtered_R.as_matrix()
        delta_oculus_local = origin_R.T @ current_R
        delta_robot_local = self._map_rotation_matrix(delta_oculus_local)
        return R.from_matrix(delta_robot_local).as_rotvec()

    def _map_rotation_matrix(self, delta_oculus: np.ndarray) -> np.ndarray:
        return self._rot_T @ delta_oculus @ self._rot_T.T

    def _seed_filter(self, quest_pose: QuestPose) -> None:
        q = [quest_pose.qx, quest_pose.qy, quest_pose.qz, quest_pose.qw]
        self._filtered_pos = np.array(
            [quest_pose.x, quest_pose.y, quest_pose.z], dtype=float
        )
        self._filtered_R = R.from_quat(q)

    def _filtered_quest_T(self, quest_pose: QuestPose) -> np.ndarray:
        current_pos = np.array([quest_pose.x, quest_pose.y, quest_pose.z], dtype=float)
        current_R = R.from_quat(
            [quest_pose.qx, quest_pose.qy, quest_pose.qz, quest_pose.qw]
        )

        pos_alpha = clamp(self.cfg.filter_ratio_pos, 0.0, 0.99)
        rot_alpha = clamp(self.cfg.filter_ratio_rot, 0.0, 0.99)

        if self._filtered_pos is None or self._filtered_R is None:
            self._filtered_pos = current_pos
            self._filtered_R = current_R
        else:
            self._filtered_pos = (
                pos_alpha * self._filtered_pos + (1.0 - pos_alpha) * current_pos
            )
            delta_R = current_R * self._filtered_R.inv()
            self._filtered_R = (
                R.from_rotvec((1.0 - rot_alpha) * delta_R.as_rotvec())
                * self._filtered_R
            )

        T = np.eye(4)
        T[:3, 3] = self._filtered_pos
        T[:3, :3] = self._filtered_R.as_matrix()
        return T

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

    def _sync_accum_pos_from_target(self, pose: List[float]) -> None:
        self._accum_delta_pos = np.array(
            [
                pose[0] - self.robot_origin[0],
                pose[1] - self.robot_origin[1],
                pose[2] - self.robot_origin[2],
            ],
            dtype=float,
        )

    def _deadband_and_step_limit(
        self, raw_target: List[float]
    ) -> List[float]:
        cur = self._last_target
        pos_dist = norm3([raw_target[i] - cur[i] for i in range(3)])
        rot_diffs = [angle_diff_deg(raw_target[i], cur[i]) for i in range(3, 6)]
        rot_dist = norm3(rot_diffs)

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
            for axis, i in enumerate(range(3, 6)):
                tgt[i] = normalize_angle_deg(cur[i] + rot_diffs[axis] * r)
        else:
            for i in range(3, 6):
                tgt[i] = raw_target[i]

        return tgt

    def _info_for_target(self, target: List[float]) -> Dict[str, float]:
        rot_diffs = [
            angle_diff_deg(target[i], self._last_target[i]) for i in range(3, 6)
        ]
        return {
            "delta_pos_mm": float(np.linalg.norm(self._accum_delta_pos)),
            "delta_rot_deg": float(math.degrees(np.linalg.norm(self._accum_delta_rot))),
            "sent_step_mm": float(
                math.sqrt(
                    (target[0] - self._last_target[0]) ** 2
                    + (target[1] - self._last_target[1]) ** 2
                    + (target[2] - self._last_target[2]) ** 2
                )
            ),
            "sent_step_deg": float(norm3(rot_diffs)),
        }

    def mapping_text(self) -> str:
        return (
            f"pos_scale={self.cfg.position_scale}, rot_scale={self.cfg.rotation_scale}, "
            f"rot_mode={self.cfg.rotation_mode}, "
            f"filter=({self.cfg.filter_ratio_pos},{self.cfg.filter_ratio_rot}), "
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
QuestTeleopMapper = ToolFrameQuestTeleopMapper
DirectTeleopConfig = QuestTeleopConfig
DirectTeleopMapper = ToolFrameQuestTeleopMapper
