#!/usr/bin/env python3
"""
Quest3 UDP -> Dobot CR5A tool-frame teleoperation with an online target
governor.

This script keeps the latest-only Quest input model.  Each control tick uses
only receiver.poll_latest(), maps that pose to a raw Cartesian target, then
lets a velocity-limited command target chase the raw target online.
"""

import argparse
import math
import queue
import sys
import threading
import time
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from dobot_teleop.dobot_dashboard import (
    DobotDashboard,
    DobotDashboardError,
    format_pose,
)
from dobot_teleop.quest_udp import QuestUdpReceiver
from dobot_teleop.toolframe_mapping import QuestTeleopConfig, QuestTeleopMapper
from dobot_teleop.teleop_action_stream import TeleopAction, TeleopActionPublisher
from dobot_teleop.transforms import (
    ToolOffsetConfig,
    format_tool_offset_config,
)

from servoj_toolframe_teleop import (
    check_joint_limits,
    format_joints,
    make_config,
    should_send_target,
)


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def norm3(values: Iterable[float]) -> float:
    values = list(values)
    return math.sqrt(values[0] ** 2 + values[1] ** 2 + values[2] ** 2)


def shortest_angle_diff_deg(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def pose_delta_mm_deg(a: List[float], b: List[float]) -> Tuple[float, float]:
    pos_delta = norm3([a[i] - b[i] for i in range(3)])
    R_a = R.from_euler("XYZ", a[3:], degrees=True).as_matrix()
    R_b = R.from_euler("XYZ", b[3:], degrees=True).as_matrix()
    R_err = R_a.T @ R_b
    rot_delta = math.degrees(float(np.linalg.norm(R.from_matrix(R_err).as_rotvec())))
    return pos_delta, rot_delta


def pose_action_delta_mm_deg(reference: List[float], target: List[float]) -> List[float]:
    """Return a base-frame 6D target delta in the recorder's mm/degree units."""
    position = [float(target[index] - reference[index]) for index in range(3)]
    reference_R = R.from_euler("XYZ", reference[3:], degrees=True).as_matrix()
    target_R = R.from_euler("XYZ", target[3:], degrees=True).as_matrix()
    rotation = R.from_matrix(target_R @ reference_R.T).as_euler("XYZ", degrees=True)
    return [*position, *map(float, rotation)]


class PgeModbusGripper:
    """PGE gripper controlled through Dobot tool RS485 / Modbus-RTU."""

    REG_INIT_CMD = 0x0100
    REG_FORCE = 0x0101
    REG_POSITION = 0x0103
    REG_SPEED = 0x0104
    REG_INIT_STATUS = 0x0200
    COMMAND_DELAY_S = 0.08

    def __init__(
        self,
        client: DobotDashboard,
        slave_id: int,
        baud: int,
        parity: str,
        data_bit: int,
        stop_bit: int,
        force: int,
        speed: int,
        open_position: int,
        close_position: int,
        trigger_threshold: float,
        init_timeout_s: float,
    ):
        self.client = client
        self.slave_id = slave_id
        self.baud = baud
        self.parity = parity
        self.data_bit = data_bit
        self.stop_bit = stop_bit
        self.force = force
        self.speed = speed
        self.open_position = open_position
        self.close_position = close_position
        self.trigger_threshold = trigger_threshold
        self.init_timeout_s = init_timeout_s
        self.index: Optional[int] = None
        self.closed: Optional[bool] = None

    def initialize(self, init_value: int = 1) -> None:
        print("Initializing PGE gripper via tool RS485 / Modbus-RTU.")
        self._send("SetToolPower(1)", "SetToolPower")
        time.sleep(0.5)
        self._send("SetToolMode(1,0)", "SetToolMode")
        self._send(
            f'SetTool485({self.baud},"{self.parity}",{self.stop_bit})',
            "SetTool485",
        )
        time.sleep(0.2)
        response = self._send(
            "ModbusRTUCreate("
            f'{self.slave_id},{self.baud},"{self.parity}",'
            f"{self.data_bit},{self.stop_bit})",
            "ModbusRTUCreate",
        )
        values = self.client.values(response)
        if not values:
            raise DobotDashboardError(
                f"Cannot parse ModbusRTUCreate index from response: {response}"
            )
        self.index = int(values[0])
        print(f"Gripper Modbus index={self.index}")

        self.write_u16(self.REG_INIT_CMD, init_value)
        self.wait_initialized()
        self.write_u16(self.REG_FORCE, self.force)
        self.write_u16(self.REG_SPEED, self.speed)
        self.command_closed(False, force=True)

    def close(self) -> None:
        if self.index is not None:
            self._send(f"ModbusClose({self.index})", "ModbusClose", require_ok=False)
            self.index = None

    def update_from_trigger(self, trigger_value: float) -> None:
        want_closed = trigger_value >= self.trigger_threshold
        self.command_closed(want_closed)

    def command_closed(self, closed: bool, force: bool = False) -> None:
        if self.index is None:
            return
        if not force and self.closed == closed:
            return
        position = self.close_position if closed else self.open_position
        self.write_u16(self.REG_POSITION, position)
        self.closed = closed
        state = "closed" if closed else "open"
        print(f"Gripper {state}: trigger position={position}")

    def write_u16(self, addr: int, value: int) -> None:
        if self.index is None:
            raise DobotDashboardError("Gripper Modbus index is not initialized")
        command = f"SetHoldRegs({self.index},{addr},1,{{{value}}},U16)"
        response = ""
        for attempt in range(3):
            response = self.client.command(command)
            if self.client.error_id(response) == 0:
                time.sleep(self.COMMAND_DELAY_S)
                return
            if attempt < 2:
                print(
                    f"WARNING: SetHoldRegs addr={addr} value={value} failed, "
                    f"retrying: {response}"
                )
                time.sleep(0.25)
        self.client.require_ok(response, "SetHoldRegs")

    def read_u16(self, addr: int) -> int:
        if self.index is None:
            raise DobotDashboardError("Gripper Modbus index is not initialized")
        response = self._send(
            f"GetHoldRegs({self.index},{addr},1,U16)",
            "GetHoldRegs",
        )
        values = self.client.values(response)
        if not values:
            raise DobotDashboardError(
                f"GetHoldRegs returned no value for addr={addr}: {response}"
            )
        return int(values[0])

    def wait_initialized(self) -> None:
        deadline = time.monotonic() + self.init_timeout_s
        last_status: Optional[int] = None
        last_error: Optional[Exception] = None
        while time.monotonic() <= deadline:
            try:
                status = self.read_u16(self.REG_INIT_STATUS)
                last_status = status
                last_error = None
                if status == 1:
                    print("Gripper init status=1 (initialized).")
                    return
                if status == 2:
                    print("Gripper init status=2 (initializing).")
                else:
                    print(f"Gripper init status={status}.")
            except DobotDashboardError as exc:
                last_error = exc
                print(f"WARNING: cannot read gripper init status yet: {exc}")
            time.sleep(0.5)

        if last_status is None and last_error is not None:
            print(
                "WARNING: gripper init status was not readable; continuing with "
                f"write-only control after {self.init_timeout_s:.1f}s."
            )
            return
        raise DobotDashboardError(
            f"Gripper initialization timed out after {self.init_timeout_s:.1f}s; "
            f"last status={last_status}"
        )

    def _send(self, command: str, name: str, require_ok: bool = True) -> str:
        response = self.client.command(command)
        if require_ok:
            self.client.require_ok(response, name)
        return response


class RawToolFrameQuestTeleopMapper(QuestTeleopMapper):
    """Reuse the existing tool-frame mapping but return unclipped raw targets.

    Workspace and total-displacement safety limits still live in the parent
    mapper.  Only the old per-tick Cartesian step limiter is bypassed because
    this script applies velocity limiting in CartesianTargetGovernor.
    """

    def _deadband_and_step_limit(self, raw_target: List[float]) -> List[float]:
        return list(raw_target)


class CartesianTargetGovernor:
    """Velocity-limited tracker for Cartesian pose targets."""

    def __init__(
        self,
        max_linear_speed_mm_s: float,
        max_angular_speed_deg_s: float,
    ):
        self.max_linear_speed_mm_s = max_linear_speed_mm_s
        self.max_angular_speed_deg_s = max_angular_speed_deg_s
        self._cmd_pos: Optional[np.ndarray] = None
        self._cmd_R: Optional[np.ndarray] = None

    def reset(self, robot_pose: List[float]) -> None:
        self._cmd_pos = np.array(robot_pose[:3], dtype=float)
        self._cmd_R = R.from_euler("XYZ", robot_pose[3:], degrees=True).as_matrix()

    def current_target(self) -> List[float]:
        if self._cmd_pos is None or self._cmd_R is None:
            raise RuntimeError("CartesianTargetGovernor.reset() must be called first")
        euler = R.from_matrix(self._cmd_R).as_euler("XYZ", degrees=True)
        return [
            float(self._cmd_pos[0]),
            float(self._cmd_pos[1]),
            float(self._cmd_pos[2]),
            float(euler[0]),
            float(euler[1]),
            float(euler[2]),
        ]

    def update(self, raw_target: List[float], dt: float) -> List[float]:
        if self._cmd_pos is None or self._cmd_R is None:
            self.reset(raw_target)
            return self.current_target()

        max_pos_step_mm = max(self.max_linear_speed_mm_s, 0.0) * max(dt, 0.0)
        max_rot_step_rad = math.radians(
            max(self.max_angular_speed_deg_s, 0.0) * max(dt, 0.0)
        )

        raw_pos = np.array(raw_target[:3], dtype=float)
        pos_delta = raw_pos - self._cmd_pos
        pos_norm = float(np.linalg.norm(pos_delta))
        if max_pos_step_mm <= 0.0:
            pos_delta = np.zeros(3)
        elif pos_norm > max_pos_step_mm:
            pos_delta *= max_pos_step_mm / pos_norm
        self._cmd_pos = self._cmd_pos + pos_delta

        R_raw = R.from_euler("XYZ", raw_target[3:], degrees=True).as_matrix()
        R_err = self._cmd_R.T @ R_raw
        rotvec = R.from_matrix(R_err).as_rotvec()
        rot_norm = float(np.linalg.norm(rotvec))
        if max_rot_step_rad <= 0.0:
            rotvec = np.zeros(3)
        elif rot_norm > max_rot_step_rad:
            rotvec *= max_rot_step_rad / rot_norm
        self._cmd_R = self._cmd_R @ R.from_rotvec(rotvec).as_matrix()

        return self.current_target()

    def lag_to(self, raw_target: List[float]) -> Tuple[float, float]:
        if self._cmd_pos is None or self._cmd_R is None:
            return 0.0, 0.0
        raw_pos = np.array(raw_target[:3], dtype=float)
        lag_pos = float(np.linalg.norm(raw_pos - self._cmd_pos))
        R_raw = R.from_euler("XYZ", raw_target[3:], degrees=True).as_matrix()
        R_err = self._cmd_R.T @ R_raw
        lag_rot = math.degrees(float(np.linalg.norm(R.from_matrix(R_err).as_rotvec())))
        return lag_pos, lag_rot


class CartesianSafetyEnvelope:
    """Clamp Cartesian targets to the same static safety fence as teleop.

    ``QuestTeleopMapper`` already applies these limits while mapping Quest
    poses.  Other command sources (for example an autonomous policy) do not
    go through that mapper, so this small reusable wrapper keeps the workspace
    and total-displacement checks identical instead of introducing a second,
    unrelated safety policy.

    Poses use the Dobot Dashboard convention: XYZ in mm and Rx/Ry/Rz in
    degrees, with XYZ Euler angle order matching ``QuestTeleopMapper``.
    """

    def __init__(
        self,
        *,
        max_total_translation_mm: float,
        max_total_rotation_deg: float,
        workspace_min_x_mm: float,
        workspace_max_x_mm: float,
        workspace_min_y_mm: float,
        workspace_max_y_mm: float,
        workspace_min_z_mm: float,
        workspace_max_z_mm: float,
    ):
        # Match QuestTeleopMapper._cap_norm(): a non-positive total limit
        # disables that particular origin-relative fence.
        self.max_total_translation_mm = max_total_translation_mm
        self.max_total_rotation_deg = max_total_rotation_deg
        self.workspace_min = np.array(
            [workspace_min_x_mm, workspace_min_y_mm, workspace_min_z_mm],
            dtype=float,
        )
        self.workspace_max = np.array(
            [workspace_max_x_mm, workspace_max_y_mm, workspace_max_z_mm],
            dtype=float,
        )
        if np.any(self.workspace_min > self.workspace_max):
            raise ValueError("workspace minimum must not exceed workspace maximum")
        self._origin_pos: Optional[np.ndarray] = None
        self._origin_R: Optional[np.ndarray] = None

    def reset(self, robot_pose: List[float]) -> None:
        if len(robot_pose) != 6:
            raise ValueError("robot pose must contain 6 values")
        self._origin_pos = np.asarray(robot_pose[:3], dtype=float)
        self._origin_R = R.from_euler("XYZ", robot_pose[3:], degrees=True).as_matrix()

    def apply(self, raw_target: List[float]) -> List[float]:
        """Return a finite, workspace- and origin-bounded Cartesian target."""
        if self._origin_pos is None or self._origin_R is None:
            raise RuntimeError("CartesianSafetyEnvelope.reset() must be called first")
        if len(raw_target) != 6 or not np.all(np.isfinite(raw_target)):
            raise ValueError("target pose must contain six finite values")

        position = np.asarray(raw_target[:3], dtype=float)
        delta = position - self._origin_pos
        delta_norm = float(np.linalg.norm(delta))
        if (
            self.max_total_translation_mm > 0.0
            and delta_norm > self.max_total_translation_mm
        ):
            delta *= self.max_total_translation_mm / delta_norm
        position = self._origin_pos + delta
        position = np.clip(position, self.workspace_min, self.workspace_max)

        target_R = R.from_euler("XYZ", raw_target[3:], degrees=True).as_matrix()
        relative_rotvec = R.from_matrix(self._origin_R.T @ target_R).as_rotvec()
        relative_angle = float(np.linalg.norm(relative_rotvec))
        max_angle = math.radians(self.max_total_rotation_deg)
        if max_angle > 0.0 and relative_angle > max_angle:
            relative_rotvec *= max_angle / relative_angle
        limited_R = self._origin_R @ R.from_rotvec(relative_rotvec).as_matrix()
        euler = R.from_matrix(limited_R).as_euler("XYZ", degrees=True)
        return [*map(float, position), *map(float, euler)]


class TimingStats:
    def __init__(self):
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.last_ms = 0.0

    def add(self, elapsed_ms: float) -> None:
        self.count += 1
        self.total_ms += elapsed_ms
        self.max_ms = max(self.max_ms, elapsed_ms)
        self.last_ms = elapsed_ms

    @property
    def avg_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count

    def summary(self, name: str) -> str:
        if self.count == 0:
            return f"{name}=n/a"
        return (
            f"{name}=last {self.last_ms:.1f}ms "
            f"avg {self.avg_ms:.1f}ms max {self.max_ms:.1f}ms n={self.count}"
        )

    def reset(self) -> None:
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.last_ms = 0.0


def plan_joint_step(
    desired_joints: List[float],
    last_sent_joints: List[float],
    max_joint_step_deg: float,
) -> Tuple[List[float], float, bool]:
    planned = []
    max_abs_step = 0.0
    limited = False

    for desired, current in zip(desired_joints, last_sent_joints):
        delta = shortest_angle_diff_deg(desired, current)
        clipped = clamp(delta, -max_joint_step_deg, max_joint_step_deg)
        if abs(clipped - delta) > 1e-9:
            limited = True
        max_abs_step = max(max_abs_step, abs(clipped))
        planned.append(current + clipped)

    return planned, max_abs_step, limited


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direct Quest3 UDP to Dobot ServoJ/ServoP tool-frame teleop with target governor."
    )
    parser.add_argument("--robot-ip", required=True, help="Dobot controller IP")
    parser.add_argument("--dashboard-port", type=int, default=29999)

    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=5005)

    parser.add_argument("--servo-mode", choices=("joint", "cartesian"),
                        default="joint",
                        help="joint = ServoJ (IK pre-check + joint limits); "
                             "cartesian = ServoP")
    parser.add_argument("--command-rate", type=float, default=10.0)
    parser.add_argument("--servo-t", type=float, default=0.10)
    parser.add_argument("--servo-aheadtime", type=float, default=50.0)
    parser.add_argument("--servo-gain", type=float, default=500.0)

    parser.add_argument("--max-linear-speed-mm-s", type=float, default=30.0)
    parser.add_argument("--max-angular-speed-deg-s", type=float, default=15.0)
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=30.0)

    parser.add_argument("--position-scale", type=float, default=0.20)
    parser.add_argument("--rotation-scale", type=float, default=0.50)
    parser.add_argument("--rotation-mode", choices=("frame-delta", "origin-delta"),
                        default="frame-delta")
    parser.add_argument("--filter-ratio-pos", type=float, default=0.0)
    parser.add_argument("--filter-ratio-rot", type=float, default=0.0)
    parser.add_argument("--target-deadband-mm", type=float, default=5.0)
    parser.add_argument("--target-deadband-deg", type=float, default=3.0)
    parser.add_argument("--max-step-mm", type=float, default=0.8)
    parser.add_argument("--max-step-deg", type=float, default=0.50)
    parser.add_argument("--max-total-translation-mm", type=float, default=500.0)
    parser.add_argument("--max-total-rotation-deg", type=float, default=90.0)
    parser.add_argument("--workspace-min-x-mm", type=float, default=-700.0)
    parser.add_argument("--workspace-max-x-mm", type=float, default=700.0)
    parser.add_argument("--workspace-min-y-mm", type=float, default=-1000.0)
    parser.add_argument("--workspace-max-y-mm", type=float, default=350.0)
    parser.add_argument("--workspace-min-z-mm", type=float, default=50.0)
    parser.add_argument("--workspace-max-z-mm", type=float, default=800.0)

    parser.add_argument("--pos-transform", type=float, nargs=9, default=None,
                        metavar=("P00", "P01", "P02", "P10", "P11",
                                 "P12", "P20", "P21", "P22"))
    parser.add_argument("--rot-transform", type=float, nargs=9, default=None,
                        metavar=("R00", "R01", "R02", "R10", "R11",
                                 "R12", "R20", "R21", "R22"))
    parser.add_argument("--pos-map-x", default=None)
    parser.add_argument("--pos-map-y", default=None)
    parser.add_argument("--pos-map-z", default=None)
    parser.add_argument("--pos-sign-x", type=float, default=None)
    parser.add_argument("--pos-sign-y", type=float, default=None)
    parser.add_argument("--pos-sign-z", type=float, default=None)

    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--clear-error", action="store_true")
    parser.add_argument("--auto-enable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-targets", action="store_true")
    parser.add_argument("--log-joints", action="store_true")
    parser.add_argument("--log-timing", action="store_true",
                        help="Print InverseKin and ServoJ/ServoP TCP timing")
    parser.add_argument("--timing-log-interval", type=float, default=2.0,
                        help="Seconds between timing summary logs")
    parser.add_argument("--log-quest", action="store_true")
    parser.add_argument("--verbose-tcp", action="store_true")
    parser.add_argument("--ignore-deadman", action="store_true")
    parser.add_argument("--publish-action-stream", action="store_true",
                        help="Publish latest-only CR5A 7D actions for the read-only PI0 recorder")
    parser.add_argument("--action-stream-host", default="127.0.0.1",
                        help="UDP host for --publish-action-stream")
    parser.add_argument("--action-stream-port", type=int, default=5010,
                        help="UDP port for --publish-action-stream")
    parser.add_argument("--collision-level", type=int, default=1,
                        help="Dobot collision detection sensitivity: 0=off, 1~5 (higher=more sensitive). "
                             "Default 1 for safe teleop. Set 0 to disable temporarily for debugging.")
    parser.add_argument("--enable-gripper", action="store_true")
    parser.add_argument("--gripper-slave-id", type=int, default=1)
    parser.add_argument("--gripper-baud", type=int, default=115200)
    parser.add_argument("--gripper-parity", choices=("N", "E", "O"), default="N")
    parser.add_argument("--gripper-data-bit", type=int, default=8)
    parser.add_argument("--gripper-stop-bit", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gripper-force", type=int, default=50)
    parser.add_argument("--gripper-speed", type=int, default=50)
    parser.add_argument("--gripper-open-position", type=int, default=1000)
    parser.add_argument("--gripper-close-position", type=int, default=0)
    parser.add_argument("--gripper-trigger-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-init-value", type=lambda value: int(value, 0),
                        default=1,
                        help="PGE init value: 1=single direction, 0xA5=full init")
    parser.add_argument("--gripper-init-timeout", type=float, default=8.0,
                        help="Seconds to wait for PGE init status register 0x0200")

    # ── Tool offset (for action stream metadata) ──────────────────────────
    parser.add_argument("--use-gripper-center-pose", action="store_true", default=True,
                        help="Record gripper-center poses in action stream metadata")
    parser.add_argument("--no-use-gripper-center-pose", action="store_false",
                        dest="use_gripper_center_pose")
    parser.add_argument("--tool-offset-x-mm", type=float, default=0.0)
    parser.add_argument("--tool-offset-y-mm", type=float, default=0.0)
    parser.add_argument("--tool-offset-z-mm", type=float, default=160.0)
    parser.add_argument("--tool-offset-rx-deg", type=float, default=0.0)
    parser.add_argument("--tool-offset-ry-deg", type=float, default=0.0)
    parser.add_argument("--tool-offset-rz-deg", type=float, default=0.0)
    parser.add_argument("--controller-tool-offset-already-set", action="store_true")

    return parser.parse_args()


def start_keyboard_thread(command_queue):
    def worker():
        while True:
            line = sys.stdin.readline()
            if line == "":
                return
            command_queue.put(line.strip().lower())

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def enable_teleop(client, mapper, governor, latest_pose, args):
    if latest_pose is None:
        print("Cannot enable: no Quest UDP pose received yet.")
        return False, None

    if args.dry_run:
        robot_pose = [0.0, 0.0, 300.0, -180.0, 0.0, 90.0]
        robot_joints = [0.0] * 6
        print(f"DRY RUN robot origin: {format_pose(robot_pose)}")
    else:
        robot_pose = client.get_pose()
        robot_joints = client.get_angle()
        print(f"Robot origin: {format_pose(robot_pose)}")
        print(f"Robot joints: {format_joints(robot_joints)}")

    workspace_checks = [
        ("X", robot_pose[0], args.workspace_min_x_mm, args.workspace_max_x_mm),
        ("Y", robot_pose[1], args.workspace_min_y_mm, args.workspace_max_y_mm),
        ("Z", robot_pose[2], args.workspace_min_z_mm, args.workspace_max_z_mm),
    ]
    outside = [
        f"{name}={value:.1f} not in [{lower:.1f}, {upper:.1f}]"
        for name, value, lower, upper in workspace_checks
        if value < lower or value > upper
    ]
    if outside:
        print(
            "Cannot enable: robot origin is outside configured workspace: "
            + "; ".join(outside)
        )
        print("Move the robot inside the workspace or loosen the workspace limits.")
        return False, None

    mapper.reset(latest_pose, robot_pose)
    governor.reset(robot_pose)
    print(
        "Teleop enabled and aligned. "
        f"Quest origin=({latest_pose.x:.3f},{latest_pose.y:.3f},{latest_pose.z:.3f}) "
        f"RG buttons={list(latest_pose.buttons.keys())}"
    )
    return True, robot_joints


def reseed_on_deadman_press(client, mapper, governor, latest_pose, args):
    """Align all motion state to the robot at the RG rising edge.

    This avoids the first command after a fresh grip from containing stale
    Quest, governor, or joint-limiter state gathered before the operator was
    actually holding the deadman switch.
    """
    if args.dry_run:
        robot_pose = governor.current_target()
        robot_joints = [0.0] * 6
    else:
        robot_pose = client.get_pose()
        robot_joints = client.get_angle()

    mapper.reset(latest_pose, robot_pose)
    governor.reset(robot_pose)
    print(f"  ↳ clutch: aligned to current robot pose={format_pose(robot_pose)}")
    return robot_joints


def main():
    args = parse_args()
    if not 20 <= args.gripper_force <= 100:
        raise ValueError("--gripper-force must be in [20, 100]")
    if not 1 <= args.gripper_speed <= 100:
        raise ValueError("--gripper-speed must be in [1, 100]")
    if not 0 <= args.gripper_open_position <= 1000:
        raise ValueError("--gripper-open-position must be in [0, 1000]")
    if not 0 <= args.gripper_close_position <= 1000:
        raise ValueError("--gripper-close-position must be in [0, 1000]")
    if args.gripper_init_value not in (1, 0xA5):
        raise ValueError("--gripper-init-value must be 1 or 0xA5")
    if args.gripper_init_timeout <= 0.0:
        raise ValueError("--gripper-init-timeout must be positive")

    period = 1.0 / max(args.command_rate, 1.0)
    max_pos_step_mm = args.max_linear_speed_mm_s * period
    max_rot_step_deg = args.max_angular_speed_deg_s * period
    max_joint_step_deg = args.max_joint_speed_deg_s * period

    if abs(args.servo_t - period) > 0.02:
        print(
            f"Warning: servo_t={args.servo_t:.3f}s but command period={period:.3f}s. "
            "ServoJ/ServoP is usually smoother when these match."
        )

    config: QuestTeleopConfig = make_config(args)
    mapper = RawToolFrameQuestTeleopMapper(config)
    governor = CartesianTargetGovernor(
        args.max_linear_speed_mm_s,
        args.max_angular_speed_deg_s,
    )
    receiver = QuestUdpReceiver(args.udp_host, args.udp_port)
    client = DobotDashboard(
        args.robot_ip,
        args.dashboard_port,
        timeout=0.6,
        verbose=args.verbose_tcp,
    )
    action_publisher = (
        TeleopActionPublisher(args.action_stream_host, args.action_stream_port)
        if args.publish_action_stream
        else None
    )
    gripper = None

    # ── Tool offset configuration ──────────────────────────────────────────
    tool_offset_cfg = ToolOffsetConfig(
        enabled=args.use_gripper_center_pose,
        xyz_mm=(args.tool_offset_x_mm, args.tool_offset_y_mm, args.tool_offset_z_mm),
        rpy_deg=(args.tool_offset_rx_deg, args.tool_offset_ry_deg, args.tool_offset_rz_deg),
        frame_name="gripper_center",
        controller_tool_offset_already_set=args.controller_tool_offset_already_set,
    )

    command_queue = queue.Queue()
    start_keyboard_thread(command_queue)

    enabled = False
    latest_pose = None
    first_pose_printed = False
    was_rg_pressed = False
    last_status_time = time.monotonic()
    last_target_log_time = 0.0
    last_quest_log_time = 0.0
    last_timing_log_time = 0.0
    stop_requested = False
    last_sent_joints = None
    joint_lag_active = False
    ik_timing = TimingStats()
    servo_timing = TimingStats()
    last_action_stream_log_time = 0.0

    print(f"Quest UDP listening on {args.udp_host}:{args.udp_port}")
    print(f"Dobot dashboard target: {args.robot_ip}:{args.dashboard_port}")
    print(f"Servo mode: {args.servo_mode.upper()}")
    print("Rotation composition: tool-frame target_R = origin_R @ delta_R")
    print(f"Mapping: {mapper.mapping_text()}")
    print(format_tool_offset_config(tool_offset_cfg))
    print(
        "Governor: "
        f"linear={args.max_linear_speed_mm_s:.1f}mm/s "
        f"angular={args.max_angular_speed_deg_s:.1f}deg/s "
        f"joint={args.max_joint_speed_deg_s:.1f}deg/s "
        f"steps=({max_pos_step_mm:.2f}mm, {max_rot_step_deg:.2f}deg, "
        f"{max_joint_step_deg:.2f}deg)"
    )
    print(
        "Keys: e=align+enable, p=pause, g=GetPose, c=ClearError, "
        "s=Stop, q=quit"
    )
    if args.ignore_deadman:
        print("WARNING: deadman switch disabled; Quest motion will drive the robot.")
    if args.enable_gripper:
        print(
            "Gripper enabled: Quest right trigger controls PGE gripper "
            f"(threshold={args.gripper_trigger_threshold:.2f})"
        )
    if action_publisher is not None:
        print(f"Teleop action stream enabled: udp://{args.action_stream_host}:{args.action_stream_port}")

    try:
        if args.dry_run:
            print("DRY RUN: Dobot TCP commands will not be sent.")
        else:
            client.connect()
            print(f"Dobot dashboard connected via {client.backend_name()}.")
            if args.clear_error:
                print(client.clear_error())
            if args.enable_robot:
                print(client.enable_robot())
            if args.collision_level is not None:
                print(client.set_collision_level(args.collision_level))
            if args.enable_gripper:
                gripper = PgeModbusGripper(
                    client=client,
                    slave_id=args.gripper_slave_id,
                    baud=args.gripper_baud,
                    parity=args.gripper_parity,
                    data_bit=args.gripper_data_bit,
                    stop_bit=args.gripper_stop_bit,
                    force=args.gripper_force,
                    speed=args.gripper_speed,
                    open_position=args.gripper_open_position,
                    close_position=args.gripper_close_position,
                    trigger_threshold=args.gripper_trigger_threshold,
                    init_timeout_s=args.gripper_init_timeout,
                )
                gripper.initialize(init_value=args.gripper_init_value)

        next_tick = time.monotonic()
        while not stop_requested:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.005))
                continue
            next_tick += period

            new_pose = receiver.poll_latest()
            if new_pose is not None:
                latest_pose = new_pose
                if not first_pose_printed:
                    first_pose_printed = True
                    print(
                        "First Quest pose received: "
                        f"x={new_pose.x:.3f} y={new_pose.y:.3f} z={new_pose.z:.3f} "
                        f"buttons={list(new_pose.buttons.keys())} "
                        f"from {new_pose.address[0]}:{new_pose.address[1]}"
                    )
                    if args.auto_enable and not enabled:
                        enabled, last_sent_joints = enable_teleop(
                            client, mapper, governor, latest_pose, args
                        )
                        joint_lag_active = False
                        was_rg_pressed = False

            while True:
                try:
                    key = command_queue.get_nowait()
                except queue.Empty:
                    break

                if key == "e":
                    enabled, last_sent_joints = enable_teleop(
                        client, mapper, governor, latest_pose, args
                    )
                    joint_lag_active = False
                    was_rg_pressed = False
                elif key == "p":
                    enabled = False
                    joint_lag_active = False
                    was_rg_pressed = False
                    if args.dry_run:
                        print("Teleop paused. DRY RUN: Stop skipped.")
                    else:
                        print(client.stop())
                        print("Teleop paused.")
                elif key == "g":
                    if args.dry_run:
                        print("DRY RUN: GetPose skipped.")
                    else:
                        print(f"Current pose: {format_pose(client.get_pose())}")
                        print(f"Current joints: {format_joints(client.get_angle())}")
                elif key == "c":
                    if args.dry_run:
                        print("DRY RUN: ClearError skipped.")
                    else:
                        print(client.clear_error())
                elif key == "s":
                    enabled = False
                    joint_lag_active = False
                    was_rg_pressed = False
                    if args.dry_run:
                        print("DRY RUN: Stop skipped.")
                    else:
                        print(client.stop())
                elif key == "q":
                    enabled = False
                    was_rg_pressed = False
                    stop_requested = True
                    break
                elif key:
                    print(f"Unknown key: {key}")

            if latest_pose is None:
                if now - last_status_time >= 3.0:
                    print("No Quest UDP pose yet. Start the Quest app and streaming.")
                    last_status_time = now
                continue

            if args.log_quest and now - last_quest_log_time >= 0.5:
                if mapper.quest_origin is None:
                    dx_m = dy_m = dz_m = 0.0
                else:
                    dx_m = latest_pose.x - mapper.quest_origin.x
                    dy_m = latest_pose.y - mapper.quest_origin.y
                    dz_m = latest_pose.z - mapper.quest_origin.z
                buttons_str = ", ".join(
                    f"{k}={v:.2f}" for k, v in latest_pose.buttons.items()
                )
                raw_keys = ",".join(sorted(latest_pose.raw.keys()))
                print(
                    "quest "
                    f"pos=({latest_pose.x:.3f},{latest_pose.y:.3f},{latest_pose.z:.3f})m "
                    f"quat=({latest_pose.qx:.3f},{latest_pose.qy:.3f},"
                    f"{latest_pose.qz:.3f},{latest_pose.qw:.3f}) "
                    f"dpos=({dx_m * 1000.0:.1f},{dy_m * 1000.0:.1f},"
                    f"{dz_m * 1000.0:.1f})mm "
                    f"buttons=[{buttons_str}] "
                    f"raw_keys=[{raw_keys}]"
                )
                last_quest_log_time = now

            if enabled:
                rg_val = latest_pose.buttons.get("RG", 0.0)
                rg_pressed = args.ignore_deadman or rg_val > 0.5

                # ── 离合器: RG按下瞬间，消除governor追赶滞后 ──────
                if rg_pressed and not was_rg_pressed:
                    try:
                        last_sent_joints = reseed_on_deadman_press(
                            client, mapper, governor, latest_pose, args
                        )
                    except DobotDashboardError as exc:
                        print(f"Cannot align on RG press: {exc}")
                        was_rg_pressed = rg_pressed
                        continue
                    joint_lag_active = False
                    was_rg_pressed = rg_pressed
                    continue
                was_rg_pressed = rg_pressed

                raw_target, info = mapper.target_from_quest(
                    latest_pose, rg_pressed=rg_pressed
                )
                raw_step_pos = info["sent_step_mm"]
                raw_step_rot = info.get("sent_step_deg", 0.0)

                if rg_pressed:
                    prev_cmd_target = governor.current_target()
                    cmd_target = governor.update(raw_target, period)
                    cmd_lag_pos, cmd_lag_rot = governor.lag_to(raw_target)
                    cmd_step_pos, cmd_step_rot = pose_delta_mm_deg(
                        prev_cmd_target, cmd_target
                    )
                else:
                    prev_cmd_target = governor.current_target()
                    cmd_target = governor.current_target()
                    cmd_lag_pos, cmd_lag_rot = 0.0, 0.0
                    cmd_step_pos, cmd_step_rot = 0.0, 0.0
                    joint_lag_active = False

                raw_changed = rg_pressed and should_send_target(info)
                cmd_moved = rg_pressed and (
                    cmd_step_pos > 1e-3 or cmd_step_rot > 1e-3
                )
                governor_lag_active = (
                    rg_pressed and (cmd_lag_pos > 1e-3 or cmd_lag_rot > 1e-3)
                )
                send_target = (
                    raw_changed or cmd_moved or governor_lag_active or joint_lag_active
                )
                planned_joints = None
                desired_joints = None
                joint_step = 0.0
                ik_ms = None
                servo_ms = None
                servo_sent = False
                stream_current_joints = (
                    None if last_sent_joints is None else list(last_sent_joints)
                )

                if send_target and not args.dry_run:
                    if args.servo_mode == "joint":
                        try:
                            t0 = time.perf_counter()
                            desired_joints = client.inverse_kin(cmd_target)
                            # 不传 joint_near: 控制器默认根据当前关节角就近选解
                            ik_ms = (time.perf_counter() - t0) * 1000.0
                            ik_timing.add(ik_ms)
                        except DobotDashboardError:
                            print(
                                f"IK unsolvable for cmd={format_pose(cmd_target)}, "
                                f"raw={format_pose(raw_target)}, "
                                f"raw_step_pos={raw_step_pos:.2f}mm, "
                                f"raw_step_rot={raw_step_rot:.2f}deg, skipping"
                            )
                            continue

                        ok, limit_msg = check_joint_limits(desired_joints)
                        if not ok:
                            print(
                                f"Joint-limit skip: {limit_msg}  "
                                f"cmd={format_pose(cmd_target)} raw={format_pose(raw_target)}"
                            )
                            continue

                        if last_sent_joints is None:
                            last_sent_joints = client.get_angle()

                        planned_joints, joint_step, joint_lag_active = plan_joint_step(
                            desired_joints,
                            last_sent_joints,
                            max_joint_step_deg,
                        )

                        ok, limit_msg = check_joint_limits(planned_joints)
                        if not ok:
                            print(
                                f"Joint-limit skip after rate limit: {limit_msg}  "
                                f"planned=[{format_joints(planned_joints)}]"
                            )
                            continue

                        try:
                            t0 = time.perf_counter()
                            client.servoj(
                                planned_joints,
                                t=args.servo_t,
                                aheadtime=args.servo_aheadtime,
                                gain=args.servo_gain,
                            )
                            servo_ms = (time.perf_counter() - t0) * 1000.0
                            servo_timing.add(servo_ms)
                            last_sent_joints = planned_joints
                            servo_sent = True
                        except DobotDashboardError as exc:
                            try:
                                mode = client.robot_mode()
                            except Exception as mode_exc:
                                mode = f"unavailable ({mode_exc})"
                            print(
                                f"ServoJ rejected: "
                                f"cmd={format_pose(cmd_target)}, "
                                f"raw={format_pose(raw_target)}, "
                                f"planned=[{format_joints(planned_joints)}], "
                                f"robot_mode={mode}, response={exc}, skipping"
                            )
                            # Auto-clear collision mode so teleop can continue
                            if mode == 11:
                                try:
                                    print("  ↳ robot_mode=11 (COLLISION), auto-clearing...")
                                    client.clear_error()
                                    print("  ↳ ClearError OK, robot should resume.")
                                except DobotDashboardError as ce:
                                    print(f"  ↳ ClearError also failed: {ce}")
                            continue
                    else:
                        try:
                            t0 = time.perf_counter()
                            client.servop(
                                cmd_target,
                                t=args.servo_t,
                                aheadtime=args.servo_aheadtime,
                                gain=args.servo_gain,
                            )
                            servo_ms = (time.perf_counter() - t0) * 1000.0
                            servo_timing.add(servo_ms)
                            servo_sent = True
                        except DobotDashboardError:
                            try:
                                mode = client.robot_mode()
                            except Exception as mode_exc:
                                mode = f"unavailable ({mode_exc})"
                            try:
                                angles = client.get_angle()
                                angle_text = format_joints(angles)
                            except Exception as angle_exc:
                                angle_text = f"unavailable ({angle_exc})"
                            print(
                                "ServoP rejected target: "
                                f"cmd={format_pose(cmd_target)}, "
                                f"raw={format_pose(raw_target)}, "
                                f"raw_step_pos={raw_step_pos:.2f}mm, "
                                f"raw_step_rot={raw_step_rot:.2f}deg, "
                                f"cmd_lag_pos={cmd_lag_pos:.1f}mm, "
                                f"cmd_lag_rot={cmd_lag_rot:.1f}deg, "
                                f"robot_mode={mode}, joints=[{angle_text}], skipping"
                            )
                            continue

                gripper_cmd = QuestTeleopMapper.gripper_from_trigger(latest_pose)
                if args.enable_gripper:
                    if args.dry_run:
                        pass
                    elif gripper is not None:
                        gripper.update_from_trigger(gripper_cmd)

                if servo_sent:
                    action = pose_action_delta_mm_deg(prev_cmd_target, cmd_target) + [gripper_cmd]
                else:
                    # A mapper target that failed validation or was not sent is
                    # not a valid robot action label.
                    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_cmd]
                if action_publisher is not None:
                    stream_action = TeleopAction(
                        timestamp=time.time(),
                        seq=action_publisher.next_seq(),
                        source="quest_governor_teleop",
                        action=tuple(action),
                        current_pose=tuple(prev_cmd_target),
                        target_pose=tuple(cmd_target),
                        current_joints=(
                            None if stream_current_joints is None else tuple(stream_current_joints)
                        ),
                        deadman=rg_pressed,
                        servo_sent=servo_sent,
                        gripper_command=gripper_cmd,
                    )
                    published = action_publisher.publish(stream_action)
                    if now - last_action_stream_log_time >= 1.0:
                        print(
                            f"Published action seq={stream_action.seq} servo_sent={servo_sent} "
                            f"deadman={rg_pressed} action={stream_action.action} "
                            f"udp={'ok' if published else 'dropped'}"
                        )
                        last_action_stream_log_time = now

                if args.log_targets and now - last_target_log_time >= 0.5:
                    joint_str = ""
                    if args.log_joints:
                        if args.dry_run:
                            joint_str = " joints=[dry-run]"
                        elif planned_joints is not None:
                            joint_str = f" joints=[{format_joints(planned_joints)}]"
                        else:
                            try:
                                angles = client.get_angle()
                                joint_str = f" joints=[{format_joints(angles)}]"
                            except Exception as exc:
                                joint_str = f" joints=[unavailable: {exc}]"
                    deadman_str = (
                        " [NO-DEADMAN]"
                        if args.ignore_deadman
                        else (" [RG]" if rg_pressed else " [--]")
                    )
                    timing_str = ""
                    if args.log_timing:
                        if args.dry_run:
                            timing_str = " ik_ms=dry-run servo_ms=dry-run"
                        else:
                            ik_text = "n/a" if ik_ms is None else f"{ik_ms:.1f}"
                            servo_text = (
                                "n/a" if servo_ms is None else f"{servo_ms:.1f}"
                            )
                            timing_str = (
                                f" ik_ms={ik_text} servo_ms={servo_text}"
                            )
                    print(
                        f"raw {format_pose(raw_target)} | "
                        f"cmd {format_pose(cmd_target)} "
                        f"raw_step_pos={raw_step_pos:.2f}mm "
                        f"raw_step_rot={raw_step_rot:.2f}deg "
                        f"cmd_lag_pos={cmd_lag_pos:.1f}mm "
                        f"cmd_lag_rot={cmd_lag_rot:.1f}deg "
                        f"joint_step={joint_step:.2f}deg"
                        f"{joint_str}"
                        f"{timing_str}"
                        f"{deadman_str}"
                    )
                    last_target_log_time = now

                if (
                    args.log_timing
                    and not args.dry_run
                    and now - last_timing_log_time >= args.timing_log_interval
                    and (ik_timing.count > 0 or servo_timing.count > 0)
                ):
                    print(
                        "timing "
                        f"{ik_timing.summary('IK')} | "
                        f"{servo_timing.summary(args.servo_mode.upper())}"
                    )
                    ik_timing.reset()
                    servo_timing.reset()
                    last_timing_log_time = now

            elif now - last_status_time >= 3.0:
                age = now - latest_pose.timestamp
                buttons_str = ", ".join(
                    f"{k}={v}" for k, v in latest_pose.buttons.items()
                )
                print(
                    f"Quest receiving. count={latest_pose.count}, "
                    f"last=({latest_pose.x:.3f},{latest_pose.y:.3f},{latest_pose.z:.3f}), "
                    f"age={age:.2f}s, buttons=[{buttons_str}]. "
                    "Press e to align+enable."
                )
                last_status_time = now

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except (OSError, DobotDashboardError) as exc:
        print(f"ERROR: {exc}")
    finally:
        if gripper is not None:
            try:
                gripper.close()
            except Exception:
                pass
        if not args.dry_run:
            try:
                client.stop()
            except Exception:
                pass
            client.close()
        receiver.close()
        if action_publisher is not None:
            action_publisher.close()
        print("Direct teleop stopped.")


if __name__ == "__main__":
    main()
