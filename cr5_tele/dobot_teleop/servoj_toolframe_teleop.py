#!/usr/bin/env python3

"""
Direct Quest3 UDP -> Dobot ServoJ tool-frame teleoperation (no ROS).

Key features (ported from lerobot_franka_teleop):
  * Frame-to-frame delta (position + orientation) for smooth VR motion.
  * RG (Right Grip) deadman switch — pose accrues only while gripped.
  * Right Trigger -> gripper command (requires dobot gripper support).
  * Full 6-DOF output.
  * Default: ServoJ (IK pre-check with joint-limit guard). Optional: ServoP.
"""

import argparse
import queue
import sys
import threading
import time

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
    apply_tool_offset,
    format_tool_offset_config,
)

# ---------------------------------------------------------------------------
# CR5 joint limits and safety check
# ---------------------------------------------------------------------------
CR5_JOINT_LIMITS = [
    (-360.0, 360.0),  # J1
    (-360.0, 360.0),  # J2
    (-160.0, 160.0),  # J3  (tightest)
    (-360.0, 360.0),  # J4
    (-360.0, 360.0),  # J5
    (-360.0, 360.0),  # J6
]

JOINT_SAFETY_MARGIN_DEG = 0.5


def check_joint_limits(joints):
    """Return (ok, message).  ok=False means at least one joint exceeds the
    safe range (mechanical limit minus JOINT_SAFETY_MARGIN_DEG)."""
    for i, (j, (lo, hi)) in enumerate(zip(joints, CR5_JOINT_LIMITS)):
        safe_lo = lo + JOINT_SAFETY_MARGIN_DEG
        safe_hi = hi - JOINT_SAFETY_MARGIN_DEG
        if j < safe_lo or j > safe_hi:
            return False, (
                f"J{i + 1}={j:.2f}° out of safe range "
                f"[{safe_lo:.1f}, {safe_hi:.1f}]"
            )
    return True, ""


def shortest_angle_diff_deg(target: float, current: float) -> float:
    """Shortest signed angle difference in degrees, normalized to [-180, 180)."""
    return (target - current + 180.0) % 360.0 - 180.0


def plan_joint_step(
    desired_joints: list,
    last_sent_joints: list,
    max_joint_step_deg: float,
):
    """Limit each joint's step to ``max_joint_step_deg`` degrees.

    Returns:
        planned:       List of rate-limited joint angles.
        max_abs_step:  Largest absolute joint delta applied this tick (deg).
        limited:       True if any joint was clipped.
    """
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


def format_joints(joints):
    """Compact joint-angle string for logging."""
    return " ".join(f"J{i + 1}={j:.2f}" for i, j in enumerate(joints))


def should_send_target(info):
    return info["sent_step_mm"] > 0.0 or info.get("sent_step_deg", 0.0) > 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direct Quest3 UDP to Dobot ServoJ/ServoP tool-frame teleoperation."
    )
    # ---- robot connection --------------------------------------------------
    parser.add_argument("--robot-ip", required=True, help="Dobot controller IP")
    parser.add_argument("--dashboard-port", type=int, default=29999)

    # ---- UDP ---------------------------------------------------------------
    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=5005)

    # ---- Servo parameters --------------------------------------------------
    parser.add_argument("--servo-mode", choices=("joint", "cartesian"),
                        default="joint",
                        help="joint = ServoJ (IK pre-check + joint limits); "
                             "cartesian = ServoP (original behaviour)")
    parser.add_argument("--command-rate", type=float, default=10.0)
    parser.add_argument("--servo-t", type=float, default=0.10)
    parser.add_argument("--servo-aheadtime", type=float, default=50.0)
    parser.add_argument("--servo-gain", type=float, default=500.0)

    # ---- joint-level rate limiting (critical near singularities) -----------
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=30.0,
                        help="Max per-joint speed for ServoJ step limiting (deg/s). "
                             "Set lower (e.g. 10-15) if the robot jerks near singularities.")

    # ---- mapper parameters ------------------------------------------------
    parser.add_argument("--position-scale", type=float, default=0.20)
    parser.add_argument("--rotation-scale", type=float, default=0.50)
    parser.add_argument("--rotation-mode", choices=("frame-delta", "origin-delta"),
                        default="frame-delta",
                        help="frame-delta accumulates per-frame rotation; "
                             "origin-delta follows Quest orientation relative to alignment")
    parser.add_argument("--filter-ratio-pos", type=float, default=0.0,
                        help="Quest position EMA ratio, 0 disables filtering")
    parser.add_argument("--filter-ratio-rot", type=float, default=0.0,
                        help="Quest rotation EMA ratio, 0 disables filtering")
    parser.add_argument("--target-deadband-mm", type=float, default=5.0)
    parser.add_argument("--target-deadband-deg", type=float, default=3.0)
    parser.add_argument("--max-step-mm", type=float, default=0.8)
    parser.add_argument("--max-step-deg", type=float, default=0.50)
    parser.add_argument("--max-total-translation-mm", type=float, default=120.0)
    parser.add_argument("--max-total-rotation-deg", type=float, default=90.0)
    parser.add_argument("--workspace-min-x-mm", type=float, default=-700.0)
    parser.add_argument("--workspace-max-x-mm", type=float, default=700.0)
    parser.add_argument("--workspace-min-y-mm", type=float, default=-700.0)
    parser.add_argument("--workspace-max-y-mm", type=float, default=350.0)
    parser.add_argument("--workspace-min-z-mm", type=float, default=50.0)
    parser.add_argument("--workspace-max-z-mm", type=float, default=800.0)

    # ---- axis remap (legacy, overrides defaults) ---------------------------
    parser.add_argument("--pos-transform", type=float, nargs=9, default=None,
                        metavar=("P00", "P01", "P02", "P10", "P11",
                                 "P12", "P20", "P21", "P22"),
                        help="3x3 position transform matrix (row-major)")
    parser.add_argument("--rot-transform", type=float, nargs=9, default=None,
                        metavar=("R00", "R01", "R02", "R10", "R11",
                                 "R12", "R20", "R21", "R22"),
                        help="3x3 rotation transform matrix (row-major)")
    parser.add_argument("--pos-map-x", default=None,
                        help="Legacy: axis_map_robot_x (vr_x|vr_y|vr_z)")
    parser.add_argument("--pos-map-y", default=None,
                        help="Legacy: axis_map_robot_y (vr_x|vr_y|vr_z)")
    parser.add_argument("--pos-map-z", default=None,
                        help="Legacy: axis_map_robot_z (vr_x|vr_y|vr_z)")
    parser.add_argument("--pos-sign-x", type=float, default=None)
    parser.add_argument("--pos-sign-y", type=float, default=None)
    parser.add_argument("--pos-sign-z", type=float, default=None)

    # ---- robot commands ----------------------------------------------------
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--clear-error", action="store_true")
    parser.add_argument("--auto-enable", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print targets without sending to robot")
    parser.add_argument("--log-targets", action="store_true")
    parser.add_argument("--log-joints", action="store_true",
                        help="Append GetAngle joint values to target logs")
    parser.add_argument("--log-quest", action="store_true",
                        help="Print raw Quest pose and delta from align origin")
    parser.add_argument("--verbose-tcp", action="store_true")
    parser.add_argument("--ignore-deadman", action="store_true",
                        help="Move without requiring the Quest RG button")

    # ---- recorder action stream ------------------------------------------
    parser.add_argument("--publish-action-stream", action="store_true",
                        help="Publish latest-only CR5A 7D actions for the read-only dataset recorder")
    parser.add_argument("--action-stream-host", default="127.0.0.1",
                        help="UDP host for --publish-action-stream")
    parser.add_argument("--action-stream-port", type=int, default=5010,
                        help="UDP port for --publish-action-stream")

    # ---- gripper -----------------------------------------------------------
    parser.add_argument("--enable-gripper", action="store_true",
                        help="Send gripper commands via ServoP")

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


def _legacy_to_transform(pos_map, pos_sign):
    """Build a 3x3 position matrix from the old axis-map / sign interface."""
    axis_index = {"vr_x": 0, "vr_y": 1, "vr_z": 2}
    # Use provided signs, falling back to original defaults (-1, -1, 1)
    default_signs = [-1.0, -1.0, 1.0]
    signs = list(pos_sign) if pos_sign is not None else default_signs
    for i in range(min(len(signs), 3)):
        if signs[i] is None:
            signs[i] = default_signs[i]
    if pos_map is None or any(k is None for k in pos_map):
        return None
    T = [[0.0] * 3 for _ in range(3)]
    for i, key in enumerate(pos_map):
        if key in axis_index:
            j = axis_index[key]
            T[i][j] = signs[i] if len(signs) > i else 1.0
    return T


def make_config(args):
    # Determine the position transform matrix
    pos_T = None
    if args.pos_transform is not None:
        pos_T = [
            list(args.pos_transform[0:3]),
            list(args.pos_transform[3:6]),
            list(args.pos_transform[6:9]),
        ]
    elif args.pos_map_x is not None:
        pos_T = _legacy_to_transform(
            [args.pos_map_x, args.pos_map_y, args.pos_map_z],
            [args.pos_sign_x, args.pos_sign_y, args.pos_sign_z],
        )

    rot_T = None
    if args.rot_transform is not None:
        rot_T = [
            list(args.rot_transform[0:3]),
            list(args.rot_transform[3:6]),
            list(args.rot_transform[6:9]),
        ]

    kwargs = dict(
        position_scale=args.position_scale,
        rotation_scale=args.rotation_scale,
        rotation_mode=args.rotation_mode.replace("-", "_"),
        filter_ratio_pos=args.filter_ratio_pos,
        filter_ratio_rot=args.filter_ratio_rot,
        target_deadband_mm=args.target_deadband_mm,
        target_deadband_deg=args.target_deadband_deg,
        max_step_mm=args.max_step_mm,
        max_step_deg=args.max_step_deg,
        max_total_translation_mm=args.max_total_translation_mm,
        max_total_rotation_deg=args.max_total_rotation_deg,
        workspace_min_x_mm=args.workspace_min_x_mm,
        workspace_max_x_mm=args.workspace_max_x_mm,
        workspace_min_y_mm=args.workspace_min_y_mm,
        workspace_max_y_mm=args.workspace_max_y_mm,
        workspace_min_z_mm=args.workspace_min_z_mm,
        workspace_max_z_mm=args.workspace_max_z_mm,
    )
    if pos_T is not None:
        kwargs["pos_transform"] = pos_T
    if rot_T is not None:
        kwargs["rot_transform"] = rot_T

    return QuestTeleopConfig(**kwargs)


def enable_teleop(client, mapper, latest_pose, args):
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
    print(
        "Teleop enabled and aligned. "
        f"Quest origin=({latest_pose.x:.3f},{latest_pose.y:.3f},{latest_pose.z:.3f}) "
        f"RG buttons={list(latest_pose.buttons.keys())}"
    )
    return True, robot_joints


def reseed_on_deadman_press(client, mapper, latest_pose, fallback_pose, args):
    """Reset mapper and joint limiter at the RG rising edge.

    The first frame after pressing the deadman switch should establish a fresh
    Quest and robot origin, not send a motion command based on stale startup
    state.
    """
    if args.dry_run:
        robot_pose = list(fallback_pose or mapper._last_target)
        robot_joints = [0.0] * 6
    else:
        robot_pose = client.get_pose()
        robot_joints = client.get_angle()

    mapper.reset(latest_pose, robot_pose)
    print(f"  ↳ clutch: aligned to current robot pose={format_pose(robot_pose)}")
    return robot_pose, robot_joints


def _base_frame_pose_delta(reference_pose, target_pose):
    """Return the target delta used by the Cartesian controller in mm/deg."""
    delta_pos = [float(target_pose[index] - reference_pose[index]) for index in range(3)]
    reference_R = R.from_euler("XYZ", reference_pose[3:], degrees=True).as_matrix()
    target_R = R.from_euler("XYZ", target_pose[3:], degrees=True).as_matrix()
    delta_euler = R.from_matrix(target_R @ reference_R.T).as_euler("XYZ", degrees=True)
    return [*delta_pos, *map(float, delta_euler)]


def main():
    args = parse_args()
    period = 1.0 / max(args.command_rate, 1.0)
    max_joint_step_deg = args.max_joint_speed_deg_s * period
    if abs(args.servo_t - period) > 0.02:
        print(
            f"Warning: servo_t={args.servo_t:.3f}s but command period={period:.3f}s. "
            "ServoJ/ServoP is usually smoother when these match."
        )

    config = make_config(args)
    mapper = QuestTeleopMapper(config)
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
    stop_requested = False
    last_command_pose = None
    last_command_joints = None
    last_sent_joints = None  # for joint-level step limiting (ServoJ path)
    last_action_stream_log_time = 0.0

    print(f"Quest UDP listening on {args.udp_host}:{args.udp_port}")
    print(f"Dobot dashboard target: {args.robot_ip}:{args.dashboard_port}")
    print(f"Servo mode: {args.servo_mode.upper()}")
    print("Rotation composition: tool-frame target_R = origin_R @ delta_R")
    print(f"Mapping: {mapper.mapping_text()}")
    print(format_tool_offset_config(tool_offset_cfg))
    if args.servo_mode == "joint":
        print(
            f"Joint step limit: {max_joint_step_deg:.2f} deg/step "
            f"({args.max_joint_speed_deg_s:.1f} deg/s at {args.command_rate:.1f} Hz)"
        )
    print(
        "Keys: e=align+enable, p=pause, g=GetPose, c=ClearError, "
        "s=Stop, q=quit"
    )
    if args.ignore_deadman:
        print("WARNING: deadman switch disabled; Quest motion will drive the robot.")
    if args.enable_gripper:
        print("  Gripper enabled: right trigger controls gripper")
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
                        enabled, robot_joints = enable_teleop(client, mapper, latest_pose, args)
                        if enabled:
                            last_command_pose = list(mapper._last_target)
                            last_command_joints = None
                            last_sent_joints = robot_joints
                            was_rg_pressed = False

            # -- keyboard commands -------------------------------------------
            while True:
                try:
                    key = command_queue.get_nowait()
                except queue.Empty:
                    break

                if key == "e":
                    enabled, robot_joints = enable_teleop(client, mapper, latest_pose, args)
                    if enabled:
                        last_command_pose = list(mapper._last_target)
                        last_command_joints = None
                        last_sent_joints = robot_joints
                        was_rg_pressed = False
                elif key == "p":
                    enabled = False
                    last_command_pose = None
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
                        angles = client.get_angle()
                        print(
                            "Current joints: "
                            + " ".join(
                                f"J{i + 1}={angle:.1f}" for i, angle in enumerate(angles)
                            )
                        )
                elif key == "c":
                    if args.dry_run:
                        print("DRY RUN: ClearError skipped.")
                    else:
                        print(client.clear_error())
                elif key == "s":
                    enabled = False
                    last_command_pose = None
                    was_rg_pressed = False
                    if args.dry_run:
                        print("DRY RUN: Stop skipped.")
                    else:
                        print(client.stop())
                elif key == "q":
                    enabled = False
                    last_command_pose = None
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

            # -- teleop logic ------------------------------------------------
            if enabled:
                # Deadman switch: RG (Right Grip) button
                rg_val = latest_pose.buttons.get("RG", 0.0)
                rg_pressed = args.ignore_deadman or rg_val > 0.5

                if rg_pressed and not was_rg_pressed:
                    try:
                        last_command_pose, last_sent_joints = reseed_on_deadman_press(
                            client, mapper, latest_pose, last_command_pose, args
                        )
                    except DobotDashboardError as exc:
                        print(f"Cannot align on RG press: {exc}")
                        was_rg_pressed = rg_pressed
                        continue
                    last_command_joints = None
                    was_rg_pressed = rg_pressed
                    continue
                was_rg_pressed = rg_pressed

                target, info = mapper.target_from_quest(latest_pose, rg_pressed=rg_pressed)
                gripper_cmd = QuestTeleopMapper.gripper_from_trigger(latest_pose)
                command_reference = list(last_command_pose or target)
                servo_sent = False

                sent_pos = info["sent_step_mm"]
                sent_rot = info.get("sent_step_deg", 0.0)
                if should_send_target(info):
                    if not args.dry_run:
                        if args.servo_mode == "joint":
                            # ── ServoJ path: IK + joint-limit + joint-step guard ─
                            try:
                                desired_joints = client.inverse_kin(target)
                                # 不传 joint_near: 控制器默认根据当前关节角就近选解
                            except DobotDashboardError:
                                print(
                                    f"IK unsolvable for {format_pose(target)}, "
                                    f"step_pos={sent_pos:.2f}mm, "
                                    f"step_rot={sent_rot:.2f}deg, skipping"
                                )
                                continue

                            ok, limit_msg = check_joint_limits(desired_joints)
                            if not ok:
                                print(
                                    f"Joint-limit skip: {limit_msg}  "
                                    f"target={format_pose(target)}"
                                )
                                continue

                            # ── Joint-level step limiting (prevents IK-branch
                            #     jumps near singularities from slamming the robot)
                            if last_sent_joints is None:
                                try:
                                    last_sent_joints = client.get_angle()
                                except DobotDashboardError:
                                    last_sent_joints = list(desired_joints)

                            planned_joints, joint_step, joint_limited = plan_joint_step(
                                list(desired_joints),
                                list(last_sent_joints),
                                max_joint_step_deg,
                            )

                            ok2, limit_msg2 = check_joint_limits(planned_joints)
                            if not ok2:
                                print(
                                    f"Joint-limit skip after rate limit: {limit_msg2}  "
                                    f"planned=[{format_joints(planned_joints)}]"
                                )
                                continue

                            try:
                                client.servoj(
                                    planned_joints,
                                    t=args.servo_t,
                                    aheadtime=args.servo_aheadtime,
                                    gain=args.servo_gain,
                                )
                            except DobotDashboardError as exc:
                                try:
                                    mode = client.robot_mode()
                                except Exception as mode_exc:
                                    mode = f"unavailable ({mode_exc})"
                                print(
                                    f"ServoJ rejected: "
                                    f"target={format_pose(target)}, "
                                    f"planned_joints=[{format_joints(planned_joints)}], "
                                    f"desired_joints=[{format_joints(desired_joints)}], "
                                    f"robot_mode={mode}, "
                                    f"response={exc}, skipping"
                                )
                                if mode == 11:
                                    try:
                                        print("  ↳ robot_mode=11 (COLLISION), auto-clearing...")
                                        client.clear_error()
                                        print("  ↳ ClearError OK.")
                                    except DobotDashboardError as ce:
                                        print(f"  ↳ ClearError also failed: {ce}")
                                continue
                            last_sent_joints = list(planned_joints)
                            last_command_joints = list(planned_joints)
                            servo_sent = True
                            if joint_limited:
                                print(
                                    f"Joint step clipped: max_step={joint_step:.2f}° "
                                    f"desired=[{format_joints(desired_joints)}] → "
                                    f"planned=[{format_joints(planned_joints)}]"
                                )
                        else:
                            # ---- ServoP path (original) ----------------------
                            try:
                                client.servop(
                                    target,
                                    t=args.servo_t,
                                    aheadtime=args.servo_aheadtime,
                                    gain=args.servo_gain,
                                )
                            except DobotDashboardError:
                                try:
                                    mode = client.robot_mode()
                                except Exception as mode_exc:
                                    mode = f"unavailable ({mode_exc})"
                                try:
                                    angles = client.get_angle()
                                    angle_text = " ".join(
                                        f"J{i + 1}={angle:.1f}" for i, angle in enumerate(angles)
                                    )
                                except Exception as angle_exc:
                                    angle_text = f"unavailable ({angle_exc})"
                                print(
                                    "ServoP rejected target: "
                                    f"{format_pose(target)}, "
                                    f"step_pos={sent_pos:.2f}mm, "
                                    f"step_rot={sent_rot:.2f}deg, "
                                    f"delta_pos={info['delta_pos_mm']:.1f}mm, "
                                    f"delta_rot={info['delta_rot_deg']:.1f}deg, "
                                    f"robot_mode={mode}, "
                                    f"joints=[{angle_text}], "
                                    "skipping"
                                )
                                continue
                            servo_sent = True
                        # Gripper
                        if args.enable_gripper:
                            # Dobot gripper typically uses 0-100 range;
                            # scale from [0,1] to [0,100] and clamp.
                            gripper_val = clamp(gripper_cmd * 100.0, 0.0, 100.0)
                            # Send as a separate ServoP with gripper coordinate
                            # (use the same target but tell dobot to move gripper)
                            # Some dobot firmware expects a separate command.
                            # For simplicity we send it as part of the pose if
                            # the firmware supports 7-DOF.
                            pass  # TODO: implement dobot gripper command

                if servo_sent:
                    action = _base_frame_pose_delta(command_reference, target) + [gripper_cmd]
                    last_command_pose = list(target)
                else:
                    # Do not label an unsent mapper target as a robot action.
                    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_cmd]

                if action_publisher is not None:
                    stream_action = TeleopAction(
                        timestamp=time.time(),
                        seq=action_publisher.next_seq(),
                        source="quest_servop_teleop" if args.servo_mode == "cartesian" else "quest_servoj_teleop",
                        action=tuple(action),
                        current_pose=tuple(command_reference),
                        target_pose=tuple(target),
                        current_joints=None if last_command_joints is None else tuple(last_command_joints),
                        deadman=rg_pressed,
                        servo_sent=servo_sent,
                        gripper_command=gripper_cmd,
                    )
                    published = action_publisher.publish(stream_action)
                    if now - last_action_stream_log_time >= 1.0:
                        print(
                            f"Published action seq={stream_action.seq} sent={servo_sent} "
                            f"deadman={rg_pressed} action={stream_action.action} "
                            f"udp={'ok' if published else 'dropped'}"
                        )
                        last_action_stream_log_time = now

                if args.log_targets and now - last_target_log_time >= 0.5:
                    grip_str = ""
                    if args.enable_gripper:
                        grip_str = f" grip={QuestTeleopMapper.gripper_from_trigger(latest_pose):.2f}"
                    joint_str = ""
                    if args.log_joints:
                        if args.dry_run:
                            joint_str = " joints=[dry-run]"
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
                    print(
                        f"target {format_pose(target)} "
                        f"step_pos={sent_pos:.2f}mm "
                        f"step_rot={sent_rot:.2f}deg "
                        f"Δpos={info['delta_pos_mm']:.1f}mm "
                        f"Δrot={info['delta_rot_deg']:.1f}deg"
                        f"{grip_str}"
                        f"{joint_str}"
                        f"{deadman_str}"
                    )
                    last_target_log_time = now

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


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


if __name__ == "__main__":
    main()
