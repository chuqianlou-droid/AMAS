#!/usr/bin/env python3
"""Record read-only CR5A state and dual RealSense RGB into a raw PI0 dataset."""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# 让 scripts/dataset/ 下的脚本能找到 dobot_teleop 包和顶层遥操作模块
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np

from dobot_teleop.cr5a_pi0_schema import CR5A_ACTION_DIM, CR5A_ACTION_KEYS, build_cr5a_observation, validate_cr5a_action
from dobot_teleop.dobot_dashboard import DobotDashboard, DobotDashboardError, format_pose
from dobot_teleop.lerobot_writer import LerobotWriter
from dobot_teleop.realsense_dual_rgb_provider import DualRealSenseRGBProvider
from dobot_teleop.teleop_action_stream import TeleopAction, TeleopActionSubscriber
from dobot_teleop.transforms import (
    ToolOffsetConfig,
    apply_tool_offset,
    format_pose_diff,
    format_tool_offset_config,
    remove_tool_offset,
)


@dataclass(frozen=True)
class ActionSample:
    """A recorder-ready action plus the stream metadata for one observation."""

    action: np.ndarray
    timestamp: float
    age_ms: float
    deadman: bool
    servo_sent: bool
    gripper_command: float
    controller_pose: np.ndarray | None = None
    controller_joints: np.ndarray | None = None
    gripper_center_pose: np.ndarray | None = None
    gripper_center_target: np.ndarray | None = None


def teleop_action_sample(
    action: TeleopAction | None, *, now_s: float, max_action_age_ms: float
) -> tuple[ActionSample | None, str | None]:
    """Turn a latest-only UDP action into a validated sample or a drop reason."""
    if action is None:
        return None, "no_action"
    age_ms = (now_s - action.timestamp) * 1000.0
    if age_ms < -1000.0 or age_ms > max_action_age_ms:
        return None, "stale_action"
    return (
        ActionSample(
            action=validate_cr5a_action(action.action),
            timestamp=action.timestamp,
            age_ms=age_ms,
            deadman=action.deadman,
            servo_sent=action.servo_sent,
            gripper_command=float(np.clip(action.gripper_command, 0.0, 1.0)),
            controller_pose=np.asarray(action.current_pose, dtype=np.float32),
            controller_joints=(
                None
                if action.current_joints is None
                else np.asarray(action.current_joints, dtype=np.float32)
            ),
            gripper_center_pose=(
                None
                if action.gripper_center_pose is None
                else np.asarray(action.gripper_center_pose, dtype=np.float32)
            ),
            gripper_center_target=(
                None
                if action.gripper_center_target is None
                else np.asarray(action.gripper_center_target, dtype=np.float32)
            ),
        ),
        None,
    )


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write PNG frames. Install it with: pip install Pillow") from exc
    return Image


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    Image = _require_pillow()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def _episode_path(root: Path, episode_index: int | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if episode_index is None:
        indices = []
        for path in root.glob("episode_*"):
            try:
                indices.append(int(path.name.rsplit("_", 1)[1]))
            except ValueError:
                continue
        episode_index = max(indices, default=-1) + 1
    path = root / f"episode_{episode_index:06d}"
    if path.exists():
        raise FileExistsError(f"Episode directory already exists: {path}")
    return path


def write_raw_episode(
    episode_dir: Path | str,
    *,
    images_d435: Iterable[np.ndarray],
    images_d415: Iterable[np.ndarray],
    tcp_pose: np.ndarray,
    joints: np.ndarray,
    gripper: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    prompt: str,
    action_timestamps: np.ndarray | None = None,
    action_age_ms: np.ndarray | None = None,
    deadman: np.ndarray | None = None,
    servo_sent: np.ndarray | None = None,
    gripper_action: np.ndarray | None = None,
    metadata: dict | None = None,
) -> Path:
    """Persist one validated raw episode in the documented local format."""
    episode_dir = Path(episode_dir)
    if episode_dir.exists():
        raise FileExistsError(f"Episode directory already exists: {episode_dir}")
    frames_d435, frames_d415 = list(images_d435), list(images_d415)
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
    actions = np.asarray(actions, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    frame_count = len(timestamps)
    action_timestamps = (
        timestamps.copy()
        if action_timestamps is None
        else np.asarray(action_timestamps, dtype=np.float64).reshape(-1)
    )
    action_age_ms = (
        np.zeros(frame_count, dtype=np.float32)
        if action_age_ms is None
        else np.asarray(action_age_ms, dtype=np.float32).reshape(-1)
    )
    deadman = (
        np.zeros(frame_count, dtype=np.bool_)
        if deadman is None
        else np.asarray(deadman, dtype=np.bool_).reshape(-1)
    )
    servo_sent = (
        np.zeros(frame_count, dtype=np.bool_)
        if servo_sent is None
        else np.asarray(servo_sent, dtype=np.bool_).reshape(-1)
    )
    gripper_action = (
        actions[:, 6].astype(np.float32, copy=True)
        if gripper_action is None
        else np.asarray(gripper_action, dtype=np.float32).reshape(-1)
    )
    expected_shapes = {
        "tcp_pose": (frame_count, 6),
        "joints": (frame_count, 6),
        "gripper": (frame_count,),
        "actions": (frame_count, CR5A_ACTION_DIM),
        "action_timestamps": (frame_count,),
        "action_age_ms": (frame_count,),
        "deadman": (frame_count,),
        "servo_sent": (frame_count,),
        "gripper_action": (frame_count,),
    }
    actual_shapes = {
        "tcp_pose": tcp_pose.shape,
        "joints": joints.shape,
        "gripper": gripper.shape,
        "actions": actions.shape,
        "action_timestamps": action_timestamps.shape,
        "action_age_ms": action_age_ms.shape,
        "deadman": deadman.shape,
        "servo_sent": servo_sent.shape,
        "gripper_action": gripper_action.shape,
    }
    if len(frames_d435) != frame_count or len(frames_d415) != frame_count:
        raise ValueError("Image frame counts must match timestamps")
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual_shapes[name]}")
    for name, values in {
        "tcp_pose": tcp_pose,
        "joints": joints,
        "gripper": gripper,
        "actions": actions,
        "action_timestamps": action_timestamps,
        "action_age_ms": action_age_ms,
        "gripper_action": gripper_action,
    }.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or inf")
    for action in actions:
        validate_cr5a_action(action)
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")

    episode_dir.mkdir(parents=True)
    d435_dir = episode_dir / "images_d435"
    d415_dir = episode_dir / "images_d415"
    d435_dir.mkdir()
    d415_dir.mkdir()
    for index, (d435, d415) in enumerate(zip(frames_d435, frames_d415, strict=True)):
        observation = build_cr5a_observation(
            d435, d415, tcp_pose[index], joints[index], float(gripper[index]), prompt
        )
        _write_rgb_png(d435_dir / f"{index:06d}.png", observation["images"]["image_primary"])
        _write_rgb_png(d415_dir / f"{index:06d}.png", observation["images"]["image_secondary"])

    np.savez_compressed(
        episode_dir / "steps.npz",
        tcp_pose=tcp_pose,
        joints=joints,
        gripper=gripper,
        actions=actions,
        timestamps=timestamps,
        action_timestamps=action_timestamps,
        action_age_ms=action_age_ms,
        deadman=deadman,
        servo_sent=servo_sent,
        gripper_action=gripper_action,
    )
    first_image = build_cr5a_observation(frames_d435[0], frames_d415[0], tcp_pose[0])["images"]["image_primary"]
    info = {
        "format": "cr5a_pi0_raw_v1",
        "frame_count": frame_count,
        "prompt": prompt,
        "action_keys": CR5A_ACTION_KEYS,
        "action_units": {"translation": "mm", "rotation": "deg", "gripper": "[0, 1]"},
        "d435_image_shape": list(first_image.shape),
        "d415_image_shape": list(
            build_cr5a_observation(frames_d435[0], frames_d415[0], tcp_pose[0])["images"]["image_secondary"].shape
        ),
        "gripper_note": "TODO: recorder currently stores 0.0 because no CR5A gripper readback is wired.",
        "control_note": "Recorder is read-only; robot motion is controlled only by the Quest teleop process.",
    }
    if metadata:
        info.update(metadata)
    (episode_dir / "metadata.json").write_text(json.dumps(info, indent=2, ensure_ascii=True) + "\n")
    return episode_dir


def _mock_action(mode: str, current_pose: np.ndarray, previous_pose: np.ndarray | None) -> np.ndarray:
    if mode == "zero":
        return np.zeros(CR5A_ACTION_DIM, dtype=np.float32)
    if mode == "tiny_x":
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if mode == "tiny_z":
        return np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if mode == "pose_delta":
        if previous_pose is None:
            return np.zeros(CR5A_ACTION_DIM, dtype=np.float32)
        delta = current_pose - previous_pose
        delta[3:6] = (delta[3:6] + 180.0) % 360.0 - 180.0
        return np.concatenate([delta, np.array([0.0], dtype=np.float32)]).astype(np.float32)
    raise ValueError(f"Unsupported mock action mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--dashboard-port", type=int, default=29999)
    parser.add_argument("--d415-serial", default="021422061498")
    parser.add_argument("--d435-serial", default="801312070525")
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--format", choices=("raw", "lerobot"), default="raw",
                        help="Output format: 'raw' = cr5a_pi0_raw_v1 (legacy), "
                             "'lerobot' = LeRobot v2.1 parquet (directly usable for training)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--record-rate", type=float, default=None, help="Defaults to --camera-fps")
    parser.add_argument("--action-source", choices=("mock", "teleop"), default="mock",
                        help="mock keeps recorder-only format tests; teleop reads the Quest action UDP stream")
    parser.add_argument("--mock-action", choices=("zero", "tiny_x", "tiny_z", "pose_delta"), default="zero")
    parser.add_argument("--teleop-action-host", default="127.0.0.1",
                        help="UDP host for --action-source teleop")
    parser.add_argument("--teleop-action-port", type=int, default=5010,
                        help="UDP port for --action-source teleop")
    parser.add_argument("--teleop-state-source", choices=("stream", "dashboard"), default="stream",
                        help="stream avoids a second Dashboard connection; dashboard reads GetPose/GetAngle directly")
    parser.add_argument("--max-action-age-ms", type=float, default=200.0,
                        help="Treat teleop actions older than this as invalid")
    parser.add_argument("--drop-no-action", action="store_true",
                        help="Skip frames when no fresh teleop action is available")
    parser.add_argument("--record-only-when-deadman", action="store_true",
                        help="For teleop source, save only frames whose action reports deadman=true")
    parser.add_argument("--save-png-frames", action="store_true",
                        help="Also export each frame's d415/d435 images as PNG files for visual inspection")

    # ── Tool offset configuration ─────────────────────────────────────────
    parser.add_argument("--use-gripper-center-pose", action="store_true", default=True,
                        help="Transform TCP pose to gripper-center pose before recording (default)")
    parser.add_argument("--no-use-gripper-center-pose", action="store_false",
                        dest="use_gripper_center_pose",
                        help="Record raw TCP pose without applying tool offset")
    parser.add_argument("--tool-offset-x-mm", type=float, default=0.0,
                        help="Tool offset X in TCP local frame (mm)")
    parser.add_argument("--tool-offset-y-mm", type=float, default=0.0,
                        help="Tool offset Y in TCP local frame (mm)")
    parser.add_argument("--tool-offset-z-mm", type=float, default=160.0,
                        help="Tool offset Z in TCP local frame (mm, default 160)")
    parser.add_argument("--tool-offset-rx-deg", type=float, default=0.0)
    parser.add_argument("--tool-offset-ry-deg", type=float, default=0.0)
    parser.add_argument("--tool-offset-rz-deg", type=float, default=0.0)
    parser.add_argument("--controller-tool-offset-already-set", action="store_true",
                        help="Controller already reports gripper-center; skip software offset")
    parser.add_argument("--log-pose-diff", action="store_true",
                        help="Print raw_tcp vs gripper_center diff at 1 Hz during recording")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_sec <= 0:
        raise SystemExit("--duration-sec must be positive")
    record_rate = float(args.camera_fps if args.record_rate is None else args.record_rate)
    if record_rate <= 0:
        raise SystemExit("--record-rate must be positive")
    if args.max_action_age_ms < 0:
        raise SystemExit("--max-action-age-ms must not be negative")

    # ── Tool offset configuration ──────────────────────────────────────────
    tool_offset_cfg = ToolOffsetConfig(
        enabled=args.use_gripper_center_pose,
        xyz_mm=(args.tool_offset_x_mm, args.tool_offset_y_mm, args.tool_offset_z_mm),
        rpy_deg=(args.tool_offset_rx_deg, args.tool_offset_ry_deg, args.tool_offset_rz_deg),
        frame_name="gripper_center",
        controller_tool_offset_already_set=args.controller_tool_offset_already_set,
    )
    print(format_tool_offset_config(tool_offset_cfg))
    if tool_offset_cfg.controller_tool_offset_already_set:
        print(
            "WARNING: --controller-tool-offset-already-set is ON. "
            "Software offset is SKIPPED; raw TCP pose will be recorded as-is. "
            "Ensure the Dobot controller tool offset is correctly configured "
            "so that GetPose already returns the gripper-center pose."
        )

    # ── LeRobot writer (only for --format lerobot) ──────────────────────────
    lerobot_writer: LerobotWriter | None = None
    if args.format == "lerobot":
        lerobot_writer = LerobotWriter(args.output_dir, fps=int(record_rate))
        if not lerobot_writer.open_dataset():
            lerobot_writer.start_dataset()
            print(f"LeRobot dataset created at {args.output_dir}")
        else:
            print(f"Appending to existing LeRobot dataset at {args.output_dir} "
                  f"(episodes so far: {lerobot_writer._episode_index})")

    episode_dir = _episode_path(args.output_dir, args.episode_index) if args.format == "raw" else None
    # Dobot controllers often allow only one Dashboard client.  During a real
    # demonstration the teleop process owns that connection, so the recorder
    # defaults to the pose/joint reference included in its UDP action stream.
    client = (
        DobotDashboard(args.robot_ip, args.dashboard_port, timeout=0.8)
        if args.action_source == "mock" or args.teleop_state_source == "dashboard"
        else None
    )
    camera = DualRealSenseRGBProvider(
        args.d415_serial, args.d435_serial, args.camera_width, args.camera_height, args.camera_fps
    )
    action_subscriber = (
        TeleopActionSubscriber(args.teleop_action_host, args.teleop_action_port)
        if args.action_source == "teleop"
        else None
    )
    images_d435: list[np.ndarray] = []
    images_d415: list[np.ndarray] = []
    tcp_poses: list[np.ndarray] = []
    gripper_center_poses: list[np.ndarray] = []  # primary training pose
    joints_list: list[np.ndarray] = []
    grippers: list[float] = []
    actions: list[np.ndarray] = []
    timestamps: list[float] = []
    action_timestamps: list[float] = []
    action_ages_ms: list[float] = []
    deadman_states: list[bool] = []
    servo_sent_states: list[bool] = []
    gripper_actions: list[float] = []
    previous_pose: np.ndarray | None = None
    interrupted = False
    failure: Exception | None = None
    frames_dropped_no_action = 0
    frames_dropped_stale_action = 0
    frames_dropped_deadman = 0
    frames_seen = 0
    last_pose_diff_log_time = 0.0

    print("CR5A PI0 recorder: read-only mode. No ServoP, ServoJ, or gripper command will be sent.")
    if action_subscriber is not None:
        print(f"Action source: teleop UDP {args.teleop_action_host}:{args.teleop_action_port}")
        print(f"Teleop state source: {args.teleop_state_source}")
    else:
        print(f"Action source: mock ({args.mock_action})")
    print("Keys: s=stop and save, Ctrl+C=interrupt and save")

    # ── Non-blocking keyboard watcher ─────────────────────────────────────
    stop_requested = False
    def _keyboard_watcher():
        nonlocal stop_requested
        while not stop_requested:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                line = sys.stdin.readline()
                if line.strip().lower() == 's':
                    stop_requested = True
                    break
    kb_thread = threading.Thread(target=_keyboard_watcher, daemon=True)
    kb_thread.start()

    try:
        if client is not None:
            client.connect()
        camera.start()
        deadline = time.monotonic() + args.duration_sec
        period = 1.0 / record_rate
        next_tick = time.monotonic()
        while time.monotonic() < deadline and not stop_requested:
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            next_tick += period
            frames_seen += 1
            sample_now_s = time.time()
            if action_subscriber is not None:
                action_subscriber.poll_latest()
                sample, reason = teleop_action_sample(
                    action_subscriber.latest,
                    now_s=sample_now_s,
                    max_action_age_ms=args.max_action_age_ms,
                )
                if sample is None:
                    if reason == "stale_action":
                        frames_dropped_stale_action += 1
                    else:
                        frames_dropped_no_action += 1
                    if args.drop_no_action:
                        continue
                    sample = ActionSample(
                        action=np.zeros(CR5A_ACTION_DIM, dtype=np.float32),
                        timestamp=sample_now_s,
                        age_ms=-1.0,
                        deadman=False,
                        servo_sent=False,
                        gripper_command=0.0,
                    )
                if args.record_only_when_deadman and not sample.deadman:
                    frames_dropped_deadman += 1
                    continue
            else:
                sample = None
            d435, d415 = camera.get_rgb_images()
            if args.action_source == "teleop" and args.teleop_state_source == "stream":
                if sample.controller_pose is None or sample.controller_joints is None:
                    frames_dropped_no_action += 1
                    continue
                raw_tcp = sample.controller_pose
                joints = sample.controller_joints
            else:
                assert client is not None
                raw_tcp = np.asarray(client.get_pose(), dtype=np.float32)
                joints = np.asarray(client.get_angle(), dtype=np.float32)

            # ── Apply tool offset: TCP → gripper center ──────────────────
            raw_tcp_list = raw_tcp.tolist()
            gripper_center_list, _ = apply_tool_offset(raw_tcp_list, tool_offset_cfg)
            # Primary pose used for dataset: gripper_center.
            # raw_tcp is still recorded for debug.
            pose = np.asarray(gripper_center_list, dtype=np.float32)

            if args.action_source == "mock":
                sample = ActionSample(
                    action=_mock_action(args.mock_action, pose, previous_pose),
                    timestamp=sample_now_s,
                    age_ms=0.0,
                    deadman=False,
                    servo_sent=False,
                    gripper_command=0.0,
                )
            assert sample is not None
            build_cr5a_observation(d435, d415, pose, joints, sample.gripper_command, args.prompt)
            images_d435.append(d435)
            images_d415.append(d415)
            tcp_poses.append(raw_tcp)                          # raw TCP (debug)
            gripper_center_poses.append(pose)                  # gripper_center (primary)
            joints_list.append(joints)
            grippers.append(0.0)  # TODO: add CR5A gripper position readback when the wrapper exposes it.
            actions.append(validate_cr5a_action(sample.action))
            timestamps.append(time.time())
            action_timestamps.append(sample.timestamp)
            action_ages_ms.append(sample.age_ms)
            deadman_states.append(sample.deadman)
            servo_sent_states.append(sample.servo_sent)
            gripper_actions.append(sample.gripper_command)
            previous_pose = pose

            # ── Periodic pose diff logging (1 Hz) ────────────────────────
            now_log = time.monotonic()
            if args.log_pose_diff and tool_offset_cfg.effective() and now_log - last_pose_diff_log_time >= 1.0:
                print(format_pose_diff(raw_tcp_list, gripper_center_list))
                last_pose_diff_log_time = now_log

            if len(timestamps) == 1 or len(timestamps) % 30 == 0:
                print(
                    f"Frame {len(timestamps)} | action_age={sample.age_ms:.1f}ms | "
                    f"deadman={sample.deadman} | servo_sent={sample.servo_sent} | "
                    f"action={np.array2string(sample.action, precision=3)}"
                )
    except KeyboardInterrupt:
        interrupted = True
        stop_requested = True
        print("Recording interrupted (Ctrl+C); saving captured frames.")
    except (OSError, RuntimeError, ValueError, DobotDashboardError) as exc:
        failure = exc
        print(f"Recording stopped due to error: {exc}")
    finally:
        camera.close()
        if client is not None:
            client.close()
        if action_subscriber is not None:
            action_subscriber.close()

    if timestamps:
        if args.format == "lerobot":
            # ── 直接写入 LeRobot 格式 ──────────────────────────────────────
            assert lerobot_writer is not None
            # observation.state = joints(6) + gripper(1) = 7
            # （不包含 tcp_pose，它是 joints 的 FK 结果，action 里也有位姿信息）
            obs_state = np.concatenate(
                [
                    np.stack(joints_list).astype(np.float32),   # (N, 6)
                    np.asarray(grippers, dtype=np.float32).reshape(-1, 1),  # (N, 1)
                ],
                axis=1,
            )  # (N, 7)
            actions_arr = np.stack(actions).astype(np.float32)
            ep_idx = lerobot_writer.add_episode(
                obs_state=obs_state,
                actions=actions_arr,
                d415_images=images_d415,
                d435_images=images_d435,
                prompt=args.prompt,
            )
            print(f"Saved {len(timestamps)} frames → LeRobot episode {ep_idx} in {args.output_dir}")
            # ── 可选: 同时导出 PNG 方便肉眼检查 ──────────────────────────
            if args.save_png_frames:
                from PIL import Image as PILImage
                png_dir = Path(args.output_dir) / f"episode_{ep_idx:06d}_png"
                d415_dir = png_dir / "d415"
                d435_dir = png_dir / "d435"
                d415_dir.mkdir(parents=True, exist_ok=True)
                d435_dir.mkdir(parents=True, exist_ok=True)
                for i, (img415, img435) in enumerate(zip(images_d415, images_d435)):
                    PILImage.fromarray(np.asarray(img415, dtype=np.uint8), mode="RGB").save(d415_dir / f"{i:06d}.png")
                    PILImage.fromarray(np.asarray(img435, dtype=np.uint8), mode="RGB").save(d435_dir / f"{i:06d}.png")
                print(f"  ↳ PNG frames → {png_dir}")
            nonzero_ratio = float(np.mean(np.any(np.abs(actions_arr) > 1e-6, axis=1)))
            print(f"Nonzero action ratio: {nonzero_ratio:.3f}")
        else:
            # ── 原有 raw 格式 ─────────────────────────────────────────────
            assert episode_dir is not None
            write_raw_episode(
                episode_dir,
                images_d435=images_d435,
                images_d415=images_d415,
                tcp_pose=np.stack(gripper_center_poses),  # primary = gripper_center
                joints=np.stack(joints_list),
                gripper=np.asarray(grippers),
                actions=np.stack(actions),
                timestamps=np.asarray(timestamps),
                prompt=args.prompt,
                action_timestamps=np.asarray(action_timestamps),
                action_age_ms=np.asarray(action_ages_ms),
                deadman=np.asarray(deadman_states),
                servo_sent=np.asarray(servo_sent_states),
                gripper_action=np.asarray(gripper_actions),
                metadata={
                    "completed": failure is None and not interrupted,
                    "robot": "dobot_cr5a",
                    "action_source": args.action_source,
                    "action_dim": CR5A_ACTION_DIM,
                    "duration_sec": args.duration_sec,
                    "mock_action": args.mock_action,
                    "d415_serial": args.d415_serial,
                    "d435_serial": args.d435_serial,
                    "record_rate_hz": record_rate,
                    "teleop_action_host": args.teleop_action_host if action_subscriber is not None else None,
                    "teleop_action_port": args.teleop_action_port if action_subscriber is not None else None,
                    "max_action_age_ms": args.max_action_age_ms,
                    "teleop_state_source": args.teleop_state_source if action_subscriber is not None else "dashboard",
                    "pose_frame": tool_offset_cfg.frame_name if tool_offset_cfg.effective() else "tcp",
                    "tool_offset_mm": list(tool_offset_cfg.xyz_mm) if tool_offset_cfg.effective() else None,
                    "state_note": (
                        "Teleop stream state is the controller's last command reference, not an extra GetPose sample."
                        if args.action_source == "teleop" and args.teleop_state_source == "stream"
                        else "State was read directly through Dobot Dashboard GetPose/GetAngle."
                    ),
                },
            )
            print(f"Saved {len(timestamps)} frames to {episode_dir}")
            nonzero_ratio = float(np.mean(np.any(np.abs(np.stack(actions)) > 1e-6, axis=1)))
            print(f"Nonzero action ratio: {nonzero_ratio:.3f}")
    print(f"Frames seen: {frames_seen}")
    print(f"Frames dropped no action: {frames_dropped_no_action}")
    print(f"Frames dropped stale action: {frames_dropped_stale_action}")
    print(f"Frames dropped deadman: {frames_dropped_deadman}")
    if stop_requested:
        print("Recording stopped by user (s key).")
    if failure is not None:
        raise SystemExit(1)
    if not timestamps:
        print("No frames were saved. The recorder remained alive; start teleop or remove a drop filter and retry.")


if __name__ == "__main__":
    main()
