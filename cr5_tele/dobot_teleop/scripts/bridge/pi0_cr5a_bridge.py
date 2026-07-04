#!/usr/bin/env python3
"""Run an OpenPI/PI0 action stream on a Dobot CR5A through ServoP.

This is deliberately a bridge, not a CR5A training configuration.  OpenPI's
``actions`` array is defined by the checkpoint's output transforms, so there
is no universally correct mapping from an OpenPI action to a CR5A pose.  Pick
that mapping explicitly with ``--action-format`` and the scale/frame options
below.  The bridge refuses malformed actions and routes every target through
the existing CartesianSafetyEnvelope and CartesianTargetGovernor before a
ServoP command can be sent.

The normal inference path is an OpenPI websocket policy server.  An observation
provider is a small Python function that owns the checkpoint-specific camera,
state and prompt schema.  See ``--help`` and ``load_observation_provider`` for
its contract.  ``--actions-jsonl`` is useful for validating calibration and
safety in ``--dry-run`` without a policy server or robot.
"""

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

# 让 scripts/bridge/ 下的脚本能找到 dobot_teleop 包和顶层遥操作模块
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
from scipy.spatial.transform import Rotation as R

from dobot_teleop.dobot_dashboard import DobotDashboard, DobotDashboardError, format_pose
from dobot_teleop.transforms import (
    ToolOffsetConfig,
    format_tool_offset_config,
    remove_tool_offset,
)

# These are intentionally imported from the existing governor rather than
# reimplemented here.
from toolframe_governor_teleop import (  # noqa: E402
    CartesianSafetyEnvelope,
    CartesianTargetGovernor,
    PgeModbusGripper,
)
from servoj_toolframe_teleop import (  # noqa: E402
    check_joint_limits,
    format_joints,
    plan_joint_step,
)


Pose = List[float]
ObservationProvider = Callable[[Dict[str, Any]], Dict[str, Any]]
CR5A_ACTION_DIM = 7
ALOHA_ACTION_DIM = 14


def _canonical_action_format(action_format: str) -> str:
    """Return the explicit CR5A mode used for the ServoP safety gate.

    ``delta`` and ``absolute_pose`` are the public CR5A names.  The longer
    names remain accepted for existing calibration commands, but normalize to
    one of these two modes before an action can reach ServoP.
    """
    if action_format in {"delta", "cartesian_delta_mm_deg", "cartesian_delta_m_rad", "normalized_cartesian_delta"}:
        return "delta"
    if action_format in {"absolute_pose", "cartesian_absolute_mm_deg", "cartesian_absolute_m_rad"}:
        return "absolute_pose"
    raise ValueError(f"Unsupported CR5A action format: {action_format}")


def _matrix_from_flat(values: Optional[List[float]], name: str) -> np.ndarray:
    matrix = np.eye(3) if values is None else np.asarray(values, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values")
    # Axis maps should preserve vector lengths.  Reflections are allowed since
    # their conjugation also maps rotations correctly.
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must be an orthonormal 3x3 axis transform")
    return matrix


def _finite_pose(pose: Iterable[float], name: str) -> Pose:
    values = [float(value) for value in pose]
    if len(values) != 6 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain six finite values")
    return values


def _format_timestamp(timestamp_s: float) -> str:
    local_time = time.localtime(timestamp_s)
    milliseconds = int((timestamp_s - int(timestamp_s)) * 1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", local_time) + f".{milliseconds:03d}"


def _pose_lag_mm_deg(a: Pose, b: Pose) -> Tuple[float, float]:
    a = _finite_pose(a, "pose a")
    b = _finite_pose(b, "pose b")
    pos_lag = float(np.linalg.norm(np.asarray(a[:3]) - np.asarray(b[:3])))
    R_a = R.from_euler("XYZ", a[3:], degrees=True).as_matrix()
    R_b = R.from_euler("XYZ", b[3:], degrees=True).as_matrix()
    rot_lag = math.degrees(float(np.linalg.norm(R.from_matrix(R_a.T @ R_b).as_rotvec())))
    return pos_lag, rot_lag


def _smooth_pose(previous: Optional[Pose], target: Pose, alpha: float) -> Pose:
    target = _finite_pose(target, "target pose")
    if previous is None or alpha >= 1.0:
        return target
    if alpha <= 0.0:
        return _finite_pose(previous, "previous pose")

    previous = _finite_pose(previous, "previous pose")
    prev_pos = np.asarray(previous[:3], dtype=float)
    target_pos = np.asarray(target[:3], dtype=float)
    pos = prev_pos + alpha * (target_pos - prev_pos)

    prev_R = R.from_euler("XYZ", previous[3:], degrees=True).as_matrix()
    target_R = R.from_euler("XYZ", target[3:], degrees=True).as_matrix()
    delta_rotvec = R.from_matrix(prev_R.T @ target_R).as_rotvec()
    smoothed_R = prev_R @ R.from_rotvec(alpha * delta_rotvec).as_matrix()
    euler = R.from_matrix(smoothed_R).as_euler("XYZ", degrees=True)
    return [*map(float, pos), *map(float, euler)]


class Pi0ActionAdapter:
    """Convert one configured policy action into a raw CR5A Cartesian target."""

    DELTA_FORMATS = {
        "cartesian_delta_mm_deg",
        "cartesian_delta_m_rad",
        "normalized_cartesian_delta",
    }

    def __init__(
        self,
        *,
        action_format: str,
        position_transform: np.ndarray,
        rotation_transform: np.ndarray,
        delta_frame: str,
        rotation_delta_representation: str,
        normalized_linear_scale_mm: float,
        normalized_angular_scale_deg: float,
        gripper_index: Optional[int],
        gripper_scale: float,
        gripper_offset: float,
    ):
        self.action_format = action_format
        self.position_transform = position_transform
        self.rotation_transform = rotation_transform
        self.delta_frame = delta_frame
        self.rotation_delta_representation = rotation_delta_representation
        self.normalized_linear_scale_mm = normalized_linear_scale_mm
        self.normalized_angular_scale_deg = normalized_angular_scale_deg
        self.gripper_index = gripper_index
        self.gripper_scale = gripper_scale
        self.gripper_offset = gripper_offset

    def adapt(self, action: Any, reference_pose: Pose) -> Tuple[Pose, Optional[float]]:
        """Return ``(raw_target_pose, optional_gripper_trigger)``.

        Delta formats accept the first six components as
        ``[dx, dy, dz, d_rx, d_ry, d_rz]``.  The optional gripper value is read
        at ``--gripper-index`` (6 by default for CR5A's seventh component).
        Absolute formats are already expected in CR5A base coordinates; no
        axis transform is applied to them because an absolute-frame calibration
        cannot be inferred safely from a generic OpenPI checkpoint.
        """
        values = np.asarray(action, dtype=float).reshape(-1)
        if len(values) != CR5A_ACTION_DIM or not np.all(np.isfinite(values)):
            raise ValueError(
                "CR5A Cartesian policy action must contain exactly seven finite values "
                "[dx, dy, dz, dRx, dRy, dRz, gripper], "
                f"got {values!r}"
            )
        reference_pose = _finite_pose(reference_pose, "reference pose")

        gripper = None
        if self.gripper_index is not None:
            if self.gripper_index < 0 or self.gripper_index >= len(values):
                raise ValueError(
                    f"gripper index {self.gripper_index} is outside action dimension {len(values)}"
                )
            gripper = float(np.clip(values[self.gripper_index] * self.gripper_scale + self.gripper_offset, 0.0, 1.0))

        if _canonical_action_format(self.action_format) == "absolute_pose":
            if self.action_format == "cartesian_absolute_m_rad":
                pose = [*map(float, values[:3] * 1000.0), *map(float, np.degrees(values[3:6]))]
            else:
                pose = [*map(float, values[:6])]
            return pose, gripper

        delta_position, delta_rotation = self._decode_delta(values[:6])
        delta_position = self.position_transform @ delta_position
        delta_rotation_matrix = self._rotation_matrix_from_delta(delta_rotation)
        # Coordinate conversion of an SO(3) delta is a conjugation, not an
        # element-wise Euler-axis swap.
        delta_rotation_matrix = (
            self.rotation_transform @ delta_rotation_matrix @ self.rotation_transform.T
        )

        reference_R = R.from_euler("XYZ", reference_pose[3:], degrees=True).as_matrix()
        if self.delta_frame == "tool":
            position = np.asarray(reference_pose[:3]) + reference_R @ delta_position
            target_R = reference_R @ delta_rotation_matrix
        else:
            position = np.asarray(reference_pose[:3]) + delta_position
            target_R = delta_rotation_matrix @ reference_R
        euler = R.from_matrix(target_R).as_euler("XYZ", degrees=True)
        return [*map(float, position), *map(float, euler)], gripper

    def _decode_delta(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.action_format == "cartesian_delta_m_rad":
            return values[:3] * 1000.0, np.degrees(values[3:])
        if self.action_format == "normalized_cartesian_delta":
            return (
                values[:3] * self.normalized_linear_scale_mm,
                values[3:] * self.normalized_angular_scale_deg,
            )
        return values[:3], values[3:]

    def _rotation_matrix_from_delta(self, delta_degrees: np.ndarray) -> np.ndarray:
        if self.rotation_delta_representation == "rotvec":
            return R.from_rotvec(np.radians(delta_degrees)).as_matrix()
        return R.from_euler("XYZ", delta_degrees, degrees=True).as_matrix()


def load_observation_provider(spec: str) -> ObservationProvider:
    """Load ``/path/to/provider.py:function``.

    The function receives a context dict with ``cr5_pose_mm_deg``,
    ``cr5_joints_deg``, ``command_pose_mm_deg``, ``timestamp_s`` and
    ``instruction``.  It must return exactly the observation dictionary
    expected by the selected OpenPI checkpoint.  This explicit boundary is
    intentional: camera names, image layout, state order and normalization are
    checkpoint-specific (TODO: add a CR5A checkpoint config when one exists).
    """
    try:
        file_name, function_name = spec.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError("--observation-provider must be /path/to/file.py:function") from exc
    file_path = Path(file_name).expanduser().resolve()
    if not file_path.is_file():
        raise ValueError(f"observation provider file does not exist: {file_path}")
    module_spec = importlib.util.spec_from_file_location("pi0_cr5a_observation_provider", file_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"cannot import observation provider: {file_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    provider = getattr(module, function_name, None)
    if not callable(provider):
        raise ValueError(f"observation provider {spec!r} is not callable")
    return provider


class JsonlActionSource:
    """Finite action source for dry runs and calibration replay."""

    def __init__(self, path: str, action_key: str):
        self._lines = iter(Path(path).expanduser().read_text().splitlines())
        self._action_key = action_key

    def infer(self, _observation: Dict[str, Any]) -> Dict[str, Any]:
        for line in self._lines:
            if line.strip():
                item = json.loads(line)
                return item if isinstance(item, dict) else {self._action_key: item}
        raise StopIteration


def make_policy_source(args: argparse.Namespace) -> Any:
    if args.actions_jsonl:
        return JsonlActionSource(args.actions_jsonl, args.policy_action_key)

    client_source = _find_openpi_client_source(Path(__file__).resolve())
    if str(client_source) not in sys.path:
        sys.path.insert(0, str(client_source))
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import openpi_client. Install OpenPI's client package or use --actions-jsonl."
        ) from exc
    return WebsocketClientPolicy(args.policy_host, args.policy_port)


def _find_openpi_client_source(start: Path) -> Path:
    for parent in (start, *start.parents):
        candidate = parent / "openpi" / "packages" / "openpi-client" / "src"
        if candidate.is_dir():
            return candidate
    # Fall back to the repository layout used when this script lives under
    # dobot_teleop/scripts/bridge and the checkout root is three levels up.
    return start.parents[3] / "openpi" / "packages" / "openpi-client" / "src"


def extract_action_array(response: Dict[str, Any], action_key: str) -> np.ndarray:
    """Normalize a websocket response to a finite ``(horizon, action_dim)`` array."""
    if action_key not in response:
        raise ValueError(f"policy response has no {action_key!r} key; keys={list(response)}")
    actions = np.asarray(response[action_key], dtype=float)
    if actions.ndim == 1:
        actions = actions[None, :]
    elif actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2:
        raise ValueError(f"policy {action_key!r} must have shape (horizon, action_dim), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("policy action chunk contains non-finite values")
    return actions


def classify_cr5a_action_chunk(actions: np.ndarray, *, action_format: str, execute: bool) -> str:
    """Classify and gate a policy action chunk before it can reach ServoP."""
    if actions.ndim != 2:
        raise ValueError(f"policy actions must have shape (horizon, action_dim), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("policy action chunk contains non-finite values")

    action_dim = actions.shape[-1]
    if action_dim == ALOHA_ACTION_DIM:
        if execute:
            raise ValueError(
                "Received ALOHA-style 14D action, not CR5A Cartesian 7D action. Refuse to execute."
            )
        return "aloha_14d"
    if action_dim != CR5A_ACTION_DIM:
        raise ValueError(
            f"Received {action_dim}D policy action, expected CR5A Cartesian 7D action "
            "[dx, dy, dz, dRx, dRy, dRz, gripper]."
        )
    if _canonical_action_format(action_format) not in {"delta", "absolute_pose"}:
        raise ValueError("CR5A ServoP requires --action-format delta or --action-format absolute_pose")
    return "cr5a_cartesian_7d"


def extract_action_chunk(response: Dict[str, Any], action_key: str, horizon: int) -> Deque[np.ndarray]:
    """Compatibility helper for callers that already hold a validated action chunk."""
    actions = extract_action_array(response, action_key)
    return deque(np.array(action, dtype=float) for action in actions[: max(horizon, 1)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-ip", required=True, help="Dobot controller IP")
    parser.add_argument("--dashboard-port", type=int, default=29999)
    parser.add_argument("--policy-host", default="127.0.0.1", help="OpenPI websocket server host")
    parser.add_argument("--policy-port", type=int, default=8000)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--observation-provider", help="/path/provider.py:function for websocket inference")
    source.add_argument("--actions-jsonl", help="JSONL action chunks; intended for dry-run/calibration")
    parser.add_argument("--policy-action-key", default="actions")
    parser.add_argument("--open-loop-horizon", type=int, default=1,
                        help="Actions to execute from a returned PI0 action chunk before querying again")
    parser.add_argument("--async-policy", action=argparse.BooleanOptionalAction, default=True,
                        help="Run the next policy inference in the background while executing queued actions")
    parser.add_argument("--policy-prefetch-remaining", type=int, default=4,
                        help="Start async inference when this many queued actions remain")
    parser.add_argument("--instruction", default="", help="Passed unchanged to the observation provider")

    parser.add_argument(
        "--action-format",
        choices=(
            "delta",
            "absolute_pose",
            "cartesian_delta_mm_deg",
            "cartesian_delta_m_rad",
            "normalized_cartesian_delta",
            "cartesian_absolute_mm_deg",
            "cartesian_absolute_m_rad",
        ),
        default="delta",
    )
    parser.add_argument("--delta-frame", choices=("base", "tool"), default="tool")
    parser.add_argument("--rotation-delta-representation", choices=("euler", "rotvec"), default="euler")
    parser.add_argument("--normalized-linear-scale-mm", type=float, default=2.0)
    parser.add_argument("--normalized-angular-scale-deg", type=float, default=2.0)
    parser.add_argument("--policy-pos-transform", type=float, nargs=9, default=None, metavar="P")
    parser.add_argument("--policy-rot-transform", type=float, nargs=9, default=None, metavar="R")
    parser.add_argument("--gripper-index", type=int, default=6,
                        help="Action component used as a [0,1] PGE trigger; use -1 to disable")
    parser.add_argument("--gripper-scale", type=float, default=1.0)
    parser.add_argument("--gripper-offset", type=float, default=0.0)

    parser.add_argument("--servo-mode", choices=("cartesian", "joint"), default="cartesian",
                        help="cartesian = ServoP; joint = IK + joint limiting + ServoJ")
    parser.add_argument("--command-rate", type=float, default=10.0)
    parser.add_argument("--servo-t", type=float, default=None, help="Defaults to one command period")
    parser.add_argument("--servo-aheadtime", type=float, default=50.0)
    parser.add_argument("--servo-gain", type=float, default=500.0)
    parser.add_argument("--target-lowpass-alpha", type=float, default=1.0,
                        help="Low-pass factor for ServoP targets: 1 disables smoothing, smaller is smoother")
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=30.0,
                        help="Max per-joint speed for ServoJ mode")
    parser.add_argument("--max-linear-speed-mm-s", type=float, default=30.0)
    parser.add_argument("--max-angular-speed-deg-s", type=float, default=15.0)
    parser.add_argument("--max-control-dt", type=float, default=0.10,
                        help="Upper bound on elapsed time used by the target governor")
    parser.add_argument("--max-total-translation-mm", type=float, default=120.0)
    parser.add_argument("--max-total-rotation-deg", type=float, default=90.0)
    parser.add_argument("--workspace-min-x-mm", type=float, default=-700.0)
    parser.add_argument("--workspace-max-x-mm", type=float, default=700.0)
    parser.add_argument("--workspace-min-y-mm", type=float, default=-700.0)
    parser.add_argument("--workspace-max-y-mm", type=float, default=350.0)
    parser.add_argument("--workspace-min-z-mm", type=float, default=50.0)
    parser.add_argument("--workspace-max-z-mm", type=float, default=800.0)

    parser.add_argument("--clear-error", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Arm ServoP output; otherwise the bridge is dry-run only")
    parser.add_argument("--max-actions", type=int, default=0, help="Stop after N actions (0 means run until interrupted/source ends)")
    parser.add_argument("--log-targets", action="store_true")
    parser.add_argument("--verbose-tcp", action="store_true")
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
    parser.add_argument("--gripper-trigger-threshold", type=float, default=0.7,
                        help="Close gripper when the policy gripper value is at least this value")
    parser.add_argument("--gripper-open-threshold", type=float, default=0.25,
                        help="Open gripper only when the policy gripper value is at most this value")
    parser.add_argument("--gripper-min-command-interval-s", type=float, default=0.4,
                        help="Minimum time between open/close state changes")
    parser.add_argument("--gripper-close-delay-s", type=float, default=0.0,
                        help="Require a sustained close request for this long before closing")
    parser.add_argument("--gripper-close-max-lag-mm", type=float, default=0.0,
                        help="If positive, defer closing while the sent ServoP target lags the policy target by more than this")
    parser.add_argument("--gripper-init-value", type=lambda value: int(value, 0), default=1)
    parser.add_argument("--gripper-init-timeout", type=float, default=8.0)

    # ── Tool offset (model outputs gripper_center → convert to TCP for ServoP) ─
    parser.add_argument("--use-gripper-center-pose", action="store_true", default=True,
                        help="Model outputs gripper-center poses; convert to TCP before ServoP")
    parser.add_argument("--no-use-gripper-center-pose", action="store_false",
                        dest="use_gripper_center_pose",
                        help="Model outputs TCP poses; send directly to ServoP")
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
                        help="Print gripper_center → tcp conversion at 1 Hz")

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    # Fail before opening a robot connection if an invalid format was supplied.
    _canonical_action_format(args.action_format)
    if args.command_rate <= 0 or args.max_control_dt <= 0:
        raise ValueError("--command-rate and --max-control-dt must be positive")
    if args.servo_t is not None and args.servo_t < 0.02:
        raise ValueError("--servo-t must be >= 0.02 for Dobot ServoJ/ServoP")
    if not 20.0 <= args.servo_aheadtime <= 100.0:
        raise ValueError("--servo-aheadtime must be in [20, 100] for Dobot ServoJ/ServoP")
    if not 200.0 <= args.servo_gain <= 1000.0:
        raise ValueError("--servo-gain must be in [200, 1000] for Dobot ServoJ/ServoP")
    if not 0.0 <= args.target_lowpass_alpha <= 1.0:
        raise ValueError("--target-lowpass-alpha must be in [0, 1]")
    if args.servo_mode == "joint" and args.max_joint_speed_deg_s <= 0.0:
        raise ValueError("--max-joint-speed-deg-s must be positive in ServoJ mode")
    if args.open_loop_horizon <= 0:
        raise ValueError("--open-loop-horizon must be positive")
    if args.policy_prefetch_remaining < 0:
        raise ValueError("--policy-prefetch-remaining must not be negative")
    if args.max_actions < 0:
        raise ValueError("--max-actions must not be negative")
    if args.gripper_index < -1:
        raise ValueError("--gripper-index must be -1 or a non-negative index")
    if args.enable_gripper and args.gripper_index == -1:
        raise ValueError("--enable-gripper requires --gripper-index >= 0")
    if args.enable_gripper and not 20 <= args.gripper_force <= 100:
        raise ValueError("--gripper-force must be in [20, 100]")
    if not 0.0 <= args.gripper_open_threshold <= args.gripper_trigger_threshold <= 1.0:
        raise ValueError("--gripper-open-threshold must be <= --gripper-trigger-threshold, both in [0, 1]")
    if args.gripper_min_command_interval_s < 0.0:
        raise ValueError("--gripper-min-command-interval-s must not be negative")
    if args.gripper_close_delay_s < 0.0:
        raise ValueError("--gripper-close-delay-s must not be negative")
    if args.gripper_close_max_lag_mm < 0.0:
        raise ValueError("--gripper-close-max-lag-mm must not be negative")
    if args.gripper_init_value not in (1, 0xA5):
        raise ValueError("--gripper-init-value must be 1 or 0xA5")


def _make_safety_envelope(args: argparse.Namespace) -> CartesianSafetyEnvelope:
    return CartesianSafetyEnvelope(
        max_total_translation_mm=args.max_total_translation_mm,
        max_total_rotation_deg=args.max_total_rotation_deg,
        workspace_min_x_mm=args.workspace_min_x_mm,
        workspace_max_x_mm=args.workspace_max_x_mm,
        workspace_min_y_mm=args.workspace_min_y_mm,
        workspace_max_y_mm=args.workspace_max_y_mm,
        workspace_min_z_mm=args.workspace_min_z_mm,
        workspace_max_z_mm=args.workspace_max_z_mm,
    )


def _require_origin_in_workspace(origin_pose: Pose, args: argparse.Namespace) -> None:
    origin_pose = _finite_pose(origin_pose, "CR5A origin pose")
    checks = (
        ("X", origin_pose[0], args.workspace_min_x_mm, args.workspace_max_x_mm),
        ("Y", origin_pose[1], args.workspace_min_y_mm, args.workspace_max_y_mm),
        ("Z", origin_pose[2], args.workspace_min_z_mm, args.workspace_max_z_mm),
    )
    outside = [
        f"{axis}={value:.1f} not in [{lower:.1f}, {upper:.1f}]"
        for axis, value, lower, upper in checks
        if value < lower or value > upper
    ]
    if outside:
        raise ValueError(
            "Refusing to arm: CR5A origin is outside the configured workspace: "
            + "; ".join(outside)
        )


def main() -> None:
    args = parse_args()
    _validate_args(args)

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
            "Software offset is SKIPPED.  The model's gripper-center output "
            "will be sent directly to ServoP without conversion. "
            "Ensure the Dobot controller tool offset is correctly configured."
        )
    if tool_offset_cfg.effective():
        print(
            "[TOOL_OFFSET] PI0 outputs gripper_center targets. "
            "Each target will be converted to TCP (subtract local-Z offset) "
            "BEFORE sending to the robot.  No 16 cm double-offset."
        )

    period = 1.0 / args.command_rate
    servo_t = period if args.servo_t is None else args.servo_t
    max_joint_step_deg = args.max_joint_speed_deg_s * period
    adapter = Pi0ActionAdapter(
        action_format=args.action_format,
        position_transform=_matrix_from_flat(args.policy_pos_transform, "--policy-pos-transform"),
        rotation_transform=_matrix_from_flat(args.policy_rot_transform, "--policy-rot-transform"),
        delta_frame=args.delta_frame,
        rotation_delta_representation=args.rotation_delta_representation,
        normalized_linear_scale_mm=args.normalized_linear_scale_mm,
        normalized_angular_scale_deg=args.normalized_angular_scale_deg,
        gripper_index=None if args.gripper_index == -1 else args.gripper_index,
        gripper_scale=args.gripper_scale,
        gripper_offset=args.gripper_offset,
    )
    provider = load_observation_provider(args.observation_provider) if args.observation_provider else None
    source = make_policy_source(args)
    client = DobotDashboard(args.robot_ip, args.dashboard_port, timeout=0.6, verbose=args.verbose_tcp)
    gripper: Optional[PgeModbusGripper] = None
    safety = _make_safety_envelope(args)
    governor = CartesianTargetGovernor(args.max_linear_speed_mm_s, args.max_angular_speed_deg_s)
    action_queue: Deque[np.ndarray] = deque()
    simulated_pose = [0.0, -300.0, 300.0, -180.0, 0.0, 90.0]
    simulated_joints = [0.0] * 6
    smoothed_command_target: Optional[Pose] = None
    gripper_close_requested_at: Optional[float] = None
    last_sent_joints: Optional[List[float]] = None
    last_policy_timestamp_s = 0.0
    last_policy_latency_ms = 0.0
    policy_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if args.async_policy else None
    pending_policy: Optional[concurrent.futures.Future] = None
    sent = 0

    def build_policy_observation() -> Dict[str, Any]:
        actual_pose = client.get_pose() if args.execute else list(origin_pose)
        actual_joints = client.get_angle() if args.execute else list(origin_joints)
        context = {
            "timestamp_s": time.time(),
            "cr5_pose_mm_deg": actual_pose,
            "cr5_joints_deg": actual_joints,
            "command_pose_mm_deg": governor.current_target(),
            "instruction": args.instruction,
        }
        observation = provider(context) if provider is not None else {}
        if not isinstance(observation, dict):
            raise ValueError("observation provider must return a dictionary")
        return observation

    def run_policy_inference(observation: Dict[str, Any], request_timestamp_s: float) -> Tuple[np.ndarray, float, float]:
        policy_start = time.perf_counter()
        response = source.infer(observation)
        policy_latency_ms = (time.perf_counter() - policy_start) * 1000.0
        raw_actions = extract_action_array(response, args.policy_action_key)
        return raw_actions, request_timestamp_s, policy_latency_ms

    def submit_policy_request() -> concurrent.futures.Future:
        observation = build_policy_observation()
        request_timestamp_s = time.time()
        if policy_executor is None:
            future: concurrent.futures.Future = concurrent.futures.Future()
            try:
                future.set_result(run_policy_inference(observation, request_timestamp_s))
            except Exception as exc:
                future.set_exception(exc)
            return future
        print(f"[PI0-CR5A] async policy request ts={_format_timestamp(request_timestamp_s)}")
        return policy_executor.submit(run_policy_inference, observation, request_timestamp_s)

    def action_queue_from_policy_result(result: Tuple[np.ndarray, float, float]) -> Tuple[Deque[np.ndarray], float, float]:
        raw_actions, policy_timestamp_s, policy_latency_ms = result
        print(
            "[PI0-CR5A] "
            f"policy_ts={_format_timestamp(policy_timestamp_s)} "
            f"policy_ms={policy_latency_ms:.1f} "
            f"raw actions shape = {raw_actions.shape}"
        )
        action_kind = classify_cr5a_action_chunk(
            raw_actions, action_format=args.action_format, execute=args.execute
        )
        if action_kind == "aloha_14d":
            raise RuntimeError("Received ALOHA 14D action; cannot execute on CR5A.")
        return (
            deque(np.array(action, dtype=float) for action in raw_actions[: max(args.open_loop_horizon, 1)]),
            policy_timestamp_s,
            policy_latency_ms,
        )

    print("PI0/OpenPI -> CR5A bridge")
    print(
        f"action={args.action_format}, delta_frame={args.delta_frame}, "
        f"servo_mode={args.servo_mode}, rate={args.command_rate:.1f} Hz"
    )
    if args.servo_mode == "joint":
        print(
            f"ServoJ joint step limit: {max_joint_step_deg:.2f} deg/step "
            f"({args.max_joint_speed_deg_s:.1f} deg/s)"
        )
    print(
        "Policy scheduling: "
        f"async={args.async_policy}, open_loop_horizon={args.open_loop_horizon}, "
        f"prefetch_remaining={args.policy_prefetch_remaining}"
    )
    print(
        "Safety: "
        f"workspace=([{args.workspace_min_x_mm:.0f},{args.workspace_max_x_mm:.0f}], "
        f"[{args.workspace_min_y_mm:.0f},{args.workspace_max_y_mm:.0f}], "
        f"[{args.workspace_min_z_mm:.0f},{args.workspace_max_z_mm:.0f}]) mm; "
        f"origin limits={args.max_total_translation_mm:.1f} mm / {args.max_total_rotation_deg:.1f} deg; "
        f"governor={args.max_linear_speed_mm_s:.1f} mm/s / {args.max_angular_speed_deg_s:.1f} deg/s"
    )
    if not args.execute:
        print("DRY RUN: robot, gripper, and Dobot network commands are disabled. Pass --execute to arm motion.")

    try:
        if args.execute:
            client.connect()
            print(f"Dobot dashboard connected via {client.backend_name()}.")
            if args.clear_error:
                print(client.clear_error())
            if args.enable_robot:
                print(client.enable_robot())
            origin_pose = client.get_pose()
            origin_joints = client.get_angle()
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
                    open_threshold=args.gripper_open_threshold,
                    min_command_interval_s=args.gripper_min_command_interval_s,
                )
                gripper.initialize(args.gripper_init_value)
        else:
            origin_pose, origin_joints = simulated_pose, simulated_joints

        _require_origin_in_workspace(origin_pose, args)
        safety.reset(origin_pose)
        governor.reset(origin_pose)
        last_sent_joints = list(origin_joints)
        print(f"CR5A command origin: {format_pose(origin_pose)}")
        next_tick = time.monotonic()
        # The first policy action is a normal command-period increment, not a
        # zero-length no-op caused by initialization timing.
        last_tick = next_tick - period
        while args.max_actions == 0 or sent < args.max_actions:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.005))
                continue
            next_tick += period
            dt = min(max(now - last_tick, 0.0), args.max_control_dt)
            last_tick = now

            if pending_policy is not None and pending_policy.done():
                try:
                    action_queue, last_policy_timestamp_s, last_policy_latency_ms = action_queue_from_policy_result(
                        pending_policy.result()
                    )
                finally:
                    pending_policy = None

            if not action_queue:
                if pending_policy is None:
                    pending_policy = submit_policy_request()
                action_queue, last_policy_timestamp_s, last_policy_latency_ms = action_queue_from_policy_result(
                    pending_policy.result()
                )
                pending_policy = None

            if (
                args.async_policy
                and pending_policy is None
                and len(action_queue) <= args.policy_prefetch_remaining
            ):
                pending_policy = submit_policy_request()

            action = action_queue.popleft()
            model_target, gripper_trigger = adapter.adapt(action, governor.current_target())

            # ── Convert gripper_center → TCP for robot command ───────────
            # The model outputs targets in gripper_center space (same frame as
            # training data).  Dobot motion commands expect TCP targets only.
            # T_base_tcp_cmd = T_base_gripper_des @ inv(T_tcp_gripper)
            if tool_offset_cfg.effective():
                raw_target = remove_tool_offset(model_target, tool_offset_cfg)
            else:
                raw_target = list(model_target)

            fenced_target = safety.apply(raw_target)
            command_target = governor.update(fenced_target, dt)
            command_target = _smooth_pose(smoothed_command_target, command_target, args.target_lowpass_alpha)
            smoothed_command_target = command_target

            gripper_deferred = False
            gripper_lag_mm = 0.0
            if gripper is not None and gripper_trigger is not None:
                close_requested = gripper_trigger >= args.gripper_trigger_threshold
                if close_requested and gripper.closed is not True:
                    if gripper_close_requested_at is None:
                        gripper_close_requested_at = now
                    close_age_s = now - gripper_close_requested_at
                    gripper_lag_mm, _ = _pose_lag_mm_deg(command_target, fenced_target)
                    gripper_deferred = (
                        close_age_s < args.gripper_close_delay_s
                        or (
                            args.gripper_close_max_lag_mm > 0.0
                            and gripper_lag_mm > args.gripper_close_max_lag_mm
                        )
                    )
                else:
                    gripper_close_requested_at = None

            planned_joints: Optional[List[float]] = None
            joint_step = 0.0
            joint_limited = False
            if args.execute:
                if args.servo_mode == "joint":
                    try:
                        desired_joints = client.inverse_kin(command_target)
                    except DobotDashboardError:
                        print(f"IK unsolvable for ServoJ target: {format_pose(command_target)}, skipping")
                        continue

                    ok, limit_msg = check_joint_limits(desired_joints)
                    if not ok:
                        print(
                            f"Joint-limit skip: {limit_msg}; "
                            f"target={format_pose(command_target)} desired=[{format_joints(desired_joints)}]"
                        )
                        continue

                    if last_sent_joints is None:
                        last_sent_joints = client.get_angle()
                    planned_joints, joint_step, joint_limited = plan_joint_step(
                        desired_joints,
                        last_sent_joints,
                        max_joint_step_deg,
                    )

                    ok, limit_msg = check_joint_limits(planned_joints)
                    if not ok:
                        print(
                            f"Joint-limit skip after rate limit: {limit_msg}; "
                            f"planned=[{format_joints(planned_joints)}]"
                        )
                        continue

                    try:
                        client.servoj(
                            planned_joints,
                            t=servo_t,
                            aheadtime=args.servo_aheadtime,
                            gain=args.servo_gain,
                        )
                        last_sent_joints = planned_joints
                    except DobotDashboardError as exc:
                        try:
                            mode = client.robot_mode()
                        except Exception as mode_exc:
                            mode = f"unavailable ({mode_exc})"
                        print(
                            f"ServoJ rejected: target={format_pose(command_target)}, "
                            f"planned=[{format_joints(planned_joints)}], "
                            f"robot_mode={mode}, response={exc}, skipping"
                        )
                        if mode == 11:
                            try:
                                print("  ↳ robot_mode=11 (COLLISION), auto-clearing...")
                                client.clear_error()
                            except DobotDashboardError as clear_exc:
                                print(f"  ↳ ClearError also failed: {clear_exc}")
                        continue
                else:
                    client.servop(command_target, t=servo_t, aheadtime=args.servo_aheadtime, gain=args.servo_gain)

                if gripper is not None and gripper_trigger is not None and not gripper_deferred:
                    gripper.update_from_trigger(gripper_trigger)
            if args.log_targets or not args.execute:
                action_timestamp_s = time.time()
                log_parts = [
                    f"ts={_format_timestamp(action_timestamp_s)}",
                    f"policy_ts={_format_timestamp(last_policy_timestamp_s)}",
                    f"policy_ms={last_policy_latency_ms:.1f}",
                    f"action[{sent}]={np.array2string(action, precision=4)}",
                ]
                if tool_offset_cfg.effective():
                    log_parts.append(f"model(gripper) {format_pose(model_target)}")
                    log_parts.append(f"→tcp {format_pose(raw_target)}")
                else:
                    log_parts.append(f"raw {format_pose(raw_target)}")
                log_parts.append(f"fenced {format_pose(fenced_target)}")
                if args.servo_mode == "joint":
                    log_parts.append(f"ServoJ target {format_pose(command_target)}")
                    if planned_joints is not None:
                        limit_text = " limited" if joint_limited else ""
                        log_parts.append(
                            f"joints [{format_joints(planned_joints)}] step={joint_step:.2f}deg{limit_text}"
                        )
                else:
                    log_parts.append(f"ServoP {format_pose(command_target)}")
                if gripper_trigger is not None:
                    log_parts.append(f"grip={gripper_trigger:.2f}")
                    if gripper_deferred:
                        log_parts.append(f"grip_deferred lag={gripper_lag_mm:.1f}mm")
                print(" | ".join(log_parts))
            sent += 1

    except StopIteration:
        print("Action source exhausted.")
    except KeyboardInterrupt:
        print("Interrupted.")
    except (OSError, DobotDashboardError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    finally:
        if policy_executor is not None:
            policy_executor.shutdown(wait=False, cancel_futures=True)
        if gripper is not None:
            try:
                gripper.close()
            except Exception:
                pass
        if args.execute:
            try:
                client.stop()
            except Exception:
                pass
            client.close()


if __name__ == "__main__":
    main()
