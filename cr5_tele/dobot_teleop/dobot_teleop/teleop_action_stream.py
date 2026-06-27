"""Latest-only localhost UDP stream for CR5A Quest teleoperation actions.

The publisher is deliberately fire-and-forget: a slow or absent recorder can
never block the robot control loop.  The recorder validates timestamps and
keeps only the newest received packet.
"""

from __future__ import annotations

import json
import math
import socket
from dataclasses import dataclass
from typing import Any, Sequence


ACTION_DIM = 7
POSE_DIM = 6


def _finite_vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class TeleopAction:
    """One planned or successfully sent CR5A control action.

    ``current_pose`` is the controller's last successfully sent Cartesian
    target (TCP frame).  It intentionally avoids an extra synchronous
    ``GetPose`` call in the teleop loop.

    ``gripper_center_pose`` and ``gripper_center_target`` (optional) are the
    same poses transformed to the gripper-center frame via the tool offset.
    When tool offset is active, the recorder uses these for training labels.
    """

    timestamp: float
    seq: int
    source: str
    action: tuple[float, ...]
    current_pose: tuple[float, ...]
    target_pose: tuple[float, ...]
    current_joints: tuple[float, ...] | None
    deadman: bool
    servo_sent: bool
    gripper_command: float
    gripper_center_pose: tuple[float, ...] | None = None
    gripper_center_target: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        if self.seq < 0:
            raise ValueError("seq must be non-negative")
        if not self.source:
            raise ValueError("source must not be empty")
        _finite_vector(self.action, ACTION_DIM, "action")
        _finite_vector(self.current_pose, POSE_DIM, "current_pose")
        _finite_vector(self.target_pose, POSE_DIM, "target_pose")
        if self.current_joints is not None:
            _finite_vector(self.current_joints, POSE_DIM, "current_joints")
        if self.gripper_center_pose is not None:
            _finite_vector(self.gripper_center_pose, POSE_DIM, "gripper_center_pose")
        if self.gripper_center_target is not None:
            _finite_vector(self.gripper_center_target, POSE_DIM, "gripper_center_target")
        if not math.isfinite(float(self.gripper_command)):
            raise ValueError("gripper_command must be finite")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "timestamp": float(self.timestamp),
            "seq": int(self.seq),
            "source": self.source,
            "action": list(self.action),
            "current_pose": list(self.current_pose),
            "target_pose": list(self.target_pose),
            "current_joints": None if self.current_joints is None else list(self.current_joints),
            "deadman": bool(self.deadman),
            "servo_sent": bool(self.servo_sent),
            "gripper_command": float(self.gripper_command),
        }
        if self.gripper_center_pose is not None:
            result["gripper_center_pose"] = list(self.gripper_center_pose)
        if self.gripper_center_target is not None:
            result["gripper_center_target"] = list(self.gripper_center_target)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeleopAction":
        joints = data.get("current_joints")
        gc_pose = data.get("gripper_center_pose")
        gc_target = data.get("gripper_center_target")
        return cls(
            timestamp=float(data["timestamp"]),
            seq=int(data["seq"]),
            source=str(data["source"]),
            action=_finite_vector(data["action"], ACTION_DIM, "action"),
            current_pose=_finite_vector(data["current_pose"], POSE_DIM, "current_pose"),
            target_pose=_finite_vector(data["target_pose"], POSE_DIM, "target_pose"),
            current_joints=None if joints is None else _finite_vector(joints, POSE_DIM, "current_joints"),
            deadman=bool(data["deadman"]),
            servo_sent=bool(data["servo_sent"]),
            gripper_command=float(data.get("gripper_command", 0.0)),
            gripper_center_pose=(
                None if gc_pose is None else _finite_vector(gc_pose, POSE_DIM, "gripper_center_pose")
            ),
            gripper_center_target=(
                None if gc_target is None else _finite_vector(gc_target, POSE_DIM, "gripper_center_target")
            ),
        )


class TeleopActionPublisher:
    """Non-blocking UDP JSON publisher used only by the teleop process."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5010):
        self.address = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._seq = 0

    def publish(self, action: TeleopAction) -> bool:
        try:
            self._socket.sendto(json.dumps(action.to_dict(), separators=(",", ":")).encode("utf-8"), self.address)
        except (OSError, TypeError, ValueError):
            return False
        self._seq = max(self._seq, action.seq + 1)
        return True

    def next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def close(self) -> None:
        self._socket.close()


class TeleopActionSubscriber:
    """UDP receiver that discards malformed packets and returns newest valid action."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5010):
        self.address = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(self.address)
        self._socket.setblocking(False)
        self.latest: TeleopAction | None = None
        self.invalid_packets = 0

    def poll_latest(self) -> TeleopAction | None:
        newest = None
        while True:
            try:
                data, _address = self._socket.recvfrom(8192)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                message = json.loads(data.decode("utf-8"))
                newest = TeleopAction.from_dict(message)
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                self.invalid_packets += 1
        if newest is not None:
            self.latest = newest
        return newest

    def close(self) -> None:
        self._socket.close()
