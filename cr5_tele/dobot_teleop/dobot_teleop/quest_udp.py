#!/usr/bin/env python3

import json
import socket
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class QuestPose:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    buttons: Dict[str, float]  # button name -> value (0/1 or trigger 0..1)
    timestamp: float
    count: int
    address: Tuple[str, int]
    raw: Dict[str, object] = field(default_factory=dict)


class QuestUdpReceiver:
    """Latest-only UDP receiver for the Quest Unity app.

    poll_latest() drains the socket and returns only the newest packet. This is
    deliberate for teleoperation: stale controller poses are worse than skipped
    intermediate poses.

    Expected JSON format from Unity app:
    {
        "x": float, "y": float, "z": float,
        "qx": float, "qy": float, "qz": float, "qw": float,
        "RG": 0/1, "A": 0/1, "B": 0/1,
        "rightTrig": float (0..1), "leftTrig": float (0..1),
        ...
    }
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5005):
        self.host = host
        self.port = port
        self.count = 0
        self.latest: Optional[QuestPose] = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.setblocking(False)

    def close(self) -> None:
        self.sock.close()

    def poll_latest(self) -> Optional[QuestPose]:
        newest = None
        while True:
            try:
                data, address = self.sock.recvfrom(4096)
            except BlockingIOError:
                break

            newest = self._decode(data, address)

        if newest is not None:
            self.latest = newest
        return newest

    def _decode(self, data: bytes, address: Tuple[str, int]) -> QuestPose:
        msg = json.loads(data.decode("utf-8"))
        self.count += 1

        button_aliases = {
            "RG": [
                "RG",
                "rightGrip",
                "RightGrip",
                "rightGripButton",
                "gripButton",
                "grip",
                "SecondaryHandTrigger",
            ],
            "A": ["A", "buttonA"],
            "B": ["B", "buttonB"],
            "rightTrig": ["rightTrig", "rightTrigger", "trigger", "IndexTrigger"],
            "leftTrig": ["leftTrig", "leftTrigger"],
            "LG": ["LG", "leftGrip", "LeftGrip", "leftGripButton"],
        }
        buttons: Dict[str, float] = {}
        for name, aliases in button_aliases.items():
            for key in aliases:
                if key not in msg:
                    continue
                val = msg[key]
                buttons[key] = float(val) if not isinstance(val, bool) else (1.0 if val else 0.0)
                if name != key:
                    buttons[name] = buttons[key]
                break

        return QuestPose(
            x=float(msg.get("x", 0.0)),
            y=float(msg.get("y", 0.0)),
            z=float(msg.get("z", 1.0)),
            qx=float(msg.get("qx", 0.0)),
            qy=float(msg.get("qy", 0.0)),
            qz=float(msg.get("qz", 0.0)),
            qw=float(msg.get("qw", 1.0)),
            buttons=buttons,
            timestamp=time.monotonic(),
            count=self.count,
            address=address,
            raw=msg,
        )
