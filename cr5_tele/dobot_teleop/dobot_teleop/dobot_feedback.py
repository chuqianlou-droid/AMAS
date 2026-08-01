#!/usr/bin/env python3
"""Dobot feedback-port client for live robot state sampling.

The Dashboard port (29999) is often owned by the teleoperation process.  The
feedback port (30004/30005) streams read-only robot state and can be used by
the recorder to align images with physical robot feedback without stealing the
control connection.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


class DobotFeedbackError(RuntimeError):
    pass


class DobotFeedbackClient:
    """Read ToolVectorActual and QActual from Dobot's feedback port."""

    TEST_VALUE = 0x123456789ABCDEF

    def __init__(
        self,
        host: str,
        port: int = 30004,
        timeout: float = 0.8,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._api = None

    def connect(self) -> None:
        try:
            probe = socket.create_connection((self.host, self.port), timeout=self.timeout)
            probe.close()
        except OSError as exc:
            raise DobotFeedbackError(
                f"Cannot connect to Dobot feedback at {self.host}:{self.port}: {exc}"
            ) from exc

        api_class = self._load_official_feedback_api()
        self._api = api_class(self.host, self.port)
        if getattr(self._api, "socket_dobot", 0) == 0:
            raise DobotFeedbackError(f"Dobot feedback socket was not opened: {self.host}:{self.port}")

    def close(self) -> None:
        if self._api is not None:
            api = self._api
            try:
                api.close()
            finally:
                if hasattr(api, "socket_dobot"):
                    api.socket_dobot = 0
                self._api = None

    def backend_name(self) -> str:
        return "official DobotApiFeedBack"

    @staticmethod
    def _load_official_feedback_api():
        workspace_root = Path(__file__).resolve().parents[2]
        sdk_dir = workspace_root / "TCP-IP-Python-V4"
        sdk_file = sdk_dir / "dobot_api.py"
        if not sdk_file.exists():
            raise DobotFeedbackError(
                f"Official Dobot SDK not found: {sdk_file}. "
                "Put TCP-IP-Python-V4 next to dobot_teleop."
            )

        sdk_dir_str = str(sdk_dir)
        if sdk_dir_str not in sys.path:
            sys.path.insert(0, sdk_dir_str)

        try:
            from dobot_api import DobotApiFeedBack
        except Exception as exc:
            raise DobotFeedbackError(
                f"Failed to import official Dobot SDK from {sdk_dir}: {exc}"
            ) from exc

        return DobotApiFeedBack

    @staticmethod
    def _finite_vector(values: Iterable[float], size: int, name: str) -> List[float]:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise DobotFeedbackError(f"{name} must contain {size} finite values, got {array!r}")
        return [float(value) for value in array]

    def read_state(self, attempts: int = 5) -> Tuple[List[float], List[float]]:
        """Return ``(tool_vector_actual, q_actual)`` from the latest valid packet."""
        if self._api is None:
            raise DobotFeedbackError("Dobot feedback is not connected")

        last_error: Exception | None = None
        for _ in range(max(attempts, 1)):
            try:
                feedback = self._api.feedBackData()
                if feedback is None or len(feedback) == 0:
                    continue
                if int(feedback["TestValue"][0]) != self.TEST_VALUE:
                    continue
                pose = self._finite_vector(feedback["ToolVectorActual"][0], 6, "ToolVectorActual")
                joints = self._finite_vector(feedback["QActual"][0], 6, "QActual")
                return pose, joints
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise DobotFeedbackError(f"Failed to read feedback state: {last_error}") from last_error
        raise DobotFeedbackError("Failed to read a valid Dobot feedback packet")
