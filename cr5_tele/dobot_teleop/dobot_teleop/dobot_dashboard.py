#!/usr/bin/env python3

import re
import socket
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional


class DobotDashboardError(RuntimeError):
    pass


class DobotDashboard:
    """Dobot CR V4 dashboard client backed only by the official SDK."""

    def __init__(
        self,
        host: str,
        port: int = 29999,
        timeout: float = 0.6,
        verbose: bool = False,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.verbose = verbose
        self._api = None

    def connect(self) -> None:
        try:
            probe = socket.create_connection((self.host, self.port), timeout=self.timeout)
            probe.close()
        except OSError as exc:
            raise DobotDashboardError(
                f"Cannot connect to Dobot dashboard at {self.host}:{self.port}: {exc}"
            ) from exc

        api_class = self._load_official_dashboard_api()
        self._api = api_class(self.host, self.port)

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
        return "official DobotApiDashboard"

    @staticmethod
    def _load_official_dashboard_api():
        workspace_root = Path(__file__).resolve().parents[2]
        sdk_dir = workspace_root / "TCP-IP-Python-V4"
        sdk_file = sdk_dir / "dobot_api.py"
        if not sdk_file.exists():
            raise DobotDashboardError(
                f"Official Dobot SDK not found: {sdk_file}. "
                "Put TCP-IP-Python-V4 next to dobot_teleop."
            )

        sdk_dir_str = str(sdk_dir)
        if sdk_dir_str not in sys.path:
            sys.path.insert(0, sdk_dir_str)

        try:
            from dobot_api import DobotApiDashboard
        except Exception as exc:
            raise DobotDashboardError(
                f"Failed to import official Dobot SDK from {sdk_dir}: {exc}"
            ) from exc

        return DobotApiDashboard

    def command(self, command: str) -> str:
        if self._api is None:
            raise DobotDashboardError("Dobot dashboard is not connected")

        line = command.strip()
        if self.verbose:
            print(f">> {line}")
        response = self._api.sendRecvMsg(line)
        if self.verbose:
            print(f"<< {response}")
        return response

    @staticmethod
    def error_id(response: str) -> Optional[int]:
        match = re.match(r"\s*(-?\d+)\s*,", response)
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def values(response: str) -> List[float]:
        match = re.search(r"\{([^}]*)\}", response)
        if not match:
            return []
        values = []
        for item in match.group(1).split(","):
            item = item.strip()
            if item:
                values.append(float(item))
        return values

    def require_ok(self, response: str, command_name: str) -> str:
        error_id = self.error_id(response)
        if error_id != 0:
            raise DobotDashboardError(
                f"{command_name} failed, error_id={error_id}, response={response}"
            )
        return response

    def enable_robot(self) -> str:
        response = self._api.EnableRobot()
        return self.require_ok(response, "EnableRobot")

    def clear_error(self) -> str:
        response = self._api.ClearError()
        return self.require_ok(response, "ClearError")

    def stop(self) -> str:
        return self._api.Stop()

    def emergency_stop(self) -> str:
        return self._api.EmergencyStop(1)

    def robot_mode(self) -> Optional[int]:
        response = self._api.RobotMode()
        self.require_ok(response, "RobotMode")
        values = self.values(response)
        return int(values[0]) if values else None

    def get_error(self, language: str = "zh_cn") -> Any:
        return self._api.GetError(language)

    def get_angle(self) -> List[float]:
        response = self._api.GetAngle()
        self.require_ok(response, "GetAngle")
        values = self.values(response)
        if len(values) < 6:
            raise DobotDashboardError(f"GetAngle returned no joint values: {response}")
        return values[:6]

    def inverse_kin(self, pose_mm_deg: Iterable[float]) -> List[float]:
        pose = list(pose_mm_deg)
        if len(pose) != 6:
            raise ValueError("InverseKin pose must contain 6 values")
        response = self._api.InverseKin(*pose)
        self.require_ok(response, "InverseKin")
        values = self.values(response)
        if len(values) < 6:
            raise DobotDashboardError(f"InverseKin returned no joint values: {response}")
        return values[:6]

    def get_pose(self, user: Optional[int] = None, tool: Optional[int] = None) -> List[float]:
        if user is None and tool is None:
            response = self._api.GetPose()
        elif user is not None and tool is not None:
            response = self._api.GetPose(user=user, tool=tool)
        else:
            raise ValueError("GetPose requires either both user/tool or neither")

        self.require_ok(response, "GetPose")
        values = self.values(response)
        if len(values) < 6:
            raise DobotDashboardError(f"GetPose returned no pose values: {response}")
        return values[:6]

    def servop(
        self,
        pose_mm_deg: Iterable[float],
        t: float,
        aheadtime: float,
        gain: float,
    ) -> str:
        pose = list(pose_mm_deg)
        if len(pose) != 6:
            raise ValueError("ServoP pose must contain 6 values")

        response = self._api.ServoP(
            pose[0],
            pose[1],
            pose[2],
            pose[3],
            pose[4],
            pose[5],
            t=t,
            aheadtime=aheadtime,
            gain=gain,
        )
        self.require_ok(response, "ServoP")
        return response


def format_pose(pose: Iterable[float]) -> str:
    values = list(pose)
    return (
        f"X={values[0]:.1f} Y={values[1]:.1f} Z={values[2]:.1f} "
        f"Rx={values[3]:.1f} Ry={values[4]:.1f} Rz={values[5]:.1f}"
    )
