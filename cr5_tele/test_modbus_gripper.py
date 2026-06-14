#!/usr/bin/env python3
import argparse
import re
import socket
import sys
import time
from typing import List, Optional


REG_INIT_CMD = 0x0100
REG_FORCE = 0x0101
REG_POSITION = 0x0103
REG_SPEED = 0x0104

REG_INIT_STATUS = 0x0200
REG_GRIP_STATUS = 0x0201
REG_CURRENT_POSITION = 0x0202

DEFAULT_OPEN_POSITION = 1000
DEFAULT_CLOSE_POSITION = 0


class DashboardError(RuntimeError):
    pass


class DobotDashboardSocket:
    def __init__(self, ip: str, port: int, timeout: float, command_wait: float):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.command_wait = command_wait
        self.sock: Optional[socket.socket] = None

    def __enter__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        print(f"\n连接机器人 Dashboard: {self.ip}:{self.port}")
        self.sock.connect((self.ip, self.port))
        print("连接成功。")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def send(self, cmd: str, require_ok: bool = True) -> str:
        if self.sock is None:
            raise DashboardError("Dashboard socket is not connected")

        print(f">>> {cmd}")
        self.sock.sendall(cmd.encode("utf-8"))
        time.sleep(self.command_wait)

        chunks = []
        while True:
            try:
                data = self.sock.recv(4096)
            except socket.timeout:
                break

            if not data:
                break

            text = data.decode("utf-8", errors="ignore")
            chunks.append(text)
            if ";" in text:
                break

        response = "".join(chunks).strip()
        print(f"<<< {response}")

        if require_ok:
            error_id = parse_error_id(response)
            if error_id != 0:
                raise DashboardError(
                    f"命令失败: {cmd}, error_id={error_id}, response={response}"
                )

        return response


def parse_error_id(response: str) -> Optional[int]:
    match = re.match(r"\s*(-?\d+)\s*,", response or "")
    if not match:
        return None
    return int(match.group(1))


def parse_values(response: str) -> List[int]:
    match = re.search(r"\{([^}]*)\}", response or "")
    if not match:
        return []

    values = []
    for item in match.group(1).split(","):
        item = item.strip()
        if item:
            values.append(int(float(item)))
    return values


def setup_modbus(client: DobotDashboardSocket, args) -> int:
    print("\n=== 初始化 Dobot 末端 RS485 / Modbus-RTU 通道 ===")
    client.send("SetToolPower(1)")
    time.sleep(args.power_delay)

    client.send("SetToolMode(1)")
    client.send(f'SetTool485({args.baud},"{args.parity}",{args.stop_bit})')

    response = client.send(
        "ModbusRTUCreate("
        f'{args.slave_id},{args.baud},"{args.parity}",{args.data_bit},{args.stop_bit}'
        ")"
    )
    values = parse_values(response)
    if not values:
        raise DashboardError(f"无法解析 ModbusRTUCreate 返回的 index: {response}")

    index = values[0]
    print(f"Modbus master index: {index}")
    return index


def write_u16(client: DobotDashboardSocket, index: int, addr: int, value: int) -> None:
    client.send(f"SetHoldRegs({index},{addr},1,{{{value}}},U16)")


def read_u16(client: DobotDashboardSocket, index: int, addr: int) -> Optional[int]:
    response = client.send(f"GetHoldRegs({index},{addr},1,U16)")
    values = parse_values(response)
    return values[0] if values else None


def initialize_gripper(client: DobotDashboardSocket, index: int, args) -> None:
    print("\n=== 初始化夹爪 ===")
    write_u16(client, index, REG_INIT_CMD, args.init_value)

    deadline = time.monotonic() + args.init_timeout
    while time.monotonic() < deadline:
        time.sleep(args.status_interval)
        status = read_u16(client, index, REG_INIT_STATUS)
        print(f"初始化状态: {status_text(status)} ({status})")
        if status == 1:
            print("夹爪初始化成功。")
            return

    raise DashboardError("等待夹爪初始化成功超时")


def configure_motion(client: DobotDashboardSocket, index: int, args) -> None:
    print("\n=== 设置夹爪力值和速度 ===")
    write_u16(client, index, REG_FORCE, args.force)
    write_u16(client, index, REG_SPEED, args.speed)


def move_gripper(
    client: DobotDashboardSocket,
    index: int,
    position: int,
    label: str,
    args,
) -> None:
    print(f"\n=== {label}: position={position} ===")
    write_u16(client, index, REG_POSITION, position)
    time.sleep(args.move_wait)
    print_feedback(client, index)


def print_feedback(client: DobotDashboardSocket, index: int) -> None:
    print("\n=== 读取夹爪反馈 ===")
    init_status = read_u16(client, index, REG_INIT_STATUS)
    grip_status = read_u16(client, index, REG_GRIP_STATUS)
    current_pos = read_u16(client, index, REG_CURRENT_POSITION)
    print(f"初始化状态: {status_text(init_status)} ({init_status})")
    print(f"夹持状态  : {grip_text(grip_status)} ({grip_status})")
    print(f"当前位置  : {current_pos}")


def status_text(value: Optional[int]) -> str:
    return {
        0: "未初始化",
        1: "初始化成功",
        2: "初始化中",
    }.get(value, "未知")


def grip_text(value: Optional[int]) -> str:
    return {
        0: "运动中",
        1: "到达位置",
        2: "夹住物体",
        3: "物体掉落",
    }.get(value, "未知")


def clamp_args(args) -> None:
    if not 20 <= args.force <= 100:
        raise ValueError("--force 必须在 20..100 之间")
    if not 1 <= args.speed <= 100:
        raise ValueError("--speed 必须在 1..100 之间")
    if not 0 <= args.open_position <= 1000:
        raise ValueError("--open-position 必须在 0..1000 之间")
    if not 0 <= args.close_position <= 1000:
        raise ValueError("--close-position 必须在 0..1000 之间")
    if args.init_value not in (1, 0xA5):
        raise ValueError("--init-value 只能是 1 或 165(0xA5)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dobot CR5A 末端 RS485 / Modbus-RTU PGE 夹爪测试脚本"
    )
    parser.add_argument("--ip", required=True, help="机器人 IP，例如 192.168.5.1")
    parser.add_argument("--port", type=int, default=29999, help="Dashboard 端口")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--init", action="store_true", help="初始化夹爪并读取反馈")
    action.add_argument("--open", action="store_true", help="打开夹爪")
    action.add_argument("--close", action="store_true", help="闭合夹爪")
    action.add_argument("--cycle", action="store_true", help="初始化 -> 打开 -> 闭合 -> 打开")
    action.add_argument("--status", action="store_true", help="只读取夹爪反馈")

    parser.add_argument("--slave-id", type=int, default=1)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--parity", choices=("N", "E", "O"), default="N")
    parser.add_argument("--data-bit", type=int, default=8)
    parser.add_argument("--stop-bit", type=int, choices=(1, 2), default=1)

    parser.add_argument("--force", type=int, default=50, help="力值，默认 50")
    parser.add_argument("--speed", type=int, default=50, help="速度，默认 50")
    parser.add_argument("--open-position", type=int, default=DEFAULT_OPEN_POSITION)
    parser.add_argument("--close-position", type=int, default=DEFAULT_CLOSE_POSITION)
    parser.add_argument(
        "--init-value",
        type=lambda value: int(value, 0),
        default=1,
        help="初始化写入值：1=单方向初始化，0xA5=完整初始化",
    )

    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--command-wait", type=float, default=0.1)
    parser.add_argument("--power-delay", type=float, default=0.5)
    parser.add_argument("--init-timeout", type=float, default=8.0)
    parser.add_argument("--status-interval", type=float, default=0.5)
    parser.add_argument("--move-wait", type=float, default=1.0)
    parser.add_argument("--cycle-wait", type=float, default=1.0)
    return parser.parse_args()


def print_header(args) -> None:
    print("===================================")
    print("Dobot CR5A 末端 RS485 Modbus 夹爪测试脚本")
    print("===================================")
    print(f"Robot IP : {args.ip}")
    print(f"Port     : {args.port}")
    print(f"Slave ID : {args.slave_id}")
    print(f"Serial   : {args.baud},{args.data_bit},{args.parity},{args.stop_bit}")
    print(f"Force    : {args.force}")
    print(f"Speed    : {args.speed}")
    print("===================================")


def main() -> int:
    args = parse_args()
    clamp_args(args)
    print_header(args)
    input("确认夹爪附近没有手指和障碍物后，按 Enter 开始测试...")

    try:
        with DobotDashboardSocket(
            args.ip,
            args.port,
            timeout=args.timeout,
            command_wait=args.command_wait,
        ) as client:
            index = None
            try:
                index = setup_modbus(client, args)

                if args.status:
                    print_feedback(client, index)
                    return 0

                if args.init:
                    initialize_gripper(client, index, args)
                    print_feedback(client, index)
                    return 0

                if args.open:
                    configure_motion(client, index, args)
                    move_gripper(client, index, args.open_position, "打开夹爪", args)
                    return 0

                if args.close:
                    configure_motion(client, index, args)
                    move_gripper(client, index, args.close_position, "闭合夹爪", args)
                    return 0

                if args.cycle:
                    initialize_gripper(client, index, args)
                    configure_motion(client, index, args)
                    move_gripper(client, index, args.open_position, "打开夹爪", args)
                    time.sleep(args.cycle_wait)
                    move_gripper(client, index, args.close_position, "闭合夹爪", args)
                    time.sleep(args.cycle_wait)
                    move_gripper(client, index, args.open_position, "再次打开夹爪", args)
                    return 0
            finally:
                if index is not None:
                    print("\n=== 释放 Modbus 主站 ===")
                    client.send(f"ModbusClose({index})", require_ok=False)

    except (DashboardError, OSError, ValueError) as exc:
        print(f"\n错误: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
