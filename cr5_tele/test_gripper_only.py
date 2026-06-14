#!/usr/bin/env python3
import argparse
import socket
import time
import re


def send_cmd(sock, cmd, wait=0.1):
    """
    向 Dobot Dashboard 端口发送一条命令，并打印返回值。
    """
    print(f">>> {cmd}")
    sock.sendall(cmd.encode("utf-8"))

    time.sleep(wait)

    try:
        data = sock.recv(4096).decode("utf-8", errors="ignore").strip()
    except socket.timeout:
        data = ""

    print(f"<<< {data}")
    return data


def is_success(resp):
    """
    Dobot 通常成功返回类似：
    0,{...},Command();
    第一个数字为 0 表示成功。
    """
    if not resp:
        return False

    m = re.match(r"\s*(-?\d+)\s*,", resp)
    if m:
        return int(m.group(1)) == 0

    return "0," in resp or "OK" in resp.upper()


def test_tool_gripper(sock, do_index, cycles, interval):
    """
    测试机械臂末端航插 Tool DO。
    适用于夹爪接在机械臂末端法兰旁边的小圆形航插。
    """
    print("\n=== 测试模式：末端航插 Tool DO ===")

    send_cmd(sock, "SetToolPower(1)")
    time.sleep(0.5)

    # 先尝试 ToolDOInstant，如果不支持，再尝试 ToolDOExecute
    close_cmd_1 = f"ToolDOInstant({do_index},1)"
    open_cmd_1 = f"ToolDOInstant({do_index},0)"

    close_cmd_2 = f"ToolDOExecute({do_index},1)"
    open_cmd_2 = f"ToolDOExecute({do_index},0)"

    print("\n先测试命令是否可用...")
    resp = send_cmd(sock, open_cmd_1)

    if is_success(resp):
        close_cmd = close_cmd_1
        open_cmd = open_cmd_1
        print("使用命令：ToolDOInstant")
    else:
        print("ToolDOInstant 可能不支持，改试 ToolDOExecute")
        resp = send_cmd(sock, open_cmd_2)
        close_cmd = close_cmd_2
        open_cmd = open_cmd_2
        print("使用命令：ToolDOExecute")

    print("\n开始循环开合夹爪，请注意手指不要靠近夹爪。")
    for i in range(cycles):
        print(f"\n--- 第 {i + 1}/{cycles} 次 ---")

        print("夹爪闭合")
        send_cmd(sock, close_cmd)
        time.sleep(interval)

        print("夹爪张开")
        send_cmd(sock, open_cmd)
        time.sleep(interval)

    print("\n测试结束，最后保持夹爪张开。")
    send_cmd(sock, open_cmd)


def test_cabinet_gripper(sock, do_index, cycles, interval):
    """
    测试控制柜 DO。
    适用于夹爪接在控制柜 I/O 端子排 DO1/DO2 上。
    """
    print("\n=== 测试模式：控制柜 DO ===")

    # 先尝试 DOInstant，如果不支持，再尝试 DOExecute
    close_cmd_1 = f"DOInstant({do_index},1)"
    open_cmd_1 = f"DOInstant({do_index},0)"

    close_cmd_2 = f"DOExecute({do_index},1)"
    open_cmd_2 = f"DOExecute({do_index},0)"

    print("\n先测试命令是否可用...")
    resp = send_cmd(sock, open_cmd_1)

    if is_success(resp):
        close_cmd = close_cmd_1
        open_cmd = open_cmd_1
        print("使用命令：DOInstant")
    else:
        print("DOInstant 可能不支持，改试 DOExecute")
        resp = send_cmd(sock, open_cmd_2)
        close_cmd = close_cmd_2
        open_cmd = open_cmd_2
        print("使用命令：DOExecute")

    print("\n开始循环开合夹爪，请注意手指不要靠近夹爪。")
    for i in range(cycles):
        print(f"\n--- 第 {i + 1}/{cycles} 次 ---")

        print("夹爪闭合")
        send_cmd(sock, close_cmd)
        time.sleep(interval)

        print("夹爪张开")
        send_cmd(sock, open_cmd)
        time.sleep(interval)

    print("\n测试结束，最后保持夹爪张开。")
    send_cmd(sock, open_cmd)


def main():
    parser = argparse.ArgumentParser(description="Dobot CR5A gripper only test")
    parser.add_argument("--ip", required=True, help="机器人 IP，例如 192.168.5.1")
    parser.add_argument("--port", type=int, default=29999, help="Dashboard 端口，默认 29999")
    parser.add_argument(
        "--mode",
        choices=["tool", "cabinet"],
        required=True,
        help="tool=末端航插，cabinet=控制柜 DO",
    )
    parser.add_argument("--do", type=int, default=1, help="DO 编号，默认 1")
    parser.add_argument("--cycles", type=int, default=3, help="开合次数，默认 3")
    parser.add_argument("--interval", type=float, default=1.0, help="开合间隔秒数，默认 1.0")

    args = parser.parse_args()

    print("===================================")
    print("Dobot CR5A 夹爪单独测试脚本")
    print("===================================")
    print(f"Robot IP : {args.ip}")
    print(f"Port     : {args.port}")
    print(f"Mode     : {args.mode}")
    print(f"DO index : {args.do}")
    print("===================================")

    input("确认夹爪附近没有手指和障碍物后，按 Enter 开始测试...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        print(f"\n连接机器人 Dashboard: {args.ip}:{args.port}")
        sock.connect((args.ip, args.port))
        print("连接成功。")

        if args.mode == "tool":
            test_tool_gripper(sock, args.do, args.cycles, args.interval)
        else:
            test_cabinet_gripper(sock, args.do, args.cycles, args.interval)

    print("\n脚本结束。")


if __name__ == "__main__":
    main()
