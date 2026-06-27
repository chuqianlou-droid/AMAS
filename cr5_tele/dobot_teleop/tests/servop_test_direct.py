#!/usr/bin/env python3

import argparse
import math
import time

from dobot_teleop.dobot_dashboard import DobotDashboard, format_pose


def parse_args():
    parser = argparse.ArgumentParser(description="Direct Dobot ServoP smoke test.")
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--dashboard-port", type=int, default=29999)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.0)
    parser.add_argument("--speed-mm-s", type=float, default=50.0)
    parser.add_argument("--min-t", type=float, default=0.4)
    parser.add_argument("--max-t", type=float, default=0.8)
    parser.add_argument("--aheadtime", type=float, default=50.0)
    parser.add_argument("--gain", type=float, default=200.0)
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--clear-error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    client = DobotDashboard(args.robot_ip, args.dashboard_port)
    client.connect()
    try:
        print(f"Dobot dashboard connected via {client.backend_name()}.")
        if args.clear_error:
            print(client.clear_error())
        if args.enable_robot:
            print(client.enable_robot())

        current = client.get_pose()
        target = list(current)
        target[0] += args.dx
        target[1] += args.dy
        target[2] += args.dz

        distance = math.sqrt(args.dx * args.dx + args.dy * args.dy + args.dz * args.dz)
        t = distance / max(args.speed_mm_s, 1e-6)
        t = min(max(t, args.min_t), args.max_t)

        print(f"Current: {format_pose(current)}")
        print(f"Target : {format_pose(target)}")
        print(f"distance={distance:.1f} mm, t={t:.3f}s")
        print(client.servop(target, t=t, aheadtime=args.aheadtime, gain=args.gain))
        time.sleep(t + 0.2)
    finally:
        client.close()


if __name__ == "__main__":
    main()
