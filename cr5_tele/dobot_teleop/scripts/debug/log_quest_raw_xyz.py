#!/usr/bin/env python3
"""Record raw Quest/Unity UDP x/y/z poses for axis calibration."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import select
import socket
import sys
import time
from typing import Any


def _float_from(msg: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(msg.get(key, default))
    except (TypeError, ValueError):
        return default


def _button_value(msg: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key not in msg:
            continue
        value = msg[key]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _stdin_key_available() -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    return sys.stdin.readline().strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="UDP listen host.")
    parser.add_argument("--port", type=int, default=5005, help="UDP listen port.")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="CSV output path. Defaults to logs/quest_raw_xyz_<timestamp>.csv.",
    )
    parser.add_argument("--print-hz", type=float, default=10.0, help="Terminal print rate.")
    parser.add_argument("--timeout-s", type=float, default=1.0, help="Socket timeout.")
    args = parser.parse_args()

    if args.output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = pathlib.Path("logs") / f"quest_raw_xyz_{stamp}.csv"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.settimeout(args.timeout_s)

    print(f"Listening for raw Quest/Unity UDP on {args.host}:{args.port}")
    print(f"Writing CSV to {args.output.resolve()}")
    print("Move the controller along one physical axis at a time. Press Enter+r+Enter to reset origin, Ctrl+C to stop.")

    origin: tuple[float, float, float] | None = None
    seq = 0
    last_print_t = 0.0
    print_period = 1.0 / max(args.print_hz, 1e-6)

    fieldnames = [
        "seq",
        "wall_time",
        "mono_time",
        "src_ip",
        "src_port",
        "x_m",
        "y_m",
        "z_m",
        "dx_mm",
        "dy_mm",
        "dz_mm",
        "qx",
        "qy",
        "qz",
        "qw",
        "RG",
        "rightTrig",
        "raw_json",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                key = _stdin_key_available()
                if key == "r":
                    origin = None
                    print("Origin reset requested; next packet becomes the new zero.")

                try:
                    data, address = sock.recvfrom(65535)
                except socket.timeout:
                    continue

                try:
                    msg = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    print(f"Skipping non-JSON packet from {address}: {exc}")
                    continue

                now = time.time()
                mono = time.monotonic()
                x = _float_from(msg, "x", 0.0)
                y = _float_from(msg, "y", 0.0)
                z = _float_from(msg, "z", 0.0)
                if origin is None:
                    origin = (x, y, z)
                    print(f"Origin set to x={x:.4f} y={y:.4f} z={z:.4f}")

                dx_mm = (x - origin[0]) * 1000.0
                dy_mm = (y - origin[1]) * 1000.0
                dz_mm = (z - origin[2]) * 1000.0
                rg = _button_value(msg, "RG", "rightGrip", "RightGrip", "SecondaryHandTrigger")
                right_trig = _button_value(
                    msg,
                    "rightTrig",
                    "rightTrigger",
                    "rightIndexTrigger",
                    "trigger",
                    "IndexTrigger",
                    "RIndexTrigger",
                    "SecondaryIndexTrigger",
                )

                seq += 1
                writer.writerow(
                    {
                        "seq": seq,
                        "wall_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                        "mono_time": f"{mono:.6f}",
                        "src_ip": address[0],
                        "src_port": address[1],
                        "x_m": f"{x:.9f}",
                        "y_m": f"{y:.9f}",
                        "z_m": f"{z:.9f}",
                        "dx_mm": f"{dx_mm:.3f}",
                        "dy_mm": f"{dy_mm:.3f}",
                        "dz_mm": f"{dz_mm:.3f}",
                        "qx": f"{_float_from(msg, 'qx', 0.0):.9f}",
                        "qy": f"{_float_from(msg, 'qy', 0.0):.9f}",
                        "qz": f"{_float_from(msg, 'qz', 0.0):.9f}",
                        "qw": f"{_float_from(msg, 'qw', 1.0):.9f}",
                        "RG": f"{rg:.3f}",
                        "rightTrig": f"{right_trig:.3f}",
                        "raw_json": json.dumps(msg, ensure_ascii=False, sort_keys=True),
                    }
                )

                if mono - last_print_t >= print_period:
                    print(
                        f"seq={seq:06d} pos=({x:+.4f},{y:+.4f},{z:+.4f})m "
                        f"dpos=({dx_mm:+7.1f},{dy_mm:+7.1f},{dz_mm:+7.1f})mm "
                        f"RG={rg:.2f} trig={right_trig:.2f}"
                    )
                    last_print_t = mono

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            sock.close()

    print(f"Saved {seq} packets to {args.output.resolve()}")


if __name__ == "__main__":
    main()
