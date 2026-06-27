#!/usr/bin/env python3
"""
Convert CR5A teleoperation dataset (cr5a_pi0_raw_v1 format) to LeRobot v2.1 format.

This is a batch conversion tool.  For live recording directly into LeRobot format,
use record_cr5a_pi0_dataset.py --format lerobot.

Usage:
    # 使用默认路径
    python3 convert_to_lerobot.py

    # 指定输入/输出路径
    python3 convert_to_lerobot.py --input-dir datasets/cr5a_demo_test --output-dir datasets/cr5a_lerobot
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 让 scripts/dataset/ 下的脚本能找到 dobot_teleop 包
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
from PIL import Image as PILImage

from dobot_teleop.lerobot_writer import LerobotWriter

FPS = 15


def load_episode(ep_dir: Path) -> dict:
    """Load a single episode from the raw cr5a_pi0_raw_v1 format."""
    meta = json.loads((ep_dir / "metadata.json").read_text())
    steps = np.load(ep_dir / "steps.npz", allow_pickle=True)

    n_frames = int(steps["tcp_pose"].shape[0])

    # Build observation.state: joints(6) + gripper(1) = 7
    # （不包含 tcp_pose，它是 joints 的 FK 结果，action 里也有位姿信息）
    joints = steps["joints"]
    gripper = steps["gripper"].reshape(-1, 1)
    obs_state = np.concatenate([joints, gripper], axis=1).astype(np.float32)

    actions = steps["actions"].astype(np.float32)

    # Load images as numpy arrays
    d415_dir = ep_dir / "images_d415"
    d435_dir = ep_dir / "images_d435"
    d415_paths = sorted(d415_dir.glob("*.png"))
    d435_paths = sorted(d435_dir.glob("*.png"))

    assert len(d415_paths) == n_frames, (
        f"d415 image count mismatch in {ep_dir.name}: "
        f"{len(d415_paths)} images vs {n_frames} steps"
    )
    assert len(d435_paths) == n_frames, (
        f"d435 image count mismatch in {ep_dir.name}: "
        f"{len(d435_paths)} images vs {n_frames} steps"
    )

    # Load all images into numpy arrays
    d415_imgs = [np.array(PILImage.open(p).convert("RGB"), dtype=np.uint8) for p in d415_paths]
    d435_imgs = [np.array(PILImage.open(p).convert("RGB"), dtype=np.uint8) for p in d435_paths]

    return {
        "n_frames": n_frames,
        "prompt": meta.get("prompt", ""),
        "obs_state": obs_state,
        "actions": actions,
        "d415_images": d415_imgs,
        "d435_images": d435_imgs,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert CR5A raw dataset to LeRobot v2.1 format")
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path(_PROJ_ROOT) / "datasets" / "cr5a_demo_test",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(_PROJ_ROOT) / "datasets" / "cr5a_lerobot",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CR5A Raw → LeRobot v2.1 Converter")
    print("=" * 60)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    # ── Discover episodes ──────────────────────────────────────────────────
    episode_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
    )
    if not episode_dirs:
        print(f"ERROR: No episode directories found in {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(episode_dirs)} episodes in {input_dir}")

    # ── Load all episodes ──────────────────────────────────────────────────
    all_episodes = []
    total_frames = 0
    for ep_dir in episode_dirs:
        ep_data = load_episode(ep_dir)
        all_episodes.append(ep_data)
        print(f"  {ep_dir.name}: {ep_data['n_frames']} frames, "
              f"prompt='{ep_data['prompt'][:50]}'")
        total_frames += ep_data["n_frames"]

    print(f"\nTotal frames: {total_frames}")

    # ── Write to LeRobot format ─────────────────────────────────────────────
    writer = LerobotWriter(output_dir, fps=FPS)
    writer.start_dataset()

    for ep_data in all_episodes:
        ep_idx = writer.add_episode(
            obs_state=ep_data["obs_state"],
            actions=ep_data["actions"],
            d415_images=ep_data["d415_images"],
            d435_images=ep_data["d435_images"],
            prompt=ep_data["prompt"],
        )
        print(f"  → episode_{ep_idx:06d}.parquet ({ep_data['n_frames']} frames)")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output: {output_dir}")
    print(f"Episodes: {len(all_episodes)}")
    print(f"Total frames: {total_frames}")
    print("=" * 60)


if __name__ == "__main__":
    main()
