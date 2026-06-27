#!/usr/bin/env python3
"""Inspect a raw CR5A PI0 episode and validate its on-disk schema."""

from __future__ import annotations

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

from dobot_teleop.cr5a_pi0_schema import CR5A_ACTION_DIM


REQUIRED_STEP_KEYS = ("tcp_pose", "joints", "gripper", "actions", "timestamps")
ACTION_METADATA_KEYS = ("action_timestamps", "action_age_ms", "deadman", "servo_sent", "gripper_action")


def _read_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to inspect PNG frames. Install it with: pip install Pillow") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_episode(episode_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata_path = episode_dir / "metadata.json"
    steps_path = episode_dir / "steps.npz"
    if not metadata_path.is_file() or not steps_path.is_file():
        raise FileNotFoundError("Episode must contain metadata.json and steps.npz")
    metadata = json.loads(metadata_path.read_text())
    with np.load(steps_path) as steps:
        missing = [key for key in REQUIRED_STEP_KEYS if key not in steps]
        if missing:
            raise ValueError(f"steps.npz is missing required keys: {missing}")
        arrays = {key: np.asarray(steps[key]) for key in REQUIRED_STEP_KEYS}
        arrays.update({key: np.asarray(steps[key]) for key in ACTION_METADATA_KEYS if key in steps})
    return metadata, arrays


def validate_episode(episode_dir: Path) -> dict[str, np.ndarray]:
    _metadata, arrays = load_episode(episode_dir)
    frame_count = arrays["timestamps"].shape[0]
    expected_shapes = {
        "tcp_pose": (frame_count, 6),
        "joints": (frame_count, 6),
        "gripper": (frame_count,),
        "actions": (frame_count, CR5A_ACTION_DIM),
    }
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise ValueError(f"{key} has shape {arrays[key].shape}; expected {expected}")
    for key in ACTION_METADATA_KEYS:
        if key in arrays and arrays[key].shape != (frame_count,):
            raise ValueError(f"{key} has shape {arrays[key].shape}; expected {(frame_count,)}")
    for key, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key} contains NaN or inf")
    for camera in ("d415", "d435"):
        frames = sorted((episode_dir / f"images_{camera}").glob("*.png"))
        if len(frames) != frame_count:
            raise ValueError(f"images_{camera} has {len(frames)} frames; expected {frame_count}")
    return arrays


def _make_preview(episode_dir: Path) -> np.ndarray:
    d435 = _read_image(sorted((episode_dir / "images_d435").glob("*.png"))[0])
    d415 = _read_image(sorted((episode_dir / "images_d415").glob("*.png"))[0])
    if d435.shape[0] != d415.shape[0]:
        raise ValueError(f"Cannot concatenate image heights {d435.shape[0]} and {d415.shape[0]}")
    return np.concatenate([d435, d415], axis=1)


def _display_preview(preview: np.ndarray, *, show: bool, save_path: Path | None) -> None:
    if save_path is not None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to save a preview PNG") from exc
        Image.fromarray(preview, mode="RGB").save(save_path)
        print(f"Preview written to {save_path}")
    if show:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Preview display skipped: matplotlib is not installed. Use --save-preview for headless inspection.")
            return
        plt.figure("CR5A D435 | D415")
        plt.imshow(preview)
        plt.axis("off")
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--save-preview", type=Path)
    parser.add_argument("--no-show", action="store_true", help="Validate and print only; useful on headless hosts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata, arrays = load_episode(args.episode_dir)
    arrays = validate_episode(args.episode_dir)
    preview = _make_preview(args.episode_dir)
    actions = arrays["actions"]
    print(f"episode: {args.episode_dir}")
    print(f"episode length: {len(actions)}")
    print(f"D435/D415 image size: {preview.shape[0]} x {preview.shape[1] // 2} RGB")
    print(f"tcp_pose shape: {arrays['tcp_pose'].shape}")
    print(f"joints shape: {arrays['joints'].shape}")
    print(f"action shape: {actions.shape}")
    print(f"action min: {np.array2string(actions.min(axis=0), precision=4)}")
    print(f"action max: {np.array2string(actions.max(axis=0), precision=4)}")
    if "action_age_ms" in arrays:
        print(f"action age ms: {arrays['action_age_ms'].min():.1f} .. {arrays['action_age_ms'].max():.1f}")
    if "deadman" in arrays:
        print(f"deadman frames: {int(np.count_nonzero(arrays['deadman']))}/{len(actions)}")
    if "servo_sent" in arrays:
        print(f"Servo sent frames: {int(np.count_nonzero(arrays['servo_sent']))}/{len(actions)}")
    print(f"prompt: {metadata.get('prompt', '')}")
    _display_preview(preview, show=not args.no_show, save_path=args.save_preview)


if __name__ == "__main__":
    main()
