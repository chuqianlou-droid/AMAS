#!/usr/bin/env python3
"""
Reusable LeRobot v2.1 dataset writer for CR5A teleoperation data.

Can be used both during live recording (record_cr5a_pi0_dataset.py)
and for batch conversion (convert_to_lerobot.py).

Usage (recording):
    writer = LerobotWriter(output_dir, fps=15)
    writer.start_dataset()
    for each episode:
        writer.add_episode(
            obs_state=np.ndarray (N,13),   # joints(6)+pose(6)+gripper(1)
            actions=np.ndarray (N,7),       # [dx,dy,dz,dRx,dRy,dRz,gripper]
            d415_images=list[np.ndarray],   # (H,W,3) uint8
            d435_images=list[np.ndarray],   # (H,W,3) uint8
            prompt="Grab the bottle...",
        )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

# ── LeRobot v2.1 constants ─────────────────────────────────────────────────
CODEBASE_VERSION = "v2.1"
ROBOT_TYPE = "dobot_cr5a"
DEFAULT_CHUNK_SIZE = 1000

# observation.state = joints(6) + gripper(1) = 7
# （不包含 tcp_pose，因为它是 joints 的 FK 结果，属于冗余信息）
# action = [dx, dy, dz, dRx, dRy, dRz, gripper] = 7
FEATURES: dict = {
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"],
    },
    "observation.images.d415": {
        "dtype": "image",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.d435": {
        "dtype": "image",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
}

DEFAULT_FEATURES: dict = {
    "timestamp":    {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index":  {"dtype": "int64",   "shape": (1,), "names": None},
    "episode_index":{"dtype": "int64",   "shape": (1,), "names": None},
    "index":        {"dtype": "int64",   "shape": (1,), "names": None},
    "task_index":   {"dtype": "int64",   "shape": (1,), "names": None},
}

ALL_FEATURES = {**FEATURES, **DEFAULT_FEATURES}


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict: dict = {}
    for key, value in d.items():
        parts = key.split(sep)
        dd = outdict
        for part in parts[:-1]:
            if part not in dd:
                dd[part] = {}
            dd = dd[part]
        dd[parts[-1]] = value
    return outdict


class LerobotWriter:
    """Incrementally writes CR5A episodes into a LeRobot v2.1 dataset directory.

    Call ``start_dataset()`` to create a fresh dataset, or ``open_dataset()`` to
    append to an existing one. Then ``add_episode()`` for every episode.
    """

    def __init__(self, root: str | Path, fps: int = 15) -> None:
        self.root = Path(root)
        self.fps = fps
        self._started = False
        self._episode_index: int = 0
        self._total_frames: int = 0
        self._tasks: dict[int, str] = {}
        self._task_to_index: dict[str, int] = {}

    # ── public API ────────────────────────────────────────────────────────

    def start_dataset(self) -> None:
        """Create (or overwrite) the dataset directory and write initial metadata."""
        if self.root.exists():
            import shutil
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(exist_ok=True)
        self._write_initial_info()
        self._started = True

    def open_dataset(self) -> bool:
        """Open an existing dataset for appending episodes.

        Returns True if the dataset exists and was loaded successfully,
        False if it doesn't exist yet (caller should use start_dataset()).
        """
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            return False

        info = json.loads(info_path.read_text())
        self._episode_index = info["total_episodes"]
        self._total_frames = info["total_frames"]
        self.fps = info["fps"]

        # Load existing tasks
        tasks_path = self.root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._tasks[entry["task_index"]] = entry["task"]
                    self._task_to_index[entry["task"]] = entry["task_index"]

        self._started = True
        return True

    def add_episode(
        self,
        *,
        obs_state: np.ndarray,          # (N, 13) float32
        actions: np.ndarray,            # (N, 7) float32
        d415_images: Sequence[np.ndarray],  # list of (H, W, 3) uint8
        d435_images: Sequence[np.ndarray],  # list of (H, W, 3) uint8
        prompt: str = "",
    ) -> int:
        """Write one episode and return its episode_index.

        Args:
            obs_state:  (N, 13) — joints(6) + tcp_pose(6) + gripper(1)
            actions:    (N, 7)  — [dx, dy, dz, dRx, dRy, dRz, gripper]
            d415_images: N × (H, W, 3) uint8 numpy arrays
            d435_images: N × (H, W, 3) uint8 numpy arrays
            prompt:     task description string
        """
        if not self._started:
            raise RuntimeError("Call start_dataset() before add_episode()")

        n = int(obs_state.shape[0])
        ep_idx = self._episode_index
        global_offset = self._total_frames

        # ── resolve task ─────────────────────────────────────────────────
        if prompt and prompt not in self._task_to_index:
            task_idx = len(self._tasks)
            self._tasks[task_idx] = prompt
            self._task_to_index[prompt] = task_idx
            self._append_jsonlines(
                {"task_index": task_idx, "task": prompt}, "meta/tasks.jsonl"
            )
        task_idx = self._task_to_index.get(prompt, 0)

        # ── build parquet dict ────────────────────────────────────────────
        ep_dict = self._build_parquet_dict(
            obs_state, actions, d415_images, d435_images, n, ep_idx, global_offset, task_idx, self.fps
        )

        # ── compute stats ─────────────────────────────────────────────────
        ep_buffer = {
            "observation.state": obs_state,
            "action": actions,
        }
        ep_stats = self._compute_episode_stats(ep_buffer)
        serialized_stats = self._serialize_stats(ep_stats)

        # ── write parquet ─────────────────────────────────────────────────
        chunk = ep_idx // DEFAULT_CHUNK_SIZE
        parquet_dir = self.root / "data" / f"chunk-{chunk:03d}"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = parquet_dir / f"episode_{ep_idx:06d}.parquet"
        self._write_parquet(ep_dict, parquet_path)

        # ── update metadata ───────────────────────────────────────────────
        self._append_jsonlines(
            {"episode_index": ep_idx, "tasks": [prompt] if prompt else [], "length": n},
            "meta/episodes.jsonl",
        )
        self._append_jsonlines(
            {"episode_index": ep_idx, "stats": serialized_stats},
            "meta/episodes_stats.jsonl",
        )

        # Update info.json
        self._episode_index += 1
        self._total_frames += n
        self._update_info()

        # Re-compute global stats from all per-episode stats
        self._update_global_stats()

        return ep_idx

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_parquet_dict(
        obs_state: np.ndarray,
        actions: np.ndarray,
        d415_images: Sequence[np.ndarray],
        d435_images: Sequence[np.ndarray],
        n: int,
        ep_idx: int,
        global_offset: int,
        task_idx: int,
        fps: int = 15,
    ) -> dict:
        """Build the dict fed to HuggingFace Dataset.from_dict()."""
        from PIL import Image as PILImage

        obs_list = [obs_state[i] for i in range(n)]
        act_list = [actions[i] for i in range(n)]

        # numpy arrays → PIL Images (datasets.Image() expects PIL)
        d415_pil = [PILImage.fromarray(np.asarray(img, dtype=np.uint8), mode="RGB") for img in d415_images]
        d435_pil = [PILImage.fromarray(np.asarray(img, dtype=np.uint8), mode="RGB") for img in d435_images]

        # synthetic timestamps: frame_index / fps
        timestamps = [i / float(fps) for i in range(n)]
        frame_indices = list(range(n))
        ep_indices = [ep_idx] * n
        indices = list(range(global_offset, global_offset + n))
        task_indices = [task_idx] * n

        return {
            "observation.state": obs_list,
            "action": act_list,
            "observation.images.d415": d415_pil,
            "observation.images.d435": d435_pil,
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": ep_indices,
            "index": indices,
            "task_index": task_indices,
        }

    @staticmethod
    def _write_parquet(ep_dict: dict, output_path: Path) -> None:
        import datasets
        from datasets.table import embed_table_storage

        hf_features: dict = {}
        for key, ft in ALL_FEATURES.items():
            if ft["dtype"] == "image":
                hf_features[key] = datasets.Image()
            elif ft["shape"] == (1,):
                hf_features[key] = datasets.Value(dtype=ft["dtype"])
            elif len(ft["shape"]) == 1:
                hf_features[key] = datasets.Sequence(
                    length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
                )
            elif len(ft["shape"]) == 2:
                hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
            elif len(ft["shape"]) == 3:
                hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
            else:
                raise ValueError(f"Unsupported feature shape: {ft}")

        hf_features_obj = datasets.Features(hf_features)
        ep_dict_filtered = {k: v for k, v in ep_dict.items() if k in hf_features}
        ds = datasets.Dataset.from_dict(ep_dict_filtered, features=hf_features_obj, split="train")

        fmt = ds.format
        ds = ds.with_format("arrow")
        ds = ds.map(embed_table_storage, batched=False)
        ds = ds.with_format(**fmt)

        ds.to_parquet(str(output_path))

    @staticmethod
    def _compute_episode_stats(ep_buffer: dict) -> dict:
        """Compute per-episode stats matching LeRobot's get_feature_stats format."""
        stats = {}
        for key, ft in FEATURES.items():
            if ft["dtype"] in ("image", "video"):
                continue
            values = ep_buffer[key]
            if isinstance(values, list):
                values = np.array(values)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            n = values.shape[0]
            keepdims = values.ndim == 1
            stats[key] = {
                "min": values.min(axis=0, keepdims=keepdims),
                "max": values.max(axis=0, keepdims=keepdims),
                "mean": values.mean(axis=0, keepdims=keepdims),
                "std": values.std(axis=0, keepdims=keepdims),
                "count": np.array([n]),
            }
        return stats

    @staticmethod
    def _serialize_stats(stats: dict) -> dict:
        """Convert numpy arrays in stats to lists/dicts for JSON."""
        serialized = {}
        for key, value in _flatten_dict(stats).items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = value.item()
            elif isinstance(value, (int, float)):
                serialized[key] = value
            else:
                raise TypeError(f"Unsupported type in stats: {type(value)}")
        return _unflatten_dict(serialized)

    def _load_all_episode_stats(self) -> list[dict] | None:
        """Load all per-episode stats from episodes_stats.jsonl."""
        path = self.root / "meta" / "episodes_stats.jsonl"
        if not path.exists():
            return None

        entries = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                stats = entry["stats"]
                stats = {
                    key: np.array(value) for key, value in _flatten_dict(stats).items()
                }
                entries.append(_unflatten_dict(stats))
        return entries

    def _update_global_stats(self) -> None:
        """Recompute aggregated stats from all per-episode stats."""
        all_stats = self._load_all_episode_stats()
        if all_stats is None or len(all_stats) == 0:
            return

        feature_names = list(all_stats[0].keys())
        aggregated = {}
        for key in feature_names:
            mins = np.stack([s[key]["min"] for s in all_stats])
            maxs = np.stack([s[key]["max"] for s in all_stats])
            means = np.stack([s[key]["mean"] for s in all_stats])
            stds = np.stack([s[key]["std"] for s in all_stats])

            aggregated[key] = {
                "min": mins.min(axis=0),
                "max": maxs.max(axis=0),
                "mean": means.mean(axis=0),
                "std": stds.mean(axis=0),
            }

        self._write_json(self._serialize_stats(aggregated), "meta/stats.json")

    def _write_initial_info(self) -> None:
        """Write a fresh info.json with zero totals."""
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": ROBOT_TYPE,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": DEFAULT_CHUNK_SIZE,
            "fps": self.fps,
            "splits": {"train": "0:0"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": ALL_FEATURES,
        }
        self._write_json(info, "meta/info.json")

    def _update_info(self) -> None:
        """Rewrite info.json with current totals."""
        total_chunks = max(1, (self._episode_index - 1) // DEFAULT_CHUNK_SIZE + 1) if self._episode_index > 0 else 1
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": ROBOT_TYPE,
            "total_episodes": self._episode_index,
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": 0,
            "total_chunks": total_chunks,
            "chunks_size": DEFAULT_CHUNK_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{self._episode_index}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": ALL_FEATURES,
        }
        self._write_json(info, "meta/info.json")

    # ── file I/O helpers ──────────────────────────────────────────────────

    def _write_json(self, data: dict, rel_path: str) -> None:
        fpath = self.root / rel_path
        fpath.parent.mkdir(exist_ok=True, parents=True)
        with open(fpath, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _append_jsonlines(self, data: dict, rel_path: str) -> None:
        fpath = self.root / rel_path
        fpath.parent.mkdir(exist_ok=True, parents=True)
        with open(fpath, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
