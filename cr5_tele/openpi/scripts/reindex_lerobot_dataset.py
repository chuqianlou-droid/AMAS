#!/usr/bin/env python3
"""Reindex an existing LeRobot dataset after removing missing episodes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return root / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if name not in table.column_names:
        return table
    idx = table.column_names.index(name)
    return table.set_column(idx, name, pa.array(values))


def _fix_huggingface_metadata(table: pa.Table) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    raw_hf_metadata = metadata.get(b"huggingface")
    if raw_hf_metadata is None:
        return table

    hf_metadata = json.loads(raw_hf_metadata)

    def fix_types(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("_type") == "List":
                value["_type"] = "Sequence"
            for child in value.values():
                fix_types(child)
        elif isinstance(value, list):
            for child in value:
                fix_types(child)

    fix_types(hf_metadata)
    metadata[b"huggingface"] = json.dumps(hf_metadata).encode()
    return table.replace_schema_metadata(metadata)


def _rewrite_episode_parquet(src: Path, dst: Path, new_episode_index: int, global_offset: int) -> int:
    table = pq.read_table(src)
    num_rows = table.num_rows
    table = _replace_column(table, "episode_index", np.full(num_rows, new_episode_index, dtype=np.int64))
    table = _replace_column(table, "frame_index", np.arange(num_rows, dtype=np.int64))
    table = _replace_column(table, "index", np.arange(global_offset, global_offset + num_rows, dtype=np.int64))
    table = _fix_huggingface_metadata(table)

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return num_rows


def _aggregate_stats(episode_stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not episode_stats:
        return {}

    keys = list(episode_stats[0]["stats"].keys())
    aggregated: dict[str, Any] = {}
    for key in keys:
        mins, maxs, means, stds, counts = [], [], [], [], []
        for row in episode_stats:
            stats = row["stats"][key]
            count = np.asarray(stats.get("count", [1]), dtype=np.float64)
            mins.append(np.asarray(stats["min"], dtype=np.float64))
            maxs.append(np.asarray(stats["max"], dtype=np.float64))
            means.append(np.asarray(stats["mean"], dtype=np.float64))
            stds.append(np.asarray(stats["std"], dtype=np.float64))
            counts.append(float(count.reshape(-1)[0]))

        counts_arr = np.asarray(counts, dtype=np.float64)
        means_arr = np.stack(means)
        stds_arr = np.stack(stds)
        weights = counts_arr.reshape((len(counts_arr),) + (1,) * (means_arr.ndim - 1))
        mean = (means_arr * weights).sum(axis=0) / counts_arr.sum()
        second_moment = ((stds_arr**2 + means_arr**2) * weights).sum(axis=0) / counts_arr.sum()

        aggregated[key] = {
            "min": np.stack(mins).min(axis=0).tolist(),
            "max": np.stack(maxs).max(axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(second_moment - mean**2, 0.0)).tolist(),
        }

    return aggregated


def reindex_dataset(input_dir: Path, output_dir: Path, *, overwrite: bool = False) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("Refusing to reindex in place. Use a different --output-dir.")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    info = _read_json(input_dir / "meta" / "info.json")
    episodes = {row["episode_index"]: row for row in _read_jsonl(input_dir / "meta" / "episodes.jsonl")}
    episode_stats = {row["episode_index"]: row for row in _read_jsonl(input_dir / "meta" / "episodes_stats.jsonl")}
    tasks = _read_jsonl(input_dir / "meta" / "tasks.jsonl")

    existing_old_indices = []
    missing_old_indices = []
    for old_idx in range(int(info["total_episodes"])):
        if _data_path(input_dir, info, old_idx).is_file():
            existing_old_indices.append(old_idx)
        else:
            missing_old_indices.append(old_idx)

    if not existing_old_indices:
        raise ValueError(f"No episode parquet files found under {input_dir}")

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(existing_old_indices)} existing episodes")
    if missing_old_indices:
        print("Skipping missing episodes:", ", ".join(str(i) for i in missing_old_indices))

    new_episodes = []
    new_episode_stats = []
    total_frames = 0
    output_info = dict(info)
    output_info["total_episodes"] = 0
    output_info["total_frames"] = 0
    output_info["total_chunks"] = 1
    output_info["splits"] = {"train": "0:0"}

    for new_idx, old_idx in enumerate(existing_old_indices):
        src = _data_path(input_dir, info, old_idx)
        dst = _data_path(output_dir, output_info, new_idx)
        num_rows = _rewrite_episode_parquet(src, dst, new_idx, total_frames)

        episode_row = dict(episodes.get(old_idx, {"tasks": [], "length": num_rows}))
        episode_row["episode_index"] = new_idx
        episode_row["length"] = num_rows
        new_episodes.append(episode_row)

        if old_idx in episode_stats:
            stats_row = dict(episode_stats[old_idx])
            stats_row["episode_index"] = new_idx
            new_episode_stats.append(stats_row)

        print(f"  old {old_idx:06d} -> new {new_idx:06d}: {num_rows} frames")
        total_frames += num_rows

    chunks_size = int(output_info["chunks_size"])
    total_episodes = len(existing_old_indices)
    output_info["total_episodes"] = total_episodes
    output_info["total_frames"] = total_frames
    output_info["total_chunks"] = max(1, (total_episodes - 1) // chunks_size + 1)
    output_info["splits"] = {"train": f"0:{total_episodes}"}

    output_meta = output_dir / "meta"
    _write_json(output_meta / "info.json", output_info)
    _write_jsonl(output_meta / "episodes.jsonl", new_episodes)
    _write_jsonl(output_meta / "episodes_stats.jsonl", new_episode_stats)
    _write_jsonl(output_meta / "tasks.jsonl", tasks)
    _write_json(output_meta / "stats.json", _aggregate_stats(new_episode_stats))

    print("Done.")
    print(f"total_episodes = {total_episodes}")
    print(f"total_frames = {total_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        reindex_dataset(args.input_dir, args.output_dir, overwrite=args.overwrite)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
