#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ffprobe_num_frames(video_path: Path) -> int | None:
    if shutil.which("ffprobe") is None:
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]

    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return int(out)
    except Exception:
        return None


def validate_dataset(dataset_dir: Path) -> None:
    dataset_dir = dataset_dir.expanduser()

    meta_dir = dataset_dir / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    modality_path = meta_dir / "modality.json"

    required = [info_path, episodes_path, tasks_path, modality_path]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required meta file: {p}")

    info = json.loads(info_path.read_text())
    episodes = read_jsonl(episodes_path)
    tasks = read_jsonl(tasks_path)

    state_dim = info["features"]["observation.state"]["shape"][0]
    action_dim = info["features"]["action"]["shape"][0]
    image_shape = info["features"]["observation.images.ego_view"]["shape"]
    chunks_size = int(info.get("chunks_size", 1000))

    total_rows = 0
    errors = []

    print("[dataset]", dataset_dir)
    print("[meta]")
    print("  total_episodes:", info.get("total_episodes"))
    print("  total_frames:", info.get("total_frames"))
    print("  total_tasks:", info.get("total_tasks"))
    print("  fps:", info.get("fps"))
    print("  state_dim:", state_dim)
    print("  action_dim:", action_dim)
    print("  image_shape:", image_shape)
    print("  tasks:", [t["task"] for t in tasks])

    for ep in episodes:
        episode_index = int(ep["episode_index"])
        expected_len = int(ep["length"])
        episode_chunk = episode_index // chunks_size

        parquet_path = (
            dataset_dir
            / "data"
            / f"chunk-{episode_chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )

        video_path = (
            dataset_dir
            / "videos"
            / f"chunk-{episode_chunk:03d}"
            / "observation.images.ego_view"
            / f"episode_{episode_index:06d}.mp4"
        )

        if not parquet_path.exists():
            errors.append(f"Missing parquet: {parquet_path}")
            continue

        if not video_path.exists():
            errors.append(f"Missing video: {video_path}")
            continue

        df = pd.read_parquet(parquet_path)
        rows = len(df)
        total_rows += rows

        if rows != expected_len:
            errors.append(
                f"Episode {episode_index}: parquet rows {rows} != episodes.jsonl length {expected_len}"
            )

        required_cols = [
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "annotation.human.action.task_description",
            "next.reward",
            "next.done",
        ]

        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Episode {episode_index}: missing column {col}")

        if rows > 0:
            first_state_dim = len(df["observation.state"].iloc[0])
            first_action_dim = len(df["action"].iloc[0])

            if first_state_dim != state_dim:
                errors.append(
                    f"Episode {episode_index}: state dim {first_state_dim} != info dim {state_dim}"
                )

            if first_action_dim != action_dim:
                errors.append(
                    f"Episode {episode_index}: action dim {first_action_dim} != info dim {action_dim}"
                )

            done_count = int(df["next.done"].sum())
            if done_count != 1:
                errors.append(f"Episode {episode_index}: done count {done_count} != 1")

            if not bool(df["next.done"].iloc[-1]):
                errors.append(f"Episode {episode_index}: last next.done is not True")

            unique_episode_indices = sorted(df["episode_index"].unique().tolist())
            if unique_episode_indices != [episode_index]:
                errors.append(
                    f"Episode {episode_index}: episode_index values {unique_episode_indices}"
                )

        video_frames = ffprobe_num_frames(video_path)
        video_frame_text = "unknown"
        if video_frames is not None:
            video_frame_text = str(video_frames)
            if video_frames != rows:
                errors.append(
                    f"Episode {episode_index}: video frames {video_frames} != parquet rows {rows}"
                )

        print(
            f"[episode {episode_index:06d}] rows={rows} "
            f"state_dim={len(df['observation.state'].iloc[0]) if rows else 'NA'} "
            f"action_dim={len(df['action'].iloc[0]) if rows else 'NA'} "
            f"video_frames={video_frame_text} "
            f"task={ep.get('tasks')}"
        )

    if total_rows != int(info.get("total_frames", -1)):
        errors.append(
            f"total rows {total_rows} != info total_frames {info.get('total_frames')}"
        )

    if len(episodes) != int(info.get("total_episodes", -1)):
        errors.append(
            f"episodes count {len(episodes)} != info total_episodes {info.get('total_episodes')}"
        )

    if len(tasks) != int(info.get("total_tasks", -1)):
        errors.append(
            f"tasks count {len(tasks)} != info total_tasks {info.get('total_tasks')}"
        )

    print("[summary]")
    print("  episodes:", len(episodes))
    print("  rows:", total_rows)
    print("  errors:", len(errors))

    if errors:
        print("\n[errors]")
        for e in errors:
            print(" -", e)
        raise RuntimeError("Dataset validation failed.")

    print("[done] dataset validation passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_dataset(args.dataset_dir)


if __name__ == "__main__":
    main()

