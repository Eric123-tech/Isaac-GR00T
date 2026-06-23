#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


EXPECTED_STATE_DIM = 40
EXPECTED_ACTION_DIM = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a synthetic ARTIMANO dataset.")
    parser.add_argument("--dataset-path", default="demo_data/artimano_synthetic")
    parser.add_argument(
        "--check-gr00t-video-loader",
        action="store_true",
        help="Also decode one frame via gr00t.utils.video_utils, which requires torchcodec.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required metadata file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def print_tree_summary(dataset_path: Path) -> None:
    print(f"Dataset path: {dataset_path}")
    for rel_dir in ["meta", "data/chunk-000", "videos/chunk-000/observation.images.front", "videos/chunk-000/observation.images.wrist"]:
        path = dataset_path / rel_dir
        if not path.exists():
            raise FileNotFoundError(f"Missing required directory: {path}")
        files = sorted(p.name for p in path.iterdir() if p.is_file())
        print(f"  {rel_dir}: {len(files)} file(s)")


def first_parquet(dataset_path: Path) -> Path:
    parquet_files = sorted((dataset_path / "data" / "chunk-000").glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_path / 'data/chunk-000'}")
    return parquet_files[0]


def validate_parquet(parquet_path: Path) -> None:
    df = pd.read_parquet(parquet_path)
    print(f"First parquet: {parquet_path}")
    print(f"Rows: {len(df)}")
    print("Columns:")
    for column in df.columns:
        print(f"  - {column}")

    required_columns = {
        "observation.state",
        "action",
        "timestamp",
        "annotation.human.task_description",
        "task_index",
        "episode_index",
        "index",
        "next.reward",
        "next.done",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise AssertionError(f"Missing required parquet columns: {missing}")

    state = np.asarray(df["observation.state"].iloc[0], dtype=np.float32)
    action = np.asarray(df["action"].iloc[0], dtype=np.float32)
    print(f"State dim: {state.shape[0]}")
    print(f"Action dim: {action.shape[0]}")
    assert state.shape == (EXPECTED_STATE_DIM,), f"Expected state dim {EXPECTED_STATE_DIM}, got {state.shape}"
    assert action.shape == (EXPECTED_ACTION_DIM,), f"Expected action dim {EXPECTED_ACTION_DIM}, got {action.shape}"

    task_value = df["annotation.human.task_description"].iloc[0]
    assert int(task_value) == 0, "annotation.human.task_description should store task index 0"
    assert bool(df["next.done"].iloc[-1]) is True, "Final frame next.done must be True"


def validate_video_with_cv2(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing video file: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"OpenCV could not read first frame from: {path}")
        print(f"Read video frame via OpenCV: {path} shape={frame.shape}")
    finally:
        cap.release()


def validate_gr00t_video_loader(path: Path) -> None:
    from gr00t.utils.video_utils import get_frames_by_indices

    frames = get_frames_by_indices(str(path), np.array([0]))
    assert frames.shape[0] == 1, f"Expected one decoded frame, got {frames.shape}"
    print(f"Read video frame via GR00T video loader: {path} shape={frames.shape}")


def validate_videos(dataset_path: Path, modality: dict, info: dict, check_gr00t_loader: bool) -> None:
    video_path_pattern = info["video_path"]
    chunk_idx = 0
    episode_idx = 0
    for key, meta in modality["video"].items():
        original_key = meta.get("original_key", f"observation.images.{key}")
        video_rel_path = video_path_pattern.format(
            episode_chunk=chunk_idx,
            video_key=original_key,
            episode_index=episode_idx,
        )
        video_path = dataset_path / video_rel_path
        validate_video_with_cv2(video_path)
        if check_gr00t_loader:
            validate_gr00t_video_loader(video_path)


def validate_modality(modality: dict) -> None:
    print("Modality:")
    print(json.dumps(modality, indent=2))
    assert modality["state"]["left_hand"] == {"start": 0, "end": 20}
    assert modality["state"]["right_hand"] == {"start": 20, "end": 40}
    assert modality["action"]["left_hand"] == {"start": 0, "end": 20}
    assert modality["action"]["right_hand"] == {"start": 20, "end": 40}
    assert modality["video"]["front"]["original_key"] == "observation.images.front"
    assert modality["video"]["wrist"]["original_key"] == "observation.images.wrist"
    assert (
        modality["annotation"]["human.task_description"]["original_key"]
        == "annotation.human.task_description"
    )


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    print_tree_summary(dataset_path)
    info = read_json(dataset_path / "meta" / "info.json")
    modality = read_json(dataset_path / "meta" / "modality.json")
    validate_modality(modality)
    validate_parquet(first_parquet(dataset_path))
    validate_videos(dataset_path, modality, info, args.check_gr00t_video_loader)
    print("ARTIMANO dataset validation passed.")


if __name__ == "__main__":
    main()
