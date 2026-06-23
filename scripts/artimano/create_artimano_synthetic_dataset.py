#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd


TASK_DESCRIPTION = "pick up the object with the artimano hand"
CHUNK_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a synthetic GR00T-compatible LeRobot v2 ARTIMANO dataset."
    )
    parser.add_argument("--output-dir", default="demo_data/artimano_synthetic")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--frames-per-episode", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--left-hand-dim", type=int, default=20)
    parser.add_argument("--right-hand-dim", type=int, default=20)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def ensure_clean_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    (output_dir / "meta").mkdir(parents=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True)
    (output_dir / "videos" / "chunk-000" / "observation.images.front").mkdir(parents=True)
    (output_dir / "videos" / "chunk-000" / "observation.images.wrist").mkdir(parents=True)


def smooth_joint_trajectory(
    *,
    rng: np.random.Generator,
    frames: int,
    dim: int,
    episode_id: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    trajectory = np.zeros((frames, dim), dtype=np.float32)

    for joint_id in range(dim):
        frequency = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = rng.uniform(0.15, 0.65)
        offset = rng.uniform(-0.15, 0.15)
        trajectory[:, joint_id] = (
            offset
            + amplitude * np.sin(2.0 * np.pi * frequency * t + phase)
            + 0.25 * amplitude * np.sin(2.0 * np.pi * (frequency * 0.5) * t + phase * 0.5)
        )

    noise = rng.normal(loc=0.0, scale=0.01, size=(frames, dim)).astype(np.float32)
    random_walk = np.cumsum(noise, axis=0)
    random_walk -= random_walk.mean(axis=0, keepdims=True)
    trajectory += 0.05 * random_walk
    trajectory += np.float32(0.01 * episode_id)
    return trajectory.astype(np.float32)


def future_smoothed_targets(state: np.ndarray, lookahead: int = 4) -> np.ndarray:
    action = np.empty_like(state, dtype=np.float32)
    for frame_idx in range(len(state)):
        start = frame_idx + 1
        end = min(len(state), frame_idx + lookahead + 1)
        if start >= len(state):
            action[frame_idx] = state[-1]
        else:
            action[frame_idx] = state[start:end].mean(axis=0)
    return action.astype(np.float32)


def make_frame(
    *,
    state: np.ndarray,
    frame_idx: int,
    episode_id: int,
    height: int,
    width: int,
    camera: str,
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if camera == "front":
        frame[:] = (32, 45, 56)
        color = (90, 210, 235)
        secondary = (230, 160, 70)
    else:
        frame[:] = (45, 35, 48)
        color = (190, 130, 245)
        secondary = (80, 220, 140)

    x = int((np.sin(state[0]) * 0.35 + 0.5) * (width - 1))
    y = int((np.cos(state[1]) * 0.35 + 0.5) * (height - 1))
    radius = max(8, min(height, width) // 18)

    cv2.circle(frame, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(
        frame,
        (width // 4, height // 2 - radius),
        (3 * width // 4, height // 2 + radius),
        secondary,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"ARTIMANO {camera} ep{episode_id:02d} f{frame_idx:03d}",
        (12, max(24, height - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return frame


def write_video(
    path: Path,
    *,
    states: np.ndarray,
    episode_id: int,
    fps: int,
    height: int,
    width: int,
    camera: str,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame_idx, state in enumerate(states):
            frame_rgb = make_frame(
                state=state,
                frame_idx=frame_idx,
                episode_id=episode_id,
                height=height,
                width=width,
                camera=camera,
            )
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def joint_names(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{idx:02d}.pos" for idx in range(dim)]


def build_info(
    *,
    num_episodes: int,
    frames_per_episode: int,
    fps: int,
    left_hand_dim: int,
    right_hand_dim: int,
    image_height: int,
    image_width: int,
) -> dict:
    total_dim = left_hand_dim + right_hand_dim
    names = joint_names("left_hand", left_hand_dim) + joint_names("right_hand", right_hand_dim)
    video_info = {
        "dtype": "video",
        "shape": [image_height, image_width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": image_height,
            "video.width": image_width,
            "video.codec": "mp4v",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    return {
        "codebase_version": "v2.1",
        "robot_type": "artimano_synthetic",
        "total_episodes": num_episodes,
        "total_frames": num_episodes * frames_per_episode,
        "total_tasks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "names": names, "shape": [total_dim]},
            "observation.state": {"dtype": "float32", "names": names, "shape": [total_dim]},
            "observation.images.front": video_info,
            "observation.images.wrist": video_info,
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "annotation.human.task_description": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "next.reward": {"dtype": "float32", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
        },
        "total_chunks": 1,
        "total_videos": num_episodes * 2,
    }


def build_modality(left_hand_dim: int, right_hand_dim: int) -> dict:
    total_dim = left_hand_dim + right_hand_dim
    return {
        "state": {
            "left_hand": {"start": 0, "end": left_hand_dim},
            "right_hand": {"start": left_hand_dim, "end": total_dim},
        },
        "action": {
            "left_hand": {"start": 0, "end": left_hand_dim},
            "right_hand": {"start": left_hand_dim, "end": total_dim},
        },
        "video": {
            "front": {"original_key": "observation.images.front"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {
            "human.task_description": {
                "original_key": "annotation.human.task_description",
            }
        },
    }


def write_episode(
    *,
    output_dir: Path,
    rng: np.random.Generator,
    episode_id: int,
    global_start_idx: int,
    frames_per_episode: int,
    fps: int,
    left_hand_dim: int,
    right_hand_dim: int,
    image_height: int,
    image_width: int,
) -> None:
    total_dim = left_hand_dim + right_hand_dim
    state = smooth_joint_trajectory(
        rng=rng,
        frames=frames_per_episode,
        dim=total_dim,
        episode_id=episode_id,
    )
    action = future_smoothed_targets(state)

    rows = []
    for frame_idx in range(frames_per_episode):
        rows.append(
            {
                "observation.state": state[frame_idx],
                "action": action[frame_idx],
                "timestamp": np.float32(frame_idx / fps),
                "frame_index": np.int64(frame_idx),
                "episode_index": np.int64(episode_id),
                "index": np.int64(global_start_idx + frame_idx),
                "task_index": np.int64(0),
                "annotation.human.task_description": np.int64(0),
                "next.reward": np.float32(1.0 if frame_idx == frames_per_episode - 1 else 0.0),
                "next.done": bool(frame_idx == frames_per_episode - 1),
            }
        )

    parquet_path = output_dir / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, engine="pyarrow", index=False)

    for camera, video_key in [
        ("front", "observation.images.front"),
        ("wrist", "observation.images.wrist"),
    ]:
        video_path = (
            output_dir
            / "videos"
            / "chunk-000"
            / video_key
            / f"episode_{episode_id:06d}.mp4"
        )
        write_video(
            video_path,
            states=state,
            episode_id=episode_id,
            fps=fps,
            height=image_height,
            width=image_width,
            camera=camera,
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if args.frames_per_episode <= 16:
        raise ValueError("--frames-per-episode must be greater than the 16-step action horizon")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.left_hand_dim <= 0 or args.right_hand_dim <= 0:
        raise ValueError("--left-hand-dim and --right-hand-dim must be positive")
    if args.image_height <= 0 or args.image_width <= 0:
        raise ValueError("--image-height and --image-width must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    ensure_clean_output_dir(output_dir, overwrite=args.overwrite)

    rng = np.random.default_rng(args.seed)
    episodes = []
    for episode_id in range(args.num_episodes):
        write_episode(
            output_dir=output_dir,
            rng=rng,
            episode_id=episode_id,
            global_start_idx=episode_id * args.frames_per_episode,
            frames_per_episode=args.frames_per_episode,
            fps=args.fps,
            left_hand_dim=args.left_hand_dim,
            right_hand_dim=args.right_hand_dim,
            image_height=args.image_height,
            image_width=args.image_width,
        )
        episodes.append(
            {
                "episode_index": episode_id,
                "tasks": [TASK_DESCRIPTION],
                "length": args.frames_per_episode,
            }
        )

    write_json(
        output_dir / "meta" / "info.json",
        build_info(
            num_episodes=args.num_episodes,
            frames_per_episode=args.frames_per_episode,
            fps=args.fps,
            left_hand_dim=args.left_hand_dim,
            right_hand_dim=args.right_hand_dim,
            image_height=args.image_height,
            image_width=args.image_width,
        ),
    )
    write_json(output_dir / "meta" / "modality.json", build_modality(args.left_hand_dim, args.right_hand_dim))
    write_jsonl(output_dir / "meta" / "tasks.jsonl", [{"task_index": 0, "task": TASK_DESCRIPTION}])
    write_jsonl(output_dir / "meta" / "episodes.jsonl", episodes)

    print(f"Wrote ARTIMANO synthetic dataset to {output_dir}")
    print(f"Episodes: {args.num_episodes}; frames/episode: {args.frames_per_episode}")
    print(f"State/action dim: {args.left_hand_dim + args.right_hand_dim}")


if __name__ == "__main__":
    main()
