import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def natural_key(path: Path):
    stem = path.stem
    return int(stem) if stem.isdigit() else stem


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    else:
        x = np.asarray(x)
    return x.astype(np.float32)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_rgb_files(rgb_dir: Path, max_frames: int | None = None):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(rgb_dir.glob(f"*{ext}"))

    files = sorted(files, key=natural_key)

    if max_frames is not None:
        files = files[:max_frames]

    if len(files) < 2:
        raise RuntimeError(f"Need at least 2 images, found {len(files)} in {rgb_dir}")

    return files


def get_raw_mano(anno_obj):
    if not isinstance(anno_obj, dict):
        raise TypeError(f"Expected annotation object to be dict, got {type(anno_obj)}")

    if "raw_mano" not in anno_obj:
        raise KeyError(f"Cannot find 'raw_mano'. Top-level keys: {list(anno_obj.keys())}")

    return anno_obj["raw_mano"]


def get_frame_mano(raw_mano, frame_id: int):
    if frame_id in raw_mano:
        return raw_mano[frame_id]

    frame_id_str = str(frame_id)
    if frame_id_str in raw_mano:
        return raw_mano[frame_id_str]

    raise KeyError(f"frame_id {frame_id} not found in raw_mano")


def extract_prop_from_mano(frame_mano, mode: str) -> np.ndarray:
    lh_pose = to_numpy(frame_mano["lh__pose_coeffs"]).reshape(16, 4)
    rh_pose = to_numpy(frame_mano["rh__pose_coeffs"]).reshape(16, 4)
    lh_tsl = to_numpy(frame_mano["lh__tsl"]).reshape(3)
    rh_tsl = to_numpy(frame_mano["rh__tsl"]).reshape(3)

    if mode == "fingers120":
        # 15 joints * 4 for each hand. Drop wrist/global quaternion.
        prop = np.concatenate(
            [
                lh_pose[1:].reshape(-1),
                rh_pose[1:].reshape(-1),
            ],
            axis=0,
        )

    elif mode == "full128":
        # 16 MANO quaternions * 4 for each hand.
        # Total dim = 64 + 64 = 128.
        # This avoids the GR00T N1.7 negative padding issue from 134-dim action.
        prop = np.concatenate(
            [
                lh_pose.reshape(-1),
                rh_pose.reshape(-1),
            ],
            axis=0,
        )

    elif mode == "full134":
        # Debug only. This may exceed GR00T N1.7 action dimension limit.
        prop = np.concatenate(
            [
                lh_pose.reshape(-1),
                lh_tsl,
                rh_pose.reshape(-1),
                rh_tsl,
            ],
            axis=0,
        )

    else:
        raise ValueError(f"Unknown prop mode: {mode}")

    return prop.astype(np.float32)


def build_props_from_rgb(raw_mano, rgb_files, mode: str):
    props = []
    valid_rgb_files = []
    oakink_frame_ids = []

    for img_path in rgb_files:
        frame_id = int(img_path.stem)

        try:
            frame_mano = get_frame_mano(raw_mano, frame_id)
        except KeyError:
            print(f"[warn] skip RGB frame without raw_mano: {img_path.name}")
            continue

        prop = extract_prop_from_mano(frame_mano, mode=mode)

        props.append(prop)
        valid_rgb_files.append(img_path)
        oakink_frame_ids.append(frame_id)

    if len(props) < 2:
        raise RuntimeError("Too few RGB frames have matching raw_mano frames.")

    props = np.stack(props, axis=0).astype(np.float32)
    return valid_rgb_files, props, oakink_frame_ids


def write_video(rgb_files, out_path: Path, fps: int, image_size: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (image_size, image_size),
    )

    kept = 0
    for p in rgb_files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[warn] failed to read image: {p}")
            continue

        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        writer.write(img)
        kept += 1

    writer.release()

    if kept != len(rgb_files):
        raise RuntimeError(f"Video wrote {kept} frames, expected {len(rgb_files)}")

    return kept


def write_parquet(out_path: Path, props: np.ndarray, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # state[t] = prop[t]
    # action[t] = prop[t+1]
    # GR00T's RELATIVE action processor converts this absolute target to
    # action[t] - state[t] during training.
    # Therefore rows = T - 1.
    T = props.shape[0]
    num_rows = T - 1

    rows = []
    for t in range(num_rows):
        state = props[t].astype(np.float32)
        action = props[t + 1].astype(np.float32)

        rows.append(
            {
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "timestamp": float(t / fps),
                "frame_index": int(t),
                "episode_index": 0,
                "index": int(t),
                "task_index": 0,
                "annotation.human.action.task_description": 0,
                "next.reward": 0.0,
                "next.done": bool(t == num_rows - 1),
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)

    return num_rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_meta(
    output_dir: Path,
    task: str,
    num_rows: int,
    fps: int,
    image_size: int,
    prop_dim: int,
    robot_type: str,
):
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {
                "task_index": 0,
                "task": task,
            }
        ],
    )

    write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {
                "episode_index": 0,
                "tasks": [task],
                "length": num_rows,
            }
        ],
    )

    info = {
        "codebase_version": "v2.0",
        "robot_type": robot_type,

        "total_episodes": 1,
        "total_frames": num_rows,
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": 1000,

        "fps": fps,
        "video": True,

        "splits": {
            "train": "0:1",
        },

        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",

        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [prop_dim],
                "names": ["hand"],
            },
            "action": {
                "dtype": "float32",
                "shape": [prop_dim],
                "names": ["hand"],
            },
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "channel"],
            },
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [1],
                "names": ["task_index"],
            },
        },
    }

    with (meta_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    modality = {
        "state": {
            "hand": {
                "start": 0,
                "end": prop_dim,
            }
        },
        "action": {
            "hand": {
                "start": 0,
                "end": prop_dim,
            }
        },
        "video": {
            "ego_view": {
                "original_key": "observation.images.ego_view"
            }
        },
        "annotation": {
            "human.action.task_description": {}
        },
    }

    with (meta_dir / "modality.json").open("w") as f:
        json.dump(modality, f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--anno_pkl", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--task", default="Cap the bottle.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_frames", type=int, default=300)

    parser.add_argument(
        "--prop_mode",
        choices=["full128", "fingers120", "full134"],
        default="full128",
    )

    args = parser.parse_args()

    rgb_dir = Path(args.rgb_dir).expanduser()
    anno_pkl = Path(args.anno_pkl).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not rgb_dir.exists() or not rgb_dir.is_dir():
        raise FileNotFoundError(f"rgb_dir does not exist or is not a directory: {rgb_dir}")

    if not anno_pkl.exists() or not anno_pkl.is_file():
        raise FileNotFoundError(f"anno_pkl does not exist or is not a file: {anno_pkl}")

    if args.prop_mode == "full134":
        print("[warn] full134 is for debugging only. It may exceed GR00T N1.7 action dimension limit.")

    print(f"[rgb_dir] {rgb_dir}")
    print(f"[anno_pkl] {anno_pkl}")
    print(f"[output_dir] {output_dir}")
    print(f"[prop_mode] {args.prop_mode}")

    rgb_files = load_rgb_files(rgb_dir, max_frames=args.max_frames)

    anno_obj = load_pickle(anno_pkl)
    raw_mano = get_raw_mano(anno_obj)

    valid_rgb_files, props, oakink_frame_ids = build_props_from_rgb(
        raw_mano=raw_mano,
        rgb_files=rgb_files,
        mode=args.prop_mode,
    )

    # Since action[t] uses prop[t+1] as the absolute target, parquet has T-1 rows.
    # To keep video frame count equal to parquet row count, use first T-1 images.
    video_rgb_files = valid_rgb_files[:-1]
    parquet_props = props

    prop_dim = props.shape[1]

    print(f"[num input rgb] {len(rgb_files)}")
    print(f"[num matched rgb] {len(valid_rgb_files)}")
    print(f"[first oakink frame id] {oakink_frame_ids[0]}")
    print(f"[last oakink frame id] {oakink_frame_ids[-1]}")
    print(f"[prop shape] {props.shape}")
    print(f"[prop dim] {prop_dim}")
    print(f"[fps] {args.fps}")

    video_path = (
        output_dir
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / "episode_000000.mp4"
    )

    num_video_frames = write_video(
        rgb_files=video_rgb_files,
        out_path=video_path,
        fps=args.fps,
        image_size=args.image_size,
    )

    parquet_path = output_dir / "data" / "chunk-000" / "episode_000000.parquet"

    num_rows = write_parquet(
        out_path=parquet_path,
        props=parquet_props,
        fps=args.fps,
    )

    if num_video_frames != num_rows:
        raise RuntimeError(f"video frames {num_video_frames} != parquet rows {num_rows}")

    write_meta(
        output_dir=output_dir,
        task=args.task,
        num_rows=num_rows,
        fps=args.fps,
        image_size=args.image_size,
        prop_dim=prop_dim,
        robot_type=f"oakink2_mano_{args.prop_mode}",
    )

    print("[done]")
    print(f"video:   {video_path}")
    print(f"parquet: {parquet_path}")
    print(f"rows:    {num_rows}")
    print(f"dim:     {prop_dim}")


if __name__ == "__main__":
    main()
