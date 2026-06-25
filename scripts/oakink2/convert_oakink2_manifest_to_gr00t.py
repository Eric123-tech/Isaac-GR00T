#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from convert_one_oakink2_to_gr00t_realprop import (
    load_rgb_files,
    load_pickle,
    get_raw_mano,
    build_props_from_rgb,
    write_video,
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.expanduser().open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_valid_manifest_row(row: dict) -> tuple[bool, str]:
    if not row.get("has_rgb_dir", False):
        return False, "missing_rgb_dir"
    if not row.get("has_anno_pkl", False):
        return False, "missing_anno_pkl"
    if row.get("task_lookup_status") != "matched":
        return False, "missing_task"
    if not row.get("task"):
        return False, "empty_task"
    if int(row.get("num_rgb_frames", 0)) <= 0:
        return False, "zero_rgb_frames"
    return True, "valid"


def write_episode_parquet(
    out_path: Path,
    props,
    fps: int,
    episode_index: int,
    task_index: int,
    global_index_start: int,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # state[t] = prop[t]
    # action[t] = prop[t+1]
    # GR00T's RELATIVE action processor converts this absolute target to
    # action[t] - state[t] during training.
    # rows = T - 1
    T = props.shape[0]
    num_rows = T - 1

    rows = []
    for t in range(num_rows):
        rows.append(
            {
                "observation.state": props[t].astype("float32").tolist(),
                "action": props[t + 1].astype("float32").tolist(),
                "timestamp": float(t / fps),
                "frame_index": int(t),
                "episode_index": int(episode_index),
                "index": int(global_index_start + t),
                "task_index": int(task_index),
                "annotation.human.action.task_description": int(task_index),
                "next.reward": 0.0,
                "next.done": bool(t == num_rows - 1),
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return num_rows


def write_meta(
    output_dir: Path,
    episodes_meta: list[dict],
    tasks: list[str],
    total_frames: int,
    fps: int,
    image_size: int,
    prop_dim: int,
    robot_type: str,
    chunks_size: int = 1000,
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    task_rows = [
        {
            "task_index": i,
            "task": task,
        }
        for i, task in enumerate(tasks)
    ]
    write_jsonl(meta_dir / "tasks.jsonl", task_rows)

    episode_rows = [
        {
            "episode_index": ep["episode_index"],
            "tasks": [ep["task"]],
            "length": ep["length"],
        }
        for ep in episodes_meta
    ]
    write_jsonl(meta_dir / "episodes.jsonl", episode_rows)

    total_episodes = len(episodes_meta)

    info = {
        "codebase_version": "v2.0",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": total_episodes,
        "total_chunks": max(1, (total_episodes + chunks_size - 1) // chunks_size),
        "chunks_size": chunks_size,
        "fps": fps,
        "video": True,
        "splits": {
            "train": f"0:{total_episodes}",
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


def convert_manifest(
    manifest: Path,
    output_dir: Path,
    fps: int,
    image_size: int,
    max_frames: int | None,
    prop_mode: str,
    overwrite: bool,
) -> None:
    manifest = manifest.expanduser()
    output_dir = output_dir.expanduser()

    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"output_dir already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    rows = read_jsonl(manifest)

    skipped = []
    failed = []
    successes = []

    task_to_index: dict[str, int] = {}
    tasks: list[str] = []

    episodes_meta = []
    total_frames = 0
    global_index = 0
    prop_dim = None

    valid_rows = []
    for row in rows:
        ok, reason = is_valid_manifest_row(row)
        if ok:
            valid_rows.append(row)
        else:
            skipped.append(
                {
                    "encoded_key": row.get("encoded_key"),
                    "reason": reason,
                }
            )

    print(f"[manifest] {manifest}")
    print(f"[rows] total={len(rows)} valid={len(valid_rows)} skipped={len(skipped)}")
    print(f"[output_dir] {output_dir}")
    print(f"[prop_mode] {prop_mode}")

    for new_episode_index, row in enumerate(valid_rows):
        encoded_key = row["encoded_key"]
        rgb_dir = Path(row["rgb_dir"]).expanduser()
        anno_pkl = Path(row["anno_pkl"]).expanduser()
        task = row["task"]

        if task not in task_to_index:
            task_to_index[task] = len(tasks)
            tasks.append(task)

        task_index = task_to_index[task]
        episode_chunk = new_episode_index // 1000

        try:
            rgb_files = load_rgb_files(rgb_dir, max_frames=max_frames)
            anno_obj = load_pickle(anno_pkl)
            raw_mano = get_raw_mano(anno_obj)

            valid_rgb_files, props, oakink_frame_ids = build_props_from_rgb(
                raw_mano=raw_mano,
                rgb_files=rgb_files,
                mode=prop_mode,
            )

            if prop_dim is None:
                prop_dim = int(props.shape[1])
            elif prop_dim != int(props.shape[1]):
                raise RuntimeError(
                    f"Inconsistent prop dim: got {props.shape[1]}, expected {prop_dim}"
                )

            # Match previous convention: video frame count equals parquet row count.
            video_rgb_files = valid_rgb_files[:-1]

            video_path = (
                output_dir
                / "videos"
                / f"chunk-{episode_chunk:03d}"
                / "observation.images.ego_view"
                / f"episode_{new_episode_index:06d}.mp4"
            )
            parquet_path = (
                output_dir
                / "data"
                / f"chunk-{episode_chunk:03d}"
                / f"episode_{new_episode_index:06d}.parquet"
            )

            num_video_frames = write_video(
                rgb_files=video_rgb_files,
                out_path=video_path,
                fps=fps,
                image_size=image_size,
            )

            num_rows = write_episode_parquet(
                out_path=parquet_path,
                props=props,
                fps=fps,
                episode_index=new_episode_index,
                task_index=task_index,
                global_index_start=global_index,
            )

            if num_video_frames != num_rows:
                raise RuntimeError(
                    f"video frames {num_video_frames} != parquet rows {num_rows}"
                )

            episodes_meta.append(
                {
                    "episode_index": new_episode_index,
                    "encoded_key": encoded_key,
                    "task": task,
                    "task_index": task_index,
                    "length": num_rows,
                    "first_oakink_frame_id": int(oakink_frame_ids[0]),
                    "last_oakink_frame_id": int(oakink_frame_ids[-1]),
                    "num_input_rgb": len(rgb_files),
                    "num_matched_rgb": len(valid_rgb_files),
                    "video_path": str(video_path),
                    "parquet_path": str(parquet_path),
                }
            )

            successes.append(
                {
                    "episode_index": new_episode_index,
                    "encoded_key": encoded_key,
                    "task": task,
                    "rows": num_rows,
                    "prop_dim": int(prop_dim),
                    "video": str(video_path),
                    "parquet": str(parquet_path),
                }
            )

            total_frames += num_rows
            global_index += num_rows

            print(
                f"[ok] episode={new_episode_index:06d} rows={num_rows} "
                f"dim={prop_dim} task={task!r}"
            )

        except Exception as e:
            failed.append(
                {
                    "encoded_key": encoded_key,
                    "rgb_dir": str(rgb_dir),
                    "anno_pkl": str(anno_pkl),
                    "reason": repr(e),
                }
            )
            print(f"[failed] {encoded_key}: {repr(e)}")

    if len(successes) == 0:
        raise RuntimeError(
            f"No episodes were converted. skipped={len(skipped)}, failed={len(failed)}"
        )

    assert prop_dim is not None

    write_meta(
        output_dir=output_dir,
        episodes_meta=episodes_meta,
        tasks=tasks,
        total_frames=total_frames,
        fps=fps,
        image_size=image_size,
        prop_dim=prop_dim,
        robot_type=f"oakink2_mano_{prop_mode}",
    )

    report = {
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "fps": fps,
        "image_size": image_size,
        "max_frames": max_frames,
        "prop_mode": prop_mode,
        "total_manifest_rows": len(rows),
        "valid_manifest_rows": len(valid_rows),
        "successful_episodes": len(successes),
        "skipped_rows": len(skipped),
        "failed_rows": len(failed),
        "total_frames": total_frames,
        "prop_dim": prop_dim,
        "tasks": tasks,
        "successes": successes,
        "skipped": skipped,
        "failed": failed,
    }

    report_path = output_dir / "meta" / "conversion_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("[done]")
    print(f"successful episodes: {len(successes)}")
    print(f"skipped rows:        {len(skipped)}")
    print(f"failed rows:         {len(failed)}")
    print(f"total frames:        {total_frames}")
    print(f"prop dim:            {prop_dim}")
    print(f"report:              {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an OakInk2 JSONL manifest into a GR00T/LeRobot-style dataset."
    )

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_frames", type=int, default=300)

    parser.add_argument(
        "--prop_mode",
        choices=["full128", "fingers120", "full134"],
        default="full128",
    )

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prop_mode == "full134":
        print(
            "[warn] full134 is for debugging only. "
            "It may exceed GR00T N1.7 action dimension limit."
        )

    convert_manifest(
        manifest=args.manifest,
        output_dir=args.output_dir,
        fps=args.fps,
        image_size=args.image_size,
        max_frames=args.max_frames,
        prop_mode=args.prop_mode,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
