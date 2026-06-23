#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from urllib.parse import unquote


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_task_targets(path: Path) -> dict:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"task_target.json not found: {path}")

    data = json.loads(path.read_text())

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected task_target.json to be a dict, but got {type(data)}"
        )

    return data


def decode_oakink2_key(encoded_key: str) -> tuple[str, str, str]:
    """
    Example:
    encoded_key:
        scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56

    decoded_key:
        scene_01__A003++seq__49a8305e104d29e3816a__2023-04-15-09-41-56

    task_key:
        scene_01__A003/seq__49a8305e104d29e3816a__2023-04-15-09-41-56
    """
    decoded_key = unquote(encoded_key)
    task_key = decoded_key.replace("++", "/")

    if "++" in decoded_key:
        scene, seq = decoded_key.split("++", 1)
    else:
        # Fallback for unexpected names.
        scene = decoded_key
        seq = ""

    return decoded_key, task_key, scene, seq


def count_rgb_frames(rgb_dir: Path) -> int:
    if not rgb_dir.exists() or not rgb_dir.is_dir():
        return 0

    count = 0
    for p in rgb_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            count += 1
    return count


def build_manifest(
    oakink2_root: Path,
    task_target_json: Path,
    camera: str,
    output: Path,
    max_trajectories: int | None = None,
) -> list[dict]:
    oakink2_root = oakink2_root.expanduser()
    output = output.expanduser()

    data_root = oakink2_root / "extracted" / "data"
    anno_root = oakink2_root / "extracted" / "anno_preview"

    if not data_root.exists():
        raise FileNotFoundError(f"RGB data root not found: {data_root}")

    if not anno_root.exists():
        print(f"[warning] anno_preview root not found: {anno_root}")

    task_targets = load_task_targets(task_target_json)

    traj_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if max_trajectories is not None:
        traj_dirs = traj_dirs[:max_trajectories]

    rows: list[dict] = []

    for episode_index, traj_dir in enumerate(traj_dirs):
        encoded_key = traj_dir.name
        decoded_key, task_key, scene, seq = decode_oakink2_key(encoded_key)

        rgb_dir = data_root / encoded_key / camera
        anno_pkl = anno_root / f"{encoded_key}.pkl"

        has_rgb_dir = rgb_dir.exists() and rgb_dir.is_dir()
        has_anno_pkl = anno_pkl.exists() and anno_pkl.is_file()
        num_rgb_frames = count_rgb_frames(rgb_dir)

        task = task_targets.get(task_key)
        task_lookup_status = "matched" if task is not None else "missing"

        row = {
            "episode_index": episode_index,
            "encoded_key": encoded_key,
            "decoded_key": decoded_key,
            "task_key": task_key,
            "scene": scene,
            "seq": seq,
            "rgb_dir": str(rgb_dir),
            "anno_pkl": str(anno_pkl),
            "camera": camera,
            "task": task,
            "task_lookup_status": task_lookup_status,
            "num_rgb_frames": num_rgb_frames,
            "has_rgb_dir": has_rgb_dir,
            "has_anno_pkl": has_anno_pkl,
        }
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rows


def print_summary(rows: list[dict], output: Path) -> None:
    total = len(rows)
    matched_task = sum(r["task_lookup_status"] == "matched" for r in rows)
    missing_task = sum(r["task_lookup_status"] == "missing" for r in rows)
    available_rgb = sum(r["has_rgb_dir"] for r in rows)
    available_anno = sum(r["has_anno_pkl"] for r in rows)
    rgb_nonzero = sum(r["num_rgb_frames"] > 0 for r in rows)

    valid = sum(
        r["task_lookup_status"] == "matched"
        and r["has_rgb_dir"]
        and r["has_anno_pkl"]
        and r["num_rgb_frames"] > 0
        for r in rows
    )

    print("[done]")
    print(f"output:              {output}")
    print(f"total trajectories:  {total}")
    print(f"matched tasks:       {matched_task}")
    print(f"missing tasks:       {missing_task}")
    print(f"has rgb dir:         {available_rgb}")
    print(f"has anno pkl:        {available_anno}")
    print(f"rgb frame count > 0: {rgb_nonzero}")
    print(f"valid rows:          {valid}")

    print("\n[first 3 rows]")
    for row in rows[:3]:
        print(json.dumps(row, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a JSONL manifest for locally available OakInk2 trajectories."
    )

    parser.add_argument(
        "--oakink2-root",
        type=Path,
        required=True,
        help="OakInk2 root, e.g. datasets/oakink2",
    )
    parser.add_argument(
        "--task-target-json",
        type=Path,
        required=True,
        help="Path to ManipTrans/data/OakInk-v2/program/task_target.json",
    )
    parser.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Camera folder name, e.g. 104422070969",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Optional limit for quick debugging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = build_manifest(
        oakink2_root=args.oakink2_root,
        task_target_json=args.task_target_json,
        camera=args.camera,
        output=args.output,
        max_trajectories=args.max_trajectories,
    )

    print_summary(rows, args.output.expanduser())


if __name__ == "__main__":
    main()
