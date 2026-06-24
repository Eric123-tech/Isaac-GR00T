#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TASK_TARGET_JSON = REPO_ROOT / "scripts" / "oakink2" / "task_target.json"
DEFAULT_OAKINK2_ROOT = Path("datasets/oakink2")
DEFAULT_CAMERA = "104422070969"
DEFAULT_MANIFEST_OUTPUT = Path(
    "datasets/oakink2/manifests/oakink2_available_trajectories.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128"
)
DEFAULT_OUTPUT_BASENAME = "direct_hand_manifest_1_mano_full128"
SAMPLE_TASK_KEY = (
    "scene_01__A003/seq__49a8305e104d29e3816a__2023-04-15-09-41-56"
)


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_oakink2_manifest import (  # noqa: E402
    build_manifest,
    count_rgb_frames,
    decode_oakink2_key,
    load_task_targets,
    normalize_oakink2_task_key,
    resolve_anno_pkl,
)
from convert_oakink2_manifest_to_gr00t import (  # noqa: E402
    convert_manifest,
    is_valid_manifest_row,
    write_jsonl,
)
from validate_gr00t_lerobot_dataset import validate_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    task_target_default = (
        DEFAULT_TASK_TARGET_JSON if DEFAULT_TASK_TARGET_JSON.exists() else None
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run the OakInk2 extracted-layout check, manifest, GR00T conversion, and "
            "validation pipeline."
        )
    )
    parser.add_argument(
        "--oakink2-root",
        type=Path,
        default=DEFAULT_OAKINK2_ROOT,
        help="OakInk2 root directory. Default: datasets/oakink2",
    )
    parser.add_argument(
        "--task-target-json",
        type=Path,
        default=task_target_default,
        required=task_target_default is None,
        help=(
            "Task target JSON. Defaults to scripts/oakink2/task_target.json "
            "when that file exists."
        ),
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=DEFAULT_CAMERA,
        help=f"OakInk2 camera folder name. Default: {DEFAULT_CAMERA}",
    )
    parser.add_argument(
        "--task-key",
        type=str,
        default=None,
        help=(
            "Optional trajectory key. Accepts scene/seq, scene++seq, or "
            f"scene%2B%2Bseq forms, e.g. {SAMPLE_TASK_KEY}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "GR00T/LeRobot output dataset. Defaults to "
            "datasets/oakink2/gr00t_lerobot/<seq5>_direct_hand_manifest_1_mano_full128 "
            "when --task-key is set, otherwise "
            f"{DEFAULT_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_OUTPUT,
        help=f"Manifest JSONL output. Default: {DEFAULT_MANIFEST_OUTPUT}",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument(
        "--prop-mode",
        choices=["full128"],
        default="full128",
        help="Proprioception representation. Currently locked to full128.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing converted output dataset.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help=(
            "Deprecated compatibility flag. The pipeline always requires an "
            "existing extracted OakInk2 layout."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check inputs and print planned actions without writing outputs.",
    )

    args = parser.parse_args()
    args.output_dir = resolve_output_dir(args.output_dir, args.task_key)
    validate_args(args)
    return args


def task_key_short_id(task_key: str) -> str:
    normalized_task_key = normalize_oakink2_task_key(task_key)
    match = re.search(r"(?:^|/)seq__([^_/]+?)__", normalized_task_key)
    if not match:
        raise ValueError(
            f"Could not extract a sequence id from --task-key {task_key!r}. "
            "Expected a segment like seq__49a8305e104d29e3816a__2023-04-15-09-41-56."
        )

    seq_id = match.group(1)
    if len(seq_id) < 5:
        raise ValueError(
            f"Sequence id in --task-key is shorter than 5 characters: {seq_id!r}"
        )

    return seq_id[:5]


def resolve_output_dir(output_dir: Path | None, task_key: str | None) -> Path:
    if output_dir is not None:
        return output_dir

    if task_key is None:
        return DEFAULT_OUTPUT_DIR

    short_id = task_key_short_id(task_key)
    return Path("datasets/oakink2/gr00t_lerobot") / (
        f"{short_id}_{DEFAULT_OUTPUT_BASENAME}"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.task_target_json is None:
        raise ValueError(
            "No task target JSON was provided and scripts/oakink2/task_target.json "
            "does not exist. Pass --task-target-json explicitly."
        )

    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")

    if args.image_size <= 0:
        raise ValueError(f"--image-size must be positive, got {args.image_size}")

    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError(f"--max-frames must be at least 2, got {args.max_frames}")


def extracted_roots(oakink2_root: Path) -> tuple[Path, Path, Path]:
    extracted_root = oakink2_root.expanduser() / "extracted"
    return extracted_root, extracted_root / "data", extracted_root / "anno_preview"


def has_extracted_layout(oakink2_root: Path) -> bool:
    _, data_root, anno_root = extracted_roots(oakink2_root)
    return data_root.is_dir() and anno_root.is_dir()


def ensure_extracted(oakink2_root: Path, skip_extract: bool, dry_run: bool) -> bool:
    del skip_extract, dry_run
    oakink2_root = oakink2_root.expanduser()
    _, data_root, anno_root = extracted_roots(oakink2_root)

    print("[layout]")
    print(f"  oakink2_root: {oakink2_root}")
    print(f"  expected data: {data_root}")
    print(f"  expected anno: {anno_root}")

    if has_extracted_layout(oakink2_root):
        print("  extracted layout exists")
        return True

    raise FileNotFoundError(
        "Expected manually extracted OakInk2 layout is missing. "
        f"Place RGB frames under {data_root}/<encoded_key>/<camera>/ and "
        f"annotation preview files under {anno_root}/<encoded_key>.pkl. "
        "This pipeline no longer reads raw/ or extracts archives automatically."
    )


def build_manifest_preview(
    oakink2_root: Path,
    task_target_json: Path,
    camera: str,
) -> list[dict]:
    oakink2_root = oakink2_root.expanduser()
    data_root = oakink2_root / "extracted" / "data"
    anno_root = oakink2_root / "extracted" / "anno_preview"

    if not data_root.exists():
        raise FileNotFoundError(f"RGB data root not found: {data_root}")

    task_targets = load_task_targets(task_target_json)
    rows = []

    traj_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    for episode_index, traj_dir in enumerate(traj_dirs):
        encoded_key = traj_dir.name
        decoded_key, task_key, scene, seq = decode_oakink2_key(encoded_key)
        rgb_dir = data_root / encoded_key / camera
        anno_pkl = resolve_anno_pkl(anno_root, encoded_key)

        has_rgb_dir = rgb_dir.exists() and rgb_dir.is_dir()
        has_anno_pkl = anno_pkl.exists() and anno_pkl.is_file()
        task = task_targets.get(task_key)

        rows.append(
            {
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
                "task_lookup_status": "matched" if task is not None else "missing",
                "num_rgb_frames": count_rgb_frames(rgb_dir),
                "has_rgb_dir": has_rgb_dir,
                "has_anno_pkl": has_anno_pkl,
            }
        )

    return rows


def row_matches_task_key(row: dict, normalized_task_key: str) -> bool:
    candidate_values = [
        row.get("task_key"),
        row.get("decoded_key"),
        row.get("encoded_key"),
        row.get("anno_pkl"),
        row.get("rgb_dir"),
    ]

    for value in candidate_values:
        if not value:
            continue
        try:
            if normalize_oakink2_task_key(str(value)) == normalized_task_key:
                return True
        except ValueError:
            continue

    return False


def select_manifest_rows(rows: list[dict], task_key: str | None) -> list[dict]:
    if task_key is not None:
        normalized_task_key = normalize_oakink2_task_key(task_key)
        matching_rows = [
            row for row in rows if row_matches_task_key(row, normalized_task_key)
        ]
        if not matching_rows:
            examples = []
            for row in rows[:5]:
                for value in [row.get("task_key"), row.get("encoded_key")]:
                    if value and value not in examples:
                        examples.append(value)
            raise ValueError(
                f"No manifest rows matched --task-key {task_key!r}. "
                f"Normalized target: {normalized_task_key!r}. "
                f"First available keys: {examples}"
            )

        invalid = []
        selected = []
        for row in matching_rows:
            ok, reason = is_valid_manifest_row(row)
            if ok:
                selected.append(row)
            else:
                invalid.append(
                    {
                        "encoded_key": row.get("encoded_key"),
                        "reason": reason,
                    }
                )

        if not selected:
            raise RuntimeError(
                f"Rows matched --task-key {task_key!r}, but none are valid: {invalid}"
            )
    else:
        selected = [row for row in rows if is_valid_manifest_row(row)[0]]
        if not selected:
            raise RuntimeError("No valid OakInk2 manifest rows were found.")

    for episode_index, row in enumerate(selected):
        row["episode_index"] = episode_index

    return selected


def write_selected_manifest(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path = manifest_path.expanduser()
    write_jsonl(manifest_path, rows)


def print_manifest_summary(all_rows: list[dict], selected_rows: list[dict]) -> None:
    valid_rows = [row for row in all_rows if is_valid_manifest_row(row)[0]]
    print(f"  total rows: {len(all_rows)}")
    print(f"  valid rows: {len(valid_rows)}")
    print(f"  selected rows: {len(selected_rows)}")

    tasks = sorted({row["task"] for row in selected_rows if row.get("task")})
    print(f"  selected tasks: {tasks}")


def run_manifest_stage(args: argparse.Namespace) -> list[dict]:
    print("[manifest]")
    print(f"  task_target_json: {args.task_target_json}")
    print(f"  camera: {args.camera}")
    print(f"  manifest_output: {args.manifest_output}")

    if args.dry_run:
        rows = build_manifest_preview(
            oakink2_root=args.oakink2_root,
            task_target_json=args.task_target_json,
            camera=args.camera,
        )
        selected_rows = select_manifest_rows(rows, args.task_key)
        print_manifest_summary(rows, selected_rows)
        print("  dry-run: manifest would be written here")
        return selected_rows

    rows = build_manifest(
        oakink2_root=args.oakink2_root,
        task_target_json=args.task_target_json,
        camera=args.camera,
        output=args.manifest_output,
    )
    selected_rows = select_manifest_rows(rows, args.task_key)
    write_selected_manifest(args.manifest_output, selected_rows)
    print_manifest_summary(rows, selected_rows)
    return selected_rows


def read_json(path: Path) -> dict:
    with path.expanduser().open("r") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.expanduser().open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_convert_stage(args: argparse.Namespace) -> None:
    print("[convert]")
    print(f"  output_dir: {args.output_dir}")
    print(f"  fps: {args.fps}")
    print(f"  image_size: {args.image_size}")
    print(f"  max_frames: {args.max_frames}")
    print(f"  prop_mode: {args.prop_mode}")

    if args.dry_run:
        print("  dry-run: conversion would be run here")
        return

    convert_manifest(
        manifest=args.manifest_output,
        output_dir=args.output_dir,
        fps=args.fps,
        image_size=args.image_size,
        max_frames=args.max_frames,
        prop_mode=args.prop_mode,
        overwrite=args.overwrite,
    )


def run_validate_stage(args: argparse.Namespace) -> None:
    print("[validate]")
    print(f"  dataset_dir: {args.output_dir}")

    if args.dry_run:
        print("  dry-run: validation would be run here")
        return

    validate_dataset(args.output_dir)


def dataset_name_from_output(output_dir: Path) -> str:
    return output_dir.expanduser().name


def run_id_from_dataset_name(dataset_name: str) -> str:
    suffix = f"_{DEFAULT_OUTPUT_BASENAME}"
    if dataset_name.endswith(suffix):
        run_id = dataset_name[: -len(suffix)]
    else:
        run_id = dataset_name

    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("_")
    return run_id or "oakink2"


def suggested_upload_command(output_dir: Path) -> str:
    dataset_name = dataset_name_from_output(output_dir)
    remote_dir = (
        f"david@SERVER_IP:/mnt/data/haoyu_data/oakink2/gr00t_lerobot/"
        f"{dataset_name}/"
    )
    return f"""cd ~/linux_projects/Isaac-GR00T
rsync -avhP --partial {output_dir}/ {remote_dir}"""


def suggested_server_finetune_command(output_dir: Path) -> str:
    dataset_name = dataset_name_from_output(output_dir)
    run_id = run_id_from_dataset_name(dataset_name)
    server_dataset_path = f"datasets/oakink2/gr00t_lerobot/{dataset_name}"
    server_output_dir = (
        f"/mnt/data/haoyu_data/gr00t_outputs/oakink2_{run_id}_projector_only"
    )

    return f"""cd ~/Desktop/haoyu/Isaac-GR00T
rm -rf {server_output_dir}
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
uv run accelerate launch --num_processes 1 --mixed_precision bf16 \\
  gr00t/experiment/launch_finetune.py \\
  --base-model-path nvidia/GR00T-N1.7-3B \\
  --dataset-path {server_dataset_path} \\
  --embodiment-tag NEW_EMBODIMENT \\
  --modality-config-path examples/ARTIMANO/oakink2_mano_full134_config.py \\
  --num-gpus 1 \\
  --output-dir {server_output_dir} \\
  --max-steps 2000 \\
  --save-steps 100 \\
  --global-batch-size 1 \\
  --gradient-accumulation-steps 1 \\
  --dataloader-num-workers 0 \\
  --no-use-wandb \\
  --no-tune-llm \\
  --no-tune-visual \\
  --tune-projector \\
  --no-tune-diffusion-model"""


def print_done_summary(args: argparse.Namespace, selected_rows: list[dict]) -> None:
    print("[done]")

    if args.dry_run:
        print("  dry-run complete")
        print(f"  selected episodes: {len(selected_rows)}")
        print(f"  manifest path: {args.manifest_output}")
        print(f"  output path: {args.output_dir}")
        return

    report_path = args.output_dir.expanduser() / "meta" / "conversion_report.json"
    info_path = args.output_dir.expanduser() / "meta" / "info.json"
    tasks_path = args.output_dir.expanduser() / "meta" / "tasks.jsonl"

    report = read_json(report_path)
    info = read_json(info_path)
    tasks = [row["task"] for row in read_jsonl(tasks_path)]

    print(f"  manifest path: {args.manifest_output}")
    print(f"  output path: {args.output_dir}")
    print(f"  episodes: {info.get('total_episodes')}")
    print(f"  frames: {info.get('total_frames')}")
    print(f"  fps: {info.get('fps')}")
    print(f"  prop_mode: {report.get('prop_mode')}")
    print(f"  dim: {report.get('prop_dim')}")
    print(f"  tasks: {tasks}")
    print("  suggested upload command:")
    print(suggested_upload_command(args.output_dir))
    print("  suggested server projector-only finetune:")
    print(suggested_server_finetune_command(args.output_dir))


def main() -> None:
    args = parse_args()

    ensure_extracted(
        oakink2_root=args.oakink2_root,
        skip_extract=args.skip_extract,
        dry_run=args.dry_run,
    )

    selected_rows = run_manifest_stage(args)
    run_convert_stage(args)
    run_validate_stage(args)
    print_done_summary(args, selected_rows)


if __name__ == "__main__":
    main()
