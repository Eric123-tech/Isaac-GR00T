#!/usr/bin/env python3
"""Focused left_hand_qpos bottleneck audit for ARTIMANO rendered-RGB GR00T data.

Audit only: this script does not modify the dataset, modality config, training
code, stats.json, or relative_stats.json.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from copy import deepcopy
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = Path("/home/david/IsaacLab")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ISAACLAB_ROOT / "scripts/gr00t_bridge") not in sys.path:
    sys.path.insert(0, str(ISAACLAB_ROOT / "scripts/gr00t_bridge"))

DATASET_PATH = Path(
    "/mnt/data/haoyu_data/oakink2/gr00t_lerobot/"
    "fccd8_artimano_torch_wrist_qpos_isaac_render_rgb_row68_window_relstats_recomputed"
)
RENDERED_FRAME_DIR = Path("/mnt/data/haoyu_data/isaaclab_assets/oakink2_fccd8/rendered_rgb_fixed_row68_window_frames")
SIDE_CAR_NAME = "isaac_render_rgb_row68_window_state_action.npz"
CONFIG_PATH = REPO_ROOT / "examples/ARTIMANO/oakink2_artimano_bimanual_config.py"
CONFIG_VAR = "oakink2_artimano_bimanual_config"
EMBODIMENT_TAG = "NEW_EMBODIMENT"
OUT_DIR = Path("/mnt/data/haoyu_data/isaaclab_assets/oakink2_fccd8/left_qpos_bottleneck_audit")

ROWS_TRAINING = [0, 1, 20, 40, 68, 100, 140, 144]
ROWS_PRED = [0, 20, 40, 68, 100, 140]
HORIZON = 16
TASK_DESCRIPTION = "Cap the bottle."

QPOS_SLICES = {
    "left_hand_qpos": slice(18, 40),
    "right_hand_qpos": slice(40, 62),
}
STATE_LAYOUT = {
    "left_wrist_pose": slice(0, 9),
    "right_wrist_pose": slice(9, 18),
    "left_hand_qpos": slice(18, 40),
    "right_hand_qpos": slice(40, 62),
}
MT_ORDER = [
    "j_index1y", "j_index1z", "j_index2", "j_index3",
    "j_middle1y", "j_middle1z", "j_middle2", "j_middle3",
    "j_pinky1y", "j_pinky1z", "j_pinky2", "j_pinky3",
    "j_ring1y", "j_ring1z", "j_ring2", "j_ring3",
    "j_thumb1x", "j_thumb1y", "j_thumb1z", "j_thumb2y", "j_thumb2z", "j_thumb3",
]
ISAACLAB_ORDER = [
    "j_index1y", "j_middle1y", "j_pinky1y", "j_ring1y", "j_thumb1x",
    "j_index1z", "j_middle1z", "j_pinky1z", "j_ring1z", "j_thumb1y",
    "j_index2", "j_middle2", "j_pinky2", "j_ring2", "j_thumb1z",
    "j_index3", "j_middle3", "j_pinky3", "j_ring3",
    "j_thumb2y", "j_thumb2z", "j_thumb3",
]
MT_TO_ISAACLAB = [MT_ORDER.index(name) for name in ISAACLAB_ORDER]
ISAACLAB_TO_MT = [ISAACLAB_ORDER.index(name) for name in MT_ORDER]
FINGER_BLOCKS = {
    "index": [0, 1, 2, 3],
    "middle": [4, 5, 6, 7],
    "pinky": [8, 9, 10, 11],
    "ring": [12, 13, 14, 15],
    "thumb": [16, 17, 18, 19, 20, 21],
}


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default) + "\n")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_sidecar_or_parquet(dataset_path: Path):
    sidecar = dataset_path / "retarget" / SIDE_CAR_NAME
    if sidecar.exists():
        data = np.load(sidecar, allow_pickle=True)
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        frame_indices = np.asarray(data.get("frame_indices", np.arange(len(states))), dtype=np.int64)
        source_frame_ids = np.asarray(data["source_frame_ids"], dtype=np.int64) if "source_frame_ids" in data else None
        return states, actions, frame_indices, source_frame_ids, f"sidecar npz: {sidecar}"

    parquet_path = dataset_path / "data/chunk-000/episode_000000.parquet"
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        frame_indices = (
            np.asarray(table.column("frame_index").to_numpy(zero_copy_only=False), dtype=np.int64)
            if "frame_index" in table.column_names
            else np.arange(len(states), dtype=np.int64)
        )
        return states.reshape(len(states), -1), actions.reshape(len(actions), -1), frame_indices, None, f"parquet: {parquet_path}"
    except Exception as exc:
        raise RuntimeError(f"Could not load {sidecar} or parquet {parquet_path}: {exc}") from exc


def load_task_description(dataset_path: Path) -> str:
    path = dataset_path / "meta/tasks.jsonl"
    if not path.exists():
        return TASK_DESCRIPTION
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("task_index", 0)) == 0:
            return str(record.get("task", TASK_DESCRIPTION))
    return TASK_DESCRIPTION


def qpos_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        values = values.reshape(values.shape[0], -1)
    diffs = np.diff(values, axis=0) if len(values) > 1 else np.zeros((0, values.shape[1]), dtype=np.float64)
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "range": np.max(values, axis=0) - np.min(values, axis=0),
        "rms": np.sqrt(np.mean(values * values, axis=0)),
        "std_norm": float(np.linalg.norm(np.std(values, axis=0))),
        "range_norm": float(np.linalg.norm(np.max(values, axis=0) - np.min(values, axis=0))),
        "trajectory_length": float(np.sum(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.0,
        "delta_mean": np.mean(diffs, axis=0) if len(diffs) else np.zeros(values.shape[1]),
        "delta_std": np.std(diffs, axis=0) if len(diffs) else np.zeros(values.shape[1]),
        "delta_max_abs": np.max(np.abs(diffs), axis=0) if len(diffs) else np.zeros(values.shape[1]),
        "delta_std_norm": float(np.linalg.norm(np.std(diffs, axis=0))) if len(diffs) else 0.0,
    }


def audit_raw_qpos(states: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    side_stats: dict[str, dict[str, Any]] = {}
    for side, sl in QPOS_SLICES.items():
        side_name = "left" if side.startswith("left") else "right"
        state_q = states[:, sl]
        action_q = actions[:, sl]
        rel_q = action_q - state_q
        side_stats[f"{side_name}_state"] = qpos_stats(state_q)
        side_stats[f"{side_name}_action"] = qpos_stats(action_q)
        side_stats[f"{side_name}_relative"] = qpos_stats(rel_q)
        next_err = action_q[:-1] - state_q[1:]
        summary[f"{side_name}_action_equals_next_state_l2"] = float(np.linalg.norm(next_err))
        summary[f"{side_name}_action_equals_next_state_max_abs"] = float(np.max(np.abs(next_err))) if next_err.size else 0.0
        for kind, arr in [("state", state_q), ("action", action_q), ("relative", rel_q)]:
            stats = qpos_stats(arr)
            for j, name in enumerate(MT_ORDER):
                rows.append(
                    {
                        "side": side_name,
                        "kind": kind,
                        "joint_index": j,
                        "joint_name": name,
                        "mean": float(stats["mean"][j]),
                        "std": float(stats["std"][j]),
                        "min": float(stats["min"][j]),
                        "max": float(stats["max"][j]),
                        "range": float(stats["range"][j]),
                        "rms": float(stats["rms"][j]),
                        "delta_mean": float(stats["delta_mean"][j]),
                        "delta_std": float(stats["delta_std"][j]),
                        "delta_max_abs": float(stats["delta_max_abs"][j]),
                        "trajectory_length": float(stats["trajectory_length"]),
                        "std_norm": float(stats["std_norm"]),
                        "delta_std_norm": float(stats["delta_std_norm"]),
                    }
                )
    left_state = side_stats["left_state"]
    right_state = side_stats["right_state"]
    left_rel = side_stats["left_relative"]
    right_rel = side_stats["right_relative"]
    summary.update(
        {
            "left_std_norm": left_state["std_norm"],
            "right_std_norm": right_state["std_norm"],
            "left_std_norm_over_right": left_state["std_norm"] / (right_state["std_norm"] + 1e-12),
            "left_delta_std_norm": left_rel["std_norm"],
            "right_delta_std_norm": right_rel["std_norm"],
            "left_delta_std_norm_over_right": left_rel["std_norm"] / (right_rel["std_norm"] + 1e-12),
            "left_trajectory_length": left_state["trajectory_length"],
            "right_trajectory_length": right_state["trajectory_length"],
            "left_trajectory_length_over_right": left_state["trajectory_length"] / (right_state["trajectory_length"] + 1e-12),
        }
    )
    write_csv(OUT_DIR / "raw_qpos_stats.csv", rows)
    write_json(OUT_DIR / "raw_qpos_stats.json", {"summary": summary, "source": "states/actions sidecar", "rows": rows})
    return {"rows": rows, "summary": summary, "side_stats": side_stats}

def relative_stats_std_array(rel_stats: dict[str, Any], side: str) -> np.ndarray:
    arr = np.asarray(rel_stats[side]["std"], dtype=np.float64)
    if arr.ndim == 1:
        if arr.size == 22:
            return arr.reshape(1, 22)
        if arr.size % 22 == 0:
            return arr.reshape(arr.size // 22, 22)
    if arr.shape[-1] != 22:
        return arr.reshape(-1, 22)
    return arr.reshape(-1, 22)


def recompute_horizon_relative_qpos_std(states: np.ndarray, actions: np.ndarray, sl: slice, horizon: int) -> np.ndarray:
    rows = []
    for h in range(horizon):
        rel_samples = []
        for t in range(len(states)):
            idx = t + h
            if idx >= len(actions):
                break
            rel_samples.append(actions[idx, sl] - states[t, sl])
        if not rel_samples:
            rows.append(np.zeros(22, dtype=np.float64))
        else:
            rows.append(np.std(np.asarray(rel_samples, dtype=np.float64), axis=0))
    return np.asarray(rows, dtype=np.float64)


def audit_stats_json(states: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    stats = json.loads((DATASET_PATH / "meta/stats.json").read_text())
    rel_stats = json.loads((DATASET_PATH / "meta/relative_stats.json").read_text())
    rows: list[dict[str, Any]] = []
    flags: list[str] = []
    rel_by_side: dict[str, np.ndarray] = {}
    recomputed_rel_by_side: dict[str, np.ndarray] = {}
    for side, sl in QPOS_SLICES.items():
        side_name = "left" if side.startswith("left") else "right"
        current_raw_std = np.asarray(stats["observation.state"]["std"], dtype=np.float64)[sl]
        recomputed_raw_std = np.std(states[:, sl], axis=0).astype(np.float64)
        current_rel_std = relative_stats_std_array(rel_stats, side)
        recomputed_rel_std = recompute_horizon_relative_qpos_std(states, actions, sl, current_rel_std.shape[0])
        rel_by_side[side_name] = current_rel_std
        recomputed_rel_by_side[side_name] = recomputed_rel_std
        for h in range(current_rel_std.shape[0]):
            for j, name in enumerate(MT_ORDER):
                raw_ratio = current_raw_std[j] / (recomputed_raw_std[j] + 1e-12)
                rel_ratio = current_rel_std[h, j] / (recomputed_rel_std[h, j] + 1e-12)
                row_flags = []
                for label, value in [("stats_std", current_raw_std[j]), ("relative_stats_std", current_rel_std[h, j])]:
                    if value < 1e-5:
                        row_flags.append(f"{label}<1e-5")
                if current_rel_std[h, j] > 10.0 * np.median(current_rel_std[h] + 1e-12):
                    row_flags.append("relative_stats_std_gt_10x_horizon_median")
                if raw_ratio > 2 or raw_ratio < 0.5:
                    row_flags.append("stats/recomputed_raw_ratio_outside_[0.5,2]")
                if rel_ratio > 2 or rel_ratio < 0.5:
                    row_flags.append("relative_stats/recomputed_ratio_outside_[0.5,2]")
                record = {
                    "side": side_name,
                    "horizon": h,
                    "joint_index": j,
                    "joint_name": name,
                    "stats_json_state_std": float(current_raw_std[j]),
                    "recomputed_raw_state_std": float(recomputed_raw_std[j]),
                    "stats_json_state_std_over_recomputed": float(raw_ratio),
                    "relative_stats_std": float(current_rel_std[h, j]),
                    "recomputed_relative_target_std": float(recomputed_rel_std[h, j]),
                    "relative_stats_std_over_recomputed": float(rel_ratio),
                    "flags": ";".join(row_flags),
                }
                rows.append(record)
                if row_flags:
                    flags.append(f"{side_name}:h{h}:{name}:{','.join(row_flags)}")
    if "left" in rel_by_side and "right" in rel_by_side:
        lr_ratio = rel_by_side["left"] / (rel_by_side["right"] + 1e-12)
        for h in range(lr_ratio.shape[0]):
            for j, ratio in enumerate(lr_ratio[h]):
                if ratio > 2 or ratio < 0.5:
                    flags.append(f"left/right_relative_stats_ratio:h{h}:{MT_ORDER[j]}:{ratio:.4g}")
                    for row in rows:
                        if row["side"] == "left" and row["horizon"] == h and row["joint_index"] == j:
                            row["flags"] = (row["flags"] + ";" if row["flags"] else "") + "left/right_relative_stats_ratio_outside_[0.5,2]"
                            break
        lr_ratio_summary = {
            "min": float(np.min(lr_ratio)),
            "max": float(np.max(lr_ratio)),
            "median": float(np.median(lr_ratio)),
        }
    else:
        lr_ratio_summary = {"min": None, "max": None, "median": None}
    report = [
        "qpos stats.json / relative_stats.json audit",
        "",
        "relative_stats compared per horizon step and joint using action[t+h] - state[t].",
        f"flags_count = {len(flags)}",
        f"left/right relative_stats ratio summary = {lr_ratio_summary}",
        *[f"- {flag}" for flag in flags[:300]],
    ]
    if len(flags) > 300:
        report.append(f"... {len(flags) - 300} more flags omitted")
    write_csv(OUT_DIR / "qpos_stats_relative_stats_comparison.csv", rows)
    write_text(OUT_DIR / "qpos_stats_relative_stats_report.txt", "\n".join(report) + "\n")
    return {"rows": rows, "flags": flags, "left_right_relative_stats_ratio": lr_ratio_summary}

def load_config_module(path: Path):
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    MODALITY_CONFIGS.pop("new_embodiment", None)
    spec = importlib.util.spec_from_file_location("artimano_bottleneck_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit_training_targets(rows_to_sample: list[int]) -> dict[str, Any]:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.state_action.state_action_processor import StateActionProcessor

    module = load_config_module(CONFIG_PATH)
    modality_config = deepcopy(getattr(module, CONFIG_VAR))
    embodiment = EmbodimentTag.resolve(EMBODIMENT_TAG)
    loader = LeRobotEpisodeLoader(DATASET_PATH, modality_config)
    stats = loader.get_dataset_statistics()
    processor = StateActionProcessor(
        modality_configs={embodiment.value: modality_config},
        statistics={embodiment.value: stats},
        use_relative_action=True,
        clip_outliers=True,
    )
    traj = loader[0]
    records: list[dict[str, Any]] = []
    debug: dict[str, Any] = {}
    flags: list[str] = []
    for row in rows_to_sample:
        if row + HORIZON > len(traj):
            flags.append(f"row {row} skipped; horizon exceeds trajectory length {len(traj)}")
            continue
        sample = extract_step_data(traj, row, modality_config, embodiment, allow_padding=False)
        state_dict = {k: np.asarray(v, dtype=np.float32) for k, v in sample.states.items()}
        action_dict = {k: np.asarray(v, dtype=np.float32) for k, v in sample.actions.items()}
        _norm_state, norm_action = processor.apply(state=state_dict, action=action_dict, embodiment_tag=embodiment.value)
        row_debug: dict[str, Any] = {}
        for key in ["left_hand_qpos", "right_hand_qpos"]:
            raw_chunk = np.asarray(action_dict[key], dtype=np.float64).reshape(HORIZON, -1)
            state_ref = np.asarray(state_dict[key], dtype=np.float64).reshape(-1)
            rel_chunk = raw_chunk - state_ref[None, :]
            norm_chunk = np.asarray(norm_action[key], dtype=np.float64).reshape(HORIZON, -1)
            side = "left" if key.startswith("left") else "right"
            rec = {
                "row": row,
                "side": side,
                "raw_target_chunk_std_norm": float(np.linalg.norm(np.std(raw_chunk, axis=0))),
                "relative_target_chunk_std_norm": float(np.linalg.norm(np.std(rel_chunk, axis=0))),
                "relative_target_chunk_abs_max": float(np.max(np.abs(rel_chunk))),
                "normalized_target_mean": float(np.mean(norm_chunk)),
                "normalized_target_std": float(np.std(norm_chunk)),
                "normalized_target_min": float(np.min(norm_chunk)),
                "normalized_target_max": float(np.max(norm_chunk)),
                "normalized_target_abs_max": float(np.max(np.abs(norm_chunk))),
                "normalized_has_nan_or_inf": bool(not np.all(np.isfinite(norm_chunk))),
            }
            if rec["normalized_has_nan_or_inf"]:
                flags.append(f"{key} row {row}: normalized target has NaN/Inf")
            if rec["normalized_target_std"] < 1e-6:
                flags.append(f"{key} row {row}: normalized target nearly zero")
            records.append(rec)
            row_debug[key] = {"raw": raw_chunk, "relative_manual": rel_chunk, "normalized": norm_chunk}
        debug[f"row_{row}"] = row_debug
    # cross-side ratio flags
    by_row = {}
    for rec in records:
        by_row.setdefault(rec["row"], {})[rec["side"]] = rec
    for row, sides in by_row.items():
        if "left" in sides and "right" in sides:
            ratio = sides["left"]["normalized_target_std"] / (sides["right"]["normalized_target_std"] + 1e-12)
            sides["left"]["left_normalized_std_over_right"] = ratio
            sides["right"]["left_normalized_std_over_right"] = ratio
            if ratio > 2 or ratio < 0.5:
                flags.append(f"row {row}: left/right normalized qpos std ratio {ratio:.4g}")
    write_csv(OUT_DIR / "training_batch_qpos_target_stats.csv", records)
    np.savez_compressed(OUT_DIR / "training_batch_qpos_target_debug.npz", **debug)
    return {"records": records, "flags": flags}


def ensure_hwc_rgb_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D image, got {arr.shape}")
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise RuntimeError(f"Expected RGB, got {arr.shape}")
    if arr.dtype != np.uint8:
        if arr.size and float(arr.max()) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def load_png_or_video_frame(row: int) -> np.ndarray:
    path = RENDERED_FRAME_DIR / f"frame_{row:06d}.png"
    if path.exists():
        from PIL import Image

        return ensure_hwc_rgb_uint8(np.asarray(Image.open(path).convert("RGB")))
    video_path = DATASET_PATH / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, row)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {row} from {video_path}")
    return ensure_hwc_rgb_uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def split_state_dict(state62: np.ndarray) -> dict[str, np.ndarray]:
    state62 = np.asarray(state62, dtype=np.float32).reshape(62)
    return {key: state62[sl] for key, sl in STATE_LAYOUT.items()}


def build_obs(state62: np.ndarray, rgb: np.ndarray, task_description: str) -> dict[str, Any]:
    pieces = split_state_dict(state62)
    return {
        "video": {"ego_view": ensure_hwc_rgb_uint8(rgb)[None, None, ...]},
        "state": {key: value[None, None, :].astype(np.float32) for key, value in pieces.items()},
        "language": {"annotation.human.action.task_description": [[task_description]]},
    }


def action_to_state62(action: dict[str, Any], horizon: int) -> np.ndarray:
    arrays = [
        np.asarray(action["left_wrist_pose"], dtype=np.float32)[0, :horizon],
        np.asarray(action["right_wrist_pose"], dtype=np.float32)[0, :horizon],
        np.asarray(action["left_hand_qpos"], dtype=np.float32)[0, :horizon],
        np.asarray(action["right_hand_qpos"], dtype=np.float32)[0, :horizon],
    ]
    return np.concatenate(arrays, axis=1).astype(np.float32)


def maybe_load_saved_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = OUT_DIR / "checkpoint4000_predictions_vs_gt.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return np.asarray(data["rows"], dtype=np.int64), np.asarray(data["pred"], dtype=np.float32), np.asarray(data["gt"], dtype=np.float32)


def run_or_load_predictions(states: np.ndarray, actions: np.ndarray, source_frame_ids: np.ndarray | None, args) -> dict[str, Any]:
    saved = maybe_load_saved_predictions() if not args.force_server else None
    if saved is not None:
        pred_rows, pred, gt = saved
        print(f"[INFO] Loaded saved predictions from {OUT_DIR / 'checkpoint4000_predictions_vs_gt.npz'}")
    else:
        if args.no_server:
            raise RuntimeError("Prediction audit requires saved predictions or a running GR00T server. Remove --no-server to call server.")
        from lightweight_gr00t_client import LightweightGR00TClient

        client = LightweightGR00TClient(host=args.host, port=args.port, timeout_ms=args.timeout_ms)
        client.ping()
        pred_chunks = []
        gt_chunks = []
        pred_rows = []
        task_description = load_task_description(DATASET_PATH)
        for row in ROWS_PRED:
            if row + HORIZON > len(actions):
                continue
            rgb = load_png_or_video_frame(row)
            obs = build_obs(states[row], rgb, task_description)
            client.reset()
            t0 = time.time()
            action, _info = client.get_action(obs)
            infer_dt = time.time() - t0
            pred62 = action_to_state62(action, HORIZON)
            gt62 = actions[row:row + HORIZON]
            pred_chunks.append(pred62)
            gt_chunks.append(gt62)
            pred_rows.append(row)
            print(f"[INFO] predicted row={row} source_frame_id={int(source_frame_ids[row]) if source_frame_ids is not None else row} infer_dt={infer_dt:.3f}s")
        pred_rows = np.asarray(pred_rows, dtype=np.int64)
        pred = np.asarray(pred_chunks, dtype=np.float32)
        gt = np.asarray(gt_chunks, dtype=np.float32)
        np.savez_compressed(OUT_DIR / "checkpoint4000_predictions_vs_gt.npz", rows=pred_rows, pred=pred, gt=gt)
    return audit_prediction_errors(states, actions, source_frame_ids, pred_rows, pred, gt)


def audit_prediction_errors(states: np.ndarray, actions: np.ndarray, source_frame_ids: np.ndarray | None, pred_rows: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    by_row_h: list[dict[str, Any]] = []
    joint_abs_errors: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    baseline_rows: list[dict[str, Any]] = []
    for i, row in enumerate(pred_rows):
        for h in range(pred.shape[1]):
            pred_left = pred[i, h, 18:40]
            pred_right = pred[i, h, 40:62]
            gt_left = gt[i, h, 18:40]
            gt_right = gt[i, h, 40:62]
            left_abs = np.abs(pred_left - gt_left)
            right_abs = np.abs(pred_right - gt_right)
            joint_abs_errors["left"].append(left_abs)
            joint_abs_errors["right"].append(right_abs)
            by_row_h.append(
                {
                    "row": int(row),
                    "source_frame_id": int(source_frame_ids[row]) if source_frame_ids is not None else int(row),
                    "horizon": h,
                    "left_qpos_l2": float(np.linalg.norm(pred_left - gt_left)),
                    "right_qpos_l2": float(np.linalg.norm(pred_right - gt_right)),
                    "left_qpos_rmse_per_dim": float(np.sqrt(np.mean((pred_left - gt_left) ** 2))),
                    "right_qpos_rmse_per_dim": float(np.sqrt(np.mean((pred_right - gt_right) ** 2))),
                }
            )
    left_err = np.asarray(joint_abs_errors["left"])
    right_err = np.asarray(joint_abs_errors["right"])
    by_joint = []
    for side, err in [("left", left_err), ("right", right_err)]:
        for j, name in enumerate(MT_ORDER):
            by_joint.append(
                {
                    "side": side,
                    "joint_index": j,
                    "joint_name": name,
                    "mean_abs_error": float(np.mean(err[:, j])),
                    "rmse": float(np.sqrt(np.mean(err[:, j] ** 2))),
                    "max_abs_error": float(np.max(err[:, j])),
                }
            )
    mean_left_q = np.mean(states[:, 18:40], axis=0)
    mean_right_q = np.mean(states[:, 40:62], axis=0)
    for i, row in enumerate(pred_rows):
        gt_left = gt[i, :, 18:40]
        gt_right = gt[i, :, 40:62]
        hold_left = np.repeat(states[row, 18:40][None, :], gt.shape[1], axis=0)
        hold_right = np.repeat(states[row, 40:62][None, :], gt.shape[1], axis=0)
        mean_left = np.repeat(mean_left_q[None, :], gt.shape[1], axis=0)
        mean_right = np.repeat(mean_right_q[None, :], gt.shape[1], axis=0)
        for side, pred_q, gt_q, mean_q, hold_q in [
            ("left", pred[i, :, 18:40], gt_left, mean_left, hold_left),
            ("right", pred[i, :, 40:62], gt_right, mean_right, hold_right),
        ]:
            for name, candidate in [("model", pred_q), ("mean", mean_q), ("hold", hold_q), ("next_state_gt", gt_q)]:
                diff = candidate - gt_q
                baseline_rows.append(
                    {
                        "row": int(row),
                        "side": side,
                        "baseline": name,
                        "l2": float(np.linalg.norm(diff)),
                        "rmse_per_dim": float(np.sqrt(np.mean(diff * diff))),
                    }
                )
    write_csv(OUT_DIR / "checkpoint4000_qpos_error_by_row_horizon.csv", by_row_h)
    write_csv(OUT_DIR / "checkpoint4000_qpos_error_by_joint.csv", by_joint)
    write_csv(OUT_DIR / "checkpoint4000_qpos_baselines.csv", baseline_rows)
    summary: dict[str, Any] = {
        "rows": pred_rows.tolist(),
        "mean_left_qpos_l2_per_step": float(np.mean([r["left_qpos_l2"] for r in by_row_h])),
        "mean_right_qpos_l2_per_step": float(np.mean([r["right_qpos_l2"] for r in by_row_h])),
        "left_over_right_l2_ratio": float(np.mean([r["left_qpos_l2"] for r in by_row_h]) / (np.mean([r["right_qpos_l2"] for r in by_row_h]) + 1e-12)),
        "baseline_mean_by_side": {},
    }
    for side in ["left", "right"]:
        side_rows = [r for r in baseline_rows if r["side"] == side]
        for baseline in ["model", "mean", "hold", "next_state_gt"]:
            vals = [r["l2"] for r in side_rows if r["baseline"] == baseline]
            summary["baseline_mean_by_side"][f"{side}_{baseline}_l2"] = float(np.mean(vals))
    write_json(OUT_DIR / "checkpoint4000_qpos_error_summary.json", summary)
    return {"pred_rows": pred_rows, "pred": pred, "gt": gt, "summary": summary, "by_row_h": by_row_h}


def audit_order_swap(pred_info: dict[str, Any]) -> dict[str, Any]:
    pred = pred_info["pred"]
    gt = pred_info["gt"]
    pred_left = pred[:, :, 18:40].reshape(-1, 22)
    pred_right = pred[:, :, 40:62].reshape(-1, 22)
    gt_left = gt[:, :, 18:40].reshape(-1, 22)
    gt_right = gt[:, :, 40:62].reshape(-1, 22)
    candidates: list[dict[str, Any]] = []

    def add(name: str, candidate: np.ndarray, target: np.ndarray = gt_left):
        diff = candidate - target
        candidates.append({"candidate": name, "left_l2": float(np.linalg.norm(diff)), "left_rmse": float(np.sqrt(np.mean(diff * diff)))})

    add("identity", pred_left)
    add("left_right_swap_pred_left_vs_gt_right", pred_left, gt_right)
    add("left_right_swap_pred_right_vs_gt_left", pred_right, gt_left)
    add("pred_left_mt_to_isaaclab", pred_left[:, MT_TO_ISAACLAB])
    add("pred_left_isaaclab_to_mt", pred_left[:, ISAACLAB_TO_MT])
    block_names = list(FINGER_BLOCKS.keys())
    for perm in permutations(block_names):
        indices: list[int] = []
        for name in perm:
            indices.extend(FINGER_BLOCKS[name])
        add("finger_blocks_" + "-".join(perm), pred_left[:, indices])
    sign_rows = []
    for j, name in enumerate(MT_ORDER):
        pos_l2 = float(np.linalg.norm(pred_left[:, j] - gt_left[:, j]))
        neg_l2 = float(np.linalg.norm(-pred_left[:, j] - gt_left[:, j]))
        sign_rows.append(
            {
                "joint_index": j,
                "joint_name": name,
                "positive_l2": pos_l2,
                "negative_l2": neg_l2,
                "negative_improvement_frac": (pos_l2 - neg_l2) / (pos_l2 + 1e-12),
                "suspicious_sign_flip": bool(neg_l2 < 0.8 * pos_l2),
            }
        )
    candidates_sorted = sorted(candidates, key=lambda r: r["left_l2"])
    identity = next(r for r in candidates if r["candidate"] == "identity")
    best = candidates_sorted[0]
    improvement = (identity["left_l2"] - best["left_l2"]) / (identity["left_l2"] + 1e-12)
    for row in candidates:
        row["improvement_over_identity_frac"] = (identity["left_l2"] - row["left_l2"]) / (identity["left_l2"] + 1e-12)
    write_csv(OUT_DIR / "left_qpos_order_swap_candidates.csv", sorted(candidates, key=lambda r: r["left_l2"]))
    write_csv(OUT_DIR / "left_qpos_sign_flip_diagnostic.csv", sign_rows)
    report = [
        "left qpos joint-order / swap audit",
        "",
        f"identity_l2 = {identity['left_l2']:.8f}",
        f"best_candidate = {best['candidate']}",
        f"best_l2 = {best['left_l2']:.8f}",
        f"best_improvement_over_identity = {improvement:.4%}",
        "",
        "Top candidates:",
    ]
    for row in candidates_sorted[:15]:
        report.append(f"- {row['candidate']}: l2={row['left_l2']:.8f}, improvement={row['improvement_over_identity_frac']:.4%}")
    suspicious = [r for r in sign_rows if r["suspicious_sign_flip"]]
    report.extend(["", "Suspicious sign-flip joints:"])
    report.extend([f"- {r['joint_name']}: improvement={r['negative_improvement_frac']:.4%}" for r in suspicious] or ["- none"])
    write_text(OUT_DIR / "left_qpos_order_swap_audit_report.txt", "\n".join(report) + "\n")
    return {"identity": identity, "best": best, "improvement": improvement, "suspicious_sign_flips": suspicious}


def audit_visibility(states: np.ndarray, source_frame_ids: np.ndarray | None):
    from PIL import Image, ImageDraw, ImageFont

    images = []
    report_rows = []
    for row in ROWS_PRED:
        rgb = load_png_or_video_frame(row)
        image = Image.fromarray(rgb).resize((256, 256))
        draw = ImageDraw.Draw(image)
        source = int(source_frame_ids[row]) if source_frame_ids is not None else row
        draw.rectangle((0, 0, 255, 22), fill=(0, 0, 0))
        draw.text((4, 4), f"row {row} / src {source}", fill=(255, 255, 255))
        images.append(image)
        # Conservative automated label: rendered ego view is close-up, but no calibrated projection is used here.
        report_rows.append(
            {
                "row": row,
                "source_frame_id": source,
                "left_hand_visible": "partial/manual_check_required",
                "right_hand_visible": "partial/manual_check_required",
                "left_hand_occluded_by_object_or_right_hand": "partial/manual_check_required",
                "left_hand_near_image_boundary": "manual_check_required",
                "right_hand_near_image_boundary": "manual_check_required",
            }
        )
    cols = 3
    rows_n = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * 256, rows_n * 256), (40, 40, 40))
    for i, image in enumerate(images):
        sheet.paste(image, ((i % cols) * 256, (i // cols) * 256))
    sheet.save(OUT_DIR / "left_right_hand_visibility_contact_sheet.png")
    lines = [
        "left/right hand visibility audit",
        "",
        "No calibrated camera projection was used in this offline audit; labels are conservative and require manual inspection of the contact sheet.",
        f"contact_sheet = {OUT_DIR / 'left_right_hand_visibility_contact_sheet.png'}",
        "",
    ]
    for row in report_rows:
        lines.append(
            f"row={row['row']} source_frame_id={row['source_frame_id']} left={row['left_hand_visible']} right={row['right_hand_visible']} "
            f"left_occlusion={row['left_hand_occluded_by_object_or_right_hand']} left_boundary={row['left_hand_near_image_boundary']} right_boundary={row['right_hand_near_image_boundary']}"
        )
    write_text(OUT_DIR / "left_right_hand_visibility_report.txt", "\n".join(lines) + "\n")
    return {"rows": report_rows, "contact_sheet": str(OUT_DIR / "left_right_hand_visibility_contact_sheet.png")}


def classify_results(raw: dict[str, Any], stats: dict[str, Any], training: dict[str, Any], pred: dict[str, Any], order: dict[str, Any], visibility: dict[str, Any]) -> dict[str, str]:
    raw_summary = raw["summary"]
    gt_motion = "PASS"
    if max(raw_summary["left_std_norm_over_right"], raw_summary["left_delta_std_norm_over_right"], raw_summary["left_trajectory_length_over_right"]) > 1.5:
        gt_motion = "WARN"
    if max(raw_summary["left_std_norm_over_right"], raw_summary["left_delta_std_norm_over_right"], raw_summary["left_trajectory_length_over_right"]) > 3.0:
        gt_motion = "FAIL"

    stats_result = "PASS"
    mismatch_stats = [
        f for f in stats["flags"]
        if "stats/recomputed_raw_ratio_outside" in f
        or "relative_stats/recomputed_ratio_outside" in f
        or "relative_stats_std_gt_10x_horizon_median" in f
    ]
    distribution_stats = [f for f in stats["flags"] if f not in mismatch_stats]
    if distribution_stats:
        stats_result = "WARN"
    if mismatch_stats:
        stats_result = "FAIL"

    training_result = "PASS"
    if training["flags"]:
        training_result = "WARN"
    if any("NaN/Inf" in f for f in training["flags"]):
        training_result = "FAIL"

    pred_summary = pred["summary"]
    left_model = pred_summary["baseline_mean_by_side"].get("left_model_l2", float("inf"))
    left_mean = pred_summary["baseline_mean_by_side"].get("left_mean_l2", float("inf"))
    left_hold = pred_summary["baseline_mean_by_side"].get("left_hold_l2", float("inf"))
    prediction_result = "PASS"
    if pred_summary["left_over_right_l2_ratio"] > 1.2:
        prediction_result = "WARN"
    if left_model >= min(left_mean, left_hold) * 0.98:
        prediction_result = "WARN"
    if left_model > min(left_mean, left_hold) * 1.1:
        prediction_result = "FAIL"

    order_result = "PASS"
    if order["improvement"] > 0.2:
        order_result = "FAIL"
    elif order["improvement"] > 0.1 or order["suspicious_sign_flips"]:
        order_result = "WARN"

    visibility_result = "WARN"  # no calibrated projection/manual labels are available in this offline script
    overall = "PASS"
    values = [gt_motion, stats_result, training_result, prediction_result, order_result, visibility_result]
    if "FAIL" in values:
        overall = "FAIL"
    elif "WARN" in values:
        overall = "WARN"
    return {
        "LEFT_QPOS_GT_MOTION_RESULT": gt_motion,
        "LEFT_QPOS_STATS_RESULT": stats_result,
        "LEFT_QPOS_TRAINING_TARGET_RESULT": training_result,
        "LEFT_QPOS_PREDICTION_RESULT": prediction_result,
        "LEFT_QPOS_ORDER_SWAP_RESULT": order_result,
        "LEFT_QPOS_VISIBILITY_RESULT": visibility_result,
        "OVERALL_LEFT_QPOS_BOTTLENECK_AUDIT_RESULT": overall,
    }


def write_final_report(raw, stats, training, pred, order, visibility, results, data_source: str):
    lines = [
        "ARTIMANO left_hand_qpos bottleneck audit",
        "",
        f"dataset = {DATASET_PATH}",
        f"data_source = {data_source}",
        f"rendered_frame_dir = {RENDERED_FRAME_DIR}",
        f"modality_config = {CONFIG_PATH}",
        "",
        "A. Raw GT qpos motion",
        f"left_std_norm/right = {raw['summary']['left_std_norm_over_right']:.6f}",
        f"left_delta_std_norm/right = {raw['summary']['left_delta_std_norm_over_right']:.6f}",
        f"left_trajectory_length/right = {raw['summary']['left_trajectory_length_over_right']:.6f}",
        f"left action[t] vs state[t+1] qpos L2 = {raw['summary']['left_action_equals_next_state_l2']:.8g}",
        f"right action[t] vs state[t+1] qpos L2 = {raw['summary']['right_action_equals_next_state_l2']:.8g}",
        "",
        "B. stats.json / relative_stats.json",
        f"stats flags count = {len(stats['flags'])}",
        "",
        "C. Actual GR00T training-batch qpos targets",
        f"training target flags count = {len(training['flags'])}",
        *[f"- {flag}" for flag in training["flags"][:20]],
        "",
        "D. checkpoint-4000 prediction-vs-GT qpos",
        f"mean_left_qpos_l2_per_step = {pred['summary']['mean_left_qpos_l2_per_step']:.8f}",
        f"mean_right_qpos_l2_per_step = {pred['summary']['mean_right_qpos_l2_per_step']:.8f}",
        f"left/right qpos L2 ratio = {pred['summary']['left_over_right_l2_ratio']:.6f}",
        f"left model L2 = {pred['summary']['baseline_mean_by_side'].get('left_model_l2', float('nan')):.8f}",
        f"left mean baseline L2 = {pred['summary']['baseline_mean_by_side'].get('left_mean_l2', float('nan')):.8f}",
        f"left hold baseline L2 = {pred['summary']['baseline_mean_by_side'].get('left_hold_l2', float('nan')):.8f}",
        "",
        "E. Joint-order / swap audit",
        f"identity_l2 = {order['identity']['left_l2']:.8f}",
        f"best_candidate = {order['best']['candidate']}",
        f"best_l2 = {order['best']['left_l2']:.8f}",
        f"best_improvement_over_identity = {order['improvement']:.4%}",
        "",
        "F. Visibility audit",
        f"contact_sheet = {visibility['contact_sheet']}",
        "visibility_result_note = WARN because calibrated projection/manual labels are not available in this offline audit.",
        "",
        "Final result lines:",
    ]
    for key, value in results.items():
        lines.append(f"{key} = {value}")
    write_text(OUT_DIR / "left_qpos_bottleneck_report.txt", "\n".join(lines) + "\n")
    write_json(
        OUT_DIR / "left_qpos_bottleneck_summary.json",
        {
            "dataset": str(DATASET_PATH),
            "data_source": data_source,
            "raw_summary": raw["summary"],
            "stats_flags": stats["flags"],
            "training_flags": training["flags"],
            "prediction_summary": pred["summary"],
            "order_summary": {"identity": order["identity"], "best": order["best"], "improvement": order["improvement"], "suspicious_sign_flips": order["suspicious_sign_flips"]},
            "visibility": visibility,
            "results": results,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--no-server", action="store_true", help="Do not call GR00T server; require saved predictions.")
    parser.add_argument("--force-server", action="store_true", help="Ignore saved predictions and query server again.")
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    states, actions, frame_indices, source_frame_ids, data_source = load_sidecar_or_parquet(DATASET_PATH)
    print(f"[INFO] Loaded states shape = {states.shape}")
    print(f"[INFO] Loaded actions shape = {actions.shape}")
    print(f"[INFO] data_source = {data_source}")
    if states.shape[1] != 62 or actions.shape[1] != 62:
        raise RuntimeError(f"Expected 62D states/actions, got {states.shape}, {actions.shape}")

    raw = audit_raw_qpos(states, actions)
    stats = audit_stats_json(states, actions)
    training = audit_training_targets(ROWS_TRAINING)
    pred = run_or_load_predictions(states, actions, source_frame_ids, args)
    order = audit_order_swap(pred)
    visibility = audit_visibility(states, source_frame_ids)
    results = classify_results(raw, stats, training, pred, order, visibility)
    write_final_report(raw, stats, training, pred, order, visibility, results, data_source)

    print(f"[INFO] Saved audit outputs to: {OUT_DIR}")
    for key, value in results.items():
        print(f"{key} = {value}")


if __name__ == "__main__":
    main()
