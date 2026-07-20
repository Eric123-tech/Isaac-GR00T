#!/usr/bin/env python3
"""Audit rendered-RGB row68-window dataset at GR00T training-batch level.

This script is intentionally read-only with respect to the dataset.  It checks
raw LeRobot metadata, actual GR00T LeRobotEpisodeLoader/extract_step_data
semantics, StateActionProcessor action target transforms, boundary behavior,
FPS/timestamp metadata, and stats/relative_stats consistency.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_PATH = Path(
    os.environ.get(
        "DATASET_PATH",
        "/mnt/data/haoyu_data/oakink2/gr00t_lerobot/"
        "fccd8_artimano_torch_wrist_qpos_isaac_render_rgb_row68_window",
    )
)
MODALITY_CONFIG_PATH = REPO_ROOT / "examples/ARTIMANO/oakink2_artimano_bimanual_config.py"
EMBODIMENT_TAG_NAME = "NEW_EMBODIMENT"
ROWS = [0, 1, 20, 40, 68, 100, 140, 144, 145, 150, 159]
NON_BOUNDARY_ROWS = [0, 1, 20, 40, 68, 100, 140]
BOUNDARY_ROWS = [144, 145, 150, 159]
ACTION_HORIZON = 16
SEGMENTS = {
    "left_wrist_pose": slice(0, 9),
    "right_wrist_pose": slice(9, 18),
    "left_hand_qpos": slice(18, 40),
    "right_hand_qpos": slice(40, 62),
    "wrist_xyz_only": np.r_[0:3, 9:12],
    "wrist_rot6d_only": np.r_[3:9, 12:18],
    "qpos_only": np.r_[18:62],
}
KEY_SLICES = {
    "left_wrist_pose": slice(0, 9),
    "right_wrist_pose": slice(9, 18),
    "left_hand_qpos": slice(18, 40),
    "right_hand_qpos": slice(40, 62),
}
OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        "/mnt/data/haoyu_data/isaaclab_assets/oakink2_fccd8/gr00t_training_batch_audit",
    )
)
REPORT_PATH = OUTPUT_DIR / "action_chunk_alignment_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "action_chunk_alignment_summary.json"
ROWS_CSV_PATH = OUTPUT_DIR / "action_chunk_alignment_rows.csv"
STATS_REPORT_PATH = OUTPUT_DIR / "stats_relative_stats_audit_report.txt"
DEBUG_NPZ_PATH = OUTPUT_DIR / "debug_expected_vs_actual_chunks.npz"
WINDOW_STATS_PREVIEW_PATH = OUTPUT_DIR / "recomputed_window_stats_preview.json"
WINDOW_REL_STATS_PREVIEW_PATH = OUTPUT_DIR / "recomputed_window_relative_stats_preview.json"
VIDEO_PATH = DATASET_PATH / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
SIDE_CAR_NAME = "isaac_render_rgb_row68_window_state_action.npz"
NUM_TOL = 1e-5
CLEARLY_BETTER_RATIO = 0.5
STATS_WARN_MEAN_ABS_DIFF = 1e-3
REL_STATS_WARN_MEAN_ABS_DIFF = 1e-3


@dataclass
class SegmentMetric:
    row: int
    candidate_offset: int
    representation: str
    segment: str
    l2: float
    mse: float
    max_abs: float


@dataclass
class RowAlignmentSummary:
    row: int
    sampled_by_sharded_dataset: bool
    raw_available: bool
    transformed_available: bool
    raw_best_offset: int | None
    transformed_best_offset: int | None
    raw_offset0_l2: float | None
    transformed_offset0_l2: float | None
    transformed_best_l2: float | None
    boundary_behavior: str


def load_modality_config() -> None:
    spec = importlib.util.spec_from_file_location("oakink2_artimano_bimanual_config", MODALITY_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load modality config: {MODALITY_CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def as_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr.reshape(-1)


def stack_column(series) -> np.ndarray:
    return np.stack([as_float_array(v) for v in series.to_list()], axis=0).astype(np.float32)


def load_raw_parquet(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, Any]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required in the GR00T uv environment for this audit") from exc

    parquet_path = dataset_path / "data/chunk-000/episode_000000.parquet"
    df = pd.read_parquet(parquet_path)
    if "observation.state" not in df.columns or "action" not in df.columns:
        raise RuntimeError(f"Expected observation.state/action in {parquet_path}; columns={list(df.columns)}")
    states = stack_column(df["observation.state"])
    actions = stack_column(df["action"])
    frame_indices = df["frame_index"].to_numpy(dtype=np.int64) if "frame_index" in df.columns else np.arange(len(df), dtype=np.int64)
    timestamps = df["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in df.columns else None
    return states, actions, frame_indices, timestamps, df


def load_sidecar(dataset_path: Path) -> dict[str, Any] | None:
    sidecar_path = dataset_path / "retarget" / SIDE_CAR_NAME
    if not sidecar_path.exists():
        return None
    data = np.load(sidecar_path, allow_pickle=True)
    return {"path": str(sidecar_path), "keys": list(data.files), **{k: data[k] for k in data.files}}


def run_text_command(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def probe_video(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_read_frames": int(stream.get("nb_read_frames") or stream.get("nb_frames") or -1),
    }


def decode_video_frame_ffmpeg(video_path: Path, row: int) -> np.ndarray:
    props = probe_video(video_path)
    width = props["width"]
    height = props["height"]
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{int(row)})",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    arr = np.frombuffer(result.stdout, dtype=np.uint8)
    expected = height * width * 3
    if arr.size != expected:
        raise RuntimeError(f"ffmpeg decoded {arr.size} bytes for row {row}, expected {expected}")
    return arr.reshape(height, width, 3).copy()


def ensure_rgb_uint8(frame: Any) -> np.ndarray:
    try:
        from PIL import Image
    except Exception:
        Image = None
    if Image is not None and isinstance(frame, Image.Image):
        frame = np.asarray(frame.convert("RGB"))
    else:
        frame = np.asarray(frame)
    frame = np.squeeze(frame)
    if frame.ndim == 3 and frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim != 3:
        raise RuntimeError(f"Expected image with 3 dims, got {frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise RuntimeError(f"Expected RGB image, got {frame.shape}")
    if frame.dtype != np.uint8:
        if frame.size and float(frame.max()) <= 1.5:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def split_62(chunk: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(chunk[..., sl], dtype=np.float32) for key, sl in KEY_SLICES.items()}


def concat_action_dict(action_dict: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    return np.concatenate([np.asarray(action_dict[key], dtype=np.float32) for key in keys], axis=-1)


def metric(diff: np.ndarray) -> tuple[float, float, float]:
    diff64 = np.asarray(diff, dtype=np.float64)
    return float(np.linalg.norm(diff64)), float(np.mean(diff64 * diff64)), float(np.max(np.abs(diff64)))


def segment_metrics(actual: np.ndarray, expected: np.ndarray, row: int, offset: int, representation: str) -> list[SegmentMetric]:
    out = []
    for name, idx in SEGMENTS.items():
        diff = actual[:, idx] - expected[:, idx]
        l2, mse, max_abs = metric(diff)
        out.append(SegmentMetric(row, offset, representation, name, l2, mse, max_abs))
    return out


def get_candidate_chunk(actions: np.ndarray, row: int, offset: int, horizon: int) -> np.ndarray | None:
    start = row + offset
    end = start + horizon
    if start < 0 or end > len(actions):
        return None
    return actions[start:end].astype(np.float32)


def transform_chunk(processor, embodiment_value: str, state62: np.ndarray, chunk62: np.ndarray) -> np.ndarray:
    state_dict = {key: arr[None, :] for key, arr in split_62(state62).items()}
    action_dict = split_62(chunk62)
    _, norm_action = processor.apply(state=state_dict, action=action_dict, embodiment_tag=embodiment_value)
    return concat_action_dict(norm_action, list(KEY_SLICES.keys()))


def relative_chunk(processor, modality_config, state62: np.ndarray, chunk62: np.ndarray) -> dict[str, np.ndarray]:
    state_dict = split_62(state62)
    action_dict = split_62(chunk62)
    rel = {}
    action_keys = modality_config["action"].modality_keys
    action_configs = modality_config["action"].action_configs
    for key, action_config in zip(action_keys, action_configs):
        state_key = action_config.state_key or key
        rel[key] = processor._convert_to_relative_action(  # noqa: SLF001 - audit script intentionally checks native path
            action=action_dict[key],
            reference_state=state_dict[state_key],
            action_type=action_config.type,
            action_format=action_config.format,
        ).astype(np.float32)
    return rel


def stats_for_array(arr: np.ndarray) -> dict[str, Any]:
    arr64 = np.asarray(arr, dtype=np.float64)
    return {
        "mean": np.mean(arr64, axis=0).tolist(),
        "std": np.std(arr64, axis=0).tolist(),
        "min": np.min(arr64, axis=0).tolist(),
        "max": np.max(arr64, axis=0).tolist(),
        "q01": np.quantile(arr64, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr64, 0.99, axis=0).tolist(),
    }


def recompute_window_stats(states: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    return {
        "observation.state": stats_for_array(states),
        "action": stats_for_array(actions),
    }


def recompute_relative_stats(processor, modality_config, states: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    usable = len(actions) - ACTION_HORIZON + 1
    per_key = {key: [] for key in KEY_SLICES}
    for row in range(usable):
        chunk = actions[row : row + ACTION_HORIZON]
        rel = relative_chunk(processor, modality_config, states[row], chunk)
        for key, value in rel.items():
            per_key[key].append(value)
    return {key: stats_for_array(np.stack(values, axis=0)) for key, values in per_key.items()}


def compare_stat_group(current: dict[str, Any], recomputed: dict[str, Any], label: str) -> dict[str, Any]:
    out = {"label": label, "stat_metrics": {}, "dimension_match": True, "max_mean_abs_diff": 0.0}
    for stat_name in ["mean", "std", "min", "max", "q01", "q99"]:
        if stat_name not in current or stat_name not in recomputed:
            out["stat_metrics"][stat_name] = {"missing": True}
            out["dimension_match"] = False
            continue
        cur = np.asarray(current[stat_name], dtype=np.float64)
        rec = np.asarray(recomputed[stat_name], dtype=np.float64)
        if cur.shape != rec.shape:
            out["stat_metrics"][stat_name] = {"shape_current": list(cur.shape), "shape_recomputed": list(rec.shape), "shape_match": False}
            out["dimension_match"] = False
            continue
        diff = cur - rec
        mean_abs = float(np.mean(np.abs(diff)))
        out["stat_metrics"][stat_name] = {
            "shape": list(cur.shape),
            "shape_match": True,
            "mean_abs_diff": mean_abs,
            "max_abs_diff": float(np.max(np.abs(diff))),
        }
        out["max_mean_abs_diff"] = max(out["max_mean_abs_diff"], mean_abs)
    return out


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_modality_config()

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset, extract_step_data
    from gr00t.data.state_action.state_action_processor import StateActionProcessor

    embodiment = EmbodimentTag.resolve(EMBODIMENT_TAG_NAME)
    embodiment_value = embodiment.value
    modality_config = MODALITY_CONFIGS[embodiment_value]
    action_keys = modality_config["action"].modality_keys

    report_lines: list[str] = []
    report_lines.append("GR00T rendered-RGB row68-window training-batch alignment audit")
    report_lines.append("")
    report_lines.append(f"dataset = {DATASET_PATH}")
    report_lines.append(f"modality_config = {MODALITY_CONFIG_PATH}")
    report_lines.append(f"embodiment = {EMBODIMENT_TAG_NAME} ({embodiment_value})")
    report_lines.append(f"rows = {ROWS}")
    report_lines.append(f"ACTION_HORIZON = {ACTION_HORIZON}")
    report_lines.append("")

    summary: dict[str, Any] = {
        "dataset": str(DATASET_PATH),
        "modality_config": str(MODALITY_CONFIG_PATH),
        "embodiment": embodiment_value,
        "rows": ROWS,
        "action_horizon": ACTION_HORIZON,
        "errors": [],
    }

    states = actions = frame_indices = timestamps = raw_df = None
    try:
        states, actions, frame_indices, timestamps, raw_df = load_raw_parquet(DATASET_PATH)
        sidecar = load_sidecar(DATASET_PATH)
        info = json.loads((DATASET_PATH / "meta/info.json").read_text())
        modality_meta = json.loads((DATASET_PATH / "meta/modality.json").read_text())
        video_props = probe_video(VIDEO_PATH)
        action_next_diff = actions[:-1] - states[1:]
        action_state_diff = actions[:-1] - states[:-1]
        timestamp_step = None if timestamps is None or len(timestamps) < 2 else np.diff(timestamps)

        raw_sanity = {
            "states_shape": list(states.shape),
            "actions_shape": list(actions.shape),
            "frame_index_is_0_to_159": bool(np.array_equal(frame_indices, np.arange(len(frame_indices)))),
            "timestamp_step_mean": None if timestamp_step is None else float(np.mean(timestamp_step)),
            "timestamp_step_max_abs_from_0p1": None if timestamp_step is None else float(np.max(np.abs(timestamp_step - 0.1))),
            "info_fps": info.get("fps"),
            "video": video_props,
            "action_equals_next_state_mean_abs": float(np.mean(np.abs(action_next_diff))),
            "action_equals_next_state_max_abs": float(np.max(np.abs(action_next_diff))),
            "action_minus_current_state_mean_abs": float(np.mean(np.abs(action_state_diff))),
            "action_minus_current_state_max_abs": float(np.max(np.abs(action_state_diff))),
            "sidecar": None if sidecar is None else {"path": sidecar["path"], "keys": sidecar["keys"]},
        }
        summary["raw_sanity"] = raw_sanity
        report_lines.append("A. Raw parquet/video sanity")
        report_lines.append(json.dumps(raw_sanity, indent=2))
        report_lines.append("")
    except Exception as exc:
        summary["errors"].append(f"raw_sanity_failed: {type(exc).__name__}: {exc}")
        report_lines.append("A. Raw parquet/video sanity FAILED")
        report_lines.append(traceback.format_exc())
        report_lines.append("")

    loader = episode_df = shard_dataset = processor = None
    dataset_inspection: dict[str, Any] = {}
    row_summaries: list[RowAlignmentSummary] = []
    segment_rows: list[SegmentMetric] = []
    debug_npz: dict[str, Any] = {}

    try:
        loader = LeRobotEpisodeLoader(DATASET_PATH, modality_config)
        loader_stats = loader.get_dataset_statistics()
        processor = StateActionProcessor(
            modality_configs={embodiment_value: modality_config},
            statistics={embodiment_value: loader_stats},
            use_relative_action=True,
            clip_outliers=True,
        )
        shard_dataset = ShardedSingleStepDataset(
            DATASET_PATH,
            embodiment,
            modality_config,
            shard_size=1024,
            episode_sampling_rate=1.0,
            seed=42,
            allow_padding=False,
        )
        episode_df = loader[0]
        effective_len = shard_dataset.get_effective_episode_length(0)
        sampled_rows = sorted(int(x) for shard in shard_dataset.sharded_episodes for ep, inds in shard if ep == 0 for x in inds)
        dataset_inspection = {
            "loader_episode_count": len(loader),
            "loader_episode_length_0": loader.get_episode_length(0),
            "sharded_dataset_num_shards": len(shard_dataset),
            "sharded_dataset_total_sampled_steps": int(sum(shard_dataset.shard_lengths)),
            "effective_episode_length": int(effective_len),
            "sampled_row_min": int(min(sampled_rows)) if sampled_rows else None,
            "sampled_row_max": int(max(sampled_rows)) if sampled_rows else None,
            "sampled_contains_0_to_144_exactly": sampled_rows == list(range(145)),
            "action_horizon_from_dataset": int(shard_dataset.action_horizon),
            "video_keys": modality_config["video"].modality_keys,
            "state_keys": modality_config["state"].modality_keys,
            "action_keys": action_keys,
            "language_keys": modality_config["language"].modality_keys,
            "action_delta_indices": modality_config["action"].delta_indices,
            "state_delta_indices": modality_config["state"].delta_indices,
            "video_delta_indices": modality_config["video"].delta_indices,
            "action_configs": [
                {
                    "key": key,
                    "rep": str(cfg.rep),
                    "type": str(cfg.type),
                    "format": str(cfg.format),
                    "state_key": cfg.state_key,
                }
                for key, cfg in zip(action_keys, modality_config["action"].action_configs)
            ],
            "use_relative_action": True,
            "actions_normalized_by_StateActionProcessor": True,
            "stats_file": str(DATASET_PATH / "meta/stats.json"),
            "relative_stats_file": str(DATASET_PATH / "meta/relative_stats.json"),
            "episode_df_columns": list(episode_df.columns),
        }
        summary["dataset_inspection"] = dataset_inspection
        report_lines.append("B. GR00T actual dataloader inspection")
        report_lines.append(json.dumps(dataset_inspection, indent=2))
        report_lines.append("")

        rgb_checks = []
        for row in NON_BOUNDARY_ROWS + [144]:
            if row >= len(episode_df) or "video.ego_view" not in episode_df.columns:
                continue
            actual_rgb = ensure_rgb_uint8(episode_df["video.ego_view"].iloc[row])
            expected_rgb = decode_video_frame_ffmpeg(VIDEO_PATH, row)
            diff = np.abs(actual_rgb.astype(np.int16) - expected_rgb.astype(np.int16))
            rgb_checks.append(
                {
                    "row": row,
                    "actual_shape": list(actual_rgb.shape),
                    "expected_shape": list(expected_rgb.shape),
                    "mean_abs_diff": float(np.mean(diff)),
                    "max_abs_diff": int(np.max(diff)),
                }
            )
        summary["rgb_checks"] = rgb_checks
        report_lines.append("RGB row checks via actual LeRobotEpisodeLoader video.ego_view vs ffmpeg decoded MP4")
        report_lines.append(json.dumps(rgb_checks, indent=2))
        report_lines.append("")
    except Exception as exc:
        summary["errors"].append(f"dataset_inspection_failed: {type(exc).__name__}: {exc}")
        report_lines.append("B. GR00T actual dataloader inspection FAILED")
        report_lines.append(traceback.format_exc())
        report_lines.append("")

    if states is not None and actions is not None and episode_df is not None and processor is not None:
        report_lines.append("C/D/E. Sample-level semantic, segment, and boundary checks")
        sampled_by_dataset = set(range(dataset_inspection.get("effective_episode_length", 0)))
        for row in ROWS:
            row_summary = RowAlignmentSummary(
                row=row,
                sampled_by_sharded_dataset=row in sampled_by_dataset,
                raw_available=False,
                transformed_available=False,
                raw_best_offset=None,
                transformed_best_offset=None,
                raw_offset0_l2=None,
                transformed_offset0_l2=None,
                transformed_best_l2=None,
                boundary_behavior="not_checked",
            )
            try:
                vla = extract_step_data(episode_df, row, modality_config, embodiment, allow_padding=False)
                actual_raw = concat_action_dict(vla.actions, action_keys)
                actual_state = concat_action_dict(vla.states, modality_config["state"].modality_keys).reshape(-1)
                state_l2, _mse, state_max = metric(actual_state - states[row])
                row_summary.raw_available = True
                actual_transformed = transform_chunk(processor, embodiment_value, actual_state, actual_raw)
                row_summary.transformed_available = True

                raw_total_l2 = {}
                transformed_total_l2 = {}
                for offset in [-1, 0, 1]:
                    cand = get_candidate_chunk(actions, row, offset, ACTION_HORIZON)
                    if cand is None:
                        continue
                    l2, mse, max_abs = metric(actual_raw - cand)
                    raw_total_l2[offset] = l2
                    segment_rows.extend(segment_metrics(actual_raw, cand, row, offset, "raw_absolute"))
                    cand_transformed = transform_chunk(processor, embodiment_value, actual_state, cand)
                    tl2, tmse, tmax_abs = metric(actual_transformed - cand_transformed)
                    transformed_total_l2[offset] = tl2
                    segment_rows.extend(segment_metrics(actual_transformed, cand_transformed, row, offset, "processed_relative_normalized"))
                    if offset == 0:
                        debug_npz[f"row_{row:04d}_actual_raw"] = actual_raw
                        debug_npz[f"row_{row:04d}_expected_raw_offset0"] = cand
                        debug_npz[f"row_{row:04d}_actual_processed"] = actual_transformed
                        debug_npz[f"row_{row:04d}_expected_processed_offset0"] = cand_transformed

                if raw_total_l2:
                    row_summary.raw_best_offset = min(raw_total_l2, key=raw_total_l2.get)
                    row_summary.raw_offset0_l2 = raw_total_l2.get(0)
                if transformed_total_l2:
                    row_summary.transformed_best_offset = min(transformed_total_l2, key=transformed_total_l2.get)
                    row_summary.transformed_offset0_l2 = transformed_total_l2.get(0)
                    row_summary.transformed_best_l2 = transformed_total_l2[row_summary.transformed_best_offset]
                row_summary.boundary_behavior = "full_chunk_available_no_padding"
                report_lines.append(
                    f"[ROW {row}] sampled={row_summary.sampled_by_sharded_dataset} "
                    f"state_l2={state_l2:.6g} state_max={state_max:.6g} "
                    f"raw_l2_by_offset={raw_total_l2} transformed_l2_by_offset={transformed_total_l2}"
                )
            except Exception as exc:
                row_summary.boundary_behavior = f"extract_no_padding_failed: {type(exc).__name__}: {exc}"
                try:
                    vla_pad = extract_step_data(episode_df, row, modality_config, embodiment, allow_padding=True)
                    padded_raw = concat_action_dict(vla_pad.actions, action_keys)
                    last_action = actions[-1]
                    zero_chunk = np.zeros_like(padded_raw)
                    repeat_last_chunk = np.repeat(last_action[None, :], ACTION_HORIZON, axis=0)
                    row_summary.boundary_behavior += (
                        f"; allow_padding=True works; padded_chunk_first_last_l2="
                        f"{metric(padded_raw[-1] - last_action)[0]:.6g}; "
                        f"vs_all_zero_l2={metric(padded_raw - zero_chunk)[0]:.6g}; "
                        f"vs_repeat_last_l2={metric(padded_raw - repeat_last_chunk)[0]:.6g}"
                    )
                except Exception as pad_exc:
                    row_summary.boundary_behavior += f"; allow_padding_failed: {type(pad_exc).__name__}: {pad_exc}"
                report_lines.append(f"[ROW {row}] sampled={row_summary.sampled_by_sharded_dataset} {row_summary.boundary_behavior}")
            row_summaries.append(row_summary)
        report_lines.append("")

    # F. FPS/timestamp audit
    fps_result = "FAIL"
    fps_details: dict[str, Any] = {}
    if states is not None:
        try:
            info = json.loads((DATASET_PATH / "meta/info.json").read_text())
            video_props = probe_video(VIDEO_PATH)
            timestamp_step = None if timestamps is None or len(timestamps) < 2 else np.diff(timestamps)
            fps_details = {
                "meta_info_fps": info.get("fps"),
                "video_r_frame_rate": video_props.get("r_frame_rate"),
                "video_avg_frame_rate": video_props.get("avg_frame_rate"),
                "timestamp_step_mean": None if timestamp_step is None else float(np.mean(timestamp_step)),
                "timestamp_step_max_abs_from_0p1": None if timestamp_step is None else float(np.max(np.abs(timestamp_step - 0.1))),
                "dataloader_delta_indices_are_fixed_frame_offsets": True,
                "actual_action_chunk_indices": "row + delta_indices, delta_indices=list(range(16))",
                "interpretation": "GR00T LeRobotEpisodeLoader/extract_step_data uses integer row offsets, not time-based sampling, for state/action/video delta_indices.",
            }
            fps_result = "PASS"
            if info.get("fps") != 10 or video_props.get("nb_read_frames") != 160:
                fps_result = "WARN"
            if timestamp_step is not None and float(np.max(np.abs(timestamp_step - 0.1))) > 1e-6:
                fps_result = "WARN"
        except Exception as exc:
            fps_details = {"error": f"{type(exc).__name__}: {exc}"}
            fps_result = "FAIL"
    summary["fps_timestamp"] = fps_details
    report_lines.append("F. FPS / timestamp audit")
    report_lines.append(json.dumps(fps_details, indent=2))
    report_lines.append("")

    # G. stats audit
    stats_result = "FAIL"
    stats_report_lines = ["Stats and relative_stats audit", "", f"dataset = {DATASET_PATH}", ""]
    stats_summary: dict[str, Any] = {}
    try:
        current_stats = json.loads((DATASET_PATH / "meta/stats.json").read_text())
        current_rel_stats = json.loads((DATASET_PATH / "meta/relative_stats.json").read_text())
        current_rel_stats.pop("__fingerprints__", None)
        recomputed_stats = recompute_window_stats(states, actions)
        recomputed_rel_stats = recompute_relative_stats(processor, modality_config, states, actions)
        json_dump(WINDOW_STATS_PREVIEW_PATH, recomputed_stats)
        json_dump(WINDOW_REL_STATS_PREVIEW_PATH, recomputed_rel_stats)

        stat_comparisons = []
        for key in ["observation.state", "action"]:
            stat_comparisons.append(compare_stat_group(current_stats.get(key, {}), recomputed_stats[key], key))
        rel_comparisons = []
        for key in KEY_SLICES:
            rel_comparisons.append(compare_stat_group(current_rel_stats.get(key, {}), recomputed_rel_stats[key], f"relative_action.{key}"))
        stats_summary = {
            "stats_preview": str(WINDOW_STATS_PREVIEW_PATH),
            "relative_stats_preview": str(WINDOW_REL_STATS_PREVIEW_PATH),
            "stats_comparisons": stat_comparisons,
            "relative_stats_comparisons": rel_comparisons,
        }
        dim_ok = all(item["dimension_match"] for item in stat_comparisons + rel_comparisons)
        stat_large = any(item["max_mean_abs_diff"] > STATS_WARN_MEAN_ABS_DIFF for item in stat_comparisons)
        rel_large = any(item["max_mean_abs_diff"] > REL_STATS_WARN_MEAN_ABS_DIFF for item in rel_comparisons)
        if not dim_ok:
            stats_result = "FAIL"
        elif stat_large or rel_large:
            stats_result = "WARN"
        else:
            stats_result = "PASS"
        stats_report_lines.append(json.dumps(stats_summary, indent=2))
    except Exception as exc:
        stats_summary = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        stats_report_lines.append(json.dumps(stats_summary, indent=2))
        stats_result = "FAIL"
    summary["stats_audit"] = stats_summary
    STATS_REPORT_PATH.write_text("\n".join(stats_report_lines) + "\n")
    report_lines.append("G. Stats and relative_stats audit")
    report_lines.append(f"details = {STATS_REPORT_PATH}")
    report_lines.append(json.dumps({"result": stats_result}, indent=2))
    report_lines.append("")

    # PASS/WARN/FAIL aggregation
    action_pass = True
    action_warn = False
    action_reasons = []
    row_summary_dicts = [asdict(item) for item in row_summaries]
    for item in row_summaries:
        if item.row in NON_BOUNDARY_ROWS:
            if item.transformed_best_offset != 0 or item.transformed_offset0_l2 is None or item.transformed_offset0_l2 > NUM_TOL:
                action_pass = False
                action_reasons.append(f"row {item.row}: transformed best offset={item.transformed_best_offset}, offset0_l2={item.transformed_offset0_l2}")
    for item in row_summaries:
        if item.row in [145, 150, 159] and item.sampled_by_sharded_dataset:
            action_warn = True
            action_reasons.append(f"boundary row {item.row} unexpectedly sampled")
    action_result = "PASS" if action_pass else "FAIL"
    if action_result == "PASS" and action_warn:
        action_result = "WARN"

    rgb_result = "FAIL"
    rgb_checks = summary.get("rgb_checks", [])
    if rgb_checks:
        rgb_result = "PASS" if all(item["mean_abs_diff"] < 5.0 for item in rgb_checks) else "FAIL"

    if action_result == "FAIL" or rgb_result == "FAIL" or fps_result == "FAIL" or stats_result == "FAIL":
        overall = "FAIL"
    elif action_result == "WARN" or fps_result == "WARN" or stats_result == "WARN":
        overall = "WARN"
    else:
        overall = "PASS"

    summary.update(
        {
            "row_alignment": row_summary_dicts,
            "action_reasons": action_reasons,
            "segment_metrics_count": len(segment_rows),
            "results": {
                "ACTION_CHUNK_ALIGNMENT_RESULT": action_result,
                "RGB_ROW_ALIGNMENT_RESULT": rgb_result,
                "FPS_TIMESTAMP_RESULT": fps_result,
                "STATS_RELATIVE_STATS_RESULT": stats_result,
                "OVERALL_TRAINING_BATCH_AUDIT_RESULT": overall,
            },
        }
    )

    # CSV + debug npz
    with ROWS_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(segment_rows[0]).keys()) if segment_rows else ["empty"])
        writer.writeheader()
        for item in segment_rows:
            writer.writerow(asdict(item))
    if debug_npz:
        np.savez_compressed(DEBUG_NPZ_PATH, **debug_npz)

    report_lines.append("Final per-row alignment summary")
    report_lines.append(json.dumps(row_summary_dicts, indent=2))
    report_lines.append("")
    report_lines.append("Action alignment reasons")
    report_lines.extend([f"- {reason}" for reason in action_reasons] or ["- none"])
    report_lines.append("")
    report_lines.append(f"ACTION_CHUNK_ALIGNMENT_RESULT = {action_result}")
    report_lines.append(f"RGB_ROW_ALIGNMENT_RESULT = {rgb_result}")
    report_lines.append(f"FPS_TIMESTAMP_RESULT = {fps_result}")
    report_lines.append(f"STATS_RELATIVE_STATS_RESULT = {stats_result}")
    report_lines.append(f"OVERALL_TRAINING_BATCH_AUDIT_RESULT = {overall}")
    REPORT_PATH.write_text("\n".join(report_lines) + "\n")
    json_dump(SUMMARY_PATH, summary)

    print(f"[INFO] Saved report to: {REPORT_PATH}")
    print(f"[INFO] Saved summary to: {SUMMARY_PATH}")
    print(f"[INFO] Saved per-row CSV to: {ROWS_CSV_PATH}")
    print(f"[INFO] Saved stats report to: {STATS_REPORT_PATH}")
    print(f"ACTION_CHUNK_ALIGNMENT_RESULT = {action_result}")
    print(f"RGB_ROW_ALIGNMENT_RESULT = {rgb_result}")
    print(f"FPS_TIMESTAMP_RESULT = {fps_result}")
    print(f"STATS_RELATIVE_STATS_RESULT = {stats_result}")
    print(f"OVERALL_TRAINING_BATCH_AUDIT_RESULT = {overall}")


if __name__ == "__main__":
    main()
