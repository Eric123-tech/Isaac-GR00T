#!/usr/bin/env python3
"""Audit fccd8 rendered-RGB LeRobot semantics for GR00T finetuning.

Task 6 from gr00t_artimano_isaac_render_rgb_plan_v2.md.
Run from the GR00T repository with:

    uv run python scripts/ARTIMANO/audit_fccd8_render_rgb_gr00t_input_semantics.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path("/mnt/data/haoyu_data/oakink2/gr00t_lerobot/fccd8_artimano_torch_wrist_qpos_isaac_render_rgb_row68_window")
SOURCE_DATASET_ROOT = Path("/mnt/data/haoyu_data/oakink2/gr00t_lerobot/fccd8_artimano_torch_wrist_qpos")
RENDERED_FRAME_DIR = Path("/mnt/data/haoyu_data/isaaclab_assets/oakink2_fccd8/rendered_rgb_fixed_row68_window_frames")
MODALITY_CONFIG_PATH = REPO_ROOT / "examples/ARTIMANO/oakink2_artimano_bimanual_config.py"
EMBODIMENT_TAG_NAME = "NEW_EMBODIMENT"
OUT_ROOT = Path("/mnt/data/haoyu_data/isaaclab_assets/oakink2_fccd8")
REPORT_PATH = OUT_ROOT / "audit_render_rgb_gr00t_input_semantics_report.txt"
SUMMARY_JSON = OUT_ROOT / "audit_render_rgb_gr00t_input_semantics_summary.json"
CONTACT_SHEET_PATH = OUT_ROOT / "audit_render_rgb_gr00t_input_contact_sheet.png"
RGB_CHECK_ROW000 = OUT_ROOT / "audit_rgb_channel_check_row000.png"
RGB_CHECK_ROW068 = OUT_ROOT / "audit_rgb_channel_check_row068.png"
DATALOADER_IMAGE_ROW000 = OUT_ROOT / "audit_gr00t_dataloader_image_row000.png"
DATALOADER_IMAGE_ROW068 = OUT_ROOT / "audit_gr00t_dataloader_image_row068.png"
SIDE_CAR = DATASET_ROOT / "retarget/isaac_render_rgb_row68_window_state_action.npz"
TASK_TEXT = "Cap the bottle."
START_ROW = 68
NUM_ROWS = 160
FPS = 10
VIDEO_COMPARE_ROWS = [0, 1, 20, 40, 68, 100, 140, 159]
SEGMENTS = {
    "left_wrist_pose": slice(0, 9),
    "right_wrist_pose": slice(9, 18),
    "left_hand_qpos": slice(18, 40),
    "right_hand_qpos": slice(40, 62),
}
REQUIRED_COLUMNS = [
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "annotation.human.action.task_description",
]


class Audit:
    def __init__(self):
        self.checks: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {
            "dataset": str(DATASET_ROOT),
            "source_dataset": str(SOURCE_DATASET_ROOT),
            "rendered_frame_dir": str(RENDERED_FRAME_DIR),
            "modality_config_path": str(MODALITY_CONFIG_PATH),
            "embodiment_tag": EMBODIMENT_TAG_NAME,
            "checks": self.checks,
            "warnings": [],
        }
        self.lines: list[str] = []

    def check(self, name: str, passed: bool, details: str = ""):
        self.checks.append({"name": name, "passed": bool(passed), "details": details})
        prefix = "[PASS]" if passed else "[FAIL]"
        self.lines.append(f"{prefix} {name}: {details}")

    def warn(self, message: str):
        self.summary["warnings"].append(message)
        self.lines.append(f"[WARN] {message}")

    def section(self, title: str):
        self.lines.extend(["", f"## {title}"])

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "name") and hasattr(value, "value"):
        return f"{value.__class__.__name__}.{value.name}({value.value})"
    return value


def list_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [str(path.relative_to(root)) for path in sorted(root.rglob("*")) if path.is_file()]


def table_to_vector(table, column: str) -> np.ndarray:
    if column in table.column_names:
        values = table.column(column).to_pylist()
        arr = np.asarray(values, dtype=np.float32)
        return arr.reshape(arr.shape[0], -1)
    prefix = f"{column}."
    cols = [name for name in table.column_names if name.startswith(prefix)]
    if not cols:
        raise KeyError(f"Missing vector column {column}")
    cols = sorted(cols, key=lambda name: int(name[len(prefix):]))
    return np.stack([np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=np.float32) for name in cols], axis=1)


def read_scalar_column(table, column: str):
    if column not in table.column_names:
        return None
    return table.column(column).to_pylist()


def load_tasks(path: Path) -> dict[int, str]:
    tasks = {}
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            tasks[int(rec["task_index"])] = str(rec.get("task", rec.get("task_description", "")))
    return tasks


def stats_for(name: str, arr: np.ndarray) -> dict[str, Any]:
    arr64 = arr.astype(np.float64)
    return {
        "name": name,
        "shape": list(arr.shape),
        "min": float(np.min(arr64)),
        "max": float(np.max(arr64)),
        "mean": float(np.mean(arr64)),
        "std": float(np.std(arr64)),
    }


def segment_l2_errors(a: np.ndarray, b: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for name, sl in SEGMENTS.items():
        diff = a[:, sl].astype(np.float64) - b[:, sl].astype(np.float64)
        per_row = np.linalg.norm(diff, axis=1)
        out[name] = {"mean_l2": float(np.mean(per_row)), "max_l2": float(np.max(per_row))}
    return out


def decode_video_frame(video_path: Path, row: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read frame {row} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def load_png_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def to_rgb_array(image: Any) -> np.ndarray:
    if hasattr(image, "convert"):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr.copy()


def image_diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"mean_abs": float(np.mean(diff)), "max_abs": float(np.max(diff))}


def draw_labeled_tile(img: np.ndarray, lines: list[str], width: int = 256, label_h: int = 42) -> Image.Image:
    pil = Image.fromarray(img).convert("RGB")
    pil.thumbnail((width, width), Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (width, width + label_h), (255, 255, 255))
    tile.paste(pil, ((width - pil.width) // 2, label_h))
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    y = 2
    for line in lines:
        draw.text((4, y), line, fill=(0, 0, 0), font=font)
        y += 13
    return tile


def save_contact_sheet(rows: list[int], video_path: Path, frame_indices: np.ndarray, source_rows: np.ndarray | None):
    tiles = []
    for row in rows:
        img = decode_video_frame(video_path, row)
        original_row = int(source_rows[row]) if source_rows is not None and row < len(source_rows) else START_ROW + row
        labels = [f"local row {row}", f"orig row {original_row}", f"frame_index {int(frame_indices[row])}"]
        tiles.append(draw_labeled_tile(img, labels))
    cols = 4
    rows_count = math.ceil(len(tiles) / cols)
    w, h = tiles[0].size
    sheet = Image.new("RGB", (cols * w, rows_count * h), (230, 230, 230))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * w, (idx // cols) * h))
    sheet.save(CONTACT_SHEET_PATH)


def save_rgb_channel_check(row: int, video_path: Path, out_path: Path) -> dict[str, float]:
    png = load_png_rgb(RENDERED_FRAME_DIR / f"frame_{row:06d}.png")
    video_rgb = decode_video_frame(video_path, row)
    diff = image_diff(video_rgb, png)
    gap = 12
    label_h = 32
    left = draw_labeled_tile(png, [f"source PNG row {row}", "RGB"], width=256, label_h=label_h)
    right = draw_labeled_tile(video_rgb, [f"video decoded row {row}", "cv2 BGR->RGB"], width=256, label_h=label_h)
    canvas = Image.new("RGB", (left.width + gap + right.width, max(left.height, right.height)), (250, 250, 250))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    canvas.save(out_path)
    return diff


def import_modality_config():
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(MODALITY_CONFIG_PATH.parent))
    spec = importlib.util.spec_from_file_location("oakink2_artimano_bimanual_config", MODALITY_CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag

    tag = EmbodimentTag[EMBODIMENT_TAG_NAME]
    return MODALITY_CONFIGS[tag.value], tag


def modality_action_summary(modality_configs) -> list[dict[str, str]]:
    action_cfg = modality_configs["action"]
    rows = []
    for key, cfg in zip(action_cfg.modality_keys, action_cfg.action_configs):
        rows.append(
            {
                "key": key,
                "rep": getattr(cfg.rep, "name", str(cfg.rep)),
                "type": getattr(cfg.type, "name", str(cfg.type)),
                "format": getattr(cfg.format, "name", str(cfg.format)),
                "state_key": str(cfg.state_key),
            }
        )
    return rows


def check_stats_shapes(stats: dict[str, Any], audit: Audit) -> dict[str, Any]:
    result = {}
    for top_key, expected_dim in [("observation.state", 62), ("action", 62)]:
        entry = stats.get(top_key)
        ok = isinstance(entry, dict)
        shapes = {}
        if ok:
            for stat_key in ["mean", "std", "min", "max"]:
                values = entry.get(stat_key)
                shapes[stat_key] = len(values) if hasattr(values, "__len__") else None
                ok = ok and shapes[stat_key] == expected_dim
        audit.check(f"stats {top_key} dimensions", ok, json.dumps(shapes))
        result[top_key] = shapes
    return result


def inspect_actual_gr00t_loader(audit: Audit, modality_configs, embodiment_tag):
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset, extract_step_data

    loader = LeRobotEpisodeLoader(dataset_path=DATASET_ROOT, modality_configs=modality_configs)
    episode_df = loader[0]
    audit.check("LeRobotEpisodeLoader constructs episode", len(episode_df) == NUM_ROWS, f"len={len(episode_df)}")

    samples = {}
    for row in [0, 68]:
        vla = extract_step_data(episode_df, row, modality_configs, embodiment_tag)
        img = to_rgb_array(vla.images["ego_view"][0])
        if row == 0:
            Image.fromarray(img).save(DATALOADER_IMAGE_ROW000)
        elif row == 68:
            Image.fromarray(img).save(DATALOADER_IMAGE_ROW068)
        state_shapes = {k: list(v.shape) for k, v in vla.states.items()}
        action_shapes = {k: list(v.shape) for k, v in vla.actions.items()}
        raw_png = load_png_rgb(RENDERED_FRAME_DIR / f"frame_{row:06d}.png")
        diff = image_diff(img, raw_png)
        samples[str(row)] = {
            "available_episode_columns": list(episode_df.columns),
            "image_shape": list(img.shape),
            "image_dtype": str(img.dtype),
            "image_min": int(img.min()),
            "image_max": int(img.max()),
            "image_mean": float(np.mean(img)),
            "image_vs_source_png": diff,
            "state_shapes": state_shapes,
            "action_shapes": action_shapes,
            "language_text": vla.text,
            "embodiment": getattr(vla.embodiment, "name", str(vla.embodiment)),
        }
        audit.check(
            f"GR00T loader row {row} sees rendered RGB",
            diff["mean_abs"] < 8.0,
            f"mean_abs={diff['mean_abs']:.4f}, max_abs={diff['max_abs']:.1f}",
        )
        audit.check(f"GR00T loader row {row} language", vla.text == TASK_TEXT, repr(vla.text))
        audit.check(
            f"GR00T loader row {row} state/action shapes",
            state_shapes == {
                "left_wrist_pose": [1, 9],
                "right_wrist_pose": [1, 9],
                "left_hand_qpos": [1, 22],
                "right_hand_qpos": [1, 22],
            }
            and action_shapes == {
                "left_wrist_pose": [16, 9],
                "right_wrist_pose": [16, 9],
                "left_hand_qpos": [16, 22],
                "right_hand_qpos": [16, 22],
            },
            f"state={state_shapes}, action={action_shapes}",
        )

    sharded = ShardedSingleStepDataset(
        dataset_path=DATASET_ROOT,
        embodiment_tag=embodiment_tag,
        modality_configs=modality_configs,
        shard_size=1024,
        episode_sampling_rate=0.1,
        seed=42,
        allow_padding=False,
    )
    audit.check("ShardedSingleStepDataset constructs", len(sharded) > 0, f"num_shards={len(sharded)}, shard_lengths={sharded.shard_lengths.tolist()}")
    return samples


def main():
    audit = Audit()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    audit.lines.append("# fccd8 rendered-RGB GR00T input semantics audit")
    audit.lines.append(f"dataset = {DATASET_ROOT}")
    audit.lines.append(f"source_dataset = {SOURCE_DATASET_ROOT}")
    audit.lines.append(f"rendered_frame_dir = {RENDERED_FRAME_DIR}")
    audit.lines.append(f"modality_config_path = {MODALITY_CONFIG_PATH}")
    audit.lines.append(f"embodiment_tag = {EMBODIMENT_TAG_NAME}")

    audit.section("Dataset structure")
    required_files = [
        DATASET_ROOT / "data/chunk-000/episode_000000.parquet",
        DATASET_ROOT / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
        DATASET_ROOT / "meta/tasks.jsonl",
        DATASET_ROOT / "meta/episodes.jsonl",
        DATASET_ROOT / "meta/info.json",
        DATASET_ROOT / "meta/stats.json",
        DATASET_ROOT / "meta/relative_stats.json",
    ]
    for path in required_files:
        audit.check(f"exists {path.relative_to(DATASET_ROOT)}", path.exists(), str(path))
    discovered = {
        "meta": list_files(DATASET_ROOT / "meta"),
        "data/chunk-000": list_files(DATASET_ROOT / "data/chunk-000"),
        "videos/chunk-000/observation.images.ego_view": list_files(DATASET_ROOT / "videos/chunk-000/observation.images.ego_view"),
    }
    audit.summary["discovered_files"] = discovered
    audit.lines.append(json.dumps(discovered, indent=2))

    audit.section("Parquet schema")
    parquet_path = DATASET_ROOT / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(parquet_path)
    df = pd.read_parquet(parquet_path)
    audit.summary["parquet_columns"] = table.column_names
    audit.lines.append("columns = " + json.dumps(table.column_names))
    for column in REQUIRED_COLUMNS:
        audit.check(f"parquet column {column}", column in table.column_names, "")
    audit.check("row count = 160", table.num_rows == NUM_ROWS, f"row_count={table.num_rows}")
    states = table_to_vector(table, "observation.state")
    actions = table_to_vector(table, "action")
    frame_index = np.asarray(df["frame_index"], dtype=np.int64) if "frame_index" in df else np.asarray([], dtype=np.int64)
    index_col = np.asarray(df["index"], dtype=np.int64) if "index" in df else np.asarray([], dtype=np.int64)
    episode_index = np.asarray(df["episode_index"], dtype=np.int64) if "episode_index" in df else np.asarray([], dtype=np.int64)
    timestamp = np.asarray(df["timestamp"], dtype=np.float64) if "timestamp" in df else np.asarray([], dtype=np.float64)
    task_index = np.asarray(df["task_index"], dtype=np.int64) if "task_index" in df else None
    audit.check("frame_index is 0..159", np.array_equal(frame_index, np.arange(NUM_ROWS)), f"first={frame_index[:3].tolist()}, last={frame_index[-3:].tolist()}")
    audit.check("index is 0..159", np.array_equal(index_col, np.arange(NUM_ROWS)), f"first={index_col[:3].tolist()}, last={index_col[-3:].tolist()}")
    audit.check("episode_index all 0", episode_index.size == NUM_ROWS and np.all(episode_index == 0), f"unique={np.unique(episode_index).tolist() if episode_index.size else []}")
    expected_ts = np.arange(NUM_ROWS, dtype=np.float64) / FPS
    audit.check("timestamp monotonic frame_index/10", timestamp.size == NUM_ROWS and np.allclose(timestamp, expected_ts, atol=1e-8), f"first={timestamp[:3].tolist() if timestamp.size else []}")

    audit.section("State/action semantics")
    audit.check("observation.state shape", states.shape == (NUM_ROWS, 62), str(states.shape))
    audit.check("action shape", actions.shape == (NUM_ROWS, 62), str(actions.shape))
    seg_stats = {name: {"state": stats_for(name, states[:, sl]), "action": stats_for(name, actions[:, sl])} for name, sl in SEGMENTS.items()}
    audit.summary["segment_stats"] = seg_stats
    audit.lines.append(json.dumps(seg_stats, indent=2))
    next_diff = actions[:-1] - states[1:]
    next_l2 = np.linalg.norm(next_diff.astype(np.float64), axis=1)
    next_summary = {"mean_l2": float(np.mean(next_l2)), "max_l2": float(np.max(next_l2)), "segments": segment_l2_errors(actions[:-1], states[1:])}
    audit.summary["next_state_action_check"] = next_summary
    audit.check("action[i] == state[i+1] rows 0..158", next_summary["max_l2"] < 1e-7, json.dumps(next_summary))

    source_table = pq.read_table(SOURCE_DATASET_ROOT / "data/chunk-000/episode_000000.parquet")
    source_states = table_to_vector(source_table, "observation.state")
    source_actions = table_to_vector(source_table, "action")
    audit.check("output row0 == original row68 state", np.linalg.norm(states[0] - source_states[START_ROW]) < 1e-7, str(float(np.linalg.norm(states[0] - source_states[START_ROW]))))
    audit.check("output row159 == original row227 state", np.linalg.norm(states[-1] - source_states[START_ROW + NUM_ROWS - 1]) < 1e-7, str(float(np.linalg.norm(states[-1] - source_states[START_ROW + NUM_ROWS - 1]))))
    audit.check("output action0 == original action68", np.linalg.norm(actions[0] - source_actions[START_ROW]) < 1e-7, str(float(np.linalg.norm(actions[0] - source_actions[START_ROW]))))

    audit.section("Rendered video/frame alignment")
    video_path = DATASET_ROOT / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
    cap = cv2.VideoCapture(str(video_path))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    video_props = {"frame_count": video_frame_count, "width": video_width, "height": video_height, "fps": video_fps}
    audit.summary["video_properties"] = video_props
    audit.check("video frame count = 160", video_frame_count == NUM_ROWS, json.dumps(video_props))
    audit.check("video resolution 256x256", video_width == 256 and video_height == 256, json.dumps(video_props))
    audit.check("video fps = 10", abs(video_fps - FPS) < 1e-3, json.dumps(video_props))
    frame_diffs = {}
    for row in VIDEO_COMPARE_ROWS:
        png = load_png_rgb(RENDERED_FRAME_DIR / f"frame_{row:06d}.png")
        vid = decode_video_frame(video_path, row)
        frame_diffs[str(row)] = image_diff(vid, png)
    audit.summary["video_vs_png_diffs"] = frame_diffs
    audit.lines.append(json.dumps(frame_diffs, indent=2))
    max_mean_diff = max(item["mean_abs"] for item in frame_diffs.values())
    audit.check("video frames match rendered PNGs within compression tolerance", max_mean_diff < 8.0, f"max_mean_abs={max_mean_diff:.4f}")
    source_rows = None
    if SIDE_CAR.exists():
        sidecar = np.load(SIDE_CAR, allow_pickle=True)
        source_rows = np.asarray(sidecar["source_rows"], dtype=np.int64) if "source_rows" in sidecar else None
    save_contact_sheet(VIDEO_COMPARE_ROWS, video_path, frame_index, source_rows)
    audit.check("saved contact sheet", CONTACT_SHEET_PATH.exists(), str(CONTACT_SHEET_PATH))

    audit.section("RGB channel/order")
    rgb000 = save_rgb_channel_check(0, video_path, RGB_CHECK_ROW000)
    rgb068 = save_rgb_channel_check(68, video_path, RGB_CHECK_ROW068)
    audit.summary["rgb_channel_checks"] = {"0": rgb000, "68": rgb068}
    audit.check("RGB channel row000 plausible", rgb000["mean_abs"] < 8.0, json.dumps(rgb000))
    audit.check("RGB channel row068 plausible", rgb068["mean_abs"] < 8.0, json.dumps(rgb068))

    audit.section("Task language / task index")
    tasks = load_tasks(DATASET_ROOT / "meta/tasks.jsonl")
    audit.summary["tasks"] = tasks
    audit.check("task index 0 maps to Cap the bottle.", tasks.get(0) == TASK_TEXT, json.dumps(tasks))
    if task_index is not None:
        audit.check("every row task_index = 0", np.all(task_index == 0), f"unique={np.unique(task_index).tolist()}")
    ann_col = read_scalar_column(table, "annotation.human.action.task_description")
    if ann_col is not None:
        if all(isinstance(x, (int, np.integer)) for x in ann_col):
            ok = all(int(x) == 0 for x in ann_col)
            detail = "numeric task index column resolves via tasks.jsonl"
        else:
            ok = all(str(x) == TASK_TEXT for x in ann_col)
            detail = "string task text column"
        audit.check("annotation.human.action.task_description resolves correctly", ok, detail)
    audit.lines.append(f'GR00T task_description = "{TASK_TEXT}"')

    audit.section("Modality config")
    modality_configs, embodiment_tag = import_modality_config()
    audit.summary["modality_config"] = to_jsonable(modality_configs)
    expected_video = modality_configs["video"].modality_keys == ["ego_view"]
    expected_state = modality_configs["state"].modality_keys == ["left_wrist_pose", "right_wrist_pose", "left_hand_qpos", "right_hand_qpos"]
    expected_action = modality_configs["action"].modality_keys == ["left_wrist_pose", "right_wrist_pose", "left_hand_qpos", "right_hand_qpos"]
    audit.check("modality video key observation.images.ego_view", expected_video, str(modality_configs["video"].modality_keys))
    audit.check("modality state keys match dataset", expected_state, str(modality_configs["state"].modality_keys))
    audit.check("modality action keys match dataset", expected_action, str(modality_configs["action"].modality_keys))
    audit.check("embodiment tag NEW_EMBODIMENT", embodiment_tag.name == EMBODIMENT_TAG_NAME, str(embodiment_tag))
    action_summary = modality_action_summary(modality_configs)
    audit.summary["action_config_summary"] = action_summary
    audit.lines.append(json.dumps(action_summary, indent=2))
    action_ok = action_summary == [
        {"key": "left_wrist_pose", "rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROT6D", "state_key": "left_wrist_pose"},
        {"key": "right_wrist_pose", "rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROT6D", "state_key": "right_wrist_pose"},
        {"key": "left_hand_qpos", "rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "left_hand_qpos"},
        {"key": "right_hand_qpos", "rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "right_hand_qpos"},
    ]
    audit.check("action representation semantics", action_ok, "raw action absolute next-state; GR00T processor uses relative internally; server returns absolute postprocessed action")

    audit.section("Normalization/statistics")
    stats = json.loads((DATASET_ROOT / "meta/stats.json").read_text())
    rel_stats = json.loads((DATASET_ROOT / "meta/relative_stats.json").read_text())
    stats_shape_summary = check_stats_shapes(stats, audit)
    audit.summary["stats_shape_summary"] = stats_shape_summary
    rel_keys = [k for k in rel_stats.keys() if k != "__fingerprints__"]
    audit.summary["relative_stats_keys"] = rel_keys
    expected_rel_keys = ["left_wrist_pose", "right_wrist_pose", "left_hand_qpos", "right_hand_qpos"]
    rel_ok = sorted(rel_keys) == sorted(expected_rel_keys)
    audit.check("relative_stats keys present", rel_ok, str(rel_keys))
    rel_dims_ok = True
    rel_dims = {}
    for key in expected_rel_keys:
        entry = rel_stats.get(key, {})
        rel_dims[key] = {stat_key: (len(entry.get(stat_key, [[]])[0]) if entry.get(stat_key) and isinstance(entry.get(stat_key), list) and isinstance(entry.get(stat_key)[0], list) else None) for stat_key in ["mean", "std", "min", "max"]}
        expected = 9 if "wrist" in key else 22
        rel_dims_ok = rel_dims_ok and all(dim == expected for dim in rel_dims[key].values())
    audit.summary["relative_stats_dims"] = rel_dims
    audit.check("relative_stats dimensional consistency", rel_dims_ok, json.dumps(rel_dims))
    source_stats = json.loads((SOURCE_DATASET_ROOT / "meta/stats.json").read_text())
    copied_stats = stats.get("__fingerprints__") == source_stats.get("__fingerprints__") and stats.get("observation.state", {}).get("mean") == source_stats.get("observation.state", {}).get("mean")
    if copied_stats:
        audit.warn("stats appear to be copied from the original full dataset. This may be acceptable for compatibility, but for row-window overfitting, recomputed window stats may improve normalization consistency.")
        stats_origin = "copied_from_original_full_dataset"
    else:
        stats_origin = "recomputed_or_modified_for_row_window"
    audit.summary["stats_origin"] = stats_origin

    audit.section("Actual GR00T dataloader sample")
    try:
        loader_samples = inspect_actual_gr00t_loader(audit, modality_configs, embodiment_tag)
        audit.summary["gr00t_loader_samples"] = loader_samples
        audit.lines.append(json.dumps(loader_samples, indent=2))
        dataloader_ok = True
    except Exception as exc:
        dataloader_ok = False
        audit.summary["gr00t_loader_error"] = f"{type(exc).__name__}: {exc}"
        audit.check("GR00T dataloader sample can be constructed", False, f"{type(exc).__name__}: {exc}")
    audit.check("dataloader answers rendered RGB/task/state/action questions", dataloader_ok, "see gr00t_loader_samples")

    pass_criteria = {
        "dataset_has_160_rows": table.num_rows == NUM_ROWS,
        "video_has_160_frames": video_frame_count == NUM_ROWS,
        "video_matches_pngs": max_mean_diff < 8.0,
        "frame_index_0_159": np.array_equal(frame_index, np.arange(NUM_ROWS)),
        "local_row0_maps_original_row68": np.linalg.norm(states[0] - source_states[START_ROW]) < 1e-7,
        "state_action_62d": states.shape == (NUM_ROWS, 62) and actions.shape == (NUM_ROWS, 62),
        "action_equals_next_state": next_summary["max_l2"] < 1e-7,
        "task_text_cap_the_bottle": tasks.get(0) == TASK_TEXT,
        "modality_config_matches": expected_video and expected_state and expected_action and action_ok,
        "gr00t_dataloader_sample_ok": dataloader_ok,
    }
    audit.summary["pass_criteria"] = pass_criteria
    final_pass = all(pass_criteria.values()) and audit.passed
    audit.summary["AUDIT_RESULT"] = "PASS" if final_pass else "FAIL"
    audit.lines.extend(["", f"AUDIT_RESULT = {audit.summary['AUDIT_RESULT']}", "", "Pass criteria:", json.dumps(to_jsonable(pass_criteria), indent=2)])

    REPORT_PATH.write_text("\n".join(audit.lines) + "\n")
    SUMMARY_JSON.write_text(json.dumps(to_jsonable(audit.summary), indent=2) + "\n")
    print(f"[INFO] Saved report to: {REPORT_PATH}")
    print(f"[INFO] Saved summary to: {SUMMARY_JSON}")
    print(f"[INFO] Saved contact sheet to: {CONTACT_SHEET_PATH}")
    print(f"[INFO] Saved RGB channel checks to: {RGB_CHECK_ROW000}, {RGB_CHECK_ROW068}")
    print(f"[INFO] Saved dataloader images to: {DATALOADER_IMAGE_ROW000}, {DATALOADER_IMAGE_ROW068}")
    print(f"AUDIT_RESULT = {audit.summary['AUDIT_RESULT']}")
    if not final_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
