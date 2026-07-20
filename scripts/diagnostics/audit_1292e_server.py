#!/usr/bin/env python
"""Read-only server-side audit for the 1292e GR00T dataset/checkpoint pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

from transformers import AutoProcessor

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader


ACTION_STATE_KEYS = [
    "left_wrist_pose",
    "right_wrist_pose",
    "left_hand_qpos",
    "right_hand_qpos",
]
SPECIAL_OUTPUTS = {"bundle_manifest.sha256"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def numeric_leaf_summary(arr_like: Any) -> dict[str, Any] | None:
    try:
        arr = np.asarray(arr_like)
    except Exception:
        return None
    if arr.dtype == object:
        return None
    if not np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
        return None
    arr64 = arr.astype(np.float64, copy=False)
    flat = arr64.reshape(-1) if arr64.shape else arr64.reshape(1)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(flat)) if flat.size else None,
        "max": float(np.max(flat)) if flat.size else None,
        "mean": float(np.mean(flat)) if flat.size else None,
        "std": float(np.std(flat)) if flat.size else None,
    }


def walk_structure(obj: Any, prefix: str = "") -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    structure: dict[str, str] = {}
    numeric: dict[str, dict[str, Any]] = {}
    if isinstance(obj, dict):
        structure[prefix or "."] = "dict"
        for key in sorted(obj):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            child_structure, child_numeric = walk_structure(obj[key], child_prefix)
            structure.update(child_structure)
            numeric.update(child_numeric)
    elif isinstance(obj, list):
        summary = numeric_leaf_summary(obj)
        if summary is not None:
            structure[prefix] = "numeric_leaf"
            numeric[prefix] = summary
        else:
            structure[prefix] = "list"
            if obj:
                child_structure, child_numeric = walk_structure(obj[0], f"{prefix}[0]")
                structure.update(child_structure)
                numeric.update(child_numeric)
    else:
        summary = numeric_leaf_summary(obj)
        if summary is not None:
            structure[prefix] = "numeric_leaf"
            numeric[prefix] = summary
        else:
            structure[prefix] = type(obj).__name__
    return structure, numeric


def flatten_numeric(obj: Any, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if isinstance(obj, dict):
        for key in sorted(obj):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_numeric(obj[key], child_prefix))
        return out
    summary = numeric_leaf_summary(obj)
    if summary is not None:
        out[prefix] = np.asarray(obj)
    return out


def save_npz_numeric(path: Path, numeric: dict[str, np.ndarray]) -> None:
    safe = {key.replace(".", "__").replace("/", "__"): np.asarray(value) for key, value in numeric.items()}
    np.savez_compressed(path, **safe)


def audit_dataset_files(dataset_path: Path) -> dict[str, Any]:
    files = []
    total_bytes = 0
    rel_hashes = {}
    for path in sorted(p for p in dataset_path.rglob("*") if p.is_file()):
        rel = path.relative_to(dataset_path).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        files.append(rel)
        total_bytes += size
        rel_hashes[rel] = {"sha256": digest, "bytes": size}
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(rel_hashes[rel]["sha256"].encode("ascii"))
        h.update(b"\n")
    special = {
        key: rel_hashes.get(key)
        for key in [
            "meta/stats.json",
            "meta/info.json",
            "meta/tasks.jsonl",
            "data/chunk-000/episode_000000.parquet",
            "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
        ]
    }
    return {
        "dataset_path": str(dataset_path),
        "file_count": len(files),
        "total_byte_count": total_bytes,
        "tree_content_hash": h.hexdigest(),
        "files": rel_hashes,
        "sorted_relative_file_list": files,
        "special_files": special,
    }


def import_modality_config(config_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import modality config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[config_path.stem] = module
    spec.loader.exec_module(module)


def get_training_modality_configs(config_path: Path) -> dict[str, Any]:
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    tag = EmbodimentTag.NEW_EMBODIMENT.value
    if tag not in MODALITY_CONFIGS:
        import_modality_config(config_path)
    if tag not in MODALITY_CONFIGS:
        raise RuntimeError(f"Missing modality config for {tag}")
    return MODALITY_CONFIGS[tag]


def image_to_rgb_uint8(img: Any) -> np.ndarray:
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"))
    else:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.size and float(np.max(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def load_remote_semantic_stats(dataset_path: Path, config_path: Path) -> dict[str, Any]:
    modality_configs = get_training_modality_configs(config_path)
    loader = LeRobotEpisodeLoader(dataset_path, modality_configs=modality_configs)
    return {EmbodimentTag.NEW_EMBODIMENT.value: loader.get_dataset_statistics()}


def export_episode_samples(dataset_path: Path, config_path: Path, frames: list[int], output_dir: Path) -> dict[str, Any]:
    modality_configs = get_training_modality_configs(config_path)
    loader = LeRobotEpisodeLoader(dataset_path, modality_configs=modality_configs)
    df = loader[0]
    n = len(df)
    samples_meta = {"episode_length": n, "requested_frames": frames, "frames": {}}
    npz: dict[str, np.ndarray] = {}
    for frame in frames:
        if frame < 0 or frame >= n:
            samples_meta["frames"][str(frame)] = {"error": f"out_of_range episode_length={n}"}
            continue
        row = df.iloc[frame]
        frame_key = f"frame_{frame:03d}"
        frame_meta: dict[str, Any] = {}
        for key in ["timestamp", "frame_index", "episode_index"]:
            if key in row:
                val = row[key]
                frame_meta[key] = val.item() if isinstance(val, np.generic) else val
        lang_col = "language.annotation.human.action.task_description"
        if lang_col in row:
            frame_meta["task_text"] = str(row[lang_col])
        for kind in ["state", "action"]:
            for name in ACTION_STATE_KEYS:
                col = f"{kind}.{name}"
                if col in row:
                    arr = np.asarray(row[col], dtype=np.float32)
                    npz[f"{frame_key}_{col.replace('.', '_')}"] = arr
                    frame_meta[col] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        video_col = "video.ego_view"
        if video_col in row:
            rgb = image_to_rgb_uint8(row[video_col])
            npz[f"{frame_key}_video_ego_view_rgb"] = rgb
            Image.fromarray(rgb).save(output_dir / f"rgb_frame_{frame:03d}.png")
            frame_meta["video.ego_view"] = {"shape": list(rgb.shape), "dtype": str(rgb.dtype)}
        samples_meta["frames"][str(frame)] = frame_meta
    np.savez_compressed(output_dir / "episode0_samples.npz", **npz)
    (output_dir / "episode0_samples.json").write_text(
        json.dumps(json_safe(samples_meta), indent=2, sort_keys=True), encoding="utf-8"
    )
    return samples_meta


def load_processor_stats(checkpoint_path: Path) -> tuple[Any, dict[str, Any]]:
    import gr00t.model  # noqa: F401 register custom processor/model

    processor_dir = (
        checkpoint_path / "processor"
        if (checkpoint_path / "processor").is_dir()
        and not (checkpoint_path / "processor_config.json").exists()
        else checkpoint_path
    )
    processor = AutoProcessor.from_pretrained(processor_dir)
    processor.eval()
    state_action_processor = getattr(processor, "state_action_processor", None)
    stats = getattr(processor, "statistics", None)
    if not stats and state_action_processor is not None:
        stats = getattr(state_action_processor, "statistics", None)
    norm_params = getattr(state_action_processor, "norm_params", {}) if state_action_processor else {}
    modality_configs = processor.get_modality_configs() if hasattr(processor, "get_modality_configs") else {}
    stats_file = processor_dir / "statistics.json"
    source_file = str(stats_file) if stats_file.exists() else None
    metadata = {
        "processor_dir": str(processor_dir),
        "processor_class": f"{processor.__class__.__module__}.{processor.__class__.__name__}",
        "state_action_processor_class": f"{state_action_processor.__class__.__module__}.{state_action_processor.__class__.__name__}"
        if state_action_processor is not None
        else None,
        "source_stats_file": source_file,
        "source_stats_file_sha256": sha256_file(stats_file) if stats_file.exists() else None,
        "normalization_mode": {
            "use_percentiles": getattr(state_action_processor, "use_percentiles", None),
            "clip_outliers": getattr(state_action_processor, "clip_outliers", None),
            "apply_sincos_state_encoding": getattr(
                state_action_processor, "apply_sincos_state_encoding", None
            ),
            "use_relative_action": getattr(state_action_processor, "use_relative_action", None),
        },
        "loaded_modality_keys": {
            tag: {mod: list(cfg.modality_keys) for mod, cfg in mods.items()}
            for tag, mods in modality_configs.items()
        },
        "loaded_state_stat_keys": sorted(
            flatten_numeric(stats or {}).keys()
        ),
        "loaded_action_norm_param_keys": sorted(flatten_numeric(norm_params).keys()),
    }
    return processor, {"statistics": stats or {}, "norm_params": norm_params, "metadata": metadata}


def compare_numeric(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, Any]:
    matches = []
    missing_in_b = []
    extra_in_b = []
    for key in sorted(a):
        if key not in b:
            missing_in_b.append(key)
            continue
        arr_a = np.asarray(a[key], dtype=np.float64)
        arr_b = np.asarray(b[key], dtype=np.float64)
        if arr_a.shape != arr_b.shape:
            matches.append({"path": key, "shape_a": list(arr_a.shape), "shape_b": list(arr_b.shape), "shape_mismatch": True})
            continue
        diff = np.abs(arr_a - arr_b)
        matches.append(
            {
                "path": key,
                "shape": list(arr_a.shape),
                "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
                "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
            }
        )
    for key in sorted(b):
        if key not in a:
            extra_in_b.append(key)
    comparable = [m for m in matches if not m.get("shape_mismatch")]
    max_diff = max((m["max_abs_diff"] for m in comparable), default=None)
    classification = (
        "CHECKPOINT_STATS_SOURCE_UNRESOLVED"
        if not b
        else "REMOTE_CHECKPOINT_MATCHES_REMOTE_DATASET"
        if comparable and max_diff is not None and max_diff <= 1e-8 and not missing_in_b and not extra_in_b
        else "REMOTE_CHECKPOINT_DIFFERS_FROM_REMOTE_DATASET"
    )
    return {
        "matches": matches,
        "missing_in_checkpoint": missing_in_b,
        "extra_in_checkpoint": extra_in_b,
        "max_abs_diff_overall": max_diff,
        "classification": classification,
    }


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}


def read_text_head_if_exists(path: Path, max_chars: int = 200000) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def audit_training_provenance(dataset_path: Path, checkpoint_path: Path, run_output_path: Path) -> dict[str, Any]:
    files = {
        "checkpoint_experiment_cfg_conf": checkpoint_path / "experiment_cfg" / "conf.yaml",
        "checkpoint_experiment_cfg_config": checkpoint_path / "experiment_cfg" / "config.yaml",
        "checkpoint_experiment_dataset_statistics": checkpoint_path / "experiment_cfg" / "dataset_statistics.json",
        "checkpoint_wandb_config": checkpoint_path / "wandb_config.json",
        "checkpoint_processor_config": checkpoint_path / "processor_config.json",
        "checkpoint_statistics": checkpoint_path / "statistics.json",
        "parent_wandb_config": run_output_path / "wandb_config.json",
    }
    out = {"files": {}}
    text_blob = ""
    for name, path in files.items():
        stat = None
        if path.exists():
            stat = {"mtime": path.stat().st_mtime, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        out["files"][name] = {"path": str(path), "exists": path.exists(), "stat": stat}
        content = read_text_head_if_exists(path)
        if content is not None:
            out["files"][name]["head"] = content[:4000]
            text_blob += "\n" + content
    indicators = {
        "dataset_path_in_configs": str(dataset_path) in text_blob,
        "base_model_mentions": sorted(set(line.strip() for line in text_blob.splitlines() if "base" in line.lower() and "model" in line.lower()))[:30],
        "dataset_path_mentions": sorted(set(line.strip() for line in text_blob.splitlines() if "dataset" in line.lower()))[:50],
        "modality_config_mentions": sorted(set(line.strip() for line in text_blob.splitlines() if "modality" in line.lower()))[:30],
        "embodiment_mentions": sorted(set(line.strip() for line in text_blob.splitlines() if "embodiment" in line.lower()))[:30],
    }
    out["indicators"] = indicators
    watched = [dataset_path / "meta" / "stats.json"]
    watched += sorted((dataset_path / "data").rglob("*.parquet"))[:20]
    watched += sorted((dataset_path / "videos").rglob("*.mp4"))[:20]
    watched += [checkpoint_path / "processor_config.json", checkpoint_path / "statistics.json"]
    out["modification_times"] = {
        str(p): p.stat().st_mtime for p in watched if p.exists() and p.is_file()
    }
    return out


def assert_no_bundle_overwrite(output_dir: Path, frames: list[int]) -> None:
    planned = [
        "server_audit.json",
        "dataset_file_hashes.json",
        "episode0_samples.npz",
        "episode0_samples.json",
        "checkpoint_loaded_stats.npz",
        "checkpoint_loaded_stats.json",
        "bundle_manifest.sha256",
    ]
    planned += [f"rgb_frame_{frame:03d}.png" for frame in frames]
    existing = [name for name in planned if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing bundle files: " + ", ".join(existing)
        )


def write_bundle_manifest(output_dir: Path) -> str:
    entries = []
    for path in sorted(p for p in output_dir.iterdir() if p.is_file() and p.name not in SPECIAL_OUTPUTS):
        entries.append((path.name, sha256_file(path)))
    data = "".join(f"{digest}  {rel}\n" for rel, digest in entries)
    manifest_hash = sha256_bytes(data.encode("utf-8"))
    (output_dir / "bundle_manifest.sha256").write_text(data, encoding="utf-8")
    return manifest_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--run-output-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument(
        "--modality-config-path",
        type=Path,
        default=Path("examples/ARTIMANO/oakink2_artimano_bimanual_config.py"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert_no_bundle_overwrite(args.output_dir, args.frames)

    stats_path = args.dataset_path / "meta" / "stats.json"
    stats_raw = stats_path.read_bytes()
    stats_obj = json.loads(stats_raw)
    structure, numeric_summary = walk_structure(stats_obj)
    remote_dataset_numeric = flatten_numeric(stats_obj)
    remote_raw_sha = sha256_bytes(stats_raw)
    remote_canonical_sha = sha256_bytes(canonical_json_bytes(stats_obj))

    dataset_manifest = audit_dataset_files(args.dataset_path)
    remote_semantic_stats = load_remote_semantic_stats(args.dataset_path, args.modality_config_path)
    remote_semantic_numeric = flatten_numeric(remote_semantic_stats)
    remote_semantic_structure, remote_semantic_numeric_summary = walk_structure(remote_semantic_stats)
    episode_meta = export_episode_samples(
        args.dataset_path, args.modality_config_path, args.frames, args.output_dir
    )

    _processor, processor_data = load_processor_stats(args.checkpoint_path)
    loaded_stats = processor_data["statistics"]
    loaded_numeric = flatten_numeric(loaded_stats)
    loaded_structure, loaded_numeric_summary = walk_structure(loaded_stats)
    loaded_canonical_sha = sha256_bytes(canonical_json_bytes(json_safe(loaded_stats)))
    save_npz_numeric(args.output_dir / "checkpoint_loaded_stats.npz", loaded_numeric)

    comparison = compare_numeric(remote_semantic_numeric, loaded_numeric)
    provenance = audit_training_provenance(args.dataset_path, args.checkpoint_path, args.run_output_path)

    server_audit = {
        "dataset_path": str(args.dataset_path),
        "checkpoint_path": str(args.checkpoint_path),
        "remote_stats_raw_sha256": remote_raw_sha,
        "remote_stats_canonical_json_sha256": remote_canonical_sha,
        "remote_stats_structure": structure,
        "remote_stats_numeric_summary": numeric_summary,
        "remote_semantic_stats_structure": remote_semantic_structure,
        "remote_semantic_stats_numeric_summary": remote_semantic_numeric_summary,
        "checkpoint_loaded_stats_canonical_sha256": loaded_canonical_sha,
        "checkpoint_vs_remote_dataset_stats": comparison,
        "training_provenance": provenance,
        "episode0_samples": episode_meta,
    }
    (args.output_dir / "server_audit.json").write_text(
        json.dumps(json_safe(server_audit), indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "dataset_file_hashes.json").write_text(
        json.dumps(json_safe(dataset_manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    checkpoint_stats_json = {
        "metadata": processor_data["metadata"],
        "loaded_stats_structure": loaded_structure,
        "loaded_stats_numeric_summary": loaded_numeric_summary,
        "loaded_processor_stats_canonical_sha256": loaded_canonical_sha,
        "remote_comparison": comparison,
    }
    (args.output_dir / "checkpoint_loaded_stats.json").write_text(
        json.dumps(json_safe(checkpoint_stats_json), indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest_hash = write_bundle_manifest(args.output_dir)
    print(f"remote_raw_stats_sha256 = {remote_raw_sha}")
    print(f"remote_canonical_stats_sha256 = {remote_canonical_sha}")
    print(f"checkpoint_normalization_source = {processor_data['metadata']['source_stats_file']}")
    print(f"checkpoint_loaded_stats_canonical_sha256 = {loaded_canonical_sha}")
    print(f"server_side_consistency = {comparison['classification']}")
    print(f"dataset_file_count = {dataset_manifest['file_count']}")
    print(f"dataset_tree_hash = {dataset_manifest['tree_content_hash']}")
    print(f"bundle_manifest_hash = {manifest_hash}")
    print(f"bundle_output_dir = {args.output_dir}")


if __name__ == "__main__":
    main()
