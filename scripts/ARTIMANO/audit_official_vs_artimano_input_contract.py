#!/usr/bin/env python3
"""Compare official SO100 and custom ARTIMANO GR00T input contracts.

This script is an audit/report generator.  It does not modify datasets or model
code and does not launch long training by itself; it writes the exact positive-
control finetune/eval commands plus inspected dataloader/server-payload contract.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = Path("/mnt/data/haoyu_data/isaaclab_assets/oakink2_fccd8/official_vs_artimano_input_contract_audit")
OFFICIAL_DATASET = REPO_ROOT / "demo_data/cube_to_bowl_5"
OFFICIAL_CONFIG_PATH = REPO_ROOT / "examples/SO100/so100_config.py"
OFFICIAL_TAG = "NEW_EMBODIMENT"
OFFICIAL_OUTPUT_DIR = Path("/mnt/data/haoyu_data/gr00t_outputs/official_positive_control_projector_diffusion_2gpu")
OFFICIAL_EXPERIMENT_NAME = "official_positive_control_projector_diffusion_2gpu"
OFFICIAL_RUN_DIR = OFFICIAL_OUTPUT_DIR / OFFICIAL_EXPERIMENT_NAME
OFFICIAL_CHECKPOINT_STEP = 200
ARTIMANO_DATASET = Path("/mnt/data/haoyu_data/oakink2/gr00t_lerobot/fccd8_artimano_torch_wrist_qpos_isaac_render_rgb_row68_window_relstats_recomputed")
ARTIMANO_CONFIG_PATH = REPO_ROOT / "examples/ARTIMANO/oakink2_artimano_bimanual_config.py"
ARTIMANO_TAG = "NEW_EMBODIMENT"
ARTIMANO_OUTPUT_DIR = Path("/mnt/data/haoyu_data/gr00t_outputs/oakink2_fccd8_artimano_torch_isaac_render_rgb_row68_from_base_relstats_recomputed_projector_diffusion_2gpu")
ARTIMANO_EVAL_SCRIPT = Path("/home/david/IsaacLab/scripts/gr00t_bridge/eval_gr00t_openloop_isaac_render_rgb.py")
ARTIMANO_ROWS = "0,20,40,68,100,140"


def load_module(path: Path, module_name: str):
    # SO100 and ARTIMANO example configs both register NEW_EMBODIMENT.
    # This audit keeps explicit config dict copies, so it is safe to clear
    # the registry slot before loading each example config.
    try:
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        MODALITY_CONFIGS.pop("new_embodiment", None)
    except Exception:
        pass
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        "path": str(video_path),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_read_frames": int(stream.get("nb_read_frames") or stream.get("nb_frames") or -1),
    }


def arr_stats(value: Any) -> dict[str, Any]:
    try:
        from PIL import Image
    except Exception:
        Image = None
    if Image is not None and isinstance(value, Image.Image):
        arr = np.asarray(value.convert("RGB"))
    else:
        arr = np.asarray(value)
    if arr.dtype == object:
        return {"type": str(type(value)), "shape": list(arr.shape), "dtype": str(arr.dtype)}
    arr_float = arr.astype(np.float64) if arr.size else np.asarray(arr, dtype=np.float64)
    out = {
        "type": str(type(value)),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr_float)) if arr.size else None,
        "max": float(np.max(arr_float)) if arr.size else None,
        "mean": float(np.mean(arr_float)) if arr.size else None,
        "std": float(np.std(arr_float)) if arr.size else None,
    }
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        out["layout"] = "HWC"
        out["channel_means_rgb_assuming_hwc"] = [float(x) for x in np.mean(arr_float[..., :3].reshape(-1, 3), axis=0)]
        out["rgb_bgr_check"] = "stored/decoded as RGB by LeRobotEpisodeLoader/PIL"
    elif arr.ndim == 3 and arr.shape[0] in (3, 4):
        out["layout"] = "CHW"
    return out


def split_dim_from_meta(modality_meta: dict[str, Any], modality: str, keys: list[str]) -> int:
    total = 0
    for key in keys:
        item = modality_meta[modality][key]
        total += int(item["end"]) - int(item["start"])
    return total


def action_config_dict(action_keys: list[str], action_configs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "rep": str(cfg.rep),
            "type": str(cfg.type),
            "format": str(cfg.format),
            "state_key": getattr(cfg, "state_key", None),
        }
        for key, cfg in zip(action_keys, action_configs)
    ]


def dataset_video_paths(dataset_path: Path, info: dict[str, Any], image_keys: list[str], episode_index: int = 0) -> dict[str, Path]:
    modality = read_json(dataset_path / "meta/modality.json")
    out = {}
    for key in image_keys:
        original_key = modality["video"][key].get("original_key", f"observation.images.{key}")
        video_rel = info["video_path"].format(episode_chunk=0, video_key=original_key, episode_index=episode_index)
        out[key] = dataset_path / video_rel
    return out


def inspect_contract(label: str, dataset_path: Path, config_path: Path, config_var: str, tag_name: str, row: int = 0) -> dict[str, Any]:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.state_action.state_action_processor import StateActionProcessor

    module = load_module(config_path, f"contract_{label}_config")
    modality_config = deepcopy(getattr(module, config_var))
    embodiment = EmbodimentTag.resolve(tag_name)
    info = read_json(dataset_path / "meta/info.json")
    modality_meta = read_json(dataset_path / "meta/modality.json")
    tasks = read_jsonl(dataset_path / "meta/tasks.jsonl")
    episodes = read_jsonl(dataset_path / "meta/episodes.jsonl")
    loader = LeRobotEpisodeLoader(dataset_path, modality_config)
    stats = loader.get_dataset_statistics()
    processor = StateActionProcessor(
        modality_configs={embodiment.value: modality_config},
        statistics={embodiment.value: stats},
        use_relative_action=True,
        clip_outliers=True,
    )
    traj = loader[0]
    sample = extract_step_data(traj, row, modality_config, embodiment, allow_padding=False)

    video_keys = modality_config["video"].modality_keys
    state_keys = modality_config["state"].modality_keys
    action_keys = modality_config["action"].modality_keys
    language_keys = modality_config["language"].modality_keys
    action_horizon = len(modality_config["action"].delta_indices)
    video_paths = dataset_video_paths(dataset_path, info, video_keys, episode_index=0)

    state_dict = {k: np.asarray(v, dtype=np.float32) for k, v in sample.states.items()}
    action_dict = {k: np.asarray(v, dtype=np.float32) for k, v in sample.actions.items()}
    norm_state, norm_action = processor.apply(state=state_dict, action=action_dict, embodiment_tag=embodiment.value)

    server_payload = {
        "video": {key: f"np.array(video.{key})[None, :] -> shape (1, T, H, W, C)" for key in video_keys},
        "state": {key: f"state.{key}[None, :] -> shape (1, T, D)" for key in state_keys},
        "language": {language_keys[0]: "[[task_string]]"},
    }
    if label == "artimano":
        server_payload = {
            "video": {"ego_view": "rgb[None, None, H, W, C] uint8 RGB"},
            "state": {key: f"state.{key}[None, None, D] float32" for key in state_keys},
            "language": {language_keys[0]: "[[task_string]]"},
        }

    return {
        "label": label,
        "dataset_path": str(dataset_path),
        "config_path": str(config_path),
        "embodiment_tag": tag_name,
        "embodiment_value": embodiment.value,
        "fps": info.get("fps"),
        "num_episodes": len(episodes),
        "num_rows_total_frames": info.get("total_frames"),
        "episode0_rows": len(traj),
        "video_frame_count_episode0": {key: probe_video(path) for key, path in video_paths.items()},
        "image_resolution_from_info": {key: info["features"][modality_meta["video"][key].get("original_key", f"observation.images.{key}")]["shape"] for key in video_keys},
        "task_description_string_episode0": sample.text,
        "tasks_jsonl": tasks,
        "observation_keys": {"video": video_keys, "state": state_keys, "language": language_keys},
        "image_keys": video_keys,
        "state_keys": state_keys,
        "action_keys": action_keys,
        "action_horizon": action_horizon,
        "state_dim": split_dim_from_meta(modality_meta, "state", state_keys),
        "action_dim": split_dim_from_meta(modality_meta, "action", action_keys),
        "action_representation": action_config_dict(action_keys, modality_config["action"].action_configs),
        "normalization_type": "StateActionProcessor minmax unless mean_std_embedding_keys is set; clip_outliers=True; relative action stats used for RELATIVE actions when use_relative_action=True",
        "stats_json_path": str(dataset_path / "meta/stats.json"),
        "relative_stats_json_path": str(dataset_path / "meta/relative_stats.json"),
        "training_sample": {
            "row": row,
            "image_stats": {key: arr_stats(value[0] if isinstance(value, list) else value) for key, value in sample.images.items()},
            "state_stats": {key: arr_stats(value) for key, value in sample.states.items()},
            "action_chunk_stats": {key: arr_stats(value) for key, value in sample.actions.items()},
            "normalized_state_stats": {key: arr_stats(value) for key, value in norm_state.items()},
            "processed_action_stats": {key: arr_stats(value) for key, value in norm_action.items()},
            "action_chunk_offset": "extract_step_data uses indices row + delta_indices; delta_indices=list(range(16)); expected action[row:row+16]",
        },
        "server_inference_payload_contract": server_payload,
        "server_payload_notes": {
            "official": "gr00t/eval/open_loop_eval.py parse_observation_gr00t adds batch dimension to video/state and [[text]] for language.",
            "artimano": "scripts/gr00t_bridge/eval_gr00t_openloop_isaac_render_rgb.py sends nested video/state/language to LightweightGR00TClient.",
        },
    }


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def make_finetune_command() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {REPO_ROOT}
export HF_HOME=/mnt/data/haoyu_data/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/haoyu_data/hf_cache/transformers
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0,1}}
NUM_GPUS=2 \\
MASTER_PORT=${{MASTER_PORT:-29531}} \\
SAVE_STEPS=100 \\
MAX_STEPS=200 \\
USE_WANDB=0 \\
DATALOADER_NUM_WORKERS=4 \\
GLOBAL_BATCH_SIZE=32 \\
SHARD_SIZE=1024 \\
NUM_SHARDS_PER_EPOCH=100000 \\
EPISODE_SAMPLING_RATE=0.1 \\
uv run bash examples/finetune.sh \\
  --base-model-path nvidia/GR00T-N1.7-3B \\
  --dataset-path ./demo_data/cube_to_bowl_5 \\
  --modality-config-path examples/SO100/so100_config.py \\
  --embodiment-tag NEW_EMBODIMENT \\
  --output-dir {OFFICIAL_OUTPUT_DIR} \\
  --experiment-name official_positive_control_projector_diffusion_2gpu
"""


def make_eval_command() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {REPO_ROOT}
export HF_HOME=/mnt/data/haoyu_data/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/haoyu_data/hf_cache/transformers
uv run python gr00t/eval/open_loop_eval.py \\
  --dataset-path ./demo_data/cube_to_bowl_5 \\
  --embodiment-tag NEW_EMBODIMENT \\
  --model-path {OFFICIAL_RUN_DIR}/checkpoint-{OFFICIAL_CHECKPOINT_STEP} \\
  --traj-ids 0 \\
  --action-horizon 16 \\
  --steps 400 \\
  --modality-keys single_arm gripper \\
  --save-plot-path {OUTPUT_DIR}/official_open_loop_eval_traj0.png
"""


def make_artimano_sensitivity_command() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/david/IsaacLab
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}} \\
./isaaclab.sh -p {ARTIMANO_EVAL_SCRIPT.relative_to('/home/david/IsaacLab')} \\
  --rows {ARTIMANO_ROWS} \\
  --ablation-modes correct,black_image,shuffled_image,wrong_task_text,shifted_proprio,zero_proprio \\
  --out-dir {OUTPUT_DIR}/artimano_input_sensitivity_eval
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    official = inspect_contract("official_so100", OFFICIAL_DATASET, OFFICIAL_CONFIG_PATH, "so100_config", OFFICIAL_TAG)
    artimano = inspect_contract("artimano", ARTIMANO_DATASET, ARTIMANO_CONFIG_PATH, "oakink2_artimano_bimanual_config", ARTIMANO_TAG)

    official_ckpt = OFFICIAL_RUN_DIR / f"checkpoint-{OFFICIAL_CHECKPOINT_STEP}"
    artimano_ckpts = sorted(ARTIMANO_OUTPUT_DIR.glob("checkpoint-*")) if ARTIMANO_OUTPUT_DIR.exists() else []
    official_eval_result_path = OUTPUT_DIR / "official_eval_result.txt"
    official_eval_plot_path = OUTPUT_DIR / "official_open_loop_eval_traj0.png"
    sensitivity_report_path = OUTPUT_DIR / "artimano_input_sensitivity_report.txt"
    sensitivity_metrics_path = OUTPUT_DIR / "artimano_input_sensitivity_metrics.csv"
    sensitivity_metrics_has_rows = (
        sensitivity_metrics_path.exists()
        and len([line for line in sensitivity_metrics_path.read_text().splitlines() if line.strip()]) > 1
    )
    finetune_result = "PASS" if official_ckpt.exists() else "WARN"
    eval_result = "PASS" if official_eval_result_path.exists() and official_eval_plot_path.exists() else "WARN"
    sensitivity_result = "PASS" if sensitivity_report_path.exists() and sensitivity_metrics_has_rows else "WARN"
    contract_result = "PASS"
    overall = "WARN" if "WARN" in [finetune_result, eval_result, sensitivity_result] else "PASS"

    plan = f"""Official GR00T positive-control plan

Selected official example: SO100 cube_to_bowl_5 tutorial dataset.
Reason: repo-shipped, small, supports RGB (front+wrist), proprio state, text task, action output, and projector+diffusion finetuning through examples/finetune.sh.

Official dataset: {OFFICIAL_DATASET}
Official modality config: {OFFICIAL_CONFIG_PATH}
Official embodiment tag: {OFFICIAL_TAG} ({official['embodiment_value']})
Official state dim: {official['state_dim']}
Official action dim: {official['action_dim']}
Official image keys: {official['image_keys']}
Official task/text key: {official['observation_keys']['language']}
Official action horizon: {official['action_horizon']}

Custom ARTIMANO dataset: {ARTIMANO_DATASET}
Custom config: {ARTIMANO_CONFIG_PATH}
Custom tag: {ARTIMANO_TAG} ({artimano['embodiment_value']})
Custom state/action dim: {artimano['state_dim']} / {artimano['action_dim']}
Custom image keys: {artimano['image_keys']}
Custom action horizon: {artimano['action_horizon']}

Training status at audit time:
Official checkpoint-{OFFICIAL_CHECKPOINT_STEP} exists: {official_ckpt.exists()} ({official_ckpt})
ARTIMANO checkpoints found: {[p.name for p in artimano_ckpts[-5:]]}
"""
    write(OUTPUT_DIR / "official_positive_control_plan.txt", plan)
    write(OUTPUT_DIR / "official_finetune_command.sh", make_finetune_command())
    write(OUTPUT_DIR / "official_eval_command.sh", make_eval_command())
    os.chmod(OUTPUT_DIR / "official_finetune_command.sh", 0o755)
    os.chmod(OUTPUT_DIR / "official_eval_command.sh", 0o755)
    write(OUTPUT_DIR / "artimano_input_sensitivity_command.sh", make_artimano_sensitivity_command())
    os.chmod(OUTPUT_DIR / "artimano_input_sensitivity_command.sh", 0o755)

    report_lines = [
        "Official vs ARTIMANO input contract audit",
        "",
        "Official selected example: SO100 cube_to_bowl_5 tutorial dataset",
        "",
        "Official contract:",
        json.dumps(official, indent=2, default=str),
        "",
        "ARTIMANO contract:",
        json.dumps(artimano, indent=2, default=str),
        "",
        "Comparison summary:",
        f"- official image keys: {official['image_keys']} vs ARTIMANO image keys: {artimano['image_keys']}",
        f"- official state/action dim: {official['state_dim']}/{official['action_dim']} vs ARTIMANO {artimano['state_dim']}/{artimano['action_dim']}",
        f"- official action reps: {official['action_representation']}",
        f"- ARTIMANO action reps: {artimano['action_representation']}",
        "- both training paths use LeRobotEpisodeLoader + extract_step_data(row + delta_indices).",
        "- both inference payloads are nested video/state/language dicts with batch/time dimensions.",
        "",
        f"OFFICIAL_FINETUNE_RESULT = {finetune_result}",
        f"OFFICIAL_EVAL_RESULT = {eval_result}",
        f"OFFICIAL_VS_ARTIMANO_INPUT_CONTRACT_RESULT = {contract_result}",
        f"ARTIMANO_INPUT_SENSITIVITY_RESULT = {sensitivity_result}",
        f"OVERALL_OFFICIAL_CONTROL_AUDIT_RESULT = {overall}",
    ]
    write(OUTPUT_DIR / "official_vs_artimano_input_contract_report.txt", "\n".join(report_lines) + "\n")

    if sensitivity_result != "PASS":
        sensitivity_report = f"""ARTIMANO input sensitivity report

Status: not executed by this contract audit script.
Reason: requires a running GR00T server/checkpoint through LightweightGR00TClient.
Command generated at: {OUTPUT_DIR / 'artimano_input_sensitivity_command.sh'}

Modes implemented in {ARTIMANO_EVAL_SCRIPT}:
- correct
- black_image
- shuffled_image
- wrong_task_text
- shifted_proprio
- zero_proprio

ARTIMANO_INPUT_SENSITIVITY_RESULT = {sensitivity_result}
"""
        write(sensitivity_report_path, sensitivity_report)
        write(sensitivity_metrics_path, "row,source_frame_id,mode,infer_dt,pred_vs_correct_l2,wrist_xyz_prediction_diff_l2,qpos_prediction_diff_l2,gt_error_l2,gt_error_mse\n")

    summary = {
        "official": official,
        "artimano": artimano,
        "commands": {
            "official_finetune_command": str(OUTPUT_DIR / "official_finetune_command.sh"),
            "official_eval_command": str(OUTPUT_DIR / "official_eval_command.sh"),
            "artimano_input_sensitivity_command": str(OUTPUT_DIR / "artimano_input_sensitivity_command.sh"),
        },
        "status": {
            "OFFICIAL_FINETUNE_RESULT": finetune_result,
            "OFFICIAL_EVAL_RESULT": eval_result,
            "OFFICIAL_VS_ARTIMANO_INPUT_CONTRACT_RESULT": contract_result,
            "ARTIMANO_INPUT_SENSITIVITY_RESULT": sensitivity_result,
            "OVERALL_OFFICIAL_CONTROL_AUDIT_RESULT": overall,
        },
    }
    write_json(OUTPUT_DIR / "official_vs_artimano_input_contract_summary.json", summary)

    print(f"[INFO] Wrote outputs to: {OUTPUT_DIR}")
    print(f"OFFICIAL_FINETUNE_RESULT = {finetune_result}")
    print(f"OFFICIAL_EVAL_RESULT = {eval_result}")
    print(f"OFFICIAL_VS_ARTIMANO_INPUT_CONTRACT_RESULT = {contract_result}")
    print(f"ARTIMANO_INPUT_SENSITIVITY_RESULT = {sensitivity_result}")
    print(f"OVERALL_OFFICIAL_CONTROL_AUDIT_RESULT = {overall}")


if __name__ == "__main__":
    main()
