#!/usr/bin/env python
"""Deterministic transport sentinel server for 1292e ManipTrans ↔ GR00T checks."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.policy.policy import BasePolicy
from gr00t.policy.server_client import PolicyServer

SCHEMA_VERSION = "1292e_transport_sentinel_v1"
TASK_TEXT = "1292e transport sentinel v1"
JOINT_ORDER = [
    "j_index1y",
    "j_index1z",
    "j_index2",
    "j_index3",
    "j_middle1y",
    "j_middle1z",
    "j_middle2",
    "j_middle3",
    "j_pinky1y",
    "j_pinky1z",
    "j_pinky2",
    "j_pinky3",
    "j_ring1y",
    "j_ring1z",
    "j_ring2",
    "j_ring3",
    "j_thumb1x",
    "j_thumb1y",
    "j_thumb1z",
    "j_thumb2y",
    "j_thumb2z",
    "j_thumb3",
]
STATE_KEYS = ["left_wrist_pose", "right_wrist_pose", "left_hand_qpos", "right_hand_qpos"]
ACTION_KEYS = STATE_KEYS
LANGUAGE_KEY = "annotation.human.action.task_description"
VIDEO_KEY = "ego_view"
RGB_SAMPLE_PIXELS = [(0, 0), (0, 639), (479, 0), (479, 639), (100, 200)]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(value: Any) -> str | None:
    try:
        arr = np.ascontiguousarray(np.asarray(value))
    except Exception:
        return None
    return sha256_bytes(arr.tobytes())


def array_summary(value: Any) -> dict[str, Any]:
    try:
        arr = np.asarray(value)
    except Exception as exc:
        return {"error": str(exc), "type": type(value).__name__}
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": array_sha256(arr),
    }
    if arr.size and np.issubdtype(arr.dtype, np.number):
        summary["min"] = float(np.min(arr))
        summary["max"] = float(np.max(arr))
    return summary


def axis_angle_to_rotmat(axis: list[float] | np.ndarray, angle: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=np.float64)
    axis_arr = axis_arr / np.linalg.norm(axis_arr)
    x, y, z = axis_arr
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    tr = float(np.trace(rot))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def rotmat_to_rot6d_rows(rot: np.ndarray) -> np.ndarray:
    return np.asarray(rot[:2, :].reshape(6), dtype=np.float64)


def rot6d_rows_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    row1 = np.asarray(rot6d[:3], dtype=np.float64)
    row2 = np.asarray(rot6d[3:6], dtype=np.float64)
    r1 = row1 / max(np.linalg.norm(row1), 1e-12)
    r2 = row2 - np.dot(row2, r1) * r1
    r2 = r2 / max(np.linalg.norm(r2), 1e-12)
    r3 = np.cross(r1, r2)
    return np.stack([r1, r2, r3], axis=0)


def rotation_geodesic(a: np.ndarray, b: np.ndarray) -> float:
    rel = a @ b.T
    cos_angle = (np.trace(rel) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def diff_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    if actual_arr.shape != expected_arr.shape:
        return {"shape_mismatch": True, "actual_shape": list(actual_arr.shape), "expected_shape": list(expected_arr.shape)}
    diff = np.abs(actual_arr - expected_arr)
    return {
        "shape_mismatch": False,
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
    }


def make_rgb_pattern() -> np.ndarray:
    y, x = np.indices((480, 640), dtype=np.uint16)
    rgb = np.empty((480, 640, 3), dtype=np.uint8)
    rgb[..., 0] = (x % 256).astype(np.uint8)
    rgb[..., 1] = (y % 256).astype(np.uint8)
    rgb[..., 2] = ((3 * x + 5 * y) % 256).astype(np.uint8)
    return rgb


def make_upload_expected() -> dict[str, Any]:
    left_rot = axis_angle_to_rotmat([1.0, 2.0, 3.0], 0.4)
    right_rot = axis_angle_to_rotmat([-2.0, 1.0, 1.0], -0.35)
    left_wrist = np.concatenate([[0.111, -0.222, 0.333], rotmat_to_rot6d_rows(left_rot)]).astype(np.float32)
    right_wrist = np.concatenate([[-0.444, 0.555, 0.666], rotmat_to_rot6d_rows(right_rot)]).astype(np.float32)
    left_qpos = np.array([0.001 * (i + 1) for i in range(22)], dtype=np.float32)
    right_qpos = np.array([-0.0015 * (i + 1) for i in range(22)], dtype=np.float32)
    return {
        "left_rot": left_rot,
        "right_rot": right_rot,
        "left_quat_xyzw": rotmat_to_quat_xyzw(left_rot),
        "right_quat_xyzw": rotmat_to_quat_xyzw(right_rot),
        "state": OrderedDict(
            [
                ("left_wrist_pose", left_wrist),
                ("right_wrist_pose", right_wrist),
                ("left_hand_qpos", left_qpos),
                ("right_hand_qpos", right_qpos),
            ]
        ),
        "canonical_62d": np.concatenate([left_wrist, right_wrist, left_qpos, right_qpos]).astype(np.float32),
        "rgb": make_rgb_pattern()[None, None, ...],
        "task_text": TASK_TEXT,
    }


def sentinel_spec() -> dict[str, Any]:
    expected = make_upload_expected()
    return {
        "schema_version": SCHEMA_VERSION,
        "joint_order": JOINT_ORDER,
        "left_wrist_xyz": expected["state"]["left_wrist_pose"][:3],
        "left_wrist_quat_xyzw": expected["left_quat_xyzw"],
        "left_wrist_rot6d_rows": expected["state"]["left_wrist_pose"][3:9],
        "right_wrist_xyz": expected["state"]["right_wrist_pose"][:3],
        "right_wrist_quat_xyzw": expected["right_quat_xyzw"],
        "right_wrist_rot6d_rows": expected["state"]["right_wrist_pose"][3:9],
        "left_qpos": expected["state"]["left_hand_qpos"],
        "right_qpos": expected["state"]["right_hand_qpos"],
        "rgb_shape": [1, 1, 480, 640, 3],
        "rgb_formula": "R=x%256,G=y%256,B=(3*x+5*y)%256",
        "task_text": TASK_TEXT,
        "response_horizon": 16,
    }


def make_return_action() -> OrderedDict[str, np.ndarray]:
    left_wrist = np.zeros((1, 16, 9), dtype=np.float32)
    right_wrist = np.zeros((1, 16, 9), dtype=np.float32)
    left_qpos = np.zeros((1, 16, 22), dtype=np.float32)
    right_qpos = np.zeros((1, 16, 22), dtype=np.float32)
    for h in range(16):
        left_rot = axis_angle_to_rotmat([1, 0, 1], 0.02 * (h + 1))
        right_rot = axis_angle_to_rotmat([0, 1, -1], -0.025 * (h + 1))
        left_wrist[0, h] = np.concatenate(
            [[0.400 + 0.001 * h, -0.200 - 0.002 * h, 0.800 + 0.003 * h], rotmat_to_rot6d_rows(left_rot)]
        )
        right_wrist[0, h] = np.concatenate(
            [[-0.300 - 0.001 * h, 0.250 + 0.002 * h, 0.700 - 0.003 * h], rotmat_to_rot6d_rows(right_rot)]
        )
        left_qpos[0, h] = np.array([0.002 * (i + 1) + 0.0001 * h for i in range(22)], dtype=np.float32)
        right_qpos[0, h] = np.array([-0.0025 * (i + 1) - 0.0002 * h for i in range(22)], dtype=np.float32)
    return OrderedDict(
        [
            ("left_wrist_pose", left_wrist),
            ("right_wrist_pose", right_wrist),
            ("left_hand_qpos", left_qpos),
            ("right_hand_qpos", right_qpos),
        ]
    )


def extract_nested(observation: dict[str, Any], modality: str, key: str) -> Any:
    if modality in observation and isinstance(observation[modality], dict) and key in observation[modality]:
        return observation[modality][key]
    flat_key = f"{modality}.{key}"
    if flat_key in observation:
        return observation[flat_key]
    return None


def array_to_bt_vector(value: Any, dim: int) -> tuple[np.ndarray | None, bool, dict[str, Any]]:
    if value is None:
        return None, False, {"missing": True}
    arr = np.asarray(value)
    exact = list(arr.shape) == [1, 1, dim]
    if exact:
        return np.asarray(arr[0, 0], dtype=np.float32), True, array_summary(arr)
    squeezed = np.squeeze(arr)
    if squeezed.shape == (dim,):
        summary = array_summary(arr)
        summary["expected_exact_shape"] = [1, 1, dim]
        return np.asarray(squeezed, dtype=np.float32), False, summary
    summary = array_summary(arr)
    summary["expected_exact_shape"] = [1, 1, dim]
    return None, False, summary


def extract_task_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    arr = np.asarray(value, dtype=object)
    if arr.size == 0:
        return None
    return str(arr.reshape(-1)[0])


def rgb_compare(received: Any, expected: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"received": array_summary(received), "expected_sha256": array_sha256(expected)}
    if received is None:
        out["pass"] = False
        out["error"] = "missing video.ego_view"
        return out
    arr = np.asarray(received)
    exact_shape = list(arr.shape) == list(expected.shape)
    out["exact_shape_match"] = exact_shape
    if arr.dtype != np.uint8:
        out["dtype_match"] = False
    else:
        out["dtype_match"] = True
    if exact_shape:
        arr_u8 = arr.astype(np.uint8, copy=False)
        diff = np.abs(arr_u8.astype(np.int16) - expected.astype(np.int16))
        out.update(
            {
                "sha256": array_sha256(arr_u8),
                "max_pixel_diff": int(np.max(diff)),
                "mean_pixel_diff": float(np.mean(diff)),
                "per_channel_mae": [float(np.mean(diff[..., c])) for c in range(3)],
                "sample_pixels": {
                    f"{y},{x}": {
                        "received": arr_u8[0, 0, y, x].tolist(),
                        "expected": expected[0, 0, y, x].tolist(),
                    }
                    for y, x in RGB_SAMPLE_PIXELS
                },
            }
        )
        out["pass"] = out["dtype_match"] and out["max_pixel_diff"] == 0
        alt = {}
        base = expected[0, 0]
        got = arr_u8[0, 0]
        alt["rgb_bgr_max_diff"] = int(np.max(np.abs(got[..., ::-1].astype(np.int16) - base.astype(np.int16))))
        alt["horizontal_flip_max_diff"] = int(np.max(np.abs(got[:, ::-1].astype(np.int16) - base.astype(np.int16))))
        alt["vertical_flip_max_diff"] = int(np.max(np.abs(got[::-1, :].astype(np.int16) - base.astype(np.int16))))
        alt["transpose_hw_max_diff"] = None
        if got.transpose(1, 0, 2).shape == base.shape:
            alt["transpose_hw_max_diff"] = int(np.max(np.abs(got.transpose(1, 0, 2).astype(np.int16) - base.astype(np.int16))))
        out["alternative_hypotheses"] = alt
        return out
    out["pass"] = False
    alt = {}
    squeezed = np.squeeze(arr)
    if squeezed.shape == (3, 480, 640):
        chw_to_hwc = squeezed.transpose(1, 2, 0).astype(np.uint8, copy=False)
        alt["chw_to_hwc_max_diff"] = int(np.max(np.abs(chw_to_hwc.astype(np.int16) - expected[0, 0].astype(np.int16))))
    else:
        alt["chw_to_hwc_max_diff"] = None
    out["alternative_hypotheses"] = alt
    return out


def compare_upload(observation: dict[str, Any]) -> dict[str, Any]:
    expected = make_upload_expected()
    received_keys_order = list(observation.keys())
    received_field_order: list[str] = []
    received_shapes: dict[str, Any] = {}
    received_hashes: dict[str, Any] = {}
    extracted: dict[str, np.ndarray] = {}
    exact_shape_ok = True

    for state_key, dim in [("left_wrist_pose", 9), ("right_wrist_pose", 9), ("left_hand_qpos", 22), ("right_hand_qpos", 22)]:
        raw = extract_nested(observation, "state", state_key)
        field_name = f"state.{state_key}"
        received_field_order.append(field_name)
        vec, exact_shape, summary = array_to_bt_vector(raw, dim)
        received_shapes[field_name] = summary
        received_hashes[field_name] = summary.get("sha256")
        exact_shape_ok = exact_shape_ok and exact_shape
        if vec is not None:
            extracted[state_key] = vec

    flat_actual = None
    if all(key in extracted for key in STATE_KEYS):
        flat_actual = np.concatenate([extracted[key] for key in STATE_KEYS]).astype(np.float32)

    fieldwise: dict[str, Any] = {}
    for state_key in STATE_KEYS:
        if state_key in extracted:
            fieldwise[state_key] = diff_summary(extracted[state_key], expected["state"][state_key])
        else:
            fieldwise[state_key] = {"missing_or_bad_shape": True}

    for prefix in ["left", "right"]:
        key = f"{prefix}_wrist_pose"
        if key in extracted:
            actual_rot = rot6d_rows_to_rotmat(extracted[key][3:9])
            expected_rot = expected[f"{prefix}_rot"]
            fieldwise[f"{prefix}_wrist_xyz"] = diff_summary(extracted[key][:3], expected["state"][key][:3])
            fieldwise[f"{prefix}_wrist_rot6d"] = diff_summary(extracted[key][3:9], expected["state"][key][3:9])
            fieldwise[f"{prefix}_wrist_rotation_geodesic_rad"] = rotation_geodesic(actual_rot, expected_rot)

    for side in ["left", "right"]:
        qkey = f"{side}_hand_qpos"
        if qkey in extracted:
            joints = {}
            exp = expected["state"][qkey]
            actual = extracted[qkey]
            for idx, joint_name in enumerate(JOINT_ORDER):
                joints[joint_name] = {
                    "actual": float(actual[idx]),
                    "expected": float(exp[idx]),
                    "abs_diff": float(abs(float(actual[idx]) - float(exp[idx]))),
                }
            fieldwise[f"{qkey}_by_joint_name"] = joints

    flat_summary = {"missing": True}
    if flat_actual is not None:
        flat_summary = diff_summary(flat_actual, expected["canonical_62d"])

    task_raw = extract_nested(observation, "language", LANGUAGE_KEY)
    task_text = extract_task_text(task_raw)
    task_comparison = {
        "received": task_text,
        "expected": TASK_TEXT,
        "equal": task_text == TASK_TEXT,
        "raw_summary": str(type(task_raw).__name__),
    }

    video_raw = extract_nested(observation, "video", VIDEO_KEY)
    rgb = rgb_compare(video_raw, expected["rgb"])
    received_field_order.append(f"video.{VIDEO_KEY}")
    received_shapes[f"video.{VIDEO_KEY}"] = array_summary(video_raw) if video_raw is not None else {"missing": True}
    received_hashes[f"video.{VIDEO_KEY}"] = received_shapes[f"video.{VIDEO_KEY}"].get("sha256")
    received_field_order.append(f"language.{LANGUAGE_KEY}")

    alternatives: dict[str, Any] = {}
    if flat_actual is not None:
        swapped = np.concatenate(
            [extracted["right_wrist_pose"], extracted["left_wrist_pose"], extracted["right_hand_qpos"], extracted["left_hand_qpos"]]
        )
        alternatives["left_right_swapped_flat62_max_diff"] = diff_summary(swapped, expected["canonical_62d"])
        reversed_qpos = flat_actual.copy()
        reversed_qpos[18:40] = reversed_qpos[18:40][::-1]
        reversed_qpos[40:62] = reversed_qpos[40:62][::-1]
        alternatives["qpos_reversed_flat62_max_diff"] = diff_summary(reversed_qpos, expected["canonical_62d"])

    pass_conditions = [
        exact_shape_ok,
        flat_actual is not None,
        flat_summary.get("max_abs_diff", 1.0) == 0.0,
        task_comparison["equal"],
        rgb.get("pass", False),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "UPLOAD_SENTINEL_PASS" if all(pass_conditions) else "UPLOAD_SENTINEL_FAIL",
        "pass": bool(all(pass_conditions)),
        "received_observation_keys_order": received_keys_order,
        "received_field_order_checked": received_field_order,
        "received_shapes_dtypes_hashes": received_shapes,
        "received_array_sha256_values": received_hashes,
        "fieldwise_upload_differences": fieldwise,
        "flattened_62d_difference": flat_summary,
        "task_comparison": task_comparison,
        "rgb_comparison": rgb,
        "alternative_failure_hypotheses": alternatives,
    }


class TransportSentinelPolicy(BasePolicy):
    def __init__(self, output_dir: Path):
        super().__init__(strict=True)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.output_dir / "server_transport_audit.jsonl"
        self.request_index = 0
        self.spec = sentinel_spec()
        self.spec_sha256 = sha256_bytes(canonical_json_bytes(self.spec))
        self.return_action = make_return_action()
        self.return_sha256 = sha256_bytes(canonical_json_bytes({k: v for k, v in self.return_action.items()}))
        self.modality_configs = {
            "video": ModalityConfig(delta_indices=[0], modality_keys=[VIDEO_KEY]),
            "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
            "action": ModalityConfig(
                delta_indices=list(range(16)),
                modality_keys=ACTION_KEYS,
                action_configs=[
                    ActionConfig(ActionRepresentation.RELATIVE, ActionType.EEF, ActionFormat.XYZ_ROT6D, "left_wrist_pose"),
                    ActionConfig(ActionRepresentation.RELATIVE, ActionType.EEF, ActionFormat.XYZ_ROT6D, "right_wrist_pose"),
                    ActionConfig(ActionRepresentation.RELATIVE, ActionType.NON_EEF, ActionFormat.DEFAULT, "left_hand_qpos"),
                    ActionConfig(ActionRepresentation.RELATIVE, ActionType.NON_EEF, ActionFormat.DEFAULT, "right_hand_qpos"),
                ],
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
        }

    def append_record(self, endpoint: str, record: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_index": self.request_index,
            "endpoint": endpoint,
            **record,
        }
        self.request_index += 1
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")
            f.flush()

    def handle_ping(self) -> dict[str, Any]:
        response = {"status": "ok", "message": "Transport sentinel server is running", "schema_version": SCHEMA_VERSION}
        self.append_record("ping", {"pass": True, "classification": "RETURN_SENTINEL_READY"})
        return response

    def check_observation(self, observation: dict[str, Any]) -> None:
        assert isinstance(observation, dict), f"observation must be dict, got {type(observation)}"

    def check_action(self, action: dict[str, Any]) -> None:
        for key, shape in {
            "left_wrist_pose": (1, 16, 9),
            "right_wrist_pose": (1, 16, 9),
            "left_hand_qpos": (1, 16, 22),
            "right_hand_qpos": (1, 16, 22),
        }.items():
            arr = np.asarray(action[key])
            assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
            assert arr.dtype == np.float32, f"{key} dtype {arr.dtype} != float32"

    def _get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None):
        upload = compare_upload(observation)
        response_hashes = {key: array_sha256(value) for key, value in self.return_action.items()}
        response_shapes = {key: array_summary(value) for key, value in self.return_action.items()}
        info = {
            "schema_version": SCHEMA_VERSION,
            "sentinel_spec_sha256": self.spec_sha256,
            "upload_classification": upload["classification"],
            "upload_pass": upload["pass"],
            "received_observation_keys_order": upload["received_observation_keys_order"],
            "received_shapes_dtypes": upload["received_shapes_dtypes_hashes"],
            "received_array_sha256_values": upload["received_array_sha256_values"],
            "fieldwise_upload_differences": upload["fieldwise_upload_differences"],
            "flattened_62d_difference": upload["flattened_62d_difference"],
            "rgb_comparison": upload["rgb_comparison"],
            "task_comparison": upload["task_comparison"],
            "alternative_failure_hypotheses": upload["alternative_failure_hypotheses"],
            "response_action_keys_order": list(self.return_action.keys()),
            "response_array_shapes_dtypes": response_shapes,
            "response_array_sha256_values": response_hashes,
            "return_sentinel_sha256": self.return_sha256,
            "return_classification": "RETURN_SENTINEL_READY",
        }
        self.append_record(
            "get_action",
            {
                "pass": upload["pass"],
                "classification": upload["classification"],
                "upload_comparison": upload,
                "received_hashes": upload["received_array_sha256_values"],
                "returned_action_hashes": response_hashes,
            },
        )
        return self.return_action, info

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.append_record("reset", {"pass": True, "classification": "RETURN_SENTINEL_READY"})
        return {"status": "reset", "schema_version": SCHEMA_VERSION}

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        self.append_record("get_modality_config", {"pass": True, "classification": "RETURN_SENTINEL_READY"})
        return self.modality_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    policy = TransportSentinelPolicy(args.output_dir)
    print(f"schema_version = {SCHEMA_VERSION}")
    print(f"sentinel_spec_sha256 = {policy.spec_sha256}")
    print(f"output_path = {policy.audit_path}")
    print(f"listening = {args.host}:{args.port}")
    with PolicyServer(policy=policy, host=args.host, port=args.port) as server:
        server.register_endpoint("ping", policy.handle_ping, requires_input=False)
        server.run()


if __name__ == "__main__":
    main()
