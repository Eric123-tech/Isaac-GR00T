#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from retargeting.oakink2_layer.smplx import SMPLXLayer


DEFAULT_OAKINK2_ROOT = Path("datasets/oakink2")
DEFAULT_CAMERA = "104422070969"
DEFAULT_TASK_TARGET_JSON = Path("scripts/oakink2/task_target.json")
DEFAULT_ASSET_ROOT = Path("third_party/maniptrans_assets")
DEFAULT_OUTPUT_ROOT = Path("datasets/oakink2/gr00t_lerobot")
DEFAULT_SERVER_SSH = "david@SERVER_IP"
DEFAULT_SERVER_REPO_DIR = "~/Desktop/haoyu/Isaac-GR00T"
DEFAULT_SERVER_DATA_ROOT = "/mnt/data/haoyu_data/oakink2/gr00t_lerobot"
DEFAULT_SERVER_OUTPUT_ROOT = "/mnt/data/haoyu_data/gr00t_outputs"
MODALITY_CONFIG_PATH = "examples/ARTIMANO/oakink2_artimano_bimanual_config.py"
CHUNKS_SIZE = 1000
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
OAKINK2_KEY_PATTERN = re.compile(r"(scene_[^/]+?)(?:\+\+|/)(seq__[^/]+)")

WRIST_POSE_DIM = 9

ARTIMANO_BODY_NAMES = [
    "palm",
    "index1y",
    "index1z",
    "index2",
    "index3",
    "index_tip",
    "middle1y",
    "middle1z",
    "middle2",
    "middle3",
    "middle_tip",
    "pinky1y",
    "pinky1z",
    "pinky2",
    "pinky3",
    "pinky_tip",
    "ring1y",
    "ring1z",
    "ring2",
    "ring3",
    "ring_tip",
    "thumb1x",
    "thumb1y",
    "thumb1z",
    "thumb2y",
    "thumb2z",
    "thumb3",
    "thumb_tip",
]

ARTIMANO_DOF_NAMES = [
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

HAND2DEX_MAPPING = {
    "wrist": ["palm"],
    "thumb_proximal": ["thumb1x", "thumb1y", "thumb1z"],
    "thumb_intermediate": ["thumb2y", "thumb2z"],
    "thumb_distal": ["thumb3"],
    "thumb_tip": ["thumb_tip"],
    "index_proximal": ["index1y", "index1z"],
    "index_intermediate": ["index2"],
    "index_distal": ["index3"],
    "index_tip": ["index_tip"],
    "middle_proximal": ["middle1y", "middle1z"],
    "middle_intermediate": ["middle2"],
    "middle_distal": ["middle3"],
    "middle_tip": ["middle_tip"],
    "ring_proximal": ["ring1y", "ring1z"],
    "ring_intermediate": ["ring2"],
    "ring_distal": ["ring3"],
    "ring_tip": ["ring_tip"],
    "pinky_proximal": ["pinky1y", "pinky1z"],
    "pinky_intermediate": ["pinky2"],
    "pinky_distal": ["pinky3"],
    "pinky_tip": ["pinky_tip"],
}

DEX2HAND_MAPPING = {
    dex_name: hand_name
    for hand_name, dex_names in HAND2DEX_MAPPING.items()
    for dex_name in dex_names
}


@dataclass
class ResolvedOakInk2Inputs:
    task_key: str
    short_id: str
    rgb_dir: Path
    anno_pkl: Path
    task: str


@dataclass
class JointDef:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_t: torch.Tensor
    origin_r: torch.Tensor
    axis: torch.Tensor
    lower: float
    upper: float


@dataclass
class SideTargets:
    side: str
    wrist_pos: torch.Tensor
    wrist_rotmat: torch.Tensor
    link_targets: torch.Tensor
    link_names: list[str]
    weights: torch.Tensor


@dataclass
class SideRetargetResult:
    side: str
    wrist_pose: np.ndarray
    qpos: np.ndarray
    wrist_pos: np.ndarray
    wrist_rotmat: np.ndarray
    target_link_pos: np.ndarray
    opt_link_pos: np.ndarray
    dof_names: list[str]
    body_names: list[str]
    final_loss: float


def read_pickle(path: Path) -> Any:
    with path.expanduser().open("rb") as f:
        return pickle.load(f)


def load_task_targets(path: Path) -> dict:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"task_target.json not found: {path}")

    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"Expected task_target.json to be a dict, but got {type(data)}")
    return data


def normalize_oakink2_task_key(key: str) -> str:
    """
    Normalize OakInk2 trajectory spellings to task_target.json form:
        scene_.../seq__...
    """
    text = str(key).strip().strip("'\"").replace("\\", "/").rstrip("/")
    if text.endswith(".pkl"):
        text = text[:-4]

    decoded = unquote(text).rstrip("/")
    if decoded.endswith(".pkl"):
        decoded = decoded[:-4]

    match = OAKINK2_KEY_PATTERN.search(decoded)
    if match is None:
        raise ValueError(
            f"Could not normalize OakInk2 trajectory key {key!r}. "
            "Expected scene_.../seq__..., scene_...++seq__..., or scene_...%2B%2Bseq__..."
        )

    scene, seq = match.groups()
    return f"{scene}/{seq}"


def anno_pkl_candidates(anno_root: Path, encoded_key: str) -> list[Path]:
    decoded_key = unquote(encoded_key)
    candidate_keys = []
    for key in [encoded_key, decoded_key, quote(decoded_key, safe="")]:
        if key not in candidate_keys:
            candidate_keys.append(key)
    return [anno_root / f"{key}.pkl" for key in candidate_keys]


def resolve_anno_pkl(anno_root: Path, encoded_key: str) -> Path:
    candidates = anno_pkl_candidates(anno_root, encoded_key)
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def natural_key(path: Path):
    stem = path.stem
    return int(stem) if stem.isdigit() else stem


def load_rgb_files(rgb_dir: Path, max_frames: int | None = None) -> list[Path]:
    files = []
    for ext in IMAGE_EXTS:
        files.extend(rgb_dir.glob(f"*{ext}"))

    files = sorted(files, key=natural_key)
    if max_frames is not None:
        files = files[:max_frames]
    if len(files) < 2:
        raise RuntimeError(f"Need at least 2 images, found {len(files)} in {rgb_dir}")
    return files


def write_video(rgb_files: list[Path], out_path: Path, fps: int, image_size: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (image_size, image_size),
    )

    kept = 0
    for path in rgb_files:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[warn] failed to read image: {path}")
            continue
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        writer.write(image)
        kept += 1

    writer.release()
    if kept != len(rgb_files):
        raise RuntimeError(f"Video wrote {kept} frames, expected {len(rgb_files)}")
    return kept


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_key_short_id(task_key: str) -> str:
    match = re.search(r"seq__([0-9a-fA-F]{5})", task_key)
    if match is not None:
        return match.group(1)
    return re.sub(r"[^A-Za-z0-9]+", "_", task_key).strip("_")[:16]


def default_output_dir(task_key: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{task_key_short_id(task_key)}_artimano_torch_wrist_qpos"


def find_trajectory_dir(oakink2_root: Path, task_key: str) -> Path:
    data_root = oakink2_root.expanduser() / "extracted" / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"OakInk2 RGB data root not found: {data_root}")

    normalized = normalize_oakink2_task_key(task_key)
    for candidate in sorted(p for p in data_root.iterdir() if p.is_dir()):
        try:
            if normalize_oakink2_task_key(candidate.name) == normalized:
                return candidate
        except ValueError:
            continue

    raise FileNotFoundError(
        f"Could not find RGB trajectory dir for {task_key!r} under {data_root}"
    )


def resolve_inputs(args: argparse.Namespace) -> ResolvedOakInk2Inputs:
    oakink2_root = args.oakink2_root.expanduser()
    task_key = normalize_oakink2_task_key(args.task_key)
    traj_dir = find_trajectory_dir(oakink2_root, task_key)
    rgb_dir = traj_dir / args.camera
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Camera RGB dir not found: {rgb_dir}")

    anno_root = oakink2_root / "extracted" / "anno_preview"
    anno_pkl = resolve_anno_pkl(anno_root, traj_dir.name)
    if not anno_pkl.is_file():
        raise FileNotFoundError(f"Annotation pkl not found: {anno_pkl}")

    task_targets = load_task_targets(args.task_target_json)
    task = task_targets.get(task_key)
    if task is None:
        raise KeyError(f"Task key {task_key!r} not found in {args.task_target_json}")

    return ResolvedOakInk2Inputs(
        task_key=task_key,
        short_id=task_key_short_id(task_key),
        rgb_dir=rgb_dir,
        anno_pkl=anno_pkl,
        task=task,
    )


def frame_id_from_rgb(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"RGB file stem is not an integer frame id: {path}") from exc


def dict_get_frame(data: dict, frame_id: int) -> Any:
    if frame_id in data:
        return data[frame_id]
    key = str(frame_id)
    if key in data:
        return data[key]
    raise KeyError(frame_id)


def select_frame_aligned_rgb(
    anno: dict,
    rgb_files: list[Path],
    object_id: str | None,
    require_object: bool,
) -> tuple[list[Path], list[int], str | None]:
    if "raw_smplx" not in anno:
        raise KeyError(f"Annotation has no raw_smplx. Keys: {list(anno.keys())}")

    if object_id is None and require_object:
        obj_list = list(anno.get("obj_list", []))
        if not obj_list:
            raise KeyError("Annotation has no obj_list and --object-id was not set")
        object_id = obj_list[0]

    if object_id is not None:
        if "obj_transf" not in anno or object_id not in anno["obj_transf"]:
            available = list(anno.get("obj_transf", {}).keys())
            raise KeyError(f"object_id {object_id!r} not found. Available: {available}")

    valid_rgb = []
    valid_ids = []
    for path in rgb_files:
        frame_id = frame_id_from_rgb(path)
        try:
            dict_get_frame(anno["raw_smplx"], frame_id)
            if object_id is not None:
                dict_get_frame(anno["obj_transf"][object_id], frame_id)
        except KeyError:
            continue
        valid_rgb.append(path)
        valid_ids.append(frame_id)

    if len(valid_ids) < 2:
        raise RuntimeError(f"Need at least 2 aligned frames, got {len(valid_ids)}")
    return valid_rgb, valid_ids, object_id


def parse_vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float32)
    return np.asarray([float(x) for x in text.split()], dtype=np.float32)


def rpy_to_matrix_np(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def fixed_axis_angle_to_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    axis = F.normalize(axis, dim=0)
    x, y, z = axis
    zero = torch.zeros((), dtype=theta.dtype, device=theta.device)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, dtype=theta.dtype, device=theta.device)
    sin = torch.sin(theta).view(-1, 1, 1)
    cos = torch.cos(theta).view(-1, 1, 1)
    return eye.unsqueeze(0) + sin * k.unsqueeze(0) + (1.0 - cos) * (k @ k).unsqueeze(0)


def make_transform(origin_t: torch.Tensor, origin_r: torch.Tensor, batch_size: int) -> torch.Tensor:
    transform = torch.eye(4, dtype=origin_t.dtype, device=origin_t.device).repeat(batch_size, 1, 1)
    transform[:, :3, :3] = origin_r
    transform[:, :3, 3] = origin_t
    return transform


class TorchUrdfChain:
    def __init__(self, urdf_path: Path, device: str):
        self.urdf_path = urdf_path.expanduser()
        self.device = torch.device(device)
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        self.joints = self._parse_joints(self.urdf_path)
        self.ordered_joints = self._topological_order(self.joints)
        self.dof_names = [joint.name for joint in self.ordered_joints if joint.joint_type != "fixed"]
        missing = [name for name in ARTIMANO_DOF_NAMES if name not in self.dof_names]
        if missing:
            raise RuntimeError(f"URDF is missing expected Artimano DOFs: {missing}")
        self.dof_index = {name: i for i, name in enumerate(ARTIMANO_DOF_NAMES)}

        lower = []
        upper = []
        joint_by_name = {joint.name: joint for joint in self.ordered_joints}
        for name in ARTIMANO_DOF_NAMES:
            lower.append(joint_by_name[name].lower)
            upper.append(joint_by_name[name].upper)
        self.lower = torch.tensor(lower, dtype=torch.float32, device=self.device)
        self.upper = torch.tensor(upper, dtype=torch.float32, device=self.device)

    def _parse_joints(self, urdf_path: Path) -> list[JointDef]:
        root = ET.parse(urdf_path).getroot()
        joints: list[JointDef] = []
        for elem in root.findall("joint"):
            name = elem.attrib["name"]
            joint_type = elem.attrib.get("type", "fixed")
            parent = elem.find("parent").attrib["link"]
            child = elem.find("child").attrib["link"]

            origin = elem.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))

            axis_elem = elem.find("axis")
            axis = parse_vec(axis_elem.attrib.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0))

            limit_elem = elem.find("limit")
            lower = float(limit_elem.attrib.get("lower", "0.0")) if limit_elem is not None else 0.0
            upper = float(limit_elem.attrib.get("upper", "0.0")) if limit_elem is not None else 0.0

            joints.append(
                JointDef(
                    name=name,
                    joint_type=joint_type,
                    parent=parent,
                    child=child,
                    origin_t=torch.tensor(xyz, dtype=torch.float32, device=self.device),
                    origin_r=torch.tensor(rpy_to_matrix_np(rpy), dtype=torch.float32, device=self.device),
                    axis=torch.tensor(axis, dtype=torch.float32, device=self.device),
                    lower=lower,
                    upper=upper,
                )
            )
        return joints

    @staticmethod
    def _topological_order(joints: list[JointDef]) -> list[JointDef]:
        known = {"palm"}
        remaining = list(joints)
        ordered: list[JointDef] = []
        while remaining:
            progressed = False
            for joint in list(remaining):
                if joint.parent in known:
                    ordered.append(joint)
                    known.add(joint.child)
                    remaining.remove(joint)
                    progressed = True
            if not progressed:
                unresolved = [(joint.parent, joint.child) for joint in remaining]
                raise RuntimeError(f"Could not topologically sort URDF joints: {unresolved}")
        return ordered

    def forward(self, qpos: torch.Tensor, link_names: list[str]) -> torch.Tensor:
        batch_size = int(qpos.shape[0])
        transforms: dict[str, torch.Tensor] = {
            "palm": torch.eye(4, dtype=qpos.dtype, device=qpos.device).repeat(batch_size, 1, 1)
        }

        for joint in self.ordered_joints:
            parent_tf = transforms[joint.parent]
            origin_tf = make_transform(joint.origin_t.to(qpos.device), joint.origin_r.to(qpos.device), batch_size)

            if joint.joint_type == "fixed":
                motion_tf = torch.eye(4, dtype=qpos.dtype, device=qpos.device).repeat(batch_size, 1, 1)
            elif joint.joint_type in {"revolute", "continuous"}:
                theta = qpos[:, self.dof_index[joint.name]]
                motion_tf = torch.eye(4, dtype=qpos.dtype, device=qpos.device).repeat(batch_size, 1, 1)
                motion_tf[:, :3, :3] = fixed_axis_angle_to_matrix(joint.axis.to(qpos.device), theta)
            else:
                raise NotImplementedError(f"Unsupported joint type {joint.joint_type!r} in {joint.name}")

            transforms[joint.child] = parent_tf @ origin_tf @ motion_tf

        missing = [name for name in link_names if name not in transforms]
        if missing:
            raise RuntimeError(f"URDF forward pass did not produce links: {missing}")
        return torch.stack([transforms[name][:, :3, 3] for name in link_names], dim=1)


def rotation_6d_to_matrix_rows(rot6d: torch.Tensor) -> torch.Tensor:
    row1 = F.normalize(rot6d[..., 0:3], dim=-1)
    row2 = rot6d[..., 3:6]
    row2 = row2 - (row1 * row2).sum(dim=-1, keepdim=True) * row1
    row2 = F.normalize(row2, dim=-1)
    row3 = torch.cross(row1, row2, dim=-1)
    return torch.stack([row1, row2, row3], dim=-2)


def matrix_to_rotation_6d_rows(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def matrix_to_xyz_rot6d_np(pos: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    rot6d = rotmat[:, :2, :].reshape(rotmat.shape[0], 6)
    return np.concatenate([pos, rot6d], axis=1).astype(np.float32)


def default_qpos(chain: TorchUrdfChain) -> torch.Tensor:
    qpos = torch.ones(len(ARTIMANO_DOF_NAMES), dtype=torch.float32, device=chain.lower.device) * (math.pi / 50.0)
    if qpos.numel() > 9:
        qpos[8] = 0.8
        qpos[9] = 0.05
    return torch.clamp(qpos, chain.lower + 1e-4, chain.upper - 1e-4)


def qpos_to_raw(qpos: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    ratio = (qpos - lower) / (upper - lower)
    ratio = torch.clamp(ratio, 1e-4, 1.0 - 1e-4)
    return torch.log(ratio / (1.0 - ratio))


def raw_to_qpos(raw: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return lower + torch.sigmoid(raw) * (upper - lower)


def aa_to_rotmat_np(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)).astype(
        np.float32
    )


def maniptrans_world_transform(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    rot = aa_to_rotmat_np(np.array([0.0, 0.0, -math.pi / 2.0], dtype=np.float32)) @ aa_to_rotmat_np(
        np.array([math.pi / 2.0, 0.0, 0.0], dtype=np.float32)
    )
    trans = np.array([0.0, 0.0, 0.415], dtype=np.float32)
    return (
        torch.tensor(rot, dtype=torch.float32, device=device),
        torch.tensor(trans, dtype=torch.float32, device=device),
    )


def transform_points(points: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    return points @ rot.T + trans


def transform_rotmats(rotmats: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    return rot.unsqueeze(0) @ rotmats


def build_smplx_batch(anno: dict, frame_ids: list[int], device: str) -> dict[str, torch.Tensor]:
    raw_smplx = anno["raw_smplx"]
    frames = [dict_get_frame(raw_smplx, frame_id) for frame_id in frame_ids]

    smplx_data: dict[str, list[torch.Tensor]] = {key: [] for key in frames[0].keys()}
    for frame in frames:
        for key, value in frame.items():
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            smplx_data[key].append(tensor.to(device=device, dtype=torch.float32))

    return {key: torch.cat(values, dim=0) for key, values in smplx_data.items()}


def create_smplx_layer(asset_root: Path, device: str) -> SMPLXLayer:
    model_path = asset_root.expanduser() / "smplx" / "SMPLX_NEUTRAL.npz"
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL-X model not found: {model_path}")
    layer = SMPLXLayer(
        str(model_path),
        dtype=torch.float32,
        rot_mode="quat",
        num_betas=300,
        gender="neutral",
    )
    return layer.to(device)


def extract_smplx_outputs(asset_root: Path, anno: dict, frame_ids: list[int], device: str):
    smplx_layer = create_smplx_layer(asset_root, device)
    smplx_data = build_smplx_batch(anno, frame_ids, device)
    with torch.no_grad():
        return smplx_layer(**smplx_data)


def side_mano_joints(smplx_results, side: str) -> dict[str, torch.Tensor]:
    if side == "right":
        return {
            "index_proximal": smplx_results.joints[:, 40].detach(),
            "index_intermediate": smplx_results.joints[:, 41].detach(),
            "index_distal": smplx_results.joints[:, 42].detach(),
            "index_tip": smplx_results.vertices[:, 7706].detach(),
            "middle_proximal": smplx_results.joints[:, 43].detach(),
            "middle_intermediate": smplx_results.joints[:, 44].detach(),
            "middle_distal": smplx_results.joints[:, 45].detach(),
            "middle_tip": smplx_results.vertices[:, 7818].detach(),
            "pinky_proximal": smplx_results.joints[:, 46].detach(),
            "pinky_intermediate": smplx_results.joints[:, 47].detach(),
            "pinky_distal": smplx_results.joints[:, 48].detach(),
            "pinky_tip": smplx_results.vertices[:, 8046].detach(),
            "ring_proximal": smplx_results.joints[:, 49].detach(),
            "ring_intermediate": smplx_results.joints[:, 50].detach(),
            "ring_distal": smplx_results.joints[:, 51].detach(),
            "ring_tip": smplx_results.vertices[:, 7929].detach(),
            "thumb_proximal": smplx_results.joints[:, 52].detach(),
            "thumb_intermediate": smplx_results.joints[:, 53].detach(),
            "thumb_distal": smplx_results.joints[:, 54].detach(),
            "thumb_tip": smplx_results.vertices[:, 8096].detach(),
        }

    if side == "left":
        return {
            "index_proximal": smplx_results.joints[:, 25].detach(),
            "index_intermediate": smplx_results.joints[:, 26].detach(),
            "index_distal": smplx_results.joints[:, 27].detach(),
            "index_tip": smplx_results.vertices[:, 4970].detach(),
            "middle_proximal": smplx_results.joints[:, 28].detach(),
            "middle_intermediate": smplx_results.joints[:, 29].detach(),
            "middle_distal": smplx_results.joints[:, 30].detach(),
            "middle_tip": smplx_results.vertices[:, 5082].detach(),
            "pinky_proximal": smplx_results.joints[:, 31].detach(),
            "pinky_intermediate": smplx_results.joints[:, 32].detach(),
            "pinky_distal": smplx_results.joints[:, 33].detach(),
            "pinky_tip": smplx_results.vertices[:, 5310].detach(),
            "ring_proximal": smplx_results.joints[:, 34].detach(),
            "ring_intermediate": smplx_results.joints[:, 35].detach(),
            "ring_distal": smplx_results.joints[:, 36].detach(),
            "ring_tip": smplx_results.vertices[:, 5193].detach(),
            "thumb_proximal": smplx_results.joints[:, 37].detach(),
            "thumb_intermediate": smplx_results.joints[:, 38].detach(),
            "thumb_distal": smplx_results.joints[:, 39].detach(),
            "thumb_tip": smplx_results.vertices[:, 5362].detach(),
        }

    raise ValueError(f"Unknown side: {side}")


def side_wrist_index(side: str) -> int:
    if side == "right":
        return 21
    if side == "left":
        return 20
    raise ValueError(f"Unknown side: {side}")


def link_weight(link_name: str) -> float:
    hand_name = DEX2HAND_MAPPING[link_name]
    if "tip" in hand_name:
        if "thumb" in hand_name:
            return 25.0
        if "index" in hand_name:
            return 20.0
        if "middle" in hand_name:
            return 10.0
        if "ring" in hand_name:
            return 7.0
        if "pinky" in hand_name:
            return 5.0
    return 1.0


def object_positions_for_frames(anno: dict, object_id: str | None, frame_ids: list[int], device: str) -> torch.Tensor | None:
    if object_id is None:
        return None
    obj_map = anno["obj_transf"][object_id]
    mats = [np.asarray(dict_get_frame(obj_map, frame_id), dtype=np.float32) for frame_id in frame_ids]
    return torch.tensor(np.stack(mats)[:, :3, 3], dtype=torch.float32, device=device)


def build_side_targets(
    smplx_results,
    side: str,
    device: str,
    apply_maniptrans_frame: bool,
    object_positions: torch.Tensor | None,
    object_safety_offset_m: float,
) -> SideTargets:
    mano_joints = side_mano_joints(smplx_results, side)
    wrist_index = side_wrist_index(side)

    wrist_pos = smplx_results.joints[:, wrist_index].detach()
    middle_pos = mano_joints["middle_proximal"]
    wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25
    wrist_rotmat = smplx_results.transform_abs[:, wrist_index, :3, :3].detach()
    offset_reference_pos = middle_pos.to(device=device, dtype=torch.float32)

    link_targets = []
    weights = []
    for link_name in ARTIMANO_BODY_NAMES:
        hand_name = DEX2HAND_MAPPING[link_name]
        if hand_name == "wrist":
            link_targets.append(wrist_pos)
        else:
            link_targets.append(mano_joints[hand_name])
        weights.append(link_weight(link_name))
    link_targets_tensor = torch.stack(link_targets, dim=1).to(device=device, dtype=torch.float32)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    wrist_pos = wrist_pos.to(device=device, dtype=torch.float32)
    wrist_rotmat = wrist_rotmat.to(device=device, dtype=torch.float32)

    if apply_maniptrans_frame:
        frame_rot, frame_trans = maniptrans_world_transform(device)
        wrist_pos = transform_points(wrist_pos, frame_rot, frame_trans)
        wrist_rotmat = transform_rotmats(wrist_rotmat, frame_rot)
        link_targets_tensor = transform_points(link_targets_tensor, frame_rot, frame_trans)
        offset_reference_pos = transform_points(offset_reference_pos, frame_rot, frame_trans)
        if object_positions is not None:
            object_positions = transform_points(object_positions, frame_rot, frame_trans)

    if object_positions is not None and object_safety_offset_m > 0.0:
        direction = offset_reference_pos - object_positions
        direction = F.normalize(direction, dim=-1)
        offset = direction * float(object_safety_offset_m)
        wrist_pos = wrist_pos + offset
        link_targets_tensor = link_targets_tensor + offset[:, None, :]

    return SideTargets(
        side=side,
        wrist_pos=wrist_pos,
        wrist_rotmat=wrist_rotmat,
        link_targets=link_targets_tensor,
        link_names=list(ARTIMANO_BODY_NAMES),
        weights=weights_tensor,
    )


def urdf_for_side(asset_root: Path, side: str) -> Path:
    if side == "left":
        return asset_root.expanduser() / "mano_urdf" / "lh_mano.urdf"
    if side == "right":
        return asset_root.expanduser() / "mano_urdf" / "rh_mano.urdf"
    raise ValueError(f"Unknown side: {side}")


def fit_artimano_side(
    targets: SideTargets,
    urdf_path: Path,
    args: argparse.Namespace,
) -> SideRetargetResult:
    chain = TorchUrdfChain(urdf_path, args.device)
    num_frames = int(targets.wrist_pos.shape[0])

    q_init = default_qpos(chain).unsqueeze(0).repeat(num_frames, 1)
    q_raw = qpos_to_raw(q_init, chain.lower, chain.upper).clone().detach().requires_grad_(True)

    target_wrist_rot6d = matrix_to_rotation_6d_rows(targets.wrist_rotmat)
    if args.fix_wrist:
        wrist_pos_param = targets.wrist_pos
        wrist_rot6d_param = target_wrist_rot6d
        params = [{"params": [q_raw], "lr": args.lr_qpos}]
    else:
        wrist_pos_param = targets.wrist_pos.clone().detach().requires_grad_(True)
        wrist_rot6d_param = target_wrist_rot6d.clone().detach().requires_grad_(True)
        params = [
            {"params": [q_raw], "lr": args.lr_qpos},
            {"params": [wrist_pos_param, wrist_rot6d_param], "lr": args.lr_wrist},
        ]

    optimizer = torch.optim.Adam(params)
    previous_loss = None
    final_loss = math.inf

    for step in range(1, args.ik_iters + 1):
        qpos = raw_to_qpos(q_raw, chain.lower, chain.upper)
        local_link_pos = chain.forward(qpos, ARTIMANO_BODY_NAMES)
        wrist_rotmat = rotation_6d_to_matrix_rows(wrist_rot6d_param)
        pred_link_pos = (wrist_rotmat @ local_link_pos.transpose(-1, -2)).transpose(-1, -2)
        pred_link_pos = pred_link_pos + wrist_pos_param[:, None, :]

        dist = torch.linalg.norm(pred_link_pos - targets.link_targets, dim=-1)
        fit_loss = (dist * targets.weights[None]).mean()

        loss = fit_loss
        if not args.fix_wrist and args.wrist_anchor_weight > 0.0:
            wrist_pos_loss = torch.linalg.norm(wrist_pos_param - targets.wrist_pos, dim=-1).mean()
            wrist_rot_loss = ((wrist_rotmat - targets.wrist_rotmat) ** 2).mean()
            loss = loss + args.wrist_anchor_weight * (wrist_pos_loss + wrist_rot_loss)

        if args.smooth_weight > 0.0 and num_frames > 1:
            q_smooth = ((qpos[1:] - qpos[:-1]) ** 2).mean()
            wrist_smooth = ((wrist_pos_param[1:] - wrist_pos_param[:-1]) ** 2).mean()
            loss = loss + args.smooth_weight * (q_smooth + wrist_smooth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu().item())
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == args.ik_iters):
            print(
                f"[ik:{targets.side}] iter={step:04d} "
                f"loss={final_loss:.6f} fit={float(fit_loss.detach().cpu().item()):.6f}"
            )

        if args.early_stop_delta > 0.0 and args.log_every > 0 and step % args.log_every == 0:
            if previous_loss is not None and abs(previous_loss - final_loss) < args.early_stop_delta:
                print(f"[ik:{targets.side}] early stop at iter={step}, delta={abs(previous_loss - final_loss):.6g}")
                break
            previous_loss = final_loss

    with torch.no_grad():
        qpos = raw_to_qpos(q_raw, chain.lower, chain.upper)
        local_link_pos = chain.forward(qpos, ARTIMANO_BODY_NAMES)
        wrist_rotmat = rotation_6d_to_matrix_rows(wrist_rot6d_param)
        pred_link_pos = (wrist_rotmat @ local_link_pos.transpose(-1, -2)).transpose(-1, -2)
        pred_link_pos = pred_link_pos + wrist_pos_param[:, None, :]

    wrist_pos_np = wrist_pos_param.detach().cpu().numpy().astype(np.float32)
    wrist_rot_np = wrist_rotmat.detach().cpu().numpy().astype(np.float32)
    return SideRetargetResult(
        side=targets.side,
        wrist_pose=matrix_to_xyz_rot6d_np(wrist_pos_np, wrist_rot_np),
        qpos=qpos.detach().cpu().numpy().astype(np.float32),
        wrist_pos=wrist_pos_np,
        wrist_rotmat=wrist_rot_np,
        target_link_pos=targets.link_targets.detach().cpu().numpy().astype(np.float32),
        opt_link_pos=pred_link_pos.detach().cpu().numpy().astype(np.float32),
        dof_names=list(ARTIMANO_DOF_NAMES),
        body_names=list(ARTIMANO_BODY_NAMES),
        final_loss=final_loss,
    )


def build_state_action_arrays(
    left: SideRetargetResult,
    right: SideRetargetResult,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    lengths = {left.wrist_pose.shape[0], right.wrist_pose.shape[0], left.qpos.shape[0], right.qpos.shape[0]}
    if len(lengths) != 1:
        raise RuntimeError(
            "Retargeted left/right wrist/qpos lengths are inconsistent: "
            f"left_wrist={left.wrist_pose.shape[0]} right_wrist={right.wrist_pose.shape[0]} "
            f"left_qpos={left.qpos.shape[0]} right_qpos={right.qpos.shape[0]}"
        )

    start = 0
    slices: dict[str, tuple[int, int]] = {}
    parts = []
    for name, array in [
        ("left_wrist_pose", left.wrist_pose),
        ("right_wrist_pose", right.wrist_pose),
        ("left_hand_qpos", left.qpos),
        ("right_hand_qpos", right.qpos),
    ]:
        end = start + int(array.shape[1])
        slices[name] = (start, end)
        parts.append(array.astype(np.float32))
        start = end

    return np.concatenate(parts, axis=1).astype(np.float32), slices


def write_episode_parquet(out_path: Path, props: np.ndarray, fps: int, task_index: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_rows = int(props.shape[0] - 1)
    rows = []
    for t in range(num_rows):
        rows.append(
            {
                "observation.state": props[t].astype("float32").tolist(),
                "action": props[t + 1].astype("float32").tolist(),
                "timestamp": float(t / fps),
                "frame_index": int(t),
                "episode_index": 0,
                "index": int(t),
                "task_index": int(task_index),
                "annotation.human.action.task_description": int(task_index),
                "next.reward": 0.0,
                "next.done": bool(t == num_rows - 1),
            }
        )

    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return num_rows


def write_meta(
    output_dir: Path,
    task: str,
    total_frames: int,
    fps: int,
    image_size: int,
    prop_dim: int,
    slices: dict[str, tuple[int, int]],
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": task}])
    write_jsonl(
        meta_dir / "episodes.jsonl",
        [{"episode_index": 0, "tasks": [task], "length": int(total_frames)}],
    )

    info = {
        "codebase_version": "v2.0",
        "robot_type": "oakink2_artimano_torch_bimanual_wrist_qpos",
        "total_episodes": 1,
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": int(fps),
        "video": True,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [int(prop_dim)],
                "names": list(slices.keys()),
            },
            "action": {
                "dtype": "float32",
                "shape": [int(prop_dim)],
                "names": list(slices.keys()),
            },
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [int(image_size), int(image_size), 3],
                "names": ["height", "width", "channel"],
            },
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [1],
                "names": ["task_index"],
            },
        },
    }

    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    modality = {
        "state": {key: {"start": int(start), "end": int(end)} for key, (start, end) in slices.items()},
        "action": {key: {"start": int(start), "end": int(end)} for key, (start, end) in slices.items()},
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.action.task_description": {}},
    }

    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")


def write_retarget_artifacts(
    output_dir: Path,
    frame_ids: list[int],
    object_id: str | None,
    left: SideRetargetResult,
    right: SideRetargetResult,
) -> Path:
    retarget_dir = output_dir / "retarget"
    retarget_dir.mkdir(parents=True, exist_ok=True)
    out_path = retarget_dir / "artimano_torch_bimanual_retarget.npz"
    np.savez_compressed(
        out_path,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        object_id=np.asarray("" if object_id is None else object_id),
        left_wrist_pose=left.wrist_pose,
        right_wrist_pose=right.wrist_pose,
        left_hand_qpos=left.qpos,
        right_hand_qpos=right.qpos,
        left_wrist_pos=left.wrist_pos,
        right_wrist_pos=right.wrist_pos,
        left_wrist_rotmat=left.wrist_rotmat,
        right_wrist_rotmat=right.wrist_rotmat,
        left_target_link_pos=left.target_link_pos,
        right_target_link_pos=right.target_link_pos,
        left_opt_link_pos=left.opt_link_pos,
        right_opt_link_pos=right.opt_link_pos,
        left_dof_names=np.asarray(left.dof_names),
        right_dof_names=np.asarray(right.dof_names),
        left_body_names=np.asarray(left.body_names),
        right_body_names=np.asarray(right.body_names),
        left_final_loss=np.asarray(left.final_loss, dtype=np.float32),
        right_final_loss=np.asarray(right.final_loss, dtype=np.float32),
    )
    return out_path


def print_next_steps(
    args: argparse.Namespace,
    output_dir: Path,
    inputs: ResolvedOakInk2Inputs,
    report: dict,
) -> None:
    if args.retarget_only:
        print("[next steps]")
        print("  --retarget-only was set, so only the Artimano IK artifact was written.")
        print(f"  retarget_npz: {report['retarget_npz']}")
        print("  To write a GR00T/LeRobot dataset, rerun this script without --retarget-only.")
        return

    dataset_name = output_dir.name
    server_dataset_path = f"{args.server_data_root.rstrip('/')}/{dataset_name}"
    train_output_name = args.train_output_name or f"oakink2_{inputs.short_id}_artimano_torch_projector_diffusion_2gpu"
    server_output_dir = f"{args.server_output_root.rstrip('/')}/{train_output_name}"

    print("\n[next steps]")
    print("# Note")
    print("# Do not run scripts/oakink2/run_oakink2_to_gr00t_pipeline.py for this Artimano flow.")
    print("# That older pipeline writes MANO prop vectors; this script already did:")
    print("# extracted OakInk2 -> SMPL-X/MANO FK -> Artimano IK -> GR00T/LeRobot dataset.")

    print("\n# 1) Validate the generated LeRobot dataset locally")
    print("cd ~/linux_projects/Isaac-GR00T")
    print("uv run python scripts/oakink2/validate_gr00t_lerobot_dataset.py \\")
    print(f"  --dataset_dir {output_dir}")

    print("\n# 2) Upload generated LeRobot dataset")
    print("cd ~/linux_projects/Isaac-GR00T")
    print("rsync -avhP --partial \\")
    print(f"  {output_dir}/ \\")
    print(f"  {args.server_ssh}:{server_dataset_path}/")

    print("\n# 3) Upload the GR00T modality config and retargeting code")
    print("rsync -avhP --relative \\")
    print(f"  {MODALITY_CONFIG_PATH} \\")
    print("  scripts/oakink2/retarget_oakink2_artimano_torch_to_gr00t.py \\")
    print("  scripts/oakink2/retargeting/ \\")
    print(f"  {args.server_ssh}:{args.server_repo_dir}/")

    print("\n# 4) On the server, finetune GR00T")
    print(f"cd {args.server_repo_dir}")
    print(f"rm -rf {server_output_dir}")
    print("CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\")
    print("uv run torchrun --nproc_per_node=2 --master_port=29500 \\")
    print("  gr00t/experiment/launch_finetune.py \\")
    print("  --base-model-path nvidia/GR00T-N1.7-3B \\")
    print(f"  --dataset-path {server_dataset_path} \\")
    print("  --embodiment-tag NEW_EMBODIMENT \\")
    print(f"  --modality-config-path {MODALITY_CONFIG_PATH} \\")
    print("  --num-gpus 2 \\")
    print(f"  --output-dir {server_output_dir} \\")
    print("  --max-steps 2000 \\")
    print("  --save-steps 100 \\")
    print("  --global-batch-size 2 \\")
    print("  --gradient-accumulation-steps 1 \\")
    print("  --dataloader-num-workers 0 \\")
    print("  --no-use-wandb \\")
    print("  --no-tune-llm \\")
    print("  --no-tune-visual \\")
    print("  --tune-projector \\")
    print("  --tune-diffusion-model")


def convert_one(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested --device {args.device}, but torch.cuda.is_available() is False")

    inputs = resolve_inputs(args)
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else default_output_dir(inputs.task_key)
    asset_root = args.asset_root.expanduser()

    require_object = args.object_safety_offset_m > 0.0 or args.object_id is not None
    rgb_files = load_rgb_files(inputs.rgb_dir, max_frames=args.max_frames)
    anno = read_pickle(inputs.anno_pkl)
    valid_rgb_files, frame_ids, object_id = select_frame_aligned_rgb(
        anno=anno,
        rgb_files=rgb_files,
        object_id=args.object_id,
        require_object=require_object,
    )

    print("[inputs]")
    print(f"  task_key:       {inputs.task_key}")
    print(f"  rgb_dir:        {inputs.rgb_dir}")
    print(f"  anno_pkl:       {inputs.anno_pkl}")
    print(f"  task:           {inputs.task}")
    print(f"  object_id:      {object_id}")
    print(f"  matched frames: {len(frame_ids)}")
    print(f"  asset_root:     {asset_root}")
    print(f"  output_dir:     {output_dir}")

    for path in [
        asset_root / "smplx" / "SMPLX_NEUTRAL.npz",
        asset_root / "mano_urdf" / "lh_mano.urdf",
        asset_root / "mano_urdf" / "rh_mano.urdf",
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Required asset not found: {path}")

    if args.dry_run:
        print("[dry-run] stopping before SMPL-X FK / Artimano IK")
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir already exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[smplx] running OakInk2 SMPL-X FK")
    smplx_results = extract_smplx_outputs(asset_root, anno, frame_ids, args.device)
    object_positions = object_positions_for_frames(anno, object_id, frame_ids, args.device)

    print("[targets] building left/right MANO keypoint targets")
    left_targets = build_side_targets(
        smplx_results=smplx_results,
        side="left",
        device=args.device,
        apply_maniptrans_frame=args.apply_maniptrans_frame,
        object_positions=object_positions,
        object_safety_offset_m=args.object_safety_offset_m,
    )
    right_targets = build_side_targets(
        smplx_results=smplx_results,
        side="right",
        device=args.device,
        apply_maniptrans_frame=args.apply_maniptrans_frame,
        object_positions=object_positions,
        object_safety_offset_m=args.object_safety_offset_m,
    )

    print("[ik] left Artimano")
    left = fit_artimano_side(left_targets, urdf_for_side(asset_root, "left"), args)
    print("[ik] right Artimano")
    right = fit_artimano_side(right_targets, urdf_for_side(asset_root, "right"), args)

    retarget_npz = write_retarget_artifacts(
        output_dir=output_dir,
        frame_ids=frame_ids,
        object_id=object_id,
        left=left,
        right=right,
    )

    props, slices = build_state_action_arrays(left, right)
    prop_dim = int(props.shape[1])
    num_rows = int(props.shape[0] - 1)

    video_path = output_dir / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4"
    parquet_path = output_dir / "data" / "chunk-000" / "episode_000000.parquet"

    if not args.retarget_only:
        print("[lerobot] writing video/parquet/meta")
        num_video_frames = write_video(
            rgb_files=valid_rgb_files[:-1],
            out_path=video_path,
            fps=args.fps,
            image_size=args.image_size,
        )
        written_rows = write_episode_parquet(
            out_path=parquet_path,
            props=props,
            fps=args.fps,
            task_index=0,
        )
        if num_video_frames != written_rows:
            raise RuntimeError(f"video frames {num_video_frames} != parquet rows {written_rows}")
        write_meta(
            output_dir=output_dir,
            task=inputs.task,
            total_frames=written_rows,
            fps=args.fps,
            image_size=args.image_size,
            prop_dim=prop_dim,
            slices=slices,
        )
        num_rows = written_rows

    report = {
        "task_key": inputs.task_key,
        "short_id": inputs.short_id,
        "rgb_dir": str(inputs.rgb_dir),
        "anno_pkl": str(inputs.anno_pkl),
        "object_id": object_id,
        "task": inputs.task,
        "device": args.device,
        "fps": args.fps,
        "image_size": args.image_size,
        "max_frames": args.max_frames,
        "ik_iters": args.ik_iters,
        "fix_wrist": args.fix_wrist,
        "wrist_anchor_weight": args.wrist_anchor_weight,
        "smooth_weight": args.smooth_weight,
        "apply_maniptrans_frame": args.apply_maniptrans_frame,
        "object_safety_offset_m": args.object_safety_offset_m,
        "num_matched_frames": len(frame_ids),
        "num_rows": num_rows,
        "prop_dim": prop_dim,
        "slices": {key: [int(v[0]), int(v[1])] for key, v in slices.items()},
        "left_final_loss": left.final_loss,
        "right_final_loss": right.final_loss,
        "dof_names": ARTIMANO_DOF_NAMES,
        "body_names": ARTIMANO_BODY_NAMES,
        "retarget_npz": str(retarget_npz),
        "video": str(video_path) if not args.retarget_only else None,
        "parquet": str(parquet_path) if not args.retarget_only else None,
        "modality_config_path": MODALITY_CONFIG_PATH,
    }
    report_path = output_dir / "meta" / "conversion_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[done]")
    print(f"  output_dir:       {output_dir}")
    print(f"  retarget_npz:     {retarget_npz}")
    print(f"  rows:             {num_rows}")
    print(f"  prop_dim:         {prop_dim}")
    print(f"  slices:           {report['slices']}")
    print(f"  left_final_loss:  {left.final_loss:.6f}")
    print(f"  right_final_loss: {right.final_loss:.6f}")
    print(f"  report:           {report_path}")
    if args.print_next_steps:
        print_next_steps(args=args, output_dir=output_dir, inputs=inputs, report=report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pure PyTorch offline OakInk2 MANO/SMPL-X -> Artimano wrist+qpos retargeting, "
            "with optional GR00T/LeRobot dataset export."
        )
    )
    parser.add_argument("--oakink2-root", type=Path, default=DEFAULT_OAKINK2_ROOT)
    parser.add_argument("--task-target-json", type=Path, default=DEFAULT_TASK_TARGET_JSON)
    parser.add_argument("--task-key", type=str, required=True)
    parser.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--object-id", type=str, default=None)

    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=300)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ik-iters", type=int, default=2000)
    parser.add_argument("--lr-qpos", type=float, default=4e-4)
    parser.add_argument("--lr-wrist", type=float, default=8e-4)
    parser.add_argument("--smooth-weight", type=float, default=1e-3)
    parser.add_argument("--wrist-anchor-weight", type=float, default=2.0)
    parser.add_argument("--early-stop-delta", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--fix-wrist", action="store_true")
    parser.add_argument("--apply-maniptrans-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--object-safety-offset-m", type=float, default=0.0)

    parser.add_argument("--retarget-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-next-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--server-ssh", type=str, default=DEFAULT_SERVER_SSH)
    parser.add_argument("--server-repo-dir", type=str, default=DEFAULT_SERVER_REPO_DIR)
    parser.add_argument("--server-data-root", type=str, default=DEFAULT_SERVER_DATA_ROOT)
    parser.add_argument("--server-output-root", type=str, default=DEFAULT_SERVER_OUTPUT_ROOT)
    parser.add_argument("--train-output-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_one(args)


if __name__ == "__main__":
    main()
