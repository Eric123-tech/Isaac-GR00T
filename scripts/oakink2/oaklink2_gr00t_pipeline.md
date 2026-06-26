# OakInk2 to Artimano GR00T Pipeline

This is the current OakInk2 data path used for Artimano GR00T finetuning:

```text
OakInk2 extracted RGB + anno_preview raw_smplx
    -> SMPL-X/MANO FK
    -> Artimano wrist pose + hand qpos IK
    -> GR00T / LeRobot v2 dataset
    -> local dataset validation
    -> server upload
    -> GR00T finetune
```

Do not use the old MANO prop-vector pipeline for Artimano training. The old
`full128/full134` data represented MANO quaternion pose, not Artimano qpos.

## Current Scripts

Keep these files:

```text
scripts/oakink2/retarget_oakink2_artimano_torch_to_gr00t.py
scripts/oakink2/retargeting/
scripts/oakink2/task_target.json
scripts/oakink2/validate_gr00t_lerobot_dataset.py
examples/ARTIMANO/oakink2_artimano_bimanual_config.py
third_party/maniptrans_assets/README.md
third_party/maniptrans_assets/.gitignore
```

The retargeting assets under `third_party/maniptrans_assets/` are local-only by
default. Do not commit SMPL-X model files or mesh assets to GitHub.

## Expected Local Layout

```text
~/linux_projects/Isaac-GR00T
~/linux_projects/Isaac-GR00T/datasets/oakink2 -> ~/datasets/oakink2
```

Expected OakInk2 extracted layout:

```text
datasets/oakink2/extracted/data/
datasets/oakink2/extracted/anno_preview/
```

Expected local retargeting assets:

```text
third_party/maniptrans_assets/smplx/SMPLX_NEUTRAL.npz
third_party/maniptrans_assets/mano_urdf/lh_mano.urdf
third_party/maniptrans_assets/mano_urdf/rh_mano.urdf
```

## Convert One Trajectory

Example:

```bash
uv run python scripts/oakink2/retarget_oakink2_artimano_torch_to_gr00t.py \
  --oakink2-root datasets/oakink2 \
  --task-target-json scripts/oakink2/task_target.json \
  --camera 104422070969 \
  --task-key scene_01__A005%2B%2Bseq__fccd810e2fd673b25c86__2023-04-15-14-30-47 \
  --fps 10 \
  --image-size 256 \
  --max-frames 300 \
  --device cuda:0 \
  --ik-iters 6000 \
  --overwrite
```

Default output:

```text
datasets/oakink2/gr00t_lerobot/fccd8_artimano_torch_wrist_qpos
```

The script writes both:

```text
retarget/artimano_torch_bimanual_retarget.npz
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.ego_view/episode_000000.mp4
meta/
```

State/action layout:

```text
left_wrist_pose:  [0, 9]
right_wrist_pose: [9, 18]
left_hand_qpos:   [18, 40]
right_hand_qpos:  [40, 62]
```

Raw parquet action semantics:

```text
observation.state[t] = Artimano prop[t]
action[t] = Artimano prop[t + 1]
```

The GR00T modality config uses `ActionRepresentation.RELATIVE`, so GR00T
converts the absolute next-state label to relative action during training.

## Validate Locally

```bash
uv run python scripts/oakink2/validate_gr00t_lerobot_dataset.py \
  --dataset_dir datasets/oakink2/gr00t_lerobot/fccd8_artimano_torch_wrist_qpos
```

Expected checks:

```text
state_dim: 62
action_dim: 62
errors: 0
dataset validation passed
```

The converter also prints this validation command automatically at the end.

## Upload To Server

The converter prints the exact upload commands. Default server settings:

```text
server ssh:         david@SERVER_IP
server repo:        ~/Desktop/haoyu/Isaac-GR00T
server data root:   /mnt/data/haoyu_data/oakink2/gr00t_lerobot
server output root: /mnt/data/haoyu_data/gr00t_outputs
```

You can override them:

```bash
--server-ssh user@host
--server-repo-dir ~/Desktop/haoyu/Isaac-GR00T
--server-data-root /mnt/data/haoyu_data/oakink2/gr00t_lerobot
--server-output-root /mnt/data/haoyu_data/gr00t_outputs
--train-output-name my_run_name
```

## Server Finetune

The converter prints the exact finetune command. It uses:

```text
CUDA_VISIBLE_DEVICES=0,1
uv run torchrun --nproc_per_node=2 --master_port=29500
--embodiment-tag NEW_EMBODIMENT
--modality-config-path examples/ARTIMANO/oakink2_artimano_bimanual_config.py
--num-gpus 2
--global-batch-size 2
--tune-projector
--tune-diffusion-model
```

For a one-trajectory smoke test, first verify the projector+diffusion checkpoint
overfits the train trajectory better than zero-action. If it cannot beat
zero-action on the same trajectory, inspect the dataset/action semantics before
scaling up.
