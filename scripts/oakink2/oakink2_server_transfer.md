# OakInk2 -> Artimano -> GR00T Server Transfer

This document summarizes the files needed to process OakInk2 trajectories into an Artimano GR00T/LeRobot dataset, upload them to the server, and run GR00T finetuning.

## Goal

Pipeline:

```text
OakInk2 extracted data
  -> SMPL-X/MANO FK
  -> Artimano IK retargeting
  -> Artimano wrist pose + hand qpos
  -> GR00T / LeRobot dataset
  -> server finetune / open-loop evaluation
```

Do not use the older `run_oakink2_to_gr00t_pipeline.py` flow for Artimano. That older path writes direct MANO prop vectors. The current Artimano flow is:

```text
scripts/oakink2/retarget_oakink2_artimano_torch_to_gr00t.py
```

## Required Code

Upload these if the server does not already have the same git branch.

```text
scripts/oakink2/retarget_oakink2_artimano_torch_to_gr00t.py
scripts/oakink2/retargeting/
scripts/oakink2/task_target.json
scripts/oakink2/validate_gr00t_lerobot_dataset.py
scripts/oakink2/oaklink2_gr00t_pipeline.md
scripts/oakink2/oakink2_server_transfer.md
examples/ARTIMANO/oakink2_artimano_bimanual_config.py
third_party/maniptrans_assets/
```

Approximate local sizes:

```text
scripts/oakink2                 272K
examples/ARTIMANO               32K
third_party/maniptrans_assets   105M
```

`third_party/maniptrans_assets/` is required for retargeting. It contains:

```text
smplx/SMPLX_NEUTRAL.npz
smplx_extra/body_upper_idx.pt
mano_urdf/lh_mano.urdf
mano_urdf/rh_mano.urdf
mano_urdf/lh_urdf_meshes/
mano_urdf/rh_urdf_meshes/
```

## Required Data

There are two useful upload modes.

### Mode A: Server Only Finetunes Existing LeRobot Dataset

Upload only the generated dataset:

```text
datasets/oakink2/gr00t_lerobot/
```

Local size currently:

```text
datasets/oakink2/gr00t_lerobot  7.4M
```

This is enough for GR00T finetuning if the LeRobot dataset was already generated locally.

### Mode B: Server Re-runs Retargeting From OakInk2 Extracted Data

Upload extracted OakInk2 data:

```text
datasets/oakink2/extracted/anno_preview/
datasets/oakink2/extracted/data/
```

Local size currently:

```text
datasets/oakink2/extracted  4.7G
```

This is needed if the server should run:

```text
OakInk2 extracted -> Artimano retarget -> LeRobot dataset
```

## Optional Object Assets For IsaacLab Replay

If the server only finetunes GR00T, object mesh assets are not required by the GR00T trainer. If the server also replays in IsaacLab / IsaacGym with objects, upload the object assets from ManipTrans.

For the `fccd8` trajectory:

```text
scene_01__A005++seq__fccd810e2fd673b25c86__2023-04-15-14-30-47
```

object ids:

```text
O02@0015@00019
O02@0015@00020
```

Minimum object asset folders:

```text
/home/eric/linux_projects/ManipTrans/data/OakInk-v2/object_preview/align_ds/O02@0015@00019/
/home/eric/linux_projects/ManipTrans/data/OakInk-v2/object_preview/align_ds/O02@0015@00020/
/home/eric/linux_projects/ManipTrans/data/OakInk-v2/coacd_object_preview/align_ds/O02@0015@00019/
/home/eric/linux_projects/ManipTrans/data/OakInk-v2/coacd_object_preview/align_ds/O02@0015@00020/
```

Note: `O02@0015@00019` may need a manually generated/copied `coacd_object_preview/.../scan.urdf` before IsaacLab replay.

## Local Retarget Command

Example for `fccd8`:

```bash
cd ~/linux_projects/Isaac-GR00T

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

Expected output:

```text
datasets/oakink2/gr00t_lerobot/fccd8_artimano_torch_wrist_qpos/
```

Important files inside the output:

```text
data/chunk-000/episode_000000.parquet
meta/
retarget/artimano_torch_bimanual_retarget.npz
videos/chunk-000/observation.images.ego_view/episode_000000.mp4
```

Validate locally:

```bash
cd ~/linux_projects/Isaac-GR00T

uv run python scripts/oakink2/validate_gr00t_lerobot_dataset.py \
  --dataset_dir datasets/oakink2/gr00t_lerobot/fccd8_artimano_torch_wrist_qpos
```

## Upload Commands

Replace `SERVER_IP` before running.

```bash
cd ~/linux_projects

SERVER=david@SERVER_IP
REMOTE_REPO=/home/david/Desktop/haoyu/Isaac-GR00T
REMOTE_DATA=/mnt/data/haoyu_data/oakink2/gr00t_lerobot
```

Do not set `REMOTE_REPO=~/Desktop/...` on the local machine. Bash expands `~` to the local user home, for example `/home/eric`, before `rsync` sends the path to the server.

### Upload Code And Retarget Assets

Use this when the server does not already have the current branch/files.

```bash
ssh "$SERVER" "mkdir -p \
  '$REMOTE_REPO/scripts/oakink2' \
  '$REMOTE_REPO/scripts/oakink2/retargeting' \
  '$REMOTE_REPO/examples/ARTIMANO' \
  '$REMOTE_REPO/third_party/maniptrans_assets'"

rsync -avhP --partial \
  Isaac-GR00T/scripts/oakink2/retarget_oakink2_artimano_torch_to_gr00t.py \
  Isaac-GR00T/scripts/oakink2/task_target.json \
  Isaac-GR00T/scripts/oakink2/validate_gr00t_lerobot_dataset.py \
  Isaac-GR00T/scripts/oakink2/oaklink2_gr00t_pipeline.md \
  Isaac-GR00T/scripts/oakink2/oakink2_server_transfer.md \
  "$SERVER:$REMOTE_REPO/scripts/oakink2/"

rsync -avhP --partial \
  Isaac-GR00T/scripts/oakink2/retargeting/ \
  "$SERVER:$REMOTE_REPO/scripts/oakink2/retargeting/"

rsync -avhP --partial \
  Isaac-GR00T/examples/ARTIMANO/oakink2_artimano_bimanual_config.py \
  "$SERVER:$REMOTE_REPO/examples/ARTIMANO/"

rsync -avhP --partial \
  Isaac-GR00T/third_party/maniptrans_assets/ \
  "$SERVER:$REMOTE_REPO/third_party/maniptrans_assets/"
```

### Upload Existing Generated LeRobot Dataset

Use this for Mode A.

```bash
ssh "$SERVER" "mkdir -p '$REMOTE_DATA/fccd8_artimano_torch_wrist_qpos'"

rsync -avhP --partial \
  Isaac-GR00T/datasets/oakink2/gr00t_lerobot/fccd8_artimano_torch_wrist_qpos/ \
  "$SERVER:$REMOTE_DATA/fccd8_artimano_torch_wrist_qpos/"
```

### Upload Extracted OakInk2 Dataset

Use this for Mode B.

```bash
ssh "$SERVER" "mkdir -p '$REMOTE_REPO/datasets/oakink2/extracted'"

rsync -avhP --partial \
  Isaac-GR00T/datasets/oakink2/extracted/ \
  "$SERVER:$REMOTE_REPO/datasets/oakink2/extracted/"
```

### Upload Optional Object Assets For IsaacLab Replay

```bash
REMOTE_OBJ=/mnt/data/haoyu_data/oakink2_fccd8

ssh "$SERVER" "mkdir -p \
  '$REMOTE_OBJ/object_preview/align_ds/O02@0015@00019' \
  '$REMOTE_OBJ/object_preview/align_ds/O02@0015@00020' \
  '$REMOTE_OBJ/coacd_object_preview/align_ds/O02@0015@00019' \
  '$REMOTE_OBJ/coacd_object_preview/align_ds/O02@0015@00020'"

rsync -avhP --partial \
  ManipTrans/data/OakInk-v2/object_preview/align_ds/O02@0015@00019/ \
  "$SERVER:$REMOTE_OBJ/object_preview/align_ds/O02@0015@00019/"

rsync -avhP --partial \
  ManipTrans/data/OakInk-v2/object_preview/align_ds/O02@0015@00020/ \
  "$SERVER:$REMOTE_OBJ/object_preview/align_ds/O02@0015@00020/"

rsync -avhP --partial \
  ManipTrans/data/OakInk-v2/coacd_object_preview/align_ds/O02@0015@00019/ \
  "$SERVER:$REMOTE_OBJ/coacd_object_preview/align_ds/O02@0015@00019/"

rsync -avhP --partial \
  ManipTrans/data/OakInk-v2/coacd_object_preview/align_ds/O02@0015@00020/ \
  "$SERVER:$REMOTE_OBJ/coacd_object_preview/align_ds/O02@0015@00020/"
```

## Server Retarget Command

Run this only if extracted OakInk2 data was uploaded to the server.

```bash
cd ~/Desktop/haoyu/Isaac-GR00T

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

Validate on the server:

```bash
cd ~/Desktop/haoyu/Isaac-GR00T

uv run python scripts/oakink2/validate_gr00t_lerobot_dataset.py \
  --dataset_dir datasets/oakink2/gr00t_lerobot/fccd8_artimano_torch_wrist_qpos
```

## Server Finetune Command

This assumes the dataset is available at:

```text
/mnt/data/haoyu_data/oakink2/gr00t_lerobot/fccd8_artimano_torch_wrist_qpos
```

Run two-GPU projector + diffusion finetuning:

```bash
cd ~/Desktop/haoyu/Isaac-GR00T

rm -rf /mnt/data/haoyu_data/gr00t_outputs/oakink2_fccd8_artimano_torch_projector_diffusion_2gpu

CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun --nproc_per_node=2 --master_port=29500 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /mnt/data/haoyu_data/oakink2/gr00t_lerobot/fccd8_artimano_torch_wrist_qpos \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/ARTIMANO/oakink2_artimano_bimanual_config.py \
  --num-gpus 2 \
  --output-dir /mnt/data/haoyu_data/gr00t_outputs/oakink2_fccd8_artimano_torch_projector_diffusion_2gpu \
  --max-steps 2000 \
  --save-steps 100 \
  --global-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 0 \
  --no-use-wandb \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model
```

## Quick Decision Guide

Use this if only training:

```text
Upload code if needed
Upload third_party/maniptrans_assets if server lacks it
Upload generated gr00t_lerobot/fccd8_artimano_torch_wrist_qpos
Run finetune
```

Use this if regenerating data on the server:

```text
Upload code
Upload third_party/maniptrans_assets
Upload datasets/oakink2/extracted
Run retarget script on server
Validate LeRobot dataset
Run finetune
```

Use this if doing IsaacLab object replay:

```text
Also upload object_preview and coacd_object_preview object assets
Make sure missing coacd URDFs are generated before upload
```
