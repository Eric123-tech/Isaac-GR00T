# OakInk2 to GR00T / LeRobot Pipeline

This document describes the current OakInk2 data preparation workflow for GR00T-N1.7 finetuning.

The current verified pipeline is:

```text
OakInk2 ego-view RGB
+ OakInk2 anno_preview raw_mano
+ task text from task_target.json
    -> GR00T / LeRobot-style dataset
    -> dataset validation
    -> GR00T-N1.7 projector-only smoke finetune
```

The current verified smoke test result is:

```text
Dataset: direct_hand_manifest_1_mano_full128
Episodes: 1
Rows: 299
State dim: 128
Action dim: 128
Video frames: 299
Validation errors: 0
Training: projector-only, 20/20 steps completed
```

Do not modify NVIDIA model source files for this data pipeline. Keep changes limited to:

```text
scripts/oakink2/
examples/ARTIMANO/
```

---

## 1. Machine paths

### Local machine

Local repository:

```bash
~/linux_projects/Isaac-GR00T
```

OakInk2 data entry inside the repository:

```bash
~/linux_projects/Isaac-GR00T/datasets/oakink2
```

Recommended real local data location:

```bash
~/datasets/oakink2
```

Recommended local symlink:

```bash
cd ~/linux_projects/Isaac-GR00T
mkdir -p datasets
ln -sfn ~/datasets/oakink2 datasets/oakink2
readlink -f datasets/oakink2
```

Expected output:

```text
/home/eric/datasets/oakink2
```

### Server

Server repository:

```bash
~/Desktop/haoyu/Isaac-GR00T
```

OakInk2 data entry inside the repository:

```bash
~/Desktop/haoyu/Isaac-GR00T/datasets/oakink2
```

Recommended real server data location:

```bash
/mnt/data/haoyu_data/oakink2
```

Recommended server symlink:

```bash
cd ~/Desktop/haoyu/Isaac-GR00T
mkdir -p datasets
ln -sfn /mnt/data/haoyu_data/oakink2 datasets/oakink2
readlink -f datasets/oakink2
```

Expected output:

```text
/mnt/data/haoyu_data/oakink2
```

Recommended server training output location:

```bash
/mnt/data/haoyu_data/gr00t_outputs
```

---

## 2. OakInk2 directory layout

The pipeline expects this structure:

```text
datasets/oakink2/
├── extracted/
│   ├── data/
│   ├── anno_preview/
│   └── program/
├── manifests/
│   └── oakink2_available_trajectories.jsonl
└── gr00t_lerobot/
    └── direct_hand_manifest_1_mano_full128/
```

The verified trajectory is:

```text
scene_01__A003/seq__49a8305e104d29e3816a__2023-04-15-09-41-56
```

Encoded folder name:

```text
scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56
```

Verified camera:

```text
104422070969
```

Expected RGB folder:

```text
datasets/oakink2/extracted/data/scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56/104422070969
```

Expected annotation preview file:

```text
datasets/oakink2/extracted/anno_preview/scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56.pkl
```

---

## 3. Manual extracted layout convention

At the current stage, OakInk2 files are manually downloaded and manually extracted outside this pipeline.

The pipeline no longer reads:

```text
datasets/oakink2/raw/
```

It also does not auto-extract `.zip`, `.tar`, `.tar.gz`, or `.tgz` files.

### RGB files

RGB frames should be placed directly under:

```text
datasets/oakink2/extracted/data/<encoded_key>/<camera>/
```

Example:

```text
datasets/oakink2/extracted/data/scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56/104422070969/
```

This folder should contain image frames such as:

```text
000003.png
000004.png
...
```

Supported image extensions:

```text
.png
.jpg
.jpeg
```

### Annotation preview files

The usable annotation preview `.pkl` file should be placed directly under:

```text
datasets/oakink2/extracted/anno_preview/<encoded_key>.pkl
```

Example:

```text
datasets/oakink2/extracted/anno_preview/scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56.pkl
```

The converter reads the `raw_mano` field from this pickle file.

Expected `raw_mano` frame fields include:

```text
lh__pose_coeffs
rh__pose_coeffs
lh__tsl
rh__tsl
lh__betas
rh__betas
```

The current `full128` pipeline only uses:

```text
lh__pose_coeffs
rh__pose_coeffs
```

---

## 4. Task text source

Task text comes from ManipTrans:

```text
~/linux_projects/ManipTrans/data/OakInk-v2/program/task_target.json
```

The structure is a dictionary:

```json
{
  "scene_01__A003/seq__49a8305e104d29e3816a__2023-04-15-09-41-56": "Cap the bottle."
}
```

Task lookup rule:

```python
from urllib.parse import unquote

decoded_key = unquote(encoded_key)
task_key = decoded_key.replace("++", "/")
task = task_targets.get(task_key)
```

Example:

```text
encoded_key:
scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56

decoded_key:
scene_01__A003++seq__49a8305e104d29e3816a__2023-04-15-09-41-56

task_key:
scene_01__A003/seq__49a8305e104d29e3816a__2023-04-15-09-41-56

task:
Cap the bottle.
```

---

## 5. Current scripts

The current OakInk2 pipeline uses these scripts:

```text
scripts/oakink2/inspect_anno_preview.py
scripts/oakink2/build_oakink2_manifest.py
scripts/oakink2/convert_one_oakink2_to_gr00t_realprop.py
scripts/oakink2/convert_oakink2_manifest_to_gr00t.py
scripts/oakink2/validate_gr00t_lerobot_dataset.py
scripts/oakink2/run_oakink2_to_gr00t_pipeline.py
```

### One-command pipeline

Script:

```text
scripts/oakink2/run_oakink2_to_gr00t_pipeline.py
```

Purpose:

```text
Check the manually extracted layout, generate the manifest, convert to GR00T / LeRobot format, and validate the dataset.
```

Known one-trajectory command:

```bash
uv run python scripts/oakink2/run_oakink2_to_gr00t_pipeline.py \
  --oakink2-root datasets/oakink2 \
  --task-target-json scripts/oakink2/task_target.json \
  --camera 104422070969 \
  --task-key scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56 \
  --manifest-output datasets/oakink2/manifests/oakink2_available_trajectories.jsonl \
  --fps 10 \
  --image-size 256 \
  --max-frames 300 \
  --prop-mode full128 \
  --overwrite
```

With this `--task-key`, the default output path is:

```text
datasets/oakink2/gr00t_lerobot/49a83_direct_hand_manifest_1_mano_full128
```

The `49a83` prefix is taken from the first five characters after `seq__`.
Pass `--output-dir` explicitly to override this naming rule.

`--task-key` accepts all of these equivalent forms:

```text
scene_01__A003/seq__49a8305e104d29e3816a__2023-04-15-09-41-56
scene_01__A003++seq__49a8305e104d29e3816a__2023-04-15-09-41-56
scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56
scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56.pkl
```

Internally these are normalized to the `task_target.json` form:

```text
scene_01__A003/seq__49a8305e104d29e3816a__2023-04-15-09-41-56
```

The script always requires these directories to already exist:

```text
datasets/oakink2/extracted/data
datasets/oakink2/extracted/anno_preview
```

The old `--skip-extract` flag is still accepted for command compatibility, but it no longer changes behavior.

Use `--dry-run` to check paths and selected trajectories without writing the manifest or converted dataset:

```bash
uv run python scripts/oakink2/run_oakink2_to_gr00t_pipeline.py \
  --task-key scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56 \
  --dry-run
```

The one-command script intentionally locks `--prop-mode` to:

```text
full128
```

Use the lower-level manifest and converter scripts below for debugging or for experiments with other proprioception definitions.

After a successful one-command run for:

```text
scene_01__A001%2B%2Bseq__9e46184387ba83f60895__2023-04-27-18-43-58
```

the script prints the matching upload command:

```bash
cd ~/linux_projects/Isaac-GR00T
rsync -avhP --partial datasets/oakink2/gr00t_lerobot/9e461_direct_hand_manifest_1_mano_full128/ david@SERVER_IP:/mnt/data/haoyu_data/oakink2/gr00t_lerobot/9e461_direct_hand_manifest_1_mano_full128/
```

and the matching server projector-only finetune command:

```bash
cd ~/Desktop/haoyu/Isaac-GR00T
rm -rf /mnt/data/haoyu_data/gr00t_outputs/oakink2_9e461_projector_only
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --num_processes 1 --mixed_precision bf16 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path datasets/oakink2/gr00t_lerobot/9e461_direct_hand_manifest_1_mano_full128 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/ARTIMANO/oakink2_mano_full134_config.py \
  --num-gpus 1 \
  --output-dir /mnt/data/haoyu_data/gr00t_outputs/oakink2_9e461_projector_only \
  --max-steps 2000 \
  --save-steps 100 \
  --global-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 0 \
  --no-use-wandb \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --no-tune-diffusion-model
```

---

## 6. Inspect annotation preview

Script:

```text
scripts/oakink2/inspect_anno_preview.py
```

Purpose:

```text
Inspect the internal structure of an OakInk2 anno_preview pickle file.
```

Example:

```bash
uv run python scripts/oakink2/inspect_anno_preview.py \
  --pkl datasets/oakink2/extracted/anno_preview/scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56.pkl
```

Use this script when checking whether `raw_mano` exists or when debugging annotation fields.

---

## 7. Build OakInk2 manifest

Script:

```text
scripts/oakink2/build_oakink2_manifest.py
```

Purpose:

```text
Scan locally available extracted OakInk2 RGB and anno_preview files, resolve task text, and write a JSONL manifest.
```

Command:

```bash
uv run python scripts/oakink2/build_oakink2_manifest.py \
  --oakink2-root datasets/oakink2 \
  --task-target-json ~/linux_projects/ManipTrans/data/OakInk-v2/program/task_target.json \
  --camera 104422070969 \
  --output datasets/oakink2/manifests/oakink2_available_trajectories.jsonl
```

Expected output for the current local setup:

```text
[done]
output:              datasets/oakink2/manifests/oakink2_available_trajectories.jsonl
total trajectories:  1
matched tasks:       1
missing tasks:       0
has rgb dir:         1
has anno pkl:        1
rgb frame count > 0: 1
valid rows:          1
```

Each manifest row contains:

```json
{
  "episode_index": 0,
  "encoded_key": "scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56",
  "decoded_key": "scene_01__A003++seq__49a8305e104d29e3816a__2023-04-15-09-41-56",
  "task_key": "scene_01__A003/seq__49a8305e104d29e3816a__2023-04-15-09-41-56",
  "scene": "scene_01__A003",
  "seq": "seq__49a8305e104d29e3816a__2023-04-15-09-41-56",
  "rgb_dir": "datasets/oakink2/extracted/data/<encoded_key>/104422070969",
  "anno_pkl": "datasets/oakink2/extracted/anno_preview/<encoded_key>.pkl",
  "camera": "104422070969",
  "task": "Cap the bottle.",
  "task_lookup_status": "matched",
  "num_rgb_frames": 972,
  "has_rgb_dir": true,
  "has_anno_pkl": true
}
```

If task lookup fails:

```json
{
  "task": null,
  "task_lookup_status": "missing"
}
```

---

## 8. Convert one OakInk2 trajectory

Script:

```text
scripts/oakink2/convert_one_oakink2_to_gr00t_realprop.py
```

Purpose:

```text
Convert a single OakInk2 trajectory into a GR00T / LeRobot-style dataset.
```

This was the original debug script. It is still useful for single-trajectory testing.

Example:

```bash
RGB_DIR='datasets/oakink2/extracted/data/scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56/104422070969'

ANNO_PKL='datasets/oakink2/extracted/anno_preview/scene_01__A003%2B%2Bseq__49a8305e104d29e3816a__2023-04-15-09-41-56.pkl'

uv run python scripts/oakink2/convert_one_oakink2_to_gr00t_realprop.py \
  --rgb_dir "$RGB_DIR" \
  --anno_pkl "$ANNO_PKL" \
  --output_dir datasets/oakink2/gr00t_lerobot/direct_hand_1_mano_full128 \
  --task "Cap the bottle." \
  --fps 10 \
  --image_size 256 \
  --max_frames 300 \
  --prop_mode full128
```

Expected output:

```text
[prop shape] (300, 128)
[prop dim] 128
rows: 299
dim: 128
```

This script writes only one episode.

---

## 9. Convert manifest to GR00T / LeRobot dataset

Script:

```text
scripts/oakink2/convert_oakink2_manifest_to_gr00t.py
```

Purpose:

```text
Read the JSONL manifest and convert all valid rows into one GR00T / LeRobot-style dataset.
```

Current one-trajectory command:

```bash
uv run python scripts/oakink2/convert_oakink2_manifest_to_gr00t.py \
  --manifest datasets/oakink2/manifests/oakink2_available_trajectories.jsonl \
  --output_dir datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128 \
  --fps 10 \
  --image_size 256 \
  --max_frames 300 \
  --prop_mode full128 \
  --overwrite
```

Expected output:

```text
[manifest] datasets/oakink2/manifests/oakink2_available_trajectories.jsonl
[rows] total=1 valid=1 skipped=0
[output_dir] datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128
[prop_mode] full128
[ok] episode=000000 rows=299 dim=128 task='Cap the bottle.'
[done]
successful episodes: 1
skipped rows:        0
failed rows:         0
total frames:        299
prop dim:            128
```

Generated dataset:

```text
datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
├── videos/
│   └── chunk-000/
│       └── observation.images.ego_view/
│           └── episode_000000.mp4
└── meta/
    ├── info.json
    ├── modality.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    └── conversion_report.json
```

---

## 10. Validate generated GR00T / LeRobot dataset

Script:

```text
scripts/oakink2/validate_gr00t_lerobot_dataset.py
```

Purpose:

```text
Check consistency between meta files, parquet files, and video files.
```

Command:

```bash
uv run python scripts/oakink2/validate_gr00t_lerobot_dataset.py \
  --dataset_dir datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128
```

Verified output:

```text
[dataset] datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128
[meta]
  total_episodes: 1
  total_frames: 299
  total_tasks: 1
  fps: 10
  state_dim: 128
  action_dim: 128
  image_shape: [256, 256, 3]
  tasks: ['Cap the bottle.']
[episode 000000] rows=299 state_dim=128 action_dim=128 video_frames=299 task=['Cap the bottle.']
[summary]
  episodes: 1
  rows: 299
  errors: 0
[done] dataset validation passed
```

---

## 11. Proprioception and action definition

Current working mode:

```text
prop_mode = full128
```

Definition:

```text
observation.state[t] = concat(
    lh__pose_coeffs[t].reshape(16 * 4),
    rh__pose_coeffs[t].reshape(16 * 4)
)
```

Dimension:

```text
left hand MANO pose:  16 * 4 = 64
right hand MANO pose: 16 * 4 = 64
total: 128
```

Action:

```text
action[t] = observation.state[t + 1]
```

The raw parquet action stores the absolute next-state target. The GR00T
finetune path uses `ActionRepresentation.RELATIVE`, so its state/action
processor converts the target to `action[t] - observation.state[t]` during
training.

Therefore, if `T` matched RGB/MANO frames are used:

```text
parquet rows = T - 1
video frames = T - 1
```

Current verified conversion:

```text
input frames used: 300
parquet rows: 299
video frames: 299
state dim: 128
action dim: 128
```

### Do not use full134 for GR00T smoke finetuning

`full134` means:

```text
left MANO pose 64
left translation 3
right MANO pose 64
right translation 3
total 134
```

This caused a GR00T-N1.7 negative padding issue because the action dimension exceeded the expected limit.

Use:

```text
full128
```

for current GR00T smoke tests.

---

## 12. GR00T modality config

Current config:

```text
examples/ARTIMANO/oakink2_mano_full134_config.py
```

Although the filename contains `full134`, the config can be used with `full128` because the dataset metadata provides the actual action/state dimensions.

Important modality keys:

```text
video:
  ego_view

state:
  hand

action:
  hand

language:
  annotation.human.action.task_description
```

Dataset feature keys:

```text
observation.images.ego_view
observation.state
action
annotation.human.action.task_description
```

---

## 13. Upload generated dataset to server

Code is synchronized with git.

Data is synchronized with `rsync`.

### Upload converted GR00T dataset

From local machine:

```bash
rsync -avhP --partial \
  datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128/ \
  david@SERVER_IP:/mnt/data/haoyu_data/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128/
```

Single-line version:

```bash
rsync -avhP --partial datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128/ david@SERVER_IP:/mnt/data/haoyu_data/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128/
```

Do not use `-p 2222` unless the server SSH service is actually running on port 2222.

### Upload manifest

From local machine:

```bash
rsync -avhP --partial \
  datasets/oakink2/manifests/oakink2_available_trajectories.jsonl \
  david@SERVER_IP:/mnt/data/haoyu_data/oakink2/manifests/oakink2_available_trajectories.jsonl
```

### Upload extracted RGB and anno files if server-side conversion is needed

Usually, the server does not need extracted OakInk2 files if the converted GR00T dataset has already been uploaded.

If the server needs to rebuild the dataset, upload extracted data:

```bash
rsync -avhP --partial \
  datasets/oakink2/extracted/data/ \
  david@SERVER_IP:/mnt/data/haoyu_data/oakink2/extracted/data/

rsync -avhP --partial \
  datasets/oakink2/extracted/anno_preview/ \
  david@SERVER_IP:/mnt/data/haoyu_data/oakink2/extracted/anno_preview/
```

If the server also needs to build the manifest, make sure `task_target.json` is available. Recommended server location:

```text
/mnt/data/haoyu_data/oakink2/extracted/program/task_target.json
```

Then use that server path in `--task-target-json`.

---

## 14. Validate dataset on server

On the server:

```bash
cd ~/Desktop/haoyu/Isaac-GR00T

uv run python scripts/oakink2/validate_gr00t_lerobot_dataset.py \
  --dataset_dir datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128
```

Expected result:

```text
[done] dataset validation passed
```

---

## 15. GR00T projector-only smoke finetune

The verified training setup is:

```text
GR00T-N1.7
accelerate launch
mixed_precision bf16
single visible GPU
projector-only finetuning
```

Run on server:

```bash
cd ~/Desktop/haoyu/Isaac-GR00T

rm -rf /mnt/data/haoyu_data/gr00t_outputs/oakink2_mano_smoke_projector_only

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --num_processes 1 --mixed_precision bf16 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/ARTIMANO/oakink2_mano_full134_config.py \
  --num-gpus 1 \
  --output-dir /mnt/data/haoyu_data/gr00t_outputs/oakink2_mano_smoke_projector_only \
  --max-steps 20 \
  --save-steps 20 \
  --global-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 0 \
  --no-use-wandb \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --no-tune-diffusion-model
```

Successful output should include:

```text
100%|...| 20/20
Model saved to /mnt/data/haoyu_data/gr00t_outputs/oakink2_mano_smoke_projector_only
Training completed!
```

This verifies:

```text
dataset loading
statistics generation
relative statistics generation
shard caching
dataloader
model forward
backward
optimizer step
checkpoint save
```

---

## 16. Diffusion finetuning note

Full diffusion finetuning with:

```text
--tune-diffusion-model
```

caused CUDA OOM on one 5090.

The OOM happened at:

```text
optimizer.step()
Adam state initialization
```

This means the data pipeline is already working; the issue is optimizer state memory for too many trainable parameters.

For future diffusion finetuning, use one of:

```text
DeepSpeed ZeRO-2 / ZeRO-3
FSDP
fewer trainable modules
smaller trainable submodules
```

Do not treat diffusion OOM as a data conversion failure.

---

## 17. Standard local workflow

Use this when new OakInk2 trajectories have been downloaded and extracted.

```bash
cd ~/linux_projects/Isaac-GR00T
```

Build manifest:

```bash
uv run python scripts/oakink2/build_oakink2_manifest.py \
  --oakink2-root datasets/oakink2 \
  --task-target-json ~/linux_projects/ManipTrans/data/OakInk-v2/program/task_target.json \
  --camera 104422070969 \
  --output datasets/oakink2/manifests/oakink2_available_trajectories.jsonl
```

Convert manifest to GR00T dataset:

```bash
uv run python scripts/oakink2/convert_oakink2_manifest_to_gr00t.py \
  --manifest datasets/oakink2/manifests/oakink2_available_trajectories.jsonl \
  --output_dir datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128 \
  --fps 10 \
  --image_size 256 \
  --max_frames 300 \
  --prop_mode full128 \
  --overwrite
```

Validate:

```bash
uv run python scripts/oakink2/validate_gr00t_lerobot_dataset.py \
  --dataset_dir datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128
```

Upload to server:

```bash
rsync -avhP --partial datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128/ david@SERVER_IP:/mnt/data/haoyu_data/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128/
```

---

## 18. Standard server workflow

```bash
cd ~/Desktop/haoyu/Isaac-GR00T
```

Pull latest code:

```bash
git pull
```

Validate uploaded dataset:

```bash
uv run python scripts/oakink2/validate_gr00t_lerobot_dataset.py \
  --dataset_dir datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128
```

Run projector-only smoke finetune:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --num_processes 1 --mixed_precision bf16 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/ARTIMANO/oakink2_mano_full134_config.py \
  --num-gpus 1 \
  --output-dir /mnt/data/haoyu_data/gr00t_outputs/oakink2_mano_smoke_projector_only \
  --max-steps 20 \
  --save-steps 20 \
  --global-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 0 \
  --no-use-wandb \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --no-tune-diffusion-model
```

---

## 19. Instructions for future Codex edits

When modifying this pipeline, Codex should follow these rules:

1. Do not modify NVIDIA model source files.
2. Keep OakInk2 pipeline code under:

```text
scripts/oakink2/
```

3. Keep GR00T modality configuration under:

```text
examples/ARTIMANO/
```

4. Do not commit data files under:

```text
datasets/oakink2/
```

5. Preserve the current working `full128` representation unless explicitly changing the action/state definition.
6. Preserve direct RGB frame-id to `raw_mano` frame-id matching.
7. Preserve LeRobot metadata fields required by GR00T-N1.7:

```text
data_path
video_path
total_videos
total_chunks
chunks_size
splits
features
```

8. Always validate generated datasets with:

```bash
uv run python scripts/oakink2/validate_gr00t_lerobot_dataset.py \
  --dataset_dir <dataset_dir>
```

9. Treat projector-only smoke finetune as the first training test.
10. Treat diffusion OOM as a training-memory issue, not a dataset-conversion issue.

---

## 20. Scaling to more trajectories

The current manifest contains one complete trajectory because only one trajectory has been locally downloaded and extracted.

To scale up:

1. Download more OakInk2 RGB trajectory data.
2. Download the matching `anno_preview` pickle files.
3. Place RGB frames under:

```text
datasets/oakink2/extracted/data/<encoded_key>/<camera>/
```

4. Place annotation preview files under:

```text
datasets/oakink2/extracted/anno_preview/<encoded_key>.pkl
```

5. Re-run:

```bash
uv run python scripts/oakink2/build_oakink2_manifest.py \
  --oakink2-root datasets/oakink2 \
  --task-target-json ~/linux_projects/ManipTrans/data/OakInk-v2/program/task_target.json \
  --camera 104422070969 \
  --output datasets/oakink2/manifests/oakink2_available_trajectories.jsonl
```

6. Check `valid rows`.
7. Convert the manifest.
8. Validate the generated dataset.
9. Upload converted dataset to server.
10. Run projector-only smoke finetune.

Suggested dataset names:

```text
direct_hand_manifest_1_mano_full128
direct_hand_manifest_5_mano_full128
direct_hand_manifest_20_mano_full128
direct_hand_manifest_50_mano_full128
```

Do not start full diffusion finetuning until projector-only training works on the larger dataset.

---

## 21. Known good baseline

Known good dataset:

```text
datasets/oakink2/gr00t_lerobot/direct_hand_manifest_1_mano_full128
```

Known good validation result:

```text
episodes: 1
rows: 299
state_dim: 128
action_dim: 128
video_frames: 299
errors: 0
```

Known good training result:

```text
GR00T-N1.7
projector-only
20/20 steps completed
train_loss around 1.34
checkpoint saved successfully
```

This is the reference baseline for future changes.
