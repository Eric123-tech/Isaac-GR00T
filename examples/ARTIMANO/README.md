# ARTIMANO Synthetic Dataset

This example creates a GR00T-flavored LeRobot v2 dataset for a synthetic ARTIMANO hand embodiment. It is intended to validate dataset structure, modality configuration, and statistics generation before real Artimano data is available.

## Generate synthetic dataset

```bash
uv run python scripts/artimano/create_artimano_synthetic_dataset.py \
  --output-dir demo_data/artimano_synthetic \
  --num-episodes 5 \
  --frames-per-episode 200 \
  --fps 30 \
  --left-hand-dim 20 \
  --right-hand-dim 20
```

## Validate dataset

```bash
uv run python scripts/artimano/validate_artimano_dataset.py \
  --dataset-path demo_data/artimano_synthetic
```

## Generate GR00T statistics

```bash
uv run python gr00t/data/stats.py \
  --dataset-path ./demo_data/artimano_synthetic \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/ARTIMANO/artimano_config.py
```

## Optional smoke fine-tune

This can OOM on 12GB GPUs.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ./demo_data/artimano_synthetic \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/ARTIMANO/artimano_config.py \
  --num-gpus 1 \
  --output-dir /tmp/artimano_synthetic_smoke \
  --save-total-limit 1 \
  --save-steps 20 \
  --max-steps 20 \
  --global-batch-size 1 \
  --dataloader-num-workers 2
```

The generated dataset is ignored by Git. Commit the scripts and config, not the generated `demo_data/artimano_synthetic/` files.
