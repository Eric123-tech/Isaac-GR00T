#!/usr/bin/env python3
"""Plot Hugging Face Trainer loss and learning-rate histories."""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def find_state_files(run_dir):
    root_file = run_dir / "trainer_state.json"
    checkpoint_files = sorted(
        run_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(CHECKPOINT_PATTERN.match(path.parent.name).group(1))
        if CHECKPOINT_PATTERN.match(path.parent.name)
        else -1,
    )
    return ([root_file] if root_file.is_file() else []) + checkpoint_files


def checkpoint_rank(state_file):
    match = CHECKPOINT_PATTERN.match(state_file.parent.name)
    return int(match.group(1)) if match else -1


def load_records(state_files):
    records = {}
    errors = []
    for state_file in state_files:
        try:
            data = json.loads(state_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{state_file}: {exc}")
            continue

        rank = checkpoint_rank(state_file)
        for entry in data.get("log_history", []):
            if not isinstance(entry, dict) or not all(
                key in entry for key in ("step", "loss", "learning_rate")
            ):
                continue
            try:
                step_value = float(entry["step"])
                loss = float(entry["loss"])
                learning_rate = float(entry["learning_rate"])
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (step_value, loss, learning_rate)):
                continue
            step = int(step_value) if step_value.is_integer() else step_value
            if step not in records or rank > records[step][0]:
                records[step] = (rank, loss, learning_rate)
    return records, errors


def moving_average(values, window):
    return [
        sum(values[index - window + 1 : index + 1]) / window
        for index in range(window - 1, len(values))
    ]


def save_plots(records, output_dir, smooth_window):
    steps = sorted(records)
    losses = [records[step][1] for step in steps]
    learning_rates = [records[step][2] for step in steps]
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_figure, loss_axis = plt.subplots()
    loss_axis.plot(steps, losses, label="Raw loss", linewidth=1)
    if 1 < smooth_window <= len(losses):
        loss_axis.plot(
            steps[smooth_window - 1 :],
            moving_average(losses, smooth_window),
            label=f"{smooth_window}-step moving average",
            linewidth=2,
        )
        loss_axis.legend()
    loss_axis.set_xlabel("Training Step")
    loss_axis.set_ylabel("Training Loss")
    loss_axis.set_title("Training Loss vs Global Step")
    loss_axis.grid(True, alpha=0.3)
    loss_figure.tight_layout()
    loss_path = output_dir / "loss_vs_steps.png"
    loss_figure.savefig(loss_path, dpi=150)
    plt.close(loss_figure)

    learning_rate_figure, learning_rate_axis = plt.subplots()
    learning_rate_axis.plot(steps, learning_rates, linewidth=1)
    learning_rate_axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    learning_rate_axis.set_xlabel("Training Step")
    learning_rate_axis.set_ylabel("Learning Rate")
    learning_rate_axis.set_title("Learning Rate vs Global Step")
    learning_rate_axis.grid(True, alpha=0.3)
    learning_rate_figure.tight_layout()
    learning_rate_path = output_dir / "learning_rate_vs_steps.png"
    learning_rate_figure.savefig(learning_rate_path, dpi=150)
    plt.close(learning_rate_figure)
    return loss_path, learning_rate_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=20)
    args = parser.parse_args()
    if args.smooth_window < 0:
        parser.error("--smooth-window must be non-negative")

    run_dir = args.run_dir.expanduser().resolve()
    state_files = find_state_files(run_dir) if run_dir.is_dir() else []
    print("Trainer state files found:")
    for state_file in state_files:
        print(f"  {state_file}")

    records, errors = load_records(state_files)
    if not records:
        checked = [run_dir / "trainer_state.json", run_dir / "checkpoint-*/trainer_state.json"]
        message = "No usable per-step records found. Files checked:\n" + "\n".join(
            f"  {path}" for path in (state_files or checked)
        )
        if errors:
            message += "\nRead errors:\n" + "\n".join(f"  {error}" for error in errors)
        raise SystemExit(message)

    loss_path, learning_rate_path = save_plots(
        records, run_dir / "training_curves", args.smooth_window
    )
    steps = sorted(records)
    print(f"Unique loss points: {len(steps)}")
    print(f"Unique learning-rate points: {len(steps)}")
    print(f"First logged step: {steps[0]}")
    print(f"Last logged step: {steps[-1]}")
    print(f"Loss figure: {loss_path}")
    print(f"Learning-rate figure: {learning_rate_path}")


if __name__ == "__main__":
    main()
