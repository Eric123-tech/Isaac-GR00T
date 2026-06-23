# AGENTS.md — Isaac GR00T N1.7

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.10 (dGPU, Orin); Python 3.12 (Thor, DGX Spark — see deployment dir)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** setuptools (see `pyproject.toml`)
- **CI:** internal GitLab CI (`.gitlab-ci.yml` + includes under `ci/`, not shipped to the public GitHub EA repo); public GitHub Actions (`.github/workflows/`)

## Quick-start commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run CPU tests
python -m pytest tests/ -m "not gpu" -v --timeout=300

# Run GPU tests
python -m pytest tests/ -m gpu -v --timeout=300

# Build package
uv build

# Validate lockfile
uv lock --locked
```

## Code style

- Formatter: `ruff format` (double quotes, spaces, line-length 100)
- Linter: `ruff check` with rules E, F, I (ignores E501)
- Config lives in `pyproject.toml` under `[tool.ruff]`
- Run `pre-commit run --all-files` before committing

## Directory layout

```
gr00t/              # Main package
  configs/          #   Training, data, and model configs
  data/             #   Data loading, embodiment tags, dataset processing
  eval/             #   Evaluation (run_gr00t_server.py)
  experiment/       #   Training pipeline (launch_finetune.py, trainer.py)
  model/            #   Model architecture (N1.7, base, modules)
  policy/           #   Policy inference (Gr00tPolicy, server/client)
examples/           # Per-embodiment example configs and READMEs
scripts/            # Deployment, conversion, and utility scripts
  deployment/       #   Platform install scripts (dgpu, orin, thor, spark)
tests/              # pytest suite (markers: gpu, not gpu)
getting_started/    # User-facing guides and notebooks
```

## Key entry points

- **Fine-tune:** `bash examples/finetune.sh --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>`
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>`
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Testing

- Test markers: `gpu` (requires GPU), default is CPU-safe
- Fixtures live in `tests/fixtures/` and `demo_data/`
- CI runs CPU and GPU tests in separate jobs with 300s timeout

## Deployment platforms

- **dGPU (H100, A100, RTX):** CUDA 12.8 — install via `scripts/deployment/dgpu/install_deps.sh`, container via top-level `docker/Dockerfile` (supports x86_64 and aarch64)
- **Jetson Orin:** CUDA 12.6 — install via `scripts/deployment/orin/install_deps.sh`, container via `scripts/deployment/orin/Dockerfile`
- **Jetson Thor:** CUDA 13.0 — install via `scripts/deployment/thor/install_deps.sh`, container via `scripts/deployment/thor/Dockerfile`
- **DGX Spark:** CUDA 13.0 — install via `scripts/deployment/spark/install_deps.sh`, container via `scripts/deployment/spark/Dockerfile`

Each Jetson/Spark platform ships an `activate_*.sh` helper (`scripts/activate_orin.sh`, `scripts/activate_spark.sh`, `scripts/activate_thor.sh`) that exports platform-specific library paths. For dGPU, the standard `source .venv/bin/activate` is sufficient.

## Codex agent instructions

This `AGENTS.md` file is the primary project guidance file for Codex and other coding agents.

When working in this repository, Codex should also read the adjacent `CLAUDE.md` file if it exists and treat it as additional project guidance. Apply both files together, with the following precedence:

1. Direct user instructions in the current conversation.
2. `AGENTS.md`.
3. `CLAUDE.md`.
4. General agent defaults.

If `AGENTS.md` and `CLAUDE.md` conflict, follow `AGENTS.md`. If `CLAUDE.md` only adds compatible project notes, commands, conventions, or workflow hints, use them.

Do not overwrite or delete `CLAUDE.md` just because this repository also has `AGENTS.md`; keep both files so Claude- and Codex-based workflows can coexist.

## Imported behavioral guidelines

The following guidelines are imported from
`https://raw.githubusercontent.com/forrestchang/andrej-karpathy-skills/main/CLAUDE.md`
and merged into this repository's agent instructions.

These rules are intended to reduce common LLM coding mistakes. Merge them with the project-specific instructions above as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them; do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what is confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Do not "improve" adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it; do not delete it.

When your changes create orphans:

- Remove imports, variables, and functions that your changes made unused.
- Do not remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria such as "make it work" require constant clarification.

These guidelines are working if there are fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
