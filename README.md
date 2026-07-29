# AV - Autonomous Driving Research

This repository contains experimental autonomous driving work in the CARLA simulator.
The main focus areas are "A to B" and "Chase" using reinforcement learning and A3C-style training.

## Folder `A3C`

The `A3C` folder contains an accelerated A3C training implementation for CARLA.
It includes:
- `new_hogwild_train_a3c_carla.py` — main PyTorch A3C training script,
- `new_hogwild_a3c.py` — A3C algorithm with parallel workers,
- `new_hogwild_carla_wrapper.py` — CARLA environment wrapper,
- `new_hogwild_run_a3c.py` — worker orchestration, supervision, and restart logic,
- `new_hogwild_training_logger.py` and `new_hogwild_system_monitor.py` — structured logging and monitoring,
- `new_hogwild_prepare_output_dir.py` and `new_hogwild_train.slurm` — output preparation and HPC job support.

Key acceleration features in `A3C`:
- multi-process Hogwild-style training,
- shared actor-critic model with shared RMSprop optimizer,
- active gradient clipping, NaN protection, and improved logging,
- separation of CARLA logic from training loops for better modularity,
- SLURM/cluster support with automatic CARLA server startup and cleanup.

## Repository structure

- `A3C/` — main accelerated A3C implementation and supporting tools,
- `A3C/docs/` — reference documentation for running the trainer and for the
  logged values,
- `A_to_B_GPU_34/` — earlier A3C versions for the A-to-B task,
- `Gif/` — visual examples of driving and chase scenarios.

## How to use

Training runs as a single `sbatch` call. The SLURM script starts one CARLA
server per worker, waits for the RPC ports, launches the trainer, and cleans up
on exit.

### 1. Point the code at your paths

```bash
cd A3C
cp new_hogwild_train_paths.json.example new_hogwild_train_paths.json
```

Fill in `VENV`, `PROJECT_DIR`, `MULTISERVER_SCRIPT`, and the three `WANDB_*`
keys. All six must be present; use `null`, not `""`, for W&B values you skip.
The script exits immediately if the file is missing.

Two paths are **not** read from that file and have to be set separately:

- `new_hogwild_train.slurm`, line 15 — `PROJECT_DIR` is hardcoded there, because
  that is what finds the JSON file in the first place.
- `settings.py` — `CARLA_PATH` and `CARLA_EGG_PATH`, only needed if you start
  CARLA yourself instead of through `MULTISERVER_SCRIPT`.

Also review the `#SBATCH` header: `--account`, `--partition`, and `--time` are
set for one specific allocation.

### 2. Launch

```bash
# 6 workers, one CARLA server per GPU
sbatch --gpus=6 new_hogwild_train.slurm -w 6 --workers-per-gpu 1 --servers-per-gpu 1

# continue an earlier run
sbatch --gpus=6 new_hogwild_train.slurm -w 6 -r runs/a3c_hogwild_6w_20260729_120000_123456

# pass rare arguments straight to the trainer
sbatch --gpus=6 new_hogwild_train.slurm -w 6 -- --lr 5e-5
```

One CARLA server is started per worker, so keep `--gpus` consistent with
`--servers-per-gpu`. Every option has a default — only step 1 is mandatory.

| Option | Default | Effect |
|---|---|---|
| `-w`, `--workers NUM` | `1` | Workers, and therefore CARLA servers |
| `--workers-per-gpu NUM` | `1` | Learner workers per GPU |
| `--servers-per-gpu NUM` | = workers-per-gpu | CARLA servers per GPU; lower it first if VRAM runs out |
| `-s`, `--scenario "N [N...]"` | `14` | Route set |
| `-r`, `--resume DIR` | fresh run | Continue an earlier run directory |
| `--steps NUM` | `10000000` | Environment-step budget |
| `--no-wandb` | off | Disable W&B; local logging is unaffected |
| `--testing` | off | Evaluation mode, no gradient updates |
| `--` | — | Everything after this goes to the Python trainer |

Full option list, including GPU placement, ports, recovery, and logging flags:
[`A3C/docs/running.md`](A3C/docs/running.md).

### 3. Read the results

Each run directory holds checkpoints, `a3c_training.log`, `carla_servers.log`,
`gpu_dmon.log`, and a `logs/` tree of JSONL records — the complete history of the
run. W&B, when enabled, mirrors a curated subset of it.

- [`A3C/docs/running.md`](A3C/docs/running.md) — every parameter, what it
  controls, and the startup/resume flow.
- [`A3C/docs/logging.md`](A3C/docs/logging.md) — every logged value, what it
  means, where it lands, and how often.


## Requirements

- Python 3.x
- PyTorch with optional CUDA support for GPU
- CARLA 0.9.x
- required packages are listed in `requirements.txt` and `Pipfile`
