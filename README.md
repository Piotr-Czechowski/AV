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
- `new_hogwild_prepare_output_dir.py` and `new_hogwild_train.slurm` — output preparation and HPC job support,
- `new_hogwild_train_paths.sh.example` — template for the single file holding every user-specific path.

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
cp new_hogwild_train_paths.sh.example new_hogwild_train_paths.sh
```

This is the only file to edit, and it has **one required line** — the path to
your CARLA Apptainer image:

```sh
export CARLA_CONTAINER_IMAGE=/net/tscratch/people/you/carla_0.9.15.sif
```

Everything else is derived from where you submit from, or has a working
default; the file documents each option next to the value it would override.
The SLURM script sources the file and the Python modules read the resulting
environment, so nothing else in the repository carries a machine-specific path.

The file is looked up next to the directory you submit from, so submit as
`cd A3C && sbatch new_hogwild_train.slurm`, or set `CONFIG_FILE` to its path.
It is gitignored, so your paths and W&B settings stay local. Running anything
outside `sbatch` — the trainer, the CARLA multiserver — means sourcing it
first: `source new_hogwild_train_paths.sh`.

One thing still lives outside it: the `#SBATCH` header of
`new_hogwild_train.slurm` — `--account`, `--partition`, and `--time` are set for
one specific allocation. SLURM parses those directives before any code runs, so
they cannot come from JSON; override them on the command line
(`sbatch --account=... --partition=...`) or edit the header once.

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

Everything a run produces lands in one directory under `RUNS_DIR`: checkpoints,
`a3c_training.log`, `carla_servers.log`, `server_logs/`, `gpu_dmon.log`,
`slurm.log` (a link to the SLURM job output), saved episode frames under
`episodes/`, and a `logs/` tree of JSONL records — the complete history of the
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
