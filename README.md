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
- `A_to_B_GPU_34/` — earlier A3C versions for the A-to-B task,
- `Gif/` — visual examples of driving and chase scenarios,
- `comparison.md` — comparison of different A3C implementations in the repository.

## How to use

1. Configure the CARLA executable and `.egg` paths in `A3C/settings.py`.
2. Run the appropriate training script from `A3C/`.
3. For cluster use, launch `A3C/new_hogwild_train.slurm` or prepare the output directory with `A3C/new_hogwild_prepare_output_dir.py`.

## Requirements

- Python 3.x
- PyTorch with optional CUDA support for GPU
- CARLA 0.9.x
- required packages are listed in `requirements.txt` and `Pipfile`
