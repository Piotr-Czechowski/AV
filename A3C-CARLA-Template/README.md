# A3C CARLA Template

Hogwild A3C for CARLA 0.9.15. Training never starts the simulator. You bring CARLA up, then run `train_a3c.py` against the RPC ports.

Default blueprint: Town03, scenario 14, 10 discrete actions, semantic camera 250x250.

## Requirements

- Python 3.10 or 3.11
- CARLA 0.9.15 server matching the `carla==0.9.15` client
- GPU recommended (CPU workers are possible, CARLA still wants a GPU)

```bash
cd A3C-CARLA-Template
cp env.example .env
python -m pip install -e .
# optional
python -m pip install -e ".[wandb]"
python -m pip install -e ".[resources]"
```

Runtime third-party deps: `carla==0.9.15`, `numpy`, `torch`, `opencv-python`, `networkx`. `wandb` and `psutil` (`--log-resources`) are extras. Install a CUDA build of PyTorch for your GPU before or instead of the generic `torch` wheel.

## Run

Same training command everywhere. Only the CARLA start changes.

```bash
# local, two terminals
./examples/local/run_servers.sh -w 2 --outdir runs/demo
./examples/run_train.sh -w 2 --outdir runs/demo

# docker (Linux)
./examples/docker/run_servers.sh -w 2 --outdir runs/demo
./examples/run_train.sh -w 2 --outdir runs/demo

# HPC: one job starts servers then trains
sbatch examples/hpc/train.slurm -w 2
```

Details: `docs/launch.md`.

## What to change

| Piece | File |
|---|---|
| Host, map, ports, scenario defaults | `settings.py` and `.env` |
| Discrete actions, legacy reward | `rl_configuration.py` |
| Network | `models/shared_actor_critic.py` |
| Algorithm / run-shape | `train_a3c.py` CLI |
| How CARLA is spawned | `carla_multiserver_launcher.py --runtime` |

See `docs/configuration.md`, `docs/launch.md`, `docs/logging.md`.
