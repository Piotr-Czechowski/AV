# A3C CARLA Template

Hogwild A3C for CARLA 0.9.15. Training never starts the simulator. You bring CARLA up, then run `train_a3c.py` against the RPC ports.

Default blueprint: Town03, scenario 14, 10 discrete actions, semantic camera 250×250.

## Requirements

- Python 3.10 or 3.11
- CARLA 0.9.15 server matching the `carla==0.9.15` client
- GPU recommended (CPU workers are possible; CARLA still wants a GPU)

```bash
git clone <this-repo>
cd A3C-CARLA-Template
cp env.example .env
# Install a CUDA build of PyTorch for your GPU first (or instead of the
# generic torch wheel pulled below).
python -m pip install -e ".[wandb,resources]"
```

Runtime deps: `carla==0.9.15`, `numpy`, `torch`, `opencv-python`, `networkx`.
Extras: `wandb`; `resources` = `psutil` + `pynvml` for `--log-resources`.

Always launch training from this directory so `models.shared_actor_critic` imports (there is no `models/__init__.py`).

## Run

Same training command everywhere. Only the CARLA start changes. Defaults are **1 worker / 1 GPU**. Two workers need two GPUs: `-w 2` and `--gpus=2` (Slurm) plus matching launcher count.

```bash
# local, two terminals
./examples/local/run_servers.sh -w 1 --outdir runs/demo
./examples/run_train.sh -w 1 --outdir runs/demo

# docker (Linux, --network host + --ipc=host)
./examples/docker/run_servers.sh -w 1 --outdir runs/demo
./examples/run_train.sh -w 1 --outdir runs/demo

# HPC: one job starts servers then trains
sbatch examples/hpc/train.slurm -w 1
```

Details: `docs/launch.md`.

## What to change

| Piece | File |
|---|---|
| Host, map, ports, scenario, episode cap | `settings.py` and `.env` |
| Discrete actions, legacy reward | `rl_configuration.py` |
| Town03 spawn/goal tables | `town03.py` (copy to `town04.py` for another map) |
| Network | `models/shared_actor_critic.py` |
| Algorithm / run-shape | `train_a3c.py` CLI |
| How CARLA is spawned | `carla_multiserver_launcher.py --runtime` |

See `docs/configuration.md`, `docs/launch.md`, `docs/logging.md`.

W&B is opt-in: install the extra, set `WANDB_API_KEY`, omit `--no-wandb`.

Smoke tests (no CARLA):

```bash
python -m unittest tests.test_template_smoke
```
