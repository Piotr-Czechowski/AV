# A3C CARLA Template

Hogwild A3C for CARLA 0.9.15. One command starts the CARLA servers, trains, and cleans up. Two pipelines are supported:

- **Docker** for a Linux workstation or server,
- **Apptainer** for a Slurm cluster.

Default blueprint: Town03, scenario 14, 10 discrete actions, semantic camera 250×250.

## Requirements

- Linux x86_64, Python 3.10, CARLA 0.9.15
- NVIDIA GPU
- Docker pipeline: Docker with the NVIDIA Container Toolkit
- Apptainer pipeline: Apptainer (`apptainer exec --nv`)

The `carla==0.9.15` client wheel exists for Python 3.10 and not for 3.11, so the template pins 3.10. The client and the server image must be the same CARLA version. A mismatch is an error.

```bash
git clone <this-repo>
cd A3C-CARLA-Template
cp env.example .env            # set CARLA_CONTAINER_IMAGE
# In a Python 3.10 environment. Install a CUDA build of PyTorch for your GPU
# first (or instead of the generic torch wheel pulled below).
python -m pip install -e ".[wandb,resources]"
```

Runtime deps: `carla==0.9.15`, `numpy`, `torch`, `opencv-python`, `networkx`.
Extras: `wandb`; `resources` = `psutil` + `pynvml` for `--log-resources`.

## Run

```bash
# Docker
docker pull carlasim/carla:0.9.15
./examples/docker/train.sh -w 1

# Apptainer on Slurm (edit the #SBATCH header first)
apptainer build carla_0.9.15.sif docker://carlasim/carla:0.9.15
sbatch examples/apptainer/train.slurm -w 1
```

`-w N` starts N workers and N CARLA servers. Several servers can share one GPU: `-w 2 --servers-per-gpu 2`. Every argument the script does not know goes to `train_a3c.py` unchanged, for example `--scenario 14 15 --steps 2000000`.

Each run gets its own directory, `runs/a3c_<N>w_<date>`. A new run refuses a directory that already holds a run.

```bash
./examples/docker/train.sh -w 1 -r runs/a3c_1w_20260101_120000                             # continue that run
./examples/docker/train.sh -w 1 --init-from runs/a3c_1w_20260101_120000/checkpoint.pth     # new run from its weights
```

Details: `docs/launch.md`.

## What to change

| Piece | File |
|---|---|
| CARLA image, ports, W&B | `.env` |
| Map, scenario, camera, episode limit defaults | `settings.py` |
| Observations, actions, legacy reward | `rl_configuration.py` |
| Network | `model.py` |
| Town03 spawn/goal tables | `town03.py` (copy to `town04.py` for another map) |
| Algorithm / run shape | `train_a3c.py` CLI and its `DEFAULT_*` constants |
| How CARLA is started and supervised | `carla_multiserver_launcher.py` |

`docs/configuration.md` has the full "what do I change, in which file" table. `docs/extending.md` has recipes for a new network, a new input, another action table, another camera, and a new map. `docs/logging.md` describes every logged value.

Evaluation is a separate program, `evaluation.py`. It is a stub for now: its docstring describes what it will do.

W&B is opt-in: install the extra, set `WANDB_API_KEY`, omit `--no-wandb`.

Smoke tests (no CARLA):

```bash
python -m unittest tests.test_template_smoke
```
