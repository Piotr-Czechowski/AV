# A3C CARLA Template

A template for training a driving agent in CARLA 0.9.15 with A3C (asynchronous advantage actor-critic, Hogwild updates). One command starts the CARLA servers, trains, and cleans up.

The training loop is separate from the agent definition. You can replace the network, the observations, the actions, the reward, and the map without touching the algorithm.

Default setup: Town03, scenario 14 (right turns), semantic camera 250×250, 10 discrete actions, one worker.

## How it works

```
pipeline script                      examples/docker/train.sh or examples/apptainer/train.slurm
├── carla_multiserver_launcher.py    starts N CARLA servers, restarts a crashed or hung one
└── train_a3c.py                     main process: shared global model, worker supervisor
    ├── worker 0 .. N-1              own CARLA server, own local model, asynchronous updates
    └── telemetry process            events.jsonl and Weights & Biases
```

- **One worker, one simulator.** Each worker drives in its own CARLA server and collects short rollouts.
- **Shared model.** The global model and its optimizer live in CPU shared memory. Every worker computes gradients on a local copy and applies them to the global model without waiting for the others.
- **Built to survive failures.** A crashed or hung simulator is restarted, a dead worker is restarted, and a run that was stopped continues from its last checkpoint.
- **Complete local logs.** Every episode and every update is written as JSONL. Weights & Biases is optional and receives a subset.

## Requirements

- Linux x86_64, Python 3.10, CARLA 0.9.15, NVIDIA GPU
- Docker pipeline: Docker with the NVIDIA Container Toolkit
- Apptainer pipeline: Apptainer (`apptainer exec --nv`)

The `carla==0.9.15` client wheel exists for Python 3.10 and not for 3.11, so the template pins 3.10. The client and the server must be the same CARLA version. A mismatch is an error.

```bash
git clone <this-repo>
cd A3C-CARLA-Template
cp env.example .env            # set CARLA_CONTAINER_IMAGE
# In a Python 3.10 environment. Install a CUDA build of PyTorch for your GPU
# first, or pip pulls the generic torch wheel.
python -m pip install -e ".[wandb,resources]"
```

Runtime dependencies: `carla==0.9.15`, `numpy`, `torch`, `opencv-python`, `networkx`.
Extras: `wandb`, and `resources` (`psutil`, `pynvml`) for `--log-resources`.

## Run

```bash
# Docker
docker pull carlasim/carla:0.9.15
./examples/docker/train.sh -w 1

# Apptainer on Slurm (set partition and account in the #SBATCH header first)
apptainer build carla_0.9.15.sif docker://carlasim/carla:0.9.15
sbatch examples/apptainer/train.slurm -w 1
```

`-w N` starts N workers and N CARLA servers. Several servers can share one GPU: `-w 2 --servers-per-gpu 2`. Every argument the script does not know goes to `train_a3c.py` unchanged, for example `--scenario 14 15 --steps 2000000`.

Each run gets its own directory under `runs/`. A new run refuses a directory that already holds a run.

```bash
./examples/docker/train.sh -w 1 -r runs/a3c_1w_20260101_120000                            # continue that run
./examples/docker/train.sh -w 1 --init-from runs/a3c_1w_20260101_120000/checkpoint.pth    # new run from its weights
```

Weights & Biases is opt-in: install the `wandb` extra, set `WANDB_API_KEY` in `.env`, and do not pass `--no-wandb`.

## Repository map

| File | Role |
|---|---|
| `train_a3c.py` | Entry point: CLI, default values, run directory, signals, last checkpoint |
| `run_a3c.py` | Worker supervisor: restart, backoff, rollback |
| `a3c_core.py` | Global network, shared optimizers, worker loop, loss, checkpoints |
| `model.py` | The network (`SharedActorCritic`, `build_model`) |
| `rl_configuration.py` | Observation spec, action table, legacy reward |
| `carla_wrapper.py` | `reset()` / `step()` for the worker: observation dict, action repeat, episode limit, shaped reward |
| `carla_env.py` | CARLA client: world, route, vehicle, sensors |
| `town03.py` | Spawn and goal tables for Town03 (one file per map) |
| `carla_navigation/` | Route planner from the CARLA agents package (MIT) |
| `settings.py` | `.env` loader and experiment defaults |
| `training_logger.py`, `timing_utils.py` | JSONL logs, telemetry process, W&B, resource sampler, phase timer |
| `carla_multiserver_launcher.py` | Starts and supervises the CARLA servers |
| `examples/` | The two pipeline scripts |
| `evaluation.py` | Stub. Its docstring describes the planned evaluation program |
| `tests/` | Smoke tests that need no CARLA: `python -m unittest tests.test_template_smoke` |

## Documentation

| Document | Content |
|---|---|
| `docs/architecture.md` | Processes, life of a run, run directory, checkpoints and resume, failure handling |
| `docs/algorithm.md` | What a worker does, the loss, the schedules, the network, algorithm parameters |
| `docs/environment.md` | Simulation, episode, observation, actions, reward, routes and scenarios |
| `docs/configuration.md` | Where each value is defined and which file to edit to change it |
| `docs/launch.md` | Docker and Apptainer pipelines, ports, GPU placement, server health checks |
| `docs/logging.md` | Every logged value, local files, W&B metrics, events |
| `docs/how_to_adjust.md` | How to adjust the template to your own agent: new network, input, action table, reward, camera, map |
