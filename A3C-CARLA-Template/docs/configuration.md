# Configuration

Four layers. Each value has one place of definition.

| Layer | Holds | Example |
|---|---|---|
| `.env` | This machine: CARLA image, ports, W&B key | `CARLA_CONTAINER_IMAGE` |
| `settings.py` | Defaults of the experiment | map, scenario, camera, episode limit |
| `rl_configuration.py` | What the agent sees and does, and the legacy reward | `ACTIONS`, `observation_spec` |
| `train_a3c.py` | CLI flags for values that change between runs, `DEFAULT_*` constants for the rest | `--lr`, `DEFAULT_REWARD_CLIP` |

Precedence: a CLI flag wins over its default from `settings.py`, which in turn reads `.env`.

Some values are constants on purpose and have no flag. A flag exists only for a value that really changes between runs. To change a constant, edit it in the file named below. `python train_a3c.py --help` lists every flag.

## What do I change, and where

| I want to change | Where | Details |
|---|---|---|
| CARLA image, entrypoint inside the image | `.env`: `CARLA_CONTAINER_IMAGE`, `CARLA_BINARY` | `docs/launch.md` |
| First port, port step | `.env`: `CARLA_START_PORT`, `CARLA_PORT_STEP`; per run `--start-port` | `docs/launch.md` |
| Workers, servers per GPU, run directory | pipeline options | `docs/launch.md` |
| Slurm resources and time limit | `#SBATCH` header of `examples/apptainer/train.slurm`, or `sbatch` flags | `docs/launch.md` |
| CARLA command-line flags, health-check timing, Apptainer adapter offset | constants at the top of `carla_multiserver_launcher.py` | `docs/launch.md` |
| Map, scenario, camera, resolution | CLI `--map-name`, `--scenario`, `--camera`, `--res`; defaults in `settings.py` | `docs/environment.md` |
| Episode length limit, action repeat | CLI `--episode-max-decisions`, `--action-repeat`; defaults in `settings.py` | `docs/environment.md` |
| Reward mode | CLI `--reward-mode legacy\|shaped` | `docs/environment.md` |
| Legacy reward terms, off-route distance | `rl_configuration.py`: `REWARD_FROM_*`, `OFFROUTE_THRESHOLD_M`, `reward_function` | `docs/environment.md` |
| Shaped-reward coefficients | `DEFAULT_REWARD_*` constants in `train_a3c.py` | `docs/environment.md` |
| Spawn points and goals | `town03.py`, one file per map | `docs/environment.md` |
| Simulation step, timeouts, ego vehicle, camera mount, goal radius | constants at the top of `carla_env.py` | `docs/environment.md` |
| Actions, observation keys and shapes | `rl_configuration.py`: `ACTIONS`, `observation_spec` | `docs/how_to_adjust.md` |
| Network | `model.py`: `build_model` | `docs/how_to_adjust.md` |
| Steps, learning rate, gamma, rollout length, entropy schedule, gradient clip | CLI; defaults are `DEFAULT_*` constants in `train_a3c.py` | `docs/algorithm.md` |
| Optimizer internals | constants in `train_a3c.py` | `docs/algorithm.md` |
| Checkpoint frequency, resume, fine-tune | CLI `--save-frequency`, `--resume`, `--init-from` | `docs/architecture.md` |
| Worker restart limits, backoff, reconnect waits | CLI `--max-restarts-per-worker`, `--carla-timeout-wait`, ... | `docs/architecture.md` |
| What is logged, W&B project and run name | CLI `--log-*`, `--wandb-*`, `--no-wandb`; `.env`: `WANDB_*` | `docs/logging.md` |

## `.env`

Copy `env.example` to `.env`. The file is gitignored.

| Variable | Meaning |
|---|---|
| `CARLA_CONTAINER_IMAGE` | Docker tag or `.sif` path. Required. An empty value or `CHANGE_ME` is rejected |
| `CARLA_BINARY` | Entrypoint inside the image, default `/home/carla/CarlaUE4.sh` |
| `CARLA_START_PORT`, `CARLA_PORT_STEP` | First RPC port (2000) and distance between two servers (5) |
| `CARLA_MAP` | Default of `--map-name` (`Town03`) |
| `VENV` | Python environment that the pipeline script activates on the host |
| `CLIENT_CONTAINER_IMAGE`, `CLIENT_PYTHON` | Apptainer only: run training inside this image with this interpreter |
| `APPTAINER_BIND` | Apptainer only: extra bind mounts. Apptainer reads it itself |
| `WANDB_API_KEY` | Turns W&B on. `--no-wandb` still turns it off |
| `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME` | Optional W&B names. The project defaults to `a3c-carla` |

Two programs read the file, so keep it to plain `KEY=value` lines:

- The pipeline scripts `source` it. A value in `.env` then replaces one that is already exported in the shell.
- `settings.py` loads it when you run a Python program directly. A variable that is already in the environment then wins over `.env`.

## `settings.py`

| Name | Default | Meaning |
|---|---|---|
| `CARLA_HOST` | `localhost` | Fixed: both pipelines run the servers on the machine that trains |
| `PORT`, `PORT_STEP` | 2000, 5 | From `CARLA_START_PORT` and `CARLA_PORT_STEP` |
| `MAP_NAME` | `Town03` | From `CARLA_MAP`. Default of `--map-name` |
| `CAMERA_TYPE`, `RES` | `semantic`, 250 | Defaults of `--camera` and `--res` |
| `SCENARIO` | `[14]` | Default of `--scenario` |
| `ACTION_REPEAT` | 2 | Default of `--action-repeat` |
| `EPISODE_MAX_DECISIONS` | 200 | Default of `--episode-max-decisions` |
| `SHOW_CAM`, `DRAW` | `False` | Debug switches that `CarlaEnv` reads directly |

## `train_a3c.py`

Flags that shape a run. The other flags are listed in the document of their topic.

| Flag | Default | Meaning |
|---|---|---|
| `--num-workers` | 1 | Worker processes. Worker `i` uses port `start port + i * port step` |
| `--start-port`, `--port-step` | from `settings.py` | Port grid of the servers |
| `--workers-per-gpu` | 1 | Worker models per GPU. `0` puts them on the CPU |
| `--worker-gpu-start` | 0 | First GPU that gets worker models |
| `--outdir` | `runs/a3c_<N>w_<date>` | Directory of a new run |
| `--resume DIR`, `--init-from PATH` | not set | Continue a run, or start a new one from saved weights |
| `--save-frequency` | 100 000 | Global steps between checkpoints |
| `--save-worker-checkpoints` | off | Also keep a checkpoint copy per worker |

The pipeline scripts set `--num-workers`, `--start-port`, `--outdir`, and `--resume` themselves (`docs/launch.md`).

`train_a3c.py` joins the CLI, the constants without a flag (`_apply_config_defaults`), the number of actions, and the observation spec into one config namespace. The core and the wrapper read only this namespace. To add a value, see `docs/how_to_adjust.md`.
