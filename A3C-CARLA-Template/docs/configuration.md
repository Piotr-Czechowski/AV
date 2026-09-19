# Configuration

Three layers. CLI wins over `settings.py`. `.env` wins only when the process environment is empty for that key.

## `.env`

Copy `env.example`. Machine and secrets:

- `CARLA_RUNTIME` — `apptainer`, `docker`, or `native` (launcher default if `--runtime` omitted)
- `CARLA_CONTAINER_IMAGE` — `.sif` or Docker tag; required for apptainer/docker
- `CARLA_BINARY` — entrypoint, default `/home/carla/CarlaUE4.sh`
- `CARLA_PATH` — host directory for native `CarlaUE4.sh`
- `CARLA_HOST` — RPC host Python workers connect to (default `localhost`)
- `CARLA_START_PORT`, `CARLA_PORT_STEP`, `CARLA_MAP` — optional defaults for `settings.py`
- `WANDB_*` — project, entity, API key

`.env` is gitignored. `settings.py` loads it on import (`load_dotenv`) and does not override variables already in the environment. `train_a3c.py` and `carla_multiserver_launcher.py` import that loader.

## `settings.py`

Experiment / environment defaults used when CLI does not say otherwise:

- `CARLA_HOST`, `PORT`, `PORT_STEP`
- `MAP_NAME` (default Town03)
- `CAMERA_TYPE`, `RES`, `SCENARIO`, `SPAWNING_TYPE`, `ACTION_TYPE`
- `STEP_COUNTER` (CarlaEnv episode cap; wrapper also caps at `--episode-max-decisions`, default 100)
- `SHOW_CAM`, `DRAW`

No learning-rate or gamma here. Those stay on the `train_a3c.py` CLI.

## `rl_configuration.py`

- `Actions` / `ACTION_CONTROL` / `ACTIONS_NAMES` — keep these three aligned
- `reward_function` and `REWARD_FROM_*` static terms used by the legacy reward path

`--reward-mode shaped` uses coefficients from `train_a3c.py` defaults, not these static terms.

## `train_a3c.py` CLI

Run shape and algorithm. Notable connection flags:

- `--carla-host` — where workers connect (does not start CARLA)
- `--map-name`
- `--start-port`, `--port-step`, `--num-workers`
- `--scenario`, `--camera`, `--res`

`a3c_core.py` does not import `settings.py`. The entry point passes a config namespace.
