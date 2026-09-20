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
- `WANDB_API_KEY` — **required** to enable W&B (opt-in). Also `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_RUN_NAME`. `--no-wandb` still disables.

`.env` is gitignored. `settings.py` loads it on import (`load_dotenv`) and does not override variables already in the environment. `train_a3c.py` and `carla_multiserver_launcher.py` import that loader.

## `settings.py`

Experiment / environment defaults used when CLI does not say otherwise:

- `CARLA_HOST`, `PORT`, `PORT_STEP`
- `MAP_NAME` (default Town03)
- `CAMERA_TYPE`, `RES`, `SCENARIO`, `SPAWNING_TYPE`, `ACTION_TYPE`
- `ACTION_REPEAT` (default 2), `EPISODE_MAX_DECISIONS` (default 200)
- `STEP_COUNTER` = `EPISODE_MAX_DECISIONS * ACTION_REPEAT` (CarlaEnv tick cap)
- Wrapper `--episode-max-decisions` uses the same `EPISODE_MAX_DECISIONS` default (existing flag; not a new one)
- `SHOW_CAM`, `DRAW`

No learning-rate or gamma here. Those stay on the `train_a3c.py` CLI.

Default `--reward-mode` is `legacy` (the original `rl_configuration.reward_function` path). `shaped` is opt-in on the CLI.

RGB and semantic cameras share the same FOV (75).

## Maps / scenarios

Spawn and goal tables live in `<map_name.lower()>.py` next to `carla_env.py`. Default: `town03.py` with `SCENARIOS[14]` (and 10–16). `carla_env` does `importlib.import_module(map_name.lower())`. Missing file or missing scenario id is a `ValueError` — it will not silently use Town03 indices on another map.

A new city: copy `town03.py` → `town04.py` and replace spawn/goal indices. No env edit.

Commented tuples inside the SC14/SC15 lists are unused variants; keep them there.

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
- `--episode-max-decisions` (default from `settings.EPISODE_MAX_DECISIONS`)
- `--reward-mode` `legacy` | `shaped`

`a3c_core.py` does not import `settings.py`. The entry point passes a config namespace.

Checkpoints: only last `checkpoint.pth` (plus sidecar step). `--resume DIR` loads the highest-step last checkpoint. There is no `best_checkpoint.pth`.
