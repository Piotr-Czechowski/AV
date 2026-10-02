# Configuration

Four layers. Each value has one place of definition.

1. `.env`: this machine (CARLA image, ports, W&B key).
2. `settings.py`: defaults of the experiment (map, scenario, camera, episode limit).
3. `rl_configuration.py`: what the agent sees and does, and the legacy reward.
4. `train_a3c.py`: the CLI for values that change between runs, and `DEFAULT_*` constants for the rest.

The CLI wins over `settings.py`. `.env` never overrides a variable that is already in the process environment.

Some values are constants on purpose and have no flag. A flag exists only for a value that really changes between runs. To change a constant, edit it in the file named below.

## What do I change, and where

| I want to change | Where |
|---|---|
| CARLA image, entrypoint inside the image | `.env`: `CARLA_CONTAINER_IMAGE`, `CARLA_BINARY` |
| First port, port step | `.env`: `CARLA_START_PORT`, `CARLA_PORT_STEP`; per run `--start-port` |
| Map | `.env`: `CARLA_MAP`; per run `--map-name` |
| W&B key, project, entity, run name | `.env`: `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`; per run `--wandb-*`, `--no-wandb` |
| Workers, steps, seed, run directory | pipeline options and `train_a3c.py` CLI |
| Scenario, camera, resolution | CLI `--scenario`, `--camera`, `--res`; defaults in `settings.py` |
| Episode length limit, action repeat | CLI `--episode-max-decisions`, `--action-repeat`; defaults in `settings.py` |
| Learning rate, gamma, rollout length, entropy schedule, gradient clip | CLI; defaults are the `DEFAULT_*` constants in `train_a3c.py` |
| Optimizer internals (RMSprop alpha and eps, Adam betas and eps) | constants in `train_a3c.py` (no flag) |
| Reward mode | CLI `--reward-mode legacy\|shaped` |
| Shaped-reward coefficients | `DEFAULT_REWARD_*` constants in `train_a3c.py` (no flag) |
| Legacy reward terms, off-route distance that ends an episode | `rl_configuration.py`: `REWARD_FROM_*`, `OFFROUTE_THRESHOLD_M`, `reward_function` |
| Actions | `rl_configuration.py`: `ACTIONS` |
| Observation keys and shapes | `rl_configuration.py`: `observation_spec` (see `docs/extending.md`) |
| Network | `model.py` (`build_model`) |
| Spawn points and goals | `town03.py` (one file per map) |
| Simulation step, client timeout, ego vehicle, camera mount and FOV, camera timeout, goal radius | constants at the top of `carla_env.py` (no flag) |
| Scale of the speed observation | `SPEED_SCALE` in `carla_wrapper.py` (no flag) |
| Debug drawing, camera preview window | `settings.py`: `DRAW`, `SHOW_CAM` |
| Worker restart limits and backoff, CARLA reconnect waits | CLI (`--max-restarts-per-worker`, `--carla-timeout-wait`, ...); defaults in `train_a3c.py` |
| How long training waits for the servers at start | `SERVER_READY_TIMEOUT_S` in `train_a3c.py` (no flag) |
| CARLA command-line flags, health-check timing, Apptainer adapter offset, Docker container name prefix | constants at the top of `carla_multiserver_launcher.py` (no flag) |
| Time training gets to stop on a signal | `GRACEFUL_SHUTDOWN_WAIT` at the top of the pipeline script |
| Slurm resources and time limit | `#SBATCH` header of `examples/apptainer/train.slurm`, or `sbatch` flags |

## `.env`

Copy `env.example`. The file is gitignored.

- `CARLA_CONTAINER_IMAGE`: Docker tag or `.sif` path. Required. `CHANGE_ME` is rejected.
- `CARLA_BINARY`: entrypoint inside the image, default `/home/carla/CarlaUE4.sh`.
- `CARLA_START_PORT`, `CARLA_PORT_STEP`, `CARLA_MAP`: optional.
- `APPTAINER_BIND`: extra bind mounts. Apptainer reads it itself.
- `WANDB_API_KEY`: required to turn W&B on. `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME` are optional. `--no-wandb` still turns it off.

## `settings.py`

- `CARLA_HOST`: always `localhost`. Both pipelines run the servers on the machine that trains.
- `PORT`, `PORT_STEP`: from `CARLA_START_PORT` (2000) and `CARLA_PORT_STEP` (5).
- `MAP_NAME` (Town03), `CAMERA_TYPE`, `RES`, `SCENARIO`: defaults of `--map-name`, `--camera`, `--res`, `--scenario`.
- `ACTION_REPEAT` (2): world ticks per agent decision.
- `EPISODE_MAX_DECISIONS` (200): default of `--episode-max-decisions`. This is the only episode length limit. `0` turns it off.
- `SHOW_CAM`, `DRAW`: debug switches that `CarlaEnv` reads directly.

## `rl_configuration.py`

- `ACTIONS`: one table of `(name, throttle, brake, steer)`. The row index is the action id. The number of actions and the vehicle control come from it.
- `observation_spec(config)`: shape and dtype of every observation key. The network is built from it and the wrapper checks every observation against it.
- `reward_function` and the `REWARD_FROM_*` terms: the legacy reward path.

`--reward-mode shaped` uses the `DEFAULT_REWARD_*` constants in `train_a3c.py` and not these terms.

## Maps and scenarios

Spawn and goal tables live in `<map_name.lower()>.py` next to `carla_env.py`. The default is `town03.py`. A missing file or a missing scenario id is an error. The env never falls back to Town03 indices on another map.

A new map: copy `town03.py` to `town04.py`, replace the spawn and goal indices, and run with `--map-name Town04`.

Commented tuples inside the scenario lists are unused variants. Keep them there.

## `train_a3c.py`

`a3c_core.py` and `carla_wrapper.py` do not read `settings.py` or argparse. `train_a3c.py` builds one config namespace from the CLI, the off-CLI constants (`_apply_config_defaults`), and the agent interface (`n_actions`, `obs_spec`). Every field is required: the core and the wrapper have no fallback values, so a missing field is an error and not a silent second default.

### New run, resume, fine-tune

| Flag | Directory | Weights | Optimizer, counters, schedules | W&B run |
|---|---|---|---|---|
| `--outdir DIR` (or none) | new: `DIR` or `runs/a3c_<N>w_<date>` | random | new | new |
| `--resume DIR` | continues in `DIR` | from `DIR/checkpoint.pth` | continued | the same run |
| `--init-from PATH` | new (as in the first row) | from `PATH` | new | new |

- A new run refuses a directory that already holds a run (`checkpoint.pth`, `checkpoints/`, `logs/`, or `resume_state.json`). Choose another directory, delete it, or use `--resume`.
- `--resume DIR` without a checkpoint in `DIR` is an error.
- `--init-from PATH` takes a `checkpoint.pth` file. Only `checkpoint['model']` is read. Use it to fine-tune: the learning-rate and entropy schedules start from step 0.
- At the end of a session the last checkpoint is written only when the session made at least one optimizer update. A run that is stopped before it learned anything leaves the existing checkpoint untouched.

Checkpoints: only the last `checkpoint.pth` (plus a step sidecar). There is no `best_checkpoint.pth`.
