# Running the trainer, and what every parameter does

`README.md` covers the normal path: fill in the paths JSON, fix `PROJECT_DIR` in
the SLURM header, `sbatch`. This document explains what happens underneath and
what each parameter actually changes.

## The paths file

`new_hogwild_train.slurm` reads `new_hogwild_train_paths.json` from the
directory named in its own line-15 `PROJECT_DIR`, and exits with an error if the
file is missing or if any key is absent. Use `null` rather than `""` for values
you want left unset.

| Key | Meaning |
|---|---|
| `VENV` | Virtualenv to activate. Skipped silently if the directory does not exist, so a typo here shows up later as a missing-import error. |
| `PROJECT_DIR` | Absolute path of the `A3C/` directory on the cluster. Becomes the working directory and the parent of `runs/`. |
| `MULTISERVER_SCRIPT` | Absolute path to the CARLA multiserver launcher (`carla_athena_multiserver_v3*.py`). Ignored with `--no-carla`. |
| `WANDB_PROJECT` | W&B project name. |
| `WANDB_RUN_NAME` | W&B run name, or `null` for `a3c-<workers>w-<id>`. |
| `WANDB_ENTITY` | W&B team or user, or `null` for your default. |

`--venv`, `--project-dir`, and `--multiserver-script` override the first three
for a single job without editing the file.

Two paths live outside the JSON: `PROJECT_DIR` in the SLURM header (it is what
locates the JSON), and `CARLA_PATH` / `CARLA_EGG_PATH` in `settings.py`, needed
only when CARLA is started from Python rather than through
`MULTISERVER_SCRIPT`.

## What a run consists of

One `sbatch` call produces this process tree:

```
SLURM job
├── CARLA multiserver          one CARLA server per worker, own RPC port
├── nvidia-smi dmon            GPU trace -> gpu_dmon.log
└── new_hogwild_train_a3c_carla.py
    ├── A3CTelemetry           sole owner of events.jsonl and W&B
    ├── RunMonitor (thread)    node-wide resource sampling
    └── supervisor
        └── worker 0..N-1      each with its own CARLA server and local model
```

Workers share one CPU model and one optimizer in shared memory (Hogwild). Each
worker keeps a local copy on its GPU, collects `--rollout-length` transitions,
computes gradients locally, copies them into the shared model, and applies an
asynchronous optimizer step. There is no lock by default.

Startup order matters: CARLA servers must be listening before the trainer
starts, which is why the SLURM script polls every expected port before
launching Python.

## Choosing the run shape

`--workers` is the primary knob. Each worker needs its own CARLA server, and
CARLA is the expensive part — a server costs several GB of VRAM. `--gpus` in the
`sbatch` header, `--workers-per-gpu`, and `--servers-per-gpu` have to add up:

```
workers          <= gpus * workers-per-gpu
workers          <= gpus * servers-per-gpu
```

If CARLA runs out of VRAM, lower `--servers-per-gpu` before anything else.
Learner workers are small; CARLA servers are not.

More workers means more gradient staleness: a worker computes gradients against
a model version that other workers have already moved. In practice 8–16 workers
is a reasonable range.

## Run shape parameters

| Parameter | Default | What it controls |
|---|---|---|
| `--num-workers` | `2` | Parallel A3C workers. Set through `-w` in the SLURM script. |
| `--workers-per-gpu` | `2` | Learner processes per GPU. |
| `--worker-gpu-start` | `0` | First GPU index used by learners. |
| `--start-port`, `--port-step` | `2000`, `100` | RPC ports of the CARLA servers. Worker *i* connects to `start-port + i * port-step`. |
| `--scenario` | `14` | Route set. Full list of ids is documented in `settings.py`. Several ids can be given. |
| `--camera` | `semantic` | `semantic` or `rgb`. Changes the observation, so switching invalidates old checkpoints. |
| `--res` | `250` | Camera resolution fed to the network. Also invalidates old checkpoints. |
| `--seed` | `52` | RNG seed. Does not make a Hogwild run deterministic — worker interleaving is not reproducible. |
| `--steps` | `10000000` | Global environment-step budget. The run stops when the shared counter reaches it. |
| `--testing` | off | Evaluation mode: rollouts are collected but no gradients are applied. |

## Algorithm parameters

| Parameter | Default | What it controls |
|---|---|---|
| `--optimizer` | `shared-rmsprop` | `shared-rmsprop` or `shared-adam`. Both keep their state in shared memory. RMSprop is the A3C default and is more tolerant of stale gradients. |
| `--lr` | `1e-4` | Learning rate. Too high shows up as exploding `train/gradient_norm_pre_clip` and a collapsing `train/entropy`. |
| `--rollout-length` | `20` | Transitions collected before each update. Longer means less biased returns and fewer, larger updates; shorter means faster feedback and more noise. |
| `--gamma` | `0.99` | Discount factor. At 0.99 the horizon is roughly 100 steps. Lower it if episodes are short. |
| `--value-loss-coef` | `1.0` | Weight of the critic loss inside the total loss. Raise it if `train/v_loss` stays flat and high. |
| `--beta-start` | `0.02` | Entropy bonus at the start. Higher means more exploration. |
| `--beta-end` | `0.002` | Entropy bonus after annealing. |
| `--beta-anneal-frac` | `0.6` | Fraction of `--steps` over which the bonus decays from start to end. |
| `--beta` | unset | Constant entropy coefficient. Overrides the whole schedule above. |
| `--max-grad-norm` | `5.0` | Gradient clipping threshold. `0` disables it. Check `train/grad_clipped`: constant 1 means the threshold is too tight. |
| `--no-normalize-advantages` | normalization on | Turns off per-rollout advantage normalization. With 20-step rollouts normalization is usually helpful. |
| `--weight-decay` | `0.0` | L2 regularization. |
| `--reward-scale` | `0.0` | Multiplies rewards before the loss. `0` means no scaling. |
| `--sync-every-n-updates` | `1` | How often a worker pulls fresh weights from the shared model. Raising it reduces contention at the cost of more staleness. |
| `--hogwild-lock-updates` | off | Serializes optimizer steps. Removes the Hogwild race at a throughput cost; useful when diagnosing instability. |
| `--gc-interval` | `10` | Episodes between forced garbage collection and CUDA cache clearing. |

Optimizer internals (`RMSPROP_ALPHA`, `RMSPROP_EPS`, `ADAM_BETA1`,
`ADAM_BETA2`, `ADAM_EPS`) are deliberately not CLI arguments — they are
module-level constants in `new_hogwild_train_a3c_carla.py` and are filled in as
defaults when missing.

## Environment and reward parameters

| Parameter | Default | What it controls |
|---|---|---|
| `--action-repeat` | `2` | Simulator ticks each chosen action is held for. Higher means coarser control but faster wall-clock progress. |
| `--episode-max-decisions` | `100` | Hard cap on decisions per episode. Combined with `--action-repeat` this sets the real episode length. |
| `--world-reload-interval` | `0` | Episodes between full CARLA world reloads. `0` disables it. Use it if the simulator degrades over long runs. |
| `--reward-mode` | `legacy` | `legacy` uses the original reward. `shaped` switches on the decomposed reward whose components are logged separately. |

The shaping coefficients (`DEFAULT_REWARD_PROGRESS_COEF`,
`DEFAULT_REWARD_GOAL_BONUS`, `DEFAULT_REWARD_COLLISION_PENALTY`,
`DEFAULT_REWARD_CLIP`, and the rest) are **not** CLI arguments. They are
module-level constants near the top of `new_hogwild_train_a3c_carla.py` and are
edited in place. Every one of them is recorded in `logs/metadata.json`, so past
runs stay reproducible.

## Recovery parameters

CARLA crashes. The supervisor is built around that.

| Parameter | Default | What it controls |
|---|---|---|
| `--worker-check-interval` | `5.0` s | How often the supervisor checks whether workers are alive. |
| `--max-restarts-per-worker` | `160` | Restart budget per worker. Exhausting it emits `worker_give_up` and leaves that worker down. |
| `--rapid-crash-threshold` | `3` | Crashes inside the step window that trigger a rollback. |
| `--rapid-crash-window-steps` | `100` | Width of that window, in global steps. |
| `--carla-timeout-wait` | `60.0` s | How long a worker waits for the simulator before declaring a timeout. |
| `--carla-server-start-period` | `30.0` s | Grace period given to a starting CARLA server. |
| `--carla-restart-backoff-max` | `300.0` s | Ceiling of the restart backoff. |
| `--max-connect-retries`, `--connect-retry-wait` | `5`, `30.0` s | Connection attempts to a CARLA server and the pause between them. |

A rapid-crash burst usually means NaN weights. The supervisor then reloads the
last checkpoint without NaNs into the shared model and emits a `rollback` event.

## Checkpointing and resume

| Parameter | Default | What it controls |
|---|---|---|
| `--save-frequency` | `100000` | Global steps between checkpoints. |
| `--save-worker-checkpoints` | off | Additionally keep a per-worker checkpoint, which widens the rollback choices. |
| `--outdir` | auto | Explicit output directory. |
| `--resume DIR` | fresh run | Continue an existing run directory. |

`--resume` restores model weights, optimizer state, the global step counter,
episode counters, best reward, cumulative training time, and the W&B run id from
`wandb_run_id.txt`, so the resumed session continues the same W&B chart.

Note that `global_t` steps back to the last checkpoint on resume, so the JSONL
files can contain overlapping ranges. The `training_start` / `training_end`
event pair in `events.jsonl` marks where one session ends and the next begins.

Changing `--steps`, `--lr`, `--gamma`, `--rollout-length`, `--optimizer`,
`--weight-decay`, or any `--beta-*` value on resume prints a `[RESUME] WARNING`
line but is allowed. Changing `--res` or `--camera` will not work: the
checkpoint no longer matches the network.

## Logging and monitoring parameters

| Parameter | Default | What it controls |
|---|---|---|
| `--no-wandb` | off | Disables W&B. Local JSONL is unaffected, and `events.jsonl` is still written. |
| `--wandb-project`, `--wandb-run-name`, `--wandb-entity` | from paths JSON | W&B destination. |
| `--log-steps` | off | One record per environment step. Enormous; use only for short debugging runs. |
| `--log-update-arrays` | off | Keeps raw advantage/value/reward/entropy arrays inside update records. |
| `--diag-log-interval` | `100` | Updates between timing snapshots. |
| `--diag-log-wall-s` | `60.0` | Seconds between timing snapshots, whichever comes first. |
| `--monitor-interval` | `10.0` s | Resource sampling period. |
| `--no-system-monitor` | off | Disables both the node and the per-worker resource monitors. |
| `--no-gpu-monitor` | off | Keeps the system monitor but drops GPU sampling. |
| `--verbose-env-logs` | off | Keeps per-episode CARLA stdout output. |
| `--save-episodes`, `--save-episode-interval` | none, `0` | Episodes whose camera frames are written to disk. |

See [`logging.md`](logging.md) for what each recorded value means.

## Shutdown

The SLURM header sets `--signal=SIGUSR1@90`, so the job receives a signal 90
seconds before its time limit. The trainer then stops workers, writes
`resume_state.json` and the `training_end` event, flushes telemetry, and closes
the W&B run. `--graceful-shutdown-wait` (default 75 s) is how long the SLURM
script waits for that before killing the process.

The consequence: a run that hits its wall-clock limit still ends with a usable
checkpoint and a complete log. A run killed with `scancel -s KILL` does not.

## First-run sanity check

Start small before committing a long allocation:

```bash
sbatch --gpus=1 new_hogwild_train.slurm -w 2 --workers-per-gpu 2 --servers-per-gpu 2 -- --steps 20000
```

Then verify:

- `logs/events.jsonl` contains `training_start`, two `worker_start`, and finally
  `training_end` plus `final_summary`;
- `logs/worker_0/episodes.jsonl` grows;
- `final_summary` reports `queue_drops: 0`;
- the W&B run shows `episode/reward` and `train/pi_loss`.
