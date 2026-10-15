# Architecture

Which processes run, what happens between the start and the end of a run, what a run leaves on disk, and how failures are handled.

## Processes

```
pipeline script
├── carla_multiserver_launcher.py    one supervisor thread per server
│   └── CARLA server i               RPC port = start port + i * port step
└── train_a3c.py                     main process
    ├── A3CWorker i                  one process per worker
    │   └── CarlaA3CWrapper -> CarlaEnv -> carla.Client("localhost", port i)
    └── telemetry process            events.jsonl, W&B
```

| Process | Code | Owns |
|---|---|---|
| Launcher | `carla_multiserver_launcher.py` | The CARLA servers. Restarts one that exits or hangs (`docs/launch.md`). |
| Main | `train_a3c.py`, `run_a3c.py` | The global model, the shared optimizer, and the run counters, all in CPU shared memory. Supervises the workers and writes the last checkpoint. |
| Worker | `a3c_core.py` | One CARLA client, one local copy of the model on its device, its own log files. |
| Telemetry | `training_logger.py` | `logs/events.jsonl` and every call to W&B. |

Three boundaries keep the parts replaceable:

- `train_a3c.py` never starts the simulator. It waits for servers that already listen on the expected ports.
- `a3c_core.py` and `carla_wrapper.py` read neither `settings.py` nor the command line. `train_a3c.py` builds one config namespace and passes it down. Every field is required: a missing one raises `AttributeError` instead of falling back to a second default.
- The core does not look inside an observation. It moves the observation dict from the wrapper to the model. `rl_configuration.py` and `model.py` define what the dict holds and what the network does with it.

## Life of a run

1. **Config.** `train_a3c.py` parses the CLI and adds the constants that have no flag, the number of actions, the observation spec, and the worker-to-device map.
2. **Run directory.** A new run creates its directory and refuses one that already holds a run. `--resume` continues in an existing one.
3. **Global network.** The model and the optimizer are created on the CPU and placed in shared memory. `--init-from` loads weights. `--resume` loads weights, optimizer state, and counters.
4. **Telemetry.** The telemetry process starts and writes the `training_start` event.
5. **Wait for CARLA.** Training polls every server port until it answers, for at most `SERVER_READY_TIMEOUT_S` (600 s). Training time counts from this moment.
6. **Train.** One process per worker starts (start method `spawn`). From here the main process only supervises: every `--worker-check-interval` seconds it looks for dead workers.
7. **Stop.** The run stops when the step counter reaches `--steps`, on SIGINT, SIGTERM, or SIGUSR1, or when every worker has been given up. Only the main process handles signals. It sets a shared event, and each worker stops after its current step. A worker that has not exited after 10 s is killed.
8. **Final state.** The main process writes `resume_state.json` and `checkpoint.pth`, emits `training_end`, and stops the telemetry process, which writes `final_summary` last.

## Run directory

```
runs/a3c_<N>w_<date>[_<job id>]/
├── checkpoint.pth              model, optimizer, counters: the latest state only
├── checkpoint_step.txt         global step of checkpoint.pth
├── resume_state.json           counters, elapsed training time, arguments of the last session
├── args.txt                    launch arguments (args_resume.txt for a resumed session)
├── logs/                       JSONL records, see docs/logging.md
├── wandb_run_id.txt, wandb/    W&B run id for --resume, W&B's own files      (W&B only)
├── checkpoints/worker_<id>/    extra checkpoint copies                       (--save-worker-checkpoints)
├── episodes/<episode>-<port>/  camera frames                                 (--save-episodes, --save-episode-interval)
├── a3c_training.log            output of train_a3c.py                        (pipeline)
├── carla_servers.log           output of the launcher                        (pipeline)
├── server_logs/                launcher events, raw output of each server    (pipeline)
└── gpu_dmon.log, slurm.log     GPU monitor, link to the Slurm job log        (Apptainer pipeline)
```

## Checkpoints and resume

`checkpoint.pth` holds the model weights, the optimizer state, and the run counters (steps, episodes, updates, best reward, recent rewards). It is written to a temporary file and renamed, so an interrupted write leaves the previous file intact.

It is written at two moments:

- every `--save-frequency` global steps (default 100 000, `0` turns it off), by the worker whose step crosses the boundary,
- at the end of a session, by the main process, only when the session made at least one update. A run stopped before it learned anything keeps its old checkpoint.

Only the latest state is kept. A new best episode reward is logged (`[BEST]`, `is_new_best`) and no separate best-model file is written. A checkpoint is never written while the global parameters contain NaN.

| Flag | Directory | Weights | Optimizer, counters, schedules | W&B run |
|---|---|---|---|---|
| `--outdir DIR` (or none) | new: `DIR` or `runs/a3c_<N>w_<date>` | random | new | new |
| `--resume DIR` | continues in `DIR` | from the checkpoint in `DIR` | continued | the same run |
| `--init-from PATH` | new, as in the first row | from `PATH` | new | new |

- A new run refuses a directory that holds `checkpoint.pth`, `checkpoints/`, `logs/`, or `resume_state.json`.
- `--resume DIR` needs a checkpoint in `DIR`. It takes the one with the highest step among `checkpoint.pth` and `checkpoints/worker_*/checkpoint.pth`.
- `--resume` does not restore the arguments of the earlier session. Pass the same flags again. Training prints a warning when `--steps`, `--lr`, `--beta-*`, `--rollout-length`, `--gamma`, `--weight-decay`, or `--optimizer` differ from the saved ones.
- `--init-from PATH` reads only the weights. The learning-rate and entropy schedules start from step 0, which makes it the flag for fine-tuning.
- Both flags work only with a checkpoint of the same network.

## Failure handling

| Failure | Reaction |
|---|---|
| A CARLA server exits or hangs | The launcher restarts it (`docs/launch.md`). The worker notices through the next two rows. |
| A CARLA call times out (`... waiting for the simulator`) | The worker emits `crash_recovery`, drops the rollout, waits `--carla-timeout-wait`, and reconnects. A connection is tried `--max-connect-retries` times, `--connect-retry-wait` apart. |
| No camera frame within 2 s | The worker emits `camera_timeout`, drops the rollout, and starts a new episode. |
| NaN in the gradients | The worker emits `nan_gradient`, skips the update, and copies the global weights again. |
| Any other exception in a worker | The worker emits `worker_crash` and its process dies. The supervisor emits `worker_restart` and starts it again after a wait. |
| A worker dies again and again | See below: rollback, then give up. |
| The launcher dies | The pipeline script prints the end of the launcher log, stops training, and exits with code 1. |

**Worker restart.** A restarted worker builds a new local model, copies the global weights, and connects to the same port. The weights and the counters live in the main process, so only the unfinished rollout is lost. The wait before restart number `k` of a worker is `--carla-server-start-period * 2^(k-1)`, capped at `--carla-restart-backoff-max`: 30, 60, 120, 240, 300 s.

**Rollback.** A crash that comes fewer than `--rapid-crash-window-steps` global steps after the same worker's previous crash counts as rapid. After `--rapid-crash-threshold` rapid crashes in a row the supervisor loads the newest readable, NaN-free checkpoint into the global network and emits `rollback`.

**Give up.** A worker that reached `--max-restarts-per-worker` is not started again (`worker_give_up`). The other workers keep training. The run stops when no worker is left.

| Flag | Default | Meaning |
|---|---|---|
| `--worker-check-interval` | 5 s | How often the supervisor looks for dead workers |
| `--max-restarts-per-worker` | 160 | Restarts before a worker is given up |
| `--carla-server-start-period` | 30 s | First wait before a restart, doubled each time |
| `--carla-restart-backoff-max` | 300 s | Longest wait before a restart |
| `--rapid-crash-window-steps` | 100 | A crash this soon after the previous one is rapid |
| `--rapid-crash-threshold` | 3 | Rapid crashes in a row that trigger a rollback |
| `--carla-timeout-wait` | 60 s | Wait before a worker reconnects after a simulator timeout |
| `--max-connect-retries` | 5 | Connection attempts of a worker |
| `--connect-retry-wait` | 30 s | Wait between connection attempts |

The client timeout for one CARLA call (`CLIENT_TIMEOUT_S`, 120 s) and the camera timeout (`CAMERA_TIMEOUT_S`, 2 s) are constants at the top of `carla_env.py`.
