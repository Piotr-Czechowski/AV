# Logging

What gets logged, where, and how often. Two independent destinations:

- **Local JSONL** under `<run directory>/logs/`. The complete record of the run: every value exactly as training produced it. This is the source of truth.
- **Weights & Biases**, when enabled. A selected subset, one record forwarded as one point. Nothing is averaged, batched, or downsampled.

Each training process writes its own files, so local logging needs no locking. Records for W&B and all events travel through one queue to a single telemetry process. That path is best effort: when the queue (10 000 records) is full, W&B loses points and the local files stay complete.

`NaN` and infinities become `null` in JSON and are not sent to W&B.

## Files

| File | Written by | One line per |
|---|---|---|
| `logs/worker_<id>/episodes.jsonl` | each worker | finished episode |
| `logs/worker_<id>/updates.jsonl` | each worker | optimizer update |
| `logs/worker_<id>/timing.jsonl` | each worker | timing window |
| `logs/worker_<id>/steps.jsonl` | each worker | environment step, only with `--log-steps` |
| `logs/worker_<id>/resources.jsonl` | each worker | resource sample, only with `--log-resources` |
| `logs/worker_-1/resources.jsonl` | main process | resource sample with GPU usage, only with `--log-resources` |
| `logs/events.jsonl` | telemetry process | lifecycle or health event |
| `logs/metadata.json` | main process | rewritten at the start of every session: arguments, model name, parameter count |

Every record carries `kind` (record type), `ts` (timestamp), `worker` (source worker, `-1` for the main process), and `global_t` where it applies. A resumed run appends to the same files. The `training_start` and `training_end` events mark the sessions.

The console output of training is `a3c_training.log` in the run directory. Its lines are tagged, for example `[BEST]`, `[TIMING]`, `[SAVE]`, `[RESTART]`, `[ROLLBACK]`, `[NaN]`, and a final `[BENCHMARK]` line with steps per second.

## Step, update, episode

Training writes on three cadences, from densest to sparsest:

- **step**: every `env.step()`. Only with `--log-steps`, only to `steps.jsonl`, never to W&B. Fields: `action`, `value`, `entropy`, `reward`, `done`, `step_in_ep`, `speed_kmh`, `route_dist`, `goal_dist`, `maneuver`, `reward_components`. It is the only view inside an episode and grows by gigabytes per hour with many workers, so it is off by default.
- **update**: every `--rollout-length` steps or at episode end, whichever comes first. Losses, gradients, and statistics of that one rollout.
- **episode**: at episode end. The outcome of the whole episode, plus the run-wide best and rolling means at that moment.

`timing` and `system` records follow their own clocks.

## Episode values

All of them reach W&B.

| Field | W&B metric | Meaning |
|---|---|---|
| `global_episode` | `episode/id` | Episode counter shared by all workers |
| `global_t` | `global_step` | Global step at episode end. The x-axis of every chart |
| `total_reward` | `episode/reward`, `worker_<id>/reward` | Sum of rewards in this episode, before `--reward-scale` |
| `mean_reward` | `episode/reward_per_step` | `total_reward / steps`, comparable across episode lengths |
| `steps` | `episode/length`, `worker_<id>/episode_length` | Decisions taken in the episode |
| `duration_s` | `episode/duration_s` | Wall-clock seconds. A rising value usually means CARLA is degrading |
| `reached_goal` | `episode/success` | 1 when the goal was reached, else 0 |
| `max_speed_kmh` | `episode/max_speed_kmh` | Highest speed reached |
| `min_route_dist` | `episode/min_route_distance` | Closest approach to the planned route |
| `goal_dist` | `episode/goal_distance` | Distance to the goal at episode end. The main progress signal on failed episodes |
| `collisions` | `episode/collisions` | Collision events recorded by CARLA |
| `local_mean_reward` | `worker_<id>/reward_mean_100` | Mean reward of this worker's last 100 episodes |
| `global_mean_reward` | `global/reward_mean` | Mean reward of the run's last `100 * workers` episodes |
| `best_reward` | `global/best_reward` | Best single-episode reward so far |
| `action_counts` | `episode/action_<n>_fraction` | How often each action was chosen, sent as a fraction. A value near 1.0 means the policy collapsed onto one action |
| `reward_components` | `episode/reward_<name>` | Episode total of each reward term (see below) |
| `is_new_best`, `port` | not sent | Local only |

## Update values

One record per optimizer update, covering that one rollout. Each value reaches W&B twice: as `train/<metric>` (all workers interleaved) and as `worker_<id>/train/<metric>` (that worker alone).

| Field | W&B metric | Meaning |
|---|---|---|
| `pi_loss` | `train/pi_loss` | Policy loss. Noisy by nature: watch the trend |
| `v_loss` | `train/v_loss` | Value loss, already multiplied by `--value-loss-coef` |
| `total_loss` | `train/total_loss` | The optimized objective |
| `gradient_norm_pre_clip` | `train/gradient_norm_pre_clip` | Gradient norm before clipping. Compare with `--max-grad-norm` |
| `gradient_norm` | `train/gradient_norm` | Gradient norm after clipping |
| `grad_clipped` | `train/grad_clipped` | 1 when clipping changed this update. Its average in W&B is the clipping rate |
| `ent_mean` | `train/entropy` | Policy entropy. A fall towards 0 means exploration is dying |
| `entropy_coef` | `train/entropy_coef` | Entropy coefficient at this step |
| `lr` | `train/lr` | Learning rate at this step |
| `advantages_mean`, `advantages_std` | `train/advantages_mean`, `train/advantages_std` | Statistics of the advantages before normalization |
| `val_mean`, `val_std` | `train/value_mean`, `train/value_std` | Statistics of the critic output: the value scale the agent believes in |
| `rew_mean`, `rew_sum` | `train/reward_mean`, `train/reward_sum` | Rewards of this rollout, after `--reward-scale` |
| `trajectory_length` | `train/trajectory_length` | Steps in this update. Below `--rollout-length` when the episode ended early |
| `reward_<name>_sum`, `reward_<name>_mean` | `train/reward_<name>_sum`, `train/reward_<name>_mean` | Rollout total and mean of each reward term |
| `update`, `is_terminal` | not sent | Global update number, and whether the rollout ended the episode. Local only |
| `advantages`, `values`, `rewards`, `entropies` | not sent | Raw arrays, local only, and only with `--log-update-arrays` |

## Reward components

Every term of the reward is logged under its own name, so you can see which one dominates.

- `--reward-mode legacy` has one component, `legacy`.
- `--reward-mode shaped` has `progress`, `target_speed`, `route_penalty`, `time_penalty`, `goal_bonus`, `collision_penalty`, `offroute_penalty`, `lane_invasion_penalty`, and `total`. `total` is the sum after clipping. It equals `episode/reward`, so the episode view does not send it again.

The formulas are in `docs/environment.md`. A term added to the shaped reward appears in the logs without further changes.

## Resource values

Only with `--log-resources`, sampled every `--log-resources-interval` seconds. Needs `psutil`, and `pynvml` for the GPU values (`pip install -e ".[resources]"`).

| Field | W&B metric | Meaning |
|---|---|---|
| `proc_cpu_percent` | `system/proc_cpu_percent`, per worker `worker_<id>/system/proc_cpu_percent` | CPU usage of the logging process |
| `proc_rss_gb` | `system/proc_rss_gb`, per worker `worker_<id>/system/proc_rss_gb` | RAM of the logging process. Steady growth means a leak |
| `gpus[n].util_percent` | `system/gpu<n>_util_percent` | GPU utilization, sampled by the main process |
| `gpus[n].mem_used_gb` | `system/gpu<n>_mem_used_gb` | GPU memory in use, sampled by the main process |

## Timing values

`timing.jsonl` only, not sent to W&B. Each record holds `avg_ms`, `count`, and `total_s` for the phases of the worker loop since the previous record: `sync`, `env_reset`, `forward`, `env_step`, `loss_compute`, `backward`, `optim_update`, `checkpoint_save`. The same numbers are printed as a `[TIMING]` line. Look here when steps per second drop.

A record is written when the global update counter reaches a multiple of `--diag-log-interval`, or at an episode end once `--diag-log-wall-s` seconds have passed since the worker's last record.

## Events

Single occurrences, not periodic measurements. Workers send them through the queue, and the telemetry process is the only writer of `logs/events.jsonl`. The full payload (error message, checkpoint path, layer names) stays local. W&B receives only a running counter for the six health events. The counter starts from 0 in every session.

| Event | Emitted when | W&B counter |
|---|---|---|
| `training_start` | A session begins: worker count, device map, resume flag | |
| `training_end` | A session ends: steps, elapsed time, restart counts, error | |
| `worker_start` | A worker process is started at the beginning of a session | |
| `worker_restart` | The supervisor found a dead worker | `health/worker_restarts_total` |
| `worker_give_up` | A worker reached `--max-restarts-per-worker` | `health/workers_given_up_total` |
| `worker_crash` | Unhandled exception inside a worker | `health/worker_crashes_total` |
| `rollback` | Rapid crashes: the global network was restored from a checkpoint (`success` tells whether it worked) | `health/rollbacks_total` |
| `nan_gradient` | NaN in the gradients: the update was skipped | `health/nan_updates_total` |
| `camera_timeout` | No camera frame in time: the episode was dropped | `health/camera_timeouts_total` |
| `crash_recovery` | CARLA call timed out: the worker reconnects | |
| `checkpoint_save` | A periodic checkpoint was written | |
| `wandb_unavailable` | W&B was not turned off, and the package is missing or `WANDB_API_KEY` is empty | |
| `wandb_init_failed`, `wandb_metric_setup_failed` | W&B startup failed. The run continues with local logs | |
| `final_summary` | Last record of a session: final counters, best reward, elapsed time, health counts, `queue_drops`, `wandb_errors` | copied to the W&B summary |

`final_summary` tells whether the W&B view was complete. `queue_drops` counts records dropped because the queue was full. `wandb_errors` counts failed W&B calls, which are swallowed so they cannot interrupt training. Both are written with `--no-wandb` too.

## Weights & Biases

W&B is on when three things hold: the `wandb` package is installed, `WANDB_API_KEY` is set, and `--no-wandb` is not passed.

- Project, entity, and run name come from `--wandb-project`, `--wandb-entity`, `--wandb-run-name`, or from `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`. The default run name is `a3c-<N>w-<W&B run id>`.
- Every metric is plotted against `global_step`.
- The run id is stored in `wandb_run_id.txt`. `--resume` continues the same W&B run. `--init-from` starts a new one.
- Only the telemetry process imports `wandb`, so a W&B problem cannot stop a worker.

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--log-steps` | off | Write `steps.jsonl` |
| `--log-update-arrays` | off | Keep the raw per-step arrays in update records |
| `--log-resources` | off | Sample CPU, RAM, and GPU usage |
| `--log-resources-interval` | 10 s | Time between resource samples |
| `--diag-log-interval` | 100 | Updates between timing records. `0` turns this trigger off |
| `--diag-log-wall-s` | 60 s | Longest time between timing records. `0` turns this trigger off |
| `--wandb-project`, `--wandb-entity`, `--wandb-run-name` | from `.env` | W&B names |
| `--no-wandb` | off | Turns W&B off |
