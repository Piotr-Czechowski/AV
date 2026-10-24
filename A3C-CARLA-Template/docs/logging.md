# Logging

What gets logged, where, and how often. Two independent destinations:

- **Local JSONL** under `<run directory>/logs/`. The complete record of the run: every value exactly as training produced it. This is the source of truth.
- **Weights & Biases**, when enabled. A selected subset, one record forwarded as one point. Nothing is averaged, batched, or downsampled.

Each training process writes its own files, so local logging needs no locking. Records for W&B and all events travel through one queue to a single telemetry process. That path is best effort: when the queue (10 000 records) is full, W&B loses points and the local files stay complete.

`NaN` and infinities become `null` in JSON and are not sent to W&B.

## Quick map: where do I find ...

Local fields are in `logs/worker_<id>/episodes.jsonl` unless noted; `<id>` is
the worker number. Every field is explained in "Episode values".

| Question | Local field | W&B |
|---|---|---|
| Mean reward, last 100 episodes (run-wide / one worker) | `global_mean_reward_100` / `local_mean_reward` | `global/reward_mean_100` / `worker_<id>/reward_mean_100` |
| Mean reward, last 100 x workers episodes | `global_mean_reward` | `global/reward_mean` |
| Distance from the target | `goal_dist` at episode end (per step in `steps.jsonl` with `--log-steps`) | `episode/goal_distance`, `worker_<id>/goal_distance` |
| Min and max reward of a worker | `local_reward_min_100`, `local_reward_max_100` (last 100 episodes); `local_worst_reward`, `local_best_reward` (all time) | `worker_<id>/reward_min_100`, `.../reward_max_100`, `.../worst_reward`, `.../best_reward` |
| Min and max reward of the run | `global_reward_min(_100)`, `global_reward_max(_100)`, `worst_reward`, `best_reward` | `global/reward_min(_100)`, `global/reward_max(_100)`, `global/worst_reward`, `global/best_reward` |
| Global steps and time to reach a reward | `logs/events.jsonl`: `reward_threshold_reached`; `final_summary` | run summary: `threshold/<signal>/<label>/global_step`, `.../elapsed_training_s` |
| Training time | `elapsed_training_s` | `global/elapsed_training_s` |

Each worker writes only its own file, so run-wide values are repeated in
every worker's records. For one run-wide series, merge the
`worker_*/episodes.jsonl` files and sort by `global_t`.

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
- **episode**: at episode end. The outcome of the whole episode, plus the reward statistics at that moment (see "Reward statistics").

`timing` and `system` records follow their own clocks.

## Reward statistics

[`reward_stats.py`](../reward_stats.py) keeps ring buffers of **episode rewards** (`total_reward`)
in shared memory and attaches the results to each episode record:

- **worker window**: the last 100 episodes of one worker (`local_*_100`),
- **run window**: the last 100 episodes of all workers (`global_*_100`),
- **long run window**: the last 100 x `--num-workers` episodes of all workers
  (`global_mean_reward`, `global_reward_min`, `global_reward_max`; the same
  as the run window with one worker),
- **all time**: best and worst reward, per worker and for the run.

Notes:

- Only completed episodes count. An episode cut off by shutdown or `--steps`
  is written with `"interrupted": true`, without reward statistics, is not
  sent to W&B, and cannot reach a threshold. An episode skipped after a
  camera timeout is never logged; only the `camera_timeout` event records it.
- Windows survive worker restarts, are stored in `checkpoint.pth`, and are
  restored by `--resume` (`--init-from` starts them empty). A window holds
  fewer episodes at the start of a run; its statistics then cover what
  exists. Episodes enter in the order they finish, while `global_episode` is
  assigned at episode start, so it can be out of order between workers.
- `--resume` with a different `--num-workers`: workers present in both runs
  keep their windows, new ones start empty, and the long run window keeps its
  most recent episodes.
- A checkpoint from before this feature holds one run-wide window.
  `--resume` rebuilds the run and long run windows and `best_reward` from it.
  Per-worker windows, best and worst values start empty, and the run-wide
  `worst_reward` is the lowest reward in the rebuilt window.
- The statistics are read under a lock shared with the workers. If a killed
  worker still holds it, the final checkpoint and summary wait 5 s, print
  `[REWARD_STATS] lock busy; reading without it`, and read without the lock.

## Episode values

All of them reach W&B, except for an interrupted episode, which stays local.

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
| `goal_dist` | `episode/goal_distance`, `worker_<id>/goal_distance` | Distance to the goal at episode end. The main progress signal on failed episodes |
| `collisions` | `episode/collisions` | Collision events recorded by CARLA |
| `local_mean_reward` | `worker_<id>/reward_mean_100` | Mean reward of this worker's last 100 episodes |
| `local_reward_min_100`, `local_reward_max_100` | `worker_<id>/reward_min_100`, `worker_<id>/reward_max_100` | Lowest and highest reward in that window |
| `local_best_reward`, `local_worst_reward` | `worker_<id>/best_reward`, `worker_<id>/worst_reward` | Highest and lowest reward of this worker, all time |
| `global_mean_reward_100`, `global_reward_min_100`, `global_reward_max_100` | `global/reward_mean_100`, `global/reward_min_100`, `global/reward_max_100` | Mean, lowest, and highest reward of the run's last 100 episodes |
| `global_mean_reward`, `global_reward_min`, `global_reward_max` | `global/reward_mean`, `global/reward_min`, `global/reward_max` | Mean, lowest, and highest reward of the run's last `100 * workers` episodes |
| `best_reward`, `worst_reward` | `global/best_reward`, `global/worst_reward` | Highest and lowest single-episode reward of the run so far |
| `elapsed_training_s` | `global/elapsed_training_s` | Active training time at the end of this episode (see "Time") |
| `action_counts` | `episode/action_<n>_fraction` | How often each action was chosen, sent as a fraction. A value near 1.0 means the policy collapsed onto one action |
| `reward_components` | `episode/reward_<name>` | Episode total of each reward term (see below) |
| `interrupted` | not sent | `true` only for an episode cut off by shutdown or `--steps`. Such a record has no reward statistics |
| `is_new_best`, `port` | not sent | Local only |

## Reward thresholds

A threshold answers "how many global steps and how much training time did the
run need to reach reward X". Set values on the command line (default: none):

```bash
./examples/docker/train.sh -w 4 --reward-thresholds 0 50 100
```

Each value is watched through three signals:

| Signal | Reached when |
|---|---|
| `episode_reward` | one episode's `total_reward` is `>=` the threshold |
| `reward_mean_100` | the mean of the run window (last 100 episodes) is `>=` the threshold |
| `reward_mean` | the mean of the long run window (last 100 x workers) is `>=` the threshold |

The two means count only once their window is full. Each (signal, threshold)
pair fires **at most once per run**, at the first episode that satisfies it.
Values must be finite, and two values that share the first six significant
digits (1000000 and 1000001) are rejected; an invalid value stops the launch
with `ERROR: --reward-thresholds: ...` before the run directory is created.
`args.txt`, `metadata.json`, and the W&B config show the sorted, deduplicated
list.

Where the result lands:

1. **`logs/events.jsonl`**: a `reward_threshold_reached` event when it
   happens. Events are best effort: one is dropped after a 1 s wait on a full
   telemetry queue, and a killed worker can lose events it has not handed
   over. `global_t`, `elapsed_training_s`, and `worker` are those of the
   episode that reached the threshold.
2. **W&B run summary**, updated at once: `threshold/<signal>/<label>/reached`
   plus `.../global_step`, `.../elapsed_training_s`, `.../global_episode`,
   `.../value`, `.../worker`.
3. **`final_summary`** (event and W&B summary), built from shared memory, so
   it is the authoritative copy: the same keys for every configured pair. A
   pair never reached has `reached = false` and `null` for the other five
   fields, which also overwrites stale values in a resumed W&B run.

`<label>` is the threshold with `-` written as `neg`, `.` as `p`, and `+`
dropped (after `{:g}`), because W&B reads a dot in a metric name as nesting:
`0.5` is `0p5`, `-50` is `neg50`, `100` is `100`, `1000000` is `1e06`. The
`threshold` field of the event stays the plain number.

```json
{"signal": "reward_mean_100", "threshold": 50.0, "value": 51.2, "global_t": 412000, "global_episode": 3350, "elapsed_training_s": 5321.4, "worker": 2, "event": "reward_threshold_reached", "kind": "event", "ts": "2026-01-01T14:03:22.118"}
```

On `--resume`, reached thresholds come back from the checkpoint and do not
fire again, and new times continue the clock stored in the checkpoint (see
"Time"). A hit reached after the last checkpoint and lost in a crash fires
again and appears twice in `events.jsonl`. Hits for thresholds no longer
configured are not restored; they stay only in the old events. A threshold
that is new on resume fires at the first episode that satisfies it.
`--reward-thresholds` is compared with the previous session's `training_args`
in `resume_state.json`; a difference prints a `[RESUME] WARNING`.

## Time

| Clock | Meaning |
|---|---|
| `elapsed_training_s` (episode records, threshold events, `global/elapsed_training_s`, `cumulative_elapsed_s` in `training_end`) | Active training time: counted from the moment all CARLA servers answer, **summed over resumed sessions**; excludes the server wait and the time between sessions. `--resume` continues from the value stored in `checkpoint.pth`, which is right even after a hard kill. Only a checkpoint from before this field falls back to `resume_state.json` or the last `training_end` event, which exist only after a graceful exit. |
| `ts` (every record) | Wall-clock time of the machine. Includes the CARLA server wait and gaps between sessions, so it is not training time. |
| W&B relative time (`_runtime`) | Measured from the start of the W&B run, which begins before the servers are ready. Not the training clock. |

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

The formulas are in [`docs/environment.md`](environment.md). A term added to the shaped reward appears in the logs without further changes.

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

Single occurrences, not periodic measurements. Workers send them through the queue, and the telemetry process is the only writer of `logs/events.jsonl`. The full payload (error message, checkpoint path, layer names) stays local. W&B receives only a running counter for the six health events, plus the run-summary keys of `reward_threshold_reached`. The health counters start from 0 in every session.

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
| `reward_threshold_reached` | A configured reward threshold was reached for the first time (see "Reward thresholds") | W&B run summary `threshold/...` keys |
| `checkpoint_save` | A periodic checkpoint was written | |
| `wandb_unavailable` | W&B was not turned off, and the package is missing or `WANDB_API_KEY` is empty | |
| `wandb_init_failed`, `wandb_metric_setup_failed` | W&B startup failed. The run continues with local logs | |
| `final_summary` | Last record of a session (keys below) | copied to the W&B summary |

`final_summary` holds:

- `final_global_step`, `final_global_update`, `session_elapsed_s`.
- `best_reward`, `worst_reward`: all workers, all time.
- `final_global_mean_reward`, `final_reward_min`, `final_reward_max`: the
  last 100 x workers episodes at the end of the run.
- `final_reward_mean_100`, `final_reward_min_100`, `final_reward_max_100`:
  the last 100 episodes at the end of the run.
- `threshold/<signal>/<label>/...` for every configured threshold (see
  "Reward thresholds").
- `queue_drops`, `wandb_errors` (below) and the `health/*` counters.

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
| `--reward-thresholds` | none | Reward values whose first crossing is logged (see "Reward thresholds") |
