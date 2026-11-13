# Logging

What gets logged, where, and how often. Two independent destinations:

- **Local JSONL** under `<run directory>/logs/`. The complete record of the run: every value exactly as training produced it. This is the source of truth.
- **Weights & Biases**, when enabled. The values that matter for analysis, one record forwarded as one point. Nothing is averaged, batched, or downsampled.

**One name per value.** A local field has the name of its W&B metric. A value of one worker is `worker_<id>/<name>` in W&B and `<name>` in `logs/worker_<id>/*.jsonl`, where the file and the `worker` field already name the worker; a run-wide value (`episode/*`, `global/*`, `train/lr`, ...) has the same name in both. Everything sent to W&B is also in the local files; the local files hold more.

Each training process writes its own files, so local logging needs no locking. Records for W&B and all events travel through one queue to a single telemetry process. That path is best effort: when the queue (10 000 records) is full, W&B loses points and the local files stay complete.

`NaN` and infinities become `null` in JSON and are not sent to W&B.

## Quick map: where do I find ...

Fields are in `logs/worker_<id>/episodes.jsonl` unless noted. Every field is explained in "Episode values".

| Question | Field (local) | W&B |
|---|---|---|
| Reward of an episode | `reward` | chart `worker_<id>/reward` |
| Mean reward, last 100 episodes of a worker | `reward_mean_100` | chart `worker_<id>/reward_mean_100` |
| Mean reward of the run (mean of the worker means) | `global/reward_mean_100` | chart `global/reward_mean_100` |
| Distance from the target at episode end | `goal_distance` | chart `worker_<id>/goal_distance` |
| Mean distance from the target, last 100 episodes (worker / run) | `goal_distance_mean_100` / `global/goal_distance_mean_100` | charts of the same names |
| Min and max reward and distance of a worker, whole run | `reward_min`, `reward_max`, `goal_distance_min`, `goal_distance_max` | run summary `worker_<id>/reward_min`, ... |
| Min and max of the run (lowest and highest worker value) | `global/reward_min`, `global/reward_max`, `global/goal_distance_min`, `global/goal_distance_max` | run summary, the same names |
| Global steps and time to reach a reward | `logs/events.jsonl`: `reward_threshold_reached`; `final_summary` | run summary `threshold/<signal>/<label>/global_step`, `.../elapsed_training_s` |
| Training time | `global/elapsed_training_s` | chart `global/elapsed_training_s` |
| Reward of every step | `reward` in `steps.jsonl` (with `--log-steps`) | not sent, by design |

Each worker writes only its own file, so run-wide values are repeated in
every worker's records. For one run-wide series, merge the
`worker_*/episodes.jsonl` files and sort by `global_step`.

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

Every record carries `kind` (record type), `ts` (timestamp), `worker` (source worker, `-1` for the main process), and `global_step` where it applies. A resumed run appends to the same files. The `training_start` and `training_end` events mark the sessions.

The console output of training is `a3c_training.log` in the run directory. Its lines are tagged, for example `[BEST]`, `[THRESHOLD]`, `[TIMING]`, `[SAVE]`, `[RESTART]`, `[ROLLBACK]`, `[NaN]`, and a final `[BENCHMARK]` line with steps per second.

## Step, update, episode

Training writes on three cadences, from densest to sparsest:

- **step**: every `env.step()`. Only with `--log-steps`, only to `steps.jsonl`, never to W&B. Fields: `action`, `value`, `entropy`, `reward`, `done`, `step_in_ep`, `global_episode`, `local_t`, `speed_kmh`, `route_dist`, `goal_dist`, `maneuver`. It is the only view inside an episode and grows by gigabytes per hour with many workers, so it is off by default.
- **update**: every `--rollout-length` steps or at episode end, whichever comes first. Losses, gradients, and statistics of that one rollout.
- **episode**: at episode end. The outcome of the whole episode, plus the reward statistics at that moment (see "Reward statistics").

`timing` and `system` records follow their own clocks.

Only the reward the agent trains on is logged (`reward`, `train/reward_mean`, `train/reward_sum`). Its terms, the `legacy` value or the shaped components, are not logged anywhere.

## Reward statistics

[`reward_stats.py`](../reward_stats.py) keeps these values per worker, in
shared memory, and attaches them to each episode record:

- **window**: the episode reward and the final goal distance of the worker's
  last 100 episodes (`reward_mean_100`, `goal_distance_mean_100`),
- **run minimum and maximum** of both, over the whole run (`reward_min`,
  `reward_max`, `goal_distance_min`, `goal_distance_max`).

The run-wide `global/*` values are aggregates of the workers, so they show
training as a whole:

| Field | Value |
|---|---|
| `global/reward_mean_100`, `global/goal_distance_mean_100` | Mean of the worker means. A worker without a value does not count |
| `global/reward_min`, `global/goal_distance_min` | Lowest worker minimum |
| `global/reward_max`, `global/goal_distance_max` | Highest worker maximum |

Notes:

- Only completed episodes count. An episode cut off by shutdown or `--steps`
  is written with `"interrupted": true`, without reward statistics, is not
  sent to W&B, and cannot reach a threshold. An episode skipped after a
  camera timeout is never logged; only the `camera_timeout` event records it.
  A missing or infinite goal distance does not enter the distance values.
- The statistics survive worker restarts, are stored in `checkpoint.pth`, and
  are restored by `--resume` (`--init-from` starts them empty). A window
  holds fewer episodes at the start of a run; its mean then covers what
  exists. Episodes enter in the order they finish, while `episode/id` is
  assigned at episode start, so it can be out of order between workers.
- `--resume` with a different `--num-workers`: workers present in both runs
  keep their values, new ones start empty.
- An older checkpoint: one without the run minimum and maximum takes the
  reward ones from its best and worst values and the distance ones from the
  saved window. One from before `reward_stats` holds a single run-wide
  window, which has no worker; its statistics start empty.
- The statistics are read under a lock shared with the workers. If a killed
  worker still holds it, the final checkpoint and summary wait 5 s, print
  `[REWARD_STATS] lock busy; reading without it`, and read without the lock.

## Episode values

An interrupted episode stays local. The run minimum and maximum are
run-summary values (one per run, updated when it changes); the other sent
values are charts.

| Field | W&B | Meaning |
|---|---|---|
| `global_step` | x-axis | Global step at episode end. The x-axis of every chart |
| `episode/id` | chart | Episode counter shared by all workers |
| `reward` | chart `worker_<id>/reward` | Sum of rewards in this episode, before `--reward-scale` |
| `episode_length` | chart `worker_<id>/episode_length` | Decisions taken in the episode |
| `goal_distance` | chart `worker_<id>/goal_distance` | Distance to the goal at episode end. The main progress signal on failed episodes |
| `episode/reward_per_step` | chart | `reward / episode_length`, comparable across episode lengths |
| `episode/duration_s` | chart | Wall-clock seconds. A rising value usually means CARLA is degrading |
| `episode/success` | chart | 1 when the goal was reached, else 0 |
| `episode/max_speed_kmh` | chart | Highest speed reached |
| `episode/min_route_distance` | chart | Closest approach to the planned route |
| `episode/collisions` | chart | Collision events recorded by CARLA |
| `episode/action_<n>_fraction` | chart | Share of decisions that chose action `n`. A value near 1.0 means the policy collapsed onto one action |
| `reward_mean_100` | chart `worker_<id>/reward_mean_100` | Mean reward of this worker's last 100 episodes |
| `goal_distance_mean_100` | chart `worker_<id>/goal_distance_mean_100` | Mean goal distance of this worker's last 100 episodes |
| `reward_min`, `reward_max` | summary `worker_<id>/reward_min`, `worker_<id>/reward_max` | Lowest and highest episode reward of this worker in the run |
| `goal_distance_min`, `goal_distance_max` | summary `worker_<id>/goal_distance_min`, `worker_<id>/goal_distance_max` | Lowest and highest goal distance of this worker in the run |
| `global/reward_mean_100` | chart | Mean of the workers' `reward_mean_100` |
| `global/goal_distance_mean_100` | chart | Mean of the workers' `goal_distance_mean_100` |
| `global/reward_min`, `global/reward_max` | summary | Lowest worker minimum, highest worker maximum |
| `global/goal_distance_min`, `global/goal_distance_max` | summary | The same for the goal distance |
| `global/elapsed_training_s` | chart | Active training time at the end of this episode (see "Time") |
| `interrupted` | not sent | `true` only for an episode cut off by shutdown or `--steps`. Such a record has no reward statistics |
| `is_new_best`, `port` | not sent | `is_new_best`: the reward beats every earlier episode of the run (the `[BEST]` line) |

## Reward thresholds

A threshold answers "how many global steps and how much training time did the
run need to reach reward X", that is, how long the agent needed to learn the
scenario. Set values on the command line (default: none):

```bash
./examples/docker/train.sh -w 4 --reward-thresholds 100 150 200
```

Each value is watched through two signals per worker and one run-wide:

| Signal | Reached when |
|---|---|
| `worker_<id>/episode_reward` | one episode's `reward` of that worker is `>=` the threshold |
| `worker_<id>/reward_mean_100` | that worker's `reward_mean_100` is `>=` the threshold, once its window is full |
| `reward_mean_100` | `global/reward_mean_100` is `>=` the threshold, once every worker's window is full |

`reward_mean_100` answers "when did the agent learn it"; one lucky episode
already fires `episode_reward`. Pick thresholds near the reward of an episode
that reaches the goal: thresholds the first random episodes already pass say
nothing. Each (signal, threshold) pair fires **at most once per run**, at the
first episode that satisfies it. Values must be finite, and two values that
share the first six significant digits (1000000 and 1000001) are rejected; an
invalid value stops the launch with `ERROR: --reward-thresholds: ...` before
the run directory is created. `args.txt`, `metadata.json`, and the W&B config
show the sorted, deduplicated list.

Where the result lands:

1. **`logs/events.jsonl`**: a `reward_threshold_reached` event when it
   happens, with `signal`, `threshold`, `value`, `global_step`,
   `global_episode`, `elapsed_training_s`, and `worker` of the episode that
   reached it. Events are best effort: one is dropped after a 1 s wait on a
   full telemetry queue, and a killed worker can lose events it has not
   handed over.
2. **W&B run summary**, updated at once: `threshold/<signal>/<label>/reached`
   plus `.../global_step`, `.../elapsed_training_s`, `.../global_episode`,
   `.../value`; per worker `threshold/worker_<id>/<signal>/...`. The worker
   is in the key, so the summary has no `worker` field. Summary values are
   not charts: in W&B they are on the run's **Overview** page (Summary), and
   they can be added as columns of the **runs table** to compare runs.
3. **`final_summary`** (event and W&B summary), built from shared memory, so
   it is the authoritative copy: the same keys for every configured pair. A
   pair never reached has `reached = false` and `null` for the other four
   fields, which also overwrites stale values in a resumed W&B run.

`<label>` is the threshold with `-` written as `neg`, `.` as `p`, and `+`
dropped (after `{:g}`), because W&B reads a dot in a metric name as nesting:
`0.5` is `0p5`, `-50` is `neg50`, `100` is `100`, `1000000` is `1e06`. The
`threshold` field of the event stays the plain number.

```json
{"signal": "reward_mean_100", "threshold": 150.0, "value": 150.2, "global_step": 412000, "global_episode": 3350, "elapsed_training_s": 5321.4, "worker": 2, "event": "reward_threshold_reached", "kind": "event", "ts": "2026-01-01T14:03:22.118"}
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
| `global/elapsed_training_s` (episode records), `elapsed_training_s` (threshold events and summary), `cumulative_elapsed_s` (`training_end`) | Active training time: counted from the moment all CARLA servers answer, **summed over resumed sessions**; excludes the server wait and the time between sessions. `--resume` continues from the value stored in `checkpoint.pth`, which is right even after a hard kill. Only a checkpoint from before this field falls back to `resume_state.json` or the last `training_end` event, which exist only after a graceful exit. |
| `ts` (every record) | Wall-clock time of the machine. Includes the CARLA server wait and gaps between sessions, so it is not training time. |
| W&B relative time (`_runtime`) | Measured from the start of the W&B run, which begins before the servers are ready. Not the training clock. |

## Update values

One record per optimizer update, covering that one rollout. The schedules all workers share are sent as they are; the values of one worker's rollout as `worker_<id>/train/<name>`.

| Field | W&B | Meaning |
|---|---|---|
| `train/pi_loss` | chart `worker_<id>/train/pi_loss` | Policy loss. Noisy by nature: watch the trend |
| `train/v_loss` | chart `worker_<id>/train/v_loss` | Value loss, already multiplied by `--value-loss-coef` |
| `train/total_loss` | chart `worker_<id>/train/total_loss` | The optimized objective |
| `train/entropy` | chart `worker_<id>/train/entropy` | Policy entropy. A fall towards 0 means exploration is dying |
| `train/advantages_mean`, `train/advantages_std` | charts `worker_<id>/train/...` | Statistics of the advantages before normalization |
| `train/value_mean`, `train/value_std` | charts `worker_<id>/train/...` | Statistics of the critic output: the value scale the agent believes in |
| `train/reward_mean` | chart `worker_<id>/train/reward_mean` | Mean reward of this rollout as the loss uses it, after `--reward-scale` |
| `train/entropy_coef` | chart | Entropy coefficient at this step |
| `train/lr` | chart | Learning rate at this step |
| `train/gradient_norm_pre_clip` | not sent | Gradient norm before clipping. Compare with `--max-grad-norm` |
| `train/gradient_norm` | not sent | Gradient norm after clipping |
| `train/grad_clipped` | not sent | 1 when clipping changed this update. Its average is the clipping rate |
| `train/reward_sum` | not sent | Sum of the rewards of this rollout, after `--reward-scale` |
| `train/trajectory_length` | not sent | Steps in this update. Below `--rollout-length` when the episode ended early |
| `update`, `is_terminal` | not sent | Global update number, and whether the rollout ended the episode |
| `advantages`, `values`, `rewards`, `entropies` | not sent | Raw arrays, only with `--log-update-arrays` |

## Resource values

Only with `--log-resources`, sampled every `--log-resources-interval` seconds. Needs `psutil`, and `pynvml` for the GPU values (`pip install -e ".[resources]"`).

| Field | W&B | Meaning |
|---|---|---|
| `system/proc_cpu_percent` | chart; a worker's as `worker_<id>/system/proc_cpu_percent` | CPU usage of the logging process |
| `system/proc_rss_gb` | chart; a worker's as `worker_<id>/system/proc_rss_gb` | RAM of the logging process. Steady growth means a leak |
| `gpus[n].util_percent` | not sent | GPU utilization, sampled by the main process. W&B records GPU usage itself (`system/gpu.<n>.*`) |
| `gpus[n].mem_used_gb` | not sent | GPU memory in use, sampled by the main process |

## Timing values

`timing.jsonl` only, not sent to W&B. Each record holds `avg_ms`, `count`, and `total_s` for the phases of the worker loop since the previous record: `sync`, `env_reset`, `forward`, `env_step`, `loss_compute`, `backward`, `optim_update`, `checkpoint_save`. The same numbers are printed as a `[TIMING]` line. Look here when steps per second drop.

A record is written when the global update counter reaches a multiple of `--diag-log-interval`, or at an episode end once `--diag-log-wall-s` seconds have passed since the worker's last record.

## Events

Single occurrences, not periodic measurements. Workers send them through the queue, and the telemetry process is the only writer of `logs/events.jsonl`. Events stay local: W&B receives only the run-summary keys of `reward_threshold_reached` and of `final_summary`. The six health events are counted into `final_summary`; the counters start from 0 in every session.

| Event | Emitted when | Also in |
|---|---|---|
| `training_start` | A session begins: worker count, device map, resume flag | |
| `training_end` | A session ends: steps, elapsed time, restart counts, error | |
| `worker_start` | A worker process is started at the beginning of a session | |
| `worker_restart` | The supervisor found a dead worker | `final_summary` counter `health/worker_restarts_total` |
| `worker_give_up` | A worker reached `--max-restarts-per-worker` | `final_summary` counter `health/workers_given_up_total` |
| `worker_crash` | Unhandled exception inside a worker | `final_summary` counter `health/worker_crashes_total` |
| `rollback` | Rapid crashes: the global network was restored from a checkpoint (`success` tells whether it worked) | `final_summary` counter `health/rollbacks_total` |
| `nan_gradient` | NaN in the gradients: the update was skipped | `final_summary` counter `health/nan_updates_total` |
| `camera_timeout` | No camera frame in time: the episode was dropped | `final_summary` counter `health/camera_timeouts_total` |
| `crash_recovery` | CARLA call timed out: the worker reconnects | |
| `reward_threshold_reached` | A configured reward threshold was reached for the first time (see "Reward thresholds") | W&B run summary, `threshold/...` keys |
| `checkpoint_save` | A periodic checkpoint was written | |
| `wandb_unavailable` | W&B was not turned off, and the package is missing or `WANDB_API_KEY` is empty | |
| `wandb_init_failed`, `wandb_metric_setup_failed` | W&B startup failed. The run continues with local logs | |
| `final_summary` | Last record of a session (keys below) | W&B run summary, without the counters |

`final_summary` holds, under the W&B summary names:

- `final_global_step`, `final_global_update`, `session_elapsed_s`.
- The run minimum and maximum, per worker and of the run
  (`worker_<id>/reward_min`, ..., `global/goal_distance_max`).
- `threshold/<signal>/<label>/...` for every configured threshold (see
  "Reward thresholds").
- Local only: `queue_drops`, `wandb_errors` (below), and the `health/*`
  counters.

`final_summary` tells whether the W&B view was complete. `queue_drops` counts records dropped because the queue was full. `wandb_errors` counts failed W&B calls, which are swallowed so they cannot interrupt training. Both are written with `--no-wandb` too.

## Weights & Biases

W&B is on when three things hold: the `wandb` package is installed, `WANDB_API_KEY` is set, and `--no-wandb` is not passed.

- Project, entity, and run name come from `--wandb-project`, `--wandb-entity`, `--wandb-run-name`, or from `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`. The default run name is `a3c-<N>w-<W&B run id>`.
- Every chart is plotted against `global_step`.
- Each value has one place: `worker_<id>/*` for one worker, `global/*` for aggregates over the workers and the training time, `train/*` for the schedules all workers share, `episode/*` for the other values of an episode. The run minimum and maximum and the thresholds are run-summary values, not charts: see the run's Overview page or the runs table.
- The W&B config holds all training arguments and `algorithm: A3C`.
- Not sent to W&B, by design, to limit the data sent with many workers: the reward of every step, the gradient norm and clipping, the rollout reward sum and length, the health events and counters, and the GPU samples of `--log-resources`. They stay in the local files. Gradient histograms are not logged at all.
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
