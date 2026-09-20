# What gets logged, where, and how often

Two independent destinations:

- **Local JSONL** under `<run_dir>/logs/`. Complete record of the run: every
  value exactly as training produced it. This is the source of truth.
- **Weights & Biases**, when enabled. A curated subset, forwarded one record to
  one point — never averaged, batched, or downsampled. Best effort: if the
  internal queue fills up, W&B loses points while local files stay complete.

`NaN` and infinities become `null` in JSON (strict JSON cannot hold them) and
are simply not sent to W&B.

## Files

| File | Written by | One line per |
|---|---|---|
| `logs/worker_<id>/episodes.jsonl` | each worker | finished episode |
| `logs/worker_<id>/updates.jsonl` | each worker | optimizer update |
| `logs/worker_<id>/timing.jsonl` | each worker | diagnostic window (`--diag-log-interval` updates or `--diag-log-wall-s` seconds) |
| `logs/worker_<id>/steps.jsonl` | each worker | environment step — **only with `--log-steps`** |
| `logs/worker_<id>/system.jsonl` | each worker | resource sample, every `--monitor-interval` s |
| `logs/system.jsonl` | main process | node-wide resource sample, every `--monitor-interval` s |
| `logs/events.jsonl` | telemetry process only | lifecycle or health event |
| `logs/metadata.json` | main process | written once at startup |

Every record carries `kind` (record type), `ts` (timestamp), `worker` (source
worker, `-1` for run-level) and, where meaningful, `global_t`.

## Step, update, episode

Training writes on three cadences, from densest to sparsest:

- **step** — every `env.step()`. Written only with `--log-steps`, only to
  `steps.jsonl`, never to W&B: `action`, `value`, `entropy`, `reward`, `done`,
  `step_in_ep`, plus `speed_kmh`, `route_dist`, `goal_dist`, `maneuver` and
  `reward_components` when the environment reports them. The only view inside
  an episode, and gigabytes per hour with 16 workers — hence off by default.
- **update** — every `--rollout-length` steps (default 20) *or* at episode end,
  whichever comes first. Losses, gradients, and statistics of that one rollout.
  This is the dense W&B path: roughly 16 points per second with 16 workers.
- **episode** — at episode end. Outcome of the whole episode, plus the
  run-wide bests and rolling means as they stood at that moment.

`timing` and `system` records follow their own clocks and are independent of
all three.

## Episode values

Written on every episode end, aggregating the whole episode. All of them reach
W&B.

| Field | W&B metric | Meaning |
|---|---|---|
| `global_episode` | `episode/id` | Episode counter shared by all workers |
| `global_t` | `global_step` | Total environment steps across the run; the x-axis of every chart |
| `total_reward` | `episode/reward`, `worker_<id>/reward` | Sum of rewards collected in this episode |
| `mean_reward` | `episode/reward_per_step` | `total_reward / steps` — reward density, comparable across episode lengths |
| `steps` | `episode/length`, `worker_<id>/episode_length` | Decisions taken before the episode ended |
| `duration_s` | `episode/duration_s` | Wall-clock seconds. Rising values usually mean CARLA is degrading |
| `reached_goal` | `episode/success` | 1 when the goal was reached, else 0 |
| `max_speed_kmh` | `episode/max_speed_kmh` | Fastest speed reached |
| `min_route_dist` | `episode/min_route_distance` | Closest approach to the reference route |
| `goal_dist` | `episode/goal_distance` | Distance to the goal at episode end. The main progress signal on failed episodes |
| `collisions` | `episode/collisions` | Collision events recorded by CARLA |
| `local_mean_reward` | `worker_<id>/reward_mean_100` | This worker's mean reward over its last 100 episodes |
| `global_mean_reward` | `global/reward_mean` | Run-wide rolling mean reward |
| `best_reward` | `global/best_reward` | Best single-episode reward so far |
| `action_counts` | `episode/action_<n>_fraction` | How often each discrete action was chosen. Sent as a fraction; a value near 1.0 means the policy collapsed onto one action |
| `reward_components` | `episode/reward_<name>` | Per-episode total of each reward term (see below) |
| `is_new_best`, `port` | — | Local only |

## Update values

Written on every optimizer update, covering that one rollout rather than the
episode around it. All of them reach W&B twice: once as `train/<metric>` (all
workers interleaved) and once as `worker_<id>/train/<metric>` (that worker
alone).

| Field | W&B metric | Meaning |
|---|---|---|
| `pi_loss` | `train/pi_loss` | Policy loss. Noisy by nature; watch the trend, not single points |
| `v_loss` | `train/v_loss` | Value-function loss — how badly the critic predicts returns |
| `total_loss` | `train/total_loss` | Combined optimized objective |
| `gradient_norm` | `train/gradient_norm` | Gradient norm **after** clipping |
| `gradient_norm_pre_clip` | `train/gradient_norm_pre_clip` | Gradient norm **before** clipping. Compare with `--max-grad-norm` |
| `grad_clipped` | `train/grad_clipped` | 1 when clipping actually bound this update. Average it in W&B to get the clipping rate |
| `ent_mean` | `train/entropy` | Policy entropy. Falling towards 0 means exploration is dying |
| `entropy_coef` | `train/entropy_coef` | Current entropy bonus weight (annealed, see `--beta-*`) |
| `lr` | `train/lr` | Current learning rate |
| `advantages_mean`, `advantages_std` | `train/advantages_mean`, `train/advantages_std` | Advantage statistics of this rollout. With `--no-normalize-advantages` off, mean sits near 0 |
| `val_mean`, `val_std` | `train/value_mean`, `train/value_std` | Critic output statistics — the value scale the agent believes in |
| `rew_mean`, `rew_sum` | `train/reward_mean`, `train/reward_sum` | Scaled rewards inside this rollout |
| `trajectory_length` | `train/trajectory_length` | Transitions in this update; below `--rollout-length` means the episode ended early |
| `reward_<name>_sum`, `reward_<name>_mean` | `train/reward_<name>_sum`, `train/reward_<name>_mean` | Per-rollout totals and means of each reward term |
| `update`, `is_terminal` | — | Local only |
| `advantages`, `values`, `rewards`, `entropies` | — | Raw arrays, local only, and only with `--log-update-arrays` |

## Reward components

Present only with `--reward-mode shaped`. Each term is logged separately so you
can see which one dominates.

| Component | Meaning |
|---|---|
| `progress` | Progress made along the route |
| `target_speed` | Closeness to the target speed |
| `route_penalty` | Penalty for drifting off the reference route |
| `time_penalty` | Constant per-step cost, pushes for shorter episodes |
| `goal_bonus` | One-off bonus for reaching the goal |
| `collision_penalty` | One-off penalty per new collision |
| `offroute_penalty` | Penalty for exceeding the off-route threshold |
| `lane_invasion_penalty` | Penalty per lane invasion |
| `total` | Sum after clipping. Duplicates `episode/reward`, so it is not sent separately |

## Resource values

Sampled every `--monitor-interval` seconds (default 10).

| Field | W&B metric | Meaning |
|---|---|---|
| `system.cpu_percent_mean` | `system/cpu_percent_mean` | CPU load averaged across cores |
| `system.cpu_percent_max` | `system/cpu_percent_max` | Busiest single core |
| `system.mem_percent` | `system/mem_percent` | RAM in use |
| `system.mem_available_gb` | `system/mem_available_gb` | RAM left. Falling steadily means a leak |
| `system.swap_used_gb` | `system/swap_used_gb` | Swap in use. Anything above 0 usually means trouble |
| `gpus[n].util_percent` | `system/gpu<n>_util_percent` | GPU utilization |
| `gpus[n].mem_used_gb` | `system/gpu<n>_mem_used_gb` | GPU memory in use |
| `carla.*`, `mem_used_gb`, `cpu_freq_mhz`, `total_procs`, `total_rss_gb` | — | Local only: CARLA process stats, PIDs, extra counters |

Per-worker resource usage goes to `logs/worker_<id>/system.jsonl` and never
reaches W&B.

## Timing values

Written to `logs/worker_<id>/timing.jsonl` only — **not** sent to W&B. Each
record holds `avg_ms`, `count`, and `total_s` for the phases of the training
loop (`forward`, `backward`, `optim_update`, `sync`, `env_step`,
`checkpoint_save`). This is where you look when steps-per-second drops.

## Events

Discrete occurrences rather than periodic measurements. Written to the single
shared `logs/events.jsonl` by the telemetry process — workers never write it
themselves. Full payload (error message, checkpoint path, layer names) stays
local; W&B receives only a running counter, and only for the six health types.

| Event | Emitted when | W&B counter |
|---|---|---|
| `training_start` | Session begins — worker count, GPU map, resume flag | — |
| `training_end` | Session ends — steps, elapsed time, restart counts, error | — |
| `worker_start` | A worker process is started or restarted | — |
| `worker_restart` | A worker died and is being restarted | `health/worker_restarts_total` |
| `worker_give_up` | `--max-restarts-per-worker` exceeded; worker abandoned | `health/workers_given_up_total` |
| `worker_crash` | Unhandled exception inside a worker | `health/worker_crashes_total` |
| `rollback` | Rapid crash burst; global network restored from checkpoint | `health/rollbacks_total` |
| `nan_gradient` | NaN gradients detected; update skipped | `health/nan_updates_total` |
| `camera_timeout` | Camera queue timed out; episode dropped | `health/camera_timeouts_total` |
| `crash_recovery` | CARLA server timeout; worker reconnected | — |
| `checkpoint_save` | Periodic or best-model checkpoint written | — |
| `wandb_unavailable` | W&B requested but the package is not installed | — |
| `wandb_init_failed`, `wandb_metric_setup_failed` | W&B startup failed; run continues locally | — |
| `final_summary` | Written last: final counters, best reward, elapsed time, `queue_drops`, `wandb_errors` | mirrored into W&B summary |

## Health of the logging itself

`final_summary` reports two counters that tell you whether the remote view was
complete:

- `queue_drops` — records dropped because the telemetry queue was full. Non-zero
  means W&B has gaps; local files do not.
- `wandb_errors` — failed W&B calls, swallowed so they cannot interrupt training.

Both are written to `events.jsonl` even with `--no-wandb`.
