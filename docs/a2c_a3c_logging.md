# Logging: A2C and the A3C template

What the A2C script ([`A_to_B_GPU_34/a2c_rgb_try2.py`](../A_to_B_GPU_34/a2c_rgb_try2.py), started by [`A_to_B_GPU_34/train_basic_a2c.slurm`](../A_to_B_GPU_34/train_basic_a2c.slurm)) and the A3C template ([`A3C-CARLA-Template/`](../A3C-CARLA-Template/)) send to W&B, and where they differ. The template's own reference is [`A3C-CARLA-Template/docs/logging.md`](../A3C-CARLA-Template/docs/logging.md).

## Naming rule

- A2C has one worker and no worker prefix: A3C `worker_<id>/<name>` is A2C `<name>`.
- An A3C run-wide value (`global/<name>`) aggregates the workers. Over one worker it is the worker value itself, so A2C logs that value once, under the per-worker name. Only `global/elapsed_training_s`, which has no per-worker form, keeps its name.
- `episode/*`, `train/lr`, and `train/entropy_coef` have the same names in both.
- Each value has one place in a run; nothing is logged twice under two names.
- The template's local JSONL fields carry the W&B names without `worker_<id>/` (the file names the worker), so a template field, its W&B metric, and the A2C metric share one name.

## Metrics

Charts are plotted against `global_step`. "Summary" means the W&B run summary: one value per run, updated when it changes, shown on the run's Overview page and available as a runs-table column, not as a chart.

| Metric | A3C template | A2C |
|---|---|---|
| Episode reward | `worker_<id>/reward` | `reward` |
| Episode length | `worker_<id>/episode_length` | `episode_length` |
| Distance from the goal at episode end | `worker_<id>/goal_distance` | `goal_distance` |
| Mean reward, last 100 episodes | `worker_<id>/reward_mean_100`; run: `global/reward_mean_100`, the mean of the worker means | `reward_mean_100` |
| Mean distance from the goal, last 100 episodes | `worker_<id>/goal_distance_mean_100`; run: `global/goal_distance_mean_100` | `goal_distance_mean_100` |
| Min and max reward and distance, whole run (summary) | `worker_<id>/reward_min`, `.../reward_max`, `.../goal_distance_min`, `.../goal_distance_max`; run: `global/...`, the lowest and highest worker value | `reward_min`, `reward_max`, `goal_distance_min`, `goal_distance_max` |
| Steps and time to reach a reward (summary) | `threshold/worker_<id>/episode_reward/<label>/*`, `threshold/worker_<id>/reward_mean_100/<label>/*`; run: `threshold/reward_mean_100/<label>/*`, once every worker has 100 episodes | `threshold/episode_reward/<label>/*`, `threshold/reward_mean_100/<label>/*` |
| Training time | `global/elapsed_training_s` | `global/elapsed_training_s` |
| Episode details | `episode/id`, `episode/reward_per_step`, `episode/duration_s`, `episode/success`, `episode/max_speed_kmh`, `episode/min_route_distance`, `episode/collisions`, `episode/action_<n>_fraction` | the same |
| Losses and rollout statistics, every update | `worker_<id>/train/pi_loss`, `v_loss`, `total_loss`, `entropy`, `advantages_mean`, `advantages_std`, `value_mean`, `value_std`, `reward_mean` | `train/<the same>` |
| Schedules, every update | `train/lr`, `train/entropy_coef` | `train/lr`, `train/entropy_coef` |
| End of session (summary) | `final_global_step`, `final_global_update`, `session_elapsed_s` | `final_global_step`, `session_elapsed_s` |

A threshold summary entry has `reached`, `global_step`, `elapsed_training_s`, `global_episode`, and `value`; `<label>` is the threshold with `-` as `neg` and `.` as `p`.

## Differences

| Topic | A3C template | A2C |
|---|---|---|
| Reward of every step | Local only: `steps.jsonl`, with `--log-steps` | W&B `step_reward` |
| Gradient histograms | Not logged | W&B `gradients/actor/<parameter>`, `gradients/critic/<parameter>`, every update |
| Gradient norm and clipping, rollout reward sum and length | Local only: `updates.jsonl` | Not logged; A2C does not clip |
| Health events and counters, `queue_drops`, `wandb_errors` | Local only: `events.jsonl` | Not logged |
| Number of optimizer updates | `final_global_update` | Not logged |
| Local files | JSONL under `<run directory>/logs/`, with the W&B names and more | None: W&B only |
| Process CPU and RAM | With `--log-resources`: `system/proc_*`, `worker_<id>/system/proc_*` | Not logged |
| Run-wide aggregates | `global/*` over the workers | Not needed: one worker |
| `train/entropy_coef` | Annealed schedule | Constant: `1.0` with `USE_ENTROPY`, else `0.0` |
| W&B config | Every training argument, `algorithm: A3C` | Fixed values under A3C names, `algorithm: A2C` |

Both log only the reward the networks train on, never its terms, and both keep W&B's own system metrics (`system/cpu`, `system/gpu.<n>.*`).

## Empty `worker` section in W&B

A W&B project shows panel sections for the keys of all its runs. The older A3C in [`A_to_B_GPU_34/new_hogwild_train_a3c_carla.py`](../A_to_B_GPU_34/new_hogwild_train_a3c_carla.py) logs `worker/<id>/...`. In a project that holds such runs, the `worker` section stays empty for template and A2C runs. A separate project avoids it (`--wandb-project`, or `WANDB_PROJECT` in `A3C-CARLA-Template/.env`).
