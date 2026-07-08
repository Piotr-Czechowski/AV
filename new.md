Below is a complete walk‑through of the logging infrastructure that lives in the A3C/Hogwild implementation you provided.  
Everything that is written to disk or sent to **Weights & Biases (W&B)** can be found in one of three places:

| Destination | Where it is created / written | What data is stored |
|-------------|------------------------------|---------------------|
| **Local JSONL files** (`runs/<run‑id>/logs/...`) | `TrainingLogger` + `RunMonitor`/`WorkerMonitor` | Episode statistics, update statistics, step statistics (optional), timing, system metrics, events, checkpoints, resume state |
| **W&B** | `wandb_logger_process()` (separate process) | Hyper‑parameters, per‑step / per‑episode / per‑update metrics, checkpoint information, crash / NaN events |
| **Checkpoint files** (`checkpoint.pth`, `best_checkpoint.pth`, `checkpoints/worker_*/*.pth`) | `GlobalNetwork.save()`, `GlobalNetwork.save_boundary_checkpoint()` | Full model + optimizer state, global counters, worker statistics |

Below we detail each component and the exact fields that are logged.

---

## 1. TrainingLogger – JSONL logging

`TrainingLogger` is instantiated once per worker (and once for the “global” logger with `worker_id=-1`).  
All calls to its public methods write a single line of JSON to an open file handle (`self._<name>_f`) and also push a dictionary onto the shared `log_queue`. The queue is read by the W&B subprocess.

| Method | File written | Fields written (JSON) | Notes |
|--------|--------------|-----------------------|-------|
| `write_metadata()` | `<run>/logs/metadata.json` | `start_time`, `args`, `model`, `n_params`, `n_workers`, plus any extra key/value pairs passed in `**extra` | Called once at the very beginning of training. |
| `log_step()` | `<run>/logs/worker_<id>/steps.jsonl` (if `log_steps=True`) | <ul><li>`global_t`</li><li>`local_t`</li><li>`global_episode`</li><li>`step_in_ep`</li><li>`action`</li><li>`value`</li><li>`entropy`</li><li>`reward`</li><li>`done`</li><li>Optional: `speed_kmh`, `route_dist`, `goal_dist`, `maneuver`</li></ul> | Also pushes the same dictionary (minus `ts` and `worker`) onto `log_queue`. |
| `log_episode()` | `<run>/logs/worker_<id>/episodes.jsonl` | <ul><li>`global_episode`</li><li>`global_t`</li><li>`total_reward`</li><li>`steps`</li><li>Optional: `duration_s`, `max_speed_kmh`, `min_route_dist`, `goal_dist`, `collisions`, `reached_goal`, `action_counts`, `port`, `local_mean_reward`, `global_mean_reward`, `is_new_best`, `reward_components`</li></ul> | Also pushes the same dictionary onto `log_queue`. |
| `log_update()` | `<run>/logs/worker_<id>/updates.jsonl` | <ul><li>`update` (count)</li><li>`global_t`</li><li>`trajectory_length`</li><li>`is_terminal`</li><li>`pi_loss`, `v_loss`, `total_loss`</li><li>`gradient_norm`</li><li>`lr`</li><li>Optional: `advantages_mean`, `advantages_std`, `val_mean`, `val_std`, `ent_mean`, `rew_mean`, `rew_sum`</li><li>Optional arrays if `log_update_arrays=True`: `advantages`, `values`, `rewards`, `entropies`</li></ul> | Also pushes the same dictionary onto `log_queue`. |
| `log_timing()` | `<run>/logs/worker_<id>/timing.jsonl` | <ul><li>`window_updates` (optional)</li><li>`ops`: dict of op → `{avg_ms, count, total_s}`</li></ul> | Also pushes the same dictionary onto `log_queue`. |
| `log_event()` | `<run>/logs/events.jsonl` | Arbitrary key/value pairs; always includes `ts`, `worker` (default -1). Common events: `training_start`, `training_end`, `worker_start`, `worker_restart`, `rollback`, `crash_recovery`, `nan_gradient`, `checkpoint_save`, etc. | No queue push – these are only written to the global events file. |
| `log_checkpoint()` | `<run>/logs/events.jsonl` (via `log_event`) | `event='checkpoint_save'`, `path`, `global_t`, plus optional `checkpoint_kind`. | Also writes a line to the worker’s own `updates.jsonl`? No – only event. |
| `close()` | Flushes all open file handles. | — |

### Queue → W&B

The `log_queue` is read by `wandb_logger_process()`.  
For each record popped from the queue:

1. The dictionary is stripped of its `worker_id` key (if present).  
2. All remaining keys are turned into a flat metric dictionary.  
3. If a `worker_id` was supplied, every key **except** `episode` and `global_step` gets prefixed with `worker/<id>/`.  
4. The resulting metrics dict is sent to W&B via `wandb.log(metrics)`.

Thus the following metrics appear in the W&B run:

| Metric | Source |
|--------|--------|
| `episode`, `global_step`, `global_t` | From any of the log_* methods (step, episode, update, timing). |
| `action`, `value`, `entropy`, `reward`, `done`, `speed_kmh`, `route_dist`, `goal_dist`, `maneuver` | From `log_step`. |
| `total_reward`, `steps`, `duration_s`, `max_speed_kmh`, `min_route_dist`, `goal_dist`, `collisions`, `reached_goal`, `action_counts`, `port`, `local_mean_reward`, `global_mean_reward`, `is_new_best`, `reward_components` | From `log_episode`. |
| `update`, `trajectory_length`, `is_terminal`, `pi_loss`, `v_loss`, `total_loss`, `gradient_norm`, `lr`, `advantages_*`, `val_*`, `ent_*`, `rew_*` | From `log_update`. |
| Timing ops (`avg_ms`, `count`, `total_s`) | From `log_timing`. |

All hyper‑parameters are logged once at the start via `wandb.init(..., config=_config_to_dict(config))`.

---

## 2. System Monitoring – RunMonitor & WorkerMonitor

Both classes inherit from `_BaseMonitor` and write a line of JSON to their respective files every `interval` seconds.

| Monitor | File written | Fields |
|---------|--------------|--------|
| **RunMonitor** (`<run>/logs/system.jsonl`) | <ul><li>`system`: dict with CPU, memory, swap, process counts, etc.</li><li>Optional: `carla`: dict of CARLA server stats (CPU %, RSS, alive count).</li><li>Optional: `gpus`: list of GPU usage & memory per device (if pynvml available).</li></ul> |
| **WorkerMonitor** (`<run>/logs/worker_<id>/system.jsonl`) | <ul><li>`cpu_percent`</li><li>`rss_gb`, `vms_gb`</li><li>`num_threads`</li><li>`ctx_voluntary`, `ctx_involuntary`</li></ul> |

These files are **not** sent to W&B; they are purely for local diagnostics.

---

## 3. Checkpointing & Resume State

| File | Created by | Contents |
|------|------------|----------|
| `<run>/checkpoint.pth` | `GlobalNetwork.save_boundary_checkpoint()` (every `save_frequency` steps) | Full state dict: `global_step`, `global_episode`, `total_updates`, `last_checkpoint_boundary`, `best_reward`, `global_mean_reward`, `worker_mean_rewards`, `recent_rewards`, `recent_reward_count`, `recent_reward_index`, plus `model.state_dict()`, `optimizer.state_dict()` |
| `<run>/checkpoint_step.txt` | Same as above | Integer step number of the checkpoint. |
| `<run>/checkpoints/worker_<id>/checkpoint.pth` | Optional, if `save_worker_checkpoints=True` | Same format as global checkpoint but for a single worker’s local model. |
| `<run>/resume_state.json` | `_write_resume_state()` at end of training | <ul><li>`global_step`, `global_episode`, `total_updates`</li><li>`elapsed_training_s`, `last_session_elapsed_s`, `last_session_start_ts`, `last_session_end_ts`</li><li>`training_args`: dict of hyper‑parameters used in the run</li><li>`timestamp` (end time)</li></ul> |
| `<run>/metadata.json` | `TrainingLogger.write_metadata()` at start | <ul><li>`start_time`</li><li>`args` (hyper‑parameters)</li><li>`model`, `n_params`, `n_workers`</li></ul> |

The resume logic (`find_latest_checkpoint()`, `rollback_global_network()`) uses these files to restore the last good state when a worker crashes or a rapid‑crash burst is detected.

---

## 4. Event Logging

All high‑level events that are not per‑step/episode metrics are written to `<run>/logs/events.jsonl` via `TrainingLogger.log_event()` and `_append_event()`.  
Typical entries include:

| Event | Typical fields |
|-------|----------------|
| `training_start` | `global_t`, `n_workers`, `resumed`, `worker_gpus` |
| `training_end` | `global_t`, `session_steps`, `session_elapsed_s`, `cumulative_elapsed_s`, `active_time_accounting=True` |
| `worker_start` | `worker`, `port`, `device` |
| `worker_restart` | `worker`, `restart_count`, `global_t` |
| `rollback` | `worker`, `rapid_crash_count`, `global_t`, `success` |
| `crash_recovery` | `global_t`, `error` |
| `nan_gradient` | `global_t`, `nan_count`, `nan_layers` |
| `checkpoint_save` | `path`, `global_t` |
| `model_save` / `model_load` / `checkpoint_save` | `path`, `global_t` |

These events are **only** written to the local JSONL file; they are not forwarded to W&B.

---

## 5. Summary of What Is Logged Where

| Data | Local File(s) | W&B |
|------|---------------|-----|
| Hyper‑parameters (config dict) | `metadata.json` | `wandb.init(..., config=…)` |
| Global counters (`global_step`, `global_episode`, etc.) | `events.jsonl` (training_start/end), `resume_state.json` | W&B metrics (`global_step`, `episode`) |
| Per‑step data (`action`, `value`, `entropy`, `reward`, `done`, speed, route distance, goal distance, maneuver) | `<worker>/steps.jsonl` | W&B metrics (prefixed with worker id if present) |
| Episode summary (`total_reward`, `steps`, `duration_s`, `max_speed_kmh`, `min_route_dist`, `goal_dist`, `collisions`, `reached_goal`, `action_counts`, `port`, `local_mean_reward`, `global_mean_reward`, `is_new_best`, `reward_components`) | `<worker>/episodes.jsonl` | W&B metrics |
| Update statistics (`pi_loss`, `v_loss`, `total_loss`, `gradient_norm`, `lr`, advantage/val/reward stats) | `<worker>/updates.jsonl` | W&B metrics |
| Timing per phase (`sync`, `reset`, `forward`, `env_step`, `backward`, `optim_update`) | `<worker>/timing.jsonl` | W&B metrics (ops with avg_ms, count, total_s) |
| System resource usage (CPU, memory, GPU, CARLA server stats) | `system.jsonl` (global & per‑worker) | **Not** sent to W&B |
| Checkpoints (`checkpoint.pth`, `best_checkpoint.pth`, worker checkpoints) | Files on disk | **Not** sent to W&B |
| Resume state (`resume_state.json`) | File on disk | **Not** sent to W&B |
| Events (training start/end, worker restart, rollback, crash recovery, NaN guard, checkpoint events) | `events.jsonl` | **Not** sent to W&B |

---

## 6. Where in the Code Each Piece Is Created

1. **Metadata & Config** – `TrainingLogger.write_metadata()` called once in `main()`.  
2. **Per‑step / Episode / Update / Timing** – Methods of `TrainingLogger` are invoked inside the worker loop (`A3CWorker.run`).  
   * `log_step()` after each environment step.  
   * `log_episode()` when an episode ends.  
   * `log_update()` after a gradient update.  
   * `log_timing()` periodically (every `diag_log_interval` updates or wall‑time).  
3. **Events** – `TrainingLogger.log_event()` is called for global events (`training_start`, `training_end`) and for checkpointing (`checkpoint_save`).  
4. **Worker / Run Monitors** – instantiated in `main()`. They start their own threads that write to `system.jsonl` every `interval`.  
5. **Checkpointing** – `GlobalNetwork.save_boundary_checkpoint()` is called from `A3CWorker._save_checkpoint()` and from the supervisor when a worker restarts or rolls back.  
6. **Resume State** – `_write_resume_state()` is called at the very end of training in `main()`.  
7. **W&B Logging** – The queue (`log_queue`) is created in `main()`. Each call to `TrainingLogger.log_*` pushes a dict onto it. A separate process runs `wandb_logger_process`, which pulls from the queue and logs metrics to W&B.

---

## 7. Practical Take‑aways

* **If you want to see per‑step metrics**: look at `<run>/logs/worker_<id>/steps.jsonl` or the corresponding W&B run (metrics prefixed with `worker/<id>/`).  
* **Episode summaries** are in `<run>/logs/worker_<id>/episodes.jsonl`.  
* **Gradient statistics** live in `<run>/logs/worker_<id>/updates.jsonl`.  
* **Timing breakdowns** are in `<run>/logs/worker_<id>/timing.jsonl`.  
* **System health** (CPU, memory, GPU) is in `<run>/logs/system.jsonl` and per‑worker `system.jsonl`.  
* **Checkpoint files** (`checkpoint.pth`, `best_checkpoint.pth`) are the only binary artifacts; they can be loaded with `torch.load()` if you need to resume or inspect a model.  
* **W&B** will automatically show all metrics that were pushed via the queue, plus the hyper‑parameters from the config dict.

Feel free to let me know if you’d like a deeper dive into any particular log field or how to parse these files programmatically!
