# A3C Multi-GPU CARLA — `new_*` pipeline

This document covers three things: how the merge between the user's
PyTorch baseline (`a3c_improved.py`) and the Chainer reference
(`async-rl-backup-12-4-2026-before-spliting-model-and-changing-loss/`)
was performed; how the resulting pipeline executes end-to-end; and the
full parameter reference + a single command to run it.

---

## 1. How the merge was done

Two implementations contributed:

- **`AV/A_to_B_GPU_34/a3c_improved.py`** — single-file PyTorch baseline.
  Already reliable as the *core training loop*: GPU→GPU parameter-server
  architecture, A3C math on PyTorch, CARLA-specific control flow with
  the existing `nets/a2c.py` networks. Weak on observability,
  recovery, parametrisation, and resume.
- **`async-rl-backup-12-4-2026-before-spliting-model-and-changing-loss/`**
  — Chainer reference. Strong scaffolding: modular file layout,
  structured JSONL logging, system + GPU monitoring, supervisor with
  rapid-crash rollback, full resume pipeline, fully CLI-driven config.
  Framework-incompatible (Chainer); we cannot import it directly.

**Strategy:** keep the PyTorch substrate from `a3c_improved.py`; port
the *non-framework* layers around it from async-rl, translating the
shared-memory bits to CUDA-aware equivalents.

### 1.1 Inherited from `a3c_improved.py`

| Feature | Where it landed in `new_*` |
|---|---|
| `torch` + `torch.multiprocessing` substrate, `mp.set_start_method('spawn')` | [new_train_a3c_carla.py](AV/A_to_B_GPU_34/new_train_a3c_carla.py) `main()` |
| GPU→GPU parameter-server: global net on `cuda:0`, workers on `cuda:1+` | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `GlobalNetwork`, `A3CWorker` |
| `SharedAdam` (Adam state pre-allocated on the param device, no `share_memory_()` for CUDA) | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `SharedAdam` |
| `transfer_grads_gpu_to_gpu` direct grad copy via `.to(dev, non_blocking=False)` | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `transfer_grads` |
| Three-lock multiproc design: `update_lock` / `stats_lock` / `save_lock` (see §1.4 — `update_lock` is a CUDA-imposed deviation from canonical Hogwild! A3C) | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `GlobalNetwork` |
| A3C math: `Transition` namedtuple, n-step bootstrap, advantage `.detach()`, Huber `F.smooth_l1_loss` value loss, trajectory-mean entropy, `total_loss = pi + v − β·H` | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `compute_and_apply_gradients` |
| `Categorical(logits=logits)` sampling | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `get_action` |
| `clip_grad_norm_(max_grad_norm)` before pushing grads | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) |
| CARLA call sequence: `step_apply_action → world.tick → image_queue drain → env.step` (extended to *double tick* — see §1.3) | [new_carla_wrapper.py](AV/A_to_B_GPU_34/new_carla_wrapper.py) `step` |
| `nets.a2c.DiscreteActor`/`Critic`, `carla_env.CarlaEnv`, `ACTIONS.ACTIONS` | imported, never modified |
| `gc.collect() + torch.cuda.empty_cache()` every N episodes | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `run` |
| W&B in a subprocess via `mp.Queue` so worker I/O never blocks | [new_train_a3c_carla.py](AV/A_to_B_GPU_34/new_train_a3c_carla.py) `wandb_logger_process` |
| `try / except RuntimeError if "time-out" in str(e): reconnect` worker pattern | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) (extended in §1.3) |
| `mp.Value` / `mp.Array` for shared stats and rolling reward EMAs | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) |
| `worker_mean_rewards[worker_id]` restored on worker (re)start | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `run` |

### 1.2 Ported from async-rl (framework-translated to PyTorch)

| Feature | Source in async-rl | Where it lives in `new_*` |
|---|---|---|
| Modular split — algorithm vs supervisor vs entry vs logger vs monitor vs profiler vs CARLA wrapper | one file per concern | 9 separate `new_*.py` |
| Step-based global counter + `--steps` budget; loop terminates on `global_t > steps` | `run_a3c.py:96-103` | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `inc_step`, [new_run_a3c.py](AV/A_to_B_GPU_34/new_run_a3c.py) supervisor |
| Linear LR decay `(steps − t − 1) / steps × lr` per step | `run_a3c.py:106-107` | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `GlobalNetwork.set_lr_for_step` |
| **NaN gradient guard**: detect → skip step → resync from global → log `nan_gradient` | `a3c.py:154-175` | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `has_nan_grads` + worker |
| **NaN-safe checkpoint write** | `run_a3c.py:188-192` | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `GlobalNetwork.save(nan_safe=True)` |
| **Rollback to last good checkpoint on rapid crash** (`< rapid_crash_window_steps` apart, `≥ rapid_crash_threshold` times) | `run_a3c.py:255-316` | [new_run_a3c.py](AV/A_to_B_GPU_34/new_run_a3c.py) `rollback_global_network` |
| `max_restarts_per_worker` cap | `train_a3c_carla.py:625-628` | [new_run_a3c.py](AV/A_to_B_GPU_34/new_run_a3c.py) `run_with_restart` |
| Supervisor with rapid-crash detection + per-worker restart counters | `run_a3c.py:319-396` | [new_run_a3c.py](AV/A_to_B_GPU_34/new_run_a3c.py) `run_with_restart` |
| Resume pipeline: `resume_state.json` (atomic `tmp + os.replace`), arg-drift warnings, W&B id reuse, episode counter persistence | `run_a3c.py:216-239`, `train_a3c_carla.py:716-781` | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `_save_checkpoint`, [new_train_a3c_carla.py](AV/A_to_B_GPU_34/new_train_a3c_carla.py) `main()` resume branch |
| `find_latest_checkpoint` (top-level + `checkpoints/worker_<i>/`) | `train_a3c_carla.py:665-688` | [new_run_a3c.py](AV/A_to_B_GPU_34/new_run_a3c.py) `find_latest_checkpoint` |
| Per-worker checkpoints | `run_a3c.py:193-203` | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `_save_checkpoint` |
| **Step-based** save frequency (was episode-based in `a3c_improved.py`) | `run_a3c.py:184-213` | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) inner loop |
| `TrainingLogger`: per-worker `episodes.jsonl` + `updates.jsonl` + `timing.jsonl` + (opt) `steps.jsonl`, shared `events.jsonl` + `metadata.json`, line-buffered | `training_logger.py` | [new_training_logger.py](AV/A_to_B_GPU_34/new_training_logger.py) |
| `RunMonitor` + `WorkerMonitor` (psutil) — system + CARLA-process + per-worker stats | `system_monitor.py` | [new_system_monitor.py](AV/A_to_B_GPU_34/new_system_monitor.py) |
| `TimingAccumulator` per-phase profiler | `timing_utils.py` | [new_timing_utils.py](AV/A_to_B_GPU_34/new_timing_utils.py) |
| `prepare_output_dir` — args + git status/log/diff snapshot | `prepare_output_dir.py` | [new_prepare_output_dir.py](AV/A_to_B_GPU_34/new_prepare_output_dir.py) |
| `CarlaA3CWrapper` abstraction: `_connect_with_retries`, `is_server_alive` (via `world.get_snapshot()`), `reconnect`, per-episode CARLA stats | `train_a3c_carla.py:61-313` | [new_carla_wrapper.py](AV/A_to_B_GPU_34/new_carla_wrapper.py) |
| Per-episode action histogram + max speed + route dist + goal dist + collisions + reached_goal | `train_a3c_carla.py:182-224` | [new_carla_wrapper.py](AV/A_to_B_GPU_34/new_carla_wrapper.py) `_action_counts`, `_ep_*` |
| `--reward-scale` hook | `a3c.py:80-81` | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `compute_and_apply_gradients` |
| End-to-end frame saving: `--save-episodes` / `--save-episode-interval` predicate + `state_observer.image.save_to_disk(...)` per step + `<project_dir>/episodes/<run_id>/<ep>-<port>/<step>.jpeg` layout (identical to `async-rl/episodes/`) | `train_a3c_carla.py:163-169`, `283-294` | [new_carla_wrapper.py](AV/A_to_B_GPU_34/new_carla_wrapper.py) `_should_save_this_episode`, `_save_frame`, `_ensure_save_dir`, `_frames_dir` — see §2.6 |
| Argparse-driven CLI for every knob; drop `from settings import …` | `train_a3c_carla.py:906-947` | [new_train_a3c_carla.py](AV/A_to_B_GPU_34/new_train_a3c_carla.py) `build_parser` |
| `--testing` mode (argmax, no updates, no checkpoints) | async-rl convention | [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) |
| Default `weight_decay=1e-2` (matches `a3c_improved.py`; CLI-overridable) | n/a — kept from a3c_improved.py | CLI flag |
| Worker-restart with `core.*` cleanup + `--carla-server-start-period` wait | `train_a3c_carla.py:837-846` | [new_run_a3c.py](AV/A_to_B_GPU_34/new_run_a3c.py) |
| `events.jsonl` lifecycle log (worker_start, worker_restart, rollback, checkpoint_save, training_start/end, …) | `train_a3c_carla.py:567-574` | [new_training_logger.py](AV/A_to_B_GPU_34/new_training_logger.py) `log_event`, [new_run_a3c.py](AV/A_to_B_GPU_34/new_run_a3c.py) `_append_event` |

### 1.3 New / synthesised in `new_*`

| Feature | Why it was needed |
|---|---|
| **Disambiguated `RuntimeError` handling** in the worker — separates `"waiting for the simulator"` (server dead → reconnect) from `"camera image"` / other `"time-out"` (queue underflow → retry without reconnect). | First test run looped on a camera-queue underflow that was being mis-classified as a server timeout — burning 60 s per step on a useless reconnect. |
| **Confirmed double-tick CARLA step pattern**: `apply → drain → tick → apply → tick → step`. | `CarlaEnv.step()` consumes two `image_queue.get(timeout=2.0)` per call (verified via [carla_env.py:1267-1274](AV/A_to_B_GPU_34/carla_env.py#L1267-L1274) and the `# 2 frames are put on the queue between two consecutive steps` comment in [carla_env_c.py:1274](AV/A_to_B_GPU_34/carla_env_c.py#L1274)). The async-rl wrapper already did this; we ported it. |
| **`pynvml` GPU stats** in `RunMonitor` | async-rl's monitor was psutil-only; per-device util % + memory is useful on multi-GPU runs. |
| **Named-args slurm wrapper** ([new_train.slurm](AV/A_to_B_GPU_34/new_train.slurm)) | Combines `test_a3c_carla.slurm`'s arg-parsing convention with `new_train_a3c_carla.py`'s flags. Auto-launches `carla_athena_multiserver_v3.py`, `lsof`-polls until ports LISTEN, traps signals to clean up. One sbatch call = full pipeline. |
| **`mp.Value('l')` (long)** for global step counter | The Chainer counter was a CPU-only `mp.Value('l', 0)`; we keep that — the LR schedule, budget check, and supervisor exit all read it. |
| **Worker **`signal.SIGINT, signal.SIG_IGN`** | Ensure Ctrl+C in the slurm tail hits the main process only; supervisor handles graceful shutdown. |
| **End-to-end image-save fix**: `--save-episodes` originally only set `_save_images=True` but never wrote files (the wrapper passed `save_image=` to `CarlaEnv` whose dump branches are commented out). | Audit against async-rl revealed the missing `state_observer.image.save_to_disk(...)` call; ported it into `_save_frame()` and wired `outdir` through `main → supervisor → worker → wrapper`. |

---

## 2. How the pipeline works (call-by-call)

When you `sbatch new_train.slurm -w 1 …`:

### 2.1 Slurm wrapper — [new_train.slurm](AV/A_to_B_GPU_34/new_train.slurm)

1. SBATCH headers parsed (partition, gpus, time, `--signal=SIGUSR1@90`, …).
2. `while [[ $# -gt 0 ]]; case "$1" in …` parses every `-w / --workers / --lr / --t-max / -r / --resume / --save-episodes / …` into shell variables.
3. `module load …`, `source ${VENV}/bin/activate`, then export `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONUNBUFFERED=1`, `PYTHONFAULTHANDLER=1`, `NCCL_DEBUG=INFO`, `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`.
4. Output dir picked: `--resume DIR` ⇒ keep DIR; `--outdir DIR` ⇒ override; otherwise `PROJECT_DIR/runs/a3c_<N>w_<ts>_<jobid>`.
5. `cleanup()` trap registered for `SIGINT`/`SIGTERM`/`SIGUSR1`/`EXIT` — kills the training process, the CARLA supervisor, `nvidia-smi dmon`, and `pkill CarlaUE4` as a safety net.
6. `nvidia-smi dmon -s pucvmet -o DT` started in background → `<outdir>/gpu_dmon.log`.
7. `nohup python carla_athena_multiserver_v3.py --num-servers=N --servers-per-gpu=M --start-port=2000 --port-step=100 …` started → `<outdir>/carla_servers.log`. PID stored as `CARLA_PID`.
8. `lsof -nP -iTCP:<port> -sTCP:LISTEN` polled, up to 60 × 10 s, for every expected port. On failure: tail of `carla_servers.log` printed, exit 1.
9. Python entry invoked: `python -u new_train_a3c_carla.py --num-workers $N … --outdir $OUTPUT_DIR`. stdout → `<outdir>/a3c_training.log`; `tail --pid` mirrors it to slurm stdout.

### 2.2 Python entry — [new_train_a3c_carla.py](AV/A_to_B_GPU_34/new_train_a3c_carla.py) `main()`

1. `build_parser().parse_args()` → `args` (every knob from §3.1).
2. `torch.manual_seed`, `np.random.seed`, `torch.cuda.manual_seed_all` from `args.seed`.
3. `args.n_actions` defaulted to `len(ac.ACTIONS_NAMES)` if not set.
4. `_assign_worker_gpus(num_workers, workers_per_gpu, worker_gpu_start)` → list of `cuda:N` strings stored in `args.worker_gpus`. Falls back to all-`cuda:0` on 1-GPU nodes.
5. `prepare_output_dir(args, args.outdir, resume=bool(args.resume))` writes `args.txt` + `git-status.txt` + `git-log.txt` + `git-diff.txt` (or `_resume`-suffixed copies on resume).
6. `cfg = SimpleNamespace(**vars(args))` — passed verbatim to workers; nothing is read from a global module.
7. `mp.set_start_method('spawn', force=True)`, `shutdown_event = mp.Event()`, optional `log_queue = mp.Queue(maxsize=1000)` for W&B.
8. `GlobalNetwork(cfg, [res, res, 3], n_actions, 1)` constructed on `cuda:0`:
   - Actor + Critic on `cuda:0`.
   - Two `SharedAdam` optimisers with state pre-allocated on `cuda:0`.
   - `update_lock`, `stats_lock`, `save_lock`.
   - `mp.Value` counters (`global_step`, `global_episode`, `total_updates`, `best_reward`, `global_mean_reward`) and `mp.Array('d', [0.0] * num_workers)` for per-worker rolling means.
9. **If `--resume`:** `find_latest_checkpoint(outdir)` scans top-level + every `checkpoints/worker_<i>/` for the newest `.pth`; `global_network.load(path)` restores weights, optimiser state, *and every counter*. `resume_state.json` read; `--steps`, `--lr`, `--entropy_coef`, `--t-max`, `--gamma`, `--weight-decay` compared against saved values, drift logged to stderr.
10. `TrainingLogger.write_metadata(...)` writes `<outdir>/logs/metadata.json` (run config + actor/critic param counts).
11. A throwaway `TrainingLogger(worker_id=-1)` writes a `training_start` event to `events.jsonl` and closes.
12. `RunMonitor(outdir, monitor_interval, track_carla=True, track_gpu=…).start()` — daemon thread sampling system + CARLA processes + (if `pynvml` installed and `--no-gpu-monitor` not set) per-GPU util/memory every `monitor_interval` seconds → `<outdir>/logs/system.jsonl`.
13. W&B subprocess started: stable run id (new UUID on first run, restored from `<outdir>/wandb_run_id.txt` on resume); the subprocess drains `log_queue` and namespaces records under `worker/<id>/…`.
14. `run_with_restart(global_network, cfg, outdir, shutdown_event, log_queue, run_id)` — supervisor loop.

### 2.3 Supervisor — [new_run_a3c.py](AV/A_to_B_GPU_34/new_run_a3c.py) `run_with_restart()`

1. `_start_worker(...)` instantiates and starts `A3CWorker(mp.Process)` for each worker_id. Per worker, a `worker_start` event is appended to `events.jsonl`.
2. Loop: every `--worker-check-interval` seconds:
   - If `cfg.steps > 0 and global_step.value >= cfg.steps`: `shutdown_event.set()`, exit loop.
   - For each worker: if not `is_alive()`:
     - `worker.join(timeout=2)`. Increment `restart_counts[i]`.
     - Compute `current_step − last_crash_step[i]`; if `< rapid_crash_window_steps`, increment `rapid_crash_count[i]`, else reset it. Update `last_crash_step[i]`.
     - `worker_restart` event written.
     - Remove any `core.*` files.
     - If `restart_counts[i] >= max_restarts_per_worker`: log `worker_give_up`, do not relaunch.
     - If `rapid_crash_count[i] >= rapid_crash_threshold`: `rollback_global_network(global_network, outdir, worker_idx=i)`:
       - Walk every checkpoint newest-first (per-worker first, then top-level, then other per-worker).
       - `torch.load` → `state_dict_load` into `global_network.actor` + `critic`.
       - If `has_nan_params(actor) or has_nan_params(critic)`: skip this checkpoint.
       - Optionally restore optimiser state.
       - On success: `rollback` event written, `rapid_crash_count[i]` reset.
     - Sleep `--carla-server-start-period`, `_start_worker(i)` again.
3. On `KeyboardInterrupt` / `shutdown_event.set()`: terminate stragglers, return `restart_counts`.

### 2.4 Worker — [new_a3c.py](AV/A_to_B_GPU_34/new_a3c.py) `A3CWorker.run()`

After `mp.Process.start()`:

1. `signal.signal(SIGINT, SIG_IGN)` so Ctrl+C only hits the main process.
2. `_init_networks()` builds local `actor` + `critic` on `cuda:N`, calls `sync_with_global()` which copies global params GPU→GPU into local and `torch.cuda.synchronize(self.device)`.
3. `TrainingLogger(outdir, worker_id, log_steps, log_update_arrays)` opens per-worker JSONL files.
4. `WorkerMonitor(outdir, worker_id, monitor_interval).start()` — daemon thread → `<outdir>/logs/worker_<i>/system.jsonl`.
5. `TimingAccumulator()` initialised.
6. Per-worker rolling-mean reward restored from `global_network.worker_mean_rewards[worker_id]`.
7. `CarlaA3CWrapper(port, scenario, camera, res, …)` constructed:
   - `_connect_with_retries` calls `CarlaEnv(...)` up to `--max-connect-retries` times with `--connect-retry-wait` s sleeps between failures.
8. **Outer loop** — `while not shutdown_event.is_set():`
   a. `inc_episode()` → `current_episode` (atomic, under `stats_lock`). `env.global_episode = current_episode`. `sync_with_global()`.
   b. `state, speed, maneuver = env.reset()` (under `timer.record('env_reset')`):
      - `CarlaEnv.reset()` does `step_apply_action(3)`, 15 ticks, drain, 1 tick, get one image → returns `(front_camera, speed_tensor)`.
      - Wrapper normalises `state /= 255`, `speed /= 100`, drops the leading `[1, …]` batch dim if present.
   c. **Inner loop** — `while not done`:
      - `inc_step(1)` → `global_t` (atomic). If `> args.steps`: `shutdown_event.set()`, break.
      - Build tensors on `self.device`.
      - `action, value, entropy = get_action(state_t, speed_t, maneuver_t, testing)` (under `timer.record('forward')`). The `actor`/`critic` forwards run on the worker's GPU; a `Categorical(logits).sample()` (or `argmax` if `--testing`) yields the action; `Transition(value, log_prob, entropy, action)` appended.
      - `next_state, next_speed, next_maneuver, reward, done, info = env.step(action)` (under `timer.record('env_step')`). Wrapper does **`apply → drain → tick → apply → tick → env.step`**: `CarlaEnv.step()` consumes the two camera frames produced by the two ticks.
      - Optional `tlogger.log_step(...)` if `--log-steps`.
      - **Update trigger** — every `--t-max` env steps or on `done`:
        - `compute_and_apply_gradients(next_state_t, done, next_speed_t, next_maneuver_t, tlogger, timer, global_t)`:
          1. `R = 0` if `done` else `critic(next_state_t)`. Reverse-accumulate n-step returns with `--reward-scale` applied: `R = r/scale + γ·R`.
          2. `policy_loss = − Σ log_prob × advantage`, `value_loss = Σ value_loss_coef × Huber(V, R)`, `total_loss = pi + v − entropy_coef × mean(entropy)`.
          3. `actor.zero_grad(); critic.zero_grad(); total_loss.backward()` (under `timer.record('backward')`).
          4. `clip_grad_norm_(actor); clip_grad_norm_(critic)` if `max_grad_norm > 0`.
          5. **NaN guard**: `has_nan_grads(actor) + has_nan_grads(critic)`. On hit ⇒ `tlogger.log_nan(...)`, `sync_with_global()`, clear buffers, **return** without touching the global model.
          6. Acquire `update_lock`. `set_lr_for_step(global_t)` pushes the linearly-decayed LR into both optimisers under the lock. `transfer_grads(actor → global.actor, cuda:0)`; same for critic. `actor_optimizer.step(); critic_optimizer.step(); zero_grad(); zero_grad()`. Release the lock. (Under `timer.record('optim_update')`.)
          7. `inc_updates()`. `tlogger.log_update(update_count, global_t, traj_len, pi/v/total loss, grad_norm, lr, advantages, values, entropies, rewards)`.
        - `sync_with_global()` if `local_updates % --sync-every-n-updates == 0`.
      - Step-based **checkpoint**: when `global_t // save_frequency` crosses a new boundary, `_save_checkpoint(global_t, tlogger)`:
        - `GlobalNetwork.save(path, global_t, nan_safe=True)` writes `<outdir>/checkpoints/worker_<i>/checkpoint.pth` + top-level `<outdir>/checkpoint.pth`. Save is refused if any param is NaN.
        - `checkpoint_step.txt` files written next to each.
        - `<outdir>/resume_state.json` written atomically (`tmp + os.replace`) — global step, training args, timestamp.
        - `tlogger.log_checkpoint(path, global_t)` event.
   d. End of episode: `gc.collect() + torch.cuda.empty_cache()` every `--gc-interval`. `_log_episode(tlogger, env, …)` writes one JSON line to `episodes.jsonl` (reward, steps, action histogram, max speed, route dist, goal dist, collisions, reached_goal, local/global mean rewards, is_new_best) and pushes the same record to the W&B queue.
   e. Every `--diag-log-interval` episodes: `tlogger.log_timing(...)` + `timer.log_and_reset(...)` — one JSON line of profiler stats per phase.
9. **`RuntimeError` handler:**
   - `"waiting for the simulator"` in message → CARLA server dead. `tlogger.log_crash_recovery(...)`, `env.reconnect()` (sleep `carla-timeout-wait` s, drop `world`/`client` refs, build a fresh `CarlaEnv`), clear buffers, continue outer loop.
   - `"camera image"` or any other `"time-out"` → camera-queue underflow, server is fine. `camera_timeout` event written, buffers cleared, no reconnect.
   - Anything else → reraise (the supervisor sees `is_alive() == False` and restarts the worker).
10. `finally`: stop `WorkerMonitor`, close `TrainingLogger`, null out `env.world` / `env.client`. Print termination line.

### 2.5 Shutdown

- Step budget reached or `KeyboardInterrupt` ⇒ `shutdown_event.set()` ⇒ supervisor exits the loop ⇒ `main()` drains the W&B queue, stops `RunMonitor`, writes `training_end` event, prints benchmark line, exits.
- The slurm `cleanup()` trap fires on `EXIT`: kills the (already-exited) training process, kills the CARLA supervisor, kills `nvidia-smi dmon`, `pkill CarlaUE4`.

### 2.6 Episode frame saving (`--save-episodes`, `--save-episode-interval`)

Identical pipeline to async-rl's per-step JPEG dump
(`async-rl/train_a3c_carla.py:284-294`), and *identical on-disk layout*
to `async-rl/episodes/` (the pointer dir, e.g.
`/net/tscratch/people/plgbartoszkawa/async-rl/episodes/a3c_ff_10w_20260323_135722/1000-2000/{1,2,3,…}.jpeg`).

**Plumbing**

| Hop | File / class | Effect |
|---|---|---|
| `--save-episodes "1 10 700"` | argparse | `args.save_episodes = [1, 10, 700]` |
| `run_id = basename(outdir)` | [new_train_a3c_carla.py:314](AV/A_to_B_GPU_34/new_train_a3c_carla.py#L314) | run_id derived once (e.g. `a3c_1w_20260424_114623_2545139`) |
| → supervisor | [new_run_a3c.py:171-174](AV/A_to_B_GPU_34/new_run_a3c.py#L171-L174) | `outdir` + `run_id` forwarded to every worker |
| → worker constructor | [new_a3c.py:238-241](AV/A_to_B_GPU_34/new_a3c.py#L238-L241) | stored as `self.outdir`, `self.run_id` |
| → wrapper | [new_a3c.py:573-574](AV/A_to_B_GPU_34/new_a3c.py#L573-L574) | `CarlaA3CWrapper(..., outdir=self.outdir, run_id=self.run_id)` |
| `env.global_episode = current_episode` *before* `env.reset()` | [new_a3c.py:583-589](AV/A_to_B_GPU_34/new_a3c.py#L583-L589) | wrapper sees correct episode number from frame 1 (cleaner than async-rl which sets it after reset) |

**Predicate** — `CarlaA3CWrapper._should_save_this_episode()` ([new_carla_wrapper.py:111-118](AV/A_to_B_GPU_34/new_carla_wrapper.py#L111-L118)):

```
self.global_episode in self._save_episodes  →  save
or
--save-episode-interval > 0 and global_episode % interval == 0  →  save
else                                                            →  skip
```

The result is cached in `self._save_images` for the duration of the
episode (re-evaluated at next `reset()`); the per-episode save dir
cache (`_save_dir_cached`) is nulled so the new dir gets a single
`os.makedirs(exist_ok=True)`.

**Save call** — `_save_frame()` ([new_carla_wrapper.py](AV/A_to_B_GPU_34/new_carla_wrapper.py)):

1. Early-exit if `_save_images is False` or no `state_observer`.
2. Read `carla_img = self.env.state_observer.image`. CarlaEnv writes
   the *latest* `carla.Image` here at the end of every `reset()` and
   `step()` ([carla_env.py:1190](AV/A_to_B_GPU_34/carla_env.py#L1190),
   [carla_env.py:1275](AV/A_to_B_GPU_34/carla_env.py#L1275)). After
   the wrapper's two-tick `step()`, this is the *second* image —
   i.e. the one the model just received as `next_state`.
3. `_ensure_save_dir()` calls `os.makedirs(exist_ok=True)` once per
   episode and returns the cached
   `<project_dir>/episodes/<run_id>/<global_episode>-<port>/`.
4. `carla_img.save_to_disk(os.path.join(ep_dir, f'{step_count}.jpeg'))`
   — CARLA's native JPEG writer; honours the in-place
   `image.convert(CityScapesPalette)` already done by
   `process_semantic_img`, so saved frames look like the policy's
   input, not the raw camera buffer.
5. IO errors are caught; `self._save_failures` is incremented (visible
   to anyone introspecting the wrapper) but the worker keeps going.

**When `_save_frame` runs** — only inside `step()`, after `env.step()`
returns and `_update_maneuver()` ran. First saved frame is `1.jpeg`
(after the first action), matching async-rl exactly. No spawn frame.

**Output layout (byte-for-byte match with `async-rl/episodes/`)**

Base is `<project_dir>` — i.e. the directory holding `new_carla_wrapper.py`,
which under default slurm settings is
`/net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34`. Frames sit
*above* the per-run `outdir` so different runs share a single
`episodes/` root and the run-name level alone disambiguates them:

```
<project_dir>/episodes/<run_id>/<global_episode>-<port>/
    1.jpeg     ← after first env.step
    2.jpeg
    3.jpeg
    ...
```

For `sbatch ... new_train.slurm -w 1 -s 14 --save-episodes "700"`
(run_id `a3c_1w_20260424_114623_2545139`, port 2000), frames land at:

```
/net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/episodes/
    a3c_1w_20260424_114623_2545139/
        700-2000/
            1.jpeg
            2.jpeg
            ...
```

Compare with the reference run in
`/net/tscratch/people/plgbartoszkawa/async-rl/episodes/a3c_ff_10w_20260323_135722/1000-2000/{1,2,3,…}.jpeg`
— identical hierarchy.

**Multi-worker note** — episode counter is global (atomic
`inc_episode()` on `mp.Value`), so a save_episode is hit by exactly
one worker. If you want multiple workers' versions of the same
"episode 700", use `--save-episode-interval 100` instead — every
worker independently saves the episodes whose global number is a
multiple of 100.

### 2.7 Output layout produced by one run

```
<outdir>/
    args.txt, git-status.txt, git-log.txt, git-diff.txt
    a3c_training.log              ← python stdout (tailed live to slurm stdout)
    carla_servers.log             ← multi-server supervisor stdout
    gpu_dmon.log                  ← nvidia-smi dmon
    checkpoint.pth                ← top-level (most recent NaN-safe)
    checkpoint_step.txt
    resume_state.json
    wandb_run_id.txt
    logs/
        metadata.json             ← run config snapshot (one-time)
        events.jsonl              ← shared lifecycle (training_start, worker_*,
                                    rollback, checkpoint_save, training_end)
        system.jsonl              ← RunMonitor: host CPU/mem/swap, CARLA procs, GPUs
        worker_0/
            episodes.jsonl
            updates.jsonl
            timing.jsonl
            steps.jsonl           ← only if --log-steps
            system.jsonl          ← WorkerMonitor: per-worker RSS/threads/ctxsw
        worker_1/ ...
    checkpoints/
        worker_0/
            checkpoint.pth
            checkpoint_step.txt
        worker_1/ ...

# saved frames live ABOVE outdir, alongside it (matching async-rl):
<project_dir>/                    ← e.g. AV/A_to_B_GPU_34/
    episodes/                     ← only populated if --save-episodes / --save-episode-interval
        <run_id>/                 ← same name as outdir's basename
            <global_episode>-<port>/
                1.jpeg            ← after first env.step
                2.jpeg
                ...
```

---

## 3. Parameter reference + final command

Every flag below is accepted by both the slurm wrapper and the Python
entry. "Slurm default" is the value the wrapper sets when you don't
pass it; "Python default" is what `new_train_a3c_carla.py --help`
reports. Slurm forwards the value verbatim. Where they differ, the
slurm value wins.

### 3.1 Topology / GPU

| Flag | Slurm default | Python default | What it does |
|---|---|---|---|
| `-w` / `--workers` (slurm) / `--num-workers` (py) | `1` | `2` | Number of A3C workers. Determines the number of CARLA servers spawned by the multi-server. |
| `--workers-per-gpu` | `1` | `2` | Workers (and CARLA servers) per GPU. |
| `--worker-gpu-start` | `0` | `1` | First `cuda:N` index used for workers. Use `1` to keep `cuda:0` reserved for the global net on multi-GPU. |
| `--global-device` | (forward python default) | `cuda:0` | Device hosting the parameter-server `GlobalNetwork`. |
| `--servers-per-gpu` *(slurm only)* | `= --workers-per-gpu` | — | Override CARLA-side placement; rarely needed. |

### 3.2 CARLA

| Flag | Slurm default | Python default | What it does |
|---|---|---|---|
| `--start-port` | `2000` | `2000` | First CARLA RPC port. |
| `--port-step` | `100` | `100` | Port increment between servers. |
| `-s` / `--scenario` | `14` | `[14]` | Scenario id(s); space-separated for multi-value (`-s "14 15 16"`). |
| `--camera` | `semantic` | `semantic` | `rgb` or `semantic`. |
| `--res` | `250` | `250` | Square image resolution. |
| `--action-type` *(py only)* | — | `discrete` | Passed to `CarlaEnv.action_space`. |
| `--mp-density` *(py only)* | — | `25` | `CarlaEnv` map-density. |
| `--n-actions` *(py only)* | — | `len(ACTIONS_NAMES)` | Override action count. |
| `--no-carla` *(slurm only)* | off | — | Skip launching CARLA (assume already up). |

### 3.3 Algorithm

| Flag | Slurm default | Python default | What it does |
|---|---|---|---|
| `--t-max` | `20` | `20` | Rollout length per gradient update. |
| `--gamma` | `0.99` | `0.99` | Discount factor. |
| `--lr` | `1e-4` | `1e-4` | Starting LR; linearly decays to 0 across `--steps`. |
| `--beta` | `0.01` | `0.01` | Entropy coefficient (`entropy_coef` internally). |
| `--value-loss-coef` | `1.0` | `1.0` | Multiplier on Huber value loss inside `total_loss`. |
| `--weight-decay` | `1e-2` | `1e-2` | Adam weight decay. Matches `a3c_improved.py`. |
| `--max-grad-norm` | `40.0` | `40.0` | `clip_grad_norm_` threshold; `≤ 0` disables. |
| `--reward-scale` | unset (flag not forwarded) | `0.0` | Divide raw reward by this; `0` / unset = disabled. The slurm wrapper only forwards the flag when `REWARD_SCALE` is set explicitly, so by default the Python entry runs with its `0.0` (off). |
| `--steps` | `10000000` | `10000000` | Total env-step budget across all workers (drives LR decay + supervisor exit). |
| `--sync-every-n-updates` | `1` | `1` | Re-pull global weights into the worker every N local updates. |
| `--gc-interval` | `10` | `10` | Episodes between `gc.collect() + empty_cache()`. |
| `--testing` | off | off | Evaluation mode: argmax actions, no updates, no checkpoints. |

### 3.4 Checkpointing / resume

| Flag | Slurm default | Python default | What it does |
|---|---|---|---|
| `--save-frequency` | `20000` | `20000` | Steps between checkpoints; `0` disables. |
| `--outdir` | auto (`runs/a3c_<N>w_<ts>_<jobid>`) | `None` (tempdir) | Output directory. |
| `-r` / `--resume` | empty | `None` | Resume from this previous outdir (loads checkpoint, restores counters, reuses W&B id, warns on arg drift). |

### 3.5 Supervisor / recovery

| Flag | Slurm default | Python default | What it does |
|---|---|---|---|
| `--max-restarts-per-worker` | `160` | `160` | Hard cap on per-worker restarts. (Bumped 8× from the old `20` because long CARLA runs hit the limit during a 10 h job.) |
| `--rapid-crash-threshold` | `3` | `3` | Rapid crashes before triggering a global rollback. |
| `--rapid-crash-window-steps` | `100` | `100` | Steps within which crashes count as "rapid". |
| `--carla-timeout-wait` | `60` | `60.0` | Sleep before reconnecting after a CARLA server timeout. |
| `--carla-server-start-period` | `30` | `30.0` | Wait between worker death and relaunch (lets CARLA boot). |
| `--max-connect-retries` | `5` | `5` | `CarlaA3CWrapper` connection attempts on cold start. |
| `--connect-retry-wait` | `30` | `30.0` | Sleep between those attempts. |
| `--worker-check-interval` *(py only)* | — | `5.0` | Supervisor poll interval. |

### 3.6 Logging / monitoring

| Flag | Slurm default | Python default | What it does |
|---|---|---|---|
| `--log-steps` | off | off | Write one JSON line per env step to `steps.jsonl`. |
| `--log-update-arrays` | off | off | Include full per-step advantage/value/entropy/reward arrays in `updates.jsonl`. |
| `--diag-log-interval` | `100` | `100` | Episodes between `timing.jsonl` flush + profiler reset. |
| `--monitor-interval` | `10` | `10.0` | RunMonitor / WorkerMonitor sampling period (s). |
| `--no-system-monitor` | off | off | Disable both monitors. |
| `--no-gpu-monitor` | off | off | Disable the pynvml GPU sampling block in RunMonitor. |
| `--wandb-project` | `a3c-carla` | `A_to_B` | W&B project name. |
| `--wandb-run-name` | empty → auto | `None` → auto | W&B run name. |
| `--no-wandb` | off | off | Skip the W&B subprocess entirely. |

### 3.7 Image saving

| Flag | Slurm default | Python default | What it does |
|---|---|---|---|
| `--save-episodes` | empty | `None` | Space-separated episode numbers whose frames get saved (`--save-episodes "1 10 100 700"`). |
| `--save-episode-interval` | `0` | `0` | Also save frames every N episodes; `0` disables. |

### 3.8 Misc

| Flag | Slurm default | Python default | What it does |
|---|---|---|---|
| `--seed` | `52` | `52` | RNG seed (torch + numpy + cuda). |
| `--venv` *(slurm only)* | `/net/tscratch/people/plgbartoszkawa/venv` | — | Venv to activate. |
| `--project-dir` *(slurm only)* | `/net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34` | — | Where the `new_*` files live. |
| `--multiserver-script` *(slurm only)* | `/net/tscratch/people/plgbartoszkawa/carla_athena_multiserver_v3.py` | — | CARLA multi-server launcher path. |

### 3.9 SBATCH headers (built-in defaults of `new_train.slurm`)

Override on the command line *before* the script path
(`sbatch --gpus=2 --mem=50G new_train.slurm …`).

| Header | Default |
|---|---|
| `--partition` | `plgrid-gpu-a100` |
| `--account` | `plgdyplomanci7-gpu-a100` |
| `--nodes` | `1` |
| `--ntasks-per-node` | `1` (the Python app uses `mp.Process` internally) |
| `--cpus-per-task` | `7` |
| `--mem` | `25G` |
| `--time` | `20:50:00` |
| `--gpus` | `1` (set to `ceil(num_workers / workers_per_gpu)` when scaling) |
| `--job-name` | `new-a3c-carla` |
| `--output` / `--error` | `new-a3c-carla-log-%J.txt` |
| `--signal` | `SIGUSR1@90` |

### 3.10 Final command

A complete, ready-to-paste invocation for one worker on one GPU,
default everything else, with frame-saving on episode 700:

```bash
sbatch --ntasks-per-node=2 --cpus-per-task=7 --mem=20G --time=3:00:00 \
       --gpus=1 --job-name=a3c-carla-1w_gpu \
       /net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/new_train.slurm \
       -w 1 -s 14 --save-episodes "700" --workers-per-gpu 2
```

Override anything inline; for example a longer run with a custom LR
and explicit run name:

```bash
sbatch --gpus=1 --mem=25G --time=20:00:00 \
       /net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/new_train.slurm \
       -w 1 --workers-per-gpu 1 --lr 3e-4 --t-max 10 --steps 5000000 \
       --wandb-run-name exp_lr3e4_t10
```

Resume that same run later:

```bash
sbatch /net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/new_train.slurm \
       -w 1 --workers-per-gpu 1 \
       -r /net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/runs/a3c_1w_<ts>_<jobid>
```
