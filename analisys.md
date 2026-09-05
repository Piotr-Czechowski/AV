# A3C Implementation Analysis

## 1. Overall Architecture

The system implements **Asynchronous Advantage Actor-Critic (A3C)** with Hogwild parallelism for CARLA autonomous driving. Key architectural components:

### Process Structure
```
main() new_hogwild_train_a3c_carla.py
├── Telemetry Process (owns events.jsonl + W&B)
├── Run Monitor (node-wide CPU/MEM/GPU sampling)
└── Supervisor
    └── Worker Processes (A3CWorker per worker_id)
    └── Each with CarlaA3CWrapper + local model
```

### Core Files & Responsibilities

| File | Purpose |
|------|---------|
| `new_hogwild_train_a3c_carla.py` | Main entry, arg parsing, pipeline orchestration |
| `new_hogwild_a3c.py` | Core A3C: `A3CWorker`, `SharedActorCritic`, gradient flow |
| `new_hogwild_carla_wrapper.py` | Adapter between CarlaEnv and A3C worker |
| `carla_env.py` | CARLA setup, scenarios, sensors, reward function |
| `new_hogwild_training_logger.py` | JSONL logging, W&B telemetry, record projection |
| `new_hogwild_system_monitor.py` | CPU/MEM/GPU sampling, CARLA process stats |
| `new_hogwild_run_a3c.py` | Worker monitoring, restart, rollback logic |
| `nets/a2c.py` | `SharedActorCritic` neural network architecture |
| `carla_athena_multiserver_v3.py` | Multi-CARLA server management |
| `A3C/tests/test_logging_design.py` | Logging design validation tests |

## 2. Normal Training Flow

### Entry Point (`main()` in `new_hogwild_train_a3c_carla.py`)

```python
def main():
    # 1. Parse arguments and setup config
    args = _apply_config_defaults(build_parser().parse_args())
    
    # 2. Create output directory
    run_output_dir = prepare_output_dir(args, ...)
    
    # 3. Initialize global network (SharedActorCritic + optimizer)
    global_network = GlobalNetwork(config, state_shape, action_shape, critic_shape)
    
    # 4. Start telemetry process (owns events.jsonl + W&B)
    log_queue = mp.Queue(maxsize=TELEMETRY_QUEUE_SIZE)
    telemetry_process = mp.Process(
        target=telemetry_process_main,
        kwargs={...}
    )
    telemetry_process.start()
    
    # 5. Create training logger (per-worker JSONL files)
    events = TrainingLogger(
        run_output_dir, worker_id=-1, telemetry_queue=log_queue, ...
    )
    
    # 6. Start run monitor (CPU/MEM/GPU sampling)
    run_monitor = RunMonitor(run_output_dir, ...)
    run_monitor.start()
    
    # 7. Run with restart supervision
    restart_counts = run_with_restart(
        global_network, config, run_output_dir, shutdown_event,
        log_queue=log_queue, dropped_counter=...
    )
```

### Worker Loop (`A3CWorker.run()` in `new_hogwild_a3c.py`)

Each worker follows this sequence:

```
1. Initialize: sync local model from global, set seed
   ↓
2. Reset: env.reset() → initial state, speed, maneuver
   ↓
3. Step loop (until done or shutdown):
   a. Get action: get_action(state, speed, maneuver)
      ↓
   b. Execute: env.step(action) → (next_state, reward, done, info)
      ↓
   c. Store transition in trajectory
      ↓
   d. Every rollout-length steps (default 20):
      - Compute loss: policy + value - entropy
      - Backward pass: total_loss.backward()
      - Clip gradients: clip_gradients_and_measure()
      - Transfer to global: transfer_local_gradients_to_global()
      - Optimizer step: global_network.optimizer.step()
      - Log update: training_logger.log_update()
      - Increment global step: global_network.increment_global_step()
   e. On episode end:
      - Log episode: training_logger.log_episode()
      - Reset environment: env.reset()
   f. Periodic: checkpoint saving, timing diagnostics, GC
```

### Gradient Flow
```
Worker local model → compute loss (policy_loss + value_loss - entropy_coef*entropy_mean) → 
gradients → transfer_local_gradients_to_global() → 
Hogwild optimizer step → global model updated → 
all workers see new weights asynchronously
```

## 3. Crash Handling Mechanisms

### A. Worker-Detected Crashes (`A3CWorker.run()`)

```python
except RuntimeError as e:
    msg = str(e)
    if 'waiting for the simulator' in msg:
        # CARLA server timeout - full reconnect needed
        training_logger.log_crash_recovery(global_t, error)
        env.reconnect()  # Tear down, wait, reconnect with retries
        clear rollout buffers
    elif 'camera image' in msg or 'time-out' in msg:
        # Camera queue timeout - just skip episode
        log_event('camera_timeout', ...)
        clear rollout buffers
        start new episode
    else:
        raise  # Re-raise unexpected errors
```

### B. Supervisor Monitoring (`run_with_restart()` in `new_hogwild_run_a3c.py`)

The supervisor checks workers every `worker_check_interval` (default 5s):

```
1. Worker died → restart_counts[i] += 1
   ↓
2. Check rapid crash pattern within rapid_crash_window_steps (default 100 steps):
   - If current_step - last_crash_step[i] < 100: rapid_crash_count[i] += 1
   - Else: rapid_crash_count[i] = 0
   ↓
3. If rapid_crash_count[i] >= rapid_crash_threshold (default 3):
   → rollback_global_network() → loads last non-NaN checkpoint
   → Emit 'rollback' event to telemetry with rapid_crash_count
   → rapid_crash_count[i] = 0
   ↓
4. If restart_counts[i] >= max_restarts_per_worker (default 160):
   → Emit 'worker_give_up' event → don't restart this worker
   ↓
5. Exponential backoff before restart:
   wait_time = min(max_wait, base_wait * 2^(restart_counts[i] - 1))
   - Default: 30s → 60s → 120s → 240s → ...
```

### C. Rollback Mechanism (`rollback_global_network()` in `new_hogwild_run_a3c.py`)

Tries checkpoints in order:
1. **Crashing worker's own checkpoint** (if exists): `checkpoints/worker_{idx}/checkpoint.pth`
2. **Top-level checkpoint**: `checkpoint.pth` in run dir
3. **Other workers' checkpoints** (newest first): iterate `checkpoints/` directory

For each candidate:
- Load state dict into global model
- Check for NaN params via `has_nan_params()`
- If clean: restore optimizer state (`load_state_dict`), return success
- If NaN: skip to next candidate

If no usable checkpoint found: print "[ROLLBACK] no usable checkpoint found", return False

### D. Event Emission (`_emit_event()` in `new_hogwild_run_a3c.py`)

All significant events go through telemetry queue to `events.jsonl` and W&B:

```
worker_start, worker_restart, worker_give_up
rollback, nan_gradient, camera_timeout
training_start, training_end, final_summary
```

Each event carries `kind`, `ts`, `worker` (source worker, -1 for run-level), and `global_t`.

## 4. Mermaid Visualization

```mermaid
flowchart TD
    %% Main Entry
    M[main() new_hogwild_train_a3c_carla.py] --> C[Create output dir]
    M --> G[Create GlobalNetwork<br/>SharedActorCritic + optimizer]
    M --> T[Start Telemetry Process<br/>owns events.jsonl + W&B]
    M --> L[Start TrainingLogger<br/>per-worker JSONL files]
    M --> R[Start RunMonitor<br/>node-wide resource sampling]
    
    %% Workers
    G -->|spawns| W[Worker Processes<br/>A3CWorker per worker_id]
    W -->|each worker runs| S[A3CWorker.run() loop]
    
    %% Worker Loop
    S --> I[Initialize networks<br/>sync with global]
    S --> E[Reset environment<br/>CarlaA3CWrapper.reset()]
    S --> L1[For each step:]
    L1 --> A[Get action from local model<br/>get_action()]
    L1 --> E[Execute action<br/>env.step(action)]
    L1 --> S[Store transition<br/>trajectory append]
    L1 --> U[Every rollout-length steps:<br/>compute_and_apply_gradients()]
    U --> Z[Zero gradients<br/>model.zero_grad()]
    U --> B[Backward pass<br/>total_loss.backward()]
    U --> C[Clip gradients<br/>clip_gradients_and_measure()]
    U --> O[Transfer gradients<br/>transfer_local_gradients_to_global()]
    U --> S[Optimizer step<br/>global_network.optimizer.step()]
    U --> L[Log update<br/>training_logger.log_update()]
    U --> F[Increment global step<br/>global_network.increment_global_step()]
    
    %% Episode end
    S --> D[Episode done?]
    D -- Yes --> L2[Log episode metrics<br/>training_logger.log_episode()]
    L2 --> R[Reset environment<br/>env.reset()]
    L2 --> E2[Increment episode count]
    L2 --> T[Send to telemetry<br/>project_update_to_wandb()]
    
    %% Supervisor monitoring
    S -->|Supervisor checks| P[run_with_restart() new_hogwild_run_a3c.py]
    P -->|Every worker_check_interval| WL[Check worker aliveness]
    WL -- Worker alive --> Continue[Continue training]
    WL -- Worker dead --> R1[restart_counts[i] += 1]
    R1 --> RC[Check rapid crash pattern]
    RC -->|Within window, count >= threshold| RB[rollback_global_network()]
    RB --> RL[Emit 'rollback' event to telemetry]
    RL --> RC2[Reset rapid_crash_count[i] = 0]
    RC -->|Not rapid crash| RC3[Check max restarts]
    RC3 -->|Under limit| R2[Restart worker with exponential backoff]
    RC3 -->|At limit| GU[Emit 'worker_give_up' event]
    GU --> SU[Training continues with reduced workers]
    
    %% Crash handling within worker
    S -->|RuntimeError caught| CH[Exception handler in A3CWorker.run()]
    CH -->|'waiting for simulator'| RC4[log_crash_recovery()]
    RC4 --> R5[env.reconnect() with retries]
    R5 --> S6[If successful: continue]
    R5 --> F6[If failed: worker process exits]
    
    CH -->|'camera image/time-out'| CT[log_event('camera_timeout')]
    CT --> CB[Clear rollout buffers]
    CB --> E2[New episode]
    
    %% Telemetry routing
    T -->|Records from workers| TE[run_telemetry_loop()]
    TE -->|Event kind| EV[events.jsonl writer]
    TE -->|Update kind| UW[project_update_to_wandb()]
    TE -->|Episode kind| EQ[project_episode_to_wandb()]
    TE -->|System kind| SY[project_system_to_wandb()]
    
    %% Logging
    L -->|Writes to| LJ[logs/worker_<id>/*.jsonl]
    L -->|Metadata| MD[logs/metadata.json]
    L -->|Events| EV2[logs/events.jsonl]
    
    style M fill:#e1f5fe,stroke:#01579b
    style W fill:#fff3e0,stroke:#ef6c00
    style S fill:#f3e5f5,stroke:#7b1fa2
    style P fill:#e8f5e9,stroke:#2e7d32
```

## 5. Example Call Structure

### Typical SLURM Submission

```bash
sbatch --gpus=4 new_hogwild_train.slurm -w 8 --workers-per-gpu 2 \
  --scenario 14 --camera semantic --res 250 --steps 1000000
```

### Execution Flow

1. **SLURM script** parses arguments, sets up environment
2. **Starts CARLA servers** via `carla_athena_multiserver_v3.py` (one per worker)
3. **Waits for ports** to become LISTEN (polls every 10s, timeout 600s)
4. **Launches training**: `python -u new_hogwild_train_a3c_carla.py` with remaining args
5. **Training process**:
   - Parses config, creates output directory
   - Initializes global network on CPU
   - Starts telemetry process (owns W&B + events.jsonl)
   - Starts run monitor (CPU/MEM/GPU sampling)
   - Starts worker processes (one per worker_id)
6. **Training proceeds** with Hogwild parallelism
7. **On completion/interruption**:
   - Supervisor gracefully shuts down workers
   - Writes `resume_state.json` with cumulative stats
   - Emits `training_end` + `final_summary` events
   - Closes W&B run (if enabled)
   - Exits with proper cleanup

### Key Parameters Affecting Flow

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--num-workers` | 2 | Parallel workers/CARLA servers |
| `--workers-per-gpu` | 2 | Learners per GPU |
| `--rollout-length` | 20 | Steps between global updates |
| `--gamma` | 0.99 | Discount factor for returns |
| `--max-grad-norm` | 5.0 | Gradient clipping threshold |
| `--hogwild-lock-updates` | off | Serialize optimizer steps |
| `--rapid-crash-threshold` | 3 | Crashes within window triggering rollback |
| `--max-restarts-per-worker` | 160 | Restart budget per worker |
| `--reward-mode` | legacy | legacy/shaped reward decomposition |
| `--no-wandb` | off | Disable W&B telemetry |
| `--log-steps` | off | Per-step logging (very verbose) |

## 6. Crash Scenario Walkthroughs

### Scenario A: Single CARLA Server Timeout

```
Worker detects timeout → log_crash_recovery() → env.reconnect() → 
_if successful_: Continue episode from saved state
_if failed after max_connect_retries_: Worker process exits → 
Supervisor detects death → restart_counts[i] += 1 → 
If under max_restarts: Exponential backoff wait → Restart worker
If at limit: worker_give_up event → Training continues with N-1 workers
```

### Scenario B: Rapid Crash Burst (NaN Weights)

```
Multiple worker crashes within 100 global steps → 
rapid_crash_count[i] increments → 
If count >= 3: rollback_global_network() → 
Loads last non-NaN checkpoint → 
Emits 'rollback' event with rapid_crash_count → 
rapid_crash_count reset → Workers continue from restored state
If NaN persists: Eventually rollback to older checkpoint
```

### Scenario C: Camera Queue Timeout

```
Worker detects "time-out waiting for camera image" → 
log_event('camera_timeout', ...) → 
Clear rollout buffers → 
Episode counter increments → 
New episode starts → Training continues 
(No checkpoint saved for dropped episode)
```

### Scenario D: Worker Exhausts Restart Budget

```
Repeated crashes → restart_counts[i] increments each time → 
When >= 160: Emit 'worker_give_up' → 
Worker NOT restarted → 
Training continues with reduced workforce → 
System may become unstable with fewer active workers
```

The design is heavily fault-tolerant, expecting CARLA crashes as a normal occurrence rather than exceptional cases. The supervisor's restart/rollback mechanisms ensure training continuity even with intermittent CARLA server issues.

## 7. Telemetry & Logging Design

Two parallel logging paths:

1. **Local JSONL**: Each worker writes `logs/worker_<id>/episodes.jsonl`, `updates.jsonl`, `timing.jsonl`, `system.jsonl`
2. **Telemetry**: All records forwarded to single consumer process owning `events.jsonl` and W&B

Record kinds: `step`, `episode`, `update`, `timing`, `system`, `event`

Each record carries `kind`, `ts`, `worker` (-1 for run-level), and `global_t`.

### Reward Components (shaped mode)

8 components logged separately:
- `progress`, `target_speed`, `route_penalty`, `time_penalty`
- `goal_bonus`, `collision_penalty`, `offroute_penalty`, `lane_invasion_penalty`
- `total` (clipped sum, duplicates `episode/reward`)

### W&B Projection Functions

- `project_update_to_wandb()`: Projects optimizer updates with `train/<metric>` and `worker_<id>/train/<metric>`
- `project_episode_to_wandb()`: Projects episode outcomes with whitelisted metrics
- `project_system_to_wandb()`: Projects resource samples with system metrics

The telemetry path never averages, downsamples, or alters what training produced - each record becomes its own W&B point.

### Final Summary

The `final_summary` event reports two critical counters:
- `queue_drops` - records dropped because telemetry queue was full (W&B gaps, local files complete)
- `wandb_errors` - failed W&B calls, swallowed so they cannot interrupt training

Both written to `events.jsonl` even with `--no-wandb`.

---

This analysis covers the complete A3C implementation across all provided files, with particular attention to the crash handling mechanisms, normal training flow, and system architecture. The mermaid visualization illustrates the complete process tree and crash recovery flow.
