# 🧠 A3C Hogwild! — Complete Implementation Guide

> **Project:** Autonomous vehicle driving in CARLA using A3C / Hogwild!
> **Directory:** `/net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/`
> **Version:** `new_hogwild_*` (new, multi-process GPU implementation)

---

## Table of Contents

1. [Theoretical Background — A3C from Scratch](#1-theoretical-background--a3c-from-scratch)
2. [System Architecture Map](#2-system-architecture-map)
3. [File-by-File Walkthrough in Startup Order](#3-file-by-file-walkthrough-in-startup-order)
   - 3.10 [`carla_env.py` — Low-Level CARLA Environment](#310-carla_envpy--low-level-carla-environment)
4. [Full Episode Flow](#4-full-episode-flow)
5. [A3C Mathematics — Formulas and Explanations](#5-a3c-mathematics--formulas-and-explanations)
6. [Hogwild! — The Missing Lock is Intentional](#6-hogwild--the-missing-lock-is-intentional)
7. [Checkpointing and Resume](#7-checkpointing-and-resume)
8. [Comparison Table: Old vs New Implementation](#8-comparison-table-old-vs-new-implementation)
9. [SLURM Launch Example](#9-slurm-launch-example)
10. [Common Problems and Debugging](#10-common-problems-and-debugging)
11. [What We Changed in This Refactor](#11-what-we-changed-in-this-refactor)

---

## 1. Theoretical Background — A3C from Scratch

### 1.1 What is A3C?

**A3C** (Asynchronous Advantage Actor-Critic) is a reinforcement learning algorithm proposed by DeepMind in 2016 (Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning"). It combines three key concepts:

- **Actor-Critic**: simultaneous learning of a policy (actor) and a value function (critic)
- **Advantage**: reducing variance of the policy gradient estimator
- **Asynchronous**: parallel workers with no synchronisation between them

### 1.2 A3C Mathematics

#### N-step Returns

Instead of waiting for the end of an episode, A3C uses an n-step estimator:

```
R_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^(n-1)·r_{t+n-1} + γⁿ·V(s_{t+n})
```

Where:
- `r_t` — reward at step `t`
- `γ ∈ [0,1]` — discount factor (default 0.99)
- `V(s_{t+n})` — bootstrap: critic's estimate of the final state (or 0 if episode ended)
- `n = rollout_length` — rollout horizon (default 20 steps)

In code (`new_hogwild_a3c.py`, lines 648-654):
```python
# new_hogwild_a3c.py, fragment of compute_and_apply_gradients
for reward in reversed(self.rewards):
    r = reward / scale if scale and scale != 0.0 else reward
    rewards_scaled.insert(0, r)
    discounted_return = r + self.config.gamma * discounted_return
    returns.insert(0, discounted_return)
```

#### Advantage

```
A(s_t, a_t) = R_t - V(s_t)
```

Advantage tells us whether the taken action was better (A > 0) or worse (A < 0) than average. This reduces variance of the policy gradient.

Advantage normalisation (enabled by default):
```
A_norm = (A - mean(A)) / (std(A) + ε)
```

#### Policy Gradient Loss

```
L_π = -E[log π(a_t|s_t) · A(s_t, a_t)]
```

By minimising this loss, we increase the probability of actions with high advantage.

#### Value Loss (Smooth L1)

```
L_V = c_v · SmoothL1(V(s_t), R_t)
```

Where `c_v = 1.0` (default). Smooth L1 is less sensitive to outliers than MSE:
```
SmoothL1(x, y) = 0.5(x-y)²        if |x-y| < 1
               = |x-y| - 0.5       otherwise
```

#### Entropy Regularisation

```
H(π) = -Σ_a π(a|s) · log π(a|s)
```

Entropy prevents the policy from "freezing" too early. The higher the entropy bonus, the more exploratory the agent.

#### Total Loss

```
L = L_π + c_v · L_V - β · H(π)
```

Where:
- `c_v = 1.0` — value loss weight (value_loss_coef)
- `β` — entropy coefficient, linearly annealed from `β_start=0.02` to `β_end=0.002` over the first 60% of steps

### 1.3 A3C vs A2C — Key Differences

| Feature | A2C (synchronous) | A3C (asynchronous) |
|---|---|---|
| Synchronisation | Waits for all workers | No waiting — Hogwild! |
| Stability | More stable (lower variance) | More chaotic, but faster |
| Throughput | Limited by the slowest worker | No bottleneck |
| Race conditions | None | Possible, but tolerated |
| Model updates | Synchronous batches | Each worker updates the global model immediately |

### 1.4 Hogwild! — Asynchronous SGD Updates

Hogwild! (Recht et al., 2011) is a parallel SGD technique without locks. Key observation: in sparse problems, gradient collisions are rare and have negligible impact on convergence. A3C extends this to dense parameters — variance from race conditions is treated as additional noise that effectively acts as regularisation.

### 1.5 ASCII — General A3C Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           GLOBAL MODEL (CPU)            │
                    │                                          │
                    │  SharedActorCritic / SharedRMSprop       │
                    │  Parameters in shared memory             │
                    │  (torch.share_memory_())                 │
                    └───────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │   Worker 0       │  │   Worker 1       │  │   Worker N       │
    │   GPU:cuda:0     │  │   GPU:cuda:0     │  │   GPU:cuda:1     │
    │   port: 2000     │  │   port: 2100     │  │   port: 2N00     │
    │                  │  │                  │  │                  │
    │ 1. sync_global() │  │ 1. sync_global() │  │ 1. sync_global() │
    │ 2. rollout T     │  │ 2. rollout T     │  │ 2. rollout T     │
    │ 3. compute_grads │  │ 3. compute_grads │  │ 3. compute_grads │
    │ 4. transfer grads│  │ 4. transfer grads│  │ 4. transfer grads│
    │ 5. optim.step()  │  │ 5. optim.step()  │  │ 5. optim.step()  │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                     │
             │    CarlaEnv         │    CarlaEnv         │    CarlaEnv
             │    port:2000        │    port:2100        │    port:2N00
             ▼                     ▼                     ▼
    [CARLA server 0]      [CARLA server 1]      [CARLA server N]
```

---

## 2. System Architecture Map

### 2.1 All Modules and Their Dependencies

```
new_hogwild_train.slurm
    │
    └──► carla_athena_multiserver_v3.py  (external script)
    │        Starts N CARLA servers
    │
    └──► new_hogwild_train_a3c_carla.py  (ENTRY POINT)
              │
              ├──► new_hogwild_prepare_output_dir.py
              │         Creates output directory
              │
              ├──► new_hogwild_training_logger.py  (TrainingLogger)
              │         Writes JSONL logs
              │
              ├──► new_hogwild_system_monitor.py  (RunMonitor)
              │         Monitors CPU/GPU/RAM/CARLA processes
              │
              ├──► new_hogwild_a3c.py  (GlobalNetwork)
              │         Global model in shared memory
              │
              └──► new_hogwild_run_a3c.py  (run_with_restart)
                        Supervisor — launches and restarts workers
                        │
                        └──► new_hogwild_a3c.py  (A3CWorker)
                                  Each worker:
                                  │
                                  ├──► new_hogwild_carla_wrapper.py  (CarlaA3CWrapper)
                                  │         CARLA environment wrapper
                                  │         │
                                  │         └──► carla_env.py  (external)
                                  │
                                  ├──► new_hogwild_training_logger.py
                                  │         Worker logs
                                  │
                                  ├──► new_hogwild_system_monitor.py  (WorkerMonitor)
                                  │         Worker process monitoring
                                  │
                                  └──► new_hogwild_timing_utils.py  (TimingAccumulator)
                                            Timing of loop phases
```

### 2.2 Data Flow — Global CPU Model vs Worker GPU

```
┌──────────────────────────────────────────────────────────────┐
│                    SHARED MEMORY (CPU)                        │
│                                                              │
│  GlobalNetwork.model (SharedActorCritic)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │   CNN      │  │   trunk    │  │  policy    │             │
│  │  weights   │  │  weights   │  │  head      │             │
│  └────────────┘  └────────────┘  └────────────┘             │
│                                                              │
│  SharedRMSprop (optimizer state: square_avg, step)           │
│  All tensors: .share_memory_() → mmap                        │
└──────────────┬───────────────────────────────────────────────┘
               │                           ▲
    sync_with_global()        transfer_local_gradients_to_global()
    CPU → GPU: local_param.data.copy_(global_param)
               │    GPU → CPU: global_param.grad = local_param.grad.to(cpu).clone()
               │                           │
               ▼                           │
┌─────────────────────────────┐            │
│   Worker (GPU: cuda:0)      │            │
│                             │            │
│  local_model = copy of      │  gradient  │
│  SharedActorCritic on GPU   │ ─────────►│  optimizer.step() on CPU
│                             │            │  updates the global model
│  Rollout rollout_length steps:      │            │
│  - forward pass (GPU)       │            │
│  - collect transitions      │            │
│  - compute loss (GPU)       │            │
│  - loss.backward() (GPU)    │            │
└─────────────────────────────┘
```

---

## 3. File-by-File Walkthrough in Startup Order

### 3.1 `new_hogwild_train.slurm` — SLURM Entry Point

Bash script launched by SLURM. Initialises the entire training environment.

#### SLURM Directives (lines 1-13)

```bash
# new_hogwild_train.slurm, lines 1-13
#SBATCH --partition=plgrid-gpu-a100   # Partition with A100 cards
#SBATCH --nodes=1                     # Single compute node
#SBATCH --ntasks-per-node=1           # One main SLURM task
#SBATCH --cpus-per-task=7             # 7 CPUs per task (1 main + N workers)
#SBATCH --mem=25G                     # 25 GB RAM
#SBATCH --time=20:50:00               # Maximum time: ~21h
#SBATCH --gpus=1                      # 1 GPU (can be overridden with sbatch --gpus=N)
#SBATCH --account=plgdyplomanci7-gpu-a100
#SBATCH --job-name=new-hogwild-a3c-carla
#SBATCH --output=new-hogwild-a3c-carla-log-%J.txt  # Single file for stdout+stderr
#SBATCH --signal=SIGUSR1@90           # Send SIGUSR1 90s before time limit
```

The `--signal=SIGUSR1@90` directive is critical — Python handles this signal and performs a graceful shutdown, saving state before the job is killed.

#### Startup Sequence

```
SLURM allocates resources
       │
       ▼
Load modules (GCCcore, Python 3.11, CUDA 12.4, cuDNN 9.2)
       │
       ▼
Activate venv (/net/tscratch/people/plgbartoszkawa/venv)
       │
       ▼
Set environment variables:
  OMP_NUM_THREADS=1, MKL_NUM_THREADS=1
  PYTHONUNBUFFERED=1, PYTHONFAULTHANDLER=1
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
       │
       ▼
Create output directory:
  runs/a3c_hogwild_{N}w_{TIMESTAMP}_{SLURM_JOB_ID}/
       │
       ▼
Launch nvidia-smi dmon (GPU watchdog in background)
       │
       ▼
Start N CARLA servers via carla_athena_multiserver_v3.py
       │
       ▼
Wait until all ports (2000, 2100, ...) are in LISTEN state
  (poll every 10s, max 60 attempts = 600s)
       │
       ▼
Build and launch training command:
  python -u new_hogwild_train_a3c_carla.py [args] >> a3c_training.log &
       │
       ▼
tail -f a3c_training.log (live log view)
       │
       ▼
Wait for Python to finish (wait $TRAIN_PID)
       │
       ▼
cleanup(): SIGTERM Python → wait 75s → SIGKILL → pkill CarlaUE4
```

#### Cleanup Mechanism

The `cleanup()` function is registered as a trap on `SIGINT`, `SIGTERM`, `SIGUSR1`, `EXIT`. When SLURM sends `SIGUSR1` 90s before the time limit:
1. Python receives the signal via Python's `signal.signal(SIGUSR1, handler)`
2. The handler sets `shutdown_event` — workers finish their current episodes
3. SLURM sends `SIGTERM` to the whole process after 90s
4. `cleanup()` kills remaining processes and CARLA servers

---

### 3.2 `new_hogwild_prepare_output_dir.py` — Directory Structure

Simple function that creates the output directory and saves launch arguments.

```python
# new_hogwild_prepare_output_dir.py, lines 8-31
def prepare_output_dir(args, user_specified_dir=None, resume=False):
```

#### What It Creates

```
runs/a3c_hogwild_2w_20240512_143022_12345/
├── args.txt                     # JSON with full launch configuration
├── args_resume.txt              # (resume only) new arguments
├── checkpoint.pth               # Latest global checkpoint
├── checkpoint_step.txt          # Step number of latest checkpoint
├── best_checkpoint.pth          # Best checkpoint (by reward)
├── resume_state.json            # Resume state (elapsed_time, step, etc.)
├── a3c_training.log             # Python stdout/stderr (via SLURM)
├── carla_servers.log            # CARLA server logs
├── gpu_dmon.log                 # nvidia-smi dmon output
├── wandb_run_id.txt             # Weights & Biases run ID
├── checkpoints/
│   ├── worker_0/                # Per-worker checkpoints (optional)
│   │   ├── checkpoint.pth
│   │   └── checkpoint_step.txt
│   └── worker_1/ ...
└── logs/
    ├── metadata.json            # Config, architecture, parameter count
    ├── events.jsonl             # Global events (start, end, restart, rollback)
    ├── system.jsonl             # System metrics (CPU, RAM, GPU, CARLA)
    ├── worker_0/
    │   ├── episodes.jsonl       # Per-episode logs
    │   ├── updates.jsonl        # Per-update logs (losses, gradients, lr)
    │   ├── steps.jsonl          # Per-step logs (optional, --log-steps)
    │   ├── timing.jsonl         # Timing profiles of loop phases
    │   └── system.jsonl         # Worker process resource usage
    └── worker_1/ ...
```

With `resume=True`, `args_resume.txt` is created instead of `args.txt` to avoid overwriting the original arguments.

---

### 3.3 `new_hogwild_train_a3c_carla.py` — Python Entry Point

Main Python file. Parses arguments, initialises everything, and starts training.

#### Configuration Constants (uppercase)

Constants are deliberately separated from the CLI — normal SLURM runs don't need to provide them:

| Constant | Default Value | Description |
|---|---|---|
| `DEFAULT_NUM_WORKERS` | 2 | Number of parallel A3C workers |
| `DEFAULT_WORKERS_PER_GPU` | 2 | Workers per GPU |
| `DEFAULT_ROLLOUT_LENGTH` | 20 | Rollout horizon (steps before update) |
| `DEFAULT_GAMMA` | 0.99 | Discount factor |
| `DEFAULT_LR` | 1e-4 | Initial learning rate |
| `DEFAULT_BETA_START` | 0.02 | Initial entropy coefficient |
| `DEFAULT_BETA_END` | 0.002 | Final entropy coefficient |
| `DEFAULT_BETA_ANNEAL_FRAC` | 0.6 | Fraction of steps for β annealing |
| `DEFAULT_VALUE_LOSS_COEF` | 1.0 | Value loss weight (c_v) |
| `DEFAULT_MAX_GRAD_NORM` | 40.0 | Gradient clipping |
| `DEFAULT_NORMALIZE_ADVANTAGES` | True | Advantage normalisation |
| `DEFAULT_STEPS` | 10_000_000 | Total step budget |
| `DEFAULT_ACTION_REPEAT` | 2 | How many times to repeat action in CARLA |
| `DEFAULT_EPISODE_MAX_DECISIONS` | 100 | Max decisions per episode |
| `DEFAULT_SAVE_FREQUENCY` | 100_000 | Steps between checkpoints |
| `DEFAULT_REWARD_CLIP` | 50.0 | Reward clip after shaping |

#### GPU Assignment to Workers

```python
# new_hogwild_train_a3c_carla.py, line 443-461
def _assign_worker_gpus(num_workers, workers_per_gpu, worker_gpu_start):
```

Assignment logic:
```
Example: 4 workers, 2 workers_per_gpu, GPUs: [0, 1]

expanded = [cuda:0, cuda:0, cuda:1, cuda:1]
            W0       W1       W2      W3
```

If there are more workers than GPU slots, the pattern repeats cyclically.

#### GlobalNetwork Initialisation

```python
# new_hogwild_train_a3c_carla.py, line 500-502
global_network = GlobalNetwork(
    config, state_shape=[args.res, args.res, 3],
    action_shape=args.n_actions, critic_shape=1)
```

The model lives in CPU memory with `share_memory()` enabled — every forked subprocess sees the same tensors via the OS mmap mechanism.

#### Signal Handling

```python
# new_hogwild_train_a3c_carla.py, line 427-440
for signame in ('SIGTERM', 'SIGUSR1', 'SIGINT'):
    signal.signal(sig, _handle_signal)
```

The handler sets `shutdown_event` — all workers check this in their loops and stop after finishing the current step/episode.

#### WandB in a Separate Process

WandB logs metrics via a dedicated subprocess `WandBLogger` that receives data through `mp.Queue`. Isolation prevents training from hanging if WandB fails (e.g., no internet on the HPC node).

---

### 3.4 `new_hogwild_a3c.py` — THE HEART OF THE IMPLEMENTATION

The most important file. Contains the entire A3C algorithm.

#### SharedRMSprop and SharedAdam — Mathematics

**RMSprop** adapts the learning rate per parameter:

```
E[g²]_t = α · E[g²]_{t-1} + (1-α) · g²_t
θ_{t+1} = θ_t - η / (√E[g²]_t + ε) · g_t
```

Where:
- `α = 0.99` (rmsprop_alpha) — moving average momentum for squared gradients
- `ε = 1e-5` (rmsprop_eps) — numerical stabilisation
- `η` — learning rate (linearly decays to 0)

**Adam** uses two moments:

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t          (first moment)
v_t = β₂·v_{t-1} + (1-β₂)·g²_t         (second moment)
m̂_t = m_t / (1 - β₁ᵗ)                  (bias correction)
v̂_t = v_t / (1 - β₂ᵗ)                  (bias correction)
θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)
```

**Key: Shared State**

```python
# new_hogwild_a3c.py, lines 61-68 (SharedRMSprop.share_memory)
for group in self.param_groups:
    for p in group['params']:
        state = self.state[p]
        for key in ('step', 'square_avg'):
            value = state.get(key)
            if torch.is_tensor(value) and value.device.type == 'cpu':
                value.share_memory_()
```

Optimizer state tensors (`square_avg`, `step`) are moved to shared memory. Every worker sees and updates the same state — this is the essence of Hogwild!. Races on `square_avg` are possible, but their impact on convergence is negligible.

#### SharedActorCritic — Network Architecture

```
Input: semantic image [3, 250, 250]
       + speed [1]
       + maneuver [1] (as one-hot [3])
            │
            ▼
┌──────────────────────────────────────────┐
│           CNN (shared trunk)             │
│  Conv2d(3→32, k=5, s=2, p=2) + ReLU     │  [32, 125, 125]
│  Conv2d(32→64, k=3, s=2, p=1) + ReLU    │  [64, 63, 63]
│  Conv2d(64→128, k=3, s=2, p=1) + ReLU   │  [128, 32, 32]
│  Conv2d(128→256, k=3, s=2, p=1) + ReLU  │  [256, 16, 16]
│  AdaptiveAvgPool2d(4,4)                  │  [256, 4, 4] = 4096
└──────────────────────────────────────────┘
            │
 flatten → [4096]
            │
      ┌─────┴────────────────┐
      │    speed_fc           │  maneuver_fc
      │  Linear(1→32) + ReLU │  Linear(3→32) + ReLU
      └─────┬────────────────┘
            │
      cat([4096, 32, 32]) = [4160]
            │
    ┌───────┴────────────────┐
    │       trunk            │
    │  Linear(4160→512)+ReLU │
    │  Linear(512→256)+ReLU  │
    └──────┬─────────────────┘
           │
     ┌─────┴──────┐
     ▼             ▼
policy head    value head
Linear(256→N)  Linear(256→1)
(logits)       (V(s))
```

- `policy head` → `Categorical(logits=...)` → `action_distribution.sample()` → discrete action
- `value head` → `V(s)` → used for bootstrap and advantage calculation

#### GlobalNetwork — CPU Shared Memory

```python
# new_hogwild_a3c.py, lines 257-307
class GlobalNetwork:
```

GlobalNetwork is not a PyTorch model — it's a container holding:

- `self.model` — `SharedActorCritic` with `.share_memory()`
- `self.optimizer` — `SharedRMSprop` with `.share_memory()`
- `self.global_step` — `mp.Value('l', 0)` — atomic step counter
- `self.global_episode` — `mp.Value('l', 0)` — atomic episode counter
- `self.total_updates` — `mp.Value('l', 0)` — model update counter
- `self.best_reward` — `mp.Value('d', -inf)` — best reward seen
- `self.recent_rewards` — `mp.Array('d', [0.0]*100)` — circular buffer of 100 latest rewards
- `self.worker_mean_rewards` — `mp.Array('d', [0.0]*N)` — mean reward per worker

**Why CPU, not GPU?**
- Shared memory only works on CPU (`share_memory_()` requires pinned memory)
- Workers on different GPUs might not see the same memory
- CPU→CPU gradient transfer is fast (copy+mmap, no PCIe)
- The optimizer runs on CPU with minimal cost

#### A3CWorker.run() — Full Flow

```python
# new_hogwild_a3c.py, line 862
def run(self):
```

Detailed flow of one worker iteration:

```
START run()
   │
   ├─ Set environment variables (OMP_NUM_THREADS=1, etc.)
   ├─ Set seed for RNG (seed + worker_id * 1009, unique per worker)
   ├─ _init_networks() — create local model copy on GPU
   ├─ Initialise TrainingLogger, WorkerMonitor, TimingAccumulator
   │
   ▼ EPISODE LOOP (until shutdown_event)
   │
   ├─ _clear_rollout_buffers() — clear trajectory, rewards
   ├─ increment_global_episode() — atomically increment global counter
   ├─ sync_with_global() — copy weights CPU → GPU
   ├─ env.reset() → state, speed, maneuver
   │
   ▼ STEP LOOP (until done and not shutdown)
   │
   ├─ increment_global_step(1) — atomically increment global_step
   ├─ Check if global_t > config.steps → if so: shutdown_event.set()
   │
   ├─ Prepare tensors: state_tensor, speed_tensor, maneuver_tensor → to GPU
   │
   ├─ get_action(state_tensor, speed_tensor, maneuver_tensor)
   │     ├─ model forward pass (GPU)
   │     ├─ Categorical(logits=logits).sample()
   │     └─ Append to trajectory: Transition(value, log_prob, entropy, action)
   │
   ├─ env.step(action)
   │     ├─ action_repeat times: step_apply_action + world.tick()
   │     ├─ get next_state, reward, done, info
   │     └─ shape_reward() → reward_f, components
   │
   ├─ Append reward_f to self.rewards
   ├─ episode_total_reward += reward_f
   │
   ├─ If steps_since_last_update >= rollout_length or done:
   │     compute_and_apply_gradients() — UPDATE
   │     if not done: sync_with_global()
   │     steps_since_last_update = 0
   │
   ├─ If global_t % save_frequency == 0:
   │     _save_checkpoint()
   │
   ▼ END OF EPISODE
   │
   ├─ gc.collect() every gc_interval episodes
   ├─ _log_episode() — write to JSONL
   └─ Check is_new_best → save best_checkpoint.pth
```

#### compute_and_apply_gradients() — Algorithm Core

```python
# new_hogwild_a3c.py, line 626
def compute_and_apply_gradients(self, final_state_tensor, done, ...):
```

**Step by step:**

```
STEP 1: Bootstrap
   if done:
       discounted_return = 0
   else:
       discounted_return = V(s_{t+n})  [forward pass without gradient, torch.no_grad()]

STEP 2: N-step Returns (backwards through rewards)
   for reward in reversed(self.rewards):
       discounted_return = reward + gamma * discounted_return
       returns.prepend(discounted_return)

STEP 3: Assemble batch from trajectory
   values    = [V(s_t), V(s_{t+1}), ..., V(s_{t+n-1})]
   log_probs = [log π(a_t|s_t), ...]
   entropies = [H(π(·|s_t)), ...]

STEP 4: Compute Advantage
   advantages = returns - values.detach()
   if normalize_advantages:
       advantages = (advantages - mean) / (std + 1e-8)

STEP 5: Losses
   policy_loss = -mean(log_probs * advantages.view(-1))
   value_loss  = value_loss_coef * smooth_l1(values, returns)
   entropy_mean = mean(entropies)
   entropy_coef = β(global_t)   [linear annealing]
   total_loss = policy_loss + value_loss - entropy_coef * entropy_mean

STEP 6: Backprop (GPU)
   model.zero_grad()
   total_loss.backward()
   clip_grad_norm_(model.parameters(), max_grad_norm=40.0)

STEP 7: Check NaN gradients
   if nan_grads: log_nan(), sync_with_global(), return

STEP 8: Transfer gradients GPU → CPU
   transfer_local_gradients_to_global(local_model, global_model, device='cpu')

STEP 9: Update global model (CPU)
   [optional: with global_network.update_lock:]
   set_lr_for_step(global_t)   [linear decay to 0]
   global_optimizer.step()
   global_optimizer.zero_grad()

STEP 10: Save metrics
   increment_total_updates(), log_update(...)
   _clear_rollout_buffers()
```

#### transfer_local_gradients_to_global() — GPU → CPU

```python
# new_hogwild_a3c.py, lines 184-199
def transfer_local_gradients_to_global(local_model, global_model, global_device):
    for local_param, global_param in zip(local_model.parameters(),
                                         global_model.parameters()):
        if local_param.grad is None:
            global_param.grad = None
        else:
            global_param.grad = local_param.grad.detach().to(
                global_device, non_blocking=False).clone()
```

**Why `.clone()`?**
Gradients are not in shared memory. Each worker assigns its own tensor as `global_param.grad`. Without `.clone()` the grad would be a reference to a GPU tensor — after `zero_grad()` clears it, the value would be lost. With `.clone()` each worker has its own CPU buffer that the optimizer reads and then clears.

#### sync_with_global() — CPU → GPU

```python
# new_hogwild_a3c.py, lines 580-595
def sync_with_global(self):
    for global_param, local_param in zip(
            self.global_network.model.parameters(),
            self.model.parameters()):
        local_param.data.copy_(
            global_param.data.to(self.device, non_blocking=True))
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)
```

`non_blocking=True` enables asynchronous copy; `synchronize()` guarantees the copy is complete before continuing. Synchronisation is only needed for CUDA — without it, the forward pass might use stale weights.

#### ASCII Animation of One Worker Iteration

```
Worker 0 (GPU:0)                  GlobalNetwork (CPU)
─────────────────────────────────────────────────────
sync_with_global()
  CPU params ──copy──► GPU params

  step t=1:  obs → forward → (action, value, entropy)
             env.step(action) → next_obs, reward
             trajectory.append(), rewards.append()

  step t=2:  ... (repeat)

  step t=rollout_length:
             discounted_return = V(s_T) [bootstrap]
             cumulative_return_T-1 = r_T-1 + γ·discounted_return
             ...
             cumulative_return_1 = r_1 + γ·cumulative_return_2

             advantages = R - V
             normalize(advantages)

             loss = policy_loss + value_loss - β·entropy
             loss.backward()          [GPU backprop]
             clip_grad_norm_()

             transfer_local_gradients_to_global()  [GPU grad → CPU grad]
                                 ─────────────────────────►
                                         global_param.grad = local_grad.clone()
                                         optimizer.step()    [Adam/RMSprop]
                                         optimizer.zero_grad()
                                 ◄─────────────────────────
sync_with_global()               [updated weights]
  CPU params ──copy──► GPU params

  step t=rollout_length+1: obs → forward → (action, value, entropy)
  ...
```

---

### 3.5 `new_hogwild_run_a3c.py` — Supervisor

The supervisor manages the worker lifecycle — starts, monitors, and restarts workers.

#### Launch Pattern

```python
# new_hogwild_run_a3c.py, line 177
def run_with_restart(global_network, config, run_output_dir, shutdown_event, ...):
```

**Startup:** For each `i` in `range(num_workers)`:
- port = `start_port + port_step * i`
- device = `worker_gpus[i]` (e.g. `cuda:0`)
- Create `A3CWorker(...)` and call `.start()` (fork process)

**Monitor loop** (every `worker_check_interval=5s`):
```
check global_step >= steps → if so: shutdown_event.set()

for each worker i:
    if w.is_alive(): continue

    w.join(timeout=2)
    restart_counts[i] += 1
    current_step = global_step.value

    compute rapid_crash:
    if current_step - last_crash_step[i] < rapid_crash_window_steps:
        rapid_crash_count[i] += 1
    else:
        rapid_crash_count[i] = 0

    if restart_counts[i] >= max_restarts_per_worker:
        "giving up" — do not restart

    if rapid_crash_count[i] >= rapid_crash_threshold:
        rollback_global_network() — restore last checkpoint

    wait_time = min(max_wait, base_wait * 2^(restart_counts[i]-1))
    # Exponential backoff: 30s, 60s, 120s, ... max 300s
    sleep(wait_time) with shutdown_event check

    launch new worker
```

#### Exponential Backoff

```
restart_count=1: wait = 30s
restart_count=2: wait = 60s
restart_count=3: wait = 120s
restart_count=4: wait = 240s
restart_count=5+: wait = 300s (max_backoff)
```

This protects against fast-crash loops that could exhaust resources.

#### rollback_global_network()

When a worker crashes `rapid_crash_threshold=3` times within `rapid_crash_window_steps=100` steps:

```python
# new_hogwild_run_a3c.py, line 55
def rollback_global_network(global_network, run_output_dir, worker_idx=None):
```

Checkpoint candidate priority:
1. Crashing worker's checkpoint (`checkpoints/worker_{i}/checkpoint.pth`)
2. Global checkpoint (`checkpoint.pth`)
3. Other workers' checkpoints (newest first)

For each candidate:
- Load `state_dict`
- Check for NaN parameters (`has_nan_params`)
- If OK — load model + optimizer, return `True`
- If NaN — move to next candidate

---

### 3.6 `new_hogwild_carla_wrapper.py` — Environment Adapter

The wrapper isolates the raw `CarlaEnv` API from A3C logic. Workers only see:
- `reset() → (state_np, speed_f, maneuver_int)`
- `step(action) → (next_np, next_speed_f, maneuver, reward_f, done, info)`

#### Action Repeat

```python
# new_hogwild_carla_wrapper.py, lines 369-372
for _ in range(self._action_repeat):
    self.env.step_apply_action(int(action))
    self.env.world.tick()
```

With default `action_repeat=2`, each worker decision is executed over two simulator ticks. Effects:
- Smoother physics (CARLA physics runs at 20 Hz, decisions at ~10 Hz)
- Effectively halves the agent's decision frequency
- One worker step = `action_repeat` simulator steps

#### Shaped Reward — Components

`shaped` mode (default) combines 8 components:

```
reward = progress            (progress toward goal)
       + target_speed        (penalty/bonus for speed)
       + route_penalty       (penalty for deviating from route)
       + time_penalty        (constant cost per step)
       + goal_bonus          (bonus for reaching goal)
       + collision_penalty   (penalty for collision)
       + offroute_penalty    (penalty for leaving route)
       + lane_invasion_penalty (penalty for lane crossing)
```

**Detailed formulas:**

```python
progress = clip(prev_goal_dist - current_goal_dist, -5.0, 5.0)
         * reward_progress_coef  # default 1.0
# Example: was 50m from goal, now 47m → progress = 3.0

target = max(1.0, reward_target_speed_kmh)  # default 20 km/h
speed_score = 1.0 - min(2.0, |speed_kmh - target| / target)
            * reward_target_speed_coef  # default 1.0
# Example: driving at 20 km/h → speed_score = 1.0
# Example: driving at 0 km/h  → speed_score = 1.0 - 1.0 = 0.0
# Example: driving at 40 km/h → speed_score = 1.0 - 2.0 = -1.0

route_penalty = -reward_route_penalty_coef * min(route_dist, offroute_threshold)
# Always negative; penalty grows with distance from route up to threshold (10m)

time_penalty = -reward_time_penalty  # -0.01 per step

goal_bonus = +50.0  # if done and distance_from_goal < 3.0m

collision_penalty = -50.0  # if new_collision (new collision since last step)

offroute_penalty = -25.0  # if route_distance >= offroute_threshold (10m)

lane_invasion_penalty = -5.0 * lane_invasions  # per lane crossing event

# Final clip:
reward = clip(sum(components), -50.0, +50.0)
```

**Numerical example — good step:**
```
Car drives at 20 km/h, approached goal by 2m, no collision, route OK

progress           =  1.0 * 2.0   =  +2.00
target_speed       =  1.0 * 1.0   =  +1.00
route_penalty      = -0.1 * 0.0   =  -0.00  (route: 0m from centerline)
time_penalty       =        -0.01 =  -0.01
goal_bonus         =         0.0  =   0.00
collision_penalty  =         0.0  =   0.00
offroute_penalty   =         0.0  =   0.00
lane_invasion      =         0.0  =   0.00
─────────────────────────────────────────
TOTAL                              =  +2.99
```

**Numerical example — bad step:**
```
Car is stationary, 12m off route, crossed lane marking

progress           =  1.0 * 0.0   =   0.00
target_speed       =  1.0 * 0.0   =   0.00  (stationary → score=0)
route_penalty      = -0.1 * 10.0  =  -1.00  (max at offroute=10m)
time_penalty       =        -0.01 =  -0.01
goal_bonus         =         0.0  =   0.00
collision_penalty  =         0.0  =   0.00
offroute_penalty   =        -25.0 =  -25.0   (route_dist=12 >= 10)
lane_invasion      = -5.0 * 1     =  -5.00
─────────────────────────────────────────
TOTAL                              =  -31.01
```

#### reset_episode_state vs reload_world

```python
# new_hogwild_carla_wrapper.py, lines 258-300
def reset(self):
    ...
    full_reload = self._world_reload_interval > 0 and \
        self.episode % self._world_reload_interval == 0
    state, speed = self.env.reset(
        save_image=save_images, episode=self.global_episode,
        reload_world=full_reload)
```

- **Normal reset** (`reload_world=False`): fast — reset vehicle position, clear collision history, new spawn. Takes ~0.5s.
- **Full reload** (`reload_world=True`): full CARLA map reload. Used every `world_reload_interval` episodes (default 0 = disabled). Takes ~5-10s, but eliminates simulator error accumulation.

---

### 3.7 `new_hogwild_training_logger.py` — JSONL Logger

Each worker has its own `TrainingLogger` instance. Files are opened with `buffering=1` (line buffering) — every line is immediately flushed to disk. Data survives process crashes.

#### Record Types

**episodes.jsonl** — one record per completed episode:
```json
{
  "ts": "2024-05-12T14:30:22.123",
  "worker": 0,
  "global_episode": 1042,
  "global_t": 500210,
  "total_reward": 47.3,
  "steps": 87,
  "mean_reward": 0.544,
  "reached_goal": false,
  "duration_s": 12.4,
  "max_speed_kmh": 24.1,
  "min_route_dist": 0.8,
  "goal_dist": 23.4,
  "collisions": 0,
  "local_mean_reward": 12.4,
  "global_mean_reward": 15.1,
  "is_new_best": false,
  "action_counts": [5, 32, 15, 8, 12, 7, 3, 5],
  "reward_components": {"progress": 12.0, "target_speed": 30.1, ...}
}
```

**updates.jsonl** — one record per model update (every rollout_length steps):
```json
{
  "ts": "2024-05-12T14:30:22.300",
  "worker": 0,
  "update": 25010,
  "global_t": 500200,
  "trajectory_length": 20,
  "is_terminal": false,
  "pi_loss": -0.234,
  "v_loss": 0.156,
  "total_loss": -0.051,
  "gradient_norm": 12.4,
  "lr": 0.000095,
  "entropy_coef": 0.018,
  "advantages_mean": 0.002,
  "advantages_std": 1.0,
  "val_mean": 14.2,
  "rew_mean": 1.4,
  "rew_sum": 28.0,
  "reward_progress_sum": 18.0,
  "reward_target_speed_sum": 15.0
}
```

**events.jsonl** — global events:
```json
{"ts": "2024-05-12T14:00:00", "event": "training_start", "n_workers": 2}
{"ts": "2024-05-12T14:15:00", "event": "checkpoint_save", "path": "...", "global_t": 100000}
{"ts": "2024-05-12T14:20:00", "event": "worker_crash", "worker": 1, "error": "CARLA timeout"}
{"ts": "2024-05-12T14:20:30", "event": "worker_restart", "worker": 1, "restart_count": 1}
{"ts": "2024-05-12T14:30:00", "event": "nan_gradient", "nan_count": 3, "nan_layers": ["cnn.0.weight"]}
{"ts": "2024-05-12T14:59:30", "event": "training_end", "cumulative_elapsed_s": 3570}
```

**timing.jsonl** — loop phase profiles (every `diag_log_interval=100` updates):
```json
{
  "ts": "2024-05-12T14:30:22",
  "worker": 0,
  "window_updates": 100,
  "ops": {
    "backward":       {"avg_ms": 45.2, "count": 100, "total_s": 4.52},
    "env_reset":      {"avg_ms": 512.0, "count": 5,  "total_s": 2.56},
    "env_step":       {"avg_ms": 48.3, "count": 2000,"total_s": 96.6},
    "forward":        {"avg_ms": 8.1,  "count": 2000,"total_s": 16.2},
    "loss_compute":   {"avg_ms": 12.4, "count": 100, "total_s": 1.24},
    "optim_update":   {"avg_ms": 3.2,  "count": 100, "total_s": 0.32},
    "sync":           {"avg_ms": 6.8,  "count": 105, "total_s": 0.71}
  }
}
```

---

### 3.8 `new_hogwild_timing_utils.py` — Profiler

Simple time accumulator with minimal overhead:

```python
# new_hogwild_timing_utils.py, lines 17-53
class TimingAccumulator:
    __slots__ = ('_totals', '_counts')
```

Using `__slots__` eliminates instance dictionary overhead — important when called this frequently (every step, every phase).

**Measured phases:**
- `sync` — weight synchronisation time CPU→GPU
- `env_reset` — CARLA environment reset time
- `forward` — forward pass time (GPU inference)
- `env_step` — CARLA step time (action_repeat ticks + observation)
- `loss_compute` — loss computation time
- `backward` — backpropagation time
- `optim_update` — optimizer update time + gradient transfer
- `checkpoint_save` — checkpoint save time

These metrics identify the bottleneck: if `env_step >> forward`, the environment is the bottleneck. If `backward >> optim_update`, the GPU is under-powered for this network.

---

### 3.9 `new_hogwild_system_monitor.py` — System Monitor

Two monitors running as daemon threads (they don't block shutdown):

#### RunMonitor

One per run. Samples every `monitor_interval=10s`:
- **CPU**: `cpu_percent_mean`, `cpu_percent_max`, `cpu_count_busy`, `cpu_freq_mhz`
- **RAM**: `mem_used_gb`, `mem_available_gb`, `mem_percent`, `swap_used_gb`
- **CARLA processes**: find PIDs via `psutil.process_iter(['name', 'cmdline'])`, checking for `CarlaUE4` in name. Per process: `cpu_percent`, `rss_gb`
- **GPU** (via pynvml): `util_percent`, `mem_used_gb`, `mem_total_gb` per device

Writes to `logs/system.jsonl`.

#### WorkerMonitor

One per worker. Samples the worker PID every 10s:
- `cpu_percent`, `rss_gb`, `vms_gb`, `num_threads`
- `ctx_voluntary`, `ctx_involuntary` — context switches

Writes to `logs/worker_{id}/system.jsonl`.

**Why `daemon=True` for monitors?**
Daemon threads are automatically killed when the main process exits — no explicit stopping needed on crash. The `stop()` method is called in each worker's `finally` block.

---

### 3.10 `carla_env.py` — Low-Level CARLA Environment

`carla_env.py` is a **reference file** — it is NOT part of the `new_hogwild_*` family, but all workers communicate with CARLA exclusively through `CarlaEnv`. The `CarlaA3CWrapper` uses `CarlaEnv` as a library and exposes only the clean `reset()` / `step()` interface to A3C. This section documents the CARLA layer in as much detail as the training layer — without it, incidents like "camera timeout" or "reward = 0 the entire episode" are difficult to debug.

#### 3.10.1 What is `CarlaEnv`

`CarlaEnv` manages:
- Connection to the CARLA UE4 process (`carla.Client("localhost", port)`, 120s timeout).
- Loading the `Town03` map and setting **synchronous mode** (`fixed_delta_seconds=0.1`, `max_substeps=10`).
- Spawning a Tesla Model 3 (the only ego-vehicle), sensors (RGB or semantic camera, collision, lane_invasion), and the spectator.
- Planning a route between the spawn point and goal via `GlobalRoutePlanner` — producing a list of `(Waypoint, RoadOption)` pairs and a list of intermediate "middle goals" dense at `mp_density` metres.
- Computing the legacy reward (`reward_function` from `utils.py` + `static_reward_mp` as a bonus for passing intermediate waypoints).

The CARLA client and server communicate via RPC. Each A3C worker runs its own `CarlaEnv` instance on a unique port (e.g. 2000, 2100, ...), which physically isolates simulation worlds between workers.

#### 3.10.2 `CarlaEnv.__init__` (lines 105-211)

```python
# carla_env.py, lines 105-211
def __init__(self, scenario, action_space='discrete', resX=250, resY=250,
             camera='semantic', port=port, manual_control=False,
             spawn_point=False, terminal_point=False, mp_density=25,
             verbose=False):
    self.client = carla.Client("localhost", port)
    self.client.set_timeout(120.0)
    self.verbose = bool(verbose)
    ...
    self.world = self.client.load_world('Town03')
    self._apply_sync_settings()
    self.blueprint_library = self.world.get_blueprint_library()
    self.map = self.world.get_map()
    self.scenario_list = scenario
    self.scenario = self.scenario_list[0]
    self.middle_goals = []
    self.middle_goals_density = mp_density
    self.create_scenario(self.sp, self.tp, self.middle_goals_density)
    self.spectator = self.set_spectator()
    self.goal_location_trans, self.goal_location_loc, self.route = \
        self.plan_the_route()
    self.action_space = self.create_action_space(action_space)
    ...
    self.state_observer = StateObserver()
    self.planner = None
    self.number_of_resets = 0
    self.car_decisions = []
```

What happens here, in order:
1. **CARLA server connection**: `carla.Client` with a 120s timeout (enough for `world.tick()` under load — `CarlaA3CWrapper` additionally retries `max_connect_retries` times on failure).
2. **Synchronous mode** (`_apply_sync_settings`): without this, CARLA runs at native speed and the camera/physics desynchronise from the worker. `fixed_delta_seconds=0.1` = one frame every 100ms of virtual time; `max_substeps=10` limits how many physics sub-steps CARLA performs in one `world.tick()`.
3. **Scenario selection**: `scenario` is a number/list; each number maps to a different set of spawn points and goals in `Town03` (via `create_scenario`). Scenarios 12-15 randomly pick goals from `MAP_POINTS_SC*` (e.g. `MAP_POINTS_SC14` — right/straight route).
4. **Route planning**: `GlobalRoutePlanner` (CARLA) generates a list of waypoints. **Maneuvers** (`car_decisions`) are extracted from it as a sequence of `LEFT(0)/STRAIGHT(1)/RIGHT(2)`, filtering out `LANEFOLLOW` and `CHANGELANE*`. These maneuvers are later passed to `SharedActorCritic` as a one-hot vector `[3]`.
5. **Middle goals**: for long route segments, intermediate waypoints are added every `mp_density=25 m`. Each intermediate goal gives `mp_reward` (legacy reward bonus) when the vehicle comes within <3 m (`static_reward_mp`).
6. **Actions** (`create_action_space`): discrete mode creates a list from `settings.ACTIONS` (list of action names from `ACTIONS.py`). Continuous mode leaves `action_space='continuous'` and uses `car_control_continuous` with `(throttle, steer, brake)` instead of a predefined move table.
7. **State observer**: `StateObserver()` — a helper object that holds the "latest frame" as an `image` attribute. The wrapper reads `state_observer.image` to save a frame to disk (`carla.Image.save_to_disk`).
8. **`verbose=False` by default** (line 117): this parameter was added **to fix a regression** — `AttributeError: 'CarlaEnv' object has no attribute 'verbose'` from an earlier version. Previously `verbose` was only read inside `__init__` but some methods referenced `self.verbose`. Now it is explicitly assigned.

#### 3.10.3 `_apply_sync_settings` (lines 212-218)

```python
def _apply_sync_settings(self):
    self.settings = self.world.get_settings()
    self.settings.synchronous_mode = True
    self.settings.fixed_delta_seconds = 0.1
    self.settings.max_substep_delta_time = 0.01
    self.settings.max_substeps = 10
    self.world.apply_settings(self.settings)
```

Consequences of synchronisation:
- CARLA does not advance on its own. Every step in the world requires `world.tick()` (called by the wrapper).
- `fixed_delta_seconds=0.1` means one `world.tick()` = 100ms of virtual time — regardless of how long rendering takes. This allows episodes to be replayed deterministically.
- `max_substep_delta_time=0.01` and `max_substeps=10` tell CARLA: physics sub-steps can be 10ms each and there can be up to 10 (= max 100ms = `fixed_delta_seconds`). This keeps physics stable even when `delta_seconds` is large.

#### 3.10.4 `reset(reload_world=True/False)` (lines 1169-1237)

```python
def reset(self, episode, save_image=False, reload_world=True):
    if reload_world:
        self.reload_world()
    else:
        self.reset_episode_state()

    if self.scenario:
        self.scenario = random.choice(self.scenario_list)
        self.create_scenario(self.sp, self.tp, self.middle_goals_density)

    self.plan_the_route()
    self.spawn_car(spawning_type, episode)
    self.set_spectator()
    if self.camera_type == 'rgb':
        self.add_rgb_camera()
    elif self.camera_type == 'semantic':
        self.add_semantic_camera()
    self.add_collision_sensor()
    self.add_line_invasion_sensor()

    # warm-up: apply action 3 (brake) and 15 ticks so the camera starts emitting frames
    self.step_apply_action(3)
    for i in range(15):
        self.world.tick()

    # clear camera queue of initial frames (they are underexposed)
    while not self.image_queue.empty():
        _ = self.image_queue.get()

    self.world.tick()
    image = self._get_latest_camera_image(timeout=2.0)
    self.state_observer.image = image

    if self.camera_type == 'rgb':
        self.process_rgb_img(image)
    else:
        self.process_semantic_img(image)
    return self.front_camera, float(self.speed)
```

**`reload_world=True`** calls `reload_world()`: `self.client.reload_world()` — a full map reload in CARLA. This takes 5-10 seconds. All actors are destroyed, the map is loaded fresh, and blueprint_library and map are re-fetched. Necessary periodically (`world_reload_interval` in the wrapper) because CARLA accumulates garbage after hundreds of episodes (e.g. RAM growth, broken debug string tables).

**`reload_world=False`** calls `reset_episode_state()` — the CHEAPER path.

#### 3.10.5 `reset_episode_state()` (NEW method, lines 1141-1158)

```python
def reset_episode_state(self):
    self.destroy_agents()
    self.actor_list = []
    if hasattr(self, 'image_queue'):
        while not self.image_queue.empty():
            _ = self.image_queue.get()
    self.collision_history_list = []
    self.invasion_history_list = []
    self.middle_goals = []
    self.step_counter = 0
    self.stat_reward_mp = []
    self.front_camera = None
    self.preview_camera = None
    self.preview_camera_enabled = False
    self.is_junction = False
    self.done = False
    self.speed = 0
    self.prev_speed = 0
```

Step by step:
1. **`destroy_agents()`** — destroys all actors from `self.actor_list` (vehicle, camera, collision, lane invasion). Calls `actor.stop()` on sensors, `actor.destroy()` on vehicles. Essential because CARLA does not remove actors automatically.
2. **`self.actor_list = []`** — clears the list.
3. **Flushing `image_queue`** — empties the queue of incoming camera frames from the previous episode (there may be dozens of unread frames from the `camera.listen` hook).
4. **Resetting history** — `collision_history_list`, `invasion_history_list`, `middle_goals`, `stat_reward_mp` (list of `[location, redeem_flag]` pairs for intermediate goals).
5. **Resetting state** — `step_counter = 0`, `done = False`, speeds zeroed, camera nulled, `is_junction = False`.

**Why cheaper than `reload_world`**: we skip `client.reload_world()` (5-10s) and `world.load_world('Town03')`. The CARLA world keeps running — we just remove the episode's actors and start planning a new route in the same map. This reduces reset time from ~7s to ~0.5s.

#### 3.10.6 `_get_latest_camera_image()` (NEW method, lines 1160-1167)

```python
def _get_latest_camera_image(self, timeout=2.0):
    try:
        image = self.image_queue.get(timeout=timeout)
    except queue.Empty:
        raise RuntimeError("time-out waiting for camera image")
    while not self.image_queue.empty():
        image = self.image_queue.get()
    return image
```

What it does:
- Waits up to 2 seconds for the first frame from `image_queue`. If none arrives, raises `RuntimeError("time-out waiting for camera image")`.
- Then **reads all subsequent frames** without blocking (`while not image_queue.empty()`), overwriting the `image` variable — only the **latest** frame is returned.

What problem it solves:
- CARLA sends camera frames to the callback `camera.listen(image_queue.put)` automatically, regardless of whether the worker reads them. After several `world.tick()` calls in one step (e.g. `action_repeat=2`), multiple frames sit in the queue.
- Without this method, the worker would receive the oldest frame instead of the newest — the agent would see the world "one tick behind".
- Second behaviour: when `image_queue` is empty (camera not yet started), the 2-second timeout gives a readable error message instead of blocking forever.

`A3CWorker` catches this `RuntimeError`: if the message contains "camera image" or "time-out", it treats it as a soft crash and skips to the next episode **without reconnecting to the server** (just `world.tick()` and the camera will recover its rhythm). The other case — "waiting for the simulator" — requires a full reconnect (`env.reconnect()`).

#### 3.10.7 Returning `float(self.speed)` in `reset()` and `step()` (lines 1237 and 1310)

```python
return self.front_camera, float(self.speed)          # reset
...
return self.front_camera, reward, self.done, route_distance, \
       speed_value, distance_from_goal              # step
```

`self.speed` is a scalar `float` computed by `calculate_speed()` from vehicle velocity (`3.6 * sqrt(vx² + vy² + vz²)`). In `step()` it is repackaged as `speed_value = float(self.speed)`.

**Why not a tensor**: in previous versions, speed was sometimes a CUDA tensor. This caused problems on machines where `CUDA_VISIBLE_DEVICES` differed between CARLA processes and workers — tensors from `CarlaEnv` had a device that `A3CWorker` couldn't see. Converting to `float` at the CarlaEnv → wrapper → worker boundary eliminates the issue: each process builds its own tensor locally (`torch.tensor([[speed]], device=self.device)`).

#### 3.10.8 `last_invasion_counter` in `step()` (line 1286)

```python
if len(self.invasion_history_list) != 0:
    invasion_counter = 1
else:
    invasion_counter = 0
self.last_invasion_counter = invasion_counter
self.invasion_history_list = []
```

`last_invasion_counter` is an instance attribute set on every step. **The wrapper consumes it** (`new_hogwild_carla_wrapper.py`):
```python
lane_invasions = getattr(self.env, 'last_invasion_counter', 0)
```
and passes it to `_shape_reward` as the `lane_invasion_penalty`. Without this attribute, the wrapper would have to inspect `invasion_history_list` BEFORE `step()` clears it — impossible because the wrapper sees `step()` as an atomic operation.

#### 3.10.9 `state_observer.image = image` (lines 1228 and 1301)

```python
self.state_observer.image = image
# carla.Image — raw CARLA object with save_to_disk() method
```

`StateObserver` is a simple container — no logic, just an `image` attribute. In `reset()` and `step()`, `CarlaEnv` simply stores the latest CARLA frame as an attribute.

**Why the wrapper, not the environment, decides on saving**: saving frames to JPEG is optional, depends on the episode (`_save_episodes`, `_save_episode_interval`) and path policy (`<project>/episodes/<run_id>/<global_episode>-<port>/<step>.jpeg`). The CARLA environment knows nothing about run_id or global episodes — those are wrapper attributes. The environment only exposes "the latest CARLA frame" via `state_observer.image`; the wrapper retrieves it in `_save_frame()` and calls `carla_img.save_to_disk(...)` itself. The CARLA environment remains stateless with respect to saving.

#### 3.10.10 Action Space: Discrete vs Continuous (lines 507-513, 1000-1030)

```python
def create_action_space(self, action_space):
    if action_space == 'discrete':
        self.action_space = [getattr(ac, action) for action in settings.ACTIONS]
        return self.action_space
    else:
        self.action_space = action_space
        return self.action_space
```

- **`discrete`** (default in A3C): `settings.ACTIONS` is a list of action names (e.g. `'ACCEL_LEFT'`, `'BRAKE'`). Each maps to a `(throttle, brake, steer)` triple via `ACTIONS.ACTION_CONTROL`. The worker chooses an action index (Categorical), and `car_control_discrete(action)` sets the vehicle controls.
- **`continuous`**: `action` is a vector `[gas/brake, steer]`. `car_control_continuous` splits `action[0]` into `throttle` (positive part) and `brake` (negative part), `action[1]` into `steer`. NOT used in the current A3C (Categorical requires discrete actions), but the code is ready for extension to a continuous policy (e.g. PPO with Gaussian).

#### 3.10.11 Reward — `reward_function` + `static_reward_mp` (lines 1247-1310)

```python
reward, done = reward_function(
    self.collision_history_list, invasion_counter,
    self.speed, route_distance,
    mp_static_reward, terminal_state_reward,
    on_junction, self.prev_speed)
```

`reward_function` (from `utils.py`) computes the legacy reward combining:
- penalty for collisions (`collision_history_list`),
- penalty for lane violations (`invasion_counter`),
- bonus/penalty for speed and route distance (`route_distance`),
- bonus for passing an intermediate goal (`mp_static_reward`),
- terminal bonus for reaching the goal (`terminal_state_reward`),
- speed difference (`prev_speed → speed`) as a driving stability signal.

This is the **legacy reward**. The wrapper receives it from `step()` as the first returned element, but then:
- In `reward_mode='legacy'` mode (`new_hogwild_carla_wrapper.py`, `_shape_reward`): the wrapper passes `reward` unchanged, adding component `{'legacy': reward}`.
- In `reward_mode='shaped'` mode (default): the wrapper IGNORES `reward` and computes its own reward from 8 components (progress, target_speed, route_penalty, time_penalty, goal_bonus, collision_penalty, offroute_penalty, lane_invasion_penalty), using `route_distance`, `speed_kmh`, `distance_from_goal`, `collisions`, `lane_invasions` received from `step()`.

In short: `CarlaEnv` produces "raw" sensor data + legacy reward; the wrapper decides whether and how to transform them into the A3C learning signal.

#### 3.10.12 Image Queue in Cameras (lines 528, 588)

```python
# add_rgb_camera (line 528) and add_semantic_camera (line 587)
self.image_queue = queue.Queue()
rgb_cam.listen(self.image_queue.put)         # rgb
semantic_cam_sensor.listen(self.image_queue.put)  # semantic
```

A standard Python queue (`queue.Queue`, not multiprocessing). `camera.listen(callback)` registers a callback invoked asynchronously by CARLA. The FIFO queue has unlimited capacity.

**The wrapper manages the queue**: before each `step()`, the wrapper flushes the queue:
```python
# new_hogwild_carla_wrapper.py, step()
if hasattr(self.env, 'image_queue'):
    while not self.env.image_queue.empty():
        self.env.image_queue.get()
for _ in range(self._action_repeat):
    self.env.step_apply_action(int(action))
    self.env.world.tick()
```
Why: if the queue had frames from the previous step, `step()` in CarlaEnv would retrieve an old frame as the "current" observation, desynchronising the agent from the world. After flushing the queue + `world.tick()`, we are guaranteed the first frame after the tick is fresh.

#### 3.10.13 Sensors — collision, lane_invasion, RGB, semantic

```python
# add_collision_sensor (lines 665-673)
col_sensor_bp = self.blueprint_library.find('sensor.other.collision')
col_sensor_bp = self.world.spawn_actor(col_sensor_bp, self.transform,
                                        attach_to=self.vehicle)
col_sensor_bp.listen(lambda data: self.collision_data_registering(data))
```

- **Collision sensor**: callback appends the event to `self.collision_history_list`. The wrapper reads `len(env.env.collision_history_list)` to log the number of collisions.
- **Lane invasion sensor**: callback appends the event to `self.invasion_history_list`. CarlaEnv consolidates this into `last_invasion_counter` (0 or 1 per step).
- **RGB / Semantic camera**: standard camera (RGB in natural colours) vs semantic (each object class = different colour after `cc.CityScapesPalette`). Default is semantic — easier to learn from, as the image does not depend on world lighting or colours.

The wrapper integrates everything: `collisions = len(env.env.collision_history_list)`, `lane_invasions = getattr(env.env, 'last_invasion_counter', 0)`, camera = `state_observer.image` (or via `front_camera` tensor).

#### 3.10.14 `step_counter` and `STEP_COUNTER` (line 1298)

```python
if self.step_counter >= how_many_steps:
    self.done = True
```

`how_many_steps = settings.STEP_COUNTER` (global constant from `settings.py`). This is the **hard CARLA environment limit** — regardless of A3C policy, after `STEP_COUNTER` ticks `CarlaEnv` sets `done=True`. This is a safeguard against infinite episodes when the agent drives in circles.

**Vs `episode_max_decisions` in the wrapper**: the wrapper has its own decision counter (`self.step_count`) and its own limit (`episode_max_decisions` in `new_hogwild_carla_wrapper.py`, default 100). The wrapper ends the episode when:
```python
if self._episode_max_decisions > 0 and \
        self.step_count >= self._episode_max_decisions:
    done = True
```
The wrapper counts **agent decisions**; `CarlaEnv` counts all ticks (including those from `action_repeat`). With `action_repeat=2` and `episode_max_decisions=100`, CarlaEnv sees up to 200 ticks = `step_counter` grows to 200. The `STEP_COUNTER` limit is typically much higher (e.g. 500), so in practice the wrapper limit dominates.

#### 3.10.15 ASCII — Full Flow: `wrapper.reset()` → `CarlaEnv.reset()` → ticks → `wrapper.step()`

```
CarlaA3CWrapper.reset()                       CarlaEnv                         CARLA server (UE4)
─────────────────────────────────────────────────────────────────────────────────────
self.episode += 1
self._episode_total_reward = 0.0
self._save_dir_cached = None

full_reload = (world_reload_interval > 0
               and episode % interval == 0)

env.reset(save_image=save_images,           ──────►  reset(reload_world=full_reload)
          episode=global_episode,                       │
          reload_world=full_reload)                     │ if full_reload:
                                                        │     reload_world()  ──────►  client.reload_world()
                                                        │     # 5-10 s, fresh map
                                                        │ else:
                                                        │     reset_episode_state()
                                                        │     # ~0.5 s
                                                        │
                                                        │ create_scenario(...)
                                                        │ plan_the_route()
                                                        │ spawn_car(...)         ──►  world.spawn_actor(tesla)
                                                        │ add_*_camera()         ──►  camera.listen(queue.put)
                                                        │ add_collision_sensor() ──►  collision.listen(...)
                                                        │ add_line_invasion_sensor()
                                                        │
                                                        │ step_apply_action(3)   # brake — warm-up
                                                        │ for _ in range(15):    ──►  15 × world.tick()
                                                        │     world.tick()
                                                        │
                                                        │ # flush camera queue
                                                        │ while not queue.empty(): queue.get()
                                                        │ world.tick()           ──►  one tick
                                                        │ image = _get_latest_camera_image()
                                                        │ state_observer.image = image
                                                        │ process_*_img(image) → self.front_camera (tensor)
                                                        │
                                              ◄────────  return front_camera, float(self.speed)

state_np = _state_to_chw_float(state)
speed_f = _speed_to_float(speed) / 100.0
self._prev_goal_dist = self._current_goal_distance()

return state_np, speed_f, self._current_maneuver


─── STEP LOOP ─────────────────────────────────────────────────────────────────────
wrapper.step(action):

  # flush camera queue of frames between steps
  while not env.image_queue.empty():
      env.image_queue.get()

  for _ in range(action_repeat=2):
      env.step_apply_action(action)         ──►  vehicle.apply_control(...)
      env.world.tick()                       ──►  CARLA simulates 100 ms
                                                  + emits frames to queue

  next_state, reward, done,
  route_dist, next_speed,
  dist_from_goal = env.step(...)            ──────►  step(episode, step, save_image):
                                                       │ calculate_distance()
                                                       │ calculate_route_distance()
                                                       │ self.speed = calculate_speed()
                                                       │ static_reward_mp()
                                                       │ # done if reached mp/terminal
                                                       │
                                                       │ # save lane_invasion count
                                                       │ self.last_invasion_counter = ...
                                                       │ self.invasion_history_list = []
                                                       │
                                                       │ reward, done = reward_function(...)
                                                       │
                                                       │ if step_counter >= STEP_COUNTER:
                                                       │     self.done = True
                                                       │
                                                       │ image = _get_latest_camera_image(2.0)
                                                       │ # waits until frame appears in queue
                                                       │ # or RuntimeError → wrapper catches and restarts episode
                                                       │ state_observer.image = image
                                                       │ process_*_img(image) → self.front_camera
                                                       │
                                              ◄────────  return front_camera, reward, done,
                                                                route_dist, float(self.speed),
                                                                dist_from_goal

  next_np = _state_to_chw_float(next_state)
  collisions = len(env.collision_history_list)
  lane_invasions = getattr(env, 'last_invasion_counter', 0)
  reward_f, components = _shape_reward(reward, ...)
  if step_count >= episode_max_decisions: done = True
  _save_frame()  # if _save_images and state_observer.image available
  return next_np, next_speed_f, maneuver, reward_f, done, info
```

#### 3.10.16 Table: Wrapper Layer vs CarlaEnv Layer

| Aspect | `CarlaEnv` (carla_env.py) | `CarlaA3CWrapper` (new_hogwild_carla_wrapper.py) |
|---|---|---|
| CARLA server connection | `carla.Client("localhost", port)` with 120s timeout | Retry × `max_connect_retries`, waits `connect_retry_wait` between attempts, `reconnect()` on failure |
| CARLA synchronous mode | `_apply_sync_settings()`: `fixed_delta_seconds=0.1`, `max_substeps=10` | Uses by default — does not override settings |
| Vehicle spawning | `spawn_car(spawning_type, episode)` — Tesla Model 3 | Does not spawn itself; passes `global_episode` in `env.reset(episode=...)` |
| Camera (RGB/semantic) | `add_rgb_camera()` / `add_semantic_camera()` with `image_queue = queue.Queue()` | Flushes queue before each `step()` so agent sees the freshest frame |
| Latest frame | `_get_latest_camera_image(timeout=2.0)` — drains queue, returns last | Catches `RuntimeError("time-out waiting for camera image")` and restarts episode without reconnect |
| Collision/lane sensors | Callbacks append to `collision_history_list`, `invasion_history_list` | Reads `len(env.collision_history_list)`, `env.last_invasion_counter` for `_shape_reward` penalties |
| Reset | `reload_world()` (full) or `reset_episode_state()` (lightweight) | Chooses mode via `world_reload_interval`: full every N episodes, lightweight otherwise |
| Returned observation | `self.front_camera`: tensor `[1, 3, H, W]` float | `_state_to_chw_float`: → numpy `[3, H, W]`, /255.0, NaN and [0,1] range check |
| Speed | `float(self.speed)` in km/h | Divides by 100 (normalisation to ~[0,1]) before entering `SharedActorCritic` |
| Maneuvers | `car_decisions` from `plan_the_route()` (list of 0/1/2) | Tracks current maneuver index in `_update_maneuver()` per step, passes to agent |
| Reward | Legacy: `reward_function(...)` from `utils.py` + `static_reward_mp` | `legacy` mode: pass through; `shaped` mode: ignore, compute 8 own components |
| Goal distance | `calculate_distance()` returns `(distance, vehicle_location)` | Caches `_prev_goal_dist` to compute `progress = prev - curr` |
| Step limit | `step_counter >= STEP_COUNTER` (hard env limit, ~500) | `step_count >= episode_max_decisions` (typically 100 decisions × `action_repeat=2`) |
| Saving frames to disk | Only `state_observer.image = image` — exposes `carla.Image` object | `_save_frame()` with `_should_save_this_episode()`, `_save_dir_cached`, `image.save_to_disk(path)` |
| Actor lifecycle | `destroy_agents()` in `reload_world()` / `reset_episode_state()` | Does not manage actors; just holds `self.env` reference and clears `env.env.world = None` in finally |
| Action mode | `create_action_space('discrete'|'continuous')` | Passes `action_space=config.action_type` (always `'discrete'` in A3C) |
| Error handling | Raises `RuntimeError("time-out ...")` from `_get_latest_camera_image` or CARLA timeouts | Catches two error types: "waiting for the simulator" → reconnect; "camera image" → skip episode |
| `verbose` | `__init__(verbose=False)` — controls `self.log.success/warn` and print in `spawn_car` | `verbose_env_logs` in config → passed to `CarlaEnv(verbose=...)` |

---

## 4. Full Episode Flow

```
                    SUPERVISOR (new_hogwild_run_a3c.py)
                          │
                          │ worker.start()
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    A3CWorker.run()                              │
│                                                                 │
│  1. INITIALISATION (one-time)                                   │
│     ├─ _init_networks() → LocalSharedActorCritic on cuda:0      │
│     ├─ sync_with_global() → copy weights CPU → GPU             │
│     ├─ Initialise TrainingLogger, WorkerMonitor, TimingAccumulator│
│     └─ Initialise CarlaA3CWrapper (connect to CARLA port:2000)  │
│                                                                 │
│  ══════════════════ EPISODE LOOP ═══════════════════           │
│                                                                 │
│  2. EPISODE SETUP                                               │
│     ├─ _clear_rollout_buffers()  [trajectory=[], rewards=[]]   │
│     ├─ increment_global_episode() → global_episode = 1042      │
│     ├─ sync_with_global()  [CPU→GPU params copy, ~6ms]         │
│     │                                                           │
│     └─ env.reset()                                             │
│           ├─ Reset internal state (episode_total_reward=0, etc.)│
│           ├─ Decide: full_reload? (every world_reload_interval) │
│           ├─ CarlaEnv.reset() → teleport vehicle               │
│           ├─ Normalise obs: [H,W,C] → [C,H,W], /255.0         │
│           └─ Return: state_np, speed_f, maneuver_int           │
│                                                                 │
│  ══════════════════ STEP LOOP ════════════════════             │
│  (rollout_length=20 steps or episode end)                       │
│                                                                 │
│  3. ONE STEP:                                                   │
│     ├─ increment_global_step(1) → global_t = 500210            │
│     │                                                           │
│     ├─ FORWARD PASS (GPU, ~8ms)                                │
│     │   ├─ state_tensor = torch.from_numpy(state).unsqueeze(0).cuda│
│     │   ├─ logits, value = model(state_tensor, speed_tensor, maneuver_tensor)│
│     │   ├─ action_distribution = Categorical(logits=logits)    │
│     │   ├─ action = action_distribution.sample() [stochastic]  │
│     │   ├─ log_prob = action_distribution.log_prob(action)      │
│     │   ├─ entropy = action_distribution.entropy()              │
│     │   └─ trajectory.append(Transition(value, log_prob, ent)) │
│     │                                                           │
│     ├─ ENV STEP (CARLA, ~48ms)                                 │
│     │   ├─ For _ in range(action_repeat=2):                     │
│     │   │   ├─ env.step_apply_action(action)  [send to CARLA]  │
│     │   │   └─ world.tick()  [one simulation tick]             │
│     │   ├─ Get: next_state, reward, done, info                 │
│     │   ├─ shape_reward() → reward_f, components               │
│     │   └─ Return: next_np, next_speed_f, maneuver, done       │
│     │                                                           │
│     ├─ rewards.append(reward_f)                                │
│     ├─ episode_total_reward += reward_f                        │
│     │                                                           │
│     └─ If steps_since_last_update >= rollout_length or done:   │
│           COMPUTE AND APPLY GRADIENTS (see section 3.4)        │
│           ├─ ~12ms loss_compute                                │
│           ├─ ~45ms backward                                    │
│           └─ ~3ms optim_update                                 │
│                                                                 │
│  4. END OF EPISODE                                             │
│     ├─ gc.collect() every gc_interval=10 episodes             │
│     ├─ _log_episode() → episodes.jsonl                        │
│     │   └─ update_stats() → global_mean_reward, is_new_best  │
│     │                                                           │
│     └─ If is_new_best:                                        │
│           save(best_checkpoint.pth)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼ (next episode: go back to step 2)
```

---

## 5. A3C Mathematics — Formulas and Explanations

### 5.1 Complete Equations

**N-step Return (bootstrap):**
```
R_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^(n-1)·r_{t+n-1} + γⁿ·V(s_{t+n})

if episode ended:
    R_t = r_t + γ·r_{t+1} + ... + γ^(n-1)·r_{t+n-1}
    (no bootstrap: V(terminal) = 0)
```

**Advantage:**
```
A(s_t, a_t) = R_t - V_θ(s_t)

after normalisation:
    A_norm = (A - μ_A) / (σ_A + 1e-8)
    where μ_A = mean(A_0,...,A_{T-1}), σ_A = std(A_0,...,A_{T-1})
```

**Policy Loss:**
```
L_π = -1/T · Σ_{t=0}^{T-1} log π_θ(a_t|s_t) · A_norm(s_t, a_t)

gradients: ∇_θ L_π = -1/T · Σ ∇_θ log π_θ(a_t|s_t) · A_norm(s_t, a_t)
```

**Value Loss (Smooth L1 / Huber):**
```
L_V = c_v · 1/T · Σ_{t=0}^{T-1} SmoothL1(V_θ(s_t), R_t)

SmoothL1(x, y) = { 0.5·(x-y)²        if |x-y| < 1
                 { |x-y| - 0.5        if |x-y| ≥ 1
```

**Entropy:**
```
H(π_θ(·|s_t)) = -Σ_{a∈A} π_θ(a|s_t) · log π_θ(a|s_t)

(for categorical distribution with logits: PyTorch Categorical.entropy())
```

**Annealing β (entropy coefficient):**
```
β(t) = β_start + min(1, t/t_anneal) · (β_end - β_start)

where:
  β_start = 0.02
  β_end   = 0.002
  t_anneal = steps * beta_anneal_frac = 10M * 0.6 = 6M steps

At t=0:       β = 0.020
At t=3M:      β = 0.011 (halfway through annealing)
At t=6M:      β = 0.002 (end of annealing)
At t>6M:      β = 0.002 (constant)
```

**Total Loss:**
```
L_total = L_π + c_v · L_V - β(t) · H̄(π)

where H̄(π) = 1/T · Σ H(π_θ(·|s_t))  (mean entropy in the batch)
```

**Learning Rate (linear decay):**
```
lr(t) = lr_0 · max(0, (T_max - t - 1) / T_max)

For T_max=10M:
  t=0:    lr = 1e-4
  t=5M:   lr = 0.5e-4
  t=10M:  lr ≈ 0
```

**Gradient Clipping:**
```
g̃ = g · min(1, max_grad_norm / ‖g‖₂)

where max_grad_norm = 40.0
```

### 5.2 Policy Gradient — Intuition

The policy gradient says: "make the action that turned out better than expected more probable". Advantage is key — without it, we would reward every action proportionally to the absolute reward, not to how good it was *relative to the state's value*.

- `A > 0`: action was better than expected → increase its probability
- `A < 0`: action was worse than expected → decrease its probability
- `A ≈ 0`: action was in line with expectations → no strong update

---

## 6. Hogwild! — The Missing Lock is Intentional

### 6.1 Why the Missing Update Lock is OK

Classic SGD requires a lock when multiple threads update the same parameters. Hogwild! proves that for **sparse gradients** (each update touches different parameters) collisions are rare and do not affect convergence.

In A3C gradients are not sparse, but:
1. Races are rare (a worker executes rollout_length=20 steps before each update)
2. The effect of an incorrect write is like additional noise to SGD — and SGD is already stochastic
3. Experimentally, A3C converges despite races (Mnih et al. 2016)

The code provides the `--hogwild-lock-updates` option (flag on `GlobalNetwork.update_lock`):
```python
# new_hogwild_a3c.py, lines 726-728
update_ctx = self.global_network.update_lock \
    if self.config.hogwild_lock_updates else contextlib.nullcontext()
with update_ctx:
    ...optimizer.step()...
```

By default the lock is **disabled** — this is Hogwild! in its pure form.

### 6.2 What Races Are Possible and Why Tolerated

| Race | Description | Impact |
|---|---|---|
| Gradient overwrite | Worker A writes `global_param.grad` before Worker B reads it | Worker B updates the model with A's gradient — "stale gradient". Effect like momentum |
| square_avg overwrite (RMSprop) | Two workers update `square_avg` simultaneously | One write is lost. Effect: sub-noise in lr estimation |
| step counter | `step` tensor may be off by +/-1 | Negligible impact on bias correction |
| zero_grad race | Worker A calls zero_grad(), Worker B just wrote grad | B's grad is lost. Effect like a skipped update |

All these races have limited impact because:
- They affect scalar/gradient tensors, not entire model parameters
- Model parameters are updated by `optimizer.step()` (which reads, not naively overwrites parameters)
- Race noise < noise from the stochastic CARLA environment

### 6.3 What the New Version (new_hogwild_*) Improves Over the Old

The old implementation used `global_param.grad = local_param.grad` without `.clone()`. This meant that after `zero_grad()` the gradient was zeroed out. The new version:

```python
# New version — safe buffer:
global_param.grad = local_param.grad.detach().to(cpu, non_blocking=False).clone()
```

`.clone()` creates a new CPU buffer for each transfer — the optimizer has its own tensor that `zero_grad()` clears only after use.

---

## 7. Checkpointing and Resume

### 7.1 Exact Structure of checkpoint.pth

```python
state = {
    # Metadata
    'global_step': 500000,            # Total steps across all workers
    'global_episode': 1042,           # Total episodes
    'total_updates': 25010,           # Total model updates
    'last_checkpoint_boundary': 5,    # Number of last boundary (step//save_freq)
    'best_reward': 127.4,             # Best episode reward so far
    'global_mean_reward': 34.2,       # Mean of last 100 episodes
    'worker_mean_rewards': [32.1, 36.3],  # Per-worker means
    'recent_rewards': [45.2, 12.1, ...],  # Circular buffer of 100 latest rewards
    'recent_reward_count': 100,       # How many elements are in the buffer
    'recent_reward_index': 1142,      # Index of next position in buffer

    # Model (for shared arch)
    'model': model.state_dict(),      # Network weights (CNN + trunk + heads)
    'optimizer': optimizer.state_dict(), # Optimizer state (square_avg, step)
}
```

### 7.2 How Boundary Checkpointing Works

```python
# new_hogwild_a3c.py, line 402-460
def save_boundary_checkpoint(self, run_output_dir, global_t, worker_id=None):
    save_frequency = int(getattr(self.config, 'save_frequency', 0))
    boundary = int(global_t) // save_frequency
    if boundary <= self.last_checkpoint_boundary.value:
        return False, None  # This boundary already saved by another worker
    ...
```

The boundary mechanism ensures that a checkpoint is saved exactly once per `save_frequency` steps, regardless of which worker reaches the boundary first. The worker that saves boundary `N` sets `last_checkpoint_boundary = N`, blocking other workers from duplicating it.

### 7.3 How Resume Works

```
sbatch new_hogwild_train.slurm --resume /path/to/old/run/

1. OUTPUT_DIR = /path/to/old/run/
2. prepare_output_dir() opens the existing directory (does not create a new one)
3. args_resume.txt → saves NEW configuration alongside the old one
4. find_latest_checkpoint(run_output_dir) finds checkpoint.pth
5. global_network.load(checkpoint.pth)
   ├─ Loads model.state_dict()
   ├─ Loads optimizer.state_dict() + .share_memory()
   ├─ Restores global_step, global_episode, best_reward, etc.
6. _read_resume_state() reads elapsed_training_s from resume_state.json
   (or from events.jsonl if resume_state.json is absent)
7. elapsed_offset = previous training time
8. Launch run_with_restart() — workers start from global_step = 500000
```

Resume warnings: code detects changes in key parameters (lr, gamma, rollout_length, optimizer, weight_decay, beta_*) and prints a WARNING, but does not block. The checkpoint MUST contain the key `model` (and optionally `optimizer`) — `GlobalNetwork.load()` raises `ValueError` with a readable message if the key is missing (e.g. attempting to resume a legacy `actor`+`critic` checkpoint from an older version of the code).

### 7.4 best_checkpoint.pth

Saved when a new episode has a higher reward than `global_network.best_reward`:

```python
# new_hogwild_a3c.py, line 820-825
if is_new_best:
    best_path = os.path.join(self.run_output_dir, 'best_checkpoint.pth')
    if self.global_network.save(best_path, global_t=global_t):
        training_logger.log_checkpoint(path=best_path, global_t=global_t,
                               checkpoint_kind='best')
```

`best_checkpoint.pth` has the identical structure to `checkpoint.pth`. This file is used for evaluation — it represents the model with the best performance achieved so far, not the latest state (which may be worse after a regression).

---

## 8. Comparison Table: Old vs New Implementation

| Aspect | Old implementation (async-rl) | New implementation (new_hogwild_*) |
|---|---|---|
| **Gradient transfer** | `global_param.grad = local_param.grad` (no clone, risk of zeroing) | `global_param.grad = local_param.grad.detach().to(cpu).clone()` (safe buffer) |
| **Network architecture** | Separate Actor + Critic (two networks) | `SharedActorCritic` with one shared CNN trunk |
| **Network input** | Image only | Image + speed + maneuver (navigation context) |
| **Optimizer state** | Not always in shared memory | Explicit `.share_memory_()` on all state tensors |
| **Reward** | Simple reward from CarlaEnv | Complex shape_reward: 8 components with clipping |
| **Supervisor** | Basic restart | Exponential backoff + rollback on rapid crash |
| **Rollback** | None | `rollback_global_network()` with candidate priority |
| **Resume** | None or partial | Full resume: model + optimizer + elapsed_time + stats |
| **Logging** | CSV or basic files | JSONL per-worker: episodes, updates, steps, timing, system |
| **System Monitor** | None | RunMonitor (CPU/RAM/GPU/CARLA) + WorkerMonitor per PID |
| **Timing** | None | `TimingAccumulator` per phase (forward, backward, sync, env) |
| **Entropy annealing** | Constant β | Linear annealing β_start → β_end over β_anneal_frac |
| **LR scheduling** | None or manual | Automatic linear decay to 0 |
| **NaN detection** | None | `model_has_nan_gradients()` + `has_nan_params()` with logging |
| **CARLA action repeat** | None | Configurable `action_repeat` (default 2 ticks) |
| **WandB** | None | Subprocess WandBLogger with isolated queue |
| **Episode saving** | None | Configurable frame saving to `episodes/` |
| **Graceful shutdown** | SIGKILL | SIGUSR1/SIGTERM → shutdown_event → save resume_state |
| **GPU assignment** | None | Configurable `workers_per_gpu`, `worker_gpu_start` |
| **World reload** | None | `world_reload_interval` to eliminate simulator errors |

---

## 9. SLURM Launch Example

### 9.1 Fresh Run — 2 Workers, 1 GPU

```bash
sbatch \
  --gpus=1 \
  new_hogwild_train.slurm \
  --workers 2 \
  --workers-per-gpu 2 \
  --scenario 14 \
  --no-wandb
```

**Explanation of each argument:**

| Argument | Value | Description |
|---|---|---|
| `--gpus=1` | 1 | SLURM: allocate 1 A100 GPU |
| `--workers 2` | 2 | Launch 2 A3CWorker processes |
| `--workers-per-gpu 2` | 2 | Both workers on one GPU (cuda:0) |
| `--scenario 14` | 14 | CARLA scenario (urban route) |
| `--no-wandb` | - | Disable Weights & Biases logging |

Generates directory: `runs/a3c_hogwild_2w_20240512_143022_12345/`

### 9.2 Advanced Run — 6 Workers, 6 GPUs

```bash
sbatch \
  --gpus=6 \
  --cpus-per-task=14 \
  --mem=60G \
  new_hogwild_train.slurm \
  --workers 6 \
  --workers-per-gpu 1 \
  --servers-per-gpu 1 \
  --scenario 14 \
  --steps 20000000 \
  --seed 42 \
  -- --lr 5e-5 --rollout-length 30 --beta-start 0.03
```

**After `--` come direct Python arguments:**

| Python argument | Value | Description |
|---|---|---|
| `--lr 5e-5` | 5e-5 | Lower learning rate (more stable, slower) |
| `--rollout-length 30` | 30 | Longer rollout (better reward propagation) |
| `--beta-start 0.03` | 0.03 | Higher initial entropy (more exploration) |

### 9.3 Resuming a Previous Run

```bash
sbatch \
  --gpus=1 \
  new_hogwild_train.slurm \
  --workers 2 \
  --resume /net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/runs/a3c_hogwild_2w_20240512_143022_12345
```

### 9.4 What a Live Training Log Looks Like

```
===================================================================
=== A3C CARLA training (new_hogwild_* pipeline) ===
===================================================================
  job id            : 12345
  host              : ares-gpu01
  visible GPUs      : 0
  outdir            : .../runs/a3c_hogwild_2w_20240512_143022_12345
  workers           : 2
  workers-per-gpu   : 2
-------------------------------------------------------------------
[SLURM] starting 2 CARLA server(s)...
[SLURM] CARLA supervisor PID=98765 -> .../carla_servers.log
[SLURM] waiting for ports: 2000 2100
[SLURM] all CARLA ports LISTEN after 4 check(s).
[RUN] outdir: .../runs/a3c_hogwild_2w_20240512_143022_12345
[W0] starting on cuda:0 port 2000
[W1] starting on cuda:0 port 2100
[W0] step 20 sync 6.2ms | forward 8.1ms | env_step 47.3ms
[W1] step 20 sync 6.5ms | forward 8.4ms | env_step 46.9ms
[2024-05-12 14:32:01.123] [TIMING] W0 backward:44.2ms(n=100) env_step:47.3ms(n=2000) forward:8.1ms(n=2000)
[BEST] W0 new best reward 127.40 at Ep42
[2024-05-12 14:45:00] [TIMING] W1 sync:6.8ms(n=1050) optim_update:3.2ms(n=1050)
[BENCHMARK] session: 1000000 steps in 3600s (277.78 steps/s); cumulative: 1000000 steps
```

---

## 10. Common Problems and Debugging

### 10.1 How to Read JSONL Logs

**Quick view of latest episodes:**
```bash
tail -20 logs/worker_0/episodes.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f\"Ep{r['global_episode']:4d} | R={r['total_reward']:7.1f} | steps={r['steps']:3d} | goal={'Y' if r['reached_goal'] else 'N'}\")
"
```

**Loss analysis:**
```bash
python3 -c "
import json
losses = []
with open('logs/worker_0/updates.jsonl') as f:
    for line in f:
        r = json.loads(line)
        losses.append((r['update'], r['pi_loss'], r['v_loss'], r['gradient_norm']))
# Last 10 updates:
for u, pi, v, g in losses[-10:]:
    print(f'update={u:6d} pi_loss={pi:8.4f} v_loss={v:8.4f} gradient_norm={g:6.2f}')
"
```

### 10.2 What to Watch in events.jsonl

```bash
# Count event types:
python3 -c "
import json, collections
counts = collections.Counter()
with open('logs/events.jsonl') as f:
    for line in f:
        r = json.loads(line)
        counts[r.get('event','?')] += 1
for event, count in counts.most_common():
    print(f'{event:30s}: {count}')
"
```

Typical output from a running training:
```
checkpoint_save               : 10
training_start                : 1
training_end                  : 0    ← still running
worker_restart                : 3    ← CARLA crashes, normal
camera_timeout                : 1
nan_gradient                  : 0    ← no NaN, good!
rollback                      : 0    ← no rollbacks, good!
```

### 10.3 How to Tell if the Model is Learning

**Positive signals:**
1. `total_reward` in `episodes.jsonl` trends upward over time (trend, not every episode)
2. `reached_goal` in `episodes.jsonl` starts appearing as non-zero
3. `pi_loss` is negative and decreasing (more negative = more confident policy)
4. `v_loss` decreasing (better value predictions)
5. `advantages_mean ≈ 0` and `advantages_std > 0` (advantage normalisation working)
6. `ent_mean` decreasing over time (policy becoming less random)
7. `gradient_norm < max_grad_norm=40` most of the time (no gradient explosion)

**Negative signals:**
1. `total_reward` flat for many episodes (no learning or stuck in local optimum)
2. `nan_gradient` in events.jsonl (numerical instability → reduce lr)
3. `rollback` in events.jsonl (frequent crashes → check CARLA logs)
4. `gradient_norm ≈ max_grad_norm` all the time (gradients always clipped → reduce lr or increase max_grad_norm)
5. `ent_mean` drops too fast at the start (too fast convergence → increase beta_start)
6. `reached_goal = false` always for 10k+ episodes (reward too small or agent too slow → check reward_goal_bonus)

**GPU monitoring:**
```bash
tail -5 logs/system.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    if 'gpus' in r:
        for g in r['gpus']:
            print(f'GPU{g[\"index\"]}: util={g[\"util_percent\"]}% mem={g[\"mem_used_gb\"]:.1f}/{g[\"mem_total_gb\"]:.1f}GB')
"
```

**Training speed monitoring:**
```bash
python3 -c "
import json
steps_times = []
with open('logs/worker_0/updates.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if 'global_t' in r and 'ts' in r:
            steps_times.append((r['global_t'], r['ts']))
if len(steps_times) > 10:
    from datetime import datetime
    t1, ts1 = steps_times[0]
    t2, ts2 = steps_times[-1]
    dt = (datetime.fromisoformat(ts2) - datetime.fromisoformat(ts1)).total_seconds()
    print(f'Speed: {(t2-t1)/dt:.1f} steps/s (total across all workers)')
"
```

### 10.4 Diagnosing Slow Training — Timing

When `env_step >> forward` (e.g. 100ms vs 8ms):
- Bottleneck: CARLA simulator (physics, rendering, network)
- Solution: check CPU/GPU load on CARLA, consider reducing `mp_density`

When `backward >> forward` (e.g. 100ms vs 8ms):
- Network is large relative to GPU capacity
- Solution: reduce `res` (camera resolution) or change architecture

When `sync >> forward` (e.g. 20ms vs 8ms):
- Too many CPU→GPU copies
- Solution: increase `sync_every_n_updates` (synchronise less frequently)

When `optim_update >> backward`:
- Contention on shared memory (lock or PCIe)
- Solution: check `--hogwild-lock-updates`, reduce workers per GPU

---

## 11. What We Changed in This Refactor

### 11.1 Variable and Function Renames

The main goal was readability — every name should be self-explanatory without consulting documentation:

| Old Name | New Name | Location |
|---|---|---|
| `cfg` | `config` | All files |
| `tlogger` | `training_logger` | `new_hogwild_a3c.py` |
| `worker_mon` | `worker_monitor` | `new_hogwild_a3c.py` |
| `run_mon` | `run_monitor` | `new_hogwild_train_a3c_carla.py` |
| `wandb_proc` | `wandb_logger_process` | `new_hogwild_train_a3c_carla.py` |
| `t0` | `phase_start_time` | `new_hogwild_a3c.py` |
| `t_max` / `T_MAX` | `rollout_length` | All files |
| `t_since_update` | `steps_since_last_update` | `new_hogwild_a3c.py` |
| `ep_reward` | `episode_total_reward` | `new_hogwild_a3c.py` |
| `ep_steps` | `episode_step_count` | `new_hogwild_a3c.py` |
| `outdir` | `run_output_dir` | All files (CLI flag `--outdir` kept) |
| `R` | `discounted_return` | `new_hogwild_a3c.py` |
| `lp` / `gp` | `local_param` / `global_param` | `new_hogwild_a3c.py` |
| `nan_layers` | `layers_with_nan_grad` | `new_hogwild_a3c.py` |
| `update_n` | `update_number` | `new_hogwild_a3c.py` |
| `lr_now` | `current_learning_rate` | `new_hogwild_a3c.py` |
| `anneal_steps` | `entropy_annealing_steps` | `new_hogwild_a3c.py` |
| `frac` | `progress_fraction` | `new_hogwild_a3c.py` |
| `rec` | `log_record` | Logger files |
| `has_nan_grads` | `model_has_nan_gradients` | `new_hogwild_a3c.py` |
| `make_shared_optimizer` | `create_shared_optimizer` | `new_hogwild_a3c.py` |
| `entropy_coef_for_step` | `compute_entropy_coefficient_for_step` | `new_hogwild_a3c.py` |
| `summarize_component_rollout` | `summarize_reward_components` | `new_hogwild_a3c.py` |
| `transfer_grads` | `transfer_local_gradients_to_global` | `new_hogwild_a3c.py` |
| `global_grad_norm` | `compute_total_gradient_norm` | `new_hogwild_a3c.py` |
| `_ep_*` attributes | `_episode_*` attributes | `new_hogwild_carla_wrapper.py` |

### 11.2 Legacy `model_arch` Removal

The previous implementation supported two model architectures: `'shared'` (single `SharedActorCritic`) and `'legacy'` (separate `DeepDiscreteActor` + `DeepCritic`). All `if model_arch == 'shared': ... else: ...` branches have been removed. Only `SharedActorCritic` remains.

**What was removed:**
- Imports `from nets.a2c import DiscreteActor as DeepDiscreteActor` and `Critic as DeepCritic`
- `GlobalNetwork` fields: `self.actor`, `self.critic`, `self.actor_optimizer`, `self.critic_optimizer`, `self.model_arch`
- `A3CWorker` fields: `self.actor`, `self.critic`
- CLI argument `--model-arch` from argparse
- All conditional branches for `'legacy'` in `_init_networks`, `sync_with_global`, `get_action`, `compute_and_apply_gradients`, `_checkpoint_state_unlocked`, `_has_nan_params`, `set_lr_for_step`, `load`, `rollback_global_network`

### 11.3 Impact on Checkpoint Structure

**Old checkpoint** could contain:
```python
{
    'model_arch': 'shared' | 'legacy',
    # if shared:
    'model': ..., 'optimizer': ...,
    # if legacy:
    'actor': ..., 'critic': ...,
    'actor_optimizer': ..., 'critic_optimizer': ...,
    ...
}
```

**New checkpoint** always contains:
```python
{
    # no 'model_arch' field
    'model': model.state_dict(),       # SharedActorCritic weights
    'optimizer': optimizer.state_dict(), # SharedRMSprop/SharedAdam state
    'global_step': ...,
    'global_episode': ...,
    'total_updates': ...,
    'last_checkpoint_boundary': ...,
    'best_reward': ...,
    'global_mean_reward': ...,
    'worker_mean_rewards': [...],
    'recent_rewards': [...],
    'recent_reward_count': ...,
    'recent_reward_index': ...,
}
```

**Backward compatibility**: old `actor`+`critic` checkpoints cannot be resumed with the new code. `GlobalNetwork.load()` raises `ValueError` with a clear message if the `model` key is missing. To use an old checkpoint, the weights would need to be manually migrated to `SharedActorCritic` format.

**JSONL field change**: `grad_norm_actor` + `grad_norm_critic` → single `gradient_norm` field in `updates.jsonl`.

---

## Summary

The `new_hogwild_*` implementation is a production-quality A3C system for CARLA with the following key properties:

```
Strengths:
├─ Hogwild! without locks → maximum throughput
├─ Shared memory: model + optimizer on CPU → no PCIe overhead
├─ SharedActorCritic: shared CNN trunk → fewer parameters, consistent representations
├─ Shaped reward with 8 components → dense learning signal
├─ Full supervisor with rollback → resilience to CARLA crashes
├─ JSONL logs with line buffering → data survives crashes
└─ Full resume with elapsed_time tracking → multi-session HPC training

Limitations / trade-offs:
├─ Global model on CPU → GPU→CPU transfer per update (~3ms)
├─ Hogwild! variance → may need more steps than A2C
└─ CARLA is single-threaded → env_step ≈ 50ms (main bottleneck)
```

Understanding the data flow: **Global CPU Model** ← synchronisation ← **Worker GPU** → gradients → **Global CPU Optimizer** → updated model → next iteration, is the key to debugging, tuning and extending this implementation.

---

*Generated by analysis of `new_hogwild_*` source code in `/net/tscratch/people/plgbartoszkawa/AV/A_to_B_GPU_34/`.*
