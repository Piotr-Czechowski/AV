# A3C implementation comparison

This file compares three A3C implementations side by side:

1. `async-rl-backup-12-4-2026-before-spliting-model-and-changing-loss/`
   - Main CARLA path: `train_a3c_carla.py`
   - Core algorithm: `a3c.py`, `run_a3c.py`, `async_rl.py`, `rmsprop_async.py`
   - Operations: `training_logger.py`, `system_monitor.py`, `prepare_output_dir.py`
2. `AV/A_to_B_GPU_34/a3c_improved_1.py`
   - Main monolithic PyTorch implementation
   - Model classes: `AV/A_to_B_GPU_34/nets/a2c.py`
3. `AV/A_to_B_GPU_34/new_hogwild_*`
   - Merged implementation: `new_hogwild_train_a3c_carla.py`, `new_hogwild_a3c.py`,
     `new_hogwild_carla_wrapper.py`, `new_hogwild_run_a3c.py`,
     `new_hogwild_training_logger.py`, `new_hogwild_system_monitor.py`,
     `new_hogwild_prepare_output_dir.py`, `new_hogwild_train.slurm`

Evidence snippets below are short excerpts from the current source files. They
are included so each claim is tied to code, not only a textual judgment.

## Short Verdict

The current `new_hogwild_*` implementation is the best base for future CARLA
A3C work. It keeps the useful PyTorch Hogwild idea from `a3c_improved_1.py`,
but fixes the largest algorithmic and operational weaknesses by adding a shared
actor-critic model, shared RMSprop, LR decay, correct entropy handling, active
gradient clipping, NaN guards, structured JSONL logs, checkpoint/resume,
rollback, signal handling, monitoring, and a SLURM/CARLA launch pipeline.

Two implementation gaps were found and are documented here, but source code is
left unchanged so `comparison.md` remains the only modified file:

1. `new_hogwild_prepare_output_dir.py` does not restore the backup
   implementation's run provenance snapshot: `git-status*.txt`,
   `git-log*.txt`, and `git-diff*.txt`.
2. `new_hogwild_run_a3c.py` logs `worker_give_up` after
   `max_restarts_per_worker`, but the dead worker remains in the monitored map,
   so the supervisor can repeatedly process the same dead worker.

`async-rl` is still the best correctness reference for classic A3C mechanics:
RMSpropAsync, n-step returns, per-step entropy, detached policy advantage,
gradient clipping, NaN guard, and optional LSTM are all explicit. Its main
drawback is the old Chainer stack and manual shared-memory plumbing.

`a3c_improved_1.py` is a useful first PyTorch multi-GPU prototype, but it is
not the best training implementation as-is. It has hardcoded paths, monolithic
structure, separate Actor/Critic CNNs, BatchNorm/Dropout in a batch-size-1
Hogwild setup, shared gradient buffers, fixed LR, no global step budget stop,
inactive gradient clipping, and incorrect entropy/value coefficient use.

## Summary Matrix

| Category | async-rl backup | a3c_improved_1.py | new_hogwild_* | Best approach |
|---|---|---|---|---|
| Runtime/framework | Chainer/CuPy plus manual multiprocessing | PyTorch multiprocessing | PyTorch multiprocessing split into focused files | `new_hogwild_*` |
| Model topology | Shared actor-critic, optional LSTM | Separate Actor and Critic CNNs | Shared actor-critic trunk | `new_hogwild_*`; `async-rl` if LSTM is needed |
| Batch-size-1 safety | No BatchNorm/Dropout | BatchNorm and Dropout in Actor/Critic | No BatchNorm/Dropout | `new_hogwild_*` / `async-rl` |
| CARLA boundary | Wrapper exists but still packs speed/maneuver as image channels | CARLA logic inside worker loop | Dedicated wrapper with validation/reward/action repeat | `new_hogwild_*` |
| Action repeat | Applies action twice and ticks twice | Applies action once, ticks twice | Configurable `action_repeat`, default 2 | `new_hogwild_*` |
| Episode cap | Low-level CARLA cap; decision count implicit | Low-level cap; learner decisions differ | Explicit decision cap | `new_hogwild_*` |
| Rollout/returns | Classic n-step return | n-step return | n-step return with rollout config | Tie, with `new_hogwild_*` most configurable |
| Policy advantage | Detached by `float(advantage.data)` | Uses critic graph in actor loss | Explicit `values.detach()` plus optional normalization | `new_hogwild_*` |
| Value loss | Squared error / 2, coefficient path | Smooth L1, coefficient constant unused | Smooth L1 with active coefficient | `new_hogwild_*` |
| Entropy | Correct per-step beta entropy | Latest distribution only, coefficient unused | Per-transition entropy with schedule | `new_hogwild_*` |
| Optimizer | Shared RMSpropAsync | SharedAdam only | SharedRMSprop default, SharedAdam optional | `new_hogwild_*` |
| LR schedule | Linear decay | Fixed LR | Linear decay | `new_hogwild_*` / `async-rl` |
| Gradient transfer | Chainer grad copy to shared model | Shared `.grad` buffers | Process-local cloned grads assigned before step | `new_hogwild_*` |
| Gradient clipping | Active Chainer hook | Defined but commented out | Active PyTorch clipping | `new_hogwild_*` / `async-rl` |
| NaN protection | Grad and checkpoint guards | None visible | Grad and checkpoint guards | `new_hogwild_*` |
| Counters/stop | Locked global step, stops at budget | Global steps updated per episode, no step budget stop | Locked decision step, supervisor stop | `new_hogwild_*` |
| Checkpoint/resume | Strong HDF5 + opt + sidecar | Hardcoded `.pth`, shallow resume | Unified `.pth` + counters + sidecar | `new_hogwild_*` |
| Provenance | Args + git status/log/diff | Weak/hardcoded | Args only; git status/log/diff missing | `async-rl` today; `new_hogwild_*` after porting this feature |
| Restart/rollback | Restart and rollback, but give-up worker remains monitored | Restart indefinitely | Restart cap and rollback, but give-up worker remains monitored | `new_hogwild_*` overall, with give-up tracking fix recommended |
| Logging/monitoring | Structured JSONL and system monitor | CSV/text/W&B/profiler | Structured JSONL, W&B process, GPU/system monitor | `new_hogwild_*` |
| HPC operations | Assumes CARLA servers already exist | Assumes CARLA servers already exist | SLURM starts CARLA, waits ports, handles cleanup | `new_hogwild_*` |

## Detailed Evidence By Category

### 1. Runtime and file organization

**Best:** `new_hogwild_*`.

Why it matters: split files make the merged implementation easier to inspect,
test, and change without touching unrelated CARLA or training logic.

`async-rl` has useful separation between algorithm, runner, logger, monitor, and
CARLA adapter, but it is built on Chainer. `a3c_improved_1.py` is PyTorch but
puts configuration, model supervision, worker loop, logging, checkpointing, and
main in one large script. `new_hogwild_*` keeps PyTorch while splitting the
implementation by responsibility.

`async-rl` proof:

```python
# async-rl-backup-.../train_a3c_carla.py
def run_a3c_carla(processes, make_env, model_opt, phi, t_max=1, gamma=GAMMA,
                  beta=1e-2, steps=8 * 10 ** 7, args={}, reward_scale=0.0,
                  restart_on_crash=False, save_frequency=0, resume_dir=None,
                  outdir_holder=None):
    ...
    shared_params = async_rl.share_params_as_shared_arrays(model)
    shared_states = async_rl.share_states_as_shared_arrays(opt)
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
MODEL_LOAD_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/carla_to_chainer_013_13_12w_10u_10s_profiled.pth'
MODEL_SAVE_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/carla_to_chainer_013_13_12w_10u_10s_profiled'
EXP_ID = "carla_to_chainer_013_13_12w_10u_10s_profiled_continued_3.pth"

NUM_WORKERS = 12
...
class GlobalNetwork:
...
class A3CWorker(mp.Process):
...
def handle_workers(global_network, log_queue=None, shutdown_event=None):
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_train_a3c_carla.py
from new_hogwild_prepare_output_dir import prepare_output_dir
from new_hogwild_training_logger import TrainingLogger
from new_hogwild_system_monitor import RunMonitor
from new_hogwild_a3c import GlobalNetwork
from new_hogwild_run_a3c import run_with_restart, find_latest_checkpoint
```

### 2. Model topology and perception reuse

**Best:** `new_hogwild_*` for current CARLA training; `async-rl` if temporal
memory is required.

Why it matters: one shared trunk avoids duplicated CNN work and keeps policy and
value learning aligned on the same visual features.

The backup and merged versions use one actor-critic representation, so image
features are computed once and shared by policy/value heads. The improved
version builds separate Actor and Critic networks and therefore duplicates CNN
work and optimizer state.

`async-rl` proof:

```python
# async-rl-backup-.../train_a3c_carla.py
class Base(chainer.Chain, a3c.A3CModel):
    ...
    self.fusion_fc1 = L.Linear(8192 + 32 + 32, 512)
    self.fusion_fc2 = L.Linear(512, 256)

    self.pi_fc = L.Linear(256, n_actions)
    self.v_fc = L.Linear(256, 1)
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
self.actor = DeepDiscreteActor(state_shape, action_shape, 'cpu').to(self.device)
self.critic = DeepCritic(state_shape, critic_shape, 'cpu').to(self.device)

self.actor.share_memory()
self.critic.share_memory()
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
class SharedActorCritic(nn.Module):
    """Batch-size-1 friendly actor-critic with one shared visual trunk."""
    ...
    self.trunk = nn.Sequential(
        nn.Linear(256 * 4 * 4 + 32 + 32, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
    )
    self.policy = nn.Linear(256, action_shape)
    self.value = nn.Linear(256, critic_shape)
```

### 3. BatchNorm, Dropout, and batch-size-1 safety

**Best:** `new_hogwild_*` and `async-rl`.

Why it matters: batch-size-1 online RL makes BatchNorm buffers and Dropout noise
hard to control, especially with separate worker-local models.

CARLA A3C workers effectively run small online batches. BatchNorm buffers in
local worker models are not gradients and are not copied back like parameters;
they can become stale or inconsistent. Dropout adds more stochasticity to an
already stochastic policy-gradient system. The merged implementation keeps the
speed/maneuver fusion idea but removes BatchNorm and Dropout.

`async-rl` proof:

```python
# async-rl-backup-.../train_a3c_carla.py
self.stem = L.Convolution2D(3, 32, 3, pad=1)
self.down1 = L.Convolution2D(32, 64, 3, stride=2, pad=1)
self.res1 = ResBlock(64)
...
h = F.relu(self.fusion_fc1(fused))
h = F.relu(self.fusion_fc2(h))
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/nets/a2c.py
self.cnn = nn.Sequential(
    nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    ...
    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(),
)
...
self.fc = nn.Sequential(
    nn.Linear(512 * 4 * 4 + 32 + 32, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
self.cnn = nn.Sequential(
    nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((4, 4)),
)
```

### 4. Observation handling and CARLA boundary

**Best:** `new_hogwild_*`.

Why it matters: keeping reset, step, normalization, and reward logic in a wrapper
keeps the worker focused on A3C updates.

The backup has a wrapper, but it packs speed and maneuver into extra spatial
channels. The improved implementation handles much of CARLA directly inside
the worker loop. The merged wrapper keeps algorithm code cleaner and validates
observation shape/range before the worker sees data.

`async-rl` proof:

```python
# async-rl-backup-.../train_a3c_carla.py
def phi(obs):
    img = obs[:3]
    speed_val = float(obs[3, 0, 0])
    maneuver_val = float(obs[4, 0, 0])
    ...
    speed_channel = np.full((1, IMG_SIZE, IMG_SIZE), speed_val, dtype=np.float32)
    man_channel = np.full((1, IMG_SIZE, IMG_SIZE), maneuver_val, dtype=np.float32)
    return np.concatenate([img_out, speed_channel, man_channel], axis=0)
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
env = CarlaEnv(
    scenario=SCENARIO, spawn_point=False, terminal_point=False,
    mp_density=25, port=self.port,
    action_space=ACTION_TYPE, camera=CAMERA_TYPE,
    resX=250, resY=250, manual_control=False
)
...
state = state / 255.0
speed = speed / 100.0
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_carla_wrapper.py
def _state_to_chw_float(self, state):
    ...
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError('expected observation [3,H,W], got {}'.format(arr.shape))
    if not np.isfinite(arr).all():
        raise ValueError('observation contains NaN or inf')
    if arr.size and arr.max() > 1.0:
        arr = arr / 255.0
    if arr.min() < -1e-4 or arr.max() > 1.0 + 1e-4:
        raise ValueError('observation outside [0,1]: min={} max={}'.format(...))
    return np.ascontiguousarray(arr)
```

### 5. Speed and maneuver input

**Best:** `new_hogwild_*`.

Why it matters: scalar speed and clamped one-hot maneuver branches represent
driving context directly without fake spatial channels.

The backup's learned maneuver embedding is good, but speed and maneuver are
encoded as fake spatial channels. The improved and merged versions pass scalar
speed and one-hot maneuver to separate branches. The merged version also clamps
maneuver ids to the valid range.

`async-rl` proof:

```python
# async-rl-backup-.../train_a3c_carla.py
speed = state[:, 3:4, 0:1, 0:1]
speed = F.reshape(speed, (state.shape[0], 1))
maneuver_float = state[:, 4, 0, 0]
maneuver = maneuver_float.data.astype(np.int32)
...
man = self.man_embed(maneuver)
man = F.relu(self.man_fc(man))
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/nets/a2c.py
self.speed_fc = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
self.manouver_fc = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
...
speed_features = self.speed_fc(speed)
manouver = F.one_hot(manouver, num_classes=3).float().to(self.device)
manouver_features = self.manouver_fc(manouver)
combined = torch.cat([cnn_features, speed_features, manouver_features], dim=1)
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
self.speed_fc = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True))
self.maneuver_fc = nn.Sequential(nn.Linear(num_maneuvers, 32), nn.ReLU(inplace=True))
...
maneuver = maneuver.to(self.device, dtype=torch.long).view(-1)
maneuver = maneuver.clamp(0, self.num_maneuvers - 1)
maneuver = F.one_hot(maneuver, num_classes=self.num_maneuvers).float()
```

### 6. Action repeat and episode length

**Best:** `new_hogwild_*`.

Why it matters: action repeat holds the same control for two CARLA ticks, making
vehicle response visible while keeping episodes near 100 learner decisions.

The backup applies the same action twice. The improved implementation applies
the action once but ticks CARLA twice, which changes control persistence. The
merged version makes this explicit with `action_repeat` and an explicit learner
decision cap.

`async-rl` proof:

```python
# async-rl-backup-.../train_a3c_carla.py
self.env.step_apply_action(action)
...
self.env.world.tick()
self.env.step_apply_action(action)
self.env.world.tick()
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
env.step_apply_action(action)
...
env.world.tick()
# env.step_apply_action(action)
env.world.tick()
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_carla_wrapper.py
self._action_repeat = max(1, int(action_repeat))
self._episode_max_decisions = int(episode_max_decisions)
...
for _ in range(self._action_repeat):
    self.env.step_apply_action(int(action))
    self.env.world.tick()
...
if self._episode_max_decisions > 0 and self.step_count >= self._episode_max_decisions:
    done = True
```

### 7. Rollout length and n-step returns

**Best:** tie on the core n-step idea; `new_hogwild_*` is best operationally
because rollout length is a named CLI/config setting and logged.

Why it matters: rollout length sets the bias/variance and synchronization cost
of each update, so it should be explicit and reproducible.

All three implementations compute reverse n-step bootstrapped returns. The
differences are configurability and loss handling after returns are computed.

`async-rl` proof:

```python
# async-rl-backup-.../a3c.py
if (is_state_terminal and self.t_start < self.t) \
        or self.t - self.t_start == self.t_max:
    ...
    for i in reversed(range(self.t_start, self.t)):
        R *= self.gamma
        R += self.past_rewards[i]
        v = self.past_values[i]
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
for reward in reversed(self.rewards):
    R = reward + self.gamma * R
    returns.insert(0, R)
...
if not TESTING and (step_count >= T_MAX or done):
    self.compute_and_apply_gradients(...)
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
for reward in reversed(self.rewards):
    r = reward / scale if scale and scale != 0.0 else reward
    rewards_scaled.insert(0, r)
    discounted_return = r + self.config.gamma * discounted_return
    returns.insert(0, discounted_return)
...
if not self.config.testing and \
        (steps_since_last_update >= self.config.rollout_length or done):
```

### 8. Policy advantage

**Best:** `new_hogwild_*`.

Why it matters: detaching the advantage prevents actor loss from accidentally
training the critic through the policy-gradient path.

The policy-gradient advantage should not backpropagate policy loss into the
critic value estimate. The backup does this by converting the Chainer variable
to a float. The improved implementation uses `G_t - V_s` directly in the actor
loss, which keeps critic graph involvement and then relies on zeroing/separate
backward behavior. The merged implementation is explicit and also supports
advantage normalization.

`async-rl` proof:

```python
# async-rl-backup-.../a3c.py
advantage = R - v
log_prob = self.past_action_log_prob[i]
entropy = self.past_action_entropy[i]
pi_loss -= log_prob * float(advantage.data)
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
for G_t, V_s, log_prob in zip(returns, values, log_probs):
    td_err = G_t - V_s
    actor_losses.append(-log_prob * td_err)
    critic_losses.append(F.smooth_l1_loss(V_s, G_t))
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
advantages_tensor = returns_tensor - values_tensor.detach()
policy_advantages = advantages_tensor
if getattr(self.config, 'normalize_advantages', True) and policy_advantages.numel() > 1:
    advantages_mean = policy_advantages.mean()
    advantages_std = policy_advantages.std(unbiased=False)
    policy_advantages = (policy_advantages - advantages_mean) / advantages_std.clamp_min(1e-8)
policy_loss = -(log_probs_tensor * policy_advantages.view(-1)).mean()
```

### 9. Value loss and coefficient use

**Best:** `new_hogwild_*`.

Why it matters: an active value coefficient lets value learning be tuned instead
of silently ignoring a declared hyperparameter.

The backup uses squared error divided by two and applies a coefficient. The
improved implementation defines `VALUE_LOSS_COEF` but does not use it. The
merged implementation uses Smooth L1 and multiplies by `value_loss_coef`.

`async-rl` proof:

```python
# async-rl-backup-.../a3c.py
v_loss += (v - R) ** 2 / 2
...
if self.v_loss_coef != 1.0:
    v_loss *= self.v_loss_coef
total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
VALUE_LOSS_COEF = 1.0
...
critic_losses.append(F.smooth_l1_loss(V_s, G_t))
...
critic_loss = torch.stack(critic_losses).mean()
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
value_loss = self.config.value_loss_coef * F.smooth_l1_loss(
    values_tensor, returns_tensor)
total_loss = policy_loss + value_loss \
    - entropy_coef * entropy_mean
```

### 10. Entropy regularization

**Best:** `new_hogwild_*`.

Why it matters: entropy must be applied per sampled transition with a known
coefficient, otherwise exploration pressure is wrong.

Entropy should be computed for every sampled transition and scaled by a known
coefficient. The backup does this with constant beta. The improved version
defines `ENTROPY_COEF`, but subtracts the latest distribution's entropy with no
coefficient. The merged version stores per-transition entropy and supports
constant or scheduled beta.

`async-rl` proof:

```python
# async-rl-backup-.../a3c.py
self.past_action_entropy[self.t] = pout.entropy
...
entropy = self.past_action_entropy[i]
pi_loss -= self.beta * entropy
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
ENTROPY_COEF = 0.01
...
if USE_ENTROPY:
    actor_loss = actor_loss - self.action_distribution.entropy().mean()
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
entropy = action_distribution.entropy()
self.trajectory.append(Transition(
    value_s=value, log_prob_a=log_prob, entropy=entropy,
    action=int(action_np)))
...
entropy_coef = compute_entropy_coefficient_for_step(self.config, global_t)
total_loss = policy_loss + value_loss - entropy_coef * entropy_mean
```

### 11. Optimizer and learning-rate schedule

**Best:** `new_hogwild_*`.

Why it matters: shared RMSprop matches classic A3C behavior, while an explicit
LR schedule makes long training runs more stable.

Classic A3C used shared RMSprop. The backup implements this in Chainer and
decays LR linearly. The improved implementation uses shared Adam only and keeps
LR fixed. The merged version defaults to shared RMSprop, keeps shared Adam
available, and decays LR by global step.

`async-rl` proof:

```python
# async-rl-backup-.../rmsprop_async.py
class RMSpropAsyncRule(optimizer.UpdateRule):
    def update_core_cpu(self, param):
        grad = param.grad
        ms = self.state['ms']
        hp = self.hyperparam
        ms *= hp.alpha
        ms += (1 - hp.alpha) * grad * grad
        param.data -= hp.lr * grad / numpy.sqrt(ms + hp.eps)
```

```python
# async-rl-backup-.../run_a3c.py
agent.optimizer.lr = (
    args['steps'] - global_t - 1) / args['steps'] * args['lr']
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
self.actor_optimizer = SharedAdam(self.actor.parameters(), lr=LR, weight_decay=1e-2)
self.critic_optimizer = SharedAdam(self.critic.parameters(), lr=LR, weight_decay=1e-2)
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
def create_shared_optimizer(params, config):
    optimizer = getattr(config, 'optimizer', 'shared-rmsprop')
    if optimizer == 'shared-rmsprop':
        return SharedRMSprop(params, lr=config.lr, alpha=..., epsilon=...)
    if optimizer == 'shared-adam':
        return SharedAdam(params, lr=config.lr, betas=(...), epsilon=...)
```

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
def set_lr_for_step(self, global_t):
    progress_fraction = max(
        0.0, (self.config.steps - global_t - 1) / self.config.steps)
    lr = progress_fraction * self.config.lr
    self.optimizer.set_lr(lr)
    return lr
```

### 12. Shared memory and gradient transfer

**Best:** `new_hogwild_*`.

Why it matters: process-local gradient tensors avoid workers overwriting shared
`.grad` buffers between gradient copy and optimizer step.

The backup manually maps Chainer parameters and optimizer state to shared
`RawArray` buffers. The improved PyTorch version shares `.grad` buffers, which
creates a race: another worker can overwrite `.grad` before the optimizer step
finishes. The merged version shares parameters and optimizer state, but keeps
grad tensors process-local and assigns a cloned CPU grad immediately before
`optimizer.step()`.

`async-rl` proof:

```python
# async-rl-backup-.../async_rl.py
def share_params_as_shared_arrays(link):
    shared_arrays = extract_params_as_shared_arrays(link)
    set_shared_params(link, shared_arrays)
    return shared_arrays

def share_states_as_shared_arrays(optimizer):
    shared_arrays = extract_states_as_shared_arrays(optimizer)
    set_shared_states(optimizer, shared_arrays)
    return shared_arrays
```

```python
# async-rl-backup-.../a3c.py
self.shared_model.zerograds()
copy_param.copy_grad(target_link=self.shared_model, source_link=self.model)
self.optimizer.update()
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
for p in self.actor.parameters():
    if p.grad is None:
        p.grad = torch.zeros_like(p.data)
    ...
    p.grad.share_memory_()
...
def transfer_grads_to_shared(local_model, shared_model):
    ...
    sp.grad.copy_(lp.grad.detach().to("cpu"))
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
def transfer_local_gradients_to_global(local_model, global_model, global_device):
    """Copy local gradients into the shared model for this process.

    Parameters and optimizer state are shared. Gradient buffers are not shared:
    each worker assigns its own cloned .grad tensors before optimizer.step().
    """
    ...
    global_param.grad = local_param.grad.detach().to(
        global_device, non_blocking=False).clone()
```

### 13. Gradient clipping and NaN update protection

**Best:** `new_hogwild_*`; `async-rl` is also strong.

Why it matters: clipping and NaN skips prevent one bad rollout from corrupting
shared parameters in a long multi-worker CARLA run.

Both backup and merged implementations actively clip or guard gradients. The
improved implementation defines clipping constants but comments out clipping
and has no comparable NaN guard before shared update.

`async-rl` proof:

```python
# async-rl-backup-.../train_a3c_carla.py
opt = rmsprop_async.RMSpropAsync(lr=args.lr, eps=0.1, alpha=0.99)
opt.setup(model)
opt.add_hook(chainer.optimizer.GradientClipping(40))
```

```python
# async-rl-backup-.../a3c.py
if nan_layers:
    self._nan_count += 1
    ...
    self.sync_parameters()
    self.model.unchain_backward()
else:
    self.shared_model.zerograds()
    copy_param.copy_grad(...)
    self.optimizer.update()
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
MAX_GRAD_NORM = 40.0
...
# gradient clipping (GPU)
# torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
# torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
if self.config.max_grad_norm > 0:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                   self.config.max_grad_norm)

layers_with_nan_grad = has_nan_grads(self.model)
if layers_with_nan_grad:
    training_logger.log_nan(...)
    self.sync_with_global()
    self._clear_rollout_buffers()
    return None
```

### 14. Global counters and stopping behavior

**Best:** `new_hogwild_*`.

Why it matters: step-based counters make LR decay, checkpoints, W&B logging, and
job stopping line up with actual learner decisions.

The backup increments a locked step counter each loop and stops at `steps`. The
improved implementation updates `global_steps` only after an episode, and the
worker loop has no global step budget termination. The merged version increments
global step once per learner decision and the supervisor uses it for stopping.

`async-rl` proof:

```python
# async-rl-backup-.../run_a3c.py
with counter.get_lock():
    counter.value += 1
global_t = counter.value

if global_t > args['steps']:
    break
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
def update_stats(self, worker_id, mean_reward, episode_reward, episode_length):
    is_new_best = False
    with self.stats_lock:
        self.global_steps.value += episode_length
        ...

def increment_updates(self):
    self.total_updates.value += 1
    return self.total_updates.value
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
def increment_global_step(self, n=1):
    with self.global_step.get_lock():
        self.global_step.value += n
        return self.global_step.value
...
global_t = self.global_network.increment_global_step(1)
if self.config.steps > 0 and global_t > self.config.steps:
    self.shutdown_event.set()
    break
```

```python
# AV/A_to_B_GPU_34/new_hogwild_run_a3c.py
if config.steps > 0 and \
        global_network.global_step.value >= config.steps:
    shutdown_event.set()
    break
```

### 15. Checkpoint, resume, and provenance

**Best:** `async-rl` for provenance today; `new_hogwild_*` would become best if
it ports the same git snapshot behavior.

Why it matters: checkpoints are only fully reproducible when they include model
state, optimizer state, counters, resume metadata, and code provenance.

The backup saves model, optimizer, checkpoint step, resume sidecar, and git
state. The improved version saves actor/critic/optimizers to one hardcoded
path and can load it, but run-state resume is shallow. The merged version saves
model, optimizer, counters, reward buffers, stats, resume sidecar, and W&B id,
but it does not currently save git status/log/diff files.

`async-rl` proof:

```python
# async-rl-backup-.../prepare_output_dir.py
with open(os.path.join(outdir, 'args{}.txt'.format(suffix)), 'w') as f:
    args_dict = args if isinstance(args, dict) else vars(args)
    f.write(json.dumps(args_dict))

with open(os.path.join(outdir, 'git-status{}.txt'.format(suffix)), 'w') as f:
    f.write(subprocess.getoutput('git status'))
with open(os.path.join(outdir, 'git-log{}.txt'.format(suffix)), 'w') as f:
    f.write(subprocess.getoutput('git log'))
with open(os.path.join(outdir, 'git-diff{}.txt'.format(suffix)), 'w') as f:
    f.write(subprocess.getoutput('git diff'))
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
def save(self, path):
    state = {
        "actor": self.actor.state_dict(),
        "actor_optimizer": self.actor_optimizer.state_dict(),
        "critic": self.critic.state_dict(),
        "critic_optimizer": self.critic_optimizer.state_dict(),
        "global_steps": self.global_steps.value,
    }
    torch.save(state, path + ".pth")
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
def _checkpoint_state_unlocked(self, global_t=None):
    state = {
        'global_step': global_t if global_t is not None else self.global_step.value,
        'global_episode': self.global_episode.value,
        'total_updates': self.total_updates.value,
        'best_reward': self.best_reward.value,
        'worker_mean_rewards': list(self.worker_mean_rewards[:]),
        'model': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
    }
    return state
```

```python
# AV/A_to_B_GPU_34/new_hogwild_prepare_output_dir.py
args_dict = args if isinstance(args, dict) else vars(args)
with open(os.path.join(run_output_dir,
                       'args{}.txt'.format(suffix)), 'w') as f:
    json.dump(args_dict, f, indent=2, default=str)

return run_output_dir
```

### 16. Restart supervision and rollback

**Best:** `new_hogwild_*` overall, but it still needs one supervisor fix.

Why it matters: CARLA workers crash often enough that restart caps and rollback
must recover cleanly without repeatedly handling the same dead process.

The backup CARLA supervisor restarts workers and rolls back after rapid crashes,
but after a worker exceeds max restarts it only `continue`s, leaving the dead
worker in the monitored map. The improved implementation restarts workers but
does not enforce its `MAX_RETRIES` constant. The merged implementation includes
rollback, max restart enforcement, and exponential backoff, but it has the same
give-up monitoring issue as the backup: after logging `worker_give_up`, it
continues without removing or disabling the dead worker.

`async-rl` proof:

```python
# async-rl-backup-.../train_a3c_carla.py
if restart_counts[i] >= max_restarts_per_worker:
    print('[RESTART] worker {} exceeded max restarts'.format(i))
    continue

if rapid_crash_count[i] >= rapid_crash_threshold:
    ok = run_a3c._rollback_shared_params(
        shared_params, model_opt, outdir,
        shared_states=shared_states,
        worker_idx=i)
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
MAX_RETRIES = 3
...
if not worker.is_alive():
    restart_counts[worker_id] += 1
    ...
    worker = A3CWorker(...)
    worker.start()
    workers[worker_id] = worker
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_run_a3c.py
if restart_counts[i] >= config.max_restarts_per_worker:
    print('[RESTART] W{} exceeded max restarts ({}), '
          'giving up'.format(
              i, config.max_restarts_per_worker),
          flush=True)
    _append_event(run_output_dir, event='worker_give_up',
                  worker=i,
                  restart_count=restart_counts[i])
    continue
```

```python
# AV/A_to_B_GPU_34/new_hogwild_run_a3c.py
if rapid_crash_count[i] >= config.rapid_crash_threshold:
    ok = rollback_global_network(global_network, run_output_dir, worker_idx=i)
    _append_event(run_output_dir, event='rollback', worker=i,
                  rapid_crash_count=rapid_crash_count[i],
                  global_t=current_step, success=ok)
```

### 17. Logging, update diagnostics, and system monitoring

**Best:** `new_hogwild_*`.

Why it matters: structured logs and monitors make failed or unstable training
runs diagnosable after the job has already ended.

The backup has strong JSONL logs and system monitoring. The improved version
uses text logs, CSV, W&B, and profiler dumps, but lacks structured update/system
logs. The merged version keeps JSONL, adds PyTorch tensor serialization support,
optional full update arrays, W&B subprocess logging, and GPU monitoring via
NVML when available.

`async-rl` proof:

```python
# async-rl-backup-.../training_logger.py
class TrainingLogger:
    def __init__(self, outdir, worker_id, log_steps=True,
                 log_update_arrays=False):
        self.log_dir = os.path.join(outdir, 'logs',
                                    'worker_{}'.format(worker_id))
        self._episodes_f = self._open('episodes.jsonl')
        self._updates_f = self._open('updates.jsonl')
        self._timing_f = self._open('timing.jsonl')
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
LOG_FILE = 'log.csv'
...
with open(LOG_FILE, 'a', newline='') as f:
    csv.writer(f).writerow([
        self.worker_id, episode_num, ep_reward, self.mean_reward,
        global_mean, episode_length, self.total_steps,
        self.global_network.total_updates.value, global_step,
        distance_from_target
    ])
...
profiler.dump_stats(f'profile_worker_{self.worker_id}.out')
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_training_logger.py
try:
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
except ImportError:
    pass
...
self._episodes_f = self._open('episodes.jsonl')
self._updates_f = self._open('updates.jsonl')
self._timing_f = self._open('timing.jsonl')
self._steps_f = self._open('steps.jsonl') if log_steps else None
```

```python
# AV/A_to_B_GPU_34/new_hogwild_system_monitor.py
if self.track_gpu and _NVML_AVAILABLE:
    log_record['gpus'] = _sample_gpus()
```

### 18. Reward handling

**Best:** `new_hogwild_*`.

Why it matters: legacy reward preserves comparability, while optional shaped
components show exactly why reward changes during experiments.

The backup and improved versions use the environment reward, with optional
reward scaling in the backup agent. The merged wrapper keeps legacy reward as
the default for comparability, but adds optional shaped reward with named
components, making reward debugging much easier.

`async-rl` proof:

```python
# async-rl-backup-.../a3c.py
if self.reward_scale:
    reward = reward / self.reward_scale
...
self.past_rewards[self.t - 1] = reward
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
next_state, reward, done, _, next_speed, distance_from_target = env.step(
    save_image=save_images, episode=current_episode, step=episode_step
)
...
self.rewards.append(reward)
ep_reward += reward
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_train_a3c_carla.py
DEFAULT_REWARD_MODE = 'legacy'
...
p.add_argument('--reward-mode', type=str,
               default=DEFAULT_REWARD_MODE,
               choices=['shaped', 'legacy'])
```

```python
# AV/A_to_B_GPU_34/new_hogwild_carla_wrapper.py
if self._reward_mode == 'legacy':
    components = {'legacy': float(legacy_reward)}
    return float(legacy_reward), components
...
components = {
    'progress': self._reward_progress_coef * progress,
    'target_speed': self._reward_target_speed_coef * speed_score,
    'route_penalty': -self._reward_route_penalty_coef * ...,
    'time_penalty': -self._reward_time_penalty,
    'goal_bonus': self._reward_goal_bonus if reached_goal else 0.0,
    ...
}
```

### 19. W&B, signals, and cluster operation

**Best:** `new_hogwild_*`.

Why it matters: SLURM signals, W&B resume ids, port checks, and cleanup are what
make a CARLA training run survive real HPC scheduling.

The backup uses W&B in the main run path and can reuse a run id on resume. The
improved version uses a separate W&B logger process, but hardcodes the run id
and name. The merged version keeps the subprocess idea, stores/reuses the W&B
run id, handles `SIGTERM`/`SIGUSR1`/`SIGINT`, and includes a SLURM script that
starts CARLA servers, waits for ports, launches training, tails logs, and cleans
up.

`async-rl` proof:

```python
# async-rl-backup-.../train_a3c_carla.py
wandb.init(
    project=args.get('wandb_project', 'a3c-carla'),
    config=args,
    id=wandb_id,
    name='a3c-carla-{}w-{}'.format(processes, wandb_id),
    resume='allow',
    reinit=True,
)
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
wandb.init(
    project="A_to_B",
    name="synchr_test3_11",
    resume="allow",
    id=EXP_ID,
    config={
        "learning_rate": LR,
        "num_workers": NUM_WORKERS,
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_train_a3c_carla.py
for signame in ('SIGTERM', 'SIGUSR1', 'SIGINT'):
    sig = getattr(signal, signame, None)
    if sig is not None:
        signal.signal(sig, _handle_signal)
...
if args.resume and os.path.exists(wandb_id_file):
    with open(wandb_id_file) as f:
        wandb_id = f.read().strip()
else:
    wandb_id = uuid.uuid4().hex[:8]
```

```bash
# AV/A_to_B_GPU_34/new_hogwild_train.slurm
#SBATCH --signal=SIGUSR1@90
...
nohup python -u "${MULTISERVER_SCRIPT}" \
    --num-servers "${NUM_WORKERS}" \
    --servers-per-gpu "${SERVERS_PER_GPU}" \
    --start-port "${START_PORT}" \
    --port-step "${PORT_STEP}" \
    > "${CARLA_LOG}" 2>&1 &
...
TRAIN_CMD=(python -u new_hogwild_train_a3c_carla.py
    --num-workers "${NUM_WORKERS}"
    --workers-per-gpu "${WORKERS_PER_GPU}"
    --start-port "${START_PORT}"
    --port-step "${PORT_STEP}"
    --outdir "${OUTPUT_DIR}"
)
```

### 20. Temporal memory

**Best:** `async-rl` if the task requires memory across observations.

Why it matters: feed-forward policies are faster, but recurrent state is the
reference option when single observations do not contain enough driving context.

The backup has an optional LSTM model with careful `keep_same_state` bootstrap
handling. The improved and merged implementations are feed-forward only. For
current throughput-focused CARLA training, feed-forward is simpler. If route
history or partial observability becomes a limiting factor, the merged
implementation should add an optional GRU/LSTM path modeled after the backup.

`async-rl` proof:

```python
# async-rl-backup-.../train_a3c_carla.py
class A3CLSTM(Base):
    def __init__(self, n_actions, n_maneuvers=N_MANEUVERS):
        super().__init__(n_actions, n_maneuvers)
        with self.init_scope():
            self.lstm = L.LSTM(256, 256)

    def pi_and_v(self, state, keep_same_state=False):
        ...
        if keep_same_state:
            prev_h, prev_c = self.lstm.h, self.lstm.c
            h = self.lstm(h)
            self.lstm.h, self.lstm.c = prev_h, prev_c
```

`a3c_improved_1.py` proof:

```python
# AV/A_to_B_GPU_34/a3c_improved_1.py
self.actor = DeepDiscreteActor(state_shape, action_shape, self.device).to(self.device)
self.critic = DeepCritic(state_shape, critic_shape, self.device).to(self.device)
...
logits = self.actor(obs, speed, manouver)
value = self.critic(obs, speed, manouver)
```

`new_hogwild_*` proof:

```python
# AV/A_to_B_GPU_34/new_hogwild_a3c.py
self.model = SharedActorCritic(
    state_shape, action_shape, critic_shape,
    self.device).to(self.device)
...
logits, value = self.model(obs, speed, maneuver)
action_distribution = Categorical(logits=logits)
```

## Merge Completeness Check

The merged `new_hogwild_*` implementation contains the important ideas from
both sources:

| Source idea | Present in `new_hogwild_*` | Evidence |
|---|---|---|
| PyTorch CPU global model plus GPU workers from `a3c_improved_1.py` | Yes | `GlobalNetwork.model.share_memory()` and each worker creates a local `SharedActorCritic` on its assigned device. |
| Hogwild shared optimizer state | Yes | `SharedRMSprop` / `SharedAdam` call `share_memory()` on optimizer state tensors. |
| Avoid shared `.grad` race | Yes, improved beyond both sources | `transfer_local_gradients_to_global()` clones local grads instead of sharing `.grad` buffers. |
| A3C-style shared RMSprop from `async-rl` | Yes | `DEFAULT_OPTIMIZER = 'shared-rmsprop'` and `create_shared_optimizer()`. |
| LR decay from `async-rl` | Yes | `GlobalNetwork.set_lr_for_step()`. |
| Entropy beta from `async-rl` | Yes, improved | Per-transition entropy plus scheduled beta. |
| Gradient clipping from `async-rl` | Yes | `clip_grad_norm_()` active in `compute_and_apply_gradients()`. |
| NaN gradient/checkpoint guard from `async-rl` | Yes | `has_nan_grads()`, `has_nan_params()`, save refusal. |
| Structured JSONL logging from `async-rl` | Yes | `new_hogwild_training_logger.py`. |
| System monitor from `async-rl` | Yes, extended | `new_hogwild_system_monitor.py` adds optional NVML GPU stats. |
| Checkpoint/resume from `async-rl` | Yes, converted to PyTorch | Unified `.pth` checkpoint plus `resume_state.json`. |
| Git provenance from `async-rl` | No | `new_hogwild_prepare_output_dir.py` writes args only; port git status/log/diff capture if reproducibility matters. |
| Restart/rollback from `async-rl` | Partly | `new_hogwild_run_a3c.py` has rollback and restart caps, but give-up worker tracking still needs a fix. |
| Separate CARLA wrapper idea | Yes, improved | `new_hogwild_carla_wrapper.py` isolates reset/step/reward/action repeat/validation. |
| HPC launch workflow | New addition | `new_hogwild_train.slurm`. |

## Remaining Tradeoffs and Recommendations

1. Use `new_hogwild_*` as the main implementation.
2. Keep `--reward-mode legacy` when comparing against old runs. Use
   `--reward-mode shaped` only as a deliberate new experiment.
3. If old `a3c_improved_1.py` checkpoints matter, write a one-time converter
   from separate Actor/Critic checkpoints into `SharedActorCritic` where layer
   names and shapes match. The merged loader intentionally rejects that legacy
   format today.
4. If partial observability becomes a problem, add optional recurrent memory to
   `SharedActorCritic`, using the backup `A3CLSTM` behavior as the reference.
5. For instability debugging, run `new_hogwild_train_a3c_carla.py` with
   `--hogwild-lock-updates`, lower `--rollout-length`, and enable
   `--log-update-arrays`.
6. For long CARLA runs where actor/sensor leakage appears, use a positive
   `--world-reload-interval`.
7. Port the backup provenance snapshot into `new_hogwild_prepare_output_dir.py`
   if run reproducibility matters.
8. Fix `new_hogwild_run_a3c.py` so workers that exceed restart limits are
   disabled or removed from the monitored worker set.
