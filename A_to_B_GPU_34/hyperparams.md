# Hyperparameter and training changes from `a2c_rgb_try2.py` to `new_hogwild*` A3C

This file compares the old single-agent A2C implementation in
`a2c_rgb_try2.py` with the current Hogwild A3C implementation split across
`new_hogwild_train_a3c_carla.py`, `new_hogwild_a3c.py`,
`new_hogwild_carla_wrapper.py`, and `new_hogwild_run_a3c.py`.

The previous notes were mostly correct, but one important detail needed
correction: the new implementation contains a shaped reward mode, but the
current default is still `legacy`, not `shaped`.

## Summary table

| Area | Old `a2c_rgb_try2.py` | New `new_hogwild*` A3C |
| --- | --- | --- |
| Training style | One `DeepActorCriticAgent` process with local actor and critic. | Multiple asynchronous workers update one shared global network in CPU shared memory. |
| Network | Separate `DeepDiscreteActor` and `DeepCritic` models, each with its own feature extractor. | One `SharedActorCritic` with shared CNN/speed/maneuver trunk and separate policy/value heads. |
| Optimizer | Two standard `torch.optim.Adam` optimizers. | One shared optimizer: default `shared-rmsprop`, optional `shared-adam`. |
| Learning rate | `settings.LR = 1e-4`, fixed for the whole run. | `DEFAULT_LR = 1e-4`, linearly decayed toward `0` over `DEFAULT_STEPS = 10_000_000`. |
| Weight decay | `weight_decay=1e-2` on both Adam optimizers. | `DEFAULT_WEIGHT_DECAY = 0.0`. |
| Discount | `settings.GAMMA = 0.99`. | `DEFAULT_GAMMA = 0.99`. No change in default value. |
| Rollout length | Optimizes after `5` steps or episode end. | `DEFAULT_ROLLOUT_LENGTH = 20`, then update or episode end. |
| Entropy | `settings.USE_ENTROPY = True`; subtracts raw entropy with implicit coefficient `1.0`. | Scheduled coefficient: `beta_start=0.02`, `beta_end=0.002`, annealed over `60%` of training steps. `--beta` can force a constant coefficient. |
| Advantage handling | Uses raw TD error, `td_target - critic_prediction`. | Computes return minus value and normalizes advantages by rollout mean/std by default. |
| Value loss | Smooth L1 critic loss, optimized separately. | Smooth L1 value loss multiplied by `DEFAULT_VALUE_LOSS_COEF = 1.0` and included in one total loss. |
| Gradient clipping | Not present. | `DEFAULT_MAX_GRAD_NORM = 5.0`; local gradients are clipped before being copied to global model. |
| NaN protection | Not present. | Checks NaN gradients before optimizer step, skips bad rollout, resyncs local model; checkpoints also avoid saving NaN parameters. |
| Reward mode | Uses `CarlaEnv` legacy reward from `utils.reward_function`. | Wrapper supports `legacy` and `shaped`; current default is `DEFAULT_REWARD_MODE = 'legacy'`. |
| Reward scaling | Not present. | `DEFAULT_REWARD_SCALE = 0.0`, meaning no scaling by default; non-zero values divide rollout rewards before return calculation. |

## What was added and how

### 1. Hogwild A3C workers and shared global model

The old code creates one agent containing separate actor and critic models.
Each optimizer step updates only that local pair of models.

The new implementation creates a `GlobalNetwork` in `new_hogwild_a3c.py`.
Its model is moved to CPU and placed in shared memory with
`self.model.share_memory()`. Each `A3CWorker` owns a local copy, interacts with
its own CARLA environment, computes gradients locally, then copies those
gradients into the shared global model before calling the shared optimizer.

The number and placement of workers are controlled from
`new_hogwild_train_a3c_carla.py`:

- `DEFAULT_NUM_WORKERS = 2`
- `DEFAULT_WORKERS_PER_GPU = 2`
- `DEFAULT_WORKER_GPU_START = 0`
- `DEFAULT_START_PORT = 2000`
- `DEFAULT_PORT_STEP = 100`

This is an implementation-level change, not just a hyperparameter change: the
training loop changed from one synchronous local learner to asynchronous A3C
workers.

### 2. Shared actor-critic architecture

Old A2C imports `DiscreteActor` and `Critic` from `nets/a2c.py` and constructs
them separately:

- `self.actor = DeepDiscreteActor(...)`
- `self.critic = DeepCritic(...)`

Because they are separate modules, image/speed/maneuver feature extraction is
learned twice.

New A3C uses one `SharedActorCritic` model. It processes:

- image input through one CNN,
- speed through a small `Linear(1, 32)` branch,
- maneuver through a one-hot branch with `Linear(num_maneuvers, 32)`,
- concatenated features through a shared fully connected trunk,
- then splits only at the final heads: `policy` and `value`.

So the actor and critic now share the expensive representation learning and
only differ in the final output layers.

### 3. Optimizer change: shared RMSprop by default

Old A2C uses two independent Adam optimizers:

- actor Adam, `lr=1e-4`, `weight_decay=1e-2`
- critic Adam, `lr=1e-4`, `weight_decay=1e-2`

New A3C uses a shared optimizer over the single shared model:

- default `DEFAULT_OPTIMIZER = 'shared-rmsprop'`
- optional `--optimizer shared-adam`
- default RMSprop internals: `alpha=0.99`, `eps=1e-5`
- default Adam internals if selected: `betas=(0.9, 0.999)`, `eps=1e-8`
- default `DEFAULT_WEIGHT_DECAY = 0.0`

This was added by defining `SharedRMSprop` and `SharedAdam` classes whose state
tensors are moved into shared memory. `create_shared_optimizer(...)` selects the
implementation from the config.

### 4. Learning rate decay

Old A2C reads `settings.LR = 1e-4` once and leaves the Adam learning rates
constant. It logs the optimizer LR after each step, but does not modify it.

New A3C keeps the same initial default LR, `DEFAULT_LR = 1e-4`, but changes it
during training. `GlobalNetwork.set_lr_for_step(global_t)` computes:

```text
lr = config.lr * max(0, (config.steps - global_t - 1) / config.steps)
```

With the default `DEFAULT_STEPS = 10_000_000`, this linearly decays the learning
rate from almost `1e-4` at the start toward `0` at the end of the step budget.
The scheduler is applied immediately before the global optimizer step.

### 5. Longer n-step rollout

Old A2C accumulates rewards and trajectory entries, then calls `optimize(...)`
when:

```text
step_num >= 5 or done
```

New A3C uses:

```text
steps_since_last_update >= config.rollout_length or done
```

with `DEFAULT_ROLLOUT_LENGTH = 20`.

Both versions use bootstrapped n-step returns when the episode has not ended.
The difference is that the A3C default waits for up to 20 decisions per update,
which gives a longer return horizon per optimizer update than the old 5-step
setting.

### 6. Advantage normalization

Old A2C calculates:

```text
td_err = td_target - critic_prediction
actor_loss += -log_prob * td_err
```

The raw TD error is used directly as the policy advantage.

New A3C calculates:

```text
advantages = returns - values.detach()
```

Then, by default, it normalizes the rollout advantages:

```text
advantages = (advantages - mean) / max(std, 1e-8)
```

This is controlled by `DEFAULT_NORMALIZE_ADVANTAGES = True`; it can be disabled
with `--no-normalize-advantages`.

Important detail: normalized advantages are used for the policy loss only. The
value loss still trains against the unnormalized return targets.

### 7. Entropy coefficient changed from implicit `1.0` to scheduled beta

Old A2C has a boolean `settings.USE_ENTROPY = True`. When enabled, it computes:

```text
actor_loss = mean(policy_losses) - entropy.mean()
```

There is no explicit entropy coefficient, so the effective coefficient is
`1.0`, which is large compared with common actor-critic settings.

New A3C uses an explicit coefficient in the total loss:

```text
total_loss = policy_loss + value_loss - entropy_coef * entropy_mean
```

The default coefficient is scheduled by
`compute_entropy_coefficient_for_step(...)`:

- `DEFAULT_BETA_START = 0.02`
- `DEFAULT_BETA_END = 0.002`
- `DEFAULT_BETA_ANNEAL_FRAC = 0.6`

With `DEFAULT_STEPS = 10_000_000`, entropy decays linearly from `0.02` to
`0.002` during the first `6,000,000` global steps and then stays at `0.002`.
Passing `--beta X` sets both start and end to `X`, giving a constant entropy
coefficient.

### 8. Combined loss and single backward pass

Old A2C calculates actor and critic losses separately:

- actor loss backward with `retain_graph=True`, then actor optimizer step,
- critic loss backward, then critic optimizer step.

New A3C builds one scalar:

```text
total_loss = policy_loss + value_loss_coef * smooth_l1(value, return)
             - entropy_coef * entropy
```

In code, the coefficient is applied when `value_loss` is created:

```text
value_loss = config.value_loss_coef * smooth_l1_loss(...)
```

Then `total_loss.backward()` is called once on the local shared actor-critic
copy. After that, gradients are clipped, checked for NaNs, copied into the
global network, and applied by the shared optimizer.

### 9. Gradient clipping

Old A2C does not call `clip_grad_norm_`.

New A3C clips local model gradients before they are transferred to the global
model:

```text
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

The default is `DEFAULT_MAX_GRAD_NORM = 5.0`. Setting `--max-grad-norm 0` or a
negative value effectively disables this block.

### 10. NaN guards and rollback/recovery support

Old A2C has no explicit NaN protection. If a backward pass creates NaN
gradients, the optimizer can apply them.

New A3C adds several safety points:

- `has_nan_grads(model)` checks local gradients after backward and clipping.
- If NaN gradients are found, the worker logs them, skips the optimizer update,
  clears the rollout, and resynchronizes from the global model.
- `has_nan_params(model)` prevents saving checkpoints with NaN parameters.
- `new_hogwild_run_a3c.py` can restart dead workers and roll the global network
  back to the latest usable non-NaN checkpoint after rapid repeated crashes.

These are stability mechanisms rather than learning hyperparameters, but they
directly affect whether unstable runs can recover.

### 11. Reward mode and shaped reward coefficients

Old A2C uses the reward returned by `CarlaEnv`, which comes from
`utils.reward_function(...)`. That legacy reward combines collision/off-route
termination, lane invasion reward, a sine-based speed reward, and a route
distance reward.

New A3C wraps `CarlaEnv` with `CarlaA3CWrapper`. The wrapper can either pass the
legacy reward through or replace it with shaped components.

Current default in `new_hogwild_train_a3c_carla.py`:

```text
DEFAULT_REWARD_MODE = 'legacy'
```

Therefore, by default, the A3C run still trains on the legacy scalar reward.

When started with `--reward-mode shaped`, the wrapper computes a reward from
named components:

- progress toward the goal, clipped to `[-5, 5]`,
- target-speed score around `DEFAULT_REWARD_TARGET_SPEED_KMH = 20.0`,
- route-distance penalty,
- fixed time penalty,
- goal bonus,
- collision penalty,
- off-route penalty,
- lane-invasion penalty.

The default shaped coefficients are:

- `DEFAULT_REWARD_PROGRESS_COEF = 1.0`
- `DEFAULT_REWARD_TARGET_SPEED_COEF = 1.0`
- `DEFAULT_REWARD_ROUTE_PENALTY_COEF = 0.1`
- `DEFAULT_REWARD_TIME_PENALTY = 0.01`
- `DEFAULT_REWARD_GOAL_BONUS = 50.0`
- `DEFAULT_REWARD_COLLISION_PENALTY = 50.0`
- `DEFAULT_REWARD_OFFROUTE_PENALTY = 25.0`
- `DEFAULT_REWARD_LANE_INVASION_PENALTY = 5.0`
- `DEFAULT_REWARD_TARGET_SPEED_KMH = 20.0`
- `DEFAULT_REWARD_OFFROUTE_THRESHOLD = 10.0`
- `DEFAULT_REWARD_CLIP = 50.0`

The wrapper also logs the per-component reward breakdown, which the old A2C
training loop did not provide.

### 12. Reward scaling hook

New A3C adds `DEFAULT_REWARD_SCALE = 0.0`. The implementation treats `0.0` as
"do not scale". If set to a non-zero value, rollout rewards are divided by that
scale before discounted returns are computed.

Old A2C has no equivalent reward scaling hook.

## Correctness check

The corrected comparison is:

- correct: gradient clipping was added with default norm `5.0`;
- correct: advantage normalization was added and is enabled by default;
- correct: entropy changed from boolean/raw entropy to a scheduled coefficient;
- correct: linear learning-rate decay was added;
- correct: NaN gradient/parameter guards were added;
- correct: the network changed from separate actor/critic modules to one shared actor-critic;
- correct: losses are now combined into one backward pass;
- correct: optimizer changed from two standard Adam optimizers with `1e-2` weight decay to one shared optimizer with default no weight decay;
- correct: rollout length changed from `5` to `20`;
- corrected: shaped reward support was added, but the current default run mode is `legacy`, so shaped reward is available only when `--reward-mode shaped` is selected.
