# Algorithm

Asynchronous advantage actor-critic (A3C) with Hogwild updates. The code is in `a3c_core.py`, the network in `model.py`.

## Shared state

- **Global model.** One actor-critic network on the CPU, in shared memory. Every worker process sees the same parameter tensors.
- **Shared optimizer.** `SharedRMSprop` (default) or `SharedAdam`. Its state tensors are in shared memory too, so all workers update the same running averages.
- **Local models.** Each worker has its own copy of the network on its device (a GPU or the CPU). It acts and computes gradients with this copy.
- **Gradients are not shared.** A worker copies its local gradients to the CPU and assigns them to the global model inside its own process, then calls `optimizer.step()`.
- **No lock.** Workers update the global parameters without waiting for each other, so updates can interleave. This is the Hogwild scheme. `--hogwild-lock-updates` makes the updates take turns.
- **Counters.** The global step, episode, and update counters are shared values. One global step is one agent decision of any worker. `--steps` and both schedules count global steps.

## Worker loop

1. **Episode start.** Copy the global weights to the local model. Reset the environment.
2. **Step.** Increase the global step counter. Run the local model on the observation and sample an action from the categorical policy. Step the environment. Keep the value, the log-probability, the entropy, and the reward.
3. **Update.** After `--rollout-length` steps, or earlier when the episode ends, turn the rollout into one optimizer update (next section).
4. **Sync.** After every `--sync-every-n-updates` updates, copy the global weights to the local model again.
5. Repeat from step 2 until the episode ends, then from step 1.

Each worker limits PyTorch and BLAS to one CPU thread and seeds its generators with `--seed + 1009 * worker id`.

## One update

For a rollout of `n` steps:

```
R_n = 0                              if the episode ended in this rollout
R_n = V(s_n)                         otherwise: bootstrap from the critic, no gradient
R_t = r_t + gamma * R_{t+1}          for t = n-1 ... 0
A_t = R_t - V(s_t)                   V detached

policy loss = -mean( log pi(a_t | s_t) * A_t )
value loss  = value_loss_coef * smooth_l1( V(s_t), R_t )
loss        = policy loss + value loss - beta * mean( entropy of pi(. | s_t) )
```

- **Episode end includes the length limit.** An episode cut by `--episode-max-decisions` is treated as finished and gets no bootstrap value.
- **Advantage normalization.** In the policy loss, `A_t` is normalized to zero mean and unit standard deviation within the rollout. `--no-normalize-advantages` turns this off. The value loss always uses the plain returns.
- **Reward scale.** With `--reward-scale X` other than 0, every reward is divided by `X` before the returns are computed.
- **Clipping.** The gradient is computed on the local model and its global norm is clipped to `--max-grad-norm` (`0` turns clipping off).
- **NaN guard.** When any gradient holds NaN, the update is skipped and the worker copies the global weights again.
- **Apply.** The worker sets the learning rate for the current global step, copies its gradients to the global model, and steps the shared optimizer.

## Schedules

Both schedules are functions of the global step `t` and of `--steps`.

- **Learning rate.** Linear decay from `--lr` to 0 at `--steps`: `lr * (steps - t - 1) / steps`.
- **Entropy coefficient `beta`.** Linear from `--beta-start` to `--beta-end` over the first `--beta-anneal-frac * steps` steps, constant afterwards. `--beta X` sets a constant value.
- With `--steps 0` the run has no step limit, the learning rate stays at `--lr`, and `beta` stays at `--beta-end`.

`--resume` continues both schedules, because it restores the step counter. `--init-from` starts them from step 0. Changing `--steps` on resume moves both schedules.

## Network

`SharedActorCritic` in `model.py`, about 2.65 million parameters with the default inputs:

```
image [3, H, W]  -> 4 x (Conv, stride 2, ReLU): 32, 64, 128, 256 channels -> average pool to 4 x 4 -> 4096
speed [1]        -> Linear 32, ReLU
maneuver         -> one-hot of 3 -> Linear 32, ReLU
concatenate 4160 -> Linear 512, ReLU -> Linear 256, ReLU -> policy head: n_actions logits
                                                         -> value head: 1 value
```

- Actor and critic share everything except the two heads.
- The network is feed-forward and sees one frame per decision. It keeps no state between steps.
- The adaptive pooling makes it independent of the image size.

`a3c_core.py` builds the global model and every local copy through `build_model`. To train another network, see `docs/how_to_adjust.md`.

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--steps` | 10 000 000 | Global steps of the whole run, summed over all workers. `0` means no limit |
| `--rollout-length` | 20 | Steps per update |
| `--gamma` | 0.99 | Discount factor |
| `--lr` | 1e-4 | Initial learning rate |
| `--optimizer` | `shared-rmsprop` | Or `shared-adam` |
| `--beta-start`, `--beta-end` | 0.02, 0.002 | Entropy coefficient at the start and at the end of the annealing |
| `--beta-anneal-frac` | 0.6 | Part of `--steps` over which `beta` is annealed |
| `--beta` | not set | Constant entropy coefficient, replaces the schedule |
| `--value-loss-coef` | 1.0 | Weight of the value loss |
| `--max-grad-norm` | 5.0 | Gradient norm limit |
| `--no-normalize-advantages` | off | Turns advantage normalization off |
| `--reward-scale` | 0 | Divides rewards. `0` means no scaling |
| `--weight-decay` | 0 | Weight decay of the optimizer |
| `--sync-every-n-updates` | 1 | Updates between two copies of the global weights |
| `--hogwild-lock-updates` | off | Applies updates one at a time |
| `--gc-interval` | 10 | Episodes between garbage collection and GPU cache release in a worker |
| `--seed` | 52 | Base random seed |

The optimizer internals have no flag. They are constants in `train_a3c.py`: RMSprop `alpha` 0.99 and `eps` 1e-5, Adam betas 0.9 and 0.999 and `eps` 1e-8.
