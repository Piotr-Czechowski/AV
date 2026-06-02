"""Core A3C/Hogwild training code for CARLA.

The global model and optimizer live in CPU shared memory. Each worker owns a
local model on its assigned device, collects a short rollout, computes a loss
locally, copies gradients to the global model, and applies an asynchronous
optimizer update.

This module intentionally does not read argparse or settings.py. The entry
point passes a plain config namespace with every value workers need.
"""

import contextlib
import os
import time
import gc
import random
from collections import namedtuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions.categorical import Categorical

from new_hogwild_timing_utils import TimingAccumulator
from new_hogwild_training_logger import TrainingLogger
from new_hogwild_system_monitor import WorkerMonitor
from new_hogwild_carla_wrapper import CarlaA3CWrapper


Transition = namedtuple(
    "Transition", ["value_s", "log_prob_a", "entropy", "action"])


# ---------------------------------------------------------------------------
# Shared optimizers with CPU state for Hogwild workers
# ---------------------------------------------------------------------------

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, eps=epsilon,
                         weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1, device=p.device)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
        self.share_memory()

    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    value = state.get(key)
                    if torch.is_tensor(value) and value.device.type == 'cpu':
                        value.share_memory_()


class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-4, alpha=0.99, epsilon=1e-5,
                 weight_decay=0.0):
        super().__init__(params, lr=lr, alpha=alpha, eps=epsilon,
                         weight_decay=weight_decay, momentum=0.0,
                         centered=False)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1, device=p.device)
                state['square_avg'] = torch.zeros_like(p.data)
        self.share_memory()

    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                for key in ('step', 'square_avg'):
                    value = state.get(key)
                    if torch.is_tensor(value) and value.device.type == 'cpu':
                        value.share_memory_()


class SharedActorCritic(nn.Module):
    """Batch-size-1 friendly actor-critic with one shared visual trunk."""

    def __init__(self, input_shape, action_shape, critic_shape=1,
                 device=torch.device('cpu'), num_maneuvers=3):
        super().__init__()
        self.device = device
        self.num_maneuvers = num_maneuvers
        in_channels = int(input_shape[2])

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
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
        )
        self.maneuver_fc = nn.Sequential(
            nn.Linear(num_maneuvers, 32),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 32 + 32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(256, action_shape)
        self.value = nn.Linear(256, critic_shape)

    def forward(self, x, speed=None, maneuver=None):
        x = x.to(self.device, dtype=torch.float32)
        features = self.cnn(x).flatten(1)

        if speed is None:
            speed = torch.zeros((x.size(0), 1), device=self.device)
        else:
            speed = speed.to(self.device, dtype=torch.float32) \
                .view(x.size(0), -1)
            if speed.size(1) != 1:
                speed = speed[:, :1]
        speed_features = self.speed_fc(speed)

        if maneuver is None:
            maneuver = torch.ones((x.size(0),), dtype=torch.long,
                                  device=self.device)
        else:
            maneuver = maneuver.to(self.device, dtype=torch.long).view(-1)
        maneuver = maneuver.clamp(0, self.num_maneuvers - 1)
        maneuver = F.one_hot(maneuver,
                             num_classes=self.num_maneuvers).float()
        maneuver_features = self.maneuver_fc(maneuver)

        hidden = self.trunk(torch.cat(
            [features, speed_features, maneuver_features], dim=1))
        return self.policy(hidden), self.value(hidden)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_nan_grads(model):
    """Return the list of parameter names whose .grad contains NaN."""
    nan_names = []
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any().item():
            nan_names.append(name)
    return nan_names


def has_nan_params(model):
    for _, p in model.named_parameters():
        if torch.isnan(p.data).any().item():
            return True
    return False


def transfer_local_gradients_to_global(local_model, global_model,
                                       global_device):
    """Copy local gradients into the shared model for this process.

    Parameters and optimizer state are shared. Gradient buffers are not shared:
    each worker assigns its own cloned .grad tensors before optimizer.step().
    """
    for local_param, global_param in zip(local_model.parameters(),
                                         global_model.parameters()):
        if local_param.grad is None:
            global_param.grad = None
        else:
            # The .grad attribute is intentionally process-local. Parameters
            # and optimizer state share storage; sharing grad buffers would
            # let workers overwrite each other's pending gradients.
            global_param.grad = local_param.grad.detach().to(
                global_device, non_blocking=False).clone()


def compute_total_gradient_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad * p.grad).item())
    return total ** 0.5


def create_shared_optimizer(params, config):
    optimizer = getattr(config, 'optimizer', 'shared-rmsprop')
    if optimizer == 'shared-rmsprop':
        return SharedRMSprop(
            params, lr=config.lr,
            alpha=getattr(config, 'rmsprop_alpha', 0.99),
            epsilon=getattr(config, 'rmsprop_eps', 1e-5),
            weight_decay=config.weight_decay)
    if optimizer == 'shared-adam':
        return SharedAdam(
            params, lr=config.lr,
            betas=(getattr(config, 'adam_beta1', 0.9),
                   getattr(config, 'adam_beta2', 0.999)),
            epsilon=getattr(config, 'adam_eps', 1e-8),
            weight_decay=config.weight_decay)
    raise ValueError('unsupported optimizer: {}'.format(optimizer))


def compute_entropy_coefficient_for_step(config, global_t):
    start = float(getattr(config, 'beta_start',
                         getattr(config, 'entropy_coef', 0.0)))
    end = float(getattr(config, 'beta_end', start))
    steps = int(getattr(config, 'steps', 0))
    progress_fraction = float(getattr(config, 'beta_anneal_frac', 0.0))
    entropy_annealing_steps = max(1, int(steps * progress_fraction)) \
        if steps > 0 and progress_fraction > 0 else 1
    if steps <= 0 or progress_fraction <= 0:
        return end
    interpolation_factor = min(
        1.0, max(0.0, float(global_t) / float(entropy_annealing_steps)))
    return start + interpolation_factor * (end - start)


def summarize_reward_components(component_rollout):
    if not component_rollout:
        return {}
    keys = sorted({k for comp in component_rollout for k in comp.keys()})
    summary = {}
    for key in keys:
        values = [float(comp.get(key, 0.0)) for comp in component_rollout]
        summary['reward_{}_sum'.format(key)] = float(np.sum(values))
        summary['reward_{}_mean'.format(key)] = float(np.mean(values))
    return summary


# ---------------------------------------------------------------------------
# Global network (CPU shared-memory Hogwild)
# ---------------------------------------------------------------------------

class GlobalNetwork:
    def __init__(self, config, state_shape, action_shape, critic_shape):
        self.config = config
        self.device = torch.device('cpu')

        self.model = SharedActorCritic(
            state_shape, action_shape, critic_shape,
            self.device).to(self.device)
        self.model.share_memory()
        self.optimizer = create_shared_optimizer(
            self.model.parameters(), config)

        # Disabled by default; useful when checking whether optimizer updates
        # are racing in a specific run.
        self.update_lock = mp.Lock()
        self.stats_lock = mp.Lock()
        self.save_lock = mp.Lock()

        self.global_step = mp.Value('l', 0)
        self.global_episode = mp.Value('l', 0)
        self.total_updates = mp.Value('l', 0)
        self.last_checkpoint_boundary = mp.Value('l', 0)
        self.best_reward = mp.Value('d', -float('inf'))
        self.global_mean_reward = mp.Value('d', 0.0)
        self._reward_buf_size = config.num_workers * 100
        self.worker_mean_rewards = mp.Array('d', [0.0] * config.num_workers)
        self.recent_rewards = mp.Array('d', [0.0] * self._reward_buf_size)
        self.recent_reward_count = mp.Value('l', 0)
        self.recent_reward_index = mp.Value('l', 0)

    # -- counters --

    def increment_global_step(self, n=1):
        with self.global_step.get_lock():
            self.global_step.value += n
            return self.global_step.value

    def increment_global_episode(self):
        with self.stats_lock:
            self.global_episode.value += 1
            return self.global_episode.value

    def increment_total_updates(self):
        with self.total_updates.get_lock():
            self.total_updates.value += 1
            return self.total_updates.value

    def update_stats(self, worker_id, mean_reward, episode_reward):
        with self.stats_lock:
            is_new_best = False
            if episode_reward > self.best_reward.value:
                self.best_reward.value = episode_reward
                is_new_best = True
            self.worker_mean_rewards[worker_id] = mean_reward
            idx = self.recent_reward_index.value % self._reward_buf_size
            self.recent_rewards[idx] = float(episode_reward)
            self.recent_reward_index.value += 1
            self.recent_reward_count.value = min(
                self._reward_buf_size, self.recent_reward_count.value + 1)
            recent = self.recent_rewards[:self.recent_reward_count.value]
            if recent:
                self.global_mean_reward.value = sum(recent) / len(recent)
            return self.global_mean_reward.value, is_new_best

    # -- LR scheduling (linear decay from config.lr to 0) --

    def set_lr_for_step(self, global_t):
        if self.config.steps <= 0:
            return self.config.lr
        progress_fraction = max(
            0.0, (self.config.steps - global_t - 1) / self.config.steps)
        lr = progress_fraction * self.config.lr
        self.optimizer.set_lr(lr)
        return lr

    # -- save / load --

    def _has_nan_params(self):
        return has_nan_params(self.model)

    def _checkpoint_state_unlocked(self, global_t=None):
        state = {
            'global_step': global_t if global_t is not None
                           else self.global_step.value,
            'global_episode': self.global_episode.value,
            'total_updates': self.total_updates.value,
            'last_checkpoint_boundary': self.last_checkpoint_boundary.value,
            'best_reward': self.best_reward.value,
            'global_mean_reward': self.global_mean_reward.value,
            'worker_mean_rewards': list(self.worker_mean_rewards[:]),
            'recent_rewards': list(self.recent_rewards[:]),
            'recent_reward_count': self.recent_reward_count.value,
            'recent_reward_index': self.recent_reward_index.value,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return state

    def save(self, path, global_t=None, nan_safe=True):
        if nan_safe and self._has_nan_params():
            print('[SAVE] refusing to save - NaN in global params',
                  flush=True)
            return False
        with self.save_lock:
            state = self._checkpoint_state_unlocked(global_t=global_t)
            torch.save(state, path)
            return True

    def save_boundary_checkpoint(self, run_output_dir, global_t,
                                 worker_id=None):
        save_frequency = int(getattr(self.config, 'save_frequency', 0))
        if save_frequency <= 0:
            return False, None
        boundary = int(global_t) // save_frequency
        if boundary <= 0:
            return False, None

        with self.save_lock:
            if boundary <= self.last_checkpoint_boundary.value:
                return False, None
            if self._has_nan_params():
                print('[SAVE] refusing boundary checkpoint - NaN in global '
                      'params', flush=True)
                return False, None

            state = self._checkpoint_state_unlocked(global_t=global_t)
            global_checkpoint_path = os.path.join(
                run_output_dir, 'checkpoint.pth')
            torch.save(state, global_checkpoint_path)
            with open(os.path.join(run_output_dir,
                                   'checkpoint_step.txt'), 'w') as f:
                f.write(str(global_t))

            if getattr(self.config, 'save_worker_checkpoints', False) and \
                    worker_id is not None:
                worker_output_dir = os.path.join(
                    run_output_dir, 'checkpoints',
                    'worker_{}'.format(worker_id))
                os.makedirs(worker_output_dir, exist_ok=True)
                torch.save(state, os.path.join(
                    worker_output_dir, 'checkpoint.pth'))
                with open(os.path.join(worker_output_dir,
                                       'checkpoint_step.txt'), 'w') as f:
                    f.write(str(global_t))

            import json
            state_path = os.path.join(run_output_dir, 'resume_state.json')
            previous_resume_state = {}
            try:
                if os.path.exists(state_path):
                    with open(state_path) as sf:
                        previous_resume_state = json.load(sf)
            except (OSError, ValueError, TypeError):
                previous_resume_state = {}

            resume_state = {
                'global_step': global_t,
                'training_args': vars(self.config)
                if not isinstance(self.config, dict) else self.config,
                'timestamp': datetime.now().strftime(
                    '%Y-%m-%dT%H:%M:%S.%f')[:-3],
            }
            for key in ('elapsed_training_s', 'last_session_elapsed_s',
                        'last_session_start_ts', 'last_session_end_ts'):
                if key in previous_resume_state:
                    resume_state[key] = previous_resume_state[key]
            tmp_path = state_path + '.tmp'
            with open(tmp_path, 'w') as sf:
                json.dump(resume_state, sf, indent=2, default=str)
            os.replace(tmp_path, state_path)

            self.last_checkpoint_boundary.value = boundary
            return True, global_checkpoint_path

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        with self.save_lock:
            if 'model' not in state:
                raise ValueError(
                    "checkpoint at {} is missing the 'model' key; the legacy "
                    "DiscreteActor+Critic checkpoint format is no longer "
                    "supported - only SharedActorCritic checkpoints can be "
                    "loaded".format(path))
            self.model.load_state_dict(state['model'])
            if 'optimizer' in state:
                try:
                    self.optimizer.load_state_dict(state['optimizer'])
                    self.optimizer.share_memory()
                except Exception as e:
                    print('[LOAD] optimizer restore failed: {}'.format(e),
                          flush=True)
            self.global_step.value = int(state.get('global_step', 0))
            self.global_episode.value = int(state.get('global_episode', 0))
            self.total_updates.value = int(state.get('total_updates', 0))
            self.last_checkpoint_boundary.value = int(
                state.get('last_checkpoint_boundary',
                          self.global_step.value //
                          max(1, int(getattr(self.config,
                                             'save_frequency', 1)))))
            self.best_reward.value = float(
                state.get('best_reward', -float('inf')))
            self.global_mean_reward.value = float(
                state.get('global_mean_reward', 0.0))
            for i, value in enumerate(state.get('worker_mean_rewards', [])):
                if i < len(self.worker_mean_rewards):
                    self.worker_mean_rewards[i] = float(value)
            for i, value in enumerate(state.get('recent_rewards', [])):
                if i < len(self.recent_rewards):
                    self.recent_rewards[i] = float(value)
            self.recent_reward_count.value = min(
                self._reward_buf_size, int(state.get('recent_reward_count', 0)))
            self.recent_reward_index.value = int(
                state.get('recent_reward_index', 0))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class A3CWorker(mp.Process):
    def __init__(self, worker_id, global_network, config, port, device,
                 run_output_dir, shutdown_event, log_queue=None, run_id=''):
        super().__init__()
        self.worker_id = worker_id
        self.global_network = global_network
        self.config = config
        self.port = port
        self.device_str = device
        self.run_output_dir = run_output_dir
        self.shutdown_event = shutdown_event
        self.log_queue = log_queue
        self.run_id = run_id

        self.device = None
        self.model = None
        self.trajectory = []
        self.rewards = []
        self.reward_components = []
        self.mean_reward = 0.0
        self.episode_rewards_history = []
        self.local_updates = 0
        self.total_steps = 0
        self.episode_count = 0
        self._initialized = False

    # -- setup --

    def _init_networks(self):
        if self._initialized:
            return
        self.device = torch.device(self.device_str)
        state_shape = [self.config.res, self.config.res, 3]
        critic_shape = 1
        action_shape = self.config.n_actions

        self.model = SharedActorCritic(
            state_shape, action_shape, critic_shape,
            self.device).to(self.device)
        self.sync_with_global()
        self._initialized = True

    def _clear_rollout_buffers(self):
        """Drop transitions collected since the last optimizer update."""
        self.trajectory.clear()
        self.rewards.clear()
        self.reward_components.clear()

    def sync_with_global(self):
        """Refresh this worker's local model from CPU shared parameters."""
        for global_param, local_param in zip(
                self.global_network.model.parameters(),
                self.model.parameters()):
            local_param.data.copy_(
                global_param.data.to(self.device, non_blocking=True))
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    # -- action --

    def get_action(self, obs, speed, maneuver, testing=False):
        """Sample or choose an action and store policy terms for training."""
        logits, value = self.model(obs, speed, maneuver)
        action_distribution = Categorical(logits=logits)

        if testing:
            action = action_distribution.probs.argmax(dim=-1)
        else:
            action = action_distribution.sample()

        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        action_np = action.cpu().numpy().squeeze(0)

        if not testing:
            self.trajectory.append(Transition(
                value_s=value, log_prob_a=log_prob, entropy=entropy,
                action=int(action_np)))

        return int(action_np), float(value.detach().cpu().item()), \
               float(entropy.detach().cpu().item())

    # -- loss and update --

    def compute_and_apply_gradients(self, final_state_tensor, done,
                                    final_speed_tensor, maneuver_tensor,
                                    training_logger, timer, global_t):
        """Turn the current rollout into one global optimizer update."""
        if not self.trajectory:
            return None

        phase_start_time = timer.start()
        with torch.no_grad():
            # Bootstrap from the critic unless this rollout ended the episode.
            if done:
                discounted_return = torch.zeros(1, 1, device=self.device)
            else:
                _, discounted_return = self.model(
                    final_state_tensor, final_speed_tensor, maneuver_tensor)

        returns = []
        rewards_scaled = []
        scale = self.config.reward_scale
        for reward in reversed(self.rewards):
            # reward_scale=0 means "use wrapper reward as-is".
            r = reward / scale if scale and scale != 0.0 else reward
            rewards_scaled.insert(0, r)
            discounted_return = r + self.config.gamma * discounted_return
            returns.insert(0, discounted_return)

        batch = Transition(*zip(*self.trajectory))
        values = batch.value_s
        log_probs = batch.log_prob_a
        entropies = batch.entropy

        returns_tensor = torch.cat(
            [r.detach().view(1, -1) for r in returns], dim=0)
        values_tensor = torch.cat([v.view(1, -1) for v in values], dim=0)
        log_probs_tensor = torch.cat(
            [lp.view(-1) for lp in log_probs], dim=0)
        entropies_tensor = torch.cat(
            [e.view(-1) for e in entropies], dim=0)

        advantages_tensor = returns_tensor - values_tensor.detach()
        policy_advantages = advantages_tensor
        if getattr(self.config, 'normalize_advantages', True) and \
                policy_advantages.numel() > 1:
            advantages_mean = policy_advantages.mean()
            advantages_std = policy_advantages.std(unbiased=False)
            policy_advantages = (policy_advantages - advantages_mean) / \
                advantages_std.clamp_min(1e-8)

        policy_loss = -(log_probs_tensor *
                        policy_advantages.view(-1)).mean()
        value_loss = self.config.value_loss_coef * F.smooth_l1_loss(
            values_tensor, returns_tensor)
        entropy_mean = entropies_tensor.mean()
        entropy_coef = compute_entropy_coefficient_for_step(
            self.config, global_t)
        total_loss = policy_loss + value_loss \
            - entropy_coef * entropy_mean
        timer.record('loss_compute', phase_start_time)

        phase_start_time = timer.start()
        self.model.zero_grad()
        total_loss.backward()
        timer.record('backward', phase_start_time)

        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.config.max_grad_norm)

        layers_with_nan_grad = has_nan_grads(self.model)
        if layers_with_nan_grad:
            training_logger.log_nan(
                global_t=global_t,
                nan_count=len(layers_with_nan_grad),
                nan_layers=layers_with_nan_grad)
            print('[NaN] W{} NaN grad at step {} in {} layers - skipping '
                  'update, re-sync from global'.format(
                      self.worker_id, global_t,
                      len(layers_with_nan_grad)), flush=True)
            self.sync_with_global()
            self._clear_rollout_buffers()
            return None

        gradient_norm = compute_total_gradient_norm(self.model)

        phase_start_time = timer.start()
        update_ctx = self.global_network.update_lock \
            if self.config.hogwild_lock_updates else contextlib.nullcontext()
        with update_ctx:
            current_learning_rate = self.global_network.set_lr_for_step(
                global_t)
            transfer_local_gradients_to_global(
                self.model, self.global_network.model,
                self.global_network.device)
            self.global_network.optimizer.step()
            self.global_network.optimizer.zero_grad()
        timer.record('optim_update', phase_start_time)

        update_number = self.global_network.increment_total_updates()
        self.local_updates += 1
        adv_values = [float(x) for x
                      in advantages_tensor.detach().cpu().view(-1)]
        val_values = [float(x) for x
                      in values_tensor.detach().cpu().view(-1)]
        component_summary = summarize_reward_components(
            self.reward_components)

        training_logger.log_update(
            update_count=update_number,
            global_t=global_t,
            trajectory_length=len(self.trajectory),
            is_terminal=bool(done),
            pi_loss=float(policy_loss.detach().cpu().item()),
            v_loss=float(value_loss.detach().cpu().item()),
            total_loss=float(total_loss.detach().cpu().item()),
            gradient_norm=gradient_norm,
            lr=current_learning_rate,
            entropy_coef=entropy_coef,
            advantages=adv_values,
            values=val_values,
            entropies=[float(e.detach().cpu().item()) for e in entropies],
            rewards=rewards_scaled,
            **component_summary
        )

        self._clear_rollout_buffers()
        return update_number

    # -- checkpointing (runs from the worker on step boundaries) --

    def _save_checkpoint(self, global_t, training_logger):
        ok, checkpoint_path = self.global_network.save_boundary_checkpoint(
            self.run_output_dir, global_t=global_t,
            worker_id=self.worker_id)
        if not ok:
            return False

        training_logger.log_checkpoint(path=checkpoint_path,
                                       global_t=global_t)
        return True

    # -- episode logging --

    def _log_episode(self, training_logger, env, global_episode, global_t,
                     episode_total_reward, episode_step_count, duration_s):
        self.episode_rewards_history.append(episode_total_reward)
        window = min(100, len(self.episode_rewards_history))
        self.mean_reward = float(
            np.mean(self.episode_rewards_history[-window:]))
        global_mean, is_new_best = self.global_network.update_stats(
            self.worker_id, self.mean_reward, episode_total_reward)

        action_counts = env._action_counts.tolist() \
            if hasattr(env, '_action_counts') else None

        training_logger.log_episode(
            global_episode=global_episode,
            global_t=global_t,
            total_reward=episode_total_reward,
            steps=episode_step_count,
            duration_s=duration_s,
            max_speed_kmh=getattr(env, '_episode_max_speed', None),
            min_route_dist=getattr(env, '_episode_min_route_dist', None),
            goal_dist=getattr(env, '_episode_goal_dist', None),
            reached_goal=getattr(env, '_episode_reached_goal', False),
            action_counts=action_counts,
            collisions=len(env.env.collision_history_list)
                       if hasattr(env, 'env') and
                       hasattr(env.env, 'collision_history_list') else None,
            port=env.port,
            local_mean_reward=self.mean_reward,
            global_mean_reward=global_mean,
            is_new_best=is_new_best,
            reward_components=getattr(env, '_episode_reward_components', None),
        )

        if is_new_best:
            print('[BEST] W{} new best reward {:.2f} at Ep{}'.format(
                self.worker_id, episode_total_reward, global_episode),
                flush=True)
            best_path = os.path.join(
                self.run_output_dir, 'best_checkpoint.pth')
            if self.global_network.save(best_path, global_t=global_t):
                training_logger.log_checkpoint(
                    path=best_path, global_t=global_t,
                    checkpoint_kind='best')

        if self.log_queue is not None:
            try:
                log_record = {
                    'worker_id': self.worker_id,
                    'episode': global_episode,
                    'global_step': global_t,
                    'reward': episode_total_reward,
                    'local_mean_reward': self.mean_reward,
                    'global_mean_reward': global_mean,
                    'episode_length': episode_step_count,
                    'total_steps': self.total_steps,
                }
                if hasattr(env, '_episode_max_speed'):
                    log_record['max_speed_kmh'] = env._episode_max_speed
                if hasattr(env, '_episode_min_route_dist'):
                    log_record['min_route_dist'] = env._episode_min_route_dist
                if hasattr(env, '_episode_goal_dist'):
                    log_record['distance_from_target'] = env._episode_goal_dist
                if hasattr(env, '_episode_reward_components'):
                    for k, v in env._episode_reward_components.items():
                        log_record['reward_component_{}'.format(k)] = v
                if action_counts is not None:
                    total_a = sum(action_counts)
                    if total_a > 0:
                        log_record['most_chosen_action_pct'] = \
                            max(action_counts) / total_a
                    for i, c in enumerate(action_counts):
                        log_record['action_{}'.format(i)] = c
                self.log_queue.put_nowait(log_record)
            except Exception:
                pass

    # -- main loop --

    def run(self):
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        try:
            import sys
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

        seed = int(getattr(self.config, 'seed', 0)) + self.worker_id * 1009
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and \
                str(self.device_str).startswith('cuda'):
            try:
                torch.cuda.set_device(torch.device(self.device_str))
                torch.cuda.manual_seed(seed)
            except Exception:
                pass

        print('[W{}] starting on {} port {}'.format(
            self.worker_id, self.device_str, self.port), flush=True)

        self._init_networks()

        training_logger = TrainingLogger(
            self.run_output_dir, self.worker_id,
            log_steps=self.config.log_steps,
            log_update_arrays=self.config.log_update_arrays)

        worker_monitor = WorkerMonitor(
            self.run_output_dir, self.worker_id,
            interval=self.config.monitor_interval)
        worker_monitor.start()

        timer = TimingAccumulator()
        last_diag_wall = time.time()
        last_diag_update = 0

        # restore per-worker mean reward from logs if present
        try:
            prev = self.global_network.worker_mean_rewards[self.worker_id]
            if prev != 0.0:
                self.mean_reward = prev
                self.episode_rewards_history = [prev] * 100
        except Exception:
            pass

        env = CarlaA3CWrapper(
            port=self.port, scenario=self.config.scenario,
            camera=self.config.camera, resX=self.config.res,
            resY=self.config.res,
            action_space=self.config.action_type,
            mp_density=self.config.mp_density,
            max_connect_retries=self.config.max_connect_retries,
            connect_retry_wait=self.config.connect_retry_wait,
            reconnect_wait=self.config.carla_timeout_wait,
            save_episodes=self.config.save_episodes,
            save_episode_interval=self.config.save_episode_interval,
            run_id=self.run_id, n_actions=self.config.n_actions,
            run_output_dir=self.run_output_dir,
            action_repeat=self.config.action_repeat,
            episode_max_decisions=self.config.episode_max_decisions,
            world_reload_interval=self.config.world_reload_interval,
            reward_mode=self.config.reward_mode,
            reward_progress_coef=self.config.reward_progress_coef,
            reward_target_speed_coef=self.config.reward_target_speed_coef,
            reward_route_penalty_coef=self.config.reward_route_penalty_coef,
            reward_time_penalty=self.config.reward_time_penalty,
            reward_goal_bonus=self.config.reward_goal_bonus,
            reward_collision_penalty=self.config.reward_collision_penalty,
            reward_offroute_penalty=self.config.reward_offroute_penalty,
            reward_lane_invasion_penalty=
            self.config.reward_lane_invasion_penalty,
            reward_target_speed_kmh=self.config.reward_target_speed_kmh,
            reward_offroute_threshold=self.config.reward_offroute_threshold,
            reward_clip=self.config.reward_clip,
            verbose_env_logs=getattr(self.config, 'verbose_env_logs', False),
        )

        try:
            while not self.shutdown_event.is_set():
                try:
                    self._clear_rollout_buffers()
                    self.episode_count += 1
                    current_episode = \
                        self.global_network.increment_global_episode()
                    env.global_episode = current_episode

                    phase_start_time = timer.start()
                    self.sync_with_global()
                    timer.record('sync', phase_start_time)

                    phase_start_time = timer.start()
                    state, speed, maneuver = env.reset()
                    timer.record('env_reset', phase_start_time)

                    done = False
                    episode_total_reward = 0.0
                    episode_step_count = 0
                    steps_since_last_update = 0
                    ep_start = time.time()

                    while not done and not self.shutdown_event.is_set():
                        global_t = self.global_network.increment_global_step(
                            1)
                        if self.config.steps > 0 and \
                                global_t > self.config.steps:
                            self.shutdown_event.set()
                            break

                        episode_step_count += 1
                        steps_since_last_update += 1
                        self.total_steps += 1

                        state_tensor = torch.from_numpy(state).float() \
                            .unsqueeze(0).to(self.device)
                        speed_tensor = torch.tensor(
                            [[speed]], dtype=torch.float32,
                            device=self.device)
                        maneuver_tensor = torch.tensor(
                            [maneuver], device=self.device)

                        phase_start_time = timer.start()
                        action, value_f, entropy_f = self.get_action(
                            state_tensor, speed_tensor, maneuver_tensor,
                            testing=self.config.testing)
                        timer.record('forward', phase_start_time)

                        phase_start_time = timer.start()
                        next_state, next_speed, next_maneuver, \
                            reward, done, info = env.step(action)
                        timer.record('env_step', phase_start_time)

                        self.rewards.append(reward)
                        self.reward_components.append(
                            info.get('reward_components', {}))
                        episode_total_reward += reward

                        if self.config.log_steps:
                            training_logger.log_step(
                                global_t=global_t,
                                local_t=self.total_steps,
                                global_episode=current_episode,
                                step_in_ep=episode_step_count,
                                action=action,
                                value=value_f,
                                entropy=entropy_f,
                                reward=reward,
                                done=done,
                                speed_kmh=info.get('speed_kmh'),
                                route_dist=info.get('route_distance'),
                                goal_dist=info.get('distance_from_goal'),
                                maneuver=maneuver,
                                reward_components=info.get(
                                    'reward_components'),
                            )

                        if not self.config.testing and \
                                (steps_since_last_update >=
                                 self.config.rollout_length or done):
                            next_state_tensor = torch.from_numpy(
                                next_state).float().unsqueeze(0).to(
                                    self.device)
                            next_speed_tensor = torch.tensor(
                                [[next_speed]], dtype=torch.float32,
                                device=self.device)
                            next_maneuver_tensor = torch.tensor(
                                [next_maneuver], device=self.device)

                            self.compute_and_apply_gradients(
                                next_state_tensor, done, next_speed_tensor,
                                next_maneuver_tensor, training_logger,
                                timer, global_t)

                            if not done and self.local_updates % \
                                    self.config.sync_every_n_updates == 0:
                                phase_start_time = timer.start()
                                self.sync_with_global()
                                timer.record('sync', phase_start_time)
                            if self.config.diag_log_interval > 0:
                                update_number = \
                                    self.global_network.total_updates.value
                                timing_window = \
                                    self.config.diag_log_interval
                                should_log_timing = (
                                    update_number > last_diag_update and
                                    update_number > 0 and
                                    update_number % timing_window == 0)
                                if should_log_timing:
                                    stats = timer.get_stats()
                                    if stats:
                                        training_logger.log_timing(
                                            stats,
                                            window_updates=timing_window)
                                    timer.log_and_reset(
                                        'W{}'.format(self.worker_id))
                                    last_diag_wall = time.time()
                                    last_diag_update = update_number
                            steps_since_last_update = 0

                        state = next_state
                        speed = next_speed
                        maneuver = next_maneuver

                        if self.config.save_frequency > 0 and \
                                global_t % self.config.save_frequency == 0:
                            phase_start_time = timer.start()
                            if self._save_checkpoint(
                                    global_t, training_logger):
                                timer.record('checkpoint_save',
                                             phase_start_time)

                    # end of episode
                    if self.episode_count % self.config.gc_interval == 0:
                        gc.collect()
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()

                    duration = time.time() - ep_start
                    self._log_episode(
                        training_logger, env, current_episode,
                        self.global_network.global_step.value,
                        episode_total_reward, episode_step_count, duration)

                    if self.config.diag_log_wall_s > 0 and \
                            time.time() - last_diag_wall >= \
                            self.config.diag_log_wall_s:
                        stats = timer.get_stats()
                        if stats:
                            training_logger.log_timing(
                                stats,
                                window_updates=
                                self.global_network.total_updates.value)
                        timer.log_and_reset('W{}'.format(self.worker_id))
                        last_diag_wall = time.time()

                except RuntimeError as e:
                    msg = str(e)
                    # Two different timeouts surface as RuntimeError here:
                    #   (a) "time-out of <N>ms while waiting for the
                    #       simulator" - CARLA server is unreachable;
                    #       we need a full reconnect + new CarlaEnv.
                    #   (b) "time-out waiting for camera image" -
                    #       image_queue.get timed out, server is fine;
                    #       a world.tick() or two is usually enough,
                    #       so we just reset the episode, no reconnect.
                    if 'waiting for the simulator' in msg:
                        print('[W{}] CARLA server timeout: reconnecting'
                              .format(self.worker_id), flush=True)
                        training_logger.log_crash_recovery(
                            global_t=self.global_network.global_step.value,
                            error=e)
                        env.reconnect()
                        self._clear_rollout_buffers()
                    elif 'camera image' in msg or 'time-out' in msg:
                        print('[W{}] camera queue timeout: '
                              'skipping episode'.format(self.worker_id),
                              flush=True)
                        training_logger.log_event(
                            'camera_timeout',
                            global_t=self.global_network.global_step.value,
                            error=msg)
                        self._clear_rollout_buffers()
                    else:
                        raise

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print('[W{}] crashed: {}'.format(self.worker_id, e), flush=True)
            training_logger.log_event(
                'worker_crash',
                global_t=self.global_network.global_step.value,
                error=str(e))
            raise
        finally:
            try:
                worker_monitor.stop()
            except Exception:
                pass
            try:
                training_logger.close()
            except Exception:
                pass
            try:
                if env and env.env:
                    env.env.world = None
                    env.env.client = None
            except Exception:
                pass
            print('[W{}] terminated'.format(self.worker_id), flush=True)
