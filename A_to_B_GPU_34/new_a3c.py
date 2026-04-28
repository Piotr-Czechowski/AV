"""A3C core: SharedAdam, GlobalNetwork, A3CWorker.

GPU->GPU parameter-server design preserved from the previous
a3c_improved.py, with every async-rl improvement wired in:

  * global step counter drives stopping criterion and LR schedule
  * linear LR decay to zero
  * NaN-gradient detection skips the update and re-syncs the worker
  * NaN-safe checkpoint writes
  * per-worker checkpoints (in addition to global) for rollback
  * per-phase TimingAccumulator instrumentation
  * structured JSONL logging (episodes + updates + timing + events + steps)
  * per-episode action-distribution + CARLA scalar stats
  * configurable entropy coefficient, value-loss coefficient, weight decay,
    reward scale, gradient clipping, sync frequency

Nothing in this file reads from a global config module – every knob is
passed through `cfg` (a plain namespace) so the worker is fully
parametrised from the CLI entry point.
"""

import os
import time
import csv
import gc
import glob
from collections import namedtuple
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions.categorical import Categorical

from nets.a2c import DiscreteActor as DeepDiscreteActor
from nets.a2c import Critic as DeepCritic

from new_timing_utils import TimingAccumulator
from new_training_logger import TrainingLogger
from new_system_monitor import WorkerMonitor
from new_carla_wrapper import CarlaA3CWrapper


Transition = namedtuple(
    "Transition", ["value_s", "log_prob_a", "entropy", "action"])


# ---------------------------------------------------------------------------
# SharedAdam – state lives on the global device (cuda:0)
# ---------------------------------------------------------------------------

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step']       = torch.zeros(1, device=p.device)
                state['exp_avg']    = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr


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


def transfer_grads(local_model, global_model, global_device):
    """Copy gradients from worker GPU to global GPU (non_blocking=False)."""
    for local_p, global_p in zip(local_model.parameters(),
                                 global_model.parameters()):
        if local_p.grad is not None:
            global_p.grad = local_p.grad.to(global_device, non_blocking=False)


def global_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad * p.grad).item())
    return total ** 0.5


# ---------------------------------------------------------------------------
# Global network (parameter server on cuda:0)
# ---------------------------------------------------------------------------

class GlobalNetwork:
    def __init__(self, cfg, state_shape, action_shape, critic_shape):
        self.cfg = cfg
        self.device = torch.device(cfg.global_device)

        self.actor = DeepDiscreteActor(state_shape, action_shape,
                                       self.device).to(self.device)
        self.critic = DeepCritic(state_shape, critic_shape,
                                 self.device).to(self.device)

        self.actor_optimizer = SharedAdam(
            self.actor.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.critic_optimizer = SharedAdam(
            self.critic.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.update_lock = mp.Lock()
        self.stats_lock = mp.Lock()
        self.save_lock = mp.Lock()

        self.global_step = mp.Value('l', 0)
        self.global_episode = mp.Value('l', 0)
        self.total_updates = mp.Value('l', 0)
        self.best_reward = mp.Value('d', -float('inf'))
        self.global_mean_reward = mp.Value('d', 0.0)
        self.worker_mean_rewards = mp.Array('d', [0.0] * cfg.num_workers)

    # -- counters --

    def inc_step(self, n=1):
        with self.global_step.get_lock():
            self.global_step.value += n
            return self.global_step.value

    def inc_episode(self):
        with self.stats_lock:
            self.global_episode.value += 1
            return self.global_episode.value

    def inc_updates(self):
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
            active = [r for r in self.worker_mean_rewards[:] if r != 0]
            if active:
                self.global_mean_reward.value = sum(active) / len(active)
            return self.global_mean_reward.value, is_new_best

    # -- LR scheduling (linear decay from cfg.lr to 0) --

    def set_lr_for_step(self, global_t):
        if self.cfg.steps <= 0:
            return self.cfg.lr
        frac = max(0.0, (self.cfg.steps - global_t - 1) / self.cfg.steps)
        lr = frac * self.cfg.lr
        self.actor_optimizer.set_lr(lr)
        self.critic_optimizer.set_lr(lr)
        return lr

    # -- save / load --

    def save(self, path, global_t=None, nan_safe=True):
        if nan_safe and (has_nan_params(self.actor)
                         or has_nan_params(self.critic)):
            print('[SAVE] refusing to save – NaN in global params',
                  flush=True)
            return False
        with self.save_lock:
            state = {
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'global_step': global_t if global_t is not None
                               else self.global_step.value,
                'global_episode': self.global_episode.value,
                'total_updates': self.total_updates.value,
                'best_reward': self.best_reward.value,
                'global_mean_reward': self.global_mean_reward.value,
            }
            torch.save(state, path)
            return True

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        with self.save_lock:
            self.actor.load_state_dict(state['actor'])
            self.critic.load_state_dict(state['critic'])
            if 'actor_optimizer' in state:
                try:
                    self.actor_optimizer.load_state_dict(
                        state['actor_optimizer'])
                    self.critic_optimizer.load_state_dict(
                        state['critic_optimizer'])
                except Exception as e:
                    print('[LOAD] optimizer restore failed: {}'.format(e),
                          flush=True)
            self.global_step.value = int(state.get('global_step', 0))
            self.global_episode.value = int(state.get('global_episode', 0))
            self.total_updates.value = int(state.get('total_updates', 0))
            self.best_reward.value = float(
                state.get('best_reward', -float('inf')))
            self.global_mean_reward.value = float(
                state.get('global_mean_reward', 0.0))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class A3CWorker(mp.Process):
    def __init__(self, worker_id, global_network, cfg, port, device,
                 outdir, shutdown_event, log_queue=None, run_id=''):
        super().__init__()
        self.worker_id = worker_id
        self.global_network = global_network
        self.cfg = cfg
        self.port = port
        self.device_str = device
        self.outdir = outdir
        self.shutdown_event = shutdown_event
        self.log_queue = log_queue
        self.run_id = run_id

        self.device = None
        self.actor = None
        self.critic = None
        self.trajectory = []
        self.rewards = []
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
        state_shape = [self.cfg.res, self.cfg.res, 3]
        critic_shape = 1
        action_shape = self.cfg.n_actions

        self.actor = DeepDiscreteActor(state_shape, action_shape,
                                       self.device).to(self.device)
        self.critic = DeepCritic(state_shape, critic_shape,
                                 self.device).to(self.device)
        self.sync_with_global()
        self._initialized = True

    def sync_with_global(self):
        for gp, lp in zip(self.global_network.actor.parameters(),
                          self.actor.parameters()):
            lp.data.copy_(gp.data.to(self.device, non_blocking=True))
        for gp, lp in zip(self.global_network.critic.parameters(),
                          self.critic.parameters()):
            lp.data.copy_(gp.data.to(self.device, non_blocking=True))
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    # -- action --

    def get_action(self, obs, speed, maneuver, testing=False):
        logits = self.actor(obs, speed, maneuver)
        value = self.critic(obs, speed, maneuver)
        dist = Categorical(logits=logits)

        if testing:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
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
                                    tlogger, timer, global_t):
        if not self.trajectory:
            return None

        t0 = timer.start()
        with torch.no_grad():
            if done:
                R = torch.zeros(1, 1, device=self.device)
            else:
                R = self.critic(final_state_tensor, final_speed_tensor,
                                maneuver_tensor)

        returns = []
        rewards_scaled = []
        scale = self.cfg.reward_scale
        for reward in reversed(self.rewards):
            r = reward / scale if scale and scale != 0.0 else reward
            rewards_scaled.insert(0, r)
            R = r + self.cfg.gamma * R
            returns.insert(0, R)

        batch = Transition(*zip(*self.trajectory))
        values = batch.value_s
        log_probs = batch.log_prob_a
        entropies = batch.entropy

        policy_loss = torch.zeros(1, device=self.device)
        value_loss = torch.zeros(1, device=self.device)

        adv_values = []
        val_values = []

        for G_t, V_s, log_prob in zip(returns, values, log_probs):
            advantage = G_t - V_s.detach()
            policy_loss = policy_loss - log_prob * advantage
            value_loss = value_loss + self.cfg.value_loss_coef \
                * F.smooth_l1_loss(V_s, G_t)
            adv_values.append(float(advantage.detach().cpu().item()))
            val_values.append(float(V_s.detach().cpu().item()))

        entropy_mean = torch.stack(list(entropies)).mean()
        total_loss = policy_loss + value_loss \
            - self.cfg.entropy_coef * entropy_mean
        timer.record('loss_compute', t0)

        t0 = timer.start()
        self.actor.zero_grad()
        self.critic.zero_grad()
        total_loss.backward()
        timer.record('backward', t0)

        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                           self.cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                           self.cfg.max_grad_norm)

        nan_layers = has_nan_grads(self.actor) + has_nan_grads(self.critic)
        if nan_layers:
            tlogger.log_nan(global_t=global_t,
                            nan_count=len(nan_layers),
                            nan_layers=nan_layers)
            print('[NaN] W{} NaN grad at step {} in {} layers – skipping '
                  'update, re-sync from global'.format(
                      self.worker_id, global_t, len(nan_layers)), flush=True)
            self.sync_with_global()
            self.trajectory.clear()
            self.rewards.clear()
            return None

        grad_n_actor = global_grad_norm(self.actor)
        grad_n_critic = global_grad_norm(self.critic)
        grad_n_total = (grad_n_actor ** 2 + grad_n_critic ** 2) ** 0.5

        t0 = timer.start()
        with self.global_network.update_lock:
            lr_now = self.global_network.set_lr_for_step(global_t)
            transfer_grads(self.actor, self.global_network.actor,
                           self.global_network.device)
            transfer_grads(self.critic, self.global_network.critic,
                           self.global_network.device)
            self.global_network.actor_optimizer.step()
            self.global_network.critic_optimizer.step()
            self.global_network.actor_optimizer.zero_grad()
            self.global_network.critic_optimizer.zero_grad()
        timer.record('optim_update', t0)

        update_n = self.global_network.inc_updates()
        self.local_updates += 1

        tlogger.log_update(
            update_count=update_n,
            global_t=global_t,
            traj_len=len(self.trajectory),
            is_terminal=bool(done),
            pi_loss=float(policy_loss.detach().cpu().item()),
            v_loss=float(value_loss.detach().cpu().item()),
            total_loss=float(total_loss.detach().cpu().item()),
            grad_norm=grad_n_total,
            lr=lr_now,
            advantages=adv_values,
            values=val_values,
            entropies=[float(e.detach().cpu().item()) for e in entropies],
            rewards=rewards_scaled,
        )

        self.trajectory.clear()
        self.rewards.clear()
        return update_n

    # -- checkpointing (runs from the worker on step boundaries) --

    def _save_checkpoint(self, global_t, tlogger):
        wdir = os.path.join(self.outdir, 'checkpoints',
                            'worker_{}'.format(self.worker_id))
        os.makedirs(wdir, exist_ok=True)
        worker_ckpt = os.path.join(wdir, 'checkpoint.pth')
        ok = self.global_network.save(worker_ckpt, global_t=global_t)
        if not ok:
            return

        global_ckpt = os.path.join(self.outdir, 'checkpoint.pth')
        self.global_network.save(global_ckpt, global_t=global_t)

        with open(os.path.join(wdir, 'checkpoint_step.txt'), 'w') as f:
            f.write(str(global_t))
        with open(os.path.join(self.outdir, 'checkpoint_step.txt'), 'w') as f:
            f.write(str(global_t))

        resume_state = {
            'global_step': global_t,
            'training_args': vars(self.cfg) if not isinstance(self.cfg, dict)
                             else self.cfg,
            'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
        }
        import json
        state_path = os.path.join(self.outdir, 'resume_state.json')
        tmp_path = state_path + '.tmp'
        with open(tmp_path, 'w') as sf:
            json.dump(resume_state, sf, indent=2, default=str)
        os.replace(tmp_path, state_path)

        tlogger.log_checkpoint(path=global_ckpt, global_t=global_t)

    # -- episode logging --

    def _log_episode(self, tlogger, env, global_episode, global_t,
                     ep_reward, ep_steps, duration_s):
        self.episode_rewards_history.append(ep_reward)
        window = min(100, len(self.episode_rewards_history))
        self.mean_reward = float(
            np.mean(self.episode_rewards_history[-window:]))
        global_mean, is_new_best = self.global_network.update_stats(
            self.worker_id, self.mean_reward, ep_reward)

        action_counts = env._action_counts.tolist() \
            if hasattr(env, '_action_counts') else None

        tlogger.log_episode(
            global_episode=global_episode,
            global_t=global_t,
            total_reward=ep_reward,
            steps=ep_steps,
            duration_s=duration_s,
            max_speed_kmh=getattr(env, '_ep_max_speed', None),
            min_route_dist=getattr(env, '_ep_min_route_dist', None),
            goal_dist=getattr(env, '_ep_goal_dist', None),
            reached_goal=getattr(env, '_ep_reached_goal', False),
            action_counts=action_counts,
            collisions=len(env.env.collision_history_list)
                       if hasattr(env, 'env') and
                       hasattr(env.env, 'collision_history_list') else None,
            port=env.port,
            local_mean_reward=self.mean_reward,
            global_mean_reward=global_mean,
            is_new_best=is_new_best,
        )

        if is_new_best:
            print('[BEST] W{} new best reward {:.2f} at Ep{}'.format(
                self.worker_id, ep_reward, global_episode), flush=True)

        if self.log_queue is not None:
            try:
                rec = {
                    'worker_id': self.worker_id,
                    'episode': global_episode,
                    'global_step': global_t,
                    'reward': ep_reward,
                    'local_mean_reward': self.mean_reward,
                    'global_mean_reward': global_mean,
                    'episode_length': ep_steps,
                    'total_steps': self.total_steps,
                }
                if hasattr(env, '_ep_max_speed'):
                    rec['max_speed_kmh'] = env._ep_max_speed
                if hasattr(env, '_ep_min_route_dist'):
                    rec['min_route_dist'] = env._ep_min_route_dist
                if hasattr(env, '_ep_goal_dist'):
                    rec['distance_from_target'] = env._ep_goal_dist
                if action_counts is not None:
                    total_a = sum(action_counts)
                    if total_a > 0:
                        rec['most_chosen_action_pct'] = max(action_counts) \
                            / total_a
                    for i, c in enumerate(action_counts):
                        rec['action_{}'.format(i)] = c
                self.log_queue.put_nowait(rec)
            except Exception:
                pass

    # -- main loop --

    def run(self):
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            import sys
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

        print('[W{}] starting on {} port {}'.format(
            self.worker_id, self.device_str, self.port), flush=True)

        self._init_networks()

        tlogger = TrainingLogger(
            self.outdir, self.worker_id,
            log_steps=self.cfg.log_steps,
            log_update_arrays=self.cfg.log_update_arrays)

        worker_mon = WorkerMonitor(self.outdir, self.worker_id,
                                   interval=self.cfg.monitor_interval)
        worker_mon.start()

        timer = TimingAccumulator()

        # restore per-worker mean reward from logs if present
        try:
            prev = self.global_network.worker_mean_rewards[self.worker_id]
            if prev != 0.0:
                self.mean_reward = prev
                self.episode_rewards_history = [prev] * 100
        except Exception:
            pass

        last_save_boundary = (
            self.global_network.global_step.value // self.cfg.save_frequency
            if self.cfg.save_frequency > 0 else 0)

        env = CarlaA3CWrapper(
            port=self.port, scenario=self.cfg.scenario,
            camera=self.cfg.camera, resX=self.cfg.res, resY=self.cfg.res,
            action_space=self.cfg.action_type,
            mp_density=self.cfg.mp_density,
            max_connect_retries=self.cfg.max_connect_retries,
            connect_retry_wait=self.cfg.connect_retry_wait,
            reconnect_wait=self.cfg.carla_timeout_wait,
            save_episodes=self.cfg.save_episodes,
            save_episode_interval=self.cfg.save_episode_interval,
            run_id=self.run_id, n_actions=self.cfg.n_actions,
            outdir=self.outdir,
        )

        try:
            while not self.shutdown_event.is_set():
                try:
                    self.trajectory.clear()
                    self.rewards.clear()
                    self.episode_count += 1
                    current_episode = self.global_network.inc_episode()
                    env.global_episode = current_episode

                    self.sync_with_global()

                    t0 = timer.start()
                    state, speed, maneuver = env.reset()
                    timer.record('env_reset', t0)

                    done = False
                    ep_reward = 0.0
                    ep_steps = 0
                    t_since_update = 0
                    ep_start = time.time()

                    last_state = state
                    last_speed = speed
                    last_maneuver = maneuver

                    while not done and not self.shutdown_event.is_set():
                        global_t = self.global_network.inc_step(1)
                        if self.cfg.steps > 0 and global_t > self.cfg.steps:
                            self.shutdown_event.set()
                            break

                        ep_steps += 1
                        t_since_update += 1
                        self.total_steps += 1

                        state_t = torch.from_numpy(state).float() \
                            .unsqueeze(0).to(self.device)
                        speed_t = torch.tensor([[speed]], dtype=torch.float32,
                                               device=self.device)
                        maneuver_t = torch.tensor([maneuver],
                                                  device=self.device)

                        t0 = timer.start()
                        action, value_f, entropy_f = self.get_action(
                            state_t, speed_t, maneuver_t,
                            testing=self.cfg.testing)
                        timer.record('forward', t0)

                        t0 = timer.start()
                        next_state, next_speed, next_maneuver, \
                            reward, done, info = env.step(action)
                        timer.record('env_step', t0)

                        self.rewards.append(reward)
                        ep_reward += reward

                        if self.cfg.log_steps:
                            tlogger.log_step(
                                global_t=global_t,
                                local_t=self.total_steps,
                                global_episode=current_episode,
                                step_in_ep=ep_steps,
                                action=action,
                                value=value_f,
                                entropy=entropy_f,
                                reward=reward,
                                done=done,
                                speed_kmh=info.get('speed_kmh'),
                                route_dist=info.get('route_distance'),
                                goal_dist=info.get('distance_from_goal'),
                                maneuver=maneuver,
                            )

                        if not self.cfg.testing and \
                                (t_since_update >= self.cfg.t_max or done):
                            next_state_t = torch.from_numpy(next_state) \
                                .float().unsqueeze(0).to(self.device)
                            next_speed_t = torch.tensor(
                                [[next_speed]], dtype=torch.float32,
                                device=self.device)
                            next_maneuver_t = torch.tensor(
                                [next_maneuver], device=self.device)

                            self.compute_and_apply_gradients(
                                next_state_t, done, next_speed_t,
                                next_maneuver_t, tlogger, timer, global_t)

                            if not done and self.local_updates % \
                                    self.cfg.sync_every_n_updates == 0:
                                self.sync_with_global()
                            t_since_update = 0

                        state = next_state
                        speed = next_speed
                        maneuver = next_maneuver
                        last_state, last_speed, last_maneuver = \
                            state, speed, maneuver

                        # step-based checkpoint
                        if self.cfg.save_frequency > 0:
                            cur_boundary = global_t // self.cfg.save_frequency
                            if cur_boundary > last_save_boundary:
                                last_save_boundary = cur_boundary
                                self._save_checkpoint(global_t, tlogger)

                    # end of episode
                    if self.episode_count % self.cfg.gc_interval == 0:
                        gc.collect()
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()

                    duration = time.time() - ep_start
                    self._log_episode(
                        tlogger, env, current_episode,
                        self.global_network.global_step.value,
                        ep_reward, ep_steps, duration)

                    if self.cfg.diag_log_interval > 0 and \
                            self.episode_count % self.cfg.diag_log_interval == 0:
                        stats = timer.get_stats()
                        if stats:
                            tlogger.log_timing(
                                stats,
                                window_updates=self.cfg.diag_log_interval)
                        timer.log_and_reset('W{}'.format(self.worker_id))

                except RuntimeError as e:
                    msg = str(e)
                    # Two different timeouts surface as RuntimeError here:
                    #   (a) "time-out of <N>ms while waiting for the
                    #       simulator" — CARLA server is unreachable;
                    #       we need a full reconnect + new CarlaEnv.
                    #   (b) "time-out waiting for camera image" —
                    #       image_queue.get timed out, server is fine;
                    #       a world.tick() or two is usually enough,
                    #       so we just reset the episode, no reconnect.
                    if 'waiting for the simulator' in msg:
                        print('[W{}] CARLA server timeout: reconnecting'
                              .format(self.worker_id), flush=True)
                        tlogger.log_crash_recovery(
                            global_t=self.global_network.global_step.value,
                            error=e)
                        env.reconnect()
                        self.trajectory.clear()
                        self.rewards.clear()
                    elif 'camera image' in msg or 'time-out' in msg:
                        print('[W{}] camera queue timeout: '
                              'skipping episode'.format(self.worker_id),
                              flush=True)
                        tlogger.log_event(
                            'camera_timeout',
                            global_t=self.global_network.global_step.value,
                            error=msg)
                        self.trajectory.clear()
                        self.rewards.clear()
                    else:
                        raise

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print('[W{}] crashed: {}'.format(self.worker_id, e), flush=True)
            tlogger.log_event('worker_crash',
                              global_t=self.global_network.global_step.value,
                              error=str(e))
            raise
        finally:
            try:
                worker_mon.stop()
            except Exception:
                pass
            try:
                tlogger.close()
            except Exception:
                pass
            try:
                if env and env.env:
                    env.env.world = None
                    env.env.client = None
            except Exception:
                pass
            print('[W{}] terminated'.format(self.worker_id), flush=True)
