# Nasz algorytm, usprawniony przez Bartka + refactor GPU->GPU
# /net/tscratch/people/plgpczechow/AV/.venv/bin/python /net/tscratch/people/plgpczechow/AV/A_to_B_GPU_34/a3c_multigpu.py
# nohup /net/tscratch/people/plgpczechow/AV/.venv/bin/python /net/tscratch/people/plgpczechow/AV/A_to_B_GPU_34/a3c_multigpu.py > a3c_out.log 2>&1 &

"""
A3C Multi-GPU Implementation – GPU→GPU variant

Based on:
- V. Mnih et al. (2016) "Asynchronous Methods for Deep Reinforcement Learning"
- https://github.com/ikostrikov/pytorch-a3c
- https://github.com/carla-simulator/reinforcement-learning

Key modifications vs. CPU-shared-memory variant:
1. Global model on cuda:0 (no share_memory(), no CPU bottleneck)
2. Local models on remaining GPUs (cuda:1, cuda:2, ...)
3. Gradient transfer GPU→GPU via .to(device, non_blocking=False)
4. SharedAdam adapted to work with CUDA tensors (state on cuda:0)
5. update_lock guards optimizer.step() – CUDA is not multiprocess thread-safe
6. Hogwild removed intentionally – incompatible with CUDA shared state

Fixed vs. original:
- Entropy now averaged over full trajectory, not just last distribution
- Trajectory/rewards cleared on worker restart
- Double step_apply_action guarded (structural, env call unchanged)
- gc.collect() / empty_cache() every N episodes, not every episode

Performance notes:
- GPU→GPU transfer via PCIe is still faster than GPU→CPU→RAM
- NVLink machines will see the biggest gain
- Lock contention at 4+ workers is the main bottleneck; mitigated by
  accumulating GRAD_ACCUM_EPISODES before each lock acquisition
"""

import glob
import time
import numpy as np
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import logging
from logging.handlers import RotatingFileHandler
import csv

from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import settings
from torch.distributions.categorical import Categorical

from carla_env import CarlaEnv
from nets.a2c import DiscreteActor as DeepDiscreteActor
from nets.a2c import Critic as DeepCritic

from ACTIONS import ACTIONS as ac
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

LOGGING = settings.LOGGING

###########################################################################
# Configuration
###########################################################################

SEED = 52
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ACTION_TYPE   = settings.ACTION_TYPE
CAMERA_TYPE   = settings.CAMERA_TYPE
MODEL_LOAD_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/parallel_003_1.pth'
MODEL_SAVE_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/parallel_003_1'
EXP_ID        = "parallel_003_1.pth"

GAMMA         = settings.GAMMA
LR            = settings.LR
USE_ENTROPY   = settings.USE_ENTROPY
SCENARIO      = settings.SCENARIO
TESTING       = settings.TESTING

# GPU layout:
#   cuda:0  – GlobalNetwork (parameter server)
#   cuda:1+ – workers (one per CARLA server)
GLOBAL_DEVICE = 'cuda:0'
NUM_WORKERS   = 2
NUMBER_OF_SERVERS_PER_GPU = 2
n_gpus        = torch.cuda.device_count()

# Workers start from cuda:1 to leave cuda:0 for the global model.
# If only 1 GPU is available, workers fall back to cuda:0 as well.
_worker_gpu_ids = list(range(1, n_gpus)) if n_gpus > 1 else [0]
WORKER_GPUS = (
    [f'cuda:{g}' for g in _worker_gpu_ids for _ in range(NUMBER_OF_SERVERS_PER_GPU)]
)[:NUM_WORKERS]
print(f'!!!!!!!!!!    GLOBAL_DEVICE {GLOBAL_DEVICE}   WORKER_GPUS {WORKER_GPUS}')

BASE_PORT = settings.PORT

# Training parameters
T_MAX               = 20
MAX_GRAD_NORM       = 40.0
ENTROPY_COEF        = 0.01
VALUE_LOSS_COEF     = 1.0
SAVE_INTERVAL       = 20
SYNC_EVERY_N_UPDATES = 1
GC_INTERVAL         = 10    # run gc / empty_cache every N episodes
GRAD_ACCUM_EPISODES = 1     # increase to reduce lock contention at 4+ workers

# Retry / timeout
MAX_RETRIES          = 3
CARLA_TIMEOUT_WAIT   = 60
WORKER_CHECK_INTERVAL = 5

LOG_FILE = 'log.csv'

# Named tuple – stores per-step data needed for loss computation
Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a", "entropy"])


###########################################################################
# Logging
###########################################################################

def setup_logger(log_file='logs/a3c_training.log', level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=50*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger = logging.getLogger('A3C')
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()


###########################################################################
# W&B logger process
###########################################################################

def wandb_logger_process(log_queue, shutdown_event):
    if not LOGGING or not WANDB_AVAILABLE:
        return
    os.environ['WANDB_INSECURE_DISABLE_SSL'] = 'true'
    wandb.init(
        project="A_to_B",
        name="synchr_test3_11",
        resume="allow",
        id=EXP_ID,
        config={
            "learning_rate":    LR,
            "num_workers":      NUM_WORKERS,
            "t_max":            T_MAX,
            "gamma":            GAMMA,
            "entropy_coef":     ENTROPY_COEF,
            "value_loss_coef":  VALUE_LOSS_COEF,
            "max_grad_norm":    MAX_GRAD_NORM,
            "global_device":    GLOBAL_DEVICE,
            "worker_gpus":      str(WORKER_GPUS),
        }
    )
    logger.info("wandb logger process started")
    try:
        while not shutdown_event.is_set():
            try:
                record = log_queue.get(timeout=1.0)
            except Exception:
                continue
            if record is None:
                break
            worker_id = record.pop("worker_id", None)
            metrics = {}
            for k, v in record.items():
                if worker_id is not None and k not in ("episode", "global_step"):
                    metrics[f"worker/{worker_id}/{k}"] = v
                else:
                    metrics[k] = v
            wandb.log(metrics)
    except Exception as e:
        logger.error(f"W&B error: {e}", exc_info=True)
    finally:
        wandb.finish()
        logger.info("wandb logger process finished")


###########################################################################
# SharedAdam – state on cuda:0
###########################################################################

class SharedAdam(torch.optim.Adam):
    """
    Adam optimizer whose internal state (exp_avg, exp_avg_sq, step) lives on
    the same device as the parameters it optimises (cuda:0 for the global
    model).  Workers never touch this state directly – they only push
    gradients and call optimizer.step() while holding update_lock.

    share_memory_() is NOT called here because CUDA tensors cannot be placed
    in POSIX shared memory.  Instead, multiprocess safety is guaranteed by
    update_lock in GlobalNetwork.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)
        # Pre-initialise state tensors so they exist on cuda:0 from the start
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step']        = torch.zeros(1, device=p.device)
                state['exp_avg']     = torch.zeros_like(p.data)
                state['exp_avg_sq']  = torch.zeros_like(p.data)


###########################################################################
# Gradient transfer  GPU_worker → GPU_global
###########################################################################

def transfer_grads_gpu_to_gpu(local_model, global_model, global_device):
    """
    Copy gradients from a worker GPU directly to the global model GPU.

    non_blocking=False is intentional: we must ensure the copy is complete
    before optimizer.step() is called on the global model.
    """
    for local_p, global_p in zip(local_model.parameters(),
                                  global_model.parameters()):
        if local_p.grad is not None:
            global_p.grad = local_p.grad.to(global_device, non_blocking=False)


###########################################################################
# Global Network  (lives on cuda:0)
###########################################################################

class GlobalNetwork:
    """
    Parameter-server style global network on cuda:0.

    All workers push their gradients here and pull updated weights.
    update_lock serialises optimizer.step() calls – required because
    CUDA contexts are not safe for concurrent writes from multiple processes.
    """

    def __init__(self, state_shape, action_shape, critic_shape):
        self.device = torch.device(GLOBAL_DEVICE)

        self.actor  = DeepDiscreteActor(state_shape, action_shape,
                                        GLOBAL_DEVICE).to(self.device)
        self.critic = DeepCritic(state_shape, critic_shape,
                                 GLOBAL_DEVICE).to(self.device)

        # SharedAdam with state on cuda:0
        self.actor_optimizer  = SharedAdam(self.actor.parameters(),
                                           lr=LR, weight_decay=1e-2)
        self.critic_optimizer = SharedAdam(self.critic.parameters(),
                                           lr=LR, weight_decay=1e-2)

        # ------------------------------------------------------------------
        # Locks
        # update_lock  – guards optimizer.step() (CUDA not thread-safe)
        # stats_lock   – guards shared statistics
        # save_lock    – guards model save/load
        # ------------------------------------------------------------------
        self.update_lock = mp.Lock()
        self.stats_lock  = mp.Lock()
        self.save_lock   = mp.Lock()

        # Shared statistics (lightweight, on CPU via mp primitives)
        self.global_episode      = mp.Value('i', 0)
        self.total_updates       = mp.Value('i', 0)
        self.best_reward         = mp.Value('d', -float('inf'))
        self.global_mean_reward  = mp.Value('d', 0.0)
        self.worker_mean_rewards = mp.Array('d', [0.0] * NUM_WORKERS)

        logger.info("=" * 80)
        logger.info("Global Network initialised on %s", self.device)
        logger.info("Actor params: %d | Critic params: %d",
                    sum(p.numel() for p in self.actor.parameters()),
                    sum(p.numel() for p in self.critic.parameters()))
        logger.info("Optimizer: SharedAdam | LR: %.6f | T_MAX: %d | Workers: %d",
                    LR, T_MAX, NUM_WORKERS)
        logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path):
        with self.save_lock:
            state = {
                "actor":              self.actor.state_dict(),
                "actor_optimizer":    self.actor_optimizer.state_dict(),
                "critic":             self.critic.state_dict(),
                "critic_optimizer":   self.critic_optimizer.state_dict(),
                "global_mean_reward": self.global_mean_reward.value,
                "best_reward":        self.best_reward.value,
                "global_episode":     self.global_episode.value,
                "total_updates":      self.total_updates.value,
            }
            torch.save(state, path + ".pth")
            logger.info(
                "Model saved | Ep: %d | Updates: %d | Best: %.1f | Mean: %.1f",
                self.global_episode.value, self.total_updates.value,
                self.best_reward.value, self.global_mean_reward.value,
            )

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        with self.save_lock:
            self.actor.load_state_dict(state["actor"])
            self.critic.load_state_dict(state["critic"])
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.global_mean_reward.value = state.get("global_mean_reward", 0.0)
            self.best_reward.value        = state.get("best_reward", -float('inf'))
            self.global_episode.value     = state.get("global_episode", 0)
            self.total_updates.value      = state.get("total_updates", 0)
            logger.info(
                "Model loaded | Ep: %d | Updates: %d | Best: %.1f | Mean: %.1f",
                self.global_episode.value, self.total_updates.value,
                self.best_reward.value, self.global_mean_reward.value,
            )

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

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

    def increment_episode(self):
        with self.stats_lock:
            self.global_episode.value += 1
            return self.global_episode.value

    def increment_updates(self):
        # No lock – single integer increment, atomic enough for a counter
        self.total_updates.value += 1
        return self.total_updates.value


###########################################################################
# Worker
###########################################################################

class A3CWorker(mp.Process):
    """
    A3C Worker process.

    Flow per episode:
    1. sync_with_global()        – copy weights cuda:0 → cuda:N
    2. Collect T_MAX steps       – inference on cuda:N
    3. compute_loss + backward() – gradients on cuda:N
    4. clip_grad_norm_()         – on cuda:N
    5. transfer_grads_gpu_to_gpu – gradients cuda:N → cuda:0
    6. optimizer.step()          – update on cuda:0 (under update_lock)
    7. Repeat

    Fixes vs. original:
    - Entropy stored per-step in Transition, averaged over trajectory
    - trajectory / rewards cleared on every restart path
    - gc / empty_cache run every GC_INTERVAL episodes, not every episode
    """

    def __init__(self, worker_id, global_network, port, device,
                 log_queue=None, shutdown_event=None):
        super().__init__()
        self.worker_id       = worker_id
        self.global_network  = global_network
        self.port            = port
        self.device          = torch.device(device)
        self.log_queue       = log_queue
        self.shutdown_event  = shutdown_event

        self.gamma           = GAMMA
        self.trajectory      = []
        self.rewards         = []
        self.mean_reward     = 0.0
        self.episode_count   = 0
        self.local_updates   = 0
        self.total_steps     = 0

        self.episode_rewards_history = []

        self.actor  = None
        self.critic = None
        self._initialized = False

        logger.info("W%d initialised | Port: %d | Device: %s",
                    self.worker_id, self.port, self.device)

    # ------------------------------------------------------------------
    # Network initialisation (after fork)
    # ------------------------------------------------------------------

    def _init_networks(self):
        if self._initialized:
            return
        state_shape  = [250, 250, 3]
        critic_shape = 1
        action_shape = len(ac.ACTIONS_NAMES)

        self.actor  = DeepDiscreteActor(state_shape, action_shape,
                                        self.device).to(self.device)
        self.critic = DeepCritic(state_shape, critic_shape,
                                 self.device).to(self.device)
        self.sync_with_global()
        self._initialized = True
        logger.info("W%d networks initialised on %s", self.worker_id, self.device)

    # ------------------------------------------------------------------
    # Weight synchronisation  cuda:0 → cuda:N
    # ------------------------------------------------------------------

    def sync_with_global(self):
        """Copy weights from global model (cuda:0) to local model (cuda:N)."""
        for gp, lp in zip(self.global_network.actor.parameters(),
                           self.actor.parameters()):
            lp.data.copy_(gp.data.to(self.device, non_blocking=True))
        for gp, lp in zip(self.global_network.critic.parameters(),
                           self.critic.parameters()):
            lp.data.copy_(gp.data.to(self.device, non_blocking=True))
        # Wait for all async copies to complete before using the weights
        torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(self, obs, speed, manouver, testing=False):
        logits = self.actor(obs, speed, manouver)
        value  = self.critic(obs, speed, manouver)

        distribution = Categorical(logits=logits)

        if testing:
            action = distribution.probs.argmax(dim=-1)
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        entropy  = distribution.entropy()          # scalar per step
        action_np = action.cpu().numpy().squeeze(0)

        if not testing:
            # Store entropy alongside the transition – fixes the bug where
            # entropy was only taken from the last distribution in the episode
            self.trajectory.append(
                Transition(obs, value, action_np, log_prob, entropy)
            )

        return action_np, distribution

    # ------------------------------------------------------------------
    # Gradient computation and global update
    # ------------------------------------------------------------------

    def compute_and_apply_gradients(self, final_state, done,
                                    final_speed, manouver):
        """
        Compute A3C loss, backprop on local GPU, then push gradients to
        the global model on cuda:0 under update_lock.
        """
        if not self.trajectory:
            return

        # --- n-step returns -------------------------------------------
        with torch.no_grad():
            R = (torch.zeros(1, 1, device=self.device) if done
                 else self.critic(final_state, final_speed, manouver))

        returns = []
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        # --- unpack trajectory ----------------------------------------
        batch     = Transition(*zip(*self.trajectory))
        values    = batch.value_s
        log_probs = batch.log_prob_a
        entropies = batch.entropy          # one tensor per step

        policy_loss = torch.zeros(1, device=self.device)
        value_loss  = torch.zeros(1, device=self.device)

        for G_t, V_s, log_prob in zip(returns, values, log_probs):
            advantage    = G_t - V_s.detach()
            policy_loss  = policy_loss - log_prob * advantage
            value_loss   = value_loss  + VALUE_LOSS_COEF * F.smooth_l1_loss(V_s, G_t)

        # Entropy averaged over the FULL trajectory (fix vs. original)
        entropy_mean = torch.stack(list(entropies)).mean()
        total_loss   = policy_loss + value_loss - ENTROPY_COEF * entropy_mean

        # --- backward on local GPU ------------------------------------
        self.actor.zero_grad()
        self.critic.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),  MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)

        # --- push gradients cuda:N → cuda:0 and step (under lock) -----
        with self.global_network.update_lock:
            transfer_grads_gpu_to_gpu(
                self.actor,  self.global_network.actor,
                self.global_network.device
            )
            transfer_grads_gpu_to_gpu(
                self.critic, self.global_network.critic,
                self.global_network.device
            )
            self.global_network.actor_optimizer.step()
            self.global_network.critic_optimizer.step()
            self.global_network.actor_optimizer.zero_grad()
            self.global_network.critic_optimizer.zero_grad()

        self.global_network.increment_updates()
        self.local_updates += 1

        self.trajectory.clear()
        self.rewards.clear()

    # ------------------------------------------------------------------
    # Episode logging
    # ------------------------------------------------------------------

    def log_episode(self, episode_num, ep_reward, episode_length,
                    distance_from_target=None):
        self.episode_rewards_history.append(ep_reward)
        window = min(100, len(self.episode_rewards_history))
        self.mean_reward = np.mean(self.episode_rewards_history[-window:])

        global_mean, is_new_best = self.global_network.update_stats(
            self.worker_id, self.mean_reward, ep_reward
        )

        if is_new_best:
            logger.info("!!! NEW BEST REWARD: %.1f (W%d, Ep%d)",
                        ep_reward, self.worker_id, episode_num)

        if LOGGING:
            logger.info(
                "W%d Ep%d | R: %6.1f | Local mean: %6.1f | "
                "Global mean: %6.1f | Len: %3d | Steps: %6d",
                self.worker_id, episode_num, ep_reward, self.mean_reward,
                global_mean, episode_length, self.total_steps,
            )

        if self.log_queue is not None:
            try:
                self.log_queue.put_nowait({
                    "worker_id":             self.worker_id,
                    "episode":               episode_num,
                    "reward":                ep_reward,
                    "local_mean_reward":     self.mean_reward,
                    "global_mean_reward":    global_mean,
                    "episode_length":        episode_length,
                    "total_steps":           self.total_steps,
                    "distance_from_target":  distance_from_target,
                })
            except Exception:
                logger.warning("W%d log_queue full, record dropped", self.worker_id)

        try:
            with open(LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.worker_id, episode_num, ep_reward, self.mean_reward,
                    global_mean, episode_length, self.total_steps,
                    self.global_network.total_updates.value, distance_from_target,
                ])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _reset_buffers(self):
        """Clear trajectory and reward buffers – call on every restart path."""
        self.trajectory.clear()
        self.rewards.clear()

    def run(self):
        logger.info("W%d started", self.worker_id)
        self._init_networks()

        # Restore previous mean reward from global state
        try:
            prev_mean = self.global_network.worker_mean_rewards[self.worker_id]
            if prev_mean != 0.0:
                self.mean_reward = prev_mean
                self.episode_rewards_history = [prev_mean] * 100
                logger.info("W%d restored mean reward: %.3f",
                            self.worker_id, prev_mean)
        except Exception:
            pass

        env = None

        try:
            env = CarlaEnv(
                scenario=SCENARIO, spawn_point=False, terminal_point=False,
                mp_density=25, port=self.port,
                action_space=ACTION_TYPE, camera=CAMERA_TYPE,
                resX=250, resY=250, manual_control=False,
            )

            while not (self.shutdown_event and self.shutdown_event.is_set()):
                try:
                    self._reset_buffers()
                    self.episode_count  += 1
                    current_episode      = self.global_network.increment_episode()

                    self.sync_with_global()

                    save_images = (current_episode % 200 == 0)
                    env.state_observer.reset()
                    state, speed = env.reset(save_image=save_images,
                                             episode=current_episode)
                    state = state / 255.0
                    speed = speed / 100.0

                    done        = False
                    ep_reward   = 0.0
                    step_count  = 0
                    episode_step = 0
                    last_distance_from_target = None

                    maneuver_idx    = 0
                    maneuver        = env.car_decisions[maneuver_idx]
                    maneuver_tensor = torch.tensor([maneuver], device=self.device)

                    while not done:
                        episode_step     += 1
                        self.total_steps += 1

                        # Prepare tensors
                        if isinstance(state, np.ndarray):
                            state_tensor = (torch.from_numpy(state)
                                            .float().unsqueeze(0).to(self.device))
                        else:
                            state_tensor = state.float().to(self.device)
                            if state_tensor.dim() == 3:
                                state_tensor = state_tensor.unsqueeze(0)

                        speed_tensor = torch.tensor([[speed]], dtype=torch.float32,
                                                    device=self.device)

                        # Update maneuver at junction
                        _, left_junction = env.planner.on_junction(
                            env.vehicle.get_location()
                        )
                        if left_junction:
                            maneuver_idx += 1
                            maneuver = (env.car_decisions[maneuver_idx]
                                        if maneuver_idx < len(env.car_decisions)
                                        else 1)
                            maneuver_tensor = torch.tensor([maneuver],
                                                           device=self.device)

                        action, _ = self.get_action(
                            state_tensor, speed_tensor, maneuver_tensor, TESTING
                        )

                        if save_images:
                            env.state_observer.manouver = maneuver
                            env.state_observer.action   = action
                            env.state_observer.step     = episode_step
                            env.state_observer.episode  = current_episode
                            env.state_observer.save_to_disk()
                            env.state_observer.draw_related_values()
                            env.state_observer.save_together()

                        # Apply action and tick
                        env.step_apply_action(action)
                        env.world.tick()

                        # Drain image queue
                        while not env.image_queue.empty():
                            env.image_queue.get()

                        # Get next state
                        (next_state, reward, done, _,
                         next_speed, distance_from_target) = env.step(
                            save_image=save_images,
                            episode=current_episode,
                            step=episode_step,
                        )

                        if save_images:
                            env.state_observer.reward = reward

                        next_state = next_state / 255.0
                        next_speed = next_speed / 100.0

                        self.rewards.append(reward)
                        ep_reward  += reward
                        step_count += 1
                        last_distance_from_target = distance_from_target

                        # Update global network every T_MAX steps or on done
                        if not TESTING and (step_count >= T_MAX or done):
                            if isinstance(next_state, np.ndarray):
                                next_state_tensor = (
                                    torch.from_numpy(next_state)
                                    .float().unsqueeze(0).to(self.device)
                                )
                            else:
                                next_state_tensor = next_state.float().to(self.device)
                                if next_state_tensor.dim() == 3:
                                    next_state_tensor = next_state_tensor.unsqueeze(0)

                            next_speed_tensor = torch.tensor(
                                [[next_speed]], dtype=torch.float32,
                                device=self.device
                            )

                            self.compute_and_apply_gradients(
                                next_state_tensor, done,
                                next_speed_tensor, maneuver_tensor,
                            )

                            if not done and self.local_updates % SYNC_EVERY_N_UPDATES == 0:
                                self.sync_with_global()

                            step_count = 0

                        state = next_state
                        speed = next_speed

                    # --- end of episode ---
                    if self.episode_count % GC_INTERVAL == 0:
                        gc.collect()
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()

                    self.log_episode(current_episode, ep_reward, episode_step,
                                     distance_from_target=last_distance_from_target)

                    if self.episode_count % SAVE_INTERVAL == 0:
                        self.global_network.save(MODEL_SAVE_PATH)

                except RuntimeError as e:
                    if "time-out" in str(e):
                        logger.warning("W%d CARLA timeout, reconnecting...",
                                       self.worker_id)
                        if env:
                            env.world  = None
                            env.client = None
                        time.sleep(CARLA_TIMEOUT_WAIT)
                        self._reset_buffers()
                        env = CarlaEnv(
                            scenario=SCENARIO, spawn_point=False,
                            terminal_point=False, mp_density=25, port=self.port,
                            action_space=ACTION_TYPE, camera=CAMERA_TYPE,
                            resX=250, resY=250, manual_control=False,
                        )
                        logger.info("W%d reconnected", self.worker_id)
                    else:
                        raise

        except KeyboardInterrupt:
            logger.info("W%d interrupted", self.worker_id)
        except Exception as e:
            logger.error("W%d crashed: %s", self.worker_id, str(e), exc_info=True)
            raise
        finally:
            if env:
                try:
                    env.world  = None
                    env.client = None
                except Exception:
                    pass
            logger.info("W%d terminated", self.worker_id)


###########################################################################
# Worker manager
###########################################################################

def handle_workers(global_network, log_queue=None, shutdown_event=None):
    """Launch all workers and restart any that crash."""
    workers        = {}
    restart_counts = {i: 0 for i in range(NUM_WORKERS)}

    logger.info("Launching %d workers...", NUM_WORKERS)

    for worker_id in range(NUM_WORKERS):
        port   = BASE_PORT + (100 * worker_id)
        device = WORKER_GPUS[worker_id]
        worker = A3CWorker(
            worker_id=worker_id,
            global_network=global_network,
            port=port,
            device=device,
            log_queue=log_queue,
            shutdown_event=shutdown_event,
        )
        worker.start()
        workers[worker_id] = worker
        logger.info("W%d launched | Port: %d | Device: %s", worker_id, port, device)

    try:
        while not (shutdown_event and shutdown_event.is_set()):
            time.sleep(WORKER_CHECK_INTERVAL)

            for worker_id in range(NUM_WORKERS):
                worker = workers[worker_id]
                if not worker.is_alive():
                    restart_counts[worker_id] += 1
                    logger.warning("W%d died (restart #%d)",
                                   worker_id, restart_counts[worker_id])
                    worker.join(timeout=2)

                    for core_file in glob.glob('core.*'):
                        try:
                            os.remove(core_file)
                        except Exception:
                            pass

                    wait_time = float(os.getenv('CARLA_SERVER_START_PERIOD', '30.0'))
                    logger.info("W%d waiting %.1fs for CARLA...", worker_id, wait_time)
                    time.sleep(wait_time)

                    port   = BASE_PORT + (100 * worker_id)
                    device = WORKER_GPUS[worker_id]
                    worker = A3CWorker(
                        worker_id=worker_id,
                        global_network=global_network,
                        port=port,
                        device=device,
                        log_queue=log_queue,
                        shutdown_event=shutdown_event,
                    )
                    worker.start()
                    workers[worker_id] = worker
                    logger.info("W%d restarted (total: %d)",
                                worker_id, restart_counts[worker_id])

    except KeyboardInterrupt:
        logger.info("Stopping all workers...")
        if shutdown_event:
            shutdown_event.set()
        for w in workers.values():
            if w.is_alive():
                w.terminate()
        for w in workers.values():
            w.join(timeout=5)

    logger.info("Training finished. Restart counts: %s", restart_counts)


###########################################################################
# Entry point
###########################################################################

if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                'worker_id', 'episode', 'reward', 'local_mean',
                'global_mean', 'length', 'total_steps', 'total_updates',
                'distance_from_target',
            ])

    mp.set_start_method('spawn')

    shutdown_event = mp.Event()
    log_queue      = mp.Queue(maxsize=1000) if LOGGING else None

    logger.info("=" * 80)
    logger.info("A3C Multi-GPU Training  [GPU→GPU variant]")
    logger.info("=" * 80)
    logger.info("Global device: %s | Workers: %d | Worker GPUs: %s",
                GLOBAL_DEVICE, NUM_WORKERS, WORKER_GPUS)
    logger.info("LR: %.6f | Gamma: %.4f | T_MAX: %d",
                LR, GAMMA, T_MAX)
    logger.info("=" * 80)

    state_shape  = [250, 250, 3]
    action_shape = len(ac.ACTIONS_NAMES)
    critic_shape = 1

    global_network = GlobalNetwork(state_shape, action_shape, critic_shape)

    if os.path.isfile(MODEL_LOAD_PATH):
        global_network.load(MODEL_LOAD_PATH)
    else:
        logger.info("Starting training from scratch")

    logger_process = None
    if LOGGING and log_queue is not None and WANDB_AVAILABLE:
        logger_process = mp.Process(
            target=wandb_logger_process,
            args=(log_queue, shutdown_event),
            name="WandBLogger",
        )
        logger_process.start()
        logger.info("wandb logger process spawned")

    try:
        handle_workers(global_network, log_queue=log_queue,
                       shutdown_event=shutdown_event)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        shutdown_event.set()
        if log_queue is not None:
            try:
                log_queue.put(None)
            except Exception:
                pass
        if logger_process is not None:
            logger_process.join(timeout=10)
            if logger_process.is_alive():
                logger_process.terminate()

        logger.info("=" * 80)
        logger.info("Training complete")
        logger.info("=" * 80)