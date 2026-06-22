"""
A3C Multi-GPU Implementation


Key modifications for Multi-GPU:
1. Global model on CPU (shared memory)
2. Local models on GPU (for fast inference)
3. Gradient transfer GPU->CPU by copying (not reference)
4. SharedAdam with shared state in shared memory
5. Hogwild! style updates (no locks on weight updates)

Performance optimizations:
- T_MAX = 10 (fewer GPU<->CPU transfers)
- SYNC_EVERY_N_UPDATES = 1 (sync after every update)
- SAVE_INTERVAL = 50
"""

import glob
import uuid
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
import argparse
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
FILE_LOGGING = settings.FILE_LOGGING # zapis do pliku logu


###########################################################################
# Configuration
###########################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default = 1)
    parser.add_argument("--workers-per-gpu", type=int, default = 2)
    parser.add_argument("--worker-gpu-start", type=int, default = 0)
    parser.add_argument("--start-port", type=int, default = 2000)
    parser.add_argument("--port-step", type=int, default = 100)
    parser.add_argument("--seed", type=int, default = 52)
    parser.add_argument('--scenario', type=int, nargs='+', default=[14])
    parser.add_argument("--outdir", type=str, default='.')
    parser.add_argument("--resume", action="store_true")    #Only resumes the current model, not the wandb logging nor the parameters
    parser.add_argument("--testing", action="store_true")   
    parser.add_argument("--wandb-name", type=str, default="no-name")


    return parser.parse_args()

ARGS = parse_args()


# use same seed
SEED = ARGS.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# global settings
ACTION_TYPE = settings.ACTION_TYPE
CAMERA_TYPE = settings.CAMERA_TYPE


GAMMA = settings.GAMMA
LR = settings.LR
USE_ENTROPY = settings.USE_ENTROPY
SCENARIO = ARGS.scenario
TESTING = settings.TESTING

# A3C specific settings
NUM_WORKERS = ARGS.num_workers
NUMBER_OF_SERVERS_PER_GPU = ARGS.workers_per_gpu
n_gpus = torch.cuda.device_count()
WORKER_GPUS = ([f'cuda:{g}' for g in range(n_gpus) for _ in range(NUMBER_OF_SERVERS_PER_GPU)])[:NUM_WORKERS]
print(f'!!!!!!!!!!    WORKER_GPUS {WORKER_GPUS}')
BASE_PORT = ARGS.start_port

# Training parameters
T_MAX = 10
MAX_GRAD_NORM = 40.0
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 1.0
SAVE_INTERVAL = 50
SYNC_EVERY_N_UPDATES = 1

# retrying settings
MAX_RETRIES = 3
CARLA_TIMEOUT_WAIT = 60
WORKER_CHECK_INTERVAL = 5

# Logging
LOG_FILE = 'log.csv'

Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])


def wandb_logger_process(log_queue, shutdown_event, wandb_id):
    if not LOGGING or not WANDB_AVAILABLE:
        return
    os.environ['WANDB_INSECURE_DISABLE_SSL'] = 'true'
    wandb.init(
        project="A_to_B",
        name=ARGS.wandb_name,
        resume="allow",
        id=wandb_id,
        config={
            "learning_rate": LR,
            "num_workers": NUM_WORKERS,
            "t_max": T_MAX,
            "gamma": GAMMA,
            "entropy_coef": ENTROPY_COEF,
            "value_loss_coef": VALUE_LOSS_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
        }
    )
    wandb.define_metric("*", step_metric="global_step")
    if settings.FILE_LOGGING:
        logger.info("wandb logger process started")

    try:
        while not shutdown_event.is_set():
            try:
                record = log_queue.get(timeout=1.0)
            except:
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


class SharedAdam(torch.optim.Adam):
    """
    Shared Adam optimizer for A3C.

    State (exp_avg, exp_avg_sq, step) is shared between workers,
    ensuring consistent adaptive learning rates.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                # place in shared memory
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


def setup_logger(log_file='logs/a3c_training.log', level=logging.INFO):
    """Setup unified logger for all components"""
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
# Gradient Transfer Function
###########################################################################

def transfer_grads_to_shared(local_model, shared_model):
    for lp, sp in zip(local_model.parameters(), shared_model.parameters()):
        # ensure sp.grad exists
        if sp.grad is None:
            sp.grad = torch.zeros_like(sp.data)

        if lp.grad is None:
            sp.grad.zero_()
        else:
            sp.grad.copy_(lp.grad.detach().to("cpu"))


###########################################################################
# Global network
###########################################################################

class GlobalNetwork:
    """
    Global network in shared memory (CPU).

    Architecture:
    - actor and critic on CPU with share_memory()
    - SharedAdam with shared state
    - No locks on weight updates (Hogwild!)
    """

    def __init__(self, state_shape, action_shape, critic_shape):
        self.device = torch.device('cpu')

        # networks on CPU
        self.actor = DeepDiscreteActor(state_shape, action_shape, 'cpu').to(self.device)
        self.critic = DeepCritic(state_shape, critic_shape, 'cpu').to(self.device)

        # shared memory
        self.actor.share_memory()
        self.critic.share_memory()

        # --- Preallocate shared grad buffers (do this once) ---
        for p in self.actor.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            else:
                p.grad.detach_()
                p.grad.zero_()
            p.grad.share_memory_()

        for p in self.critic.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            else:
                p.grad.detach_()
                p.grad.zero_()
            p.grad.share_memory_()
# -----------------------------------------------------

        # optimizers with shared state
        self.actor_optimizer = SharedAdam(self.actor.parameters(), lr=LR, weight_decay=1e-2)
        self.critic_optimizer = SharedAdam(self.critic.parameters(), lr=LR, weight_decay=1e-2)

        # shared statistics
        self.global_episode = mp.Value('i', 0)
        self.total_updates = mp.Value('i', 0)
        self.global_steps = mp.Value('i', 0)  # total env steps across all workers
        self.best_reward = mp.Value('d', -float('inf'))
        self.global_mean_reward = mp.Value('d', 0.0)
        self.worker_mean_rewards = mp.Array('d', [0.0] * NUM_WORKERS)

        # locks only for stats and save/load
        self.stats_lock = mp.Lock()
        self.save_lock = mp.Lock()
        if settings.FILE_LOGGING:
            logger.info("=" * 80)
            logger.info("Global Network initialized")
            logger.info("Actor params: %d | Critic params: %d",
                    sum(p.numel() for p in self.actor.parameters()),
                    sum(p.numel() for p in self.critic.parameters()))
            logger.info("Optimizer: SharedAdam | LR: %.6f | T_MAX: %d | Workers: %d",
                    LR, T_MAX, NUM_WORKERS)
            logger.info("=" * 80)

    def save(self, path):
        """Save global network and optimizer state"""
        with self.save_lock:
            state = {
                "actor": self.actor.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "global_mean_reward": self.global_mean_reward.value,
                "best_reward": self.best_reward.value,
                "global_episode": self.global_episode.value,
                "total_updates": self.total_updates.value,
                "global_steps": self.global_steps.value, 
            }
            torch.save(state, path + ".pth")
            if settings.FILE_LOGGING:
                logger.info("Model saved | Ep: %d | Updates: %d | GSteps: %d | Best: %.1f | Mean: %.1f",
                        self.global_episode.value, self.total_updates.value,
                        self.global_steps.value,
                        self.best_reward.value, self.global_mean_reward.value)

    def load(self, path):
        """Load global network and optimizer state"""
        state = torch.load(path, map_location=self.device)

        with self.save_lock:
            self.actor.load_state_dict(state["actor"])
            self.critic.load_state_dict(state["critic"])
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])

            self.global_mean_reward.value = state.get("global_mean_reward", 0.0)
            self.best_reward.value = state.get("best_reward", -float('inf'))
            self.global_episode.value = state.get("global_episode", 0)
            self.total_updates.value = state.get("total_updates", 0)
            self.global_steps.value = state.get("global_steps", 0)    # <-- ZMIANA 2

            if settings.FILE_LOGGING:
                logger.info("Model loaded | Ep: %d | Updates: %d | GSteps: %d | Best: %.1f | Mean: %.1f",
                        self.global_episode.value, self.total_updates.value,
                        self.global_steps.value,
                        self.best_reward.value, self.global_mean_reward.value)

                
    def update_stats(self, worker_id, mean_reward, episode_reward, episode_length):
        is_new_best = False
        with self.stats_lock:
            self.global_steps.value += episode_length  # ← dodane
            if episode_reward > self.best_reward.value:
                self.best_reward.value = episode_reward
                is_new_best = True
            self.worker_mean_rewards[worker_id] = mean_reward
            active = [r for r in self.worker_mean_rewards[:] if r != 0]
            if active:
                self.global_mean_reward.value = sum(active) / len(active)
            return self.global_mean_reward.value, is_new_best

    def increment_episode(self):
        """Increment global episode counter"""
        with self.stats_lock:
            self.global_episode.value += 1
            return self.global_episode.value

    def increment_updates(self):
        """Increment global updates counter"""
        self.total_updates.value += 1
        return self.total_updates.value


###########################################################################
# Worker
###########################################################################

class A3CWorker(mp.Process):
    """
    A3C Worker process running on GPU.

    Flow:
    1. sync_with_global() - copy weights CPU->GPU
    2. Collect T_MAX steps of experience (GPU)
    3. Compute loss and backward (GPU)
    4. transfer_grads_to_shared() - copy gradients GPU->CPU
    5. optimizer.step() - update weights (CPU, shared memory)
    6. Repeat
    """

    def __init__(self, worker_id, global_network, port, device,
                 log_queue=None, shutdown_event=None):
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id
        self.global_network = global_network
        self.port = port
        self.device = torch.device(device)
        self.log_queue = log_queue
        self.shutdown_event = shutdown_event

        self.gamma = GAMMA

        self.trajectory = []
        self.rewards = []
        self.mean_reward = 0.0
        self.episode_count = 0
        self.local_updates = 0
        self.total_steps = 0

        self.episode_rewards_history = []

        # networks (initialized in run() after fork)
        self.actor = None
        self.critic = None

        self._initialized = False
        if settings.FILE_LOGGING:
            logger.info("W%d initialized | Port: %d | Device: %s",
                    self.worker_id, self.port, self.device)

    def _init_networks(self):
        """Initialize local networks on GPU (after fork)"""
        if self._initialized:
            return

        state_shape = [250, 250, 3]
        critic_shape = 1
        action_shape = len(ac.ACTIONS_NAMES)

        self.actor = DeepDiscreteActor(state_shape, action_shape, self.device).to(self.device)
        self.critic = DeepCritic(state_shape, critic_shape, self.device).to(self.device)

        self.sync_with_global()
        self._initialized = True
        if settings.FILE_LOGGING:
            logger.info("W%d networks initialized on %s", self.worker_id, self.device)

    def sync_with_global(self):
        """Copy weights from global network (CPU) to local network (GPU)"""
        self.actor.load_state_dict(self.global_network.actor.state_dict())
        self.critic.load_state_dict(self.global_network.critic.state_dict())

    def get_action(self, obs, speed, manouver, testing=False):
        """Get action from policy"""
        logits = self.actor(obs, speed, manouver)
        value = self.critic(obs, speed, manouver)

        distribution = Categorical(logits=logits)
        self.action_distribution = distribution

        if testing:
            action = distribution.probs.argmax(dim=-1)
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        action_np = action.cpu().numpy().squeeze(0)

        if not testing:
            self.trajectory.append(Transition(obs, value, action_np, log_prob))

        return action_np, distribution

    def compute_and_apply_gradients(self, final_state, done, final_speed, manouver):
        """
        Compute gradients and update global model.

        Steps:
        1. Calculate n-step returns
        2. Compute policy + value + entropy loss
        3. backward() on local model (GPU)
        4. Clip gradients (GPU)
        5. Transfer gradients GPU->CPU
        6. optimizer.step() on global model (CPU)
        """
        if len(self.trajectory) == 0:
            return

        # calculate n-step returns
        returns = []
        with torch.no_grad():
            if done:
                R = torch.zeros(1, 1, device=self.device)
            else:
                R = self.critic(final_state, final_speed, manouver)

        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        # compute losses
        batch = Transition(*zip(*self.trajectory))
        values = batch.value_s
        log_probs = batch.log_prob_a

        actor_losses = []
        critic_losses = []

        for G_t, V_s, log_prob in zip(returns, values, log_probs):
            td_err = G_t - V_s
            actor_losses.append(-log_prob * td_err)
            critic_losses.append(F.smooth_l1_loss(V_s, G_t))
        
        actor_loss = torch.stack(actor_losses).mean()
        if USE_ENTROPY:
            actor_loss = actor_loss - self.action_distribution.entropy().mean()
        critic_loss = torch.stack(critic_losses).mean()
 
        # dwa osobne backward()
        self.actor.zero_grad()
        actor_loss.backward(retain_graph=True)
 
        self.critic.zero_grad()
        critic_loss.backward()

        # gradient clipping (GPU)
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)

        # transfer gradients GPU->CPU
        transfer_grads_to_shared(self.actor, self.global_network.actor)
        transfer_grads_to_shared(self.critic, self.global_network.critic)

        # optimizer step (CPU, shared memory)
        self.global_network.actor_optimizer.step()
        self.global_network.critic_optimizer.step()

        self.global_network.increment_updates()
        self.local_updates += 1

        self.trajectory.clear()
        self.rewards.clear()

    def log_episode(self, episode_num, ep_reward, episode_length, distance_from_target=None):
        """Log episode statistics"""
        self.episode_rewards_history.append(ep_reward)

        window = min(100, len(self.episode_rewards_history))
        self.mean_reward = np.mean(self.episode_rewards_history[-window:])

        # global_mean, is_new_best = self.global_network.update_stats(
        #     self.worker_id, self.mean_reward, ep_reward
        # )
        # global_step = self.global_network.global_steps.value
        global_mean, is_new_best = self.global_network.update_stats(
            self.worker_id, self.mean_reward, ep_reward, episode_length
        )
        global_step = self.global_network.global_steps.value

        if is_new_best and settings.FILE_LOGGING:
            logger.info("!!! NEW BEST REWARD: %.1f (W%d, Ep%d)",
                      ep_reward, self.worker_id, episode_num)

        if settings.FILE_LOGGING:
            logger.info("W%d Ep%d | R: %6.1f | Local mean: %6.1f | Global mean: %6.1f | Len: %3d | Steps: %6d",
                    self.worker_id, episode_num, ep_reward, self.mean_reward,
                    global_mean, episode_length, self.total_steps)

        if self.log_queue is not None:
            try:
                self.log_queue.put_nowait({
                    "worker_id": self.worker_id,
                    "episode": episode_num,
                    "global_step": global_step,              # <-- ZMIANA 3
                    "reward": ep_reward,
                    "local_mean_reward": self.mean_reward,
                    "global_mean_reward": global_mean,
                    "episode_length": episode_length,
                    "total_steps": self.total_steps,
                    "distance_from_target": distance_from_target,
                })
            except:
                pass

        try:
            with open(LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.worker_id, episode_num, ep_reward, self.mean_reward,
                    global_mean, episode_length, self.total_steps,
                    self.global_network.total_updates.value, global_step,   # <-- ZMIANA 4
                    distance_from_target
                ])
        except:
            pass

    def run(self):
            # --- CPU thread limiting (critical for multi-process scaling) ---
        import os
        import cProfile
        import pstats
        import io
        
        profiler = cProfile.Profile()
        profiler.enable()
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # ---------------------------------------------------------------
        if settings.FILE_LOGGING:
            logger.info("W%d started", self.worker_id)

        self._init_networks()

        # restore previous mean reward
        try:
            prev_mean = self.global_network.worker_mean_rewards[self.worker_id]
            if prev_mean != 0.0:
                self.mean_reward = prev_mean
                self.episode_rewards_history = [prev_mean] * 100
                if settings.FILE_LOGGING:
                    logger.info("W%d restored mean reward from global: %.3f",
                            self.worker_id, prev_mean)
        except:
            pass

        env = None

        try:
            # initialize CARLA environment
            env = CarlaEnv(
                scenario=SCENARIO, spawn_point=False, terminal_point=False,
                mp_density=25, port=self.port,
                action_space=ACTION_TYPE, camera=CAMERA_TYPE,
                resX=250, resY=250, manual_control=False
            )

            while not (self.shutdown_event and self.shutdown_event.is_set()):
                try:
                    # update episode counter
                    self.episode_count += 1
                    current_episode = self.global_network.increment_episode()

                    # sync with global network
                    self.sync_with_global()

                    # reset environment
                    save_images = (current_episode % 200 == 0)
                    env.state_observer.reset()
                    state, speed = env.reset(save_image=save_images, episode=current_episode)

                    # normalize inputs
                    state = state / 255.0
                    speed = speed / 100.0

                    done = False
                    ep_reward = 0.0
                    step_count = 0
                    episode_step = 0
                    last_distance_from_target = None

                    maneuver_idx = 0
                    maneuver = env.car_decisions[maneuver_idx]
                    maneuver_tensor = torch.tensor([maneuver], device=self.device)

                    # run episode
                    while not done:
                        episode_step += 1
                        self.total_steps += 1
                        # self.global_network.global_steps.value += 1


                        # prepare state tensor
                        if isinstance(state, np.ndarray):
                            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                        else:
                            state_tensor = state.float().to(self.device)
                            if state_tensor.dim() == 3:
                                state_tensor = state_tensor.unsqueeze(0)

                        speed_tensor = torch.tensor([[speed]], dtype=torch.float32, device=self.device)

                        # update maneuver if needed
                        _, left_junction = env.planner.on_junction(env.vehicle.get_location())
                        if left_junction:
                            maneuver_idx += 1
                            if maneuver_idx < len(env.car_decisions):
                                maneuver = env.car_decisions[maneuver_idx]
                            else:
                                maneuver = 1
                            maneuver_tensor = torch.tensor([maneuver], device=self.device)

                        # get action
                        action, self.last_distribution = self.get_action(
                            state_tensor, speed_tensor, maneuver_tensor, TESTING
                        )

                        # save observation if needed
                        # if save_images:
                        #     env.state_observer.manouver = maneuver
                        #     env.state_observer.action = action
                        #     env.state_observer.step = episode_step
                        #     env.state_observer.episode = current_episode
                        #     env.state_observer.save_to_disk()
                        #     env.state_observer.draw_related_values()
                        #     env.state_observer.save_together()

                        # execute action in environment
                        env.step_apply_action(action)

                        # clear image queue
                        while not env.image_queue.empty():
                            _ = env.image_queue.get()

                        # CARLA ticks
                        env.world.tick()
                        # env.step_apply_action(action)
                        env.world.tick()

                        # get next state
                        next_state, reward, done, _, next_speed, distance_from_target = env.step(
                            save_image=save_images, episode=current_episode, step=episode_step
                        )

                        if save_images:
                            env.state_observer.reward = reward

                        next_state = next_state / 255.0
                        next_speed = next_speed / 100.0

                        self.rewards.append(reward)
                        ep_reward += reward
                        step_count += 1
                        last_distance_from_target = distance_from_target
                        # update global network periodically
                        if not TESTING and (step_count >= T_MAX or done):
                            if isinstance(next_state, np.ndarray):
                                next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
                            else:
                                next_state_tensor = next_state.float().to(self.device)
                                if next_state_tensor.dim() == 3:
                                    next_state_tensor = next_state_tensor.unsqueeze(0)

                            next_speed_tensor = torch.tensor([[next_speed]], dtype=torch.float32, device=self.device)

                            self.compute_and_apply_gradients(
                                next_state_tensor, done, next_speed_tensor, maneuver_tensor
                            )

                            # sync after update
                            if not done and self.local_updates % SYNC_EVERY_N_UPDATES == 0:
                                self.sync_with_global()

                            step_count = 0

                        state = next_state
                        speed = next_speed
                    if self.episode_count % 50 == 0:
                        profiler.disable()
                        profiler.dump_stats(f'profile_worker_{self.worker_id}.out')
                        profiler.enable()
                    # episode finished
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                    self.log_episode(current_episode, ep_reward, episode_step, distance_from_target=last_distance_from_target)

                    if self.episode_count % SAVE_INTERVAL == 0:
                        self.global_network.save(MODEL_SAVE_PATH)

                except RuntimeError as e:
                    if "time-out" in str(e):
                        logger.warning("W%d CARLA timeout, reconnecting...",
                                     self.worker_id)

                        # reconnect
                        if env:
                            env.world = None
                            env.client = None

                        time.sleep(CARLA_TIMEOUT_WAIT)

                        env = CarlaEnv(
                            scenario=SCENARIO, spawn_point=False, terminal_point=False,
                            mp_density=25, port=self.port,
                            action_space=ACTION_TYPE, camera=CAMERA_TYPE,
                            resX=250, resY=250, manual_control=False
                        )
                        if settings.FILE_LOGGING:
                            logger.info("W%d reconnected", self.worker_id)
                    else:
                        raise

        except KeyboardInterrupt:
            if settings.FILE_LOGGING:
                logger.info("W%d interrupted", self.worker_id)
            else:
                pass
        except Exception as e:
            logger.error("W%d crashed: %s", self.worker_id, str(e), exc_info=True)
            raise
        finally:
            profiler.disable()
            profiler.dump_stats(f'profile_worker_{self.worker_id}.out')
            if env:
                try:
                    env.world = None
                    env.client = None
                except:
                    pass
            if settings.FILE_LOGGING:
                logger.info("W%d terminated", self.worker_id)


###########################################################################
# Handle workers
###########################################################################

def handle_workers(global_network, log_queue=None, shutdown_event=None):
    """
    Launch and monitor workers.
    Automatically restart crashed workers.
    """
    workers = {}
    restart_counts = {i: 0 for i in range(NUM_WORKERS)}
    if settings.FILE_LOGGING:
        logger.info("Launching %d workers...", NUM_WORKERS)

    # start workers
    for worker_id in range(NUM_WORKERS):
        port = BASE_PORT + (100 * worker_id)
        device = WORKER_GPUS[worker_id]

        worker = A3CWorker(
            worker_id=worker_id,
            global_network=global_network,
            port=port,
            device=device,
            log_queue=log_queue,
            shutdown_event=shutdown_event
        )
        worker.start()
        workers[worker_id] = worker
        if settings.FILE_LOGGING:
            logger.info("W%d launched | Port: %d | Device: %s", worker_id, port, device)

    # monitor workers
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
                        except:
                            pass

                    wait_time = float(os.getenv('CARLA_SERVER_START_PERIOD', '30.0'))
                    if settings.FILE_LOGGING:
                        logger.info("W%d waiting %.1fs for CARLA...", worker_id, wait_time)
                    time.sleep(wait_time)

                    port = BASE_PORT + (100 * worker_id)
                    device = WORKER_GPUS[worker_id]

                    worker = A3CWorker(
                        worker_id=worker_id,
                        global_network=global_network,
                        port=port,
                        device=device,
                        log_queue=log_queue,
                        shutdown_event=shutdown_event
                    )
                    worker.start()
                    workers[worker_id] = worker
                    if settings.FILE_LOGGING:
                        logger.info("W%d restarted (total restarts: %d)",
                              worker_id, restart_counts[worker_id])

    except KeyboardInterrupt:
        if settings.FILE_LOGGING:
            logger.info("Stopping all workers...")
        if shutdown_event:
            shutdown_event.set()
        for worker in workers.values():
            if worker.is_alive():
                worker.terminate()
        for worker in workers.values():
            worker.join(timeout=5)
    if settings.FILE_LOGGING:
        logger.info("Training finished. Restart counts: %s", restart_counts)


###########################################################################
# Main
###########################################################################

if __name__ == "__main__":
    # initialize CSV log
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                'worker_id', 'episode', 'reward', 'local_mean',
                'global_mean', 'length', 'total_steps', 'total_updates',
                'global_step', 'distance_from_target'                      # <-- ZMIANA 5
            ])
    
    wandb_id = None
    wandb_id_file = os.path.join(ARGS.outdir , "wandb_run_id.txt")
    if ARGS.resume and os.path.exists(wandb_id_file):
        with open(wandb_id_file, "r") as f:
            wandb_id = f.read().strip()

    else:
        wandb_id = uuid.uuid4().hex[:8]
        try:
            with open(wandb_id_file, "w") as f:
                f.write(wandb_id)
        except OSError:
            pass


    mp.set_start_method('spawn')

    shutdown_event = mp.Event()
    log_queue = mp.Queue(maxsize=1000) if LOGGING else None

    if settings.FILE_LOGGING:
        logger.info("=" * 80)
        logger.info("A3C Multi-GPU Training")
        logger.info("=" * 80)
        logger.info("Workers: %d | GPUs: %s", NUM_WORKERS, WORKER_GPUS)
        logger.info("LR: %.6f | Gamma: %.4f | T_MAX: %d steps",
                LR, GAMMA, T_MAX)
        logger.info("=" * 80)

    # initialize global network
    state_shape = [250, 250, 3]
    action_shape = len(ac.ACTIONS_NAMES)
    critic_shape = 1

    global_network = GlobalNetwork(state_shape, action_shape, critic_shape)

    # load existing model if available
    if ARGS.resume:
        # try to find latest checkpoint
        pth_files = sorted(
            glob.glob(os.path.join(ARGS.outdir, "*.pth")),
            key=os.path.getmtime
        )

        if len(pth_files) > 0:
            checkpoint_path = pth_files[-1]  # newest file
            print(f"[RESUME] Loading checkpoint: {checkpoint_path}")

            try:
                global_network.load(checkpoint_path.replace(".pth", ""))
            except Exception as e:
                print(f"[RESUME] Failed to load checkpoint: {e}")
        else:
            print("[RESUME] No checkpoint found in outdir, starting fresh")


    # W&B logger
    logger_process = None
    if LOGGING and log_queue is not None and WANDB_AVAILABLE:
        logger_process = mp.Process(
            target=wandb_logger_process,
            args=(log_queue, shutdown_event, wandb_id),
            name="WandBLogger"
        )
        logger_process.start()
        if settings.FILE_LOGGING:
            logger.info("wandb logger process spawned")

    # start training
    try:
        handle_workers(global_network, log_queue=log_queue, shutdown_event=shutdown_event)
    except KeyboardInterrupt:
        if settings.FILE_LOGGING:
            logger.info("Interrupted")
        else:
            pass
    finally:
        shutdown_event.set()

        if log_queue is not None:
            try:
                log_queue.put(None)
            except:
                pass

        if logger_process is not None:
            logger_process.join(timeout=10)
            if logger_process.is_alive():
                logger_process.terminate()
        if settings.FILE_LOGGING:
            logger.info("=" * 80)
            logger.info("Training complete")
            logger.info("=" * 80)
