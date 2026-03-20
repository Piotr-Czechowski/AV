# Nasz algorytm, poprawiony przez Bartka oraz Piotrka, dzialajacy na srodowisku frozen lake
# ZMIANA: wracamy do klasycznego A3C n-step -> update sieci co T_MAX krokow (albo na końcu epizodu)
# Wykluczamy GPU. - Liczymy wszystko na CPU
# Frozen Lake
# Naprawiona Entropia - teraz liczymy średnią entropię z całego rollout, a nie tylko ostatniego kroku

"""
A3C Multi-GPU Implementation (FrozenLake) - N-STEP UPDATES (T_MAX)

Based on:
- V. Mnih et al. (2016) "Asynchronous Methods for Deep Reinforcement Learning"
- https://github.com/ikostrikov/pytorch-a3c

Key modifications for Multi-GPU:
1. Global model on CPU (shared memory)
2. Local models on GPU (for fast inference)
3. Gradient transfer GPU->CPU by copying (not reference)
4. SharedAdam with shared state in shared memory
5. Hogwild! style updates (no locks on weight updates)

Update mode in this version:
- Network update happens every T_MAX steps OR when episode ends (n-step returns with bootstrap).
"""

import glob
import time
import numpy as np
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gc
import logging
from logging.handlers import RotatingFileHandler
import csv

from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions.categorical import Categorical

import gymnasium as gym

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


###########################################################################
# Configuration (FrozenLake instead of CARLA/settings.py)
###########################################################################

LOGGING = True

# use same seed
SEED = 52
torch.manual_seed(SEED)
np.random.seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# FrozenLake env config
ENV_ID = "FrozenLake-v1"
MAP_NAME = "4x4"        # "4x4" or "8x8"
IS_SLIPPERY = True      # classic FrozenLake

# Model paths (keep similar naming style)
MODEL_LOAD_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/frozenlake_parallel_016.pth'
MODEL_SAVE_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/frozenlake_parallel_016'
EXP_ID = "frozenlake_parallel_016.pth"

# RL hyperparams
GAMMA = 0.99
LR = 1e-4
TESTING = False  # jeśli True -> greedy action (argmax)

# A3C specific settings
# NUM_WORKERS = 10
# NUMBER_OF_SERVERS_PER_GPU = 2
# n_gpus = torch.cuda.device_count()
# WORKER_GPUS = ([f'cuda:{g}' for g in range(n_gpus) for _ in range(NUMBER_OF_SERVERS_PER_GPU)])[:NUM_WORKERS]
# if len(WORKER_GPUS) < NUM_WORKERS:
#     WORKER_GPUS += ['cpu'] * (NUM_WORKERS - len(WORKER_GPUS))
# print(f'!!!!!!!!!!    WORKER_GPUS {WORKER_GPUS}')

NUM_WORKERS = 4
WORKER_GPUS = ['cpu'] * NUM_WORKERS
print(f'!!!!!!!!!!    WORKER_GPUS {WORKER_GPUS}')

# Training parameters
T_MAX = 5
MAX_GRAD_NORM = 40.0
ENTROPY_COEF = 0.03
VALUE_LOSS_COEF = 1.0
SAVE_INTERVAL = 20
SYNC_EVERY_N_UPDATES = 1  # po ilu update'ach robić sync (dla n-step)

# worker monitoring
WORKER_CHECK_INTERVAL = 5

# Logging
LOG_FILE = 'log.csv'

# Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])
Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a", "entropy"])


###########################################################################
# W&B logging (LEAVE AS-IS, as requested)
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
            "learning_rate": LR,
            "num_workers": NUM_WORKERS,
            "t_max": T_MAX,
            "gamma": GAMMA,
            "entropy_coef": ENTROPY_COEF,
            "value_loss_coef": VALUE_LOSS_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
            "update_mode": "n_step",
        }
    )

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


###########################################################################
# SharedAdam
###########################################################################

class SharedAdam(torch.optim.Adam):
    """
    Shared Adam optimizer for A3C.

    State (exp_avg, exp_avg_sq, step) is shared between workers,
    ensuring consistent adaptive learning rates.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


###########################################################################
# Logger
###########################################################################

def setup_logger(log_file='logs/a3c_training.log', level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=5)
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
        if sp.grad is None:
            sp.grad = torch.zeros_like(sp.data)

        if lp.grad is None:
            sp.grad.zero_()
        else:
            # sp.grad.copy_(lp.grad.detach().to("cpu"))
            sp.grad.copy_(lp.grad.detach())

###########################################################################
# FrozenLake Networks
###########################################################################

class DiscreteActor(nn.Module):
    """
    Actor for FrozenLake:
    - input: one-hot state [B, n_states]
    - output: logits [B, n_actions]
    """
    def __init__(self, n_states, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, s_onehot):
        return self.net(s_onehot)


class Critic(nn.Module):
    """
    Critic for FrozenLake:
    - input: one-hot state [B, n_states]
    - output: value [B, 1]
    """
    def __init__(self, n_states, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s_onehot):
        return self.net(s_onehot)


def one_hot_state(state_int: int, n_states: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(1, n_states, dtype=torch.float32, device=device)
    x[0, state_int] = 1.0
    return x


###########################################################################
# Global network
###########################################################################

class GlobalNetwork:
    """
    Global network in shared memory (CPU) for FrozenLake.
    """

    def __init__(self, n_states: int, n_actions: int):
        self.device = torch.device('cpu')

        self.actor = DiscreteActor(n_states, n_actions).to(self.device)
        self.critic = Critic(n_states).to(self.device)

        self.actor.share_memory()
        self.critic.share_memory()

        # Preallocate shared grad buffers
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

        self.actor_optimizer = SharedAdam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = SharedAdam(self.critic.parameters(), lr=LR)

        self.global_episode = mp.Value('i', 0)
        self.total_updates = mp.Value('i', 0)
        self.best_reward = mp.Value('d', -float('inf'))
        self.global_mean_reward = mp.Value('d', 0.0)
        self.worker_mean_rewards = mp.Array('d', [0.0] * NUM_WORKERS)

        self.stats_lock = mp.Lock()
        self.save_lock = mp.Lock()

        logger.info("=" * 80)
        logger.info("Global Network initialized (FrozenLake) | UPDATE: n_step (T_MAX)")
        logger.info("Actor params: %d | Critic params: %d",
                    sum(p.numel() for p in self.actor.parameters()),
                    sum(p.numel() for p in self.critic.parameters()))
        logger.info("Optimizer: SharedAdam | LR: %.6f | T_MAX: %d | Workers: %d",
                    LR, T_MAX, NUM_WORKERS)
        logger.info("=" * 80)

    def save(self, path):
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
                "env_id": ENV_ID,
                "map_name": MAP_NAME,
                "is_slippery": IS_SLIPPERY,
                "update_mode": "n_step",
                "t_max": T_MAX,
            }
            torch.save(state, path + ".pth")
            logger.info("Model saved | Ep: %d | Updates: %d | Best: %.3f | Mean: %.3f",
                        self.global_episode.value, self.total_updates.value,
                        self.best_reward.value, self.global_mean_reward.value)

    def load(self, path):
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

            logger.info("Model loaded | Ep: %d | Updates: %d | Best: %.3f | Mean: %.3f",
                        self.global_episode.value, self.total_updates.value,
                        self.best_reward.value, self.global_mean_reward.value)

    def update_stats(self, worker_id, mean_reward, episode_reward):
        is_new_best = False
        with self.stats_lock:
            if episode_reward > self.best_reward.value:
                self.best_reward.value = episode_reward
                is_new_best = True

            self.worker_mean_rewards[worker_id] = mean_reward
            self.global_mean_reward.value = float(sum(self.worker_mean_rewards[:]) / len(self.worker_mean_rewards[:]))

            return self.global_mean_reward.value, is_new_best

    def increment_episode(self):
        with self.stats_lock:
            self.global_episode.value += 1
            return self.global_episode.value

    def increment_updates(self):
        self.total_updates.value += 1
        return self.total_updates.value


###########################################################################
# Worker
###########################################################################

class A3CWorker(mp.Process):
    """
    FrozenLake worker.
    UPDATE MODE: n-step (update co T_MAX albo gdy done).
    """

    def __init__(self, worker_id, global_network, device,
                 log_queue=None, shutdown_event=None):
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id
        self.global_network = global_network
        # self.device = torch.device(device)
        self.device = torch.device('cpu')
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

        self.actor = None
        self.critic = None
        self._initialized = False

        # ostatnia dystrybucja akcji (dla entropy)
        self.action_distribution = None

        logger.info("W%d initialized | Device: %s", self.worker_id, self.device)

    def _make_env(self):
        env = gym.make(ENV_ID, map_name=MAP_NAME, is_slippery=IS_SLIPPERY)
        env.reset(seed=SEED + self.worker_id)
        env.action_space.seed(SEED + self.worker_id)
        return env

    def _init_networks(self, n_states, n_actions):
        if self._initialized:
            return
        self.actor = DiscreteActor(n_states, n_actions).to(self.device)
        self.critic = Critic(n_states).to(self.device)
        self.sync_with_global()
        self._initialized = True
        logger.info("W%d networks initialized on %s", self.worker_id, self.device)

    def sync_with_global(self):
        self.actor.load_state_dict(self.global_network.actor.state_dict())
        self.critic.load_state_dict(self.global_network.critic.state_dict())

    def get_action(self, s_onehot, testing=False):
        logits = self.actor(s_onehot)
        value = self.critic(s_onehot)

        # dist = Categorical(logits=logits)
        # self.action_distribution = dist

        # if testing:
        #     action = dist.probs.argmax(dim=-1)
        # else:
        #     action = dist.sample()

        # log_prob = dist.log_prob(action)
        # action_int = int(action.item())

        # if not testing:
        #     self.trajectory.append(Transition(s_onehot, value, action_int, log_prob))
        dist = Categorical(logits=logits)

        if testing:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()          # <<< DODANE
        action_int = int(action.item())

        if not testing:
            self.trajectory.append(
                Transition(s_onehot, value, action_int, log_prob, entropy)   # <<< ZMIENIONE
            )

        return action_int

    def compute_and_apply_gradients(self, final_state_oh, done: bool):
        """
        n-step update:
        - bootstrap from V(s_{t+1}) if not done
        - otherwise bootstrap = 0
        """
        if len(self.trajectory) == 0:
            return

        returns = []
        with torch.no_grad():
            if done:
                R = torch.zeros(1, 1, device=self.device)
            else:
                R = self.critic(final_state_oh)

        for reward in reversed(self.rewards):
            R = torch.tensor([[reward]], device=self.device, dtype=torch.float32) + self.gamma * R
            returns.insert(0, R)

        batch = Transition(*zip(*self.trajectory))
        values = batch.value_s
        log_probs = batch.log_prob_a
        entropies = batch.entropy

        policy_loss = 0.0
        value_loss = 0.0

        for G_t, V_s, log_prob in zip(returns, values, log_probs):
            advantage = G_t - V_s.detach()
            policy_loss = policy_loss - log_prob * advantage
            value_loss = value_loss + VALUE_LOSS_COEF * F.smooth_l1_loss(V_s, G_t)

        # entropy = self.action_distribution.entropy().mean() if self.action_distribution is not None else 0.0
        entropy = torch.stack(entropies).mean()   # <<< NOWA ENTROPIA Z CAŁEGO ROLLOUT
        total_loss = policy_loss + value_loss - ENTROPY_COEF * entropy

        self.actor.zero_grad()
        self.critic.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)

        transfer_grads_to_shared(self.actor, self.global_network.actor)
        transfer_grads_to_shared(self.critic, self.global_network.critic)

        self.global_network.actor_optimizer.step()
        self.global_network.critic_optimizer.step()

        self.global_network.increment_updates()
        self.local_updates += 1

        self.trajectory.clear()
        self.rewards.clear()

    def log_episode(self, episode_num, ep_reward, episode_length):
        self.episode_rewards_history.append(ep_reward)
        window = min(100, len(self.episode_rewards_history))
        self.mean_reward = float(np.mean(self.episode_rewards_history[-window:]))

        global_mean, is_new_best = self.global_network.update_stats(
            self.worker_id, self.mean_reward, ep_reward
        )

        if is_new_best:
            logger.info("!!! NEW BEST REWARD: %.3f (W%d, Ep%d)",
                        ep_reward, self.worker_id, episode_num)

        if LOGGING:
            logger.info(
                "W%d Ep%d | R: %6.3f | Local mean: %6.3f | Global mean: %6.3f | Len: %3d | Steps: %6d | Updates: %6d",
                self.worker_id, episode_num, ep_reward, self.mean_reward,
                global_mean, episode_length, self.total_steps,
                self.global_network.total_updates.value
            )

        if self.log_queue is not None:
            try:
                self.log_queue.put_nowait({
                    "worker_id": self.worker_id,
                    "episode": episode_num,
                    "reward": ep_reward,
                    "local_mean_reward": self.mean_reward,
                    "global_mean_reward": global_mean,
                    "episode_length": episode_length,
                    "total_steps": self.total_steps,
                    "total_updates": self.global_network.total_updates.value,
                })
            except:
                pass

        try:
            with open(LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.worker_id, episode_num, ep_reward, self.mean_reward,
                    global_mean, episode_length, self.total_steps,
                    self.global_network.total_updates.value
                ])
        except:
            pass

    def run(self):
        # CPU thread limiting
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        logger.info("W%d started", self.worker_id)

        env = self._make_env()
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self._init_networks(n_states, n_actions)

        # restore previous mean reward
        try:
            prev_mean = self.global_network.worker_mean_rewards[self.worker_id]
            if prev_mean != 0.0:
                self.mean_reward = prev_mean
                self.episode_rewards_history = [prev_mean] * 100
                logger.info("W%d restored mean reward from global: %.3f", self.worker_id, prev_mean)
        except:
            pass

        try:
            while not (self.shutdown_event and self.shutdown_event.is_set()):
                self.episode_count += 1
                current_episode = self.global_network.increment_episode()

                # sync at episode start
                self.sync_with_global()

                obs, _ = env.reset()
                done = False
                ep_reward = 0.0
                episode_step = 0
                step_count = 0

                while not done and not (self.shutdown_event and self.shutdown_event.is_set()):
                    episode_step += 1
                    self.total_steps += 1
                    step_count += 1

                    s_oh = one_hot_state(int(obs), n_states, self.device)
                    action = self.get_action(s_oh, testing=TESTING)

                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    r = float(reward)
                    self.rewards.append(r)
                    ep_reward += r

                    # update co T_MAX albo na końcu epizodu
                    if not TESTING and (step_count >= T_MAX or done):
                        next_oh = one_hot_state(int(next_obs), n_states, self.device)
                        self.compute_and_apply_gradients(next_oh, done)

                        # sync po update (jak w oryginale)
                        if not done and self.local_updates % SYNC_EVERY_N_UPDATES == 0:
                            self.sync_with_global()

                        step_count = 0

                    obs = next_obs

                gc.collect()
                # if self.device.type == 'cuda':
                #     torch.cuda.empty_cache()

                self.log_episode(current_episode, ep_reward, episode_step)

                if self.episode_count % SAVE_INTERVAL == 0:
                    self.global_network.save(MODEL_SAVE_PATH)

        except KeyboardInterrupt:
            logger.info("W%d interrupted", self.worker_id)
        except Exception as e:
            logger.error("W%d crashed: %s", self.worker_id, str(e), exc_info=True)
            raise
        finally:
            env.close()
            logger.info("W%d terminated", self.worker_id)


###########################################################################
# Handle workers
###########################################################################

def handle_workers(global_network, log_queue=None, shutdown_event=None):
    workers = {}
    restart_counts = {i: 0 for i in range(NUM_WORKERS)}

    logger.info("Launching %d workers...", NUM_WORKERS)

    for worker_id in range(NUM_WORKERS):
        device = WORKER_GPUS[worker_id]

        worker = A3CWorker(
            worker_id=worker_id,
            global_network=global_network,
            device=device,
            log_queue=log_queue,
            shutdown_event=shutdown_event
        )
        worker.start()
        workers[worker_id] = worker
        logger.info("W%d launched | Device: %s", worker_id, device)

    try:
        while not (shutdown_event and shutdown_event.is_set()):
            time.sleep(WORKER_CHECK_INTERVAL)

            for worker_id in range(NUM_WORKERS):
                worker = workers[worker_id]

                if not worker.is_alive():
                    restart_counts[worker_id] += 1
                    logger.warning("W%d died (restart #%d)", worker_id, restart_counts[worker_id])

                    worker.join(timeout=2)

                    for core_file in glob.glob('core.*'):
                        try:
                            os.remove(core_file)
                        except:
                            pass

                    device = WORKER_GPUS[worker_id]
                    worker = A3CWorker(
                        worker_id=worker_id,
                        global_network=global_network,
                        device=device,
                        log_queue=log_queue,
                        shutdown_event=shutdown_event
                    )
                    worker.start()
                    workers[worker_id] = worker
                    logger.info("W%d restarted (total restarts: %d)", worker_id, restart_counts[worker_id])

    except KeyboardInterrupt:
        logger.info("Stopping all workers...")
        if shutdown_event:
            shutdown_event.set()
        for worker in workers.values():
            if worker.is_alive():
                worker.terminate()
        for worker in workers.values():
            worker.join(timeout=5)

    logger.info("Training finished. Restart counts: %s", restart_counts)


###########################################################################
# Main
###########################################################################

if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                'worker_id', 'episode', 'reward', 'local_mean',
                'global_mean', 'length', 'total_steps', 'total_updates'
            ])

    mp.set_start_method('spawn')

    shutdown_event = mp.Event()
    log_queue = mp.Queue(maxsize=1000) if LOGGING else None

    logger.info("=" * 80)
    logger.info("A3C Multi-GPU Training (FrozenLake) | UPDATE: n_step (T_MAX)")
    logger.info("=" * 80)
    logger.info("Env: %s | map=%s | slippery=%s", ENV_ID, MAP_NAME, IS_SLIPPERY)
    logger.info("Workers: %d | Devices: %s", NUM_WORKERS, WORKER_GPUS)
    logger.info("LR: %.6f | Gamma: %.4f | T_MAX: %d", LR, GAMMA, T_MAX)
    logger.info("=" * 80)

    # get env sizes once
    tmp_env = gym.make(ENV_ID, map_name=MAP_NAME, is_slippery=IS_SLIPPERY)
    n_states = tmp_env.observation_space.n
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    global_network = GlobalNetwork(n_states, n_actions)

    if os.path.isfile(MODEL_LOAD_PATH):
        global_network.load(MODEL_LOAD_PATH)
    else:
        logger.info("Starting training from scratch")

    # W&B logger
    logger_process = None
    if LOGGING and log_queue is not None and WANDB_AVAILABLE:
        logger_process = mp.Process(
            target=wandb_logger_process,
            args=(log_queue, shutdown_event),
            name="WandBLogger"
        )
        logger_process.start()
        logger.info("wandb logger process spawned")

    try:
        handle_workers(global_network, log_queue=log_queue, shutdown_event=shutdown_event)
    except KeyboardInterrupt:
        logger.info("Interrupted")
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

        logger.info("=" * 80)
        logger.info("Training complete")
        logger.info("=" * 80)