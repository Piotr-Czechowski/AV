# /net/tscratch/people/plgpczechow/AV/.venv/bin/python /net/tscratch/people/plgpczechow/AV/A_to_B_GPU_34/a3c.py
#nohup /net/tscratch/people/plgpczechow/AV/.venv/bin/python /net/tscratch/people/plgpczechow/AV/A_to_B_GPU_34/a3c.py > a3c_out.log 2>&1 &

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
import carla
from datetime import datetime

from nets.a2c import DiscreteActor as DeepDiscreteActor
from nets.a2c import Critic as DeepCritic

from ACTIONS import ACTIONS as ac
from utils import ColoredPrint
import wandb
LOGGING = settings.LOGGING

###########################################################################
# Configuration
###########################################################################

# use same seed
SEED = 52
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# global settings
ACTION_TYPE = settings.ACTION_TYPE
CAMERA_TYPE = settings.CAMERA_TYPE
MODEL_LOAD_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/carla_to_chainer_003.pth'
MODEL_SAVE_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/carla_to_chainer_003'
EXP_ID = "carla_to_chainer_003.pth"

GAMMA = settings.GAMMA
LR = settings.LR
USE_ENTROPY = settings.USE_ENTROPY
SCENARIO = settings.SCENARIO
TESTING = settings.TESTING

# A3C specific settings
NUM_WORKERS = 1
NUMBER_OF_SERVERS_PER_GPU = 1
n_gpus = torch.cuda.device_count()
WORKER_GPUS = ([f'cuda:{g}' for g in range(n_gpus) for _ in range(NUMBER_OF_SERVERS_PER_GPU)])[:NUM_WORKERS]
print(f'!!!!!!!!!!    WORKER_GPUS {WORKER_GPUS}')
BASE_PORT = settings.PORT

# Training parameters
UPDATE_INTERVAL = 5
MAX_GRAD_NORM = 1.0
SAVE_INTERVAL = 1

# retrying settings
MAX_RETRIES = 3
CARLA_TIMEOUT_WAIT = 60
WORKER_CHECK_INTERVAL = 5

# Logging
LOG_FILE = 'log.csv'

Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])


def wandb_logger_process(log_queue):
    if not LOGGING:
        return
    os.environ['WANDB_INSECURE_DISABLE_SSL'] = 'true'
    wandb.init(
        project="A_to_B",
        name="synchr_test3_11",
        resume="allow",
        # id=EXP_ID,
        config={"learning_rate": LR}
    )

    logger.info("wandb logger process started")

    while True:
        record = log_queue.get()
        if record is None:
            break

        wandb_log_time = time.time()

        worker_id = record.pop("worker_id", None)

        metrics = {}
        for k, v in record.items():
            if worker_id is not None and k not in ("episode", "global_step"):
                metrics[f"worker/{worker_id}/{k}"] = v
            else:
                metrics[k] = v

        wandb.log(metrics)

        logger.critical(f"[TIMING TOTAL] Logging metrics to W&B took total: {time.time() - wandb_log_time:.4f}s")

    wandb.finish()
    logger.info("wandb logger process finished")

class SharedAdam(torch.optim.Adam):
    """
    Shared Adam optimizer for A3C
    """
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.tensor(0)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
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
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()


###########################################################################
# Global network
###########################################################################

class GlobalNetwork:
    """
    Global network shared across all A3C workers
    """
    
    def __init__(self, state_shape, action_shape, critic_shape, device='cpu'):
        init_start = time.time()
        logger.critical(f"[TIMING] GlobalNetwork.__init__ started at {time.strftime('%H:%M:%S')}")
        
        self.device = device
        
        net_init_start = time.time()
        self.actor = DeepDiscreteActor(state_shape, action_shape, device).to(device)
        self.critic = DeepCritic(state_shape, critic_shape, device).to(device)
        logger.critical(f"[TIMING] Actor+Critic initialization took {time.time() - net_init_start:.4f}s")
        
        share_start = time.time()
        self.actor.share_memory()
        self.critic.share_memory()
        logger.critical(f"[TIMING] share_memory() took {time.time() - share_start:.4f}s")
        
        opt_start = time.time()
        self.actor_optimizer = SharedAdam(self.actor.parameters(), lr=LR, weight_decay=1e-2)
        self.critic_optimizer = SharedAdam(self.critic.parameters(), lr=LR, weight_decay=1e-2)
        logger.critical(f"[TIMING] Optimizer initialization took {time.time() - opt_start:.4f}s")
        
        self.global_episode = mp.Value('i', 0)
        self.total_updates = mp.Value('i', 0)
        self.best_reward = mp.Value('d', -float('inf'))
        self.global_mean_reward = mp.Value('d', 0.0)
        self.worker_mean_rewards = mp.Array('d', [0.0] * NUM_WORKERS)
        
        self.lock = mp.Lock()
        
        logger.info("=" * 80)
        logger.info("Global Network initialized")
        logger.info("Actor params: %d | Critic params: %d",
                   sum(p.numel() for p in self.actor.parameters()),
                   sum(p.numel() for p in self.critic.parameters()))
        logger.info("Optimizer: SharedAdam | LR: %.6f", LR)
        logger.info("=" * 80)
        
        logger.critical(f"[TIMING TOTAL] GlobalNetwork.__init__ total time: {time.time() - init_start:.4f}s")
    
    def update_from_worker(self, worker_id, actor_pack, critic_pack):
        """
        Update global network parameters using gradients from worker
        """

        update_start = time.time()
        logger.critical(f"[TIMING] W{worker_id} update_from_worker started at {time.strftime('%H:%M:%S')}")
        
        actor_names, actor_grads = actor_pack
        critic_names, critic_grads = critic_pack
        
        lock_start = time.time()
        with self.lock:
            lock_acquired = time.time()
            logger.critical(f"[TIMING] W{worker_id} lock acquisition took {lock_acquired - lock_start:.4f}s")
            
            # update actor
            actor_update_start = time.time()
            self.actor_optimizer.zero_grad()
            name2param_actor = dict(self.actor.named_parameters())
            for name, grad in zip(actor_names, actor_grads):
                if grad is None:
                    continue
                p = name2param_actor.get(name, None)
                if p is None:
                    raise RuntimeError(f"[A3C] Unknown actor param name from worker: {name}")
                p.grad = grad.to(self.device)
            
            self.actor_optimizer.step()
            logger.critical(f"[TIMING] W{worker_id} actor update took {time.time() - actor_update_start:.4f}s")
            
            # update critic
            critic_update_start = time.time()
            self.critic_optimizer.zero_grad()
            name2param_critic = dict(self.critic.named_parameters())

            for name, grad in zip(critic_names, critic_grads):
                if grad is None:
                    continue
                p = name2param_critic.get(name, None)
                if p is None:
                    raise RuntimeError(f"[A3C] Unknown critic param name from worker: {name}")
                p.grad = grad.to(self.device)
            
            self.critic_optimizer.step()
            logger.critical(f"[TIMING] W{worker_id} critic update took {time.time() - critic_update_start:.4f}s")
            
            self.total_updates.value += 1
            
            if self.total_updates.value % 100 == 0:
                actor_grad_norm = sum(g.norm().item() if g is not None else 0 for g in actor_grads)
                critic_grad_norm = sum(g.norm().item() if g is not None else 0 for g in critic_grads)
                logger.debug("W%d | Update #%d | A_grad: %.3f | C_grad: %.3f",
                           worker_id, self.total_updates.value, actor_grad_norm, critic_grad_norm)
        
        logger.critical(f"[TIMING TOTAL] W{worker_id} update_from_worker total time: {time.time() - update_start:.4f}s")
    
    def save(self, path):
        """Save global network and optimizer state"""
        save_start = time.time()
        logger.critical(f"[TIMING] GlobalNetwork.save started at {time.strftime('%H:%M:%S')}")
        
        with self.lock:
            state = {
                "actor": self.actor.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "global_mean_reward": self.global_mean_reward.value,
                "best_reward": self.best_reward.value,
                "global_episode": self.global_episode.value,
                "total_updates": self.total_updates.value
            }
            torch.save(state, path + ".pth")
            logger.info("Model saved | Ep: %d | Updates: %d | Best: %.1f | Mean: %.1f",
                       self.global_episode.value, self.total_updates.value,
                       self.best_reward.value, self.global_mean_reward.value)
        
        logger.critical(f"[TIMING TOTAL] GlobalNetwork.save took total {time.time() - save_start:.4f}s")
    
    def load(self, path):
        """Load global network and optimizer state"""
        load_start = time.time()
        logger.critical(f"[TIMING] GlobalNetwork.load started at {time.strftime('%H:%M:%S')}")
        
        state = torch.load(path, map_location=self.device)
        
        with self.lock:
            self.actor.load_state_dict(state["actor"])
            self.critic.load_state_dict(state["critic"])
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            
            self.global_mean_reward.value = state.get("global_mean_reward", 0.0)
            self.best_reward.value = state.get("best_reward", -float('inf'))
            self.global_episode.value = state.get("global_episode", 0)
            self.total_updates.value = state.get("total_updates", 0)
            
            logger.info("Model loaded | Ep: %d | Updates: %d | Best: %.1f | Mean: %.1f",
                       self.global_episode.value, self.total_updates.value,
                       self.best_reward.value, self.global_mean_reward.value)
        
        logger.critical(f"[TIMING] GlobalNetwork.load took total {time.time() - load_start:.4f}s")


###########################################################################
# Worker
###########################################################################

class A3CWorker(mp.Process):
    """
    A3C Worker process
    """
    
    def __init__(self, worker_id, global_network, port, device, log_queue=None):
        init_start = time.time()
        logger.critical(f"[TIMING] W{worker_id} A3CWorker.__init__ started at {time.strftime('%H:%M:%S')}")
        
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id
        self.global_network = global_network
        self.port = port
        self.device = torch.device(device)
        self.log_queue = log_queue

        self.gamma = GAMMA
        self.use_entropy = USE_ENTROPY
        
        self.trajectory = []
        self.rewards = []
        self.mean_reward = 0
        self.episode_count = 0
        self.steps_since_update = 0
        self.total_steps = 0
        self.total_updates_sent = 0
        
        self.episode_rewards_history = []
        self.episode_lengths = []
        self.retry_count = 0
        
        state_shape = [250, 250, 3]
        critic_shape = 1
        action_shape = len(ac.ACTIONS_NAMES)
        
        net_init_start = time.time()
        self.actor = DeepDiscreteActor(state_shape, action_shape, self.device).to(self.device)
        self.critic = DeepCritic(state_shape, critic_shape, self.device).to(self.device)
        logger.critical(f"[TIMING] W{worker_id} local networks initialization took {time.time() - net_init_start:.4f}s")
        
        self.accumulated_actor_grads = []
        self.accumulated_critic_grads = []
        
        logger.info("W%d initialized | Port: %d | Device: %s", 
                   self.worker_id, self.port, self.device)
        
        logger.critical(f"[TIMING TOTAL] W{worker_id} A3CWorker.__init__ total time: {time.time() - init_start:.4f}s")
    
    def sync_with_global(self):
        """Copy weights from global network to local network"""
        sync_start = time.time()
        logger.critical(f"[TIMING] W{self.worker_id} sync_with_global started at {time.strftime('%H:%M:%S')}")
        
        with self.global_network.lock:
            self.actor.load_state_dict(self.global_network.actor.state_dict())
            self.critic.load_state_dict(self.global_network.critic.state_dict())
        
        logger.critical(f"[TIMING TOTAL] W{self.worker_id} sync_with_global took total {time.time() - sync_start:.4f}s")
    
    def get_action(self, obs, speed, manouver, testing=False):
        """Get action from policy"""
        action_start = time.time()
        
        forward_start = time.time()
        logits = self.actor(obs, speed, manouver)
        value = self.critic(obs, speed, manouver)
        logger.critical(f"[TIMING] W{self.worker_id} forward pass (actor+critic) took {time.time() - forward_start:.4f}s")
        
        distribution = Categorical(logits=logits)
        self.action_distribution = distribution
        
        if testing:
            with torch.no_grad():
                action = distribution.probs.argmax(dim=-1)
        else:
            action = distribution.sample()
        
        log_prob = distribution.log_prob(action)
        action_np = action.cpu().numpy().squeeze(0)
        
        if not testing:
            self.trajectory.append(Transition(obs, value, action_np, log_prob))
        
        logger.critical(f"[TIMING TOTAL] W{self.worker_id} get_action total time: {time.time() - action_start:.4f}s")
        return action_np, distribution
    
    def calculate_n_step_returns(self, final_state, done, final_speed, manouver):
        """Calculate n-step returns for advantage estimation"""
        calc_start = time.time()
        
        returns = []
        
        with torch.no_grad():
            if done:
                R = torch.tensor([[0]]).float().to(self.device)
            else:
                R = self.critic(final_state, final_speed, manouver)
            
            for reward in reversed(self.rewards):
                R = torch.tensor(reward).float().to(self.device) + self.gamma * R
                returns.insert(0, R)
        
        logger.critical(f"[TIMING] W{self.worker_id} calculate_n_step_returns took {time.time() - calc_start:.4f}s")
        return returns
    
    def compute_gradients(self, final_state, done, final_speed, manouver):
        """
        Compute gradients from accumulated trajectory
        """
        grad_start = time.time()
        logger.critical(f"[TIMING] W{self.worker_id} compute_gradients started at {time.strftime('%H:%M:%S')}")
        
        if len(self.trajectory) == 0:
            return None, None, None, None
        
        returns_start = time.time()
        returns = self.calculate_n_step_returns(final_state, done, final_speed, manouver)
        logger.critical(f"[TIMING] W{self.worker_id} returns calculation took {time.time() - returns_start:.4f}s")
        
        batch_start = time.time()
        batch = Transition(*zip(*self.trajectory))
        values = batch.value_s
        log_probs = batch.log_prob_a
        logger.critical(f"[TIMING] W{self.worker_id} batch preparation took {time.time() - batch_start:.4f}s")
        
        loss_calc_start = time.time()
        actor_losses = []
        critic_losses = []
        
        for G_t, V_s, log_prob in zip(returns, values, log_probs):
            advantage = G_t - V_s
            actor_losses.append(-log_prob * advantage.detach())
            critic_losses.append(F.smooth_l1_loss(V_s, G_t))
        
        actor_loss = torch.stack(actor_losses).mean()
        critic_loss = torch.stack(critic_losses).mean()
        
        if self.use_entropy:
            actor_loss = torch.stack(actor_losses).mean() - self.action_distribution.entropy().mean()
        else:
            actor_loss = torch.stack(actor_losses).mean()
        
        logger.critical(f"[TIMING] W{self.worker_id} loss calculation took {time.time() - loss_calc_start:.4f}s")
        
        backward_start = time.time()
        self.actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_grads = [p.grad.clone() if p.grad is not None else None 
                      for p in self.actor.parameters()]
        actor_names = [n for n, _ in self.actor.named_parameters()]
        logger.critical(f"[TIMING] W{self.worker_id} actor backward took {time.time() - backward_start:.4f}s")
        
        backward_start = time.time()
        self.critic.zero_grad()
        critic_loss.backward()
        critic_grads = [p.grad.clone() if p.grad is not None else None 
                       for p in self.critic.parameters()]
        critic_names = [n for n, _ in self.critic.named_parameters()]
        logger.critical(f"[TIMING] W{self.worker_id} critic backward took {time.time() - backward_start:.4f}s")
        
        self.trajectory.clear()
        self.rewards.clear()
        
        logger.critical(f"[TIMING TOTAL] W{self.worker_id} compute_gradients total time: {time.time() - grad_start:.4f}s")
        return actor_loss.item(), critic_loss.item(), (actor_names, actor_grads), (critic_names, critic_grads)
    
    # def update_global_network(self):
    #     """Send accumulated gradients to global network"""
    #     if len(self.accumulated_actor_grads) == 0 or self.steps_since_update == 0:
    #         return
        
    #     avg_actor_grads = [g if g is not None else None 
    #                       for g in self.accumulated_actor_grads]
    #     avg_critic_grads = [g if g is not None else None 
    #                        for g in self.accumulated_critic_grads]
        
    #     self.global_network.update_from_worker(
    #         self.worker_id, avg_actor_grads, avg_critic_grads
    #     )
        
    #     self.total_updates_sent += 1
        
    #     self.accumulated_actor_grads = []
    #     self.accumulated_critic_grads = []
    #     self.steps_since_update = 0
    
    def log_episode(self, episode_num, ep_reward, episode_length, distance_from_target=None):
        """Log episode statistics"""
        self.episode_rewards_history.append(ep_reward)
        self.episode_lengths.append(episode_length)
        
        window = min(100, self.episode_count)
        self.mean_reward = np.mean(self.episode_rewards_history[-window:])
        
        with self.global_network.lock:
            if ep_reward > self.global_network.best_reward.value:
                self.global_network.best_reward.value = ep_reward
                logger.info("!!! NEW BEST REWARD: %.1f (W%d, Ep%d)",
                          ep_reward, self.worker_id, episode_num)
            
            self.global_network.worker_mean_rewards[self.worker_id] = self.mean_reward
            self.global_network.global_mean_reward.value = \
                sum(self.global_network.worker_mean_rewards[:]) / NUM_WORKERS
            
            global_mean = self.global_network.global_mean_reward.value
        
        if LOGGING:
            logger.info("W%d Ep%d | R: %6.1f | Local mean: %6.1f | Global mean: %6.1f | Len: %3d | Steps: %6d",
                    self.worker_id, episode_num, ep_reward, self.mean_reward,
                    global_mean, episode_length, self.total_steps)

        if self.log_queue is not None:
            self.log_queue.put({
                "worker_id": self.worker_id,
                "episode": episode_num,
                "reward": ep_reward,
                "local_mean_reward": self.mean_reward,
                "global_mean_reward": global_mean,
                "episode_length": episode_length,
                "total_steps": self.total_steps,
                "distance_from_target": distance_from_target,
            })
        
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([
                self.worker_id, episode_num, ep_reward, self.mean_reward,
                global_mean, episode_length, self.total_steps, self.total_updates_sent,
                distance_from_target
            ])
    
    def run(self):
        run_start = time.time()
        logger.critical(f"[TIMING] W{self.worker_id} run() started at {time.strftime('%H:%M:%S')}")
        logger.info("W%d started", self.worker_id)
        
        try:
            with self.global_network.lock:
                prev_mean = self.global_network.worker_mean_rewards[self.worker_id]
        except Exception as e:
            logger.warning("W%d cannot restore previous mean reward: %s",
                        self.worker_id, str(e))
            prev_mean = 0.0

        if prev_mean != 0.0:
            self.mean_reward = prev_mean
            self.episode_rewards_history = [prev_mean] * 100
            self.episode_count = 100
            logger.info("W%d restored mean reward from global: %.3f",
                        self.worker_id, prev_mean)
        else:
            logger.info("W%d no previous mean reward found (starting from scratch).",
                        self.worker_id)

        env = None
        episodes_to_save = (1, 2, 3, 5, 10, 50, 100, 250, 500, 1000, 2000, 2245, 2250, 2280, 2320, 2321, 2322)
        
        try:
            # initialize CARLA environment
            env_init_start = time.time()
            logger.critical(f"[TIMING] W{self.worker_id} CarlaEnv initialization started at {time.strftime('%H:%M:%S')}")
            env = CarlaEnv(
                scenario=SCENARIO, spawn_point=False, terminal_point=False,
                mp_density=25, port=self.port,
                action_space=ACTION_TYPE, camera=CAMERA_TYPE,
                resX=250, resY=250, manual_control=False
            )
            logger.critical(f"[TIMING] W{self.worker_id} CarlaEnv initialization took {time.time() - env_init_start:.4f}s")
            
            while True:
                try:
                    episode_start = time.time()
                    logger.critical(f"\n[TIMING] W{self.worker_id} ========== EPISODE START at {time.strftime('%H:%M:%S')} ==========")
                    
                    # sync with global network
                    self.sync_with_global()
                    
                    # update episode counter
                    self.episode_count += 1
                    with self.global_network.lock:
                        self.global_network.global_episode.value += 1
                        current_episode = self.global_network.global_episode.value
                    
                    # reset environment
                    reset_start = time.time()
                    logger.critical(f"[TIMING] W{self.worker_id} env.reset started at {time.strftime('%H:%M:%S')}")
                    save_images = (current_episode%200 == 0)
                    env.state_observer.reset()
                    state, speed = env.reset(save_image=save_images, episode=current_episode)
                    logger.critical(f"[TIMING] W{self.worker_id} env.reset took {time.time() - reset_start:.4f}s")
                    
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
                    maneuver_tensor = torch.tensor([maneuver]).to(self.device)
                    
                    # run episode
                    while not done:
                        step_start = time.time()
                        episode_step += 1
                        self.total_steps += 1
                        
                        # update maneuver if needed
                        maneuver_check_start = time.time()
                        on_junction, left_junction = env.planner.on_junction(env.vehicle.get_location())
                        if left_junction:
                            maneuver_idx += 1
                            if maneuver_idx < len(env.car_decisions):
                                maneuver = env.car_decisions[maneuver_idx]
                            else:
                                maneuver = 1
                            maneuver_tensor = torch.tensor([maneuver]).to(self.device)
                        logger.critical(f"[TIMING] W{self.worker_id} step {episode_step} maneuver check took {time.time() - maneuver_check_start:.4f}s")
                        
                        # get action
                        action, self.last_distribution = self.get_action(
                            state, speed, maneuver_tensor, TESTING
                        )
                        
                        # save observation if needed
                        if save_images:
                            save_start = time.time()
                            env.state_observer.manouver = maneuver
                            env.state_observer.action = action
                            env.state_observer.step = episode_step
                            env.state_observer.episode = current_episode
                            env.state_observer.save_to_disk()
                            env.state_observer.draw_related_values()
                            env.state_observer.save_together()
                            logger.critical(f"[TIMING] W{self.worker_id} step {episode_step} image saving took {time.time() - save_start:.4f}s")
                        
                        # execute action in environment
                        action_apply_start = time.time()
                        env.step_apply_action(action)
                        logger.critical(f"[TIMING] W{self.worker_id} step {episode_step} step_apply_action took {time.time() - action_apply_start:.4f}s")
                        
                        # clear image queue
                        queue_clear_start = time.time()
                        while not env.image_queue.empty():
                            _ = env.image_queue.get()
                        logger.critical(f"[TIMING] W{self.worker_id} step {episode_step} queue clear took {time.time() - queue_clear_start:.4f}s")
                        
                        # CARLA ticks
                        tick_start = time.time()
                        env.world.tick()
                        env.step_apply_action(action)
                        env.world.tick()
                        logger.critical(f"[TIMING] W{self.worker_id} step {episode_step} 2x world.tick() took {time.time() - tick_start:.4f}s")
                        
                        # get next state
                        env_step_start = time.time()
                        next_state, reward, done, _, speed, distance_from_target = env.step(
                            save_image=save_images, episode=current_episode, step=episode_step
                        )
                        logger.critical(f"[TIMING] W{self.worker_id} step {episode_step} env.step() took {time.time() - env_step_start:.4f}s")
                        
                        if save_images:
                            env.state_observer.reward = reward

                        next_state = next_state/255.0
                        speed = speed / 100.0
                        
                        self.rewards.append(reward)
                        ep_reward += reward
                        step_count += 1
                        last_distance_from_target = distance_from_target
                        
                        # update global network periodically
                        if not TESTING and (step_count >= UPDATE_INTERVAL or done):
                            grad_and_update_start = time.time()
                            logger.critical(f"[TIMING] W{self.worker_id} step {episode_step} gradient computation+update started")
                            
                            actor_loss, critic_loss, (actor_names, actor_grads), (critic_names, critic_grads) = self.compute_gradients(
                                next_state, done, speed, maneuver_tensor
                            )
                            
                            if actor_grads is not None:
                                self.global_network.update_from_worker(
                                    self.worker_id,
                                    (actor_names, actor_grads),
                                    (critic_names, critic_grads)
                                )
                                self.sync_with_global()
                            
                            logger.critical(f"[TIMING] W{self.worker_id} step {episode_step} gradient computation+update took {time.time() - grad_and_update_start:.4f}s")
                            
                            step_count = 0
                        
                        state = next_state
                        
                        logger.critical(f"[TIMING] W{self.worker_id} step {episode_step} TOTAL step time: {time.time() - step_start:.4f}s")
                    
                    # episode finished
                    cleanup_start = time.time()
                    logger.critical(f"[TIMING] W{self.worker_id} episode cleanup started at {time.strftime('%H:%M:%S')}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.critical(f"[TIMING] W{self.worker_id} gc.collect() + cuda.empty_cache() took {time.time() - cleanup_start:.4f}s")
                    
                    self.log_episode(current_episode, ep_reward, episode_step, distance_from_target=last_distance_from_target)
                    
                    if self.episode_count % SAVE_INTERVAL == 0:
                        self.global_network.save(MODEL_SAVE_PATH)
                    
                    self.retry_count = 0
                    
                    logger.critical(f"[TIMING] W{self.worker_id} ========== EPISODE TOTAL TIME: {time.time() - episode_start:.4f}s ==========\n")
                    
                except RuntimeError as e:
                    if "time-out" in str(e):
                        self.retry_count += 1
                        logger.warning("W%d CARLA timeout (attempt %d/%d)",
                                     self.worker_id, self.retry_count, MAX_RETRIES)
                        
                        if self.retry_count >= MAX_RETRIES:
                            logger.error("W%d max retries exceeded", self.worker_id)
                            raise
                        
                        # reconnect
                        reconnect_start = time.time()
                        logger.critical(f"[TIMING] W{self.worker_id} reconnect started at {time.strftime('%H:%M:%S')}")
                        
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
                        logger.info("W%d reconnected", self.worker_id)
                        logger.critical(f"[TIMING] W{self.worker_id} reconnect took {time.time() - reconnect_start:.4f}s")
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
                    env.world = None
                    env.client = None
                except:
                    pass
            logger.info("W%d terminated", self.worker_id)
            logger.critical(f"[TIMING] W{self.worker_id} run() total time: {time.time() - run_start:.4f}s")


###########################################################################
# Handle one worker
###########################################################################

def handle_workers(global_network, log_queue=None):
    """
    Launch and monitor workers.
    Automatically restart crashed workers.
    """
    workers = {}
    restart_counts = {i: 0 for i in range(NUM_WORKERS)}
    
    logger.info("Launching %d workers...", NUM_WORKERS)
    
    # start workers
    for worker_id in range(NUM_WORKERS):
        launch_start = time.time()
        port = BASE_PORT + (100 * worker_id)
        device = WORKER_GPUS[worker_id]
        
        worker = A3CWorker(worker_id, global_network, port, device, log_queue)
        worker.start()
        workers[worker_id] = worker
        logger.info("W%d launched | Port: %d | Device: %s", worker_id, port, device)
        logger.critical(f"[TIMING] W{worker_id} launch took {time.time() - launch_start:.4f}s")
    
    # monitor workers
    try:
        while True:
            time.sleep(WORKER_CHECK_INTERVAL)
            
            for worker_id in range(NUM_WORKERS):
                worker = workers[worker_id]
                
                if not worker.is_alive():
                    restart_start = time.time()
                    logger.critical(f"[TIMING] W{worker_id} restart process started at {time.strftime('%H:%M:%S')}")
                    
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
                    logger.info("W%d waiting %.1fs for CARLA...", worker_id, wait_time)
                    time.sleep(wait_time)
                    
                    port = BASE_PORT + (100 * worker_id)
                    device = WORKER_GPUS[worker_id]
                    
                    worker = A3CWorker(worker_id, global_network, port, device, log_queue)
                    worker.start()
                    workers[worker_id] = worker
                    logger.info("W%d restarted (total restarts: %d)",
                              worker_id, restart_counts[worker_id])
                    
                    logger.critical(f"[TIMING] W{worker_id} restart process took {time.time() - restart_start:.4f}s")
    
    except KeyboardInterrupt:
        logger.info("Stopping all workers...")
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
    main_start = time.time()
    logger.critical(f"[TIMING] Main program started at {time.strftime('%H:%M:%S')}")
    
    # initialize CSV log
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                'worker_id', 'episode', 'reward', 'local_mean', 
                'global_mean', 'length', 'total_steps', 'updates_sent',
                'distance_from_target'
            ])
    
    mp.set_start_method('spawn')
    log_queue = None
    logger_process = None
    if LOGGING:
        log_queue = mp.Queue()
        logger_process = mp.Process(
            target=wandb_logger_process,
            args=(log_queue,),
            daemon=True
        )
        logger_process.start()
        logger.info("wandb logger process spawned")    
    logger.info("=" * 80)
    logger.info("A3C Training")
    logger.info("=" * 80)
    logger.info("Workers: %d | GPUs: %s", NUM_WORKERS, WORKER_GPUS)
    logger.info("LR: %.6f | Gamma: %.4f | Update interval: %d steps",
               LR, GAMMA, UPDATE_INTERVAL)
    logger.info("=" * 80)
    
    # initialize global network
    state_shape = [250, 250, 3]
    action_shape = len(ac.ACTIONS_NAMES)
    critic_shape = 1
    
    global_network = GlobalNetwork(state_shape, action_shape, critic_shape, device='cpu')
    
    # load existing model if available
    if os.path.isfile(MODEL_LOAD_PATH):
        global_network.load(MODEL_LOAD_PATH)
    else:
        logger.info("Starting training from scratch")
    
    # start training
    try:
        handle_workers(global_network, log_queue=log_queue)
    finally:
        if LOGGING and log_queue is not None:
            log_queue.put(None)
            logger_process.join(timeout=10)
    
    logger.info("=" * 80)
    logger.info("Training complete")
    logger.info("=" * 80)
    
    logger.critical(f"[TIMING] Main program total time: {time.time() - main_start:.4f}s")