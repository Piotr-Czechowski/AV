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
MODEL_LOAD_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/synchr_test3.pth'
MODEL_SAVE_PATH = 'A_to_B_GPU_34/PC_models/currently_trained/synchr_test3'

GAMMA = settings.GAMMA
LR = settings.LR
USE_ENTROPY = settings.USE_ENTROPY
ENTROPY_COEF = 0.001  # entropy regularization coefficient
SCENARIO = settings.SCENARIO
TESTING = settings.TESTING

# A3C specific settings
NUM_WORKERS = 4
NUMBER_OF_SERVERS_PER_GPU = 1
n_gpus = torch.cuda.device_count()
WORKER_GPUS = ([f'cuda:{g}' for g in range(n_gpus) for _ in range(NUMBER_OF_SERVERS_PER_GPU)])[:NUM_WORKERS]
BASE_PORT = settings.PORT

# Training parameters
UPDATE_INTERVAL = 5  # accumulate gradients for N steps before update
MAX_GRAD_NORM = 1.0  # gradient clipping threshold - used to avoid too big gradients
SAVE_INTERVAL = 5  # save model every N episodes (per worker)

# retrying settings
MAX_RETRIES = 3
CARLA_TIMEOUT_WAIT = 60
WORKER_CHECK_INTERVAL = 5

# Logging
LOG_FILE = 'log.csv'

Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])


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
                # move to shared memory (CPU tensors)
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
    
    # File handler with rotation
    file_handler = RotatingFileHandler(log_file, maxBytes=50*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    
    # Console handler
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
        self.device = device
        
        self.actor = DeepDiscreteActor(state_shape, action_shape, device).to(device)
        self.critic = DeepCritic(state_shape, critic_shape, device).to(device)
        
        # share network parameters in memory
        self.actor.share_memory()
        self.critic.share_memory()
        
        # shared optimizers with shared momentum
        self.actor_optimizer = SharedAdam(self.actor.parameters(), lr=LR, weight_decay=1e-2)
        self.critic_optimizer = SharedAdam(self.critic.parameters(), lr=LR, weight_decay=1e-2)
        
        # global statistics (thread-safe with mp.Value/Array)
        self.global_episode = mp.Value('i', 0)
        self.total_updates = mp.Value('i', 0)
        self.best_reward = mp.Value('d', -float('inf'))
        self.global_mean_reward = mp.Value('d', 0.0)
        self.worker_mean_rewards = mp.Array('d', [0.0] * NUM_WORKERS)
        
        # lock for thread-safe updates
        self.lock = mp.Lock()
        
        logger.info("=" * 80)
        logger.info("Global Network initialized")
        logger.info("Actor params: %d | Critic params: %d",
                   sum(p.numel() for p in self.actor.parameters()),
                   sum(p.numel() for p in self.critic.parameters()))
        logger.info("Optimizer: SharedAdam | LR: %.6f", LR)
        logger.info("=" * 80)
    
    def update_from_worker(self, worker_id, actor_grads, critic_grads):
        """
        Update global network parameters using gradients from worker
        """
        with self.lock:
            # update actor
            self.actor_optimizer.zero_grad()
            for param, grad in zip(self.actor.parameters(), actor_grads):
                if grad is not None:
                    param.grad = grad.to(self.device)
            
            # gradient clipping for stability for actor
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            self.actor_optimizer.step()
            
            # update critic
            self.critic_optimizer.zero_grad()
            for param, grad in zip(self.critic.parameters(), critic_grads):
                if grad is not None:
                    param.grad = grad.to(self.device)
            
            # gradient clipping for stability for critic
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.critic_optimizer.step()
            
            self.total_updates.value += 1
            
            # log gradient norms for debugging
            if self.total_updates.value % 100 == 0:
                actor_grad_norm = sum(g.norm().item() if g is not None else 0 for g in actor_grads)
                critic_grad_norm = sum(g.norm().item() if g is not None else 0 for g in critic_grads)
                logger.debug("W%d | Update #%d | A_grad: %.3f | C_grad: %.3f",
                           worker_id, self.total_updates.value, actor_grad_norm, critic_grad_norm)
    
    def save(self, path):
        """Save global network and optimizer state"""
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
    
    def load(self, path):
        """Load global network and optimizer state"""
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


###########################################################################
# Worker
###########################################################################

class A3CWorker(mp.Process):
    """
    A3C Worker process
    """
    
    def __init__(self, worker_id, global_network, port, device):
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id
        self.global_network = global_network
        self.port = port
        self.device = torch.device(device)
        
        # hyperparameters
        self.gamma = GAMMA
        self.use_entropy = USE_ENTROPY
        self.entropy_coef = ENTROPY_COEF
        
        # training state
        self.trajectory = []
        self.rewards = []
        self.mean_reward = 0
        self.episode_count = 0
        self.steps_since_update = 0
        self.total_steps = 0
        self.total_updates_sent = 0
        
        # statistics
        self.episode_rewards_history = []
        self.episode_lengths = []
        self.retry_count = 0
        
        # networks (local copies on GPU for fast forward passes)
        state_shape = [200, 200, 3]
        critic_shape = 1
        action_shape = len(ac.ACTIONS_NAMES)
        
        self.actor = DeepDiscreteActor(state_shape, action_shape, self.device).to(self.device)
        self.critic = DeepCritic(state_shape, critic_shape, self.device).to(self.device)
        
        # gradient accumulation buffers
        self.accumulated_actor_grads = []
        self.accumulated_critic_grads = []
        
        logger.info("W%d initialized | Port: %d | Device: %s", 
                   self.worker_id, self.port, self.device)
    
    def sync_with_global(self):
        """Copy weights from global network to local network"""
        with self.global_network.lock:
            self.actor.load_state_dict(self.global_network.actor.state_dict())
            self.critic.load_state_dict(self.global_network.critic.state_dict())
    
    def get_action(self, obs, speed, manouver, testing=False):
        """Get action from policy"""
        # Forward pass
        logits = self.actor(obs, speed, manouver)
        value = self.critic(obs, speed, manouver)
        
        distribution = Categorical(logits=logits)
        
        if testing:
            with torch.no_grad():
                action = distribution.probs.argmax(dim=-1)
        else:
            action = distribution.sample()
        
        log_prob = distribution.log_prob(action)
        action_np = action.cpu().numpy().squeeze(0)
        
        if not testing:
            self.trajectory.append(Transition(obs, value, action_np, log_prob))
        
        return action_np, distribution
    
    def calculate_n_step_returns(self, final_state, done, final_speed, manouver):
        """Calculate n-step returns for advantage estimation"""
        returns = []
        
        with torch.no_grad():
            if done:
                R = torch.tensor([[0]]).float().to(self.device)
            else:
                R = self.critic(final_state, final_speed, manouver)
            
            for reward in reversed(self.rewards):
                R = torch.tensor(reward).float().to(self.device) + self.gamma * R
                returns.insert(0, R)
        
        return returns
    
    def compute_gradients(self, final_state, done, final_speed, manouver):
        """
        Compute gradients from accumulated trajectory
        """
        if len(self.trajectory) == 0:
            return None, None
        
        # calculate returns
        returns = self.calculate_n_step_returns(final_state, done, final_speed, manouver)
        
        # extract trajectory components
        batch = Transition(*zip(*self.trajectory))
        values = batch.value_s
        log_probs = batch.log_prob_a
        
        # calculate losses
        actor_losses = []
        critic_losses = []
        
        for G_t, V_s, log_prob in zip(returns, values, log_probs):
            advantage = G_t - V_s
            actor_losses.append(-log_prob * advantage.detach())
            critic_losses.append(F.smooth_l1_loss(V_s, G_t))
        
        actor_loss = torch.stack(actor_losses).mean()
        critic_loss = torch.stack(critic_losses).mean()
        
        # add entropy bonus for exploration
        if self.use_entropy and hasattr(self, 'last_distribution'):
            entropy = self.last_distribution.entropy().mean()
            actor_loss = actor_loss - self.entropy_coef * entropy
        
        # compute gradients
        self.actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_grads = [p.grad.clone() if p.grad is not None else None 
                      for p in self.actor.parameters()]
        
        self.critic.zero_grad()
        critic_loss.backward()
        critic_grads = [p.grad.clone() if p.grad is not None else None 
                       for p in self.critic.parameters()]
        
        # accumulate gradients
        if len(self.accumulated_actor_grads) == 0:
            self.accumulated_actor_grads = actor_grads
            self.accumulated_critic_grads = critic_grads
        else:
            for i in range(len(actor_grads)):
                if actor_grads[i] is not None:
                    if self.accumulated_actor_grads[i] is not None:
                        self.accumulated_actor_grads[i] += actor_grads[i]
                    else:
                        self.accumulated_actor_grads[i] = actor_grads[i]
            
            for i in range(len(critic_grads)):
                if critic_grads[i] is not None:
                    if self.accumulated_critic_grads[i] is not None:
                        self.accumulated_critic_grads[i] += critic_grads[i]
                    else:
                        self.accumulated_critic_grads[i] = critic_grads[i]
        
        # clear trajectory after computation
        self.trajectory.clear()
        self.rewards.clear()
        
        return actor_loss.item(), critic_loss.item()
    
    def update_global_network(self):
        """Send accumulated gradients to global network"""
        if len(self.accumulated_actor_grads) == 0 or self.steps_since_update == 0:
            return
        
        # average accumulated gradients 
        avg_actor_grads = [(g*UPDATE_INTERVAL)/self.steps_since_update if g is not None else None 
                          for g in self.accumulated_actor_grads]
        avg_critic_grads = [(g*UPDATE_INTERVAL) / self.steps_since_update if g is not None else None 
                           for g in self.accumulated_critic_grads]
        
        # send to global network
        self.global_network.update_from_worker(
            self.worker_id, avg_actor_grads, avg_critic_grads
        )
        
        self.total_updates_sent += 1
        
        # clear accumulators
        self.accumulated_actor_grads = []
        self.accumulated_critic_grads = []
        self.steps_since_update = 0
    
    def log_episode(self, episode_num, ep_reward, episode_length):
        """Log episode statistics"""
        self.episode_rewards_history.append(ep_reward)
        self.episode_lengths.append(episode_length)
        
        # update local mean reward (rolling average over last 100 episodes)
        window = min(100, self.episode_count)
        self.mean_reward = np.mean(self.episode_rewards_history[-window:])
        
        # update global statistics
        with self.global_network.lock:
            if ep_reward > self.global_network.best_reward.value:
                self.global_network.best_reward.value = ep_reward
                logger.info("!!! NEW BEST REWARD: %.1f (W%d, Ep%d)",
                          ep_reward, self.worker_id, episode_num)
            
            self.global_network.worker_mean_rewards[self.worker_id] = self.mean_reward
            self.global_network.global_mean_reward.value = \
                sum(self.global_network.worker_mean_rewards[:]) / NUM_WORKERS
            
            global_mean = self.global_network.global_mean_reward.value
        
        # log to console
        if LOGGING:
            logger.info("W%d Ep%d | R: %6.1f | Local mean: %6.1f | Global mean: %6.1f | Len: %3d | Steps: %6d",
                    self.worker_id, episode_num, ep_reward, self.mean_reward,
                    global_mean, episode_length, self.total_steps)
            wandb.log({"episode steps": self.total_steps,"reward": ep_reward,
                        "episode": episode_num, "global_mean_reward": global_mean})
        
        # log to CSV
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([
                self.worker_id, episode_num, ep_reward, self.mean_reward,
                global_mean, episode_length, self.total_steps, self.total_updates_sent
            ])
    
    def run(self):
        """Main worker loop"""
        logger.info("W%d started", self.worker_id)
        
        env = None
        episodes_to_save = (1, 2, 3, 5, 10, 50, 100, 200, 500)
        
        try:
            # initialize CARLA environment
            env = CarlaEnv(
                scenario=SCENARIO, spawn_point=False, terminal_point=False,
                mp_density=25, port=self.port,
                action_space=ACTION_TYPE, camera=CAMERA_TYPE,
                resX=200, resY=200, manual_control=False
            )
            
            while True:
                try:
                    # sync with global network
                    self.sync_with_global()
                    
                    # update episode counter
                    self.episode_count += 1
                    with self.global_network.lock:
                        self.global_network.global_episode.value += 1
                        current_episode = self.global_network.global_episode.value
                    
                    # reset environment
                    save_images = current_episode in episodes_to_save
                    env.state_observer.reset()
                    state, speed = env.reset(save_image=save_images, episode=current_episode)
                    
                    # normalize inputs
                    state = state / 255.0
                    speed = speed / 100.0
                    
                    # episode state
                    done = False
                    ep_reward = 0.0
                    step_count = 0
                    episode_step = 0
                    
                    # maneuver tracking
                    maneuver_idx = 0
                    maneuver = env.car_decisions[maneuver_idx]
                    maneuver_tensor = torch.tensor([maneuver]).to(self.device)
                    
                    # run episode
                    while not done:
                        episode_step += 1
                        self.total_steps += 1
                        
                        # update maneuver if needed
                        on_junction, left_junction = env.planner.on_junction(env.vehicle.get_location())
                        if left_junction:
                            maneuver_idx += 1
                            if maneuver_idx < len(env.car_decisions):
                                maneuver = env.car_decisions[maneuver_idx]
                            else:
                                maneuver = 1  # Default: go straight
                            maneuver_tensor = torch.tensor([maneuver]).to(self.device)
                        
                        # get action
                        action, self.last_distribution = self.get_action(
                            state, speed, maneuver_tensor, TESTING
                        )
                        
                        # save observation if needed
                        if save_images:
                            env.state_observer.manouver = maneuver
                            env.state_observer.action = action
                            env.state_observer.step = episode_step
                            env.state_observer.episode = current_episode
                            env.state_observer.save_to_disk()
                            env.state_observer.draw_related_values()
                            env.state_observer.save_together()
                        
                        # execute action in environment
                        env.step_apply_action(action)
                        
                        # clear image queue
                        while not env.image_queue.empty():
                            _ = env.image_queue.get()
                        
                        # CARLA ticks
                        env.world.tick()
                        env.step_apply_action(action)
                        env.world.tick()
                        
                        # get next state
                        next_state, reward, done, _, speed, _ = env.step(
                            save_image=save_images, episode=current_episode, step=episode_step
                        )
                        
                        next_state = next_state/255.0
                        speed = speed / 100.0
                        
                        self.rewards.append(reward)
                        ep_reward += reward
                        step_count += 1
                        
                        # update global network periodically
                        if not TESTING and (step_count >= UPDATE_INTERVAL or done):
                            actor_loss, critic_loss = self.compute_gradients(
                                next_state, done, speed, maneuver_tensor
                            )
                            
                            if actor_loss is not None:
                                self.steps_since_update += step_count
                                self.update_global_network()
                                # sync with global network after update
                                self.sync_with_global()
                                
                                # save model periodically
                                if self.episode_count % SAVE_INTERVAL == 0:
                                    self.global_network.save(MODEL_SAVE_PATH)
                            
                            step_count = 0
                        
                        state = next_state
                    
                    # episode finished
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    self.log_episode(current_episode, ep_reward, episode_step)
                    self.retry_count = 0
                    
                except RuntimeError as e:
                    if "time-out" in str(e):
                        self.retry_count += 1
                        logger.warning("W%d CARLA timeout (attempt %d/%d)",
                                     self.worker_id, self.retry_count, MAX_RETRIES)
                        
                        if self.retry_count >= MAX_RETRIES:
                            logger.error("W%d max retries exceeded", self.worker_id)
                            raise
                        
                        # reconnect
                        if env:
                            env.world = None
                            env.client = None
                        
                        time.sleep(CARLA_TIMEOUT_WAIT)
                        
                        env = CarlaEnv(
                            scenario=SCENARIO, spawn_point=False, terminal_point=False,
                            mp_density=25, port=self.port,
                            action_space=ACTION_TYPE, camera=CAMERA_TYPE,
                            resX=200, resY=200, manual_control=False
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
                    env.world = None
                    env.client = None
                except:
                    pass
            logger.info("W%d terminated", self.worker_id)


###########################################################################
# Handle one worker
###########################################################################

def handle_workers(global_network):
    """
    Launch and monitor workers.
    Automatically restart crashed workers.
    """
    workers = {}
    restart_counts = {i: 0 for i in range(NUM_WORKERS)}
    
    logger.info("Launching %d workers...", NUM_WORKERS)
    
    # start workers
    for worker_id in range(NUM_WORKERS):
        port = BASE_PORT + (100 * worker_id)
        device = WORKER_GPUS[worker_id]
        
        worker = A3CWorker(worker_id, global_network, port, device)
        worker.start()
        workers[worker_id] = worker
        logger.info("W%d launched | Port: %d | Device: %s", worker_id, port, device)
    
    # monitor workers
    try:
        while True:
            time.sleep(WORKER_CHECK_INTERVAL)
            
            for worker_id in range(NUM_WORKERS):
                worker = workers[worker_id]
                
                if not worker.is_alive():
                    restart_counts[worker_id] += 1
                    logger.warning("W%d died (restart #%d)", 
                                 worker_id, restart_counts[worker_id])
                    
                    # clean up
                    worker.join(timeout=2)
                    
                    # remove core dumps
                    for core_file in glob.glob('core.*'):
                        try:
                            os.remove(core_file)
                        except:
                            pass
                    
                    # wait for CARLA
                    wait_time = float(os.getenv('CARLA_SERVER_START_PERIOD', '30.0'))
                    logger.info("W%d waiting %.1fs for CARLA...", worker_id, wait_time)
                    time.sleep(wait_time)
                    
                    # restart worker
                    port = BASE_PORT + (100 * worker_id)
                    device = WORKER_GPUS[worker_id]
                    
                    worker = A3CWorker(worker_id, global_network, port, device)
                    worker.start()
                    workers[worker_id] = worker
                    logger.info("W%d restarted (total restarts: %d)",
                              worker_id, restart_counts[worker_id])
    
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
    # initialize CSV log
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                'worker_id', 'episode', 'reward', 'local_mean', 
                'global_mean', 'length', 'total_steps', 'updates_sent'
            ])

    # wandb setup
    if LOGGING:
        print('Beginngin Weights and Biases initialization')
        wandb.init(
        # set the ##wandb project where this run will be logged
        project="A_to_B",
        # create or extend already logged run:
        resume="allow",
        id="synchr_test3.pth",

        # track hyperparameters and run metadata
        config={
        "name" : "synchr_test3.pth",
        "learning_rate": lr
        }
        )
        wandb.run.notes = "Town03. Img+speed+manouver. FOV = 60. speed/100.Nowy model (nie ten z blokami rezydualnymi), z predkoscia oraz manewrem na wejsciu. Scenariusz 13 - Krotkie skrety na roznych skrzyzowaniach. Slight turns like:  9: [0, 1, 0.2], #brake slight right. Gradients logged. Stara/nowa  funkcja nagrody(sin, nacisk an jazde okolo 20 km/h). Kamera (x = 0.3, z=2.5, pitch=-10)\n    " \
        "speed_reward = -1.2 + 8*math.sin(speed/10)" \
        "if route_distance < 1:" \
        "   route_distance_reward = 1" \
        "else:" \
        "   route_distance_reward = -8*math.sin(speed/10)."
    
    # set multiprocessing start method
    mp.set_start_method('spawn')
    
    logger.info("=" * 80)
    logger.info("A3C Training")
    logger.info("=" * 80)
    logger.info("Workers: %d | GPUs: %s", NUM_WORKERS, WORKER_GPUS)
    logger.info("LR: %.6f | Gamma: %.4f | Update interval: %d steps",
               LR, GAMMA, UPDATE_INTERVAL)
    logger.info("Max grad norm: %.2f | Entropy coef: %.3f",
               MAX_GRAD_NORM, ENTROPY_COEF)
    logger.info("=" * 80)
    
    # initialize global network
    state_shape = [200, 200, 3]
    action_shape = len(ac.ACTIONS_NAMES)
    critic_shape = 1
    
    global_network = GlobalNetwork(state_shape, action_shape, critic_shape, device='cpu')
    
    # load existing model if available
    if os.path.isfile(MODEL_LOAD_PATH):
        global_network.load(MODEL_LOAD_PATH)
    else:
        logger.info("Starting training from scratch")
    
    # start training
    handle_workers(global_network)
    
    logger.info("=" * 80)
    logger.info("Training complete")
    logger.info("=" * 80)