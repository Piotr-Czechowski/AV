"""
The A2C's high-level flow:
1) Initialize the actor's and critic's networks.
2) Use the current policy of the actor to gather n-step experiences from the environment and calculate the n-step return.
3) Calculate the actor's and critic's losses.
4) Perform the stochastic gradient descent optimization step to update the actor and critic parameters.
5) Repeat from step 2.
"""

import glob
import pdb
import queue
import signal
import time
import numpy as np
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc

from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import wandb
import settings
from torch.distributions.categorical import Categorical

from carla_env import CarlaEnv
import carla
from datetime import datetime
# for new save and load
from pathlib import Path


# For RGB
# from nets.a2c import Actor as DeepActor  # Continuous
from nets.a2c import SharedImageSpeedActorCritic

from ACTIONS import ACTIONS as ac
from utils import ColoredPrint

# GPU
device = torch.device(settings.SHOULD_USE_CUDA)
seed = 52
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# Global settings
port = settings.PORT
action_type = settings.ACTION_TYPE
camera_type = settings.CAMERA_TYPE
load_model = settings.LOAD_MODEL
model_incr_load = 'A_to_B_GPU_34/PC_models/currently_trained/a2c_shared_image_speed.pth'
model_incr_save = model_incr_load

gamma = settings.GAMMA
lr = settings.LR
use_entropy = settings.USE_ENTROPY
scenario = settings.SCENARIO
testing = settings.TESTING
logging = settings.LOGGING

STOP_REQUESTED = False
ACTIVE_WORKER = None


def _request_worker_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print('[SLURM] Stop requested; finishing the current episode before exit.')


def _forward_stop_to_worker(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print('[SLURM] Stop requested; forwarding to the A2C worker.')
    if ACTIVE_WORKER is not None and ACTIVE_WORKER.is_alive():
        try:
            os.kill(ACTIVE_WORKER.pid, signal.SIGUSR1)
        except ProcessLookupError:
            pass


def _normalized_speed_tensor(speed):
    return torch.as_tensor(
        [[float(speed) / 100.0]], dtype=torch.float32, device=device)


def _run_state_path(filename):
    run_dir = os.environ.get('A2C_RUN_DIR')
    return Path(run_dir) / filename if run_dir else None


def _get_wandb_run_id():
    run_id_path = _run_state_path('wandb_run_id.txt')
    if run_id_path is None:
        return None
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
        if not run_id:
            raise RuntimeError(
                'W&B run ID file is empty: {}'.format(run_id_path))
        return run_id

    run_id = wandb.util.generate_id()
    temporary_path = run_id_path.with_suffix('.tmp')
    temporary_path.write_text(run_id)
    temporary_path.replace(run_id_path)
    return run_id


def _load_run_global_step():
    step_path = _run_state_path('global_step.txt')
    if step_path is None or not step_path.exists():
        return None
    step = int(step_path.read_text().strip())
    if step < 0:
        raise ValueError('Invalid negative A2C global step in {}'.format(
            step_path))
    return step


def _save_run_global_step(step):
    step_path = _run_state_path('global_step.txt')
    if step_path is None:
        return
    temporary_path = step_path.with_suffix('.tmp')
    temporary_path.write_text(str(step))
    temporary_path.replace(step_path)


# Transition - the representation of a single transition
"""
:param s: the state
:param value_s: the critic's prediction of the value of state s
:param a: the action taken
:param log_prob_a: the logarithm of the probability of taking action a according to the actor's current policy
"""
Transition = namedtuple(
    "Transition", ["s", "value_s", "a", "log_prob_a", "entropy"])


class DeepActorCriticAgent(mp.Process):
    def __init__(self):
        """
        An Advantage Actor-Critic (A2C) self that uses a Deep Neural Network to represent it's Policy and
        the Value function
        """
        super(DeepActorCriticAgent, self).__init__()
        # Create Carla env
        self.action_type = action_type
        self.camera_type = camera_type
        self.gamma = gamma
        self.lr = lr
        self.use_entropy = use_entropy

        # env = CarlaEnv(scenario=scenario, spawn_point=False, terminal_point=False, mp_density=25, port=port,
        #                     action_space=self.action_type, camera=self.camera_type, resX=80, resY=80, manual_control=False)
        env = CarlaEnv(scenario=scenario, spawn_point=False, terminal_point=False, mp_density=25, port=port,
                       action_space=self.action_type, camera=self.camera_type, resX=250, resY=250, manual_control=False)

        self.environment = env  # Carla env
        self.trajectory = []  # Contains the trajectory of the self as a sequence of transitions
        self.rewards = []  # Contains the rewards obtained from the env at every step
        self.best_mean_reward = - float("inf")  # self's personal best mean episode reward
        self.best_reward = - float("inf")
        self.global_step_num = 0
        self.log = ColoredPrint()

        if self.action_type == 'discrete':
            self.action_shape = len(ac.ACTIONS_NAMES)
        else:
            self.log.err(f"Wrong action type: {self.action_type}, choose discrete or continuous")

        self.model = SharedImageSpeedActorCritic(
            n_actions=self.action_shape, device=device).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        self.mean_reward = 0
        self.episode = 0

    def get_action(self, obs, speed, testing=False):
        observation = {"image": obs, "speed": speed}
        if testing:
            with torch.no_grad():
                logits, value = self.model(observation)
        else:
            logits, value = self.model(observation)
        action_distribution = Categorical(logits=logits)
        action = action_distribution.probs.argmax(dim=-1) if testing \
            else action_distribution.sample()
        log_prob_a = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        if not testing:
            self.trajectory.append(Transition(
                observation, value, action, log_prob_a, entropy))
        return int(action.item())

    def calculate_n_step_return(self, n_step_rewards, final_state, done,
                                gamma, final_speed):

        """A_to_B_GPU_34/PC_models/currently_trained/synchr_sc3_30_start_sc_3.pth
        Calculates the n-step return for each state in the input-trajectory/n_step_transitions
        :param n_step_rewards: List of rewards for each step
        :param final_state: Final state in this n_step_transition/trajectory
        :param done: True rf the final state is a terminal state if not, False
        :return: The n-step return for each state in the n_step_transitions
        """
        g_t_n_s = []
        with torch.no_grad():
            if done:
                g_t_n = torch.zeros((1, 1), dtype=torch.float32, device=device)
            else:
                _, g_t_n = self.model({
                    "image": final_state,
                    "speed": final_speed,
                })
            for r_t in n_step_rewards[::-1]:  # Reverse order; From r_tpn to r_t
                g_t_n = torch.as_tensor(
                    r_t, dtype=torch.float32, device=device).view(1, 1) \
                    + gamma * g_t_n
                g_t_n_s.insert(0, g_t_n)  # n-step returns inserted to the left to maintain correct index order
            return g_t_n_s

    def calculate_loss(self, trajectory, td_targets):
        """
        Calculates the critic and actor losses using the td_targets and self.trajectory
        :param td_targets:
        :return:
        """
        n_step_trajectory = Transition(*zip(*trajectory))
        v_s_batch = n_step_trajectory.value_s  # Critic prediction of the value of state s
        log_prob_a_batch = n_step_trajectory.log_prob_a
        entropies = n_step_trajectory.entropy
        values = torch.cat([value.view(1, -1) for value in v_s_batch], dim=0)
        returns = torch.cat([target.view(1, -1) for target in td_targets], dim=0)
        log_probs = torch.cat(
            [log_prob.view(-1) for log_prob in log_prob_a_batch])
        advantages = returns - values.detach()
        actor_loss = -(log_probs * advantages.view(-1)).mean()
        if self.use_entropy:
            entropy = torch.cat([item.view(-1) for item in entropies]).mean()
            actor_loss = actor_loss - entropy
        critic_loss = F.smooth_l1_loss(values, returns)

        return actor_loss, critic_loss

    def optimize(self, final_state_rgb, done, final_speed):
        td_targets = self.calculate_n_step_return(
            self.rewards, final_state_rgb, done, self.gamma, final_speed)

        actor_loss, critic_loss = self.calculate_loss(self.trajectory, td_targets)
        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        gradient_logs = {}
        if logging:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradient_logs[f"gradients/{name}"] = wandb.Histogram(
                        param.grad.detach().cpu().numpy())
        self.optimizer.step()
        if logging:
            wandb.log({
                "global_step": self.global_step_num,
                "actor_loss": actor_loss.detach(),
                "critic_loss": critic_loss.detach(),
                **gradient_logs,
            })
        learning_rate = self.optimizer.param_groups[0]['lr']
        self.trajectory.clear()
        self.rewards.clear()

        return actor_loss, critic_loss, learning_rate, learning_rate
    
    # def save(self, name):
    #     model_file_name = name + ".pth"
    #     if logging:
    #         print(f'Model to save name {model_file_name}')
    #     self_state = {"actor": self.actor.state_dict(),
    #                    "actor_optimizer": self.actor_optimizer.state_dict(),
    #                    "critic": self.critic.state_dict(),
    #                    "critic_optimizer": self.critic_optimizer.state_dict(),
    #                    "best_mean_reward": self.best_mean_reward,
    #                    "best_reward": self.best_reward,
    #                    "episode": self.episode,
    #                    "mean_reward": self.mean_reward}
    #     if logging:
    #         print(f'Model path exists {os.path.exists(model_file_name)} model up dir exists {os.path.exists("/".join(model_file_name.split("/")[:-1]))}')
    #     torch.save(self_state, model_file_name)

    # def load(self, name):
    #     model_file_name = name
    #     self_state = torch.load(model_file_name, map_location=lambda storage, loc: storage)
    #     self.actor.load_state_dict(self_state["actor"])
    #     self.critic.load_state_dict(self_state["critic"])
    #     self.actor_optimizer.load_state_dict(self_state["actor_optimizer"])
    #     self.critic_optimizer.load_state_dict(self_state["critic_optimizer"])
    #     self.actor.to(device) #added
    #     self.critic.to(device) #added
    #     try:
    #         self.mean_reward = self_state["mean_reward"]
    #     except:
    #         pass
    #     try:
    #         self.episode = self_state["episode"]
    #     except: 
    #         pass
    #     self.best_mean_reward = self_state["best_mean_reward"]
    #     self.best_reward = self_state["best_reward"]
    #     # print("Loaded Advantage Actor-Critic model state from", model_file_name,
    #     #       " which fetched a best mean reward of:", self.best_mean_reward,
    #     #       " and an all time best reward of:", self.best_reward)
    
    
# Methods for new model
    def save(self, name: str):
        """
        Saves the state of the models, optimizers, and training progress to a .pth file.
        This version is more robust and provides clearer feedback.

        Args:
            name (str): The base name for the checkpoint file (e.g., 'checkpoints/carla_agent').
                        The .pth extension will be added automatically.
        """
        # Use pathlib for modern and reliable path handling. It automatically
        # handles different operating systems (Windows/Linux).
        save_path = Path(name).with_suffix(".pth")
        
        # Ensure the parent directory exists before trying to save the file.
        # This prevents errors if you're saving to a new folder.
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if logging:
            print(f"Saving checkpoint to: {save_path}")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_mean_reward": self.best_mean_reward,
            "best_reward": self.best_reward,
            "episode": self.episode,
            "mean_reward": self.mean_reward,
            "global_step": self.global_step_num,
        }

        # Save the checkpoint dictionary
        torch.save(checkpoint, save_path)
        if logging:
            print(f"Checkpoint saved successfully.")

    def load(self, name: str):
        """
        Loads the state of the models, optimizers, and training progress from a .pth file.
        This version includes better error handling and safer dictionary access.

        Args:
            name (str): The full path to the checkpoint file (e.g., 'checkpoints/carla_agent.pth').
        """
        load_path = Path(name)
        
        if not load_path.is_file():
            if logging:
                print(f"Error: No checkpoint found at '{load_path}'. Starting from scratch.")
            return

        if logging:
            print(f"Loading checkpoint from: {load_path}")
        
        # Load the checkpoint onto the correct device (CPU or GPU)
        checkpoint = torch.load(load_path, map_location=device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Safely load training metadata using .get() with default values.
        # This prevents errors if a checkpoint is missing a key.
        self.episode = checkpoint.get("episode", 0)
        self.mean_reward = checkpoint.get("mean_reward", -float('inf'))
        self.best_mean_reward = checkpoint.get("best_mean_reward", -float('inf'))
        self.best_reward = checkpoint.get("best_reward", -float('inf'))
        self.global_step_num = checkpoint.get("global_step", 0)

        if logging:
            print("Checkpoint loaded successfully.")
            print(f"Resumed from Episode: {self.episode}")
            print(f"Best Mean Reward: {self.best_mean_reward:.2f}, Best Reward: {self.best_reward:.2f}")
        
def handle_crash(results_queue):
    signal.signal(signal.SIGUSR1, _request_worker_stop)
    signal.signal(signal.SIGTERM, _request_worker_stop)
    if logging:
        print('Initializing Weights & Biases')
        wandb_run_id = _get_wandb_run_id()
        wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'A_to_B'),
            entity=os.environ.get('WANDB_ENTITY') or None,
            name=os.environ.get('WANDB_RUN_NAME') or None,
            id=wandb_run_id,
            resume='allow' if wandb_run_id else None,
            config={
                'algorithm': 'A2C',
                'scenario': scenario,
                'map': 'Town03',
                'camera': camera_type,
                'learning_rate': lr,
                'gamma': gamma,
                'action_type': action_type,
                'seed': seed,
            },
            notes='A2C CARLA experiment from a2c_rgb_try2.py',
        )
        wandb.define_metric('global_step')
        wandb.define_metric('*', step_metric='global_step')
    agent = DeepActorCriticAgent()
    agent.mean_reward = 0
    agent.episode = 0
    print(f'Load a model from file')
    if os.path.isfile(model_incr_load):
        # print("model istnieje i jest wgrywany.")
        agent.load(model_incr_load)
        print(f'Loaded model and episode is {agent.episode}')
    else:
        print("model jeszcze nie istnieje.")
    run_global_step = _load_run_global_step()
    if run_global_step is not None:
        agent.global_step_num = run_global_step
        print('Resuming global step from this Slurm run: {}'.format(
            run_global_step))
    elif os.environ.get('A2C_RUN_DIR'):
        agent.global_step_num = 0
    episode_rewards = []  # Every episode's reward
    prev_checkpoint_mean_ep_rew = agent.best_mean_reward
    num_improved_episodes_before_checkpoint = 0  # To keep track of the num of ep with higher perf to save model
    episodes_to_save_images = (1, 5, 10, 43, 44, 50, 150, 400, 401, 402, 1000, 1001, 1002, 1003, 2001, 2002, 2003,
                                3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010,
                                4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110,
                                6800, 6801, 6802, 6803, 6804, 6805, 6806, 6807, 6808, 6809, 6810,
                                42010, 42011, 42012, 42013, 42014, 42015, 42016, 42017, 42018, 42019, 42020)
    max_speed = 0
    distance_from_goal = 0
    while 1:
        if STOP_REQUESTED:
            break
        # with lock:
        agent.episode += 1
        print(agent.episode)
        if agent.episode >= 42030:
            break
                # SET SYNCHRONOUS MODE
        # agent.environment.settings = agent.environment.world.get_settings()
        # agent.environment.settings.synchronous_mode = True
        # agent.environment.settings.fixed_delta_seconds = 0.1
        # agent.environment.settings.max_substep_delta_time = 0.01
        # agent.environment.settings.max_substeps = 10
        # agent.environment.world.apply_settings(agent.environment.settings)
        # if agent.episode % 2 == 0:
        #     agent.environment.scenario = [5]
        #     agent.environment.scenario_list = [5]
        # else:
        #     agent.environment.scenario = [6]
        #     agent.environment.scenario_list = [6]

        save_image = True if agent.episode in episodes_to_save_images else False
        
        agent.environment.state_observer.reset()

        state_rgb, speed = agent.environment.reset(
            save_image=save_image, episode=agent.episode)

        state_rgb = state_rgb / 255.0  # resize the tensor to [0, 1]
        speed_tensor = _normalized_speed_tensor(speed)

        done = False
        ep_reward = 0.0 
        step_num = 0  # used to yield the optimize method
        actions_counter = dict()

        # Calculate how many each action was taken
        for action in ac.ACTIONS_NAMES.values():
            actions_counter[action] = 0
        episode_step=0

        episode_start_time = datetime.now()

        while not done:
            episode_step += 1

            # perform_actions +=1  #perform every 0.2 seconds
            # if perform_actions%2==1:
                # print(perform_actions)
            if speed_tensor.item() > max_speed:
                max_speed = speed_tensor.item()

            action = agent.get_action(state_rgb, speed_tensor, testing)

            # print(f'Saving image, {save_image}')
            if save_image:
                agent.environment.state_observer.action = action # To print action on the frame
                agent.environment.state_observer.step = episode_step # To print action on the frame
                agent.environment.state_observer.episode = agent.episode
                agent.environment.state_observer.save_to_disk()
                agent.environment.state_observer.draw_related_values()
                agent.environment.state_observer.save_together()

            if agent.action_type == 'discrete':
                actions_counter[ac.ACTIONS_NAMES[agent.environment.action_space[action]]] += 1
            agent.environment.step_apply_action(action)
        # else:
            while not agent.environment.image_queue.empty():
                _ = agent.environment.image_queue.get()

            agent.environment.world.tick()
            agent.environment.step_apply_action(action)
            agent.environment.world.tick()
            
            save_image = True if agent.episode in episodes_to_save_images else False

            (new_state, reward, done, route_distance, speed, distance_from_goal
             ) = agent.environment.step(
                save_image=save_image,
                episode=agent.episode,
                step=episode_step)
            agent.environment.state_observer.reward = reward
            agent.global_step_num += 1
            _save_run_global_step(agent.global_step_num)
            if logging:
                wandb.log({
                    "global_step": agent.global_step_num,
                    "step_reward": reward,
                })
            new_state = new_state / 255.0  # resize the tensor to [0, 1]
            speed_tensor = _normalized_speed_tensor(speed)
            agent.rewards.append(reward)
            ep_reward += reward
            step_num += 1
            if sum(agent.rewards) <= -6:
                done = True
            # print("Step number: ", step_num, "reward: ", reward, "ep_reward: ", ep_reward)
            if not testing and (step_num >= 5 or done):
                actor_loss, critic_loss, actor_lr, critic_lr = agent.optimize(
                    new_state, done, speed_tensor)
                step_num = 0
            state_rgb = new_state

        print(f'Episode {agent.episode} ended')
        episode_end_time = datetime.now()
        episode_time = episode_end_time - episode_start_time
        
        gc.collect()
        torch.cuda.empty_cache()
        # if agent.action_type == 'discrete':
        #     print(str(actions_counter))

        print(f'Finalizing data to log to Weights and Biases')
        episode_rewards.append(ep_reward)
        agent.mean_reward = (agent.mean_reward * (min(100, agent.episode)-1) + ep_reward)/min(100, agent.episode) #mean reward from last 100 episodes
        if ep_reward > agent.best_reward:
            agent.best_reward = ep_reward
        agent.save(model_incr_save)
        if logging:
            print('Logging values of episode steps, episode duration [s], reward, episode, mean_reward etc.')
            print(f'{datetime.now().strftime("%Y %B %d | %H:%M")}  Logging values of episode steps, episode duration [s], reward, episode, mean_reward etc.')
            print(f'Episode {agent.episode}, and mean reward {agent.mean_reward}')
            wandb.log({
                "global_step": agent.global_step_num,
                "episode steps": episode_step,
                "episode duration [s]": episode_time.seconds,
                "reward": ep_reward,
                "episode": agent.episode,
                "mean_reward": agent.mean_reward,
                "max_speed": max_speed,
                "distance_from_goal": distance_from_goal,
            })

        # print("Episode: {} \t ep_reward:{} \t mean_ep_rew:{}\t best_ep_reward:{} max_speed: {} distance_from_goal: {}".format(agent.episode,
        #                                                                                     ep_reward,
        #                                                                                     agent.mean_reward,
        #                                                                                     agent.best_reward, 
        #                                                                                     max_speed,
        #                                                                                     distance_from_goal))        
    agent.environment.world = None
    agent.environment.client = None
    if logging and wandb.run is not None:
        wandb.finish()
    results_queue.put(1)


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, _forward_stop_to_worker)
    signal.signal(signal.SIGTERM, _forward_stop_to_worker)
    mp.set_start_method('spawn')
    results_queue = mp.Queue()
    manager = mp.Manager()
    # lock = manager.Lock()
    while not STOP_REQUESTED:
        p = mp.Process(target=handle_crash, args=(results_queue,))
        ACTIVE_WORKER = p
        p.start()
        if STOP_REQUESTED:
            os.kill(p.pid, signal.SIGUSR1)
        p.join()
        try:
            results_queue.get(timeout=2)
        except queue.Empty:
            # with lock:
            if STOP_REQUESTED:
                raise SystemExit(
                    '[SLURM] A2C worker exited before graceful shutdown completed.')
            print(f'Process failed.')
            # try to remove 'core.*' files
            for core_file in glob.glob(os.path.join(os.getcwd(), 'core.*')):
               os.remove(core_file)

            # assume that the server will restart
            time.sleep(float(os.getenv('CARLA_SERVER_START_PERIOD', '30.0')))
            continue
        else:
            if STOP_REQUESTED:
                print('[SLURM] A2C checkpoint saved; stopping supervisor.')
                break
