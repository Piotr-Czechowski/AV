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
import time
import numpy as np
import os
import os

from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import wandb
import settings
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

from carla_env import CarlaEnv
import carla

# For RGB
from nets.a2c import Actor as DeepActor  # Continuous
from nets.a2c import DiscreteActor as DeepDiscreteActor  # Separate actor
from nets.a2c import Critic as DeepCritic  # Separate critic

from ACTIONS import ACTIONS as ac
from utils import ColoredPrint

# GPU
device = torch.device(settings.SHOULD_USE_CUDA)
seed = 52
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Global settings
port = settings.PORT
action_type = settings.ACTION_TYPE
camera_type = settings.CAMERA_TYPE
load_model = settings.LOAD_MODEL
model_incr_load = 'A_to_B/PC_models/currently_trained/scenario7.pth'
model_incr_save = 'A_to_B/PC_models/currently_trained/scenario7'

gamma = settings.GAMMA
lr = settings.LR
use_entropy = settings.USE_ENTROPY
scenario = settings.SCENARIO


# Transition - the representation of a single transition
"""
:param s: the state
:param value_s: the critic's prediction of the value of state s
:param a: the action taken
:param log_prob_a: the logarithm of the probability of taking action a according to the actor's current policy
"""
Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])


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

        env = CarlaEnv(scenario=scenario, spawn_point=False, terminal_point=False, mp_density=25, port=port,
                       action_space=self.action_type, camera=self.camera_type, resX=80, resY=80, manual_control=False)

        self.environment = env  # Carla env
        self.trajectory = []  # Contains the trajectory of the self as a sequence of transitions
        self.rewards = []  # Contains the rewards obtained from the env at every step
        self.policy = self.discrete_policy  # discrete or continuous

        self.best_mean_reward = - float("inf")  # self's personal best mean episode reward
        self.best_reward = - float("inf")
        self.global_step_num = 0
        self.log = ColoredPrint()

        # For continuous policy
        self.mu, self.sigma = 0, 0
        # For discrete policy
        self.logits = 0

        self.value = 0
        self.action_distribution = None

        state_shape = [80, 80, 3]
        critic_shape = 1

        if self.action_type == 'discrete':
            self.action_shape = len(ac.ACTIONS_NAMES)
            self.policy = self.discrete_policy
            self.actor = DeepDiscreteActor(state_shape, self.action_shape, device).to(device)
        elif self.action_type == 'continuous':
            self.action_shape = 2
            self.policy = self.multi_variate_gaussian_policy
            self.actor = DeepActor(state_shape, self.action_shape, device).to(device)
        else:
            self.log.err(f"Wrong action type: {self.action_type}, choose discrete or continuous")

        self.critic = DeepCritic(state_shape, critic_shape, device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def multi_variate_gaussian_policy(self, obs):
        """
        Calculates a multi-variate gaussian distribution over actions given observations
        :param obs: self's observation
        :return: policy, a distribution over actions for the given observation
        """
        mu, sigma = self.actor(obs)
        value = self.critic(obs)
        # Clamp each dim of mu based on the (low,high) limits of that action dim
        [mu[:, i].clamp_(-1, 1) for i in range(self.action_shape)]
        sigma = torch.nn.Softplus()(sigma).squeeze() + 1e-7  # Let sigma be (smoothly) +ve
        self.mu = mu.to(torch.device("cuda"))
        self.sigma = sigma.to(torch.device("cuda"))
        self.value = value.to(torch.device("cuda"))
        if len(self.mu.shape) == 0: # See if mu is a scalar
            self.mu.unsqueeze_(0)

        self.action_distribution = MultivariateNormal(self.mu, torch.eye(self.action_shape).to(device) * self.sigma,
                                                      validate_args=True)
        return self.action_distribution

    def process_action(self, action):
        if self.action_type == 'continuous':
            # Limit the action to lie between the (low, high) limits of the env
            [action[:, i].clamp_(-1, 1) for i in range(self.action_shape)]
        action = action.to(torch.device("cuda"))
        return action.cpu().numpy().squeeze(0)  # Convert to numpy ndarray, squeeze and remove the batch dimension

    def get_action(self, obs):
        action_distribution = self.policy(obs)  # Call to self.policy(obs) also populates self.value with V(obs)
        action = action_distribution.sample()

        log_prob_a = action_distribution.log_prob(action)

        action = self.process_action(action)
        # Store the n-step trajectory while training. Skip storing the trajectories in test mode
        self.trajectory.append(Transition(obs, self.value, action, log_prob_a))  # Construct the trajectory
        return action

    def discrete_policy(self, obs):
        """
        Calculates a discrete/categorical distribution over actions given observations
        :param obs: self's observation
        :return: policy, a distribution over actions for the given observation
        """
        logits = self.actor(obs)
        value = self.critic(obs)
        self.logits = logits.to(torch.device("cuda"))
        self.value = value.to(torch.device("cuda"))
        """
        The logits argument will be interpreted as unnormalized log probabilities and can therefore be any real number. 
        It will likewise be normalized so that the resulting probabilities sum to 1 along the last dimension. 
        attr:logits will return this normalized value.
        """
        self.action_distribution = Categorical(logits=self.logits)
        return self.action_distribution

    def calculate_n_step_return(self, n_step_rewards, final_state, done, gamma):
        """
        Calculates the n-step return for each state in the input-trajectory/n_step_transitions
        :param n_step_rewards: List of rewards for each step
        :param final_state: Final state in this n_step_transition/trajectory
        :param done: True rf the final state is a terminal state if not, False
        :return: The n-step return for each state in the n_step_transitions
        """
        g_t_n_s = []
        with torch.no_grad():
            g_t_n = torch.tensor([[0]]).float().to(device) if done else self.critic(final_state)
            for r_t in n_step_rewards[::-1]:  # Reverse order; From r_tpn to r_t
                g_t_n = torch.tensor(r_t).float() + gamma * g_t_n
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
        actor_losses, critic_losses, advantages = [], [], []
        for td_target, critic_prediction, log_p_a in zip(td_targets, v_s_batch, log_prob_a_batch):
            #writer.add_scalar("Value", critic_prediction, self.global_step_num)
            td_err = td_target - critic_prediction  # td_err is an unbiased estimated of Advantage
            advantages.append(td_err)
            result = - log_p_a * td_err

            actor_losses.append(result)
            critic_losses.append(F.smooth_l1_loss(critic_prediction, td_target))

        if self.use_entropy:
            actor_loss = torch.stack(actor_losses).mean() - self.action_distribution.entropy().mean()
        else:
            actor_loss = torch.stack(actor_losses).mean()

        critic_loss = torch.stack(critic_losses).mean()
        advantage = torch.stack(advantages).mean()

        #writer.add_scalar("actor_loss", actor_loss, self.global_step_num)
        #writer.add_scalar("critic_loss", critic_loss, self.global_step_num)
        #writer.add_scalar("Advantage", advantage, self.global_step_num)
        # writer.add_scalar("log_prob_actions_batch_mean", sum(log_prob_a_batch)/len(log_prob_a_batch), self.global_step_num)
        # writer.add_scalar("v_s_batch", sum(v_s_batch)/len(v_s_batch), self.global_step_num)
        # writer.add_scalar("td_targets", sum(td_targets)/len(td_targets), self.global_step_num)
        # writer.add_scalar("Entropy", self.action_distribution.entropy(), self.global_step_num)
        # writer.add_scalar("Entropy_mean", self.action_distribution.entropy().mean(), self.global_step_num)

        return actor_loss, critic_loss

    def optimize(self, final_state_rgb, done):
        td_targets = self.calculate_n_step_return(self.rewards, final_state_rgb, done, self.gamma)
        actor_loss, critic_loss = self.calculate_loss(self.trajectory, td_targets)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_lr = self.actor_optimizer.param_groups[0]['lr']
        critic_lr =self.critic_optimizer.param_groups[0]['lr']
        self.trajectory.clear()
        self.rewards.clear()

        return actor_loss, critic_loss, actor_lr, critic_lr
    
    def save(self, name):
        model_file_name = name + ".pth"
        self_state = {"actor": self.actor.state_dict(),
                       "actor_optimizer": self.actor_optimizer.state_dict(),
                       "critic": self.critic.state_dict(),
                       "critic_optimizer": self.critic_optimizer.state_dict(),
                       "best_mean_reward": self.best_mean_reward,
                       "best_reward": self.best_reward}
        torch.save(self_state, model_file_name)
        print("self's state is saved to", model_file_name)

    def load(self, name):
        model_file_name = name
        self_state = torch.load(model_file_name, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(self_state["actor"])
        self.critic.load_state_dict(self_state["critic"])
        self.actor_optimizer.load_state_dict(self_state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(self_state["critic_optimizer"])
        self.actor.to(device) #added
        self.critic.to(device) #added

        self.best_mean_reward = self_state["best_mean_reward"]
        self.best_reward = self_state["best_reward"]
        print("Loaded Advantage Actor-Critic model state from", model_file_name,
              " which fetched a best mean reward of:", self.best_mean_reward,
              " and an all time best reward of:", self.best_reward)
        
def handle_crash(results_queue, episode_idx, server_failed, mean_reward):
    wandb.init(
    # set the #wandb project where this run will be logged
    project="A_to_B",
    # create or extend already logged run:
    resume="allow",
    id="run38",  

    # track hyperparameters and run metadata
    config={
    "name" : "run38_two_right_turns",
    "learning_rate": lr
    }
    )

    agent = DeepActorCriticAgent()
    if os.path.isfile(model_incr_load):
        print("model istnieje i jest wgrywany.")
        agent.load(model_incr_load)
    else:
        print("model jeszcze nie istnieje.")

    episode_rewards = []  # Every episode's reward
    prev_checkpoint_mean_ep_rew = agent.best_mean_reward
    num_improved_episodes_before_checkpoint = 0  # To keep track of the num of ep with higher perf to save model
    
    while 1:
        # with lock:
        episode_idx.value += 1

        state_rgb = agent.environment.reset()
        state_rgb = state_rgb / 255.0  # resize the tensor to [0, 1]

        done = False
        ep_reward = 0.0 
        step_num = 0  # used to yield the optimize method
        actions_counter = dict()

        # Calculate how many each action was taken
        for action in ac.ACTIONS_NAMES.values():
            actions_counter[action] = 0

        while not done:
            action = agent.get_action(state_rgb)
            if agent.action_type == 'discrete':
                actions_counter[ac.ACTIONS_NAMES[agent.environment.action_space[action]]] += 1

            new_state, reward, done, route_distance = agent.environment.step(action)
            new_state = new_state / 255  # resize the tensor to [0, 1]

            agent.rewards.append(reward)
            ep_reward += reward
            step_num += 1
            print("Step number: ", step_num, "reward: ", reward, "ep_reward: ", ep_reward)
            #wandb.log({"step reward": reward, "Route distance": route_distance})
            if step_num >= 5 or done:
                actor_loss, critic_loss, actor_lr, critic_lr = agent.optimize(new_state, done)
                #wandb.log({ "actor_loss": actor_loss, "critic_loss": critic_loss, "actor_lr": actor_lr,"critic_lr": critic_lr})
                step_num = 0

            state_rgb = new_state
            agent.global_step_num += 1
            #writer.add_scalar("reward", reward, self.global_step_num)

        if agent.action_type == 'discrete':
            print(str(actions_counter))

        episode_rewards.append(ep_reward)
        mean_reward.value = (mean_reward.value * episode_idx.value + ep_reward)/(episode_idx.value + 1)
        
        if ep_reward > agent.best_reward:
            agent.best_reward = ep_reward
        # if np.mean(episode_rewards) > prev_checkpoint_mean_ep_rew:
        if mean_reward.value > prev_checkpoint_mean_ep_rew:
            num_improved_episodes_before_checkpoint += 1
        if num_improved_episodes_before_checkpoint >= 3:
            # prev_checkpoint_mean_ep_rew = np.mean(episode_rewards)
            # agent.best_mean_reward = np.mean(episode_rewards)
            prev_checkpoint_mean_ep_rew = mean_reward.value
            agent.best_mean_reward = mean_reward.value
            if not os.path.exists('improved_models'):
                os.mkdir('improved_models')
            save_path = os.getcwd() + '/improved_models'
            file_name = f"{episode_idx.value}_a-b_{agent.camera_type}_{agent.action_type}_gamma-{agent.gamma}_lr-{agent.lr}"
            cp_name = os.path.join(save_path, file_name)
            agent.save(cp_name) # Save the model when it improves
            num_improved_episodes_before_checkpoint = 0
        agent.save(model_incr_save)
        if episode_idx.value % 100 == 0:  # Save the model per 100 episodes
            agent.save(f"100_a-b_{agent.camera_type}_{agent.action_type}_gamma-{agent.gamma}_lr-{agent.lr}")
        if episode_idx.value % 250 == 0:
            if not os.path.exists('models'):
                os.mkdir('models')
            save_path = os.getcwd() + '/models'
            file_name = f"{episode_idx.value}_a-b_{agent.camera_type}_{agent.action_type}_gamma-{agent.gamma}_lr-{agent.lr}"
            cp_name = os.path.join(save_path, file_name)
            agent.save(cp_name)

        wandb.log({"episode": episode_idx.value, "reward": ep_reward, "learning_rate": agent.lr, "mean_reward": mean_reward.value})
        print("Episode: {} \t ep_reward:{} \t mean_ep_rew:{}\t best_ep_reward:{}".format(episode_idx.value,
                                                                                            ep_reward,
                                                                                            # np.mean(episode_rewards),
                                                                                            mean_reward.value,
                                                                                            agent.best_reward))        
    #wandb.finish()
    del world
    del client
    results_queue.put(1)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    results_queue = mp.Queue()
    manager = mp.Manager()
    episode_idx = manager.Value('i', 5988)
    # lock = manager.Lock()
    server_failed = manager.Value('i', 0)
    mean_reward = manager.Value('d', 365)
    while 1:   
        p = mp.Process(target=handle_crash, args=(results_queue, episode_idx, server_failed, mean_reward))
        p.start()
        p.join()
        if results_queue.empty():
            server_failed.value += 1
            # with lock:
            print(f'Process failed for episode {episode_idx.value} .')
            print(f'Process failed for the {server_failed.value} time.')

            # try to remove 'core.*' files
            for core_file in glob.glob(os.path.join(os.getcwd(), 'core.*')):
               os.remove(core_file)

            # assume that the server will restart
            time.sleep(float(os.getenv('CARLA_SERVER_START_PERIOD', '30.0')))
            continue
        else:
            # empty the queue
            results_queue.get()
            # with lock:
            episode_idx += 1

