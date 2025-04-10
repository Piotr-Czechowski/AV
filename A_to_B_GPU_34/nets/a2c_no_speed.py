"""
Author: Praveen Palanisamy
"""
import torch


# class Actor(torch.nn.Module):
#     def __init__(self, input_shape, actor_shape, device=torch.device("cuda")):
#         """
#         Deep convolutional Neural Network to represent Actor in an Actor-Critic algorithm
#         The Policy is parametrized using a Gaussian distribution with mean mu and variance sigma
#         The Actor's policy parameters (mu, sigma) are output by the deep CNN implemented
#         in this class.
#         :param input_shape: Shape of each of the observations
#         :param actor_shape: Shape of the actor's output. Typically the shape of the actions
#         :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
#         """
#         super(Actor, self).__init__()
#         self.device = device
#         self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0),
#                                           torch.nn.ReLU())
#         self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=0),
#                                           torch.nn.ReLU())
#         self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1, padding=0),
#                                           torch.nn.ReLU())
#         self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 18496),
#                                           torch.nn.ReLU())
#         self.actor_mu = torch.nn.Linear(18496, actor_shape)
#         self.actor_sigma = torch.nn.Linear(18496, actor_shape)

#     def forward(self, x):
#         """
#         Forward pass through the Actor network. Takes batch_size x observations as input and produces mu and sigma
#         as the outputs
#         :param x: The observations
#         :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
#         """
#         x.requires_grad_()
#         x = x.to(self.device)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = x.view(x.shape[0], -1)
#         x = self.layer4(x)
#         actor_mu = self.actor_mu(x)
#         actor_sigma = self.actor_sigma(x)
#         return actor_mu, actor_sigma


class DiscreteActor(torch.nn.Module):
    def __init__(self, input_shape, actor_shape, device=torch.device("cuda")):
        """
        Deep convolutional Neural Network to represent Actor in an Actor-Critic algorithm
        The Policy is parametrized using a categorical/discrete distribution with logits
        The Actor's policy parameters (logits) are output by the deep CNN implemented
        in this class.
        :param input_shape: Shape of each of the observations
        :param actor_shape: Shape of the actor's output. Typically the shape of the actions
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(DiscreteActor, self).__init__()
        self.device = device
        # self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0),
        #  How many channels
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=0),
                                          torch.nn.ReLU())
        # self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, stride=1, padding=0),
        #                                  torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                          torch.nn.ReLU())
        # self.layer4 = torch.nn.Sequential(torch.nn.Linear(128 * 7 * 7, 512),
        #                                   torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 22 * 22, 512),
                                          torch.nn.ReLU())

        self.speed_layer1 = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU())

        # self.speed_layer2 = torch.nn.Sequential(
        #     torch.nn.Linear(64, 64),
        #     torch.nn.ReLU()
        # )
        self.actor = torch.nn.Linear(512, actor_shape)
        # self.actor = torch.nn.Linear(512+64, actor_shape) # camera+speed

        # self.scalar_layer = torch.nn.Sequential(
        #             torch.nn.Linear(1, 64),
        #             torch.nn.ReLU())
        # powrzucać dropouty, batchnorm
        # self.logits = torch.nn.Linear(2, actor_shape)

    def forward(self, x, speed=None, manouver=None):
        """
        Forward pass through the Actor network. Takes batch_size x observations as input and produces mu and sigma
        as the outputs
        :param x: The observations
        :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
        """
        x.requires_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        speed= None
        if speed is None and manouver is None:
            actor = self.actor(x)
        elif speed is not None and manouver is None:
            speed = speed.to(self.device).view(-1, 1)
            speed = self.speed_layer1(speed)
            # speed = self.speed_layer2(speed)
            combined = torch.cat([x, speed], dim=1)
            actor = self.actor(combined)
        elif speed is not None and manouver is not None:
            pass

        # scalar = scalar.to(self.device).view(-1, 1)
        # scalar_features = self.scalar_layer(scalar)
        # combined = torch.cat([x, scalar_features], dim=1)
        # logits = self.logits(combined)

        # attention = self.attention_layer(combined)
        # logits = self.logits(attention)
        return actor


class Critic(torch.nn.Module):
    def __init__(self, input_shape, critic_shape=1, device=torch.device("cuda")):
        """
        Deep convolutional Neural Network to represent the Critic in an Actor-Critic algorithm
        :param input_shape: Shape of each of the observations
        :param critic_shape: Shape of the Critic's output. Typically 1
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(Critic, self).__init__()
        self.device = device
        # input_shape[2] instead of 6
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=0),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                          torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 22 * 22, 512),
                                          torch.nn.ReLU())
        self.speed_layer1 = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU()
        )

        # self.speed_layer2 = torch.nn.Sequential(
        #     torch.nn.Linear(64, 64),
        #     torch.nn.ReLU()
        # )
        # ----Last layers----
        self.critic = torch.nn.Linear(512, critic_shape)
        # self.critic = torch.nn.Linear(512 + 64, critic_shape)

        # self.attention_layer = torch.nn.MultiheadAttention(512+64, 2)
        # self.critic = torch.nn.Linear(2, critic_shape)

    def forward(self, x, speed=None, manouver=None):
        """
        Forward pass through the Critic network. Takes batch_size x observations as input and produces the value
        estimate as the output
        :param x: The observations
        :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
        """
        x.requires_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.layer4(x)
        speed = None
        if speed is None and manouver is None:
            critic = self.critic(x)
        elif speed is not None and manouver is None:
            speed = speed.to(self.device).view(-1, 1)
            speed = self.speed_layer1(speed)
            # speed = self.speed_layer2(speed)
            combined = torch.cat([x, speed], dim=1)
            critic = self.critic(combined)
        elif speed is not None and manouver is not None:
            pass

        # print(x.shape)
        # scalar = scalar.to(self.device).view(-1, 1)
        # scalar_features = self.scalar_layer(scalar)
        # combined = torch.cat([x, scalar_features], dim=1)

        # critic = self.critic(combined)
        # attention = self.attention_layer(combined)
        # critic = self.critic(attention)

        return critic

# class ActorCritic(torch.nn.Module):
#     def __init__(self, input_shape, actor_shape, critic_shape, device=torch.device("cuda")):
#         """
#         A single neural network architecture that represents both the actor and the critic.
#         In this way, the feature extraction layers are shared between the actor and the critic,
#         and different heads (final layers) in the same neural network are used to represent the actor and the critic.
#         Deep convolutional Neural Network to represent both policy  (Actor) and a value function (Critic).
#         The Policy is parametrized using a Gaussian distribution with mean mu and variance sigma
#         The Actor's policy parameters (mu, sigma) and the Critic's Value (value) are output by the deep CNN implemented
#         in this class.
#         :param input_shape: Shape of each of the observations
#         :param actor_shape: Shape of the actor's output. Typically the shape of the actions
#         :param critic_shape: Shape of the Critic's output. Typically 1
#         :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
#         """
#         super(ActorCritic, self).__init__()
#         self.device = device
#         self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0),
#                                           torch.nn.ReLU())
#         self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=0),
#                                           torch.nn.ReLU())
#         self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1, padding=0),
#                                           torch.nn.ReLU())
#         self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 47 * 47, 512),
#                                           torch.nn.ReLU())
#         self.actor_mu = torch.nn.Linear(512, actor_shape)
#         self.actor_sigma = torch.nn.Linear(512, actor_shape)
#         self.critic = torch.nn.Linear(512, critic_shape)

#     def forward(self, x):
#         """
#         Forward pass through the Actor-Critic network. Takes batch_size x observations as input and produces
#         mu, sigma and the value estimate
#         as the outputs
#         :param x: The observations
#         :return: Mean (actor_mu), Sigma (actor_sigma) for a Gaussian policy and the Critic's value estimate (critic)
#         """
#         x.requires_grad_()
#         x = x.to(self.device)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = x.view(x.shape[0], -1)
#         x = self.layer4(x)
#         actor_mu = self.actor_mu(x)
#         actor_sigma = self.actor_sigma(x)
#         critic = self.critic(x)
#         return actor_mu, actor_sigma, critic
