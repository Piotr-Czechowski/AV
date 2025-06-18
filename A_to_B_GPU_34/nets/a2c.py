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

##########################################################################################TO USUWAM
# class DiscreteActor(torch.nn.Module):
#     def __init__(self, input_shape, actor_shape, device=torch.device("cuda")):
#         """
#         Deep convolutional Neural Network to represent Actor in an Actor-Critic algorithm
#         The Policy is parametrized using a categorical/discrete distribution with logits
#         The Actor's policy parameters (logits) are output by the deep CNN implemented
#         in this class.
#         :param input_shape: Shape of each of the observations
#         :param actor_shape: Shape of the actor's output. Typically the shape of the actions
#         :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
#         """
#         super(DiscreteActor, self).__init__()
#         self.device = device
#         # self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0),
#                                                      #  How many channels
#         self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0),
#                                           torch.nn.ReLU())
#         self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=0),
#                                           torch.nn.ReLU())
#         # self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, stride=1, padding=0),
#         #                                  torch.nn.ReLU())
#         self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1, padding=0),
#                                           torch.nn.ReLU())
#         # self.layer4 = torch.nn.Sequential(torch.nn.Linear(128 * 7 * 7, 512),
#         #                                   torch.nn.ReLU())
#         self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 22 * 22, 512),
#                                           torch.nn.ReLU())
        
#         # self.speed_layer1 = torch.nn.Sequential(
#         #     torch.nn.Linear(1, 64),
#         #     torch.nn.ReLU())

#         # self.speed_layer2 = torch.nn.Sequential(
#         #     torch.nn.Linear(64, 64),
#         #     torch.nn.ReLU()
#         # )
#         self.actor = torch.nn.Linear(512, actor_shape)
#         # self.actor = torch.nn.Linear(512+64, actor_shape) # camera+speed

#         # self.scalar_layer = torch.nn.Sequential(
#         #             torch.nn.Linear(1, 64),
#         #             torch.nn.ReLU())
#         #powrzucać dropouty, batchnorm
#         # self.logits = torch.nn.Linear(2, actor_shape)

#     def forward(self, x, speed=None, manouver=None):
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
#         speed= None
#         if speed is None and manouver is None:
#             actor = self.actor(x)
#         elif speed is not None and manouver is None:
#             speed = speed.to(self.device).view(-1, 1)
#             speed = self.speed_layer1(speed)
#             # speed = self.speed_layer2(speed)
#             combined = torch.cat([x, speed], dim=1)
#             actor = self.actor(combined)
#         elif speed is not None and manouver is not None:
#             pass

#         # scalar = scalar.to(self.device).view(-1, 1)
#         # scalar_features = self.scalar_layer(scalar)
#         # combined = torch.cat([x, scalar_features], dim=1)
#         # logits = self.logits(combined)

#         # attention = self.attention_layer(combined)
#         # logits = self.logits(attention)
#         return actor

############################################################################################################# DOTAD USUWAM


######################################################################################## TO USUWAM
# class Critic(torch.nn.Module):
#     def __init__(self, input_shape, critic_shape=1, device=torch.device("cuda")):
#         """
#         Deep convolutional Neural Network to represent the Critic in an Actor-Critic algorithm
#         :param input_shape: Shape of each of the observations
#         :param critic_shape: Shape of the Critic's output. Typically 1
#         :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
#         """
#         super(Critic, self).__init__()
#         self.device = device
#         # input_shape[2] instead of 6
#         self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0),
#                                           torch.nn.ReLU())
#         self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=0),
#                                           torch.nn.ReLU())
#         self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1, padding=0),
#                                           torch.nn.ReLU())
#         self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 22 * 22, 512),
#                                           torch.nn.ReLU())
#         # self.speed_layer1 = torch.nn.Sequential(
#         #     torch.nn.Linear(1, 64),
#         #     torch.nn.ReLU()
#         #     )
# =======
#     def forward(self, x, speed=None, manouver=None):
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
#         # speed = None
#         if speed is None and manouver is None:
#             actor = self.actor(x)
#         elif speed is not None and manouver is None:
#             speed = speed.to(self.device).view(-1, 1)
#             speed = self.speed_layer1(speed)
#             # speed = self.speed_layer2(speed)
#             combined = torch.cat([x, speed], dim=1)
#             actor = self.actor(combined)
#         elif speed is not None and manouver is not None:
#             pass

#         # scalar = scalar.to(self.device).view(-1, 1)
#         # scalar_features = self.scalar_layer(scalar)
#         # combined = torch.cat([x, scalar_features], dim=1)
#         # logits = self.logits(combined)

#         # attention = self.attention_layer(combined)
#         # logits = self.logits(attention)
#         return actor


# class Critic(torch.nn.Module):
#     def __init__(self, input_shape, critic_shape=1, device=torch.device("cuda")):
#         """
#         Deep convolutional Neural Network to represent the Critic in an Actor-Critic algorithm
#         :param input_shape: Shape of each of the observations
#         :param critic_shape: Shape of the Critic's output. Typically 1
#         :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
#         """
#         super(Critic, self).__init__()
#         self.device = device
#         # input_shape[2] instead of 6
#         self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride=4, padding=0),
#                                           torch.nn.ReLU())
#         self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=0),
#                                           torch.nn.ReLU())
#         self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride=1, padding=0),
#                                           torch.nn.ReLU())
#         self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 22 * 22, 512),
#                                           torch.nn.ReLU())
#         self.speed_layer1 = torch.nn.Sequential(
#             torch.nn.Linear(1, 64),
#             torch.nn.ReLU()
#             )
# >>>>>>> main
        
#         # self.speed_layer2 = torch.nn.Sequential(
#         #     torch.nn.Linear(64, 64),
#         #     torch.nn.ReLU()
#         # )
#         # ----Last layers----
#         self.critic = torch.nn.Linear(512, critic_shape)
#         # self.critic = torch.nn.Linear(512+64, critic_shape)


#         # self.attention_layer = torch.nn.MultiheadAttention(512+64, 2)
#         # self.critic = torch.nn.Linear(2, critic_shape)


#     def forward(self, x, speed=None, manouver=None):
#         """
#         Forward pass through the Critic network. Takes batch_size x observations as input and produces the value
#         estimate as the output
#         :param x: The observations
#         :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
#         """
#         x.requires_grad_()
#         x = x.to(self.device)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = x.view(x.shape[0], -1)
#         # print(x.shape)
#         x = self.layer4(x)
#         speed = None
#         if speed is None and manouver is None:
#             critic = self.critic(x)
#         elif speed is not None and manouver is None:
#             speed = speed.to(self.device).view(-1, 1)
#             speed = self.speed_layer1(speed)
#             # speed = self.speed_layer2(speed)
#             combined = torch.cat([x, speed], dim=1)
#             critic = self.critic(combined)
#         elif speed is not None and manouver is not None:
#             pass

#         # print(x.shape)
#         # scalar = scalar.to(self.device).view(-1, 1)
#         # scalar_features = self.scalar_layer(scalar)
#         # combined = torch.cat([x, scalar_features], dim=1)
        
#         # critic = self.critic(combined)
#         # attention = self.attention_layer(combined)
#         # critic = self.critic(attention)

#         return critic

#################################################################DOTAD USUWAM


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

########################## NOWY MODEL, NIEZLY
# import torch.nn as nn

# class DiscreteActor(nn.Module):
#     def __init__(self, input_shape, actor_shape, device=torch.device("cuda")):
#         super(DiscreteActor, self).__init__()
#         self.device = device

#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool2d((4, 4))  # Spłaszcza do 4x4 niezależnie od wejścia
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 4 * 4, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, actor_shape)
#         )

#     def forward(self, x, speed=None, manouver=None):
#         x = x.to(self.device, dtype=torch.float32) / 255.0  # jeśli wejście to obraz (0–255)
#         x = self.cnn(x)
#         logits = self.fc(x)
#         return logits

# class Critic(nn.Module):
#     def __init__(self, input_shape, actor_shape, device=torch.device("cuda")):
#         super(Critic, self).__init__()
#         self.device = device

#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool2d((4, 4))  # Spłaszcza do 4x4 niezależnie od wejścia
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 4 * 4, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, actor_shape)
#         )

#     def forward(self, x, speed=None, manouver=None):
#         x = x.to(self.device, dtype=torch.float32) / 255.0  # jeśli wejście to obraz (0–255)
#         x = self.cnn(x)
#         logits = self.fc(x)
#         return logits

########################################### TEN CO WYZEJ ALE Z PREDKOSCIA
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DiscreteActor(nn.Module):
#     def __init__(self, input_shape, actor_shape, device=torch.device("cuda")):
#         super(DiscreteActor, self).__init__()
#         self.device = device

#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool2d((4, 4))  # Spłaszcza do 4x4 niezależnie od wejścia
#         )
        
#         # Gałąź przetwarzająca prędkość – oczekujemy tensor o wymiarze [B, 1]
#         self.speed_fc = nn.Sequential(
#             nn.Linear(1, 32),
#             nn.ReLU()
#         )

#         # self.manouver_fc = nn.Sequential(
#         #     nn.Linear(3, 32),  # zakładamy 4 możliwe manewry
#         #     nn.ReLU()
#         # )
        
#         # # Łączymy wyjście z CNN (flattenowane do 256*4*4) z przetworzoną prędkością (32)
#         # self.fc = nn.Sequential(
#         #     nn.Linear(256 * 4 * 4 + 32 + 32, 512),  # dodajemy +32 z manewru
#         #     nn.ReLU(),
#         #     nn.Dropout(0.3),
#         #     nn.Linear(512, actor_shape)
#         # )
#         in_features = 256 * 4 * 4 + 32

#         # Wspólna część (dla wszystkich manewrów)
#         self.shared_fc = nn.Sequential(
#             nn.Linear(in_features, 128),
#             nn.ReLU()
#         )

#         # Osobne głowy dla każdego manewru
#         self.head0 = nn.Linear(128, actor_shape)  # manewr 0
#         self.head1 = nn.Linear(128, actor_shape)  # manewr 1
#         self.head2 = nn.Linear(128, actor_shape)  # manewr 2

#     def forward(self, x, speed=None, manouver=None):
#         # Normalizacja obrazu oraz przetwarzanie przez CNN
#         x = x.to(self.device, dtype=torch.float32) / 255.0
#         cnn_features = self.cnn(x)
#         cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Flatten
        
#         # Obsługa speed: jeśli brak, ustawiamy tensor zerowy
#         if speed is None:
#             speed = torch.zeros((x.size(0), 1), device=self.device)
#         else:
#             speed = speed.to(self.device, dtype=torch.float32)
#             if speed.dim() == 1:
#                 speed = speed.unsqueeze(1)  # Upewnij się, że kształt to [B, 1]
        
#         speed_features = self.speed_fc(speed)

#         # manouver = F.one_hot(manouver, num_classes=3).float().to(self.device)
#         # manouver_features = self.manouver_fc(manouver)
        
#         # # Łączenie cech obrazu z cechami prędkości
#         # # combined = torch.cat([cnn_features, speed_features], dim=1)
#         # combined = torch.cat([cnn_features, speed_features, manouver_features], dim=1)

#         # logits = self.fc(combined)
#         # return logits

#         features = torch.cat([cnn_features, speed_features], dim=1)
        
#         shared_out = self.shared_fc(features)

#         # Zakładamy batch size = 1 (dla prostoty)
#         m = manouver.item()
#         if m == 0:
#             return self.head0(shared_out)
#         elif m == 1:
#             return self.head1(shared_out)
#         elif m == 2:
#             return self.head2(shared_out)
#         else:
#             raise ValueError(f"Nieznany manewr: {m}")



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Critic(nn.Module):
#     def __init__(self, input_shape, actor_shape, device=torch.device("cuda")):
#         super(Critic, self).__init__()
#         self.device = device

#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool2d((4, 4))  # Spłaszcza do 4x4 niezależnie od wejścia
#         )
        
#         # Gałąź do przetwarzania prędkości
#         self.speed_fc = nn.Sequential(
#             nn.Linear(1, 32),
#             nn.ReLU()
#         )

#         # self.manouver_fc = nn.Sequential(
#         #     nn.Linear(3, 32),  # zakładamy 4 możliwe manewry
#         #     nn.ReLU()
#         # )
#         # # Łączymy wyjście z CNN z informacją o prędkości.
#         # self.fc = nn.Sequential(
#         #     nn.Linear(256 * 4 * 4 + 32 + 32, 512),  # dodajemy +32 z manewru
#         #     nn.ReLU(),
#         #     nn.Dropout(0.3),
#         #     nn.Linear(512, actor_shape)
#         # )
#         in_features = 256 * 4 * 4 + 32

#         # Wspólna część (dla wszystkich manewrów)
#         self.shared_fc = nn.Sequential(
#             nn.Linear(in_features, 128),
#             nn.ReLU()
#         )

#         # Osobne głowy dla każdego manewru
#         self.head0 = nn.Linear(128, actor_shape)  # manewr 0
#         self.head1 = nn.Linear(128, actor_shape)  # manewr 1
#         self.head2 = nn.Linear(128, actor_shape)  # manewr 2

#     def forward(self, x, speed=None, manouver=None):
#         # Przetwarzanie obrazu oraz normalizacja
#         x = x.to(self.device, dtype=torch.float32) / 255.0
        
#         cnn_features = self.cnn(x)
#         cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
#         # Obsługa wejścia dla prędkości
#         if speed is None:
#             speed = torch.zeros((x.size(0), 1), device=self.device)
#         else:
#             speed = speed.to(self.device, dtype=torch.float32)
#             if speed.dim() == 1:
#                 speed = speed.unsqueeze(1)
        
#         speed_features = self.speed_fc(speed)
#         # manouver = F.one_hot(manouver, num_classes=3).float().to(self.device)
#         # manouver_features = self.manouver_fc(manouver)  
        
#         # # Połączenie cech obrazu z cechami prędkości
#         # # combined = torch.cat([cnn_features, speed_features], dim=1)
#         # combined = torch.cat([cnn_features, speed_features, manouver_features], dim=1)

#         # logits = self.fc(combined)
#         # return logits
#         features = torch.cat([cnn_features, speed_features], dim=1)
        
#         shared_out = self.shared_fc(features)

#         # Zakładamy batch size = 1 (dla prostoty)
#         m = manouver.item()
#         if m == 0:
#             return self.head0(shared_out)
#         elif m == 1:
#             return self.head1(shared_out)
#         elif m == 2:
#             return self.head2(shared_out)
#         else:
#             raise ValueError(f"Nieznany manewr: {m}")


##### Attention is all you need:
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteActor(nn.Module):
    def __init__(self, input_shape, actor_shape, embed_dim=128, num_heads=4, device=torch.device("cuda")):
        super(DiscreteActor, self).__init__()
        self.device = device
        self.embed_dim = embed_dim

        # CNN do ekstrakcji cech obrazu
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.cnn_out_dim = 128 * 4 * 4
        self.image_fc = nn.Linear(self.cnn_out_dim, embed_dim)
        self.speed_fc = nn.Linear(1, embed_dim)
        self.manouver_fc = nn.Embedding(3, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, actor_shape)
        )

    def forward(self, x, speed, manouver):
        # x: [B, C, H, W]
        B = x.size(0)
        x = x.to(self.device, dtype=torch.float32) / 255.0
        cnn_features = self.cnn(x).view(B, -1)
        image_emb = self.image_fc(cnn_features)  # [B, embed_dim]

        speed = speed.to(self.device, dtype=torch.float32).view(B, 1)
        speed_emb = self.speed_fc(speed)  # [B, embed_dim]

        manouver = manouver.to(self.device, dtype=torch.long)
        manouver_emb = self.manouver_fc(manouver)  # [B, embed_dim]

        # Tokeny: [B, 3, embed_dim]
        tokens = torch.stack([image_emb, speed_emb, manouver_emb], dim=1)

        # Self-attention
        attn_out, _ = self.attention(tokens, tokens, tokens)  # [B, 3, embed_dim]
        summary = attn_out[:, 0, :]  # Można też spróbować mean(attn_out, dim=1)

        return self.actor_head(summary)
    

class Critic(nn.Module):
    def __init__(self, input_shape, critic_shape, embed_dim=128, num_heads=4, device=torch.device("cuda")):
        super(Critic, self).__init__()
        self.device = device
        self.embed_dim = embed_dim

        # CNN do ekstrakcji cech obrazu
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.cnn_out_dim = 128 * 4 * 4
        self.image_fc = nn.Linear(self.cnn_out_dim, embed_dim)
        self.speed_fc = nn.Linear(1, embed_dim)
        self.manouver_fc = nn.Embedding(3, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, critic_shape)
        )

    def forward(self, x, speed, manouver):
        # x: [B, C, H, W]
        B = x.size(0)
        x = x.to(self.device, dtype=torch.float32) / 255.0
        cnn_features = self.cnn(x).view(B, -1)
        image_emb = self.image_fc(cnn_features)  # [B, embed_dim]

        speed = speed.to(self.device, dtype=torch.float32).view(B, 1)
        speed_emb = self.speed_fc(speed)  # [B, embed_dim]

        manouver = manouver.to(self.device, dtype=torch.long)
        manouver_emb = self.manouver_fc(manouver)  # [B, embed_dim]

        # Tokeny: [B, 3, embed_dim]
        tokens = torch.stack([image_emb, speed_emb, manouver_emb], dim=1)

        # Self-attention
        attn_out, _ = self.attention(tokens, tokens, tokens)  # [B, 3, embed_dim]
        summary = attn_out[:, 0, :]  # Można też spróbować mean(attn_out, dim=1)

        return self.critic_head(summary)