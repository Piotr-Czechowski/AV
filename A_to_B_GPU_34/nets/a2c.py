"""
Author: Praveen Palanisamy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# class DiscreteActor(nn.Module):
#     def __init__(self, input_shape, actor_shape, device=torch.device("cpu")):
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

#             # --- ADDED CONV LAYER ---
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             # ------------------------

#             nn.AdaptiveAvgPool2d((4, 4))  # Flattens to 4x4 regardless of input
#         )
        
#         # Branch for processing speed - expects a tensor of shape [B, 1]
#         self.speed_fc = nn.Sequential(
#             nn.Linear(1, 32),
#             nn.ReLU()
#         )

#         # Branch for processing maneuver
#         self.manouver_fc = nn.Sequential(
#             nn.Linear(3, 32),  # Assuming 3 possible maneuvers
#             nn.ReLU()
#         )
        
#         # We combine the output from CNN (flattened to 512*4*4) with processed speed (32) and maneuver (32)
#         # The input size is updated to reflect the new conv layer's output channels (512)
#         self.fc = nn.Sequential(
#             nn.Linear(512 * 4 * 4 + 32 + 32, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
            
#             # --- ADDED LINEAR LAYER ---
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             # --------------------------

#             nn.Linear(256, actor_shape)
#         )

#     def forward(self, x, speed=None, manouver=None):
#         # Normalize the image and process through CNN
#         # x = x.to(self.device, dtype=torch.float32) / 255.0
#         x = x.to(self.device, dtype=torch.float32)

#         cnn_features = self.cnn(x)
#         cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Flatten
        
#         # Handle speed: if none, set to a zero tensor
#         if speed is None:
#             speed = torch.zeros((x.size(0), 1), device=self.device)
#         else:
#             speed = speed.to(self.device, dtype=torch.float32)
#             if speed.dim() == 1:
#                 speed = speed.unsqueeze(1)  # Ensure shape is [B, 1]
        
#         speed_features = self.speed_fc(speed)

#         # One-hot encode the maneuver
#         manouver = F.one_hot(manouver, num_classes=3).float().to(self.device)
#         manouver_features = self.manouver_fc(manouver)
        
#         # Combine image, speed, and maneuver features
#         combined = torch.cat([cnn_features, speed_features, manouver_features], dim=1)

#         logits = self.fc(combined)
#         return logits
    
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Critic(nn.Module):
#     def __init__(self, input_shape, actor_shape, device=torch.device("cpu")):
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

#             # --- ADDED CONV LAYER ---
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             # ------------------------

#             nn.AdaptiveAvgPool2d((4, 4))  # Flattens to 4x4 regardless of input
#         )
        
#         # Branch for processing speed - expects a tensor of shape [B, 1]
#         self.speed_fc = nn.Sequential(
#             nn.Linear(1, 32),
#             nn.ReLU()
#         )

#         # Branch for processing maneuver
#         self.manouver_fc = nn.Sequential(
#             nn.Linear(3, 32),  # Assuming 3 possible maneuvers
#             nn.ReLU()
#         )
        
#         # We combine the output from CNN (flattened to 512*4*4) with processed speed (32) and maneuver (32)
#         # The input size is updated to reflect the new conv layer's output channels (512)
#         self.fc = nn.Sequential(
#             nn.Linear(512 * 4 * 4 + 32 + 32, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
            
#             # --- ADDED LINEAR LAYER ---
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             # --------------------------

#             nn.Linear(256, actor_shape)
#         )

#     def forward(self, x, speed=None, manouver=None):
#         # Normalize the image and process through CNN
#         # x = x.to(self.device, dtype=torch.float32) / 255.0
#         x = x.to(self.device, dtype=torch.float32)

        
#         cnn_features = self.cnn(x)
#         cnn_features = cnn_features.view(x.size(0), -1)  # Flatten
        
#         # Handle speed: if none, set to a zero tensor
#         if speed is None:
#             speed = torch.zeros((x.size(0), 1), device=self.device)
#         else:
#             speed = speed.to(self.device, dtype=torch.float32)
#             if speed.dim() == 1:
#                 speed = speed.unsqueeze(1)  # Ensure shape is [B, 1]
        
#         speed_features = self.speed_fc(speed)

#         # One-hot encode the maneuver
#         manouver = F.one_hot(manouver, num_classes=3).float().to(self.device)
#         manouver_features = self.manouver_fc(manouver)
        
#         # Combine image, speed, and maneuver features
#         combined = torch.cat([cnn_features, speed_features, manouver_features], dim=1)

#         logits = self.fc(combined)
#         return logits
    

"""
ActorCritic – wspólny trunk CNN dla aktora i krytyka.

Zastępuje oddzielne klasy DiscreteActor i Critic z nets/a2c.py.
Architektura jest identyczna jak oryginalna (te same warstwy, te same rozmiary),
ale CNN, speed_fc i manouver_fc są współdzielone – forward pass przez CNN
wykonywany jest tylko RAZ na krok.

Użycie (zamiennik dla a3c_improved_1.py):
    # zamiast:
    #   self.actor  = DeepDiscreteActor(state_shape, action_shape, device)
    #   self.critic = DeepCritic(state_shape, critic_shape, device)
    # piszemy:
    self.model = ActorCritic(state_shape, action_shape, device=device)

    # forward:
    logits, value = self.model(obs, speed, manouver)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic.

    Parametry
    ---------
    input_shape  : [H, W, C]  – np. [250, 250, 3]
    action_shape : int        – liczba dyskretnych akcji
    critic_shape : int        – zwykle 1 (skalarna wartość stanu)
    device       : torch.device lub str
    num_maneuvers: int        – liczba możliwych manewrów (domyślnie 3)
    """

    def __init__(self, input_shape, action_shape, critic_shape=1,
                 device=torch.device("cpu"), num_maneuvers=3):
        super().__init__()
        self.device = torch.device(device)
        in_channels = int(input_shape[2])

        # ------------------------------------------------------------------ #
        #  Wspólny trunk  (identyczny z oryginalnym CNN w DiscreteActor/Critic)
        # ------------------------------------------------------------------ #
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),   # → [B, 512, 4, 4]
        )

        self.speed_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
        )

        self.manouver_fc = nn.Sequential(
            nn.Linear(num_maneuvers, 32),
            nn.ReLU(),
        )

        # Wspólna warstwa przetwarzająca połączone cechy
        trunk_out = 512 * 4 * 4 + 32 + 32   # 8256
        self.trunk = nn.Sequential(
            nn.Linear(trunk_out, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------ #
        #  Oddzielne głowice
        # ------------------------------------------------------------------ #
        self.policy_head = nn.Linear(256, action_shape)   # logity akcji
        self.value_head  = nn.Linear(256, critic_shape)   # V(s)

        self.num_maneuvers = num_maneuvers

    # ---------------------------------------------------------------------- #

    def forward(self, x, speed=None, manouver=None):
        """
        Zwraca (logits, value) w jednym forward passie.

        x        : [B, H, W, C] lub [B, C, H, W] – float32, znormalizowane
        speed    : [B, 1] lub [B] lub None
        manouver : [B] LongTensor (indeksy manewrów) lub None
        """
        x = x.to(self.device, dtype=torch.float32)

        # Obsługa układu kanałów: [B, H, W, C] → [B, C, H, W]
        if x.dim() == 4 and x.shape[1] != self.cnn[0].in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()

        # CNN trunk – JEDEN forward pass
        cnn_out = self.cnn(x).flatten(1)   # [B, 512*4*4]

        # Speed
        if speed is None:
            speed = torch.zeros((x.size(0), 1), device=self.device)
        else:
            speed = speed.to(self.device, dtype=torch.float32)
            if speed.dim() == 1:
                speed = speed.unsqueeze(1)
            elif speed.shape[1] != 1:
                speed = speed[:, :1]
        speed_feat = self.speed_fc(speed)   # [B, 32]

        # Manewry – one-hot
        if manouver is None:
            manouver = torch.ones(x.size(0), dtype=torch.long,
                                  device=self.device)
        else:
            manouver = manouver.to(self.device, dtype=torch.long).view(-1)
        manouver = manouver.clamp(0, self.num_maneuvers - 1)
        manouver_oh = F.one_hot(manouver,
                                num_classes=self.num_maneuvers).float()
        manouver_feat = self.manouver_fc(manouver_oh)   # [B, 32]

        # Wspólny trunk FC
        hidden = self.trunk(
            torch.cat([cnn_out, speed_feat, manouver_feat], dim=1)
        )   # [B, 256]

        logits = self.policy_head(hidden)   # [B, action_shape]
        value  = self.value_head(hidden)    # [B, critic_shape]

        return logits, value