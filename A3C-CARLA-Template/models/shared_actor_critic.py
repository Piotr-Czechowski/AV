"""Shared actor-critic used by Hogwild A3C workers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedActorCritic(nn.Module):
    """Batch-size-1 friendly actor-critic with one shared visual trunk."""

    def __init__(self, input_shape, action_shape, critic_shape=1,
                 device=torch.device('cpu'), num_maneuvers=3):
        super().__init__()
        self.device = device
        self.num_maneuvers = num_maneuvers
        in_channels = int(input_shape[2])

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
        )
        self.maneuver_fc = nn.Sequential(
            nn.Linear(num_maneuvers, 32),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 32 + 32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(256, action_shape)
        self.value = nn.Linear(256, critic_shape)

    def forward(self, x, speed=None, maneuver=None):
        """Return ``(policy_logits, value)`` for a CHW image batch plus speed and maneuver."""
        x = x.to(self.device, dtype=torch.float32)
        features = self.cnn(x).flatten(1)

        if speed is None:
            speed = torch.zeros((x.size(0), 1), device=self.device)
        else:
            speed = speed.to(self.device, dtype=torch.float32) \
                .view(x.size(0), -1)
            if speed.size(1) != 1:
                speed = speed[:, :1]
        speed_features = self.speed_fc(speed)

        if maneuver is None:
            maneuver = torch.ones((x.size(0),), dtype=torch.long,
                                  device=self.device)
        else:
            maneuver = maneuver.to(self.device, dtype=torch.long).view(-1)
        maneuver = maneuver.clamp(0, self.num_maneuvers - 1)
        maneuver = F.one_hot(maneuver,
                             num_classes=self.num_maneuvers).float()
        maneuver_features = self.maneuver_fc(maneuver)

        hidden = self.trunk(torch.cat(
            [features, speed_features, maneuver_features], dim=1))
        return self.policy(hidden), self.value(hidden)
