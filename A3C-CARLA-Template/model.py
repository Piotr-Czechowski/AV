"""The network that trains: ``SharedActorCritic`` and ``build_model``.

``a3c_core`` builds the shared global model and every worker's local copy
through ``build_model``. To train another network, add its class to this file
and return it from ``build_model``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedActorCritic(nn.Module):
    """Batch-size-1 friendly actor-critic with one shared visual trunk."""

    def __init__(self, obs_spec, n_actions, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        in_channels = int(obs_spec['image']['shape'][0])
        speed_size = int(obs_spec['speed']['shape'][0])
        self.num_maneuvers = int(obs_spec['maneuver']['num_classes'])

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
            nn.Linear(speed_size, 32),
            nn.ReLU(inplace=True),
        )
        self.maneuver_fc = nn.Sequential(
            nn.Linear(self.num_maneuvers, 32),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 32 + 32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, obs):
        """Return ``(policy_logits [B, n_actions], value [B, 1])``.

        ``obs`` is the batched observation dict from
        ``a3c_core.obs_to_tensors``: ``image`` [B, C, H, W], ``speed``
        [B, 1], ``maneuver`` [B].
        """
        x = obs['image'].to(self.device, dtype=torch.float32)
        features = self.cnn(x).flatten(1)

        speed = obs['speed'].to(self.device, dtype=torch.float32) \
            .view(x.size(0), -1)
        speed_features = self.speed_fc(speed)

        maneuver = obs['maneuver'].to(self.device, dtype=torch.long).view(-1)
        maneuver = maneuver.clamp(0, self.num_maneuvers - 1)
        maneuver = F.one_hot(maneuver,
                             num_classes=self.num_maneuvers).float()
        maneuver_features = self.maneuver_fc(maneuver)

        hidden = self.trunk(torch.cat(
            [features, speed_features, maneuver_features], dim=1))
        return self.policy(hidden), self.value(hidden)


def build_model(obs_spec, n_actions, device):
    """Return a new, untrained network on ``device``.

    ``obs_spec`` comes from ``rl_configuration.observation_spec``.
    """
    return SharedActorCritic(obs_spec, n_actions, device).to(device)
