import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    """
    MLP mapping states to action probabilities.
    """

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.state_dim = int(np.prod(state_space.shape))
        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_space.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class ValueNetwork(nn.Module):  # critic network
    """
    MLP mapping states to scalar value estimates.
    """

    def __init__(self, state_space: gym.spaces.Box, hidden_size: int = 128):
        super().__init__()
        self.state_dim = int(np.prod(state_space.shape))
        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)
