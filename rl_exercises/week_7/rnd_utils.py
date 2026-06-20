from typing import Tuple

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401


class DualHeadValueNetwork(nn.Module):
    """
    Value network with dual heads: one for extrinsic rewards, one for intrinsic rewards.
    """

    def __init__(self, state_dim: int, hidden_size: int = 128) -> None:
        """
        Initialize dual-head value network.

        Parameters
        ----------
        state_dim : int
            Dimensionality of observation space (flat integer).
        hidden_size : int
            Hidden layer size.
        """
        super().__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc_ext = nn.Linear(hidden_size, 1)
        self.fc_int = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both extrinsic and intrinsic value estimates.

        Parameters
        ----------
        x : torch.Tensor
            State tensor, shape (batch_size, state_dim) or (state_dim,).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Extrinsic value and intrinsic value estimates.
        """
        # TODO: Apply forward pass through shared layers and both heads
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        x = ...
        value_ext = ...
        value_int = ...
        return value_ext, value_int


class TargetNetwork(nn.Module):
    """
    A simple frozen random initialized MLP target network for RND with variable layers.

    Architecture:
      Input → Linear(obs_dim→hidden_dim) → ReLU → ... → Linear(hidden_dim→output_dim)
    """

    def __init__(
        self, obs_dim: int, output_dim: int, hidden_dim: int = 64, n_layers: int = 2
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        output_dim : int
            Dimensionality of the output space.
        hidden_dim : int
            Hidden layer size.
        n_layers : int
            Number of hidden layers, by default 2.
        """
        super().__init__()
        layers = OrderedDict()

        # TODO: build hidden layers dynamically based on n_layers, using nn.Linear and nn.ReLU
        for i in range(n_layers):
            in_dim = ...
            layers[f"fc{i + 1}"] = ...
            layers[f"relu{i + 1}"] = ...

        # TODO: output layer mapping hidden_dim to output_dim
        layers["out"] = ...

        # TODO: combine all layers into self.net using nn.Sequential
        self.net = ...

        # TODO: freeze all parameters permanently (no training)
        for param in self.parameters():
            param.requires_grad = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Apply forward pass
        return ...


class PredictorNetwork(nn.Module):
    """
    A simple MLP predictor network for RND, trained to predict the output of the TargetNetwork.

    Architecture (copy of TargetNetwork with variable layers):
      Input → Linear(obs_dim→hidden_dim) → ReLU → ... → Linear(hidden_dim→output_dim)
    """

    def __init__(
        self, obs_dim: int, output_dim: int, hidden_dim: int = 64, n_layers: int = 2
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        output_dim : int
            Dimensionality of the output space.
        hidden_dim : int
            Hidden layer size.
        n_layers : int
            Number of hidden layers, by default 2.
        """
        super().__init__()
        layers = OrderedDict()

        # TODO: Build hidden layers dynamically based on n_layers, using nn.Linear and nn.ReLU (same architecture as TargetNetwork)
        for i in range(n_layers):
            in_dim = ...
            layers[f"fc{i + 1}"] = ...
            layers[f"relu{i + 1}"] = ...

        # TODO: output layer mapping hidden_dim to output_dim
        layers["out"] = ...

        # TODO: combine all layers into self.net using nn.Sequential
        self.net = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Apply forward pass
        return ...


class RewardForwardFilter:
    """Maintains a discounted running sum of rewards for intrinsic reward normalization."""

    def __init__(self, gamma: float) -> None:
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
