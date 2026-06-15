from collections import OrderedDict

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    A simple MLP mapping state → Q‐values for each action.

    Architecture:
      Input → Linear(obs_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→n_actions)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64, hidden_sizes: list = None) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        n_actions : int
            Number of discrete actions.
        hidden_dim : int
            Hidden layer size (default).
        hidden_sizes : list, optional
            Custom architecture. E.g., [32, 32] or [128, 128, 128].
            If provided, creates network with specified layer sizes.
        """
        super().__init__()
        
        # Support custom network architecture for experimentation
        if hidden_sizes is not None:
            layers = []
            prev_dim = obs_dim
            for size in hidden_sizes:
                layers.append(nn.Linear(prev_dim, size))
                layers.append(nn.ReLU())
                prev_dim = size
            layers.append(nn.Linear(prev_dim, n_actions))
            self.net = nn.Sequential(*layers)
        else:
            # Default architecture: 2 hidden layers with hidden_dim units
            self.net = nn.Sequential(
                OrderedDict(
                    [
                        ("fc1", nn.Linear(obs_dim, hidden_dim)),
                        ("relu1", nn.ReLU()),
                        ("fc2", nn.Linear(hidden_dim, hidden_dim)),
                        ("relu2", nn.ReLU()),
                        ("out", nn.Linear(hidden_dim, n_actions)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states, shape (batch, obs_dim).

        Returns
        -------
        torch.Tensor
            Q‐values, shape (batch, n_actions).
        """
        return self.net(x)
