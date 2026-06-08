"""
Tests for TargetNetwork and PredictorNetwork in rnd_networks.py.

Verifies:
 - Correct output shapes for batch and single inputs.
 - TargetNetwork has all parameters frozen.
 - PredictorNetwork has all parameters trainable and gradients flow.
 - Dynamic layer construction for variable n_layers.
 - DualHeadValueNetwork output shapes and head independence.
 - RewardForwardFilter discounted accumulation.
"""

import unittest

import numpy as np
import torch
from rl_exercises.week_7.rnd_utils import (
    DualHeadValueNetwork,
    PredictorNetwork,
    RewardForwardFilter,
    TargetNetwork,
)


class TestTargetNetwork(unittest.TestCase):
    def test_output_shape_batch(self):
        """Forward pass on a batch gives (batch, output_dim)."""
        net = TargetNetwork(obs_dim=4, output_dim=8, hidden_dim=16, n_layers=2)
        out = net(torch.zeros(3, 4))
        self.assertEqual(out.shape, (3, 8))

    def test_output_shape_single(self):
        """Forward pass on a single sample gives (1, output_dim)."""
        net = TargetNetwork(obs_dim=4, output_dim=8, hidden_dim=16, n_layers=1)
        out = net(torch.zeros(1, 4))
        self.assertEqual(out.shape, (1, 8))

    def test_all_parameters_frozen(self):
        """Every parameter in TargetNetwork must have requires_grad=False."""
        net = TargetNetwork(obs_dim=4, output_dim=8)
        for name, param in net.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"Parameter {name} should be frozen in TargetNetwork",
            )

    def test_n_layers_1(self):
        """Works correctly with a single hidden layer."""
        net = TargetNetwork(obs_dim=4, output_dim=8, hidden_dim=16, n_layers=1)
        self.assertEqual(net(torch.zeros(2, 4)).shape, (2, 8))

    def test_n_layers_3(self):
        """Works correctly with three hidden layers."""
        net = TargetNetwork(obs_dim=4, output_dim=8, hidden_dim=16, n_layers=3)
        self.assertEqual(net(torch.zeros(2, 4)).shape, (2, 8))


class TestPredictorNetwork(unittest.TestCase):
    def test_output_shape_batch(self):
        """Forward pass on a batch gives (batch, output_dim)."""
        net = PredictorNetwork(obs_dim=4, output_dim=8, hidden_dim=16, n_layers=2)
        out = net(torch.zeros(3, 4))
        self.assertEqual(out.shape, (3, 8))

    def test_all_parameters_trainable(self):
        """Every parameter in PredictorNetwork must have requires_grad=True."""
        net = PredictorNetwork(obs_dim=4, output_dim=8)
        for name, param in net.named_parameters():
            self.assertTrue(
                param.requires_grad,
                f"Parameter {name} should be trainable in PredictorNetwork",
            )

    def test_gradient_flows_through_all_params(self):
        """Gradients must reach every parameter after a backward pass."""
        net = PredictorNetwork(obs_dim=4, output_dim=8, hidden_dim=16, n_layers=2)
        net(torch.randn(2, 4)).sum().backward()
        for name, param in net.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")

    def test_n_layers_1(self):
        """Works correctly with a single hidden layer."""
        net = PredictorNetwork(obs_dim=4, output_dim=8, hidden_dim=16, n_layers=1)
        self.assertEqual(net(torch.zeros(2, 4)).shape, (2, 8))


class TestDualHeadValueNetwork(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = DualHeadValueNetwork(state_dim=4, hidden_size=16)

    def test_returns_two_tensors(self):
        result = self.net(torch.zeros(3, 4))
        self.assertEqual(len(result), 2)

    def test_output_shapes_batch(self):
        val_ext, val_int = self.net(torch.zeros(3, 4))
        self.assertEqual(val_ext.shape, (3,))
        self.assertEqual(val_int.shape, (3,))

    def test_output_shapes_single_obs(self):
        """1D input (single observation) should be handled without error."""
        val_ext, val_int = self.net(torch.zeros(4))
        self.assertEqual(val_ext.shape, (1,))
        self.assertEqual(val_int.shape, (1,))

    def test_two_heads_independent(self):
        """Ext and int heads have separate weights, so outputs should generally differ."""
        torch.manual_seed(42)
        net = DualHeadValueNetwork(state_dim=4, hidden_size=32)
        val_ext, val_int = net(torch.randn(10, 4))
        self.assertFalse(
            torch.allclose(val_ext, val_int),
            "Ext and int heads should produce different values (independent parameters)",
        )


class TestRewardForwardFilter(unittest.TestCase):
    def test_first_update_equals_input(self):
        """First call initializes rewems to the input."""
        filt = RewardForwardFilter(gamma=0.99)
        result = filt.update(np.array([2.0]))
        np.testing.assert_allclose(result, np.array([2.0]))

    def test_discounted_accumulation(self):
        """Second update: rewems = gamma * prev_rewems + new_rews."""
        filt = RewardForwardFilter(gamma=0.5)
        filt.update(np.array([2.0]))  # rewems = 2.0
        result = filt.update(np.array([1.0]))  # rewems = 0.5 * 2.0 + 1.0 = 2.0
        np.testing.assert_allclose(result, np.array([2.0]))

    def test_gamma_zero_discards_history(self):
        """With gamma=0, only the current reward matters."""
        filt = RewardForwardFilter(gamma=0.0)
        filt.update(np.array([99.0]))
        result = filt.update(np.array([1.0]))
        np.testing.assert_allclose(result, np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
