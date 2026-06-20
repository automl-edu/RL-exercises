"""
Tests for RNDPPOAgent in rnd_ppo.py.

Verifies:
 - Inherits from PPOAgent.
 - predict() returns a 5-tuple with the correct types.
 - get_rnd_bonus() returns a non-negative float and scales with rnd_reward_weight.
 - compute_gae() produces tensors of the right shape.
 - Combined advantages from compute_gae() are zero-mean and unit-std.
 - Intrinsic GAE stream is non-episodic (ignores done mask).
 - update() returns four finite float losses.
 - evaluate() returns correct results on a deterministic environment.
"""

import unittest

import gymnasium as gym
import numpy as np
import torch
from rl_exercises.week_6 import PPOAgent
from rl_exercises.week_7.rnd_ppo import RNDPPOAgent


class DummyEnv(gym.Env):
    """
    Trivial 4-obs, 2-action environment that always returns reward=1 and terminates.
    Used for deterministic unit tests.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 1.0, True, False, {}


class TestRNDPPOAgent(unittest.TestCase):
    def setUp(self):
        self.env = DummyEnv()
        self.agent = RNDPPOAgent(
            self.env,
            epochs=1,
            batch_size=4,
            rnd_hidden_size=8,
            rnd_n_layers=1,
            num_iterations_obs_norm_init=1,
            seed=0,
        )

    def _make_trajectory(self, length=8):
        """
        Build a dummy trajectory of the format expected by update():
        (state, action, logp, entropy, ext_reward, int_reward, done, next_state)
        """
        state, _ = self.env.reset()
        traj = []
        for _ in range(length):
            action, logp, ent, _, _ = self.agent.predict(state)
            next_state, ext_reward, term, trunc, _ = self.env.step(action)
            done = float(term or trunc)
            int_reward = self.agent.get_rnd_bonus(state.astype(np.float32))
            traj.append(
                (
                    state,
                    action,
                    logp,
                    ent,
                    float(ext_reward),
                    float(int_reward),
                    done,
                    next_state,
                )
            )
            state = next_state if not done else self.env.reset()[0]
        return traj

    # -------------------------------------------------------------------------
    # Inheritance
    # -------------------------------------------------------------------------

    def test_inherits_from_ppo(self):
        self.assertIsInstance(self.agent, PPOAgent)

    # -------------------------------------------------------------------------
    # predict()
    # -------------------------------------------------------------------------

    def test_predict_returns_five_tuple(self):
        """predict() must return (action, logp, entropy, val_ext, val_int)."""
        state, _ = self.env.reset(seed=0)
        result = self.agent.predict(state)
        self.assertEqual(len(result), 5)
        action, logp, ent, val_ext, val_int = result
        self.assertIsInstance(action, int)
        self.assertIsInstance(logp, torch.Tensor)
        self.assertIsInstance(ent, torch.Tensor)
        self.assertIsInstance(val_ext, torch.Tensor)
        self.assertIsInstance(val_int, torch.Tensor)

    def test_predict_action_in_action_space(self):
        state, _ = self.env.reset(seed=0)
        action, *_ = self.agent.predict(state)
        self.assertIn(action, range(self.env.action_space.n))

    # -------------------------------------------------------------------------
    # get_rnd_bonus()
    # -------------------------------------------------------------------------

    def test_get_rnd_bonus_is_nonneg_float(self):
        state, _ = self.env.reset(seed=0)
        bonus = self.agent.get_rnd_bonus(state.astype(np.float32))
        self.assertIsInstance(bonus, float)
        self.assertGreaterEqual(bonus, 0.0)

    def test_rnd_reward_weight_zero_gives_zero_bonus(self):
        """rnd_reward_weight=0 must produce exactly 0 bonus."""
        agent_zero = RNDPPOAgent(
            self.env,
            rnd_hidden_size=8,
            rnd_n_layers=1,
            rnd_reward_weight=0.0,
            num_iterations_obs_norm_init=1,
            seed=1,
        )
        state, _ = self.env.reset()
        self.assertEqual(agent_zero.get_rnd_bonus(state.astype(np.float32)), 0.0)

    # -------------------------------------------------------------------------
    # compute_gae()
    # -------------------------------------------------------------------------

    def _gae_inputs(self, T=6, dones=None):
        dones = dones if dones is not None else torch.zeros(T)
        return (
            [1.0] * T,  # rewards_ext
            [0.5] * T,  # rewards_int
            torch.zeros(T),  # values_ext
            torch.zeros(T),  # values_int
            torch.zeros(T),  # next_values_ext
            torch.zeros(T),  # next_values_int
            dones,
        )

    def test_compute_gae_output_shapes(self):
        T = 6
        combined, adv_ext, adv_int, ret_ext, ret_int = self.agent.compute_gae(
            *self._gae_inputs(T)
        )
        for name, tensor in zip(
            ("combined", "adv_ext", "adv_int", "ret_ext", "ret_int"),
            (combined, adv_ext, adv_int, ret_ext, ret_int),
        ):
            self.assertEqual(tensor.shape, (T,), f"{name} has wrong shape")

    def test_compute_gae_combined_normalized(self):
        """Combined advantages must be zero-mean and unit-std after normalization."""
        T = 8
        rewards_ext = list(range(T))
        rewards_int = [r * 0.1 for r in rewards_ext]
        combined, *_ = self.agent.compute_gae(
            rewards_ext,
            rewards_int,
            torch.zeros(T),
            torch.zeros(T),
            torch.zeros(T),
            torch.zeros(T),
            torch.zeros(T),
        )
        self.assertAlmostEqual(combined.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(combined.std(unbiased=False).item(), 1.0, places=6)

    def test_compute_gae_intrinsic_non_episodic(self):
        """
        Intrinsic GAE must ignore the done mask.
        At the step immediately before a done, the intrinsic advantage should be
        higher than the extrinsic advantage (which is cut by the done mask).
        """
        T = 3
        # done at step 1: cuts extrinsic lookahead, but intrinsic keeps bootstrapping
        dones = torch.tensor([0.0, 1.0, 0.0])
        _, adv_ext, adv_int, _, _ = self.agent.compute_gae(
            [1.0] * T,
            [1.0] * T,
            torch.zeros(T),
            torch.zeros(T),
            torch.zeros(T),
            torch.zeros(T),
            dones,
        )
        self.assertGreater(
            adv_int[1].item(),
            adv_ext[1].item(),
            "adv_int[done_step] should exceed adv_ext[done_step] for non-episodic intrinsic stream",
        )

    # -------------------------------------------------------------------------
    # update()
    # -------------------------------------------------------------------------

    def test_update_returns_four_finite_losses(self):
        """update() must return (policy_loss, value_loss, entropy_loss, rnd_loss) — all finite."""
        traj = self._make_trajectory(length=8)
        losses = self.agent.update(traj)
        self.assertEqual(len(losses), 4)
        for i, loss in enumerate(losses):
            self.assertIsInstance(loss, float, f"loss[{i}] is not a float")
            self.assertTrue(np.isfinite(loss), f"loss[{i}]={loss} is not finite")

    # -------------------------------------------------------------------------
    # evaluate()
    # -------------------------------------------------------------------------

    def test_evaluate_dummy_env(self):
        """
        On DummyEnv every episode yields reward=1.0, so mean=1.0 and std=0.0.
        """
        mean, std = self.agent.evaluate(self.env, num_episodes=4)
        self.assertAlmostEqual(mean, 1.0, places=6)
        self.assertAlmostEqual(std, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
