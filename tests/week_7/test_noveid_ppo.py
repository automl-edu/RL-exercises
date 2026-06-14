"""
Tests for NovelDPPOAgent in noveid_ppo.py.

Verifies:
 - Inherits from PPOAgent.
 - predict() returns a 5-tuple with the correct types.
 - _rnd_error() returns a non-negative float.
 - _is_first_visit() marks state as visited and resets between episodes.
 - get_noveld_bonus() is non-negative and returns 0 on revisit.
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
from rl_exercises.week_7.noveid_ppo import NovelDPPOAgent


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


class TestNovelDPPOAgent(unittest.TestCase):
    def setUp(self):
        self.env = DummyEnv()
        self.agent = NovelDPPOAgent(
            self.env,
            epochs=1,
            batch_size=4,
            rnd_hidden_size=8,
            rnd_n_layers=1,
            noveld_alpha=1.0,
            num_iterations_obs_norm_init=1,
            seed=0,
        )

    def _make_trajectory(self, length=8):
        """
        Build a dummy trajectory in the format expected by update():
        (state, action, logp, entropy, ext_reward, int_reward, done, next_state)
        """
        state, _ = self.env.reset()
        self.agent._episode_visited = set()
        prev_obs_norm = state.astype(np.float32)
        traj = []
        for _ in range(length):
            action, logp, ent, _, _ = self.agent.predict(state)
            next_state, ext_reward, term, trunc, _ = self.env.step(action)
            done = float(term or trunc)
            next_obs_norm = next_state.astype(np.float32)
            int_reward = self.agent.get_noveld_bonus(prev_obs_norm, next_obs_norm)
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
            if done:
                state, _ = self.env.reset()
                self.agent._episode_visited = set()
                prev_obs_norm = state.astype(np.float32)
            else:
                state = next_state
                prev_obs_norm = next_obs_norm
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
    # _rnd_error()
    # -------------------------------------------------------------------------

    def test_rnd_error_is_nonneg_float(self):
        """_rnd_error must return a non-negative float."""
        error = self.agent._rnd_error(np.zeros(4, dtype=np.float32))
        self.assertIsInstance(error, float)
        self.assertGreaterEqual(error, 0.0)

    # -------------------------------------------------------------------------
    # _is_first_visit()
    # -------------------------------------------------------------------------

    def test_is_first_visit_returns_true_then_false(self):
        """First call returns True; second call with same obs returns False."""
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self.agent._episode_visited = set()
        self.assertTrue(self.agent._is_first_visit(obs))
        self.assertFalse(self.agent._is_first_visit(obs))

    def test_is_first_visit_resets_on_new_episode(self):
        """Clearing _episode_visited lets the same obs count as first visit again."""
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self.agent._episode_visited = set()
        self.agent._is_first_visit(obs)  # mark visited
        self.assertFalse(self.agent._is_first_visit(obs))
        self.agent._episode_visited = set()  # new episode
        self.assertTrue(self.agent._is_first_visit(obs))

    # -------------------------------------------------------------------------
    # get_noveld_bonus()
    # -------------------------------------------------------------------------

    def test_get_noveld_bonus_is_nonneg(self):
        """NovelD bonus is always >= 0 (clipped from below)."""
        state = np.zeros(4, dtype=np.float32)
        next_state = np.ones(4, dtype=np.float32) * 0.5
        self.agent._episode_visited = set()
        bonus = self.agent.get_noveld_bonus(state, next_state)
        self.assertGreaterEqual(bonus, 0.0)

    def test_get_noveld_bonus_zero_on_revisit(self):
        """Bonus is exactly 0 when next_state has already been visited this episode."""
        state = np.zeros(4, dtype=np.float32)
        next_state = np.ones(4, dtype=np.float32) * 0.5
        self.agent._episode_visited = set()
        self.agent.get_noveld_bonus(state, next_state)  # first visit
        bonus = self.agent.get_noveld_bonus(state, next_state)  # revisit
        self.assertEqual(bonus, 0.0)

    # -------------------------------------------------------------------------
    # compute_gae()
    # -------------------------------------------------------------------------

    def _gae_inputs(self, T=6, dones=None):
        dones = dones if dones is not None else torch.zeros(T)
        return (
            [1.0] * T,
            [0.5] * T,
            torch.zeros(T),
            torch.zeros(T),
            torch.zeros(T),
            torch.zeros(T),
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
