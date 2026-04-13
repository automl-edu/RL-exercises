# Week 2: Policy and Value Iteration
This week

This week you’ll get hands‑on experience with the core dynamic‑programming algorithms—policy iteration and value iteration—on simple grid‑world style MDPs. You’ll build your own environment, add partial observability or context, and then write the solvers.

⚠ Before you start, make sure to have read the general `README.md`.
You should add your solutions to the central train and eval script.

Wherever relevant, use [RLiable](https://github.com/google-research/rliable) for plotting your results. If you wish to explore design decisions, use [HyperSweeper](https://github.com/automl/hypersweeper) for hyperparameter tuning.

## Level 1: Build your own MDP
In `my_env.py` implement a new `gymnasium.Env` that satisfies the same interface as our example `MarsRover` (in `environments.py` in the parent folder). In particular, your env must define:

- `observation_space`: `Discrete(n_states)`
- `action_space`: `Discrete(n_actions)`
- `reset(self, …) → (obs: int, info: dict)`
- `step(self, action: int) → (next_obs: int, reward: float, terminated: bool, truncated: bool, info: dict)`
- `get_reward_per_action() → np.ndarray` of shape `(n_states, n_actions)`
- `get_transition_matrix() → np.ndarray` of shape `(n_states, n_actions, n_states)`

### Example

Use the provided `MarsRover` class in `environments.py` as a template. You can copy its structure, replace the dynamics and reward logic with your own MDP, then confirm it passes the interface tests.

### Verify

Run:

```bash
pytest tests/week_2/test_my_env.py
```
## Level 2: Partial Observability or Context

Extend your Level 1 env to be partially observable or contextual:
- Partial Observability: wrap your env so that step() and reset() return a noisy or aliased observation (e.g. randomly hide or corrupt the true state).
- Contextual MDP: introduce an external context variable (a “mode” or “task ID”) that modifies rewards or transition probabilities at reset time.

### Example
We’ve provided a `MarsRoverPartialObsWrapper` in `environments.py` that adds observation noise. Use that as a guide for wrapping or extending your own env.

Can you find an interesting way to show how your wrapper makes the environment partially observable/changing based on context? 

## Level 3: Policy & Value Iteration

Complete the algorithms in policy_iteration.py and value_iteration.py under week_2/. Fill in all TODO sections so that:

- Policy Iteration converges to a stable policy on MarsRover.
- Value Iteration finds the optimal value function and greedy policy on MarsRover.

You’re also free to apply the same solvers to your own Level 1 environment once they work.