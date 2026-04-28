from __future__ import annotations

import numpy as np
from pathlib import Path
from rl_exercises.environments import ContextualMarsRover
from rl_exercises.week_2.value_iteration import value_iteration

N_POSITIONS = 5
N_ACTIONS = 2
GAMMA = 0.9


# Mode A: convex region in context-feature space
train_contexts_mode_a = [
    {"slip": s, "reward": r} for s in [0.0, 0.25, 0.55] for r in [4, 10, 20]
]

# Mode B: small/narrow variation
train_contexts_mode_b = [{"slip": s, "reward": r} for s in [0.25, 0.35] for r in [6, 8]]

# Mode C: single feature values / combinatorial interpolation
train_contexts_mode_c = [
    {"slip": 0.0, "reward": 14},  # right is attractive
    {"slip": 0.5, "reward": 14},  # right is risky but valuable
    {"slip": 0.5, "reward": 4},  # right is risky and low-value
]

# Interpolation contexts
validation_contexts = [
    {"slip": 0.15, "reward": 8},
    {"slip": 0.40, "reward": 14},
]

# Extrapolation contexts
test_contexts = [
    {"slip": 0.7, "reward": 10},
    {"slip": 0.25, "reward": 3},
    {"slip": 0.7, "reward": 3},
]


def make_env(context: dict, seed: int = 0) -> ContextualMarsRover:
    full_context = {
        "slip_termination_probability": context["slip"],
        "slip_penalty": -5.0,
        "left_reward": 3.0,
        "right_reward": context["reward"],
    }

    return ContextualMarsRover(
        context=full_context,
        expose_context=False,
        horizon=10,
        seed=seed,
    )


def obs_tuple(position: int, context: dict) -> tuple[float, ...]:
    """
    CARL-style visible observation:
        [position, context_feature_1, context_feature_2]

    Here:
        [position, slip, reward]
    """
    return (
        float(position),
        float(context["slip"]),
        float(context["reward"]),
    )


def build_hidden_mdp(contexts: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Hidden context:
        state = position

    Since the agent does not observe the context, we average transition and reward
    information across all training contexts.
    """
    T_total = np.zeros((N_POSITIONS, N_ACTIONS, N_POSITIONS))
    R_total = np.zeros((N_POSITIONS, N_ACTIONS))

    for context in contexts:
        env = make_env(context)
        T_total += env.get_transition_matrix()
        R_total += env.get_reward_per_action()

    T_avg = T_total / len(contexts)
    R_avg = R_total / len(contexts)

    return T_avg, R_avg


def build_visible_mdp(
    contexts: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict[tuple[float, ...], int]]:
    """
    Visible context:
        observation = (position, slip, reward)

    This mimics context concatenation:
        [state, context_feature_1, context_feature_2]

    Value iteration is tabular, so we map each visible observation tuple
    to an integer index internally.
    """
    obs_to_idx: dict[tuple[float, ...], int] = {}

    idx = 0
    for context in contexts:
        for position in range(N_POSITIONS):
            obs = obs_tuple(position, context)
            obs_to_idx[obs] = idx
            idx += 1

    n_states = len(obs_to_idx)

    T = np.zeros((n_states, N_ACTIONS, n_states))
    R = np.zeros((n_states, N_ACTIONS))

    for context in contexts:
        env = make_env(context)
        T_c = env.get_transition_matrix()
        R_c = env.get_reward_per_action()

        for position in range(N_POSITIONS):
            s_idx = obs_to_idx[obs_tuple(position, context)]

            for action in range(N_ACTIONS):
                R[s_idx, action] = R_c[position, action]

                for next_position in range(N_POSITIONS):
                    next_idx = obs_to_idx[obs_tuple(next_position, context)]
                    T[s_idx, action, next_idx] = T_c[position, action, next_position]

    return T, R, obs_to_idx


def closest_context(context: dict, train_contexts: list[dict]) -> dict:
    """
    For validation/test contexts that were not seen during training, choose
    the closest training context.

    This is a simple tabular approximation. A neural-network policy could
    consume continuous context values directly, but VI/PI needs known state indices.
    """
    distances = [
        abs(context["slip"] - c["slip"]) + abs(context["reward"] - c["reward"])
        for c in train_contexts
    ]
    return train_contexts[int(np.argmin(distances))]


def evaluate_policy(
    pi: np.ndarray,
    contexts: list[dict],
    visible: bool,
    train_contexts: list[dict],
    obs_to_idx: dict[tuple[float, ...], int] | None = None,
    episodes_per_context: int = 50,
) -> float:
    returns = []

    for context in contexts:
        for episode in range(episodes_per_context):
            env = make_env(context, seed=episode)
            obs, _ = env.reset()
            total_reward = 0.0

            for _ in range(env.horizon):
                if visible:
                    assert obs_to_idx is not None

                    # If this exact context was trained on, use it.
                    # If not, map it to the closest known context.
                    if obs_tuple(obs, context) in obs_to_idx:
                        policy_obs = obs_tuple(obs, context)
                    else:
                        matched_context = closest_context(context, train_contexts)
                        policy_obs = obs_tuple(obs, matched_context)

                    policy_state = obs_to_idx[policy_obs]
                else:
                    policy_state = obs

                action = int(pi[policy_state])

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)

                if terminated or truncated:
                    break

            returns.append(total_reward)

    return float(np.mean(returns))


def run_mode(name: str, train_contexts: list[dict]) -> str:
    lines = []
    lines.append(f"\n{name}")
    lines.append("=" * len(name))

    for visible in [False, True]:
        if visible:
            T, R, obs_to_idx = build_visible_mdp(train_contexts)
            label = "visible context"
        else:
            T, R = build_hidden_mdp(train_contexts)
            obs_to_idx = None
            label = "hidden context"

        V, pi = value_iteration(
            T=T,
            R_sa=R,
            gamma=GAMMA,
            seed=333,
        )

        train_return = evaluate_policy(
            pi=pi,
            contexts=train_contexts,
            visible=visible,
            train_contexts=train_contexts,
            obs_to_idx=obs_to_idx,
        )

        validation_return = evaluate_policy(
            pi=pi,
            contexts=validation_contexts,
            visible=visible,
            train_contexts=train_contexts,
            obs_to_idx=obs_to_idx,
        )

        test_return = evaluate_policy(
            pi=pi,
            contexts=test_contexts,
            visible=visible,
            train_contexts=train_contexts,
            obs_to_idx=obs_to_idx,
        )

        filename = (
            Path("rl_exercises/week_2") / f"policy_{name.lower().replace(' ', '_')}_{label.replace(' ', '_')}.npy"
        )
        np.save(filename, pi)

        lines.append(f"\n{label}")
        lines.append(f"saved policy: {filename}")
        if visible:
            reshaped_pi = pi.reshape(len(train_contexts), N_POSITIONS)
            lines.append("policy rows = contexts, columns = positions")
            lines.append(f"policy:\n{reshaped_pi}")
        else:
            lines.append(f"policy: {pi.tolist()}")
        lines.append(f"train return: {train_return:.3f}")
        lines.append(f"validation return: {validation_return:.3f}")
        lines.append(f"test return: {test_return:.3f}")

    return "\n".join(lines)


if __name__ == "__main__":
    results = []

    results.append(run_mode("Mode A", train_contexts_mode_a))
    results.append(run_mode("Mode B", train_contexts_mode_b))
    results.append(run_mode("Mode C", train_contexts_mode_c))

    result_text = "\n".join(results)

    observations = f"""
Level 3 Observations
====================

{result_text}
"""

    with open("rl_exercises/week_2/protocol.txt", "w", encoding="utf-8") as f:
        f.write(observations)

    print(observations)
