from __future__ import annotations

import numpy as np

from rl_exercises.environments import ContextualMarsRover
from rl_exercises.train_agent import evaluate
from rl_exercises.week_2.policy_iteration import PolicyIteration


# These are the contexts used to compute the policy.
TRAIN_CONTEXTS = [
    {"action_success_prob": 0.90, "left_goal_reward": 1.0, "right_goal_reward": 10.0},
    {"action_success_prob": 0.75, "left_goal_reward": 8.0, "right_goal_reward": 2.0},
    {"action_success_prob": 0.60, "left_goal_reward": 3.0, "right_goal_reward": 6.0},
]

# Validation contexts are different from training but still similar.
VALIDATION_CONTEXTS = [
    {"action_success_prob": 0.80, "left_goal_reward": 1.0, "right_goal_reward": 9.0},
    {"action_success_prob": 0.70, "left_goal_reward": 7.0, "right_goal_reward": 2.0},
]

# Test contexts are more shifted, so they check generalization.
TEST_CONTEXTS = [
    {"action_success_prob": 0.45, "left_goal_reward": 10.0, "right_goal_reward": 1.0},
    {"action_success_prob": 0.95, "left_goal_reward": 1.0, "right_goal_reward": 3.0},
]


def train_policy(context_visible: bool) -> PolicyIteration:
    # Train once on the training context set.
    env = ContextualMarsRover(
        contexts=TRAIN_CONTEXTS,
        context_visible=context_visible,
        context_change="round_robin",
        seed=0,
    )

    agent = PolicyIteration(env=env, seed=0, filename="policy_l3.npy")
    agent.update_agent()
    # Save the learned policy so we can evaluate the same policy later.
    agent.save()
    return agent


def evaluate_on_context_set(
    agent: PolicyIteration,
    contexts: list[dict[str, float]],
    context_visible: bool,
    name: str,
) -> float:
    # Recreate the environment with the requested context split.
    env = ContextualMarsRover(
        contexts=contexts,
        context_visible=context_visible,
        context_change="round_robin",
        seed=1,
    )

    mean_reward = evaluate(env=env, agent=agent, episodes=30, seed=1)
    print(f"{name}: {mean_reward:.3f}")
    return mean_reward


def run_experiment(context_visible: bool) -> dict[str, float]:
    label = "context_visible" if context_visible else "context_hidden"
    print(f"\n=== {label} ===")

    # Run the same experiment once with hidden context and once with visible context.
    agent = train_policy(context_visible=context_visible)

    results = {
        "train": evaluate_on_context_set(
            agent, TRAIN_CONTEXTS, context_visible, "train"
        ),
        "validation": evaluate_on_context_set(
            agent, VALIDATION_CONTEXTS, context_visible, "validation"
        ),
        "test": evaluate_on_context_set(
            agent, TEST_CONTEXTS, context_visible, "test"
        ),
    }

    return results


if __name__ == "__main__":
    hidden_results = run_experiment(context_visible=False)
    visible_results = run_experiment(context_visible=True)

    print("\nSummary")
    print("hidden:", hidden_results)
    print("visible:", visible_results)
