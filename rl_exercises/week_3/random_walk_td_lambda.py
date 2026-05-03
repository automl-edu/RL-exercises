from __future__ import annotations

import numpy as np
from rl_exercises.environments import RandomWalkMarsRover
from rl_exercises.week_3.td_lambda_prediction import TDLambdaPredictionAgent

TRUE_VALUES = np.array([0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0])


def run_episode(env: RandomWalkMarsRover, agent: TDLambdaPredictionAgent):
    state, _ = env.reset()
    agent.reset_traces()

    done = False
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(None)
        done = terminated or truncated

        agent.update(
            state=state,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        state = next_state


def rms_error(V: np.ndarray) -> float:
    return float(np.sqrt(np.mean((V[1:-1] - TRUE_VALUES[1:-1]) ** 2)))


def run_experiment(
    lmbda: float,
    alpha: float,
    episodes: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    env = RandomWalkMarsRover(seed=seed)
    agent = TDLambdaPredictionAgent(
        n_states=env.n_states,
        alpha=alpha,
        gamma=1.0,
        lmbda=lmbda,
    )

    for _ in range(episodes):
        run_episode(env, agent)

    return agent.V, rms_error(agent.V)


if __name__ == "__main__":
    lines = []
    lines.append("TD(lambda) Random Walk Results")
    lines.append("==============================")

    for lmbda in [0.0, 0.3, 0.8, 1.0]:
        V, error = run_experiment(
            lmbda=lmbda,
            alpha=0.1,
            episodes=10,
            seed=0,
        )

        lines.append("")
        lines.append(f"lambda={lmbda}")
        lines.append(f"V={np.round(V, 3).tolist()}")
        lines.append(f"RMS error={error:.4f}")

    observations = "\n".join(lines)

    with open("rl_exercises/week_3/observations_l3.txt", "w", encoding="utf-8") as f:
        f.write(observations)

    print(observations)
