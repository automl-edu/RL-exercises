from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from rl_exercises.environments import RandomWalkMarsRover
from rl_exercises.week_3.td_lambda_prediction import TDLambdaPredictionAgent
from rliable import library as rly

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
FIGURE_DIR = SCRIPT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

TRUE_VALUES = np.array([0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0])

LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]


def run_episode(env: RandomWalkMarsRover, agent: TDLambdaPredictionAgent) -> None:
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


def rms_error(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean((values[1:-1] - TRUE_VALUES[1:-1]) ** 2)))


def run_single_training(
    *,
    lmbda: float,
    alpha: float,
    episodes: int,
    seed: int,
) -> float:
    env = RandomWalkMarsRover(seed=seed)
    agent = TDLambdaPredictionAgent(
        n_states=env.n_states,
        alpha=alpha,
        gamma=1.0,
        lmbda=lmbda,
    )

    for _ in range(episodes):
        run_episode(env, agent)

    return rms_error(agent.V)


def collect_scores(
    *,
    lambdas: list[float],
    alphas: list[float],
    seeds: int = 100,
    episodes: int = 10,
) -> dict[str, np.ndarray]:
    """Return scores in shape (num_runs, num_alphas).

    RLiable expects arrays with runs on axis 0. We treat each alpha value
    as a small "task" for interval estimation over the alpha sweep.
    """
    scores = {}

    for lmbda in lambdas:
        lambda_scores = np.zeros((seeds, len(alphas)), dtype=float)

        for seed in range(seeds):
            for alpha_idx, alpha in enumerate(alphas):
                lambda_scores[seed, alpha_idx] = run_single_training(
                    lmbda=lmbda,
                    alpha=alpha,
                    episodes=episodes,
                    seed=seed,
                )

        scores[f"lambda={lmbda}"] = lambda_scores

    return scores


def plot_error_vs_alpha(scores: dict[str, np.ndarray]) -> None:
    """Figure 4 style: RMS error over alpha, one curve per lambda."""
    mean_scores = {label: np.mean(values, axis=0) for label, values in scores.items()}

    interval_estimates = {}

    for label, values in scores.items():
        interval_estimates[label] = []

        for alpha_idx in range(len(ALPHAS)):
            alpha_scores = values[:, alpha_idx]

            estimates, intervals = rly.get_interval_estimates(
                {label: alpha_scores[:, None]},
                lambda x: np.array([np.mean(x)]),
                reps=2000,
            )

            interval_estimates[label].append(intervals[label][:, 0])

        interval_estimates[label] = np.array(interval_estimates[label]).T

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, means in mean_scores.items():
        ci = interval_estimates[label]
        ax.plot(ALPHAS, means, marker="o", label=label)
        ax.fill_between(ALPHAS, ci[0], ci[1], alpha=0.2)

    ax.set_xlabel("alpha")
    ax.set_ylabel("RMS error")
    ax.set_title("TD(lambda) random walk: error after 10 episodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "td_lambda_error_vs_alpha.png")
    plt.close(fig)


def plot_best_error_vs_lambda(scores: dict[str, np.ndarray]) -> None:
    """Figure 5 style: best alpha per lambda."""
    best_errors = []
    best_alphas = []

    for lmbda in LAMBDAS:
        label = f"lambda={lmbda}"
        mean_per_alpha = np.mean(scores[label], axis=0)
        best_idx = int(np.argmin(mean_per_alpha))

        best_errors.append(mean_per_alpha[best_idx])
        best_alphas.append(ALPHAS[best_idx])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(LAMBDAS, best_errors, marker="o")
    ax.set_xlabel("lambda")
    ax.set_ylabel("best RMS error over alpha")
    ax.set_title("TD(lambda) random walk: best alpha per lambda")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "td_lambda_best_error_vs_lambda.png")
    plt.close(fig)

    return best_errors, best_alphas


def write_observations(
    *,
    scores: dict[str, np.ndarray],
    best_errors: list[float],
    best_alphas: list[float],
) -> None:
    lines = []
    lines.append("Level 3 Observations: TD(lambda)")
    lines.append("=================================")
    lines.append("")
    lines.append("Random-walk setup:")
    lines.append("- states A-G encoded as 0-6")
    lines.append("- nonterminal states B-F encoded as 1-5")
    lines.append("- every episode starts at D")
    lines.append("- random movement left/right with probability 0.5")
    lines.append("- left terminal outcome is 0, right terminal outcome is 1")
    lines.append("- true values are [1/6, 2/6, 3/6, 4/6, 5/6] for B-F")
    lines.append("")
    lines.append("Best alpha per lambda:")
    for lmbda, err, alpha in zip(LAMBDAS, best_errors, best_alphas):
        lines.append(f"- lambda={lmbda}: best alpha={alpha}, RMS error={err:.4f}")

    lines.append("")
    lines.append("Mean RMS error table:")
    header = "lambda | " + " | ".join(f"a={alpha}" for alpha in ALPHAS)
    lines.append(header)
    lines.append("-" * len(header))

    for lmbda in LAMBDAS:
        label = f"lambda={lmbda}"
        means = np.mean(scores[label], axis=0)
        row = f"{lmbda} | " + " | ".join(f"{v:.4f}" for v in means)
        lines.append(row)

    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "This recreates the paper's random-walk experiment in a simplified "
        "online implementation. The generated plots are in "
        "rl_exercises/week_3/figures/."
    )
    lines.append(
        "The exact numbers do not need to match the paper exactly because the "
        "implementation details, seeds, and update convention may differ."
    )

    text = "\n".join(lines)

    with open(SCRIPT_DIR / "observations_l3.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print(text)


if __name__ == "__main__":
    scores = collect_scores(
        lambdas=LAMBDAS,
        alphas=ALPHAS,
        seeds=100,
        episodes=10,
    )

    plot_error_vs_alpha(scores)
    best_errors, best_alphas = plot_best_error_vs_lambda(scores)

    write_observations(
        scores=scores,
        best_errors=best_errors,
        best_alphas=best_alphas,
    )
