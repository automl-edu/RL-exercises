import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import trim_mean

# IQM helper


def iqm(values):

    return trim_mean(values, 0.25)


# Satistics Helper


def compute_stats(final_scores):

    mean = np.mean(final_scores)

    median = np.median(final_scores)

    std = np.std(final_scores)

    se = std / np.sqrt(len(final_scores))

    ci95 = 1.96 * se

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "se": se,
        "ci95": ci95,
        "iqm": iqm(final_scores),
    }


BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "Results"

for experiment in ["low", "medium", "large"]:
    with open(RESULTS_DIR / f"{experiment}_seeds.pkl", "rb") as f:
        runs = pickle.load(f)

    curves = np.array([run["returns"] for run in runs])

    mean_curve = np.mean(curves, axis=0)

    median_curve = np.median(curves, axis=0)

    iqm_curve = np.array([iqm(curves[:, i]) for i in range(curves.shape[1])])

    steps = runs[0]["steps"]

    plt.figure(figsize=(8, 5))

    plt.plot(steps, mean_curve, label="Mean")

    plt.plot(steps, median_curve, label="Median")

    plt.plot(steps, iqm_curve, label="IQM")

    plt.xlabel("Environment Steps")

    plt.ylabel("Return")

    plt.title(f"{experiment.capitalize()} Seed Set")

    plt.legend()

    plt.savefig(RESULTS_DIR / f"{experiment}_comparison.png")

    plt.close()

    final_scores = curves[:, -1]

    stats = compute_stats(final_scores)

    print("\n")
    print(experiment.upper())
    # print(stats)

    print(f"Mean   : {stats['mean']:.2f}")
    print(f"Median : {stats['median']:.2f}")
    print(f"IQM    : {stats['iqm']:.2f}")
    print(f"Std    : {stats['std']:.2f}")
    print(f"SE     : {stats['se']:.2f}")
    print(f"95% CI : ±{stats['ci95']:.2f}")
