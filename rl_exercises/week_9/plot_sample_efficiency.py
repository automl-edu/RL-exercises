import pickle

import matplotlib.pyplot as plt
import numpy as np


def load_curve(filename):

    with open(filename, "rb") as f:
        data = pickle.load(f)

    steps = []
    returns = []

    for x in data["returns"]:
        steps.append(x["steps"])
        returns.append(x["return"])

    return np.array(steps), np.array(returns)


ppo_returns = []

for seed in range(5):
    steps, ret = load_curve(f"Results/ppo/seed_{seed}.pkl")

    ppo_returns.append(ret)

ppo_returns = np.array(ppo_returns)

ppo_mean = np.mean(ppo_returns, axis=0)

ppo_std = np.std(ppo_returns, axis=0)


dyna_returns = []

for seed in range(5):
    steps, dyna = load_curve(f"Results/dyna/seed_{seed}.pkl")

    dyna_returns.append(dyna)

dyna_returns = np.array(dyna_returns)

dyna_mean = np.mean(dyna_returns, axis=0)

dyna_std = np.std(dyna_returns, axis=0)


plt.plot(steps, ppo_mean, label="PPO")

plt.plot(steps, dyna_mean, label="Dyna PPO")

plt.fill_between(steps, ppo_mean - ppo_std, ppo_mean + ppo_std, alpha=0.2)

plt.fill_between(steps, dyna_mean - dyna_std, dyna_mean + dyna_std, alpha=0.2)

plt.xlabel("Real Environment Steps")

plt.ylabel("Average Return")

plt.legend()

plt.savefig("sample_efficiency.png")

plt.show()
