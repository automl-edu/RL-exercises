import pickle

import matplotlib.pyplot as plt
import numpy as np

all_early = []
all_late = []

for seed in range(5):
    with open(f"Results/dyna2/seed_{seed}.pkl", "rb") as f:
        data = pickle.load(f)

    all_early.append(data["multistep"][0]["errors"])

    all_late.append(data["multistep"][-1]["errors"])

all_early = np.array(all_early)
all_late = np.array(all_late)

early_mean = np.mean(all_early, axis=0)
late_mean = np.mean(all_late, axis=0)

early_std = np.std(all_early, axis=0)
late_std = np.std(all_late, axis=0)

k = np.arange(1, 21)

plt.figure(figsize=(8, 5))

plt.plot(k, early_mean, marker="o", label="Early (2500 steps)")

plt.fill_between(k, early_mean - early_std, early_mean + early_std, alpha=0.2)

plt.plot(k, late_mean, marker="o", label="Late (15000 steps)")

plt.fill_between(k, late_mean - late_std, late_mean + late_std, alpha=0.2)

plt.xlabel("Prediction Horizon k")
plt.ylabel("Multi-Step Error $E_k$")
plt.title("Multi-Step Prediction Error")

plt.grid(True)
plt.legend()

plt.savefig("multistep.png")
plt.show()
