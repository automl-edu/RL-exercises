import pickle

import matplotlib.pyplot as plt
import numpy as np

all_mse = []

for seed in range(5):
    with open(f"Results/dyna2/seed_{seed}.pkl", "rb") as f:
        data = pickle.load(f)

    mse = []

    for x in data["model_metrics"]:
        mse.append(x["state_mse"])

    all_mse.append(mse)

steps = [x["steps"] for x in data["model_metrics"]]

all_mse = np.array(all_mse)

mean = np.mean(all_mse, axis=0)
std = np.std(all_mse, axis=0)

plt.figure(figsize=(8, 5))

plt.plot(steps, mean, marker="o", label="Mean MSE")

plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

plt.xlabel("Real Environment Steps")
plt.ylabel("State Prediction MSE")
plt.title("One-Step Model Prediction Accuracy")

plt.grid(True)
plt.legend()

plt.savefig("model_accuracy.png")
plt.show()
