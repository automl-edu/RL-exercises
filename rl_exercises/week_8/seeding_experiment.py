import pickle

import gymnasium as gym
from ppo_seeding import PPOAgent

SEED_SETS = {
    "low": list(range(3)),
    "medium": list(range(10)),
    "large": list(range(30)),
}

TOTAL_STEPS = 100000


def run_seed(seed):

    env = gym.make("CartPole-v1")

    agent = PPOAgent(
        env=env,
        seed=seed,
    )

    results = agent.train(
        total_steps=TOTAL_STEPS,
        eval_interval=5000,
        eval_episodes=10,
    )

    env.close()

    return results


def main():

    for name, seeds in SEED_SETS.items():
        all_runs = []

        for seed in seeds:
            print(f"Running seed {seed}")

            run = run_seed(seed)

            all_runs.append(run)

        with open(f"{name}_seeds.pkl", "wb") as f:
            pickle.dump(all_runs, f)


if __name__ == "__main__":
    main()
