# Week 3: Model-free Control
This week you will implement you first real model-free learning algorithm, SARSA, as well as conduct some experiments concerning its hyperparameters.

Wherever relevant, use [RLiable](https://github.com/google-research/rliable) for plotting your results. If you wish to explore design decisions, use [HyperSweeper](https://github.com/automl/hypersweeper) for hyperparameter tuning.

## Level 1
### Model-free Control with SARSA
Your task is to complete the code stubs in `sarsa.py` and  `epsilon_greedy_policy` to implement the SARSA algorithm from the lecture. 
Use the provided method signatures as guidance for what is expected in the tests, but feel free to extend the implementation as needed.

## Level 2
### Hyperparameter Optimization for SARSA
Many of the insights from SARSA carry over to more advanced RL algorithms—particularly the role of hyperparameters. In this section, you will explore how tuning these hyperparameters influences the algorithm's performance.

We’ll leverage two tools:
- [Hydra](https://github.com/facebookresearch/hydra): A configuration management framework that uses YAML files for cleanly storing experiment settings.
- [HyperSweeper](https://github.com/automl/hypersweeper): A tool for running hyperparameter sweeps with various optimization strategies using Hydra Configs.

If not already installed, install [Omegaconf](https://omegaconf.readthedocs.io/en/2.3_branch/) using pip.
Install Hydra via `pip install hydra-core`.
Clone the Hypersweeper repository from the link above, then install it into your active `uv` environment.

We’ve provided a skeleton in `sarsa_sweep.py` to help you get started. 
This script initializes the Mars Rover environment from last week and runs episodes to measure returns.
Your task is to:
- Integrate it with the Hydra configuration file located at `rl_exercises/configs/agent/sarsa_sweep.py`.
- Fill out the configuration file to specify hyperparameters to sweep over.
- Use Hypersweeper to run the hyperparameter sweep. (Currently set to Random Search—you’re welcome to modify the search strategy.)
- Extend the `run_episodes()` function in `sarsa_sweep.py` to evaluate returns across multiple episodes for a more reliable performance estimate.

Finally, analyze your results by answering:
- How much does performance improve with tuned hyperparameters?
- How does the learning rate affect training steps?
- What value of $\epsilon$ yields the best performance?

*Note:* We have not provided test cases for this part. Instead, use the example templates in Hypersweeper and the starter code to guide your implementation.

## Level 3
### Implementing TD($\lambda$)
As a final challenge, implement the TD($\lambda$) algorithm in the same style as your SARSA implementation. You may choose either:
-  [Gridcore environment](https://github.com/automl/TabularTempoRL/blob/master/grid_envs.py)
- [Four Rooms environment](https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/envs/fourrooms.py)

Make $\lambda$ (the number of lookahead steps) a configurable parameter, and:
- Run experiments to analyze performance across different values of $n$.
- Perform hyperparameter tuning for TD($\lambda$) as you did with SARSA.