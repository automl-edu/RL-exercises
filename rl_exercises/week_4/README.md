# Week 6: Deep Q-Learning

This week, you will learn how to implement and run Deep Reinforcement Learning (Deep RL) pipelines by building a Deep Q-Network (DQN) from scratch. DQN combines value-based RL with deep learning, enabling agents to act in complex, high-dimensional environments using neural networks as function approximators.

As this is your first step into Deep RL, you will also be introduced to the experimental standards commonly used in the field. In particular, you should:
- Run each experiment across multiple random seeds to assess robustness.
- Report both the mean performance and a measure of variation (e.g. confidence intervals) to evaluate the reliability of your results.

Wherever relevant, use [RLiable](https://github.com/google-research/rliable) for plotting your results. If you wish to explore design decisions, use [HyperSweeper](https://github.com/automl/hypersweeper) for hyperparameter tuning.

## Level 1
### Deep Q Learning
The goal of this exercise is to help you build an understanding of how deep networks affect learning in value-based RL. 
You will train your DQN agent on the `CartPole-v1` environment.
- Complete the DQN implementation in `dqn.py` and `buffers.py` by adding the necessary logic to use the deep network implemented in `networks.py` as a function approximator. Use the provided Hydra config at `rl_exercises/configs/agent/dqn.py`
- Vary the network architecture (width, depth), the size of the replay buffer, and batch sizes. 
- For each configuration: Plot the training curve, with the number of frames on the x-axis and the mean reward on the y-axis. Title each plot based on the architecture used, and save your plots in a new folder named `plots`.

*Note*  You can use the basic tests in `tests/week_4/test_dqn.py` to verify your implementation. However, the main deliverable is the set of plots and observations from your experiments.

## Level 2
RL experiments are notoriously sensitive to random seeds. A single run can dramatically misrepresent an algorithm’s actual performance. That’s why, in Deep RL, it is common practice to:
- Run experiments across multiple seeds (typically 3–10),
- Report mean performance along with confidence intervals, and
- Prefer robust metrics like Interquartile Mean (IQM) and visual tools like score distributions or performance profiles.

In preparation for your project and further RL experiments, use [RLiable](https://github.com/google-research/rliable), a library for more robust reporting, to improve your plots. Your task is to: 
- Re-run your DQN implementation on CartPole-v1 using at least 5 different random seeds.
- Use RLiable to compute and plot training curves, IQMs, median, mean, and optimality gap across seeds. For reference, please see Figure 9 and Figure 10 of [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/pdf/2108.13264).

Discussion prompts for Presentation: 
- What changes when using RLiable vs. plain averages?
- Do you feel more confident in the results? Why or why not?

## Level 3
As discussed in the lecture, the base DQN implementation suffers from several limitations, such as Overestimation and sample inefficiency. To address these, you will incorporate two key enhancements from the [Rainbow DQN paper](https://arxiv.org/pdf/1710.02298.pdf):
- [Prioritized Experience Replay Buffer](https://arxiv.org/pdf/1511.05952.pdf): Take a look at the paper and extend `ReplayBuffer` in `buffers.py` to implement Prioritized Experience Replay.
- [Double DQN](https://arxiv.org/pdf/1509.06461): Extend your `DQN` implementation in `dqn.py` to support Double DQN.

Ablate the performance gain added by each design decision by plotting the following configurations using Rliable:
- Base DQN
- DQN + Prioritized Replay
- DQN + Double DQN
- DQN + Prioritized Replay + Double DQN