# Week 5: Policy Gradient

This week you will implement the REINFORCE policy gradient algorithm in order to learn a stochastic policy for the `CartPole-v1` environment.

## Level 1: Policy Gradient Implementation
- Complete the `Policy` class in the code with 2 Linear units to map the states to probabilities over actions.
- Implement `compute_returns` method to compute the discounted returns $G_t$ for each state in a trajectory.
- Implement the policy improvement step in `update_agent` method to update the policy given the rewards and probabilities from the last trajectory.
- Use the policy in the `predict_action` method to sample action and return its log probability.

*Note*: It's well known that this is a noisy algorithm! It's possible that most of your runs do not learn well. This can be a hyperparameter issue. Don't be discouraged and don't spend too much time debugging performance if you think your code is correct, we will improve upon this algorithm next week.

## Level 2: Empirical Understanding
Now we want to gain some more experience with this algorithm, which should be possible even if a lot of your runs don't look very promising. Here are some variations to try:
- How does the length of the trajectories affect the training?
- Does the same network architecture and learning rate work for `LunarLander-v2`?
- How is the sample complexity (how many steps it takes to solve the environment) of this algorithm related to the DQN from the last exercise?
Please write your answers in your `observations_l2.txt`.

## Level 3: DDPG
We switched gears away from DQN to work in continuous domains, but we can in fact use the idea behind Q-Learning in these spaces as well! The [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971) algorithm uses a Q-function in combination with a learned policy. Read the whole paper and focus on the question of stability: why does DDPG work where previous work failed and where do they say they have to change the original DPG algorithm? Since you already know DQN, you should be able to form an intuition on how DDPG came about.

The Level 3 task is to implement the DDPG algorithm and compare it against REINFORCE. It's fine if you only do this on one environment. Push the corresponding plots and record your thoughts in your `observations_l3.txt`.