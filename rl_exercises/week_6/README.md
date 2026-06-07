# Week 6: Actor‑Critic Methods

This module builds on your policy‑gradient foundations by guiding you through actor‑critic methods, Proximal Policy Optimization (PPO), and continuous‑action algorithms. We focus on PPO since we spend more time on this algorithm in the lecture. It will serve as a good example of how actor-critic algorithms are implemented. This is likely the week with the most coding, but since PPO is complex, it's unfortunately necessary to look at its different parts individually.

Wherever relevant, use [RLiable](https://github.com/google-research/rliable) for plotting your results. If you wish to explore design decisions, use [HyperSweeper](https://github.com/automl/hypersweeper) for hyperparameter tuning.

## Level 1: On‑Policy Actor‑Critic Baselines

1. Complete the implementation in `actor_critic.py` (the `ActorCriticAgent` class) to support all four baseline modes:
   - `none` (no baseline)
   - `avg` (running-average reward)
   - `value` (learned value function)
   - `gae` (Generalized Advantage Estimation)

2. Train your agent on **CartPole‑v1** and **LunarLander‑v3**, comparing the four baselining strategies.
3. Use RLiable to plot average return vs. steps (and confidence intervals) for each baseline.
4. Analyze the results:
   - Do some baselines learn faster or reach higher returns?
   - Provide a conceptual justification for any observed differences (e.g. variance reduction, bias–variance trade‑off).

Training scripts and defaults live in `configs/agent/actor_critic.yaml`. Push any plots and record your observations in an `observations_l1.txt`.

## Level 2: Extending PPO

1. Complete the TODOs in `ppo.py` to flesh out a working PPO agent with:
   - Clipped surrogate objective
   - Value‑function loss coefficient (`vf_coef`)
   - Entropy bonus coefficient (`ent_coef`)
   - Mini‑batch training over multiple epochs

2. Read the [blog post on PPO implementation nuances](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) (e.g. learning‑rate annealing, KL‑early stopping, value clipping). Select **two** enhancements and integrate them into your PPO. Clearly document and justify your choices in code comments.

1. Train your PPO agent on **LunarLander‑v3**, and plot performance curves (average return vs. steps, with uncertainty) using RLiable. Compare:
   - PPO v/s Actor Critic
   - PPO with vanilla settings
   - PPO with your two enhancements

Training defaults in `configs/agent/ppo.yaml`. Again, push plots and observations in `observations_l2.txt`.

## Level 3: SAC
We briefly spoke about SAC in the lecture, but will discuss the [paper](https://arxiv.org/pdf/1801.01290) in more depth. It's the preferred sample efficient method for many continuous control tasks, so it has a different performance profile than PPO. Focus on the differences to the PPO algorithm that you're now familiar with!

For Level 3, implement SAC and compare it to your PPO results. Investigate the claims of the paper. Which can you verify? Record your experiences in a `observations_l3.txt` and push all plots.