# Week 6: Actor‑Critic Methods

This module builds on your policy‑gradient foundations by guiding you through actor‑critic methods, Proximal Policy Optimization (PPO), and continuous‑action algorithms. We’ve divided the work into three levels of increasing complexity.

Throughout, use [RLiable](https://github.com/google-research/rliable) for plotting performance curves and statistical comparisons. Feel free to leverage [HyperSweeper](https://github.com/automl/hypersweeper) for hyperparameter tuning if you wish.

---

## Level 1: On‑Policy Actor‑Critic Baselines

### Task
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

Training scripts and defaults live in `configs/agent/actor_critic.yaml`.

---

## Level 2: Extending PPO

### Task
1. Complete the TODOs in `ppo.py` to flesh out a working PPO agent with:
   - Clipped surrogate objective
   - Value‑function loss coefficient (`vf_coef`)
   - Entropy bonus coefficient (`ent_coef`)
   - Mini‑batch training over multiple epochs

2. Read the [blog post on PPO implementation nuances](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) (e.g. learning‑rate annealing, KL‑early stopping, value clipping). Select **two** enhancements and integrate them into your PPO:
   - Clearly document and justify your choices in code comments.

3. Train your PPO agent on **LunarLander‑v3**, and plot performance curves (average return vs. steps, with uncertainty) using RLiable. Compare:
   - PPO v/s Actor Critic
   - PPO with vanilla settings
   - PPO with your two enhancements

Training defaults in `configs/agent/ppo.yaml`.

---

## Level 3: Continuous‑Action Control and SAC

### Task
1. Adapt your PPO implementation to work in continuous action spaces. Replace categorical policy with a Gaussian policy (mean + log‑std outputs).
2. Implement Soft Actor‑Critic (SAC) in the same codebase style:
   - Actor network producing mean and variance
   - Two Q‑value critics
   - Temperature (entropy bonus) tuning
   - Replay buffer, off‑policy updates

3. Compare PPO vs. SAC on at least one continuous‑action environment (e.g. `Pendulum-v1`):
   - Plot learning curves via RLiable
   - Discuss sample efficiency and stability differences

Use stable‑baselines3 or spinup from DeepMind/OpenAI as reference implementations.  Cite any code you adapt.

Good luck, and happy RL‑ing!
