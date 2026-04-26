# Week 2: Policy and Value Iteration

This week you will get used to RL environments in code and implement the fundamental algorithms of policy and value iteration. You'll see how your agent's behaviour changes over time and hopefully have your first successful training runs.

⚠ Before you start, make sure to have read the general `README.md`.
You should add your solutions to the central train and eval script (see the section 'Repository Structure' in the general `README.md`)

## Level 1: Policy & Value Iteration
### 1. The MarsRover Environment
In the `../environments.py` (in the parent folder) file you’ll find the first environment we’ll work with: the MarsRover. 
You have seen it as an example in the lecture: the agent can move left or right with each step and should ideally move to the rightmost state. 
Your task here is to implement the environment dynamics `get_next_state` and determine
the transition matrix `get_transition_matrix`. This is needed for the algorithms policy and value iteration.
The script `rl_exercises/week_2/mars_rover.py` is for you to play with and to check whether your implementation
makes sense. Feel free to vary it however you need, e.g. what you log and how you initialize the environment.
```bash
python rl_exercises/week_2/mars_rover.py
```

### 2. Policy Iteration for the MarsRover
In this first exercise, the environment will be deterministic, that means the rover
will always execute the given action. Your task is to implement the algorithm policy iteration.
The code stub to be completed is in `policy_iteration.py`.

You can run the exercise with:
```bash
# Policy Iteration
python rl_exercises/train_agent.py +exercise=w2_policy_iteration
```

Please note that in this exercise we work with the state-value / Q function. In principle, the same formula applies.

### 3. Value Iteration for the probabilistic MarsRover
For this second exercise, we modify the MarsRover environment, now the rover may or may not execute the requested action, the probability is 50%. 
You will complete the code in `value_iteration.py` in order
to evaluate a policy on this variation of our environment.
What happens if you try different initial policies? Will you always converge to the same policy? What if you vary gamma?

You can run the exercise with:
```bash
# Value Iteration
python rl_exercises/train_agent.py +exercise=w2_value_iteration
```

## Level 2: Decreasing Information Flow
What happens if you only have access to `step()` instead of the dynamics and reward? Do both methods still work? This setting will be what we'll work with for the rest of the semester.
Briefly describe your observations in an `observations_l2.txt` file.

## Level 3: Adding Context
The paper we'll talk about this week is [Contextualize Me - The Case for Context in Reinforcement Learning](https://arxiv.org/pdf/2202.04500). 
This paper discusses cMDPs in detail, so you will know the basics from the lecture, but there is a lot of additional content that isn't especially relevant at this point in time.
Therefore, you don't need to read the full paper, focus on these parts only:
- Abstract
- Introduction
- Section 2: cMDPs
- Section 4: RL with Context
- Section 6.2, 6.3, 6.4: Context in Training

This will mean you will miss details, discussion and related work, but the important part is to understand why there is a formalism for the concept of generalization and how that informs our training/evaluation decisions.

As for the Level 3 exercise: obviously you won't use state-of-the-art RL algorithms on the benchmark in the paper quite yet. But you will do something similar with our MarsRover:
- implement a new contextual MarsRover environment (with at least two context features). You can see how CARL works or even use that structure if you want,  but it's not required.
- define how contexts change in training (recommended as a default: round robin rotation)
- set training, testing and validation sets like in Section 6.4
- run value and/or policy iteration with or without context provided. Also try to evaluate the saved policy on the validation and test sets. What do you observe? Where do the results differ the paper and where do you see the same? Can you explain what might be going on here? Write it down in an `observations_l3.txt` file.