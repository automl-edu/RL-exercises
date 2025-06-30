# Week 10: AutoRL
This week the exercises will dive deep into hyperparameters. You're free to explore all the questions with any RL algorithms and environments you want and look into how well HPO configurations generalize, the effects of multi-fidelity optimization and what happens with context in the mix.

## Level 1: How Well Does HPO Generalize in RL?
Design a generalization setting for HPO in RL. First, describe the generalization task. Are we generalizing across seeds, environments, algorithms, etc.? What results would you expext? Now run the optimization and record your results in a *.txt file*. Do they match your expectations?

## Level 2: Multi-fidelity in RL
Partial evaluations can make HPO much more efficient, but there are also pitfalls if early success doesn't translate to good final performance. Design a settings where you think initial scores can be deceiving and describe why. Test with any multi-fidelity optimizer if good initial performances actually influence the optimization in a negative way (for example by checking the optimization history or using DeepCave to run analyses). Upload a *.txt file* with your results.

## Level 3: Context & Hyperparameters
Read the ["Hyperparameters in Contextual RL are Highly Situational"](https://arxiv.org/pdf/2212.10876) workshop paper. It shows that there likely is an interaction between context in RL and HPO. Reproduce these results with an algorithm, environment and optimizer of your choice. What do you observe? Describe your observations in a *.txt file*.
