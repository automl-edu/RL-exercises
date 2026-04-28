# Week 3: Model-free Control
This week you will implement you first real model-free learning algorithm, SARSA, and optionally its counterpart Q-learning.

Wherever relevant, use [RLiable](https://github.com/google-research/rliable) for plotting your results. If you wish to explore design decisions, use [HyperSweeper](https://github.com/automl/hypersweeper) for hyperparameter tuning. Examples for both are in the `../examples` directory.

Hint: you need to update and run `train_agent.py` with the agent you implement. For syntex, please check out the central README. Moreover, the MarsRover from the `environment.py` implemented in the last week is needed. 

## Level 1: Model-free Control with SARSA
Your task is to complete the code stubs in `sarsa_qlearning.py` and  `epsilon_greedy_policy` to implement the SARSA algorithm from the lecture. 
Use the provided method signatures as guidance for what is expected in the tests, but feel free to extend the implementation as needed.
Note that for this part of the exercise, you can ignore the q-learning part of the agent. 
Check your implementation on the MarsRover environment: can you reach similar scores as with last week's approaches?
If not, think on why that is likely the case.
```bash
python rl_exercises/train_agent.py agent=sarsa env_name=MarsRover
```


## Level 2: Q-Learning
Now add Q-learning as an alternative so SARSA to our agent. 
Compare the performance of both, can you see differences in how they learn? 
If you see that the environment is difficult to solve for both agents, try to explicitly log their states and actions. 
What is the difference in how the approach the problem?
Record your thoughts in a `observations_l2.txt` file.
```bash
python rl_exercises/train_agent.py agent=qlearning env_name=MarsRover
```


## Level 3: Implementing TD($\lambda$)
As a final challenge, implement the TD($\lambda$) algorithm in the same style as your SARSA implementation as included in the classic paper [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009). 
As for reading, only focus on the Abstract and Sections 1-3. This may seem long, but that due to formatting and the loose tone of the paper, it's not as much content as it seems. 

If you want to solve Level 3, these are your ToDos:
- implement TD($\lambda$) similarly to SARSA and Q-Learning in a new agent
- choose and example setting from the paper and implement it as an environment
- try and recreate the paper's results. Do the match your observations? Document it in a `observations_l3.txt` file