# Week 7: Exploration

This week, you'll explore exploration: random network distillation, finding ways of analyzing exploration and exploration ensembles. From this week onward, we'll focus more in explanations and questions. **This means uploading plots and notes is no longer optional**, we want to see your results!

Throughout, use [RLiable](https://github.com/google-research/rliable) for plotting performance curves and statistical comparisons. Feel free to leverage [HyperSweeper](https://github.com/automl/hypersweeper) for hyperparameter tuning if you wish.

---

## Level 1: Random Network Distillation
Implement RND as seen in the lecture and test it on at least one environment. 
You can choose any environment, but obviously you should make sure exploring novel states is actually useful.
Good choices could be LunarLander-v3 or simple versions in [MiniGrid](https://minigrid.farama.org/).

We'll use the DQN you built in week 4 and PPO in week 6 (solution pushed, but you can also use your own) as a basis with the extended RNDDQNAgent in 'rnd_dqn.py' and 'rnd_ppo.py'.
Also, you will have to implement your own RND network (rnd_networks.py).
Finish implementing these 3 files and test how well RND works compared to epsilon greedy. 
Do you think this method is a good fit for DQN and PPO? Why or why not? 
Is there a way to see the reason from your training runs?
Upload your **comparison plot** and if you find a way to verify your intuition, a **.txt note** about your solution.


## Level 2: Visualize Exploration and NoveID
Now that you have RND implemented, let implement NoveID specifically for PPO (noveid_ppo.py).
Another task is also to find a way to analyze their behavior. 
Think of a way to show the effect of exploration at different points in training. 
You can also look at papers to find established ideas.

Use your method to generate **behavior snapshots** that let you reason about exploration behavior at a minimum of three points during training and upload these. Note down your thoughts in a **.txt file**.

## Level 3: Ensemble-Based Exploration
As this week's stretch task, take a look at a different form of exploration: using a DQN ensemble. 
Follow this paper and implement its method: https://arxiv.org/pdf/2306.05483
What effect do you see compared to RND?
