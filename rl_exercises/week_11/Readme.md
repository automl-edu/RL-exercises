# Week 11: Generalization
This week is the last regular exercise week and we'll focus on working with generalization environments. The goal is for you to be able to run and evaluate agents on cMDPs.

## Level 1: Evaluation Protocol
The lecture included a proticol by Kirk et al. that evaluates training performance, in-distribution generalization and out-of-distribution generalization for each varied context feature. Use one of the [CARL](https://github.com/automl/CARL) classic control environments to design a context space with two varying features. Now partition this space into a training and test distribution. Define your training contexts, run training and then evaluate the different performance regions. Upload a *plot* visualizing them. What do you observe?

## Level 2: Model Generalization
Now that we know how to apply this evaluation protocol, let's see how well our models do! Use the dyna PPO setup from week 9 to check how well your model performs. You'll first need to design a way to properly evaluate the model. Then *plot* the performances and compare to the policy performance. Where are the differences?

## Level 3: Curricula
Find a setting where a (manual) curriculum is useful in any given environment. Why does it help? Can you show why it's helpful in your evaluation? Upload a *.txt file* discussing your setup and observations.