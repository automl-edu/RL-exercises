# Week 8: Evaluation & Testing
This week, we focus on seeding and evaluations. This means you're quite free in the algorithms and environments you can use as long as they serve to answer the question. We recommend working with options you already know well.

## Level 1: Seeding
Choose an algorithm and a cheap to run environment. Now run as many seeds as you can. For a low, medium and large amount of seeds, look at:
- the IQM, mean and median rewards over time
- the standard error, standard deviation and 95% confidence interval
For the low amount of seeds, try using several disparate seed sets. What changes in your interpretation of the results? Which influence do the number of seeds and metrics have? Push your plots and record your thoughts in `observations_l1.txt`.

## Level 2: Statistical Testing
Now perform any statistical test to compare two RL algorithms. To do so, follow these steps:
1. Choose a test (e.g. from the ones in the lecture) and significance level
2. Choose two algorithms and run at least 5 seeds each
3. Think carefully about the aggregation of the different runs if necessary
4. Perform the test
Document the test and result in `observations_l2.txt`.

## Level 3: The Statistical Precipice
The paper [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/pdf/2108.13264) shows some of the many pitfalls of RL evaluation and how this can damage progress in the field. Carefully think about your takeaways from this paper, are all their recommendations practical? Do they go far enough for you? Are you surprised by the empirical results?

Your Level 3 task is based on this paper: choose two algorithms and at least three environments. Now perform a very thorough analysis based on this paper, utilizing all the tools they propose. Document this analysis very well and describe both your observations as well as the experience performing it (do the recommendations work in practice? Would you have done it differently?). 