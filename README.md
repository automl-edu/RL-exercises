# RL_exercises
Welcome to the RL exercises repository! You will work with this repository for the duration of the course, so please take your time to familiarize yourself with its structure.

## Excercises
Each week, you'll have a task that corresponds to that week's lecture. Each task is made up of 3 levels and additional material for you to reference. You do **not** need to complete every level each week, instead think of them like this:

1. Level 1: The basics from the lecture. This is mandatory and will be tested via autograding. Level 1 will mainly help you understand the basics of the week's topic better.
2. Level 2: Here we want to increase your understanding of how the algorithms we implement work in practice. This level will often ask you to run or design additional experiments, brainstorm improvements or compare results between environments and methods. Level 2 will build your intuition on how to solve RL environments in practice.
3. Level 3: This level is mainly for those who are very motivated and want to dive deeper into the current topic. We will ask you to implement more advanced ideas from the lecture or the literature, sometimes only with a paper for guidance. Level 3 will prepare you for implementing and extending ongoing research - this also means it's a lot of work! While we believe you should attempt Level 2 most weeks, it is perfectly fine to only select one or two weeks to tackle Level 3.

Apart from the levels, we will also link material of different sorts for each week. These are not mandatory, often go beyond the lecture and will not be the topic of your exams. There are two sorts of material we provide: academic material you can use as a reference to read deeper into a topic and guidance on practical aspects of RL. You can use both at your discretion, they are intended as resources in case you need to debug your method or possibly need a more advanced method in the first place.

## Repository Structure
Each week's task and code stubs can be found in `rl_exercises`. This is where you code and where you should store the result for each week.

The `tests` directory contains all tests for all weeks in their respective subfolders. You can run all tests (though you probably never want to do that) using the command `make test` and the weekly tests with `make test-week-<week-id>`. This is what we use in the autograding as well.

Lastly, in the root directory we have the files `train_agent.py` and `evaluate_agent.py`. We will build these up in weeks 2-10 to contain all of the algorithms and options you implement. These two scripts is what we use to test your code and generate results.

## Installation
1. Clone this repository:
    * ``git clone https://github.com/automl-edu/RL-exercises.git``
2. Install the uv package manager:
    * ``pip install uv``
3. Create a new environment:
    * ``uv venv --python 3.11``
4. Activate the new env:
    * ``source .venv/bin/activate``
5. Install this repository:
    * ``make install``


## Assignments
For information on how to publish your solutions please see `ASSIGNMENTS.md`.

## Code Quality Hacks
There are a few useful commands in this repository you should probably use.
- `make format` will format all your code using the formatter and linter ruff. This will make both your and our experience better.
- `make check` will check your code for formatting, linting, typing and docstyle. We recommend running this from time to time. It will also be checked when you commit your code.

## Relevant Packages
We use some packages and frameworks in these exercises you might not be familiar with. Here are some useful introduction links in case you run into trouble:
- We use [*Git*](http://rogerdudler.github.io/git-guide/), specifically GitHub and GitHub Classroom, for these exercises. Please make sure you're familiar with the basic commands, you will use them a lot! 
- [*Hydra*](https://hydra.cc/) is an argument parser that helps us keep an overview of arguments and offers additional functionality like running [sweeps](https://hydra.cc/docs/intro/#multirun) of the same script with a range of arguments, [submit to compute clusters](https://hydra.cc/docs/plugins/submitit_launcher/) or hyperparameter tuning using [Bayesian Optimization](https://github.com/automl-private/hydra-smac-sweeper) or [alternate methods](https://github.com/facebookresearch/how-to-autorl).
- [*PyTorch*](https://pytorch.org/) is what we use for deep RL later in the exercises. You likely won't need a deep knowledge of the package, but understanding the basic functionality is useful. They have a [DQN example for RL](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) which is also the point in the lecture where we start using deep learning.
- [*RLiable*](https://github.com/google-research/rliable) is the standard library used for robust evaluation and plotting performance curves in RL. We highly encourage you to use `rliable` and its metrics (like IQM, confidence intervals, optimality gaps) for evaluating your RL agents across multiple seeds in your exercises. You can find an example usage in `rl_exercises/week_1/rliable_example.py`.
- [*JupyterLab*](https://jupyter.org/) enables interactive coding. We will use this mainly for visualizing agent behaviour and performance.
- Our [*Pre-commit conditions*](https://pre-commit.com/) contain good practice helpers for your code - including linting, formatting and typing. We're not trying to annoy you with these, we want to ensure a high code standard and encourage you to adopt general best practices. The command `make pre-commit` will check if you're ready to commit.
