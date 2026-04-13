# RL Training Demo
Here we show you how to train an agent on an environment using popular librariers.
We train and optimize a BipedalWalker (gymnasium) using a Soft Actor-Critic (SAC, https://arxiv.org/abs/1801.01290).

For this we provide a notebook and a training script you are welcome to checkout.

## Train
If you want to use the traininc script, activate your conda env and run the following in this dir (be advised, this will take quite a bit of time!):
```bash
python train_sac.py
```
Check out the folder `configs` for possible parameters. You can read [here](https://hydra.cc/docs/advanced/override_grammar/basic/) how to set parameters via the commandline.

### Visualize Training Progress
```bash
tensorboard --logdir .
```

## Replay
The notebook lets you do the same training, but inspect it closer. I will take some time again, but if you want to know a bit more about the code it can be a nice intro. Understanding it is not required at all at this point, however.

## RLiable Example
We use `rliable` to evaluate RL algorithms more robustly. You can find an example script `rliable_example.py` in this folder alongside some dummy data (`demo_data_seed_*.csv`). Run it to see how `rliable` aggregates performance and computes confidence intervals across multiple random seeds!
```bash
python rliable_example.py
```

# Exercise

The goal of this first exercise is to set up teams and learn about git and the workflow for future exercises. 
If you have not already set up this repository to try out last week's demo, please make sure to do so now!

## 1. Form teams of up to 3 students
Most exercises will require you to implement some of the techniques you learn during the course.
Git is one of the most widely used version control systems and allows you to easily collaborate with others on code from the same repository.

Exercises have to be handed in teams of up to 3 students. When you have found your partners, open the GitHub Classroom Link provided via StudIP,
create a group (you will have to name the group yourself) and both join that group. This will allow you to clone the template repository in
which you can add your solutions to this exercise sheet.

*Note*: Make sure you and your team-mates are happy with each other. GitHub Classroom does not allow to change your groups mid semester.

## 2. Get familiar with git and GitHub Classroom
To show that you are familiar with the standard git add, commit and push steps, add a file called `rl_exercises/members.txt` to your repository.
The file should contain the names of all members in the following way:

```
member 1: name1
member 2: name2
member 3: name3
```

Afterwards you can push to submit.
We make use of GitHub Classrooms autograde functionality. 
Essentially, for most exercise sheets we will require you to pass unit tests which are automatically evaluated whenever you push to GitHub. 
To demonstrate this process, for this exercise we run a test that expects the above file to be present and to contain three lines as above (make sure to replace name1, name2 and name3).
If your group has less than 3 students, just add any name you like.
You will be informed if the tests executed successfully or not, but to be sure you should run `make test-week-1` or `pytest tests/week_1` before pushing your solution.
