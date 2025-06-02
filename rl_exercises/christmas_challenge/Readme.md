# Reinforcement Learning - Christmas Challenge

In this challenge, your task is to act better than random on the `cbench-v1` benchmark dataset in the `llvm-autophase-ic-v0` environment. 
The environment only works on Linux or MacOS.
You can use [Google Colab](colab.research.google.com/) or Linux Subsystem for Windows if you do not have such a system available.

You are free to use whatever RL library and algorithms you know!

The reward space is the [normalized instruction count](https://compilergym.com/llvm/index.html#ir-instruction-count).

**The episodes have no step limit by default, so remember to use the `TimeLimit` wrapper!**

**⚠ TODO: We will provide an evaluation script!**

## Levels
### Level 1
Please optimize your algorithm on the `dijkstra` benchmark as this will determine your [leaderboard](https://github.com/facebookresearch/CompilerGym#leaderboards) (a team from 2022 is on 1st place!) position.
Use different seeds and plot training progress. If PPO does not work, vary hyperparameters or use a different algo.

Evaluate your agent(s). Plot performance, e.g. using `rliable`.

To check that your agent actually learns something, evaluate a random policy and compare evaluation performances.
(1 bonus point)

### Level 2
Optimize hyperparameters for generalization performance on the complete benchmark.
Evaluate your agent on the complete benchmark to determine generalization performance and plot your final agent accordingly, e.g. using box plots or `rliable`.
The benchmarks are listed in the cbench-v1.txt file.
You can find other benchmarks [here](https://compilergym.com/llvm/index.html#datasets).
Remember to evaluate agents trained on different seeds.
Plot performance on all benchmarks aggregated and on individual benchmarks, what do you notice?
(1 bonus point)

### Level 3
Beat the leaderboard on [leaderboard](https://github.com/facebookresearch/CompilerGym#leaderboards). (1 bonus point)

--OR--

Optimize for generalization performance on the complete benchmark. Use contextual RL 
with our benchmark library [CARL](https://github.com/automl/CARL). For this, create a CARLCompilerGymEnv and define the instance space.
Train on one instance and on all instances and compare performance. (1 bonus point)


## Presentation
The best performing team presents and gets a bonus point.🙂

## Installation
For this challenge you need to install `compiler_gym` which you can do with

```bash
pip install -U compiler_gym
```
If this does not work for you, try these [instructions](https://github.com/facebookresearch/CompilerGym/blob/development/INSTALL.md).

If you use `make install`: If you have any error like "error in gym setup command: 'extras_require' must be a dictionary whose values are strings or lists of strings containing valid project/version requirement specifiers.", lower the `setuptools` version and install a specific `wheel` version (`pip install setuptools==65.5.0 wheel==0.38.4`). (Tested on Linux Subsystem)


## Getting Started
You can run training with `python rl_exercises/train_agent.py +exercise=compilergym`.

## Extra
You can also try your RL algorithms on your own custom benchmarks such as the example in the `custom_benchmarks` directory.
