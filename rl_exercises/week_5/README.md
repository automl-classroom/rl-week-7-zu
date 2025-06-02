# Week 5: Policy Gradient

This week, you will implement your first policy-based reinforcement learning algorithm — REINFORCE — and conduct experiments to better understand when and why it performs better (or worse) than value-based methods like DQN.

Wherever relevant, use [RLiable](https://github.com/google-research/rliable) for plotting your results.  If you wish to explore design decisions, use [HyperSweeper](https://github.com/automl/hypersweeper) for hyperparameter tuning.

## Level 1
### REINFORCE Implementation
Complete the implementation in `policy_gradient.py` to train a stochastic policy using the REINFORCE algorithm. We’ve provided test cases to help you verify the correctness of your implementation. By default, this trains on `CartPole-v1`. You can change to to any environment of your choice from classic control suite.

All relevant training configurations can be found in `configs/agent/reinforce.yaml`

## Level 2
### Empirical Analysis of Policy Gradients

Investigate how various factors influence REINFORCE's performance and compare it with DQN. Use RLiable to visualize your results.

Investigate the following aspects:
- How does the trajectory length affect training stability and convergence?
- What is the impact of network architecture and learning rate?
- How does the sample complexity of REINFORCE compare to DQN?

Support your discussion with relevant plots and provide clear documentation of your insights.

## Level 3
### When Does REINFORCE Beat DQN (and Vice Versa)?

This exercise is intended to help you build a deeper intuition for when to use policy-based versus value-based reinforcement learning methods. Design a controlled experimental setup to show one of the following:
- DQN outperforms REINFORCE
- REINFORCE outperforms DQN

You may either modify existing environments or create new ones for these experiments. Use RLiable to compare the performance of the two algorithms in your setup. Explain why your chosen setup favors one algorithm over the other, based on their underlying algorithmic principles (e.g., stochasticity handling, exploration, function approximation bias). 