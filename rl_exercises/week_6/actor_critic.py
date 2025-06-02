"""
On-policy Actor-Critic supporting four baseline modes: 'none', 'avg', 'value', and 'gae',
trained for a total number of environment steps with periodic evaluation.
Adds GAE for low-variance advantage estimation.
"""

from typing import Any, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_6.networks import (  # adjust import path as needed
    Policy,
    ValueNetwork,
)
from torch.distributions import Categorical


def set_seed(env: gym.Env, seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class ActorCriticAgent(AbstractAgent):
    """
    On-policy Actor-Critic agent with configurable baselines and GAE support.

    Parameters
    ----------
    env : gym.Env
        The environment to train the agent in.
    lr_actor : float, optional
        Learning rate for the policy network (default is 5e-4).
    lr_critic : float, optional
        Learning rate for the value network (default is 1e-3).
    gamma : float, optional
        Discount factor for computing returns (default is 0.99).
    gae_lambda : float, optional
        Lambda parameter for Generalized Advantage Estimation (default is 0.95).
    seed : int, optional
        Random seed for reproducibility (default is 0).
    hidden_size : int, optional
        Hidden layer size for policy and value networks (default is 128).
    baseline_type : str, optional
        Type of baseline: 'none', 'avg', 'value', or 'gae' (default is 'value').
    baseline_decay : float, optional
        Decay factor for running average baseline (default is 0.9).
    """

    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        seed: int = 0,
        hidden_size: int = 128,
        baseline_type: str = "value",  # 'none', 'avg', 'value', or 'gae'
        baseline_decay: float = 0.9,
    ) -> None:
        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.baseline_type = baseline_type
        self.baseline_decay = baseline_decay

        # policy
        self.policy = Policy(env.observation_space, env.action_space, hidden_size)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)

        # critic for 'value' and 'gae'
        if baseline_type in ("value", "gae"):
            self.value_fn = ValueNetwork(env.observation_space, hidden_size)
            self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=lr_critic)

        # running average baseline for 'avg' -- initialized to zero
        # this is used to compute advantages in the 'avg' baseline mode
        # it is updated during training
        if baseline_type == "avg":
            self.running_return = 0.0

    def predict_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """
        Predict action from current policy given a state.

        Parameters
        ----------
        state : np.ndarray
            Current observation from the environment.
        evaluate : bool, optional
            If True, selects the action deterministically (default is False).

        Returns
        -------
        action : int
            Selected action.
        log_prob : torch.Tensor or None
            Log probability of the selected action (None if in evaluation mode).
        """

        t = torch.from_numpy(state).float()
        probs = self.policy(t).squeeze(0)
        if evaluate:
            return int(torch.argmax(probs).item()), None
        dist = Categorical(probs)
        action = dist.sample().item()
        return action, dist.log_prob(torch.tensor(action))

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute discounted returns from a list of rewards.

        Parameters
        ----------
        rewards : list of float
            Rewards collected during the episode.

        Returns
        -------
        returns : torch.Tensor
            Discounted return for each timestep.
        """

        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def compute_advantages(
        self, states: List[np.ndarray], rewards: List[float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using a learned value function.

        Parameters
        ----------
        states : list of np.ndarray
            States encountered during the trajectory.
        rewards : list of float
            Rewards collected during the episode.

        Returns
        -------
        advantages : torch.Tensor
            Advantage values for each timestep.
        returns : torch.Tensor
            Discounted returns.
        """
        # TODO: convert rewards into discounted returns

        # TODO: convert states list into a torch batch and compute state-values

        # TODO: compute raw advantages = returns - values

        # TODO: normalize advantages to zero mean and unit variance and use 1e-8 for numerical stability

        # return normalized advantages and returns
        return None

    def compute_gae(
        self,
        states: List[np.ndarray],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Parameters
        ----------
        states : list of np.ndarray
            Current states.
        rewards : list of float
            Rewards received after taking actions.
        next_states : list of np.ndarray
            Next states observed after actions.
        dones : list of bool
            Whether the episode terminated at each step.

        Returns
        -------
        advantages : torch.Tensor
            GAE advantages for each timestep.
        returns : torch.Tensor
            Target returns for training the critic.
        """

        # TODO: compute values and next_values using your value_fn

        # TODO: compute deltas: one-step TD errors

        # TODO: accumulate GAE advantages backwards

        # TODO: compute returns using advantages and values

        # TODO: normalize advantages to zero mean and unit variance and use 1e-8 for numerical stability

        # TODO: advantages, returns  # replace with actual values (detach both to avoid re-entering the graph)

        return None

    def update_agent(
        self,
        trajectory: List[Tuple[np.ndarray, int, float, np.ndarray, bool, Any]],
    ) -> Tuple[float, float]:
        """
        Update the policy and value networks using the collected trajectory.

        Parameters
        ----------
        trajectory : list of tuple
            List of (state, action, reward, next_state, done, log_prob) from one episode.

        Returns
        -------
        policy_loss : float
            Scalar loss for the policy network.
        value_loss : float
            Scalar loss for the value network (0.0 if no critic is used).
        """

        states, actions, rewards, next_states, dones, log_probs = zip(*trajectory)

        # select advantage method
        if self.baseline_type == "gae":
            adv, ret = self.compute_gae(
                list(states), list(rewards), list(next_states), list(dones)
            )
        elif self.baseline_type == "value":
            adv, ret = self.compute_advantages(list(states), list(rewards))
        elif self.baseline_type == "avg":
            # compute returns and advantages using running average baseline
            ret = self.compute_returns(list(rewards))

            # TODO: compute advantages by subtracting running return
            adv = ...

            # TODO: normalize advantages to zero mean and unit variance and use 1e-8 for numerical stability
            # (Reminder, use unbiased=False for torch tensors)

            # TODO: update running return using baseline decay
            # (x = baseline_decay * x + (1 - baseline_decay) * mean return)
        else:
            ret = self.compute_returns(list(rewards))
            adv = (ret - ret.mean()) / (ret.std(unbiased=False) + 1e-8)

        # policy update
        logp_t = torch.stack(log_probs)
        policy_loss = -(logp_t * adv).mean()
        self.policy_optimizer.zero_grad()
        if policy_loss.requires_grad:
            policy_loss.backward()
            self.policy_optimizer.step()

        # critic update
        if self.baseline_type in ("value", "gae"):
            vals = self.value_fn(
                torch.stack([torch.from_numpy(s).float() for s in states])
            )
            value_loss = F.mse_loss(vals, ret)
            self.value_optimizer.zero_grad()
            if value_loss.requires_grad:
                value_loss.backward()
                self.value_optimizer.step()
        else:
            value_loss = 0.0

        return float(policy_loss.item()), float(
            value_loss if isinstance(value_loss, float) else value_loss.item()
        )

    def evaluate(
        self, eval_env: gym.Env, num_episodes: int = 10
    ) -> Tuple[float, float]:
        """
        Evaluate policy over multiple episodes.

        Parameters
        ----------
        eval_env : gym.Env
            Environment for evaluation.
        num_episodes : int, optional
            Number of episodes to run (default is 10).

        Returns
        -------
        mean_return : float
            Average episode return.
        std_return : float
            Standard deviation of returns.
        """
        self.policy.eval()
        returns: List[float] = []
        with torch.no_grad():
            for _ in range(num_episodes):
                state, _ = eval_env.reset()
                done = False
                total_r = 0.0
                while not done:
                    action, _ = self.predict_action(state, evaluate=True)
                    state, r, term, trunc, _ = eval_env.step(action)
                    done = term or trunc
                    total_r += r
                returns.append(total_r)
        self.policy.train()
        return float(np.mean(returns)), float(np.std(returns))

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10000,
        eval_episodes: int = 5,
    ) -> None:
        """
        Train the agent for a given number of steps.

        Parameters
        ----------
        total_steps : int
            Total number of training steps.
        eval_interval : int, optional
            Interval (in steps) at which to evaluate the policy (default is 10000).
        eval_episodes : int, optional
            Number of episodes to run for each evaluation (default is 5).
        """
        eval_env = gym.make(self.env.spec.id)
        step_count = 0

        while step_count < total_steps:
            state, _ = self.env.reset()
            done = False
            trajectory: List[Any] = []

            while not done and step_count < total_steps:
                action, logp = self.predict_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                trajectory.append(
                    (state, action, float(reward), next_state, done, logp)
                )
                state = next_state
                step_count += 1

                if step_count % eval_interval == 0:
                    mean_r, std_r = self.evaluate(eval_env, num_episodes=eval_episodes)
                    print(
                        f"[Eval ] Step {step_count:6d} AvgReturn {mean_r:5.1f} ± {std_r:4.1f}"
                    )

            policy_loss, value_loss = self.update_agent(trajectory)
            total_return = sum(r for _, _, r, *_ in trajectory)
            print(
                f"[Train] Step {step_count:6d} Return {total_return:5.1f} Policy Loss {policy_loss:.3f} Value Loss {value_loss:.3f}"
            )

        print("Training complete.")


@hydra.main(
    config_path="../configs/agent/", config_name="actor-critic", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)
    agent = ActorCriticAgent(
        env,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,
        gamma=cfg.agent.gamma,
        gae_lambda=cfg.agent.get("gae_lambda", 0.95),  # FIXME
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
        baseline_type=cfg.agent.baseline_type,
        baseline_decay=cfg.agent.get("baseline_decay", 0.9),
    )
    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )


if __name__ == "__main__":
    main()
