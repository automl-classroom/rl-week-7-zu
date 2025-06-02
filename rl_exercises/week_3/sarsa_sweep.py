"""Run multiple SARSA episodes using Hydra-configured components.

This script uses Hydra to instantiate the environment, policy, and SARSA agent from config files,
then runs multiple episodes and returns the average total reward.
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


def run_episodes(agent, env, num_episodes=5):
    """Run multiple episodes using the SARSA algorithm.

    Each episode is executed with the agent's current policy. The agent updates its Q-values
    after every step using the SARSA update rule.

    Parameters
    ----------
    agent : object
        An agent implementing `predict_action` and `update_agent`.
    env : gym.Env
        The environment in which the agent interacts.
    num_episodes : int, optional
        Number of episodes to run, by default 5.

    Returns
    -------
    float
        Mean total reward across all episodes.
    """

    # TODO: Extend the run_episodes function.
    # Currently, the funciton runs only one episode and returns the total reward without discounting.
    # Extend it to run multiple episodes and store the total discounted rewards in a list.
    # Finally, return the mean discounted reward across episodes.

    total = 0.0
    state, _ = env.reset()
    done = False
    action = agent.predict_action(state)
    while not done:
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        next_action = agent.predict_action(next_state)
        agent.update_agent(state, action, reward, next_state, next_action, done)
        total += reward
        state, action = next_state, next_action
    return total


# Decorate the function with the path of the config file and the particular config to use
@hydra.main(
    config_path="../configs/agent/", config_name="sarsa_sweep", version_base="1.1"
)
def main(cfg: DictConfig) -> dict:
    """Main function to run SARSA with Hydra-configured components.

    This function sets up the environment, policy, and agent using Hydra-based
    configuration, seeds them for reproducibility, and runs multiple episodes.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing `env`, `policy`, `agent`, and optionally `seed`.

    Returns
    -------
    float
        Mean total reward across the episodes.
    """

    # Hydra-instantiate the env
    env = instantiate(cfg.env)
    # instantiate the policy (passing in env!)
    policy = instantiate(cfg.policy, env=env)
    # 3) instantiate the agent (passing in env & policy)
    agent = instantiate(cfg.agent, env=env, policy=policy)

    # 4) (optional) reseed for reproducibility
    if cfg.seed is not None:
        env.reset(seed=cfg.seed)
        env.action_space.seed(cfg.seed)

    # 5) run & return reward
    total_reward = run_episodes(agent, env, cfg.num_episodes)
    return total_reward


if __name__ == "__main__":
    main()
