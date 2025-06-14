import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from minigrid.wrappers import FlatObsWrapper
from rl_exercises.week_4.networks import QNetwork


def evaluate_policy(snapshot_q_path, env_name="MiniGrid-Empty-5x5-v0", seed=0):
    env = FlatObsWrapper(gym.make(env_name))
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim, n_actions)
    q_net.load_state_dict(torch.load(snapshot_q_path))
    q_net.eval()

    state, _ = env.reset(seed=seed)
    trajectory = []

    for _ in range(100):
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(q_net(s_tensor), dim=1).item()
        state, _, done, truncated, _ = env.step(action)
        position = env.agent_pos  # get agent x,y
        trajectory.append(position)
        if done or truncated:
            break

    return np.array(trajectory)


def plot_trajectories(trajs, steps):
    plt.figure(figsize=(12, 4))
    for i, (traj, step) in enumerate(zip(trajs, steps)):
        plt.subplot(1, 3, i + 1)
        xs, ys = zip(*traj)
        plt.hist2d(xs, ys, bins=5, range=[[0, 5], [0, 5]], cmap="Oranges", cmin=1)
        plt.title(f"Step {step}")
        plt.xticks(range(6))
        plt.yticks(range(6))
        plt.gca().invert_yaxis()
    plt.suptitle("State Visitation Heatmaps of RND Agent")
    plt.tight_layout()
    plt.savefig("rnd_exploration_snapshots.png")
    plt.show()


if __name__ == "__main__":
    steps = [5000, 25000, 50000]  # 根据你训练时的 num_frames
    trajs = [evaluate_policy(f"snapshot_q_{s}.pt") for s in steps]
    plot_trajectories(trajs, steps)
