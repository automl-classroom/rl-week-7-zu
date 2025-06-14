import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

# === 配置 ===
agents = ["dqn", "rnd_dqn"]
n_seeds = 2
data_dir = "rl_exercises/week_7/results"  # or "results"

train_scores = {}
all_steps = []

# === 读取并对齐每个 agent 的数据 ===
for agent in agents:
    rewards_all_seeds = []
    steps_all_seeds = []
    for seed in range(n_seeds):
        filename = os.path.join(data_dir, f"{agent}_training_data_seed_{seed}.csv")
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found")
            continue
        df = pd.read_csv(filename)
        rewards_all_seeds.append(df["rewards"].to_numpy())
        steps_all_seeds.append(df["steps"].to_numpy())

    if len(rewards_all_seeds) == 0:
        continue

    # 对齐长度
    min_len = min(len(r) for r in rewards_all_seeds)
    rewards_all_seeds = [r[:min_len] for r in rewards_all_seeds]
    steps_all_seeds = [s[:min_len] for s in steps_all_seeds]

    train_scores[agent] = np.stack(rewards_all_seeds)
    all_steps.append(steps_all_seeds[0])  # 默认第一个 seed 的步长为横轴

# === 对齐所有 agent 的 steps 长度（再次取最短）
min_len = min(len(s) for s in all_steps)
steps = all_steps[0][:min_len]
for agent in train_scores:
    train_scores[agent] = train_scores[agent][:, :min_len]

# === IQM 计算 ===
iqm = lambda scores: np.array(
    [metrics.aggregate_iqm(scores[:, t]) for t in range(scores.shape[1])]
)
iqm_scores, iqm_cis = get_interval_estimates(train_scores, iqm, reps=2000)

# === 绘图 ===
plot_sample_efficiency_curve(
    steps + 1,
    iqm_scores,
    iqm_cis,
    algorithms=list(train_scores.keys()),
    xlabel="Environment Steps",
    ylabel="IQM Score",
)
plt.title("Sample Efficiency Comparison: DQN vs RND-DQN")
plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=300)
plt.show()
