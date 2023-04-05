import numpy as np
from collections import defaultdict as dd
import dt_adapters.utils as utils


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def compute_eval_metrics(rollouts):
    metrics = dd(int)
    lengths = []

    success_only_metrics = [
        "num_successes",
        "max_reward_for_success",
        "max_return_for_success",
    ]

    # Iterate through each evaluation tarjectory
    for traj in rollouts:
        traj["returns_to_go"] = discount_cumsum(traj["rewards"], gamma=1.0)

        if "rewards" in traj:
            rewards = traj["rewards"]
            returns = traj["returns_to_go"]

            metrics["returns"] += returns.mean()
            metrics["rewards"] += rewards.mean()

            metrics["max_return"] += returns.max()
            metrics["max_reward"] += rewards.max()

            metrics["return_std"] = np.std(returns)

        traj_length = traj["actions"].shape[0]

        metrics["episode_length"] += traj_length
        lengths.append(traj_length)

        metrics["success_rate"] += traj["success"]

        if traj["success"]:
            metrics["num_successes"] += 1

            if "rewards" in traj:
                metrics["max_return_for_success"] += returns.max()
                metrics["max_reward_for_success"] += rewards.max()

    for k, _ in metrics.items():
        if k not in success_only_metrics:
            metrics[k] /= len(rollouts)

    metrics["max_reward_for_success"] /= metrics["num_successes"] + 1e-6
    metrics["max_return_for_success"] /= metrics["num_successes"] + 1e-6
    metrics["episode_length_std"] = np.std(lengths)

    return metrics
