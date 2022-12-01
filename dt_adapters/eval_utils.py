import numpy as np
from collections import defaultdict as dd
import dt_adapters.general_utils as general_utils


def compute_eval_metrics(rollouts):
    metrics = dd(int)
    lengths = []

    success_only_metrics = [
        "num_successes",
        "max_reward_for_success",
        "max_return_for_success",
    ]

    for traj in rollouts:
        traj["returns_to_go"] = general_utils.discount_cumsum(
            traj["rewards"], gamma=1.0
        )

        if "rewards" in traj:
            rewards = traj["rewards"]
            returns = traj["returns_to_go"]
            metrics["returns"] += returns.mean()
            metrics["rewards"] += rewards.mean()

            metrics["max_return"] += returns.max()
            metrics["max_reward"] += rewards.max()

            metrics["return_std"] = np.std(returns)

        length = traj["actions"].shape[0]

        metrics["episode_length"] += length
        lengths.append(length)

        metrics["success_rate"] += traj["traj_success"]
        if traj["traj_success"]:
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
