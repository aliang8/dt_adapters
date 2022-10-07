import numpy as np
from collections import defaultdict as dd


def compute_eval_metrics(rollouts):
    metrics = dd(int)
    lengths = []

    success_only_metrics = [
        "num_successes",
        "max_reward_for_success",
        "max_return_for_success",
    ]

    for traj in rollouts:
        if "rewards" in traj:
            rewards = traj["rewards"]
            returns = traj["returns_to_go"]
            metrics["returns"] += returns.mean()
            metrics["rewards"] += rewards.mean()

            metrics["max_return"] += returns.max()
            metrics["max_reward"] += rewards.max()

            metrics["return_std"] = np.std(returns)

        length = traj["states"].shape[0]

        metrics["episode_length"] += length
        lengths.append(length)
        success = int(length != 500)

        metrics["success_rate"] += success
        if success:
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
