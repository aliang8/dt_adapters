import numpy as np


def compute_eval_metrics(rollouts):
    metrics = {
        "success_rate": 0,
        "num_successes": 0,
        "rewards": 0,
        "episode_length": 0,
        "episode_length_std": 0,
        "max_reward": 0,
        "max_reward_for_success": 0,
    }

    lengths = []
    for traj in rollouts:
        rewards = traj["rewards"]
        metrics["rewards"] += rewards.mean()
        metrics["max_reward"] += rewards.max()
        length = traj["dones"].shape[0]
        metrics["episode_length"] += length
        lengths.append(length)
        success = int(length != 500)
        metrics["success_rate"] += success
        if success:
            metrics["num_successes"] += 1
            metrics["max_reward_for_success"] += rewards.max()

    for k, _ in metrics.items():
        if k not in ["num_successes", "max_reward_for_success"]:
            metrics[k] /= len(rollouts)

    metrics["max_reward_for_success"] /= metrics["num_successes"] + 1e-6
    metrics["episode_length_std"] = np.std(lengths)

    return metrics
