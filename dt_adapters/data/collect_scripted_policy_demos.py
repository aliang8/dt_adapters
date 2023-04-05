"""
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 dt_adapters/collect_scripted_policy_demos.py \
    --task_name pick_place \
    --data_dir /data/anthony/dt_adapters/data \
    --num_demos 10 \
"""
from PIL import Image
import numpy as np
import wandb
import random
import torch
import h5py
import os
import time
import hydra
import multiprocessing as mp
from collections import defaultdict as dd
from metaworld.policies import *


def rollout(
    env,
    agent,
    max_episode_length=np.inf,
    animated=False,
    pause_per_frame=None,
    deterministic=False,
):
    """Sample a single episode of the agent in the environment."""
    env_steps = []
    states = []
    actions = []
    rewards = []
    dones = []

    last_obs = env.reset()
    states.append(last_obs)

    frames = {
        f"{camera_name}": [
            env.sim.render(height=256, width=256, camera_name=camera_name)
        ]
        for camera_name in image_keys
    }

    agent.reset()
    episode_length = 0
    traj_success = False

    while episode_length < (max_episode_length or np.inf):
        a = agent.get_action(last_obs)
        a[:3] = np.clip(a[:3], -1, 1)  # clip action
        obs, reward, terminate, info = env.step(a)
        terminate |= bool(info["success"])

        rewards.append(reward)
        actions.append(a)

        episode_length += 1
        if terminate:
            # print("SUCCESS")
            # print(episode_length)
            dones.append(1)
            traj_success = True
            break
        else:
            dones.append(0)

        states.append(obs)
        last_obs = obs

        last_frames = {
            f"{camera_name}": env.sim.render(
                height=256, width=256, camera_name=camera_name
            )
            for camera_name in image_keys
        }

        for k, v in last_frames.items():
            frames[k].append(v)

    for k, v in frames.items():
        frames[k] = np.array(v)

    return dict(
        states=np.array(states),
        actions=np.array(actions),
        rewards=np.array(rewards),
        frames=frames,
        traj_success=traj_success,
        dones=np.array(dones),
    )


def collect_trajectories(args, task_name):
    # initialize environment
    env = env_constructor(
        env_name=task_name,
        args=args,
        device="cuda",
    )

    while total_success < args.demos_per_env:
        path = rollout(env, policy, config, max_episode_length=500)
        videos.append(path["frames"])  # TODO: needs fixing
        if path["traj_success"]:
            total_success += 1

        if results_queue is not None and path["traj_success"]:
            results_queue.put((env_name, total_success, path))
        # time.sleep(5)

    print(f"{total_success}/{config.demos_per_env} successes")


def save_trajectories(args, task_name, trajectories):
    hf = h5py.File(os.path.join(args.data_dir, task_name), "w")

    # save each trajectory into hdf5 file
    for traj_num, traj in enumerate(trajectories):
        g = hf.create_group(f"demo_{traj_num}")
        g.create_dataset("states", data=traj["states"])
        g.create_dataset("actions", data=traj["actions"])
        g.create_dataset("rewards", data=traj["rewards"])
        g.create_dataset("dones", data=traj["dones"])
        g.create_dataset("images", data=traj["images"])

    hf.close()


def main(args):
    collect_trajectories(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="collecting scripted data from metaworld"
    )
    parser.add_argument("--task_name", default="")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--num_demos", default=10, type=int)

    args = parser.parse_args()

    main(args)
