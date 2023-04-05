"""
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 dt_adapters/data/collect_scripted_policy_demos.py \
    --task_name pick-place-v2 \
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
from dt_adapters.envs.make_env import env_constructor

ENVS_AND_SCRIPTED_POLICIES = [
    # name, policy, action noise pct, success rate
    ["assembly-v2", SawyerAssemblyV2Policy(), 0.1, 0.70],
    ["basketball-v2", SawyerBasketballV2Policy(), 0.1, 0.96],
    ["bin-picking-v2", SawyerBinPickingV2Policy(), 0.1, 0.96],
    ["box-close-v2", SawyerBoxCloseV2Policy(), 0.1, 0.82],
    ["button-press-topdown-v2", SawyerButtonPressTopdownV2Policy(), 0.1, 0.93],
    ["button-press-topdown-wall-v2", SawyerButtonPressTopdownWallV2Policy(), 0.1, 0.95],
    ["button-press-v2", SawyerButtonPressV2Policy(), 0.1, 0.98],
    ["button-press-wall-v2", SawyerButtonPressWallV2Policy(), 0.1, 0.92],
    ["coffee-button-v2", SawyerCoffeeButtonV2Policy(), 0.1, 0.99],
    ["coffee-pull-v2", SawyerCoffeePullV2Policy(), 0.1, 0.82],
    ["coffee-push-v2", SawyerCoffeePushV2Policy(), 0.1, 0.88],
    ["dial-turn-v2", SawyerDialTurnV2Policy(), 0.1, 0.84],
    ["disassemble-v2", SawyerDisassembleV2Policy(), 0.1, 0.88],
    ["door-close-v2", SawyerDoorCloseV2Policy(), 0.1, 0.97],
    ["door-lock-v2", SawyerDoorLockV2Policy(), 0.1, 0.96],
    ["door-open-v2", SawyerDoorOpenV2Policy(), 0.1, 0.92],
    ["door-unlock-v2", SawyerDoorUnlockV2Policy(), 0.1, 0.97],
    ["drawer-close-v2", SawyerDrawerCloseV2Policy(), 0.1, 0.99],
    ["drawer-open-v2", SawyerDrawerOpenV2Policy(), 0.1, 0.97],
    # ["drawer-put-block-v2", SawyerDrawerPutBlockV2Policy(), 0.0, 0.97],
    ["faucet-close-v2", SawyerFaucetCloseV2Policy(), 0.1, 1.0],
    ["faucet-open-v2", SawyerFaucetOpenV2Policy(), 0.1, 0.99],
    ["hammer-v2", SawyerHammerV2Policy(), 0.1, 0.96],
    ["hand-insert-v2", SawyerHandInsertV2Policy(), 0.1, 0.86],
    ["handle-press-side-v2", SawyerHandlePressSideV2Policy(), 0.1, 0.98],
    ["handle-press-v2", SawyerHandlePressV2Policy(), 0.1, 1.0],
    ["handle-pull-v2", SawyerHandlePullV2Policy(), 0.1, 0.99],
    ["handle-pull-side-v2", SawyerHandlePullSideV2Policy(), 0.1, 0.71],
    ["peg-insert-side-v2", SawyerPegInsertionSideV2Policy(), 0.1, 0.87],
    ["lever-pull-v2", SawyerLeverPullV2Policy(), 0.1, 0.90],
    ["peg-unplug-side-v2", SawyerPegUnplugSideV2Policy(), 0.1, 0.80],
    ["pick-out-of-hole-v2", SawyerPickOutOfHoleV2Policy(), 0.1, 0.89],
    ["pick-place-v2", SawyerPickPlaceV2Policy(), 0.1, 0.83],
    ["pick-place-wall-v2", SawyerPickPlaceWallV2Policy(), 0.1, 0.83],
    ["plate-slide-back-side-v2", SawyerPlateSlideBackSideV2Policy(), 0.1, 0.95],
    ["plate-slide-back-v2", SawyerPlateSlideBackV2Policy(), 0.1, 0.94],
    ["plate-slide-side-v2", SawyerPlateSlideSideV2Policy(), 0.1, 0.78],
    ["plate-slide-v2", SawyerPlateSlideV2Policy(), 0.1, 0.97],
    ["reach-v2", SawyerReachV2Policy(), 0.1, 0.98],
    ["reach-wall-v2", SawyerReachWallV2Policy(), 0.1, 0.96],
    ["push-back-v2", SawyerPushBackV2Policy(), 0.0, 0.91],
    ["push-v2", SawyerPushV2Policy(), 0.1, 0.88],
    ["push-wall-v2", SawyerPushWallV2Policy(), 0.1, 0.82],
    ["shelf-place-v2", SawyerShelfPlaceV2Policy(), 0.1, 0.89],
    ["soccer-v2", SawyerSoccerV2Policy(), 0.1, 0.81],
    ["stick-pull-v2", SawyerStickPullV2Policy(), 0.1, 0.81],
    ["stick-push-v2", SawyerStickPushV2Policy(), 0.1, 0.95],
    ["sweep-into-v2", SawyerSweepIntoV2Policy(), 0.1, 0.86],
    ["sweep-v2", SawyerSweepV2Policy(), 0.0, 0.99],
    ["window-close-v2", SawyerWindowCloseV2Policy(), 0.1, 0.95],
    ["window-open-v2", SawyerWindowOpenV2Policy(), 0.1, 0.93],
]


def rollout(
    env,
    agent,
    max_episode_length=np.inf,
    image_dim=256,
    camera_names=[],  # list of camera names to render
):
    """Sample a single trajectory of the agent in the environment."""
    states, actions, rewards, dones = [], [], [], []

    # add the first observation
    last_obs = env.reset()
    states.append(last_obs)

    frames = {
        f"{camera_name}": [
            env.sim.render(height=image_dim, width=image_dim, camera_name=camera_name)
        ]
        for camera_name in camera_names
    }

    episode_length = 0
    success = False

    # run until either the trajectory is done or we reach max episode length
    while episode_length < (max_episode_length or np.inf):
        action = agent.get_action(last_obs)
        action[:3] = np.clip(action[:3], -1, 1)  # clip action

        # step in environment
        obs, reward, terminate, info = env.step(action)
        terminate |= bool(info["success"])

        rewards.append(reward)
        actions.append(action)

        episode_length += 1
        if terminate:
            dones.append(1)
            success = True
            break
        else:
            dones.append(0)

        states.append(obs)
        last_obs = obs

        last_frames = {
            f"{camera_name}": env.sim.render(
                height=image_dim, width=image_dim, camera_name=camera_name
            )
            for camera_name in camera_names
        }

        for k, v in last_frames.items():
            frames[k].append(v)

    for k, v in frames.items():
        frames[k] = np.array(v)

    trajectory = dict(
        states=np.array(states),
        actions=np.array(actions),
        rewards=np.array(rewards),
        images=frames,
        success=success,
        dones=np.array(dones),
    )

    return trajectory


def collect_trajectories(args, task_name, policy):
    # initialize environment
    env = env_constructor(
        domain="metaworld",
        task_name=task_name,
    )

    trajectories = []
    total_success = 0
    while total_success < args.num_demos:
        path = rollout(env, policy, max_episode_length=500, camera_names=["corner"])

        # only save trajectories if they are successful
        if path["success"]:
            trajectories.append(path)
            total_success += 1

    save_trajectories(args, task_name, trajectories)
    print(f"Saved {len(trajectories)} trajectories for {task_name}...")


def save_trajectories(args, task_name, trajectories):
    hf = h5py.File(os.path.join(args.data_dir, f"{task_name}.hdf5"), "w")

    # save each trajectory into hdf5 file
    for traj_num, traj in enumerate(trajectories):
        g = hf.create_group(f"demo_{traj_num}")
        g.create_dataset("states", data=traj["states"])
        g.create_dataset("actions", data=traj["actions"])
        g.create_dataset("rewards", data=traj["rewards"])
        g.create_dataset("dones", data=traj["dones"])

        for camera_name, v in traj["images"].items():
            g.create_dataset(f"images/{camera_name}", data=v)

    # compute average trajectory length
    print(
        "Average trajectory length:", np.mean([len(t["states"]) for t in trajectories])
    )
    print("Average reward:", np.mean([np.sum(t["rewards"]) for t in trajectories]))

    hf.close()


def main(args):
    if not args.task_name:
        # collect trajectories for every task
        for task_name, policy, _, _ in ENVS_AND_SCRIPTED_POLICIES:
            collect_trajectories(args, task_name, policy)
    else:
        policy = [
            p for n, p, _, _ in ENVS_AND_SCRIPTED_POLICIES if n == args.task_name
        ][0]
        collect_trajectories(args, args.task_name, policy)


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
