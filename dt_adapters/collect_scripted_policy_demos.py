"""
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 dt_adapters/collect_scripted_policy_demos.py \
    --config-name=data_collection \
    data_file=trajectories_all_images_multiview_50.hdf5 \
    multiprocessing=True \
    demos_per_env=50 

CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 dt_adapters/collect_scripted_policy_demos.py \
    --config-name=data_collection \
    data_file=trajectories_compositional.hdf5 \
    multiprocessing=False \
    tasks=[drawer-put-block-v2]
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
from garage.envs import GymEnv
from garage.np import discount_cumsum, stack_tensor_dict_list

from dt_adapters.mw_utils import (
    ENVS_AND_SCRIPTED_POLICIES,
    initialize_env,
    create_video_grid,
)
from dt_adapters.mw_constants import OBJECTS_TO_ENV
from dt_adapters.general_utils import split
import dt_adapters.data.utils as data_utils
from dt_adapters.envs.make_env import env_constructor
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.policies import *
from dt_adapters.models.state_embedding_net import StateEmbeddingNet


def rollout(
    env,
    agent,
    config,
    *,
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
        for camera_name in config.data.image_keys
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
            for camera_name in config.data.image_keys
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


def collect_dataset(config, envs, results_queue, wandb_run):
    for (env_name, policy, act_noise_pct, sr) in envs:
        if "v2" not in env_name:
            continue

        print(env_name)
        # env = initialize_env(
        #     env_name,
        #     obj_randomization=True,
        #     hide_goal=False,
        #     observation_mode="state_image",
        # )
        env = env_constructor(
            env_name=env_name,
            config=config,
            device="cuda",
        )

        videos = []

        total_success = 0
        # for i in range(DEMOS_PER_ENV):
        while total_success < config.demos_per_env:
            path = rollout(env, policy, config, max_episode_length=500)
            videos.append(path["frames"])  # TODO: needs fixing
            if path["traj_success"]:
                total_success += 1

            if results_queue is not None and path["traj_success"]:
                results_queue.put((env_name, total_success, path))
            # time.sleep(5)

        print(f"{total_success}/{config.demos_per_env} successes")

        if config.log_to_wandb and not config.debug:
            videos = create_video_grid(videos, height=128, width=128)
            wandb_run.log(
                {
                    f"{env_name}/rollout_videos": wandb.Video(
                        videos, fps=10, format="gif"
                    )
                }
            )


def handle_output(config, results_queue):
    hf = h5py.File(os.path.join(config.data_dir, config.data_file), "w")
    # img_preprocessor, depth_img_preprocessor = data_utils.get_preprocessor(
    #     config.vision_backbone
    # )
    # img_encoder, depth_img_encoder = data_utils.get_visual_encoders(
    #     config.vision_backbone, "cuda"
    # )

    # print("done initializing visual encoders")
    state_embedding = StateEmbeddingNet(config.state_encoder)

    count = 0
    while True:
        out = results_queue.get()

        if out is not None:
            env_name, episode, path = out

            count += 1
            print(count)
            # if not config.save_image:
            #     img_feats = data_utils.extract_image_feats(
            #         path["frames"],
            #         img_preprocessor,
            #         img_encoder,
            #         depth_img_preprocessor,
            #         depth_img_encoder,
            #         vision_backbone=config.vision_backbone,
            #     )

            #     print("done extracting features")
            #     print(img_feats.shape)

            g = hf.create_group(f"{env_name}/demo_{episode}")
            g.create_dataset("states", data=path["states"])
            g.create_dataset("actions", data=path["actions"])
            g.create_dataset("rewards", data=path["rewards"])
            g.create_dataset("dones", data=path["dones"])

            # if config.save_image:
            img = hf.create_group(f"{env_name}/demo_{episode}/img_feats")
            for k in path["frames"].keys():
                img.create_dataset(k, data=state_embedding(path["frames"][k]))
            # else:
            #     g.create_dataset("img_feats", data=img_feats.cpu().numpy())
        else:
            break
    hf.close()


@hydra.main(config_path="configs", config_name="data_collection")
def main(config):
    if config.log_to_wandb and not config.debug:
        wandb_run = wandb.init(
            name="videos",
            group="scripted_policies",
            project="dt-adapters",
            config={},
            entity="glamor",
        )
    else:
        wandb_run = None

    if config.tasks:
        envs = [env for env in ENVS_AND_SCRIPTED_POLICIES if env[0] in config.tasks]
    else:
        envs = ENVS_AND_SCRIPTED_POLICIES

    print(len(envs))

    if config.multiprocessing and not config.debug:
        torch.multiprocessing.set_start_method("spawn")
        results_queue = mp.Queue()

        proc = mp.Process(
            target=handle_output,
            args=(
                config,
                results_queue,
            ),
        )

        processes = []
        proc.start()

        num_processes = min(len(envs), config.num_processes)
        env_chunks = list(split(envs, num_processes))

        for rank in range(num_processes):
            p = mp.Process(
                target=collect_dataset,
                args=(
                    config,
                    env_chunks[rank],
                    results_queue,
                    wandb_run,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        results_queue.put(None)
        proc.join()
    else:
        collect_dataset(config, envs, None, None)


if __name__ == "__main__":
    main()
