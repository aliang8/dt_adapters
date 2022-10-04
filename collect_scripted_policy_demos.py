from gc import collect
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from tests.metaworld.envs.mujoco.sawyer_xyz.utils import trajectory_summary
from metaworld.policies import *
from mw_utils import ENVS_AND_SCRIPTED_POLICIES, initialize_env, create_video_grid
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
from mw_dataset import OBJECTS_TO_ENV
from general_utils import split
from garage.envs import GymEnv
from garage.np import discount_cumsum, stack_tensor_dict_list


def rollout(
    env,
    agent,
    *,
    max_episode_length=np.inf,
    animated=False,
    pause_per_frame=None,
    deterministic=False,
):
    """Sample a single episode of the agent in the environment."""
    env_steps = []
    agent_infos = []
    observations = []
    last_obs, episode_infos = env.reset()
    agent.reset()
    episode_length = 0
    if animated:
        env.visualize()
    while episode_length < (max_episode_length or np.inf):
        if pause_per_frame is not None:
            time.sleep(pause_per_frame)
        a = agent.get_action(last_obs)
        agent_info = {}
        if deterministic and "mean" in agent_info:
            a = agent_info["mean"]

        a[:3] = np.clip(a[:3], -1, 1)  # clip action
        es = env.step(a)
        env_steps.append(es)
        observations.append(last_obs)
        agent_infos.append(agent_info)
        episode_length += 1
        if es.last:
            break
        last_obs = es.observation

    return dict(
        episode_infos=episode_infos,
        observations=np.array(observations),
        actions=np.array([es.action for es in env_steps]),
        rewards=np.array([es.reward for es in env_steps]),
        agent_infos=stack_tensor_dict_list(agent_infos),
        env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
        dones=np.array([es.terminal for es in env_steps]),
    )


def collect_dataset(config, envs, results_queue, wandb_run):
    for (env_name, policy, act_noise_pct, sr) in envs:
        if "v2" not in env_name:
            continue

        print(env_name)
        env = initialize_env(env_name, obj_randomization=True, hide_goal=True)
        max_path_length = env.max_path_length
        env = GymEnv(env, max_episode_length=max_path_length)
        videos = []

        total_success = 0
        # for i in range(DEMOS_PER_ENV):
        while total_success < config.demos_per_env:
            path = rollout(env, policy, animated=True)
            videos.append(path["env_infos"]["frames"])
            success = path["observations"].shape[0] != 500
            if success:
                total_success += 1

            if results_queue is not None and success:
                results_queue.put((env_name, total_success, path))

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
    while True:
        out = results_queue.get()
        if out is not None:
            env_name, episode, path = out
            g = hf.create_group(f"{env_name}/demo_{episode}")
            g.create_dataset("obs", data=path["observations"])
            g.create_dataset("action", data=path["actions"])
            g.create_dataset("reward", data=path["rewards"])
            g.create_dataset("done", data=path["dones"])
            # g.create_dataset("images", data=path["env_infos"]["frames"])
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

    envs = ENVS_AND_SCRIPTED_POLICIES

    # if DEBUG:
    #     random.shuffle(envs)
    #     envs = envs[:2]

    # if FILTER_ENVS_BY_OBJ:
    #     env_names = OBJECTS_TO_ENV[FILTER_ENVS_BY_OBJ]
    #     _envs = []
    #     for row in envs:
    #         if row[0].replace("-", "_") in env_names:
    #             _envs.append(row)
    #     envs = _envs

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
                args=(config, env_chunks[rank], results_queue, wandb_run),
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
