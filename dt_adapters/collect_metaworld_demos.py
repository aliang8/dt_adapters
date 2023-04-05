import numpy as np
import h5py
import os
import hydra
from dt_adapters.models.state_embedding_net import StateEmbeddingNet
from dt_adapters.envs.obs_wrappers import MuJoCoPixelObs, StateEmbedding
from metaworld.policies import *
from dt_adapters.envs.make_env import env_constructor
from dt_adapters.mw_utils import ENVS_AND_SCRIPTED_POLICIES
from metaworld.policies.policy import Policy
from dt_adapters.envs.gym_env import GymEnv
from dt_adapters.envs.video_recorder import VideoRecorder
from typing import List
from omegaconf import DictConfig, OmegaConf


def rollout(
    domain: str,
    env: GymEnv,
    agent: Policy,
    max_episode_length: int = np.inf,
    image_keys: List[str] = [],
    image_width: int = 256,
    image_height: int = 256,
):
    """Sample a single episode of the agent in the environment."""
    states, actions, rewards, dones = [], [], [], []

    last_obs = env.reset()
    states.append(last_obs)  # add the first observation

    frames = {
        f"{camera_name}": [
            env.sim.render(
                height=image_height, width=image_width, camera_name=camera_name
            )
        ]
        for camera_name in image_keys
    }

    agent.reset()
    episode_length = 0
    traj_success = False

    while episode_length < (max_episode_length or np.inf):

        action = agent.get_action(last_obs)
        action[:3] = np.clip(action[:3], -1, 1)  # clip action
        obs, reward, terminate, info = env.step(action)

        if domain == "metaworld":
            terminate |= bool(info["success"])

        states.append(obs)
        last_obs = obs
        rewards.append(reward)
        actions.append(action)

        episode_length += 1

        if terminate:
            dones.append(1)
            traj_success = True
            break
        else:
            dones.append(0)

        last_frames = {
            f"{camera_name}": env.sim.render(
                height=image_height, width=image_width, camera_name=camera_name
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


def collect_trajectories(
    domain: str,
    data_dir: str,
    data_file: str,
    image_width: int = 256,
    image_height: int = 256,
    max_episode_length: int = 500,
    num_demos_per_env: int = 10,
    image_keys: List[str] = [],
    env_list: List[str] = [],
    vision_backbone: str = "",
    **kwargs,
):
    hf = h5py.File(os.path.join(data_dir, data_file), "w")
    state_embedding = StateEmbeddingNet(vision_backbone)

    for (env_name, policy, _, _) in env_list:
        print(env_name)

        env = env_constructor(
            domain=domain,
            env_name=env_name,
            image_keys=[],
            image_width=image_width,
            image_height=image_height,
            device="cuda",
        )

        total_successes = 0

        while total_successes < num_demos_per_env:
            path = rollout(
                domain,
                env,
                policy,
                max_episode_length=max_episode_length,
                image_keys=image_keys,
            )

            # save only successful trajectories
            if path["traj_success"]:
                g = hf.create_group(f"{env_name}/demo_{total_successes}")

                for k in ["states", "actions", "rewards", "dones"]:
                    g.create_dataset(k, data=path[k])

                # add preprocessed image features
                img_feats = hf.create_group(
                    f"{env_name}/demo_{total_successes}/img_feats"
                )
                for k in path["frames"].keys():
                    img_feats.create_dataset(k, data=state_embedding(path["frames"][k]))

                # add raw images, for visualization purposes
                imgs = hf.create_group(f"{env_name}/demo_{total_successes}/imgs")
                for k in path["frames"].keys():
                    imgs.create_dataset(k, data=path["frames"][k])

                total_successes += 1

        print(f"done with {env_name}, collected {total_successes} rollouts")
    hf.close()


@hydra.main(config_path="configs", config_name="data_collection")
def main(config):
    if not config.tasks:
        envs = ENVS_AND_SCRIPTED_POLICIES
    else:
        envs = [env for env in ENVS_AND_SCRIPTED_POLICIES if env[0] in config.tasks]
    cfg = OmegaConf.to_container(config)
    collect_trajectories(**cfg, env_list=envs)


if __name__ == "__main__":
    main()
