from dt_adapters.envs.make_env import env_constructor
from dt_adapters.mw_utils import ENVS_AND_SCRIPTED_POLICIES
import numpy as np
import h5py
import os
import hydra
from dt_adapters.models.state_embedding_net import StateEmbeddingNet
from dt_adapters.envs.obs_wrappers import MuJoCoPixelObs, StateEmbedding
from metaworld.policies import *


def rollout(
    env,
    agent,
    config,
    *,
    max_episode_length=np.inf,
):
    """Sample a single episode of the agent in the environment."""
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


def collect_dataset(config, envs):
    hf = h5py.File(os.path.join(config.data_dir, config.data_file), "w")
    state_embedding = StateEmbeddingNet(config.state_encoder)

    for (env_name, policy, _, _) in envs:
        print(env_name)

        env = env_constructor(
            env_name=env_name,
            config=config,
            device="cuda",
        )

        total_successes = 0
        total_failures = 0

        while total_successes < config.demos_per_env:
            path = rollout(env, policy, config, max_episode_length=500)

            if path["traj_success"]:
                g = hf.create_group(f"{env_name}/demo_{total_successes}")
                g.create_dataset("states", data=path["states"])
                g.create_dataset("actions", data=path["actions"])
                g.create_dataset("rewards", data=path["rewards"])
                g.create_dataset("dones", data=path["dones"])

                img = hf.create_group(f"{env_name}/demo_{total_successes}/img_feats")
                for k in path["frames"].keys():
                    img.create_dataset(k, data=state_embedding(path["frames"][k]))
                total_successes += 1
                total_failures = 0
            else:
                total_failures += 1

            # print(f"successes: {total_successes}, failures: {total_failures}")

            if total_failures > 20:
                print(f"could not solve: {env_name}")
                break

    hf.close()


@hydra.main(config_path="configs", config_name="data_collection")
def main(config):
    # envs = ENVS_AND_SCRIPTED_POLICIES
    envs = [env for env in ENVS_AND_SCRIPTED_POLICIES if env[0] in config.tasks]
    collect_dataset(config, envs)


if __name__ == "__main__":
    main()
