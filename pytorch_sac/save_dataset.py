import os
import utils
import torch
import glob
from eval import make_env
import hydra
import metaworld
import random
from tqdm import tqdm
import h5py
import numpy as np
from mujoco_py import MjRenderContextOffscreen, MjSim, load_model_from_xml
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.path.join(cfg.experiment_dir, cfg.experiment, cfg.env)
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

    def init_env_and_agent(self, env_name):
        self.cfg.env = env_name
        env = make_env(self.cfg)

        self.cfg.agent.obs_dim = env.observation_space.shape[0]
        self.cfg.agent.action_dim = env.action_space.shape[0]
        self.cfg.agent.action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max()),
        ]

        agent = hydra.utils.instantiate(self.cfg.agent, _recursive_=False)

        # load model checkpoint
        ckpt_dir = os.path.join(
            self.cfg.experiment_dir, self.cfg.experiment, env_name, "models"
        )
        files = glob.glob(f"{ckpt_dir}/*")
        files.sort(key=len)
        ckpt_file = files[-1]

        print(f"loading model from {ckpt_file}")
        state_dict = torch.load(ckpt_file)
        agent.load_from_checkpoint(state_dict)
        return env, agent

    def create_dataset(self):
        hf = h5py.File(os.path.join(self.cfg.data_dir, "trajectories.hdf5"), "w")

        envs = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())
        for env_name in tqdm(envs):
            env, agent = self.init_env_and_agent(env_name)
            average_episode_reward = 0
            success_rate = 0.0
            for episode in tqdm(range(self.cfg.demos_per_env)):
                all_obs, dones, rewards, all_next_obs = [], [], [], []

                obs = env.reset()
                agent.reset()

                done = False
                episode_reward = 0
                ep_length = 0
                while not done and ep_length < env.max_path_length:
                    with utils.eval_mode(agent):
                        action = agent.act(obs, sample=False)

                    all_obs.append(obs)
                    obs, reward, done, info = env.step(action)
                    # print(done)
                    done = info["success"]
                    # print(info)
                    if done:
                        success_rate += 1.0

                    all_next_obs.append(obs)
                    rewards.append(reward)
                    dones.append(done)

                    episode_reward += reward
                    ep_length += 1

                all_obs = np.array(all_obs)
                dones = np.array(dones)
                rewards = np.array(rewards)
                all_next_obs = np.array(all_next_obs)

                average_episode_reward += episode_reward

                g = hf.create_group(f"{env}/demo_{episode}")
                g.create_dataset("obs", data=all_obs)
                g.create_dataset("reward", data=rewards)
                g.create_dataset("done", data=dones)
                g.create_dataset("next_obs", data=all_next_obs)

            average_episode_reward /= self.cfg.demos_per_env
            success_rate /= self.cfg.demos_per_env

            print("=" * 50)
            print(f"Success rate: {success_rate}")
            print(f"Average episode reward: {average_episode_reward}")
            print("=" * 50)

        hf.close()


@hydra.main(config_path="config", config_name="eval")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.create_dataset()


if __name__ == "__main__":
    main()
