"""
Generates the behavior dataset via multiprocessing env rollouts
"""

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
import torch.multiprocessing as mp


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


class Workspace(object):
    def __init__(self, cfg):
        # self.work_dir = os.path.join(cfg.experiment_dir, cfg.experiment, cfg.env)
        # print(f"workspace: {self.work_dir}")

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

    def create_dataset(self, envs, results_queue):
        for env_name in tqdm(envs):
            env, agent = self.init_env_and_agent(env_name)
            average_episode_reward = 0
            success_rate = 0.0
            for episode in tqdm(range(self.cfg.demos_per_env)):
                all_obs, actions, dones, rewards, all_next_obs = [], [], [], [], []

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

                    actions.append(action)
                    all_next_obs.append(obs)
                    rewards.append(reward)
                    dones.append(done)

                    episode_reward += reward
                    ep_length += 1

                all_obs = np.array(all_obs)
                actions = np.array(actions)
                dones = np.array(dones)
                rewards = np.array(rewards)
                all_next_obs = np.array(all_next_obs)

                average_episode_reward += episode_reward

                if results_queue is not None:
                    results_queue.put(
                        (
                            env_name,
                            episode,
                            all_obs,
                            actions,
                            rewards,
                            dones,
                            all_next_obs,
                        )
                    )

            average_episode_reward /= self.cfg.demos_per_env
            success_rate /= self.cfg.demos_per_env

            print("=" * 50)
            print(f"Success rate: {success_rate}")
            print(f"Average episode reward: {average_episode_reward}")
            print("=" * 50)


def handle_output(cfg, results_queue):
    hf = h5py.File(os.path.join(cfg.data_dir, cfg.dataset_file), "w")
    while True:
        out = results_queue.get()
        if out is not None:
            env_name, episode, all_obs, actions, rewards, dones, all_next_obs = out
            g = hf.create_group(f"{env_name}/demo_{episode}")
            g.create_dataset("obs", data=all_obs)
            g.create_dataset("action", data=actions)
            g.create_dataset("reward", data=rewards)
            g.create_dataset("done", data=dones)
            g.create_dataset("next_obs", data=all_next_obs)
        else:
            break
    hf.close()


@hydra.main(config_path="config", config_name="eval")
def main(cfg):
    torch.multiprocessing.set_start_method("spawn")
    workspace = Workspace(cfg)

    if cfg.mp:

        results_queue = mp.Queue()

        proc = mp.Process(
            target=handle_output,
            args=(
                cfg,
                results_queue,
            ),
        )

        processes = []
        proc.start()

        envs = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())[5:10]
        # split up the envs
        env_chunks = list(split(envs, cfg.num_processes))
        print(len(env_chunks))

        for rank in range(cfg.num_processes):
            p = mp.Process(
                target=workspace.create_dataset, args=(env_chunks[rank], results_queue)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        results_queue.put(None)
        proc.join()

    else:
        envs = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())[:5]
        workspace.create_dataset(envs, results_queue=None)

    print("done...")


if __name__ == "__main__":
    main()
