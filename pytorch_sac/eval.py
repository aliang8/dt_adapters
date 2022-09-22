#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import glob
import pickle as pkl

from video import VideoRecorder
from replay_buffer import ReplayBuffer
import utils

# import dmc2gym
import hydra
import metaworld
import random
from tqdm import tqdm

# def make_env(cfg):
#     """Helper function to create dm_control environment"""
#     if cfg.env == 'ball_in_cup_catch':
#         domain_name = 'ball_in_cup'
#         task_name = 'catch'
#     else:
#         domain_name = cfg.env.split('_')[0]
#         task_name = '_'.join(cfg.env.split('_')[1:])

#     env = dmc2gym.make(domain_name=domain_name,
#                        task_name=task_name,
#                        seed=cfg.seed,
#                        visualize_reward=True)
#     env.seed(cfg.seed)
#     assert env.action_space.low.min() >= -1
#     assert env.action_space.high.max() <= 1

#     return env

from mujoco_py import MjRenderContextOffscreen, MjSim, load_model_from_xml
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


def make_env(cfg):
    SEED = cfg.seed  # some seed number here
    # benchmark = metaworld.Benchmark(seed=SEED)
    # ml1 = metaworld.ML1(cfg.env)
    # env = ml1.train_classes[cfg.env]()

    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[cfg.env]()
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    env.seed(cfg.seed)
    # task = ml1.train_tasks[0]
    # env.set_task(task)

    if env.sim._render_context_offscreen is None:
        render_context = MjRenderContextOffscreen(env.sim, device_id=0)
        env.sim.add_render_context(render_context)
    return env


class Workspace(object):
    def __init__(self, cfg):
        # self.work_dir = os.getcwd()
        self.work_dir = os.path.join(cfg.experiment_dir, cfg.experiment, cfg.env)
        os.makedirs(self.work_dir, exist_ok=True)
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]

        self.agent = hydra.utils.instantiate(cfg.agent, _recursive_=False)

        if cfg.restore_from_checkpoint:
            ckpt_dir = os.path.join(
                cfg.experiment_dir, cfg.experiment, cfg.env, "models"
            )
            files = glob.glob(f"{ckpt_dir}/*")
            files.sort(key=len)
            ckpt_file = files[-1]

            print(f"loading model from {ckpt_file}")
            state_dict = torch.load(ckpt_file)
            self.agent.load_from_checkpoint(state_dict)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def gather_rollouts(self):
        average_episode_reward = 0
        success_rate = 0.0
        for episode in tqdm(range(self.cfg.num_rollouts)):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init()
            done = False
            episode_reward = 0
            ep_length = 0
            while not done and ep_length < self.env.max_path_length:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                # print(done)
                done = info["success"]
                # print(info)
                if done:
                    success_rate += 1.0

                self.video_recorder.record(self.env)
                episode_reward += reward
                ep_length += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f"rollout_{episode}.mp4", self.step, None)
        average_episode_reward /= self.cfg.num_rollouts
        success_rate /= self.cfg.num_rollouts

        print("=" * 50)
        print(f"Success rate: {success_rate}")
        print(f"Average episode reward: {average_episode_reward}")
        print("=" * 50)


@hydra.main(config_path="config", config_name="eval")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.gather_rollouts()


if __name__ == "__main__":
    main()
