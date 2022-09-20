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
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

# import dmc2gym
import hydra
import metaworld
import random

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
        self.work_dir = os.path.join(cfg.experiment_dir, cfg.experiment)
        os.makedirs(self.work_dir, exist_ok=True)
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        self.logger = Logger(
            cfg,
            self.work_dir,
            save_tb=cfg.log_save_tb,
            save_wb=cfg.log_to_wandb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name,
        )

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

        if cfg.checkpoint_path:
            print(f"loading model from {cfg.checkpoint_path}")
            state_dict = torch.load(cfg.checkpoint_path)
            self.agent.load_from_checkpoint(state_dict)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            ep_length = 0
            while not done and ep_length < self.env.max_path_length:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
                ep_length += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f"{self.step}.mp4", self.step, self.logger)
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log("eval/episode_reward", average_episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        print("start training...")
        # self.evaluate()

        while self.step < self.cfg.num_train_steps:
            if done or episode_step == self.env.max_path_length:
                if self.step > 0:
                    self.logger.log(
                        "train/duration", time.time() - start_time, self.step
                    )
                    start_time = time.time()

                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps)
                    )

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log("eval/episode", episode, self.step)
                    self.evaluate()

                self.logger.log("train/episode_reward", episode_reward, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log("train/episode", episode, self.step)

            if self.step != 0 and self.step % self.cfg.save_every_steps == 0:
                ckpt_dir = os.path.join(self.work_dir, "models")
                os.makedirs(ckpt_dir, exist_ok=True)
                path = os.path.join(ckpt_dir, f"step_{self.step}.pt")
                print(f"saving model to {path}")
                torch.save(self.agent.state_dict(), path)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            # print(episode_step)
            # done_no_max = 0 if episode_step + 1 == self.cfg.max_episode_steps else done
            done_no_max = 0 if episode_step + 1 == self.env.max_path_length else done

            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path="config", config_name="train")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
