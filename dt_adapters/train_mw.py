import argparse
import json
import time
import os
import sys
import importlib
import wandb
import random
import torch
import glob
import h5py
import hydra
import numpy as np
from pprint import pprint
from tqdm import tqdm
from collections import OrderedDict

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler, RandomSampler


from data.demo_dataset import DemoDataset
from omegaconf import OmegaConf
from models.decision_transformer import DecisionTransformerSeparateState
from models.mlp_policy import MLPPolicy
from sampler import ImportanceWeightBatchSampler

from transformers.adapters.configuration import AdapterConfig

import general_utils
import mw_utils
import eval_utils
from garage.envs import GymEnv
from garage.np import discount_cumsum, stack_tensor_dict_list

from data.process_rlbench_data import (
    preprocess_obs,
    extract_image_feats,
    get_visual_encoders,
)

# RLbench imports
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from transformers import CLIPProcessor, CLIPVisionModel
from trainer import Trainer


class MWTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def rollout(self, use_means=False, attend_to_rtg=False, log_eval_videos=False):
        """Sample a single episode of the agent in the environment."""
        env_steps = []
        agent_infos = []
        observations = []
        last_obs, episode_infos = self.env.reset()
        self.model.reset()

        state_dim = self.model.state_dim
        act_dim = self.model.act_dim

        if log_eval_videos:
            self.env._visualize = True
        else:
            self.env._visualize = False

        state_mean = torch.from_numpy(self.dataset.state_mean).to(device=self.device)
        state_std = torch.from_numpy(self.dataset.state_std).to(device=self.device)

        states = (
            torch.from_numpy(last_obs)
            .reshape(1, state_dim)
            .to(device=self.device, dtype=torch.float32)
        )
        actions = torch.zeros((0, act_dim), device=self.device, dtype=torch.float32)
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        use_rtg_mask = torch.tensor([attend_to_rtg]).reshape(1, 1).to(self.device)

        target_return = torch.tensor(
            self.config.target_return / self.config.scale,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1)

        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
        episode_length = 0

        while episode_length < (self.env.max_path_length or np.inf):
            # add padding
            actions = torch.cat(
                [actions, torch.zeros((1, act_dim), device=self.device)], dim=0
            )
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])

            action, return_target, agent_info = self.model.get_action(
                states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                actions=actions.to(dtype=torch.float32),
                returns_to_go=target_return,
                obj_ids=self.obj_ids,
                timesteps=timesteps.to(dtype=torch.long),
                use_means=use_means,
                use_rtg_mask=use_rtg_mask,
                sample_return_dist=self.config.model.predict_return_dist,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            es = self.env.step(action)

            env_steps.append(es)
            observations.append(last_obs)
            agent_infos.append(agent_info)

            episode_length += 1
            if es.last:
                break
            last_obs = es.observation

            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=self.device, dtype=torch.long)
                    * episode_length,
                ],
                dim=1,
            )
            cur_state = (
                torch.from_numpy(last_obs).to(device=self.device).reshape(1, state_dim)
            )
            states = torch.cat([states, cur_state], dim=0)
            if self.config.model.predict_return_dist:
                # use the model's prediction of the return to go
                # follow this paper: https://openreview.net/forum?id=fwJWhOxuzV9
                target_return = torch.cat(
                    [target_return, return_target.reshape(1, 1)], dim=1
                )
            else:
                pred_return = target_return[0, -1] - (es.reward / self.config.scale)
                target_return = torch.cat(
                    [target_return, pred_return.reshape(1, 1)], dim=1
                )

            rewards[-1] = es.reward

        rewards = np.array([es.reward for es in env_steps])
        returns = general_utils.discount_cumsum(rewards, gamma=1.0)
        return dict(
            episode_infos=episode_infos,
            observations=np.array(observations),
            actions=np.array([es.action for es in env_steps]),
            rewards=rewards,
            returns_to_go=returns,
            agent_infos=stack_tensor_dict_list(agent_infos),
            env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
            dones=np.array([es.terminal for es in env_steps]),
        )

    def setup_env(self, env_name=None, task=None):
        # create env for online training
        if self.config.stage == "finetuning" and task is not None:
            print(
                f"initializing metaworld env: {env_name}, task: {task}, obj_random: {self.config.obj_randomization}"
            )
            env = mw_utils.initialize_env(env_name, self.config.obj_randomization)
            max_path_length = env.max_path_length
            self.env = GymEnv(env, max_episode_length=max_path_length)
            self.obj_ids = mw_utils.get_object_indices(self.config.env_name)
            self.obj_ids = (
                torch.tensor(self.obj_ids).long().to(self.device).unsqueeze(0)
            )
            if not self.config.train_on_offline_data:  # clear out dataset buffer
                self.dataset.trajectories = []

        else:
            self.config.num_online_rollouts = 1


@hydra.main(config_path="configs", config_name="train")
def main(config):
    OmegaConf.set_struct(config, False)
    config.update(config.general)
    trainer = MWTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
