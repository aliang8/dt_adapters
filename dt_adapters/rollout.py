import os
import gc
import csv
import glob
import wandb
import torch
import hydra
import numpy as np
import time
import collections
from pprint import pprint
import multiprocessing as mp
from collections import defaultdict as dd

import gym

from torchvision.transforms import transforms as T
from transformers import CLIPProcessor, CLIPVisionModel

try:
    import dt_adapters.envs.rlbench_env
except:
    pass

# import dt_adapters.mw_utils as mw_utils
from dt_adapters.envs.make_env import env_constructor
import dt_adapters.general_utils as general_utils
from dt_adapters.general_utils import AttrDict as AttrDict
import dt_adapters.data.utils as data_utils
from dt_adapters.envs.video_recorder import VideoRecorder
from dt_adapters.models.model import TrajectoryModel
from dt_adapters.envs.gym_env import GymEnv
from typing import Optional, Tuple, Dict, Union, List


class Rollout:
    def __init__(
        self,
        domain: str,
        env_name: str,
        num_processes: int = 0,
        num_eval_rollouts: int = 10,
        image_keys: List[str] = [],
        vision_backbone: str = None,
        image_width: int = 128,
        image_height: int = 128,
        proprio: int = 0,
        max_episode_length: int = 500,
        log_eval_videos: bool = False,
        device: str = "cpu",
    ):
        self.env = env_constructor(
            domain=domain,
            env_name=env_name,
            image_keys=image_keys,
            vision_backbone=vision_backbone,
            image_width=image_width,
            image_height=image_height,
            proprio=proprio,
            device=device,
        )

        if log_eval_videos:
            self.recorder = VideoRecorder(self.env, path="test.mp4")

        self.domain = domain
        self.num_processes = num_processes
        self.max_episode_length = max_episode_length
        self.num_eval_rollouts = num_eval_rollouts
        self.device = device

    def mp_rollouts(
        self,
        model: TrajectoryModel,
        goal_states: Optional[torch.Tensor] = None,
        goal_img_feats: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> List[AttrDict]:

        rollout_kwargs = general_utils.AttrDict(
            model=model,
            **kwargs,
        )

        eval_rollouts = []
        if self.num_processes > 0:
            # run multiple threads at the same time
            p = mp.Pool(processes=self.num_processes)

            # fetch the results of the rollout
            results = [
                p.apply_async(self.run_single_rollout, (), rollout_kwargs)
                for i in range(self.num_eval_rollouts)
            ]
            eval_rollouts = [p.get() for p in results]
            p.close()
            p.join()
        else:
            for i in range(self.num_eval_rollouts):
                if goal_states:
                    eval_rollouts.append(
                        self.run_single_rollout(
                            goal_states=goal_states[i : i + 1],
                            goal_img_feats={
                                k: goal_img_feats[k][i : i + 1] for k in goal_img_feats
                            },
                            **rollout_kwargs,
                        )
                    )
                else:
                    eval_rollouts.append(self.run_single_rollout(**rollout_kwargs))

        return eval_rollouts

    def run_single_rollout(
        self,
        model: TrajectoryModel,
        goal_states: Optional[torch.Tensor] = None,
        goal_img_feats: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AttrDict:
        """
        Run a single episode of rollout.
        For causal decoders, need to keep track of the context. The model `get_action`
        function is responsible for slicing the context.

        For visual observations, the environment wrapper will handle featurizing the image
        and concatenating it with the state information.
        """
        start = time.time()
        episode_length = 0
        traj_success = False
        self.recorder.reset()

        with torch.no_grad():
            # ============= PROCESS FIRST OBS =============
            last_obs = self.env.reset()

            if self.recorder:
                self.recorder.capture_frame()

            # create initial conditioning information
            # these tensors will store the context history for inputting to the model
            states = (
                torch.from_numpy(last_obs)
                .reshape(1, self.env.observation_dim)
                .to(device=self.device, dtype=torch.float32)
            )

            actions = torch.zeros(
                (0, self.env.action_dim), device=self.device, dtype=torch.float32
            )
            rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
            timesteps = torch.tensor([0], device=self.device, dtype=torch.long)

            while episode_length < (self.max_episode_length or np.inf):
                # add placeholder for next action
                actions = torch.cat(
                    [
                        actions,
                        torch.zeros((1, self.env.action_dim), device=self.device),
                    ],
                    dim=0,
                )

                rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])

                action = model.get_action(
                    states=states,
                    actions=actions,
                    timesteps=timesteps,
                    goal_states=goal_states,
                    goal_img_feats=goal_img_feats,
                )

                actions[-1] = action
                action = action.detach().cpu().numpy()

                obs, reward, terminate, info = self.env.step(action)

                if self.recorder:
                    self.recorder.capture_frame()

                if self.domain == "metaworld":
                    terminate |= bool(info["success"])

                if terminate:
                    traj_success = True

                episode_length += 1

                timesteps = torch.cat(
                    [
                        timesteps,
                        torch.ones((1), device=self.device, dtype=torch.long)
                        * episode_length,
                    ],
                    dim=0,
                )

                last_obs = obs

                cur_state = (
                    torch.from_numpy(last_obs)
                    .to(device=self.device)
                    .reshape(1, self.env.observation_dim)
                )
                states = torch.cat([states, cur_state], dim=0)

                rewards[-1] = reward
                if terminate:
                    break

            rollout_time = time.time() - start

        out = AttrDict(
            states=general_utils.to_numpy(states),
            actions=general_utils.to_numpy(actions),
            rewards=general_utils.to_numpy(rewards),
            rollout_time=rollout_time,
            frames=np.array(self.recorder.recorded_frames) if self.recorder else None,
            traj_success=traj_success,
        )

        # clean up
        gc.collect()
        return out
