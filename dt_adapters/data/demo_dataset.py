from torch.utils.data import Dataset, Sampler
import os
import h5py
import torch
import random
import numpy as np
from dt_adapters.mw_constants import OBJECTS_TO_ENV
from dt_adapters.mw_utils import get_object_indices
from torchvision.transforms import transforms as T
from transformers import CLIPProcessor, CLIPVisionModel

import dt_adapters.general_utils as general_utils
from dt_adapters.data.utils import get_image_feats
from dt_adapters.data.base_dataset import BaseDataset


import tqdm
import wandb
import dt_adapters.mw_utils as mw_utils
import gym

try:
    import dt_adapters.envs.rlbench_env
except:
    pass

from collections import defaultdict as dd
import multiprocessing as mp


# def rollout(config, trajectory):
#     task = trajectory["task"]
#     env = gym.make(f"{task}-vision-v0", config=config, render_mode="rgb_array")
#     obs = env.reset()

#     video = []
#     # print(trajectory["actions"].shape)
#     for action in trajectory["actions"]:
#         #     # print(action.shape)
#         #     # print(action)
#         obs, reward, terminate, info = env.step(action)
#         frame = env.render(mode="rgb_array")
#         #     # print(frame.shape)
#         #     # print(obs.keys())
#         video.append(frame)

#     env.close()
#     return video


class DemoDataset(BaseDataset):
    def __init__(self, config, stage="pretraining"):
        super().__init__(config, stage)

        all_states = []

        # load trajectories into memory
        data_file = os.path.join(config.data_dir, config.data_file)

        with h5py.File(data_file, "r") as f:
            tasks = list(f.keys())

            for task in tasks:
                if stage == "pretraining" and task not in self.config.train_tasks:
                    continue

                elif stage in "finetuning" and task not in self.config.eval_task:
                    continue

                if stage not in ["pretraining", "finetuning", "eval"]:
                    raise Exception(f"{stage} not available")

                num_demos = len(f[task].keys())

                for k, demo in f[task].items():
                    states = demo["states"][()]

                    if self.config.env_name == "metaworld" and self.config.hide_goal:
                        states[:, -3:] = 0

                    all_states.append(states)

                    traj = {
                        "states": states,
                        "actions": demo["actions"][()],
                        "timesteps": np.arange(len(states)),
                        "attention_mask": np.ones(len(states)),
                        "online": 0,
                        "task": task,
                    }

                    if self.config.env_name == "metaworld":
                        traj.update(
                            {
                                "obj_ids": get_object_indices(task),
                                "rewards": demo["rewards"][()],
                                "dones": demo["dones"][()],
                                "returns_to_go": general_utils.discount_cumsum(
                                    demo["rewards"][()], gamma=1.0
                                ),
                            }
                        )

                    if "image" in self.config.observation_mode:
                        traj["img_feats"] = demo["img_feats"][()]

                    self.trajectories.append(traj)

        # not sure if this is proper
        all_states = np.concatenate(all_states, axis=0)

        self.state_mean, self.state_std = (
            np.mean(all_states, axis=0),
            np.std(all_states, axis=0) + 1e-6,
        )


# if __name__ == "__main__":
#     mw_config = general_utils.AttrDict(
#         data_file="trajectories_all_with_images_10.hdf5",
#         state_dim=39,
#         act_dim=4,
#         context_len=50,
#         max_ep_len=500,
#         train_tasks=["pick-place-v2"],
#         finetune_tasks=[],
#         observation_mode="image",
#         hide_goal=False,
#         scale=100,
#         image_size=64,
#         vision_backbone="clip",
#         env_name="metaworld",
#     )

#     rlbench_config = general_utils.AttrDict(
#         data_dir="/data/anthony/dt_adapters/data/rlbench_data/mt15_v1",
#         data_file="rlbench_demo_feats.hdf5",
#         observation_mode="image",
#         state_dim=30,
#         act_dim=8,
#         context_len=50,
#         max_ep_len=500,
#         train_tasks=["close_fridge", "close_box", "slide_cabinet_open"],
#         finetune_tasks=[],
#         hide_goal=False,
#         scale=100,
#         image_size=64,
#         vision_backbone="clip",
#         ll_observation_mode=[
#             "joint_positions",
#             "joint_forces",
#             "gripper_open",
#             "gripper_pose",
#             "gripper_joint_positions",
#             "gripper_touch_forces",
#         ],
#         image_keys=["overhead_rgb", "overhead_depth"],
#         env_name="rlbench",
#         use_stored_states=True,
#     )

#     dataset = DemoDataset(rlbench_config)
#     datum = dataset[0]
#     print(datum.keys())
#     print(datum["img_feats"].shape)

#     videos = dd(list)
#     if rlbench_config.use_stored_states:
#         # just visualizing from stored images
#         from rlbench import ObservationConfig
#         from rlbench.action_modes.action_mode import MoveArmThenGripper
#         from rlbench.action_modes.arm_action_modes import JointVelocity
#         from rlbench.action_modes.gripper_action_modes import Discrete
#         from rlbench.backend.utils import task_file_to_task_class
#         from rlbench.environment import Environment
#         from rlbench.tasks import *

#         tasks = [
#             "CloseFridge",
#             "CloseBox",
#             "PickAndLift",
#             "TakeUmbrellaOutOfUmbrellaStand",
#         ]

#         obs_config = ObservationConfig()
#         obs_config.set_all(True)

#         env = Environment(
#             dataset_root=rlbench_config.data_dir,
#             action_mode=MoveArmThenGripper(
#                 arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
#             ),
#             obs_config=ObservationConfig(),
#             headless=False,
#         )
#         env.launch()

#         for task_name in tasks:
#             task = env.get_task(globals()[task_name])
#             demos = task.get_demos(5, live_demos=False)

#             video_list = []
#             for demo in demos:
#                 video = []
#                 for obs in demo._observations:
#                     frame = obs.overhead_rgb
#                     video.append(frame)
#                 video_list.append(np.array(video))
#             videos[task.get_name()] = video_list

#     else:
#         # executing the actions in the environment
#         torch.multiprocessing.set_start_method("spawn")
#         dataset = DemoDataset(rlbench_config)
#         trajectories = dataset.trajectories

#         filtered_trajs = []
#         counts = dd(int)
#         for traj in trajectories:
#             if counts[traj["task"]] >= 5:
#                 continue
#             filtered_trajs.append(traj)
#             counts[traj["task"]] += 1

#         trajectories = filtered_trajs

#         p = mp.Pool(processes=5)
#         print(f"{len(trajectories)} number of trajectories")

#         results = [
#             p.apply_async(rollout, args=(rlbench_config, trajectory))
#             for trajectory in trajectories
#         ]
#         eval_rollouts = [p.get() for p in results]
#         p.close()
#         p.join()
#         print("done, starting to make videos")

#         for i, video in enumerate(eval_rollouts):
#             task = trajectories[i]["task"]
#             videos[task].append(np.array(video))

#     # log the videos for each task
#     wandb.init(
#         name="rlbench_demos",
#         group="demo_videos",
#         project="dt-adapters",
#         entity="glamor",
#     )
#     for task, video_list in videos.items():
#         video_array = mw_utils.create_video_grid(video_list)
#         print(video_array.shape)

#         wandb.log(
#             {
#                 f"eval/{task}/demo_videos": wandb.Video(
#                     video_array,
#                     caption=f"",
#                     fps=10,
#                     format="gif",
#                 )
#             }
#         )
