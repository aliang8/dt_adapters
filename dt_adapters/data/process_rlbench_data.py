""" 
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 data/process_rlbench_data.py \
    --config-name=train \
    data=[base,rlbench] \
    model=[base,decision_transformer] \
    data.data_dir=/data/anthony/dt_adapters/data/rlbench_data/mt15_v1 \
    data.data_file=rlbench_demo_feats.hdf5 \
"""

import os
import h5py
import hydra
import random
from omegaconf import OmegaConf
import numpy as np
from collections import defaultdict as dd

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

from dt_adapters.data.base_dataset import BaseDataset
from dt_adapters.general_utils import AttrDict, discount_cumsum

from dt_adapters.data.utils import (
    get_image_feats,
    preprocess_obs,
    get_visual_encoders,
    extract_image_feats,
)


def process_feats(config):
    hf = h5py.File(os.path.join(config.data_dir, config.data_file), "w")

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    # for processing rgb images
    (
        img_preprocessor,
        img_encoder,
        depth_img_preprocessor,
        depth_img_encoder,
    ) = get_visual_encoders(config.image_size, "cuda")

    env = Environment(
        dataset_root=config.data_dir,
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=obs_config,
        headless=False,
    )
    env.launch()

    train_tasks = MT15_V1["train"]

    for task in train_tasks:
        task = env.get_task(task)
        task_name = task.get_name()
        print(task_name)
        task_demos = task.get_demos(-1, live_demos=False)

        for demo_idx, demo in enumerate(task_demos):
            all_obs = demo._observations

            all_actions, all_states = [], []
            img_obs = dd(list)

            for obs in all_obs:
                action, ll_state, image_info = preprocess_obs(config, obs)

                all_actions.append(action)
                all_states.append(ll_state)
                for k, v in image_info.items():
                    img_obs[k].append(v)

            states = np.stack(all_states)
            actions = np.stack(all_actions)

            img_feats = extract_image_feats(
                img_obs,
                img_preprocessor,
                img_encoder,
                depth_img_preprocessor,
                depth_img_encoder,
            )

            # write to file
            g = hf.create_group(f"{task_name}/demo_{demo_idx}")
            g.create_dataset("states", data=states)
            g.create_dataset("actions", data=actions)
            g.create_dataset("img_feats", data=img_feats)
    hf.close()


@hydra.main(config_path="../configs", config_name="train")
def main(config):
    process_feats(config.data)


if __name__ == "__main__":
    main()
