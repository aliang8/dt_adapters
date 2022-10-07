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
from general_utils import discount_cumsum
from collections import defaultdict as dd

from data.base_dataset import BaseDataset
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from general_utils import AttrDict
from data.utils import get_image_feats

from transformers import CLIPProcessor, CLIPVisionModel

import torch
from torchvision.transforms import transforms as T
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


def preprocess_obs(config, obs):
    action = np.concatenate([obs.joint_velocities, np.array([obs.gripper_open])])

    ll_state_info = [
        np.array(getattr(obs, k)).reshape(-1) for k in config.ll_state_keys
    ]
    ll_state = np.concatenate(ll_state_info)

    image_state = {k: getattr(obs, k) for k in config.image_keys}
    return action, ll_state, image_state


def extract_image_feats(
    img_obs, img_preprocessor, img_encoder, depth_img_preprocessor, depth_img_encoder
):
    all_img_feats = []
    for k, imgs in img_obs.items():
        if "rgb" in k:
            img_feat = get_image_feats(
                np.array(imgs),
                img_preprocessor,
                img_encoder,
                "clip",
            )
        if "depth" in k:
            img_feat = get_image_feats(
                np.array(imgs),
                depth_img_preprocessor,
                depth_img_encoder,
                "resnet",
            )
        all_img_feats.append(img_feat)
    all_img_feats = np.concatenate(all_img_feats, axis=-1)
    return all_img_feats


def get_visual_encoders(image_size, device):
    img_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    img_encoder.to(device).eval()
    img_preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # for processing depth images
    # self.depth_img_preprocessor = weights.transforms()
    img_transforms = T.Compose(
        [
            T.Lambda(
                lambda images: torch.stack([T.ToTensor()(image) for image in images])
            ),
            T.Resize([image_size]),
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.Lambda(lambda images: images.numpy()),
        ]
    )
    depth_img_preprocessor = img_transforms
    weights = ResNet50_Weights.DEFAULT
    depth_img_encoder = resnet50(weights=weights)
    depth_img_encoder.to(device).eval()
    return img_preprocessor, img_encoder, depth_img_preprocessor, depth_img_encoder


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
