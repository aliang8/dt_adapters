from torch.utils.data import Dataset, Sampler
import os
import h5py
import torch
import random
import numpy as np
from general_utils import discount_cumsum, AttrDict
from mw_constants import OBJECTS_TO_ENV
from mw_utils import get_object_indices
from torchvision.transforms import transforms as T
from transformers import CLIPProcessor, CLIPVisionModel

import general_utils


class MWDemoDataset(Dataset):
    def __init__(self, config, stage="pretraining"):
        self.config = config
        self.state_dim = config.state_dim
        self.act_dim = config.act_dim
        self.context_len = config.context_len
        self.max_ep_len = config.max_ep_len

        self.trajectories = []
        self.train_tasks = config.train_tasks
        self.finetune_tasks = config.finetune_tasks

        # self.img_transforms = T.Compose(
        #     [
        #         T.Lambda(
        #             lambda images: torch.stack(
        #                 [T.ToTensor()(image) for image in images]
        #             )
        #         ),
        #         T.Resize([config.image_size]),
        #         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #         T.Lambda(lambda images: images.numpy()),
        #     ]
        # )
        if self.config.vision_backbone == "clip":
            self.img_encoder = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.img_encoder.cuda().eval()
            self.img_preprocessor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

        all_states = []

        # load trajectories into memory
        data_file = os.path.join(os.environ["DATA_DIR"], config.data_file)
        with h5py.File(data_file, "r") as f:
            envs = list(f.keys())

            for env in envs:
                if stage == "pretraining" and env not in self.train_tasks:
                    continue

                elif stage == "finetuning" and env not in self.finetune_tasks:
                    continue

                if stage not in ["pretraining", "finetuning"]:
                    raise Exception(f"{stage} not available")

                num_demos = len(f[env].keys())

                for k, demo in f[env].items():
                    states = demo["obs"][()]

                    if self.config.hide_goal:
                        states[:, -3:] = 0

                    all_states.append(states)

                    traj = {
                        "states": states,
                        "obj_ids": get_object_indices(env),
                        "actions": demo["action"][()],
                        "rewards": demo["reward"][()],
                        "dones": demo["done"][()],
                        "returns_to_go": discount_cumsum(demo["reward"][()], gamma=1.0),
                        "timesteps": np.arange(len(states)),
                        "attention_mask": np.ones(len(states)),
                        "online": 0,
                    }

                    if "image" in self.config.state_keys:
                        # apply transform to images first
                        # need to reshape into LxCxHxW
                        images = demo["images"][()]
                        images = images.transpose(0, 3, 1, 2)

                        # get pretrained image features
                        if self.config.vision_backbone == "clip":
                            # for some reason clip preprocessor needs a list of images
                            list_images = [images[i] for i in range(images.shape[0])]
                            images = torch.stack(
                                list(
                                    self.img_preprocessor(
                                        images=list_images, return_tensors="pt"
                                    ).pixel_values
                                )
                            ).to("cuda")
                            with torch.no_grad():
                                outputs = self.img_encoder(pixel_values=images)
                            # last_hidden_state = outputs.last_hidden_state
                            pooled_output = outputs.pooler_output

                            # store features for image from each step
                            traj["img_feats"] = pooled_output.detach().cpu().numpy()

                        elif self.config.vision_backbone == "resnet":
                            traj["img_feats"] = self.img_preprocessor(images)

                    self.trajectories.append(traj)

        # not sure if this is proper
        all_states = np.concatenate(all_states, axis=0)

        self.state_mean, self.state_std = (
            np.mean(all_states, axis=0),
            np.std(all_states, axis=0) + 1e-6,
        )

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        si = random.randint(0, traj["rewards"].shape[0] - 1)

        # get sequences from dataset
        state = traj["states"][si : si + self.context_len].reshape(-1, self.state_dim)
        action = traj["actions"][si : si + self.context_len].reshape(-1, self.act_dim)
        reward = traj["rewards"][si : si + self.context_len].reshape(-1, 1)
        done = traj["dones"][si : si + self.context_len].reshape(-1)
        timestep = np.arange(si, si + state.shape[0]).reshape(1, -1)
        timestep[timestep >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
        rtg = discount_cumsum(traj["rewards"][si:], gamma=1.0)[
            : state.shape[0] + 1
        ].reshape(-1, 1)

        if rtg.shape[0] <= state.shape[0]:
            rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        # padding and state + reward normalization
        tlen = state.shape[0]
        state = np.concatenate(
            [np.zeros((self.context_len - tlen, self.state_dim)), state], axis=0
        )
        state = (state - self.state_mean) / self.state_std
        action = np.concatenate(
            [np.ones((self.context_len - tlen, self.act_dim)) * -10.0, action], axis=0
        )
        reward = np.concatenate(
            [np.zeros((self.context_len - tlen, 1)), reward], axis=0
        )
        done = np.concatenate([np.ones((self.context_len - tlen)) * 2, done], axis=0)
        rtg = (
            np.concatenate([np.zeros((self.context_len - tlen, 1)), rtg], axis=0)
            / self.config.scale
        )
        timestep = np.concatenate(
            [np.zeros((1, self.context_len - tlen)), timestep], axis=1
        )
        mask = np.concatenate(
            [np.zeros((self.context_len - tlen)), np.ones((tlen))], axis=0
        )

        out = {
            "states": state,
            "actions": action,
            "returns_to_go": rtg,
            "timesteps": timestep,
            "dones": done,
            "rewards": reward,
            "attention_mask": mask,
            "obj_ids": np.array(traj["obj_ids"]),
            "online": traj["online"],
        }

        if "image" in self.config.state_keys:
            img_feats_shape = traj["img_feats"].shape[-1]
            img_feats = traj["img_feats"][si : si + self.context_len].reshape(
                -1, img_feats_shape
            )
            # img_feats = torch.cat(
            #     [
            #         torch.zeros((self.context_len - tlen, img_feats_shape)).to(
            #             img_feats.device
            #         ),
            #         img_feats,
            #     ],
            #     dim=0,
            # )
            img_feats = np.concatenate(
                [
                    np.zeros((self.context_len - tlen, img_feats_shape)),
                    img_feats,
                ],
                axis=0,
            )

            out["img_feats"] = img_feats

        return out


if __name__ == "__main__":
    config = AttrDict(
        data_file="trajectories_all_with_images_10.hdf5",
        state_dim=39,
        act_dim=4,
        context_len=50,
        max_ep_len=500,
        train_tasks=["pick-place-v2"],
        finetune_tasks=[],
        state_keys=["image"],
        hide_goal=False,
        scale=100,
        image_size=64,
        vision_backbone="clip",
    )

    dataset = MWDemoDataset(config)
    dataset[0]
