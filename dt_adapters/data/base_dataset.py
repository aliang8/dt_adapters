from torch.utils.data import Dataset, Sampler

import random
import numpy as np
from dt_adapters.general_utils import discount_cumsum, AttrDict


class BaseDataset(Dataset):
    def __init__(self, config, stage="pretraining"):
        self.config = config
        self.trajectories = []

        self.state_dim = config.state_dim
        self.act_dim = config.act_dim
        self.context_len = config.context_len
        self.max_ep_len = config.max_ep_len

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        si = random.randint(0, traj["states"].shape[0] - 1)

        # get sequences from dataset
        state = traj["states"][si : si + self.context_len].reshape(-1, self.state_dim)
        action = traj["actions"][si : si + self.context_len].reshape(-1, self.act_dim)

        if "rewards" in traj:
            reward = traj["rewards"][si : si + self.context_len].reshape(-1, 1)
            rtg = discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                : state.shape[0] + 1
            ].reshape(-1, 1)

            if rtg.shape[0] <= state.shape[0]:
                rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        if "dones" in traj:
            done = traj["dones"][si : si + self.context_len].reshape(-1)

        timestep = np.arange(si, si + state.shape[0]).reshape(1, -1)
        timestep[timestep >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff

        # padding and state + reward normalization
        tlen = state.shape[0]
        state = np.concatenate(
            [np.zeros((self.context_len - tlen, self.state_dim)), state], axis=0
        )
        state = (state - self.state_mean) / self.state_std
        action = np.concatenate(
            [np.ones((self.context_len - tlen, self.act_dim)) * -10.0, action], axis=0
        )

        if "reward" in traj:
            reward = np.concatenate(
                [np.zeros((self.context_len - tlen, 1)), reward], axis=0
            )
            rtg = (
                np.concatenate([np.zeros((self.context_len - tlen, 1)), rtg], axis=0)
                / self.config.scale
            )

        if "dones" in traj:
            done = np.concatenate(
                [np.ones((self.context_len - tlen)) * 2, done], axis=0
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
            "timesteps": timestep,
            "attention_mask": mask,
            "online": traj["online"],
        }

        if "reward" in traj:
            out["returns_to_go"] = rtg
            out["rewards"] = reward

        if "dones" in traj:
            out["dones"] = done

        if "obj_ids" in traj:
            out["obj_ids"] = np.array(traj["obj_ids"])

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
