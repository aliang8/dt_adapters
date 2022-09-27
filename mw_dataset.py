from torch.utils.data import Dataset, Sampler
import os
import h5py
import random
import numpy as np
from general_utils import discount_cumsum
from mw_constants import OBJECTS_TO_ENV
from mw_utils import get_object_indices


class MWDemoDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.state_dim = cfg.state_dim
        self.act_dim = cfg.act_dim
        self.context_len = cfg.context_len
        self.max_ep_len = cfg.max_ep_len
        self.trajectories = []
        all_states = []
        # load trajectories into memory
        data_file = os.path.join(cfg.data_dir, cfg.data_file)
        with h5py.File(data_file, "r") as f:
            envs = list(f.keys())

            for env in envs:
                num_demos = len(f[env].keys())

                for k, demo in f[env].items():
                    states = demo["obs"][()]
                    all_states.append(states)

                    self.trajectories.append(
                        {
                            "states": demo["obs"][()],
                            "object_indices": get_object_indices(env),
                            "actions": demo["action"][()],
                            "rewards": demo["reward"][()],
                            "dones": demo["done"][()],
                            "returns": discount_cumsum(demo["reward"][()], gamma=1.0),
                            "timesteps": np.arange(len(states)),
                            "attention_mask": np.ones(len(states)),
                            "online": 0,
                        }
                    )

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
        rtg = np.concatenate(
            [np.zeros((self.context_len - tlen, 1)), rtg], axis=0
        )  # / scale
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
            "object_indices": np.array(traj["object_indices"]),
            "online": traj["online"],
        }

        return out
