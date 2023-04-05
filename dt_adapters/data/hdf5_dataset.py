import torch
import h5py
import os
import abc
import random
import numpy as np
import einops
from dt_adapters.general_utils import discount_cumsum, AttrDict
from typing import Union, Callable, Optional, Sequence, List, Any
from torch.utils.data import Dataset, Sampler, TensorDataset, DataLoader


class HDF5TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories loaded from an HDF5 file.
    HDF5 files are organized as /task_name/demo_i/metadata_keys


    HDF5TrajectoryDataset[i] returns: (states, actions, mask, img_feats, goal_state, goal_obs)
        states: Tensor[T, ...], T frames of states
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
        img_feats/{key}: Tensor[T, ...], T frames of image features
        goal_state: Tensor[T, ...], T frames of goal states
        goal_obs/{key}: Tensor[T, ...], T frames of goal image features
    """

    def __init__(
        self,
        context_length: int = 5,
        max_episode_length: int = 500,
        data_dir: str = "",
        data_file: str = "",
        num_demos_per_task: int = 10,
        goal_conditional: str = "",
        goal_seq_len: int = 0,
        min_goal_sep: int = 0,
        only_sample_tail: bool = False,
        image_keys: List[str] = [],
        proprio: int = 0,
        stage: str = "pretraining",
        eval_task: str = "",
        **kwargs,
    ):
        self.context_length = context_length
        self.max_episode_length = max_episode_length
        self.data_dir = data_dir
        self.data_file = data_file
        self.num_demos_per_task = num_demos_per_task

        self.goal_conditional = goal_conditional
        self.goal_seq_len = goal_seq_len
        self.min_goal_sep = min_goal_sep
        self.only_sample_tail = only_sample_tail
        self.image_keys = image_keys
        self.proprio = proprio
        self.stage = stage
        self.eval_task = eval_task

        self.all_trajectories = self.load_from_file()

    def compute_stats(self) -> Union[np.array]:
        all_states = [traj["states"] for traj in self.all_trajectories]
        all_states = np.concatenate(all_states, axis=0)

        state_mean, state_std = (
            np.mean(all_states, axis=0),
            np.std(all_states, axis=0) + 1e-6,
        )

        return state_mean, state_std

    def load_from_file(self) -> List[dict]:
        # Load trajectories from HDF5 file
        data_file = os.path.join(self.data_dir, self.data_file)

        trajectories = []

        with h5py.File(data_file, "r") as f:
            for task in list(f.keys()):
                for i, (_, demo) in enumerate(f[task].items()):
                    if self.eval_task and task != self.eval_task:
                        continue

                    # limit number of demos
                    if self.num_demos_per_task != -1 and i >= self.num_demos_per_task:
                        continue

                    trajectory = dict()

                    for k in list(demo.keys()):
                        if isinstance(demo[k], h5py.Dataset):
                            trajectory[k] = demo[k][()]

                    # add image features
                    trajectory["img_feats"] = dict()
                    for k in self.image_keys:
                        if k in demo["img_feats"]:
                            trajectory["img_feats"][k] = demo["img_feats"][k][()]
                        else:
                            print(f"{k} not available in dataset")

                    # remove the last state (added the last during rollout collection)
                    trajectory["states"] = trajectory["states"][:-1]
                    assert (
                        trajectory["states"].shape[0] == trajectory["actions"].shape[0]
                    )

                    trajectories.append(trajectory)

        return trajectories

    def __len__(self):
        return len(self.all_trajectories)

    def pad_to_length(self, array, final_length, pad_zeros=True):
        """
        Given an array of [N, D] pad with zeros until [T, D] where T is
        the target size. Returns a 2D array
        """

        # add extra dimension
        if len(array.shape) == 1:
            array = array[:, np.newaxis]

        shape = array.shape[1:]
        pad_length = final_length - array.shape[0]

        if pad_zeros:
            pad = np.zeros((pad_length, *shape))
        else:
            pad = np.ones((pad_length, *shape)) * -10

        # pad to context length
        array = np.concatenate([pad, array], axis=0)
        return array

    def __getitem__(self, idx):
        """
        Returns a segment of demonstration based on context length
        """
        traj = self.all_trajectories[idx]

        total_seq_len = traj["states"].shape[0]
        start = np.random.randint(0, total_seq_len - 1)
        end = start + self.context_length

        pad_length = self.context_length - traj["states"][start:end].shape[0]

        subseq = dict()

        # perform left padding
        for k in traj.keys():
            if k != "img_feats":
                pad_zeros = False if k in ["actions", "dones"] else True
                window = traj[k][start:end]
                subseq[k] = self.pad_to_length(window, self.context_length, pad_zeros)

        # add image features
        subseq["img_feats"] = dict()
        for img_key in self.image_keys:
            window = traj["img_feats"][img_key][start:end]
            subseq["img_feats"][img_key] = torch.Tensor(
                self.pad_to_length(window, self.context_length, True)
            )

        # add timesteps
        subseq["timesteps"] = np.arange(start, end)
        subseq["timesteps"][:pad_length] = 0

        # add mask
        subseq["attention_mask"] = np.ones((end - start))
        subseq["attention_mask"][:pad_length] = 0

        # add goal state and image observations
        if self.goal_conditional:
            valid_start_range = (
                end + self.min_goal_sep,
                total_seq_len - self.goal_seq_len,
            )

            subseq["goal_img_feats"] = dict()

            # select some future states to be the goal
            if valid_start_range[0] < valid_start_range[1]:
                if self.only_sample_tail:
                    goal_state = traj["states"][-self.goal_seq_len :]

                    for img_key in self.image_keys:
                        subseq["goal_img_feats"][img_key] = torch.Tensor(
                            traj["img_feats"][img_key][-self.goal_seq_len :]
                        )
                else:
                    start = np.random.randint(*valid_start_range)
                    end = start + self.goal_seq_len
                    goal_state = traj["states"][start:end]
                    for img_key in self.image_keys:
                        subseq["goal_img_feats"][img_key] = torch.Tensor(
                            traj["img_feats"][img_key][start:end]
                        )

            else:
                # zeros placeholder T x obs_dim
                goal_state = np.zeros((self.goal_seq_len, *subseq["states"].shape[1:]))

                for img_key in self.image_keys:
                    subseq["goal_img_feats"][img_key] = torch.zeros(
                        (self.goal_seq_len, *subseq["img_feats"][img_key].shape[1:])
                    )

            subseq["goal_states"] = goal_state
            subseq["goal_states"] = subseq["goal_states"][:, : self.proprio]

        # slice state input based on proprio
        subseq["states"] = subseq["states"][:, : self.proprio]
        return subseq


if __name__ == "__main__":
    dataset = HDF5TrajectoryDataset(
        context_length=5,
        max_episode_length=500,
        data_dir="/data/anthony/dt_adapters/data/metaworld_data",
        data_file="trajectories_resnet50_25.hdf5",
        goal_conditional="prepend",
        goal_seq_len=1,
        min_goal_sep=20,
        proprio=4,
        only_sample_tail=True,
        image_keys=["corner", "corner2"],
        eval_task="button-press-v2",
    )

    for i in range(10):
        print(dataset[i].keys())
        for k in dataset[i]:
            if not isinstance(dataset[i][k], dict):
                print(k, dataset[i][k].shape, type(dataset[i][k]))

    # loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=2,
    #     num_workers=4,
    #     drop_last=False,
    # )

    for batch in loader:
        print(batch.keys())
