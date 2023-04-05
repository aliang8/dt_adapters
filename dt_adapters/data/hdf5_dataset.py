import torch
import h5py
import os
import abc
import random
import numpy as np
import einops
import collections
from dt_adapters.utils import pad_to_length
from typing import Union, Callable, Optional, Sequence, List, Any, Dict
from torch.utils.data import Dataset, Sampler, TensorDataset, DataLoader


class HDF5TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories loaded from an HDF5 file.
    HDF5 files are organized as /task_name/demo_i/metadata_keys


    HDF5TrajectoryDataset[i] returns: (states, actions, rewards, dones)
        states: Tensor[T, ...], T frames of states
        actions: Tensor[T, ...], T frames of actions
    """

    def __init__(
        self,
        data_dir: str = "",
        context_length: int = 5,
        max_episode_length: int = 500,
        num_demos_per_task: int = 10,
        tasks: List[str] = [],
        camera_names: List[str] = [],
    ):
        self.context_length = context_length
        self.max_episode_length = max_episode_length
        self.data_dir = data_dir
        self.num_demos_per_task = num_demos_per_task
        self.tasks = tasks
        self.camera_names = camera_names

        self.task_trajectories, self.all_trajectories = self.load_from_file()

        # print statistics
        self.state_mean, self.state_std = self.compute_stats()

    def compute_stats(self) -> Union[np.array]:
        all_states = [traj["states"] for traj in self.all_trajectories]
        all_states = np.concatenate(all_states, axis=0)

        state_mean, state_std = (
            np.mean(all_states, axis=0),
            np.std(all_states, axis=0) + 1e-6,
        )

        return state_mean, state_std

    def load_from_file(self) -> Dict:
        # mapping from task name to list of trajectories
        trajectories = collections.defaultdict(list)
        all_trajectories = []

        # Load trajectories from HDF5 file
        for task_name in self.tasks:
            data_file = os.path.join(self.data_dir, f"{task_name}.hdf5")

            with h5py.File(data_file, "r") as f:
                keys = list(f["demo_0"].keys())

                for traj_index, (_, traj) in enumerate(f.items()):

                    if (
                        self.num_demos_per_task != -1
                        and traj_index >= self.num_demos_per_task
                    ):
                        continue

                    trajectory = dict()

                    for k in keys:
                        if isinstance(traj[k], h5py.Dataset):
                            trajectory[k] = traj[k][()]

                    trajectory["images"] = dict()
                    for k in self.camera_names:
                        if k in traj["images"]:
                            trajectory["images"][k] = traj["images"][k][()]
                        else:
                            print(f"{k} not available in dataset")

                    # sanity check
                    assert (
                        trajectory["states"].shape[0] == trajectory["actions"].shape[0]
                    )

                    trajectories[task_name].append(trajectory)
                    all_trajectories.append(trajectory)

        return trajectories, all_trajectories

    def __len__(self):
        return len(self.all_trajectories)

    def __getitem__(self, idx):
        """
        Returns a random segment of a trajectory of context length.
        Segment is a dictionary containing: states, actions, rewards, dones, timesteps, attention_mask
        """
        traj = self.all_trajectories[idx]
        total_seq_len = traj["states"].shape[0]

        # randint is inclusive on both ends, it's okay if we start at the end, we just need to pad
        start = np.random.randint(0, total_seq_len - 1)
        end = start + self.context_length

        # figure out how much we need to pad the sequence by to get context_length
        pad_length = self.context_length - traj["states"][start:end].shape[0]

        subseq = dict()

        # perform left padding
        for k in traj.keys():
            if k == "images":  # we don't use the image information right now
                continue
            window = traj[k][start:end]
            subseq[k] = pad_to_length(window, self.context_length)

        # add timesteps
        subseq["timesteps"] = np.arange(start, end)
        subseq["timesteps"][:pad_length] = 0

        # add mask
        subseq["attention_mask"] = np.ones((end - start))
        subseq["attention_mask"][:pad_length] = 0
        return subseq


if __name__ == "__main__":
    dataset = HDF5TrajectoryDataset(
        context_length=5,
        max_episode_length=500,
        data_dir="/data/anthony/dt_adapters/data",
        camera_names=["corner"],
        tasks=["pick-place-v2"],
    )

    for i in range(10):
        print(dataset[i].keys())
        for k in dataset[i]:
            if not isinstance(dataset[i][k], dict):
                print(k, dataset[i][k].shape, type(dataset[i][k]))
