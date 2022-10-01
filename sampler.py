import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset


class ImportanceWeightBatchSampler(Sampler):
    """
    Example:
        >>> list(ImportanceWeightBatchSampler(range(10), batch_size=3, drop_last=False))
    """

    def __init__(
        self,
        dataset,
        batch_size,
        len_temperature=None,
        return_temperature=None,
        shuffle=False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.len_temperature = len_temperature
        self.return_temperature = return_temperature

    def __iter__(self):
        # Dynamically recompute weights for trajectories based on their returns
        # or inversely with the length
        dist = self.get_sampling_dist()
        indices = np.arange(len(self.dataset))

        batch_inds = np.random.choice(
            indices,
            size=self.batch_size,
            replace=True,
            p=dist,  # reweights so we sample according to timesteps
        )

        if self.shuffle:
            random.shuffle(batch_inds)

        return iter(batch_inds)

    def get_sampling_dist(self):
        # Longer sequences are penalized
        # Favor sequences with higher return
        traj_lens = np.array(
            [len(path["states"]) for path in self.dataset.trajectories]
        )

        traj_returns = np.array(
            [path["rewards"].sum() for path in self.dataset.trajectories]
        )

        # shorter len
        assert traj_lens.max() <= 500
        inverted_lens = 500 - traj_lens + 1e-6
        weight_len = inverted_lens / sum(inverted_lens)

        # decrease the temperature term to make the distribution more
        # peaky so more like to sample trajs with lower len
        # weight_len = F.softmax(torch.tensor(weight_len) / self.len_temperature, dim=0)

        # more rewards
        traj_returns += 1e-6  # prevent NaN issues
        weight_rew = traj_returns / sum(traj_returns)

        # high retun is almost guaranteed good trajectory
        # weight_rew = F.softmax(
        #     torch.tensor(weight_rew) / self.return_temperature, dim=0
        # )

        # more weight on trajectories that have high return
        weight_avg = 0.2 * weight_len + 0.8 * weight_rew
        return weight_avg

    def get_remove_dist(self):
        traj_lens = np.array(
            [len(path["states"]) for path in self.dataset.trajectories]
        )

        traj_returns = np.array(
            [path["rewards"].sum() for path in self.dataset.trajectories]
        )

        weight_len = traj_lens / sum(traj_lens)

        traj_returns += 1e-6
        inverted_returns = traj_returns.max() - traj_returns
        weight_rew = inverted_returns / sum(inverted_returns)

        weight_avg = 0.5 * weight_len + 0.5 * weight_rew
        return weight_avg

    def __len__(self):
        return len(self.dataset)


class TestDataset(Dataset):
    def __init__(self):
        self.trajectories = []

        for i in range(10):
            self.trajectories.append(
                {"states": np.zeros((i * 100,)), "rewards": np.arange(10) * i}
            )

    def __getitem__(self, idx):
        print(idx)
        return self.trajectories[idx]

    def __len__(self):
        return len(self.trajectories)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = TestDataset()
    sampler = ImportanceWeightBatchSampler(dataset, 2)

    # for ind in sampler:
    #     print(ind)

    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=2,
        num_workers=1,
        drop_last=False,
    )

    for batch in data_loader:
        print(batch)
