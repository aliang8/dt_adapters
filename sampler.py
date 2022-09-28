import random
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset


class ImportanceWeightBatchSampler(Sampler):
    """
    Example:
        >>> list(ImportanceWeightBatchSampler(range(10), batch_size=3, drop_last=False))
    """

    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        # Dynamically recompute weights for trajectories based on their returns
        # or inversely with the length

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

        # more rewards
        traj_returns += 1e-6  # prevent NaN issues
        weight_rew = traj_returns / sum(traj_returns)
        weight_avg = 0.5 * weight_len + 0.5 * weight_rew

        indices = np.arange(len(self.dataset))

        batch_inds = np.random.choice(
            indices,
            size=self.batch_size,
            replace=True,
            p=weight_avg,  # reweights so we sample according to timesteps
        )

        if self.shuffle:
            random.shuffle(batch_inds)

        return iter(batch_inds)

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
