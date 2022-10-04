from torch.utils.data import Dataset, Sampler
import os
import h5py
import random
import numpy as np
from general_utils import discount_cumsum
# from mw_constants import OBJECTS_TO_ENV
# from mw_utils import get_object_indices

from RLBench.rlbench.action_modes.action_mode import MoveArmThenGripper
from RLBench.rlbench.action_modes.arm_action_modes import JointVelocity
from RLBench.rlbench.action_modes.gripper_action_modes import Discrete
from RLBench.rlbench.environment import Environment
from RLBench.rlbench.observation_config import ObservationConfig
from RLBench.rlbench.tasks import MT30_V1

class RLBenchDemoDataset(Dataset):
    def __init__(self, config):
        # self.config = config
        # self.state_dim = config.state_dim
        # self.act_dim = config.act_dim
        # self.context_len = config.context_len
        # self.max_ep_len = config.max_ep_len

        # self.trajectories = []
        # all_states = []

        # # load trajectories into memory
        # data_file = os.path.join(os.environ["DATA_DIR"], config.data_file)
        # with h5py.File(data_file, "r") as f:
        #     envs = list(f.keys())

        #     for env in envs:
        #         num_demos = len(f[env].keys())

        #         for k, demo in f[env].items():
        #             states = demo["obs"][()]
        #             all_states.append(states)

        #             self.trajectories.append(
        #                 {
        #                     "states": demo["obs"][()],
        #                     "obj_ids": get_object_indices(env),
        #                     "actions": demo["action"][()],
                            # "rewards": demo["reward"][()],
                            # "dones": demo["done"][()],
                            # "returns": discount_cumsum(demo["reward"][()], gamma=1.0),
        #                     "timesteps": np.arange(len(states)),
        #                     "attention_mask": np.ones(len(states)),
        #                     "online": 0,
        #                 }
        #             )

        # # not sure if this is proper
        # all_states = np.concatenate(all_states, axis=0)

        # self.state_mean, self.state_std = (
        #     np.mean(all_states, axis=0),
        #     np.std(all_states, axis=0) + 1e-6,
        # )

        live_demos = False
        DATASET = '' if live_demos else '/home/ishika/dt_adapters/rlbench_dt/data'

        obs_config = ObservationConfig()
        obs_config.set_all(True)

        env = Environment(
            dataset_root=DATASET,
            action_mode=MoveArmThenGripper(
                arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
            obs_config=ObservationConfig(),
            headless=False)
        env.launch()

        train_tasks = MT30_V1['train']

        demos=[]
        for task in train_tasks:
            task = env.get_task(ReachTarget)
            demos += task.get_demos(2, live_demos=live_demos)  # -> List[List[Observation]]

        demos = np.array(demos).flatten()
        

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

        return out
