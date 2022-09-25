from torch.utils.data import Dataset, Sampler
import os
import h5py
import random
import numpy as np
from utils import discount_cumsum

OBJECTS_TO_ENV = {
    "button": [
        "button_press_topdown_v2",
        "button_press_topdown_wall_v2",
        "button_press_v2",
        "button_press_wall_v2",
    ],
    "basketball": ["basketball_v2"],
    "round_nut": ["assembly_v2", "disassemble_v2"],
    "block": [
        "bin_picking_v2",
        "hand_insert_v2",
        "pick_out_of_hole_v2",
        "pick_place_v2",
        "pick_place_wall_v2",
        "push_back_v2",
        "push_v2",
        "push_wall_v2",
        "shelf_place_v2",
        "sweep_into_v2",
        "sweep_v2",
    ],
    "top_link": ["box_close_v2"],
    "coffee_button_start": ["coffee_button_v2"],
    "coffee_mug": ["coffee_pull_v2", "coffee_push_v2"],
    "dial": ["dial_turn_v2"],
    "door": ["door_close_v2", "door_open_v2"],  # can we merge these together?
    "door_link": ["door_lock_v2", "door_unlock_v2", "door_v2"],
    "drawer_link": ["drawer_close_v2", "drawer_open_v2"],
    "faucet_handle": ["faucet_close_v2", "faucet_open_v2"],
    "hammer": ["hammer_v2"],
    "handle": [
        "handle_press_side_v2",
        "handle_press_v2",
        "handle_pull_side_v2",
        "handle_pull_v2",
        "lever_pull_v2",
    ],
    "peg": ["peg_insert_side_v2", "peg_unplug_side_v2"],
    "puck": [
        "plate_slide_back_side_v2",
        "plate_slide_back_v2",
        "plate_slide_side_v2",
        "plate_slide_v2",
    ],
    "no_obj": ["reach_v2", "reach_wall_v2"],
    "shelf": ["shelf_place_v2"],
    "soccer_ball": ["soccer_v2"],
    "stick": ["stick_push_v2", "stick_pull_v2"],
    "window_handle": ["window_close_v2", "window_open_v2"],
}

OBJECTS = list(OBJECTS_TO_ENV.keys())
OBJECTS = ["no_obj"] + OBJECTS

ENV_TO_OBJECTS = {}
for obj, envs in OBJECTS_TO_ENV.items():
    for env in envs:
        if env not in ENV_TO_OBJECTS:
            ENV_TO_OBJECTS[env] = [obj]
        else:
            ENV_TO_OBJECTS[env].append(obj)


class MWDemoDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.state_dim = cfg.state_dim
        self.act_dim = cfg.act_dim
        self.context_len = cfg.context_len
        self.max_ep_len = cfg.max_ep_len
        self.episodes = []
        all_states = []
        # load episodes into memory
        data_file = os.path.join(cfg.data_dir, cfg.data_file)
        with h5py.File(data_file, "r") as f:
            envs = list(f.keys())

            for env in envs:
                num_demos = len(f[env].keys())

                # get object indices
                objects_in_env = ENV_TO_OBJECTS[
                    env.replace("-goal-observable", "").replace("-", "_")
                ]
                object_indices = [0, 0]
                for i, obj in enumerate(objects_in_env):
                    object_indices[i] = OBJECTS.index(obj)

                for k, demo in f[env].items():
                    states = demo["obs"][()]
                    all_states.append(states)

                    self.episodes.append(
                        {
                            "states": demo["obs"][()],
                            "object_indices": object_indices,
                            "actions": demo["action"][()],
                            "rewards": demo["reward"][()],
                            "dones": demo["done"][()],
                            "timesteps": np.arange(len(states)),
                            "attention_mask": np.ones(len(states)),
                        }
                    )

        # not sure if this is proper
        all_states = np.concatenate(all_states, axis=0)

        self.state_mean, self.state_std = (
            np.mean(all_states, axis=0),
            np.std(all_states, axis=0) + 1e-6,
        )

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        traj = self.episodes[idx]

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
        }

        return out
