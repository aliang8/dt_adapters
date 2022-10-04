import torch
import torch.nn as nn
import torch.nn.functional as F
from mw_constants import OBJECTS


class MWStateEmbeddingNet(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        pos_hand,
        goal_pos,
        obs_obj_max_len,
        num_layers,
        **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        # state embedding
        self.pos_hand = pos_hand  # for gripper
        self.goal_pos = goal_pos
        self.obs_obj_max_len = obs_obj_max_len
        self.gripper_distance_apart = 1

        self.embed_arm_state = torch.nn.Linear(
            self.pos_hand + self.gripper_distance_apart, hidden_size
        )
        self.embed_obj_id = nn.Embedding(len(OBJECTS), hidden_size)
        self.embed_goal_pos = torch.nn.Linear(goal_pos, hidden_size)
        self.embed_obj_state = torch.nn.Linear(obs_obj_max_len // 2, hidden_size)

        encoder_modules = []
        encoder_modules.append(nn.Linear(6 * hidden_size, hidden_size))
        encoder_modules.append(nn.ReLU())

        for _ in range(num_layers):
            encoder_modules.append(nn.Linear(hidden_size, hidden_size))
            encoder_modules.append(nn.ReLU())

        self.state_encoder = nn.Sequential(*encoder_modules)

    def forward(self, states, obj_ids):
        batch_size, seq_length = states.shape[0], states.shape[1]

        curr_obs = states[:, :, :18]
        prev_obs = states[:, :, 18:36]

        # split the state into subparts
        curr_state = torch.split(
            curr_obs,
            [
                self.pos_hand + self.gripper_distance_apart,
                self.obs_obj_max_len,
            ],
            dim=-1,
        )
        arm_state = curr_state[0]
        obj_states = curr_state[1]
        goal_pos = states[:, :, -3:]

        # embed each modality with a different head
        arm_state_embeddings = self.embed_arm_state(arm_state)
        goal_pos_embeddings = self.embed_goal_pos(goal_pos)

        # assume there are two objects
        # each obj state is pos + quaternion
        obj_1_state, obj_2_state = torch.chunk(obj_states, 2, dim=-1)
        obj_1_id_embeddings = (
            self.embed_obj_id(obj_ids[:, 0]).unsqueeze(1).repeat((1, seq_length, 1))
        )
        obj_2_id_embeddings = (
            self.embed_obj_id(obj_ids[:, 1]).unsqueeze(1).repeat((1, seq_length, 1))
        )

        obj_1_state_embeddings = self.embed_obj_state(obj_1_state)
        obj_2_state_embeddings = self.embed_obj_state(obj_2_state)

        concat_state_embeddings = torch.cat(
            [
                arm_state_embeddings,
                obj_1_id_embeddings,
                obj_1_state_embeddings,
                obj_2_id_embeddings,
                obj_2_state_embeddings,
                goal_pos_embeddings,
            ],
            dim=-1,
        )
        state_embeddings = self.state_encoder(concat_state_embeddings)
        return state_embeddings
