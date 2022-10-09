import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

import dt_adapters.general_utils as general_utils
from dt_adapters.mw_constants import OBJECTS


class StateEmbeddingNet(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.state_dim = self.config.state_dim
        self.hidden_size = self.config.hidden_size

        # state embedding networks
        if "image" in self.config.state_keys:
            projection_layers = []
            self.img_feat_dim = 0
            for k in self.config.image_keys:
                if "rgb" in k:
                    self.img_feat_dim += self.config.clip_feat_dim
                if "depth" in k:
                    self.img_feat_dim += self.config.resnet_feat_dim

            projection_layers.append(nn.Linear(self.img_feat_dim, self.hidden_size))

            for _ in range(self.config.num_img_proj_layers):
                projection_layers.append(nn.ReLU())
                projection_layers.append(nn.Linear(self.hidden_size, self.hidden_size))

            self.projection = nn.Sequential(*projection_layers)

        if "low_level" in self.config.state_keys:
            # self.pos_hand = self.config.pos_hand  # for gripper
            # self.goal_pos = self.config.goal_pos
            # self.obs_obj_max_len = self.config.obs_obj_max_len
            # self.gripper_distance_apart = 1

            # self.embed_arm_state = torch.nn.Linear(
            #     self.pos_hand + self.gripper_distance_apart, self.hidden_size
            # )
            # self.embed_obj_id = nn.Embedding(len(OBJECTS), self.hidden_size)
            # self.embed_goal_pos = torch.nn.Linear(
            #     self.config.goal_pos, self.hidden_size
            # )
            # self.embed_obj_state = torch.nn.Linear(
            #     self.config.obs_obj_max_len // 2, self.hidden_size
            # )
            # self.combined_embed_dim += 6 * self.hidden_size

            encoder_modules = []
            encoder_modules.append(nn.Linear(self.state_dim, self.hidden_size))

            for _ in range(self.config.num_ll_enc_layers):
                encoder_modules.append(nn.ReLU())
                encoder_modules.append(nn.Linear(self.hidden_size, self.hidden_size))

            self.ll_state_encoder = nn.Sequential(*encoder_modules)

    def forward(self, states, img_feats=None, obj_ids=None, **kwargs):
        # encode image observation
        batch_size, seq_length = states.shape[0], states.shape[1]

        encodings = []

        if "image" in self.config.state_keys:
            img_encoding = self.projection(img_feats.float())
            img_encoding = img_encoding.reshape(batch_size, seq_length, -1)
            encodings.append(img_encoding)

        if "low_level" in self.config.state_keys:
            # curr_obs = states[:, :, :18]
            # prev_obs = states[:, :, 18:36]

            # # split the state into subparts
            # curr_state = torch.split(
            #     curr_obs,
            #     [
            #         self.pos_hand + self.gripper_distance_apart,
            #         self.obs_obj_max_len,
            #     ],
            #     dim=-1,
            # )
            # arm_state = curr_state[0]
            # obj_states = curr_state[1]
            # goal_pos = states[:, :, -3:]

            # # embed each modality with a different head
            # arm_state_embeddings = self.embed_arm_state(arm_state)
            # goal_pos_embeddings = self.embed_goal_pos(goal_pos)

            # # assume there are two objects
            # # each obj state is pos + quaternion
            # obj_1_state, obj_2_state = torch.chunk(obj_states, 2, dim=-1)
            # obj_1_id_embeddings = (
            #     self.embed_obj_id(obj_ids[:, 0]).unsqueeze(1).repeat((1, seq_length, 1))
            # )
            # obj_2_id_embeddings = (
            #     self.embed_obj_id(obj_ids[:, 1]).unsqueeze(1).repeat((1, seq_length, 1))
            # )

            # obj_1_state_embeddings = self.embed_obj_state(obj_1_state)
            # obj_2_state_embeddings = self.embed_obj_state(obj_2_state)

            # concat_state_embeddings = torch.cat(
            #     [
            #         arm_state_embeddings,
            #         obj_1_id_embeddings,
            #         obj_1_state_embeddings,
            #         obj_2_id_embeddings,
            #         obj_2_state_embeddings,
            #         goal_pos_embeddings,
            #     ],
            #     dim=-1,
            # )
            # ll_state_embeddings = self.ll_state_encoder(concat_state_embeddings)
            ll_state_embeddings = self.ll_state_encoder(states)
            encodings.append(ll_state_embeddings)

        # element-wise add embeddings
        state_embeddings = torch.stack(encodings).sum(0)
        return state_embeddings
