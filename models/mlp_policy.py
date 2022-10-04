import torch
import torch.nn as nn
import torch.nn.functional as F
from models.state_vec_embedding import MWStateEmbeddingNet


class MLPPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        pos_hand,
        goal_pos,
        obs_obj_max_len,
        num_prediction_head_layers=0,
        emb_state_separate=False,
        **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.emb_state_separate = emb_state_separate

        # pretrained backbone
        if self.emb_state_separate:
            self.encoder = MWStateEmbeddingNet(
                state_dim,
                act_dim,
                hidden_size,
                pos_hand,
                goal_pos,
                obs_obj_max_len,
                **kwargs
            )
        else:
            import ipdb

            ipdb.set_trace()
            self.encoder = nn.Sequential(
                nn.Linear(state_dim + 2, hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, hidden_size),
            )

        # fine-tuneable layers
        prediction_head = []
        for _ in range(num_prediction_head_layers):
            prediction_head.append(nn.Linear(hidden_size, hidden_size))
            prediction_head.append(nn.ReLU())

        prediction_head.append(nn.Linear(hidden_size, act_dim))

        self.prediction_layer = nn.Sequential(*prediction_head)

    def freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, states, actions, obj_ids, **kwargs):
        states = states.float()
        obj_ids = obj_ids.long()

        if self.emb_state_separate:
            embedding = self.encoder(states, obj_ids)
        else:
            embedding = self.encoder(states.float())

        action_preds = self.prediction_layer(embedding)
        return None, action_preds, None, None, None

    def reset(self):
        pass

    def get_action(self, states, obj_ids, **kwargs):
        # only take last state
        states = states[-1].reshape(1, 1, self.state_dim)
        predictions = self.forward(states, None, obj_ids)
        action_pred = predictions[1].squeeze()
        return action_pred, None, {}
