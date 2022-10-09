import torch
import torch.nn as nn
import torch.nn.functional as F

from dt_adapters.models.state_embedding_net import StateEmbeddingNet
import dt_adapters.general_utils as general_utils


class MLPPolicy(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.state_dim = self.config.state_dim
        self.act_dim = self.config.act_dim
        self.hidden_size = self.config.hidden_size
        self.emb_state_separate = self.config.emb_state_separate

        # pretrained backbone
        self.encoder = StateEmbeddingNet(config.state_encoder)

        # fine-tuneable layers
        prediction_head = []
        for _ in range(self.config.num_prediction_head_layers):
            prediction_head.append(nn.Linear(self.hidden_size, self.hidden_size))
            prediction_head.append(nn.ReLU())

        prediction_head.append(nn.Linear(self.hidden_size, self.act_dim))

        self.prediction_layer = nn.Sequential(*prediction_head)

    def freeze_backbone(self):
        general_utils.freeze_module(self.encoder)

    def forward(self, states, actions, img_feats=None, obj_ids=None, **kwargs):
        states = states.float()
        obj_ids = obj_ids.long()

        if self.emb_state_separate:
            embedding = self.encoder(states, img_feats, obj_ids)
        else:
            embedding = self.encoder(states.float())

        action_preds = self.prediction_layer(embedding)
        return None, action_preds, None, None, None

    def reset(self):
        pass

    def get_action(self, states, img_feats=None, obj_ids=None, **kwargs):
        # only take last state
        states = states[-1].reshape(1, 1, self.state_dim)
        predictions = self.forward(states, None, img_feats, obj_ids)
        action_pred = predictions[1].squeeze()
        return action_pred, None, {}
