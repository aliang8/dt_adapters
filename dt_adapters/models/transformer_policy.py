import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from dt_adapters.models.model import TrajectoryModel
from dt_adapters.models.trajectory_gpt2 import TrajectoryGPT2
from dt_adapters.models.state_embedding_net import StateEmbeddingNet
from dt_adapters.mw_constants import OBJECTS
import dt_adapters.general_utils as general_utils

from collections import OrderedDict

from torch.distributions import Normal, Independent, Categorical
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


class TransformerPolicy(TrajectoryModel):
    def __init__(self, config, **kwargs):
        self.config = config

        self.config.state_dim = (
            self.config.state_encoder.proprio
            + len(self.config.state_encoder.image_keys)
            * self.config.state_encoder.r3m_feat_dim
        )

        super().__init__(
            self.config.state_dim,
            self.config.act_dim,
            max_length=self.config.max_length,
        )

        self.hidden_size = self.config.hidden_size
        gpt_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            **self.config.gpt2,
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = TrajectoryGPT2(gpt_config)

        # state embedding
        self.embed_state = torch.nn.Linear(self.config.state_dim, self.hidden_size)
        self.embed_timestep = nn.Embedding(self.config.max_ep_len, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)

        action_predictor = []
        action_predictor.append(nn.Linear(self.hidden_size, self.act_dim))
        if self.config.action_tanh:
            action_predictor.append(nn.Tanh())

        self.predict_action = nn.Sequential(*action_predictor)

    def forward(
        self, states, actions, timesteps, img_feats=None, attention_mask=None, **kwargs
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.onebs((batch_size, seq_length), dtype=torch.long).to(
                states.device
            )

        if img_feats is not None:
            # concatenate the image feat with the state feat
            # make sure the image ones come first
            img_feat = torch.cat(list(img_feats.values()), dim=-1)
            states = torch.cat([img_feat, states], dim=-1)

        states = states.float()
        actions = actions.float()
        timesteps = timesteps.long()
        attention_mask = attention_mask.long()

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings.squeeze(1)
        action_embeddings = action_embeddings + time_embeddings.squeeze(1)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 2 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            output_adapter_fusion_attentions=True,
        )

        if hasattr(transformer_outputs, "adapter_fusion_attentions"):
            adapter_fusion_attentions = transformer_outputs.adapter_fusion_attentions
        else:
            adapter_fusion_attentions = None

        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        state_reps = x[:, 0]
        action_preds = self.predict_action(state_reps)

        out = general_utils.AttrDict(
            action_preds=action_preds,
            adapter_fusion_attentions=adapter_fusion_attentions,
        )
        return out

    def reset(self):
        pass

    def get_action(self, states, actions, timesteps, img_feats=None, **kwargs):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            if states is not None:
                states = states[:, -self.max_length :]

            actions = actions[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - actions.shape[1]),
                    torch.ones(actions.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=actions.device
            ).reshape(1, -1)

            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)

            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        if img_feats is not None:
            img_feats = img_feats.reshape(1, -1, img_feats.shape[-1])

            if self.max_length is not None:
                img_feats = img_feats[:, -self.max_length :]
                img_feats = torch.cat(
                    [
                        torch.zeros(
                            (
                                img_feats.shape[0],
                                self.max_length - img_feats.shape[1],
                                img_feats.shape[-1],
                            ),
                            device=img_feats.device,
                        ),
                        img_feats,
                    ],
                    dim=1,
                ).to(dtype=torch.float32)

        model_out = self.forward(
            states=states,
            actions=actions,
            timesteps=timesteps,
            img_feats=img_feats,
            target_actions=None,
            attention_mask=attention_mask,
            **kwargs,
        )

        return model_out["action_preds"][0, -1]

    def freeze_backbone(self):
        # freeze everything
        for module in [
            self.embed_state,
            self.embed_action,
            self.embed_timestep,
            self.embed_ln,
            self.predict_action,
            # self.transformer,
        ]:
            for param in module.parameters():
                param.requires_grad = False

        if self.config.train_prediction_head:
            for param in self.predict_action.parameters():
                param.requires_grad = True

        if self.config.train_state_embeddings:
            for module in [
                self.embed_state,
                self.embed_action,
                self.embed_timestep,
                self.embed_ln,
            ]:
                for param in module.parameters():
                    param.requires_grad = True

        if self.config.freeze_bottom_n_layers > 0:
            # freeze the bottom n layers and also freeze the input embedding layers
            # train all other layers but the one closest to the input
            # also train the action prediction head
            for param in self.transformer.transformer.h[
                self.config.freeze_bottom_n_layers :
            ].parameters():
                param.requires_grad = True

            for param in self.embed_action.parameters():
                param.requires_grad = True
