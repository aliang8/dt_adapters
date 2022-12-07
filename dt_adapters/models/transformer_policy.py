import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
import einops

from dt_adapters.models.model import TrajectoryModel
from dt_adapters.models.trajectory_gpt2 import TrajectoryGPT2
from dt_adapters.models.state_embedding_net import StateEmbeddingNet
from dt_adapters.mw_constants import OBJECTS
import dt_adapters.general_utils as general_utils

from collections import OrderedDict

from torch.distributions import Normal, Independent, Categorical
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

from typing import Optional, Tuple, Dict, Union
from omegaconf import DictConfig


class TransformerPolicy(TrajectoryModel):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_length: int,
        max_episode_length: int,
        hidden_size: int,
        action_tanh: bool,
        freeze_bottom_n_layers: int = 0,
        goal_conditional: str = None,
        state_encoder: Optional[DictConfig] = None,
        gpt2_cfg: Optional[DictConfig] = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            state_dim,
            act_dim,
            max_length=max_length,
        )

        # concatenate proprio with the view embeddings
        self.effective_input_dim = (
            state_encoder.proprio
            + len(state_encoder.image_keys) * state_encoder.r3m_feat_dim
        )
        self.goal_conditional = goal_conditional

        if goal_conditional == "concat":
            self.effective_input_dim *= 2

        self.hidden_size = hidden_size
        gpt_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            **gpt2_cfg,
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = TrajectoryGPT2(gpt_config)

        # state embedding
        self.embed_state = torch.nn.Linear(self.effective_input_dim, self.hidden_size)
        self.embed_timestep = nn.Embedding(max_episode_length, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)

        action_predictor = []
        action_predictor.append(nn.Linear(self.hidden_size, self.act_dim))

        if action_tanh:
            action_predictor.append(nn.Tanh())

        self.predict_action = nn.Sequential(*action_predictor)
        self.device = device

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        goal_states: Optional[torch.Tensor] = None,
        img_feats: Union[torch.Tensor] = None,
        goal_img_feats: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> general_utils.AttrDict:
        """
        Run a forward pass given observation representations and (optionally) goals.
        Arguments:
            states: Tensor[B, T, state_dim], states
            actions: Tensor[B, T, act_dim], actions
            timesteps: Tensor[B, T], timesteps
            goal_states: Tensor[B, T_goal, state_dim] goal states
            img_feats: Union[Tensor[B, T, obs_dim]] dictionary of image observation features
            goal_img_feats: Union[Tensor[B, T_goal, obs_dim]]
            attention_mask: Tensor[B, T]
            B: batch size, T: sequence length, E: observation embedding size, G: goal size.
        Returns:
            A dictionary of outputs:
                actions: Tensor[B, T, act_dim]
                adapter_attentions
        """
        B, T = states.shape[0], states.shape[1]
        goal_seq_len = goal_states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((B, T), dtype=torch.long).to(self.device)

        if img_feats is not None:
            # concatenate the image feat with the state feat
            # make sure the image ones come first
            img_feat = torch.cat(list(img_feats.values()), dim=-1)
            states = torch.cat([img_feat, states], dim=-1)

        if self.goal_conditional:
            # [B, T_goal, obs_dim + state_dim]
            goal_img_feats = torch.cat(list(goal_img_feats.values()), dim=-1)
            goal_states = torch.cat([goal_img_feats, goal_states], dim=-1)

            if self.goal_conditional == "prepend":
                # [B, T_goal + T, obs_dim + state_dim]
                states = torch.cat([goal_states, states], dim=1)
                import ipdb

                ipdb.set_trace()
                pad = torch.zeros((B, goal_seq_len, actions.shape[-1])).to(self.device)
                actions = torch.cat([pad, actions], dim=1)

                pad = timesteps[:, 0] - 1
                timesteps = torch.cat([pad, timesteps], dim=1)

            elif self.goal_conditional == "concat":
                # repeat the goal_information T times
                goal_states = einops.repeat(
                    goal_states, "B T D -> B (T repeat) D", repeat=T
                )
                states = torch.cat([goal_states, states], dim=2)

        states = states.float()
        actions = actions.float()
        timesteps = timesteps.long()
        attention_mask = attention_mask.long()

        # embed input tokens
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        # (s_0, a_0, s_1, a_1, ...)
        # [B, 2, T, D]
        stacked_inputs = torch.stack((state_embeddings, action_embeddings), dim=1)
        stacked_inputs = einops.rearrange(stacked_inputs, "B L T D -> B (T L) D", L=2)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # [B, 2, T]
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).to(self.device)

        stacked_attention_mask = einops.rearrange(
            stacked_attention_mask, "B L T -> B (T L)", L=2
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

        # [B, 2*T, D]
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        x = einops.rearrange(x, "B (T K) D -> B K T D", K=2)

        state_reps = x[:, 0]
        action_preds = self.predict_action(state_reps)

        out = general_utils.AttrDict(
            action_preds=action_preds,
            adapter_fusion_attentions=adapter_fusion_attentions,
        )
        return out

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        goal_img_feats: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run forward pass on model to generate next action. Used during inference time.

        Arguments:
            states: Tensor[T, D_S], states
            actions: Tensor[T, D_A], actions
            timesteps: Tensor[T], timesteps
            goal_state: Tensor[T_goal, D_S] goal states
            img_feats: Union[Tensor[T, D_O]] dictionary of image observation features
            B: batch size, T: sequence length
        Returns:
            action: torch.Tensor[D_A]
        """
        # we don't care about the past rewards in this model
        # add batch dimension to each input value
        model_input = dict(
            states=einops.rearrange(states, "T D -> 1 T D"),
            actions=einops.rearrange(actions, "T D -> 1 T D"),
            timesteps=einops.rearrange(timesteps, "T -> 1 T"),
            attention_mask=torch.ones((1, actions.shape[0])).to(self.device).long(),
        )

        for key in model_input:
            if self.max_length:
                # take the most recent timesteps
                tensor = model_input[key][:, -self.max_length :]

                # pad all the values until the have the same number of steps
                pad_length = self.max_length - tensor.shape[1]
                pad_shape = tensor.shape[2:]

                model_input[key] = torch.cat(
                    [torch.zeros(1, pad_length, *pad_shape).to(self.device), tensor],
                    dim=1,
                )

        model_out = self.forward(**model_input, **kwargs)

        # B, T, D_A
        return model_out["action_preds"][0, -1]

    def freeze_backbone(self):
        # freeze everything
        for module in [
            self.embed_state,
            self.embed_action,
            self.embed_timestep,
            self.embed_ln,
            self.predict_action,
            self.transformer,
        ]:
            for param in module.parameters():
                param.requires_grad = False

        if self.freeze_bottom_n_layers > 0:
            # freeze the bottom n layers and also freeze the input embedding layers
            # train all other layers but the one closest to the input
            # also train the action prediction head
            for param in self.transformer.transformer.h[
                self.freeze_bottom_n_layers :
            ].parameters():
                param.requires_grad = True

            for param in self.embed_action.parameters():
                param.requires_grad = True
