import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
import einops

from dt_adapters.models.model import TrajectoryModel
from dt_adapters.models.trajectory_gpt2 import TrajectoryGPT2
import dt_adapters.utils as utils

from collections import OrderedDict

from torch.distributions import Normal, Independent, Categorical
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

from typing import Optional, Tuple, Dict, Union
from omegaconf import DictConfig

img_feat_dim = {"clip": 1024, "resnet50": 2048, "r3m": 2048}


class TransformerPolicy(TrajectoryModel):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_length: int,
        max_episode_length: int,
        hidden_size: int,
        action_tanh: bool,
        gpt2_cfg: Optional[DictConfig] = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            state_dim,
            act_dim,
            max_length=max_length,
        )

        self.hidden_size = hidden_size
        gpt_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            **gpt2_cfg,
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = TrajectoryGPT2(gpt_config)

        # state embedding
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
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
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> utils.AttrDict:
        """
        Run a forward pass given observation representations and (optionally) goals.
        Arguments:
            states: Tensor[B, T, state_dim], states
            actions: Tensor[B, T, act_dim], actions
            timesteps: Tensor[B, T], timesteps
            attention_mask: Tensor[B, T]
            B: batch size, T: sequence length, E: observation embedding size, G: goal size.
        Returns:
            A dictionary of outputs:
                actions: Tensor[B, T, act_dim]
        """
        import ipdb

        ipdb.set_trace()
        B, T = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((B, T), dtype=torch.long).to(self.device)

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

        # [B, 2*T, D]
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        x = einops.rearrange(x, "B (T K) D -> B K T D", K=2)

        state_reps = x[:, 0]
        action_preds = self.predict_action(state_reps)

        out = utils.AttrDict(
            action_preds=action_preds,
        )
        return out

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        goal_states: Optional[torch.Tensor] = None,
        goal_img_feats: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run forward pass on model to generate next action. Used during inference time.

        Arguments:
            states: Tensor[T, D_S], states
            actions: Tensor[T, D_A], actions
            timesteps: Tensor[T], timesteps
            goal_state: Tensor[T_goal, D_S] goal states
            goal_img_feats: Dict[str, Tensor[T, D_O]] dictionary of image observation features
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

        if goal_states is not None:
            model_input["goal_states"] = einops.rearrange(goal_states, "T D -> 1 T D")

        if goal_img_feats is not None:
            model_input["goal_img_feats"] = dict()
            for k in goal_img_feats:
                model_input["goal_img_feats"][k] = einops.rearrange(
                    goal_img_feats[k], "T D -> 1 T D"
                )

        model_out = self.forward(**model_input, **kwargs)

        # [B, T, D_A]
        return model_out["action_preds"][0, -1]
