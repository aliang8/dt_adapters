import numpy as np
import copy
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers.adapters import AdapterLayer, Fuse
import einops

from dt_adapters.models.model import TrajectoryModel
from dt_adapters.models.trajectory_gpt2 import TrajectoryGPT2
import dt_adapters.utils.utils as utils

from collections import OrderedDict

from torch.distributions import Normal, Independent, Categorical
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

from typing import Optional, Tuple, Dict, Union, List
from omegaconf import DictConfig

class TacoFusion(nn.Module):
    """
    Implementation of a LayerAgnosticFusionComposition block.
    """

    def __init__(self, model_dim: List, adapter_names):
        super(TacoFusion, self).__init__()
        # if config.hidden_size % config.num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.model_dim = model_dim
        self.adapter_names = adapter_names

        self.W_q = nn.Linear(model_dim, model_dim)
        #self.W_k = nn.Linear(model_dim, model_dim)
        self.keys = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(model_dim, len(adapter_names))))     # (N, H)

    def forward(self, input_embeddings):
        # input_embeddings: (B, L, H)

        batch_size = input_embeddings.shape[0]      # B
        seq_len = input_embeddings.shape[1]         # L

        queries = []
        for i in range(seq_len):
            query_position_i = torch.mean(input_embeddings[:, :i+1], dim=1)     # (B, H)   : take average of embeddings from position 0 toi
            queries.append(query_position_i)
        query = torch.stack(queries, dim=1)         # (B, L, H)

        query_proj = self.W_q(query).unsqueeze(2)                       # (B, L, 1, H)
        #keys_proj = self.W_k(keys).permute(0, 2, 1)                    # (B, H, N)
        keys_proj = self.keys.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)   # (B, L, H, N)
        
        scores =  torch.matmul(query_proj, keys_proj).squeeze(2)        # (B, L, N)
        scores = torch.div(scores, np.sqrt(self.model_dim)) 
        attention_weights = nn.Softmax(dim=-1)(scores)                  # (B, L, N)
        return attention_weights

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

        self.fusion_config = None
        self.taco_fusion = nn.ModuleDict({})

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

        ######################
        # If currently doing taco-fusion
        active_adapters_setup = self.transformer.active_adapters
        if type(active_adapters_setup) is transformers.adapters.Fuse \
            and self.fusion_config is not None \
            and self.fusion_config["fusion_method"] == "taco-fusion":

            fusion_adapter_names = [a for a in active_adapters_setup]   # length N, if fusing N adapters

            #query = torch.mean(stacked_inputs, dim=1)       # [B, D] -- average embeddings across all timesteps, ensures that queries for consecutive timesteps are similar
            #keys = torch.randn(query.shape[0], len(fusion_adapter_names), self.transformer.config.n_embd).to(self.device)
            taco_adapter_weights = self.taco_fusion[','.join(fusion_adapter_names)](stacked_inputs)    # (B, L, N)

            for idx,adapter_name in enumerate(fusion_adapter_names):
                adapter_weight = taco_adapter_weights[:,:,idx]
                self.transformer.apply_to_adapter_layers(lambda i, layer: self.set_taco_fusion_weight(layer=layer, adapter_name=adapter_name, adapter_weight=adapter_weight))
                #print("Applied weight {:.4f} for adapter #{}, {}".format(adapter_weight[0].item(), idx, adapter_name))

        ######################

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

        # this applies during adapter training time
        if "adapter_fusion_attentions" in transformer_outputs:
            out.adapter_fusion_attentions = transformer_outputs[
                "adapter_fusion_attentions"
            ]
        return out

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
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
                pad_vector = torch.zeros(1, pad_length, *pad_shape).to(self.device)

                # perform left padding because GPT-2 is a left-to-right model
                model_input[key] = torch.cat(
                    [pad_vector, tensor],
                    dim=1,
                )

        model_out = self.forward(**model_input, **kwargs)

        # [B, T, action_dim]
        return model_out["action_preds"][0, -1]

    def freeze_backbone(self):
        modules_to_freeze = [
            self.embed_state,
            self.embed_timestep,
            self.embed_action,
            self.embed_ln,
            self.predict_action,
            self.transformer,
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def add_taco_fusion(self, model_dim, adapter_names, fusion_config):
        adapter_names_str = ','.join(adapter_names)
        self.taco_fusion[adapter_names_str] = TacoFusion(model_dim, adapter_names)
        self.fusion_config = fusion_config
        print("Added TACo Fusion module to model")
        #import pdb; pdb.set_trace()
    
    def set_taco_fusion_weight(self, layer, adapter_name, adapter_weight):
        if type(layer) is not AdapterLayer:
            return
        if adapter_name not in layer.adapters:
            return
        layer.adapters[adapter_name].taco_adapter_weight = adapter_weight   # (B,L)