import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from models.model import TrajectoryModel
from models.trajectory_gpt2 import TrajectoryGPT2
from models.state_vec_embedding import MWStateEmbeddingNet

from collections import OrderedDict
from mw_constants import OBJECTS

from torch.distributions import Normal, Independent, Categorical
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


class DecisionTransformerSeparateState(TrajectoryModel):
    def __init__(self, config, **kwargs):
        self.config = config

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
        if self.config.emb_state_separate:
            self.embed_state = MWStateEmbeddingNet(self.config.state_encoder)
        else:
            self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)

        self.embed_timestep = nn.Embedding(self.config.max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)

        # Settings from stochastic actions
        self.stochastic = self.config.stochastic
        self.log_std_min = self.config.log_std_min
        self.log_std_max = self.config.log_std_max
        self.stochastic_tanh = self.config.stochastic_tanh
        self.remove_pos_embs = self.config.remove_pos_embs
        self.approximate_entropy_samples = self.config.approximate_entropy_samples
        self.predict_return_dist = self.config.predict_return_dist
        self.num_return_samples = self.config.num_return_samples
        self.predict_return_dist = self.config.predict_return_dist

        if self.stochastic:
            self.predict_action_mean = nn.Sequential(
                nn.Linear(self.hidden_size, self.act_dim),
            )
            self.predict_action_logstd = nn.Sequential(
                nn.Linear(self.hidden_size, self.act_dim),
            )
        else:
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(self.hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if self.config.action_tanh else [])
                )
            )

        self.predict_return = torch.nn.Linear(self.hidden_size, 1)

        if self.predict_return_dist:
            self.num_bins = int(self.config.max_return / self.config.bin_width)
            self.predict_return_logits = torch.nn.Linear(
                self.hidden_size, self.num_bins
            )

    def forward(
        self,
        states,
        actions,
        returns_to_go,
        obj_ids,
        timesteps,
        img_feats=None,
        target_actions=None,
        attention_mask=None,
        use_means=False,
        use_rtg_mask=None,
        sample_return_dist=False,
        **kwargs
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(
                states.device
            )

        states = states.float()
        actions = actions.float()
        returns_to_go = returns_to_go.float()
        obj_ids = obj_ids.long()
        timesteps = timesteps.long()
        attention_mask = attention_mask.long()
        use_rtg_mask = use_rtg_mask.long()

        if self.config.emb_state_separate:
            state_embeddings = self.embed_state(states, obj_ids, img_feats)
        else:
            state_embeddings = self.embed_state(states)

        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        if not self.remove_pos_embs:
            # time embeddings are treated similar to positional embeddings
            state_embeddings = state_embeddings + time_embeddings.squeeze(1)
            action_embeddings = action_embeddings + time_embeddings.squeeze(1)
            returns_embeddings = returns_embeddings + time_embeddings.squeeze(1)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        # don't attend to the return tokens during pretraining
        return_attn_mask = torch.clone(attention_mask)
        return_attn_mask *= use_rtg_mask.int()

        stacked_attention_mask = (
            torch.stack((return_attn_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        state_reps = x[:, 1]
        action_reps = x[:, 2]

        # get predictions
        # predict next return given state and action
        if self.predict_return_dist:
            # predict binned returns
            return_probs = self.predict_return_logits(action_reps)

            # create discrete prob distribution
            return_dist = Categorical(F.softmax(return_probs, dim=-1))

            if sample_return_dist:
                # sample N RTGs from learned distribution during evaluation
                # and pick the one with highest value
                return_preds = return_dist.sample((self.num_return_samples,))
                return_preds = torch.max(return_preds, dim=0).values
            else:
                return_preds = return_dist.probs
        else:
            return_preds = self.predict_return(action_reps)

        # predict next state given state and action
        state_preds = self.predict_state(action_reps)

        # predict next action given state
        action_log_probs = None
        entropies = None

        if self.stochastic:
            means = self.predict_action_mean(state_reps)
            log_stds = self.predict_action_logstd(state_reps)

            # Bound log of standard deviations
            log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
            stds = torch.exp(log_stds)

            if self.stochastic_tanh:
                dist = TransformedDistribution(
                    Normal(means, stds), TanhTransform(cache_size=1)
                )
                action_dist = Independent(dist, 1)
            else:
                action_dist = Independent(Normal(means, stds), 1)

            # Sample from distribution or predict mean
            if use_means:
                if self.stochastic_tanh:
                    action_preds = torch.tanh(action_dist.base_dist.base_dist.mean)
                else:
                    action_preds = action_dist.mean
            else:
                action_preds = action_dist.rsample()

            if target_actions != None:
                # Clamp target actions to prevent nans
                eps = torch.finfo(target_actions.dtype).eps
                target_actions = torch.clamp(target_actions, -1 + eps, 1 - eps)
                action_log_probs = action_dist.log_prob(target_actions)
                # entropies = action_dist.base_dist.entropy()
                if self.stochastic_tanh:
                    entropies = -action_dist.log_prob(
                        action_dist.rsample(
                            sample_shape=torch.Size([self.approximate_entropy_samples])
                        )
                    ).mean(dim=0)
                else:
                    entropies = action_dist.entropy()
        else:
            action_preds = self.predict_action(state_reps)

        return state_preds, action_preds, return_preds, action_log_probs, entropies

    def reset(self):
        pass

    def get_action(
        self,
        states,
        actions,
        returns_to_go,
        obj_ids,
        timesteps,
        use_means,
        use_rtg_mask,
        sample_return_dist,
        **kwargs
    ):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device
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
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
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

        _, action_preds, return_preds, _, _ = self.forward(
            states,
            actions,
            returns_to_go,
            obj_ids,
            timesteps,
            target_actions=None,
            attention_mask=attention_mask,
            use_means=use_means,  # use mean action during evaluation
            use_rtg_mask=use_rtg_mask,
            sample_return_dist=sample_return_dist,
            **kwargs,
        )

        return action_preds[0, -1], return_preds[0, -1], {}

    def freeze_backbone(self):
        for module in [
            self.embed_state,
            self.embed_return,
            self.embed_action,
            self.embed_timestep,
        ]:
            for param in module.parameters():
                param.requires_grad = False
