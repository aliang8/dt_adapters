import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from models.model import TrajectoryModel
from models.trajectory_gpt2 import TrajectoryGPT2

from collections import OrderedDict
from mw_constants import OBJECTS

from torch.distributions import Normal, Independent, Categorical
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


class DecisionTransformerSeparateState(TrajectoryModel):
    def __init__(
        self,
        state_dim,
        act_dim,
        pos_hand,
        goal_pos,
        obs_obj_max_len,
        hidden_size,
        component_hidden_size,
        stochastic=False,
        log_std_min=-20,
        log_std_max=2,
        remove_pos_embs=False,
        stochastic_tanh=False,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        approximate_entropy_samples=None,
        predict_return_dist=False,
        max_return=4000,
        bin_width=50,
        num_return_samples=250,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs,
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = TrajectoryGPT2(config)

        # state embedding
        self.pos_hand = pos_hand  # for gripper
        self.goal_pos = goal_pos
        self.obs_obj_max_len = obs_obj_max_len
        self.gripper_distance_apart = 1

        self.embed_arm_state = torch.nn.Linear(
            self.pos_hand + self.gripper_distance_apart, component_hidden_size
        )
        self.embed_obj_id = nn.Embedding(len(OBJECTS), component_hidden_size)
        self.embed_goal_pos = torch.nn.Linear(goal_pos, component_hidden_size)
        self.embed_obj_state = torch.nn.Linear(
            obs_obj_max_len // 2, component_hidden_size
        )

        # projection layer
        self.embed_state = torch.nn.Linear(6 * component_hidden_size, hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        # self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)

        if stochastic:
            self.predict_action_mean = nn.Sequential(
                nn.Linear(hidden_size, self.act_dim),
            )
            self.predict_action_logstd = nn.Sequential(
                nn.Linear(hidden_size, self.act_dim),
            )
        else:
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )

        self.predict_return = torch.nn.Linear(hidden_size, 1)

        # Settings from stochastic actions
        self.stochastic = stochastic
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.stochastic_tanh = stochastic_tanh
        self.remove_pos_embs = remove_pos_embs
        self.approximate_entropy_samples = approximate_entropy_samples
        self.predict_return_dist = predict_return_dist
        self.num_return_samples = num_return_samples

        if predict_return_dist:
            self.num_bins = int(max_return / bin_width)
            self.predict_return_logits = torch.nn.Linear(hidden_size, self.num_bins)

    def forward(
        self,
        states,
        actions,
        returns_to_go,
        obj_ids,
        timesteps,
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

        state_embeddings = self.embed_state(concat_state_embeddings)
        # state_embeddings = self.embed_state(states)
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
