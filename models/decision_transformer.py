from turtle import pd
import numpy as np
import copy
import torch
import torch.nn as nn

import transformers

from models.model import TrajectoryModel
from models.trajectory_gpt2 import GPT2Model

from collections import OrderedDict
from robomimic.algo.algo import PolicyAlgo
from robomimic.algo.bc import BC
from robomimic.algo import register_algo_factory_func, PolicyAlgo, RolloutPolicy
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.models.obs_nets import ObservationGroupEncoder


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        # self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(hidden_size, self.act_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(
                states.device
            )

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        # returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps.long())

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        # returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 2 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # states (0), or actions (1); i.e. x[:,0,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # return_preds = self.predict_return(
        #     x[:, 2]
        # )  # predict next return given state and action
        state_preds = self.predict_state(
            x[:, 1]
        )  # predict next state given state and action
        action_preds = self.predict_action(x[:, 0])  # predict next action given state

        return state_preds, action_preds

    def get_action(self, states, actions, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        # returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            # returns_to_go = returns_to_go[:, -self.max_length :]
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
            # returns_to_go = torch.cat(
            #     [
            #         torch.zeros(
            #             (
            #                 returns_to_go.shape[0],
            #                 self.max_length - returns_to_go.shape[1],
            #                 1,
            #             ),
            #             device=returns_to_go.device,
            #         ),
            #         returns_to_go,
            #     ],
            #     dim=1,
            # ).to(dtype=torch.float32)
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

        _, action_preds = self.forward(
            states, actions, timesteps, attention_mask=attention_mask, **kwargs
        )

        return action_preds[0, -1]


@register_algo_factory_func("dt")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the DT algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return DT, {}


class DT(BC):
    def _create_networks(self):
        self.nets = nn.ModuleDict()

        state_dim = sum(sum(list(self.obs_shapes.values()), []))
        self.nets["policy"] = DecisionTransformer(
            state_dim=state_dim, act_dim=self.ac_dim, **self.global_config.model
        )

        # Encoder for all observation groups.
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(
            self.obs_config.encoder
        )

        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        input_batch = dict()
        # input_batch = copy.deepcopy(batch)

        if self.global_config.model.vision_only:
            input_batch["obs"] = batch["obs"]
        else:
            input_batch["states"] = torch.cat(
                [batch["obs"][k2] for k2 in batch["obs"].keys()], dim=-1
            )

        # input_batch["goal_obs"] = batch.get(
        #     "goal_obs", None
        # )  # goals may not be present
        input_batch["actions"] = batch["actions"]
        input_batch["timesteps"] = batch["timesteps"]
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _forward_training(self, batch):
        predictions = OrderedDict()

        if self.global_config.model.vision_only:
            enc_outputs = self.nets["encoder"](**batch)
            batch["states"] = enc_outputs

        state_preds, action_preds = self.nets["policy"](**batch)
        predictions["actions"] = action_preds
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        # losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        # losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        # action_losses = [
        #     self.algo_config.loss.l2_weight * losses["l2_loss"],
        #     self.algo_config.loss.l1_weight * losses["l1_loss"],
        #     self.algo_config.loss.cos_weight * losses["cos_loss"],
        # ]
        action_losses = [self.algo_config.loss.l2_weight * losses["l2_loss"]]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def get_action(self, states, actions, timesteps, **kwargs):
        return self.nets["policy"].get_action(states, actions, timesteps, **kwargs)


class DTRolloutPolicy(RolloutPolicy):
    def __init__(self, policy, obs_normalization_stats=None):
        self.policy = policy
        self.obs_normalization_stats = obs_normalization_stats

    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        """
        self.policy.set_eval()
        self.policy.reset()

        self.state_dim = sum(sum(list(self.policy.obs_shapes.values()), []))
        self.act_dim = self.policy.ac_dim
        self.device = self.policy.device

        self.states = torch.zeros(
            (0, self.state_dim), device=self.device, dtype=torch.float32
        )
        self.actions = torch.zeros(
            (0, self.act_dim), device=self.device, dtype=torch.float32
        )
        self.timesteps = torch.zeros((1, 0), device=self.device, dtype=torch.long)

    def _prepare_observation(self, ob):
        ob = super()._prepare_observation(ob)
        ob = torch.cat([v for k, v in ob.items()], dim=-1)
        return ob

    def __call__(self, ob, goal=None, t=0):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension,
                and np.array values for each key)
            goal (dict): goal observation
        """
        with torch.no_grad():
            ob = self._prepare_observation(ob)
            if goal is not None:
                goal = self._prepare_observation(goal)

            cur_state = ob.to(device=self.device).reshape(1, self.state_dim)
            self.states = torch.cat([self.states, cur_state], dim=0)
            self.actions = torch.cat(
                [self.actions, torch.zeros((1, self.act_dim), device=self.device)],
                dim=0,
            )
            self.timesteps = torch.cat(
                [
                    self.timesteps,
                    torch.ones((1, 1), device=self.device, dtype=torch.long) * t,
                ],
                dim=1,
            )
            ac = self.policy.get_action(self.states, self.actions, self.timesteps)
            self.actions[-1] = ac
        return TensorUtils.to_numpy(ac)
