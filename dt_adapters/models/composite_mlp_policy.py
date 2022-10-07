"""
Implementation of composing task-agnostic policies: https://arxiv.org/pdf/1905.10681.pdf 
"""
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp_policy import MLPPolicy


class CompositeMLPPolicy(nn.Module):
    def __init__(
        self,
        base_model_ckpt_dir,
        primitive_policy_names,
        state_dim,
        act_dim,
        hidden_size,
        num_enc_layers=1,
        dropout=0.1,
        num_attn_heads=4,
        **kwargs,
    ):
        # policy ensemble composition

        # takes state and primitive policy outputs to compute
        # composite action

        self.base_model_ckpt_dir = base_model_ckpt_dir
        self.primitive_policy_names = primitive_policy_names
        self.num_enc_layers = num_enc_layers
        self.hidden_size = hidden_size

        # load in a bunch of primitive policies
        self.primitive_policies = self.load_primitive_polices()

        # encoder takes policy outputs
        # and transforms them into latent states
        self.encoder = nn.GRU(
            input_size=act_dim,
            hidden_size=hidden_size,
            num_layers=num_enc_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # takes hidden states of forward + backward net
        # + current state => hidden state
        self.decoder = nn.Linear(hidden_size * 3, hidden_size)

        self.attention_layer = torch.nn.MultiheadAttention(
            hidden_size, num_heads=num_attn_heads, dropout=dropout, batch_first=True
        )

    def load_primitive_polices(self):
        primitive_policies = []
        for i, model_name in enumerate(self.primitive_policy_names):
            ckpt_file = sorted(
                glob.glob(f"{self.base_model_ckpt_dir}/{model_name}/models/*")
            )[-1]
            print(f"loading pretrained model #{i} from {ckpt_file}")
            state_dict = torch.load(ckpt_file)
            model_config = state_dict["config"]

            model = MLPPolicy(**model_config.model)
            primitive_policies.append(model)

        return primitive_policies

    def forward(self, states, obj_ids, **kwargs):
        batch_size, seq_length = states.shape[0], states.shape[1]
        h_0 = torch.zeros(2 * self.num_enc_layers, batch_size, self.hidden_size)

        # forward pass to get each actions from primitive policies
        policy_outputs = []
        for policy in self.primitive_policies:
            policy_outputs.append(policy(states, obj_ids))

        # B x num_policies x act_dim
        policy_outputs = torch.stack(policy_outputs, dim=1)

        # output - shape B x L x 2 * H
        output, _ = self.encoder(policy_outputs, h_0)

        final_hidden_forward = output[:, -1, 0].unsqueeze(1).repeat(1, seq_length, 1)
        final_hidden_backward = output[:, -1, 1].unsqueeze(1).repeat(1, seq_length, 1)

        # states - B x L x H
        # B x L x (H + H + H)
        decoder_input = torch.cat(
            [states, final_hidden_forward, final_hidden_backward], dim=1
        )
        decoder_output = self.decoder(decoder_input)
        # B x L x H

        query, key, value = final_hidden_forward, final_hidden_backward, decoder_output
        attn_output, attn_output_weights = self.attention_layer(query, key, value)

        composite_action = attn_output_weights * policy_outputs
        return composite_action, None, _

    def reset(self):
        pass

    def get_action(self, states, obj_ids, **kwargs):
        # only take last state
        states = states[-1].reshape(1, 1, self.state_dim)
        predictions = self.forward(states, obj_ids)
        action_pred = predictions[1].squeeze()
        return action_pred, None, {}
