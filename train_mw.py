import argparse
import json
import pdb
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback
import wandb
import random
import torch
import glob
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from collections import OrderedDict

import h5py
from tqdm import tqdm
import argparse
from models.decision_transformer import DecisionTransformerSeparateState
from torch.utils.data import Dataset, Sampler
from mw_dataset import MWDemoDataset
import hydra
from utils import create_exp_prefix, KEYS_TO_USE, count_parameters
from transformers.adapters.configuration import AdapterConfig


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


def train_single_iteration(
    config,
    model,
    data_loader,
    loss_fn,
    optimizer,
    entropy_loss_fn=None,
    alpha_optimizer=None,
    log_entropy_multiplier=None,
    scheduler=None,
    device="cpu",
):
    # iterate over demos
    for batch in data_loader:

        full_states, actions, returns_to_go, timesteps, attention_mask = (
            batch["states"].to(device).float(),
            batch["actions"].to(device).float(),
            batch["returns_to_go"].to(device).float(),
            batch["timesteps"].to(device).long(),
            batch["attention_mask"].to(device).long(),
        )

        obj_ids = batch["object_indices"].to(device).long()

        action_target = torch.clone(actions)

        _, action_preds, _, action_log_probs, entropies = model.forward(
            full_states,
            actions,
            returns_to_go[:, :-1],
            obj_ids,
            # rewards,
            # rtg[:, :-1],
            timesteps,
            target_actions=action_target,
            attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        loss_fn_inputs = {
            "action_preds": action_preds,
            "action_targets": action_target,
            "a_log_probs": action_log_probs,
            "entropies": entropies,
            "log_entropy_multiplier": log_entropy_multiplier,
        }

        loss = loss_fn(**loss_fn_inputs)

        # update model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        log_dict = {"train/action_loss": loss.item()}

        # update alpha
        if log_entropy_multiplier is not None:
            entropy_multiplier_loss = entropy_loss_fn(entropies)
            alpha_optimizer.zero_grad()
            entropy_multiplier_loss.backward()
            alpha_optimizer.step()

            log_dict["train/entropy_loss"] = entropy_multiplier_loss.item()

        if scheduler is not None:
            scheduler.step()

        if config.log_to_wandb:
            wandb.log(log_dict)


@hydra.main(config_path="configs", config_name="train")
def main(config):
    print(config)

    # first set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # setup env - actually no env because offline learning :D
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # setup model
    model = DecisionTransformerSeparateState(**config.model)

    print("base model params: ", count_parameters(model))

    if config.use_adapters:
        # loading from previous checkpoint
        ckpt_file = sorted(glob.glob(f"{config.model_ckpt_dir}/models/*"))[-1]
        print(f"loading pretrained model from {ckpt_file}")
        state_dict = torch.load(ckpt_file)
        model_config = state_dict["config"]

        model = DecisionTransformerSeparateState(**model_config.model)
        del state_dict["config"]
        del state_dict["epoch"]
        model.load_state_dict(state_dict, strict=True)

        task_name = config.adapter_task_name
        cfg = config.adapter
        cfg = OmegaConf.to_container(cfg)
        cfg["nonlinearity"] = None
        cfg["reduction_factor"] = None
        adapter_config = AdapterConfig.load(**cfg)
        model.transformer.add_adapter(task_name, config=adapter_config)
        # Freeze all model weights except of those of this adapter
        model.transformer.train_adapter([task_name])

        # Set the adapters to be used in every forward pass
        model.transformer.set_active_adapters(task_name)

        print("model params using adapter: ", count_parameters(model))

    model = model.to(device)
    model.train()

    def loss_fn(**kwargs):
        loss = None
        if config.model.stochastic:
            if config.model.use_entropy:
                if config.model.target_entropy:
                    loss = -torch.mean(kwargs["a_log_probs"]) - torch.exp(
                        kwargs["log_entropy_multiplier"].detach()
                    ) * torch.mean(kwargs["entropies"])
                else:
                    loss = -torch.mean(kwargs["a_log_probs"]) - torch.mean(
                        kwargs["entropies"]
                    )
            else:
                loss = -torch.mean(kwargs["a_log_probs"])
        else:
            loss = torch.nn.MSELoss(kwargs["action_preds"], kwargs["action_target"])
        return loss

    if config.model.target_entropy:
        target_entropy = -model.act_dim
        log_entropy_multiplier = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = torch.optim.AdamW(
            [log_entropy_multiplier],
            lr=config.alpha_lr,
            weight_decay=config.alpha_weight_decay,
        )
        entropy_loss_fn = lambda entropies: torch.exp(log_entropy_multiplier) * (
            torch.mean(entropies.detach()) - target_entropy
        )
    else:
        log_entropy_multiplier = None
        alpha_optimizer = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = None

    # setup dataloader
    dataset = MWDemoDataset(config.data)
    print(len(dataset))
    print(dataset[0].keys())

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_data_workers,
        drop_last=False,
    )

    # logging stuff
    group_name = config.exp_name
    exp_prefix = f"{create_exp_prefix(config)}-{random.randint(int(1e5), int(1e6) - 1)}"

    if config.log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project="dt-adapters",
            config=config,
            entity="glamor",
        )
    ckpt_dir = os.path.join(config.output_dir, config.exp_name, exp_prefix, "models")
    os.makedirs(ckpt_dir, exist_ok=True)

    # train loop
    for epoch in tqdm(range(config.num_epochs)):

        if config.log_to_wandb:
            wandb.log({"train/epoch": epoch})

        if epoch % config.eval_every == 0:
            # run evaluation
            pass

        # save model
        if epoch % config.save_every == 0:
            path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            print(f"saving model to {path}")
            save_dict = model.state_dict()
            save_dict["epoch"] = epoch
            save_dict["config"] = config
            torch.save(save_dict, path)

        for step in range(config.num_steps_per_iter):
            loss = train_single_iteration(
                config,
                model,
                data_loader,
                loss_fn,
                optimizer,
                entropy_loss_fn,
                alpha_optimizer,
                log_entropy_multiplier,
                scheduler,
                device,
            )


if __name__ == "__main__":
    main()
