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
from torch.utils.data import DataLoader
from collections import OrderedDict

import h5py
from tqdm import tqdm
import argparse
from models.decision_transformer import DecisionTransformerSeparateState
from torch.utils.data import Dataset, Sampler
from mw_dataset import MWDemoDataset
import hydra
from utils import create_exp_prefix, KEYS_TO_USE


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


def train_single_iteration(
    config, model, data_loader, loss_fn, optimizer, scheduler, device
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

        state_preds, action_preds, return_preds = model.forward(
            full_states,
            actions,
            returns_to_go[:, :-1],
            obj_ids,
            # rewards,
            # rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        loss = loss_fn(action_preds, action_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if config.log_to_wandb:
            wandb.log({"train/action_loss": loss.item()})


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
    model = model.to(device)
    model.train()

    loss_fn = torch.nn.MSELoss()

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
                config, model, data_loader, loss_fn, optimizer, scheduler, device
            )


if __name__ == "__main__":
    main()
