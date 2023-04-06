import os
import sys
import warnings
import gym
import shutil
from gym.logger import set_level

# ignore some tf warnings and gym, they're very annoying
# need to import this before my other modules
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
set_level(40)  # only log errors

import argparse
import json
import time
import sys
import importlib
import wandb
import random
import torch
import glob
import h5py
import yaml
import hydra
import einops
import numpy as np
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from collections import OrderedDict
from omegaconf import OmegaConf
import torch.multiprocessing as mp
import seaborn as sns
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image
from dt_adapters.rollout import run_single_rollout

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler, RandomSampler

from dt_adapters.models.transformer_policy import TransformerPolicy
import dt_adapters.utils as utils
import dt_adapters.eval_utils as eval_utils
import dt_adapters.viz_utils as viz_utils
from dt_adapters.envs.make_env import env_constructor


class Trainer(object):
    """
    Trainer class
    """

    def __init__(self, config):
        utils.set_all_seeds(config.seed)

        self.config = config
        self.start_epoch = 0

        self.ckpt_dir = utils.setup_logging(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.setup_model()
        self.optimizer, self.scheduler = utils.create_optimizer(
            config.optimizer,
            self.model,
            self.config.lr,
            self.config.weight_decay,
            self.config.warmup_steps,
        )

        # setup dataset
        self.setup_dataloader()
        self.loss_fn = torch.nn.MSELoss()

        # setup env for eval
        self.env = env_constructor(
            domain="metaworld",
            task_name=self.config.data.tasks[0],
        )

    def setup_dataloader(self):
        self.dataset = hydra.utils.instantiate(self.config.data, _recursive_=False)

        self.sampler = RandomSampler(self.dataset)

        self.data_loader = DataLoader(
            dataset=self.dataset,
            sampler=self.sampler,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_data_workers,
            drop_last=False,
            persistent_workers=True,
        )

    def setup_model(self):
        # create model
        model = hydra.utils.instantiate(self.config.model, _recursive_=False)
        print("base model params: ", utils.count_parameters(model))

        # put model on device
        self.model = model.to(self.device)
        self.model.train()

        print("final model params: ", utils.count_parameters(model))
        self.model.share_memory()

    def eval(self, epoch):
        print()
        print("*" * 50)
        print("running eval")
        print("*" * 50)
        start = time.time()

        eval_rollouts = []

        for _ in range(self.config.num_eval_rollouts):
            rollout = run_single_rollout(
                self.env,
                "metaworld",
                self.model,
                self.device,
                max_episode_length=self.config.data.max_episode_length,
                save_frames=self.config.log_eval_videos,
                camera_names=self.config.data.camera_names,
            )
            eval_rollouts.append(rollout)

        # compute metrics and log
        metrics = eval_utils.compute_eval_metrics(eval_rollouts)
        print(
            f"task: {self.config.data.tasks[0]}, epoch {epoch}/{self.config.num_epochs} eval metrics:"
        )
        print(
            f"collected {len(eval_rollouts)} rollouts in {time.time() - start} seconds"
        )
        metrics["eval_rollout_time"] = time.time() - start
        pprint(metrics)

        if self.config.log_to_wandb:
            # save metrics and rollout videos to wandb
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

            if self.config.log_eval_videos:
                videos = [traj["images"]["corner"] for traj in eval_rollouts]
                viz_utils.save_videos_to_wandb(
                    videos,
                    task_name=self.config.data.tasks[0],
                    step=epoch,
                    fps=self.config.fps,
                )

    def train_single_epoch(self):
        # iterate over the entire dataset once
        # dataset should consists of (s,a) trajectories
        # the model should be able to predict the next action (a') given the current state

        for batch in self.data_loader:
            start = time.time()

            # put tensors on gpu
            batch = utils.to_device(batch, self.device)

            action_target = torch.clone(batch["actions"]).float()

            # feed inputs to model to get action predictions
            model_out = self.model.forward(**batch)

            mask = batch["attention_mask"].reshape(-1) > 0
            action_preds = model_out["action_preds"]
            action_preds = einops.rearrange(action_preds, "b t d -> (b t) d")[mask]
            action_target = einops.rearrange(action_target, "b t d -> (b t) d")[mask]

            # compute loss between prediction and ground truth actions
            action_pred_loss = self.loss_fn(action_preds, action_target)

            # compute gradients and update the modelparameters
            self.optimizer.zero_grad()
            total_loss = action_pred_loss
            total_loss.backward()
            max_norm, total_norm = utils.compute_gradient_norms(self.model)

            # clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

            log_dict = {
                "train/action_pred_loss": action_pred_loss.item(),
                "train/max_norm": max_norm,
                "train/total_norm": total_norm,
                "train/time_per_iter": time.time() - start,
            }

            # update learning rate if we are using a scheduler
            if self.scheduler is not None and self.config.use_lr_scheduler:
                self.scheduler.step()
                log_dict["train/lr"] = self.scheduler.get_last_lr()[0]

            # log metadata!
            if self.config.log_to_wandb:
                wandb.log(log_dict)

    def train(self):
        # main train loop
        # iterate over the dataset for a fixed number of epochs

        print(f"starting epoch: {self.start_epoch}")

        for epoch in tqdm(range(self.start_epoch, self.config.num_epochs)):

            if self.config.log_to_wandb:
                wandb.log({"train/epoch": epoch})

            # run evaluation to test the policy every eval_every epochs
            if self.config.eval_every > 0 and epoch % self.config.eval_every == 0:
                if self.config.skip_first_eval and epoch == 0:
                    pass
                else:
                    self.eval(epoch)

            # save model
            if epoch % self.config.save_every == 0 and epoch != 0:
                ckpt_file = os.path.join(self.ckpt_dir, "models", f"ckpt_{epoch}.pth")
                metadata = {
                    "epoch": epoch,
                }
                utils.save_model(
                    ckpt_file, self.model, self.optimizer, self.scheduler, metadata
                )

            # iterate over dataset for a number of epochs
            for _ in range(self.config.num_epochs):
                self.train_single_epoch()

        # save everything again at the end
        self.eval(epoch)
        self.save_model(epoch, self.eval_metrics)


@hydra.main(config_path="configs", config_name="base")
def main(config):
    OmegaConf.set_struct(config, False)
    config.update(config.general)
    print("=" * 50)
    print("config:")
    pprint(OmegaConf.to_container(config))
    print("=" * 50)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
