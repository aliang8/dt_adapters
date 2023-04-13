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
from datetime import datetime
from collections import OrderedDict
from omegaconf import OmegaConf
import torch.multiprocessing as mp
import seaborn as sns
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image
from dt_adapters.rollout import run_single_rollout

from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler, RandomSampler

from dt_adapters.models.transformer_policy import TransformerPolicy
import dt_adapters.utils.utils as utils
import dt_adapters.utils.eval_utils as eval_utils
import dt_adapters.utils.viz_utils as viz_utils
import dt_adapters.utils.adapter_utils as adapter_utils
from dt_adapters.envs.make_env import env_constructor


class Trainer(object):
    """
    Trainer class
    """

    def __init__(self, config):
        utils.set_all_seeds(config.seed)

        self.config = config
        self.start_epoch = 0
        self.total_training_iters = 0

        self.ckpt_dir = utils.setup_logging(self.config)

        # if eval from checkpoint file, load the config from the checkpoint file
        if self.config.stage == "eval":
            print("running eval, loading config from experiment config file")
            saved_config_file = os.path.join(
                self.config.log_dir, self.config.exp_name, "config.yaml"
            )
            if os.path.exists(saved_config_file):
                exp_config = OmegaConf.load(saved_config_file)
                self.config.model.update(exp_config.model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.use_adapter = (
            self.config.model.use_single_adapter or self.config.model.use_adapter_fusion
        )

        self.setup_model()
        self.optimizer, self.scheduler = utils.create_optimizer(
            config.optimizer,
            self.model,
            self.config.lr,
            self.config.weight_decay,
            self.config.warmup_steps,
            self.config.use_lr_scheduler,
        )

        # setup dataset
        self.setup_dataloader()
        self.loss_fn = torch.nn.MSELoss()
        self.best_eval_score = -np.inf
        self.best_eval_epoch = 0
        self.eval_metrics = None

        # setup env for eval
        if (
            self.config.stage == "finetune" and self.config.eval_every > 0
        ) or self.config.stage == "eval":
            self.env = env_constructor(
                domain="metaworld",
                task_name=self.config.data.eval_task,
            )

    def setup_dataloader(self):
        if self.config.stage == "pretraining":
            self.config.data.tasks = list(
                (
                    set(self.config.data.all_tasks)
                    - set(self.config.data.adapter_tasks)
                    - set(self.config.data.fusion_tasks)
                )
            )
            print("pretraining tasks: ", self.config.data.tasks)
        else:
            self.config.data.tasks = [self.config.data.eval_task]

        self.dataset = hydra.utils.instantiate(self.config.data, _recursive_=False)

        print(
            f"stage: {self.config.stage}, number of total trajectories in dataset: ",
            len(self.dataset),
        )

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

        # =================================
        # handle adapter stuff here
        # =================================
        adapter_library = adapter_utils.load_adapter_library(
            self.config.model.adapter_library_file
        )

        adapter_name = self.config.model.adapter_task_name

        # load from checkpoint for fine-tuning
        if self.config.load_from_ckpt:
            model, start_epoch = utils.load_model_from_ckpt(
                model,
                self.config,
                self.config.model_ckpt_dir,
                adapter_library,
                strict=False,
            )

            # continue training from a previous checkpoint
            if self.config.resume_experiment:
                self.start_epoch = start_epoch

        if self.config.stage != "pretraining":
            model.freeze_backbone()

        if self.config.stage != "eval":
            # insert new adapters
            if self.config.model.use_single_adapter:
                # create a new adapter for the task
                self.adapter_name = adapter_utils.insert_new_adapter(
                    adapter_library,
                    model,
                    adapter_name,
                    self.config.model.adapter_config.adapter,
                )

            if self.config.model.use_adapter_fusion:
                # create a fusion layer
                self.fusion_name, _ = adapter_utils.insert_new_fusion_layer(
                    adapter_library,
                    model,
                    adapter_name,
                    self.config.model.adapter_config,
                    adapter_keys_to_use=self.config.model.adapter_keys_to_use,
                )

        if self.use_adapter:
            print(model.transformer.adapter_summary())

        # put model on device
        self.model = model.to(self.device)

        if self.config.stage == "eval":
            self.model.eval()
        else:
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
                state_mean=torch.Tensor(self.dataset.state_mean).to(self.device),
                state_std=torch.Tensor(self.dataset.state_std).to(self.device),
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
        self.eval_metrics = metrics
        self.best_eval_score = max(self.best_eval_score, metrics["success_rate"])
        if self.best_eval_score == metrics["success_rate"]:
            self.best_eval_epoch = epoch
        print("best eval score: ", self.best_eval_score)

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

        # save results to json file
        eval_results_file = os.path.join(self.ckpt_dir, "eval_results.txt")
        with open(eval_results_file, "a+") as f:
            f.write(f"epoch {epoch} eval metrics: \n")
            f.write(json.dumps(metrics, indent=4))

    def compute_loss(self, batch, model_out):
        loss_dict = dict()
        mask = batch["attention_mask"].reshape(-1) > 0
        action_preds = model_out["action_preds"]
        action_target = torch.clone(batch["actions"]).float()

        # flatten the first dimension
        action_preds = einops.rearrange(action_preds, "b t d -> (b t) d")[mask]
        action_target = einops.rearrange(action_target, "b t d -> (b t) d")[mask]

        # compute loss between prediction and ground truth actions
        action_pred_loss = self.loss_fn(action_preds, action_target)
        loss_dict["action_pred_loss"] = action_pred_loss

        # for weighted-composition, add an extra entropy loss term to avoid uniform solution
        entropy_loss = 0
        fusion_method = self.config.model.adapter_config.fusion.fusion_method
        if (
            self.config.model.use_adapter_fusion
            and fusion_method == "weighted-composition"
        ):
            attn_dict = model_out["adapter_fusion_attentions"]
            attn_matrix = viz_utils.extract_attn_matrix(attn_dict)

            # want to minimize entropy
            entropy_loss = Categorical(probs=attn_matrix).entropy().mean()
            entropy_loss = (
                self.config.model.adapter_config.fusion.entropy_loss_weight
                * entropy_loss
            )
            loss_dict["entropy_loss"] = entropy_loss

        return loss_dict

    def train_single_iteration(self):
        # iterate over the entire dataset once
        # dataset should consists of (s,a) trajectories
        # the model should be able to predict the next action (a') given the current state

        for batch in self.data_loader:
            start = time.time()

            # put tensors on gpu
            batch = utils.to_device(batch, self.device)

            # feed inputs to model to get action predictions
            model_out = self.model.forward(**batch)

            # returns dictionary of inidivual loss terms
            loss_dict = self.compute_loss(batch, model_out)

            # compute gradients and update the model parameters
            self.optimizer.zero_grad()
            total_loss = sum(list(loss_dict.values()))
            total_loss.backward()
            max_norm, total_norm = utils.compute_gradient_norms(self.model)

            # clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

            log_dict = {
                "train/max_norm": max_norm,
                "train/total_norm": total_norm,
                "train/time_per_iter": time.time() - start,
            }

            log_dict.update({f"train/{k}": v.item() for k, v in loss_dict.items()})

            # update learning rate if we are using a scheduler
            if self.config.use_lr_scheduler:
                self.scheduler.step()
                log_dict["train/lr"] = self.scheduler.get_last_lr()[0]

            # log metadata!
            if self.config.log_to_wandb:
                # if we are using adapters, visualize fusion weights
                if (
                    self.config.model.use_adapter_fusion
                    and "adapter_fusion_attentions" in model_out
                ):
                    x_labels = (
                        self.config.model.adapter_config.adapters_to_use
                        + self.config.data.eval_task
                    )
                    # the adapter fusion attention is of size [bs, seq_len, n_tasks] for bert-fusion
                    # it should be of size [n_tasks] for weighted-composition
                    heatmap = viz_utils.visualize_fusion_attention(
                        self.config.model.adapter_config.fusion.fusion_method,
                        model_out["adapter_fusion_attentions"],
                        adapters_to_use=x_labels,
                    )
                    log_dict["train/fusion_attention_map"] = wandb.Image(heatmap)

                # log general metadata
                wandb.log(log_dict)

    def save_model(self, epoch):
        if self.use_adapter:
            model_ckpt_dir = os.path.join(self.ckpt_dir, "models", f"epoch_{epoch:04d}")
            os.makedirs(model_ckpt_dir, exist_ok=True)

            now = datetime.now()

            metadata = {
                "epoch": epoch,
                "total_training_iters": self.total_training_iters,
                "best_eval_score": self.best_eval_score,
                "best_eval_epoch": self.best_eval_epoch,
                "ckpt_path": str(model_ckpt_dir),
                "exp_name": self.config.exp_name,
                "time": now.strftime("%d/%m/%Y %H:%M:%S"),
                "seed": self.config.seed,
            }

            new_adapter_name = self.config.model.adapter_task_name

            # use HF util to save adapters
            adapter_utils.save_adapters(
                self.model,
                model_ckpt_dir,
                self.config.model.use_adapter_fusion,
                self.fusion_name,
                metadata,
            )
            # update the library of adapters
            adapter_utils.update_adapter_library(
                self.config.model.adapter_library_file,
                self.adapter_name,
                model_ckpt_dir,
                metadata,
            )
        else:
            model_ckpt_file = os.path.join(
                self.ckpt_dir, "models", f"ckpt_{epoch:04d}.pth"
            )

            metadata = {
                "epoch": epoch,
                "config": self.config,
                "total_training_iters": self.total_training_iters,
            }
            utils.save_model(
                model_ckpt_file, self.model, self.optimizer, self.scheduler, metadata
            )

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
                self.save_model(epoch)

            # iterate over dataset for a number of epochs
            for _ in range(self.config.num_iterations_per_epoch):
                self.train_single_iteration()
                self.total_training_iters += 1

        # run one last evaluation
        print("final evaluation...")
        if self.config.stage == "finetune":
            self.eval(epoch)

        # save the model one last time
        self.save_model(epoch)


@hydra.main(config_path="configs", config_name="base")
def main(config):
    OmegaConf.set_struct(config, False)
    config.update(config.general)
    print("=" * 50)
    print("config:")
    pprint(OmegaConf.to_container(config))
    print("=" * 50)

    trainer = Trainer(config)
    if config.stage == "eval":
        trainer.eval(0)
    else:
        trainer.train()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
