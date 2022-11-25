import os
import warnings
import gym
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
import numpy as np
from pprint import pprint
from tqdm import tqdm
from collections import OrderedDict
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from functools import partial
from pathlib import Path


from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler, RandomSampler


from dt_adapters.data.demo_dataset import DemoDataset
from dt_adapters.models.transformer_policy import TransformerPolicy
from dt_adapters.models.mlp_policy import MLPPolicy
from dt_adapters.sampler import ImportanceWeightBatchSampler
import dt_adapters.general_utils as general_utils
import dt_adapters.mw_utils as mw_utils
import dt_adapters.eval_utils as eval_utils
from dt_adapters.data.utils import get_visual_encoders
from dt_adapters.rollout import rollout
import dt_adapters.constants as constants


from transformers.adapters.configuration import AdapterConfig
import transformers.adapters.composition as ac

from garage.envs import GymEnv
from garage.np import discount_cumsum, stack_tensor_dict_list
from dt_adapters.trainer import Trainer


class CLTrainer(Trainer):
    """
    Trainer class
    """

    def __init__(self, config):
        super().__init__(config)
        self.current_task = None
        self.cl_eval_metrics = {}

    def eval(self, epoch, task=None):
        print("running eval")
        log_eval_videos = (
            epoch % self.config.log_eval_videos_every == 0 or self.config.mode == "eval"
        )
        attend_to_rtg = True if self.config.online_training else False
        task = task if task is not None else self.config.data.eval_task
        eval_rollouts = self.mp_rollout(
            task,
            self.config.num_processes,
            self.config.num_eval_rollouts,
            use_means=True,
            attend_to_rtg=attend_to_rtg,
            log_eval_videos=log_eval_videos,
        )
        eval_rollouts = [self.update_path(path) for path in eval_rollouts]

        # compute metrics and log
        metrics = eval_utils.compute_eval_metrics(eval_rollouts)
        metrics["epoch"] = epoch
        metrics["total_training_iters"] = self.total_training_iters
        metrics["total_online_rollouts"] = self.total_online_rollouts

        if self.config.log_to_wandb:
            wandb.log({f"{self.current_task}/eval/{k}": v for k, v in metrics.items()})

            if log_eval_videos:
                frame_keys = list(eval_rollouts[0]["frames"].keys())
                for key in frame_keys:
                    videos = [traj["frames"][key] for traj in eval_rollouts]
                    self.save_videos(videos, key=key)

        print("=" * 50)
        print(
            f"task: {self.config.data.eval_task}, epoch {epoch} eval out of {self.config.num_eval_rollouts} episodes"
        )
        pprint(metrics)
        print("=" * 50)
        self.eval_metrics = metrics

    def save_model(self, epoch):
        if self.config.log_outputs:
            if self.config.model.use_adapters:
                if self.config.model.use_adapter_fusion:
                    # save the fusion layer
                    self.model.save_adapter_fusion(
                        self.ckpt_dir, self.config.model.adapters_to_use
                    )
                    self.model.save_all_adapters(self.ckpt_dir)
                else:
                    # save just the adapter weights
                    self.model.transformer.save_adapter(
                        self.ckpt_dir,
                        self.config.data.eval_task,
                        meta_dict={"epoch": epoch, "config": self.config},
                    )

                # log adapter info to hub
                with open(constants.HUB_FILE, "r") as f:
                    try:
                        adapter_info = yaml.safe_load(f)
                    except yaml.YAMLError as exc:
                        print(exc)

                    key = (
                        "single_task_adapters"
                        if not self.config.model.use_adapters
                        else "adapter_fusion_layers"
                    )

                    new_adapter = {
                        "name": self.config.data.eval_task,
                        "ckpt_path": self.ckpt_dir,
                        "epoch": epoch,
                        "best_success_rate": self.best_eval_perf,
                    }

                    names = [a["name"] for a in adapter_info[key]]
                    index = names.index(self.config.data.eval_task)

                    # insert new adapter into library
                    if index == -1:
                        adapter_info[key].append(new_adapter)
                    else:
                        adapter_info[key][index] = new_adapter

                with open(constants.HUB_FILE, "w") as f:
                    yaml.safe_dump(adapter_info, f)
            else:
                path = os.path.join(self.ckpt_dir, f"{self.current_task}_latest.pt")
                print(f"saving model to {path}")
                save_dict = self.model.state_dict()
                save_dict["epoch"] = epoch
                save_dict["config"] = self.config
                torch.save(save_dict, path)

                # also save the one with best performance
                if self.eval_metrics["success_rate"] > self.best_eval_perf:
                    print("saving new best model, {self.eval_metrics['success_rate']}")
                    path = os.path.join(
                        self.ckpt_dir, f"{self.current_task}_best_eval.pt"
                    )
                    torch.save(save_dict, path)

    def train_single_iteration(self):
        # iterate over dataset
        for batch in self.data_loader:
            start = time.time()
            # put tensors on gpu
            batch = general_utils.to_device(batch, self.device)
            action_target = torch.clone(batch["actions"])

            if "online" in batch:
                batch["use_rtg_mask"] = batch["online"].reshape(-1, 1)

            model_out = self.model.forward(
                **batch,
                target_actions=action_target,
                use_means=False,  # sample during training
            )

            act_dim = model_out["action_preds"].shape[2]
            mask = batch["attention_mask"].reshape(-1) > 0
            action_preds = model_out["action_preds"].reshape(-1, act_dim)[mask]
            action_target = action_target.reshape(-1, act_dim)[mask]

            loss_fn_inputs = {
                "action_preds": action_preds,
                "action_targets": action_target,
                # "return_preds": return_preds,
                # "return_targets": return_target,
            }

            action_pred_loss, return_pred_loss = self.loss_fn(**loss_fn_inputs)

            # update model
            self.optimizer.zero_grad()
            total_loss = action_pred_loss + return_pred_loss
            total_loss.backward()

            model_params = [
                p
                for p in self.model.parameters()
                if p.requires_grad and p.grad is not None
            ]

            # compute norms
            max_norm = max([p.grad.detach().abs().max() for p in model_params])
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model_params]),
                2.0,
            ).item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

            log_dict = {
                "train/action_pred_loss": action_pred_loss.item(),
                "train/return_pred_loss": return_pred_loss.item(),
                "train/max_norm": max_norm,
                "train/total_norm": total_norm,
            }

            # update learning rates
            if self.scheduler is not None:
                self.scheduler.step()
                log_dict["train/lr"] = self.scheduler.get_last_lr()[0]

            if self.alpha_scheduler is not None:
                self.alpha_scheduler.step()
                log_dict["train/alpha_lr"] = self.alpha_scheduler.get_last_lr()[0]

            log_dict["time_per_iter"] = time.time() - start

            # add prefix
            log_dict_w_prefix = {}
            for k, v in log_dict.items():
                log_dict_w_prefix[f"{self.current_task}/{k}"] = v

            if self.config.log_to_wandb:
                wandb.log(log_dict_w_prefix)

    def train_single_task(self):
        epoch = 0

        # train loop
        for epoch in tqdm(range(self.config.num_epochs)):

            if self.config.log_to_wandb:
                wandb.log({f"{self.current_task}/train/epoch": epoch})

            # run evaluation for online_training
            if self.config.eval_every > 0 and epoch % self.config.eval_every == 0:
                if self.config.skip_first_eval and epoch == 0:
                    pass
                else:
                    self.eval(epoch)

            if (
                self.config.early_stopping
                and self.eval_metrics[self.config.early_stopping_metric]
                < self.best_eval_perf
            ):
                # early stopping
                if self.patience >= self.config.patience:
                    print(f"patience {self.config.patience} reached, early stopping")
                    break
                else:
                    self.patience += 1
                    print(f"performance did not improve, patience: {self.patience}")
            else:  # resetting the patience
                self.patience = self.config.patience

            # save model
            if epoch % self.config.save_every == 0 and epoch != 0:
                self.save_model(epoch)

            # update metrics
            if hasattr(self, "eval_metrics"):
                if self.eval_metrics["success_rate"] > self.best_eval_perf:
                    print(f"new best eval: {self.eval_metrics['success_rate']} ")
                    self.best_eval_perf = self.eval_metrics["success_rate"]
                else:
                    print(
                        f"model eval perf: {self.eval_metrics['success_rate']}, previous best: {self.best_eval_perf}"
                    )

            for _ in range(self.config.num_steps_per_epoch):
                self.train_single_iteration()
                self.total_training_iters += 1

        # run last evaluation
        self.eval(epoch, self.current_task)

        # save very last epoch
        self.save_model(epoch)

    def load_model_from_ckpt(self):
        import ipdb

        ipdb.set_trace()
        if self.config.model.model_cls == "transformer":
            model_cls = TransformerPolicy
        elif self.config.model.model_cls == "mlp_policy":
            model_cls = MLPPolicy

        if self.config.pretrained_mdl_ckpt_dir:
            ckpt_file = sorted(
                glob.glob(f"{self.config.pretrained_mdl_ckpt_dir}/models/*")
            )[-1]
            state_dict = torch.load(ckpt_file)
            model_config = state_dict["config"]

            model = model_cls(model_config.model)
            print(
                f"loading pretrained model from {self.config.pretrained_mdl_ckpt_dir}, epoch: {state_dict['epoch']}"
            )

            del state_dict["config"]
            del state_dict["epoch"]
            model.load_state_dict(state_dict, strict=True)
            model.config.update(self.config.model)
            self.config.batch_size = model_config["batch_size"]
        else:
            model = model_cls(model_config.model)

        # if self.config.load_from_ckpt and self.config.resume_experiment:
        #     import ipdb

        #     ipdb.set_trace()
        #     # loading from previous checkpoint

        #     # figure out which was the last task that had saved checkpoints
        #     ckpt_files = sorted(glob.glob(f"{self.config.model_ckpt_dir}/models/*"))

        #     for i, task in enumerate(self.config.data.cl_tasks):
        #         found = False
        #         for ckpt_file in ckpt_files:
        #             if task in ckpt_file:
        #                 found = True
        #         if not found:
        #             self.current_task = task
        #             break

        #     print(f"finished learning up to {self.current_task}, continuing from here")

        #     # load checkpoint of previous task

        print("base model params: ", general_utils.count_parameters(model))
        return model

    def setup_logging(self):
        if (
            self.config.load_from_ckpt and self.config.resume_experiment
        ):  # continue the same plot
            path = Path(self.config.model_ckpt_dir)
            exp_prefix = path.parts[-1]
            group_name = path.parts[-2]
        else:
            exp_prefix = f"exp-{random.randint(int(1e5), int(1e6) - 1)}"
            group_name = self.config.exp_name

        if self.config.log_to_wandb:
            wandb.init(
                name=exp_prefix,
                group=group_name,
                project="dt-adapters",
                config=OmegaConf.to_container(self.config),
                entity="glamor",
            )

        if self.config.log_outputs:
            self.ckpt_dir = os.path.join(
                os.environ["LOG_DIR"], self.config.exp_name, exp_prefix, "models"
            )
            print(f"saving outputs to: {self.ckpt_dir}")
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self):
        cl_tasks = self.config.data.cl_tasks

        print("started continual learning training")
        start = time.time()

        # iterate through all the continual learning tasks
        for i, task in enumerate(cl_tasks):
            # reset dataset and dataloader
            self.current_task = task
            self.config.data.eval_task = task

            print(f"cl task #{i}, training model for {task}")

            # load demos for this specific task
            self.dataset = DemoDataset(self.config.data, stage=self.config.stage)

            self.setup_dataloader()
            self.total_training_iters = 0
            self.best_eval_perf = -np.inf
            self.patience = self.config.patience
            self.eval_metrics = None

            # reset optimizer
            self.setup_optimizer()

            # train model on the current CL task
            self.train_single_task()

            # look at final eval_metrics
            self.cl_eval_metrics[task] = self.eval_metrics

            print(
                f"cl task #{i}, took {time.time() - start} seconds to finish training {task}"
            )
            start = time.time()


@hydra.main(config_path="configs", config_name="train")
def main(config):
    OmegaConf.set_struct(config, False)
    config.update(config.general)
    print("=" * 50)
    print("config:")
    pprint(OmegaConf.to_container(config))
    print("=" * 50)

    trainer = CLTrainer(config)

    if config.mode == "train":
        trainer.train()
    else:
        trainer.eval(0)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
