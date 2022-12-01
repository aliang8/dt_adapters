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
import numpy as np
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from collections import OrderedDict
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from functools import partial


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
from dt_adapters.rollout import rollout, mp_rollout
import dt_adapters.constants as constants


from transformers.adapters.configuration import AdapterConfig
import transformers.adapters.composition as ac

from garage.envs import GymEnv
from garage.np import discount_cumsum, stack_tensor_dict_list


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer(object):
    """
    Trainer class
    """

    def __init__(self, config):
        set_all_seeds(config.seed)

        self.config = config
        self.total_training_iters = 0
        self.best_eval_perf = -np.inf
        self.patience = self.config.patience
        self.eval_metrics = None
        self.start_epoch = 0

        self.setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.setup_model_and_optimizer()

        # setup dataset
        start = time.time()
        self.dataset = DemoDataset(self.config.data, stage=config.stage)
        self.dataset[0]
        print(f"dataset len: {len(self.dataset)}")
        print(f"stage: {config.stage}, took {time.time() - start} seconds to load data")

        self.setup_dataloader()

        def loss_fn(**kwargs):
            action_pred_loss = torch.nn.MSELoss()(
                kwargs["action_preds"].float(), kwargs["action_targets"].float()
            )
            return action_pred_loss

        self.loss_fn = loss_fn

    def setup_logging(self):
        self.ckpt_dir = Path(os.path.join(os.environ["LOG_DIR"], self.config.exp_name))
        if self.config.log_outputs:
            # check if this experiment already exists
            if os.path.exists(self.ckpt_dir) and not self.config.resume_experiment:
                overwrite = input(f"{self.ckpt_dir} already exists, overwrite [y/n]")

                if overwrite == "y":
                    shutil.rmtree(self.ckpt_dir)
                    os.makedirs(self.ckpt_dir, exist_ok=True)
                    os.makedirs(self.ckpt_dir / "models", exist_ok=True)
                else:
                    sys.exit()
            else:
                os.makedirs(self.ckpt_dir, exist_ok=True)
                os.makedirs(self.ckpt_dir / "models", exist_ok=True)

        if self.config.log_to_wandb:
            wandb.init(
                name=self.config.exp_name,
                # group=group_name,
                project=self.config.project_name,
                config=OmegaConf.to_container(self.config),
                entity="glamor",
            )

    def setup_dataloader(self):
        self.sampler = RandomSampler(self.dataset)

        self.data_loader = DataLoader(
            dataset=self.dataset,
            sampler=self.sampler,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_data_workers,
            drop_last=False,
            persistent_workers=True,
        )

    def save_videos(self, videos, key):
        video_array = mw_utils.create_video_grid(
            videos,
            height=self.config.image_height,
            width=self.config.image_width,
        )
        wandb.log(
            {
                f"eval/{self.config.data.eval_task}/{key}/videos": wandb.Video(
                    video_array,
                    caption=f"train_iter_{self.total_training_iters}",
                    fps=self.config.fps,
                    format="gif",
                )
            }
        )

    def eval(self, epoch):
        print("running eval")
        start = time.time()

        self.model.eval()
        log_eval_videos = (
            epoch % self.config.log_eval_videos_every == 0 or self.config.mode == "eval"
        ) and self.config.log_eval_videos

        eval_rollouts = mp_rollout(
            self.config,
            self.model,
            use_means=True,
            attend_to_rtg=False,
            log_eval_videos=log_eval_videos,
        )

        # compute metrics and log
        metrics = eval_utils.compute_eval_metrics(eval_rollouts)
        metrics["epoch"] = epoch
        metrics["total_training_iters"] = self.total_training_iters

        if self.config.log_to_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

            if log_eval_videos:
                videos = [traj["frames"] for traj in eval_rollouts]
                self.save_videos(videos, key="main")

        print("=" * 50)
        print(
            f"task: {self.config.data.eval_task}, epoch {epoch}/{self.config.num_epochs} eval out of {self.config.num_eval_rollouts} episodes"
        )
        print(
            f"collected {len(eval_rollouts)} rollouts in {time.time() - start} seconds"
        )
        pprint(metrics)
        print("=" * 50)
        return metrics

    def save_model(self, epoch, metrics):
        if self.config.log_outputs:
            if (
                hasattr(self, "eval_metrics")
                and self.eval_metrics
                and self.config.save_best_model
            ):
                if metrics["success_rate"] > self.best_eval_perf:
                    print(
                        f"saving model to {self.ckpt_dir}, new best eval: {metrics['success_rate']} "
                    )
                    self.best_eval_perf = metrics["success_rate"]
                else:
                    print(
                        f"model eval perf: {metrics['success_rate']}, previous best: {self.best_eval_perf}, not saving"
                    )
                    return

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

                general_utils.update_adapter_hub(
                    adapter_name=self.config.data.eval_task,
                    ckpt_dir=self.ckpt_dir,
                    epoch=epoch,
                    best_perf=self.best_eval_perf,
                )
            else:
                metadata = {
                    "ckpt_dir": self.ckpt_dir,
                    "epoch": epoch,
                    "config": self.config,
                }
                path = os.path.join(self.ckpt_dir, "models", f"epoch_{epoch:03d}.pt")
                general_utils.save_model(
                    path, self.model, self.optimizer, self.scheduler, metadata
                )

    def setup_model_and_optimizer(self):
        if self.config.model.model_cls == "transformer":
            model_cls = TransformerPolicy
        elif self.config.model.model_cls == "mlp_policy":
            model_cls = MLPPolicy

        model = model_cls(self.config.model)
        model = model.to(self.device)
        model.train()
        self.model = model

        # setup optimizer
        if self.config.optimizer == "adam":
            optim_cls = torch.optim.Adam
        elif self.config.optimizer == "adamw":
            optim_cls = torch.optim.AdamW
        else:
            raise NotImplementedError()

        self.optimizer = optim_cls(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        if self.config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda steps: min((steps + 1) / self.config.warmup_steps, 1),
            )
        else:
            self.scheduler = None

        if self.config.load_from_ckpt:
            model, self.start_epoch = general_utils.load_model_from_ckpt(
                model,
                self.config,
                self.config.model_ckpt_dir,
                self.optimizer,
                self.scheduler,
            )

        print("base model params: ", general_utils.count_parameters(model))

        if self.config.model.use_adapters:
            # Look at what trained adapters are already available
            with open(constants.HUB_FILE, "r") as f:
                try:
                    adapter_info = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    print(exc)

                single_task_adapters = adapter_info["single_task_adapters"]
                print("-" * 50)
                st_library = {a["name"]: a["ckpt_path"] for a in single_task_adapters}
                print(f"{len(st_library)} adapters available: ")
                pprint(st_library)
                print("-" * 50)

            task_name = self.config.model.adapter_task_name
            cfg = self.config.model.adapter
            cfg = OmegaConf.to_container(cfg)
            cfg["nonlinearity"] = None
            cfg["reduction_factor"] = None

            if self.config.model.use_adapter_fusion:
                # For now we only train a new fusion layer
                # maybe consider training a new adapter in addition to fusion
                adapters_to_use = OmegaConf.to_container(
                    self.config.model.adapters_to_use
                )

                # load adapters to use
                print("Loading adapter weights...")
                print(f"Adapters to use: {adapters_to_use}")

                # check that all adapters exist
                for adapter_name in adapters_to_use:
                    if not adapter_name in st_library:
                        raise Exception("{adapter_name} not a valid adapter")

                for adapter_name in adapters_to_use:
                    adapter_ckpt_path = st_library[adapter_name]
                    print(f"Loading {adapter_name} from {adapter_ckpt_path}")
                    adapter_name = model.transformer.load_adapter(adapter_ckpt_path)

                model.transformer.add_adapter_fusion(adapters_to_use)

                # set the fusion layer as active
                fusion_layer = ac.Fuse(*adapters_to_use)
                model.transformer.set_active_adapters(fusion_layer)

                # make sure all the other weights are frozen except fusion layer
                model.transformer.train_adapter_fusion(fusion_layer)
            else:
                if task_name in st_library:
                    print(
                        f"Trained adapter already exists for: {task_name}, will be overwriting."
                    )

                print(f"Training new adapter for: {task_name}")

                # train a new set of adapter weights
                adapter_config = AdapterConfig.load(**cfg)
                model.transformer.add_adapter(task_name, config=adapter_config)

                # freeze all model weights except of those of this adapter
                model.transformer.train_adapter([task_name])

                # set the adapters to be used in every forward pass
                model.transformer.set_active_adapters(task_name)

        if self.config.freeze_backbone:
            model.freeze_backbone()

        print(self.model)

        print("final model params: ", general_utils.count_parameters(model))
        self.model.share_memory()

    def train_single_iteration(self):
        # iterate over dataset
        for batch in self.data_loader:
            start = time.time()
            # put tensors on gpu
            batch = general_utils.to_device(batch, self.device)

            action_target = torch.clone(batch["actions"])

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
            }

            action_pred_loss = self.loss_fn(**loss_fn_inputs)

            # update model
            self.optimizer.zero_grad()
            total_loss = action_pred_loss
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
                "train/max_norm": max_norm,
                "train/total_norm": total_norm,
            }

            # update learning rates
            if self.scheduler is not None:
                self.scheduler.step()
                log_dict["train/lr"] = self.scheduler.get_last_lr()[0]

            log_dict["time_per_iter"] = time.time() - start

            if self.config.log_to_wandb:
                wandb.log(log_dict)

    def train(self):
        # train loop
        print(f"starting epoch: {self.start_epoch}")

        for epoch in tqdm(range(self.start_epoch, self.config.num_epochs)):

            if self.config.log_to_wandb:
                wandb.log({"train/epoch": epoch})

            # run evaluation
            if self.config.eval_every > 0 and epoch % self.config.eval_every == 0:
                if self.config.skip_first_eval and epoch == 0:
                    pass
                else:
                    self.eval_metrics = self.eval(epoch)

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
                self.save_model(epoch, self.eval_metrics)

            # iterate for a number of rollouts
            for _ in range(self.config.num_steps_per_epoch):
                self.train_single_iteration()
                self.total_training_iters += 1

        # save very last epoch
        self.save_model(epoch, self.eval_metrics)


@hydra.main(config_path="configs", config_name="train")
def main(config):
    OmegaConf.set_struct(config, False)
    config.update(config.general)
    print("=" * 50)
    print("config:")
    pprint(OmegaConf.to_container(config))
    print("=" * 50)

    trainer = Trainer(config)

    if config.mode == "train":
        trainer.train()
    else:
        trainer.eval(0)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
