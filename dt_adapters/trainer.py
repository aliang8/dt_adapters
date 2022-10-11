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
import hydra
import numpy as np
from pprint import pprint
from tqdm import tqdm
from collections import OrderedDict
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from functools import partial


from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler, RandomSampler


from dt_adapters.data.demo_dataset import DemoDataset
from dt_adapters.models.decision_transformer import DecisionTransformerSeparateState
from dt_adapters.models.mlp_policy import MLPPolicy
from dt_adapters.sampler import ImportanceWeightBatchSampler
import dt_adapters.general_utils as general_utils
import dt_adapters.mw_utils as mw_utils
import dt_adapters.eval_utils as eval_utils
from dt_adapters.data.utils import get_visual_encoders
from dt_adapters.rollout import rollout


from transformers.adapters.configuration import AdapterConfig

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
        self.setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.setup_model()

        # setup dataset
        start = time.time()
        self.dataset = DemoDataset(self.config.data, stage=config.stage)
        print(f"dataset len: {len(self.dataset)}")
        print(f"stage: {config.stage}, took {time.time() - start} seconds to load data")

        self.setup_dataloader()
        self.setup_optimizer()
        self.total_training_iters = 0
        self.total_online_rollouts = 0

        def loss_fn(**kwargs):
            action_pred_loss = torch.tensor(0.0)
            if config.model.stochastic:
                if config.model.use_entropy:
                    if config.model.target_entropy:
                        action_pred_loss = -torch.mean(
                            kwargs["a_log_probs"]
                        ) - torch.exp(kwargs["alpha"].detach()) * torch.mean(
                            kwargs["entropies"]
                        )
                    else:
                        action_pred_loss = -torch.mean(
                            kwargs["a_log_probs"]
                        ) - torch.mean(kwargs["entropies"])
                else:
                    action_pred_loss = -torch.mean(kwargs["a_log_probs"])
            else:
                action_pred_loss = torch.nn.MSELoss()(
                    kwargs["action_preds"].float(), kwargs["action_targets"].float()
                )

            if config.model.predict_return_dist:
                # need to first bin the target returns via quantization
                # scale = (config.model.max_return - 0) / (num_bins - 0)
                bin_width = config.model.bin_width
                num_bins = int(config.model.max_return / bin_width)
                scale = bin_width
                zero_point = 0

                binned_targets = torch.quantize_per_tensor(
                    kwargs["return_targets"].float(),
                    scale=scale,
                    zero_point=zero_point,
                    dtype=torch.quint8,
                ).int_repr()

                # use crossentropy to minimize prediction and gt return dist
                return_pred_loss = torch.nn.CrossEntropyLoss()(
                    kwargs["return_preds"].reshape(-1, num_bins),
                    binned_targets.reshape(-1),
                )
            else:
                return_pred_loss = torch.tensor(0.0)

            return action_pred_loss, return_pred_loss

        self.loss_fn = loss_fn

    def warmup_data_collection(self):
        """
        Collect online rollouts for bootstrapping the replay buffer.
        """
        print("collecting warmup trajectories to fill replay buffer...")
        start = time.time()
        for _ in tqdm(range(self.config.num_warmup_rollouts)):
            # don't attend to Returns for warmup because we are using pretrained model
            path = self.mp_rollout(
                num_processes=1,
                num_rollouts=1,
                use_means=True,
                attend_to_rtg=False,
                log_eval_videos=False,
            )[0]
            new_trajectory = self.update_path(path)
            self.dataset.trajectories.append(new_trajectory)
        print(f"took {time.time() - start} seconds for warmup collection")

        # info about initial dataset
        print("initial dataset size: ", len(self.dataset))
        metrics = eval_utils.compute_eval_metrics(self.dataset.trajectories)
        pprint(metrics)

    def setup_logging(self):
        group_name = self.config.exp_name
        exp_prefix = f"exp-{random.randint(int(1e5), int(1e6) - 1)}"
        if self.config.stage == "finetuning":
            exp_prefix = self.config.data.eval_task + exp_prefix

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

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / self.config.warmup_steps, 1)
        )

        # auto-tune entropy term
        if self.config.model.use_entropy and self.config.model.target_entropy:
            target_entropy = -self.model.act_dim

            self.alpha = torch.zeros(1, requires_grad=True, device=self.device)

            self.alpha_optimizer = torch.optim.AdamW(
                [self.alpha],
                lr=self.config.alpha_lr,
                weight_decay=self.config.alpha_weight_decay,
            )

            self.entropy_loss_fn = lambda entropies: torch.exp(self.alpha) * (
                torch.mean(entropies.detach()) - target_entropy
            )

            self.alpha_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.alpha_optimizer,
                lambda steps: min((steps + 1) / self.config.warmup_steps, 1),
            )
        else:
            self.alpha = None
            self.alpha_optimizer = None
            self.alpha_scheduler = None
            self.entropy_loss_fn = None

    def setup_dataloader(self):
        if self.config.online_training:
            # create batch sampler to rerank trajectories based on length during training
            self.sampler = ImportanceWeightBatchSampler(
                self.dataset, self.config.batch_size, shuffle=True
            )
        else:
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
        log_eval_videos = (
            epoch % self.config.log_eval_videos_every == 0 or self.config.mode == "eval"
        )
        attend_to_rtg = True if self.config.online_training else False
        eval_rollouts = self.mp_rollout(
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
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

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

    def save_model(self, epoch):
        if self.config.log_outputs:
            if self.config.model.use_adapters:
                # save just the adapter weights
                self.model.transformer.save_adapter(
                    self.ckpt_dir,
                    self.config.data.eval_task,
                    meta_dict={"epoch": epoch, "config": self.config},
                )

            path = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}.pt")
            print(f"saving model to {path}")
            save_dict = self.model.state_dict()
            save_dict["epoch"] = epoch
            save_dict["config"] = self.config
            torch.save(save_dict, path)

    def setup_model(self):
        if self.config.model.model_cls == "decision_transformer":
            model_cls = DecisionTransformerSeparateState
        elif self.config.model.model_cls == "mlp_policy":
            model_cls = MLPPolicy

        if self.config.model_ckpt_dir and self.config.load_from_ckpt:
            # loading from previous checkpoint
            ckpt_file = sorted(glob.glob(f"{self.config.model_ckpt_dir}/models/*"))[-1]
            print(f"loading pretrained model from {ckpt_file}")
            state_dict = torch.load(ckpt_file)
            model_config = state_dict["config"]

            model = model_cls(model_config.model)
            del state_dict["config"]
            del state_dict["epoch"]
            model.load_state_dict(state_dict, strict=True)
            self.config.batch_size = model_config["batch_size"]
        else:
            model = model_cls(self.config.model)

        print("base model params: ", general_utils.count_parameters(model))

        if self.config.model.use_adapters:
            task_name = self.config.model.adapter_task_name
            cfg = self.config.model.adapter
            cfg = OmegaConf.to_container(cfg)
            cfg["nonlinearity"] = None
            cfg["reduction_factor"] = None

            adapter_config = AdapterConfig.load(**cfg)
            model.transformer.add_adapter(task_name, config=adapter_config)

            # freeze all model weights except of those of this adapter
            model.transformer.train_adapter([task_name])

            # set the adapters to be used in every forward pass
            model.transformer.set_active_adapters(task_name)

        if self.config.freeze_backbone:
            model.freeze_backbone()

        self.model = model.to(self.device)
        self.model.train()
        print(self.model)
        print("final model params: ", general_utils.count_parameters(model))

        # load ll_state and image encoding networks
        if "image" in self.config.data.observation_mode:
            print("loading visual feature extractors")
            (
                self.img_preprocessor,
                self.img_encoder,
                self.depth_img_preprocessor,
                self.depth_img_encoder,
            ) = get_visual_encoders(self.config.data.image_size, "cuda")
            self.img_encoder.share_memory()
            self.depth_img_encoder.share_memory()
        else:
            self.img_encoder = None
            self.depth_img_encoder = None

        self.model.share_memory()

    def update_path(self, path):
        # "obj_ids": mw_utils.get_object_indices(self.config.env_name),
        # "dones": path["dones"],
        # add extra information to the rollout trajectory
        path.update(
            {
                "returns_to_go": general_utils.discount_cumsum(
                    path["rewards"], gamma=1.0
                ),
                "timesteps": np.arange(len(path["actions"])),
                "attention_mask": np.ones(len(path["actions"])),
                "online": 1,
            }
        )
        return path

    def train_single_iteration(self):
        # iterate over dataset
        for batch in self.data_loader:
            start = time.time()
            # put tensors on gpu
            batch = general_utils.to_device(batch, self.device)
            if "returns_to_go" in batch:
                batch["returns_to_go"] = batch["returns_to_go"][:, :-1]
                return_target = torch.clone(batch["returns_to_go"])

            if "online" in batch:
                batch["use_rtg_mask"] = batch["online"].reshape(-1, 1)

            action_target = torch.clone(batch["actions"])

            (
                _,
                action_preds,
                return_preds,
                action_log_probs,
                entropies,
            ) = self.model.forward(
                **batch,
                target_actions=action_target,
                use_means=False,  # sample during training
            )

            act_dim = action_preds.shape[2]
            mask = batch["attention_mask"].reshape(-1) > 0
            action_preds = action_preds.reshape(-1, act_dim)[mask]
            action_target = action_target.reshape(-1, act_dim)[mask]

            loss_fn_inputs = {
                "action_preds": action_preds,
                "action_targets": action_target,
                # "return_preds": return_preds,
                # "return_targets": return_target,
            }

            if self.config.model.stochastic:
                action_log_probs = action_log_probs.reshape(-1)[mask]
                entropies = entropies.reshape(-1)[mask]
                loss_fn_inputs.update(
                    {
                        "a_log_probs": action_log_probs,
                        "entropies": entropies,
                        "alpha": self.alpha,
                    }
                )

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

            if self.config.model.stochastic:
                log_dict.update(
                    {
                        "train/entropy": entropies.mean().item(),
                        "train/action_log_prob": action_log_probs.mean().item(),
                    }
                )

            # update alpha
            if self.alpha is not None:
                entropy_multiplier_loss = self.entropy_loss_fn(entropies)
                self.alpha_optimizer.zero_grad()
                entropy_multiplier_loss.backward()
                self.alpha_optimizer.step()

                log_dict["train/entropy_loss"] = entropy_multiplier_loss.item()
                log_dict["train/alpha"] = self.alpha.item()

            # update learning rates
            if self.scheduler is not None:
                self.scheduler.step()
                log_dict["train/lr"] = self.scheduler.get_last_lr()[0]

            if self.alpha_scheduler is not None:
                self.alpha_scheduler.step()
                log_dict["train/alpha_lr"] = self.alpha_scheduler.get_last_lr()[0]

            log_dict["time_per_iter"] = time.time() - start

            if self.config.log_to_wandb:
                wandb.log(log_dict)

    def train(self):
        if (
            not self.config.train_on_offline_data
            and self.config.num_warmup_rollouts > 0
        ):
            self.warmup_data_collection()

        # train loop
        for epoch in tqdm(range(self.config.num_epochs)):

            if self.config.log_to_wandb:
                wandb.log({"train/epoch": epoch})

            # save model
            if epoch % self.config.save_every == 0 and epoch != 0:
                self.save_model(epoch)

            # run evaluation for online_training
            if self.config.eval_every > 0 and epoch % self.config.eval_every == 0:
                if self.config.skip_first_eval and epoch == 0:
                    pass
                else:
                    self.eval(epoch)

            # iterate for a number of rollouts
            # for each new rollout collected, we train the model for some amount of iterations
            for _ in range(self.config.num_online_rollouts):
                # collect new rollout using stochastic policy
                if self.config.online_training:
                    with torch.no_grad():
                        import ipdb

                        ipdb.set_trace()
                        path = self.mp_rollout(
                            num_processes=1,
                            num_rollouts=1,
                            use_means=False,
                            attend_to_rtg=True,
                            log_eval_videos=False,
                        )[0]
                        self.total_online_rollouts += 1

                    # remove the trajectory with lowest reward
                    # not all low return trajectories are bad
                    # need to weigh with the traj length
                    selection_dist = self.sampler.get_remove_dist()
                    indices = np.arange(len(self.dataset))
                    ind_to_remove = np.random.choice(
                        indices,
                        size=1,
                        p=selection_dist,  # reweights so we sample according to timesteps
                    )[0]
                    del self.dataset.trajectories[ind_to_remove]

                    # create a new trajectory
                    new_trajectory = self.update_path(path)
                    self.dataset.trajectories.append(new_trajectory)

                    # refresh dataloader
                    self.setup_dataloader()

                    # log stats about current replay buffer
                    metrics = eval_utils.compute_eval_metrics(self.dataset.trajectories)
                    wandb.log({f"buffer_stats/{k}": v for k, v in metrics.items()})

                for _ in range(self.config.num_steps_per_epoch):
                    self.train_single_iteration()
                    self.total_training_iters += 1

        # save very last epoch
        self.save_model(epoch)

    def mp_rollout(
        self,
        num_processes,
        num_rollouts,
        use_means=True,
        attend_to_rtg=False,
        log_eval_videos=False,
    ):
        # torch.multiprocessing.set_start_method("spawn", force=True)
        start = time.time()

        rollout_kwargs = general_utils.AttrDict(
            config=self.config,
            model=self.model,
            img_encoder=self.img_encoder,
            depth_img_encoder=self.depth_img_encoder,
            state_mean=self.dataset.state_mean,
            state_std=self.dataset.state_std,
            device="cuda",
            use_means=use_means,
            attend_to_rtg=attend_to_rtg,
            log_eval_videos=log_eval_videos,
        )

        if num_processes > 0:
            # run multiple threads at the same time
            p = mp.Pool(processes=num_processes)

            # fetch the results of the rollout
            results = [
                p.apply_async(rollout, (), rollout_kwargs) for i in range(num_rollouts)
            ]
            eval_rollouts = [p.get() for p in results]
            p.close()
            p.join()
        else:
            for i in range(num_rollouts):
                eval_rollouts = [rollout(**rollout_kwargs)]

        print(
            f"done, got {len(eval_rollouts)} rollouts in {time.time() - start} seconds"
        )
        return eval_rollouts


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
