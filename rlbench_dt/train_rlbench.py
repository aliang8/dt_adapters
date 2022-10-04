import argparse
import json
import numpy as np
import time
import os
import sys
import wandb
import random
import torch
import glob
import h5py
import hydra
from pprint import pprint
from tqdm import tqdm
from collections import OrderedDict

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler, RandomSampler


from rlbench_dataset import RLBenchDemoDataset
# from omegaconf import OmegaConf

from models.decision_transformer import DecisionTransformerSeparateState
# from sampler import ImportanceWeightBatchSampler
from transformers.adapters.configuration import AdapterConfig

# import general_utils
# import mw_utils
# import eval_utils
# from garage.envs import GymEnv
# from garage.np import discount_cumsum, stack_tensor_dict_list

import numpy as np

from RLBench.rlbench.action_modes.action_mode import MoveArmThenGripper
from RLBench.rlbench.action_modes.arm_action_modes import JointVelocity
from RLBench.rlbench.action_modes.gripper_action_modes import Discrete
from RLBench.rlbench.environment import Environment
from RLBench.rlbench.observation_config import ObservationConfig
from RLBench.rlbench.tasks import MT15_V1

class Trainer(object):
    def __init__(self, config):
        # first set seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.config = config
        self.setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.setup_model()

        # setup dataset
        self.dataset = RLBenchDemoDataset(self.config.data)
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_env(self.config.env_name)
        self.total_training_iters = 0

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

        self.loss_fn = loss_fn

    def rollout(self, use_means=False, attend_to_rtg=False, phase="train"):
        """Sample a single episode of the agent in the environment."""
        env_steps = []
        agent_infos = []
        observations = []
        last_obs, episode_infos = self.env.reset()
        self.model.reset()

        state_dim = self.model.state_dim
        act_dim = self.model.act_dim

        if self.config.log_eval_videos and phase == "eval":
            self.env._visualize = True
        else:
            self.env._visualize = False

        state_mean = torch.from_numpy(self.dataset.state_mean).to(device=self.device)
        state_std = torch.from_numpy(self.dataset.state_std).to(device=self.device)

        states = (
            torch.from_numpy(last_obs)
            .reshape(1, state_dim)
            .to(device=self.device, dtype=torch.float32)
        )
        actions = torch.zeros((0, act_dim), device=self.device, dtype=torch.float32)
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        use_rtg_mask = torch.tensor([attend_to_rtg]).reshape(1, 1).to(self.device)

        target_return = torch.tensor(
            self.config.target_return / self.config.scale,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1)

        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
        episode_length = 0

        while episode_length < (self.env.max_path_length or np.inf):
            # add padding
            actions = torch.cat(
                [actions, torch.zeros((1, act_dim), device=self.device)], dim=0
            )
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])

            action, agent_info = self.model.get_action(
                states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                actions=actions.to(dtype=torch.float32),
                returns_to_go=target_return,
                obj_ids=self.obj_ids,
                timesteps=timesteps.to(dtype=torch.long),
                use_means=use_means,
                use_rtg_mask=use_rtg_mask,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            es = self.env.step(action)

            env_steps.append(es)
            observations.append(last_obs)
            agent_infos.append(agent_info)

            episode_length += 1
            if es.last:
                break
            last_obs = es.observation

            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=self.device, dtype=torch.long)
                    * episode_length,
                ],
                dim=1,
            )
            cur_state = (
                torch.from_numpy(last_obs).to(device=self.device).reshape(1, state_dim)
            )
            states = torch.cat([states, cur_state], dim=0)
            pred_return = target_return[0, -1] - (es.reward / self.config.scale)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            rewards[-1] = es.reward

        return dict(
            episode_infos=episode_infos,
            observations=np.array(observations),
            actions=np.array([es.action for es in env_steps]),
            rewards=np.array([es.reward for es in env_steps]),
            agent_infos=stack_tensor_dict_list(agent_infos),
            env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
            dones=np.array([es.terminal for es in env_steps]),
        )

    def setup_env(self, env_name):
        # create env for online training
        if self.config.online_training:
            env = mw_utils.initialize_env(env_name, self.config.obj_randomization)
            max_path_length = env.max_path_length
            self.env = GymEnv(env, max_episode_length=max_path_length)

            if not self.config.train_on_offline_data:  # clear out dataset buffer
                self.dataset.trajectories = []

            self.obj_ids = mw_utils.get_object_indices(self.config.env_name)
            self.obj_ids = (
                torch.tensor(self.obj_ids).long().to(self.device).unsqueeze(0)
            )

        else:
            self.config.num_online_rollouts = 1

    def warmup_data_collection(self):
        print("collecting warmup trajectories to fill replay buffer...")
        start = time.time()
        for _ in tqdm(range(self.config.num_warmup_rollouts)):
            with torch.no_grad():
                # don't attend to Returns for warmup because we are using pretrained model
                path = self.rollout(use_means=True, attend_to_rtg=False, phase="train")
            new_trajectory = self.create_traj_from_path(path)
            self.dataset.trajectories.append(new_trajectory)
        print(f"took {time.time() - start} seconds for warmup collection")

        # info about initial dataset
        print("initial dataset size: ", len(self.dataset))
        metrics = eval_utils.compute_eval_metrics(self.dataset.trajectories)
        pprint(metrics)

    def setup_logging(self):
        group_name = self.config.exp_name
        exp_prefix = f"exp-{random.randint(int(1e5), int(1e6) - 1)}"

        if self.config.log_to_wandb:
            wandb.init(
                name=exp_prefix,
                group=group_name,
                project="dt-adapters",
                config=self.config,
                entity="glamor",
            )

        self.ckpt_dir = os.path.join(
            os.environ["LOG_DIR"], self.config.exp_name, exp_prefix, "models"
        )
        print(f"saving outputs to: {self.ckpt_dir}")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def setup_model(self):
        model = DecisionTransformerSeparateState(**self.config.model)
        print("base model params: ", general_utils.count_parameters(model))

        if self.config.model_ckpt_dir and self.config.load_from_ckpt:
            # loading from previous checkpoint
            ckpt_file = sorted(glob.glob(f"{self.config.model_ckpt_dir}/models/*"))[-1]
            print(f"loading pretrained model from {ckpt_file}")
            state_dict = torch.load(ckpt_file)
            model_config = state_dict["config"]

            model = DecisionTransformerSeparateState(**model_config.model)
            del state_dict["config"]
            del state_dict["epoch"]
            model.load_state_dict(state_dict, strict=True)
            self.config.batch_size = model_config["batch_size"]

        if self.config.use_adapters:
            task_name = self.config.adapter_task_name
            cfg = self.config.adapter
            cfg = OmegaConf.to_container(cfg)
            cfg["nonlinearity"] = None
            cfg["reduction_factor"] = None

            adapter_config = AdapterConfig.load(**cfg)
            model.transformer.add_adapter(task_name, config=adapter_config)

            # freeze all model weights except of those of this adapter
            model.transformer.train_adapter([task_name])

            # set the adapters to be used in every forward pass
            model.transformer.set_active_adapters(task_name)
            print("model params using adapter: ", general_utils.count_parameters(model))

        self.model = model.to(self.device)
        self.model.train()

    def setup_dataloader(self):
        if self.config.online_training:
            # create batch sampler to rerank trajectories based on length during training
            sampler = ImportanceWeightBatchSampler(
                self.dataset, self.config.batch_size, shuffle=True
            )
        else:
            sampler = RandomSampler(self.dataset)

        self.data_loader = DataLoader(
            dataset=self.dataset,
            sampler=sampler,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_data_workers,
            drop_last=False,
        )

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

            self.log_entropy_multiplier = torch.zeros(
                1, requires_grad=True, device=self.device
            )

            self.alpha_optimizer = torch.optim.AdamW(
                [self.log_entropy_multiplier],
                lr=self.config.alpha_lr,
                weight_decay=self.config.alpha_weight_decay,
            )

            self.entropy_loss_fn = lambda entropies: torch.exp(
                self.log_entropy_multiplier
            ) * (torch.mean(entropies.detach()) - target_entropy)

            self.alpha_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.alpha_optimizer,
                lambda steps: min((steps + 1) / self.config.warmup_steps, 1),
            )
        else:
            self.log_entropy_multiplier = None
            self.alpha_optimizer = None
            self.alpha_scheduler = None
            self.entropy_loss_fn = None

    def save_videos(self, videos):
        video_array = mw_utils.create_video_grid(
            videos,
            height=self.config.image_height,
            width=self.config.image_width,
        )
        wandb.log(
            {
                f"eval/{self.config.env_name}/rollout_videos": wandb.Video(
                    video_array, fps=self.config.fps, format="gif"
                )
            }
        )

    def eval(self, epoch):
        eval_rollouts = []
        for _ in range(self.config.num_eval_rollouts):
            with torch.no_grad():
                path = self.rollout(use_means=True, attend_to_rtg=True, phase="eval")
                eval_rollouts.append(path)

        # compute metrics and log
        metrics = eval_utils.compute_eval_metrics(eval_rollouts)
        metrics["epoch"] = epoch
        metrics["total_training_iters"] = self.total_training_iters

        if self.config.log_to_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

            if self.config.log_eval_videos:
                videos = [traj["env_infos"]["frames"] for traj in eval_rollouts]
                self.save_videos(videos)

        print("=" * 50)
        print(f"epoch {epoch} eval out of {self.config.num_eval_rollouts} episodes")
        pprint(metrics)
        print("=" * 50)

    def save_model(self, epoch):
        if self.config.online_training and self.config.use_adapters:
            # save just the adapter weights
            self.model.transformer.save_adapter(
                self.ckpt_dir,
                self.config.env_name,
                meta_dict={"epoch": epoch, "config": self.config},
            )

        path = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}.pt")
        print(f"saving model to {path}")
        save_dict = self.model.state_dict()
        save_dict["epoch"] = epoch
        save_dict["config"] = self.config
        torch.save(save_dict, path)

    def create_traj_from_path(self, path):
        trajectory = {
            "states": path["observations"],
            "obj_ids": mw_utils.get_object_indices(self.config.env_name),
            "actions": path["actions"],
            "rewards": path["rewards"],
            "dones": path["dones"],
            "returns": general_utils.discount_cumsum(path["rewards"][()], gamma=1.0),
            "timesteps": np.arange(len(path["observations"])),
            "attention_mask": np.ones(len(path["observations"])),
            "online": 1,
        }
        return trajectory

    def train_single_iteration(self):
        # iterate over dataset
        for batch in self.data_loader:
            batch = general_utils.to_device(batch, self.device)
            batch["returns_to_go"] = batch["returns_to_go"][:, :-1]
            batch["use_rtg_mask"] = batch["online"].reshape(-1, 1)

            action_target = torch.clone(batch["actions"])

            _, action_preds, _, action_log_probs, entropies = self.model.forward(
                **batch,
                target_actions=action_target,
                use_means=False,  # sample during training
            )

            act_dim = action_preds.shape[2]
            mask = batch["attention_mask"].reshape(-1) > 0
            action_preds = action_preds.reshape(-1, act_dim)[mask]
            action_target = action_target.reshape(-1, act_dim)[mask]
            action_log_probs = action_log_probs.reshape(-1)[mask]
            entropies = entropies.reshape(-1)[mask]

            loss_fn_inputs = {
                "action_preds": action_preds,
                "action_targets": action_target,
                "a_log_probs": action_log_probs,
                "entropies": entropies,
                "log_entropy_multiplier": self.log_entropy_multiplier,
            }

            loss = self.loss_fn(**loss_fn_inputs)

            # update model
            self.optimizer.zero_grad()
            loss.backward()

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
                "train/action_loss": loss.item(),
                "train/entropy": entropies.mean().item(),
                "train/action_log_prob": action_log_probs.mean().item(),
                "train/max_norm": max_norm,
                "train/total_norm": total_norm,
            }

            # update alpha
            if self.log_entropy_multiplier is not None:
                entropy_multiplier_loss = self.entropy_loss_fn(entropies)
                self.alpha_optimizer.zero_grad()
                entropy_multiplier_loss.backward()
                self.alpha_optimizer.step()

                log_dict["train/entropy_loss"] = entropy_multiplier_loss.item()
                log_dict[
                    "train/log_entropy_multiplier"
                ] = self.log_entropy_multiplier.item()

            # update learning rates
            if self.scheduler is not None:
                self.scheduler.step()
                log_dict["train/lr"] = self.scheduler.get_last_lr()[0]

            if self.alpha_scheduler is not None:
                self.alpha_scheduler.step()
                log_dict["train/alpha_lr"] = self.alpha_scheduler.get_last_lr()[0]

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
            if epoch % self.config.eval_every == 0 and self.config.online_training:
                if self.config.skip_first_eval and epoch == 0:
                    continue

                self.eval(epoch)

            # iterate for a number of rollouts
            # for each new rollout collected, we train the model for some amount of iterations
            for _ in range(self.config.num_online_rollouts):
                # collect new rollout using stochastic policy
                if self.config.online_training:
                    with torch.no_grad():
                        path = self.rollout(
                            use_means=False, attend_to_rtg=True, phase="train"
                        )
                    # import ipdb

                    # ipdb.set_trace()

                    # remove the trajectory with lowest reward
                    # self.dataset.trajectories = self.dataset.trajectories[1:]
                    traj_returns = np.array(
                        [traj["rewards"].sum() for traj in self.dataset.trajectories]
                    )
                    ind_to_remove = np.argmin(traj_returns)
                    del self.dataset.trajectories[ind_to_remove]

                    # create a new trajectory
                    new_trajectory = self.create_traj_from_path(path)
                    self.dataset.trajectories.append(new_trajectory)

                    # refresh dataloader
                    self.setup_dataloader()

                for _ in range(self.config.num_steps_per_epoch):
                    self.train_single_iteration()
                    self.total_training_iters += 1


@hydra.main(config_path="configs", config_name="train")
def main(config):
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
