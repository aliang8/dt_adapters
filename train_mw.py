import argparse
import json
import time
import os
import sys
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

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler, RandomSampler


from mw_dataset import MWDemoDataset
from omegaconf import OmegaConf
from models.decision_transformer import DecisionTransformerSeparateState
from sampler import ImportanceWeightBatchSampler

from transformers.adapters.configuration import AdapterConfig

import general_utils
import mw_utils
import eval_utils
from garage.envs import GymEnv
from garage.np import discount_cumsum, stack_tensor_dict_list


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer(object):
    def __init__(self, config):
        set_all_seeds(config.seed)
        self.config = config
        self.setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.setup_model()

        # setup dataset
        self.dataset = MWDemoDataset(self.config.data, stage=config.stage)
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_env(self.config.env_name)
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

    def rollout(self, use_means=False, attend_to_rtg=False, log_eval_videos=False):
        """Sample a single episode of the agent in the environment."""
        env_steps = []
        agent_infos = []
        observations = []
        last_obs, episode_infos = self.env.reset()
        self.model.reset()

        state_dim = self.model.state_dim
        act_dim = self.model.act_dim

        if log_eval_videos:
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

            action, return_target, agent_info = self.model.get_action(
                states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                actions=actions.to(dtype=torch.float32),
                returns_to_go=target_return,
                obj_ids=self.obj_ids,
                timesteps=timesteps.to(dtype=torch.long),
                use_means=use_means,
                use_rtg_mask=use_rtg_mask,
                sample_return_dist=self.config.model.predict_return_dist,
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
            if self.config.model.predict_return_dist:
                # use the model's prediction of the return to go
                # follow this paper: https://openreview.net/forum?id=fwJWhOxuzV9
                target_return = torch.cat(
                    [target_return, return_target.reshape(1, 1)], dim=1
                )
            else:
                pred_return = target_return[0, -1] - (es.reward / self.config.scale)
                target_return = torch.cat(
                    [target_return, pred_return.reshape(1, 1)], dim=1
                )

            rewards[-1] = es.reward

        rewards = np.array([es.reward for es in env_steps])
        returns = general_utils.discount_cumsum(rewards, gamma=1.0)
        return dict(
            episode_infos=episode_infos,
            observations=np.array(observations),
            actions=np.array([es.action for es in env_steps]),
            rewards=rewards,
            returns_to_go=returns,
            agent_infos=stack_tensor_dict_list(agent_infos),
            env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
            dones=np.array([es.terminal for es in env_steps]),
        )

    def setup_env(self, env_name):
        # create env for online training
        if self.config.env_name:
            print(
                f"initializing metaworld env: {env_name}, obj_random: {self.config.obj_randomization}"
            )
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
                path = self.rollout(
                    use_means=True, attend_to_rtg=False, log_eval_videos=False
                )
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
                config=OmegaConf.to_container(self.config),
                entity="glamor",
            )

        if self.config.log_outputs:
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

    def save_videos(self, videos):
        video_array = mw_utils.create_video_grid(
            videos,
            height=self.config.image_height,
            width=self.config.image_width,
        )
        wandb.log(
            {
                f"eval/{self.config.env_name}/rollout_videos": wandb.Video(
                    video_array,
                    caption=f"train_iter_{self.total_training_iters}",
                    fps=self.config.fps,
                    format="gif",
                )
            }
        )

    def eval(self, epoch):
        eval_rollouts = []
        log_eval_videos = epoch % self.config.log_eval_videos_every == 0
        attend_to_rtg = True if self.config.online_training else False
        for _ in range(self.config.num_eval_rollouts):
            with torch.no_grad():
                path = self.rollout(
                    use_means=True,
                    attend_to_rtg=attend_to_rtg,
                    log_eval_videos=log_eval_videos,
                )
                eval_rollouts.append(path)

        # compute metrics and log
        metrics = eval_utils.compute_eval_metrics(eval_rollouts)
        metrics["epoch"] = epoch
        metrics["total_training_iters"] = self.total_training_iters
        metrics["total_online_rollouts"] = self.total_online_rollouts

        if self.config.log_to_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

            if log_eval_videos:
                videos = [traj["env_infos"]["frames"] for traj in eval_rollouts]
                self.save_videos(videos)

        print("=" * 50)
        print(f"epoch {epoch} eval out of {self.config.num_eval_rollouts} episodes")
        pprint(metrics)
        print("=" * 50)

    def save_model(self, epoch):
        if self.config.use_adapters:
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
            "returns_to_go": general_utils.discount_cumsum(
                path["rewards"][()], gamma=1.0
            ),
            "timesteps": np.arange(len(path["observations"])),
            "attention_mask": np.ones(len(path["observations"])),
            "online": 1,
        }
        return trajectory

    def train_single_iteration(self):
        # iterate over dataset
        for batch in self.data_loader:
            start = time.time()
            batch = general_utils.to_device(batch, self.device)
            batch["returns_to_go"] = batch["returns_to_go"][:, :-1]
            batch["use_rtg_mask"] = batch["online"].reshape(-1, 1)

            action_target = torch.clone(batch["actions"])
            return_target = torch.clone(batch["returns_to_go"])

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
                "return_preds": return_preds,
                "return_targets": return_target,
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
            if epoch % self.config.eval_every == 0:
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
                        path = self.rollout(
                            use_means=False,
                            attend_to_rtg=True,
                            log_eval_videos=False,
                        )
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
                    new_trajectory = self.create_traj_from_path(path)
                    self.dataset.trajectories.append(new_trajectory)

                    # refresh dataloader
                    self.setup_dataloader()

                    # log stats about current replay buffer
                    metrics = eval_utils.compute_eval_metrics(self.dataset.trajectories)
                    wandb.log({f"buffer_stats/{k}": v for k, v in metrics.items()})

                for _ in range(self.config.num_steps_per_epoch):
                    self.train_single_iteration()
                    self.total_training_iters += 1


@hydra.main(config_path="configs", config_name="train")
def main(config):
    OmegaConf.set_struct(config, False)
    config.update(config.general)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
