import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger

from evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from models.decision_transformer import DecisionTransformer
from training.act_trainer import ActTrainer
from training.seq_trainer import SequenceTrainer

from utils import discount_cumsum


def setup(config):
    exp_prefix = config.experiment.name
    group_name = f"{exp_prefix}-{config.env.env_name}"
    exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"

    if config.experiment.log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project="dt-adapters",
            config=config,
        )


def setup_optimizer(model, config):
    # setup optimizer and scheduler
    warmup_steps = config.optim_params.warmup_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim_params.learning_rate.initial,
        weight_decay=config.optim_params.learning_rate.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    return optimizer, scheduler


# def get_batch(device, batch_size=256, max_len=K):
#     batch_inds = np.random.choice(
#         np.arange(num_trajectories),
#         size=batch_size,
#         replace=True,
#         p=p_sample,  # reweights so we sample according to timesteps
#     )

#     s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
#     for i in range(batch_size):
#         traj = trajectories[int(sorted_inds[batch_inds[i]])]
#         si = random.randint(0, traj["rewards"].shape[0] - 1)

#         # get sequences from dataset
#         s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
#         a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
#         r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
#         if "terminals" in traj:
#             d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
#         else:
#             d.append(traj["dones"][si : si + max_len].reshape(1, -1))
#         timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
#         timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
#         rtg.append(
#             discount_cumsum(traj["rewards"][si:], gamma=1.0)[
#                 : s[-1].shape[1] + 1
#             ].reshape(1, -1, 1)
#         )
#         if rtg[-1].shape[1] <= s[-1].shape[1]:
#             rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

#         # padding and state + reward normalization
#         tlen = s[-1].shape[1]
#         s[-1] = np.concatenate(
#             [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
#         )
#         s[-1] = (s[-1] - state_mean) / state_std
#         a[-1] = np.concatenate(
#             [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
#         )
#         r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
#         d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
#         # rtg[-1] = (
#         #     np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
#         # )
#         timesteps[-1] = np.concatenate(
#             [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
#         )
#         mask.append(
#             np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1)
#         )

#     s = torch.from_numpy(np.concatenate(s, axis=0)).to(
#         dtype=torch.float32, device=device
#     )
#     a = torch.from_numpy(np.concatenate(a, axis=0)).to(
#         dtype=torch.float32, device=device
#     )
#     r = torch.from_numpy(np.concatenate(r, axis=0)).to(
#         dtype=torch.float32, device=device
#     )
#     d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
#     rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
#         dtype=torch.float32, device=device
#     )
#     timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
#         dtype=torch.long, device=device
#     )
#     mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

#     return s, a, r, d, rtg, timesteps, mask


def setup_env(config):
    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.data, all_obs_keys=config.all_obs_keys, verbose=True
    )

    # create environment
    envs = OrderedDict()
    if config.rollout.enabled:
        # create environments for validation runs
        env_names = [config.env_name]

        if config.additional_envs is not None:
            for name in config.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env
            print(envs[env.name])
    return envs, env_meta, shape_meta


def setup_dataloaders(config, shape_meta):
    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"]
    )
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    print(trainset[0])

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )
    return train_loader, None


def eval_episodes(
    env, model, state_dim, act_dim, num_eval_episodes=5, max_ep_len=1000, device="cpu"
):
    returns, lengths = [], []
    for _ in range(num_eval_episodes):
        with torch.no_grad():
            ret, length = evaluate_episode_rtg(
                env,
                state_dim,
                act_dim,
                model,
                max_ep_len=max_ep_len,
                # scale=scale,
                # target_return=target_rew / scale,
                # mode=mode,
                # state_mean=state_mean,
                # state_std=state_std,
                device=device,
            )
        returns.append(ret)
        lengths.append(length)
    return {"return_mean": np.mean(returns), "return_std": np.std(returns)}
    # return {
    #     f"target_{target_rew}_return_mean": np.mean(returns),
    #     f"target_{target_rew}_return_std": np.std(returns),
    #     f"target_{target_rew}_length_mean": np.mean(lengths),
    #     f"target_{target_rew}_length_std": np.std(lengths),
    # }


def train(config, device):
    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.env.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    (
        env,
        env_meta,
        shape_meta,
    ) = setup_env(config.env)

    state_dim = sum(
        sum([shape_meta["all_shapes"][k] for k in shape_meta["all_obs_keys"]], [])
    )
    act_dim = shape_meta["ac_dim"]
    model = DecisionTransformer(state_dim=state_dim, act_dim=act_dim, **config.model)
    model = model.to(device)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    optimizer, scheduler = setup_optimizer(model, config.algo)
    train_loader, val_loader = setup_dataloaders(config, shape_meta)
    loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2)

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=config.train.batch_size,
        scheduler=scheduler,
        loaders={"train": train_loader, "val": val_loader},
        loss_fn=loss_fn,
        device=device,
        eval_fns=[eval_episodes],
    )

    for iter in range(config.train.max_iters):
        outputs = trainer.train_iteration(
            num_steps=config.train.num_steps_per_iter,
            iter_num=iter + 1,
            print_logs=True,
        )
        if config.experiment.log_to_wandb:
            wandb.log(outputs)

        if iter % config.train.eval_every == 0:
            eval_outputs = eval_episodes(
                env,
                model,
                state_dim,
                act_dim,
                config.train.num_eval_episodes,
                config.train.max_ep_len,
                device,
            )


def main(args):

    # if args.config is not None:
    #     ext_cfg = json.load(open(args.config, "r"))
    #     config = config_factory(ext_cfg["algo_name"])
    #     # update config with external json - this will throw errors if
    #     # the external config has keys not present in the base algo config
    #     with config.values_unlocked():
    #         config.update(ext_cfg)
    # else:
    config = config_factory("dt")

    if args.dataset is not None:
        config.env.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        setup(config)
        train(config, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="set this flag to run a quick training run for debugging purposes",
    )

    args = parser.parse_args()
    main(args)
