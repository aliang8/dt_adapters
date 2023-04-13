import numpy as np
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import os
import sys
import wandb
import glob
import shutil
from pathlib import Path
from omegaconf import OmegaConf

from dt_adapters.utils.adapter_utils import load_adapter, load_fusion_layer


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(batch, device):
    """
    put all input from batch dictionary onto the same device
    """
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = to_device(v, device)
        else:
            batch[k] = batch[k].to(device)
    return batch


def pad_to_length(array, final_length, pad_zeros=True):
    """
    Given an array of [N, D] pad with zeros until [T, D] where T is
    the target size. Returns a 2D array
    """

    # add extra dimension
    if len(array.shape) == 1:
        array = array[:, np.newaxis]

    shape = array.shape[1:]
    pad_length = final_length - array.shape[0]

    if pad_zeros:
        pad = np.zeros((pad_length, *shape))
    else:
        pad = np.ones((pad_length, *shape)) * -10

    # pad to context length
    array = np.concatenate([pad, array], axis=0)
    return array


def save_model(ckpt_file, model, optimizer, scheduler=None, metadata=None):
    print(f"saving model to {ckpt_file}")
    save_dict = {}
    save_dict["model"] = model.state_dict()
    if optimizer:
        save_dict["optimizer"] = optimizer.state_dict()

    if scheduler:
        save_dict["scheduler"] = scheduler.state_dict()

    save_dict.update(metadata)
    torch.save(save_dict, ckpt_file)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def find_diff_dict(d1, d2, path=""):
    for k in d1:
        if k == "general":
            continue
        if k in d2:
            if type(d1[k]) is dict:
                find_diff_dict(d1[k], d2[k], "%s -> %s" % (path, k) if path else k)

            if d1[k] != d2[k] and type(d1[k]) is not dict and type(d2[k]) is not dict:
                result = [
                    "%s: " % path,
                    f"{bcolors.FAIL} - {k} : {d1[k]}{bcolors.ENDC}",
                    f"{bcolors.OKGREEN} + {k} : {d2[k]}{bcolors.ENDC}",
                ]
                print("\n".join(result))
        else:
            print("%s%s as key not in d2\n" % ("%s: " % path if path else "", k))


def find_new_keys(d1, d2, path=""):
    for k in d1:
        if k == "general":
            continue

        if k not in d2:
            print(f"{bcolors.OKBLUE} + {k}: {d1[k]} {bcolors.ENDC}")
            continue

        if type(d1[k]) is dict:
            find_new_keys(d1[k], d2[k], path=f"{path} -> {k}" if path else k)

        if k not in d2:
            result = [
                "%s: " % path,
                f"{bcolors.OKBLUE} + {k} : {d1[k]}{bcolors.ENDC}",
            ]
            print("\n".join(result))


def load_model_from_ckpt(model, cfg, model_ckpt_dir, adapter_library=None, strict=True):
    # loading from previous checkpoint
    ckpt_file = sorted(glob.glob(f"{model_ckpt_dir}/models/*"))[-1]
    print(f"loading pretrained model from {ckpt_file}")

    state_dict = torch.load(ckpt_file)
    prev_cfg = state_dict["config"]
    epoch = state_dict["epoch"]
    del state_dict["config"]
    del state_dict["epoch"]

    # find differences between old cfg and new one
    print("Differences between old config and new one:")
    find_diff_dict(OmegaConf.to_container(prev_cfg), OmegaConf.to_container(cfg))

    # find keys in new config not in the old config
    print("New keys:")
    find_new_keys(OmegaConf.to_container(cfg), OmegaConf.to_container(prev_cfg))

    model.load_state_dict(state_dict["model"], strict=strict)

    # load adapters and fusion layers here if relevant
    adapter_key = f"{cfg.model.adapter_task_name}_{cfg.exp_name}_{cfg.seed}"
    if cfg.stage == "eval" and cfg.model.use_single_adapter:
        load_adapter(model, adapter_library, adapter_key=adapter_key)

    if cfg.stage == "eval" and cfg.model.use_adapter_fusion:
        load_fusion_layer(
            model,
            adapter_library,
            adapters_to_use=cfg.model.adapters_to_use,
            task_name=adapter_key,
        )

    return model, epoch


def create_optimizer(
    optim_cls, model, lr, weight_decay, warmup_steps, use_scheduler=True
):
    # setup optimizer and scheduler
    if optim_cls == "adam":
        optim_cls = torch.optim.Adam
    elif optim_cls == "adamw":
        optim_cls = torch.optim.AdamW
    else:
        raise NotImplementedError()

    optimizer = optim_cls(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / warmup_steps, 1),
        )
    else:
        scheduler = None
    return optimizer, scheduler


def compute_gradient_norms(model):
    # get all parameters that require gradients
    model_params = [
        p for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    # compute norms
    max_norm = max([p.grad.detach().abs().max() for p in model_params])
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in model_params]),
        2.0,
    ).item()
    return max_norm, total_norm


def setup_logging(config):
    if config.stage == "pretraining":
        ckpt_dir = Path(os.path.join(config.log_dir, config.exp_name, config.seed))
    elif config.stage == "finetune" or config.stage == "eval":
        ckpt_dir = Path(
            os.path.join(
                config.log_dir, config.exp_name, config.data.eval_task, str(config.seed)
            )
        )
    config.ckpt_dir = ckpt_dir

    # check if this experiment already exists
    # create results folders
    if (
        os.path.exists(ckpt_dir)
        and not config.resume_experiment
        and not config.overwrite_folder
    ):
        overwrite = input(f"{ckpt_dir} already exists, overwrite [y/n]")

        if overwrite == "y":
            shutil.rmtree(ckpt_dir)
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(ckpt_dir / "models", exist_ok=True)
        else:
            sys.exit()
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(ckpt_dir / "models", exist_ok=True)

    # save config to yaml file
    config_file = ckpt_dir / "config.yaml"
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            OmegaConf.save(config, f)

    if config.log_to_wandb:
        wandb.init(
            name=f"{config.exp_name}_{config.data.eval_task}_{config.seed}",
            project=config.project_name,
            config=OmegaConf.to_container(config),
            entity="glamor",
        )

    return ckpt_dir


class AttrDict(Dict):
    """Extended dictionary accessible with dot notation.

    >>> ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    >>> ad.key1
    1
    >>> ad.update({'my-key': 3.14})
    >>> ad.update(new_key=42)
    >>> ad.key1 = 2
    >>> ad
    "key1":    2
    "key2":    abc
    "my-key":  3.14
    "new_key": 42
    """

    def __getattr__(self, key: str) -> Optional[Any]:
        try:
            return self[key]
        except KeyError as exp:
            raise AttributeError(f'Missing attribute "{key}"') from exp

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __repr__(self) -> str:
        if not len(self):
            return ""
        max_key_length = max(len(str(k)) for k in self)
        tmp_name = "{:" + str(max_key_length + 3) + "s} {}"
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        out = "\n".join(rows)
        return out
