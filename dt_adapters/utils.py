import numpy as np
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import os
import sys
import wandb
import shutil
from pathlib import Path
from omegaconf import OmegaConf


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def create_optimizer(optim_cls, model, lr, weight_decay, warmup_steps):
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

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1),
    )
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
    ckpt_dir = Path(os.path.join(config.log_dir, config.exp_name))

    # check if this experiment already exists
    # create results folders
    if os.path.exists(ckpt_dir) and not config.resume_experiment:
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

    if config.log_to_wandb:
        wandb.init(
            name=config.exp_name,
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
