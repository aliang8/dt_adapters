import glob
import torch
import numpy as np
from omegaconf import OmegaConf
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = to_device(v, device)
        else:
            batch[k] = batch[k].to(device)
    return batch


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


KEYS_TO_USE = [
    "seed",
    "data.context_len",
    "model.n_layer",
    "model.n_head",
    "data.data_file",
]

# chunk configs
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def create_exp_prefix(config):
    out = ""
    for key in KEYS_TO_USE:
        keys = key.split(".")
        value = config
        for k in keys:
            value = value[k]

        out += f"{key}={value}-"
    return out


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


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


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


def load_model_from_ckpt(model, cfg, model_ckpt_dir, strict=True):
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
    return model, epoch


def load_optimizer(model_ckpt_dir, optimizer, scheduler=None):
    ckpt_file = sorted(glob.glob(f"{model_ckpt_dir}/models/*"))[-1]
    state_dict = torch.load(ckpt_file)

    optimizer.load_state_dict(state_dict["optimizer"])

    if scheduler:
        scheduler.load_state_dict(state_dict["scheduler"])


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
