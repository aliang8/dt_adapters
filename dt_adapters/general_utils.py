import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union


def to_device(batch, device):
    for k, v in batch.items():
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
