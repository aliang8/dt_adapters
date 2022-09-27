import numpy as np


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


KEYS_TO_USE = ["seed", "data.context_len", "model.n_layer", "model.n_head", "data_file"]


def create_exp_prefix(config):
    out = ""
    for key in KEYS_TO_USE:
        keys = key.split(".")
        value = config
        for k in keys:
            value = value[k]

        out += f"{key}={value},"
    return out
