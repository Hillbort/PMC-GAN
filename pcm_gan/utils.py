import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True, verbose: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            print(f"device: cuda ({name})")
        return device
    device = torch.device("cpu")
    if verbose:
        print("device: cpu")
    return device


@dataclass
class ScenarioConfig:
    seq_len: int = 24
    channels: int = 4  # [pv, wind, load_e, load_h]
    cond_dim: int = 3  # [temperature, wind_speed, irradiance]
    daylight_start: int = 6
    daylight_end: int = 18


def build_daylight_mask(seq_len: int, channels: int, daylight_start: int, daylight_end: int) -> torch.Tensor:
    # daylight_start/daylight_end are in hours; map to indices based on seq_len
    start_idx = int(round(daylight_start / 24.0 * seq_len))
    end_idx = int(round(daylight_end / 24.0 * seq_len))
    mask = torch.ones(seq_len, channels, dtype=torch.float32)
    for t in range(seq_len):
        if t < start_idx or t >= end_idx:
            mask[t, 0] = 0.0  # PV is zero at night
    return mask


def make_time_features(seq_len: int) -> torch.Tensor:
    t = torch.arange(seq_len, dtype=torch.float32)
    sin_t = torch.sin(2 * math.pi * t / seq_len)
    cos_t = torch.cos(2 * math.pi * t / seq_len)
    return torch.stack([sin_t, cos_t], dim=-1)


def net_load(x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, C)
    # Default channel conventions:
    # C=4: [pv, wind, load_e, load_h]
    # C=3: [solar, wind, load]
    c = x.shape[-1]
    if c == 4:
        pv = x[..., 0]
        wind = x[..., 1]
        load_e = x[..., 2]
        load_h = x[..., 3]
        return load_e + load_h - pv - wind
    if c == 3:
        pv = x[..., 0]
        wind = x[..., 1]
        load = x[..., 2]
        return load - pv - wind
    raise ValueError(f"Unsupported channel count for net_load: {c}")


def gpd_nll(y: torch.Tensor, xi: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # y: exceedance >= 0
    beta = torch.clamp(beta, min=eps)
    xi = torch.clamp(xi, min=-0.49, max=2.0)
    if torch.any(1 + xi * y / beta <= 0):
        # invalid region, add large penalty
        return torch.full_like(y, 50.0)
    term = 1 + xi * y / beta
    nll = torch.log(beta) + (1.0 / xi + 1.0) * torch.log(term)
    return nll


def split_train_val(
    n: int,
    val_ratio: float = 0.2,
    labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    if labels is None:
        np.random.shuffle(idx)
        cut = int(n * (1 - val_ratio))
        return idx[:cut], idx[cut:]

    labels = np.asarray(labels)
    if labels.shape[0] != n:
        raise ValueError(f"labels length {labels.shape[0]} must match n {n}")

    buckets = defaultdict(list)
    for i, label in enumerate(labels.tolist()):
        buckets[label].append(i)

    train_parts = []
    val_parts = []
    for group in buckets.values():
        group_idx = np.asarray(group, dtype=np.int64)
        np.random.shuffle(group_idx)
        if group_idx.size <= 1:
            train_parts.append(group_idx)
            continue
        val_count = int(round(group_idx.size * float(val_ratio)))
        val_count = max(1, min(group_idx.size - 1, val_count))
        val_parts.append(group_idx[:val_count])
        train_parts.append(group_idx[val_count:])

    train_idx = np.concatenate(train_parts) if train_parts else np.empty((0,), dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.empty((0,), dtype=np.int64)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    return train_idx, val_idx


def update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        model_params = dict(model.named_parameters())
        for name, ema_param in ema_model.named_parameters():
            ema_param.mul_(decay).add_(model_params[name], alpha=1.0 - decay)

        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema_model.named_buffers():
            if name in model_buffers:
                ema_buffer.copy_(model_buffers[name])


def get_generator_state_dict(ckpt: dict, prefer_ema: bool = True):
    if prefer_ema and "G_ema" in ckpt:
        return ckpt["G_ema"], "G_ema"
    return ckpt["G"], "G"
