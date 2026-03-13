import copy
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data_synth import RealDataset, ddre33_cond_layout, load_real_dataset, load_ddre33_dataset
from .losses import (
    gradient_penalty_wgan_scalar,
    wgan_d_loss,
    wgan_g_loss,
    peak_event_loss,
    ramp_event_loss,
    active_ratio_loss,
    channel_stats_loss,
    tail_quantile_loss,
    rolling_stats_loss,
    acf_loss,
    corr_matrix_loss,
)
from .models import Discriminator, Generator
from .utils import ScenarioConfig, get_device, set_seed, split_train_val, update_ema_model


def parse_args():
    p = argparse.ArgumentParser(description="PCM-GAN training (real data).")
    p.add_argument("--outdir", type=str, default="pcm_gan_runs")
    p.add_argument("--data_csv", type=str, default="")
    p.add_argument(
        "--dataset",
        type=str,
        default="real",
        choices=["real", "ddre33"],
        help="Dataset type: real (time+features) or ddre33 (wide scenario files).",
    )
    p.add_argument("--pv18_csv", type=str, default="")
    p.add_argument("--pv33_csv", type=str, default="")
    p.add_argument("--wind22_csv", type=str, default="")
    p.add_argument("--wind25_csv", type=str, default="")
    p.add_argument("--pv18_labels_csv", type=str, default="")
    p.add_argument("--pv33_labels_csv", type=str, default="")
    p.add_argument("--wind22_labels_csv", type=str, default="")
    p.add_argument("--wind25_labels_csv", type=str, default="")
    p.add_argument(
        "--ddre33_date_cond",
        action="store_true",
        help="Append day-of-year sin/cos to DDRE-33 condition labels.",
    )
    p.add_argument(
        "--ddre33_static_cond",
        action="store_true",
        help="Store DDRE-33 cond as static (B, cond_dim) instead of (B, T, cond_dim).",
    )
    p.add_argument(
        "--ddre33_curve_cond",
        action="store_true",
        help="Append per-sample curve control features to DDRE-33 conditions.",
    )
    p.add_argument(
        "--ddre33_curve_cond_norm",
        type=str,
        default="none",
        choices=["none", "minmax"],
        help="Normalization for DDRE-33 curve control features.",
    )
    p.add_argument("--cond_onehot", action="store_true", help="Use one-hot climate labels.")
    p.add_argument(
        "--ddre33_mode",
        type=str,
        default="4ch",
        choices=["4ch", "2ch_pairs", "2ch_single"],
        help="DDRE-33 mode: 4ch (pv18,pv33,wind22,wind25), 2ch_pairs (pv+wind), or 2ch_single (pv18+wind22).",
    )
    p.add_argument(
        "--max_cols",
        type=int,
        default=0,
        help="Limit number of scenario columns per node (0 = all).",
    )
    p.add_argument(
        "--x_cols",
        type=str,
        default="solar_power,wind_power,load_power",
        help="Comma-separated columns for x (order matters).",
    )
    p.add_argument(
        "--cond_cols",
        type=str,
        default="DHI,DNI,GHI,Dew Point,Solar Zenith Angle,Wind Speed,Relative Humidity,Temperature",
        help="Comma-separated columns for cond.",
    )
    p.add_argument(
        "--mask_source",
        type=str,
        default="solar",
        choices=["solar", "ghi"],
        help="Daylight mask source: solar output or GHI.",
    )
    p.add_argument(
        "--resolution",
        type=str,
        default="hourly",
        choices=["hourly", "15min", "minute"],
        help="Resample resolution for daily sequences.",
    )
    p.add_argument("--resample", type=str, default="")
    p.add_argument("--seq_len", type=int, default=0)
    p.add_argument("--cond_agg", type=str, default="mean", choices=["mean", "max", "min"])
    p.add_argument(
        "--cond_norm",
        type=str,
        default="none",
        choices=["none", "minmax", "zscore"],
        help="Normalize condition features.",
    )
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_g", type=float, default=0.0, help="Generator LR (0 uses --lr).")
    p.add_argument("--lr_d", type=float, default=0.0, help="Discriminator LR (0 uses 2*--lr).")
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d_steps", type=int, default=2)
    p.add_argument("--d_model_dim", type=int, default=64)
    p.add_argument("--d_depth", type=int, default=2)
    p.add_argument("--d_heads", type=int, default=2)
    p.add_argument("--d_patch_scales", type=int, default=1)
    p.add_argument("--d_high_weight", type=float, default=0.3, help="Weight for high-frequency diff branch.")
    p.add_argument("--lambda_gp", type=float, default=1.0)
    p.add_argument("--lambda_tail", type=float, default=0.08)
    p.add_argument("--lambda_tailq", type=float, default=None, help="(deprecated) use --lambda_tail")
    p.add_argument("--tail_q", type=float, default=0.95)
    p.add_argument("--lambda_stats", type=float, default=0.05)
    p.add_argument("--lambda_roll", type=float, default=0.10, help="Weight for rolling mean/std loss.")
    p.add_argument("--lambda_acf", type=float, default=0.10, help="Weight for ACF loss (lags 1..N).")
    p.add_argument("--lambda_corr", type=float, default=0.02, help="Weight for correlation-matrix loss.")
    p.add_argument("--lambda_peak_evt", type=float, default=0.0, help="Weight for peak event loss.")
    p.add_argument("--lambda_ramp_evt", type=float, default=0.0, help="Weight for extreme ramp loss.")
    p.add_argument("--lambda_active_evt", type=float, default=0.0, help="Weight for active-ratio loss.")
    p.add_argument("--roll_win", type=int, default=8, help="Window size for rolling stats loss.")
    p.add_argument("--acf_lags", type=int, default=6, help="Max lag for ACF loss.")
    p.add_argument("--event_peak_temp", type=float, default=20.0, help="Soft peak-time temperature.")
    p.add_argument(
        "--event_active_threshold",
        type=float,
        default=0.05,
        help="Threshold for active-ratio event loss.",
    )
    p.add_argument(
        "--event_active_sharpness",
        type=float,
        default=30.0,
        help="Sigmoid sharpness for active-ratio event loss.",
    )
    p.add_argument(
        "--tail_qs",
        type=str,
        default="0.9,0.95,0.99",
        help="Comma-separated quantiles for tail loss (overrides --tail_q).",
    )
    p.add_argument("--lambda_wind_adv", type=float, default=1.5)
    p.add_argument(
        "--extreme_resample",
        dest="extreme_resample",
        action="store_true",
        help="Enable weighted resampling to increase extreme samples in each epoch.",
    )
    p.add_argument(
        "--no_extreme_resample",
        dest="extreme_resample",
        action="store_false",
        help="Disable weighted extreme resampling.",
    )
    p.set_defaults(extreme_resample=True)
    p.add_argument("--extreme_q", type=float, default=0.95, help="Quantile for extreme score.")
    p.add_argument("--extreme_alpha", type=float, default=0.3, help="Weight scale for extreme resampling.")
    p.add_argument(
        "--extreme_joint_channels",
        type=str,
        default="auto",
        help="Channels used for extreme risk scoring. Comma indices (e.g. '0,1') or 'auto'.",
    )
    p.add_argument(
        "--mask_mode",
        type=str,
        default="data",
        choices=["data", "none"],
        help="Mask mode: data (use data-derived PV mask) or none (all ones).",
    )
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay for generator weights. Set <=0 to disable.",
    )
    p.add_argument(
        "--ema_start_epoch",
        type=int,
        default=0,
        help="Start EMA updates from this epoch index (0-based).",
    )
    p.add_argument(
        "--ch_weights",
        type=str,
        default="1,1,1",
        help="Comma-separated per-channel weights (e.g., '1,2,2').",
    )
    p.add_argument(
        "--no_channel_mixer",
        action="store_true",
        help="Disable generator channel_mixer to reduce PV dominance.",
    )
    p.add_argument(
        "--no_baseline_residual",
        action="store_true",
        help="Disable baseline+residual generator decomposition.",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="DataLoader prefetch factor per worker (only used when num_workers > 0).",
    )
    p.add_argument(
        "--cache_gpu",
        action="store_true",
        help="Cache full training data on GPU to reduce CPU bottleneck.",
    )
    p.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="If >0, set torch.set_num_threads to limit CPU usage.",
    )
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable AMP even when CUDA is available.",
    )
    p.add_argument(
        "--log_batch_every",
        type=int,
        default=0,
        help="If >0, print batch stats every N batches.",
    )
    p.add_argument(
        "--debug_nan",
        action="store_true",
        help="Print NaN/Inf diagnostics for the first batch.",
    )
    p.add_argument(
        "--metric_bins",
        type=int,
        default=20,
        help="Bins for histogram distribution error metrics.",
    )
    p.add_argument(
        "--metric_quantiles",
        type=str,
        default="0.1,0.5,0.9,0.95,0.99",
        help="Comma-separated quantiles for quantile error metrics.",
    )
    p.add_argument(
        "--train_metrics_every",
        type=int,
        default=0,
        help="Compute expensive training-set distribution metrics every N batches (0 disables).",
    )
    p.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Run validation metrics every N epochs.",
    )
    p.add_argument(
        "--torch_compile",
        action="store_true",
        help="Enable torch.compile for Generator/Discriminator on supported PyTorch versions.",
    )
    return p.parse_args()


def _parse_quantiles(s: str):
    qs = []
    for v in s.split(","):
        v = v.strip()
        if not v:
            continue
        q = float(v)
        if q <= 0.0 or q >= 1.0:
            raise ValueError(f"quantile must be in (0,1), got {q}")
        qs.append(q)
    return qs or [0.5]


def _safe_mean(values):
    if not values:
        return float("nan")
    return float(np.mean(values))


def _hist_l1(x_fake, x_real, bins=20, eps=1e-6):
    # x_*: (B, T, C) in [0,1] for DDRE-33
    c = x_fake.size(-1)
    errs = []
    for ch in range(c):
        xf = x_fake[..., ch].reshape(-1)
        xr = x_real[..., ch].reshape(-1)
        h_f = torch.histc(xf, bins=bins, min=0.0, max=1.0)
        h_r = torch.histc(xr, bins=bins, min=0.0, max=1.0)
        h_f = h_f / (h_f.sum() + eps)
        h_r = h_r / (h_r.sum() + eps)
        errs.append(torch.abs(h_f - h_r).mean())
    return torch.stack(errs).mean()


def _quantile_mae(x_fake, x_real, qs):
    # per-channel quantile MAE averaged across channels and quantiles
    c = x_fake.size(-1)
    errs = []
    for q in qs:
        qf = torch.quantile(x_fake.reshape(-1, c), q, dim=0)
        qr = torch.quantile(x_real.reshape(-1, c), q, dim=0)
        errs.append(torch.abs(qf - qr).mean())
    return torch.stack(errs).mean()


def _volatility_err(x_fake, x_real, eps=1e-6):
    # Volatility error: std and ramp std across channels
    xf = x_fake.float()
    xr = x_real.float()
    std_err = torch.abs(xf.std(dim=1, unbiased=False) - xr.std(dim=1, unbiased=False)).mean()
    if xf.size(1) > 1:
        df = xf[:, 1:] - xf[:, :-1]
        dr = xr[:, 1:] - xr[:, :-1]
        ramp_std_err = torch.abs(
            df.std(dim=1, unbiased=False) - dr.std(dim=1, unbiased=False)
        ).mean()
    else:
        ramp_std_err = xf.sum() * 0.0
    return std_err, ramp_std_err


def _corr_err(x_fake, x_real, eps=1e-6):
    c = x_fake.size(-1)
    xf = x_fake.reshape(-1, c).float()
    xr = x_real.reshape(-1, c).float()
    xf = xf - xf.mean(dim=0, keepdim=True)
    xr = xr - xr.mean(dim=0, keepdim=True)
    cov_f = (xf.transpose(0, 1) @ xf) / max(xf.size(0) - 1, 1)
    cov_r = (xr.transpose(0, 1) @ xr) / max(xr.size(0) - 1, 1)
    std_f = torch.sqrt(torch.clamp(torch.diag(cov_f), min=eps))
    std_r = torch.sqrt(torch.clamp(torch.diag(cov_r), min=eps))
    corr_f = cov_f / (std_f[:, None] * std_f[None, :] + eps)
    corr_r = cov_r / (std_r[:, None] * std_r[None, :] + eps)
    return torch.mean(torch.abs(corr_f - corr_r))


def _extreme_weights(data_np, ch_indices=None, q=0.95, alpha=0.3):
    # data_np: (N, T, C)
    n = data_np.shape[0]
    q = float(q)
    q = min(max(q, 0.5), 0.999)
    if ch_indices is None:
        ch_indices = [0]
    cmax = data_np.shape[-1] - 1
    ch_indices = [int(max(0, min(int(ch), cmax))) for ch in ch_indices]
    if not ch_indices:
        ch_indices = [0]
    risks = np.zeros(n, dtype=np.float32)
    q_low = 1.0 - q
    for i in range(n):
        ch_risks = []
        low_masks = []
        for ch in ch_indices:
            s = data_np[i, :, ch]
            # Two-sided tail proxy in [0,1]-normalized data:
            # emphasize both very high and very low extremes.
            q_hi = float(np.quantile(s, q))
            q_lo = float(np.quantile(s, q_low))
            tail = max(q_hi, 1.0 - q_lo)
            if s.shape[0] > 1:
                rs = np.abs(np.diff(s))
                rq = float(np.quantile(rs, q))
            else:
                rq = 0.0
            ch_risks.append(0.6 * tail + 0.4 * rq)
            low_masks.append(s <= q_lo)
        if len(ch_risks) >= 2:
            joint_low = float(np.mean(np.logical_and.reduce(low_masks)))
            risks[i] = 0.9 * float(np.mean(ch_risks)) + 0.1 * joint_low
        else:
            risks[i] = float(ch_risks[0])
    order = np.argsort(np.argsort(risks))
    rank_norm = order.astype(np.float32) / max(n - 1, 1)
    w = 0.7 + float(alpha) * rank_norm
    return np.clip(w, 1e-3, None).astype(np.float32)


def _parse_extreme_channels(spec, num_channels, dataset, ddre33_mode):
    s = str(spec).strip().lower()
    if s in ("", "auto"):
        if num_channels <= 1:
            return [0]
        if dataset == "ddre33":
            if ddre33_mode in ("2ch_single", "2ch_pairs"):
                return [0, 1]
            # 4ch: include all renewable channels.
            return list(range(num_channels))
        # real default: first two channels (typically solar+wind)
        return [0, 1]

    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            idx = int(p)
        except ValueError as exc:
            raise ValueError(f"Invalid extreme_joint_channels token: {p}") from exc
        if idx < 0 or idx >= num_channels:
            raise ValueError(
                f"extreme_joint_channels index out of range: {idx}, channels={num_channels}"
            )
        out.append(idx)
    if not out:
        out = [0]
    return out

def _ks_stat(x_fake, x_real, bins=50, eps=1e-6):
    # Kolmogorov-Smirnov statistic via histogram CDF
    c = x_fake.size(-1)
    errs = []
    for ch in range(c):
        xf = x_fake[..., ch].reshape(-1)
        xr = x_real[..., ch].reshape(-1)
        h_f = torch.histc(xf, bins=bins, min=0.0, max=1.0)
        h_r = torch.histc(xr, bins=bins, min=0.0, max=1.0)
        cdf_f = torch.cumsum(h_f, dim=0) / (h_f.sum() + eps)
        cdf_r = torch.cumsum(h_r, dim=0) / (h_r.sum() + eps)
        errs.append(torch.max(torch.abs(cdf_f - cdf_r)))
    return torch.stack(errs).mean()


def _wasserstein_hist(x_fake, x_real, bins=50, eps=1e-6):
    # Approximate 1D Wasserstein distance via CDF L1 over histogram
    c = x_fake.size(-1)
    errs = []
    bin_width = 1.0 / max(bins, 1)
    for ch in range(c):
        xf = x_fake[..., ch].reshape(-1)
        xr = x_real[..., ch].reshape(-1)
        h_f = torch.histc(xf, bins=bins, min=0.0, max=1.0)
        h_r = torch.histc(xr, bins=bins, min=0.0, max=1.0)
        cdf_f = torch.cumsum(h_f, dim=0) / (h_f.sum() + eps)
        cdf_r = torch.cumsum(h_r, dim=0) / (h_r.sum() + eps)
        errs.append(torch.sum(torch.abs(cdf_f - cdf_r)) * bin_width)
    return torch.stack(errs).mean()


def _build_split_labels(cond_np, dataset, ddre33_mode, cond_onehot):
    if dataset != "ddre33" or not cond_onehot:
        return None
    cond_mean = cond_np if cond_np.ndim == 2 else cond_np.mean(axis=1)
    if ddre33_mode in ("2ch_pairs", "2ch_single"):
        if cond_mean.shape[1] < 10:
            return None
        pv = np.argmax(cond_mean[:, :6], axis=1)
        wind = np.argmax(cond_mean[:, 6:10], axis=1)
        return (pv * 10 + wind).astype(np.int64)
    if cond_mean.shape[1] < 20:
        return None
    pv18 = np.argmax(cond_mean[:, :6], axis=1)
    pv33 = np.argmax(cond_mean[:, 6:12], axis=1)
    wind22 = np.argmax(cond_mean[:, 12:16], axis=1)
    wind25 = np.argmax(cond_mean[:, 16:20], axis=1)
    return (pv18 + 6 * (pv33 + 6 * (wind22 + 4 * wind25))).astype(np.int64)


def _evaluate_generator(
    model,
    loader,
    device,
    z_dim,
    metric_bins,
    metric_quantiles,
    acf_lags,
    fixed_latents=None,
    use_amp=False,
    event_peak_temp=20.0,
    event_active_threshold=0.05,
    event_active_sharpness=30.0,
):
    hist_err_list = []
    q_err_list = []
    ks_list = []
    wass_list = []
    std_err_list = []
    ramp_std_err_list = []
    acf_err_list = []
    corr_err_list = []
    peak_evt_err_list = []
    ramp_evt_err_list = []
    active_evt_err_list = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for x, c, mask in loader:
            x = x.to(device, non_blocking=device.type == "cuda")
            c = c.to(device, non_blocking=device.type == "cuda")
            mask = mask.to(device, non_blocking=device.type == "cuda")
            if fixed_latents is not None:
                z = fixed_latents[offset : offset + x.size(0)].to(device)
            else:
                z = torch.randn(x.size(0), z_dim, device=device)
            offset += x.size(0)
            with torch.amp.autocast("cuda", enabled=use_amp):
                x_fake = model(z, c, mask)
            x_fake_f = x_fake.float()
            x_real_f = x.float()
            hist_err_list.append(float(_hist_l1(x_fake_f, x_real_f, bins=metric_bins).item()))
            q_err_list.append(float(_quantile_mae(x_fake_f, x_real_f, metric_quantiles).item()))
            ks_list.append(float(_ks_stat(x_fake_f, x_real_f, bins=metric_bins).item()))
            wass_list.append(float(_wasserstein_hist(x_fake_f, x_real_f, bins=metric_bins).item()))
            std_e, ramp_std_e = _volatility_err(x_fake_f, x_real_f)
            std_err_list.append(float(std_e.item()))
            ramp_std_err_list.append(float(ramp_std_e.item()))
            acf_err_list.append(float(acf_loss(x_fake_f, x_real_f, max_lag=acf_lags).item()))
            corr_err_list.append(float(_corr_err(x_fake_f, x_real_f).item()))
            peak_evt_err_list.append(float(peak_event_loss(x_fake_f, x_real_f, peak_temp=event_peak_temp).item()))
            ramp_evt_err_list.append(float(ramp_event_loss(x_fake_f, x_real_f).item()))
            active_evt_err_list.append(
                float(
                    active_ratio_loss(
                        x_fake_f,
                        x_real_f,
                        threshold=event_active_threshold,
                        sharpness=event_active_sharpness,
                    ).item()
                )
            )

    if not q_err_list:
        return None
    metrics = {
        "hist_err": float(np.mean(hist_err_list)),
        "q_err": float(np.mean(q_err_list)),
        "ks": float(np.mean(ks_list)),
        "w1": float(np.mean(wass_list)),
        "std_err": float(np.mean(std_err_list)),
        "ramp_std_err": float(np.mean(ramp_std_err_list)),
        "acf_err": float(np.mean(acf_err_list)),
        "corr_err": float(np.mean(corr_err_list)),
        "peak_evt_err": float(np.mean(peak_evt_err_list)),
        "ramp_evt_err": float(np.mean(ramp_evt_err_list)),
        "active_evt_err": float(np.mean(active_evt_err_list)),
    }
    metrics["model_score"] = (
        0.35 * metrics["q_err"]
        + 0.25 * metrics["acf_err"]
        + 0.2 * metrics["corr_err"]
        + 0.2 * metrics["ramp_std_err"]
    )
    return metrics

def main():
    args = parse_args()
    set_seed(args.seed)
    lambda_tail = args.lambda_tail if args.lambda_tailq is None else args.lambda_tailq

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.resolution == "hourly":
        default_resample = "H"
        default_seq_len = 24
    elif args.resolution == "15min":
        default_resample = "15T"
        default_seq_len = 96
    else:
        default_resample = "T"
        default_seq_len = 1440

    resample_rule = args.resample or default_resample
    seq_len = args.seq_len or default_seq_len
    res_label = "1min" if args.resolution == "minute" else args.resolution

    if args.dataset == "ddre33":
        x_cols, cond_cols = ddre33_cond_layout(
            args.ddre33_mode,
            args.cond_onehot,
            args.ddre33_date_cond,
            args.ddre33_curve_cond,
        )
    else:
        x_cols = [c.strip() for c in args.x_cols.split(",") if c.strip()]
        cond_cols = [c.strip() for c in args.cond_cols.split(",") if c.strip()]

    ch_weights = [float(v.strip()) for v in args.ch_weights.split(",") if v.strip()]
    if len(ch_weights) != len(x_cols):
        if args.dataset == "ddre33" and args.ch_weights == "1,1,1":
            ch_weights = [1.0] * len(x_cols)
        else:
            raise ValueError(f"ch_weights length {len(ch_weights)} must match channels {len(x_cols)}")

    scfg = ScenarioConfig(seq_len=seq_len, channels=len(x_cols), cond_dim=len(cond_cols))
    if args.dataset == "real":
        if not args.data_csv:
            raise ValueError("Real data CSV is required. Provide --data_csv.")
        data, cond, mask, x_min, x_max, cond_stats = load_real_dataset(
            args.data_csv,
            seq_len=scfg.seq_len,
            x_cols=x_cols,
            cond_cols=cond_cols,
            resample_rule=resample_rule,
            cond_agg=args.cond_agg,
            x_agg="sum",
            mask_source=args.mask_source,
            cond_norm=args.cond_norm,
        )
    else:
        need = [
            args.pv18_csv,
            args.pv33_csv,
            args.wind22_csv,
            args.wind25_csv,
            args.pv18_labels_csv,
            args.pv33_labels_csv,
            args.wind22_labels_csv,
            args.wind25_labels_csv,
        ]
        if not all(need):
            raise ValueError("DDRE-33 requires pv/wind CSVs and labels CSVs.")
        data, cond, mask, x_min, x_max, cond_stats = load_ddre33_dataset(
            pv18_csv=args.pv18_csv,
            pv33_csv=args.pv33_csv,
            wind22_csv=args.wind22_csv,
            wind25_csv=args.wind25_csv,
            pv18_labels_csv=args.pv18_labels_csv,
            pv33_labels_csv=args.pv33_labels_csv,
            wind22_labels_csv=args.wind22_labels_csv,
            wind25_labels_csv=args.wind25_labels_csv,
            seq_len=scfg.seq_len,
            resample_rule=resample_rule,
            one_hot=args.cond_onehot,
            mode=args.ddre33_mode,
            max_cols=args.max_cols,
            normalize=False,
            add_date_cond=args.ddre33_date_cond,
            static_cond=args.ddre33_static_cond,
            add_curve_cond=args.ddre33_curve_cond,
            curve_cond_norm=args.ddre33_curve_cond_norm,
        )
        if args.ddre33_curve_cond:
            print(
                "[info] ddre33 curve_cond enabled: appended summary features are added to cond only; "
                f"power sequences keep original DDRE-33 scale, curve_cond_norm={args.ddre33_curve_cond_norm}"
            )
            if args.ddre33_curve_cond_norm == "minmax":
                print(
                    "[warn] curve_cond_norm=minmax only re-normalizes appended curve control features. "
                    "If DDRE-33 source curves are already normalized and you want absolute feature semantics, "
                    "prefer --ddre33_curve_cond_norm none."
                )
        if args.ddre33_mode == "4ch":
            print(
                "[warn] ddre33 4ch mode is available, but the current channel-specific priors/losses are tuned "
                "mainly for paired PV/Wind experiments. For formal thesis results, prefer 2ch_single or 2ch_pairs."
            )
    if args.mask_mode == "none":
        mask = np.ones_like(data, dtype=np.float32)

    device = get_device()
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.train_metrics_every <= 0:
        print("[info] train_metrics_every<=0: expensive train metrics will be skipped and logged as NaN")
    if args.eval_every > 1:
        print(f"[info] eval_every={args.eval_every}: validation metrics refresh every {args.eval_every} epochs")

    split_labels = _build_split_labels(cond, args.dataset, args.ddre33_mode, args.cond_onehot)
    if split_labels is not None:
        print(f"[info] using stratified train/val split over {len(np.unique(split_labels))} label groups")
    idx_train, idx_val = split_train_val(len(data), args.val_ratio, labels=split_labels)
    if idx_val.size == 0:
        print("[warn] validation split is empty, falling back to random split")
        idx_train, idx_val = split_train_val(len(data), args.val_ratio)
    if args.cache_gpu and device.type == "cuda":
        data_t = torch.from_numpy(data).to(device, non_blocking=True)
        cond_t = torch.from_numpy(cond).to(device, non_blocking=True)
        mask_t = torch.from_numpy(mask).to(device, non_blocking=True)
        train_ds = RealDataset(data_t[idx_train], cond_t[idx_train], mask_t[idx_train])
        val_ds = RealDataset(data_t[idx_val], cond_t[idx_val], mask_t[idx_val])
        pin = False
        num_workers = 0
    else:
        train_ds = RealDataset(data[idx_train], cond[idx_train], mask[idx_train])
        val_ds = RealDataset(data[idx_val], cond[idx_val], mask[idx_val])
        pin = device.type == "cuda"
        num_workers = args.num_workers

    sampler = None
    if args.extreme_resample:
        risk_channels = _parse_extreme_channels(
            args.extreme_joint_channels, scfg.channels, args.dataset, args.ddre33_mode
        )
        print(f"[info] extreme resample channels: {risk_channels}")
        w_np = _extreme_weights(
            data[idx_train],
            ch_indices=risk_channels,
            q=args.extreme_q,
            alpha=args.extreme_alpha,
        )
        sampler = WeightedRandomSampler(
            torch.from_numpy(w_np).double(),
            num_samples=len(w_np),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
    )
    val_fixed_latents = torch.randn(len(val_ds), args.z_dim) if len(val_ds) > 0 else None

    G = Generator(
        seq_len=scfg.seq_len,
        cond_dim=scfg.cond_dim,
        z_dim=args.z_dim,
        model_dim=128,
        depth=4,
        heads=4,
        channels=scfg.channels,
        use_channel_mixer=not args.no_channel_mixer,
        use_baseline_residual=not args.no_baseline_residual,
    ).to(device)
    G_ema = None
    use_ema = 0.0 < float(args.ema_decay) < 1.0
    if use_ema:
        G_ema = copy.deepcopy(G).eval()
        for p in G_ema.parameters():
            p.requires_grad_(False)
    D = Discriminator(
        seq_len=scfg.seq_len,
        cond_dim=scfg.cond_dim,
        model_dim=args.d_model_dim,
        depth=args.d_depth,
        heads=args.d_heads,
        channels=scfg.channels,
        patch_scales=args.d_patch_scales,
        high_weight=args.d_high_weight,
    ).to(device)
    if args.torch_compile and hasattr(torch, "compile"):
        G = torch.compile(G)
        D = torch.compile(D)
    lr_g = args.lr_g if args.lr_g > 0 else args.lr
    lr_d = args.lr_d if args.lr_d > 0 else (2.0 * args.lr)
    g_opt = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.9))

    # adversarial training
    metrics_path = outdir / "metrics.csv"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
            if (
                "model_score" not in first
                or "val_model_score" not in first
                or "peak_evt_err" not in first
                or "val_peak_evt_err" not in first
            ):
                legacy = outdir / "metrics_legacy.csv"
                metrics_path.replace(legacy)
                print(f"[info] legacy metrics moved to {legacy}")
        except Exception:
            pass
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if scfg.channels == 2:
                writer.writerow(
                    [
                        "epoch",
                        "d_loss",
                        "g_loss",
                        "g_adv",
                        "g_tailq",
                        "pv_mean",
                        "wind_mean",
                        "hist_err",
                        "q_err",
                        "ks",
                        "w1",
                        "std_err",
                        "ramp_std_err",
                        "acf_err",
                        "corr_err",
                        "peak_evt_err",
                        "ramp_evt_err",
                        "active_evt_err",
                        "model_score",
                        "val_hist_err",
                        "val_q_err",
                        "val_ks",
                        "val_w1",
                        "val_std_err",
                        "val_ramp_std_err",
                        "val_acf_err",
                        "val_corr_err",
                        "val_peak_evt_err",
                        "val_ramp_evt_err",
                        "val_active_evt_err",
                        "val_model_score",
                    ]
                )
            elif scfg.channels == 3:
                writer.writerow(
                    [
                        "epoch",
                        "d_loss",
                        "g_loss",
                        "g_adv",
                        "g_tailq",
                        "solar_mean",
                        "wind_mean",
                        "load_mean",
                        "acf_err",
                        "corr_err",
                        "peak_evt_err",
                        "ramp_evt_err",
                        "active_evt_err",
                        "model_score",
                        "val_hist_err",
                        "val_q_err",
                        "val_ks",
                        "val_w1",
                        "val_std_err",
                        "val_ramp_std_err",
                        "val_acf_err",
                        "val_corr_err",
                        "val_peak_evt_err",
                        "val_ramp_evt_err",
                        "val_active_evt_err",
                        "val_model_score",
                    ]
                )
            else:
                writer.writerow(
                    [
                        "epoch",
                        "d_loss",
                        "g_loss",
                        "g_adv",
                        "g_tailq",
                        "pv_mean",
                        "wind_mean",
                        "load_e_mean",
                        "load_h_mean",
                        "acf_err",
                        "corr_err",
                        "peak_evt_err",
                        "ramp_evt_err",
                        "active_evt_err",
                        "model_score",
                        "val_hist_err",
                        "val_q_err",
                        "val_ks",
                        "val_w1",
                        "val_std_err",
                        "val_ramp_std_err",
                        "val_acf_err",
                        "val_corr_err",
                        "val_peak_evt_err",
                        "val_ramp_evt_err",
                        "val_active_evt_err",
                        "val_model_score",
                    ]
                )

    q_list = _parse_quantiles(args.metric_quantiles)
    tail_qs = _parse_quantiles(args.tail_qs) if args.tail_qs else [args.tail_q]
    best_score = float("inf")
    last_val_metrics = None

    def _build_checkpoint(best_metric_value):
        ckpt = {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "cfg": {
                "seq_len": scfg.seq_len,
                "channels": scfg.channels,
                "cond_dim": scfg.cond_dim,
                "z_dim": args.z_dim,
                "use_baseline_residual": not args.no_baseline_residual,
                "x_min": x_min.tolist(),
                "x_max": x_max.tolist(),
                "best_score": best_metric_value,
                "best_score_split": "val" if len(val_ds) > 0 else "train",
                "generator_state_preference": "G_ema" if G_ema is not None else "G",
                "train_params": {
                    "lr": args.lr,
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                    "d_steps": args.d_steps,
                    "lambda_gp": args.lambda_gp,
                    "lambda_tail": lambda_tail,
                    "lambda_stats": args.lambda_stats,
                    "lambda_roll": args.lambda_roll,
                    "lambda_acf": args.lambda_acf,
                    "lambda_corr": args.lambda_corr,
                    "lambda_peak_evt": args.lambda_peak_evt,
                    "lambda_ramp_evt": args.lambda_ramp_evt,
                    "lambda_active_evt": args.lambda_active_evt,
                    "lambda_wind_adv": args.lambda_wind_adv,
                    "tail_qs": tail_qs,
                    "ema_decay": float(args.ema_decay),
                    "ema_start_epoch": int(args.ema_start_epoch),
                    "val_ratio": float(args.val_ratio),
                    "event_peak_temp": float(args.event_peak_temp),
                    "event_active_threshold": float(args.event_active_threshold),
                    "event_active_sharpness": float(args.event_active_sharpness),
                },
                **{k: v.tolist() if hasattr(v, "tolist") else v for k, v in cond_stats.items()},
            },
        }
        if G_ema is not None:
            ckpt["G_ema"] = G_ema.state_dict()
        return ckpt

    for epoch in tqdm(
        range(args.epochs),
        desc="training",
        ncols=80,
        ascii=True,
        bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        g_losses = []
        d_losses = []
        g_adv_list = []
        g_tailq_list = []
        pv_list = []
        wind_list = []
        le_list = []
        lh_list = []
        load_list = []
        hist_err_list = []
        q_err_list = []
        ks_list = []
        wass_list = []
        std_err_list = []
        ramp_std_err_list = []
        acf_err_list = []
        corr_err_list = []
        peak_evt_err_list = []
        ramp_evt_err_list = []
        active_evt_err_list = []
        G.train()
        D.train()
        for batch_idx, (x, c, mask) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=pin)
            c = c.to(device, non_blocking=pin)
            mask = mask.to(device, non_blocking=pin)
            c_global = c if c.dim() == 2 else c.mean(dim=1)
            if args.debug_nan:
                def _stats(name, t):
                    t_f = t.float()
                    finite = torch.isfinite(t_f)
                    if not finite.all():
                        bad = (~finite).sum().item()
                        print(f"[debug] {name} has {bad} non-finite values")
                    print(
                        f"[debug] {name} min={t_f.min().item():.6f} max={t_f.max().item():.6f} "
                        f"mean={t_f.mean().item():.6f}"
                    )
                _stats("x", x)
                _stats("cond", c)
                _stats("mask", mask)
            if args.log_batch_every and batch_idx % args.log_batch_every == 0:
                print(
                    f"[batch {batch_idx}/{len(train_loader)}] "
                    f"x_min={x.min().item():.4f} x_max={x.max().item():.4f} x_mean={x.mean().item():.4f}"
                )

            # update D (NO AMP)
            for _ in range(args.d_steps):
                z = torch.randn(x.size(0), args.z_dim, device=device)
                with torch.no_grad():
                    x_fake = G(z, c, mask)
                x_real = x
                d_real = D(x_real, c_global)
                d_fake = D(x_fake, c_global)
                d_loss = wgan_d_loss(d_real, d_fake)
                if args.debug_nan:
                    if not torch.isfinite(d_real).all() or not torch.isfinite(d_fake).all():
                        print("[debug] non-finite d_real/d_fake")
                    if not torch.isfinite(d_loss):
                        print("[debug] non-finite d_loss")
                eps = torch.rand(x.size(0), 1, 1, device=device, dtype=x_real.dtype)
                x_hat = eps * x_real + (1 - eps) * x_fake
                x_hat = x_hat.requires_grad_(True)
                d_hat = D(x_hat, c_global)
                gp = gradient_penalty_wgan_scalar(d_hat, x_hat)
                d_loss_total = d_loss + args.lambda_gp * gp
                if not torch.isfinite(d_loss_total):
                    print("[warn] non-finite d_loss_total, skipping D step")
                    d_opt.zero_grad(set_to_none=True)
                    continue
                d_opt.zero_grad(set_to_none=True)
                d_loss_total.backward()
                d_opt.step()
                d_losses.append(float(d_loss_total.item()))

            # update G
            z = torch.randn(x.size(0), args.z_dim, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                x_fake = G(z, c, mask)
            with torch.amp.autocast("cuda", enabled=False):
                x_fake_f = x_fake.float()
                x_real_f = x.float()
                c_global_f = c_global.float()
                d_fake = D(x_fake_f, c_global_f)
                g_adv = wgan_g_loss(d_fake)
                # Tail loss: full sequence, multi-quantile
                g_tailq = tail_quantile_loss(x_fake_f, x_real_f, q=tail_qs, channel_weights=ch_weights)
                # PV stats (daylight only)
                if mask is not None and mask.numel() > 0:
                    day_mask = mask[..., 0] > 0.5
                    if day_mask.any():
                        xf_pv = x_fake_f[..., 0][day_mask]
                        xr_pv = x_real_f[..., 0][day_mask]
                        pv_stats = torch.abs(xf_pv.mean() - xr_pv.mean()) + torch.abs(
                            xf_pv.std(unbiased=False) - xr_pv.std(unbiased=False)
                        )
                    else:
                        pv_stats = x_fake_f.sum() * 0.0
                else:
                    pv_stats = x_fake_f.sum() * 0.0
                # Wind stats (full + ramp)
                if x_fake_f.size(-1) > 1:
                    wind_stats = channel_stats_loss(
                        x_fake_f,
                        x_real_f,
                        channels=[1],
                        include_ramp=True,
                    ) * args.lambda_wind_adv
                else:
                    wind_stats = x_fake_f.sum() * 0.0
                g_stats = pv_stats + wind_stats
                g_roll = (
                    rolling_stats_loss(x_fake_f, x_real_f, win=args.roll_win)
                    if args.lambda_roll > 0
                    else x_fake_f.sum() * 0.0
                )
                g_acf = (
                    acf_loss(x_fake_f, x_real_f, max_lag=args.acf_lags)
                    if args.lambda_acf > 0
                    else x_fake_f.sum() * 0.0
                )
                g_corr = (
                    corr_matrix_loss(x_fake_f, x_real_f)
                    if args.lambda_corr > 0
                    else x_fake_f.sum() * 0.0
                )
                g_peak_evt = (
                    peak_event_loss(x_fake_f, x_real_f, peak_temp=args.event_peak_temp)
                    if args.lambda_peak_evt > 0
                    else x_fake_f.sum() * 0.0
                )
                g_ramp_evt = (
                    ramp_event_loss(x_fake_f, x_real_f)
                    if args.lambda_ramp_evt > 0
                    else x_fake_f.sum() * 0.0
                )
                g_active_evt = (
                    active_ratio_loss(
                        x_fake_f,
                        x_real_f,
                        threshold=args.event_active_threshold,
                        sharpness=args.event_active_sharpness,
                    )
                    if args.lambda_active_evt > 0
                    else x_fake_f.sum() * 0.0
                )
                tail_weight = lambda_tail * min(1.0, (epoch + 1) / max(1.0, 0.3 * args.epochs))
                g_loss = (
                    g_adv
                    + tail_weight * g_tailq
                    + args.lambda_stats * g_stats
                    + args.lambda_roll * g_roll
                    + args.lambda_acf * g_acf
                    + args.lambda_corr * g_corr
                    + args.lambda_peak_evt * g_peak_evt
                    + args.lambda_ramp_evt * g_ramp_evt
                    + args.lambda_active_evt * g_active_evt
                )
            if args.debug_nan:
                if not torch.isfinite(x_fake).all():
                    print("[debug] non-finite x_fake")
                if not torch.isfinite(g_loss):
                    print("[debug] non-finite g_loss")
            if not torch.isfinite(g_loss):
                print("[warn] non-finite g_loss, skipping G step")
                g_opt.zero_grad(set_to_none=True)
                continue
            g_opt.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(g_opt)
            scaler.update()
            if G_ema is not None and epoch >= int(args.ema_start_epoch):
                update_ema_model(G_ema, G, float(args.ema_decay))
            g_losses.append(float(g_loss.item()))
            g_adv_list.append(float(g_adv.item()))
            g_tailq_list.append(float(g_tailq.item()))
            pv_list.append(x_fake_f[..., 0].mean().item())
            if scfg.channels > 1:
                wind_list.append(x_fake_f[..., 1].mean().item())
            # distribution error metrics on batch (DDRE-33 data is already in [0,1])
            if args.train_metrics_every > 0 and batch_idx % args.train_metrics_every == 0:
                with torch.no_grad():
                    hist_err_list.append(float(_hist_l1(x_fake_f, x_real_f, bins=args.metric_bins).item()))
                    q_err_list.append(float(_quantile_mae(x_fake_f, x_real_f, q_list).item()))
                    ks_list.append(float(_ks_stat(x_fake_f, x_real_f, bins=args.metric_bins).item()))
                    wass_list.append(float(_wasserstein_hist(x_fake_f, x_real_f, bins=args.metric_bins).item()))
                    std_e, ramp_std_e = _volatility_err(x_fake_f, x_real_f)
                    std_err_list.append(float(std_e.item()))
                    ramp_std_err_list.append(float(ramp_std_e.item()))
                    acf_err_list.append(float(acf_loss(x_fake_f, x_real_f, max_lag=args.acf_lags).item()))
                    corr_err_list.append(float(_corr_err(x_fake_f, x_real_f).item()))
                    peak_evt_err_list.append(
                        float(peak_event_loss(x_fake_f, x_real_f, peak_temp=args.event_peak_temp).item())
                    )
                    ramp_evt_err_list.append(float(ramp_event_loss(x_fake_f, x_real_f).item()))
                    active_evt_err_list.append(
                        float(
                            active_ratio_loss(
                                x_fake_f,
                                x_real_f,
                                threshold=args.event_active_threshold,
                                sharpness=args.event_active_sharpness,
                            ).item()
                        )
                    )
            if scfg.channels == 3:
                load_list.append(x_fake_f[..., 2].mean().item())
            elif scfg.channels >= 4:
                le_list.append(x_fake_f[..., 2].mean().item())
                lh_list.append(x_fake_f[..., 3].mean().item())

        d_loss_m = float(np.mean(d_losses))
        g_loss_m = float(np.mean(g_losses))
        g_adv_m = float(np.mean(g_adv_list))
        g_tailq_m = float(np.mean(g_tailq_list))
        pv_m = float(np.mean(pv_list))
        wind_m = float(np.mean(wind_list)) if wind_list else 0.0
        if scfg.channels == 3:
            load_m = float(np.mean(load_list))
        elif scfg.channels >= 4:
            le_m = float(np.mean(le_list))
            lh_m = float(np.mean(lh_list))

        hist_err_m = _safe_mean(hist_err_list)
        q_err_m = _safe_mean(q_err_list)
        ks_m = _safe_mean(ks_list)
        wass_m = _safe_mean(wass_list)
        std_err_m = _safe_mean(std_err_list)
        ramp_std_err_m = _safe_mean(ramp_std_err_list)
        acf_err_m = _safe_mean(acf_err_list)
        corr_err_m = _safe_mean(corr_err_list)
        peak_evt_err_m = _safe_mean(peak_evt_err_list)
        ramp_evt_err_m = _safe_mean(ramp_evt_err_list)
        active_evt_err_m = _safe_mean(active_evt_err_list)
        model_score = (
            0.35 * q_err_m + 0.25 * acf_err_m + 0.2 * corr_err_m + 0.2 * ramp_std_err_m
            if hist_err_list
            else float("nan")
        )
        run_eval = args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1)
        if run_eval:
            last_val_metrics = _evaluate_generator(
                G_ema if G_ema is not None else G,
                val_loader,
                device=device,
                z_dim=args.z_dim,
                metric_bins=args.metric_bins,
                metric_quantiles=q_list,
                acf_lags=args.acf_lags,
                fixed_latents=val_fixed_latents,
                use_amp=use_amp,
                event_peak_temp=args.event_peak_temp,
                event_active_threshold=args.event_active_threshold,
                event_active_sharpness=args.event_active_sharpness,
            )
        val_metrics = last_val_metrics
        val_hist_err_m = val_metrics["hist_err"] if val_metrics is not None else float("nan")
        val_q_err_m = val_metrics["q_err"] if val_metrics is not None else float("nan")
        val_ks_m = val_metrics["ks"] if val_metrics is not None else float("nan")
        val_wass_m = val_metrics["w1"] if val_metrics is not None else float("nan")
        val_std_err_m = val_metrics["std_err"] if val_metrics is not None else float("nan")
        val_ramp_std_err_m = val_metrics["ramp_std_err"] if val_metrics is not None else float("nan")
        val_acf_err_m = val_metrics["acf_err"] if val_metrics is not None else float("nan")
        val_corr_err_m = val_metrics["corr_err"] if val_metrics is not None else float("nan")
        val_peak_evt_err_m = val_metrics["peak_evt_err"] if val_metrics is not None else float("nan")
        val_ramp_evt_err_m = val_metrics["ramp_evt_err"] if val_metrics is not None else float("nan")
        val_active_evt_err_m = val_metrics["active_evt_err"] if val_metrics is not None else float("nan")
        val_model_score = val_metrics["model_score"] if val_metrics is not None else model_score
        best_metric = val_model_score if val_metrics is not None else model_score

        if scfg.channels == 2:
            print(
                f"[epoch {epoch+1}/{args.epochs}] "
                f"d_loss={d_loss_m:.4f} g_loss={g_loss_m:.4f} "
                f"pv={pv_m:.4f} wind={wind_m:.4f} "
                f"hist_err={hist_err_m:.4f} q_err={q_err_m:.4f} "
                f"ks={ks_m:.4f} w1={wass_m:.4f} "
                f"std_err={std_err_m:.4f} ramp_std_err={ramp_std_err_m:.4f} "
                f"acf_err={acf_err_m:.4f} corr_err={corr_err_m:.4f} "
                f"peak_evt={peak_evt_err_m:.4f} ramp_evt={ramp_evt_err_m:.4f} active_evt={active_evt_err_m:.4f} "
                f"score={model_score:.4f} "
                f"val_score={val_model_score:.4f}"
            )
        elif scfg.channels == 3:
            print(
                f"[epoch {epoch+1}/{args.epochs}] "
                f"d_loss={d_loss_m:.4f} g_loss={g_loss_m:.4f} "
                f"solar={pv_m:.4f} wind={wind_m:.4f} load={load_m:.4f} "
                f"acf_err={acf_err_m:.4f} corr_err={corr_err_m:.4f} "
                f"peak_evt={peak_evt_err_m:.4f} ramp_evt={ramp_evt_err_m:.4f} active_evt={active_evt_err_m:.4f} "
                f"score={model_score:.4f} "
                f"val_score={val_model_score:.4f}"
            )
        else:
            print(
                f"[epoch {epoch+1}/{args.epochs}] "
                f"d_loss={d_loss_m:.4f} g_loss={g_loss_m:.4f} "
                f"pv={pv_m:.4f} wind={wind_m:.4f} le={le_m:.4f} lh={lh_m:.4f} "
                f"acf_err={acf_err_m:.4f} corr_err={corr_err_m:.4f} "
                f"peak_evt={peak_evt_err_m:.4f} ramp_evt={ramp_evt_err_m:.4f} active_evt={active_evt_err_m:.4f} "
                f"score={model_score:.4f} "
                f"val_score={val_model_score:.4f}"
            )
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if scfg.channels == 2:
                writer.writerow(
                    [
                        epoch + 1,
                        d_loss_m,
                        g_loss_m,
                        g_adv_m,
                        g_tailq_m,
                        pv_m,
                        wind_m,
                        hist_err_m,
                        q_err_m,
                        ks_m,
                        wass_m,
                        std_err_m,
                        ramp_std_err_m,
                        acf_err_m,
                        corr_err_m,
                        peak_evt_err_m,
                        ramp_evt_err_m,
                        active_evt_err_m,
                        model_score,
                        val_hist_err_m,
                        val_q_err_m,
                        val_ks_m,
                        val_wass_m,
                        val_std_err_m,
                        val_ramp_std_err_m,
                        val_acf_err_m,
                        val_corr_err_m,
                        val_peak_evt_err_m,
                        val_ramp_evt_err_m,
                        val_active_evt_err_m,
                        val_model_score,
                    ]
                )
            elif scfg.channels == 3:
                writer.writerow(
                    [
                        epoch + 1,
                        d_loss_m,
                        g_loss_m,
                        g_adv_m,
                        g_tailq_m,
                        pv_m,
                        wind_m,
                        load_m,
                        acf_err_m,
                        corr_err_m,
                        peak_evt_err_m,
                        ramp_evt_err_m,
                        active_evt_err_m,
                        model_score,
                        val_hist_err_m,
                        val_q_err_m,
                        val_ks_m,
                        val_wass_m,
                        val_std_err_m,
                        val_ramp_std_err_m,
                        val_acf_err_m,
                        val_corr_err_m,
                        val_peak_evt_err_m,
                        val_ramp_evt_err_m,
                        val_active_evt_err_m,
                        val_model_score,
                    ]
                )
            else:
                writer.writerow(
                    [
                        epoch + 1,
                        d_loss_m,
                        g_loss_m,
                        g_adv_m,
                        g_tailq_m,
                        pv_m,
                        wind_m,
                        le_m,
                        lh_m,
                        acf_err_m,
                        corr_err_m,
                        peak_evt_err_m,
                        ramp_evt_err_m,
                        active_evt_err_m,
                        model_score,
                        val_hist_err_m,
                        val_q_err_m,
                        val_ks_m,
                        val_wass_m,
                        val_std_err_m,
                        val_ramp_std_err_m,
                        val_acf_err_m,
                        val_corr_err_m,
                        val_peak_evt_err_m,
                        val_ramp_evt_err_m,
                        val_active_evt_err_m,
                        val_model_score,
                    ]
                )

        if best_metric < best_score:
            best_score = best_metric
            best_ckpt = _build_checkpoint(best_score)
            best_path = outdir / f"pcm_gan_{res_label}_best.pt"
            torch.save(best_ckpt, best_path)
            print(f"saved best checkpoint {best_path} score={best_score:.6f}")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            ckpt = _build_checkpoint(best_score)
            ckpt_path = outdir / f"pcm_gan_{res_label}_epoch_{epoch+1:04d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"saved checkpoint {ckpt_path}")

    ckpt = _build_checkpoint(best_score)
    final_path = outdir / f"pcm_gan_{res_label}.pt"
    torch.save(ckpt, final_path)
    print(f"saved to {final_path}")


if __name__ == "__main__":
    main()
