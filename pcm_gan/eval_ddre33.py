import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .data_synth import load_ddre33_dataset
from .losses import rolling_stats_loss, acf_loss
from .models import Generator
from .utils import ScenarioConfig, get_generator_state_dict, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DDRE-33 GAN with fixed conditions.")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--ckpt_dir", type=str, default="", help="Directory with checkpoints.")
    p.add_argument(
        "--ckpt_glob",
        type=str,
        default="pcm_gan_15min_epoch_*.pt",
        help="Glob pattern inside ckpt_dir.",
    )
    p.add_argument("--pv18_csv", type=str, required=True)
    p.add_argument("--pv33_csv", type=str, required=True)
    p.add_argument("--wind22_csv", type=str, required=True)
    p.add_argument("--wind25_csv", type=str, required=True)
    p.add_argument("--pv18_labels_csv", type=str, required=True)
    p.add_argument("--pv33_labels_csv", type=str, required=True)
    p.add_argument("--wind22_labels_csv", type=str, required=True)
    p.add_argument("--wind25_labels_csv", type=str, required=True)
    p.add_argument("--ddre33_mode", type=str, default="2ch_single", choices=["4ch", "2ch_pairs", "2ch_single"])
    p.add_argument("--cond_onehot", action="store_true")
    p.add_argument("--ddre33_date_cond", action="store_true")
    p.add_argument("--ddre33_static_cond", action="store_true")
    p.add_argument("--ddre33_curve_cond", action="store_true")
    p.add_argument("--ddre33_curve_cond_norm", type=str, default="none", choices=["none", "minmax"])
    p.add_argument("--max_cols", type=int, default=0)
    p.add_argument("--num", type=int, default=64, help="Number of samples to evaluate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pv_climate", type=int, default=-1, help="Filter PV climate label (0-5).")
    p.add_argument("--wind_climate", type=int, default=-1, help="Filter wind climate label (0-3).")
    p.add_argument("--date_month", type=int, default=0, help="Month for date filtering (1-12).")
    p.add_argument("--date_day", type=int, default=0, help="Day for date filtering (1-31).")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--quantiles", type=str, default="0.1,0.5,0.9,0.95,0.99")
    p.add_argument("--out_csv", type=str, default="")
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


def _hist_l1(x_fake, x_real, bins=20, eps=1e-6):
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
    c = x_fake.size(-1)
    errs = []
    for q in qs:
        qf = torch.quantile(x_fake.reshape(-1, c), q, dim=0)
        qr = torch.quantile(x_real.reshape(-1, c), q, dim=0)
        errs.append(torch.abs(qf - qr).mean())
    return torch.stack(errs).mean()


def _ks_stat(x_fake, x_real, bins=50, eps=1e-6):
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


def _volatility_err(x_fake, x_real):
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


def _date_to_sin_cos(month, day):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    m = int(month)
    d = int(day)
    if m < 1 or m > 12:
        raise ValueError("month must be 1-12")
    if d < 1 or d > days_in_month[m - 1]:
        raise ValueError("invalid day for month")
    doy = sum(days_in_month[: m - 1]) + d
    angle = 2.0 * np.pi * ((doy - 1) / 365.0)
    return float(np.sin(angle)), float(np.cos(angle))


def main():
    args = parse_args()
    set_seed(args.seed)

    data, cond, mask, x_min, x_max, cond_stats = load_ddre33_dataset(
        pv18_csv=args.pv18_csv,
        pv33_csv=args.pv33_csv,
        wind22_csv=args.wind22_csv,
        wind25_csv=args.wind25_csv,
        pv18_labels_csv=args.pv18_labels_csv,
        pv33_labels_csv=args.pv33_labels_csv,
        wind22_labels_csv=args.wind22_labels_csv,
        wind25_labels_csv=args.wind25_labels_csv,
        seq_len=96,
        resample_rule="15min",
        one_hot=args.cond_onehot,
        mode=args.ddre33_mode,
        max_cols=args.max_cols,
        normalize=False,
        add_date_cond=args.ddre33_date_cond,
        static_cond=args.ddre33_static_cond,
        add_curve_cond=args.ddre33_curve_cond,
        curve_cond_norm=args.ddre33_curve_cond_norm,
    )

    cond_arr = cond
    if cond_arr.ndim == 3:
        cond_mean = cond_arr.mean(axis=1)
    else:
        cond_mean = cond_arr

    # filter by climate if requested
    idx = np.arange(len(cond_mean))
    if args.cond_onehot and cond_mean.shape[1] >= 10:
        pv_labels = np.argmax(cond_mean[:, :6], axis=1)
        wind_labels = np.argmax(cond_mean[:, 6:10], axis=1)
        if args.pv_climate >= 0:
            idx = idx[pv_labels[idx] == args.pv_climate]
        if args.wind_climate >= 0:
            idx = idx[wind_labels[idx] == args.wind_climate]

    # filter by date (using sin/cos if enabled)
    if args.ddre33_date_cond and cond_mean.shape[1] >= 12 and args.date_month and args.date_day:
        ds, dc = _date_to_sin_cos(args.date_month, args.date_day)
        date_feat = cond_mean[:, 10:12]
        d2 = (date_feat[:, 0] - ds) ** 2 + (date_feat[:, 1] - dc) ** 2
        idx = idx[np.argsort(d2[idx])]

    if idx.size == 0:
        raise ValueError("No samples match the requested filters.")

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(idx, size=min(args.num, idx.size), replace=False)

    real = torch.from_numpy(data[sel])
    cond_sel = cond_arr[sel]
    if cond_sel.ndim == 2:
        cond_sel = np.repeat(cond_sel[:, None, :], real.size(1), axis=1)
    cond_t = torch.from_numpy(cond_sel)
    mask_t = torch.from_numpy(mask[sel])

    ckpts = []
    if args.ckpt:
        ckpts = [Path(args.ckpt)]
    elif args.ckpt_dir:
        ckpts = sorted(Path(args.ckpt_dir).glob(args.ckpt_glob))
    else:
        raise ValueError("Provide --ckpt or --ckpt_dir.")

    def _epoch_from_name(name: str):
        m = None
        try:
            import re as _re
            m = _re.search(r"epoch_(\\d+)", name)
        except Exception:
            m = None
        return int(m.group(1)) if m else -1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qs = _parse_quantiles(args.quantiles)
    rows = []
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["cfg"]
        scfg = ScenarioConfig(seq_len=cfg["seq_len"], channels=cfg["channels"], cond_dim=cfg["cond_dim"])
        if scfg.seq_len != real.size(1) or scfg.channels != real.size(2):
            raise ValueError("ckpt shape mismatch with data")

        G = Generator(
            seq_len=scfg.seq_len,
            cond_dim=scfg.cond_dim,
            z_dim=cfg["z_dim"],
            model_dim=128,
            depth=4,
            heads=4,
            channels=scfg.channels,
            use_baseline_residual=bool(cfg.get("use_baseline_residual", False)),
        ).to(device)
        state_dict, _ = get_generator_state_dict(
            ckpt, prefer_ema=cfg.get("generator_state_preference", "G_ema") == "G_ema"
        )
        G.load_state_dict(state_dict, strict=False)
        G.eval()

        z = torch.randn(real.size(0), cfg["z_dim"], device=device)
        with torch.no_grad():
            fake = G(z, cond_t.to(device), mask_t.to(device)).cpu()

        hist_err = _hist_l1(fake, real, bins=args.bins).item()
        q_err = _quantile_mae(fake, real, qs).item()
        ks = _ks_stat(fake, real, bins=args.bins).item()
        w1 = _wasserstein_hist(fake, real, bins=args.bins).item()
        std_err, ramp_std_err = _volatility_err(fake, real)
        roll_err = rolling_stats_loss(fake, real).item()
        acf_err = acf_loss(fake, real, max_lag=6).item()
        mean_err = torch.abs(fake.mean(dim=1) - real.mean(dim=1)).mean().item()

        row = {
            "ckpt": ckpt_path.name,
            "epoch": _epoch_from_name(ckpt_path.name),
            "num": int(real.size(0)),
            "hist_err": hist_err,
            "q_err": q_err,
            "ks": ks,
            "w1": w1,
            "mean_err": mean_err,
            "std_err": float(std_err.item()),
            "ramp_std_err": float(ramp_std_err.item()),
            "roll_err": roll_err,
            "acf_err": acf_err,
        }
        rows.append(row)

        print(f"== {ckpt_path.name} ==")
        for k, v in row.items():
            if k in ("ckpt", "epoch", "num"):
                print(f"{k}: {v}")
            else:
                print(f"{k}: {v:.6f}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()
