import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_synth import ddre33_cond_layout, load_ddre33_dataset, load_real_dataset
from .models import Generator
from .utils import ScenarioConfig, get_device, get_generator_state_dict, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Generate fixed evaluation report (plots + CSV).")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, default="pcm_gan_eval_report")
    p.add_argument("--dataset", type=str, default="real", choices=["real", "ddre33"])
    p.add_argument("--num", type=int, default=128, help="Number of samples to evaluate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--denorm", action="store_true", help="Denormalize outputs for real dataset.")

    # real-data args
    p.add_argument("--data_csv", type=str, default="")
    p.add_argument(
        "--x_cols",
        type=str,
        default="solar_power,wind_power,load_power",
        help="Comma-separated columns for x.",
    )
    p.add_argument(
        "--cond_cols",
        type=str,
        default="DHI,DNI,GHI,Dew Point,Solar Zenith Angle,Wind Speed,Relative Humidity,Temperature",
        help="Comma-separated columns for cond.",
    )
    p.add_argument("--resolution", type=str, default="hourly", choices=["hourly", "15min", "minute"])
    p.add_argument("--resample", type=str, default="")
    p.add_argument("--seq_len", type=int, default=0)
    p.add_argument("--cond_agg", type=str, default="mean", choices=["mean", "max", "min"])
    p.add_argument("--mask_source", type=str, default="ghi", choices=["solar", "ghi"])
    p.add_argument("--cond_norm", type=str, default="none", choices=["none", "minmax", "zscore"])

    # DDRE-33 args
    p.add_argument("--pv18_csv", type=str, default="")
    p.add_argument("--pv33_csv", type=str, default="")
    p.add_argument("--wind22_csv", type=str, default="")
    p.add_argument("--wind25_csv", type=str, default="")
    p.add_argument("--pv18_labels_csv", type=str, default="")
    p.add_argument("--pv33_labels_csv", type=str, default="")
    p.add_argument("--wind22_labels_csv", type=str, default="")
    p.add_argument("--wind25_labels_csv", type=str, default="")
    p.add_argument("--cond_onehot", action="store_true")
    p.add_argument("--ddre33_date_cond", action="store_true")
    p.add_argument("--ddre33_static_cond", action="store_true")
    p.add_argument("--ddre33_curve_cond", action="store_true")
    p.add_argument("--ddre33_curve_cond_norm", type=str, default="none", choices=["none", "minmax"])
    p.add_argument("--ddre33_mode", type=str, default="2ch_single", choices=["4ch", "2ch_pairs", "2ch_single"])
    p.add_argument("--max_cols", type=int, default=0)

    p.add_argument("--acf_lags", type=int, default=24)
    p.add_argument("--quantiles", type=str, default="0.9,0.95,0.99")
    return p.parse_args()


def _parse_quantiles(s: str):
    qs = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        q = float(token)
        if not (0.0 < q < 1.0):
            raise ValueError(f"quantile must be in (0,1), got {q}")
        qs.append(q)
    return qs if qs else [0.95]


def _default_resolution(resolution: str):
    if resolution == "hourly":
        return "H", 24
    if resolution == "15min":
        return "15T", 96
    return "T", 1440


def _acf_1d(x: np.ndarray, max_lag: int):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom <= 1e-12:
        return np.zeros(max_lag, dtype=np.float64)
    vals = []
    n = x.shape[0]
    for lag in range(1, max_lag + 1):
        if lag >= n:
            vals.append(0.0)
        else:
            vals.append(float(np.sum(x[:-lag] * x[lag:]) / denom))
    return np.array(vals, dtype=np.float64)


def _safe_corrcoef(arr_2d: np.ndarray):
    # arr_2d: (N, C)
    c = arr_2d.shape[1]
    out = np.eye(c, dtype=np.float64)
    for i in range(c):
        for j in range(i + 1, c):
            xi = arr_2d[:, i]
            xj = arr_2d[:, j]
            si = float(np.std(xi))
            sj = float(np.std(xj))
            if si < 1e-12 or sj < 1e-12:
                v = 0.0
            else:
                v = float(np.corrcoef(xi, xj)[0, 1])
            out[i, j] = v
            out[j, i] = v
    return out


def _plot_distribution(real, fake, names, out_png):
    c = real.shape[-1]
    fig, axes = plt.subplots(c, 1, figsize=(8, 3 * c), squeeze=False)
    for ch in range(c):
        ax = axes[ch, 0]
        xr = real[..., ch].reshape(-1)
        xf = fake[..., ch].reshape(-1)
        lo = min(float(xr.min()), float(xf.min()))
        hi = max(float(xr.max()), float(xf.max()))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 50)
        ax.hist(xr, bins=bins, alpha=0.5, density=True, label="real")
        ax.hist(xf, bins=bins, alpha=0.5, density=True, label="fake")
        ax.set_title(f"Distribution - {names[ch]}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_temporal(real, fake, names, out_png):
    c = real.shape[-1]
    t = np.arange(real.shape[1])
    fig, axes = plt.subplots(c, 1, figsize=(10, 3 * c), squeeze=False)
    for ch in range(c):
        ax = axes[ch, 0]
        r = real[..., ch]
        f = fake[..., ch]
        r_mean = r.mean(axis=0)
        f_mean = f.mean(axis=0)
        r_p10, r_p90 = np.quantile(r, [0.1, 0.9], axis=0)
        f_p10, f_p90 = np.quantile(f, [0.1, 0.9], axis=0)
        ax.plot(t, r_mean, label="real_mean")
        ax.plot(t, f_mean, label="fake_mean")
        ax.fill_between(t, r_p10, r_p90, alpha=0.2)
        ax.fill_between(t, f_p10, f_p90, alpha=0.2)
        ax.set_title(f"Temporal Profile - {names[ch]}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_correlation(real, fake, names, out_png):
    rf = real.reshape(-1, real.shape[-1])
    ff = fake.reshape(-1, fake.shape[-1])
    corr_r = _safe_corrcoef(rf)
    corr_f = _safe_corrcoef(ff)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(corr_r, vmin=-1, vmax=1, cmap="coolwarm")
    axes[0].set_title("Real Correlation")
    axes[1].imshow(corr_f, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_title("Fake Correlation")
    for ax in axes:
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
    fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return corr_r, corr_f


def _plot_volatility_and_acf(real, fake, names, max_lag, out_png):
    c = real.shape[-1]
    std_r = np.std(real, axis=(0, 1))
    std_f = np.std(fake, axis=(0, 1))
    dr = np.diff(real, axis=1)
    df = np.diff(fake, axis=1)
    ramp_std_r = np.std(dr, axis=(0, 1))
    ramp_std_f = np.std(df, axis=(0, 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(c)
    w = 0.35
    axes[0].bar(x - w / 2, std_r, width=w, label="real_std")
    axes[0].bar(x + w / 2, std_f, width=w, label="fake_std")
    axes[0].bar(x - w / 2, ramp_std_r, width=w, bottom=std_r * 0.0, alpha=0.35, label="real_ramp_std")
    axes[0].bar(x + w / 2, ramp_std_f, width=w, bottom=std_f * 0.0, alpha=0.35, label="fake_ramp_std")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_title("Volatility")
    axes[0].legend()

    lags = np.arange(1, max_lag + 1)
    for ch in range(c):
        acf_r = _acf_1d(real[..., ch].reshape(-1), max_lag)
        acf_f = _acf_1d(fake[..., ch].reshape(-1), max_lag)
        axes[1].plot(lags, acf_r, label=f"{names[ch]}_real")
        axes[1].plot(lags, acf_f, linestyle="--", label=f"{names[ch]}_fake")
    axes[1].set_title("ACF")
    axes[1].set_xlabel("Lag")
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return std_r, std_f, ramp_std_r, ramp_std_f


def _plot_extreme_quantiles(real, fake, names, qs, out_png):
    c = real.shape[-1]
    fig, axes = plt.subplots(c, 1, figsize=(8, 3 * c), squeeze=False)
    qvals_r = np.quantile(real.reshape(-1, c), qs, axis=0)
    qvals_f = np.quantile(fake.reshape(-1, c), qs, axis=0)
    x = np.arange(len(qs))
    for ch in range(c):
        ax = axes[ch, 0]
        ax.plot(x, qvals_r[:, ch], marker="o", label="real")
        ax.plot(x, qvals_f[:, ch], marker="o", label="fake")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{q:.2f}" for q in qs])
        ax.set_title(f"Extreme Quantiles - {names[ch]}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return qvals_r, qvals_f


def main():
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    resample_default, seq_default = _default_resolution(args.resolution)
    resample_rule = args.resample or resample_default
    seq_len = args.seq_len or seq_default

    if args.dataset == "ddre33":
        x_names, cond_cols = ddre33_cond_layout(
            args.ddre33_mode,
            args.cond_onehot,
            args.ddre33_date_cond,
            args.ddre33_curve_cond,
        )
        real, cond, mask, _, _, _ = load_ddre33_dataset(
            pv18_csv=args.pv18_csv,
            pv33_csv=args.pv33_csv,
            wind22_csv=args.wind22_csv,
            wind25_csv=args.wind25_csv,
            pv18_labels_csv=args.pv18_labels_csv,
            pv33_labels_csv=args.pv33_labels_csv,
            wind22_labels_csv=args.wind22_labels_csv,
            wind25_labels_csv=args.wind25_labels_csv,
            seq_len=seq_len,
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
    else:
        x_names = [v.strip() for v in args.x_cols.split(",") if v.strip()]
        cond_cols = [v.strip() for v in args.cond_cols.split(",") if v.strip()]
        if not args.data_csv:
            raise ValueError("--data_csv is required when --dataset real")
        real, cond, mask, _, _, _ = load_real_dataset(
            args.data_csv,
            seq_len=seq_len,
            x_cols=x_names,
            cond_cols=cond_cols,
            resample_rule=resample_rule,
            cond_agg=args.cond_agg,
            x_agg="sum",
            mask_source=args.mask_source,
            cond_norm=args.cond_norm,
        )

    rng = np.random.default_rng(args.seed)
    n = min(int(args.num), len(real))
    idx = rng.choice(len(real), size=n, replace=False)
    real = real[idx]
    cond_sel = cond[idx]
    mask_sel = mask[idx]

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    scfg = ScenarioConfig(seq_len=cfg["seq_len"], channels=cfg["channels"], cond_dim=cfg["cond_dim"])
    if scfg.seq_len != real.shape[1]:
        raise ValueError(f"seq_len mismatch: ckpt={scfg.seq_len}, data={real.shape[1]}")
    if scfg.channels != real.shape[2]:
        raise ValueError(f"channel mismatch: ckpt={scfg.channels}, data={real.shape[2]}")
    if scfg.cond_dim != len(cond_cols):
        raise ValueError(f"cond_dim mismatch: ckpt={scfg.cond_dim}, cond={len(cond_cols)}")

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

    z = torch.randn(n, cfg["z_dim"], device=device)
    c = torch.from_numpy(cond_sel).to(device)
    m = torch.from_numpy(mask_sel).to(device)
    with torch.no_grad():
        fake = G(z, c, m).cpu().numpy()

    if args.denorm and args.dataset != "ddre33" and "x_min" in cfg and "x_max" in cfg:
        x_min = np.array(cfg["x_min"], dtype=np.float32)
        x_max = np.array(cfg["x_max"], dtype=np.float32)
        real = real * (x_max[None, None, :] - x_min[None, None, :]) + x_min[None, None, :]
        fake = fake * (x_max[None, None, :] - x_min[None, None, :]) + x_min[None, None, :]
        if cfg.get("x_transform") == "log1p_wind":
            ch = cfg.get("x_transform_channel", 1)
            real[..., ch] = np.maximum(np.expm1(real[..., ch]), 0.0)
            fake[..., ch] = np.maximum(np.expm1(fake[..., ch]), 0.0)

    qs = _parse_quantiles(args.quantiles)
    _plot_distribution(real, fake, x_names, outdir / "01_distribution.png")
    _plot_temporal(real, fake, x_names, outdir / "02_temporal.png")
    corr_r, corr_f = _plot_correlation(real, fake, x_names, outdir / "03_correlation.png")
    std_r, std_f, ramp_std_r, ramp_std_f = _plot_volatility_and_acf(
        real, fake, x_names, args.acf_lags, outdir / "04_volatility_acf.png"
    )
    qvals_r, qvals_f = _plot_extreme_quantiles(real, fake, x_names, qs, outdir / "05_extreme_quantiles.png")

    rows = []
    for ch, name in enumerate(x_names):
        row = {
            "channel": name,
            "mean_real": float(real[..., ch].mean()),
            "mean_fake": float(fake[..., ch].mean()),
            "std_real": float(std_r[ch]),
            "std_fake": float(std_f[ch]),
            "ramp_std_real": float(ramp_std_r[ch]),
            "ramp_std_fake": float(ramp_std_f[ch]),
        }
        for i, q in enumerate(qs):
            row[f"q{q:.2f}_real"] = float(qvals_r[i, ch])
            row[f"q{q:.2f}_fake"] = float(qvals_f[i, ch])
        rows.append(row)

    with (outdir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    corr_rows = []
    for i, ni in enumerate(x_names):
        for j, nj in enumerate(x_names):
            corr_rows.append(
                {
                    "ch_i": ni,
                    "ch_j": nj,
                    "corr_real": float(corr_r[i, j]),
                    "corr_fake": float(corr_f[i, j]),
                    "corr_abs_err": float(abs(corr_r[i, j] - corr_f[i, j])),
                }
            )
    with (outdir / "correlation_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        writer.writeheader()
        writer.writerows(corr_rows)

    np.save(outdir / "real_samples.npy", real)
    np.save(outdir / "fake_samples.npy", fake)
    print(f"saved evaluation report to: {outdir}")


if __name__ == "__main__":
    main()
