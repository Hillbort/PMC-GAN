import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_synth import load_ddre33_dataset
from .models import Generator
from .utils import ScenarioConfig, get_device, get_generator_state_dict, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate same-label distribution/envelope alignment for DDRE-33."
    )
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, default="pcm_gan_runs/eval_label_envelope")
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
    p.add_argument("--num_per_label", type=int, default=64)
    p.add_argument("--min_count", type=int, default=8)
    p.add_argument("--max_plots", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quantiles", type=str, default="0.1,0.5,0.9,0.95,0.99")
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


def _hist_l1_1d(xf, xr, bins=40, eps=1e-6):
    lo = min(float(np.min(xf)), float(np.min(xr)))
    hi = max(float(np.max(xf)), float(np.max(xr)))
    if hi <= lo:
        hi = lo + 1.0
    hf, _ = np.histogram(xf, bins=bins, range=(lo, hi), density=False)
    hr, _ = np.histogram(xr, bins=bins, range=(lo, hi), density=False)
    hf = hf.astype(np.float64)
    hr = hr.astype(np.float64)
    hf = hf / (hf.sum() + eps)
    hr = hr / (hr.sum() + eps)
    return float(np.mean(np.abs(hf - hr)))


def _ks_stat_1d(xf, xr, bins=80, eps=1e-6):
    lo = min(float(np.min(xf)), float(np.min(xr)))
    hi = max(float(np.max(xf)), float(np.max(xr)))
    if hi <= lo:
        hi = lo + 1.0
    hf, _ = np.histogram(xf, bins=bins, range=(lo, hi), density=False)
    hr, _ = np.histogram(xr, bins=bins, range=(lo, hi), density=False)
    cdf_f = np.cumsum(hf) / (np.sum(hf) + eps)
    cdf_r = np.cumsum(hr) / (np.sum(hr) + eps)
    return float(np.max(np.abs(cdf_f - cdf_r)))


def _envelope_metrics(real_ch, fake_ch, q_low=0.1, q_high=0.9, eps=1e-6):
    # real_ch/fake_ch: (N, T)
    r_lo = np.quantile(real_ch, q_low, axis=0)
    r_hi = np.quantile(real_ch, q_high, axis=0)
    f_lo = np.quantile(fake_ch, q_low, axis=0)
    f_hi = np.quantile(fake_ch, q_high, axis=0)
    env_mae = float(np.mean(0.5 * (np.abs(r_lo - f_lo) + np.abs(r_hi - f_hi))))
    inter = np.maximum(0.0, np.minimum(r_hi, f_hi) - np.maximum(r_lo, f_lo))
    union = np.maximum(r_hi, f_hi) - np.minimum(r_lo, f_lo)
    overlap = float(np.mean(inter / (union + eps)))
    return env_mae, overlap, (r_lo, r_hi, f_lo, f_hi)


def _plot_label_envelope(real, fake, label_name, out_png):
    # real/fake: (N,T,2) for 2ch case
    t = np.arange(real.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), squeeze=False)
    ch_names = ["pv", "wind"]
    for ch in range(min(2, real.shape[2])):
        ax = axes[ch, 0]
        r = real[..., ch]
        f = fake[..., ch]
        r_m = r.mean(axis=0)
        f_m = f.mean(axis=0)
        r_lo = np.quantile(r, 0.1, axis=0)
        r_hi = np.quantile(r, 0.9, axis=0)
        f_lo = np.quantile(f, 0.1, axis=0)
        f_hi = np.quantile(f, 0.9, axis=0)
        ax.plot(t, r_m, label="real_mean")
        ax.plot(t, f_m, label="fake_mean")
        ax.fill_between(t, r_lo, r_hi, alpha=0.2, label="real_p10-p90")
        ax.fill_between(t, f_lo, f_hi, alpha=0.2, label="fake_p10-p90")
        ax.set_title(f"{label_name} - {ch_names[ch]}")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    data, cond, mask, _, _, _ = load_ddre33_dataset(
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
    if data.shape[-1] != 2:
        raise ValueError("This script currently supports 2-channel mode (2ch_single/2ch_pairs).")

    cond_mean = cond if cond.ndim == 2 else cond.mean(axis=1)
    if cond_mean.shape[1] < 10:
        raise ValueError("cond_onehot is required for same-label evaluation (expected cond_dim >= 10).")

    pv_labels = np.argmax(cond_mean[:, :6], axis=1)
    wind_labels = np.argmax(cond_mean[:, 6:10], axis=1)

    # collect existing label pairs
    pair_to_indices = {}
    for i in range(data.shape[0]):
        key = (int(pv_labels[i]), int(wind_labels[i]))
        pair_to_indices.setdefault(key, []).append(i)

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    scfg = ScenarioConfig(seq_len=cfg["seq_len"], channels=cfg["channels"], cond_dim=cfg["cond_dim"])
    if scfg.seq_len != data.shape[1]:
        raise ValueError(f"seq_len mismatch: ckpt={scfg.seq_len}, data={data.shape[1]}")
    if scfg.channels != data.shape[2]:
        raise ValueError(f"channels mismatch: ckpt={scfg.channels}, data={data.shape[2]}")
    if scfg.cond_dim != cond_mean.shape[1]:
        raise ValueError(f"cond_dim mismatch: ckpt={scfg.cond_dim}, cond={cond_mean.shape[1]}")

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

    rng = np.random.default_rng(args.seed)
    qs = _parse_quantiles(args.quantiles)
    rows = []
    plotted = 0

    for (pv_c, wind_c), idxs in sorted(pair_to_indices.items()):
        if len(idxs) < int(args.min_count):
            continue
        n_sel = min(int(args.num_per_label), len(idxs))
        sel = rng.choice(np.array(idxs), size=n_sel, replace=False)
        real = data[sel]  # (N,T,2)
        cond_sel = cond[sel]
        if cond_sel.ndim == 2:
            cond_sel = np.repeat(cond_sel[:, None, :], scfg.seq_len, axis=1)
        mask_sel = mask[sel]

        z = torch.randn(n_sel, cfg["z_dim"], device=device)
        c = torch.from_numpy(cond_sel).to(device)
        m = torch.from_numpy(mask_sel).to(device)
        with torch.no_grad():
            fake = G(z, c, m).cpu().numpy()

        label_name = f"pv{pv_c}_wind{wind_c}"
        row = {
            "label": label_name,
            "pv_climate": pv_c,
            "wind_climate": wind_c,
            "count": int(n_sel),
        }

        for ch, ch_name in enumerate(["pv", "wind"]):
            xr = real[..., ch].reshape(-1)
            xf = fake[..., ch].reshape(-1)
            q_err = 0.0
            for q in qs:
                q_err += float(np.abs(np.quantile(xf, q) - np.quantile(xr, q)))
            q_err /= max(len(qs), 1)
            hist_err = _hist_l1_1d(xf, xr)
            ks = _ks_stat_1d(xf, xr)
            env_mae, env_overlap, _ = _envelope_metrics(real[..., ch], fake[..., ch], 0.1, 0.9)
            row[f"{ch_name}_q_err"] = q_err
            row[f"{ch_name}_hist_err"] = hist_err
            row[f"{ch_name}_ks"] = ks
            row[f"{ch_name}_env_mae"] = env_mae
            row[f"{ch_name}_env_overlap"] = env_overlap

        # compact composite score for ranking
        row["score"] = (
            0.25 * row["pv_q_err"]
            + 0.25 * row["wind_q_err"]
            + 0.2 * row["pv_env_mae"]
            + 0.2 * row["wind_env_mae"]
            + 0.1 * (1.0 - 0.5 * (row["pv_env_overlap"] + row["wind_env_overlap"]))
        )
        rows.append(row)

        if plotted < int(args.max_plots):
            _plot_label_envelope(real, fake, label_name, plot_dir / f"{label_name}.png")
            plotted += 1

    if not rows:
        raise ValueError("No label groups passed min_count. Try lower --min_count or increase data.")

    rows = sorted(rows, key=lambda r: float(r["score"]))
    out_csv = outdir / "label_envelope_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved label-envelope metrics: {out_csv}")
    print(f"saved plots: {plot_dir}")
    print("Top 5 labels (best score):")
    for r in rows[:5]:
        print(
            f"  {r['label']}: score={r['score']:.5f}, "
            f"pv_q={r['pv_q_err']:.5f}, wind_q={r['wind_q_err']:.5f}, "
            f"pv_env={r['pv_env_mae']:.5f}, wind_env={r['wind_env_mae']:.5f}"
        )


if __name__ == "__main__":
    main()
