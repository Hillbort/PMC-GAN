import argparse
from pathlib import Path

import numpy as np
import torch

from .data_synth import ddre33_cond_layout, load_real_dataset, load_ddre33_dataset
from .models import Generator
from .utils import ScenarioConfig, get_device, get_generator_state_dict, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="PCM-GAN test/inference (real data).")
    p.add_argument("--ckpt", type=str, default="pcm_gan_runs/pcm_gan.pt")
    p.add_argument("--outdir", type=str, default="pcm_gan_test_out")
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
    p.add_argument("--cond_onehot", action="store_true", help="Use one-hot climate labels.")
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
    p.add_argument("--num", type=int, default=16)
    p.add_argument(
        "--topk_candidates",
        type=int,
        default=1,
        help="Number of candidates sampled per condition before screening.",
    )
    p.add_argument(
        "--topk_keep",
        type=int,
        default=1,
        help="Number of kept samples per condition after screening.",
    )
    p.add_argument("--seed", type=int, default=42)
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
        "--mask_source",
        type=str,
        default="ghi",
        choices=["solar", "ghi"],
        help="Daylight mask source: solar output or GHI.",
    )
    p.add_argument("--denorm", action="store_true", help="Denormalize x using scaler in checkpoint.")
    return p.parse_args()


def _candidate_score(x_cands, x_ref):
    # x_cands: (K,T,C), x_ref: (T,C)
    ref_mean = x_ref.mean(axis=0)
    ref_std = x_ref.std(axis=0)
    if x_ref.shape[0] > 1:
        ref_ramp_std = np.diff(x_ref, axis=0).std(axis=0)
    else:
        ref_ramp_std = np.zeros_like(ref_mean)
    scores = []
    for x in x_cands:
        m = np.abs(x.mean(axis=0) - ref_mean).mean()
        s = np.abs(x.std(axis=0) - ref_std).mean()
        if x.shape[0] > 1:
            r = np.abs(np.diff(x, axis=0).std(axis=0) - ref_ramp_std).mean()
        else:
            r = 0.0
        scores.append(float(0.5 * m + 0.2 * s + 0.3 * r))
    return np.array(scores, dtype=np.float32)


def main():
    args = parse_args()
    set_seed(args.seed)
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

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    scfg = ScenarioConfig(seq_len=cfg["seq_len"], channels=cfg["channels"], cond_dim=cfg["cond_dim"])
    # For DDRE-33, if user leaves default resolution/seq_len, align to checkpoint shape.
    if args.dataset == "ddre33" and args.seq_len == 0 and not args.resample and seq_len != scfg.seq_len:
        seq_len = int(scfg.seq_len)
        if seq_len == 24:
            resample_rule = "H"
        elif seq_len == 96:
            resample_rule = "15T"
        elif seq_len == 1440:
            resample_rule = "T"

    if args.dataset == "real":
        if not args.data_csv:
            raise ValueError("Real data CSV is required. Provide --data_csv.")
        data, cond, mask, _, _, _ = load_real_dataset(
            args.data_csv,
            seq_len=seq_len,
            x_cols=x_cols,
            cond_cols=cond_cols,
            resample_rule=resample_rule,
            cond_agg=args.cond_agg,
            x_agg="sum",
            mask_source=args.mask_source,
        )
    else:
        data, cond, mask, _, _, _ = load_ddre33_dataset(
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

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(cond), size=min(args.num, len(cond)), replace=False)
    cond_sel = cond[idx]
    mask_sel = mask[idx]
    real_sel = data[idx]

    if scfg.seq_len != seq_len:
        hint = ""
        if args.dataset == "ddre33":
            hint = f". Hint: use --resolution 15min or --seq_len {scfg.seq_len} to match checkpoint."
        raise ValueError(f"seq_len mismatch: ckpt={scfg.seq_len}, data={seq_len}{hint}")
    if scfg.channels != len(x_cols):
        hint = ""
        if args.dataset == "ddre33":
            hint = ". Hint: ensure --ddre33_mode matches training checkpoint."
        raise ValueError(f"channels mismatch: ckpt={scfg.channels}, x_cols={len(x_cols)}{hint}")
    if scfg.cond_dim != len(cond_cols):
        hint = ""
        if args.dataset == "ddre33":
            hint = (
                ". Hint: DDRE-33 one-hot cond_dim is 10; with --ddre33_date_cond it is 12. "
                "Curve cond adds 4 features per channel. Ensure --cond_onehot/--ddre33_date_cond/"
                "--ddre33_static_cond/--ddre33_curve_cond match training."
            )
        raise ValueError(f"cond_dim mismatch: ckpt={scfg.cond_dim}, cond_cols={len(cond_cols)}{hint}")

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

    k = max(1, int(args.topk_candidates))
    keep = max(1, min(int(args.topk_keep), k))
    c = torch.from_numpy(cond_sel).to(device)
    m = torch.from_numpy(mask_sel).to(device)
    with torch.no_grad():
        if k == 1 and keep == 1:
            z = torch.randn(cond_sel.shape[0], cfg["z_dim"], device=device)
            x = G(z, c, m).cpu().numpy()
        else:
            picked = []
            for i in range(cond_sel.shape[0]):
                if c.dim() == 2:
                    ci = c[i : i + 1].repeat(k, 1)
                else:
                    ci = c[i : i + 1].repeat(k, 1, 1)
                mi = m[i : i + 1].repeat(k, 1, 1)
                zi = torch.randn(k, cfg["z_dim"], device=device)
                cand = G(zi, ci, mi).cpu().numpy()  # (K,T,C)
                sc = _candidate_score(cand, real_sel[i])
                ord_idx = np.argsort(sc)[:keep]
                picked.append(cand[ord_idx])
            x = np.concatenate(picked, axis=0)
            cond_sel = np.repeat(cond_sel, keep, axis=0)

    if args.denorm and args.dataset != "ddre33" and "x_min" in cfg and "x_max" in cfg:
        x_min = np.array(cfg["x_min"], dtype=np.float32)
        x_max = np.array(cfg["x_max"], dtype=np.float32)
        x = x * (x_max[None, None, :] - x_min[None, None, :]) + x_min[None, None, :]
        if cfg.get("x_transform") == "log1p_wind":
            ch = cfg.get("x_transform_channel", 1)
            x[..., ch] = np.maximum(np.expm1(x[..., ch]), 0.0)

    np.save(outdir / "samples.npy", x)
    np.save(outdir / "conds.npy", cond_sel)

    header = ",".join(x_cols)
    for i in range(x.shape[0]):
        csv_path = outdir / f"sample_{i:03d}.csv"
        np.savetxt(csv_path, x[i], delimiter=",", header=header, comments="")

    print(f"saved {x.shape[0]} samples to {outdir}")


if __name__ == "__main__":
    main()
