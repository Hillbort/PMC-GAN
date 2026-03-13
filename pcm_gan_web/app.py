import csv
import hashlib
import re
import threading
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from flask import Flask, render_template, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcm_gan.data_synth import load_real_dataset, load_ddre33_dataset
from pcm_gan.models import Generator
from pcm_gan.utils import ScenarioConfig, get_generator_state_dict, set_seed

RESULT_DIR = Path(__file__).resolve().parent / "static" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATA = ROOT / "pcm_gan_data" / "ororiginal_data" / "ERCOT_zone_1_.csv"
DDRE_DIR = ROOT / "pcm_gan_data" / "ororiginal_data"
DDRE_SUBSET_DIR = ROOT / "pcm_gan_data" / "ddre33_subset"

app = Flask(__name__)

JOBS = {}
PROCS = {}
JOBS_LOCK = threading.Lock()


def list_checkpoints():
    candidates = []
    for path in ROOT.rglob("*.pt"):
        candidates.append(path)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def resolution_defaults(resolution):
    if resolution == "hourly":
        return "H", 24
    if resolution == "15min":
        return "15T", 96
    return "T", 1440


def cache_key(csv_path, resolution, cond_norm, mask_source):
    raw = f"{csv_path}|{resolution}|{cond_norm}|{mask_source}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_or_build_cache(csv_path, resolution, cond_norm, mask_source, days=60):
    key = cache_key(csv_path, resolution, cond_norm, mask_source)
    cache_path = CACHE_DIR / f"cache_{key}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as data:
            # validate required keys; if missing, rebuild cache
            required = {"data", "cond", "mask", "x_min", "x_max"}
            if not required.issubset(set(data.files)):
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError:
                    cache_path = CACHE_DIR / f"cache_{key}_{uuid.uuid4().hex[:6]}.npz"
                data = None
            if data is not None:
                cached = {
                    "data": data["data"],
                    "cond": data["cond"],
                    "mask": data["mask"],
                    "x_min": data["x_min"],
                    "x_max": data["x_max"],
                    "cond_stats": data["cond_stats"].item() if "cond_stats" in data else {},
                    "cache_path": str(cache_path),
                    "seq_len": int(data["seq_len"]) if "seq_len" in data else None,
                    "cond_dim": int(data["cond_dim"]) if "cond_dim" in data else None,
                }
        # validate cached shapes; rebuild if mismatch
        if data is not None:
            resample_rule, seq_len = resolution_defaults(resolution)
            if (
                cached["data"].ndim != 3
                or cached["cond"].ndim != 3
                or cached["data"].shape[1] != seq_len
                or cached["cond"].shape[1] != seq_len
            ):
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError:
                    # file is in use; fall through and build a new cache file
                    cache_path = CACHE_DIR / f"cache_{key}_{uuid.uuid4().hex[:6]}.npz"
            else:
                return cached

    resample_rule, seq_len = resolution_defaults(resolution)
    x_cols = ["solar_power", "wind_power", "load_power"]
    cond_cols = [
        "DHI",
        "DNI",
        "GHI",
        "Dew Point",
        "Solar Zenith Angle",
        "Wind Speed",
        "Relative Humidity",
        "Temperature",
    ]
    data, cond, mask, x_min, x_max, cond_stats = load_real_dataset(
        csv_path,
        seq_len=seq_len,
        x_cols=x_cols,
        cond_cols=cond_cols,
        resample_rule=resample_rule,
        cond_agg="mean",
        x_agg="sum",
        mask_source=mask_source,
        cond_norm=cond_norm,
    )
    if data.shape[0] > days:
        data = data[:days]
        cond = cond[:days]
        mask = mask[:days]
    np.savez_compressed(
        cache_path,
        data=data,
        cond=cond,
        mask=mask,
        x_min=x_min,
        x_max=x_max,
        cond_stats=cond_stats,
        seq_len=np.array(seq_len),
        cond_dim=np.array(cond.shape[-1]),
    )
    return {
        "data": data,
        "cond": cond,
        "mask": mask,
        "x_min": x_min,
        "x_max": x_max,
        "cond_stats": cond_stats,
        "cache_path": str(cache_path),
    }


def cache_key_ddre33(args):
    raw = "|".join(
        [
            "ddre33",
            args.get("ddre33_mode", "2ch_single"),
            str(args.get("cond_onehot", True)),
            str(args.get("ddre33_date_cond", False)),
            str(args.get("ddre33_static_cond", False)),
            str(args.get("max_cols", 0)),
            args.get("pv18_csv", ""),
            args.get("pv33_csv", ""),
            args.get("wind22_csv", ""),
            args.get("wind25_csv", ""),
            args.get("pv18_labels_csv", ""),
            args.get("pv33_labels_csv", ""),
            args.get("wind22_labels_csv", ""),
            args.get("wind25_labels_csv", ""),
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_or_build_cache_ddre33(args):
    key = cache_key_ddre33(args)
    cache_path = CACHE_DIR / f"cache_{key}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as data:
            required = {"data", "cond", "mask", "x_min", "x_max"}
            if required.issubset(set(data.files)):
                return {
                    "data": data["data"],
                    "cond": data["cond"],
                    "mask": data["mask"],
                    "x_min": data["x_min"],
                    "x_max": data["x_max"],
                    "cond_stats": data["cond_stats"].item() if "cond_stats" in data else {},
                    "cache_path": str(cache_path),
                }
    data, cond, mask, x_min, x_max, cond_stats = load_ddre33_dataset(
        pv18_csv=args.get("pv18_csv", ""),
        pv33_csv=args.get("pv33_csv", ""),
        wind22_csv=args.get("wind22_csv", ""),
        wind25_csv=args.get("wind25_csv", ""),
        pv18_labels_csv=args.get("pv18_labels_csv", ""),
        pv33_labels_csv=args.get("pv33_labels_csv", ""),
        wind22_labels_csv=args.get("wind22_labels_csv", ""),
        wind25_labels_csv=args.get("wind25_labels_csv", ""),
        seq_len=int(args.get("seq_len", 96)),
        resample_rule="15min",
        one_hot=bool(args.get("cond_onehot", True)),
        mode=args.get("ddre33_mode", "2ch_single"),
        max_cols=int(args.get("max_cols", 0) or 0),
        normalize=False,
        add_date_cond=bool(args.get("ddre33_date_cond", False)),
        static_cond=bool(args.get("ddre33_static_cond", False)),
    )
    np.savez_compressed(
        cache_path,
        data=data,
        cond=cond,
        mask=mask,
        x_min=x_min,
        x_max=x_max,
        cond_stats=cond_stats,
        seq_len=np.array(data.shape[1]),
        cond_dim=np.array(cond.shape[-1]),
    )
    return {
        "data": data,
        "cond": cond,
        "mask": mask,
        "x_min": x_min,
        "x_max": x_max,
        "cond_stats": cond_stats,
        "cache_path": str(cache_path),
    }


def count_csv_data_rows(metrics_path):
    if not metrics_path.exists():
        return 0
    with metrics_path.open("r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    return max(0, line_count - 1)


def load_metrics(metrics_path, limit=200, start_row=0):
    if not metrics_path.exists():
        return []
    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if start_row > 0:
        rows = rows[start_row:]
    return rows[-limit:]


def tail_text(path, max_chars=4000):
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    if len(data) <= max_chars:
        return data
    return data[-max_chars:]


def extract_tqdm_progress(text):
    # Match tqdm percent like: "training:  12%|"
    matches = re.findall(r"training:\s*(\d+)%\|", text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def run_training(job_id, args):
    base_cmd = [
        "python",
        "-m",
        "pcm_gan.train",
        "--dataset",
        args.get("dataset", "real"),
        "--resolution",
        args["resolution"],
        "--mask_source",
        args["mask_source"],
        "--cond_norm",
        args["cond_norm"],
        "--epochs",
        str(args["epochs"]),
        "--batch",
        str(args["batch"]),
        "--lr",
        str(args["lr"]),
        "--z_dim",
        str(args["z_dim"]),
        "--d_steps",
        str(args["d_steps"]),
        "--lambda_gp",
        str(args["lambda_gp"]),
        "--lambda_tailq",
        str(args["lambda_tailq"]),
        "--tail_q",
        str(args["tail_q"]),
        "--lambda_stats",
        str(args["lambda_stats"]),
        "--save_every",
        str(args["save_every"]),
        "--outdir",
        args["outdir"],
    ]
    if args.get("dataset") == "ddre33":
        base_cmd.extend(
            [
                "--ddre33_mode",
                args.get("ddre33_mode", "2ch_single"),
                "--max_cols",
                str(args.get("max_cols", 0)),
                "--pv18_csv",
                args.get("pv18_csv", ""),
                "--pv33_csv",
                args.get("pv33_csv", ""),
                "--wind22_csv",
                args.get("wind22_csv", ""),
                "--wind25_csv",
                args.get("wind25_csv", ""),
                "--pv18_labels_csv",
                args.get("pv18_labels_csv", ""),
                "--pv33_labels_csv",
                args.get("pv33_labels_csv", ""),
                "--wind22_labels_csv",
                args.get("wind22_labels_csv", ""),
                "--wind25_labels_csv",
                args.get("wind25_labels_csv", ""),
            ]
        )
        if args.get("cond_onehot"):
            base_cmd.append("--cond_onehot")
        if args.get("ddre33_date_cond"):
            base_cmd.append("--ddre33_date_cond")
    else:
        base_cmd.extend(["--data_csv", args.get("data_csv", "")])
    if not args.get("use_venv"):
        base_cmd[0] = sys.executable
    cmd = conda_cmd(args.get("use_venv"), base_cmd)
    log_out = Path(args["outdir"]) / f"train_{job_id}.out.log"
    log_err = Path(args["outdir"]) / f"train_{job_id}.err.log"
    Path(args["outdir"]).mkdir(parents=True, exist_ok=True)
    with log_out.open("w", encoding="utf-8") as f_out, log_err.open("w", encoding="utf-8") as f_err:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=f_out,
            stderr=f_err,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0,
        )
        with JOBS_LOCK:
            PROCS[job_id] = proc
        code = proc.wait()
    with JOBS_LOCK:
        job = JOBS.get(job_id, {})
        if job.get("status") == "stopping":
            job["status"] = "stopped"
        else:
            job["status"] = "done" if code == 0 else "error"
        job["returncode"] = code
        JOBS[job_id] = job
        PROCS.pop(job_id, None)


def start_training_job(args):
    job_id = uuid.uuid4().hex[:8]
    metrics_path = ROOT / args["outdir"] / "metrics.csv"
    log_out = ROOT / args["outdir"] / f"train_{job_id}.out.log"
    log_err = ROOT / args["outdir"] / f"train_{job_id}.err.log"
    start_rows = count_csv_data_rows(metrics_path)
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "metrics_path": str(metrics_path),
            "log_out": str(log_out),
            "log_err": str(log_err),
            "outdir": args["outdir"],
            "start_time": time.time(),
            "total_epochs": int(args["epochs"]),
            "start_rows": start_rows,
        }
    t = threading.Thread(target=run_training, args=(job_id, args), daemon=True)
    t.start()
    return job_id


def generate_samples(args):
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

    if args.get("dataset") == "ddre33":
        seq_len = 96
        cache = load_or_build_cache_ddre33(args)
        data, cond, mask = cache["data"], cache["cond"], cache["mask"]
        cond_stats = cache["cond_stats"]
        rng = np.random.default_rng(int(args["seed"]))
        idx = rng.choice(len(cond), size=min(int(args["num"]), len(cond)), replace=False)
        cond_sel = cond[idx]
        mask_sel = mask[idx]
        real_sel = data[idx]
        if cond_sel.ndim == 2:
            cond_sel = np.repeat(cond_sel[:, None, :], seq_len, axis=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args["ckpt"], map_location=device)
        cfg = ckpt["cfg"]
        scfg = ScenarioConfig(seq_len=cfg["seq_len"], channels=cfg["channels"], cond_dim=cfg["cond_dim"])
        if scfg.seq_len != seq_len:
            raise ValueError(f"seq_len mismatch: ckpt={scfg.seq_len}, data={seq_len}")
        if scfg.channels != 2:
            raise ValueError(f"channels mismatch: ckpt={scfg.channels}, expected 2")
        if scfg.cond_dim != cond_sel.shape[2]:
            raise ValueError(f"cond_dim mismatch: ckpt={scfg.cond_dim}, cond={cond_sel.shape[2]}")

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

        c = torch.from_numpy(cond_sel).to(device)
        m = torch.from_numpy(mask_sel).to(device)
        k = max(1, int(args.get("topk_candidates", 1)))
        keep = max(1, min(int(args.get("topk_keep", 1)), k))
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
                    cand = G(zi, ci, mi).cpu().numpy()
                    sc = _candidate_score(cand, real_sel[i])
                    ord_idx = np.argsort(sc)[:keep]
                    picked.append(cand[ord_idx])
                x = np.concatenate(picked, axis=0)
                cond_sel = np.repeat(cond_sel, keep, axis=0)

        if args.get("denorm") and args.get("dataset") != "ddre33" and "x_min" in cfg and "x_max" in cfg:
            x_min = np.array(cfg["x_min"], dtype=np.float32)
            x_max = np.array(cfg["x_max"], dtype=np.float32)
            x = x * (x_max[None, None, :] - x_min[None, None, :]) + x_min[None, None, :]

        cond_mean = cond_sel.mean(axis=1)
        return x, cond_mean, cond_stats, cache.get("cache_path")

    resample_rule, seq_len = resolution_defaults(args["resolution"])
    if args.get("resample"):
        resample_rule = args["resample"]
    if args.get("seq_len"):
        seq_len = int(args["seq_len"])

    x_cols = [c.strip() for c in args["x_cols"].split(",") if c.strip()]
    cond_cols = [c.strip() for c in args["cond_cols"].split(",") if c.strip()]
    cache = load_or_build_cache(
        args["data_csv"],
        args["resolution"],
        args.get("cond_norm", "none"),
        args["mask_source"],
    )
    data, cond, mask = cache["data"], cache["cond"], cache["mask"]
    cond_stats = cache["cond_stats"]

    rng = np.random.default_rng(int(args["seed"]))
    idx = rng.choice(len(cond), size=min(int(args["num"]), len(cond)), replace=False)
    cond_sel = cond[idx]
    mask_sel = mask[idx]
    real_sel = data[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args["ckpt"], map_location=device)
    cfg = ckpt["cfg"]
    scfg = ScenarioConfig(seq_len=cfg["seq_len"], channels=cfg["channels"], cond_dim=cfg["cond_dim"])
    if scfg.seq_len != seq_len:
        raise ValueError(f"seq_len mismatch: ckpt={scfg.seq_len}, data={seq_len}")
    if scfg.channels != len(x_cols):
        raise ValueError(f"channels mismatch: ckpt={scfg.channels}, x_cols={len(x_cols)}")
    if scfg.cond_dim != len(cond_cols):
        raise ValueError(f"cond_dim mismatch: ckpt={scfg.cond_dim}, cond_cols={len(cond_cols)}")

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

    c = torch.from_numpy(cond_sel).to(device)
    m = torch.from_numpy(mask_sel).to(device)
    k = max(1, int(args.get("topk_candidates", 1)))
    keep = max(1, min(int(args.get("topk_keep", 1)), k))
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
                cand = G(zi, ci, mi).cpu().numpy()
                sc = _candidate_score(cand, real_sel[i])
                ord_idx = np.argsort(sc)[:keep]
                picked.append(cand[ord_idx])
            x = np.concatenate(picked, axis=0)
            cond_sel = np.repeat(cond_sel, keep, axis=0)

    if args.get("denorm") and args.get("dataset") != "ddre33" and "x_min" in cfg and "x_max" in cfg:
        x_min = np.array(cfg["x_min"], dtype=np.float32)
        x_max = np.array(cfg["x_max"], dtype=np.float32)
        x = x * (x_max[None, None, :] - x_min[None, None, :]) + x_min[None, None, :]
        if cfg.get("x_transform") == "log1p_wind":
            ch = cfg.get("x_transform_channel", 1)
            x[..., ch] = np.maximum(np.expm1(x[..., ch]), 0.0)

    cond_mean = cond_sel.mean(axis=1)
    return x, cond_mean, cond_stats, cache.get("cache_path")


def conda_cmd(use_venv, inner_cmd):
    if not use_venv:
        return inner_cmd
    return ["conda", "run", "-n", "venv"] + inner_cmd


def generate_samples_subprocess(args, use_venv):
    run_dir = Path(args["outdir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "pcm_gan.generate",
        "--ckpt",
        args["ckpt"],
        "--data_csv",
        args.get("data_csv", ""),
        "--dataset",
        args.get("dataset", "real"),
        "--resolution",
        args["resolution"],
        "--mask_source",
        args["mask_source"],
        "--cond_norm",
        args.get("cond_norm", "none"),
        "--num",
        str(args["num"]),
        "--topk_candidates",
        str(args.get("topk_candidates", 1)),
        "--topk_keep",
        str(args.get("topk_keep", 1)),
        "--seed",
        str(args["seed"]),
        "--outdir",
        str(run_dir),
        "--x_cols",
        args["x_cols"],
        "--cond_cols",
        args["cond_cols"],
        "--cond_agg",
        args.get("cond_agg", "mean"),
    ]
    if args.get("dataset") == "ddre33":
        cmd.extend(
            [
                "--ddre33_mode",
                args.get("ddre33_mode", "2ch_single"),
                "--max_cols",
                str(args.get("max_cols", 0)),
                "--pv18_csv",
                args.get("pv18_csv", ""),
                "--pv33_csv",
                args.get("pv33_csv", ""),
                "--wind22_csv",
                args.get("wind22_csv", ""),
                "--wind25_csv",
                args.get("wind25_csv", ""),
                "--pv18_labels_csv",
                args.get("pv18_labels_csv", ""),
                "--pv33_labels_csv",
                args.get("pv33_labels_csv", ""),
                "--wind22_labels_csv",
                args.get("wind22_labels_csv", ""),
                "--wind25_labels_csv",
                args.get("wind25_labels_csv", ""),
            ]
        )
        if args.get("cond_onehot"):
            cmd.append("--cond_onehot")
        if args.get("ddre33_date_cond"):
            cmd.append("--ddre33_date_cond")
        if args.get("ddre33_static_cond"):
            cmd.append("--ddre33_static_cond")
    if args.get("denorm") and args.get("dataset") != "ddre33":
        cmd.append("--denorm")
    final_cmd = conda_cmd(use_venv, cmd)
    proc = subprocess.run(final_cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "generate failed")

    samples_path = run_dir / "samples.npy"
    conds_path = run_dir / "conds.npy"
    if not samples_path.exists() or not conds_path.exists():
        raise RuntimeError("missing samples.npy or conds.npy from generate")
    samples = np.load(samples_path)
    conds = np.load(conds_path)
    if conds.ndim == 3:
        conds = conds.mean(axis=1)
    return samples, conds


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


@app.route("/custom_generate", methods=["POST"])
def custom_generate():
    payload = request.get_json(silent=True) or {}
    ckpt_path = payload.get("ckpt")
    cond_vec = payload.get("cond")
    pv_climate = payload.get("pv_climate")
    wind_climate = payload.get("wind_climate")
    date_month = payload.get("date_month")
    date_day = payload.get("date_day")
    date_sin = payload.get("date_sin")
    date_cos = payload.get("date_cos")
    denorm = bool(payload.get("denorm"))
    if not ckpt_path:
        return {"error": "ckpt and cond are required"}, 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    scfg = ScenarioConfig(seq_len=cfg["seq_len"], channels=cfg["channels"], cond_dim=cfg["cond_dim"])
    if cond_vec is not None and len(cond_vec) > 0:
        cond_arr = np.array(cond_vec, dtype=np.float32)
        if cond_arr.ndim != 1:
            return {"error": "cond must be a 1D array"}, 400
        if len(cond_vec) == scfg.cond_dim:
            cond_arr = np.repeat(cond_arr[None, :], scfg.seq_len, axis=0)
        elif len(cond_vec) == scfg.seq_len * scfg.cond_dim:
            cond_arr = cond_arr.reshape(scfg.seq_len, scfg.cond_dim)
        else:
            return {"error": f"cond length {len(cond_vec)} not match cond_dim {scfg.cond_dim} or seq_len*cond_dim"}, 400
    elif pv_climate is not None and wind_climate is not None and scfg.cond_dim in (10, 12):
        pv_idx = int(pv_climate)
        wind_idx = int(wind_climate)
        pv_vec = np.zeros(6, dtype=np.float32)
        wind_vec = np.zeros(4, dtype=np.float32)
        if 0 <= pv_idx < 6:
            pv_vec[pv_idx] = 1.0
        if 0 <= wind_idx < 4:
            wind_vec[wind_idx] = 1.0
        base = np.concatenate([pv_vec, wind_vec], axis=0)
        if scfg.cond_dim == 12:
            if date_sin is not None and date_cos is not None:
                ds = float(date_sin)
                dc = float(date_cos)
            elif date_month is not None and date_day is not None:
                try:
                    ds, dc = _date_to_sin_cos(date_month, date_day)
                except Exception as exc:
                    return {"error": f"invalid date: {exc}"}, 400
            else:
                ds, dc = 0.0, 1.0
            base = np.concatenate([base, np.array([ds, dc], dtype=np.float32)], axis=0)
        cond_arr = np.repeat(base[None, :], scfg.seq_len, axis=0)
    else:
        return {"error": "cond or (pv_climate, wind_climate) is required"}, 400

    # apply cond normalization if stored in ckpt
    cond_norm = cfg.get("cond_norm", "none")
    if cond_norm == "minmax":
        c_min = np.array(cfg.get("cond_min"), dtype=np.float32)
        c_max = np.array(cfg.get("cond_max"), dtype=np.float32)
        denom = np.maximum(c_max - c_min, 1e-6)
        cond_arr = (cond_arr - c_min[None, :]) / denom[None, :]
    elif cond_norm == "zscore":
        c_mean = np.array(cfg.get("cond_mean"), dtype=np.float32)
        c_std = np.maximum(np.array(cfg.get("cond_std"), dtype=np.float32), 1e-6)
        cond_arr = (cond_arr - c_mean[None, :]) / c_std[None, :]

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

    z = torch.randn(1, cfg["z_dim"], device=device)
    c = torch.from_numpy(cond_arr[None, :, :]).to(device)
    # default mask: allow solar everywhere, then rely on model
    m = torch.ones(1, scfg.seq_len, scfg.channels, device=device)
    with torch.no_grad():
        x = G(z, c, m).cpu().numpy()

    if denorm and not cfg.get("ddre33_mode") and "x_min" in cfg and "x_max" in cfg:
        x_min = np.array(cfg["x_min"], dtype=np.float32)
        x_max = np.array(cfg["x_max"], dtype=np.float32)
        x = x * (x_max[None, None, :] - x_min[None, None, :]) + x_min[None, None, :]
        if cfg.get("x_transform") == "log1p_wind":
            ch = cfg.get("x_transform_channel", 1)
            x[..., ch] = np.maximum(np.expm1(x[..., ch]), 0.0)

    return {"samples": x.tolist()}


@app.route("/compare_day", methods=["POST"])
def compare_day():
    payload = request.get_json(silent=True) or {}
    ckpt_path = payload.get("ckpt")
    data_csv = payload.get("data_csv")
    dataset = payload.get("dataset", "real")
    resolution = payload.get("resolution", "hourly")
    mask_source = payload.get("mask_source", "ghi")
    cond_norm = payload.get("cond_norm", "none")
    denorm = bool(payload.get("denorm"))
    day_index = int(payload.get("day_index", 0))
    if not ckpt_path:
        return {"error": "ckpt is required"}, 400

    if dataset == "ddre33":
        cache = load_or_build_cache_ddre33(payload)
    else:
        if not data_csv:
            return {"error": "data_csv is required"}, 400
        cache = load_or_build_cache(data_csv, resolution, cond_norm, mask_source)
    data = cache["data"]
    cond = cache["cond"]
    mask = cache["mask"]
    if day_index < 0 or day_index >= data.shape[0]:
        return {"error": "day_index out of range"}, 400

    x_cols = ["solar_power", "wind_power", "load_power"]
    real_norm = data[day_index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    scfg = ScenarioConfig(seq_len=cfg["seq_len"], channels=cfg["channels"], cond_dim=cfg["cond_dim"])
    if scfg.seq_len != real_norm.shape[0]:
        return {"error": "seq_len mismatch for compare"}, 400
    if scfg.channels != real_norm.shape[1]:
        return {"error": "channels mismatch for compare"}, 400
    if scfg.cond_dim != cond.shape[-1]:
        return {"error": "cond_dim mismatch for compare"}, 400

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
    z = torch.randn(1, cfg["z_dim"], device=device)
    if cond.ndim == 2:
        c_arr = np.repeat(cond[day_index][None, None, :], scfg.seq_len, axis=1)
    else:
        c_arr = cond[day_index][None, :, :]
    c = torch.from_numpy(c_arr).to(device)
    m = torch.from_numpy(mask[day_index][None, :, :]).to(device)
    with torch.no_grad():
        fake_norm = G(z, c, m).cpu().numpy()[0]
    if denorm and dataset != "ddre33" and "x_min" in cfg and "x_max" in cfg:
        x_min = np.array(cfg["x_min"], dtype=np.float32)
        x_max = np.array(cfg["x_max"], dtype=np.float32)
        real_norm = real_norm * (x_max[None, :] - x_min[None, :]) + x_min[None, :]
        fake_norm = fake_norm * (x_max[None, :] - x_min[None, :]) + x_min[None, :]
        if cfg.get("x_transform") == "log1p_wind":
            ch = cfg.get("x_transform_channel", 1)
            real_norm[..., ch] = np.maximum(np.expm1(real_norm[..., ch]), 0.0)
            fake_norm[..., ch] = np.maximum(np.expm1(fake_norm[..., ch]), 0.0)
    mae_pv = float(np.mean(np.abs(real_norm[:, 0] - fake_norm[:, 0])))
    mae_wind = float(np.mean(np.abs(real_norm[:, 1] - fake_norm[:, 1]))) if real_norm.shape[1] > 1 else None
    return {
        "day_index": day_index,
        "total_days": int(data.shape[0]),
        "real_norm": real_norm.tolist(),
        "fake_norm": fake_norm.tolist(),
        "cond": cond[day_index].tolist(),
        "x_cols": x_cols if dataset != "ddre33" else ["pv", "wind"],
        "mae_pv": mae_pv,
        "mae_wind": mae_wind,
    }


@app.route("/compare_climates", methods=["POST"])
def compare_climates():
    payload = request.get_json(silent=True) or {}
    ckpt_path = payload.get("ckpt")
    dataset = payload.get("dataset", "real")
    denorm = bool(payload.get("denorm"))
    seed = int(payload.get("seed", 42))
    if not ckpt_path:
        return {"error": "ckpt is required"}, 400
    if dataset != "ddre33":
        return {"error": "compare_climates only supports ddre33"}, 400

    cache = load_or_build_cache_ddre33(payload)
    data = cache["data"]
    cond = cache["cond"]
    mask = cache["mask"]
    if data.ndim != 3 or cond.ndim not in (2, 3):
        return {"error": "invalid ddre33 cache"}, 400

    if cond.ndim == 2:
        cond_mean = cond
    else:
        cond_mean = cond.mean(axis=1)
    if cond_mean.shape[1] not in (10, 12):
        return {"error": f"cond_dim {cond_mean.shape[1]} not supported (expected 10 or 12 one-hot)"}, 400

    pv_labels = np.argmax(cond_mean[:, :6], axis=1)
    wind_labels = np.argmax(cond_mean[:, 6:10], axis=1)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    scfg = ScenarioConfig(seq_len=cfg["seq_len"], channels=cfg["channels"], cond_dim=cfg["cond_dim"])
    if scfg.seq_len != data.shape[1]:
        return {"error": "seq_len mismatch for compare_climates"}, 400
    if scfg.channels != data.shape[2]:
        return {"error": "channels mismatch for compare_climates"}, 400

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

    cases = []

    def _mae(a, b, ch):
        return float(np.mean(np.abs(a[:, ch] - b[:, ch])))

    def build_onehot(pv_idx, wind_idx, seq_len, extra=None):
        pv_vec = np.zeros(6, dtype=np.float32)
        wind_vec = np.zeros(4, dtype=np.float32)
        if 0 <= pv_idx < 6:
            pv_vec[pv_idx] = 1.0
        if 0 <= wind_idx < 4:
            wind_vec[wind_idx] = 1.0
        base = np.concatenate([pv_vec, wind_vec], axis=0)
        if extra is not None:
            base = np.concatenate([base, extra], axis=0)
        return np.repeat(base[None, :], seq_len, axis=0)

    # PV climates (0-5): pick real sample with pv climate i, keep its wind climate
    for pv_idx in range(6):
        candidates = np.where(pv_labels == pv_idx)[0]
        if candidates.size == 0:
            continue
        sel = int(rng.choice(candidates, size=1)[0])
        wind_idx = int(wind_labels[sel])
        extra = None
        if cond_mean.shape[1] == 12:
            if cond.ndim == 2:
                extra = cond[sel][10:12]
            else:
                extra = cond[sel][0, 10:12]
        cond_arr = build_onehot(pv_idx, wind_idx, scfg.seq_len, extra=extra)
        z = torch.randn(1, cfg["z_dim"], device=device)
        c = torch.from_numpy(cond_arr[None, :, :]).to(device)
        m = torch.from_numpy(mask[sel][None, :, :]).to(device)
        with torch.no_grad():
            fake = G(z, c, m).cpu().numpy()[0]
        real = data[sel].copy()
        if denorm and dataset != "ddre33" and "x_min" in cfg and "x_max" in cfg:
            x_min = np.array(cfg["x_min"], dtype=np.float32)
            x_max = np.array(cfg["x_max"], dtype=np.float32)
            real = real * (x_max[None, :] - x_min[None, :]) + x_min[None, :]
            fake = fake * (x_max[None, :] - x_min[None, :]) + x_min[None, :]
        cases.append(
            {
                "name": f"PV climate {pv_idx} (wind={wind_idx})",
                "pv_climate": pv_idx,
                "wind_climate": wind_idx,
                "sample_index": sel,
                "mae_pv": _mae(real, fake, 0),
                "mae_wind": _mae(real, fake, 1) if real.shape[1] > 1 else None,
                "real": real.tolist(),
                "fake": fake.tolist(),
            }
        )

    # Wind climates (0-3): pick real sample with wind climate j, keep its pv climate
    for wind_idx in range(4):
        candidates = np.where(wind_labels == wind_idx)[0]
        if candidates.size == 0:
            continue
        sel = int(rng.choice(candidates, size=1)[0])
        pv_idx = int(pv_labels[sel])
        extra = None
        if cond_mean.shape[1] == 12:
            if cond.ndim == 2:
                extra = cond[sel][10:12]
            else:
                extra = cond[sel][0, 10:12]
        cond_arr = build_onehot(pv_idx, wind_idx, scfg.seq_len, extra=extra)
        z = torch.randn(1, cfg["z_dim"], device=device)
        c = torch.from_numpy(cond_arr[None, :, :]).to(device)
        m = torch.from_numpy(mask[sel][None, :, :]).to(device)
        with torch.no_grad():
            fake = G(z, c, m).cpu().numpy()[0]
        real = data[sel].copy()
        if denorm and dataset != "ddre33" and "x_min" in cfg and "x_max" in cfg:
            x_min = np.array(cfg["x_min"], dtype=np.float32)
            x_max = np.array(cfg["x_max"], dtype=np.float32)
            real = real * (x_max[None, :] - x_min[None, :]) + x_min[None, :]
            fake = fake * (x_max[None, :] - x_min[None, :]) + x_min[None, :]
        cases.append(
            {
                "name": f"Wind climate {wind_idx} (pv={pv_idx})",
                "pv_climate": pv_idx,
                "wind_climate": wind_idx,
                "sample_index": sel,
                "mae_pv": _mae(real, fake, 0),
                "mae_wind": _mae(real, fake, 1) if real.shape[1] > 1 else None,
                "real": real.tolist(),
                "fake": fake.tolist(),
            }
        )

    return {"cases": cases, "count": len(cases), "seq_len": int(scfg.seq_len), "channels": int(scfg.channels)}


@app.route("/", methods=["GET", "POST"])
def index():
    checkpoints = list_checkpoints()
    selected = None
    error = None
    results = {}
    train_results = {}

    form_data = {
        "action": "test",
        "ckpt": str(checkpoints[0]) if checkpoints else "",
        "train_data_csv": str(DEFAULT_DATA) if DEFAULT_DATA.exists() else "",
        "test_data_csv": str(DEFAULT_DATA) if DEFAULT_DATA.exists() else "",
        "train_resolution": "hourly",
        "test_resolution": "hourly",
        "train_mask_source": "solar",
        "test_mask_source": "ghi",
        "cond_norm": "minmax",
        "train_dataset": "real",
        "dataset": "real",
        "train_ddre33_mode": "2ch_single",
        "ddre33_mode": "2ch_single",
        "ddre_data_source": "ororiginal_data",
        "train_cond_onehot": "on",
        "cond_onehot": "on",
        "train_ddre33_date_cond": "on",
        "ddre33_date_cond": "on",
        "train_ddre33_static_cond": "on",
        "ddre33_static_cond": "on",
        "train_max_cols": "100",
        "max_cols": "100",
        "train_pv18_csv": str(DDRE_DIR / "node_18_PV.csv"),
        "train_pv33_csv": str(DDRE_DIR / "node_33_PV.csv"),
        "train_wind22_csv": str(DDRE_DIR / "node_22_wind.csv"),
        "train_wind25_csv": str(DDRE_DIR / "node_25_wind.csv"),
        "train_pv18_labels_csv": str(DDRE_DIR / "node_18_PV_labels_climate.csv"),
        "train_pv33_labels_csv": str(DDRE_DIR / "node_33_PV_labels_climate.csv"),
        "train_wind22_labels_csv": str(DDRE_DIR / "node_22_wind_labels.csv"),
        "train_wind25_labels_csv": str(DDRE_DIR / "node_25_wind_labels.csv"),
        "pv18_csv": str(DDRE_DIR / "node_18_PV.csv"),
        "pv33_csv": str(DDRE_DIR / "node_33_PV.csv"),
        "wind22_csv": str(DDRE_DIR / "node_22_wind.csv"),
        "wind25_csv": str(DDRE_DIR / "node_25_wind.csv"),
        "pv18_labels_csv": str(DDRE_DIR / "node_18_PV_labels_climate.csv"),
        "pv33_labels_csv": str(DDRE_DIR / "node_33_PV_labels_climate.csv"),
        "wind22_labels_csv": str(DDRE_DIR / "node_22_wind_labels.csv"),
        "wind25_labels_csv": str(DDRE_DIR / "node_25_wind_labels.csv"),
        "num": "8",
        "topk_candidates": "1",
        "topk_keep": "1",
        "seed": "42",
        "test_denorm": "on",
        "test_use_venv": "",
        "train_use_venv": "",
        "epochs": "10",
        "batch": "32",
        "lr": "0.0002",
        "z_dim": "64",
        "d_steps": "2",
        "lambda_gp": "5.0",
        "lambda_tailq": "0.1",
        "tail_q": "0.95",
        "lambda_stats": "0.05",
        "save_every": "5",
        "outdir": "pcm_gan_runs",
    }

    if request.method == "POST":
        try:
            action = request.form.get("action", "test")
            for key in form_data:
                if key in request.form:
                    form_data[key] = request.form.get(key)
            selected = form_data["ckpt"]

            if action == "train":
                train_args = {
                    "dataset": form_data.get("train_dataset", "real"),
                    "data_csv": form_data["train_data_csv"],
                    "resolution": form_data["train_resolution"],
                    "mask_source": form_data["train_mask_source"],
                    "cond_norm": form_data.get("cond_norm", "none"),
                    "epochs": int(form_data["epochs"]),
                    "batch": int(form_data["batch"]),
                    "lr": float(form_data["lr"]),
                    "z_dim": int(form_data["z_dim"]),
                    "d_steps": int(form_data["d_steps"]),
                    "lambda_gp": float(form_data["lambda_gp"]),
                    "lambda_tailq": float(form_data["lambda_tailq"]),
                    "tail_q": float(form_data["tail_q"]),
                    "lambda_stats": float(form_data["lambda_stats"]),
                    "save_every": int(form_data["save_every"]),
                    "outdir": form_data["outdir"],
                    "use_venv": form_data.get("train_use_venv") == "on",
                    "ddre33_mode": form_data.get("train_ddre33_mode", "2ch_single"),
                    "cond_onehot": form_data.get("train_cond_onehot") == "on",
                    "ddre33_date_cond": form_data.get("train_ddre33_date_cond") == "on",
                    "ddre33_static_cond": form_data.get("train_ddre33_static_cond") == "on",
                    "max_cols": int(form_data.get("train_max_cols") or 0),
                    "pv18_csv": form_data.get("train_pv18_csv", ""),
                    "pv33_csv": form_data.get("train_pv33_csv", ""),
                    "wind22_csv": form_data.get("train_wind22_csv", ""),
                    "wind25_csv": form_data.get("train_wind25_csv", ""),
                    "pv18_labels_csv": form_data.get("train_pv18_labels_csv", ""),
                    "pv33_labels_csv": form_data.get("train_pv33_labels_csv", ""),
                    "wind22_labels_csv": form_data.get("train_wind22_labels_csv", ""),
                    "wind25_labels_csv": form_data.get("train_wind25_labels_csv", ""),
                }
                job_id = start_training_job(train_args)
                train_results = {
                    "job_id": job_id,
                    "status": "running",
                }
            else:
                args = {
                    "ckpt": form_data["ckpt"],
                    "data_csv": form_data["test_data_csv"],
                    "dataset": form_data.get("dataset", "real"),
                    "resolution": form_data["test_resolution"],
                    "mask_source": form_data["test_mask_source"],
                    "cond_norm": form_data.get("cond_norm", "none"),
                    "num": int(form_data["num"]),
                    "topk_candidates": int(form_data.get("topk_candidates") or 1),
                    "topk_keep": int(form_data.get("topk_keep") or 1),
                    "seed": int(form_data["seed"]),
                    "denorm": form_data.get("test_denorm") == "on",
                    "use_venv": form_data.get("test_use_venv") == "on",
                    "x_cols": "solar_power,wind_power,load_power",
                    "cond_cols": "DHI,DNI,GHI,Dew Point,Solar Zenith Angle,Wind Speed,Relative Humidity,Temperature",
                    "cond_agg": "mean",
                    "ddre33_mode": form_data.get("ddre33_mode", "2ch_single"),
                    "cond_onehot": form_data.get("cond_onehot") == "on",
                    "ddre33_date_cond": form_data.get("ddre33_date_cond") == "on",
                    "ddre33_static_cond": form_data.get("ddre33_static_cond") == "on",
                    "max_cols": int(form_data.get("max_cols") or 0),
                    "pv18_csv": form_data.get("pv18_csv", ""),
                    "pv33_csv": form_data.get("pv33_csv", ""),
                    "wind22_csv": form_data.get("wind22_csv", ""),
                    "wind25_csv": form_data.get("wind25_csv", ""),
                    "pv18_labels_csv": form_data.get("pv18_labels_csv", ""),
                    "pv33_labels_csv": form_data.get("pv33_labels_csv", ""),
                    "wind22_labels_csv": form_data.get("wind22_labels_csv", ""),
                    "wind25_labels_csv": form_data.get("wind25_labels_csv", ""),
                }
                run_id = time.strftime("%Y%m%d_%H%M%S")
                run_dir = RESULT_DIR / f"run_{run_id}"
                run_dir.mkdir(parents=True, exist_ok=True)

                if args["use_venv"]:
                    args["outdir"] = str(run_dir)
                    samples, conds = generate_samples_subprocess(args, use_venv=True)
                    cond_stats, cache_path = {}, None
                else:
                    set_seed(int(form_data["seed"]))
                    samples, conds, cond_stats, cache_path = generate_samples(args)
                    np.save(run_dir / "samples.npy", samples)
                    np.save(run_dir / "conds.npy", conds)

                csv_paths = []
                header = (
                    "pv,wind"
                    if args.get("dataset") == "ddre33"
                    else "solar_power,wind_power,load_power"
                )
                for i in range(samples.shape[0]):
                    csv_path = run_dir / f"sample_{i:03d}.csv"
                    if not csv_path.exists():
                        np.savetxt(csv_path, samples[i], delimiter=",", header=header, comments="")
                    csv_paths.append(csv_path.name)

                results = {
                    "run_dir": run_dir.name,
                    "csv_files": csv_paths,
                    "samples": samples.tolist(),
                    "conds": conds.tolist(),
                    "cond_stats": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cond_stats.items()},
                    "header": header.split(","),
                    "cache_path": cache_path,
                }
        except Exception as exc:  # pragma: no cover
            error = str(exc)

    return render_template(
        "index.html",
        checkpoints=checkpoints,
        selected=selected,
        results=results,
        train_results=train_results,
        error=error,
        form_data=form_data,
    )


@app.route("/results/<run>/<path:filename>")
def serve_result(run, filename):
    run_dir = RESULT_DIR / run
    return send_from_directory(run_dir, filename, as_attachment=False)


@app.route("/train_status/<job_id>")
def train_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return {"error": "job not found"}, 404

    metrics_path = Path(job["metrics_path"])
    out_log = Path(job["log_out"])
    err_log = Path(job["log_err"])
    metrics = load_metrics(metrics_path, limit=200, start_row=int(job.get("start_rows", 0)))
    stdout_tail = tail_text(out_log, max_chars=4000)
    stderr_tail = tail_text(err_log, max_chars=4000)
    progress_pct = extract_tqdm_progress(stderr_tail) or extract_tqdm_progress(stdout_tail)
    current_epoch = 0
    total_epochs = int(job.get("total_epochs", 0) or 0)
    if progress_pct is None:
        progress_pct = 0
    if metrics:
        try:
            current_epoch = int(metrics[-1].get("epoch", 0))
        except Exception:
            current_epoch = 0
    latest = metrics[-1] if metrics else None
    return {
        "status": job["status"],
        "returncode": job.get("returncode"),
        "metrics": metrics,
        "latest": latest,
        "metrics_path": str(metrics_path),
        "stdout": stdout_tail,
        "stderr": stderr_tail,
        "elapsed": int(time.time() - job["start_time"]),
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "progress_pct": progress_pct,
    }


@app.route("/checkpoints")
def checkpoints_api():
    items = [str(p) for p in list_checkpoints()]
    return {"checkpoints": items}


@app.route("/train_stop/<job_id>", methods=["POST"])
def train_stop(job_id):
    with JOBS_LOCK:
        proc = PROCS.get(job_id)
        job = JOBS.get(job_id)
    if not job:
        return {"error": "job not found"}, 404
    if proc is None or job.get("status") != "running":
        return {"status": job.get("status", "unknown"), "message": "job not running"}
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, text=True)
        else:
            proc.terminate()
        with JOBS_LOCK:
            job["status"] = "stopping"
            JOBS[job_id] = job
        return {"status": "stopping"}
    except Exception as exc:
        return {"error": str(exc)}, 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
