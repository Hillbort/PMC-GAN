import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


@dataclass
class RunningStats:
    count: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self.count += int(x.size)
        self.sum += float(np.sum(x))
        self.sumsq += float(np.sum(x * x))
        self.min = min(self.min, float(np.min(x)))
        self.max = max(self.max, float(np.max(x)))

    def mean(self) -> float:
        return self.sum / max(self.count, 1)

    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        var = self.sumsq / self.count - (self.sum / self.count) ** 2
        return float(np.sqrt(max(var, 0.0)))


def reservoir_sample(
    rng: np.random.Generator, sample: Optional[np.ndarray], x: np.ndarray, k: int, seen: int
) -> Tuple[np.ndarray, int]:
    if k <= 0:
        return np.empty((0,), dtype=np.float64), seen
    if sample is None:
        sample = np.empty((0,), dtype=np.float64)
    x = x.astype(np.float64, copy=False)
    if sample.size < k:
        need = min(k - sample.size, x.size)
        if need > 0:
            sample = np.concatenate([sample, x[:need]])
            x = x[need:]
            seen += need
    for val in x:
        seen += 1
        j = rng.integers(0, seen)
        if j < k:
            sample[j] = val
    return sample, seen


def bin_stats(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> List[Dict[str, float]]:
    # return for each bin: count, zero_ratio, y_mean
    idx = np.digitize(x, bins) - 1
    out = []
    for i in range(len(bins) - 1):
        sel = idx == i
        xi = x[sel]
        yi = y[sel]
        if yi.size == 0:
            out.append(
                {
                    "bin_left": float(bins[i]),
                    "bin_right": float(bins[i + 1]),
                    "count": 0,
                    "zero_ratio": float("nan"),
                    "y_mean": float("nan"),
                }
            )
            continue
        out.append(
            {
                "bin_left": float(bins[i]),
                "bin_right": float(bins[i + 1]),
                "count": int(yi.size),
                "zero_ratio": float(np.mean(yi == 0)),
                "y_mean": float(np.mean(yi)),
            }
        )
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Analyze raw CSV distributions.")
    p.add_argument("--csv", type=str, default="", help="Path to ERCOT_zone_1_.csv")
    p.add_argument("--chunksize", type=int, default=200000)
    p.add_argument("--sample_n", type=int, default=200000, help="Reservoir sample size for quantiles")
    p.add_argument("--wind_col", type=str, default="wind_power")
    p.add_argument("--wind_speed_col", type=str, default="Wind Speed")
    p.add_argument("--solar_col", type=str, default="solar_power")
    p.add_argument("--ghi_col", type=str, default="GHI")
    p.add_argument("--load_col", type=str, default="load_power")
    p.add_argument("--bins", type=int, default=10, help="Number of bins for conditional stats")
    p.add_argument(
        "--wind_speed_thresholds",
        type=str,
        default="3,5,8",
        help="Comma-separated wind speed thresholds for counting (e.g., '3,5,8').",
    )
    p.add_argument("--ddre33", action="store_true", help="Analyze DDRE-33 PV/Wind wide CSVs.")
    p.add_argument("--pv18_csv", type=str, default="")
    p.add_argument("--pv33_csv", type=str, default="")
    p.add_argument("--wind22_csv", type=str, default="")
    p.add_argument("--wind25_csv", type=str, default="")
    return p.parse_args()


def _analyze_ddre33(args) -> None:
    files = {
        "pv18": args.pv18_csv,
        "pv33": args.pv33_csv,
        "wind22": args.wind22_csv,
        "wind25": args.wind25_csv,
    }
    missing = [k for k, v in files.items() if not v]
    if missing:
        raise ValueError(f"Missing DDRE-33 CSVs: {missing}")

    print("== DDRE-33 Wide CSV Stats ==")
    for key, path in files.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        vmin = np.inf
        vmax = -np.inf
        vsum = 0.0
        vsumsq = 0.0
        vcount = 0
        for chunk in pd.read_csv(path, chunksize=20000):
            if "timestamp" in chunk.columns:
                chunk = chunk.drop(columns=["timestamp"])
            chunk = chunk.apply(pd.to_numeric, errors="coerce")
            arr = chunk.to_numpy(dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            vmin = min(vmin, float(arr.min()))
            vmax = max(vmax, float(arr.max()))
            vsum += float(arr.sum())
            vsumsq += float((arr * arr).sum())
            vcount += int(arr.size)
        if vcount == 0:
            print(f"{key}: no data")
            continue
        mean = vsum / vcount
        var = max(vsumsq / vcount - mean * mean, 0.0)
        std = var ** 0.5
        in_01 = vmin >= -1e-6 and vmax <= 1.0 + 1e-6
        print(
            f"{key}: min={vmin:.6f} max={vmax:.6f} mean={mean:.6f} std={std:.6f} "
            f"in_[0,1]={in_01}"
        )


def main():
    if pd is None:
        raise RuntimeError("pandas is required. Please `pip install pandas`.")
    args = parse_args()
    rng = np.random.default_rng(42)

    if args.ddre33:
        _analyze_ddre33(args)
        return

    if not args.csv:
        raise ValueError("--csv is required unless --ddre33 is set.")

    cols = [
        args.wind_col,
        args.wind_speed_col,
        args.solar_col,
        args.ghi_col,
        args.load_col,
    ]

    stats: Dict[str, RunningStats] = {c: RunningStats() for c in cols}
    wind_zero_speed_sample = None
    wind_pos_speed_sample = None
    solar_zero_ghi_sample = None
    solar_pos_ghi_sample = None
    wind_power_sample = None
    solar_power_sample = None
    load_power_sample = None
    ghi_sample = None
    wind_speed_sample = None
    seen_wind_zero = 0
    seen_wind_pos = 0
    seen_solar_zero = 0
    seen_solar_pos = 0
    seen_wind = 0
    seen_solar = 0
    seen_load = 0
    seen_ghi = 0
    seen_wspeed = 0
    wind_speed_counts = {}
    wind_speed_samples_seen = 0

    for chunk in pd.read_csv(args.csv, usecols=lambda c: c in cols, chunksize=args.chunksize):
        for c in cols:
            if c not in chunk.columns:
                raise ValueError(f"Missing column: {c}")
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        chunk = chunk.dropna(subset=cols)
        if chunk.empty:
            continue

        for c in cols:
            stats[c].update(chunk[c].to_numpy(dtype=np.float64))

        wind = chunk[args.wind_col].to_numpy(dtype=np.float64)
        wind_speed = chunk[args.wind_speed_col].to_numpy(dtype=np.float64)
        solar = chunk[args.solar_col].to_numpy(dtype=np.float64)
        ghi = chunk[args.ghi_col].to_numpy(dtype=np.float64)
        load = chunk[args.load_col].to_numpy(dtype=np.float64)

        wind_power_sample, seen_wind = reservoir_sample(
            rng, wind_power_sample, wind, args.sample_n, seen_wind
        )
        wind_speed_sample, seen_wspeed = reservoir_sample(
            rng, wind_speed_sample, wind_speed, args.sample_n, seen_wspeed
        )
        solar_power_sample, seen_solar = reservoir_sample(
            rng, solar_power_sample, solar, args.sample_n, seen_solar
        )
        ghi_sample, seen_ghi = reservoir_sample(rng, ghi_sample, ghi, args.sample_n, seen_ghi)
        load_power_sample, seen_load = reservoir_sample(
            rng, load_power_sample, load, args.sample_n, seen_load
        )

        wz = wind == 0
        wp = wind > 0
        wind_zero_speed_sample, seen_wind_zero = reservoir_sample(
            rng, wind_zero_speed_sample, wind_speed[wz], args.sample_n, seen_wind_zero
        )
        wind_pos_speed_sample, seen_wind_pos = reservoir_sample(
            rng, wind_pos_speed_sample, wind_speed[wp], args.sample_n, seen_wind_pos
        )

        sz = solar == 0
        sp = solar > 0
        solar_zero_ghi_sample, seen_solar_zero = reservoir_sample(
            rng, solar_zero_ghi_sample, ghi[sz], args.sample_n, seen_solar_zero
        )
        solar_pos_ghi_sample, seen_solar_pos = reservoir_sample(
            rng, solar_pos_ghi_sample, ghi[sp], args.sample_n, seen_solar_pos
        )

        # Wind speed threshold counts (full data, not sampled)
        if not wind_speed_counts:
            thresholds = []
            for t in args.wind_speed_thresholds.split(","):
                t = t.strip()
                if not t:
                    continue
                try:
                    thresholds.append(float(t))
                except ValueError:
                    raise ValueError(f"Invalid threshold in --wind_speed_thresholds: {t}")
            wind_speed_counts = {thr: 0 for thr in thresholds}
        wind_speed_samples_seen += wind_speed.size
        for thr in wind_speed_counts:
            wind_speed_counts[thr] += int(np.sum(wind_speed > thr))

    def qstats(x: np.ndarray) -> Dict[str, float]:
        if x is None or x.size == 0:
            return {"p01": float("nan"), "p50": float("nan"), "p99": float("nan")}
        return {
            "p01": float(np.quantile(x, 0.01)),
            "p50": float(np.quantile(x, 0.50)),
            "p99": float(np.quantile(x, 0.99)),
        }

    print("== Basic Stats ==")
    for c in cols:
        s = stats[c]
        print(
            f"{c}: count={s.count} min={s.min:.4f} max={s.max:.4f} mean={s.mean():.4f} std={s.std():.4f}"
        )

    print("\n== Quantiles (sampled) ==")
    print(f"{args.wind_col}: {qstats(wind_power_sample)}")
    print(f"{args.wind_speed_col}: {qstats(wind_speed_sample)}")
    print(f"{args.solar_col}: {qstats(solar_power_sample)}")
    print(f"{args.ghi_col}: {qstats(ghi_sample)}")
    print(f"{args.load_col}: {qstats(load_power_sample)}")

    print("\n== Zero-power thresholds (sampled) ==")
    print(f"Wind speed when wind_power==0: {qstats(wind_zero_speed_sample)}")
    print(f"Wind speed when wind_power>0: {qstats(wind_pos_speed_sample)}")
    print(f"GHI when solar_power==0: {qstats(solar_zero_ghi_sample)}")
    print(f"GHI when solar_power>0: {qstats(solar_pos_ghi_sample)}")

    if wind_speed_counts:
        print("\n== Wind Speed Threshold Counts ==")
        for thr in sorted(wind_speed_counts.keys()):
            cnt = wind_speed_counts[thr]
            ratio = cnt / max(wind_speed_samples_seen, 1)
            print(f">{thr:g}: count={cnt} ratio={ratio:.4f}")

    # conditional bin stats
    if wind_speed_sample is not None and wind_speed_sample.size >= args.bins:
        bins = np.quantile(wind_speed_sample, np.linspace(0, 1, args.bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        print("\n== Wind power vs wind speed bins ==")
        # Use sampled pairs to avoid second pass
        # Approximate by pairing sampled wind_speed with nearest sampled wind_power (same indices)
        n = min(wind_speed_sample.size, wind_power_sample.size)
        x = wind_speed_sample[:n]
        y = wind_power_sample[:n]
        for row in bin_stats(x, y, bins):
            print(
                f"[{row['bin_left']:.3f}, {row['bin_right']:.3f}) "
                f"count={row['count']} zero_ratio={row['zero_ratio']:.3f} y_mean={row['y_mean']:.4f}"
            )

    if ghi_sample is not None and ghi_sample.size >= args.bins:
        bins = np.quantile(ghi_sample, np.linspace(0, 1, args.bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        print("\n== Solar power vs GHI bins ==")
        n = min(ghi_sample.size, solar_power_sample.size)
        x = ghi_sample[:n]
        y = solar_power_sample[:n]
        for row in bin_stats(x, y, bins):
            print(
                f"[{row['bin_left']:.3f}, {row['bin_right']:.3f}) "
                f"count={row['count']} zero_ratio={row['zero_ratio']:.3f} y_mean={row['y_mean']:.4f}"
            )


if __name__ == "__main__":
    main()
