from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pandas as pd
except Exception:
    pd = None


def _parse_label_map(labels_csv: str):
    if pd is None:
        raise RuntimeError("pandas is required for loading labels. Please `pip install pandas`.")
    df = pd.read_csv(labels_csv)
    if "Index" in df.columns and "Type" in df.columns:
        return {str(k): int(v) for k, v in zip(df["Index"].tolist(), df["Type"].tolist())}
    # Fallback: use first row of wide label format (Series_1...Series_N)
    if df.shape[0] > 0:
        row = df.iloc[0].tolist()
        return {str(c): int(v) for c, v in zip(df.columns.tolist(), row)}
    raise ValueError(f"Unsupported label file format: {labels_csv}")


def _sorted_cols(cols, prefix):
    def key_fn(c):
        if not c.startswith(prefix):
            return (1, c)
        try:
            idx = int(c.replace(prefix, "").strip())
        except Exception:
            return (1, c)
        return (0, idx)

    return sorted(cols, key=key_fn)


def _one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def _curve_control_features(x_day: np.ndarray, channel_names: List[str], active_eps: float = 0.05):
    feats = []
    names = []
    if x_day.ndim != 2:
        raise ValueError(f"x_day must be 2D, got shape {x_day.shape}")
    for ch, ch_name in enumerate(channel_names):
        s = x_day[:, ch].astype(np.float32, copy=False)
        peak = float(np.max(s)) if s.size else 0.0
        ramp = np.diff(s) if s.size > 1 else np.zeros((0,), dtype=np.float32)
        active_thr = max(float(active_eps), 0.05 * peak)
        feats.extend(
            [
                float(np.mean(s)) if s.size else 0.0,
                peak,
                float(np.std(ramp)) if ramp.size else 0.0,
                float(np.mean(s > active_thr)) if s.size else 0.0,
            ]
        )
        names.extend(
            [
                f"{ch_name}_mean",
                f"{ch_name}_peak",
                f"{ch_name}_ramp_std",
                f"{ch_name}_active_ratio",
            ]
        )
    return np.asarray(feats, dtype=np.float32), names


def ddre33_cond_layout(mode: str, one_hot: bool, add_date_cond: bool = False, add_curve_cond: bool = False):
    if mode in ("2ch_pairs", "2ch_single"):
        x_cols = ["pv", "wind"]
        if one_hot:
            cond_cols = [f"pv_c{i}" for i in range(6)] + [f"wind_c{i}" for i in range(4)]
        else:
            cond_cols = ["pv_c", "wind_c"]
    else:
        x_cols = ["pv18", "pv33", "wind22", "wind25"]
        if one_hot:
            cond_cols = [f"pv18_c{i}" for i in range(6)] + [f"pv33_c{i}" for i in range(6)]
            cond_cols += [f"wind22_c{i}" for i in range(4)] + [f"wind25_c{i}" for i in range(4)]
        else:
            cond_cols = ["pv18_c", "pv33_c", "wind22_c", "wind25_c"]
    if add_date_cond:
        cond_cols += ["doy_sin", "doy_cos"]
    if add_curve_cond:
        for name in x_cols:
            cond_cols += [
                f"{name}_mean",
                f"{name}_peak",
                f"{name}_ramp_std",
                f"{name}_active_ratio",
            ]
    return x_cols, cond_cols


class RealDataset(Dataset):
    def __init__(self, data, cond, mask):
        self.data = data if isinstance(data, torch.Tensor) else torch.from_numpy(data)
        self.cond = cond if isinstance(cond, torch.Tensor) else torch.from_numpy(cond)
        self.mask = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        c = self.cond[idx]
        if self.mask.ndim == 3:
            m = self.mask[idx]
        else:
            m = self.mask
        return x, c, m


def load_real_dataset(
    csv_path: str,
    seq_len: int,
    x_cols: List[str],
    cond_cols: List[str],
    resample_rule: str = "H",
    cond_agg: str = "mean",
    x_agg: str = "sum",
    mask_source: str = "solar",
    cond_norm: str = "none",
    norm_eps: float = 1e-6,
):
    if pd is None:
        raise RuntimeError("pandas is required for loading real data. Please `pip install pandas`.")

    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("CSV must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    missing = [c for c in (x_cols + cond_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # coerce numeric columns (non-numeric -> NaN) and drop invalid rows
    for col in x_cols + cond_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=x_cols + cond_cols)

    # pandas >= 2.2 deprecates 'H'
    if resample_rule == "H":
        resample_rule = "h"

    # resample to target resolution then build daily sequences
    if x_agg == "sum":
        x_res = df[x_cols].resample(resample_rule).sum()
    elif x_agg == "mean":
        x_res = df[x_cols].resample(resample_rule).mean()
    else:
        raise ValueError(f"Unsupported x_agg: {x_agg}")

    if cond_agg == "mean":
        c_res = df[cond_cols].resample(resample_rule).mean()
    elif cond_agg == "max":
        c_res = df[cond_cols].resample(resample_rule).max()
    elif cond_agg == "min":
        c_res = df[cond_cols].resample(resample_rule).min()
    else:
        raise ValueError(f"Unsupported cond_agg: {cond_agg}")

    df_res = x_res.join(c_res, how="inner").dropna()

    # group by date to build daily sequences
    x_list = []
    c_list = []
    mask_list = []
    for date, g in df_res.groupby(df_res.index.date):
        if len(g) != seq_len:
            continue
        x_vals = g[x_cols].to_numpy(dtype=np.float32)
        # cond is a sequence aligned with x (B, T, D)
        c_vals = g[cond_cols].to_numpy(dtype=np.float32)
        if mask_source == "solar":
            m_vals = (x_vals[:, 0] > 0).astype(np.float32)
        elif mask_source == "ghi":
            if "GHI" not in g.columns:
                raise ValueError("mask_source=ghi requires 'GHI' in cond columns.")
            m_vals = (g["GHI"].to_numpy(dtype=np.float32) > 0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported mask_source: {mask_source}")
        x_list.append(x_vals)
        c_list.append(c_vals)
        mask_list.append(m_vals)

    if not x_list:
        raise ValueError("No full daily sequences found. Check resample_rule and seq_len.")

    data = np.stack(x_list, axis=0)
    cond = np.stack(c_list, axis=0)
    # apply log1p to wind channel before min-max normalization
    wind_idx = x_cols.index("wind_power") if "wind_power" in x_cols else None
    x_transform = "none"
    x_transform_channel = None
    if wind_idx is not None:
        data[..., wind_idx] = np.log1p(np.maximum(data[..., wind_idx], 0.0))
        x_transform = "log1p_wind"
        x_transform_channel = int(wind_idx)
    # min-max normalize x to [0, 1] per channel
    x_min = data.min(axis=(0, 1))
    x_max = data.max(axis=(0, 1))
    denom = np.maximum(x_max - x_min, norm_eps)
    data = (data - x_min[None, None, :]) / denom[None, None, :]
    # optional cond normalization
    if cond_norm == "minmax":
        c_min = cond.min(axis=(0, 1))
        c_max = cond.max(axis=(0, 1))
        c_denom = np.maximum(c_max - c_min, norm_eps)
        cond = (cond - c_min[None, None, :]) / c_denom[None, None, :]
    elif cond_norm == "zscore":
        c_mean = cond.mean(axis=(0, 1))
        c_std = cond.std(axis=(0, 1))
        c_std = np.maximum(c_std, norm_eps)
        cond = (cond - c_mean[None, None, :]) / c_std[None, None, :]
    elif cond_norm != "none":
        raise ValueError(f"Unsupported cond_norm: {cond_norm}")
    mask = np.ones_like(data, dtype=np.float32)
    mask[:, :, 0] = np.stack(mask_list, axis=0)
    cond_stats = {}
    if cond_norm == "minmax":
        cond_stats = {
            "cond_min": c_min.astype(np.float32),
            "cond_max": c_max.astype(np.float32),
            "cond_norm": "minmax",
        }
    elif cond_norm == "zscore":
        cond_stats = {
            "cond_mean": c_mean.astype(np.float32),
            "cond_std": c_std.astype(np.float32),
            "cond_norm": "zscore",
        }
    else:
        cond_stats = {"cond_norm": "none"}
    cond_stats["x_transform"] = x_transform
    cond_stats["x_transform_channel"] = x_transform_channel
    return data, cond, mask, x_min.astype(np.float32), x_max.astype(np.float32), cond_stats


def load_ddre33_dataset(
    pv18_csv: str,
    pv33_csv: str,
    wind22_csv: str,
    wind25_csv: str,
    pv18_labels_csv: str,
    pv33_labels_csv: str,
    wind22_labels_csv: str,
    wind25_labels_csv: str,
    seq_len: int = 96,
    resample_rule: str = "15T",
    one_hot: bool = True,
    mode: str = "4ch",
    max_cols: int = 0,
    normalize: bool = False,
    add_date_cond: bool = False,
    static_cond: bool = True,
    add_curve_cond: bool = False,
    curve_cond_norm: str = "none",
):
    if pd is None:
        raise RuntimeError("pandas is required for loading DDRE-33 data. Please `pip install pandas`.")

    def read_series(path):
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"CSV must contain a 'timestamp' column: {path}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        # ensure numeric values
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    pv18 = read_series(pv18_csv)
    pv33 = read_series(pv33_csv)
    w22 = read_series(wind22_csv)
    w25 = read_series(wind25_csv)

    if resample_rule == "T":
        resample_rule = "min"
    if resample_rule == "15T":
        resample_rule = "15min"
    if resample_rule:
        pv18 = pv18.resample(resample_rule).mean()
        pv33 = pv33.resample(resample_rule).mean()
        w22 = w22.resample(resample_rule).mean()
        w25 = w25.resample(resample_rule).mean()

    # Fill gaps to avoid NaNs after resample
    pv18 = pv18.interpolate(method="time").ffill().bfill()
    pv33 = pv33.interpolate(method="time").ffill().bfill()
    w22 = w22.interpolate(method="time").ffill().bfill()
    w25 = w25.interpolate(method="time").ffill().bfill()

    pv18_cols = _sorted_cols(pv18.columns.tolist(), "PV_power_")
    pv33_cols = _sorted_cols(pv33.columns.tolist(), "PV_power_")
    w22_cols = _sorted_cols(w22.columns.tolist(), "wind_power_")
    w25_cols = _sorted_cols(w25.columns.tolist(), "wind_power_")

    n = min(len(pv18_cols), len(pv33_cols), len(w22_cols), len(w25_cols))
    if max_cols and max_cols > 0:
        n = min(n, int(max_cols))
    pv18_cols = pv18_cols[:n]
    pv33_cols = pv33_cols[:n]
    w22_cols = w22_cols[:n]
    w25_cols = w25_cols[:n]

    lab_pv18 = _parse_label_map(pv18_labels_csv)
    lab_pv33 = _parse_label_map(pv33_labels_csv)
    lab_w22 = _parse_label_map(wind22_labels_csv)
    lab_w25 = _parse_label_map(wind25_labels_csv)

    # climate label sizes: PV 0-5, wind 0-3
    pv_classes = 6
    wind_classes = 4

    x_list = []
    c_list = []
    m_list = []
    curve_feature_names = None
    def to_daily_map(df):
        return {k: g for k, g in df.groupby(df.index.date)}

    pv18_map = to_daily_map(pv18)
    pv33_map = to_daily_map(pv33)
    w22_map = to_daily_map(w22)
    w25_map = to_daily_map(w25)
    common_dates = sorted(set(pv18_map) & set(pv33_map) & set(w22_map) & set(w25_map))

    for d in common_dates:
        if add_date_cond:
            doy = int(getattr(d, "timetuple", lambda: None)().tm_yday) if hasattr(d, "timetuple") else None
            if doy is None:
                doy = 1
            angle = 2.0 * np.pi * (float(doy - 1) / 365.0)
            date_feat = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
        else:
            date_feat = None
        pv18_day = pv18_map[d]
        pv33_day = pv33_map[d]
        w22_day = w22_map[d]
        w25_day = w25_map[d]
        if (
            len(pv18_day) != seq_len
            or len(pv33_day) != seq_len
            or len(w22_day) != seq_len
            or len(w25_day) != seq_len
        ):
            continue
        pv18_vals = pv18_day[pv18_cols].to_numpy(dtype=np.float32)
        pv33_vals = pv33_day[pv33_cols].to_numpy(dtype=np.float32)
        w22_vals = w22_day[w22_cols].to_numpy(dtype=np.float32)
        w25_vals = w25_day[w25_cols].to_numpy(dtype=np.float32)
        for i in range(n):
            if mode == "2ch_pairs":
                # Pair A: pv18 + wind22
                x_day = np.stack([pv18_vals[:, i], w22_vals[:, i]], axis=1)
                ch_names = ["pv", "wind"]
                if one_hot:
                    c_vec = np.concatenate(
                        [
                            _one_hot(lab_pv18.get(pv18_cols[i], 0), pv_classes),
                            _one_hot(lab_w22.get(w22_cols[i], 0), wind_classes),
                        ],
                        axis=0,
                    )
                else:
                    c_vec = np.array(
                        [lab_pv18.get(pv18_cols[i], 0), lab_w22.get(w22_cols[i], 0)],
                        dtype=np.float32,
                    )
                if date_feat is not None:
                    c_vec = np.concatenate([c_vec, date_feat], axis=0)
                if add_curve_cond:
                    curve_vec, curve_names = _curve_control_features(x_day, ch_names)
                    c_vec = np.concatenate([c_vec, curve_vec], axis=0)
                    if curve_feature_names is None:
                        curve_feature_names = curve_names
                c_day = c_vec if static_cond else np.repeat(c_vec[None, :], seq_len, axis=0)
                m_day = np.ones_like(x_day, dtype=np.float32)
                m_day[:, 0] = (x_day[:, 0] > 0).astype(np.float32)
                m_day[:, 1] = (x_day[:, 1] > 0).astype(np.float32)
                x_list.append(x_day)
                c_list.append(c_day)
                m_list.append(m_day)

                # Pair B: pv33 + wind25 (treated as another sample)
                x_day = np.stack([pv33_vals[:, i], w25_vals[:, i]], axis=1)
                ch_names = ["pv", "wind"]
                if one_hot:
                    c_vec = np.concatenate(
                        [
                            _one_hot(lab_pv33.get(pv33_cols[i], 0), pv_classes),
                            _one_hot(lab_w25.get(w25_cols[i], 0), wind_classes),
                        ],
                        axis=0,
                    )
                else:
                    c_vec = np.array(
                        [lab_pv33.get(pv33_cols[i], 0), lab_w25.get(w25_cols[i], 0)],
                        dtype=np.float32,
                    )
                if date_feat is not None:
                    c_vec = np.concatenate([c_vec, date_feat], axis=0)
                if add_curve_cond:
                    curve_vec, curve_names = _curve_control_features(x_day, ch_names)
                    c_vec = np.concatenate([c_vec, curve_vec], axis=0)
                    if curve_feature_names is None:
                        curve_feature_names = curve_names
                c_day = c_vec if static_cond else np.repeat(c_vec[None, :], seq_len, axis=0)
                m_day = np.ones_like(x_day, dtype=np.float32)
                m_day[:, 0] = (x_day[:, 0] > 0).astype(np.float32)
                m_day[:, 1] = (x_day[:, 1] > 0).astype(np.float32)
                x_list.append(x_day)
                c_list.append(c_day)
                m_list.append(m_day)
            elif mode == "2ch_single":
                # Single pair only: pv18 + wind22
                x_day = np.stack([pv18_vals[:, i], w22_vals[:, i]], axis=1)
                ch_names = ["pv", "wind"]
                if one_hot:
                    c_vec = np.concatenate(
                        [
                            _one_hot(lab_pv18.get(pv18_cols[i], 0), pv_classes),
                            _one_hot(lab_w22.get(w22_cols[i], 0), wind_classes),
                        ],
                        axis=0,
                    )
                else:
                    c_vec = np.array(
                        [lab_pv18.get(pv18_cols[i], 0), lab_w22.get(w22_cols[i], 0)],
                        dtype=np.float32,
                    )
                if date_feat is not None:
                    c_vec = np.concatenate([c_vec, date_feat], axis=0)
                if add_curve_cond:
                    curve_vec, curve_names = _curve_control_features(x_day, ch_names)
                    c_vec = np.concatenate([c_vec, curve_vec], axis=0)
                    if curve_feature_names is None:
                        curve_feature_names = curve_names
                c_day = c_vec if static_cond else np.repeat(c_vec[None, :], seq_len, axis=0)
                m_day = np.ones_like(x_day, dtype=np.float32)
                m_day[:, 0] = (x_day[:, 0] > 0).astype(np.float32)
                m_day[:, 1] = (x_day[:, 1] > 0).astype(np.float32)
                x_list.append(x_day)
                c_list.append(c_day)
                m_list.append(m_day)
            else:
                x_day = np.stack(
                    [
                        pv18_vals[:, i],
                        pv33_vals[:, i],
                        w22_vals[:, i],
                        w25_vals[:, i],
                    ],
                    axis=1,
                )
                ch_names = ["pv18", "pv33", "wind22", "wind25"]
                if one_hot:
                    c_vec = np.concatenate(
                        [
                            _one_hot(lab_pv18.get(pv18_cols[i], 0), pv_classes),
                            _one_hot(lab_pv33.get(pv33_cols[i], 0), pv_classes),
                            _one_hot(lab_w22.get(w22_cols[i], 0), wind_classes),
                            _one_hot(lab_w25.get(w25_cols[i], 0), wind_classes),
                        ],
                        axis=0,
                    )
                else:
                    c_vec = np.array(
                        [
                            lab_pv18.get(pv18_cols[i], 0),
                            lab_pv33.get(pv33_cols[i], 0),
                            lab_w22.get(w22_cols[i], 0),
                            lab_w25.get(w25_cols[i], 0),
                        ],
                        dtype=np.float32,
                    )
                if date_feat is not None:
                    c_vec = np.concatenate([c_vec, date_feat], axis=0)
                if add_curve_cond:
                    curve_vec, curve_names = _curve_control_features(x_day, ch_names)
                    c_vec = np.concatenate([c_vec, curve_vec], axis=0)
                    if curve_feature_names is None:
                        curve_feature_names = curve_names
                c_day = c_vec if static_cond else np.repeat(c_vec[None, :], seq_len, axis=0)
                m_day = np.ones_like(x_day, dtype=np.float32)
                m_day[:, 0] = (x_day[:, 0] > 0).astype(np.float32)
                m_day[:, 1] = (x_day[:, 1] > 0).astype(np.float32)
                x_list.append(x_day)
                c_list.append(c_day)
                m_list.append(m_day)

    if not x_list:
        raise ValueError("No full daily sequences found. Check seq_len/resample_rule.")

    data = np.stack(x_list, axis=0)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    cond = np.stack(c_list, axis=0)
    mask = np.stack(m_list, axis=0)

    x_min = data.min(axis=(0, 1))
    x_max = data.max(axis=(0, 1))
    if normalize:
        denom = np.maximum(x_max - x_min, 1e-6)
        data = (data - x_min[None, None, :]) / denom[None, None, :]

    curve_min = None
    curve_max = None
    if add_curve_cond and curve_feature_names:
        curve_dim = len(curve_feature_names)
        if static_cond:
            curve_raw = cond[:, -curve_dim:].astype(np.float32, copy=False)
        else:
            curve_raw = cond[:, 0, -curve_dim:].astype(np.float32, copy=False)
        if curve_cond_norm == "minmax":
            curve_min = curve_raw.min(axis=0)
            curve_max = curve_raw.max(axis=0)
            curve_denom = np.maximum(curve_max - curve_min, 1e-6)
            curve_norm = (curve_raw - curve_min[None, :]) / curve_denom[None, :]
            if static_cond:
                cond[:, -curve_dim:] = curve_norm
            else:
                cond[:, :, -curve_dim:] = np.repeat(curve_norm[:, None, :], cond.shape[1], axis=1)
        elif curve_cond_norm != "none":
            raise ValueError(f"Unsupported curve_cond_norm: {curve_cond_norm}")

    cond_stats = {
        "cond_norm": "onehot" if one_hot else "none",
        "pv_classes": pv_classes,
        "wind_classes": wind_classes,
        "ddre33_mode": mode,
        "x_norm": "minmax" if normalize else "none",
        "date_cond": "doy_sin_cos" if add_date_cond else "none",
        "static_cond": bool(static_cond),
        "curve_cond": bool(add_curve_cond),
        "curve_cond_norm": curve_cond_norm,
    }
    if add_curve_cond and curve_feature_names:
        cond_stats["curve_feature_names"] = list(curve_feature_names)
        if curve_min is not None and curve_max is not None:
            cond_stats["curve_cond_min"] = curve_min.astype(np.float32)
            cond_stats["curve_cond_max"] = curve_max.astype(np.float32)
    return data, cond, mask, x_min.astype(np.float32), x_max.astype(np.float32), cond_stats
