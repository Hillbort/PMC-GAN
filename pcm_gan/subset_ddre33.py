import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


def parse_args():
    p = argparse.ArgumentParser(description="Create a small DDRE-33 subset covering all climate classes.")
    p.add_argument("--pv18_csv", type=str, required=True)
    p.add_argument("--pv33_csv", type=str, required=True)
    p.add_argument("--wind22_csv", type=str, required=True)
    p.add_argument("--wind25_csv", type=str, required=True)
    p.add_argument("--pv18_labels_csv", type=str, required=True)
    p.add_argument("--pv33_labels_csv", type=str, required=True)
    p.add_argument("--wind22_labels_csv", type=str, required=True)
    p.add_argument("--wind25_labels_csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="pcm_gan_data/ddre33_subset")
    p.add_argument("--per_class", type=int, default=1, help="Extra samples per class after coverage (>=1).")
    p.add_argument("--max_cols", type=int, default=0, help="Cap total selected columns (0 = no cap).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _read_label_map(path: Path) -> Dict[str, int]:
    df = pd.read_csv(path)
    if "Index" not in df.columns or "Type" not in df.columns:
        raise ValueError(f"Label file must have Index/Type columns: {path}")
    return {str(k): int(v) for k, v in zip(df["Index"].tolist(), df["Type"].tolist())}


def _load_columns(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path, nrows=0)
    cols = df.columns.tolist()
    if "timestamp" in cols:
        cols.remove("timestamp")
    return cols


def _idx_from_col(col: str) -> Optional[int]:
    try:
        return int(col.split("_")[-1])
    except Exception:
        return None


def _build_pairs(pv_cols: List[str], wind_cols: List[str]) -> List[Tuple[str, str]]:
    pv_map = { _idx_from_col(c): c for c in pv_cols if _idx_from_col(c) is not None }
    wind_map = { _idx_from_col(c): c for c in wind_cols if _idx_from_col(c) is not None }
    common_idx = sorted(set(pv_map) & set(wind_map))
    return [(pv_map[i], wind_map[i]) for i in common_idx]


def _greedy_cover(
    candidates: List[str],
    pv_labels: Dict[str, int],
    wind_labels: Dict[str, int],
    pv_classes: int,
    wind_classes: int,
    rng: np.random.Generator,
    per_class: int,
) -> List[str]:
    uncovered_pv = set(range(pv_classes))
    uncovered_wind = set(range(wind_classes))
    selected = []

    # greedy set cover
    while uncovered_pv or uncovered_wind:
        best = None
        best_gain = -1
        for col in candidates:
            pv = pv_labels.get(col, 0)
            wind = wind_labels.get(col, 0)
            gain = (1 if pv in uncovered_pv else 0) + (1 if wind in uncovered_wind else 0)
            if gain > best_gain:
                best_gain = gain
                best = col
        if best is None:
            break
        selected.append(best)
        pv = pv_labels.get(best, 0)
        wind = wind_labels.get(best, 0)
        uncovered_pv.discard(pv)
        uncovered_wind.discard(wind)
        candidates = [c for c in candidates if c != best]

        if best_gain <= 0:
            break

    # add extras per class (if available)
    def add_extras(label_map, n_class):
        extra = []
        for k in range(n_class):
            cols = [c for c in candidates if label_map.get(c, -1) == k]
            if not cols:
                continue
            rng.shuffle(cols)
            take = max(0, min(len(cols), max(0, per_class - 1)))
            extra.extend(cols[:take])
            for c in cols[:take]:
                candidates.remove(c)
        return extra

    per_class = 1 if per_class < 1 else per_class
    selected.extend(add_extras(pv_labels, pv_classes))
    selected.extend(add_extras(wind_labels, wind_classes))
    return selected


def _write_subset_csv(src: Path, dst: Path, cols: List[str], chunksize: int = 20000) -> None:
    first = True
    for chunk in pd.read_csv(src, chunksize=chunksize):
        if "timestamp" in chunk.columns:
            out = chunk[["timestamp"] + cols]
        else:
            out = chunk[cols]
        out.to_csv(dst, index=False, mode="w" if first else "a", header=first)
        first = False


def _write_subset_labels(src: Path, dst: Path, cols: List[str]) -> None:
    df = pd.read_csv(src)
    df = df[df["Index"].astype(str).isin(cols)]
    df.to_csv(dst, index=False)


def main():
    if pd is None:
        raise RuntimeError("pandas is required. Please `pip install pandas`.")
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    pv18_csv = Path(args.pv18_csv)
    pv33_csv = Path(args.pv33_csv)
    wind22_csv = Path(args.wind22_csv)
    wind25_csv = Path(args.wind25_csv)
    pv18_labels = Path(args.pv18_labels_csv)
    pv33_labels = Path(args.pv33_labels_csv)
    wind22_labels = Path(args.wind22_labels_csv)
    wind25_labels = Path(args.wind25_labels_csv)

    pv18_cols = _load_columns(pv18_csv)
    pv33_cols = _load_columns(pv33_csv)
    wind22_cols = _load_columns(wind22_csv)
    wind25_cols = _load_columns(wind25_csv)

    pairs_18_22 = _build_pairs(pv18_cols, wind22_cols)
    if not pairs_18_22:
        raise ValueError("No matching PV18/Wind22 scenario indices found.")

    pv18_map = _read_label_map(pv18_labels)
    wind22_map = _read_label_map(wind22_labels)

    # Use PV18 + Wind22 pairs to cover all climate classes
    pair_cols = [p for p, _ in pairs_18_22]
    selected_pv = _greedy_cover(pair_cols, pv18_map, wind22_map, 6, 4, rng, args.per_class)
    selected_idx = sorted({_idx_from_col(c) for c in selected_pv if _idx_from_col(c) is not None})

    if args.max_cols and args.max_cols > 0:
        selected_idx = selected_idx[: int(args.max_cols)]

    # Build final column lists by index
    pv18_sel = [f"PV_power_{i}" for i in selected_idx]
    pv33_sel = [f"PV_power_{i}" for i in selected_idx]
    wind22_sel = [f"wind_power_{i}" for i in selected_idx]
    wind25_sel = [f"wind_power_{i}" for i in selected_idx]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _write_subset_csv(pv18_csv, outdir / pv18_csv.name, pv18_sel)
    _write_subset_csv(pv33_csv, outdir / pv33_csv.name, pv33_sel)
    _write_subset_csv(wind22_csv, outdir / wind22_csv.name, wind22_sel)
    _write_subset_csv(wind25_csv, outdir / wind25_csv.name, wind25_sel)

    _write_subset_labels(pv18_labels, outdir / pv18_labels.name, pv18_sel)
    _write_subset_labels(pv33_labels, outdir / pv33_labels.name, pv33_sel)
    _write_subset_labels(wind22_labels, outdir / wind22_labels.name, wind22_sel)
    _write_subset_labels(wind25_labels, outdir / wind25_labels.name, wind25_sel)

    print(f"saved subset to: {outdir}")
    print(f"selected columns: {len(selected_idx)}")


if __name__ == "__main__":
    main()
