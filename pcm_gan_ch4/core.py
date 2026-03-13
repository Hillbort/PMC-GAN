from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pcm_gan.data_synth import load_ddre33_dataset
from pcm_gan.models import Generator
from pcm_gan.utils import ScenarioConfig, get_device, get_generator_state_dict, set_seed

from .knowledge import DEFAULT_EVENT_ARCHETYPES, MacroPlan, PromptSpec, parse_prompt, resolve_macro_plan


def _curve_features(x_day: np.ndarray, active_eps: float = 0.05) -> np.ndarray:
    feats: List[float] = []
    for ch in range(x_day.shape[1]):
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
    return np.asarray(feats, dtype=np.float32)


def _doy_to_sin_cos(d: date) -> Tuple[float, float]:
    doy = d.timetuple().tm_yday
    angle = 2.0 * np.pi * ((float(doy) - 1.0) / 365.0)
    return float(np.sin(angle)), float(np.cos(angle))


def _sin_cos_to_doy(sin_v: float, cos_v: float) -> int:
    angle = np.arctan2(sin_v, cos_v)
    if angle < 0:
        angle += 2.0 * np.pi
    doy = int(round(angle / (2.0 * np.pi) * 365.0)) + 1
    return max(1, min(365, doy))


def _season_from_doy(doy: int) -> str:
    dt = datetime.strptime(f"2025-{doy:03d}", "%Y-%j").date()
    month = dt.month
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def _smoothstep(x: float) -> float:
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


@dataclass
class DailyPlan:
    day_index: int
    date: str
    doy: int
    pv_label: int
    wind_label: int
    trend: float
    prototype_index: int
    source_pair: str
    curve_features: List[float]
    notes: List[str] = field(default_factory=list)


@dataclass
class OrchestrationOutput:
    prompt_spec: PromptSpec
    macro_plan: MacroPlan
    daily_plan: List[DailyPlan]
    multi_day_csv: Optional[str]
    source_mode: str
    summary_path: str


class DDRE33ReferenceBank:
    def __init__(
        self,
        pv18_csv: str,
        pv33_csv: str,
        wind22_csv: str,
        wind25_csv: str,
        pv18_labels_csv: str,
        pv33_labels_csv: str,
        wind22_labels_csv: str,
        wind25_labels_csv: str,
        max_cols: int = 0,
    ):
        data, cond, mask, x_min, x_max, _ = load_ddre33_dataset(
            pv18_csv=pv18_csv,
            pv33_csv=pv33_csv,
            wind22_csv=wind22_csv,
            wind25_csv=wind25_csv,
            pv18_labels_csv=pv18_labels_csv,
            pv33_labels_csv=pv33_labels_csv,
            wind22_labels_csv=wind22_labels_csv,
            wind25_labels_csv=wind25_labels_csv,
            seq_len=96,
            resample_rule="15min",
            one_hot=True,
            mode="2ch_single",
            max_cols=max_cols,
            normalize=False,
            add_date_cond=True,
            static_cond=True,
            add_curve_cond=False,
        )
        self.data = data.astype(np.float32, copy=False)
        self.cond = cond.astype(np.float32, copy=False)
        self.mask = mask.astype(np.float32, copy=False)
        self.x_min = np.asarray(x_min, dtype=np.float32)
        self.x_max = np.asarray(x_max, dtype=np.float32)
        self.pv_labels = np.argmax(self.cond[:, :6], axis=1).astype(np.int64)
        self.wind_labels = np.argmax(self.cond[:, 6:10], axis=1).astype(np.int64)
        self.doys = np.array([_sin_cos_to_doy(v[10], v[11]) for v in self.cond], dtype=np.int64)
        self.seasons = np.array([_season_from_doy(int(doy)) for doy in self.doys])
        self.features = np.stack([_curve_features(x) for x in self.data], axis=0).astype(np.float32)
        self._pair_to_indices: Dict[Tuple[int, int], np.ndarray] = {}
        for pv, wind in {(int(pv), int(wind)) for pv, wind in zip(self.pv_labels.tolist(), self.wind_labels.tolist())}:
            idx = np.where((self.pv_labels == pv) & (self.wind_labels == wind))[0]
            self._pair_to_indices[(pv, wind)] = idx

    def nearest_pair(self, pv_label: int, wind_label: int) -> Tuple[int, int]:
        key = (int(pv_label), int(wind_label))
        if key in self._pair_to_indices and self._pair_to_indices[key].size > 0:
            return key
        pairs = list(self._pair_to_indices)
        return min(pairs, key=lambda item: abs(item[0] - pv_label) + abs(item[1] - wind_label))

    def pair_indices(self, pv_label: int, wind_label: int, season: Optional[str] = None) -> np.ndarray:
        pv_label, wind_label = self.nearest_pair(pv_label, wind_label)
        idx = self._pair_to_indices[(pv_label, wind_label)]
        if season:
            mask = self.seasons[idx] == season
            if np.any(mask):
                return idx[mask]
        return idx

    def pair_stats(self, pv_label: int, wind_label: int, season: Optional[str] = None) -> Dict[str, np.ndarray]:
        idx = self.pair_indices(pv_label, wind_label, season=season)
        feat = self.features[idx]
        q25 = np.quantile(feat, 0.25, axis=0)
        q50 = np.quantile(feat, 0.50, axis=0)
        q75 = np.quantile(feat, 0.75, axis=0)
        return {"q25": q25.astype(np.float32), "q50": q50.astype(np.float32), "q75": q75.astype(np.float32)}

    def select_prototype(self, pv_label: int, wind_label: int, doy: int, season: Optional[str] = None) -> int:
        idx = self.pair_indices(pv_label, wind_label, season=season)
        return int(idx[np.argmin(np.abs(self.doys[idx] - int(doy)))])

    def ramp_limits(self, quantile: float = 0.995) -> np.ndarray:
        diffs = np.abs(np.diff(self.data, axis=1))
        return np.quantile(diffs, float(quantile), axis=(0, 1)).astype(np.float32)


class PCMGeneratorBridge:
    def __init__(self, ckpt_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.available = False
        self.use_curve_cond = True
        self.use_date_cond = True
        self.device = device or get_device(verbose=False)
        self.ckpt_path = ckpt_path
        if not ckpt_path:
            return
        path = Path(ckpt_path)
        if not path.exists():
            return
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt["cfg"]
        self.cfg = cfg
        self.scfg = ScenarioConfig(seq_len=cfg["seq_len"], channels=cfg["channels"], cond_dim=cfg["cond_dim"])
        self.use_curve_cond = bool(cfg.get("curve_cond", self.scfg.cond_dim >= 20))
        self.use_date_cond = cfg.get("date_cond", "none") != "none"
        self.z_dim = int(cfg["z_dim"])
        self.generator = Generator(
            seq_len=self.scfg.seq_len,
            cond_dim=self.scfg.cond_dim,
            z_dim=self.z_dim,
            model_dim=128,
            depth=4,
            heads=4,
            channels=self.scfg.channels,
            use_baseline_residual=bool(cfg.get("use_baseline_residual", False)),
        ).to(self.device)
        state_dict, _ = get_generator_state_dict(
            ckpt, prefer_ema=cfg.get("generator_state_preference", "G_ema") == "G_ema"
        )
        self.generator.load_state_dict(state_dict, strict=False)
        self.generator.eval()
        self.available = True

    def generate_day(
        self,
        cond_vec: np.ndarray,
        mask: np.ndarray,
        target_features: Optional[np.ndarray] = None,
        topk_candidates: int = 1,
        seed: int = 42,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if not self.available:
            raise RuntimeError("PCMGeneratorBridge is unavailable because checkpoint is missing.")
        rng = np.random.default_rng(seed)
        cond_t = torch.from_numpy(cond_vec[None, :].astype(np.float32)).to(self.device)
        mask_t = torch.from_numpy(mask[None, ...].astype(np.float32)).to(self.device)
        k = max(1, int(topk_candidates))
        if k == 1 or target_features is None:
            z = torch.randn(1, self.z_dim, device=self.device)
            with torch.no_grad():
                out = self.generator(z, cond_t, mask_t).cpu().numpy()[0]
            return out, {"candidate_score": float("nan"), "selected_idx": 0}

        z = torch.from_numpy(rng.standard_normal((k, self.z_dim)).astype(np.float32)).to(self.device)
        cond_batch = cond_t.repeat(k, 1)
        mask_batch = mask_t.repeat(k, 1, 1)
        with torch.no_grad():
            out = self.generator(z, cond_batch, mask_batch).cpu().numpy()
        feats = np.stack([_curve_features(x) for x in out], axis=0)
        weights = np.array([1.2, 1.2, 0.8, 0.8, 1.2, 1.2, 0.8, 0.8], dtype=np.float32)
        scores = np.mean(np.abs(feats - target_features[None, :]) * weights[None, :], axis=1)
        best = int(np.argmin(scores))
        return out[best], {"candidate_score": float(scores[best]), "selected_idx": best}


class AutonomousOrchestrationAgent:
    def __init__(self, reference_bank: DDRE33ReferenceBank, ckpt_path: Optional[str] = None):
        self.bank = reference_bank
        self.bridge = PCMGeneratorBridge(ckpt_path=ckpt_path)

    def _build_trend(self, days: int, severity: float) -> np.ndarray:
        if days <= 1:
            return np.array([float(max(0.35, severity))], dtype=np.float32)
        t = np.linspace(0.0, 1.0, days, dtype=np.float32)
        floor = 0.45 + 0.15 * float(severity)
        wave = np.clip(np.sin(np.pi * t), 0.0, None)
        trend = floor + (1.0 - floor) * wave ** 1.25
        return trend.astype(np.float32)

    def _target_curve(self, macro: MacroPlan, trend_val: float, base_stats: Dict[str, np.ndarray]) -> np.ndarray:
        cfg = DEFAULT_EVENT_ARCHETYPES[macro.event_type]
        deltas = np.asarray(cfg["feature_deltas"], dtype=np.float32)
        base = base_stats["q50"].astype(np.float32)
        spread = np.maximum(base_stats["q75"] - base_stats["q25"], 1e-3).astype(np.float32)
        target = base + float(macro.severity) * float(trend_val) * deltas + 0.10 * spread * np.sign(deltas)
        target = target.astype(np.float32)
        target[[0, 1, 3, 4, 5, 7]] = np.clip(target[[0, 1, 3, 4, 5, 7]], 0.0, 1.0)
        target[[2, 6]] = np.clip(target[[2, 6]], 0.0, 1.5)
        target[1] = max(target[1], target[0])
        target[5] = max(target[5], target[4])
        return target

    def plan_daily_conditions(self, spec: PromptSpec, macro: MacroPlan, seed: int = 42) -> List[DailyPlan]:
        rng = np.random.default_rng(seed)
        trend = self._build_trend(spec.days, spec.severity)
        stats = self.bank.pair_stats(macro.pv_label, macro.wind_label, season=macro.season)
        spread = np.maximum(stats["q75"] - stats["q25"], 1e-3).astype(np.float32)
        rho = 0.70
        uncertainty = float(DEFAULT_EVENT_ARCHETYPES[macro.event_type]["uncertainty"])
        prev = stats["q50"].astype(np.float32)
        plans: List[DailyPlan] = []
        for d in range(spec.days):
            cur_date = spec.start_date + timedelta(days=d)
            target = self._target_curve(macro, float(trend[d]), stats)
            eps = rng.normal(0.0, uncertainty * spread * (0.25 + 0.75 * float(trend[d])), size=target.shape)
            state = rho * prev + (1.0 - rho) * target + eps.astype(np.float32)
            state[[0, 1, 3, 4, 5, 7]] = np.clip(state[[0, 1, 3, 4, 5, 7]], 0.0, 1.0)
            state[[2, 6]] = np.clip(state[[2, 6]], 0.0, 1.5)
            state[1] = max(state[1], state[0])
            state[5] = max(state[5], state[4])
            doy = cur_date.timetuple().tm_yday
            prototype_idx = self.bank.select_prototype(macro.pv_label, macro.wind_label, doy, season=macro.season)
            plans.append(
                DailyPlan(
                    day_index=d + 1,
                    date=cur_date.isoformat(),
                    doy=doy,
                    pv_label=macro.pv_label,
                    wind_label=macro.wind_label,
                    trend=float(trend[d]),
                    prototype_index=prototype_idx,
                    source_pair=f"pv{macro.pv_label}_wind{macro.wind_label}",
                    curve_features=[float(v) for v in state],
                )
            )
            prev = state.astype(np.float32)
        return plans

    def _cond_vector(self, plan: DailyPlan, use_curve_cond: bool = True, use_date_cond: bool = True) -> np.ndarray:
        parts = [_one_hot(plan.pv_label, 6), _one_hot(plan.wind_label, 4)]
        if use_date_cond:
            cur_date = datetime.strptime(plan.date, "%Y-%m-%d").date()
            parts.append(np.asarray(_doy_to_sin_cos(cur_date), dtype=np.float32))
        if use_curve_cond:
            parts.append(np.asarray(plan.curve_features, dtype=np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _project_boundary(
        self,
        x0: np.ndarray,
        prev_end: np.ndarray,
        ramp_max: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ) -> np.ndarray:
        low = np.maximum(x_min, prev_end - ramp_max)
        high = np.minimum(x_max, prev_end + ramp_max)
        return np.minimum(np.maximum(x0, low), high)

    def _smooth_next_day(self, raw: np.ndarray, prev_end: np.ndarray, ramp_max: np.ndarray, window: int) -> np.ndarray:
        out = raw.copy()
        out[0] = self._project_boundary(out[0], prev_end, ramp_max, self.bank.x_min, self.bank.x_max)
        win = min(max(2, int(window)), out.shape[0])
        delta = out[0] - raw[0]
        for i in range(1, win):
            alpha = 1.0 - _smoothstep(i / max(win - 1, 1))
            out[i] = np.clip(raw[i] + alpha * delta, self.bank.x_min, self.bank.x_max)
            out[i] = np.clip(out[i], out[i - 1] - ramp_max, out[i - 1] + ramp_max)
        return out

    def stitch_days(self, daily_series: Sequence[np.ndarray], ramp_max: np.ndarray, window: int = 8) -> np.ndarray:
        stitched: List[np.ndarray] = []
        prev_end: Optional[np.ndarray] = None
        for day in daily_series:
            cur = np.asarray(day, dtype=np.float32).copy()
            if prev_end is not None:
                cur = self._smooth_next_day(cur, prev_end, ramp_max, window)
            stitched.append(cur)
            prev_end = cur[-1].copy()
        return np.concatenate(stitched, axis=0).astype(np.float32)

    def run(
        self,
        prompt: str,
        days: Optional[int],
        start_date: Optional[str],
        outdir: str,
        seed: int = 42,
        topk_candidates: int = 8,
        boundary_window: int = 8,
        ramp_quantile: float = 0.995,
    ) -> OrchestrationOutput:
        set_seed(seed)
        spec = parse_prompt(prompt, days=days, start_date=start_date)
        macro = resolve_macro_plan(spec)
        daily_plan = self.plan_daily_conditions(spec, macro, seed=seed)
        return self.run_from_plan(
            spec=spec,
            macro=macro,
            daily_plan=daily_plan,
            outdir=outdir,
            seed=seed,
            topk_candidates=topk_candidates,
            boundary_window=boundary_window,
            ramp_quantile=ramp_quantile,
        )

    def run_from_plan(
        self,
        spec: PromptSpec,
        macro: MacroPlan,
        daily_plan: Sequence[DailyPlan],
        outdir: str,
        seed: int = 42,
        topk_candidates: int = 8,
        boundary_window: int = 8,
        ramp_quantile: float = 0.995,
    ) -> OrchestrationOutput:
        set_seed(seed)
        outdir_p = Path(outdir)
        outdir_p.mkdir(parents=True, exist_ok=True)
        plots_dir = outdir_p / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        daily_series: List[np.ndarray] = []
        source_mode = "reference_fallback"
        generation_rows: List[Dict[str, object]] = []
        for i, plan in enumerate(daily_plan):
            mask = self.bank.mask[plan.prototype_index]
            cond_vec = self._cond_vector(
                plan,
                use_curve_cond=self.bridge.use_curve_cond if self.bridge.available else True,
                use_date_cond=self.bridge.use_date_cond if self.bridge.available else True,
            )
            if self.bridge.available:
                x_day, meta = self.bridge.generate_day(
                    cond_vec,
                    mask,
                    target_features=np.asarray(plan.curve_features, dtype=np.float32),
                    topk_candidates=topk_candidates,
                    seed=seed + i,
                )
                source_mode = "pcm_gan"
            else:
                x_day = self.bank.data[plan.prototype_index].copy()
                meta = {"candidate_score": float("nan"), "selected_idx": -1}
            daily_series.append(x_day.astype(np.float32))
            generation_rows.append(
                {
                    "day": plan.day_index,
                    "date": plan.date,
                    "prototype_index": plan.prototype_index,
                    "candidate_score": meta["candidate_score"],
                    "selected_idx": meta["selected_idx"],
                    "source_mode": source_mode if self.bridge.available else "reference_fallback",
                }
            )

        ramp_max = self.bank.ramp_limits(quantile=ramp_quantile)
        stitched = self.stitch_days(daily_series, ramp_max=ramp_max, window=boundary_window)

        macro_path = outdir_p / "macro_plan.json"
        spec_path = outdir_p / "prompt_spec.json"
        daily_csv = outdir_p / "daily_plan.csv"
        gen_csv = outdir_p / "generation_trace.csv"
        profile_csv = outdir_p / "multi_day_profile.csv"
        summary_path = outdir_p / "orchestration_summary.json"

        macro_path.write_text(
            json.dumps(asdict(macro), ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        spec_path.write_text(
            json.dumps(asdict(spec), ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

        with daily_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "day",
                    "date",
                    "doy",
                    "pv_label",
                    "wind_label",
                    "trend",
                    "prototype_index",
                    "pv_mean",
                    "pv_peak",
                    "pv_ramp_std",
                    "pv_active_ratio",
                    "wind_mean",
                    "wind_peak",
                    "wind_ramp_std",
                    "wind_active_ratio",
                ]
            )
            for plan in daily_plan:
                writer.writerow(
                    [
                        plan.day_index,
                        plan.date,
                        plan.doy,
                        plan.pv_label,
                        plan.wind_label,
                        plan.trend,
                        plan.prototype_index,
                        *plan.curve_features,
                    ]
                )

        with gen_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(generation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(generation_rows)

        start_dt = datetime.strptime(daily_plan[0].date, "%Y-%m-%d")
        with profile_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "day", "step_in_day", "pv", "wind"])
            for idx, row in enumerate(stitched):
                ts = start_dt + timedelta(minutes=15 * idx)
                writer.writerow([ts.isoformat(sep=" "), idx // 96 + 1, idx % 96 + 1, float(row[0]), float(row[1])])

        self._plot_daily_plan(daily_plan, plots_dir / "micro_plan.png")
        self._plot_multiday_profile(stitched, plots_dir / "multi_day_profile.png")

        summary = {
            "source_mode": source_mode,
            "days": spec.days,
            "checkpoint_used": self.bridge.ckpt_path if self.bridge.available else None,
            "ramp_max": ramp_max.tolist(),
            "boundary_window": int(boundary_window),
            "macro_plan_path": str(macro_path),
            "prompt_spec_path": str(spec_path),
            "daily_plan_csv": str(daily_csv),
            "generation_trace_csv": str(gen_csv),
            "multi_day_profile_csv": str(profile_csv),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return OrchestrationOutput(
            prompt_spec=spec,
            macro_plan=macro,
            daily_plan=daily_plan,
            multi_day_csv=str(profile_csv),
            source_mode=source_mode,
            summary_path=str(summary_path),
        )

    def _plot_daily_plan(self, daily_plan: Sequence[DailyPlan], out_png: Path) -> None:
        days = [p.day_index for p in daily_plan]
        feats = np.asarray([p.curve_features for p in daily_plan], dtype=np.float32)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        titles = ["PV level", "PV variability", "Wind level", "Wind variability"]
        series = [
            [feats[:, 0], feats[:, 1], feats[:, 3]],
            [feats[:, 2]],
            [feats[:, 4], feats[:, 5], feats[:, 7]],
            [feats[:, 6]],
        ]
        names = [
            ["pv_mean", "pv_peak", "pv_active_ratio"],
            ["pv_ramp_std"],
            ["wind_mean", "wind_peak", "wind_active_ratio"],
            ["wind_ramp_std"],
        ]
        for ax, title, sers, nms in zip(axes.reshape(-1), titles, series, names):
            for y, nm in zip(sers, nms):
                ax.plot(days, y, marker="o", label=nm)
            ax.set_title(title)
            ax.set_xlabel("day")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

    def _plot_multiday_profile(self, stitched: np.ndarray, out_png: Path) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        t = np.arange(stitched.shape[0])
        axes[0].plot(t, stitched[:, 0], label="PV", color="#d97706")
        axes[1].plot(t, stitched[:, 1], label="Wind", color="#2563eb")
        for ax in axes:
            for boundary in range(96, stitched.shape[0], 96):
                ax.axvline(boundary, color="grey", linestyle="--", linewidth=0.8)
            ax.legend()
            ax.grid(alpha=0.25)
        axes[1].set_xlabel("15-min steps")
        axes[0].set_title("Chapter 4 Multi-day Extreme Scenario")
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
