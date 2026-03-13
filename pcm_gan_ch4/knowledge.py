from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import math
import re
from typing import Dict, List, Optional, Tuple


@dataclass
class PromptSpec:
    prompt: str
    days: int
    start_date: date
    event_type: str
    severity: float
    season: str
    keywords: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class MacroPlan:
    prompt: str
    event_type: str
    severity: float
    season: str
    pv_label: int
    wind_label: int
    pv_scores: List[float]
    wind_scores: List[float]
    feasible_events: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def infer_season(start_date: date, prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ("winter", "cold season", "寒潮", "冬季")):
        return "winter"
    if any(k in p for k in ("summer", "heatwave", "高温", "夏季")):
        return "summer"
    if any(k in p for k in ("spring", "春季")):
        return "spring"
    if any(k in p for k in ("autumn", "fall", "秋季")):
        return "autumn"
    return _season_from_month(start_date.month)


DEFAULT_EVENT_ARCHETYPES: Dict[str, Dict[str, object]] = {
    "cold_snap": {
        "keywords": ["cold snap", "cold wave", "寒潮", "severe cold", "freeze", "icy"],
        "seasons": ["winter", "autumn", "spring"],
        "pv_center": 1.0,
        "wind_center": 2.4,
        "feature_deltas": [-0.20, -0.18, 0.06, -0.10, 0.08, 0.10, 0.18, 0.04],
        "uncertainty": 0.40,
    },
    "heatwave": {
        "keywords": ["heatwave", "extreme heat", "高温", "hot spell", "scorching"],
        "seasons": ["summer", "spring"],
        "pv_center": 4.8,
        "wind_center": 1.2,
        "feature_deltas": [0.12, 0.10, -0.02, 0.08, -0.05, -0.04, 0.04, -0.02],
        "uncertainty": 0.22,
    },
    "typhoon": {
        "keywords": ["typhoon", "hurricane", "tropical cyclone", "台风", "storm landfall"],
        "seasons": ["summer", "autumn"],
        "pv_center": 0.6,
        "wind_center": 3.0,
        "feature_deltas": [-0.35, -0.32, 0.08, -0.16, 0.16, 0.18, 0.22, 0.08],
        "uncertainty": 0.55,
    },
    "blizzard": {
        "keywords": ["blizzard", "snowstorm", "暴雪", "snow disaster"],
        "seasons": ["winter"],
        "pv_center": 0.3,
        "wind_center": 2.3,
        "feature_deltas": [-0.28, -0.24, 0.10, -0.18, 0.10, 0.12, 0.20, 0.06],
        "uncertainty": 0.46,
    },
    "rainstorm": {
        "keywords": ["rainstorm", "heavy rain", "暴雨", "stormy rain", "squall line"],
        "seasons": ["summer", "autumn", "spring"],
        "pv_center": 0.8,
        "wind_center": 2.2,
        "feature_deltas": [-0.26, -0.22, 0.06, -0.16, 0.08, 0.10, 0.16, 0.05],
        "uncertainty": 0.36,
    },
    "calm_cloudy": {
        "keywords": ["cloudy", "overcast", "多云", "阴天", "calm wind", "wind lull"],
        "seasons": ["spring", "summer", "autumn", "winter"],
        "pv_center": 1.4,
        "wind_center": 0.6,
        "feature_deltas": [-0.14, -0.10, -0.02, -0.08, -0.10, -0.08, -0.04, -0.06],
        "uncertainty": 0.16,
    },
    "dry_windy": {
        "keywords": ["dry windy", "windy", "大风", "gale", "strong wind"],
        "seasons": ["autumn", "winter", "spring"],
        "pv_center": 3.2,
        "wind_center": 2.9,
        "feature_deltas": [0.04, 0.04, 0.02, 0.02, 0.14, 0.16, 0.20, 0.06],
        "uncertainty": 0.30,
    },
    "neutral_extreme": {
        "keywords": [],
        "seasons": ["spring", "summer", "autumn", "winter"],
        "pv_center": 2.5,
        "wind_center": 1.8,
        "feature_deltas": [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.04, 0.0],
        "uncertainty": 0.18,
    },
}


SEVERITY_KEYWORDS = {
    "mild": 0.25,
    "moderate": 0.45,
    "severe": 0.70,
    "extreme": 0.90,
    "catastrophic": 1.00,
    "轻度": 0.25,
    "中等": 0.45,
    "严重": 0.70,
    "极端": 0.90,
}


def parse_start_date(value: Optional[str]) -> date:
    if not value:
        return datetime.now().date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_prompt(prompt: str, days: Optional[int], start_date: Optional[str]) -> PromptSpec:
    d0 = parse_start_date(start_date)
    p = prompt.lower()
    scores: List[Tuple[str, float]] = []
    keywords: List[str] = []
    for event_type, cfg in DEFAULT_EVENT_ARCHETYPES.items():
        score = 0.0
        for kw in cfg["keywords"]:
            if kw and kw.lower() in p:
                score += 1.0
                keywords.append(kw)
        scores.append((event_type, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    event_type = scores[0][0] if scores and scores[0][1] > 0 else "neutral_extreme"

    inferred_days = days
    if inferred_days is None:
        m = re.search(r"(\d+)\s*[- ]?\s*day", p)
        inferred_days = int(m.group(1)) if m else 7
    inferred_days = max(1, int(inferred_days))

    severity = 0.60
    for kw, val in SEVERITY_KEYWORDS.items():
        if kw in p:
            severity = max(severity, val)

    season = infer_season(d0, prompt)
    notes = []
    top_events = [name for name, score in scores if score > 0]
    if len(top_events) > 1:
        notes.append(
            "Multiple event keywords detected; the planner will keep the highest-score archetype and "
            "record the alternatives for conflict resolution."
        )
    return PromptSpec(
        prompt=prompt,
        days=inferred_days,
        start_date=d0,
        event_type=event_type,
        severity=float(min(max(severity, 0.0), 1.0)),
        season=season,
        keywords=sorted(set(keywords)),
        notes=notes,
    )


def _gaussian_scores(center: float, n_classes: int, sigma: float) -> List[float]:
    vals = []
    for idx in range(n_classes):
        vals.append(math.exp(-0.5 * ((idx - center) / max(sigma, 1e-6)) ** 2))
    s = sum(vals)
    return [v / s for v in vals]


def resolve_macro_plan(spec: PromptSpec) -> MacroPlan:
    cfg = DEFAULT_EVENT_ARCHETYPES[spec.event_type]
    feasible_events = [name for name, ecfg in DEFAULT_EVENT_ARCHETYPES.items() if spec.season in ecfg["seasons"]]

    notes = list(spec.notes)
    if spec.season not in cfg["seasons"]:
        notes.append(
            f"Prompt season '{spec.season}' is weakly inconsistent with archetype '{spec.event_type}'. "
            "The event is retained but marked as a relaxed knowledge resolution."
        )

    pv_center = float(cfg["pv_center"])
    wind_center = float(cfg["wind_center"])
    pv_sigma = 0.90 - 0.25 * spec.severity
    wind_sigma = 0.85 - 0.20 * spec.severity
    pv_scores = _gaussian_scores(pv_center, 6, max(0.35, pv_sigma))
    wind_scores = _gaussian_scores(wind_center, 4, max(0.30, wind_sigma))
    pv_label = int(max(range(6), key=lambda i: pv_scores[i]))
    wind_label = int(max(range(4), key=lambda i: wind_scores[i]))

    return MacroPlan(
        prompt=spec.prompt,
        event_type=spec.event_type,
        severity=spec.severity,
        season=spec.season,
        pv_label=pv_label,
        wind_label=wind_label,
        pv_scores=[float(v) for v in pv_scores],
        wind_scores=[float(v) for v in wind_scores],
        feasible_events=feasible_events,
        notes=notes,
    )
