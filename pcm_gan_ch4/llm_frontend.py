from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import request

import numpy as np

from .core import DailyPlan, DDRE33ReferenceBank
from .knowledge import MacroPlan, PromptSpec, parse_prompt, resolve_macro_plan


MANUAL_PROMPT_TEMPLATE = """You are a meteorological extreme-event planning agent for a power-system scenario generator.

Task:
Given the user request, produce a STRICT JSON object with three sections:
1. prompt_spec
2. macro_plan
3. daily_plan

Hard constraints:
- Output JSON only. Do not output Markdown. Do not wrap in code fences.
- days must equal {days}.
- start_date must be "{start_date}".
- event_type must be one of:
  ["cold_snap","heatwave","typhoon","blizzard","rainstorm","calm_cloudy","dry_windy","neutral_extreme"]
- season must be one of:
  ["winter","spring","summer","autumn"]
- pv_label must be integer in [0,5].
- wind_label must be integer in [0,3].
- daily_plan must contain exactly {days} items.
- curve_features must contain exactly 8 numbers in this fixed order:
  [pv_mean, pv_peak, pv_ramp_std, pv_active_ratio, wind_mean, wind_peak, wind_ramp_std, wind_active_ratio]
- All mean/peak/active_ratio values must be in [0,1].
- ramp_std must be non-negative.
- For every day:
  pv_peak >= pv_mean
  wind_peak >= wind_mean
- date must increase day by day from the given start_date.
- doy must be the correct day-of-year.
- prototype_index can be set to -1.
- source_pair should be formatted as "pv{{pv_label}}_wind{{wind_label}}".
- trend should represent a multi-day evolution and normally follow buildup -> peak -> decay rather than iid noise.

User request:
{prompt}

JSON schema reminder:
{{
  "prompt_spec": {{
    "prompt": "...",
    "days": {days},
    "start_date": "{start_date}",
    "event_type": "...",
    "severity": 0.0,
    "season": "...",
    "keywords": ["..."],
    "notes": ["..."]
  }},
  "macro_plan": {{
    "prompt": "...",
    "event_type": "...",
    "severity": 0.0,
    "season": "...",
    "pv_label": 0,
    "wind_label": 0,
    "pv_scores": [6 numbers],
    "wind_scores": [4 numbers],
    "feasible_events": ["..."],
    "notes": ["..."]
  }},
  "daily_plan": [
    {{
      "day_index": 1,
      "date": "{start_date}",
      "doy": 1,
      "pv_label": 0,
      "wind_label": 0,
      "trend": 0.0,
      "prototype_index": -1,
      "source_pair": "pv0_wind0",
      "curve_features": [0,0,0,0,0,0,0,0],
      "notes": []
    }}
  ]
}}
"""


def build_manual_prompt(prompt: str, days: int, start_date: str) -> str:
    return MANUAL_PROMPT_TEMPLATE.format(prompt=prompt, days=int(days), start_date=start_date)


def save_manual_prompt(path: str, prompt: str, days: int, start_date: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_manual_prompt(prompt=prompt, days=days, start_date=start_date), encoding="utf-8")
    return str(out)


def _extract_json_text(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3:
            t = "\n".join(lines[1:-1]).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON object found in LLM output.")
    return t[start : end + 1]


def _normalize_scores(values: Optional[Sequence[float]], size: int, label: int) -> List[float]:
    if values is None:
        out = [0.0] * size
        out[max(0, min(size - 1, int(label)))] = 1.0
        return out
    arr = [max(float(v), 0.0) for v in values][:size]
    if len(arr) < size:
        arr += [0.0] * (size - len(arr))
    s = sum(arr)
    if s <= 0:
        arr = [0.0] * size
        arr[max(0, min(size - 1, int(label)))] = 1.0
        return arr
    return [float(v / s) for v in arr]


def _coerce_spec(obj: Dict[str, Any], fallback_prompt: str, fallback_days: int, fallback_start_date: str) -> PromptSpec:
    parsed = parse_prompt(fallback_prompt, days=fallback_days, start_date=fallback_start_date)
    start_raw = obj.get("start_date", fallback_start_date)
    start_dt = datetime.strptime(str(start_raw), "%Y-%m-%d").date()
    return PromptSpec(
        prompt=str(obj.get("prompt", fallback_prompt)),
        days=int(obj.get("days", fallback_days)),
        start_date=start_dt,
        event_type=str(obj.get("event_type", parsed.event_type)),
        severity=float(obj.get("severity", parsed.severity)),
        season=str(obj.get("season", parsed.season)),
        keywords=[str(v) for v in obj.get("keywords", parsed.keywords)],
        notes=[str(v) for v in obj.get("notes", parsed.notes)],
    )


def _coerce_macro(obj: Dict[str, Any], spec: PromptSpec) -> MacroPlan:
    base = resolve_macro_plan(spec)
    pv_label = int(obj.get("pv_label", base.pv_label))
    wind_label = int(obj.get("wind_label", base.wind_label))
    return MacroPlan(
        prompt=str(obj.get("prompt", spec.prompt)),
        event_type=str(obj.get("event_type", spec.event_type)),
        severity=float(obj.get("severity", spec.severity)),
        season=str(obj.get("season", spec.season)),
        pv_label=max(0, min(5, pv_label)),
        wind_label=max(0, min(3, wind_label)),
        pv_scores=_normalize_scores(obj.get("pv_scores"), 6, pv_label),
        wind_scores=_normalize_scores(obj.get("wind_scores"), 4, wind_label),
        feasible_events=[str(v) for v in obj.get("feasible_events", base.feasible_events)],
        notes=[str(v) for v in obj.get("notes", base.notes)],
    )


def _coerce_curve_features(values: Sequence[Any]) -> List[float]:
    arr = [float(v) for v in values][:8]
    if len(arr) != 8:
        raise ValueError("curve_features must contain exactly 8 values.")
    arr[0] = min(max(arr[0], 0.0), 1.0)
    arr[1] = min(max(arr[1], 0.0), 1.0)
    arr[2] = max(arr[2], 0.0)
    arr[3] = min(max(arr[3], 0.0), 1.0)
    arr[4] = min(max(arr[4], 0.0), 1.0)
    arr[5] = min(max(arr[5], 0.0), 1.0)
    arr[6] = max(arr[6], 0.0)
    arr[7] = min(max(arr[7], 0.0), 1.0)
    arr[1] = max(arr[1], arr[0])
    arr[5] = max(arr[5], arr[4])
    return [float(v) for v in arr]


def normalize_plan_payload(
    payload: Dict[str, Any],
    bank: DDRE33ReferenceBank,
    fallback_prompt: str,
    fallback_days: int,
    fallback_start_date: str,
) -> Tuple[PromptSpec, MacroPlan, List[DailyPlan]]:
    spec = _coerce_spec(payload.get("prompt_spec", {}), fallback_prompt, fallback_days, fallback_start_date)
    macro = _coerce_macro(payload.get("macro_plan", {}), spec)
    daily_items = payload.get("daily_plan", [])
    if len(daily_items) != int(spec.days):
        raise ValueError(f"daily_plan length {len(daily_items)} must equal days {spec.days}")

    plans: List[DailyPlan] = []
    for idx, item in enumerate(daily_items):
        day_idx = int(item.get("day_index", idx + 1))
        cur_date = item.get("date")
        if not cur_date:
            cur_date = (spec.start_date + timedelta(days=idx)).isoformat()
        cur_dt = datetime.strptime(str(cur_date), "%Y-%m-%d").date()
        doy = int(item.get("doy", cur_dt.timetuple().tm_yday))
        pv_label = max(0, min(5, int(item.get("pv_label", macro.pv_label))))
        wind_label = max(0, min(3, int(item.get("wind_label", macro.wind_label))))
        prototype_index = int(item.get("prototype_index", -1))
        if prototype_index < 0:
            prototype_index = bank.select_prototype(pv_label, wind_label, doy, season=macro.season)
        source_pair = str(item.get("source_pair", f"pv{pv_label}_wind{wind_label}"))
        curve_features = _coerce_curve_features(item.get("curve_features", []))
        plans.append(
            DailyPlan(
                day_index=day_idx,
                date=cur_dt.isoformat(),
                doy=doy,
                pv_label=pv_label,
                wind_label=wind_label,
                trend=float(item.get("trend", 0.5)),
                prototype_index=prototype_index,
                source_pair=source_pair,
                curve_features=curve_features,
                notes=[str(v) for v in item.get("notes", [])],
            )
        )
    return spec, macro, plans


def load_plan_json(
    path: str,
    bank: DDRE33ReferenceBank,
    fallback_prompt: str,
    fallback_days: int,
    fallback_start_date: str,
) -> Tuple[PromptSpec, MacroPlan, List[DailyPlan], Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    spec, macro, plans = normalize_plan_payload(
        payload=payload,
        bank=bank,
        fallback_prompt=fallback_prompt,
        fallback_days=fallback_days,
        fallback_start_date=fallback_start_date,
    )
    return spec, macro, plans, payload


def call_openai_compatible_api(
    prompt: str,
    days: int,
    start_date: str,
    api_key: str,
    model: str,
    base_url: str,
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    body = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You are a meteorological planning agent. Output valid JSON only.",
            },
            {
                "role": "user",
                "content": build_manual_prompt(prompt=prompt, days=days, start_date=start_date),
            },
        ],
    }
    data = json.dumps(body).encode("utf-8")
    req = request.Request(
        base_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=int(timeout_sec)) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    try:
        content = result["choices"][0]["message"]["content"]
    except Exception as exc:
        raise ValueError(f"Unexpected API response structure: {result}") from exc
    return json.loads(_extract_json_text(content))


def call_api_with_env(
    prompt: str,
    days: int,
    start_date: str,
    api_key_env: str,
    model: str,
    base_url: str,
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env} is empty.")
    return call_openai_compatible_api(
        prompt=prompt,
        days=days,
        start_date=start_date,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_sec=timeout_sec,
    )


def save_api_plan(payload: Dict[str, Any], out_path: str) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
