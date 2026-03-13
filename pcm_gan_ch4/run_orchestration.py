from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from .core import AutonomousOrchestrationAgent, DDRE33ReferenceBank
from .llm_frontend import (
    call_api_with_env,
    load_plan_json,
    save_api_plan,
    save_manual_prompt,
)


def parse_args():
    p = argparse.ArgumentParser(description="Chapter 4 multi-scale extreme scenario orchestration for PCM-GAN.")
    p.add_argument("--prompt", type=str, required=True, help="Natural language prompt for the extreme event.")
    p.add_argument("--days", type=int, default=7, help="Number of days in the prolonged event.")
    p.add_argument("--start_date", type=str, default="", help="Event start date in YYYY-MM-DD.")
    p.add_argument("--ckpt", type=str, default="", help="Optional trained PCM-GAN checkpoint.")
    p.add_argument("--outdir", type=str, default="pcm_gan_ch4_runs/demo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk_candidates", type=int, default=8)
    p.add_argument("--boundary_window", type=int, default=8)
    p.add_argument("--ramp_quantile", type=float, default=0.995)
    p.add_argument("--max_cols", type=int, default=0)
    p.add_argument(
        "--planner_mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "manual_json", "deepseek_api"],
        help="Frontend planner mode.",
    )
    p.add_argument("--manual_plan_json", type=str, default="", help="External JSON plan produced manually or by API.")
    p.add_argument(
        "--prompt_template_out",
        type=str,
        default="",
        help="Optional path to export the manual web-LLM prompt template.",
    )
    p.add_argument("--api_model", type=str, default="deepseek-chat")
    p.add_argument(
        "--api_base_url",
        type=str,
        default="https://api.deepseek.com/chat/completions",
        help="OpenAI-compatible chat completions endpoint.",
    )
    p.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")
    p.add_argument("--api_timeout_sec", type=int, default=120)
    p.add_argument(
        "--save_api_plan_json",
        type=str,
        default="",
        help="Optional path to save the JSON plan returned by API.",
    )

    p.add_argument("--pv18_csv", type=str, required=True)
    p.add_argument("--pv33_csv", type=str, required=True)
    p.add_argument("--wind22_csv", type=str, required=True)
    p.add_argument("--wind25_csv", type=str, required=True)
    p.add_argument("--pv18_labels_csv", type=str, required=True)
    p.add_argument("--pv33_labels_csv", type=str, required=True)
    p.add_argument("--wind22_labels_csv", type=str, required=True)
    p.add_argument("--wind25_labels_csv", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    effective_start_date = args.start_date or datetime.now().date().isoformat()
    if args.prompt_template_out:
        save_manual_prompt(
            path=args.prompt_template_out,
            prompt=args.prompt,
            days=args.days,
            start_date=effective_start_date,
        )
        print(f"[done] prompt_template={args.prompt_template_out}")
    bank = DDRE33ReferenceBank(
        pv18_csv=args.pv18_csv,
        pv33_csv=args.pv33_csv,
        wind22_csv=args.wind22_csv,
        wind25_csv=args.wind25_csv,
        pv18_labels_csv=args.pv18_labels_csv,
        pv33_labels_csv=args.pv33_labels_csv,
        wind22_labels_csv=args.wind22_labels_csv,
        wind25_labels_csv=args.wind25_labels_csv,
        max_cols=args.max_cols,
    )
    agent = AutonomousOrchestrationAgent(bank, ckpt_path=args.ckpt or None)
    if args.planner_mode == "heuristic":
        result = agent.run(
            prompt=args.prompt,
            days=args.days,
            start_date=effective_start_date,
            outdir=args.outdir,
            seed=args.seed,
            topk_candidates=args.topk_candidates,
            boundary_window=args.boundary_window,
            ramp_quantile=args.ramp_quantile,
        )
    elif args.planner_mode == "manual_json":
        if not args.manual_plan_json:
            raise ValueError("--manual_plan_json is required when --planner_mode manual_json")
        spec, macro, daily_plan, payload = load_plan_json(
            path=args.manual_plan_json,
            bank=bank,
            fallback_prompt=args.prompt,
            fallback_days=args.days,
            fallback_start_date=effective_start_date,
        )
        result = agent.run_from_plan(
            spec=spec,
            macro=macro,
            daily_plan=daily_plan,
            outdir=args.outdir,
            seed=args.seed,
            topk_candidates=args.topk_candidates,
            boundary_window=args.boundary_window,
            ramp_quantile=args.ramp_quantile,
        )
        Path(args.outdir, "manual_plan_normalized.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        payload = call_api_with_env(
            prompt=args.prompt,
            days=args.days,
            start_date=effective_start_date,
            api_key_env=args.api_key_env,
            model=args.api_model,
            base_url=args.api_base_url,
            timeout_sec=args.api_timeout_sec,
        )
        save_path = args.save_api_plan_json or str(Path(args.outdir) / "api_plan.json")
        save_api_plan(payload, save_path)
        print(f"[done] api_plan_json={save_path}")
        spec, macro, daily_plan, _ = load_plan_json(
            path=save_path,
            bank=bank,
            fallback_prompt=args.prompt,
            fallback_days=args.days,
            fallback_start_date=effective_start_date,
        )
        result = agent.run_from_plan(
            spec=spec,
            macro=macro,
            daily_plan=daily_plan,
            outdir=args.outdir,
            seed=args.seed,
            topk_candidates=args.topk_candidates,
            boundary_window=args.boundary_window,
            ramp_quantile=args.ramp_quantile,
        )
    print(f"[done] source_mode={result.source_mode}")
    print(f"[done] summary={result.summary_path}")
    if result.multi_day_csv:
        print(f"[done] multi_day_csv={result.multi_day_csv}")


if __name__ == "__main__":
    main()
