# Chapter 4 Orchestration Module

This folder implements the simulation program for Chapter 4:

- semantic prompt parsing
- knowledge-driven macro baseline planning
- non-stationary micro-volatility autoregressive planning
- optional PCM-GAN daily generation
- Lipschitz-continuous multi-day boundary stitching

## Main entry

```bash
python -m pcm_gan_ch4.run_orchestration \
  --prompt "7-day severe cold snap with strong wind fluctuation" \
  --days 7 \
  --start_date 2025-12-15 \
  --ckpt pcm_gan_runs_full_stage2_seed42/pcm_gan_15min_best.pt \
  --pv18_csv pcm_gan_data/ororiginal_data/node_18_PV.csv \
  --pv33_csv pcm_gan_data/ororiginal_data/node_33_PV.csv \
  --wind22_csv pcm_gan_data/ororiginal_data/node_22_wind.csv \
  --wind25_csv pcm_gan_data/ororiginal_data/node_25_wind.csv \
  --pv18_labels_csv pcm_gan_data/ororiginal_data/node_18_PV_labels_climate.csv \
  --pv33_labels_csv pcm_gan_data/ororiginal_data/node_33_PV_labels_climate.csv \
  --wind22_labels_csv pcm_gan_data/ororiginal_data/node_22_wind_labels.csv \
  --wind25_labels_csv pcm_gan_data/ororiginal_data/node_25_wind_labels.csv \
  --outdir pcm_gan_ch4_runs/cold_snap_demo
```

## Current behavior

- If `--ckpt` exists, the module loads the trained PCM-GAN generator and synthesizes each day with planned surrogate conditions.
- If `--ckpt` is absent, the module still runs in `reference_fallback` mode and uses nearest DDRE-33 reference days to complete the Chapter 4 pipeline.

## Output files

- `prompt_spec.json`
- `macro_plan.json`
- `daily_plan.csv`
- `generation_trace.csv`
- `multi_day_profile.csv`
- `orchestration_summary.json`
- `plots/micro_plan.png`
- `plots/multi_day_profile.png`
