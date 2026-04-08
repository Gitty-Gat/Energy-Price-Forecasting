# Usage

Canonical CLI:

```bash
python src/pipelines/forecast.py \
  --merged data/processed/merged_exog.csv \
  --outputs results/forecasts \
  --horizons 10 20 \
  --seed 42
```

Hydra-configured equivalent:

```bash
python src/pipelines/forecast_hydra.py
```

Example Hydra overrides:

```bash
python src/pipelines/forecast_hydra.py \
  merged=/absolute/path/to/merged.csv \
  outputs=/absolute/path/to/out_dir \
  horizons=[3] \
  seed=9 \
  with_hybrid=false
```

See `docs/project-plan/VERIFICATION_MATRIX.md` for the exact commands that have been exercised locally.
