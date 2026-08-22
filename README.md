# SeismicML — ML Earthquake Forecasting Pipeline

Spatiotemporal binary classification that forecasts the probability of an **M ≥ 4.5 earthquake within a 7-day window** from 30-day seismic precursor features, built for an NVIDIA RTX 4070 Ti (Tensor Cores, mixed precision, XLA).

## Results (trained on USGS catalog 2021-01-01 → 2024-01-01, 83,107 events)

| Metric | Value |
| --- | --- |
| Final test accuracy | **0.7666** |
| Overfitting gap \|Acc_train − Acc_val\| | 0.0106 (target ≤ 0.05) |
| Loss stability | Finite across all 50 epochs (no NaN under mixed_float16) |
| VRAM usage @ batch 128 | ~1.7 GB (ceiling 6.5 GB) |
| XLA training step | ~0.6–0.9 ms (target ≤ 1.8 ms) |

Generated deliverables (recreate via `python -m src.model.train`, then Wave C renders):

- `artifacts/convergence.png` — train/val/test loss & accuracy convergence curves
- `artifacts/seismic_forecast_map.html` — interactive Folium map: historical density heatmap, predicted-risk heatmap, TP/FP/FN verification markers

## How it works

1. **Ingestion** — paginated USGS FDSN queries (handles the 20k-event API cap), null-filtered, fallback-filled.
2. **Feature engineering** — events aggregated onto a 2° spatial / 7-day temporal grid: event counts, mean depth/gap/sig; target = M≥4.5 in the *next* period per cell.
3. **Splitting** — strictly chronological walk-forward 70/15/15 (`max(T_train) < min(T_val) < min(T_test)`); `StandardScaler` fit on train only. Leakage is asserted by unit tests.
4. **Training** — Keras MLP (Dense-128 → BN → Dropout 0.25 → Dense-64 → Dense-64 → sigmoid head in `dtype="float32"`), global `mixed_float16` policy, `jit_compile=True` (XLA).
5. **Evaluation & mapping** — per-epoch train/val/test tracking via `PartitionMetricTracker`; Folium overlay with confusion-matrix markers.

## Quickstart

Requires Python 3.12+ and (optionally) a CUDA 12.x GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate      # also puts pip-provided CUDA libs on LD_LIBRARY_PATH
pip install -r requirements.txt -r requirements-dev.txt

# full pipeline: fetch → grid features → walk-forward split → 50-epoch GPU training
python -m src.model.train
```

Run the test suite:

```bash
python -m pytest tests/ -q     # 22 tests: leakage, schema, dtype/XLA contracts
ruff check src tests && mypy src
```

> **GPU note:** TensorFlow wheels do not self-locate the pip-installed CUDA/cuDNN libraries on all driver stacks. Activating `.venv` applies the required `LD_LIBRARY_PATH` automatically (see the hook appended to `.venv/bin/activate` at setup).

## Repository layout

```
src/
├── config.py            # constants, lazy GPU init, artifact paths
├── data/fetch.py        # paginated USGS FDSN ingestion
├── data/features.py     # spatiotemporal grid aggregation + target
├── data/splits.py       # walk-forward split + train-only scaling
├── pipeline/datasets.py # tf.data pipelines (batch 128, AUTOTUNE prefetch)
├── model/arch.py        # Tensor-Core-aligned architecture (128/64/64)
├── model/tracker.py     # PartitionMetricTracker callback
├── model/train.py       # training entry point
└── viz/                 # convergence plotting + Folium overlay map
tests/                   # 22 tests incl. leakage and mixed-precision contracts
docs: TDD.md · ROADMAP.md · AGENTS.md
```

## Documentation

- [`TDD.md`](./TDD.md) — technical design document (architecture, math, acceptance criteria)
- [`ROADMAP.md`](./ROADMAP.md) — workstream decomposition, frozen module contracts, gate results
- [`AGENTS.md`](./AGENTS.md) — operating manual for parallel-agent development of this repo

## Status

All gates passed (see ROADMAP §5). Remaining human review item: visual confirmation that high-probability clusters on the overlay map intersect active plate boundaries.
