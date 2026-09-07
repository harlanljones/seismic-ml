# HJ-490: Live app: Contract amendment — walk_forward_split shifted windows (W3)

## Summary

`walk_forward_split` accepts optional `train_end`/`val_end` bounds (defaults preserve frozen 70/15/15 for all existing callers). Retraining passes shifted cut points so the held-out test window is `t..t+7` with observable labels at promotion time. `src/jobs/__init__.py` exists; `src/jobs/retrain.py` consumes the shifted split.

## Changes

- `src/data/splits.py` — `walk_forward_split(n_samples, train_end=0.70, val_end=0.85)`; contiguous slices, no shuffling, deterministic
- `tests/test_splits.py` — default 70/15/15 boundaries, contiguity/completeness, determinism; shifted-bound coverage for retrain cut points
- `src/jobs/retrain.py` — passes shifted bounds for the observable-label test window

## Testing

- `python -m pytest tests/test_splits.py -q` — passed (part of 8 passed with `test_nowcast.py`)
- Existing callers (`train`, `train_and_persist_bundle`) unchanged — defaults preserve behavior

## Notes

- Leakage invariant tightens for live setting: promotion gate never uses the latest 7-day window as a hard positive set
- Unblocks HJ-491 (Phase D promotion gate)
