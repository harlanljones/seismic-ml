# HJ-491: Live app: Phase D — Retrain lifecycle & promotion gate

## Summary

Automated retraining lifecycle is implemented: `scheduled_retrain` fits a challenger on a ~3yr catalog with the shifted walk-forward split (HJ-490), and `promote_if_better` promotes only on higher test AUC with `train_val_gap <= 0.05` and finite loss. Promotion is a `latest` symlink repoint; prior-3 bundles retained for instant rollback. All `AGENTS.md` §2 invariants preserved (trainer unchanged).

## Changes

- `src/jobs/retrain.py` — `scheduled_retrain(bundle_root, days_lookback=1095)`: fetch, `build_spatiotemporal_grid`, shifted 70/15/15, `model.fit` with `PartitionMetricTracker`; writes challenger bundle, never swaps live
- `src/jobs/retrain.py` — `promote_if_better(challenger_id, bundle_root)`: AUC + gap + finite-loss gate; repoints `artifacts/model_bundle/latest` symlink
- `tests/test_lifecycle.py` — worse-AUC challenger is NOT promoted; symlink points to winner; old bundles intact

## Testing

- `python -m pytest tests/test_lifecycle.py -q` — passed (part of 11 passed with `test_serve.py`)
- Gate assertions: train/val gap ≤ 0.05, no XLA fallback, finite loss

## Notes

- Depends on HJ-487 (bundle format) and HJ-490 (shifted split); both verified present
- Timers: `deploy/retrain.timer` @03:00, `deploy/nowcast.timer` @06:00; nowcast always consumes `latest`
