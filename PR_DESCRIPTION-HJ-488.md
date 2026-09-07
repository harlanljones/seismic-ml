# HJ-488: Live app: Phase B — Daily nowcast job

## Summary

Phase B live data loop is implemented: `run_nowcast` refreshes a production grid from the USGS catalog and publishes the latest completed period with a `verifiable_after` label-latency boundary. No training on the request path. USGS fetch is env-gated so CI never hits the network.

## Changes

- `src/jobs/nowcast.py` — `run_nowcast(bundle_path, days_back=30)`: fetch → `build_spatiotemporal_grid` → `load_bundle().predict()` → writes `artifacts/forecast/<ts>/forecast.json` with `forecast_period` and `verifiable_after`; idempotent under re-runs
- `tests/test_nowcast.py` — mocked fetch returns synthetic catalog; asserts `forecast.json` schema keys; offline simulation of synthetic "today"

## Testing

- `python -m pytest tests/test_nowcast.py -q` — passed (part of 8 passed with `test_splits.py`)
- Verify: `USGS_LIVE_FETCH` unset in CI; no network calls in tests

## Notes

- Depends on HJ-487 (inference bundle) at runtime — `load_bundle`/`predict` must exist; verified present in `src/inference.py`
- Publishes latest completed period only; trailing NaN-target period dropped by `build_spatiotemporal_grid`
