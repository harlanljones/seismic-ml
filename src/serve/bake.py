"""One-time bake of the SeismicML live app into a self-contained static bundle.

Runs the REAL pipeline once on this machine (USGS catalog fetch -> grid ->
GPU training -> promote bundle -> nowcast -> held-out test-eval receipts) and
emits a single ``frontend/baked.js`` that defines ``window.SEISMIC_BAKED``
with every dataset the web app needs:

- ``forecast``: the live nowcast grid (from ``run_nowcast``)
- ``receipts``: held-out test-partition TP/FP/FN verification + scorecard
- ``events``: recent catalog events (Watch beat)
- ``model``: promoted bundle ``metadata.json``
- ``history``: available forecast snapshots under ``artifacts/forecast/``

This is a deliberate operator command (mirrors ``python -m src.model.train``):
it fetches live USGS data and trains on the GPU. It is NOT part of the serving
path — the web app never trains.

Usage::

    source .venv/bin/activate && python -m src.serve.bake

Gated by the same ``USGS_LIVE_FETCH`` switch as the jobs: set it to ``1`` for a
live fetch, or inject ``catalog=`` for a synthetic/offline bake (tests).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src import config
from src.data.features import build_spatiotemporal_grid
from src.data.fetch import fetch_seismic_catalog
from src.data.splits import walk_forward_split
from src.jobs.retrain import promote_if_better
from src.model.train import train_and_persist_bundle

if TYPE_CHECKING:
    from src.inference import ModelBundle

BAKED_JS = "frontend/baked.js"


def _live_fetch_enabled() -> bool:
    value = os.environ.get("USGS_LIVE_FETCH", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_latest_forecast() -> dict[str, Any]:
    """Read ``artifacts/forecast/latest/forecast.json`` (the nowcast output)."""
    root = config.ARTIFACTS_DIR / "forecast" / "latest"
    p = Path(os.path.realpath(root))
    with open(p / "forecast.json", encoding="utf-8") as f:
        return json.load(f)


def _load_model_metadata() -> dict[str, Any]:
    """Read ``artifacts/model_bundle/latest/metadata.json`` (promoted bundle)."""
    root = config.ARTIFACTS_DIR / "model_bundle" / "latest"
    p = Path(os.path.realpath(root))
    with open(p / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _history_records() -> list[dict[str, Any]]:
    """Enumerate forecast snapshots under ``artifacts/forecast/`` newest-first."""
    root = config.ARTIFACTS_DIR / "forecast"
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir(), reverse=True):
        if entry.name == "latest" or not entry.is_dir():
            continue
        fc_path = entry / "forecast.json"
        if not fc_path.exists():
            continue
        try:
            with open(fc_path, encoding="utf-8") as f:
                fc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        records.append(
            {
                "ts": entry.name,
                "forecast_period": fc.get("forecast_period"),
                "verifiable_after": fc.get("verifiable_after"),
                "generated_at": fc.get("generated_at"),
                "features": len(fc.get("features", [])),
            }
        )
    return records
def _receipts_payload(grid_data: pd.DataFrame, bundle: ModelBundle) -> dict[str, Any]:
    """Recompute held-out test verification with the bundle's own scaler.

    Uses the exact same walk-forward split the trainer used, transforms the
    test partition with the persisted StandardScaler, predicts once, and
    derives the full confusion matrix (including TN, which the marker layer
    suppresses) plus the marker list mirroring the offline map semantics.
    """
    from src.serve.map_server import _make_markers

    X = grid_data[config.FEATURE_COLS].to_numpy(dtype="float32")
    y = grid_data["target"].to_numpy()

    _train, _val, te_idx = walk_forward_split(len(grid_data))
    scaler = bundle["scaler"]

    X_test = scaler.transform(X[te_idx]).astype("float32")
    y_test = y[te_idx]
    probs = bundle["model"].predict(X_test, verbose=0).ravel().astype(float)
    preds = (probs >= 0.5).astype(int)

    test_slice = grid_data.iloc[te_idx].copy().reset_index(drop=True)
    test_slice["prob"] = probs
    test_slice["pred"] = preds
    test_slice["actual"] = y_test
    markers = _make_markers(test_slice)

    tn = int(np.sum((y_test == 0) & (preds == 0)))
    fp = int(np.sum((y_test == 0) & (preds == 1)))
    fn = int(np.sum((y_test == 1) & (preds == 0)))
    tp = int(np.sum((y_test == 1) & (preds == 1)))

    hit = (tp + fn) and tp / (tp + fn) * 100 or 0.0
    far = (fp + tn) and fp / (fp + tn) * 100 or 0.0

    return {
        "model_version": bundle["metadata"].get("model_version"),
        "test_acc": bundle["metadata"].get("test_acc"),
        "train_acc": bundle["metadata"].get("train_acc"),
        "val_acc": bundle["metadata"].get("val_acc"),
        "train_val_gap": bundle["metadata"].get("train_val_gap"),
        "test_auc": bundle["metadata"].get("test_auc"),
        "test_window_start": bundle["metadata"].get("train_window_start"),
        "test_window_end": bundle["metadata"].get("train_window_end"),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "hit_rate_pct": round(float(hit), 1),
        "false_alarm_rate_pct": round(float(far), 1),
        "n_markers": len(markers),
        "markers": markers,
    }
def _recent_events(catalog: pd.DataFrame, days: int, max_events: int) -> list[dict[str, Any]]:
    """Lightweight recent-event records (lon/lat/mag/time) for the Watch beat."""
    if catalog.empty:
        return []
    cutoff = catalog["time"].max() - pd.Timedelta(days=days)
    recent = catalog[catalog["time"] >= cutoff].sort_values("time", ascending=False)
    recent = recent.head(max_events)
    return [
        {"lon": float(r["lon"]), "lat": float(r["lat"]),
         "mag": float(r["mag"]), "time": str(r["time"])}
        for _, r in recent.iterrows()
    ]


def _write_baked_js(output_path: Path, payload: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    js = (
        "// Generated by src/serve/bake.py — do not edit by hand.\n"
        "// Run `python -m src.serve.bake` to regenerate from live USGS data.\n"
        "window.SEISMIC_BAKED = "
        + json.dumps(payload, indent=2, default=str)
        + ";\n"
    )
    output_path.write_text(js, encoding="utf-8")
    print(f"[bake] wrote {output_path} ({output_path.stat().st_size / 1024:.0f} KiB)")
    return output_path


def bake(
    output_path: str | Path = BAKED_JS, catalog: pd.DataFrame | None = None
) -> Path:
    """Run the real pipeline and write ``frontend/baked.js``.

    Mirrors the deploy flow: fetch (or injected) catalog -> grid -> GPU
    ``train_and_persist_bundle`` -> promote via the safe gate -> nowcast ->
    test-eval receipts -> single-file baked export.
    """
    output_path = Path(output_path)

    if catalog is None:
        if not _live_fetch_enabled():
            raise RuntimeError(
                "USGS_LIVE_FETCH not set — refusing network fetch in non-live mode"
            )
        end = datetime.now(UTC).date()
        start = end - timedelta(days=1095)
        catalog = fetch_seismic_catalog(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            min_mag=config.MIN_MAG,
        )

    grid = build_spatiotemporal_grid(
        catalog, grid_size=config.GRID_SIZE_DEG, time_step_days=config.TIME_STEP_DAYS
    )
    if grid.empty:
        raise RuntimeError("bake: empty grid — catalog window too small")

    from src.jobs.nowcast import run_nowcast

    bundle_root = config.ARTIFACTS_DIR / "model_bundle"
    bundle = train_and_persist_bundle(grid, bundle_root)
    promote_if_better(bundle["run_id"], bundle_root)

    run_nowcast(bundle_path=bundle_root / "latest", catalog=catalog)

    forecast = _load_latest_forecast()
    model_meta = _load_model_metadata()
    history = _history_records()
    receipts = _receipts_payload(grid, bundle)
    events = _recent_events(catalog, days=120, max_events=4000)

    manifest = {
        "baked_at": datetime.now(UTC).isoformat(),
        "model_version": model_meta.get("model_version"),
        "forecast_period": forecast.get("forecast_period"),
        "verifiable_after": forecast.get("verifiable_after"),
        "grid_size_deg": forecast.get("grid_size_deg"),
        "history": history,
    }
    payload = {
        "manifest": manifest,
        "forecast": forecast,
        "receipts": receipts,
        "events": events,
        "model": model_meta,
        "history": history,
    }
    return _write_baked_js(output_path, payload)


if __name__ == "__main__":
    out = bake()
    print(f"[bake] done -> {out}")