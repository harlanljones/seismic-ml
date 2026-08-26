"""Live nowcast job for SeismicML (Phase B, LIVE_APP_ROADMAP §3/§5.3).

Runs the production loop: optional USGS catalog fetch -> spatiotemporal grid
-> bundle inference -> ``artifacts/forecast/<ts>/forecast.json`` (plus a
``latest`` symlink). Network fetches are gated behind ``USGS_LIVE_FETCH`` so CI
never reaches the network.

Contract (ROADMAP §4.2): the emitted ``forecast.json`` uses the exact shape
documented in LIVE_APP_ROADMAP §3 Phase B.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from src import config
from src.data.features import build_spatiotemporal_grid
from src.data.fetch import fetch_seismic_catalog
from src.inference import load_bundle, predict


def _live_fetch_enabled() -> bool:
    value = os.environ.get("USGS_LIVE_FETCH", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _period_to_iso(period: pd.Period) -> str:
    return pd.Timestamp(period.start_time).isoformat()


def run_nowcast(
    bundle_path: str | Path,
    days_back: int = 30,
    catalog: pd.DataFrame | None = None,
) -> Path:
    """Execute one nowcast cycle and write ``forecast.json``.

    If ``catalog`` is provided it is used directly (no network). Otherwise a
    live USGS fetch is attempted only when ``USGS_LIVE_FETCH`` is truthy;
    absent that, a clear error is raised so CI can never hit the network.

    Returns the path to the written ``forecast.json``. Re-running overwrites the
    timestamped directory and repoints the ``latest`` symlink (idempotent).
    """
    bundle_path = Path(bundle_path)

    if catalog is None:
        if not _live_fetch_enabled():
            raise RuntimeError(
                "USGS_LIVE_FETCH not set — refusing network fetch in non-live mode"
            )
        end = datetime.now(UTC).date()
        start = end - timedelta(days=days_back)
        catalog = fetch_seismic_catalog(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            min_mag=config.MIN_MAG,
        )

    grid = build_spatiotemporal_grid(
        catalog,
        grid_size=config.GRID_SIZE_DEG,
        time_step_days=config.TIME_STEP_DAYS,
    )

    bundle = load_bundle(bundle_path)
    preds = predict(grid)

    if preds.empty:
        raise RuntimeError(
            "Nowcast produced no predictions; catalog window too small to form "
            "a completed period."
        )

    feature_periods = pd.PeriodIndex(preds["period"])
    latest_period = feature_periods.max()
    period_end = pd.Timestamp(latest_period.end_time)
    verifiable_after = period_end + timedelta(days=config.TIME_STEP_DAYS)

    forecast_period_iso = _period_to_iso(latest_period)
    verifiable_after_iso = verifiable_after.isoformat()
    generated_at_iso = datetime.now(UTC).isoformat()

    features = []
    for _, row in preds.iterrows():
        features.append(
            {
                "lat_bin": float(row["lat_bin"]),
                "lon_bin": float(row["lon_bin"]),
                "period": _period_to_iso(row["period"]),
                "prob": float(row["prob"]),
            }
        )

    forecast = {
        "forecast_period": forecast_period_iso,
        "verifiable_after": verifiable_after_iso,
        "generated_at": generated_at_iso,
        "grid_size_deg": float(config.GRID_SIZE_DEG),
        "model_version": str(bundle["metadata"]["model_version"]),
        "features": features,
    }

    root = config.ARTIFACTS_DIR / "forecast"
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    forecast_dir = root / ts
    forecast_dir.mkdir(parents=True, exist_ok=True)

    out_path = forecast_dir / "forecast.json"
    out_path.write_text(json.dumps(forecast, indent=2), encoding="utf-8")

    latest_link = root / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        import shutil

        shutil.rmtree(latest_link)
    os.symlink(forecast_dir, latest_link)

    return out_path
