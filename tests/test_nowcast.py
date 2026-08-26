"""Phase B nowcast job tests (LIVE_APP_ROADMAP §3/§5.3).

CPU-only, synthetic catalog injected so NO network fetch and NO USGS_LIVE_FETCH
are required. Mirrors the CUDA-disabling guard from tests/test_arch.py.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.features import build_spatiotemporal_grid
from src.jobs import nowcast
from src.model.train import train_and_persist_bundle


def _make_catalog(seed: int = 0) -> pd.DataFrame:
    """Synthetic USGS-style catalog spanning ~3 periods (21 days)."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2021-01-01")
    for _ in range(200):
        offset = int(rng.integers(0, 21))  # days within the 21-day window
        rows.append(
            {
                "time": base + pd.Timedelta(days=offset)
                + pd.Timedelta(seconds=int(rng.integers(0, 86400))),
                "lat": float(rng.choice([4.0, 6.0, 8.0])),
                "lon": float(rng.choice([-22.0, -20.0])),
                "depth": float(rng.normal(10.0, 2.0)),
                "mag": float(rng.uniform(2.5, 6.0)),
                "gap": float(rng.normal(100.0, 10.0)),
                "dmin": float(rng.uniform(0.5, 2.0)),
                "sig": float(rng.normal(50.0, 5.0)),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def bundle_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(config, "EPOCHS", 1)
    catalog = _make_catalog()
    grid = build_spatiotemporal_grid(
        catalog, grid_size=config.GRID_SIZE_DEG, time_step_days=config.TIME_STEP_DAYS
    )
    root = tmp_path / "model_bundle"
    bundle = train_and_persist_bundle(grid, root)
    return root / bundle["run_id"]


def test_run_nowcast_refuses_live_fetch_without_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="USGS_LIVE_FETCH"):
        nowcast.run_nowcast(
            bundle_path=tmp_path / "model_bundle" / "x", catalog=None
        )


def test_run_nowcast_emits_valid_forecast(
    tmp_path: Path, bundle_dir: Path, monkeypatch
) -> None:
    assert "USGS_LIVE_FETCH" not in os.environ
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)

    catalog = _make_catalog()
    out_path = nowcast.run_nowcast(
        bundle_path=bundle_dir, days_back=30, catalog=catalog
    )

    assert out_path.exists()
    assert out_path.name == "forecast.json"

    latest_link = tmp_path / "forecast" / "latest"
    assert latest_link.is_symlink()
    assert Path(os.path.realpath(latest_link)) == out_path.parent

    with open(out_path, encoding="utf-8") as f:
        forecast = json.load(f)

    for key in (
        "forecast_period",
        "verifiable_after",
        "generated_at",
        "grid_size_deg",
        "model_version",
        "features",
    ):
        assert key in forecast

    assert forecast["grid_size_deg"] == float(config.GRID_SIZE_DEG)

    rows = forecast["features"]
    assert rows
    seen_periods = set()
    for row in rows:
        for col in ("lat_bin", "lon_bin", "period", "prob"):
            assert col in row
        assert isinstance(row["lat_bin"], float)
        assert isinstance(row["lon_bin"], float)
        assert isinstance(row["period"], str)
        assert np.isfinite(row["prob"])
        assert 0.0 <= row["prob"] <= 1.0
        seen_periods.add(row["period"])

    assert len(seen_periods) >= 1

    freq = f"{config.TIME_STEP_DAYS}D"
    seen_periods_p = [pd.Timestamp(p).to_period(freq) for p in seen_periods]
    latest_period = max(seen_periods_p)
    expected_end = pd.Timestamp(latest_period.end_time) + pd.Timedelta(
        days=config.TIME_STEP_DAYS
    )
    assert forecast["verifiable_after"] == expected_end.isoformat()
    assert forecast["forecast_period"] == pd.Timestamp(
        latest_period.start_time
    ).isoformat()


def test_run_nowcast_idempotent_repoints_latest(
    tmp_path: Path, bundle_dir: Path, monkeypatch
) -> None:
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    catalog = _make_catalog()

    nowcast.run_nowcast(bundle_path=bundle_dir, catalog=catalog)
    second = nowcast.run_nowcast(bundle_path=bundle_dir, catalog=catalog)

    latest_link = tmp_path / "forecast" / "latest"
    assert Path(os.path.realpath(latest_link)) == second.parent
    with open(second, encoding="utf-8") as f:
        assert json.load(f)["features"]
