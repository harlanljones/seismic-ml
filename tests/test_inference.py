"""Phase A inference-bundle round-trip tests (LIVE_APP_ROADMAP §3).

CPU-only, synthetic data. Mirrors the CUDA-disabling guard from
``tests/test_arch.py`` / ``tests/test_datasets.py``: ``CUDA_VISIBLE_DEVICES``
must be set to ``"-1"`` BEFORE any TensorFlow (or src module importing it) is
imported.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS
from src.inference import load_bundle, predict
from src.model.train import train_and_persist_bundle


def _make_grid(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Build a chronologically sorted synthetic grid (one row per period)."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Period("2021-01-04", freq="7D")
    for i in range(n):
        rows.append(
            {
                "lat_bin": float(rng.choice([4.0, 6.0, 8.0])),
                "lon_bin": float(rng.choice([-22.0, -20.0])),
                "event_count": float(rng.integers(1, 50)),
                "mean_depth": float(rng.normal(10.0, 2.0)),
                "mean_gap": float(rng.normal(100.0, 10.0)),
                "mean_sig": float(rng.normal(50.0, 5.0)),
                "period": base + i,
                "target": int(rng.integers(0, 2)),
                "max_mag": float(rng.uniform(3.0, 5.0)),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def bundle_root(tmp_path: Path) -> Path:
    root = tmp_path / "model_bundle"
    return root


def test_train_and_persist_bundle_writes_artifacts(bundle_root: Path) -> None:
    grid = _make_grid()
    bundle = train_and_persist_bundle(grid, bundle_root)

    run_dir = bundle_root / bundle["run_id"]
    assert (run_dir / "weights" / "seismic_model.weights.h5").exists()
    assert (run_dir / "scaler.joblib").exists()
    assert (run_dir / "metadata.json").exists()


def test_load_bundle_then_predict(bundle_root: Path) -> None:
    grid = _make_grid()
    bundle = train_and_persist_bundle(grid, bundle_root)
    run_id = bundle["run_id"]

    load_bundle(bundle_root / run_id)

    preds = predict(grid)
    assert list(preds.columns) == ["lat_bin", "lon_bin", "period", "prob"]
    assert len(preds) == len(grid)
    assert np.isfinite(preds["prob"].to_numpy()).all()
    assert float(preds["prob"].min()) >= 0.0
    assert float(preds["prob"].max()) <= 1.0


def test_predict_raises_on_missing_feature(bundle_root: Path) -> None:
    grid = _make_grid()
    bundle = train_and_persist_bundle(grid, bundle_root)
    load_bundle(bundle_root / bundle["run_id"])

    dropped = grid.drop(columns=["mean_sig"])
    with pytest.raises(ValueError):
        predict(dropped)


def test_metadata_contains_eval_fields(bundle_root: Path) -> None:
    grid = _make_grid()
    bundle = train_and_persist_bundle(grid, bundle_root)
    run_dir = bundle_root / bundle["run_id"]

    with open(run_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    for key in (
        "test_auc",
        "train_acc",
        "val_acc",
        "train_val_gap",
        "model_version",
        "input_dim",
        "feature_cols",
        "train_window_start",
        "train_window_end",
    ):
        assert key in metadata

    assert metadata["feature_cols"] == list(FEATURE_COLS)
    assert isinstance(metadata["test_auc"], float)
    assert isinstance(metadata["train_val_gap"], float)
