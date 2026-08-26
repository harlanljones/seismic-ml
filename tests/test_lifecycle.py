"""Phase D lifecycle tests (LIVE_APP_ROADMAP §3/§4.3): safe promotion gate
and end-to-end scheduled retrain.

CPU-only, synthetic data. ``CUDA_VISIBLE_DEVICES`` must be ``"-1"`` BEFORE any
TensorFlow (or src module importing it) is imported.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src import config
from src.jobs import retrain


def _make_bundle(
    root: Path, run_id: str, metadata: dict[str, float]
) -> Path:
    """Write a deterministic bundle dir with metadata + scaler + dummy weights."""
    run_dir = root / run_id
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    joblib.dump(StandardScaler(), run_dir / "scaler.joblib")
    (run_dir / "weights" / "seismic_model.weights.h5").write_bytes(b"dummy")
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return run_dir


def _metadata(test_auc: float, train_val_gap: float) -> dict[str, float]:
    return {
        "test_auc": test_auc,
        "train_acc": 0.80,
        "val_acc": 0.80 - train_val_gap,
        "train_val_gap": train_val_gap,
    }


@pytest.fixture
def bundles(tmp_path: Path) -> Path:
    root = tmp_path / "model_bundle"
    root.mkdir()
    incumbent = _make_bundle(root, "incumbent-1", _metadata(0.60, 0.02))
    _make_bundle(root, "challenger-2", _metadata(0.70, 0.02))
    (root / "latest").symlink_to(incumbent, target_is_directory=True)
    return root


def test_promote_higher_auc_and_finite_gap(bundles: Path) -> None:
    ok = retrain.promote_if_better("challenger-2", bundles)
    assert ok is True
    latest = Path(os.path.realpath(bundles / "latest"))
    assert latest.name == "challenger-2"
    # old bundle intact
    assert (bundles / "incumbent-1" / "metadata.json").exists()


def test_promote_lower_auc_returns_false(bundles: Path) -> None:
    _make_bundle(bundles, "challenger-lower", _metadata(0.55, 0.02))
    ok = retrain.promote_if_better("challenger-lower", bundles)
    assert ok is False
    latest = Path(os.path.realpath(bundles / "latest"))
    assert latest.name == "incumbent-1"


def test_promote_gap_too_large_returns_false(bundles: Path) -> None:
    _make_bundle(bundles, "challenger-gap", _metadata(0.95, 0.10))
    ok = retrain.promote_if_better("challenger-gap", bundles)
    assert ok is False
    latest = Path(os.path.realpath(bundles / "latest"))
    assert latest.name == "incumbent-1"


def test_promote_bootstrap_no_incumbent(tmp_path: Path) -> None:
    root = tmp_path / "model_bundle"
    root.mkdir()
    _make_bundle(root, "challenger-first", _metadata(0.70, 0.02))
    ok = retrain.promote_if_better("challenger-first", root)
    assert ok is True
    latest = Path(os.path.realpath(root / "latest"))
    assert latest.name == "challenger-first"


def _make_catalog(seed: int = 0) -> pd.DataFrame:
    """Synthetic USGS-style catalog spanning ~70 days (>= 9 grid periods)."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2021-01-01")
    for _ in range(900):
        offset = int(rng.integers(0, 70))
        rows.append(
            {
                "time": base
                + pd.Timedelta(days=offset)
                + pd.Timedelta(seconds=int(rng.integers(0, 86400))),
                "lat": float(rng.choice([4.0, 6.0, 8.0])),
                "lon": float(rng.choice([-22.0, -20.0, -18.0])),
                "depth": float(rng.normal(10.0, 2.0)),
                "mag": float(rng.uniform(2.5, 6.0)),
                "gap": float(rng.normal(100.0, 10.0)),
                "dmin": float(rng.uniform(0.5, 2.0)),
                "sig": float(rng.normal(50.0, 5.0)),
            }
        )
    return pd.DataFrame(rows)


def test_scheduled_retrain_end_to_end(
    tmp_path: Path, monkeypatch
) -> None:
    assert "USGS_LIVE_FETCH" not in os.environ
    monkeypatch.setattr(config, "EPOCHS", 1)

    root = tmp_path / "model_bundle"
    catalog = _make_catalog()

    run_id = retrain.scheduled_retrain(root, catalog=catalog)

    assert isinstance(run_id, str) and run_id
    run_dir = root / run_id
    assert run_dir.is_dir()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "scaler.joblib").exists()
    assert (run_dir / "weights" / "seismic_model.weights.h5").exists()

    with open(run_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    assert "test_auc" in metadata
    assert "train_val_gap" in metadata


def test_scheduled_retrain_refuses_network_without_env(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="USGS_LIVE_FETCH"):
        retrain.scheduled_retrain(tmp_path / "model_bundle", catalog=None)
