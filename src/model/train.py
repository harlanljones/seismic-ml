"""Training entry point for SeismicML (workstream W5).

Contract (ROADMAP.md §4.7): GPU init first, walk-forward split (fit scaler on
train only), tf.data pipelines, mixed-precision model with XLA, fit with
PartitionMetricTracker, persist weights + history.json under artifacts/.

Phase A (LIVE_APP_ROADMAP §3) adds ``train_and_persist_bundle`` which mirrors
``train`` but persists a loadable ModelBundle into a run-specific directory.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import tensorflow as tf

from src.config import (
    ARTIFACTS_DIR,
    EPOCHS,
    FEATURE_COLS,
    GRID_SIZE_DEG,
    MIN_MAG,
    TARGET_MAG,
    TIME_STEP_DAYS,
    ensure_artifacts_dir,
    initialize_gpu,
)
from src.data.splits import scale_splits, walk_forward_split
from src.model.arch import build_4070ti_model
from src.model.tracker import PartitionMetricTracker
from src.pipeline.datasets import make_tf_dataset

if TYPE_CHECKING:
    from src.inference import ModelBundle


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
        )
        return out.stdout.decode().strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def train(grid_df: pd.DataFrame) -> tuple[tf.keras.Model, PartitionMetricTracker]:
    """Train the 4070 Ti model on a grid-feature DataFrame.

    Returns ``(model, tracker)``; persists
    ``artifacts/seismic_model.weights.h5`` and ``artifacts/history.json``.
    """
    initialize_gpu()

    X = grid_df[FEATURE_COLS].to_numpy()
    y = grid_df["target"].to_numpy()

    train_idx, val_idx, test_idx = walk_forward_split(len(grid_df))
    X_train, X_val, X_test, _scaler = scale_splits(
        X[train_idx], X[val_idx], X[test_idx]
    )
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    train_ds = make_tf_dataset(X_train, y_train, is_train=True)
    val_ds = make_tf_dataset(X_val, y_val)
    test_ds = make_tf_dataset(X_test, y_test)

    model = build_4070ti_model(input_dim=len(FEATURE_COLS))
    tracker = PartitionMetricTracker(test_dataset=test_ds)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[tracker],
        verbose=1,
    )

    ensure_artifacts_dir()
    model.save_weights(ARTIFACTS_DIR / "seismic_model.weights.h5")
    with open(ARTIFACTS_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(tracker.records, f)

    return model, tracker


def train_and_persist_bundle(
    grid_df: pd.DataFrame, bundle_root: Path
) -> "ModelBundle":
    """Train the 4070 Ti model and persist a loadable bundle on disk.

    Mirrors :func:`train` (GPU init, walk-forward split, leakage-free scaling,
    mixed-precision XLA fit with ``PartitionMetricTracker``) but writes into a
    run-specific directory under ``bundle_root`` instead of ``ARTIFACTS_DIR``:

    ``<bundle_root>/<run_id>/weights/seismic_model.weights.h5``
    ``<bundle_root>/<run_id>/scaler.joblib``
    ``<bundle_root>/<run_id>/metadata.json``

    Returns a :data:`ModelBundle` referencing the persisted artifacts.
    """
    import joblib
    from sklearn.metrics import roc_auc_score

    bundle_root = Path(bundle_root)
    initialize_gpu()

    X = grid_df[FEATURE_COLS].to_numpy()
    y = grid_df["target"].to_numpy()

    train_idx, val_idx, test_idx = walk_forward_split(len(grid_df))
    X_train, X_val, X_test, scaler = scale_splits(
        X[train_idx], X[val_idx], X[test_idx]
    )
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    train_ds = make_tf_dataset(X_train, y_train, is_train=True)
    val_ds = make_tf_dataset(X_val, y_val)
    test_ds = make_tf_dataset(X_test, y_test)

    model = build_4070ti_model(input_dim=len(FEATURE_COLS))
    tracker = PartitionMetricTracker(test_dataset=test_ds)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[tracker],
        verbose=1,
    )

    y_test_prob = model.predict(X_test, verbose=0).ravel()
    test_auc = float(roc_auc_score(y_test, y_test_prob))
    train_acc = float(tracker.records["train_acc"][-1])
    val_acc = float(tracker.records["val_acc"][-1])
    train_val_gap = abs(train_acc - val_acc)

    run_id = f"{_git_short_sha()}-{int(time.time())}"
    run_dir = bundle_root / run_id
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model.save_weights(weights_dir / "seismic_model.weights.h5")
    joblib.dump(scaler, run_dir / "scaler.joblib")

    period_min = grid_df["period"].min()
    period_max = grid_df["period"].max()
    metadata = {
        "model_version": run_id,
        "input_dim": len(FEATURE_COLS),
        "feature_cols": list(FEATURE_COLS),
        "grid_size_deg": GRID_SIZE_DEG,
        "time_step_days": TIME_STEP_DAYS,
        "min_mag": MIN_MAG,
        "target_mag": TARGET_MAG,
        "train_window_start": str(pd.Timestamp(period_min.start_time).date()),
        "train_window_end": str(pd.Timestamp(period_max.start_time).date()),
        "test_auc": test_auc,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_val_gap": train_val_gap,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "bundle_root": bundle_root,
        "run_id": run_id,
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
    }


if __name__ == "__main__":
    from src.config import END_DATE, GRID_SIZE_DEG, MIN_MAG, START_DATE, TIME_STEP_DAYS
    from src.data.features import build_spatiotemporal_grid
    from src.data.fetch import fetch_seismic_catalog

    catalog = fetch_seismic_catalog(
        start_date=START_DATE, end_date=END_DATE, min_mag=MIN_MAG
    )
    grid = build_spatiotemporal_grid(
        catalog, grid_size=GRID_SIZE_DEG, time_step_days=TIME_STEP_DAYS
    )
    trained_model, trained_tracker = train(grid)
    print(f"Final test accuracy: {trained_tracker.records['test_acc'][-1]:.4f}")
