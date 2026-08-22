"""Training entry point for SeismicML (workstream W5).

Contract (ROADMAP.md §4.7): GPU init first, walk-forward split (fit scaler on
train only), tf.data pipelines, mixed-precision model with XLA, fit with
PartitionMetricTracker, persist weights + history.json under artifacts/.
"""

import json

import pandas as pd
import tensorflow as tf

from src.config import (
    ARTIFACTS_DIR,
    EPOCHS,
    FEATURE_COLS,
    ensure_artifacts_dir,
    initialize_gpu,
)
from src.data.splits import scale_splits, walk_forward_split
from src.model.arch import build_4070ti_model
from src.model.tracker import PartitionMetricTracker
from src.pipeline.datasets import make_tf_dataset


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
