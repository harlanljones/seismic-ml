"""Serve-time inference for SeismicML (Phase A, LIVE_APP_ROADMAP §3).

CPU-safe: ``CUDA_VISIBLE_DEVICES`` is forced to ``"-1"`` at module import time,
before any TensorFlow import, so this module is importable and usable on a
GPU-less host. TensorFlow is imported lazily inside the functions that need it.

Contract (ROADMAP §4.1):
    ModelBundle  -- TypedDict describing a persisted run
    load_bundle(bundle_root) -> ModelBundle
    predict(grid_df)         -> DataFrame[lat_bin, lon_bin, period, prob]
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import joblib
import pandas as pd

if TYPE_CHECKING:
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler


class ModelBundle(TypedDict):
    """Handle to a persisted model run."""

    bundle_root: Path
    run_id: str
    model: tf.keras.Model
    scaler: StandardScaler
    metadata: dict


_CURRENT_BUNDLE: ModelBundle | None = None


def _resolve_bundle_dir(bundle_root: Path) -> Path:
    """Resolve the concrete run directory containing model weights.

    ``bundle_root`` may point directly at a run_id dir (contains
    ``weights/seismic_model.weights.h5``) or at a ``latest`` symlink (contains
    ``latest/weights/seismic_model.weights.h5``). The ``latest`` form is
    resolved through ``os.path.realpath`` so the returned path is canonical.
    """
    bundle_root = Path(bundle_root)
    direct_weights = bundle_root / "weights" / "seismic_model.weights.h5"
    if direct_weights.exists():
        return bundle_root

    latest = bundle_root / "latest"
    latest_weights = latest / "weights" / "seismic_model.weights.h5"
    if latest.exists() and latest_weights.exists():
        return Path(os.path.realpath(latest))

    raise FileNotFoundError(
        f"No model bundle found under {bundle_root}: expected "
        f"{direct_weights} or a 'latest' symlink to a run directory."
    )


def load_bundle(bundle_root: Path) -> ModelBundle:
    """Load a persisted :data:`ModelBundle` from ``bundle_root``.

    Sets the module-global current bundle so that :func:`predict` can use it
    without an explicit argument. Resolves the bundle directory robustly via
    :func:`_resolve_bundle_dir`.
    """
    from src.model.arch import build_4070ti_model

    global _CURRENT_BUNDLE

    bundle_root = Path(bundle_root)
    bundle_dir = _resolve_bundle_dir(bundle_root)

    metadata_path = bundle_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {bundle_dir}")
    with open(metadata_path, encoding="utf-8") as f:
        metadata: dict = json.load(f)

    scaler = joblib.load(bundle_dir / "scaler.joblib")

    model = build_4070ti_model(input_dim=int(metadata["input_dim"]))
    model.load_weights(bundle_dir / "weights" / "seismic_model.weights.h5")

    bundle: ModelBundle = {
        "bundle_root": bundle_root,
        "run_id": bundle_dir.name,
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
    }
    _CURRENT_BUNDLE = bundle
    return bundle


def predict(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Score a grid-feature DataFrame, returning per-bin probabilities.

    Requires :func:`load_bundle` to have been called first (sets the
    module-global bundle). The returned DataFrame has exactly the columns
    ``lat_bin, lon_bin, period, prob`` with finite ``prob`` values in ``[0, 1]``.

    Raises:
        RuntimeError: if :func:`load_bundle` has not been called.
        ValueError: if ``grid_df`` is missing required feature columns (or
            ``period``) or carries unexpected columns beyond the allowed extras
            (``target``, ``max_mag``).
    """
    bundle = _CURRENT_BUNDLE
    if bundle is None:
        raise RuntimeError("Call load_bundle(...) before predict(...)")

    metadata = bundle["metadata"]
    feature_cols = list(metadata["feature_cols"])

    required = set(feature_cols) | {"period"}
    allowed_extras = {"target", "max_mag"}
    present = set(grid_df.columns)

    missing = required - present
    if missing:
        raise ValueError(
            f"grid_df is missing required columns: {sorted(missing)}"
        )
    disallowed = present - required - allowed_extras
    if disallowed:
        raise ValueError(
            f"grid_df contains unexpected columns: {sorted(disallowed)}"
        )

    features = grid_df[feature_cols].to_numpy(dtype="float32")
    scaled = bundle["scaler"].transform(features).astype("float32")

    probs = bundle["model"].predict(scaled, verbose=0).ravel().astype(float)

    return pd.DataFrame(
        {
            "lat_bin": grid_df["lat_bin"].to_numpy(),
            "lon_bin": grid_df["lon_bin"].to_numpy(),
            "period": grid_df["period"].to_numpy(),
            "prob": probs,
        }
    )
