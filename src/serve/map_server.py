"""Layer-data adapter for the SeismicML serving surface (Phase C, §5.2).

This module is **pure Python** (no FastAPI, no Folium, no TensorFlow import at
module scope) so that ``src/viz/map.py`` can import it offline while the live
FastAPI app reuses the exact same layer semantics. It is the single source of
truth for the train-density heatmap, predicted-risk heatmap, and
confusion-matrix marker colors/labels, faithfully mirroring ``map.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# --- Centralized layer constants (preserve map.py:45-63 semantics) ---------

TRAIN_HEAT: dict[str, Any] = {
    "radius": 12,
    "blur": 10,
    "min_opacity": 0.2,
    "gradient": {0.2: "blue", 0.8: "cyan"},
}

PRED_HEAT: dict[str, Any] = {
    "radius": 16,
    "blur": 14,
    "min_opacity": 0.4,
    "gradient": {0.4: "orange", 1.0: "red"},
}


def forecast_to_geojson(forecast: dict) -> dict:
    """Convert a Phase-B ``forecast.json`` into a GeoJSON FeatureCollection.

    Each feature in ``forecast["features"]`` becomes a Point at the bin centroid
    ``[lon_bin + grid/2, lat_bin + grid/2]`` with properties
    ``prob``, ``period``, ``lat_bin``, ``lon_bin``. The FeatureCollection also
    carries the forecast-level ``forecast_period`` and ``verifiable_after``
    timestamps plus the ``grid_size_deg`` for convenience.
    """
    grid = float(forecast.get("grid_size_deg", 0.0))
    half = grid / 2.0

    features: list[dict] = []
    for feat in forecast.get("features", []):
        lat_bin = float(feat["lat_bin"])
        lon_bin = float(feat["lon_bin"])
        lon_c = lon_bin + half
        lat_c = lat_bin + half
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon_c, lat_c]},
                "properties": {
                    "prob": float(feat["prob"]),
                    "period": feat.get("period"),
                    "lat_bin": lat_bin,
                    "lon_bin": lon_bin,
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "forecast_period": forecast.get("forecast_period"),
        "verifiable_after": forecast.get("verifiable_after"),
        "grid_size_deg": grid,
        "features": features,
    }


def build_layer_payload(
    grid_data: pd.DataFrame,
    val_idx: int,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    model: Any,
) -> dict:
    """Build the three map layer payloads from test-partition data.

    Extracted verbatim (semantically) from ``src/viz/map.py`` so offline and
    live renderings match. Returns::

        {
          "train_heatmap": [[lat, lon, event_count], ...],
          "pred_heatmap":  [[lat, lon, prob], ...],
          "markers":      [{lat, lon, prob, gt, pred, color, label}, ...],
        }

    Marker colors/labels mirror the confusion matrix (map.py:66-95):
    TP green / FP orange / FN red; TN emits no marker (skipped), matching the
    offline behavior exactly.
    """
    preds_prob = model.predict(X_test_scaled).flatten()
    preds_binary = (preds_prob >= 0.5).astype(int)

    test_slice = grid_data.iloc[val_idx:].copy().reset_index(drop=True)
    test_slice["prob"] = preds_prob
    test_slice["pred"] = preds_binary
    test_slice["actual"] = y_test

    train_heatmap = (
        grid_data.iloc[:val_idx][["lat_bin", "lon_bin", "event_count"]]
        .values.tolist()
    )
    pred_heatmap = (
        test_slice[["lat_bin", "lon_bin", "prob"]].values.tolist()
    )

    markers: list[dict] = []
    for _, row in test_slice.iterrows():
        lat = float(row["lat_bin"])
        lon = float(row["lon_bin"])
        prob = float(row["prob"])
        gt = int(row["actual"])
        pred = int(row["pred"])

        if gt == 1 and pred == 1:
            color, label = "#00FF00", "True Positive (Accurate Forecast)"
        elif gt == 0 and pred == 1:
            color, label = "#FFA500", "False Positive (False Alarm)"
        elif gt == 1 and pred == 0:
            color, label = "#FF0000", "False Negative (Missed Event)"
        else:
            continue

        markers.append(
            {
                "lat": lat,
                "lon": lon,
                "prob": prob,
                "gt": gt,
                "pred": pred,
                "color": color,
                "label": label,
            }
        )

    return {
        "train_heatmap": train_heatmap,
        "pred_heatmap": pred_heatmap,
        "markers": markers,
    }
