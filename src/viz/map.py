"""Interactive Folium geospatial overlay map (workstream W7)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap

if TYPE_CHECKING:
    from tensorflow.keras import Model


def generate_spatial_overlay_map(
    grid_data: pd.DataFrame,
    val_idx: int,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    model: Model,
    output_path: str = "artifacts/seismic_forecast_map.html",
) -> folium.Map:
    """Render an interactive Leaflet map of training density, predicted risk,
    and confusion-matrix verification markers over the test partition."""
    preds_prob = model.predict(X_test_scaled).flatten()
    preds_binary = (preds_prob >= 0.5).astype(int)

    test_slice = grid_data.iloc[val_idx:].copy().reset_index(drop=True)
    test_slice["prob"] = preds_prob
    test_slice["pred"] = preds_binary
    test_slice["actual"] = y_test

    center_lat = test_slice["lat_bin"].mean()
    center_lon = test_slice["lon_bin"].mean()
    fmap = folium.Map(
        location=[center_lat, center_lon], zoom_start=4, tiles="CartoDB dark_matter"
    )

    train_heatmap_data = grid_data.iloc[:val_idx][
        ["lat_bin", "lon_bin", "event_count"]
    ].values.tolist()
    train_layer = folium.FeatureGroup(name="Historical Training Density")
    HeatMap(
        train_heatmap_data,
        radius=12,
        blur=10,
        min_opacity=0.2,
        gradient={0.2: "blue", 0.8: "cyan"},
    ).add_to(train_layer)
    train_layer.add_to(fmap)

    pred_heatmap_data = test_slice[["lat_bin", "lon_bin", "prob"]].values.tolist()
    pred_heat_layer = folium.FeatureGroup(name="Predicted Risk Heatmap (Test Set)")
    HeatMap(
        pred_heatmap_data,
        radius=16,
        blur=14,
        min_opacity=0.4,
        gradient={0.4: "orange", 1.0: "red"},
    ).add_to(pred_heat_layer)
    pred_heat_layer.add_to(fmap)

    marker_layer = folium.FeatureGroup(name="Model Verification Markers")
    for _, row in test_slice.iterrows():
        lat, lon = row["lat_bin"], row["lon_bin"]
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

        popup_html = (
            f"<b>Status:</b> {label}<br>"
            f"<b>Predicted Risk:</b> {prob:.2%}<br>"
            f"<b>Actual M>=4.5:</b> {bool(gt)}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=5 + prob * 7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=popup_html,
        ).add_to(marker_layer)

    marker_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    return fmap
