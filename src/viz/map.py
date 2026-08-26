"""Interactive Folium geospatial overlay map (workstream W7).

The layer-data computation is delegated to :func:`build_layer_payload` in
``src/serve/map_server.py`` (LIVE_APP_ROADMAP §5.2 faithful refactor) so the
offline rendering stays the single source of truth shared with the live app.
The rendered HTML output and visual semantics are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap

from src.serve.map_server import PRED_HEAT, TRAIN_HEAT, build_layer_payload

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
    payload = build_layer_payload(grid_data, val_idx, X_test_scaled, y_test, model)
    train_heatmap_data = payload["train_heatmap"]
    pred_heatmap_data = payload["pred_heatmap"]
    markers = payload["markers"]

    test_slice = grid_data.iloc[val_idx:]
    center_lat = test_slice["lat_bin"].mean()
    center_lon = test_slice["lon_bin"].mean()
    fmap = folium.Map(
        location=[center_lat, center_lon], zoom_start=4, tiles="CartoDB dark_matter"
    )

    train_layer = folium.FeatureGroup(name="Historical Training Density")
    HeatMap(train_heatmap_data, **TRAIN_HEAT).add_to(train_layer)
    train_layer.add_to(fmap)

    pred_heat_layer = folium.FeatureGroup(name="Predicted Risk Heatmap (Test Set)")
    HeatMap(pred_heatmap_data, **PRED_HEAT).add_to(pred_heat_layer)
    pred_heat_layer.add_to(fmap)

    marker_layer = folium.FeatureGroup(name="Model Verification Markers")
    for m in markers:
        popup_html = (
            f"<b>Status:</b> {m['label']}<br>"
            f"<b>Predicted Risk:</b> {m['prob']:.2%}<br>"
            f"<b>Actual M>=4.5:</b> {bool(m['gt'])}"
        )
        folium.CircleMarker(
            location=[m["lat"], m["lon"]],
            radius=5 + m["prob"] * 7,
            color=m["color"],
            fill=True,
            fill_color=m["color"],
            fill_opacity=0.8,
            popup=popup_html,
        ).add_to(marker_layer)

    marker_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    return fmap
