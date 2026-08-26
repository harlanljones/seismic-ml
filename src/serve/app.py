"""FastAPI serving surface for SeismicML (Phase C, LIVE_APP_ROADMAP §3).

Reads the latest persisted artifacts (forecast + promoted model bundle) at
request time and never trains. The ``/`` route server-renders a Leaflet page
with the forecast GeoJSON inlined so first paint needs no fetch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import src.config as _config
from src.serve.map_server import forecast_to_geojson

ARTIFACTS_DIR: Path = Path(_config.ARTIFACTS_DIR)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
templates = Jinja2Templates(directory=str(_FRONTEND_DIR))

app = FastAPI(title="SeismicML Forecast API")

app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


def _forecast_json_path() -> Path:
    p = Path(ARTIFACTS_DIR) / "forecast" / "latest" / "forecast.json"
    return Path(os.path.realpath(p))


def _model_metadata_path() -> Path:
    p = Path(ARTIFACTS_DIR) / "model_bundle" / "latest" / "metadata.json"
    return Path(os.path.realpath(p))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/forecast")
def forecast(lat: float | None = None, lon: float | None = None) -> dict:
    with open(_forecast_json_path(), encoding="utf-8") as f:
        fc = json.load(f)
    geo = forecast_to_geojson(fc)
    if lat is not None and lon is not None:
        geo = _nearest_bin(geo, lat, lon)
    return geo


def _nearest_bin(geo: dict, lat: float, lon: float) -> dict:
    feats = geo.get("features", [])
    if not feats:
        return geo
    best: dict | None = None
    best_d: float | None = None
    for ft in feats:
        coords = ft["geometry"]["coordinates"]
        d = (coords[1] - lat) ** 2 + (coords[0] - lon) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best = ft
    return {
        "type": "FeatureCollection",
        "forecast_period": geo.get("forecast_period"),
        "verifiable_after": geo.get("verifiable_after"),
        "grid_size_deg": geo.get("grid_size_deg"),
        "features": [best],
    }


@app.get("/model")
def model() -> dict:
    with open(_model_metadata_path(), encoding="utf-8") as f:
        return json.load(f)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    with open(_forecast_json_path(), encoding="utf-8") as f:
        fc = json.load(f)
    geo = forecast_to_geojson(fc)
    return templates.TemplateResponse(
        request, "index.html", {"request": request, "forecast": geo}
    )
