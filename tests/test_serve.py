"""Phase C serving-surface tests (LIVE_APP_ROADMAP §3, G_phC).

CPU-only, hermetic. Writes a temporary ``forecast.json`` and ``metadata.json``
under a temp artifacts root, points ``src.config.ARTIFACTS_DIR`` (and
``src.serve.app.ARTIFACTS_DIR``) at it, then drives the FastAPI app via
``TestClient``. No network, no training.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FORECAST_SAMPLE = {
    "forecast_period": "2026-08-26T00:00:00",
    "verifiable_after": "2026-09-02T00:00:00",
    "generated_at": "2026-08-26T06:00:00",
    "grid_size_deg": 2.0,
    "model_version": "abc123-1724659200",
    "features": [
        {"lat_bin": 4.0, "lon_bin": -22.0, "period": "2026-08-26T00:00:00", "prob": 0.71},
        {"lat_bin": 6.0, "lon_bin": -20.0, "period": "2026-08-26T00:00:00", "prob": 0.32},
        {"lat_bin": 8.0, "lon_bin": -22.0, "period": "2026-08-26T00:00:00", "prob": 0.55},
    ],
}

METADATA_SAMPLE = {
    "model_version": "abc123-1724659200",
    "input_dim": 6,
    "feature_cols": ["lat_bin", "lon_bin", "event_count", "mean_depth", "mean_gap", "mean_sig"],
    "grid_size_deg": 2.0,
    "time_step_days": 7,
    "min_mag": 2.5,
    "target_mag": 4.5,
    "train_window_start": "2021-01-01",
    "train_window_end": "2024-01-01",
}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    import src.serve.app as app_mod
    from src import config

    original = config.ARTIFACTS_DIR
    config.ARTIFACTS_DIR = tmp_path
    app_mod.ARTIFACTS_DIR = tmp_path

    fc_dir = tmp_path / "forecast" / "latest"
    fc_dir.mkdir(parents=True)
    (fc_dir / "forecast.json").write_text(json.dumps(FORECAST_SAMPLE), encoding="utf-8")

    mb_dir = tmp_path / "model_bundle" / "latest"
    mb_dir.mkdir(parents=True)
    (mb_dir / "metadata.json").write_text(json.dumps(METADATA_SAMPLE), encoding="utf-8")

    with TestClient(app_mod.app) as c:
        yield c

    config.ARTIFACTS_DIR = original


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_forecast_geojson(client: TestClient) -> None:
    resp = client.get("/forecast")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) >= 1
    feat = body["features"][0]
    assert feat["geometry"]["type"] == "Point"
    assert "prob" in feat["properties"]


def test_forecast_nearest_bin(client: TestClient) -> None:
    resp = client.get("/forecast", params={"lat": 5.0, "lon": -21.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) == 1


def test_model_metadata(client: TestClient) -> None:
    resp = client.get("/model")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("model_version", "input_dim", "feature_cols"):
        assert key in body


def test_index_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "SeismicML" in resp.text
