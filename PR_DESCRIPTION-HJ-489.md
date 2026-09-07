# HJ-489: Live app: Phase C — Serving surface (FastAPI + map_server + Leaflet)

## Summary

Serving surface is implemented: FastAPI app serves `GET /health`, `GET /forecast` (GeoJSON FeatureCollection + nearest-bin point query), `GET /model`, and `GET /` Jinja2 page. Layer payload builder extracted from `src/viz/map.py` into `src/serve/map_server.py` (faithful refactor); `map.py` is the offline caller. Frontend in `frontend/` renders train density, pred risk, and verification markers via Leaflet. No training on the request path.

## Changes

- `src/serve/app.py` — `health`, `forecast(lat, lon)`, `model`, `index`; reads latest `forecast.json`, no recompute per request
- `src/serve/map_server.py` — `forecast_to_geojson`, `build_layer_payload`, `_make_markers`; exact radius/gradient/opacity from offline map semantics
- `src/viz/map.py` — refactored to call `build_layer_payload` (offline caller, output unchanged)
- `frontend/index.html`, `frontend/app.js`, `frontend/styles.css` — Leaflet 5-beat tour + dashboard consuming `/forecast`
- `tests/test_serve.py` — starlette TestClient: `/health` 200, `/forecast` FeatureCollection, `/model` metadata keys

## Testing

- `python -m pytest tests/test_serve.py -q` — passed (part of 11 passed with `test_lifecycle.py`)
- Marker colors match offline confusion-matrix semantics (TP/FP/FN)

## Notes

- Depends on HJ-487 (inference bundle) at runtime for live predictions; serves baked `forecast.json` otherwise
- `requirements.txt` already includes `fastapi`, `uvicorn`, `jinja2`
