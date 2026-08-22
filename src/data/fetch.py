"""USGS FDSN catalog ingestion for SeismicML."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from src.config import END_DATE, MIN_MAG, START_DATE

USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
CHUNK_DAYS = 30
MIN_CHUNK_DAYS = 1.0
FLOAT_COLS = ["lon", "lat", "depth", "mag", "gap", "dmin", "sig"]


def _fetch_window(
    start: datetime,
    end: datetime,
    min_mag: float,
    records: list[dict[str, object]],
) -> None:
    """Fetch one sub-window, recursively halving on HTTP 400 over-limit."""
    params: dict[str, str | float] = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_mag,
    }
    response = requests.get(USGS_QUERY_URL, params=params, timeout=45)
    if response.status_code == 400:
        span = (end - start).total_seconds() / 86400
        if span <= MIN_CHUNK_DAYS:
            raise RuntimeError(
                f"USGS query failed even at minimum chunk ({span:.2f} days): "
                f"{response.text[:200]}"
            )
        mid = start + (end - start) / 2
        _fetch_window(start, mid, min_mag, records)
        _fetch_window(mid, end, min_mag, records)
        return
    if response.status_code != 200:
        raise RuntimeError(
            f"USGS query failed with HTTP status {response.status_code}: "
            f"{response.text[:200]}"
        )
    records.extend(_parse_features(response.json()))


def _parse_features(payload: dict[str, Any]) -> list[dict[str, object]]:
    """Convert GeoJSON features to flat record dicts, dropping null depth/mag."""
    records: list[dict[str, object]] = []
    for feature in payload["features"]:
        coords = feature["geometry"]["coordinates"]
        props = feature["properties"]
        if coords[2] is None or props["mag"] is None:
            continue
        records.append(
            {
                "time": props["time"],
                "lon": float(coords[0]),
                "lat": float(coords[1]),
                "depth": float(coords[2]),
                "mag": float(props["mag"]),
                "gap": float(props.get("gap") or 180.0),
                "dmin": float(props.get("dmin") or 1.0),
                "sig": float(props.get("sig") or 0),
            }
        )
    return records


def _postprocess(records: list[dict[str, object]]) -> pd.DataFrame:
    """Build the catalog DataFrame exactly once from all chunk records."""
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(
            {"time": pd.Series(dtype="datetime64[ns]")}
            | {col: pd.Series(dtype="float64") for col in FLOAT_COLS}
        )
    df["time"] = pd.to_datetime(df["time"], unit="ms").astype("datetime64[ns]")
    return (
        df.astype({col: "float64" for col in FLOAT_COLS})
        .drop_duplicates()  # adjacent windows share their boundary instant
        .sort_values("time")
        .reset_index(drop=True)
    )


def fetch_seismic_catalog(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    min_mag: float = MIN_MAG,
) -> pd.DataFrame:
    """Pull earthquake catalog records from the USGS FDSN event API.

    The date range is fetched in CHUNK_DAYS-day slices so no single query
    exceeds the USGS 20,000-event cap; any slice rejected with HTTP 400 is
    recursively halved. Returns a DataFrame with columns [time, lon, lat,
    depth, mag, gap, dmin, sig], sorted by time ascending with a reset
    index. Rows with null depth or null magnitude are dropped; gap/dmin/
    sig fall back to 180.0 / 1.0 / 0 respectively when absent.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    records: list[dict[str, object]] = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + timedelta(days=CHUNK_DAYS), end)
        _fetch_window(cursor, window_end, min_mag, records)
        cursor = window_end
    return _postprocess(records)
