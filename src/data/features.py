"""Spatiotemporal grid feature construction (W3)."""

import pandas as pd

from src.config import TARGET_MAG

FINAL_COLS: list[str] = [
    "lat_bin",
    "lon_bin",
    "period",
    "event_count",
    "mean_depth",
    "max_mag",
    "mean_gap",
    "mean_sig",
    "target",
]


def build_spatiotemporal_grid(
    df: pd.DataFrame,
    grid_size: float = 2.0,
    time_step_days: int = 7,
) -> pd.DataFrame:
    """Aggregate an event catalog into a (lat_bin, lon_bin, period) grid.

    Input schema: ROADMAP.md §4.1 (time, lon, lat, depth, mag, gap, dmin, sig).
    Output schema: ROADMAP.md §4.2.
    """
    work = df.copy()
    work["lat_bin"] = (work["lat"] // grid_size) * grid_size
    work["lon_bin"] = (work["lon"] // grid_size) * grid_size
    work["period"] = work["time"].dt.to_period(f"{time_step_days}D")

    grid = (
        work.groupby(["lat_bin", "lon_bin", "period"])
        .agg(
            event_count=("mag", "count"),
            mean_depth=("depth", "mean"),
            max_mag=("mag", "max"),
            mean_gap=("gap", "mean"),
            mean_sig=("sig", "mean"),
        )
        .reset_index()
    )

    # shift(-1): NaN on the last period per cell; NaN >= TARGET_MAG -> False,
    # so the final period survives dropna() with target == 0 (contract semantics).
    grid["target"] = (
        grid.groupby(["lat_bin", "lon_bin"])["max_mag"].shift(-1) >= TARGET_MAG
    ).astype(int)

    grid = grid.dropna().sort_values("period", kind="stable").reset_index(drop=True)
    return grid[FINAL_COLS]
