"""W3 contract tests: grid features, target semantics, leakage, scaling."""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS, TARGET_MAG
from src.data.features import FINAL_COLS, build_spatiotemporal_grid
from src.data.splits import scale_splits, walk_forward_split


def make_catalog(rows: list[dict]) -> pd.DataFrame:
    base = {"depth": 10.0, "mag": 3.0, "gap": 100.0, "dmin": 1.0, "sig": 50.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def week(start_day: int) -> pd.Timestamp:
    return pd.Timestamp("2021-01-04") + pd.Timedelta(days=start_day)


def multi_period_catalog(n_periods: int = 12) -> pd.DataFrame:
    rows = []
    for p in range(n_periods):
        for cell in ((5.5, -20.3), (12.1, 40.9)):
            rows.append(
                {
                    "time": week(7 * p),
                    "lat": cell[0],
                    "lon": cell[1],
                    "depth": 10.0 + p,
                    "mag": 3.0 + (p % 4) * 0.2,
                }
            )
    return make_catalog(rows)


class TestSchemaConformance:
    def test_exact_column_order(self):
        df = make_catalog(
            [
                {"time": week(0), "lat": 5.5, "lon": -20.3},
                {"time": week(7), "lat": 5.6, "lon": -20.1},
            ]
        )
        grid = build_spatiotemporal_grid(df)
        assert list(grid.columns) == FINAL_COLS

    def test_dtypes(self):
        df = multi_period_catalog()
        grid = build_spatiotemporal_grid(df)
        assert str(grid["lat_bin"].dtype) == "float64"
        assert str(grid["lon_bin"].dtype) == "float64"
        assert isinstance(grid["period"].dtype, pd.PeriodDtype)
        assert str(grid["event_count"].dtype) == "int64"
        for col in ("mean_depth", "max_mag", "mean_gap", "mean_sig"):
            assert str(grid[col].dtype) == "float64", col
        assert str(grid["target"].dtype) == "int64"

    def test_bin_edges_use_floor_division(self):
        df = make_catalog(
            [
                {"time": week(0), "lat": 5.99, "lon": -20.01},
                {"time": week(0), "lat": 4.01, "lon": -21.99},
            ]
        )
        grid = build_spatiotemporal_grid(df)
        assert set(grid["lat_bin"]) == {4.0}
        assert set(grid["lon_bin"]) == {-22.0}

    def test_aggregations(self):
        df = make_catalog(
            [
                {"time": week(0), "lat": 5.5, "lon": -20.3, "depth": 8.0, "mag": 3.0},
                {"time": week(1), "lat": 5.7, "lon": -20.1, "depth": 12.0, "mag": 4.0},
                {"time": week(2), "lat": 5.2, "lon": -20.6, "depth": 16.0, "mag": 5.0},
            ]
        )
        grid = build_spatiotemporal_grid(df)
        row = grid.iloc[0]
        assert row["event_count"] == 1
        assert row["mean_depth"] == pytest.approx(8.0)
        assert row["max_mag"] == pytest.approx(3.0)


class TestTargetCorrectness:
    def test_target_one_when_next_period_has_large_quake(self):
        # Period A: mag 3.x; period B: mag 4.8 >= threshold.
        df = make_catalog(
            [
                {"time": week(0), "lat": 5.5, "lon": -20.3, "mag": 3.1},
                {"time": week(7), "lat": 5.5, "lon": -20.3, "mag": 4.8},
            ]
        )
        grid = build_spatiotemporal_grid(df)
        assert len(grid) == 2
        by_period = dict(zip(grid["period"], grid["target"]))
        first, last = sorted(by_period)
        assert by_period[first] == 1
        # Last period per cell: shift(-1) yields NaN; NaN >= 4.5 is False,
        # so the row SURVIVES dropna() with target == 0 (frozen behavior).
        assert by_period[last] == 0

    def test_target_zero_when_next_period_below_threshold(self):
        df = make_catalog(
            [
                {"time": week(0), "lat": 5.5, "lon": -20.3, "mag": 3.1},
                {"time": week(7), "lat": 5.5, "lon": -20.3, "mag": 4.4},
            ]
        )
        grid = build_spatiotemporal_grid(df)
        assert grid["target"].tolist() == [0, 0]

    def test_threshold_is_exclusive_below_target_mag(self):
        df = make_catalog(
            [
                {"time": week(0), "lat": 5.5, "lon": -20.3, "mag": 3.1},
                {"time": week(7), "lat": 5.5, "lon": -20.3, "mag": TARGET_MAG},
            ]
        )
        grid = build_spatiotemporal_grid(df)
        assert grid.sort_values("period")["target"].iloc[0] == 1

    def test_cells_do_not_leak_targets_into_each_other(self):
        # Cell 1 gets a big quake in period B; cell 2 never does.
        df = make_catalog(
            [
                {"time": week(0), "lat": 5.5, "lon": -20.3, "mag": 3.0},
                {"time": week(7), "lat": 5.5, "lon": -20.3, "mag": 4.8},
                {"time": week(0), "lat": 30.5, "lon": 10.3, "mag": 3.0},
                {"time": week(7), "lat": 30.5, "lon": 10.3, "mag": 3.5},
            ]
        )
        grid = build_spatiotemporal_grid(df)
        cell_big = grid[(grid["lat_bin"] == 4.0)].sort_values("period")
        cell_quiet = grid[(grid["lat_bin"] == 30.0)].sort_values("period")
        assert cell_big["target"].tolist() == [1, 0]
        assert cell_quiet["target"].tolist() == [0, 0]


class TestLeakageFreeSplits:
    def test_temporal_ordering_across_splits(self):
        # 20 periods x 2 cells = 40 rows -> split boundaries land on
        # even indices, i.e., between whole periods (no straddled bins).
        grid = build_spatiotemporal_grid(multi_period_catalog(n_periods=20))
        assert len(grid) == 40
        assert grid["period"].nunique() >= 10
        train, val, test = walk_forward_split(len(grid))

        t_train = grid["period"].iloc[train]
        t_val = grid["period"].iloc[val]
        t_test = grid["period"].iloc[test]

        assert t_train.max() < t_val.min()
        assert t_val.max() < t_test.min()

    def test_split_slices_are_contiguous_and_complete(self):
        n = 100
        train, val, test = walk_forward_split(n)
        assert (train.start, train.stop) == (0, 70)
        assert (val.start, val.stop) == (70, 85)
        assert (test.start, test.stop) == (85, 100)

    def test_no_shuffle_independence(self):
        # Same input twice -> identical split boundaries (deterministic).
        a = walk_forward_split(57)
        b = walk_forward_split(57)
        assert a == b
        assert (a[0].stop, a[1].stop, a[2].stop) == (
            int(57 * 0.70),
            int(57 * 0.85),
            57,
        )


class TestScalingDiscipline:
    def test_scaler_fit_on_train_only(self):
        rng = np.random.default_rng(42)
        X_train = rng.exponential(scale=50.0, size=(200, len(FEATURE_COLS)))
        offset = np.arange(1, len(FEATURE_COLS) + 1) * 150.0
        X_val = X_train[:60] + offset
        X_test = X_train[:45] + 3 * offset

        Xtr, Xv, Xte, scaler = scale_splits(X_train, X_val, X_test)

        # Train is standardized to ~zero mean, unit variance.
        assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(Xtr.std(axis=0), 1.0, atol=1e-10)

        # Val/test were NOT refit: their means stay far from zero.
        assert np.all(np.abs(Xv.mean(axis=0)) > 1.0)
        assert np.all(np.abs(Xte.mean(axis=0)) > 1.0)

        # Scaler statistics come from the TRAIN partition only.
        assert np.allclose(scaler.mean_, X_train.mean(axis=0))
        assert np.allclose(scaler.scale_, X_train.std(axis=0))

        for arr in (Xtr, Xv, Xte):
            assert not np.isnan(arr).any()
