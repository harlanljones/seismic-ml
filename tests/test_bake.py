"""Phase C bake tests (src/serve/bake.py).

CPU-only, synthetic catalog injected so no network and no USGS_LIVE_FETCH are
required. Mirrors the CUDA-disabling guard used across the suite.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.serve import bake


def _make_catalog(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2021-01-01")
    for _ in range(700):
        offset = int(rng.integers(0, 70))
        rows.append(
            {
                "time": base
                + pd.Timedelta(days=offset)
                + pd.Timedelta(seconds=int(rng.integers(0, 86400))),
                "lat": float(rng.choice([4.0, 6.0, 8.0, 10.0])),
                "lon": float(rng.choice([-22.0, -20.0, -18.0])),
                "depth": float(rng.normal(10.0, 2.0)),
                "mag": float(rng.uniform(2.5, 6.0)),
                "gap": float(rng.normal(100.0, 10.0)),
                "dmin": float(rng.uniform(0.5, 2.0)),
                "sig": float(rng.normal(50.0, 5.0)),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def fake_artifacts(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(config, "EPOCHS", 1)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    return tmp_path


def test_bake_synthetic_emits_baked_js(
    fake_artifacts: Path, tmp_path: Path
) -> None:
    catalog = _make_catalog()
    out = tmp_path / "baked.js"
    result = bake.bake(output_path=out, catalog=catalog)

    assert result == out
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "window.SEISMIC_BAKED" in text

    start = text.index("{")
    payload = json.loads(text[start : text.rindex(";")])
    for key in ("manifest", "forecast", "receipts", "events", "model", "history"):
        assert key in payload

    receipts = payload["receipts"]
    assert set(receipts["confusion"]) == {"tp", "fp", "fn", "tn"}
    assert "markers" in receipts


def test_bake_refuses_network_without_env(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="USGS_LIVE_FETCH"):
        bake.bake(output_path=tmp_path / "baked.js", catalog=None)


def test_receipts_confusion_matches_markers(fake_artifacts: Path, tmp_path: Path) -> None:
    from src.data.features import build_spatiotemporal_grid

    catalog = _make_catalog()
    grid = build_spatiotemporal_grid(
        catalog, grid_size=config.GRID_SIZE_DEG, time_step_days=config.TIME_STEP_DAYS
    )
    bundle_root = config.ARTIFACTS_DIR / "model_bundle"
    from src.model.train import train_and_persist_bundle

    bundle = train_and_persist_bundle(grid, bundle_root)
    receipts = bake._receipts_payload(grid, bundle)

    c = receipts["confusion"]
    # TN+FN+FP+TP should equal the test partition length (exact slice math,
    # not a 15% approximation).
    from src.data.splits import walk_forward_split

    _, _, te = walk_forward_split(len(grid))
    n_test = len(grid.iloc[te])
    assert c["fp"] + c["tp"] == sum(
        1 for m in receipts["markers"] if m["gt"] == 0 and m["pred"] == 1
    ) + sum(
        1 for m in receipts["markers"] if m["gt"] == 1 and m["pred"] == 1
    )
    assert c["tn"] + c["tp"] + c["fn"] + c["fp"] == n_test