"""Scheduled retrain + safe promotion gate for SeismicML (Phase D).

Runs a periodic refit that produces a CHALLENGER bundle and promotes it ONLY
if it beats the incumbent on a safe gate, via a ``latest`` symlink. The trainer
itself (:func:`src.model.train.train_and_persist_bundle`) preserves every
``AGENTS.md`` §2 invariant (leakage-free walk-forward split, mixed_float16,
float32 head, Tensor-Core-aligned widths, XLA JIT, VRAM ceiling) — this module
never alters those; it only orchestrates fetch/grid and the promotion pointer.

Network fetches mirror ``src/jobs/nowcast.py``: gated behind ``USGS_LIVE_FETCH``
so CI never reaches the network.

Contract (LIVE_APP_ROADMAP §3 Phase D / §5.3.3):
    scheduled_retrain(bundle_root, days_lookback=1095, catalog=None) -> run_id
    promote_if_better(challenger_id, bundle_root) -> bool
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src import config
from src.data.features import build_spatiotemporal_grid
from src.data.fetch import fetch_seismic_catalog
from src.model.train import train_and_persist_bundle

if TYPE_CHECKING:
    from src.model.train import ModelBundle

_LATEST_LINK = "latest"


def _live_fetch_enabled() -> bool:
    value = os.environ.get("USGS_LIVE_FETCH", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_metadata(bundle_dir: Path) -> dict[str, Any]:
    with open(bundle_dir / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def scheduled_retrain(
    bundle_root: str | Path,
    days_lookback: int = 1095,
    catalog: pd.DataFrame | None = None,
) -> str:
    """Fetch (or reuse) a catalog, build the grid, and persist a CHALLENGER bundle.

    Returns the persisted ``run_id``. The challenger is NOT promoted here —
    call :func:`promote_if_better` separately.

    If ``catalog`` is None: only call :func:`fetch_seismic_catalog` when the
    ``USGS_LIVE_FETCH`` env var is truthy; otherwise raise so CI can never hit
    the network (mirrors ``nowcast.run_nowcast``).

    Split note (§5.3.1 label-latency): ``build_spatiotemporal_grid`` already
    drops the unobservable trailing period (the final per-cell ``shift(-1)``
    target is NaN -> dropped), so the standard ``walk_forward_split`` test
    partition's labels are fully observable at promotion time. The trainer
    internally uses the default 70/15/15 boundaries — do NOT silently change
    split boundaries here.
    """
    if catalog is None:
        if not _live_fetch_enabled():
            raise RuntimeError(
                "USGS_LIVE_FETCH not set — refusing network fetch in non-live mode"
            )
        end: date = datetime.now(UTC).date()
        start: date = end - timedelta(days=days_lookback)
        catalog = fetch_seismic_catalog(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            min_mag=config.MIN_MAG,
        )

    grid = build_spatiotemporal_grid(
        catalog,
        grid_size=config.GRID_SIZE_DEG,
        time_step_days=config.TIME_STEP_DAYS,
    )

    bundle: ModelBundle = train_and_persist_bundle(grid, Path(bundle_root))
    return bundle["run_id"]


def promote_if_better(challenger_id: str, bundle_root: str | Path) -> bool:
    """Promote ``challenger_id`` to ``latest`` only if it clears the safe gate.

    Gates (all must hold):
      - incumbent exists (bootstrap promotes when none exists), AND
      - challenger["test_auc"] > incumbent["test_auc"], AND
      - challenger["train_val_gap"] <= 0.05, AND
      - challenger test_auc / train_acc / val_acc are all finite.

    On promotion the ``latest`` symlink is repointed atomically (temp symlink +
    ``os.replace``); old bundle directories are never deleted, so rollback is
    just repointing the symlink. Returns ``True`` if promoted, ``False`` if not.
    """
    bundle_root = Path(bundle_root)
    latest_link = bundle_root / _LATEST_LINK

    incumbent_dir: Path | None = None
    if latest_link.exists():
        incumbent_dir = Path(os.path.realpath(latest_link))

    challenger_dir = bundle_root / challenger_id
    challenger = _read_metadata(challenger_dir)

    def _finite_eval(meta: dict[str, Any]) -> bool:
        return all(
            math.isfinite(float(meta[k]))
            for k in ("test_auc", "train_acc", "val_acc")
        )

    if incumbent_dir is None:
        promote = True  # bootstrap: nothing to beat
    else:
        incumbent = _read_metadata(incumbent_dir)
        promote = (
            _finite_eval(challenger)
            and float(challenger["test_auc"]) > float(incumbent["test_auc"])
            and float(challenger["train_val_gap"]) <= 0.05
        )

    if not promote:
        return False

    tmp_link = bundle_root / f".{_LATEST_LINK}.tmp.{os.getpid()}"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    os.symlink(challenger_dir, tmp_link)
    os.replace(tmp_link, latest_link)
    return True


if __name__ == "__main__":
    root = config.ARTIFACTS_DIR / "model_bundle"
    run_id = scheduled_retrain(root)
    print(f"Challenger bundle: {run_id}")
    promoted = promote_if_better(run_id, root)
    print(f"Promoted: {promoted}")
