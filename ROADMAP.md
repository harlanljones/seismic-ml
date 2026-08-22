# ROADMAP.md — SeismicML Execution Plan & Workstream Contracts

> **STATUS: COMPLETE (2026-08-21).** All workstreams delivered and all gates passed — see §5 for gate evidence. This document is retained as the record of the parallel-agent execution model and frozen contracts.

Companion to [`AGENTS.md`](./AGENTS.md) (operating rules) and [`TDD.md`](./TDD.md) (engineering source of truth). All contracts below are derived from the reference implementation in `TDD.md` §4 — signatures and schemas are FROZEN at Gate G0.

---

## 1. Target Repository Structure

See `AGENTS.md` §3 for the full tree with file ownership. Summary of importable modules:

```
src.config          (W1)   src.data.fetch      (W2)
src.data.features   (W3)   src.data.splits     (W3)
src.pipeline.datasets (W4) src.model.arch      (W5)
src.model.tracker   (W5)   src.model.train     (W5)
src.viz.metrics     (W6)   src.viz.map         (W7)
```

Dependency direction is strictly downward: `viz → model → pipeline → data → config`. No upward or sideways imports.

## 2. Shared Constants (owned by W1, consumed by all)

| Constant | Value | Notes |
| --- | --- | --- |
| `START_DATE` / `END_DATE` | `"2021-01-01"` / `"2024-01-01"` | USGS query window |
| `MIN_MAG` | `2.5` | catalog completeness floor |
| `GRID_SIZE_DEG` | `2.0` | spatial bin edge |
| `TIME_STEP_DAYS` | `7` | aggregation period |
| `TARGET_MAG` | `4.5` | forecast threshold |
| `BATCH_SIZE` | `128` | Tensor Core occupancy |
| `EPOCHS` | `50` | training length |
| `LEARNING_RATE` | `2e-3` | Adam |
| `DROPOUT_RATE` | `0.25` | regularization |
| `ARTIFACTS_DIR` | `artifacts/` | models, history JSON, plots, maps |

## 3. Workstreams

| # | Workstream | Owned files | Depends on |
| --- | --- | --- | --- |
| W1 | Config, GPU init, requirements | `requirements.txt`, `src/config.py`, `src/__init__.py`, package `__init__.py`s | none |
| W2 | USGS catalog ingestion | `src/data/fetch.py` | W2 schema only (below) |
| W3 | Grid features + walk-forward splits | `src/data/features.py`, `src/data/splits.py`, `tests/test_features.py` | W1 constants, W2 output schema |
| W4 | tf.data input pipelines | `src/pipeline/datasets.py`, `tests/test_datasets.py` | W1 constants, W3 split schema |
| W5 | Keras architecture, callback, training entry point | `src/model/arch.py`, `src/model/tracker.py`, `src/model/train.py`, `tests/test_arch.py` | W1 constants, W4 dataset spec |
| W6 | Convergence plotting | `src/viz/metrics.py` | tracker records schema (fixture-driven) |
| W7 | Folium overlay map | `src/viz/map.py` | grid schema, model `.predict()`, split index contract |

## 4. Frozen Contracts

### 4.1 Catalog DataFrame schema (W2 → W3/W7)

`fetch_seismic_catalog(start_date: str = "2021-01-01", end_date: str = "2024-01-01", min_mag: float = 2.5) -> pd.DataFrame`

**AMENDMENT (G1-integration, 2026-08-21):** USGS FDSN rejects queries matching >20,000 events with HTTP 400. `fetch_seismic_catalog` MUST internally chunk the requested date range into sub-windows that stay under the cap (e.g., ≤30-day slices; shrink adaptively on HTTP 400), fetch each chunk, concatenate, then apply the shared post-processing below exactly once. Public signature and returned schema unchanged; callers unaffected.

Columns, in order, sorted by `time` ascending, index reset:

| Column | Dtype | Source |
| --- | --- | --- |
| `time` | datetime64[ns] | `properties.time` (ms epoch) via `pd.to_datetime(..., unit="ms")` |
| `lon` | float64 | `geometry.coordinates[0]` |
| `lat` | float64 | `geometry.coordinates[1]` |
| `depth` | float64 | `geometry.coordinates[2]` (rows with null dropped) |
| `mag` | float64 | `properties.mag` (rows with null dropped) |
| `gap` | float64 | `properties.gap`, fallback `180.0` |
| `dmin` | float64 | `properties.dmin`, fallback `1.0` |
| `sig` | float64 | `properties.sig`, fallback `0` |

### 4.2 Grid feature schema (W3 → W4/W5/W7)

`build_spatiotemporal_grid(df: pd.DataFrame, grid_size: float = 2.0, time_step_days: int = 7) -> pd.DataFrame`

Group by `(lat_bin, lon_bin, period)` where `lat_bin = (lat // grid_size) * grid_size` (same for lon), `period = time.dt.to_period(f"{time_step_days}D")`. Aggregations: `event_count=("mag","count")`, `mean_depth=("depth","mean")`, `max_mag=("mag","max")`, `mean_gap=("gap","mean")`, `mean_sig=("sig","mean")`.

Target: `grid["target"] = (grid.groupby(["lat_bin","lon_bin"])["max_mag"].shift(-1) >= 4.5).astype(int)`; drop NaN rows; sort by `period`; reset index.

Final columns: `lat_bin, lon_bin, period, event_count, mean_depth, max_mag, mean_gap, mean_sig, target`.

Feature vector (`FEATURE_COLS`, defined in `config.py`): `["lat_bin", "lon_bin", "event_count", "mean_depth", "mean_gap", "mean_sig"]` → shape `(n_samples, 6)`.

### 4.3 Split contract (W3 → W4/W5/W6/W7)

`walk_forward_split(n_samples: int) -> tuple[slice, slice, slice]`

- Chronological 70/15/15: `train = slice(0, int(n*0.70))`, `val = slice(int(n*0.70), int(n*0.85))`, `test = slice(int(n*0.85), n)`.
- Invariant: temporal ordering strictly preserved; leakage check test asserts `max(T_train) < min(T_val) < min(T_test)` on period timestamps.

Scaling contract: `StandardScaler` instantiated and **fit on train only**; val/test transformed only. Expose as `scale_splits(X_train, X_val, X_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning scaled arrays plus the fitted scaler.

### 4.4 Dataset contract (W4 → W5)

`make_tf_dataset(features: np.ndarray, labels: np.ndarray, is_train: bool = False) -> tf.data.Dataset`

- `from_tensor_slices((features, labels))`; if `is_train`: `shuffle(buffer_size=4096)`; always `.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)`.
- Yield shapes: `(None, 6)` float features, `(None,)` int labels.

### 4.5 Architecture contract (W5)

`build_4070ti_model(input_dim: int) -> tf.keras.Model`

- Input `(input_dim,)` → Dense(128, relu) → BatchNorm → Dropout(0.25) → Dense(64, relu) → Dense(64, relu) → Dense(1, sigmoid, **dtype="float32"**).
- Compile: `Adam(learning_rate=2e-3)`, `loss="binary_crossentropy"`, `metrics=["accuracy", AUC(name="auc")]`, `jit_compile=True`.
- Global policy must be `mixed_float16` at call time (asserted in tests).

### 4.6 Tracker records schema (W5 → W6)

`PartitionMetricTracker(test_dataset: tf.data.Dataset)` — Keras callback. On every `on_epoch_end`, appends to `self.records`:

```
{
  "train_loss": [...], "train_acc": [...],
  "val_loss":   [...], "val_acc":   [...],
  "test_loss":  [...], "test_acc":   [...]
}
```

All six lists have equal length (= epochs completed); values are floats. Test metrics come from `model.evaluate(test_dataset, verbose=0)` each epoch. Persisted as `artifacts/history.json` by the training entry point.

### 4.7 Training entry point contract (W5)

`train(grid_df: pd.DataFrame) -> tuple[tf.keras.Model, PartitionMetricTracker]`

Orchestrates §4.3 scaling → §4.4 datasets → §4.5 model → `fit(epochs=50, callbacks=[tracker], validation_data=val_ds)`; saves model weights + `artifacts/history.json`.

### 4.8 Plotting contract (W6)

`plot_convergence_curves(metric_tracker, output_path: str = "artifacts/convergence.png") -> matplotlib.figure.Figure`

Two side-by-side axes (15×5): loss curves (train solid blue, val dashed orange, test dotted red) and accuracy curves (train green, val purple, test dotted red). Grid alpha 0.3, legends on. Saves to `output_path` AND returns the figure (caller decides whether to `show()`).

### 4.9 Map contract (W7)

`generate_spatial_overlay_map(grid_data: pd.DataFrame, val_idx: int, X_test_scaled: np.ndarray, y_test: np.ndarray, model: tf.keras.Model, output_path: str = "artifacts/seismic_forecast_map.html") -> folium.Map`

Behavior per TDD §4 Phase 4:
- Predict on `X_test_scaled`; test slice = `grid_data.iloc[val_idx:]`.
- Three layers on a CartoDB dark_matter basemap centered on test-slice bin means, zoom 4:
  1. Historical training density HeatMap (`grid_data.iloc[:val_idx]`, blue→cyan gradient),
  2. Predicted-risk HeatMap (orange→red gradient),
  3. Confusion markers for test rows: TP green `#00FF00`, FP orange `#FFA500`, FN red `#FF0000`; TN suppressed. Marker radius `5 + prob*7`; popup shows status label, predicted risk %, ground truth.
- `folium.LayerControl(collapsed=False)`; save to `output_path`; return the map object.

## 5. Execution DAG & Gates

```
G0: Orchestrator freezes this document (contracts above) ──────────────┐
                                                                       │
Wave A (fully parallel, no shared writable files):                     │
  W1 (config/reqs) ──┐                                                 │
  W2 (fetch)        ─┤── G1: integration smoke test                    │
  W3 (features/splits, tests) ──┤    (orchestrator runs merged tree:   │
  W4 (datasets, tests) ─┘       fetch → grid → split → tf.data round- │
                                 trip on synthetic fixture data)       │
  W6 (metrics, fixture records) ─┤                                     │
  W7 (map, stub model + fixture grid) ─────────────────────────────────┤
                                                                       │
Wave B:                                                                │
  W5 (arch/tracker/train, tests) ← needs merged W1+W4 ── G2a:          │
      arch contract tests pass on GPU-less CI (policy/dims/XLA flags)  │
                                                                       │
Integration run (orchestrator, real hardware):                        │
  full train (50 epochs) → artifacts/{model, history.json}             │
                                                                       │
Wave C (parallel):                                                     │
  W6 renders convergence.png from history.json                         │
  W7 renders seismic_forecast_map.html from trained model              │
                                                                       │
G2: Final acceptance audit (TDD.md §5 table) ──────────────────────────┘
```

Gate checklist:

- **G0 — PASSED (2026-08-21):** all contracts in §4 frozen before any implementation agent spawned.
- **G1 — PASSED:** merged Wave A tree passed `ruff check src tests`, `mypy src`, 17 unit tests, and the end-to-end synthetic smoke script (grid → leakage-free split → scaling → tf.data round-trip).
- **G2a — PASSED:** `pytest tests/test_arch.py` green — mixed-float16 policy active, float32 output head, layer units ∈ {128, 64}, `jit_compile=True` present. Full suite at merge: **22 passed**, ruff + mypy clean across 14 source files.
- **G2 — PASSED** with evidence from the integration run (83,107-event live catalog, RTX 4070 Ti):
  - VRAM: ~1.7 GB @ batch 128 (≤ 6.5 GB) ✔
  - XLA: compiled via XLA, no fallbacks; step time ~0.6–0.9 ms (≤ 1.8 ms) ✔
  - Numeric stability: finite loss across all 50 epochs ✔
  - Leakage-free split: asserted by `tests/test_features.py` ✔
  - Overfitting gap: |Acc_train − Acc_val| = 0.0106 (≤ 0.05); final Acc_train/val/test = 0.7581 / 0.7687 / 0.7666 ✔
  - Map calibration: TP/FP/FN markers rendered in `artifacts/seismic_forecast_map.html`; plate-boundary visual confirmation remains a human review item ☐

## 5.1 Amendment Log

| Date | Amendment | Trigger |
| --- | --- | --- |
| 2026-08-21 | §4.1: `fetch_seismic_catalog` must chunk date ranges to stay under the USGS FDSN 20,000-event cap (30-day windows, adaptive halving on HTTP 400) | Integration run failed with HTTP 400 on the full 3-year query (~83k events) |

## 6. Risk Register

| Risk | Mitigation |
| --- | --- |
| USGS API unavailable/rate-limited during W2/G1 | Contracts tested against checked-in GeoJSON fixture; live fetch only at integration |
| Class imbalance (M ≥ 4.5 targets rare) | Do NOT silently add resampling — report to orchestrator; any change requires contract amendment |
| XLA incompatibility with BatchNorm/Dropout | If compile fails, escalate to orchestrator; do not remove `jit_compile=True` locally |
| mixed_float16 NaN underflow | Float32 head already mandated (§4.5); if NaN persists, escalate — optimizer loss-scaling change is a contract amendment |
