# Live App Roadmap — SeismicML

> Companion to [`ROADMAP.md`](../ROADMAP.md) (workstreams W1–W7).
> Scope: evolve the completed offline seismic-forecast pipeline into a **live
> application** — interactive API/web frontend rendering live forecasts **and**
> an automated retraining pipeline.
>
> Ticket: [linear.app/harlanljones/HJ-481](https://linear.app/harlanljones/issue/HJ-481)

## 0. Status

- Phase A: not started — contract amendment in flight (§4.1).
- Phase B: not started (blocked on A).
- Phase C: not started (blocked on A; B for the live data loop).
- Phase D: not started (blocked on C).

---

## 1. Context & Problem Statement

The current pipeline (`TDD.md` §4) is **batch/offline**: it ingests a fixed
catalog window (`START_DATE`–`END_DATE`, currently ending 2024-01-01), builds a
grid (`src/data/features.py`), trains (`src/model/train.py`), and renders a
static Folium map (`src/viz/map.py`). There is **no** serve-time inference path,
**no** data-freshness mechanism, and **no** model-serving surface.

"Live app" requires four capabilities layered on top of the existing modules:

1. **Inference bundle** — the fitted `StandardScaler` and model weights must be
   retrievable at serve time. *(Today `scale_splits()` returns the scaler and
   `train()` discards it as `_scaler`.)*
2. **Live data loop** — a rolling nowcast must refresh features from the live
   USGS catalog on a schedule.
3. **Serving surface** — an API + interactive frontend replacing the static
   HTML map output.
4. **Model lifecycle** — safe automated retraining + promotion + rollback,
   preserving every `AGENTS.md` §2 invariant (leakage-free splits, XLA,
   mixed-precision discipline, VRAM ≤ 6.5 GB).

### Hard constraints that bound the design

- `AGENTS.md` §2 invariants are **non-negotiable**: mixed_float16 global policy,
  final head `dtype="float32"`, Tensor-Core-aligned widths, `jit_compile=True`,
  VRAM ≤ 6.5 GB @ batch 128, chronological walk-forward splits.
- The prediction horizon is `TIME_STEP_DAYS=7`; model **labels** require a full
  future window to compute (`features.py` `shift(-1)` target). A "nowcast"
  forecast published at period *t* is only verifiable ~7+ days later — this is
  the fundamental label-latency boundary.
- USGS FDSN catalog has a 20,000-event cap per query; revisions are rare but
  possible. Fetch must use delta windows and `_fetch_window`'s recursive
  halving (`src/data/fetch.py`).

---

## 2. Work Breakdown & Module Ownership

New/overlaid modules (in repo style: `src/<area>/<file>.py`) vs. existing
modules reused as-is:

| Module | Role | Owner pattern |
|---|---|---|
| `src/config.py` *(extending)* | Add `LIVE_*` constants (schedule cadence, bundle dir, lookahead horizon) | W1-adjacent |
| `src/data/fetch.py` *(reuse)* | `fetch_seismic_catalog(start, end, min_mag)` already parameterized — caller supplies live window | W2 |
| `src/data/features.py` *(reuse)* | `build_spatiotemporal_grid(df)` on trailing window | W3 |
| `src/data/splits.py` *(reuse for retrain)* | `walk_forward_split`/`scale_splits` — unchanged contract | W3 |
| `src/pipeline/datasets.py` *(reuse for inference)* | `make_tf_dataset` — inference path uses plain batched tensors instead | W4-adjacent |
| `src/inference.py` *(NEW)* | Pure inference: `load_bundle`, `predict(grid_df)`; CPU-safe | New lane |
| `src/serve/app.py` *(NEW)* | FastAPI `GET /forecast`, `GET /health`, `GET /model` | New lane |
| `src/serve/map_server.py` *(NEW)* | Expose layer data as GeoJSON to the frontend | W6-adjacent |
| `src/viz/map.py` *(reuse)* | Layer semantics (train density, pred risk, markers) | W6 |
| `src/model/arch.py` *(reuse)* | `build_4070ti_model` — unchanged | W5 |
| `src/model/train.py` *(extend)* | Add `train_and_persist_bundle()` — saves scaler + weights + metadata | W5 |
| `src/model/train.py` *(reuse)* | Existing `train(grid_df)` preserved (backward compat) | W5 |
| `src/jobs/nowcast.py` *(NEW)* | Scheduled job: fetch → build grid → predict → write artifacts | New lane |
| `src/jobs/retrain.py` *(NEW)* | Scheduled retrain with promotion gate | W5-adjacent |
| `artifacts/model_bundle/` *(NEW layout)* | `weights/`, `scaler.joblib`, `metadata.json` | New |

> **Dependency additions (documented only — `requirements.txt` NOT edited here):**
> the live app requires `fastapi`, `uvicorn`, `jinja2`, and `joblib`
> (`joblib` for scaler persistence, already used by `train_and_persist_bundle`).
> These must be added to `requirements.txt` when Phase A/C land.

> No two concurrent modules during a phase share a writable file, preserving
> `AGENTS.md` §4.3 conflict-free parallelism.

---

## 3. Phases

### Phase A — Serve-time inference foundation ⛳ required first

**Goal:** make a trained model serveable on a fixed feature vector.

**Stories / tasks:**
1. Add bundle persistence to `train.py` — a thin
   `train_and_persist_bundle(grid_df, bundle_root: Path) -> ModelBundle` that
   writes `artifacts/model_bundle/<run_id>/weights/seismic_model.weights.h5`,
   `scaler.joblib` (joblib, sklearn-safe), and `metadata.json`
   (feature_cols, grid_size_deg, input_dim, min_mag, train_window,
   model_version).
2. Add `src/inference.py`:
   ```python
   def load_bundle(bundle_root: Path) -> ModelBundle
   def predict(grid_df: pd.DataFrame) -> pd.DataFrame  # cols: lat_bin, lon_bin, period, prob
   ```
   - Builds `build_4070ti_model(input_dim)` then `load_weights` from the bundle;
     uses the persisted scaler to transform features.
   - **CPU-safe:** `os.environ["CUDA_VISIBLE_DEVICES"]="-1"` BEFORE TF import,
     matching `tests/test_arch.py`; no GPU needed at inference (~20 K params).
   - Input validation: reject any `grid_df` whose feature columns differ from
     persisted `metadata.json["feature_cols"]`.
3. Persist the scaler via `joblib.dump(scaler, ...)`, closing the
   ephemeral-scaler gap (`train()` currently discards `_scaler`).

**Contract amendments** (frozen → must be ratified before coding; §4.1):
new `ModelBundle` dir layout + `metadata.json` JSON schema; new
`load_bundle`/`predict` signatures.

**Verification (Phase A):**
- `tests/test_inference.py`: round-trip predict; scaler persistence; feature
  schema mismatch raises; CPU path returns finite probs.
- `ruff check src tests`, `mypy src`,
  `python -m pytest tests/test_inference.py -q`.

**Exit gate:** `predict(grid_df)` returns finite probabilities matching the
offline `train()` model within float tolerance, no GPU required.

---

### Phase B — Live data loop

**Goal:** refresh a production grid from the live USGS catalog each day.

**Stories:**
1. `fetch_seismic_catalog` already accepts explicit `start_date`/`end_date`;
   no internal change needed — the caller `src/jobs/nowcast.py` supplies the
   live window.
2. Add `src/jobs/nowcast.py::run_nowcast(bundle_path, days_back=30)`:
   - `end = today`, `start = end - days_back`.
   - `catalog = fetch_seismic_catalog(start, end)`.
   - `grid = build_spatiotemporal_grid(catalog)`.
   - `load_bundle(bundle_path).predict(grid)` → write
     `artifacts/forecast/<ts>/forecast.json` (grid + probs) and the HTML map
     via `generate_spatial_overlay_map` (see §5.2 for the refactor note).
3. Scheduling decision in §5.1.

**Verification (Phase B):**
- `tests/test_nowcast.py`: mocked fetch returns a synthetic catalog; job emits
  expected `forecast.json` keys. USGS call gated behind an env var so CI never
  hits the network.
- Offline simulation: replay a synthetic "today" and assert forecast schema.

**Exit gate:** `run_nowcast` produces a valid `forecast.json` from today's
catalog without training; idempotent under re-runs.

---

### Phase C — Serving surface

**Goal:** web API + interactive frontend.

**Stories:**
1. `src/serve/app.py` (FastAPI):
   - `GET /health`.
   - `GET /forecast` (full grid GeoJSON) and `GET /forecast?lat=&lon=`
     (nearest-bin point query).
   - `GET /model` (serve `metadata.json`).
   - Reads the **latest** persisted `forecast.json`; no recompute per request.
 2. Frontend: **Jinja2 server-rendered + Leaflet (CDN)**, served by FastAPI
    (`src/serve/app.py`). Replace `fmap.save(...)` static output with a
    deployable page that calls `/forecast` and renders the three layers (train
    density heatmap, pred risk heatmap, alert markers) via Leaflet — mirror the
    layer semantics in `src/viz/map.py` (radius/gradient/opacity) so the live
    view is faithful to the verified offline artifact.
3. `src/serve/map_server.py`: adapter exposing layer data as GeoJSON
   (heatmap weights + circle-marker feature collections).

**Verification (Phase C):**
- `tests/test_serve.py`: starlette TestClient; `/health` 200; `/forecast`
  returns a FeatureCollection; `/model` returns metadata keys.
- Local manual smoke: open the page over `localhost`.

**Exit gate:** an operator hits `/forecast` on a live server and sees the same
risk/marker semantics the static offline map produced; no training on the
request critical path.

---

### Phase D — Model lifecycle (automated retraining)

**Goal:** periodic refit that cannot corrupt the live model.

**Stories:**
1. `src/jobs/retrain.py::scheduled_retrain(bundle_root, days_lookback=1095)`:
   - Fetch ~3 yr catalog, `build_spatiotemporal_grid`, chronological 70/15/15
     (`walk_forward_split`, `scale_splits` — scaler fit on train only),
     `model.fit` with the existing `PartitionMetricTracker`.
   - Preserve **all** `AGENTS.md` §2 invariants (trainer unchanged).
   - Write a **challenger** bundle `artifacts/model_bundle/<run_id>/`; do **not**
     swap live yet.
2. Promotion gate `promote_if_better(challenger_id)`:
   - Compare challenger vs incumbent on the held-out test partition's AUC.
   - Promote only if challenger wins on AUC **and** `abs(Acc_train − Acc_val)
     ≤ 0.05` **and** `tracker.records` show finite loss across all epochs.
   - Promote = point `artifacts/model_bundle/latest` symlink at the winner;
     never mutate an in-flight bundle directory.
3. Rollback: retain prior-3 bundles for instant symlink rollback; keep last-30
   on cold storage if hosting permits.

**Verification (Phase D):**
- `tests/test_lifecycle.py`: challenger with worse AUC is *not* promoted;
  symlink points to winner; old bundles intact.

**Exit gate:** one cron entry can run `retrain.py` daily; promotion is safe
(no downtime, no leakage regression, AUC not degraded).

---

## 4. Contract Amendments

Frozen API contracts live in `ROADMAP.md` §4; the live app extends them.
Proposed amendments below must be ratified before the relevant phase ships.

### 4.1 Phase A contracts

```jsonc
// artifacts/model_bundle/<run_id>/metadata.json
{
  "model_version": "<git short sha or epoch-tag>",
  "input_dim": 6,               // == len(FEATURE_COLS)
  "feature_cols": ["lat_bin","lon_bin","event_count","mean_depth","mean_gap","mean_sig"],
  "grid_size_deg": 2.0,
  "time_step_days": 7,
  "min_mag": 2.5,
  "target_mag": 4.5,
  "train_window_start": "2021-01-01",
  "train_window_end": "<last catalog date of the training fit>"
}
```

```python
# src/inference.py  (frozen signatures)
from typing import TypedDict
class ModelBundle(TypedDict): ...
def load_bundle(bundle_root: Path) -> ModelBundle
def predict(grid_df: pd.DataFrame) -> pd.DataFrame   # cols: lat_bin, lon_bin, period, prob
```

### 4.2 Phase B–C contracts
- `forecast.json` schema (B produces; C serves).
- `GET /forecast` → `{ "type": "FeatureCollection", "features": [...] }`.

### 4.3 Invariant lock
- Phases must keep `AGENTS.md` §2 intact. Phase D re-asserts each gate in
  `tests/test_lifecycle.py` (no XLA fallback, finite loss, ≤ 0.05 train/val
  gap).

---

## 5. Operational Decisions

### 5.1 Scheduling / host topology — DECISION REQUIRED

| Option | Cadence | Where | Notes |
|---|---|---|---|
| Local host | cron / systemd --user timer | WSL2 + RTX 4070 Ti | GPU available for retrain; not prod traffic |
| VPS (2 vCPU/8 GB) | cron / systemd timer | Headless Linux | CPU-only inference fine (A is CPU-safe) |
| Cloudflare Cron + Workers | `@daily` Cron Triggers | Serverless | Map/weights bundle in R2; state via R2 |
| Fly.io / Render | scheduled deploy | Container host | Easiest public `/forecast` |

**Default — ACCEPTED (2026-08-26):** A+B+D run on the local box via
`systemd --user` timers (GPU reused for retrain); C is served behind
`uvicorn` + Tailscale as an alpha, migrating to a VPS later. Schemas are
frozen, so the later re-arch (e.g. moving C to a VPS) is free — no contract
change required. See §10 "Ratified decisions" for the summary.

### 5.2 Live vs static map
`generate_spatial_overlay_map` writes a **static** Folium HTML (`fmap.save`).
Phase C wants a *dynamic* page reading `/forecast` JSON. Two options:
- Minimal: Phase B also writes a JSON layer payload (small `map.py` change);
  the Phase-C page consumes JSON only; static HTML stays the offline fallback.
- Faithful: refactor `map.py` to extract a "build layer payload" helper into
  `src/serve/map_server.py` returning GeoJSON, with `map.py` as offline caller.

**Recommendation — AUTHORIZED by the orchestrator (2026-08-26):** the faithful
refactor is approved. Extract the layer-payload builder from `src/viz/map.py`
into `src/serve/map_server.py`; `src/viz/map.py` remains the offline caller.
Per `AGENTS.md` §4.3 this still routes through the orchestrator, but the
amendment is now ratified (not pending). See §10 open question #2 (resolved).

### 5.3 Data freshness cut
- Features window: trailing `days_back=30` (per TDD precursors).
- Target: M ≥ `TARGET_MAG` (4.5) within the **next** `TIME_STEP_DAYS` (7).
  A forecast at `now` is only verifiable once `[now, now+7]` completes — encode
  a `verifiable_after` field on each forecast.

#### 5.3.1 `run_nowcast` publish rules (ratified)
`run_nowcast` publishes the **LATEST COMPLETED period only**: the trailing
NaN-target period is dropped by `build_spatiotemporal_grid`, so the served
forecast covers periods whose labels are fully observable. Each emitted record
carries `verifiable_after = period_end + TIME_STEP_DAYS(7)` (the moment the
actual outcome becomes known). Inside the unverifiable window
(`now < verifiable_after`) the frontend renders markers as **"actual pending"**
rather than showing a confirmed hit/miss — preserving the label-latency
boundary (§1) and avoiding false verification in the live view.

#### 5.3.2 Label-latency mitigation (retrain split)
Because outcomes lag 7+ days, the **promotion gate (Phase D) must not use the
latest 7-day window as a hard positive set**. This is now **adopted as a
contract amendment implemented in HJ-490**: `walk_forward_split` is shifted so
the retrain held-out **test window is `t..t+7`** (labels observable at
promotion time), while train/val remain `t-30..t-7` / `t-7..t`. This is where
 the leakage invariant tightens for the live setting (`AGENTS.md` §2.1).

#### 5.3.3 Bundle versioning & timers (ratified)
- **Run identity:** `run_id = "<git_short_sha>-<epoch>"` (uniquely names each
  retrain/promote cycle under `artifacts/model_bundle/`).
- **Promotion pointer:** `artifacts/model_bundle/latest` is a **symlink** to the
  promoted `run_id` directory; live inference/nowcast always resolves `latest`.
  Never mutate an in-flight bundle — only repoint the symlink.
- **Rollback:** the **prior-3** promoted bundles are retained for instant
  symlink rollback (Phase D R3/R4).
- **Scheduling (`systemd --user` timers, local box per §5.1):**
  - `retrain.timer` @ `03:00` local, `After=network-online.target` (GPU reused).
  - `nowcast.timer` @ `06:00` local; **nowcast always consumes `latest`** so a
    promoted challenger is served automatically on the next run.

---

## 6. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Scaler/feature-schema drift between train & serve | Med | High | Persist scaler + `metadata.json`; `predict` validates columns. |
| R2 | USGS 20k cap / rate limit on daily fetch | Med | High | Delta windows + recursive halving (`_fetch_window`); env-gated CI. |
| R3 | Promotion regresses live model (AUC flip-flop) | Med | High | Challenge-first, symlink-promote, retain bundles, held-out AUC gate. |
| R4 | Train/val gap > 0.05 in a challenger (DoD violation) | Low | High | Hard gate in `promote_if_better`; block symlink switch. |
| R5 | Mixed-precision NaN at serve | Low | Med | `predict` casts inputs to float32 explicitly; final head already float32. |
| R6 | Map refactor touches W6-owned `viz/map.py` | Med | Med | Route via orchestrator (§4.3); Phase C ships JSON path first. |
| R7 | VRAM ceiling breach on co-located train+serve (4070 Ti) | Low | High | Serve is CPU-safe (A); retrain runs alone; enforce ≤ 6.5 GB. |
| R8 | Model staleness (catalog distribution shift) | Med | Med | Daily retrain cadence; surface `train_window_end` on `/model`. |

---

## 7. Gates & DAG

```
Phase A (inference bundle)  ── A.1 gate: round-trip predict, scaler persisted
   │
   ├──> Phase B (live data loop)  ── B.1 gate: run_nowcast emits forecast.json
   │         (blocked on A)
   └──> Phase C (serving surface) ── C.1 gate: /forecast GeoJSON, no training
           (blocked on A)              on request path
                │
                └──> Phase D (retrain lifecycle) ── D.1 gate: daily cron safe-
                        (blocked on C + A)          promotes via AUC gate
```

Gate checkpoints mirror `ROADMAP.md` §5 style: each is `ruff` + `mypy` +
`pytest` on the new tests, plus a doc-level acceptance row:

| Gate | Acceptance |
|---|---|
| G_phA | `predict()` finite probs match offline model < 1e-5 mean-abs diff; scaler file present; CPU-safe. |
| G_phB | `forecast.json` schema valid; nowcast idempotent; CI never hits USGS (env-gated). |
| G_phC | `/forecast` valid GeoJSON FeatureCollection; marker colors match offline confusion-matrix semantics. |
| G_phD | Challenger with lower AUC is *not* promoted; prior bundle intact; train/val gap ≤ 0.05 asserted. |

---

## 8. Verification Matrix

Commands run from repo root with the venv active (`python -m pytest`, never
bare `pytest`):

| Check | Command |
|---|---|
| Lint | `ruff check src tests` |
| Type check | `mypy src` (sklearn override per `mypy.ini`) |
| New tests | `python -m pytest tests/test_inference.py tests/test_nowcast.py tests/test_serve.py tests/test_lifecycle.py -q` |
| GPU gates (Phase D) | orchestrator-only: `python -m pytest tests/test_arch.py -m gpu -q` |

> **Requirements (documented only):** the live app adds `fastapi`, `uvicorn`,
> `jinja2`, and `joblib` to `requirements.txt` (Phase A/C). `joblib` backs the
> scaler persistence; `fastapi`/`uvicorn`/`jinja2` back the serving surface.
> `requirements.txt` itself is not modified by this doc ticket.

---

## 9. Out of scope (future tickets)

- Multi-model / ensemble serving.
- Probabilistic calibration (temperature/Platt) on the sigmoid output — the
  `prob` values are raw; a follow-on can add isotonic regression on val.
- Alerting (webhooks/email) on `pred=1 & actual=1` days.
- Plate-boundary TP intersection assertion at serving time (today a static DoD
  check; could become a served diagnostic marker).
- Batch inference endpoint beyond a single grid.

---

## 10. Open questions

### Ratified decisions (2026-08-26)

The following design questions were ratified on 2026-08-26 and recorded in the
sections below — restating them here as the authoritative summary:

1. **`predict` output schema** — columns are `lat_bin, lon_bin, period, prob`
   (§3 and §4.1 now agree; `period` keys `verifiable_after` / heatmap grouping).
2. **Host topology** — default ACCEPTED: local `systemd --user` timers for
   A+B+D (GPU reused), C behind `uvicorn`+Tailscale as alpha → VPS later (§5.1).
3. **W6 map refactor** — AUTHORIZED: extract layer-payload builder into
   `src/serve/map_server.py`; `src/viz/map.py` stays the offline caller (§5.2).
4. **Retrain split** — shifted `walk_forward_split` adopted as HJ-490; retrain
   held-out test window is `t..t+7` (§5.3.2).
5. **Nowcast publish rules** — latest completed period only, with
   `verifiable_after` and "actual pending" markers (§5.3.1).
6. **Frontend** — Jinja2 + Leaflet (CDN) served by FastAPI; `requirements.txt`
   gains `fastapi`, `uvicorn`, `jinja2`, `joblib` (§3 Phase C, §2, §8).
7. **Bundle versioning** — `run_id="<git_short_sha>-<epoch>"`, `latest` symlink,
   prior-3 retained, `retrain.timer`@03:00 / `nowcast.timer`@06:00 (§5.3.3).

---

1. **Host topology** (§5.1) — **RESOLVED**: default ACCEPTED (local
   `systemd --user` timers for A+B+D, C behind `uvicorn`+Tailscale as alpha,
   VPS later). See §5.1 and "Ratified decisions" above.
2. **W6 map refactor** (§5.2) — **RESOLVED (authorized)**: extract the
   layer-payload builder into `src/serve/map_server.py`; `src/viz/map.py`
   remains the offline caller. See §5.2 and "Ratified decisions" above.
3. **git workflow** — **CONFIRMED**: commit to
   `harlanljones/hj-481-develop-a-roadmap-for-a-live-app`, per-phase branches.
