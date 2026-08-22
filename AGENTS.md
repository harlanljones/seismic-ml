# AGENTS.md — SeismicML Agent Operating Manual

> **STATUS: Project COMPLETE (2026-08-21).** All gates passed; see `ROADMAP.md` §5. This manual is retained as the operating model for future sessions extending this codebase.

This file governs every agent session (human, orchestrator, or subagent) working in this repository. The engineering source of truth is [`TDD.md`](./TDD.md). The execution plan and workstream decomposition live in [`ROADMAP.md`](./ROADMAP.md). **Read both before writing any code.**

---

## 1. Project Context (30-second orientation)

- **What:** ML seismic prediction pipeline — spatiotemporal binary classification forecasting M ≥ 4.5 earthquakes within a 7-day window from 30-day precursor features.
- **Hardware target:** NVIDIA RTX 4070 Ti (12GB VRAM, Ada Lovelace, CC 8.9), TensorFlow 2.x / Keras, CUDA 12.x, cuDNN 8.9+, WSL2 / Linux.
- **Deliverables:** trained model with train/val/test convergence curves + interactive Folium geospatial overlay map.
- **Reference implementation:** `TDD.md` §4 contains the complete working prototype. Module contracts in `ROADMAP.md` are derived directly from it — do not deviate without an explicit contract change.

## 2. Non-Negotiable Invariants

Violating any of these is a failed review regardless of test status:

1. **No temporal data leakage.** Splits must be chronological walk-forward (70/15/15): `max(T_train) < min(T_val) < min(T_test)`. Never shuffle across split boundaries. `StandardScaler` fits on train only; val/test are transformed only.
2. **Mixed precision discipline.** Global policy `mixed_float16`; the final classification head MUST declare `dtype="float32"`. Loss must remain finite (no NaN).
3. **Tensor Core alignment.** Dense layer units in multiples of 64 (128/64/64 per TDD).
4. **VRAM ceiling ≤ 6.5 GB** at batch size 128. Memory growth enabled before any other TF op. Never call `tf.config.experimental.set_memory_growth` after GPU initialization.
5. **XLA enabled:** `jit_compile=True` in `model.compile()`. Zero compilation fallbacks permitted.
6. **Deterministic interfaces.** Function signatures and DataFrame schemas are frozen contracts (see `ROADMAP.md` §4). Changes require updating the contract first, then all dependents.

## 3. Repository Layout & File Ownership

```
seismic-ml/
├── AGENTS.md              # this file
├── ROADMAP.md             # execution plan, workstreams, contracts, gates
├── TDD.md                 # technical design document (source of truth)
├── requirements.txt       # W1
├── src/
│   ├── __init__.py
│   ├── config.py          # W1: constants, GPU init, paths        [owner: agent-w1]
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch.py       # W2: USGS catalog ingestion           [owner: agent-w2]
│   │   ├── features.py    # W3: spatiotemporal grid features     [owner: agent-w3]
│   │   └── splits.py      # W3: walk-forward split + scaling     [owner: agent-w3]
│   ├── pipeline/
│   │   └── datasets.py    # W4: tf.data input pipelines          [owner: agent-w4]
│   ├── model/
│   │   ├── arch.py        # W5: Keras architecture               [owner: agent-w5]
│   │   ├── tracker.py     # W5: PartitionMetricTracker callback  [owner: agent-w5]
│   │   └── train.py       # W5: training entry point             [owner: agent-w5]
│   └── viz/
│       ├── metrics.py     # W6: convergence plotting             [owner: agent-w6]
│       └── map.py         # W7: Folium overlay map               [owner: agent-w7]
├── tests/
│   ├── test_features.py   # leakage + schema checks              [owner: agent-w3]
│   ├── test_datasets.py                                          [owner: agent-w4]
│   └── test_arch.py       # dtype policy, layer dims, XLA        [owner: agent-w5]
├── artifacts/             # generated: models, history JSON, plots, maps (gitignored)
└── notebooks/             # scratch only; nothing here is imported by src/
```

**Ownership rule:** an agent writes ONLY inside its owned files plus its own tests. If you need a change outside your ownership, stop and report to the orchestrator instead of editing. This makes parallel execution conflict-free by construction.

## 4. Parallel-Agent Protocol

### 4.1 Roles

- **Orchestrator** (main session): owns `ROADMAP.md`, resolves contract disputes, runs gate checkpoints (`ROADMAP.md` §5), performs merges/integration. Does not implement workstream code itself.
- **Subagents** (one per workstream W1–W7): implement their owned modules against frozen contracts, run their own verification commands, report completion with evidence.

### 4.2 Subagent prompt template

Every subagent prompt MUST contain:

1. Pointers: "Read AGENTS.md §2 invariants, ROADMAP.md §4 contracts for workstream Wx."
2. Scope: the exact file list it owns.
3. The frozen contract text relevant to its inputs/outputs (do not rely on the agent finding it).
4. Verification command(s) it must pass before reporting done.
5. Explicit statement: "Do not modify files outside your scope; report blockers to the orchestrator."

### 4.3 Concurrency rules

- Contracts are written and frozen at Gate G0 BEFORE any implementation agents spawn.
- Work within a wave (see `ROADMAP.md` §5 DAG) proceeds fully in parallel — no shared writable files exist between concurrent agents.
- Integration happens only at gates, executed by the orchestrator on the merged tree.
- Any cross-cutting discovery (e.g., a needed schema change) halts that dependency chain and goes back through the orchestrator for a contract amendment — never a silent local edit.

## 5. Verification Requirements

An agent reports "done" only after ALL of these pass on its own changes:

| Check | Command | Applies to |
| --- | --- | --- |
| Lint | `ruff check src tests` | all |
| Type check | `mypy src` (lenient config in `mypy.ini`) | all |
| Unit tests | `python -m pytest tests/ -q` (bare `pytest` misses repo root on sys.path) | all |
| Schema conformance | pytest markers in `tests/test_features.py` | W2, W3 |
| Dtype/XLA assertions | `tests/test_arch.py::test_mixed_precision_contract`, `test_layer_alignment` | W5 |

GPU-dependent tests (VRAM ceiling, step timing) run only at gates G1/G2 by the orchestrator, never inside subagent loops.

### 5.1 Verified environment notes (2026-08-21)

- All Python commands assume `source .venv/bin/activate`. The activate script carries a hook exporting the pip-installed NVIDIA CUDA/cuDNN lib dirs onto `LD_LIBRARY_PATH`; without it TensorFlow silently registers no GPU on this driver stack (610.57 / CUDA UMD 13.3).
- Resolved stack: Python 3.12, TF 2.21 (`tensorflow[and-cuda]`), pandas **3.0** (newer than the TDD reference era — conform to contract semantics, not pandas-2 idioms), Keras 3.
- CPU-safe test pattern: set `os.environ["CUDA_VISIBLE_DEVICES"]="-1"` BEFORE importing tensorflow (see `tests/test_datasets.py`, `tests/test_arch.py`).
- Stub packages for mypy live in `requirements-dev.txt`; sklearn needs the `mypy.ini` ignore override.

## 6. Definition of Done (project level)

All gates passed per `TDD.md` §5 acceptance table: VRAM ≤ 6.5 GB, no XLA fallbacks, finite loss, leakage-free splits verified by test, |Acc_train − Acc_val| ≤ 0.05, TP clusters intersecting plate boundaries on the Folium map.
