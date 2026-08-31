# SeismicML — Live Forecast Tour

> **Product truth** for the SeismicML interactive web app. Captured 2026-08-30 via the
> shape/grilling interview. Companion to `docs/LIVE_APP_ROADMAP.md`.

## Platform

web — static-first, hybrid. Serves the identical frontend from any static host
(no server required) and auto-upgrades to live data when the FastAPI backend is
present (`/health` probe).

## What this is

A **public, portfolio-grade web app** that makes a real ML seismic-forecast model
legible to a **technically curious stranger** in ~2–3 minutes — and then hands
them the wheel of a full explorable dashboard.

It is a **guided story, not a dashboard-first page**. Five beats on a fixed map
stage, visitor-paced (Next/Back, flying map transitions, no autoplay):

1. **Watch** — the 30-day precursor field: "before a big one, the ground sends signals."
2. **Signal** — zoom one hot cell's precursors (event count, depth, azimuthal gap, signal) and the question posed: *M≥4.5 in the next 7 days?*
3. **Forecast** — the risk heatmap over the 7-day window, threshold-explainable.
4. **Receipts** — the held-out test verification, **radically honest**: the wins *and* the false alarms and misses, same visual weight.
5. **Playground** — "Take the controls" → full dashboard (threshold slider, drill-down, top-10, model card, forecast history).

## Audience & outcome

- **Who:** a technically curious stranger landing on a public URL, no seismic-ML context.
- **Outcome:** they *understand the model* (it reads 30-day precursors and forecasts M≥4.5 in the next 7 days), *trust it appropriately* (the honest receipts beat), and *explore freely* after the tour.
- **Success:** teaches, impresses, proves — and never oversells.

## Uniquely true

The app is backed by a real, verified spatiotemporal forecast pipeline
(`src/`), and every number on the page is real output (test acc 0.7666, TP/FP/FN
from the held-out test partition, live forecast from the USGS catalog). The
honest **label-latency loop** — each forecast is verifiable only ~7 days later —
is surfaced as prominently as the wins.

## Honesty contract (non-negotiable)

- Never oversell. Limitations, label latency, and false alarms are shown with the
  same visual weight as successes.
- The footer carries: "Research prototype — not a public warning system."
- Live forecast markers are **pending** until `verifiable_after`; constituted wins/misses
  come only from the held-out test partition.
- No fabricated metrics, ever.

## Visual world

**Geophysical Observatory** (dark). Near-black stage, CartoDB dark_matter,
glowing risk heat, IBM-Plex-mono telemetry labels, instrument-panel density, the
beat card as an HP-style panel. Established 2026-08-30 (see `.impeccable/DESIGN.md`).
Responsive-first: map stays the hero; tour + dashboard reflow on mobile/tablet.

## Stack

Frontend: vanilla HTML/CSS/JS + Leaflet + leaflet.heat (CDN), **no build step**.
Backend (live mode): FastAPI + uvicorn. Data: baked JSONs from the real pipeline
(`src/serve/bake.py`), auto-upgraded by the live API. User decision: stack
delegated to the incumbent repo (fastapi/uvicorn/jinja2 already in requirements).

## Boundaries

- All `AGENTS.md` §2 invariants preserved (leakage-free splits, mixed precision,
  float32 head, XLA, VRAM ceiling, frozen schemas).
- The frozen `forecast.json` / `metadata.json` / GeoJSON contracts are untouched.
- The app never trains; bake is a separate one-time operator command.