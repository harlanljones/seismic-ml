# HJ-492: Live app — Record ratified decisions in roadmap doc

## Changes

Updates `docs/LIVE_APP_ROADMAP.md` to reflect the implemented design:

1. **§0 Status** — marked all phases **Completed** with a retroactive note, per the actual codebase state.
2. **§10 Ratified decisions** — merged the duplicate second list into the authoritative 8-item list, adding the git-workflow convention as item #8.
3. **Cross-reference consistency** — verified `predict` output columns (`lat_bin, lon_bin, period, prob`) match across §3 Phase A, §4.1, and §10 ratified decisions.

## Verification

- `predict` columns consistent: line 106, line 248, line 411 all show `lat_bin, lon_bin, period, prob`.
- §5 decisions (host topology, map refactor, walk-forward split, publish rules, frontend, bundle versioning) all match the ratified list in §10.
- No sections rewritten — only targeted edits to §0 and §10.

## Files touched

- `docs/LIVE_APP_ROADMAP.md` (only file)