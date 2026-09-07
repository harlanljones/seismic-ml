# ft/hj-seismic-combined — consolidated changes from HJ-488…HJ-492

## Summary
All uncommitted changes from the five `ft/HJ-*` worktrees for the Seismic ML
"live app" tickets, moved onto a single branch for one-pass review. Nothing is
committed — everything below is uncommitted working-tree state, ready for you
to review and commit.

Note: the *code* for HJ-487…HJ-491 (inference bundle, nowcast job, serving
surface, split amendment, retrain lifecycle) is already on `main` (tickets are
Done). The only surviving code change is HJ-492's roadmap edit; the other
worktrees held only their PR descriptions, collected here renamed per ticket.

## Changes
- `docs/LIVE_APP_ROADMAP.md` — HJ-492: records the ratified decisions
  (7 insertions, 15 deletions) in the live-app roadmap doc.
- `PR_DESCRIPTION-HJ-488.md` — Phase B daily nowcast job description (from worktree, unchanged).
- `PR_DESCRIPTION-HJ-489.md` — Phase C serving surface description.
- `PR_DESCRIPTION-HJ-490.md` — walk_forward_split contract amendment description.
- `PR_DESCRIPTION-HJ-491.md` — Phase D retrain lifecycle & promotion gate description.
- `PR_DESCRIPTION-HJ-492.md` — HJ-492's own description (identical content also
  kept at `PR_DESCRIPTION-HJ-492.md` for traceability).

## Testing
- No executable code changed; `docs/` + markdown only.
- Sanity: `git diff --stat` shows exactly the one file above.

## Notes
- Source worktrees (`~/dev/ft/HJ-488…HJ-492`) are now fully superseded by this
  branch; safe to remove with `git worktree remove` + `git branch -D` after you
  commit this branch.
