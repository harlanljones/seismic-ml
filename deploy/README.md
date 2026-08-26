# SeismicML systemd --user deploy units

These reference units wire the Phase D retrain and Phase B nowcast jobs into
`systemd --user` timers on the local box (per LIVE_APP_ROADMAP §5.1 / §5.3.3).
They are **reference units** — they are not installed or enabled automatically.

## Layout

| Unit | Runs | Schedule |
|------|------|----------|
| `retrain.service` | `python -m src.jobs.retrain` (writes a challenger bundle, promotes via the AUC gate) | `retrain.timer` |
| `retrain.timer` | `OnCalendar=*-*-* 03:00`, `After=network-online.target` | daily 03:00 |
| `nowcast.service` | `python -m src.jobs.nowcast` (consumes `artifacts/model_bundle/latest`) | `nowcast.timer` |
| `nowcast.timer` | `OnCalendar=*-*-* 06:00` | daily 06:00 |

The `*.service` files run the relevant module under the project venv python
(`%h/dev/seismic-ml/.venv/bin/python`). Adjust the `WorkingDirectory` /
`ExecStart` paths to match the actual install location.

## Install

```bash
mkdir -p ~/.config/systemd/user
cp deploy/*.service deploy/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now retrain.timer
systemctl --user enable --now nowcast.timer
```

## Notes

- `nowcast` always consumes `artifacts/model_bundle/latest`; a promoted
  challenger is served automatically on the next nowcast run.
- `retrain` only **promotes** when the challenger beats the incumbent on the
  safe gate (higher `test_auc`, `train_val_gap <= 0.05`, finite eval metrics).
  Old bundles are never deleted — rollback is just repointing the `latest`
  symlink.
- Network fetches are gated behind `USGS_LIVE_FETCH`. Set it in the service
  environment (`Environment=USGS_LIVE_FETCH=1` under `[Service]`) for live
  hosts; leave unset in CI so no job ever hits the network.
- These units are `Type=oneshot`; timers use `Persistent=true` so a missed run
  fires on the next boot/login.
