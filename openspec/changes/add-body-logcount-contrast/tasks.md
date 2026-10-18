# Tasks: G.0-A — the corrected model's body contrast on log-compressed counts

Phase 9 G.0-A (roadmap v4.5 § Phase 9; tracker `phase9-tracking` G.0-A), after
[Logbook 099](../../../docs/experiments/logbooks/099-logcount-contrast.md) read `no_wiring_effect` on the
point worm. The maintainer's decisions of 2026-10-19: the hybrid horizon, 128 seeds, frozen floors on 16
seeds, and no count-weight arm.

- [ ] 1. **Configs** (design Decision A): `scripts/campaigns/generate_body_logcount_configs.py` derives
  092's four body configs with `connectome_source: emmons_2024_hermaphrodite` and
  `gap_weight_mode: log_count`, and never overwrites. Loader and build tests cover them.
- [ ] 2. **Harness** `scripts/analysis/body_logcount_contrast.py` (Decisions B, D and F). It provides
  `STEMS` and the `plan`, `preflight`, `table` and `score` subcommands, reads the primary and the gates
  on episodes 1–3,000, and reports the late windows and the collapse statistic beside. Tests cover the
  window, every verdict row, the gates, the refusals and seed disjointness.
- [ ] 3. **Before launch** (Decisions C, D and E):
  - smokes on 9899–9900 at 20 episodes;
  - the gate pilot on 9895–9898, configured as the panel;
  - `preflight.json` on the pilot;
  - `table.json` on the 128 panel draws;
  - the cost from the pilot's run records;
  - `/nematode-review-spec add-body-logcount-contrast`, with findings fixed or recorded.
- [ ] 4. **Registration**: `docs/experiments/logbooks/supporting/100-body-logcount/launch.md`, committed
  before any panel run, with the arms, seeds, window, reading, minimum, verdict map, joint outcomes,
  operating characteristic, gates, audit, cost, launch commands and retention. The seed-band table gains
  4001–4128, 9895–9898 and 9899–9900.
- [ ] 5. **The panel** from a separate worktree: 288 runs with the output controls. Every learning run
  must reach 6,000 episodes.
- [ ] 6. **Readout**: Logbook 100. It gives the verdict, the late window and collapse beside it, the
  STOP branch's status with the point worm, and the substrate limit. Update the tracker's G.0-A item,
  the roadmap's G.0 entry and Go/No-Go, and the CHANGELOG. The decomposition restatement registers next.
- [ ] 7. **Close-out**: full suite, `pre-commit run -a`, `openspec validate --strict`, archive, PR.
