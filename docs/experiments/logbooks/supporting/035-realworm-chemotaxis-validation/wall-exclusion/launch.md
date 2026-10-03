# Wall-proximity check of Logbook 035 — launch record

Registered 2026-10-04, before any capture ran. Phase 8b housekeeping item H.3.

## Why

The point worm's position is clamped to the square `[0, world_size_mm]` arena, so a worm heading into
an edge slides along it. Its heading, displacement and bearing to the gradient then change for a reason
that is not taxis. Wormlight found every reversal and nearly all of its "weathervaning" began at the dish
wall. Logbook 035's curves excluded near-stationary creep steps but never wall-proximal ones, so whether
part of its klinokinesis or weathervane signal is a wall effect is unknown.

## What runs

Logbook 035's three arms, re-captured because its own captures predate the artefact-retention rule:

| arm | config | role |
|---|---|---|
| MLP | `mlpppo_small_continuous2d_fick_adaptive_klinotaxis_capture.yml` | gating |
| connectome | `connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_capture.yml` | companion |
| control | `mlpppo_small_continuous2d_fick_adaptive_derivative_capture.yml` | no head-sweep |

Each is its 035 parent with `capture_behaviour: true`, and the control's `chemotaxis_mode: derivative`.
Seeds 42–49, 300 episodes, headless, through `scripts/run_campaign.py` with the output controls. 24 runs.

Analysis exactly as 035: the last 100 episodes, `--theta-sharp 0.45`, curving-rate floor 0.25× the median
stride, 80% bootstrap CI.

## Readings, fixed now

1. **Identity.** With the exclusion off, the re-captured per-seed statistics are compared with 035's
   committed `*-curves.json`. If every value matches, the re-capture is 035's data and the comparison
   below is a statement about 035. If any differs, the differences are reported, and the comparison
   below is a statement about the re-capture only, with 035 carrying the note that it could not be
   reproduced exactly.
2. **Primary margin: 1.0 mm**, the cells' `max_step_mm`. A transition with either step within one step
   length of an edge can have been clamped. Arena 20 mm, so the kept region is the central 18 × 18 mm.
3. **Sensitivity margin: 2.0 mm.** Reported, never re-read as the primary.
4. **What "moved" means**, at the primary margin, per arm: any of the four statistics changes verdict
   (REPRODUCED / PARTIAL / ABSENT), or either strategy's combined verdict changes. Means, intervals and
   the fraction of transitions kept are reported beside every verdict.
5. **Outcomes.**
   - Nothing moves: 035 stands, and its note says the curves are robust to excluding wall-proximal
     transitions at both margins.
   - A verdict moves: 035's note states which, and C.3 and D.1 carry the exclusion as a requirement
     rather than an option.
   - The control's weathervane rises toward significance with the exclusion: reported as the
     specificity control weakening, since 035's double dissociation rests on it.

## Retention (A.0)

Committed beside this file: the per-arm summary JSON at each margin (off, 1.0, 2.0) and the identity
comparison. The campaign directory and its captures are archived off-repo.
