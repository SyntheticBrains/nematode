## Why

The continuous environment clamps the worm's position to the square arena, so a worm heading into an
edge slides along it. Its heading, displacement and bearing to the gradient then change for a reason that
is not taxis. Wormlight, the sister project, found every reversal and nearly all of its apparent
weathervaning began at the dish wall. The chemotaxis validation behind Logbook 035 excludes near-stationary
creep steps but not wall-proximal ones, so part of 035's klinokinesis or weathervane signal could be a wall
effect. C.3 (body-level validation) and D.1 (edged lawns) will reuse the method, so the check has to exist
first. Phase 8b's re-plan scheduled it as housekeeping item H.3.

## What Changes

### 1. The exclusion

`behavioural_curves.away_from_walls(steps, arena_mm, margin_mm)` splits a captured run into its stretches
that keep at least the margin from every edge; no transition bridges an excluded gap. The validation
harness gains `--wall-margin-mm` with `--arena-mm`, off by default. When off, the summary is unchanged;
when on, the summary records the margin, the arena and the transitions kept.

### 2. Logbook 035, re-captured and re-read

035's three arms are committed as capture configs, one key from their parents, because 035's own captures
predate the artefact-retention rule. They are re-run on 035's seeds and episodes, checked for identity
against 035's committed statistics with the exclusion off, and read at a 1.0 mm primary margin (the cells'
`max_step_mm`) and a 2.0 mm sensitivity margin, as registered in the launch record before any run.

## Capabilities

**Modified**: `realworm-behavioural-validation`, with one added requirement: wall-proximal transitions
can be excluded.

## Impact

- `packages/quantum-nematode/quantumnematode/validation/behavioural_curves.py`: `away_from_walls`
- `scripts/analysis/behavioural_chemotaxis_validation.py`: the two flags and the summary block
- `scripts/analysis/wall_exclusion_check.py`: new; manifests, the three readings and the comparison
- `configs/scenarios/foraging/*_capture.yml`: three capture configs
- `docs/experiments/logbooks/supporting/035-realworm-chemotaxis-validation/wall-exclusion/`: the launch
  record, the summaries and the comparison
- Logbook 035: a dated note; the Phase 8 tracker: H.3
- Tests for the filter, the harness's off and on paths, and the comparison

No brain, environment or agent code changes. No committed config changes.

## Breaking Changes

None.
