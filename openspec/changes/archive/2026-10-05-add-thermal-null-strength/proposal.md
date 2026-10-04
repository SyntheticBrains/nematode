## Why

A.6 and its gap-only split ran on hard350 alone, and found about half of block V's `auc_success` lead
over the degree-preserving null coming from that null's rewired gap junctions. Block V's second cell,
the thermal-plus-foraging cell at target 20, has only A.1's initialisation control behind it. The 8b
re-plan scheduled thermal coverage (A.6t) first in 8b's carried controls, because it feeds both the
combined paper's 8a half and A.5's fallback package.

## What Changes

- **Configs.** Four new thermal configs, the chemical-only and gap-held nulls learning and frozen, each
  one key from block V's thermal current-null configs, written by the existing generator.
- **Analysis.** `scripts/analysis/thermal_null_strength.py` scores both interactions in one panel:
  `combined` (the chemical-only null, as A.6) and `split` (the gap-held null on the current null's exact
  chemical graph, as A.6's split), corrected together, against one registered minimum taken from the
  thermal cell's own committed effect.
- **Scorer.** `operating_point_surface.score_level` takes the block-V cell to score as, defaulting to its
  own, so the thermal panel reuses it unchanged.
- **Registration.** Seeds 385–512 (128), sized from the thermal cell's per-seed spread, in
  `docs/experiments/logbooks/supporting/077-thermal-null-strength/launch.md`, before any scored run.

## Capabilities

**Modified**: `architecture-comparison-protocol`, with one added requirement: a control repeated on
another cell is sized and judged on that cell's own effect.

## Impact

- `configs/scenarios/thermal_foraging/`: four configs
- `scripts/analysis/thermal_null_strength.py`: new; `operating_point_surface.py`: one keyword
- `scripts/campaigns/generate_null_strength_configs.py`: two thermal panels
- Tests for the configs, the registration constants, the verdict maps and the manifest
- After the campaign: Logbook 077, the tracker's A.6t, the roadmap's block V conditions

No package code changes.

## Breaking Changes

None.
