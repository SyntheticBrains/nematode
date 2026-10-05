## Why

A.6t (Logbook 077) was unreadable: at block V's thermal point (target 20) every learning arm saturates
above the 90% bar. Described, not a verdict, block V's thermal lead over the current null was not visible
against nulls holding the wild type's gap junctions. If a readable panel confirms that, block V's thermal
advantage is entirely the current null's rewired gap junctions, which changes the combined paper's claim.
The maintainer chose a non-saturating follow-up with the split's arms only.

## What Changes

- **A gate-only pilot**, already run on disjoint seeds 1001–1004, at food targets 25, 30, 35 and 40, chose
  target 35 under a rule fixed before its decisive pilot ran: the lowest target where every level is
  readable. Its evidence is committed as `pilot.json`.
- **Configs**: six target-35 configs, one key each from block V's target-20 thermal configs, through a new
  `generate_thermal_target_configs.py`.
- **Analysis**: `thermal_split.py` scores two readings together — the split (holding the gap junctions)
  and the lead (the wild type against the gap-held null) — against one minimum scaled from A.1's thermal
  effect by the wild type's own `auc_success` ratio.
- **Registration**: seeds 513–640, 768 runs, in `docs/experiments/logbooks/supporting/078-thermal-split/ launch.md`, with the gate preflight's output, before any scored run.

## Capabilities

**Modified**: `architecture-comparison-protocol`, with one added requirement: a difficulty pin chosen for
a follow-up is chosen by the gates alone, under a rule fixed before the pilot that decides it.

## Impact

- `configs/scenarios/thermal_foraging/`: six configs
- `scripts/campaigns/generate_thermal_target_configs.py`, `scripts/analysis/thermal_split.py`: new
- Tests for the configs, the constants against the pilot record, the verdict maps and the manifest
- After the campaign: Logbook 078, the tracker's A.6t, block V's thermal condition in the roadmap

No package code changes.

## Breaking Changes

None.
