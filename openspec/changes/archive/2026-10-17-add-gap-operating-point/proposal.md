## Why

Phase 9 opens with G.0 (roadmap v4.5 § Phase 9, D23, D24; tracker `phase9-tracking`). Phase 8's one
surviving wiring effect is gap-junction placement: the wild type learns the thermal cell faster than
a null whose gap junctions alone are rewired, +0.096 `auc_success` on the point worm
([Logbook 082](../../../docs/experiments/logbooks/082-plastic-gaps.md)) and +0.126 through the body
([Logbook 092](../../../docs/experiments/logbooks/092-thermal-body.md)). That effect has an untested
explanation belonging to the model rather than the animal. In the settling loop a gap junction is
additive drive, each pair's raw EM count divided by the square root of the two neurons' partner
counts; on Emmons 2024 the pairs ALA–CANL/R (401 each), ALA–PVDL/R (250) and CAN–PVD (100) sit far
above the weight at which a symmetric pair of tanh units latches, 26 pairs carry a count of 20 or
more, and the gap scale has never been swept under PPO. A degree-preserving gap swap moves those
outliers onto the sensory-to-motor path. So "placement matters" may mean "where a few count outliers
land". Nothing in the mechanism block (G.1–G.3b) registers until this screen has read.

## What Changes

- **Brain options** (`connectome-ppo-brain`), each byte-identical at its default:
  - `gap_weight_mode`: `count` (default, today's behaviour) | `unit` (every existing pair's weight
    set to 1 before the fan-in normalisation) | `log_count` (`log1p(count)` before normalisation).
  - `gap_weight_scale`: a positive multiplier on the normalised gap matrix, default 1.0.
  - Two wirings, and a threshold key `gap_outlier_min_count` (default 20) they read:
    `rewired_gap_outliers_held` swaps every gap pair except those with count ≥ the threshold, which
    are held; `rewired_gap_outliers_scattered` swaps only those pairs, each paired with a random
    non-outlier pair, so the outliers land elsewhere while the rest of the gap graph moves only by the
    partners they displace. The chemical graph is held exactly in both, as in the gap-only null.
  - The misleading comment claiming the normalisation keeps per-neuron gap input bounded is corrected.
- **Configs**: the thermal target-35 cell on Emmons 2024 (D23) through a generator: the anchor's four
  arms, and every G.0 arm below, each a stated delta from its parent, re-read by a loader test.
- **Analysis** (`scripts/analysis/gap_operating_point.py`), in the gate preflight's shape: the
  anchor's readout and D23's minimum rule; the panel's readings, corrected together, and the
  registered verdict map (survives / both / artefact / unresolved, every combination of the two deciding readings named); the saturation predictor; the matched-statistics table and the shortcut
  audit for every gap null (D24); the per-seed CSV.
- **The shortcut audit** (`scripts/analysis/sensory_motor_hops.py`) takes the cell's config, so it can
  audit the thermal cell's injected sensors and every gap null.
- **Runs**: the Emmons anchor pilot (seeds 9801–9816), then, under a launch record committed before
  any scored run, the G.0 panel (seeds 2401–2464); then Logbook 095.
- **Tracker**: H.2 (retention rule re-registered; 8b's campaign directories archived off-repo by the
  maintainer, 2026-10-16) and H.3 (D23 and D24 cited by this launch record) close with this change;
  G.0's items tick as they read.

## Capabilities

**Modified**: `connectome-ppo-brain` — the gap-weight options and the two outlier wirings (ADDED
requirements; every existing scenario unchanged).

## Impact

- Code: `connectome/rewiring.py` (a held set and a focal set for the undirected swap),
  `brain/arch/connectome_ppo.py` (three keys, two wiring values, the gap build), their tests.
- Scripts: `scripts/campaigns/generate_gap_operating_point_configs.py`,
  `scripts/analysis/gap_operating_point.py`, `scripts/analysis/sensory_motor_hops.py`.
- Configs: under `configs/scenarios/thermal_foraging/`, generated.
- Docs: `docs/experiments/logbooks/supporting/095-gap-operating-point/` (launch record, pilot,
  panel), Logbook 095, the tracker, the roadmap's G.0 status, CHANGELOG.
- Machine time: the pilot ≈ 1 h; the panel ≈ 896 runs ≈ 15 h at 16 workers.

## Breaking Changes

None.

## Backward Compatibility

Every new key defaults to today's behaviour, and the existing wirings draw the graphs they drew
before; tests pin both.
