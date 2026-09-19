# Tasks

## 1. The arms

- [x] 1.1 `..._hard350_eprop_readout_only_r1e4_rewired_null.yml`, one key (`plasticity_rate: 0.0001`)
  from the committed `..._readout_only_rewired_null.yml`; and `..._readout_only_r1e4.yml`, L.0's
  rate-check config, promoted to a registered arm with its header rewritten and still one key from
  `..._readout_only.yml`. Exact-key test by loading both and diffing the resolved config.
- [x] 1.2 Assert by test that the two 0.0001 pooled arms differ from the two 0.0001 wide arms in the
  `readout_width` key alone, so the 2×2 at 0.0001 is L.1's 2×2 with one key moved on every learning
  cell.
- [x] 1.3 Confirm no package code changes are needed.

## 2. The reused cells

- [ ] 2.1 **Identity check before any reuse**: at seed 1, re-run the four floors, L.1's two pooled
  learning arms and the two wide 0.0001 arms under the current path (`--no-detailed-export --no-file-log`) and compare to their committed logs on every field `read_log` parses. Identical: the
  cells are reused and the record says on what evidence. Any field differing: that arm's cell is
  re-run for seeds 1–96 and nothing of it is reused.
- [ ] 2.2 Every reused cell is read from its campaign directory through the same
  `connectome_structure_efficiency` call as the new arms — never from a committed JSON.

## 3. Harness

- [x] 3.1 `scripts/analysis/l4_rate_calibration.py`: three campaign directories in
  (`--campaign`, `--wide-rate`, `--baseline`), a manifest per (rate, width) pair,
  `connectome_structure_efficiency.analyse` once per pair and unmodified, `_two_sided`, `censoring`,
  `cell_values`, `write_manifest`, `_jsonable` imported from `l4_readout_width.py`, and the wide
  0.0001 cells read through `l4_feature_ablations.scan_rate_matched_wide` (L.1's own label regex
  rejects `_r1e4` names, so no regex is copied and `rw.scan` cannot mis-read them).
- [x] 3.2 **Read-only, asserted**: `l4_readout_width.py`, `l4_feature_ablations.py`,
  `connectome_structure_efficiency.py`, `wiring_premise.py`, with the shallow-clone skip guard.
- [x] 3.3 The primary: the interaction at 0.0001 per seed, then the committed paired test; both main
  effects at 0.0001 beside it; the four gates read first, each 0.0001 arm against its own-width
  floor; the three-way `I_1e-3 − I_1e-4` per seed as the registered secondary; the wide wiring effect
  at 0.0001 reported as a reference outside the family. **Eight tests, one BH-FDR family**, q beside
  every raw p.
- [x] 3.4 **The minimum as a decision rule**: `pool_effect_survives_the_rate` requires significance
  **and** `abs(Δ) ≥ 0.141` (half of L.1's +0.2818); significant-below-minimum reads
  `pool_effect_is_rate_specific` with *shrunk below half* named; a non-significant interaction reads
  the same with size and CI carried, as a failure to detect.
- [x] 3.5 The guards carried from L.1 and L.4: metric orientation before comparing directions;
  reachability with the verdict withheld below 5 pairs; the seed intersection under
  `--allow-incomplete` and strict completeness over every cell of every campaign otherwise; `--seeds`
  derived from `SEEDS`; rate tags held to their arms (`_r1e4` on every 0.0001 learning arm, none on
  a floor, and `_r1e2` — L.0's rate-check tag — refused outright); vocabulary derived from source.
- [x] 3.6 `auc_success` primary, `episodes_to_30pct_success` beside it with censoring per cell.
- [x] 3.7 The realised interaction sd, se, detectable effect and power at 0.141 as fields.
- [x] 3.8 Tests for 3.1–3.7: a fixture per reading, the below-minimum case, an orientation case
  through `analyse()`, the rate-tag refusal, strict completeness over three campaigns, and the
  read-only guard.

## 4. The stop clauses

- [ ] 4.1 **Pilot on seeds 101–104**, eight runs, under the output controls: both pooled 0.0001 arms
  learn above L.1's pilot pooled floors and differ from the 0.001 pooled pilot arms; the wild-type arm
  is compared to L.0's rate-check run on the same seeds. No reading at four pairs.
- [ ] 4.2 The identity check (2.1) passes or the affected cell is re-run — settled before any
  registered seed.
- [x] 4.3 `launch.md` before anything runs: the 2×2 at 0.0001 with its three sources, the readings with
  the minimum, the sensitivity table, the identity evidence, what the reading conditions and what it
  does not decide, and the honest prior.

## 5. Campaign

- [ ] 5.1 192 runs: two arms × seeds 1–96 at 3000 episodes, `--track-experiment --no-detailed-export --no-file-log`; disk (~17.6 MB per run, ~3.4 GB) recorded in `launch.md`
  before launch.
- [ ] 5.2 Per-seed CSV with all four 0.0001 cells and L.1's four 0.001 cells, the tables and the
  reading under `supporting/068-l1b-rate-calibration/`.

## 6. The record

- [ ] 6.1 Logbook 068: the 2×2 at 0.0001 beside L.1's at 0.001, the interaction with its
  minimum-effect status, the three-way, the gates, both metrics with censoring, the identity evidence,
  and every "may not be cited as".
- [ ] 6.2 The experiments index row and `CHANGELOG.md`.
- [ ] 6.3 The tracker: an L.1b entry, the dated condition on L.1's entry, and Z.1's dependency line
  noting L.1b ran first; the same condition as a dated note in Logbook 066 and in the roadmap's
  ladder paragraph. L.1's verdict is not rewritten.
- [ ] 6.4 **If `survives`**: the synthesis carries the interaction as the phase's positive and the
  sign flip as a 0.001 result.
- [ ] 6.5 **If `rate_specific`**: the synthesis carries L.1 as a positive at one pinned rate that a
  decade lower does not reproduce, in the same sentence as the claim.
- [ ] 6.6 **If `width_favours_the_shuffle_at_this_rate`**: reported as the reverse direction, not
  explained; the synthesis says the interaction's sign depends on the rate.
- [ ] 6.7 The phase-protocol note under principle 7 gains the pooled-width fact: L.0 did sweep the rate
  on the wild type at the pooled width and 0.001 won there, so what was unswept is the per-neuron
  width and the null at either width.
