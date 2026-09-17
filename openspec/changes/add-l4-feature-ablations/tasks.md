# Tasks

## 1. The arms

- [x] 1.1 Eight configs, each differing from its committed L.1 wide parent in **one key** —
  `synapse_signs: atlas` for the four L.4 arms, `enable_gap_junctions: false` for the four L.5 arms.
  Exact-key test by loading both and diffing the resolved config, as L.1 did.
- [x] 1.2 Assert by test that `synapse_signs: atlas` at the per-neuron width leaves every magnitude
  bitwise identical and changes signs only, and that `enable_gap_junctions: false` leaves **every
  parameter** bitwise identical — so each ablation is the forward pass and nothing else.
- [x] 1.3 The pool numbers as tested facts, not prose: 199 gap junctions touching the pool, 47 within
  it, all 39 pool neurons with at least one; 323 chemical inputs, 311 grounded, 275 E / 36 I.
- [x] 1.4 Confirm no package code changes are needed. Both flags exist; both build at width 39.

## 2. The baseline

- [x] 2.1 *(**passed 2026-09-18**: `wt_wide` and `rn_wide` at seed 1, re-run under the output
  controls, reproduce L.1's committed logs on all 9 parsed fields — `campaigns/export-flags-identity`.
  L.1's 384 wide-arm runs are the baseline.)* **Byte-identity check before any reuse.** Re-run `wt_wide` and `rn_wide` at one L.1 seed
  each under `--no-detailed-export --no-file-log` and compare to L.1's committed logs on every field
  `read_log` parses. Identical: L.1's 384 wide-arm runs are the baseline, and the record says so with
  the evidence. **Any** field differing: re-run the baseline in full (+384 runs) and reuse nothing.
- [x] 2.2 The baseline's four arms are read from `campaigns/readout-width` through the same
  `connectome_structure_efficiency` call as the ablated arms, not from L.1's JSON, so both halves of
  every interaction pass through the same code path.

## 3. Harness

- [x] 3.1 `scripts/analysis/l4_feature_ablations.py`, the L.1 sibling pattern: manifest builder per
  (ablation, wiring) pair, `connectome_structure_efficiency.analyse` called **once per pair** and
  unmodified, per-seed interaction, both cells' wiring effects, gates read first. **Imports** L.1's
  `_two_sided`, `censoring`, reachability, orientation and family-adjustment helpers from
  `l4_readout_width.py` rather than copying them.
- [x] 3.2 **Read-only, asserted**: `l4_readout_width.py`, `connectome_structure_efficiency.py`,
  `wiring_premise.py`, with V.4's shallow-clone skip guard.
- [x] 3.3 **The minimum effect as a decision rule**: `carries_the_effect` requires significance **and**
  `abs(Δ) ≥ 0.123` — two-thirds of the wide wiring effect +0.1852, the quantity an ablation can
  actually remove; a significant interaction below the minimum reads `inconclusive_at_this_sensitivity`
  with the shrinkage named, never `carries_the_effect`. `survives_without_it` carries the interaction's
  size and CI, because a non-significant interaction is a failure to detect.
- [x] 3.3b *(**added 2026-09-18 after the pilot, before the campaign**)* **The L.4 gains diagnostic**:
  each atlas arm's gain over its floor against the wide arm's gain over its floor, paired per seed,
  per wiring, two-sided, outside the family. Both significantly smaller: a `carries_the_effect` on
  L.4 reads *carries or unlearnable*. Motivated by the pilot — atlas gains +2.2/+2.8 against wide
  +13.7/+7.8, bimodal, floors quiet — and registered before any registered seed ran. Qualifies,
  never rescues.
- [x] 3.3a **The L.4 floors diagnostic**: the two atlas frozen floors against the two wide frozen floors
  on plateau-tail foods, two-sided, outside the family. If it fires at q ≤ 0.05, a `carries_the_effect`
  on L.4 is reported as *carries or saturates*. It qualifies; it never rescues.
- [x] 3.4 **One BH-FDR family across all ten tests**, `q` attached to every record beside its raw `p`,
  the reading comparing the interaction's **q**.
- [x] 3.5 The guards L.1's reviews added, carried: metric orientation before comparing directions;
  reachability from each ablation's retained pairs, branches withheld at an unreachable n; the seed
  intersection under `--allow-incomplete`; `--seeds` derived from `SEEDS`; verdict vocabulary derived
  from source.
- [x] 3.6 `auc_success` primary, `episodes_to_30pct_success` beside it with censoring per cell, the
  reason carried — L.1's registration, inherited.
- [x] 3.7 **Per ablation, never pooled**: a split is the informative outcome and is reported as one.
- [x] 3.8 Tests for 3.1–3.7: a fixture per reading, one significant-below-minimum case, one split, one
  orientation case through `analyse()`, and one where the L.4 diagnostic fires and the reading is
  qualified.

## 4. The structural probe

- [x] 4.1 `scripts/analysis/l4_structural_probe.py`: build the 96 rewirings from their seeds (and the
  wild type), compute mean within-class presynaptic Jaccard from `m_chem`, join to L.1's committed
  `per-seed.csv` on seed, and run the **registered** test — Spearman ρ against `rn_wide − rn_pooled`,
  one-sided positive, minimum ρ ≥ 0.3. Nothing about the correlation is computed before this file and
  its test exist.
- [x] 4.2 The descriptive companion: the wild type's Jaccard against the 96 rewirings' distribution.
- [x] 4.3 Tests: the statistic on a hand-built mask with a known Jaccard; the join refuses a seed
  missing from either side; the minimum is applied.

## 5. The stop clauses

- [x] 5.1 *(**passed 2026-09-18**: 32/32; every ablated arm differs from its wide parent after
  training — atlas 0.10/0.13 and nogap 0.71/0.73 against the wide 0.49/0.29 on `auc_success`; all four
  new floors at 1.8–4.1 foods where a no-learning policy sits; both readings correctly withheld at four
  pairs. The pilot also motivated the gains diagnostic, 3.3b.)* **Pilot on seeds 101–104** with the output controls: all eight arms run, each ablated arm
  differs from its wide parent after training, the floors sit where a no-learning policy sits. No
  reading at four pairs.
- [x] 5.2 *(passed, see 2.1.)* The byte-identity check (2.1) **passes or the baseline is re-run** —
  settled before any registered seed.
- [x] 5.4 *(**ran 2026-09-18 — outcome B**: 0.0001 learns cleanly on both wirings, +15.0/+16.1 over floor with no seed below, against +2.2/+2.8 at 0.001 and a full collapse at 0.01; see the launch record.)* *(added 2026-09-18 after the pilot, before the campaign)* **The atlas rate check**, with
  its three-outcome decision rule registered in `launch.md` and the design **before** the 16 runs:
  both atlas learning arms at 0.0001 and 0.01 on seeds 101–104, each config one key from its atlas
  parent. Outcome A or C: campaign as registered. Outcome B: atlas learning arms at the clean rate
  plus a rate-matched wide baseline (+192 runs), floors reused. The harness accepts a per-ablation
  baseline directory for that case.
- [x] 5.3 `launch.md` before anything runs: the two interactions, the readings with the minimum effect,
  the baseline decision and its evidence, the sensitivity arithmetic, the probe's registered test,
  the L.4 diagnostic, and the honest prior as the design states it.

## 6. Campaign

- [ ] 6.1 *(**reshaped by the rate check, outcome B**)* **960 runs: ten arms × seeds 1–96** at 3000
  episodes — the atlas learning arms at **0.0001**, their floors, the four nogap arms, and the two wide
  learning arms at 0.0001 as L.4's rate-matched baseline — with `--track-experiment --no-detailed-export --no-file-log`. Measured disk cost recorded in `launch.md` **before** launch.
- [ ] 6.2 Per-seed CSV with all cells, the tables and both readings under
  `supporting/067-l4-feature-ablations/`.

## 7. The record

- [ ] 7.1 Logbook 067: both ablations' tables, each reading with its minimum-effect status, the
  baseline decision, the probe's result, and every "may not be cited as".
- [ ] 7.2 The experiments index row and `CHANGELOG.md`.
- [ ] 7.3 The tracker's L.4 and L.5 entries, and the ladder paragraph in the roadmap.
- [ ] 7.4 **If either ablation reads `carries_the_effect`**: the phase's positive is restated as
  living in the named feature, and the phase-after-7 rung that manipulates that feature directly is
  named.
- [ ] 7.5 **If both read `survives_without_it`**: the effect lives in the directed chemical graph's
  connectivity, and the placed-plasticity rung is the one that follows.
