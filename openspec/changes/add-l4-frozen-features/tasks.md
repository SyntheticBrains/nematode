# Tasks

## 1. The arms

- [x] 1.1 Two configs from the committed `..._hard350_eprop_readout_only.yml` and
  `..._hard350_eprop_frozen.yml`: their `wiring: rewired_degree_preserving` counterparts. `rewire_seed`
  stays **unset**, so each seed's rewiring derives from its run seed and the wild-type and rewired arms
  pair — the same discipline V.1 and V.3 ran under.
- [x] 1.2 Exact-key test: each rewired config differs from its wild-type partner in the `wiring` key
  **alone**. The learning configs differ from their floors in **three** keys — `freeze_updates`,
  `plasticity_plastic_readout` and `plasticity_plastic_tensors` — and the last two are **required** to
  differ: a frozen arm may not declare a plastic readout, since with no update no tensor moves and "a
  plastic-readout floor" is not a thing. A guard written in R.2 refuses that combination, and the test
  asserts the guard as well as the key set. *(This task originally said "`freeze_updates` alone"; the
  test caught it.)*
- [x] 1.3 Confirm the operating point is R.2's, unchanged and written out rather than inherited:
  `plasticity_eligibility: eprop`, `plasticity_learning_signal: random`,
  `plasticity_plastic_readout: true`, `plasticity_plastic_tensors: readout_only`,
  `plasticity_node_noise: 0.0`, `plasticity_rate: 0.001`, `trace_decay: 0.9`,
  `initial_log_std: -1.0`, `forward_pass_depth: 4`, the committed anatomical readout.

## 2. The matched projection

- [x] 2.1 **Assert by test that the feedback projection `B` is identical across wirings at a seed.**
  It is so by construction — `B` is drawn from a `torch.Generator()` seeded with the run seed, the
  rewiring draws from a separate numpy generator, and rewiring preserves `n_neurons` — but that is a
  reading of the code, and the alternative confounds the wiring with the feedback path invisibly.
- [x] 2.2 Assert the rewiring preserves what the arms are matched on: neuron set and ordering, per-post
  fan-in, and the motor pool's membership — so the weight-init scale, the strict mask's shape, the
  gap-junction normalisation and the readout's inputs are the same on both sides.

## 3. Harness

- [x] 3.1 `scripts/analysis/l4_frozen_features.py`, a **sibling** module — the R.1c/R.1d/R.2 precedent.
  It drives block V's committed `connectome_structure_efficiency.py` rather than re-implementing its
  metrics: the four efficiency metrics, paired by seed, BH-FDR, `episodes_to_30pct_success` primary
  against the registered **≥ 20%** minimum.
- [x] 3.1a **`wiring_premise.py` and `connectome_structure_efficiency.py` are READ-ONLY in this change.**
  The first hard-codes its test family as `(id, cell, arm_a, arm_b, kind)` tuples and **carries block V's
  committed verdicts** for three cells — so the natural-looking implementation, adding a fourth cell to
  it, would edit a harness whose output is already on the record. The gates and the prior check below
  live in the sibling module, reusing the statistics layer; a test asserts neither committed file is
  imported for anything but its metrics and its stats.
- [x] 3.1b **Build the manifest the efficiency script consumes**, and register the mapping rather than
  deciding it in code. `analyse()` reads `arm seed out_path` lines and requires the arm names
  **`wild_type`** and **`rewired_null`** exactly, failing fast on any unpaired seed — an extra or missing
  run would otherwise shrink the shared horizon and distort every metric. This change's four labels map
  onto those two for the efficiency primary (the floors are not part of that call), and the paths are
  the campaign's own run logs: the committed parser reads them directly, verified at **3000 of 3000**
  lines on an R.2 log.
- [x] 3.2 **The two learning gates** — each wiring against its **own** frozen floor. A contrast between
  two arms that did not learn is not a wiring result.
- [x] 3.3 **The untrained-prior check** — wild-type frozen against rewired frozen, **measured here and
  not inherited**. V.1's −0.17 (q = 0.735) and V.3's −0.01 (q = 0.841) are reported beside it as
  context only: those floors were PPO-configured and ran at the default action noise (std 1.0), where
  these run at `initial_log_std: -1.0` (std 0.368), and treating a figure from one regime as
  established in another is the cross-regime comparison this change's own design forbids. A separation
  here would mean the rewiring changed the substrate before any learning, which makes the primary
  uninterpretable — and it is one of the two conditions that fire `void`.
- [x] 3.4 **Credited drift on `w_chem`, which must read 0.00 for both wirings** — the check that the
  substrate really was frozen, as it was in R.2's `readout_only` arm. It **voids** the contrast
  otherwise, and requires drift read at **every** scored seed on both arms: with one arm's checkpoints
  missing, an `all()` over a short list would report the substrate frozen on the strength of whatever
  happened to be on disk.
- [x] 3.5 The four registered readings — `wiring_is_legible`, `wiring_is_inert_as_features`,
  `below_bar`, `void` — with the consequence of each in the harness, not in prose. `void` fires on a
  failed learning gate, a separated prior, **or a substrate that did not stay frozen** — non-zero
  `w_chem` drift, or drift evidence missing for any scored seed, the two being distinguished in the
  message because "it could not be checked" is not "it held".
- [x] 3.6 The power arithmetic carried as a field in the record, not only in this registration: seed
  count, the k needed at that count, and the power against V.3's observed 66–81% win rate.
- [x] 3.7 Tests for 3.1–3.6, including a fixture in each reading and one where the prior separates.

## 4. The stop clauses

- [x] 4.1 A **pilot on disjoint seeds 101–104** before any registered seed: the arms run, the rewiring
  differs between them, `B` matches within a seed, and both floors sit where a no-learning policy sits.
  **Ran 2026-09-15, 16/16 clean**: `w_chem` drift 0.00 on both wirings, the prior does not separate
  (−2.29 foods, q = 0.875), and both learners move +16.3 and +14.4 foods off their floors on 4/4
  seeds. Its direction is **not read** and the harness withholds a verdict at that seed count — with
  4 pairs the smallest reachable one-sided p is 2⁻⁴ = 0.0625, above the gate, so no gate could have
  passed whatever the arms did. **That was a harness defect the pilot caught**: it first printed
  `void — a learning gate failed`, the same shape as the bug R.2's harness had in a more dangerous
  form. Fixed and tested.
- [x] 4.2a **The primary metric's censoring is recorded with its direction of bias.**
  `episodes_to_30pct_success` is right-censored at the horizon, which the committed instrument treats
  as an exact observation — block V's established convention, and the reason the per-seed CSV carries a
  `primary_censored` column. The bias direction is stated rather than corrected: a censored wild-type
  seed's true value is **≥** the horizon, so using the horizon **understates** how far behind it is.
  That is conservative for the registered one-sided test (wild-better) and also conservative for the
  reverse lean, so it cannot manufacture either finding. Switching to a censor-aware survival
  estimator was considered and **rejected**: it would diverge from the instrument V.1 and V.3 are
  recorded against, making this campaign non-comparable with the comparator it exists to be read
  beside, and changing the estimator after seeing the result is the post-hoc move this project's
  discipline forbids.
- [x] 4.2 **The rate check runs** on those disjoint seeds — the committed `0.001` and one decade either
  side, learning arms only. R.2 waived its registered rate check, defensibly, because its pilot was
  plainly not rate-limited; this campaign can return a **null** that closes the phase, and a null from
  an uncalibrated rate is the failure R.1c's σ calibration found on this substrate. No waiver.
- [x] 4.3 Re-score R.2's committed `readout_only` arm through this harness and confirm **17.570** foods
  and **52.61%** full clear. A mismatch means the harnesses disagree and nothing here is comparable.
  **Passes exactly**: every seed matches R.2's committed value to four decimals through the same
  `read_log`, and the two campaigns' runs are **byte-identical across all 3000 episodes**. This task
  earned itself — progress updates had reported an unexplained 0.6-food gap three times, which was a
  metric mismatch of my own (the logs' whole-run mean read against the harness's plateau-tail mean),
  not a harness disagreement.
- [x] 4.4 `launch.md` committed before anything runs, carrying the power arithmetic and the honest prior.

## 5. Campaign

- [x] 5.1 128 runs: four arms × seeds 1–32 at 3000 episodes, with `--track-experiment` so the drift
  check has weights to read.
- [x] 5.2 Per-seed CSV, the per-arm table and the verdict under `supporting/064-l4-frozen-features/`.
  The CSV carries a **`primary_censored`** column per row: the primary metric is right-censored at the
  3000-episode horizon, five wild-type seeds and one null seed sit there, and a reader has to be able
  to see which rows those are rather than take the means on trust.

## 6. The record

- [x] 6.1 Logbook 064: the four-arm table, the two gates and the prior check stated in **every** verdict
  branch, the drift column, the verdict against the four registered readings, and V.3's +23.5% reported
  beside the result as the same question under another regime — **not** as a quantitative delta.
- [x] 6.2 The experiments index row and `CHANGELOG.md`.
- [x] 6.3 The tracker's L.0 entry, and L.1's conditional promotion resolved: MUST if this reads null,
  SHOULD if positive.
- [x] 6.4 State plainly what this may **not** be cited as, per the design's last section — including
  that it does not satisfy D2's primary and cannot convert the SPLIT into a GO.
- [x] 6.5 **Not applicable**: the reading is `wiring_is_inert_as_features`, so L.4 and L.5 stay shut and
  are recorded `closed-unopened` in the tracker — against a null there is nothing for a feature
  ablation to have changed. They reopen if L.1 finds the pooling was hiding structure. **L.1 is
  promoted to MUST** instead, by the conditional registered before this ran.
