# Tasks

## 1. The arms

- [ ] 1.1 Two configs from the committed `..._hard350_eprop_readout_only.yml` and
  `..._hard350_eprop_frozen.yml`: their `wiring: rewired_degree_preserving` counterparts. `rewire_seed`
  stays **unset**, so each seed's rewiring derives from its run seed and the wild-type and rewired arms
  pair — the same discipline V.1 and V.3 ran under.
- [ ] 1.2 Exact-key test: each rewired config differs from its wild-type partner in the `wiring` key
  **alone**, and the two learning configs from the two floors in `freeze_updates` alone.
- [ ] 1.3 Confirm the operating point is R.2's, unchanged and written out rather than inherited:
  `plasticity_eligibility: eprop`, `plasticity_learning_signal: random`,
  `plasticity_plastic_readout: true`, `plasticity_plastic_tensors: readout_only`,
  `plasticity_node_noise: 0.0`, `plasticity_rate: 0.001`, `trace_decay: 0.9`,
  `initial_log_std: -1.0`, `forward_pass_depth: 4`, the committed anatomical readout.

## 2. The matched projection

- [ ] 2.1 **Assert by test that the feedback projection `B` is identical across wirings at a seed.**
  It is so by construction — `B` is drawn from a `torch.Generator()` seeded with the run seed, the
  rewiring draws from a separate numpy generator, and rewiring preserves `n_neurons` — but that is a
  reading of the code, and the alternative confounds the wiring with the feedback path invisibly.
- [ ] 2.2 Assert the rewiring preserves what the arms are matched on: neuron set and ordering, per-post
  fan-in, and the motor pool's membership — so the weight-init scale, the strict mask's shape, the
  gap-junction normalisation and the readout's inputs are the same on both sides.

## 3. Harness

- [ ] 3.1 `scripts/analysis/l4_frozen_features.py`, a **sibling** module — the R.1c/R.1d/R.2 precedent.
  It drives block V's committed `connectome_structure_efficiency.py` rather than re-implementing its
  metrics: the four efficiency metrics, paired by seed, BH-FDR, `episodes_to_30pct_success` primary
  against the registered **≥ 20%** minimum.
- [ ] 3.1a **`wiring_premise.py` and `connectome_structure_efficiency.py` are READ-ONLY in this change.**
  The first hard-codes its test family as `(id, cell, arm_a, arm_b, kind)` tuples and **carries block V's
  committed verdicts** for three cells — so the natural-looking implementation, adding a fourth cell to
  it, would edit a harness whose output is already on the record. The gates and the prior check below
  live in the sibling module, reusing the statistics layer; a test asserts neither committed file is
  imported for anything but its metrics and its stats.
- [ ] 3.1b **Build the manifest the efficiency script consumes**, and register the mapping rather than
  deciding it in code. `analyse()` reads `arm seed out_path` lines and requires the arm names
  **`wild_type`** and **`rewired_null`** exactly, failing fast on any unpaired seed — an extra or missing
  run would otherwise shrink the shared horizon and distort every metric. This change's four labels map
  onto those two for the efficiency primary (the floors are not part of that call), and the paths are
  the campaign's own run logs: the committed parser reads them directly, verified at **3000 of 3000**
  lines on an R.2 log.
- [ ] 3.2 **The two learning gates** — each wiring against its **own** frozen floor. A contrast between
  two arms that did not learn is not a wiring result.
- [ ] 3.3 **The untrained-prior check** — wild-type frozen against rewired frozen, **measured here and
  not inherited**. V.1's −0.17 (q = 0.735) and V.3's −0.01 (q = 0.841) are reported beside it as
  context only: those floors were PPO-configured and ran at the default action noise (std 1.0), where
  these run at `initial_log_std: -1.0` (std 0.368), and treating a figure from one regime as
  established in another is the cross-regime comparison this change's own design forbids. A separation
  here would mean the rewiring changed the substrate before any learning, which makes the primary
  uninterpretable — and it is one of the two conditions that fire `void`.
- [ ] 3.4 **Credited drift on `w_chem`, which must read 0.00 for both wirings** — the check that the
  substrate really was frozen, as it was in R.2's `readout_only` arm.
- [ ] 3.5 The four registered readings — `wiring_is_legible`, `wiring_is_inert_as_features`,
  `below_bar`, `void` — with the consequence of each in the harness, not in prose. `void` fires on a
  failed learning gate or a separated prior.
- [ ] 3.6 The power arithmetic carried as a field in the record, not only in this registration: seed
  count, the k needed at that count, and the power against V.3's observed 66–81% win rate.
- [ ] 3.7 Tests for 3.1–3.6, including a fixture in each reading and one where the prior separates.

## 4. The stop clauses

- [ ] 4.1 A **pilot on disjoint seeds 101–104** before any registered seed: the arms run, the rewiring
  differs between them, `B` matches within a seed, and both floors sit where a no-learning policy sits.
- [ ] 4.2 **The rate check runs** on those disjoint seeds — the committed `0.001` and one decade either
  side, learning arms only. R.2 waived its registered rate check, defensibly, because its pilot was
  plainly not rate-limited; this campaign can return a **null** that closes the phase, and a null from
  an uncalibrated rate is the failure R.1c's σ calibration found on this substrate. No waiver.
- [ ] 4.3 Re-score R.2's committed `readout_only` arm through this harness and confirm **17.570** foods
  and **52.61%** full clear. A mismatch means the harnesses disagree and nothing here is comparable.
- [ ] 4.4 `launch.md` committed before anything runs, carrying the power arithmetic and the honest prior.

## 5. Campaign

- [ ] 5.1 128 runs: four arms × seeds 1–32 at 3000 episodes, with `--track-experiment` so the drift
  check has weights to read.
- [ ] 5.2 Per-seed CSV, the per-arm table and the verdict under `supporting/064-l4-frozen-features/`.

## 6. The record

- [ ] 6.1 Logbook 064: the four-arm table, the two gates and the prior check stated in **every** verdict
  branch, the drift column, the verdict against the four registered readings, and V.3's +23.5% reported
  beside the result as the same question under another regime — **not** as a quantitative delta.
- [ ] 6.2 The experiments index row and `CHANGELOG.md`.
- [ ] 6.3 The tracker's L.0 entry, and L.1's conditional promotion resolved: MUST if this reads null,
  SHOULD if positive.
- [ ] 6.4 State plainly what this may **not** be cited as, per the design's last section — including
  that it does not satisfy D2's primary and cannot convert the SPLIT into a GO.
- [ ] 6.5 If the reading is `wiring_is_legible`, record that **L.4 and L.5 open** — their gate is this
  result reading positive.
