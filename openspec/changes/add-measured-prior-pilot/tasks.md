# Tasks: The measured-prior pilot — sign versus magnitude, and the unit-scale sweep

Phase 8 task **B.1b**. The plan is in `docs/roadmap.md` § Phase 8, **D16** and **D17**. The
decisions taken before implementation are in this change's `design.md`.

## Code

- [x] 1. **done: the null's fan-in block is reordered per neuron, wild-type covered positions first, so the existing overwrite places everything the spec names.** Original scope: **The fan-in pairing** (`connectome_ppo.py`), per design Decision A:

  - allow `per_neuron_fanin` with a measured prior, at validation and at construction, and keep
    `dense_mask` refused;
  - carry each neuron's wild-type covered positions in the rewired assignment;
  - under the fan-in draw, place the wild type's covered and then uncovered values on the null;
  - under `measured_signs`, take the magnitudes from the wild type's draw.

  `edge_order` is unchanged. No planning references in package code.

- [x] 2. **done: `default_rng([seed, tag])`.** Original scope: **The shuffle's generator** (Decision B): `measured_shuffled` permutes from a generator
  seeded from the run seed and a fixed tag.

- [x] 3. **done, 61 tests in the file (27 new). With the reorder disabled, 6 of the 8 fan-in placement and multiset tests fail; the two that pass are the random prior and the edge-order contrast, which the reorder does not touch.** Original scope: **Tests**, extending `test_connectome_weight_prior.py`:

  - under the fan-in draw, the wild type's uncovered edges are bit-identical to its random fan-in
    build, and every other parameter is identical;
  - on the null, the first *k* edges and the remaining edges are as specified, and each neuron's
    multiset equals the wild type's, under every measured prior;
  - the shared generator is unchanged under both draws;
  - at one seed the shuffle generator's stream differs from the draw generator's, which fails
    without the fix, and the permutation is unchanged across draws;
  - `dense_mask` is refused at validation and at construction;
  - the existing `edge_order` tests still pass unchanged.

## Panel

- [x] 4. **done: `scripts/analysis/measured_prior_pilot.py`; the selection is a function of the gate records alone.** Original scope: **Panel definition and analysis**, in `scripts/analysis/measured_prior_pilot.py`:
  - the stems, levels, seeds and arm map, built by a loop and not by a regex;
  - a manifest builder writing A.2's line format (`<arm> <level> <seed> <log path>`, with arms
    named `wt_learn`, `rn_learn`, `wt_frozen`, `rn_frozen`), which `learning_gates` reads, and the
    completeness check;
  - per level, the learning gates, censoring and metric choice, and the wiring gap;
  - the branch read and the multiplier selection, per Decision E: a level passes on `gate_passes`
    and not `saturated`, and the Lee branch reads the wild type's own test against its floor;
  - a per-seed CSV (`lineterminator="\n"`) and an analysis JSON.
- [x] 5. **done: `learning_gates` takes `floor_level` and `substrate_drift` takes `floor_levels`, both defaulting to A.2's rule; the gate also records each seed's plateau and floor so the CSV carries them. A.2's 190 tests pass unchanged.** Original scope: **The gate, shared rather than copied**: `operating_point_surface.learning_gates` takes the
  floor level as an argument (Decision F), and A.2's tests still pass.
- [x] 6. **done: 48 configs written, none pre-existing.** Original scope: **The config generator**: `scripts/campaigns/generate_measured_prior_configs.py` writes
  the 48 configs from their parents, adding at most two keys, with a house header. Existing files
  are left alone.
- [x] 7. **done, 74 tests: configs through the real loader, fresh seeds, every level building distinct weights with a floor built on its learning arm's weights, and the selection's eight cases.** Original scope: **Panel tests** (`tests/.../analysis/test_measured_prior_pilot.py`):
  - every config loads through the real loader and differs from its parent only in the registered
    keys;
  - every level reaches the brain: `weight_prior` and the multiplier change the constructed chemical
    weights (the swept-level requirement);
  - the seeds are fresh and disjoint;
  - the selection rule is checked on synthetic gate outcomes: the no-pass case, the tie-break, a
    level whose null fails its floor, and a saturated level;
  - the gap is never read by the selection.

## Registration and run

- [ ] 8. **Pre-launch checks**:
  - the full suite (`uv run pytest -q -m "not nightly"`);
  - `git add -A`, then `uv run pre-commit run --all-files`, judged by its exit code;
  - smoke: construct one arm per new level on both wirings, then one short `run_simulation.py` run
    each of PPO with fan-in and `measured`, and reading with `measured_signs`.
- [ ] 9. **The launch record**, `docs/experiments/logbooks/supporting/072-measured-prior-pilot/launch.md`,
  committed **before any seed runs**:
  - the arms, seeds, gate, metric rule, selection rule and branches;
  - the retention line;
  - the cost estimate.
- [ ] 10. **The campaign**:
  - `run_campaign.py` over both halves, 448 runs, with the output controls;
  - progress read with `campaign_progress.py`;
  - no branch switches until it completes.

## Records

- [ ] 11. **Score and commit** the per-seed CSV and the analysis JSON under the logbook's supporting
  directory.
- [ ] 12. **Logbook 072**, following the logbook skill: the branch taken per learner, the chosen
  multiplier per learner, the gates, and the descriptive gap. The PPO draw is stated beside every PPO
  figure.
- [ ] 13. **Discharge:**
  - the index row;
  - tracker B.1b ticked, and B.1c given the chosen multiplier per learner, the fan-in pairing, and
    any condition from branch 5, and the condition that B.1c's PPO arm inherits A.2's depth and
    initial-noise settings from a surface measured under `edge_order`;
  - a dated note at roadmap D17 and § B.1;
  - the `docs/architectures.md` row and a CHANGELOG line for the pairing.
- [ ] 14. **Close-out**: validate, archive and open the PR.
