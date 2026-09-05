# Tasks — plastic rewired-null arm

## 1. Configuration

- [x] 1.1 `connectomeppo_small_continuous2d_combined_klinotaxis_plastic_rewired_null.yml`: the plastic wild-type arm with `wiring: rewired_degree_preserving` and nothing else changed; named so the plastic parent stays a prefix.
- [x] 1.2 Minimal-delta test: exactly one added key against the plastic parent; loads; selects the three-factor rule on rewired wiring; parent unchanged.
- [x] 1.3 Smoke entry beside the other plastic arms.

## 2. Shared-substrate parity (Decision 2), tested against the real C3 plastic configs

- [x] 2.1 Test: at one seed, readout, every sensory-projection gain (food, predator, thermotaxis — the C3 cell enables all three), and `log_std` are `torch.equal` between the plastic wild-type and plastic rewired-null brains. This supersedes the default-config `test_readout_identical_across_wiring_arms` in `test_connectome_learning_rule.py` at the real configs and across all five gain matrices; leave that test in place, do not duplicate its readout-only assertion here.
- [x] 2.2 Test: `m_chem` differs; `w_chem` differs; `g_gap` differs; chemical edge count, every neuron's chemical in- and out-degree, and every neuron's gap-junction degree are equal — asserted at the **brain level on the real C3 plastic configs**. The rewiring function itself is already proved by `connectome/test_rewiring.py` (`test_preserves_chemical_in_out_degree`, `test_preserves_gap_degree`, `test_no_self_loops_or_duplicates`, `test_preserves_neuron_set_and_edge_counts`); do not re-test it, test what the brain built from it.
- [x] 2.3 Test: trace configuration and every plasticity hyperparameter are equal on both brains (read from the loaded configs).
- [x] 2.4 Test: the rewiring is deterministic under the run seed, and a different seed gives a different mask — pairing holds per seed. "Differs" is probabilistic in principle (034: "with overwhelming probability"), so **pin the seeds**: 23 and 24 are verified to differ; do not draw arbitrary pairs. Determinism at the function level is already `test_deterministic_under_same_seed`; this asserts it through the brain's default `rewire_seed` derivation from the run seed.
- [x] 2.5 **Negative-space test**: per-neuron initial incoming weight energy is *not* asserted equal, and the test file says why, so a later reader does not "fix" the parity test by adding a claim that is false.

## 3. Confinement to the null wiring (Decision 1)

- [x] 3.1 Test: after training the rewired arm, every changed `w_chem` entry lies on `m_chem`.
- [x] 3.2 Test: every non-zero eligibility-trace entry lies on `m_chem` throughout training.
- [x] 3.3 Test: the rewired arm learns (weights change) under the three-factor rule and does not under `freeze_updates` — the freeze holds on the null wiring too.
- [x] 3.4 Test: `chemical_mask_mode` is `strict` on the loaded rewired config (inherited, not restated).

## 4. Documentation

- [x] 4.1 `configs/README.md` variant vocabulary for `plastic_rewired_null`.
- [x] 4.2 `docs/architectures.md`: the plastic wiring arms and what they hold constant.
- [x] 4.3 CHANGELOG entry.
- [x] 4.4 Tracker: tick A.6 with a dated note recording the ratified arm-set decision and the deferred rewired floors.

## 5. Close-out

- [ ] 5.1 `git add -A`, then `uv run pre-commit run --all-files` **unfiltered**, exit code 0.
- [ ] 5.2 Full non-nightly suite green.
- [ ] 5.3 No planning references in tests beyond spec-scenario citations; none in any implementation code (no implementation code changes here).
- [ ] 5.4 Re-review for drift, archive, review the branch, open the PR.
