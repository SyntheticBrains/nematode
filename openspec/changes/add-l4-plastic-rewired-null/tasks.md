# Tasks — plastic rewired-null arm

## 1. Configuration

- [ ] 1.1 `connectomeppo_small_continuous2d_combined_klinotaxis_plastic_rewired_null.yml`: the plastic wild-type arm with `wiring: rewired_degree_preserving` and nothing else changed; named so the plastic parent stays a prefix.
- [ ] 1.2 Minimal-delta test: exactly one added key against the plastic parent; loads; selects the three-factor rule on rewired wiring; parent unchanged.
- [ ] 1.3 Smoke entry beside the other plastic arms.

## 2. Shared-substrate parity (Decision 2), tested against the real C3 plastic configs

- [ ] 2.1 Test: at one seed, readout, every sensory-projection gain (food, predator, thermotaxis — the C3 cell enables all three), and `log_std` are `torch.equal` between the plastic wild-type and plastic rewired-null brains.
- [ ] 2.2 Test: `m_chem` differs; `w_chem` differs; `g_gap` differs; chemical edge count and every neuron's in- and out-degree are equal.
- [ ] 2.3 Test: trace configuration and every plasticity hyperparameter are equal on both brains (read from the loaded configs).
- [ ] 2.4 Test: the rewiring is deterministic under the run seed, and a different seed gives a different mask — pairing holds per seed.
- [ ] 2.5 **Negative-space test**: per-neuron initial incoming weight energy is *not* asserted equal, and the test file says why, so a later reader does not "fix" the parity test by adding a claim that is false.

## 3. Confinement to the null wiring (Decision 1)

- [ ] 3.1 Test: after training the rewired arm, every changed `w_chem` entry lies on `m_chem`.
- [ ] 3.2 Test: every non-zero eligibility-trace entry lies on `m_chem` throughout training.
- [ ] 3.3 Test: the rewired arm learns (weights change) under the three-factor rule and does not under `freeze_updates` — the freeze holds on the null wiring too.
- [ ] 3.4 Test: `chemical_mask_mode` is `strict` on the loaded rewired config (inherited, not restated).

## 4. Documentation

- [ ] 4.1 `configs/README.md` variant vocabulary for `plastic_rewired_null`.
- [ ] 4.2 `docs/architectures.md`: the plastic wiring arms and what they hold constant.
- [ ] 4.3 CHANGELOG entry.
- [ ] 4.4 Tracker: tick A.6 with a dated note recording the ratified arm-set decision and the deferred rewired floors.

## 5. Close-out

- [ ] 5.1 `git add -A`, then `uv run pre-commit run --all-files` **unfiltered**, exit code 0.
- [ ] 5.2 Full non-nightly suite green.
- [ ] 5.3 No planning references in tests beyond spec-scenario citations; none in any implementation code (no implementation code changes here).
- [ ] 5.4 Re-review for drift, archive, review the branch, open the PR.
