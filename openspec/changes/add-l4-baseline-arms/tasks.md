# Tasks — L4 sanity-floor arms

## 1. Unmodulated Hebbian mode

- [x] 1.1 Add an unmodulated branch to the three-factor rule: `Δw = η·E − η·λ_w·w`, sharing masking, decay, clamping and the effective-change telemetry with the modulated path. The modulator is the only difference.
- [x] 1.2 Compute and report the prediction error and baseline in unmodulated mode even though the update discards them (Decision 3), so a mislabelled or non-applied arm is visible from telemetry.
- [x] 1.3 Extend the rule selection to a third value; carry the same activity-trace pairing validation as the modulated selection (the validator currently tests for the modulated value alone, so a new arm would otherwise skip the check and fail later at first update rather than at load).
- [x] 1.3a Report the **configured** rule in the PPO-only accessor error rather than the hard-coded modulated name, so an unmodulated arm is not told it is something it is not. Rename the per-step dispatch helper to match, since it now serves both plastic arms.
- [x] 1.4 Tests: update equals rate × trace before stabilisation; reward has no effect on resulting weights; the reward stream is still reported and matches what the modulated rule reports; modulated and unmodulated agree exactly when the prediction error is one; stabilisation and effective-change telemetry behave as in the modulated path.

## 2. Substrate parity across arms

- [x] 2.1 Ensure the unmodulated selection takes the anatomical readout, exactly as the modulated one does — the floors must decode identically to the arm they bound.
- [x] 2.2 Tests: plastic arm, frozen floor, and unmodulated floor constructed at one seed have byte-identical topology parameters including the readout; the frozen floor's weights are bit-identical after an episode; the unmodulated floor's weights change and are invariant to the reward stream.

## 3. Configurations

- [x] 3.1 Frozen-weights floor config: the plastic wild-type arm with updates frozen (plasticity rule, not the gradient rule — Decision 1). Name it so the plastic parent's full name stays a prefix, per the derived-variant convention.
- [x] 3.2 Unmodulated-Hebbian floor config: the plastic wild-type arm with the modulator off, named on the same rule.
- [x] 3.3 Tests: each floor config is a minimal delta from the plastic parent; both load and select their arm; the parent is unchanged.
- [x] 3.4 Add the plastic arm and both floors to the smoke-config list. No connectome config is currently smoke-tested at all, so the plastic code path — per-step dispatch, no value head, an empty rollout buffer — has never been exercised end to end through the run entry point. (Checked: nothing outside the brain reads the per-step value or the loss series, so this is insurance rather than a known break.)

## 4. Roadmap correction

- [x] 4.1 Correct D2's frozen-weights wording with a dated note: the connectome supplies edge existence; weights are drawn `N(0, 1/√(chemical in-degree))` and `syn.weight` never reaches `w_chem`. State the substrate as anatomically constrained in topology, randomly initialised in weight, and record that changing the initialisation would require re-baselining everything measured against it.
- [x] 4.2 Check for the same overstatement elsewhere in the roadmap, the tracker, and the logbooks; correct or leave with a reason.

## 5. Documentation

- [x] 5.1 `docs/architectures.md`: the unmodulated mode and what the two floors are for.
- [x] 5.2 `configs/README.md` variant vocabulary for the two floor suffixes.
- [x] 5.3 CHANGELOG entry.
- [x] 5.4 Correct the wording of the filed weight-persistence issue: the auto-save path logs at debug level rather than emitting nothing at all — invisible at the default level, but the issue should not overstate it.
- [x] 5.5 Tracker: tick A.4 with a dated note; **correct the over-claim** that the frozen arm needs weight persistence, and cross-reference the filed persistence issue against the arm it actually blocks (imitation warm-start).

## 6. Close-out

- [x] 6.1 `git add -A` **first**, then `uv run pre-commit run --all-files` green (hooks skip untracked files).
- [x] 6.2 Full non-nightly suite green.
- [x] 6.3 No implementation code or docstring references planning docs, roadmap IDs, OpenSpec changes, or logbooks.
- [ ] 6.4 Re-review for drift, archive, review the branch, open the PR.
