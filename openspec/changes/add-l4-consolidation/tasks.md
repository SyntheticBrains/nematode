# Tasks: consolidation mechanisms for the local rule

## 1. Configuration

- [ ] 1.1 Add the consolidation selector and its parameters to the shared plasticity mixin with
  load-time bounds: selector `none|anchor|rigidity|oracle`, anchor rate `[0, 1)`, stiffness `≥ 0`,
  rigidity growth `≥ 0`, decay `[0, 1)`, strength `≥ 0`, oracle reference `(0, 1]`, oracle rate
  `(0, 1]`.
- [ ] 1.2 Validator: a selector other than `none` whose own parameters leave it inert is rejected,
  with a message naming the parameter.
- [ ] 1.3 Tests: each bound; the inert-mechanism rejection for `anchor` and `rigidity`; the
  default config is unchanged field for field from today's.

## 2. The elastic anchor

- [ ] 2.1 Allocate one anchor per plastic tensor, initialised to the constructed weights, only
  when `anchor` is selected.
- [ ] 2.2 Apply `− η · κ_a · (w − a)` inside the masked update beside the decay; advance the
  anchor after the write.
- [ ] 2.3 Tests: the restoring term's sign and magnitude; an anchor rate of zero holds the anchor;
  a positive rate moves it by exactly the configured fraction; nothing is written off the edge set.

## 3. Reinforced rigidity

- [ ] 3.1 Allocate one protective variable per plastic tensor, zero at construction, only when
  `rigidity` is selected.
- [ ] 3.2 Divide the Hebbian term's rate by `1 + κ_c · c` using the pre-growth value; advance
  `c ← (1 − λ_c) c + γ_c · max(m, 0) · |E|` after the update.
- [ ] 3.3 Tests: rigidity grows where a positive modulator meets a large trace and not where the
  modulator is negative; the pre-growth divisor; decay toward zero without reinforcement; the
  unmodulated arm accumulates on the trace alone.

## 4. The oracle gate

- [ ] 4.1 Accept an episode-success flag on the rule and maintain the trailing rate; scale the
  plasticity rate by `clamp(1 − s / s_ref, 0, 1)`.
- [ ] 4.2 Forward the flag from the brain's existing `post_process_episode(episode_success=...)`
  hook; the hook stays a no-op under every other selector.
- [ ] 4.3 State in the implementation that this consumes the task's scoring rather than the
  reward stream, and is a diagnostic bound rather than a mechanism.
- [ ] 4.4 Tests: the rate reaches zero at the reference and is unchanged far below it; the flag
  reaches the rule through the brain hook; no other selector reads it.

## 5. State lifecycle and telemetry

- [ ] 5.1 `reset_state()` re-anchors the anchor to the current weights and zeroes the protective
  variable, beside the existing resets.
- [ ] 5.2 Report the effective rate multiplier, the mean anchor departure and the mean protective
  variable; record them in the shared plasticity report.
- [ ] 5.3 Tests: a loaded clone is anchored to itself and its first restoring term is zero; the
  protective variable does not survive a load; consolidation state is not written to a checkpoint;
  the telemetry keys are present under every selector with the documented inactive values.

## 6. Byte-identity and integration

- [ ] 6.1 Test: with the selector `none`, a fixed-seed run's weight trajectory is bit-identical to
  the same run before this change.
- [ ] 6.2 Test: consolidation composes with grounded signs and homeostasis in the registered
  order, with the clamp last.
- [ ] 6.3 Smoke-test entries for one consolidated config.

## 7. The screen

- [ ] 7.1 Clone-arm configs for the three variants, deriving from the existing wild-type plastic
  clone config with the rule keys and nothing else changed.
- [ ] 7.2 A screen harness reading the campaign directory and reporting, per variant, the eight
  per-seed plateau tails, the mean delta against the committed frozen-clone values, the count at
  or above, the endpoint cosine to the clone, and the mean effective rate multiplier.
- [ ] 7.3 Tests for the harness: the hold and improve rules at their boundaries; a missing run is
  reported and never imputed; the comparator values are the committed ones.
- [ ] 7.4 Run the declared pilot (seeds 1–2 over the grid) and write its grid, criterion and pins
  into the launch record.
- [ ] 7.5 Commit the launch record, then run the three screens (seeds 1–8, 2000 episodes).
- [ ] 7.6 Records under `docs/experiments/logbooks/supporting/045-l4-consolidation/`: `launch.md`,
  `screen.json`, `per-seed.csv`, `_manifest.txt`, `details.md`.

## 8. Documentation

- [ ] 8.1 `docs/architectures.md` and `configs/README.md`: the selector, its parameters and the
  oracle's status as a bound rather than a mechanism.
- [ ] 8.2 `CHANGELOG.md`.
- [ ] 8.3 Tracker and roadmap updated with the screen's outcome at close-out.
