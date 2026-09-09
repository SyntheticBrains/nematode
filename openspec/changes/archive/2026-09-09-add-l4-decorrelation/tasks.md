# Tasks: a decorrelating term for the local rule

## 1. Configuration

- [x] 1.1 Add the decorrelation selector (`none|anti_hebbian_inhibitory|oja`) and the Oja
  coefficient to the shared plasticity mixin, with load-time bounds.
- [x] 1.2 Validators: on the mixin, `oja` with a zero coefficient is rejected naming the
  coefficient; on the connectome config beside the synapse-sign validator, and duplicated as a
  brain-construction guard, `anti_hebbian_inhibitory` requires `synapse_signs: atlas`; the MLP
  config rejects `anti_hebbian_inhibitory` outright.
- [x] 1.3 Tests: every rejection, including the construction guard on a `model_copy`-derived
  config; the default config is unchanged field for field.

## 2. Anti-Hebbian inhibitory plasticity

- [x] 2.1 Negate the Hebbian term where the grounded sign is inhibitory; leave grounded excitatory
  and ungrounded synapses untouched.
- [x] 2.2 Tests: the negation is exact against the same step with the selector off; excitatory and
  ungrounded entries are untouched; the term's total magnitude is unchanged; nothing is written
  off the edge set.

## 3. Oja decorrelation

- [x] 3.1 Apply `− η · γ · y² · w` inside the masked update beside the decay, broadcasting the
  post-synaptic activity along each weight's post-synaptic axis.
- [x] 3.2 Extend the `PlasticTopology` seam with `plastic_post_activities`: the connectome exposes
  a view over the activity buffer its trace update keeps; the MLP topology retains each layer's
  post-activation at trace time. The rule reads it only under `oja`.
- [x] 3.2b Tests: both topologies satisfy the extended seam; each vector's length equals the
  weight's extent along the axis complementary to the fan-in axis; the vector equals the
  post-synaptic factor the trace was just built from; the connectome's is a view, not a copy.
- [x] 3.3 Tests: the term's sign and magnitude; an inactive unit receives none; a zero weight
  receives none; the MLP substrate takes the same path.

## 4. Signs without enforcement

- [x] 4.1 Hand the sign vector to the rule whenever signs are grounded, independent of Dale's law.
- [x] 4.2 Tests: a grounded brain's rule holds the signs with enforcement off; an ungrounded
  brain's does not; enforcement and the anti-Hebbian variant compose.

## 5. Telemetry

- [x] 5.1 Report the decorrelating term's share of the update's total absolute magnitude; record
  it in the shared plasticity report.
- [x] 5.2 Tests: the share is zero under `none`, positive under each variant, and present in the
  history record.

## 6. Byte-identity and integration

- [x] 6.1 Test: with the selector `none`, a fixed-seed run's weight trajectory is bit-identical.
- [x] 6.2 Test: the variants compose with consolidation, Dale's law and homeostasis in the
  registered order, with the clamp last.
- [x] 6.3 Smoke-test entries for one decorrelated config.

## 7. The registered test

- [x] 7.1 Four arm configs: wild-type and rewired-null grounded Hebbian under each variant, each
  one key block off its committed parent.
- [x] 7.2 An analysis harness fixing the four-arm registry, the D1–D4 family, the ordered verdict
  map, the `full_recovery` and decorrelation-share annotations, and the extension list, reading
  the sign-grounding test's committed per-seed table as the comparator.
- [x] 7.3 Tests for the harness: each verdict branch including `no_recovery`; the comparator is the
  committed table; a missing run is reported and never imputed; the annotations cannot change the
  verdict.
- [x] 7.4 Run the declared pilot for the Oja coefficient (seeds 1–2, `γ ∈ {0.01, 0.1, 1.0}`) and
  write its grid, criterion and pin into the launch record; the anti-Hebbian variant has no
  hyperparameter and records that instead.
- [x] 7.5 Commit the launch record, then run the four arms (seeds 1–16, 1000 episodes).
- [x] 7.6 Records under `docs/experiments/logbooks/supporting/046-l4-decorrelation/`: `launch.md`,
  `panel.json`, `per-seed.csv`, `curves.csv`, `_manifest.txt`, `details.md`.

## 8. Documentation

- [x] 8.1 `docs/architectures.md`: the selector, the two variants, and that the anti-Hebbian one
  requires grounded signs.
- [x] 8.2 `CHANGELOG.md`.
- [x] 8.3 Tracker and roadmap updated with the verdict at close-out.
