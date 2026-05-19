# Evolution Framework — Delta for `add-transgenerational-memory-redesign` (M6.9+ PR-A)

## ADDED Requirements

### Requirement: Sensory-Conditional Substrate (TransgenerationalMemory.bias_network)

`TransgenerationalMemory` SHALL support an optional sensory-conditional parametric bias-network in addition to the legacy per-action constant `logit_bias` (M6 fallback). When `bias_network is not None`, the substrate's effective bias is computed as `bias_network(sensory_input)` rather than from the constant `logit_bias` tensor. When `bias_network is None`, behaviour SHALL remain byte-equivalent to the M6 path.

The frozen-dataclass invariant SHALL be preserved: when a caller passes an `nn.Sequential` to the constructor, `__post_init__` SHALL `copy.deepcopy` the module and reassign via `object.__setattr__` (mirrors the M6 `logit_bias.detach().clone()` pattern). Caller-side mutations to the source module SHALL NOT bleed into the stored substrate.

#### Scenario: bias_network field accepts a torch.nn.Sequential

- **WHEN** `TransgenerationalMemory(bias_network=nn.Sequential(...), lineage_depth=0, source_genome_id="g0")` is constructed
- **THEN** the substrate SHALL store the network reference without raising
- **AND** subsequent calls to `apply_to_logits(logits, sensory_input)` SHALL return `logits + bias_network(sensory_input)` (clamped to `|x| ≤ LOGIT_BIAS_CLAMP`)

#### Scenario: legacy logit_bias path is byte-equivalent when bias_network is None

- **WHEN** `TransgenerationalMemory(logit_bias=t, bias_network=None, lineage_depth=0, source_genome_id="g0")` is constructed
- **AND** `apply_to_logits(logits, sensory_input=anything)` is called
- **THEN** the substrate SHALL return `logits + self.logit_bias` (legacy M6 behaviour)
- **AND** the `sensory_input` argument SHALL be ignored

#### Scenario: bias_network output is clamped at LOGIT_BIAS_CLAMP

- **WHEN** the bias-network's raw output exceeds `LOGIT_BIAS_CLAMP = 6.0` in absolute magnitude on any action dimension
- **THEN** `apply_to_logits` SHALL clamp the output element-wise to `[-LOGIT_BIAS_CLAMP, +LOGIT_BIAS_CLAMP]` BEFORE adding to logits
- **AND** the clamp value SHALL be 6.0 (raised from the initial M6.9+ design value of 2.0 in pilot 3 to rule out logit saturation as the source of F1+ collapse; the wider clamp did not unlock substrate signal — see \[[019-transgenerational-memory-redesign]\])

#### Scenario: bias_network is deep-copied in `__post_init__`

- **WHEN** a caller passes an `nn.Sequential` to `TransgenerationalMemory(bias_network=net, ...)`
- **AND** subsequently mutates `net` (e.g. `net[0].weight.data.zero_()`)
- **THEN** the stored substrate's `bias_network` SHALL be unaffected by the mutation
- **AND** `apply_to_logits` results SHALL match those obtained at construction time

### Requirement: Configurable Decay Shape for Substrate Cascade

`TransgenerationalInheritance.inherit_from(parents, decay_factor, decay_shape)` SHALL support three decay shapes for the bias-network weights cascade:

- `geometric` (M6 default): cumulative scale at depth `d` is `decay_factor**d`. The per-generation factor is `decay_factor` regardless of depth — byte-equivalent to M6.
- `linear`: cumulative scale at depth `d` is `max(0, 1 − d × (1 − decay_factor))`. Reaches zero at `d = 1/(1 − decay_factor)`; the per-generation factor is `cum(d+1) / cum(d)`.
- `sigmoid`: cumulative scale at depth `d` is `sigmoid(K × (M − d))` with **fixed** `K = 2.0` and `M = 1.0`. The schedule is **independent of `decay_factor`** — it is a fixed-shape sensitivity-analysis alternative to geometric/linear, not a calibrated schedule. Monotonically decreasing; `cum(0) ≈ 0.881`, `cum(1) = 0.500`, `cum(2) ≈ 0.119`, `cum(3) ≈ 0.018`. Use sigmoid when the pilot pivot table flags "decay shape too aggressive under geometric collapse"; sigmoid concentrates most of the decay between depths 1 and 2.

#### Scenario: geometric decay multiplies weights by decay_factor per generation

- **WHEN** `inherit_from([parent], decay_factor=0.6, decay_shape="geometric")` is called on a parent at `lineage_depth=0`
- **THEN** the child's bias-network weight tensors SHALL each equal `parent_weights × 0.6`
- **AND** the child's `lineage_depth` SHALL equal 1

#### Scenario: linear decay reaches zero at lineage_depth = 1/(1−decay_factor)

- **WHEN** `inherit_from([parent], decay_factor=0.6, decay_shape="linear")` is called repeatedly
- **THEN** at `lineage_depth=2` the child's weights SHALL equal `parent_weights × max(0, 1 − 2 × 0.4) = parent_weights × 0.2`
- **AND** at `lineage_depth=3` the child's weights SHALL equal `0` (clipped at zero)

#### Scenario: sigmoid decay follows a fixed slow-then-fast schedule

- **WHEN** `inherit_from([parent], decay_factor=0.6, decay_shape="sigmoid")` is called repeatedly
- **THEN** the cumulative scale at depth `d` SHALL equal `sigmoid(2.0 * (1.0 − d))` (fixed `K=2`, `M=1`; `decay_factor` is intentionally ignored by sigmoid)
- **AND** the cumulative scale SHALL be monotonically decreasing in generation index
- **AND** `cum(0) ≈ 0.881`, `cum(1) = 0.500`, `cum(2) ≈ 0.119`, `cum(3) ≈ 0.018`

### Requirement: Env-Derived F0 Probe Ring

`EvolutionLoop._build_f0_probe_params` SHALL generate probe positions from the env's actual stationary-predator coordinates rather than synthetic `BrainParams` with zero predator gradient. For each predator, the builder SHALL emit `probe_ring.count` (default 8) probe positions on a **Manhattan-distance ring** (L1 perimeter) at exact distance `predator.damage_radius + probe_ring.radius_offset` from the predator center. The L1 ring is chosen so the spec's "distance equals damage_radius + radius_offset" invariant holds for every probe regardless of `count`; a Euclidean projection would produce variable Manhattan distances and contradict the gradient-strength formula (which uses L1).

#### Scenario: probe ring uses env predator positions

- **WHEN** `_build_f0_probe_params` is called with an env containing 5 stationary predators
- **AND** `probe_ring.count = 8` (default)
- **THEN** the returned probe list SHALL contain exactly 40 probes (5 predators × 8 ring positions)
- **AND** each probe's `predator_gradient_strength` SHALL be > 0
- **AND** each probe's `predator_gradient_direction` SHALL be derivable from `atan2(predator_y − probe_y, predator_x − probe_x)`

#### Scenario: configurable count and radius_offset

- **WHEN** `_build_f0_probe_params` is called with `probe_ring.count = 4` and `probe_ring.radius_offset = 2`
- **AND** the env contains 3 stationary predators with `damage_radius = 3`
- **THEN** the returned probe list SHALL contain 12 probes (3 × 4)
- **AND** the Manhattan distance from each probe to its source predator SHALL equal `3 + 2 = 5` exactly

#### Scenario: exact L1 distance at the default count

- **WHEN** `_build_f0_probe_params` is called with the default `probe_ring.count = 8`, `radius_offset = 1`
- **AND** the env contains a stationary predator with `damage_radius = 4`
- **THEN** all 8 probes SHALL sit at exact Manhattan distance `4 + 1 = 5` from the predator (no Euclidean approximation drift)

#### Scenario: pure helper for gradient computation

- **WHEN** `_compute_probe_gradient(probe_pos=(5, 5), predator_pos=(10, 10))` is called as a pure function
- **THEN** it SHALL return `(predator_gradient_strength, predator_gradient_direction)` deterministically
- **AND** `predator_gradient_strength` SHALL equal `1.0 / (1.0 + manhattan_distance((5, 5), (10, 10))) = 1.0 / 11.0`

#### Scenario: safe_probes extends probe set with zero-pathogen response-surface samples

- **WHEN** `_build_f0_probe_params` is called with a `probe_ring.safe_probes` sub-block configured
- **THEN** `_build_safe_probes(env, safe_probes_cfg, stam_state)` SHALL be invoked AFTER the ring probes are built
- **AND** the returned probes SHALL each have `predator_gradient_strength == 0.0` and varying `food_gradient_*` values that sweep the no-pathogen response surface
- **AND** the candidate cells SHALL satisfy `min(manhattan(cell, p) for p in env.predators) >= safe_probes.min_predator_distance`
- **AND** when no cell on the grid satisfies the distance constraint, `_build_safe_probes` SHALL log a warning and return an empty list (probe extraction proceeds with ring probes only — no hard failure)

### Requirement: F0 Bias-Network Fit

`TransgenerationalMemory.extract_from_brain` SHALL fit the `bias_network` MLP from the F0 elite's empirical action distribution at the env-derived probe ring. For each probe, sample `n=10` action selections deterministically on `extraction_seed`; compute the empirical action distribution; train the MLP for 50 Adam epochs over `(sensory_input → logit_offset_from_uniform)`. When `hidden_dim == 0`, use closed-form least-squares instead of Adam (linear projection).

#### Scenario: deterministic on extraction_seed

- **WHEN** `extract_from_brain(brain, probe_params, rng_seed=42, source_genome_id="g0")` is called twice with the same brain and probe set
- **THEN** the returned `TransgenerationalMemory.bias_network.state_dict()` SHALL be bit-identical across the two calls

#### Scenario: linear projection uses closed-form least-squares

- **WHEN** the `bias_network` config has `hidden_dim = 0`
- **AND** `extract_from_brain` is called
- **THEN** the projection weights SHALL be fit via least-squares (not Adam)
- **AND** the fit SHALL complete in a single deterministic pass

#### Scenario: MLP fit converges within 50 epochs

- **WHEN** the `bias_network` config has `hidden_dim ≥ 1`
- **AND** `extract_from_brain` is called
- **THEN** Adam SHALL run for exactly 50 epochs
- **AND** the final training loss SHALL be lower than the initial loss (sanity-check convergence)

## MODIFIED Requirements

### Requirement: Transgenerational Inheritance Strategy

The `TransgenerationalInheritance` strategy SHALL apply the substrate cascade using whichever substrate form is configured (`bias_network` MLP or legacy `logit_bias` tensor). All M6 behaviours (kind() literal, gen-0 capture, F1+ derivation, lineage CSV semantics) SHALL be preserved.

#### Scenario: bias_network cascade preserves decay invariants

- **WHEN** the F0 substrate has `bias_network` set and decay_factor 0.6 under `decay_shape: geometric`
- **AND** the cascade applies through F1/F2/F3
- **THEN** the F1 bias-network weights SHALL equal `F0_weights × 0.6`
- **AND** the F2 bias-network weights SHALL equal `F0_weights × 0.36`
- **AND** the F3 bias-network weights SHALL equal `F0_weights × 0.216`

#### Scenario: legacy logit_bias cascade unchanged

- **WHEN** the F0 substrate has `bias_network = None` and `logit_bias` set
- **AND** the cascade applies through F1/F2/F3 under `decay_shape: geometric` and `decay_factor: 0.6`
- **THEN** the F1 `logit_bias` SHALL equal `F0_logit_bias × 0.6` (M6 byte-equivalent)
- **AND** the cascade SHALL be identical to the M6 path

#### Scenario: substrate form persisted in .tei.pt round-trip

- **WHEN** a substrate with `bias_network` set is serialised via `save()` then loaded via `load()`
- **THEN** the loaded substrate SHALL have a functionally-equivalent `bias_network` (same architecture, same state_dict)
- **AND** subsequent `apply_to_logits` calls SHALL produce identical outputs

### Requirement: Cross-Arm Statistical Verdict (n=4 Noise-Aware)

The M6.9+ aggregator SHALL produce a cross-arm primary verdict using both a non-parametric and a bootstrap-resampled statistical test, requiring agreement on direction AND significance. The primary verdict SHALL be GO iff:

1. The `tei_on` arm passes its per-arm gate (F1 ≥ 40% × F0, F2 ≥ 25% × F0, F3 ≥ 15% × F0, monotone non-increasing), AND
2. The paired-seed delta `tei_on − control` on F1+ retention is statistically distinguishable from zero via BOTH:
   - one-sided Wilcoxon signed-rank with p < 0.10, AND
   - ≥ 5 percentage points absolute mean delta AND non-overlapping 80% bootstrap confidence intervals (1000 resamples per seed).

Both checks (Wilcoxon AND bootstrap) MUST agree on direction (both positive in favour of `tei_on`). A bare 5pp threshold without statistical agreement SHALL NOT pass the primary verdict (n=4 is noise-bounded; M5 used n=12 for credible signal).

#### Scenario: GO when both tests agree

- **WHEN** the four paired-seed deltas `tei_on − control` are `[+0.08, +0.10, +0.07, +0.09]` (mean +0.085)
- **AND** Wilcoxon one-sided p < 0.10
- **AND** the 80% bootstrap CI of the mean delta is `[+0.04, +0.13]` (non-overlapping with zero)
- **AND** `tei_on` passes its per-arm gate
- **THEN** the cross-arm primary verdict SHALL be GO

#### Scenario: STOP when delta < 5pp even if Wilcoxon significant

- **WHEN** the four paired-seed deltas are `[+0.02, +0.03, +0.02, +0.04]` (mean +0.0275)
- **AND** Wilcoxon one-sided p = 0.06 (significant)
- **THEN** the cross-arm primary verdict SHALL be STOP (the ≥ 5pp bound is the load-bearing filter)

#### Scenario: STOP when Wilcoxon insignificant even if delta ≥ 5pp

- **WHEN** the four paired-seed deltas are `[+0.20, +0.01, -0.05, +0.10]` (mean +0.065 but noisy)
- **AND** Wilcoxon one-sided p = 0.30 (insignificant)
- **THEN** the cross-arm primary verdict SHALL be STOP (noise-bounded; the bare mean is not load-bearing)

#### Scenario: STOP when bootstrap CI overlaps zero

- **WHEN** the four paired-seed deltas are `[+0.10, +0.10, -0.05, +0.10]` (mean +0.0625)
- **AND** Wilcoxon one-sided p < 0.10
- **AND** the 80% bootstrap CI is `[-0.02, +0.14]` (overlaps zero)
- **THEN** the cross-arm primary verdict SHALL be STOP (CI overlap precludes a credible directional claim)

#### Scenario: STOP when tei_on fails its per-arm gate

- **WHEN** `tei_on` survival at F1 is below 40% × F0 (per-arm gate FAIL)
- **AND** all cross-arm statistical checks pass
- **THEN** the cross-arm primary verdict SHALL be STOP (per-arm gate is the prerequisite — substrate must retain signal within its own arm before cross-arm comparison is meaningful)

### Requirement: PR-B Trigger Decision

When the cross-arm primary verdict is GO, the aggregator SHALL emit `pr_b_trigger.md` recommending the PR-B scaffold. When the primary verdict is STOP, the aggregator SHALL emit `m6_13_punt_note.md` documenting the null finding and noting that PR-B (`transgenerational+weights` symmetric-compute control) is deferred to M6.13+ unless follow-up evidence revives the hypothesis.

#### Scenario: PR-B trigger emitted on GO

- **WHEN** the cross-arm primary verdict is GO
- **THEN** the aggregator SHALL write `pr_b_trigger.md` to the output directory
- **AND** the file SHALL contain a one-line recommendation to scaffold OpenSpec change `add-transgenerational-memory-weights`

#### Scenario: punt note emitted on STOP

- **WHEN** the cross-arm primary verdict is STOP
- **THEN** the aggregator SHALL write `m6_13_punt_note.md` to the output directory
- **AND** the file SHALL document the null finding + the M6.13 deferral rationale

### Requirement: F0 Substrate Extraction Pipeline

The F0 substrate extraction pipeline SHALL use the env-derived probe ring when `probe_ring` config is set; SHALL persist `bias_network.input_features` in the `.tei.pt` payload so F1+ workers can validate sensory-input shape at load time.

#### Scenario: bias_network input_features persisted to .tei.pt

- **WHEN** an F0 substrate with `bias_network.input_features = ["predator_gradient_strength", "food_gradient_strength"]` is saved
- **THEN** the `.tei.pt` payload SHALL include the input_features list as a metadata field
- **AND** `load()` SHALL deserialise it intact

#### Scenario: F1+ load validates input_features against current brain

- **WHEN** an F1+ worker loads a `.tei.pt` substrate whose `input_features` includes a `BrainParams` field absent from the current brain config
- **THEN** the worker SHALL raise `RuntimeError` with an operator-friendly message naming the missing field
- **AND** the run SHALL fail-fast at substrate-load time, NOT at first `apply_to_logits` call

### Requirement: Substrate-Diversity Tripwire (T2 + T4)

The pre-flight calibration smoke SHALL include a substrate-diversity tripwire (T2) and substrate-magnitude tripwire (T4) that together catch the M6 failure mode (3-of-4 calibration seeds extracting bit-identical substrates from independent F0 elites). Both checks MUST pass before pilot is unblocked.

T2 — substrate diversity — SHALL compute pairwise coefficient-of-variation across each pair of calibration seeds' flattened `bias_network.state_dict()` tensors. The pairwise CoV definition is `||W_i − W_j||_2 / mean(||W_i||_2, ||W_j||_2)`. The minimum pairwise CoV across all pairs SHALL be compared against a configurable threshold (default 5%); below threshold is a T2 trip.

T4 — substrate magnitude — SHALL compute mean `|bias_network(probe)|` over a deterministic probe set (uniform in `[-1, 1]^len(input_features)`, seeded RNG). The minimum mean-abs-output across seeds SHALL be compared against a configurable threshold (default 0.1); below threshold is a T4 trip.

#### Scenario: pairwise CoV is zero for bit-identical substrates

- **WHEN** the diversity script flattens two byte-identical `bias_network.state_dict()` tensors
- **THEN** their pairwise CoV SHALL equal 0.0 (to within float64 precision)
- **AND** the substrate-diversity verdict SHALL fail with `diversity_pass=False`

#### Scenario: pairwise CoV is scale-invariant

- **WHEN** the diversity script computes pairwise CoV on two vectors AND on the same two vectors uniformly multiplied by a positive scalar
- **THEN** both CoVs SHALL be equal (to within float precision)
- **AND** the verdict SHALL not depend on overall substrate magnitude — only on relative direction differences

#### Scenario: minimum pairwise CoV across N seeds is the reported diversity metric

- **WHEN** N ≥ 2 seeds' substrates are passed to the diversity script
- **THEN** the script SHALL report `min_pairwise_cov` as the worst-case (smallest) CoV across all `N*(N-1)/2` pairs
- **AND** the verdict SHALL trip iff `min_pairwise_cov < diversity_threshold`

#### Scenario: T4 magnitude trip on zero-output bias-network

- **WHEN** a substrate's `bias_network` produces mean `|output|` below the magnitude threshold over the deterministic probe set
- **THEN** the script SHALL fail T4 with `magnitude_pass=False`
- **AND** the overall verdict SHALL be False (both T2 and T4 must pass)

#### Scenario: empty or single-seed input fails closed

- **WHEN** the diversity script is invoked with zero or one substrate
- **THEN** the diversity verdict SHALL be False (`diversity_pass=False`)
- **AND** the script SHALL exit with status code 1, NOT misleadingly pass through

#### Scenario: campaign-root discovery walks the canonical inheritance layout

- **WHEN** the diversity script is invoked with `--campaign-root <root> --arm tei_on`
- **THEN** it SHALL discover substrate files at `<root>/tei_on/seed-<N>/<session>/inheritance/gen-000/genome-*.tei.pt`
- **AND** missing seed directories or missing F0 substrates SHALL produce a warning + skipped seed, not a hard error
