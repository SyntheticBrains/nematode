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

- **WHEN** the bias-network's raw output exceeds `LOGIT_BIAS_CLAMP = 2.0` in absolute magnitude on any action dimension
- **THEN** `apply_to_logits` SHALL clamp the output element-wise to `[-LOGIT_BIAS_CLAMP, +LOGIT_BIAS_CLAMP]` BEFORE adding to logits

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

`EvolutionLoop._build_f0_probe_params` SHALL generate probe positions from the env's actual stationary-predator coordinates rather than synthetic `BrainParams` with zero predator gradient. For each predator, the builder SHALL emit `probe_ring.count` (default 8) probe positions in a ring at distance `probe_ring.radius_offset + predator.damage_radius` from the predator center.

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
- **AND** the distance from each probe to its source predator SHALL equal `3 + 2 = 5`

#### Scenario: pure helper for gradient computation

- **WHEN** `_compute_probe_gradient(probe_pos=(5, 5), predator_pos=(10, 10), max_dist=20)` is called as a pure function
- **THEN** it SHALL return `(predator_gradient_strength, predator_gradient_direction)` deterministically
- **AND** `predator_gradient_strength` SHALL equal `1.0 / (1.0 + manhattan_distance((5, 5), (10, 10))) = 1.0 / 11.0`

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
