# Transgenerational Memory Capability

## ADDED Requirements

### Requirement: Transgenerational Memory Substrate

The system SHALL provide a `TransgenerationalMemory` dataclass in `quantumnematode/agent/transgenerational_memory.py` representing an inheritable behavioural-bias substrate. The dataclass SHALL expose:

- A `logit_bias: torch.Tensor` field of shape `(num_actions,)` and dtype `float32`, representing an additive per-action bias on actor logits. The size SHALL match the brain's `num_actions` (4 for the default `DEFAULT_ACTIONS` set in `brain/actions.py:20`; would be 6 for the `SIX_ACTIONS` set). The dataclass SHALL accept any 1-D `num_actions` size at construction (the size is not hardcoded), so a future config selecting `SIX_ACTIONS` does not require dataclass changes.
- A `lineage_depth: int` field recording the inheritance generation (0 for F0, incremented by 1 at each `inherit_from` call).
- A `source_genome_id: str` field identifying the F0 elite from which the substrate originated.

The dataclass SHALL clamp `|logit_bias[i]| ≤ 2.0` in `__post_init__` (post-clamp Boltzmann ratio bounded at `e^2 ≈ 7.4×`) so a strong bias cannot collapse exploration. The dataclass SHALL be `frozen=True` so cross-generation aliasing cannot mutate an ancestor's substrate. Because frozen dataclasses cannot reassign fields directly, the clamping pass SHALL use `object.__setattr__(self, "logit_bias", clamped_tensor)` inside `__post_init__` (the canonical Python pattern for frozen-dataclass post-init mutation).

#### Scenario: Substrate construction clamps bias values

- **GIVEN** an attempt to construct `TransgenerationalMemory` with `logit_bias = torch.tensor([5.0, -3.0, 0.5, 1.0])` (4-action default)
- **WHEN** `__post_init__` runs
- **THEN** the stored `logit_bias` SHALL equal `torch.tensor([2.0, -2.0, 0.5, 1.0])` (each element clamped to `[-2.0, 2.0]`)
- **AND** the original input tensor SHALL NOT be mutated in place

#### Scenario: Substrate shape is validated

- **GIVEN** an attempt to construct `TransgenerationalMemory` with `logit_bias.ndim != 1` or with `logit_bias.dtype != torch.float32`
- **WHEN** `__post_init__` runs
- **THEN** a `ValueError` SHALL be raised stating the expected shape and dtype
- **AND** the error message SHALL include the offending tensor's actual shape and dtype

### Requirement: Substrate Decay at Generation Boundary

`TransgenerationalMemory` SHALL provide an `inherit_from(parents: Sequence[TransgenerationalMemory], decay_factor: float) -> TransgenerationalMemory` class method (or module-level factory) that produces a child substrate as `child.logit_bias = parents[0].logit_bias * decay_factor` (top-1 elite semantics, mirroring `LamarckianInheritance`'s single-elite broadcast). The child SHALL inherit `source_genome_id` from the parent and SHALL increment `lineage_depth` by 1.

The decay SHALL be applied multiplicatively at the generation boundary inside `inherit_from`, NOT inside `prepare_episode` or per-step logic. This SHALL prevent re-decaying the same parent's substrate every episode within a generation.

#### Scenario: Single-parent decay produces geometric retention

- **GIVEN** an F0 substrate with `logit_bias = torch.tensor([0.0, 1.0, -0.5, 0.2])` (4-action default), `lineage_depth = 0`, `source_genome_id = "gid-a"`
- **WHEN** `inherit_from([f0_substrate], decay_factor=0.6)` is called once to produce F1, again on the F1 result to produce F2, and again to produce F3
- **THEN** F1 `logit_bias` SHALL be `torch.tensor([0.0, 0.6, -0.3, 0.12])` and `lineage_depth` SHALL be 1
- **AND** F2 `logit_bias` SHALL be `torch.tensor([0.0, 0.36, -0.18, 0.072])` and `lineage_depth` SHALL be 2
- **AND** F3 `logit_bias` SHALL be `torch.tensor([0.0, 0.216, -0.108, 0.0432])` and `lineage_depth` SHALL be 3
- **AND** every descendant SHALL have `source_genome_id = "gid-a"`

#### Scenario: Decay factor out of range is rejected

- **GIVEN** `inherit_from([parent], decay_factor)` called with `decay_factor < 0.0` or `decay_factor > 1.0`
- **WHEN** the method runs
- **THEN** a `ValueError` SHALL be raised stating the valid range
- **AND** the message SHALL show the offending value

#### Scenario: Empty parents list raises clearly

- **GIVEN** `inherit_from([], decay_factor=0.6)` (no parents)
- **WHEN** the method runs
- **THEN** a `ValueError` SHALL be raised stating that at least one parent substrate is required

### Requirement: Logit-Bias Application

`TransgenerationalMemory` SHALL provide an `apply_to_logits(logits: torch.Tensor) -> torch.Tensor` method that returns `logits + self.logit_bias` (broadcast over leading batch / sequence dimensions). The method SHALL NOT mutate the input tensor in place. The method SHALL preserve the input tensor's shape, dtype, and device.

#### Scenario: Logits are augmented additively without mutation

- **GIVEN** a substrate with `logit_bias = torch.tensor([0.5, -0.5, 1.0, 0.0])` and logits `torch.tensor([[1.0, 2.0, 3.0, 0.0]])` (4-action default)
- **WHEN** `apply_to_logits` is called
- **THEN** the returned tensor SHALL equal `torch.tensor([[1.5, 1.5, 4.0, 0.0]])`
- **AND** the input logits tensor SHALL be unchanged (no in-place mutation)
- **AND** the returned tensor SHALL be a distinct tensor object (not an alias)

#### Scenario: Apply preserves shape and dtype across batch dimensions

- **GIVEN** a substrate with `logit_bias` of shape `(4,)` and logits of shape `(batch, seq, 4)` and dtype `float32`
- **WHEN** `apply_to_logits` is called
- **THEN** the returned tensor SHALL have shape `(batch, seq, 4)` and dtype `float32`
- **AND** the bias SHALL be broadcast across the batch and sequence dimensions

### Requirement: F0 Telemetry-Pass Extraction

The system SHALL provide an `extract_from_brain(brain, env, probe_positions, rng_seed) -> TransgenerationalMemory` function that runs the F0 elite policy on a deterministic set of probe positions near a pathogen lawn and produces an `logit_bias` reflecting the F0 policy's action-probability deviation from a no-lawn baseline.

The extraction SHALL run AFTER fitness evaluation completes (so the F0 fitness score is computed from raw episode traces, not from the telemetry pass). The telemetry pass SHALL use a deterministic RNG seed (configurable, defaulting to a fixed sentinel) so the extraction is reproducible. The probe positions SHALL be a deterministic set built relative to the lawn position (configurable, defaulting to a ring at fixed offsets).

#### Scenario: Extraction is deterministic for a given seed

- **GIVEN** an F0 elite brain, env, and probe positions
- **WHEN** `extract_from_brain` is called twice with the same `rng_seed`
- **THEN** the two returned `TransgenerationalMemory` instances SHALL have `logit_bias` tensors equal element-wise

#### Scenario: Extraction is computed AFTER fitness evaluation

- **GIVEN** an F0 generation completing fitness evaluation
- **WHEN** the per-generation post-eval hook fires
- **THEN** fitness scores SHALL have been recorded from raw episode traces BEFORE the telemetry pass runs
- **AND** the telemetry pass SHALL use disjoint episode rollouts from those that produced the fitness scores

### Requirement: Substrate Serialisation Round-Trip

`TransgenerationalMemory` SHALL serialise to disk via `torch.save`/`torch.load` over a `.tei.pt` file extension. The deserialised instance SHALL be byte-equivalent to the original — `logit_bias` tensor equal element-wise, `lineage_depth` equal, `source_genome_id` equal.

#### Scenario: Round-trip preserves all fields

- **GIVEN** a `TransgenerationalMemory` with `logit_bias = torch.tensor([0.5, -0.3, 1.0, 0.0])` (4-action default), `lineage_depth = 2`, `source_genome_id = "gid-elite-3"`
- **WHEN** the substrate is saved to `<tmp>/foo.tei.pt` and re-loaded
- **THEN** the loaded instance SHALL satisfy `torch.equal(loaded.logit_bias, original.logit_bias)`
- **AND** `loaded.lineage_depth == 2`
- **AND** `loaded.source_genome_id == "gid-elite-3"`

#### Scenario: Loading a missing file raises FileNotFoundError

- **GIVEN** a path that does not exist
- **WHEN** the loader is called
- **THEN** a `FileNotFoundError` SHALL be raised with the missing path in the message

### Requirement: F0 Calibration Pre-Flight Gate

The system SHALL provide an F0-calibration smoke target in the M6 campaign shell that runs an F0-only single-generation pass (no F1/F2/F3) and reports the mean F0 avoidance choice index. The smoke target SHALL be invocable independently of the pilot or full campaign (e.g., `phase5_m6_transgenerational_lstmppo_klinotaxis.sh --smoke`).

The calibration SHALL be treated as a hard pre-flight gate before the M6.5 pilot or full campaign is unblocked: the mean F0 choice index from the smoke pass SHALL fall within `[0.45, 0.85]` inclusive. Values below `0.45` indicate F0 is at the chance floor (gate ratios uninterpretable); values above `0.85` indicate F0 is ceiling-saturated (F1 will inherit ~ceiling regardless of substrate). The implementer SHALL retune `damage_radius` and `ppo_train_episodes` until the smoke passes the gate before progressing M6.5 sub-tasks.

#### Scenario: Calibration smoke target reports F0 choice index

- **GIVEN** the M6 campaign shell with `--smoke` mode
- **WHEN** the calibration target runs (1 seed × pop 6 × F0-only × ~50 episodes)
- **THEN** the output SHALL include a single-line summary stating `mean_f0_choice_index = <value>`
- **AND** the output SHALL state whether the value falls within the `[0.45, 0.85]` calibratable envelope
- **AND** the exit code SHALL be 0 regardless of envelope outcome (the gate is operator-actioned, not script-actioned, so the operator can choose to retune or accept)

#### Scenario: Calibration outcome is documented in pilot/full pre-flight

- **GIVEN** an M6.5 pilot or full campaign launch
- **WHEN** the pilot/full driver is invoked
- **THEN** the driver SHALL log the latest recorded calibration outcome (F0 mean choice index and envelope pass/fail) at startup
- **AND** the operator SHALL have acknowledged the calibration value in the campaign-shell preamble before pilot/full begins (manual confirmation; the script SHALL print a notice but SHALL NOT block)
