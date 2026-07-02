# source-depletion-dynamics Specification

## Purpose

Config-gated within-episode source-depletion dynamics for area-restricted search: each food source
carries a **remaining amount** that drains as the agent feeds and scales the source's contribution to
the concentration field, so a grazed patch **flattens in place** (its near-source gradient vanishes)
until it is exhausted and removed. Disabled by default (byte-identical to the global-strength field).
It provides an environmental within-episode-memory demand — the biological twin of the artificial
bit-memory control; see [Logbook 032](../../../docs/experiments/logbooks/032-ars-source-depletion.md)
for the evaluation (the demand is short-horizon at current fidelity → null architecture separation).

## Requirements

### Requirement: Config-gated per-source depleting amplitude

The environment SHALL support a configuration-gated source-depletion dynamic, **disabled by
default**, in which each food source carries a **remaining amount** and its contribution to the food
concentration field scales with that amount. When the dynamic is disabled, the field SHALL behave
exactly as before (every source contributes the global configured strength), so existing and
discrete-grid configurations are byte-stable.

#### Scenario: Disabled by default leaves the field unchanged

- **WHEN** a configuration does not enable source-depletion
- **THEN** the food concentration field SHALL be computed exactly as before (global gradient strength per source), with no per-source amount affecting it

#### Scenario: Enabled, a source's contribution scales with its remaining amount

- **WHEN** source-depletion is enabled and a source has been partially depleted (remaining amount below its initial value)
- **THEN** that source's contribution to the concentration field SHALL be scaled by its remaining amount (a smaller bump than a full source at the same position)

### Requirement: Depletion is a once-per-step feeding event, not a sensing side effect

When source-depletion is enabled, a source's remaining amount SHALL be decremented **only** by a
food-consumption (feeding) event, applied **once per step** at the consume step, and SHALL NOT be
changed as a side effect of any sensory field read. Field reads SHALL be pure, so that sampling the
field multiple times in a step (for example the klinotaxis head-sweep, which reads the food
concentration twice) does not alter any source's amount.

#### Scenario: A consume event decrements the matched source

- **WHEN** the agent consumes a food source with source-depletion enabled
- **THEN** that source's remaining amount SHALL be decremented by the configured depletion quantum, exactly once for that consume event

#### Scenario: Sampling the field does not deplete

- **WHEN** the field is read one or more times in a step (scalar concentration, klinotaxis left/right samples, reward terms) without a consume event
- **THEN** no source's remaining amount SHALL change

### Requirement: In-place flattening and removal at exhaustion

A depleting source SHALL persist at its position with reduced amplitude until its remaining amount
crosses a removal threshold, at which point it SHALL be removed and the existing respawn contract
applies (subject to `no_respawn`). A source at or below the removal threshold SHALL NOT count as
consumable food (it SHALL NOT trigger a goal/reward).

#### Scenario: A partially-depleted source persists in place

- **WHEN** a source has been depleted but remains above the removal threshold
- **THEN** it SHALL remain at the same position with reduced field amplitude (its near-source gradient flattens), and SHALL still be consumable

#### Scenario: An exhausted source is removed

- **WHEN** a source's remaining amount crosses the removal threshold
- **THEN** it SHALL be removed, and a replacement SHALL spawn per the existing respawn policy unless `no_respawn` is set

#### Scenario: A depleted source is not consumable

- **WHEN** the agent is within the capture radius of a source whose remaining amount is at or below the removal threshold
- **THEN** no consumption / goal / reward SHALL fire for that source

#### Scenario: An exhausted source is absent from the foraging signals

- **WHEN** a source is exhausted (its remaining amount crosses the removal threshold and it is removed)
- **THEN** it SHALL be absent from every food signal — concentration, gradient, and nearest-food distance — so no foraging reward points at a spent patch (a *partially*-depleted source above the threshold remains valid food and continues to contribute to all signals)

### Requirement: Per-source amount integrity across mutations and copy

The per-source remaining-amount store SHALL stay index-aligned with the food-source positions across
every addition and removal, and SHALL be preserved by environment state-copy (a copied continuous-2D
environment SHALL carry the depletion state). The distance-zero concentration special case SHALL read
the **source's** remaining amount, not the global strength, when depletion is enabled.

#### Scenario: Amounts stay aligned across add/remove

- **WHEN** a food source is removed or a new one is spawned
- **THEN** the remaining-amount store SHALL stay index-aligned with the food-source positions

#### Scenario: Copy preserves depletion state

- **WHEN** a continuous-2D environment with depletion state is copied
- **THEN** the copy SHALL carry the same per-source remaining amounts as the source environment

#### Scenario: Distance-zero reads the source amount

- **WHEN** depletion is enabled and the agent is exactly at a depleted source's position
- **THEN** the concentration there SHALL reflect that source's reduced remaining amount, not the global strength

### Requirement: Area-restricted-search within-episode-memory evaluation

The depleting field SHALL be evaluable as a within-episode-memory demand: an evaluation runs the
architecture panel on a depletion-enabled foraging cell and reports each arm's plateau-tail foraging
success across paired seeds using the committed paired-seed statistics layer, testing whether the
recurrent/attention arms separate from the memoryless MLP (the biological twin of the bit-memory
separation). A null result SHALL be reported as such.

#### Scenario: Separation is reported when memory arms out-forage the memoryless arm

- **WHEN** the panel is evaluated on a well-calibrated depleting cell over paired seeds
- **THEN** the evaluation SHALL report per-arm plateau-tail foraging success and the pairwise deltas with BH-FDR-adjusted significance, and a separation verdict when the recurrent/attention arms significantly exceed the memoryless MLP

#### Scenario: A null result is reported as such

- **WHEN** the recurrent/attention arms do not significantly exceed the memoryless MLP on the depleting cell
- **THEN** the evaluation SHALL report a null verdict (environmental depletion alone did not induce a separable within-episode-memory demand at this fidelity) rather than a separation
