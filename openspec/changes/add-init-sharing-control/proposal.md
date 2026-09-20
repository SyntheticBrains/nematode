## Why

Phase 8's **A.1**, the control Logbook 069 named as the phase's first act, and the one published critique aimed at this design.

Block V's headline — the wild-type connectome reaching competence **23–55% sooner** than its degree-preserving rewired null under PPO, replicated on fresh rewirings — ships with a standing condition: rewiring and initialisation vary together. [Dhiman 2026](https://arxiv.org/abs/2604.04033) reports exactly that advantage dissolving in the fly under shared initialisation plus a degree-preserving null. Until this control runs, a referee reads Block V and Dhiman together and reaches for the second.

**Reading the implementation first narrows the question, and the decision it feeds was corrected because of it.** Weight initialisation is a per-edge draw of 3,709 Gaussian values in `(pre, post)` sorted order, scaled per post-synaptic neuron by `1/sqrt(chemical in-degree)`. Both graphs carry exactly 3,709 edges, so they consume the **same standard-normal stream** and the *n*-th value is identical in both. Rewiring already runs on a dedicated generator so the weight stream is untouched, and a committed test already asserts the readout and food gains are byte-identical across wirings at one seed.

So the confound is not "initialisation varies". It is **which edge the *n*-th drawn value lands on**, and the per-edge scale sequence that follows. That is what D15 names, and it is what this change controls.

## What Changes

### 1. Two weight-draw modes on the connectome brain

A new `weight_draw` config field, `edge_order` (today's behaviour, bit-identical) plus:

- **`dense_mask`** (D15 i) — one dense draw, then each edge takes its own cell scaled per post-synaptic neuron, so **every edge present in both graphs carries the identical value**.
- **`per_neuron_fanin`** (D15 ii) — each post-synaptic neuron's fan-in values are drawn together and assigned to its incoming edges in pre-synaptic-index order, so **every neuron receives the identical multiset of incoming weights** in both graphs and only the pairing differs.

The two definitions exist because neither is uniquely "the same initialisation" once the mask changes; running both is the honest answer, and either is a stronger control than the critique's own (a shared seed, three optimisation seeds, five rewirings).

### 2. A crossed panel, read as an interaction

Wiring {wild type, rewired null} × draw {`edge_order`, `dense_mask`, `per_neuron_fanin`}, on both block-V cells, **16 paired seeds**, fresh at 129–144. The primary is the **interaction** of draw mode with wiring on `episodes_to_30pct_success` — per the requirement that a manipulation crossed with a structure contrast is read as an interaction and never as a main effect. Frozen arms run at every mode, because the untrained prior is what shows whether a draw mode moves the arms before any learning.

384 runs, roughly 5–6 hours at 16 workers.

### 3. Scope, decided deliberately

A.1 answers the **pairing** question only. The across-seed coupling — the null's graph and weights both deriving from the run seed — is a different experiment, which [Logbook 065](../../../docs/experiments/logbooks/065-wiring-fresh-rewiring.md) already calls one, and it is recorded as a follow-up rather than mixed in. `rewire_seed` stays unset and is documented as equal to the run seed, which preserves V.4's fresh-rewiring property.

The `edge_order` baseline is **re-run fresh** rather than reused from V.4: the reuse rule forbids partial reuse and requires a parsed-field identity check, and PR #397 bumped five dependencies since those runs. Running all three levels on one seed set in one campaign makes the interaction a within-campaign contrast rather than a comparison across campaigns and dependency versions.

### 4. The instrument is not touched

Scoring goes through the **unmodified** `wiring_premise.py` and `connectome_structure_efficiency.py`. A new driver builds manifests and reports branches, in the mould `wiring_fresh_rewiring.py` established — new code confined to preparing inputs and reporting outputs, which is what the replication requirement permits. Editing the instrument's hard-coded cell and family tuples would forfeit the replication property V.4 rests on.

### 5. A.0 rides along

The artefact-retention rule, which must be recorded before the first Phase 8 campaign runs.

## Capabilities

**Modified**: `architecture-comparison-protocol` — **one** added requirement: a claim that two arms share an initialisation is verified by test rather than argued from construction.

One, not several. A.4 has just finished redistributing 38 single-use rules that accumulated one per rung; the discipline this change inherits is to add a requirement only where no existing one covers the case. The seven that already govern this campaign — interaction reading, power in advance, seed coupling, replication instrument, multi-panel disagreement, committed-baseline reuse, standing conditions — are cited in `design.md` rather than restated.

## Impact

**Code:**

- `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py` — the config field, the constructor parameter, the draw branch, the load-time guard
- `scripts/analysis/init_sharing_control.py` — new driver

**Tests:**

- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_connectome_weight_draw.py` — new
- `packages/quantum-nematode/tests/quantumnematode_tests/analysis/test_init_sharing_control.py` — new

**Configs:** 16 new YAMLs, one key off their parent, across four arms × two new modes × two cells. The eight committed block-V configs serve the `edge_order` level unchanged, which also preserves the panel-commit identity their test pins.

**Docs:** `openspec/changes/phase8-tracking/tasks.md` (A.0, A.1); a logbook at the close.

## Breaking Changes

None. `weight_draw` defaults to `edge_order`, which is bit-identical to the pre-option brain.

## Backward Compatibility

Every committed config and every prior result is unaffected. The combination of a new draw mode with `weight_init: count_scaled` is refused at load rather than silently defined, since the two settings are orthogonal in principle but the pairing is untested and unused.
