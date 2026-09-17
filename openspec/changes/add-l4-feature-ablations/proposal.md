# Which part of the wiring carries the effect? (L.4 + L.5)

## Why

[L.1](../../../docs/experiments/logbooks/066-l4-readout-width.md) returned `pooling_hid_structure`:
crossing readout width with wiring on `hard350` over 96 paired seeds, the interaction is **+0.2818**
on `auc_success` at q = 0.000 — at the pooled width the degree-preserving null is ahead, at the
per-neuron width the wild type is ahead, and no width main effect was detected. It is **the first
positive for the wild-type wiring under a biologically plausible learner in this phase**, and it
reopened two rungs that had been `closed-unopened` on the gate "against a null there is nothing for a
feature ablation to have changed." There now is.

Both ask the question a positive begs: **which part of the wiring carries it?** The connectome the
learner reads is a specific object — 3,709 chemical synapses with signs drawn at random, and 1,093
gap junctions built from the data — and a positive on "the wiring" does not say whether it lives in
the directed chemical graph, in the symmetric electrical one, or in the sign structure the random
draw imposes. Each ablation removes one of those and asks whether the effect survives.

- **L.4 — synapse signs as features.** `synapse_signs: atlas` replaces the coin-flip sign on 3,176
  of 3,709 chemical synapses with the sign the pre-synaptic neuron's transmitter implies; magnitudes,
  norms, readout and RNG stream are untouched (B.1 asserted this). [B.1](../../../docs/experiments/logbooks/044-l4-atlas-signs.md)
  found grounded signs left the untrained prior alone and made **Hebbian learning worse** — but every
  arm in B.1 ran a rule that could not learn, so it never asked whether signs change what the frozen
  substrate *computes as features*. L.1's learner reads exactly those features.

- **L.5 — gap junctions as features.** `enable_gap_junctions: false` zeroes the electrical matrix in
  the forward pass; every parameter is bitwise identical to the baseline. Gap junctions are the part
  of the wiring most like degree statistics — symmetric, degree-scaled — and **199 of them touch the
  39-neuron readout pool, 47 lie within it, and every pool neuron carries at least one**, so this
  ablation removes part of the readout's *immediate* input. The rewiring rewires gap junctions too,
  so both wirings lose them symmetrically. **L.5 is the clean ablation**: the forward pass and
  nothing else.

- **L.4 is not a clean removal, and the record says so.** Grounding makes the pool's inputs
  **275 excitatory to 36 inhibitory** against roughly 50/50 under the random draw — a large shift in
  net drive that can move the tanh units' operating point. If the effect vanishes under atlas, "the
  signs carried it" and "the substrate saturated" are both live. So the **atlas frozen floors against
  the wide frozen floors** is a registered diagnostic: a significant shift between two arms in which
  nothing learns says the operating point moved, and a `carries_the_effect` on L.4 is then reported as
  *carries or saturates*.

> *Does the per-neuron wiring effect survive grounding the signs? Survive removing the gap
> junctions?*

## What changes

**Each ablation is an interaction against L.1's wide baseline**, in the design L.1 validated: the
effect of interest is `(wt_ablated − rn_ablated) − (wt_wide − rn_wide)` per seed. Negative means the
ablation *removed* wiring effect; near zero means it survived; positive means it grew.

- **Eight new arms** on `hard350` at seeds **1–96**, each differing from its committed L.1 wide
  parent in **one key**:

  | arm | parent | key | learner |
  |---|---|---|---|
  | `wt_atlas` / `rn_atlas` | `wt_wide` / `rn_wide` | `synapse_signs: atlas` | readout learns, `w_chem` frozen |
  | `wt_atlas_frozen` / `rn_atlas_frozen` | the wide floors | `synapse_signs: atlas` | nothing learns |
  | `wt_nogap` / `rn_nogap` | `wt_wide` / `rn_wide` | `enable_gap_junctions: false` | readout learns |
  | `wt_nogap_frozen` / `rn_nogap_frozen` | the wide floors | `enable_gap_junctions: false` | nothing learns |

  **768 runs**, the size L.1 was — and ~15 GB rather than ~530 GB, because they run with the output
  controls L.1's crash produced.

- **The baseline is L.1's committed wide arms, not a re-run — conditional on a byte-identity check.**
  L.1's 384 wide-arm runs (`wt_wide`, `rn_wide`, their floors) are the un-ablated cells. Reusing
  them is licensed only if a re-run of one seed per arm under the new output controls reproduces
  L.1's logs field for field; if it does not, the baseline is re-run (+384) and nothing is reused.

- **A minimum effect as a decision rule, registered beside significance** — the design gap L.1
  recorded. The quantity an ablation can remove is the **wide wiring effect, +0.1852** — not L.1's
  +0.2818 interaction, which includes the pooled cells where the null was ahead; a feature that
  carried all of it would give an interaction of −0.185. `carries_the_effect` requires the
  interaction to be significant **and** to remove at least **two-thirds of that**: abs(Δ) ≥
  **0.123** — the feature is the *majority* carrier. Sized from L.1's realised spread (sd 0.416 at
  n = 96, ρ ≈ 0.08 between conditions, so treated as independent): se 0.0425, detectable 0.119 at
  80%, and **~80% power at the 0.123 minimum**. Half (0.093) would have ~59% power at 96 seeds and
  need ~160 — plus a re-run baseline, since L.1's covers seeds 1–96 only — so it is not registered.

- **A pre-registered structural probe on the open puzzle** — why the wide null got *worse* in L.1.
  No runs: 96 topology builds and L.1's committed per-seed file. Hypothesis, fixed before the
  correlation is computed: a per-neuron readout hurts a rewiring because the rewiring decorrelates
  the inputs *within* each motor class, so per-neuron weights fit seed-specific noise the class mean
  averaged out. Feasibility only has been looked at — within-class presynaptic Jaccard is 0.07–0.23
  in the wild type against 0.01–0.04 in three rewirings — and the registered test is on the 96.

- **A new harness** `scripts/analysis/l4_feature_ablations.py`, the L.1 sibling pattern: manifest
  builder per (ablation, wiring) pair, `connectome_structure_efficiency` called once per pair and
  unmodified, gates read first, one BH-FDR family across all ten registered tests, the metric-
  orientation and reachability guards L.1's reviews added, and every harness verdict name derived
  from source.

## The registered readings, per ablation

| reading | when |
|---|---|
| `carries_the_effect` | interaction significantly **negative** and abs(Δ) ≥ 0.123: the ablated feature is the majority carrier. On L.4, qualified *carries or saturates* if the floors diagnostic fires, and *carries or unlearnable* if the gains diagnostic fires *(the latter added 2026-09-18 after the pilot; see the design)* |
| `survives_without_it` | no significant interaction — **a failure to detect**, its size and CI carried — **and** the wiring effect under ablation is itself significant and positive: the effect is still there without that feature |
| `amplifies` | interaction significantly **positive**: removing the feature *helped* the wild type more. Reported, not explained |
| `inconclusive_at_this_sensitivity` | no significant interaction and the ablated wiring effect is not significant either: the panel cannot place the feature's contribution |
| `no_learning`, `insufficient_seeds` | as L.1 |

**Read per ablation, never pooled.** The two answer different questions and a split is the
informative outcome.

## Impact

- Affected specs: `plasticity-evaluation`
- Affected code: `configs/scenarios/foraging/` (eight arms), `scripts/analysis/l4_feature_ablations.py`
  (new), `scripts/analysis/l4_structural_probe.py` (new, the puzzle probe)
- **No package code changes**: both flags exist and both build at the per-neuron width, verified.
- **`l4_readout_width.py`, `connectome_structure_efficiency.py`, `wiring_premise.py` are READ-ONLY.**
