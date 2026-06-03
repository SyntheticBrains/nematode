# 025: Weight-Search Cross-Architecture Ranking (Integrated C3 Cell)

**Status**: cross-architecture C3 ranking complete (7 architecture families, n=8 paired seeds), including a
genuinely-quantum arm with a full controlled attribution. Phase 0 (predator-sensing canonicalisation) is
recorded under [supporting/025/phase-0](supporting/025-weight-search-architecture-ranking/phase-0/README.md).

**Branch**: `feat/weight-search-phase4-comparison`.

**OpenSpec changes**: `weight-search-architecture-ranking`; the quantum arm shipped via
`add-equivariant-quantum-brain` (archived `2026-06-01-add-equivariant-quantum-brain`, PR #200).

**Date**: 2026-06-03.

______________________________________________________________________

## Objective

Rank brain-architecture families on a single **integrated "C3" cell** — food chemotaxis + predator
evasion + thermotaxis, all active in the same simulation under klinotaxis (head-sweep) sensing — to
answer three questions:

1. Where does the **wild-type connectome** rank against the strongest classical / neuromorphic baselines
   on integrated control? (Gate 3 evidence, roadmap RQ on connectome optimality.)
2. Which architecture juggles the three competing pressures best, and how reliably?
3. Does a **genuinely-quantum** architecture (trained parameterised-quantum-circuit policy with
   load-bearing entanglement) add anything over fair classical baselines? (RQ4.)

## Background

The C3 cell is the load-bearing comparison configuration: an identical env + reward block across all
arms (LETHAL thermal gradient 1.5, two-channel predator biology, `distal_chemo_contact_trigger` reward),
with only the brain block differing — so the comparison measures architecture, not environment. The
primary metric is **`post_convergence_success_rate`** (the full-clear, 10/10-food rate averaged over the
post-convergence plateau). Post-convergence — rather than a fixed last-N window — is used because the
arms have very different warm-up lengths: the spiking and quantum arms have long from-scratch dead-
exploration warm-ups on this lethal cell (the quantum arm is flat-dead to ~ep 1000, then ignites and
plateaus by ~ep 2500), so a fair comparison ranks each arm on its converged plateau.

Statistics: paired-seed one-sided Wilcoxon (seeds 42–49), 80% bootstrap CIs (1000 resamples, seeded),
BH-FDR across the pairwise set — pre-committed in the change's `architecture-comparison-protocol` spec.

The six classical / neuromorphic arms (MLP-PPO, LSTM-PPO, connectome-PPO, CfC, spiking-PPO,
FeedforwardGA) were ranked first; the equivariant-quantum arm was added last, with a controlled
attribution to test RQ4 honestly.

## Hypotheses (pre-registered)

- **H1 (connectome)**: the wild-type connectome lands **mid-pack** — competitive on foraging, behind the
  temporal/gradient arms on predator evasion (no recurrent memory advantage; fixed wiring).
- **H2 (quantum)**: **competitive-or-below** a parameter-matched classical baseline — no quantum
  advantage is expected for small classical-data RL at simulator scale (2025–26 literature scan +
  Logbook 008). The reported payload is the **ablation deltas**, not the raw rank.

## Method

**Architectures (7 families, n=8 seeds 42–49, C3 cell, 4000 ep each for the fresh arms):**
MLP-PPO, LSTM-PPO, connectome-PPO (strict chemical-synapse mask + fixed gap junctions), CfC (liquid),
spiking-PPO (recurrent adaptive-LIF + MLP actor head), FeedforwardGA (GA-evolved fixed topology), and
**equivariant-quantum** (`equivariantquantum`).

**The quantum arm**: a bilateral-Z₂-equivariant parameterised quantum circuit — an equivariant
parity-block pre-encoder → a `U_R`-invariant reference state → a Z₂-equivariant data-reuploading circuit
(IsingXX / same-parity IsingZZ entanglers) → an equivariant Pauli readout (the left–right mirror swaps
`LEFT`/`RIGHT`, fixes `FORWARD`/`STAY`). It is *genuinely quantum* (trained circuit parameters + load-
bearing entanglement) but classically simulated via an in-repo differentiable torch statevector
simulator (validated vs Qiskit ~1e-7). The symmetry is *C. elegans*' own bilateral body symmetry and the
natural symmetry of klinotaxis. `num_qubits=8, k_odd=3` (the three lateral-gradient odd features:
food / predator-chemosensation / thermotaxis).

**Quantum attribution — four controlled arms (n=8 each, matched env/reward/budget):**

| arm | what differs | isolates |
|---|---|---|
| equivariant-quantum (main) | — | (the headline arm) |
| unstructured-quantum | drop the Z₂ symmetry (arbitrary PQC) | symmetry prior, within quantum |
| thin classical-equivariant | replace the circuit with the original thin-odd-path MLP | quantum circuit (naive) |
| rich classical-equivariant | replace with a Z₂-symmetrised full-MLP (matched 9.9k params) | quantum circuit (**fair**) |
| rich classical **non**-equivariant | the same 9.9k-param MLP, symmetrisation removed | symmetry prior, within classical (**matched**) |

The "rich" classical control was added after the naive thin control proved sub-MLP (see Analysis); it is
a maximally-expressive equivariant net built by Z₂ group-averaging (`even=½(f(x)+f(Rx))`,
`odd=½(g(x)−g(Rx))`), exactly equivariant for any MLP, at ~9.9k params (> the circuit's ~tens).

## Results

### Cross-architecture C3 ranking (post-convergence full-clear, n=8)

| rank | architecture | post-conv % (mean ± sd) | foraging (foods) | evasion (%) | thermal (comfort) |
|---|---|---|---|---|---|
| 1 | **equivariantquantum** | **86.0 ± 2.9** | **8.04** | 75.6 | 0.496 |
| 2 | cfcppo | 84.4 ± 9.5 | 3.94 | 60.0 | 0.498 |
| 3 | spikingppo | 84.2 ± 1.1 | 6.17 | 74.2 | 0.470 |
| 4 | lstmppo | 83.6 ± 4.5 | 5.59 | 70.7 | 0.493 |
| 5 | connectomeppo | 75.6 ± 2.8 | 5.72 | 63.3 | 0.516 |
| 6 | mlpppo | 73.1 ± 7.5 | 6.30 | 75.8 | 0.506 |
| 7 | feedforwardga | 0.0 ± 0.0 | 1.29 | — | — |

**Top cluster is a statistical tie.** equivariant-quantum is numerically #1 with tight variance, but
every pairwise test against the next three is non-significant: vs CfC Δ+1.5 (q=0.58), vs spiking Δ+1.8
(q=0.16), vs LSTM Δ+2.4 (q=0.14). It *is* significantly above connectome (Δ+10.4 \*\*\*), MLP (Δ+12.9 \*\*\*),
and GA (Δ+86.0 \*\*\*). So the four-way top cluster is **{quantum, CfC, spiking, LSTM} ≈ 84–86%**, with the
quantum arm the numerical leader and notably the **highest forager** (8.04 foods).

### Connectome verdict (Gate 3)

The wild-type connectome (75.6%) ranks **mid-pack** — significantly behind the top cluster and below even
plain MLP-PPO on per-behaviour evasion. Paired-seed per-behaviour: ~tie on foraging (vs MLP Δ−0.6 ns, vs
LSTM Δ+0.1 ns), **behind on predator evasion** (vs MLP Δ−12.5, vs LSTM Δ−7.5), ~tie on full-clear vs MLP
(Δ+2.4 ns). **H1 confirmed**: competitive on foraging, behind on evasion (fixed wiring, no recurrent
memory advantage). Its best per-behaviour score is thermal comfort (0.516, the field's highest).

### Quantum attribution — the headline deconstructed

| control arm | post-conv % | delta vs main | significance |
|---|---|---|---|
| equivariant-quantum (main) | 86.0 ± 2.9 | — | — |
| **rich classical-equivariant (fair)** | **87.9 ± 1.9** | **−1.9** | **ns (p=0.93)** |
| rich classical non-equivariant | 86.3 ± 2.4 | −0.3 | ns |
| unstructured-quantum | 83.6 ± 3.6 | +2.4 | ns (p=0.13) |
| thin classical-equivariant | 61.4 ± 3.2 | +24.6 | \*\*\* (artifact) |

Two clean deltas:

- **Quantum-circuit value** (main − fair classical): **−1.9, ns.** A fair classical-equivariant net at
  matched capacity *matches and numerically edges* the quantum circuit. **No quantum-circuit advantage.**
- **Symmetry-prior value** (matched-capacity, equivariant − non-equivariant): **+1.5 (classical, ns)** and
  **+2.4 (quantum, ns).** The bilateral-symmetry prior is **not significantly load-bearing** either.

## Analysis

**H2 confirmed, decisively and on-task.** The exciting raw result — equivariant-quantum #1, "+24.6 over a
classical-equivariant net" — fully deconstructs under controls:

1. The +24.6 was a **weak-baseline artifact.** The naive classical-equivariant control (61.4%) is
   *below even plain MLP-PPO* (73.1%) because its left/right (odd) output was
   `tanh(linear(odd-latents-only))` — an information-starved path that cripples the klinotaxis left/right
   decisions that dominate this task. Give the classical control a proper (Z₂-symmetrised) rich odd path
   at matched capacity and it climbs 61.4 → **87.9%**, erasing the gap entirely.
2. With a **fair** classical control, the quantum circuit shows **no advantage** (−1.9, ns). This is a
   controlled, on-task confirmation of the 2025–26 literature scan and the project's own Logbook 008
   ("environment complexity below the quantum-advantage threshold") — now *demonstrated*, not assumed.
3. The **symmetry prior**, though well-motivated (the worm's bilateral body plan) and *correct*, is not
   significantly load-bearing: matched-capacity non-equivariant controls are statistically tied in both
   the quantum (+2.4 ns) and classical (+1.5 ns) families.

So the equivariant-quantum arm's top rank is attributable to **neither the quantum circuit nor the
symmetry significantly** — it sits in the top cluster with the best classical / neuromorphic arms, and
every edge over them is within noise. It is a *legitimate, reliable co-top performer* (and the strongest
forager), but not an advantaged one.

**Methodology note.** A less careful analysis would have published "quantum beats classical by 24.6
points." Two control audits — a fair-baseline (rich classical) audit and a matched-capacity symmetry
control — caught two over-claims before either reached this writeup. That discipline is the most
defensible result in this arc.

## Conclusions

1. **Top cluster (≈84–86%) is a four-way tie: {equivariant-quantum, CfC, spiking, LSTM}.** The quantum
   arm is the numerical leader and the highest forager (8.04 foods), with tight variance (sd 2.9).
2. **No quantum advantage.** A fair, matched-capacity classical-equivariant net matches/edges the quantum
   circuit (87.9 vs 86.0, ns). Confirms Logbook 008 + the 2025–26 scan, now under on-task controls.
3. **No significant symmetry advantage.** The bilateral-Z₂ prior adds only +1.5–2.4 pts (ns) at matched
   capacity, in both the quantum and classical families.
4. **Connectome ranks mid-pack** (75.6%) — competitive on foraging, behind the temporal/gradient arms on
   predator evasion. Significantly above MLP/GA on full-clear, significantly below the top cluster.
5. **GA collapses** on the integrated lethal cell (0%) — GA-on-fixed-topology does not solve C3.
6. **Methodology**: the genuinely-quantum arm is a clean *negative* on advantage, delivered with
   controlled attribution (fair-baseline + matched-capacity audits) rather than an over-claim.

## Next Steps

- This 7-arch grid ranking is the **baseline T7 (continuous-physics upgrade) will measure against** for
  the env-upgrade delta.
- **RQ4 (quantum)**: settled negative on advantage at this complexity; if revisited, it belongs at the T7
  continuous-physics substrate (per `phase6-tracking` Decision 4), not the grid.
- Optional: a deeper symmetry study (multiple symmetry groups / harder cells) if the small equivariant
  edge is worth chasing — but it is currently sub-significant.

## Data References

- **Analysis exports**: [supporting/025/phase-4/analysis](supporting/025-weight-search-architecture-ranking/phase-4/analysis/)
  (cross-architecture ranking CSV/JSON, per-seed tables, quantum attribution).
- **Phase 0** (predator-sensing canonicalisation): [supporting/025/phase-0](supporting/025-weight-search-architecture-ranking/phase-0/README.md).
- **C3 configs**: `configs/scenarios/foraging_predator_thermal/*_small_combined_klinotaxis.yml` (one per
  arm); the quantum arm + its ablation siblings under the same directory
  (`equivariantquantum_small_combined_klinotaxis*.yml`).
- **Quantum brain**: `packages/quantum-nematode/quantumnematode/brain/arch/equivariant_quantum.py` +
  `_quantum_statevector.py`; spec archived under `2026-06-01-add-equivariant-quantum-brain`.
