# Design: panel 2 — the Hebbian wiring contrast

## Context

Panel 1 (Logbook 040) fixed the recipe every plastic arm runs with — both scaling switches,
homeostasis, `initial_log_std −1.0`, `plasticity_rate 1e-3` — and showed that under it the
unmodulated-Hebbian rule converges within a few hundred episodes to a fixed point of the wiring's
correlation structure whose quality depends on the seed. Its descriptive wild-type-over-rewired
Hebbian advantage (+16.5, 5/8) is the hypothesis here. Nothing in the rule changes; the design
adds sample size, a prior sweep, and an initialisation factor, and removes the arms that cannot
inform the question (the reward-modulated arms until the prior is known; the MLP, which
collapses under any local rule).

## Goals / Non-Goals

**Goals**

- Test the Hebbian wiring contrast as a registered primary with enough seeds to carry a
  ten-to-twenty-point effect against floors that vary by thirty.
- Measure the prior over policies each wiring imposes on random initialisations.
- Add anatomy the substrate discards: synapse-count-scaled magnitudes, as a factor.
- Keep panel 1's seeds 1–8 identical so the two panels' floors coincide.

**Non-Goals**

- Any change to the rule, the scaling, the noise or the homeostasis.
- Reward-modulated arms (a follow-up once the prior is characterised).
- Sign identity from transmitter data (7a-ii's atlas).

## Decisions

### D1. Count-scaled initialisation

For post-synaptic neuron `j` with incoming edges `i` carrying counts `n_i` (from Cook 2019's
`weight` field, integer synapse numbers), the chemical weight is
`w_ij = z_i · n_i / sqrt(Σ_i n_i²)`, with `z_i ~ N(0, 1)` drawn in the same order and from the
same generator as the degree-scaled draw. Its expected squared incoming norm is exactly 1 for
every neuron with inputs — the degree-scaled initialisation's expectation (`k` draws at scale
`1/√k`) — so the homeostatic targets and the bound sit where they sit in panel 1, and only the
relative magnitudes within a neuron's inputs change: proportional to count.

Ratified with Chris over a logarithmic law: linear is the direct physical reading (each contact
adds conductance) and reports the anatomy as measured, tail included (the largest input edge
carries a median 29%, 90th-percentile 50%, of a neuron's input; a log law would compress that to
18% and 35% on an assumption the data does not make). Signs remain random draws: transmitter
identity is 7a-ii's.

**Under the rewiring**, the degree-preserving swap carries each edge's count with its
pre-synaptic endpoint (the utility already does this to preserve provenance), so a rewired neuron
receives a different multiset of counts and the per-neuron normalisation is recomputed on the
rewired edge set. The rewired-null is therefore a null of the count structure as well as of the
partner identity, which is what "degree-matched scramble" should mean once counts carry weight.

**Byte-identity**: `weight_init: degree_scaled` (default) takes the existing code path with no
extra operation; the frozen-reference and wiring-arms tests keep proving it. The option is a
config field beside `wiring`, and the count-initialised configs are one key off their parents.

### D2. Arms, seeds, budgets

| arm key | wiring | init | rule |
|---|---|---|---|
| `wt_frozen`, `rn_frozen` | wild-type / rewired | degree-scaled | frozen |
| `wt_frozen_count`, `rn_frozen_count` | wild-type / rewired | count-scaled | frozen |
| `wt_hebbian`, `rn_hebbian` | wild-type / rewired | degree-scaled | unmodulated Hebbian |
| `wt_hebbian_count`, `rn_hebbian_count` | wild-type / rewired | count-scaled | unmodulated Hebbian |

- **Hebbian panel**: the four Hebbian arms on seeds **1–16**, 1000 episodes. Seeds 1–8 reproduce
  panel 1's Hebbian floors bit for bit (same configs, same seeds), so panel 1's floor values are a
  built-in check; seeds 9–16 double the sample. The budget is set from panel 1's evidence: every
  Hebbian run there sat at its final level from its first 500-episode block and stayed there for
  3000; 1000 leaves a 250-episode plateau tail after a 750-episode margin.
- **Prior sweep**: the four frozen arms on seeds **1–64**, 600 episodes. A frozen arm is a fixed
  policy, so its plateau tail is its success rate and 600 episodes give it a 150-episode
  estimate; 64 seeds give the distribution. The frozen arms on seeds 1–16 double as the
  learning-gain floors for the Hebbian panel.
- **No pilot.** Every value the arms run with is panel 1's registered pin, and the Hebbian rule
  does not use the modulator. The plateau detector still reports convergence per run; a Hebbian
  seed without a plateau at 1000 receives the single registered extension (a fresh run at 1500).
- Cost: 64 + 256 = 320 runs, about an hour on 16 workers.

### D3. Metric and statistics

The committed plateau-tail full-clear success (final quarter) and the committed paired-seed
one-sided Wilcoxon, 80% bootstrap CI and BH-FDR, through panel 1's readers. Pairing is by seed
across every arm (`rewire_seed` from the run seed, as before; the count-scaled draw consumes the
generator identically, so wild-type and rewired arms remain paired under both initialisations).

### D4. The confirmatory family (four tests, one BH-FDR family at α = 0.05)

| id | test | direction | reads |
|---|---|---|---|
| **P1** | `wt_hebbian` vs `rn_hebbian`, seeds 1–16 | wild-type > rewired | the primary: Logbook 040's signal, doubled |
| **P2** | `wt_hebbian_count` vs `rn_hebbian_count`, seeds 1–16 | wild-type > rewired | the same contrast with the anatomy's counts |
| **P3** | `wt_hebbian_count` vs `wt_hebbian`, seeds 1–16 | count > degree | does the count structure improve the wild-type fixed point? |
| **P4** | `wt_frozen` vs `rn_frozen`, seeds 1–64 | wild-type > rewired | the prior: is the wild-type's distribution of untrained policies better than its scramble's? |

A test passes at q < 0.05 with a positive mean delta. A reverse result is detected by an interval
lying entirely below zero and named, as in Logbook 034.

**Descriptive**: `wt_frozen_count` vs `wt_frozen` and the rewired equivalents; every remaining
pair; for each arm the distribution of plateau tails and the **competent fraction**, the share of
seeds whose plateau tail is at least 20% with no learning (the prior's headline number); the
learning gain of each Hebbian arm over its own frozen arm on seeds 1–16; per-seed sign counts for
P1 and P4.

### D5. The verdict map (Logbook 034's vocabulary)

From P1, in order: `wiring_specific` (P1 passes), `rewired_beats_wild_type` (P1's interval
entirely below zero), `degree_statistics` (interval spans zero), `inconclusive` (otherwise);
`insufficient_seeds` first if P1 has fewer than two common seeds. P2–P4 annotate the verdict and
never change it: the record states whether the count structure preserves the contrast (P2),
improves the wild-type fixed point (P3), and whether the prior already differs (P4). A
`wiring_specific` verdict with P4 failing is the interesting case — the wiring's advantage would
be *created* by Hebbian alignment rather than present in the untrained prior — and is named as
such in the report; a `wiring_specific` verdict with P4 passing says the advantage is already in
the prior.

Claim type: performance, throughout. Ensemble invariance is reported (sign counts); no dynamics
claim is made here.

### D6. Harness and records

`scripts/analysis/l4_panel2.py`: its own arm registry (the eight stems), panel 1's `read_log`,
`scan_campaign`, `paired` and the statistics helpers imported, the family and verdict above, the
prior-sweep analysis (distributions, competent fractions, P4), per-seed CSV and curves, a launch
record before the run, everything promoted to `supporting/041-l4-panel2/`. Tested on synthetic
logs: registry and seed ranges (Hebbian arms accept 1–16, frozen arms 1–64, nothing else), each
test's direction, the family size, every verdict row, the reverse case, the competent fraction,
and the P4 annotation.

## Risks / Trade-offs

- **The count-scaled prior may differ from the degree-scaled one enough to move the floors.**
  That is the point of the factor; P2 and the descriptive frozen pairs read it, and the
  homeostatic targets stay at 1 by construction so the rule's dynamics are unchanged.
- **Sixty-four seeds of frozen runs is the largest campaign yet.** Frozen runs are the cheapest
  (short episodes, no learning); about forty minutes.
- **Bimodal floors make P1 a high-variance test even at n = 16.** The sign count and the
  competent fractions are reported beside the test so a null is read against the distribution,
  not only the mean.
- **P4 pairs seeds across wirings, but the two arms' draws differ by construction** (a different
  edge set consumes the same generator differently). Pairing by seed is kept for consistency with
  every earlier panel; the prior sweep is also reported as two distributions.

## Open Questions

None. The plastic arms under count-scaled initialisation, and a reward-modulated re-run once the
prior is known, are the follow-up this panel sets up.
