# 122: Phase 10 — C.2's Compact Encoder: Built, Checked, and Fitted Toward the Wild Type in Two Stages

**Status**: completed (description; nothing here is a registered verdict). **C.2's encoder is built, every check of the encoder itself passes, and a staged fit brings it most of the way to the wild type.**

The checks that pass:

- the existing connectome's outputs are byte-identical with the encoder off;
- layered parity with the per-step runner holds on the wild type and the degree-preserving null;
- the encoder builds on every null wiring and under the receptor-placement null;
- a hand-set vector carries a 9.0-min post-removal decay.

No fit from the box centre held Scheer's small lawn. The probes found why:

- two defects in the box, both fixed;
- a search problem in 196 dimensions, which a staged fit solves (the lawn subset first, then every parameter).

**The staged fit:**

- **It holds the lawn.** Its answer spends 99% of the small lawn's assay on it (real worms 98%), roams and dwells in bouts, and scores 3.43 on the registered ranking score (D_fit 3.5–3.6), against the 10.6 every earlier fit stalled at.
- **What still misses** is mainly the speed distributions: dwelling is too fast and roaming too slow at their 90th percentiles. That is the gap C.1 closed with a per-state speed spread, and the encoder has one noise level for both states.
- **It hasn't converged:** it was still improving at its cap of 500 generations.

The readiness restarts did not run, so F.0b-ii has no truths yet, and W's cost projects to 232–815 hours.

> | Check | Result |
> |---|---|
> | Identity: every existing connectome record, encoder off | **10 of 10 byte-identical** |
> | Parity: state (50 steps), trajectories (300) and statistics (32 seeds, full assay), wild type and degree-preserving null | pass |
> | The nulls: every rewired wiring and the receptor-placement null, with the mutants | build, with the wild type's 196 names |
> | Cost: one generation on 16 workers | D_fit's assays 21.2 s; all four 55.6 s |
> | Feasibility: a hand-set vector's post-removal decay, against 6–12 min | **9.0 min** |
> | Readiness: a fit passing D_fit's three families in 3 of 5 populations | **not run**; the staged pilot reaches score 3.43 on the lawn (99%) |

**Date**: 2026-11-03 to 2026-11-04.

**OpenSpec change**: `add-c2-encoder`.

**Records**: [`supporting/122-c2-encoder/`](supporting/122-c2-encoder/):

- `identity.json`;
- `cost.json`, `tanh_speedup.json`;
- `feasibility.json` and `feasibility/` (the trials and their script);
- `w_sizing.json`;
- `readiness/`:
  - every pilot's record (`pilot.json`, `pilot_uncompressed.json`, `reduced.json`, `pilot_logbox.json`, `pilot_staged.json`);
  - the premature stop (`premature/`) and the first box's stall (`stall_box3/`);
  - the diagnosis (`diagnosis/`), each read with its script as run.

## Objective

C.2 is the connectome controller fitted to the real wild type ([roadmap](../../roadmap.md) § Phase 10). No encoder existed that turned the connectome brain into a function of a few hundred named parameters. This change builds one, checks it against the existing brain, and probes whether it is ready for F.0b-ii, its recovery control, before anything registers. The maintainer asked that everything W depends on be settled here, since W is the phase's longest run.

## The encoder

`encoder: compact` on the connectome brain makes it a function of 196 named parameters (132 under `encoder_modulator_tying: per_ligand`), written from a unit-box vector (`quantumnematode.connectome.encoder`):

| Block | Parameters | Count |
|---|---|---|
| wiring | excitatory, inhibitory and gap scales (box [0, 1.5]); a signed gain for each unsigned edge's presynaptic transmitter group (dopamine, serotonin, octopamine, none) | 7 |
| sensory | 11 amphid classes × concentration, dC/dt and an antisymmetric lateral gain; food contact onto CEP, ADE and PDE | 36 |
| readout | the pooled motor readout (5 outputs × VB, DB, VA, DA) and a bias per output | 25 |
| noise | the action log-SDs | 5 |
| modulator | the data layer's 104 signed strengths, and its 17 fixed-sign strengths on a log scale over [0.001, 3] | 121 |
| timescale | one for the four monoamine pools, one for the 51 neuropeptide pools | 2 |

- **The wiring is fixed from data:**
  - a chemical edge's weight is its scale × its grounded sign × its synapse count, normalised by the root sum of squares of the counts onto its postsynaptic neuron;
  - gap junctions are strength-normalised, under one scale. This is the maintainer's gap model for every arm, which replaces the degree normaliser ([Logbook 103](103-phase9-synthesis.md)).
- **No weight depends on the brain's seed.**
- **On a null,** each edge carries its own count and its origin's sign.
- **Settling:** the chemical and gap terms act through one matrix, so each settling step is one product.
- **Mutants:**
  - tph-1 and pdfr-1 are built with their knockout genes;
  - tax-4 zeroes the 21 gains of its seven classes.
- **Refusals:** the brain refuses every setting the encoder replaces, and the loader refuses a modulator step that does not come to the configured step.
- **Configs:** the configs are `*_lawns_encoded_s1_primitives*.yml`, graded under `benchmark_grade.ARMS["connectome_c2"]`.

## The checks

- **Identity** (`identity.json`): `connectome_identity_check.py` ran ten jobs on discarded seeds, on the code before any edit and at the final code, and every one is byte-identical. The jobs: the hand-placed and data-layer arms in the batched evaluator on v3's four assays (every trajectory array and each animal's final satiety and total intake), and the per-step runner on the small lawn. After review the baseline was re-recorded on the pre-change code (`c73feb22`) in a worktree, and each record now states its checked source: the commit, any uncommitted edits, and a digest of the package, the configs and the script.
- **Parity** (`test_batched_parity.py`, `test_batched_statistics.py`): on the wild type and on the degree-preserving null, a vector inside the box gives:
  - features, action means and derivatives within 1e-5 over 50 steps;
  - bouts identical over 300 steps;
  - paired statistics over the full small-lawn assay on seeds 5001–5032 with no bias detected.
- **The nulls** (`test_connectome_encoder.py`): the encoder builds on all seven rewired wirings and under the receptor-placement null, each with the tph-1, pdfr-1 and tax-4 encodings, and with the wild type's parameter names.

## The cost pilot (`cost.json`, `tanh_speedup.json`)

- **Pace:** one generation of 16 simulations of 200 animals on 16 workers takes **21.2 s** on D_fit's two assays and **55.6 s** on all four. The staged fit ran at 26–28 s per generation (stage 2 26.3 s, stage 1 28.3 s).

- **Profile:** in process, a D_fit simulation spends 3.3 s of its profiled 14.9 s settling and 2.8 s in measurement.

- **One speed-up, taken before any encoded record existed:**

  - this torch build's CPU `tanh` is not vectorised, so the encoded settling takes NumPy's float32 `tanh` wherever no gradient is taken;
  - the two agree to within one ulp, and the result does not depend on the batch;
  - **1.21×** per simulation (9.87 s against 11.95 s), with every statistic identical on the timed population.

- **The compute path:** a whole generation's settling step (3,200 animals) takes:

  - 4.9 ms on one CPU thread;
  - 5.4 ms on 18 threads;
  - 1.26 ms on MPS (2.2e-6 from the CPU).

  MPS is 3.9× one CPU thread on the settling alone. The settling is about a quarter of a simulation's profiled time, and on MPS the rest (the environment, the noise draws, conversion, measurement and grading) would run serially in one process, against the 16 workers' parallel generation. So **W runs on the CPU.**

## The feasibility of a slow decay (`feasibility.json`)

**The vector:**

- the wiring scales at zero;
- the ingestion drive exciting NSM;
- one peptide couple from NSM, nlp-54 through TRHR-1 onto 11 of the 12 VA motor neurons, read by the reversal output;
- the neuropeptide timescale 540 s.

**The result:** it decays **9.0 min** after food removal, inside the 6–12 min window, on its reported seeds. Two trials on other seeds gave 8.9 and 8.9 min.

**The decay follows the slow pool:** with the reversal bias fixed, the neuropeptide timescale at 60 s gives no decay in the window, 270 s gives 3.3 min, 540 s gives 8.9 and 1,080 s gives 52. It does not equal the timescale, though: the readout's operating point reshapes it (−0.75 gives 21 min, −1.25 gives 5.8). Its values do not depend on the box.

## The readiness probe

Every fit was sep-CMA on D_fit's assays at 200 animals per evaluation, with its answer re-simulated twice at checkpoints on fresh animals. The maintainer chose each step from the one before.

| Step | Box | Score ranked on | Generations | Best answer's score | Time on lawn | Finding |
|---|---|---|---|---|---|---|
| restart 0 | wiring 0–3, fixed strengths linear | ranking | 48 (plateau) | D_fit 5.5–5.6 | — | the plateau stopped it early |
| restart 0, measured toward the cap | same | ranking | 325 (stopped) | 10.6 from generation 50 | 0.26 | a stall, not a slow mean |
| pilot | wiring 0–1.5 | ranking | 150 | 10.62 | 0.23 | not saturation alone |
| pilot | same | uncompressed | 150 | 11.01 | 0.21–0.29 | not the score's compression |
| reduced (18 parameters) | same | uncompressed | 150 | 14.43 | 0.18 | converged off the lawn |
| pilot | fixed strengths log-scaled | ranking | 150 | 10.69 | 0.22–0.26 | still off the lawn |
| staged, stage 1 (18) | same | uncompressed | 150 | 10.0 | 0.34 | the best start yet |
| **staged, stage 2 (196)** | same | ranking | **500** | **3.43** | **0.99** | holds the lawn; still improving |

1. **The plateau rule stopped the first restart at generation 48, far from the bands.**
   - It compares 20-generation means of the population median, and its σ ≤ σ₀ guard is vacuous at σ₀.
   - Measured to generation 325, the answer's score stayed at about 10.6 from generation 50, with σ flat near 0.155. It was a stall, not a slow mean behind a flat median.
2. **The worm does not hold the small lawn.**
   - Time on the lawn is z −116 (0.26 against 0.98), and exits are z +65.
   - Those two terms dominate every score. The ranking score's compression, 10·asinh(z/10), gives a statistic at z ≈ 120 a slope of 0.09.
3. **The first box started the network saturated.** By forward passes, a common wiring scale of 1.5 (the centre) saturates 56–62% of neurons, while 0.5–0.75 leaves the network graded. Rescaling to [0, 1.5] did not, alone, lift the stall.
4. **Food contact can move the outputs** (`diagnosis/contact_reach.txt`): over 4,000 random settings of the lawn subset, the speed output's change between contact 0 and 1 has a median of 0.21 and a 90th percentile of 1.9 pre-squash.
5. **The fixed-sign strengths swamped the senses on food** (`diagnosis/contact_exists*.txt`).
   - The settings where contact slows the worm most hold Scheer's small lawn once those strengths sit at their floor: up to 0.92 of the assay on it, with 0.35 exits per on-lawn hour.
   - At the first box's centre of 1.5 the same settings reach only 0.14–0.47. On food the pools inject currents held through every settling step, while the sensory input is only the settling's initial state.
   - With the strengths on a log scale (centre 0.055), the same settings hold the lawn at the box's own start: 0.93, 0.27 exits per hour.
6. **Lawn-holders that never roam score worse than any worm off the lawn** (`diagnosis/lawn_holder_undefined.txt`).
   - The best random lawn-holder never roams, so it leaves 7 of D_fit's 14 statistics undefined: the roaming and dwelling bouts, the roaming speeds and roaming survival.
   - Its score is about 7,005, and the definedness term ranks it below every worm that leaves none undefined.
   - The full fit cannot reach a worm that holds the lawn and switches states without passing through such worms or finding the switching first.
7. **Staged, it can.**
   - Stage 1 fitted the lawn subset alone and reached the best start yet: score 10.0, 0.34 of the assay on the lawn.
   - It missed its pre-stated 0.75 bar, and stage 2 ran from it regardless, recorded as a deviation from that rule.
   - Stage 2, every parameter from stage 1's answer at σ₀ 0.1, holds the lawn from generation 75: 0.97–0.99 of the assay on it, roaming and dwelling in bouts, every statistic defined. **The encoder switches between roaming and dwelling in closed loop.**
   - Its answer sat at a score of 4.2–4.5 from generation 100 to 475 and stepped to **3.43** at 500 (D_fit 3.5–3.6). It was still improving at the cap.

**What generation 500's answer misses** (`diagnosis/final_stage2.txt`, one re-simulation):

| Statistic | z |
|---|---|
| dwelling speed, median / 90th percentile | +3.3 / **+6.5** |
| roaming speed, median / 90th percentile | −3.2 / **−5.3** |
| uniform well's roaming bouts | **−5.1** |
| small lawn's roaming fraction | **+5.3** |
| small lawn's dwelling bouts | −3.4 |

Time on the lawn (z +2.0), exits (z −2.8), roaming survival (z −1.7), the uniform well's roaming fraction and dwelling bouts, and the small lawn's roaming bouts and pre-exit enrichment are within the bar of 3.04. **The two states barely differ in speed.** The fit trades that against the bout statistics: at generation 257, roaming fraction passed and the bouts failed, and at 500 the reverse. C.1 had the same gap until it gained a per-state speed spread ([Logbook 121](121-c1-v3-fit.md)). The encoder has one action noise level for both states: it refuses the brain's state-dependent std head (`continuous_std_mode: state_dependent`).

## What this establishes, and what it does not

- **The encoder is sound as an instrument.** It reproduces the existing brain when off, matches the per-step runner, builds on every null W plans with the wild type's parameters, and its slow pools can carry a 9-minute behavioural decay.
- **It can be fitted toward the wild type, but not from the box centre in one stage.** Two box defects are fixed: the saturating wiring range, and fixed-sign strengths that swamped the senses. A staged fit then reaches a lawn-holding, state-switching worm at a score of 3.43.
- **It does not yet pass.** Its speed distributions miss, the one kind of miss C.1 also had, for want of a state-dependent speed spread. It had not converged at 500 generations.
- **Nothing here is a verdict on C.2,** and nothing is registered. The staged procedure was found on discarded seeds, one pilot each. The readiness restarts never ran, so there are no truths for F.0b-ii, no restart spread and no measured convergence generation.

## For W

- **Settled here:**

  - the strength-normalised gap model;
  - the null encoding (counts with edges, origin signs);
  - the CPU compute path;
  - the box's wiring range [0, 1.5] and log-scaled fixed-sign strengths;
  - x₀ the box centre for every arm;
  - the staged procedure, which applies identically to every arm, so no start is tuned to the wild type's routes.

- **The cost** (`w_sizing.json`):

  - a staged fit costs about 4.8 h at stage 2's 500 generations, or 8.5 h at 1,000;
  - W at 6 arms × 8–16 draws projects to **232–815 hours** on this machine.

  That is beyond what W can run as planned, so its registration needs a stopping rule, fewer draws, or a cheaper fit.

- **Not yet measured:** the generations a fit needs, the restart-to-restart spread against W's minimum (2.97 fitted, 2.05 held-out), and the tying.

- **The class-preserving left–right-symmetric null** is still to build.

## For the next change

1. **A state-dependent speed spread.** Let the encoder carry the brain's state-dependent std head (or an equivalent per-pool noise), and check that it lifts the kinematics, as C.1's spread did.
2. **The staged procedure, registered:**
   - stage 1's subset and bar (0.34 was enough to start stage 2; 0.75 was not met);
   - each stage's objective and σ₀;
   - stage 2's cap, with a stopping rule read on the answer's checkpoints, not on the population median.
3. **The readiness restarts under it,** for F.0b-ii's truths and W's spread.
4. ~~**A loader guard.**~~ *(Landed with this change's review.)* A brain config that its own class refuses used to load silently as another class's fields copied onto defaults (`modulators: true` came back `False`). The loader now validates `brain.config` against the class `name` selects first. Seven committed configs that already could not build their brain now fail at load: two foraging configs the config-loading test lists as stale, and five hybrid-cortex specials.

## Corrections and lessons

- **Check the box's centre by forward passes before fitting.** Both box defects showed in seconds of forward passes (saturation; modulator currents against the sensory input), after hours of fitting had not found them.
- **The answer's checkpoints, not the median, show progress.** In 196 dimensions the population median went flat while nothing improved, and it kept falling after the answer had settled.
- **Feasibility by random draws is not reachability by the fit.** The random lawn-holders never roam, and the score ranks them below everything. A target must be met as the benchmark defines it, every statistic defined.
- **Stopping a pool run's parent orphans its workers.** Two stopped runs left 34 idle workers (about 16 GB) for 15 hours. They are found by parent PID 1 and stopped by PID.
- **A first draft of the cost pilot crashed moving the encoder to MPS** on a float64 buffer. The timescales are float32 now, and the encoder moves to MPS.

## Addendum (2026-11-04): the glutamate signs onto AIA and ADE

**What was wrong.** The per-connection table signed all 29 glutamatergic chemical edges onto AIA and ADE excitatory. None had a physiology override, and the expression step gave none of them a sign, so each fell to the per-neuron rule, which signs glutamate positive.

López-Cruz et al. 2019 (*Neuron* 102:407) show the opposite for the pathways local search runs through: "Glutamatergic chemosensory and mechanosensory neurons generate local search by suppressing MGL-1-expressing neurons". The chemosensory set is "a set of glutamatergic chemosensory neurons that synapse onto the AIA interneurons", and the mechanosensory set "a set of glutamatergic mechanosensory neurons that synapse onto the ADE sensory-inter neurons". The paper's knockouts cover "all tax-4 chemosensory neurons" and "all mec-3 mechanosensory neurons". The quotes are from the PMC text of the published paper (PMC9161785).

So the wild type above was wired with the opposite sign on these pathways. Every readiness figure in this logbook is on that earlier wiring, and none is re-read as the corrected wiring's.

**The evidence, in the vendored rows** (`data/benchmark/lopezcruz2019_reorientations.json`; early reorientations are the mean per 2-min bin summed over bins 2–5, 6–14 min after removal). The two modules are redundant:

| Row | Early reorientations | Its experiment's control |
|---|---|---|
| glutamate lost in the tax-4 cells (Fig. 4F) | 5.7 | 7.5 |
| glutamate lost in the mec-3 cells (Fig. 4F) | 6.3 | 7.5 |
| glutamate lost in both (Fig. 4F) | 2.3 | 7.5 |
| mgl-1 (Fig. 4F) | 2.0 | 7.5 |
| glutamate lost in AWC, ASK, ASG alone (Figs. 4C–E) | 6.2, 4.1, 6.7 | 5.7, 5.2, 7.2 |

So each override's evidence is the paper's conclusion for its pathway, drawn from:

- the double knockout;
- mgl-1's loss, and its rescue in AIA and ADE;
- silencing AIA or ADE in mgl-1 mutants.

It is not a single-pathway measurement. That is the same standard as the AWC → AIY override.

**The correction.** Sixteen overrides, sign −1, citing `lopezcruz2019`, in `data/connectome/sign_overrides_physiology.csv`:

- the glutamatergic tax-4 cells onto AIA: ASE, ASG, ASK and AWC, 12 edges and 195 synapse units;
- the glutamatergic mec-3 cells onto ADE: FLP and AVM, 4 edges and 35 units.

The table now has 67 physiology signs, 1,676 by expression, 1,433 by the rule and 533 with none, and it corrects 358 of the atlas's signs.

**A limit: 13 glutamatergic inputs stay excitatory.** These are ASH, ADL, AIZ, AIM, ADA and AIB onto AIA, and OLL and IL1 onto ADE. None of the paper's cell-specific manipulations covered these cells, so only the receptor's location argues for flipping them, which is weaker than the table's standard. ASH and ADL are large inputs onto AIA (24 and 19 units onto AIAL), so AIA's glutamatergic input is now mixed.

**A second limit: the sign is a fast stand-in for MGL-1.** MGL-1 is metabotropic and slow; the fast sign captures its direction only.

**Reproducing the earlier table.** `sign_override_exclude: [lopezcruz2019]` on the connectome brain ignores these rows. The nine committed per-connection configs behind recorded runs set it, through their generators: five lawn configs and four thermal target-35 configs. `sign_scale_calibration.py`, which builds per-connection brains in code for the thermal rung's recorded calibration, sets it too. The seven encoded configs take the corrected table.

**The identity check** (`supporting/122-c2-encoder/aia_ade_signs_identity.json`): 19 of 20 jobs are byte-identical to a baseline recorded before the change. Those are every batched and per-step connectome job and the nine excluded configs' built brains, compared on their topology state (every weight, gain and wiring buffer; not the PPO critic, which no sign reaches). The one that differs is the encoded brain. Its normalised counts move from the excitatory matrix to the inhibitory one in exactly the 16 entries, at the same magnitude (`test_the_encoded_brain_changes_only_on_the_corrected_edges`).

**For the next steps.**

- The readiness restarts run on the corrected wiring.
- The perturbation family (planned) grades López-Cruz's cell-level rows against it.
- Its pan-neuronal rows will read the 13 inputs left on the rule.
