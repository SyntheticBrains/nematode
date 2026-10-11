## Overview

The connectome gains two slow, global concentrations, serotonin and PDF:

- each is released by its cited neurons' activity;
- each acts on its cited receptor cells with the receptor's sign;
- each persists across steps with a 60 s timescale.

The primary reading knocks each modulator out of the trained network at evaluation and asks whether
the worm's states shift in the mutants' published directions. The choices below were made with the
user on 2026-10-11.

## Decisions

### Decision A: The circuit, as the sources support it

From `data/neuromodulators/roaming_dwelling_circuit.json`, verified against the full texts:

| | release | receptor | sign | target cells |
|---|---|---|---|---|
| serotonin | NSM, driven by food ingestion (Rhoades 2019) | MOD-1 | inhibitory | AIY, ASI, RIF (Flavell 2013) |
| PDF | AVB, PVP, SIAV (Flavell 2013) | PDFR-1 | excitatory | AIY, RIM, RIA (Flavell 2013) |

- **Each class's left and right cells are both gated or both release.** Neurons absent from the
  connectome are reported at build and skipped.
- **AIY carries both receptors,** the convergence point in Flavell 2013's model.
- **Exclusions:** RIB is excluded, since its cited source does not contain it. AIY's MOD-1 role is
  contested by Dag 2023's acute-slowing assay, a different behaviour, and is kept as Flavell's
  roaming/dwelling assay supports it. HSN is not a release site.
- **PDF inhibits NSM** (Ji 2021 Fig 4E–G), modelled as a functional link, decided with the user after
  the spec review.
  - **Why it has to be added explicitly.** NSM is wired only to the pharynx in Emmons 2024: every
    synaptic input and output is a pharyngeal neuron (I1–I6, M1–M4, MI), and it has no gap junctions.
    No path from a receptor cell reaches it.
  - **What Ji 2021 found.** PDF signalling keeps NSM inactive during roaming, acting through the
    PDFR-1 neurons. The MOD-1 neurons inhibit NSM too. The mechanism is polysynaptic or
    extrasynaptic.
  - **How it is modelled.** An inhibitory current on NSM, proportional to the PDF concentration, with
    its own learnable strength, which the *pdfr-1* knockout removes, as in the worm.
  - **What it gives the model** is the antagonism Ji 2021 proposes for persistent states: serotonin
    inhibits the PDFR-1-expressing AIY, and PDF inhibits serotonin's source.

### Decision B: The concentrations

Each concentration `c` is a scalar carried across steps within an episode, and reset to 0 at each
episode's start:

```text
c ← c · exp(−Δt/τ) + (1 − exp(−Δt/τ)) · r
```

- **The timescale:** Δt is the step's 5 worm-seconds, and τ is fixed at 60 s, the scale of NSM's
  activity peaks. Persistence beyond it has to come from the circuit, the mechanism Ji 2021 proposes
  (mutual inhibition).
- **The release `r`:** the mean, over the release cells, of their settled activity above rest,
  `max(0, h)`, at the step's end.
- **Timing:** a step's release updates the concentration the next step sees.

### Decision C: Gating

At every settling iteration, each receptor cell gains a current of `sign · g · c` before its tanh:
inhibitory for MOD-1, excitatory for PDFR-1. NSM gains `−g_NSM · c_PDF`, PDF's inhibition of NSM.
These currents are **held**, added at every iteration, unlike sensory inputs, which set only the
initial state.

- **The strengths are learnable,** `g = softplus(ρ)`, one per receptor, with ρ initialised so that
  g = 1. PPO decides how much to use each pathway; the knockouts ask whether it did. The learned
  strengths are reported.
- **Two code paths.** The gating is applied in the rollout loop and in PPO's batched replay loop, which
  are separate. A test checks that they agree.

### Decision D: Food ingestion drives NSM

`BrainParams.food_ingested` carries the intake value of the worm's last move, the same quantity
behaviour capture records, and `BrainParams.max_food_ingested` the largest one move can yield (a
full cell's bite at the best lawn's quality). The ingestion drive is their ratio, the last move's
bite as a fraction of a full one, read the way satiety is read against its maximum. The connectome
adds `γ · drive` to NSML and NSMR as a **held current at every settling iteration**, with γ a
learnable gain initialised at 1. NSM detects ingested food (Rhoades 2019).

**Why the drive is a fraction of a full bite** (found in implementation, 2026-10-11). On D.1's
cell a full bite is worth 0.02 of intake value, while the PDF concentration settles near 0.3 to
0.45. Taken raw, the drive was some twenty times weaker than PDF's inhibition of NSM, so NSM stayed
below rest, serotonin never formed, and the MOD-1 strength took no gradient. As a fraction, a full
bite drives NSM at 1 on any cell, and in a check on the cell's untrained network both
concentrations formed (serotonin mean 0.16, peak 0.85; PDF about 0.3).

**Why the current is held.** Sensory inputs set only the initial state. NSM has no self-connection,
so an initial-state input would be gone by the end of settling, and serotonin release, read from
NSM's final activity, would not track what the worm ate. The spec review found this.

**The satiety-link arm** multiplies the drive by `(1 + (1 − s))`, where `s` is satiety as a fraction
of its maximum. The connectome reads `s` only in that arm.

### Decision E: Replay and persistence

PPO replays from the buffered preprocessed states. The step's two concentrations and its
`food_ingested` are therefore appended to the preprocessed state vector, and the topology splits them
off. Replay then sees exactly what the rollout saw.

- **The concentration update is outside the gradient.** It is computed from the rollout's settled
  activity and detached, so the concentrations are part of the state the policy responds to.
- **What learning can shape:** the receptor strengths, the ingestion gain, and through `w_chem` the
  release cells' activity. That last effect works through returns, not through a direct gradient.

### Decision F: Knockouts

`modulator_knockout` takes `none`, `tph1` (serotonin held at 0) or `pdfr1` (every PDFR-1 effect
removed: the excitatory current on AIY, RIM and RIA, and the inhibition of NSM).

**Why only two.** In this model each modulator acts through one receptor, so a *mod-1* knockout would
equal *tph-1* and a *pdf* knockout would equal *pdfr-1*. In the worm, *tph-1* also removes
serotonin's other receptors; here there are none to remove.

- **Acute:** set at evaluation on a trained, intact network. This is the primary reading.
- **Raised:** set in the training config. Reported beside.
- **When off:** with `modulators: false`, the brain is byte-identical to before, and a connectome
  checkpoint saved before this change still loads.

### Decision G: The readings

The cell is D.1's lawn cell, with stage 0's settings: entropy 0.005, Emmons 2024, depth 4. The
substrate's synapse signs are grounded in the transmitter atlas and held under PPO (Decision H).

**Gates, in order:**

1. The modulated connectome's intake beats its untrained floor's, paired by seed.
2. **The modulated connectome dwells:** D.1's dwelling share against its floor, classified at stage
   0's minimum of 0.278, passing on significant dwelling of any size (`move_wt`, or `below` with a
   positive mean). A modulated network that does not dwell has no state for a knockout to shift; it
   need not dwell as much as stage 0 did. This was decided with the user after the sign-grounded
   pilot put the panel near stage 0's bar.

**The primary readings.** Two, one per mutant, each the knockout's **mean on-lawn speed** minus the
intact network's, paired by seed:

- **acute *tph-1*:** the published direction is faster, since the mutant roams more;
- **acute *pdfr-1*:** the published direction is slower, since the mutant dwells more.

Each is classified by `mc.classify` at **0.009 mm/s**, a third of the 0.028 mm/s between real worms'
calibrated states' typical dwelling and roaming speeds. It is set from the biology, not from the pilots' effects:
those were too small and too mixed in sign to anchor one. The q values are Holm-adjusted across the
two. The verdicts: the mutant direction reproduced, the
opposite direction, below the minimum, no effect at the minimum, or unresolved.

**Why speed, not the dwelling share** (decided with the user after the first pilot, 2026-10-11). The
first pilot's policies sat near one state per seed: two seeds dwelt in every window and two roamed
with brief dwells. On the classifier's share, a dwelling seed cannot dwell more and a roaming seed
cannot dwell less, so each knockout could move only on half the seeds, and the all-seed difference
was diluted by the rest. Speed is the continuous quantity the states are classified from. It shows
shifts the classifier's threshold hides, and real worms' calibrated states give it a scale. Their
typical speeds, the exponentiated means of the HMM's log-speed emissions less its 0.001 mm/s floor,
are 0.013 mm/s dwelling and 0.041 mm/s roaming. Those are close to geometric means, while the
reading is the arithmetic mean of on-lawn window speed. The dwelling share stays as the gate and is reported
beside the speed readings.

**Reported beside, never read as verdicts:**

- each acute knockout on the dwelling share;
- the raised *tph-1* and *pdfr-1* arms against the intact one, on both measures;
- a satiety-link arm, in which NSM's ingestion drive is amplified by hunger, `(1 + (1 − satiety))`,
  as fasting amplifies NSM's food response (Rhoades 2019);
- the learned receptor strengths and ingestion gain;
- the concentrations in each state;
- the modulated connectome against stage 0's unmodulated one, paired by seed, noting that stage 0's
  signs were random.

### Decision H: Grounded signs, held under PPO

**The finding** (the first pilot, seeds 9401–9404, random signs). Stage 0 and the first modulated
arms drew every synapse's sign at random. What a receptor cell does to locomotion was then arbitrary
per seed, and nothing tied a knockout's direction to the worm's. Acute *tph-1* slowed both roaming
seeds, the opposite of the mutant, which fits an arbitrary direction. Acute *pdfr-1* slowed one, the
mutant's direction.

**The change.** The modulated arms ground signs in the transmitter atlas (`synapse_signs: atlas`) and
enforce them (`enforce_synapse_signs: true`).

- **The atlas's reach.** It grounds 3,176 of the 3,709 chemical synapses by the presynaptic cell's
  transmitter, including every output of AIY, RIF, RIM, RIA and the PDF release cells, but none of
  ASI's or NSM's.
- **Enforcement is new under PPO.** It was implemented only by the plastic rules, and under PPO the
  flag was ignored. PPO now clamps each grounded synapse back to its sign after every optimiser
  step, as the plastic rules do.
- **What grounding does not settle.** Grounding is necessary for a knockout's direction to mean
  anything, not sufficient: ungrounded synapses and learned magnitudes can still route around it.
- **Glutamate is graded excitatory throughout.** The atlas rule signs a synapse by its presynaptic
  cell's transmitter, and makes every glutamatergic synapse excitatory. In this circuit the
  best-known exception lands on a receptor cell: AWC's glutamatergic input to AIY is inhibitory,
  through the GLC-3 chloride channel (Chalasani et al. 2007). AIY carries both receptors, so its
  sensory drive here has the wrong sign at that synapse. Per-synapse receptor signs would fix it, and
  are outside this change.
- **A positive control first.** The sign-grounded connectome has not been tested on the cell. The
  pilot reads its gates before anything is registered: it must learn and dwell.

The random-sign arms and their pilot stay as committed. They are the record of why the reading
changed.

## Risks

- **The trained network may not use the pathways.** Its receptor strengths may shrink toward zero,
  and the knockouts would then read no effect. The strengths are reported so that this is visible.
- **Release does not have a direct gradient.** PPO shapes the release cells only through `w_chem` and
  returns. If the field's effect comes from the NSM ingestion drive, that is the biology's sensory
  path.
- **The cell's economics already make dwelling optimal** (D.1, stage 0). The modulators may add
  little to a policy that dwells without them. That is a finding about where states come from here,
  and the knockouts measure it.
