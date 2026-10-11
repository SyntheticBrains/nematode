# 090: Removing Serotonin or PDF From the Trained Connectome Costs It Food but Does Not Shift Its Speed the Mutants' Way (Phase 8b B.3)

**Status**: completed. Both gates pass and the panel is readable, with two registered readings:
**acute *tph-1*: no effect at the minimum; acute *pdfr-1*: unresolved at this sensitivity.**

The connectome PPO brain carries a serotonin and PDF field placed by the cited circuit, with its
synapse signs grounded in the transmitter atlas and held under PPO. Each modulator was removed at
evaluation from the trained network. Neither moved its mean on-lawn speed by the 0.009 mm/s
minimum in the mutant's direction.

| registered reading, seeds 2001–2032 | mean (mm/s) | 80% CI | Holm q | state |
|---|---|---|---|---|
| acute *tph-1* − intact, on-lawn speed (+ = faster, as *tph-1* roams more) | **−0.0015** | [−0.0078, +0.0049] | 0.86 | `no_move` → **no effect at the minimum** |
| intact − acute *pdfr-1*, on-lawn speed (+ = slower, as *pdfr-1* dwells more) | **+0.0043** | [−0.0007, +0.0091] | 0.37 | `unresolved` → **unresolved at this sensitivity** |

**Both gates pass:**

- **Intake:** +2.48 [+2.35, +2.61] over the untrained floor.
- **Dwelling:** +0.474 [+0.37, +0.58] over the floor, q = 5 × 10⁻⁶. That clears stage 0's 0.278 bar too.

**Beside the readings, never read as verdicts:**

- **The network uses both modulators.** Removing either costs it food: −0.59 [−0.79, −0.40] per
  episode without serotonin and −0.42 [−0.57, −0.26] without PDFR-1, against the intact network's
  4.25.
- **On the classified dwelling share, acute *tph-1* moves the mutant's way:** +0.11 [+0.04, +0.18] less
  dwelling, carried by the seeds that switch states.
- **Networks raised without a modulator move against the mutants:** raised without serotonin they
  are slower, raised without PDFR-1 faster.
- **PPO weakens PDF's inhibition of NSM** on 29 of 32 seeds, to a median strength of 0.39.

**Date**: 2026-10-11.

**OpenSpec change**: `add-b3-modulators`.

**Pre-registration**: [supporting/090-modulators/launch.md](supporting/090-modulators/launch.md),
committed at 9cf21b00 before any panel run. It followed two pilots, a re-plan and the spec review.

## Objective

B.3 asks whether serotonin and PDF on the connectome act as they do in the worm. In the worm,
serotonin from NSM extends dwelling through MOD-1, and PDF from AVB, PVP and SIAV extends roaming
through PDFR-1 (Flavell et al. 2013). Their mutants shift the balance: *tph-1* worms roam more, and
*pdfr-1* worms dwell more.

The question is narrower than "does a model reproduce the worm". It is: **given only the anatomy and
the modulators' placement, does a reward-trained connectome come to use them as the worm does?**

Stage 0 (Logbook 089) showed the connectome dwells on this cell without modulators, so the states
are there to modulate.

## Method

**The circuit**, from [`data/neuromodulators/`](../../../data/neuromodulators/), verified against the
full texts of Flavell 2013, Ji 2021, Dag 2023, Rhoades 2019 and Ben Arous 2009:

| | release | receptor | sign | targets |
|---|---|---|---|---|
| serotonin | NSM, driven by food ingestion | MOD-1 | inhibitory | AIY, ASI, RIF |
| PDF | AVB, PVP, SIAV | PDFR-1 | excitatory | AIY, RIM, RIA |

PDF also inhibits NSM (Ji 2021), modelled as a functional link because NSM is wired only to the
pharynx.

**The field** (design Decisions B–E):

- **Concentrations.** Each is a scalar carried across an episode's steps. Each step it decays toward
  its release cells' mean activity above rest, with τ = 60 s at a 5 s step.
- **Currents.** At every settling iteration, each receptor cell and NSM gets a held current: the
  receptor's sign × a learnable strength × the concentration.
- **Ingestion.** NSM is driven by the last move's bite, as a fraction of a full bite. Taken raw, the
  drive was twenty times weaker than PDF's inhibition, and serotonin never formed.
- **Replay.** The concentrations travel in the state vector, so PPO's replay sees what the rollout
  saw.

**The substrate.** Stage 0's settings: Emmons 2024, settling to depth 4, entropy 0.005, on D.1's lawn
cell. Its synapse signs are **grounded in the transmitter atlas and held under PPO** (Decision H):

- The atlas grounds 3,176 of 3,709 chemical synapses.
- Holding them under PPO is new. The flag was previously read only by the plastic rules, and under
  PPO it was silently ignored.

**The readings** (Decision G):

- **Acute knockouts.** The intact arm's trained weights, evaluated with serotonin held at zero
  (*tph-1*) or every PDFR-1 effect removed (*pdfr-1*). Each runs the intact arm's 30 held-out
  episodes, on the same lawns.
- **The measure** is mean on-lawn speed, oriented so the mutant's direction is positive.
- **The minimum** is 0.009 mm/s, a third of the 0.028 mm/s between real worms' calibrated dwelling and
  roaming speeds, set from the biology.
- **Classification:** `mc.classify` on each reading, with Holm across the two.

**The panel:** five arms on seeds 2001–2032, 160 runs of 3,000 episodes, all of which succeeded:

- intact;
- its untrained floor;
- raised *tph-1*;
- raised *pdfr-1*;
- a satiety link, in which NSM's drive is × (1 + hunger).

### How the reading was arrived at

Two pilots changed the panel before it was registered.

**The first pilot, random signs** (seeds 9401–9404). Two seeds dwelt in every window and two roamed.

- **Acute *tph-1* slowed both roaming seeds,** against the mutant.
- **The cause.** Under random synapse signs, what a receptor cell does to locomotion is drawn per
  seed, so a knockout's direction is arbitrary.
- **The classifier's ceiling.** A seed at 100% dwelling cannot dwell more, and one at 0% cannot dwell
  less, so the classified share lets each knockout move only half the seeds.

With the user, the reading moved to speed and the substrate to grounded signs.

**The sign-grounded pilot** (seeds 9411–9418), extended to eight seeds after the spec review so that
the dwelling gate could resolve:

- the connectome dwelt as stage 0's did: +0.495, five of eight seeds;
- the knockouts' spreads sized the panel.

## Results

### Gates

| gate | reading | rule | result |
|---|---|---|---|
| intake | intact − floor, +2.48 [+2.35, +2.61] | 80% interval above zero | **passes** |
| dwelling | intact − floor, +0.474 [+0.37, +0.58], q 5 × 10⁻⁶ | significant, any size | **passes** (`move_wt` at 0.278) |

The intact network dwells on 0.49 of its on-lawn time, against its floor's 0.013. 17 of 32 seeds
dwell more than 0.278 above their floor, against stage 0's 13 of 19.

### The registered readings

- **Acute *tph-1*: no effect at the minimum.**
  - The reading is −0.0015 [−0.0078, +0.0049] mm/s; the interval lies inside ±0.009 and includes
    zero.
  - 16 of 32 seeds read positive; 7 reach +0.009 and 8 reach −0.009.
- **Acute *pdfr-1*: unresolved.**
  - The reading is +0.0043 [−0.0007, +0.0091] mm/s; the interval spans the minimum's edge.
  - The mean is in the mutant's direction, half the minimum's size.
  - 18 of 32 seeds read positive.

**Achieved spreads,** reported beside, never used to re-read:

| | achieved sd | achieved MDE | sign-grounded pilot's sd |
|---|---|---|---|
| *tph-1* | 0.028 | 0.0125 | 0.020 |
| *pdfr-1* | 0.022 | 0.0095 | 0.032 |

*tph-1*'s interval nonetheless fits inside ±0.009, which is what `no_move` asks.

### Described beside

| arm | on-lawn dwelling share | on-lawn speed (mm/s) | intake per held-out episode | seeds with bouts of both states |
|---|---|---|---|---|
| intact | 0.49 | 0.056 | 4.25 | 12 |
| acute *tph-1* | 0.38 | 0.054 | 3.66 | 10 |
| acute *pdfr-1* | 0.44 | 0.051 | 3.84 | 10 |
| raised *tph-1* | 0.62 | 0.037 | 3.76 | 8 |
| raised *pdfr-1* | 0.32 | 0.077 | 3.90 | 6 |
| satiety link | 0.58 | 0.052 | 4.16 | 11 |
| untrained floor | 0.013 | 0.127 | 1.53 | 0 |

Paired by seed against the intact arm (80% intervals). Oriented rows read positive in the mutant's
direction:

| reading | mean [80% CI] |
|---|---|
| acute *tph-1*, intake | **−0.59 [−0.79, −0.40]** |
| acute *pdfr-1*, intake | **−0.42 [−0.57, −0.26]** |
| acute *tph-1*, dwelling share, oriented | +0.11 [+0.04, +0.18] |
| acute *pdfr-1*, dwelling share, oriented | −0.05 [−0.13, +0.03] |
| raised *tph-1*, speed, oriented | −0.019 [−0.030, −0.008] |
| raised *pdfr-1*, speed, oriented | −0.021 [−0.038, −0.005] |
| satiety link − intact, dwelling share | +0.09 [−0.06, +0.24] |
| intact − stage 0 (random signs, no field), dwelling share, seeds 2001–2019 | −0.15 [−0.32, +0.03] |

**Post hoc, on the seven seeds that switch states** (at least 3 bouts of each), acute *tph-1* cuts
dwelling on six, by +0.21 to +0.89. Their speed moves less, since dwelling is classified from turning
as well as speed. This selection was not registered and is reported as description.

**The learned strengths** start at 1. Medians across 32 seeds:

| pathway | median strength | range | seeds below 1 |
|---|---|---|---|
| MOD-1 | 1.17 | 0.88–1.97 | 4 |
| PDFR-1 | 1.17 | 0.58–2.15 | 6 |
| PDF onto NSM | **0.39** | 0.05–1.60 | **29** |
| ingestion onto NSM | 1.29 | 0.49–3.02 | 7 |

**Concentrations by state.** In the intact network, serotonin's median is 0.32 in dwelling windows
and 0.36 in roaming windows. The worm's NSM is active during dwelling; nothing here reproduces that
across seeds.

## What this establishes, and what it does not

**Established:**

- **The trained connectome uses both modulators.** Removing either costs it 10–14% of its intake.
  The field is not decoration.
- **That use is not the mutants' speed shift.** Removing serotonin leaves mean on-lawn speed within
  ±0.009 mm/s, a third of the worm's dwelling-to-roaming gap, of where it was.
- **PPO turns down PDF's inhibition of NSM, consistently.** This held on 29 of 32 seeds and in both
  pilots. The learner keeps its serotonin source free of PDF's control. The worm's Ji et al. mechanism
  depends on the reverse: PDF silencing NSM during roaming.
- **The sign-grounded connectome dwells,** as stage 0's did. The positive control holds on the new
  substrate.

**Not established:**

- **Whether PDFR-1 removal moves speed the mutant's way.** The reading is unresolved; its mean points
  that way at half the minimum.
- **That serotonin's effect on dwelling, seen on the share and on the switching seeds, is real.** It
  is beside the registered reading and partly post hoc.
- **Anything about raised mutants as mutants.** Networks trained without a modulator move against
  the mutants. That fits a learner compensating through other weights, which a real mutant cannot do
  on this scale, but it is not tested here.

## Biological fidelity

- **What the learner does with the circuit is its own.** It is given the anatomy and the modulators'
  placement and signs, and it learns their strengths for reward. It weakened the one link the worm's
  state persistence rests on. Why is not tested; a plausible reading is that on this cell a serotonin
  source that tracks food, unchecked by PDF, serves foraging better.
- **States here come from the cell's economics.** As D.1 and stage 0 found, dwelling is the best
  response to a movement cost and depleting food. Most seeds sit in one state all episode, so a
  knockout has little to switch. In the worm, states switch every few minutes. Here only 7 of 32
  seeds switch like that, and that is where serotonin's removal shows.
- **Signs come from the presynaptic transmitter.** The atlas rule makes every glutamatergic synapse
  excitatory, while AWC's glutamatergic input to AIY, a receptor cell for both modulators, is
  inhibitory (Chalasani et al. 2007). H.2's per-connection sign table has that synapse right (51
  physiology-measured connections, 1,676 expression-predicted), but no brain reads it yet.
- **Release and receptors are minimal.** There is one receptor per modulator, no HSN, no other
  serotonin or PDF receptors, and a fixed 60 s timescale. *tph-1* here removes only MOD-1's effect.
- **Comparison with other models.** BAAIWorm (Zhao et al. 2024) and modWorm (2025) are fitted models
  that reproduce behaviour and in-vivo ablations (modWorm: 19 of 20 scenarios). This logbook asks a
  different question, whether a learner given the placement comes to use the modulators as the worm
  does. The answer on this cell is partly: it uses them, not in the mutants' way.

## Next

- **B.3 is met.** The modulator field exists, is tested, and has its registered reading. The answer is
  negative for *tph-1* and open for *pdfr-1*.
- **Candidates, none scheduled:**
  - **Per-connection signs.** Put H.2's table under the field, so that AWC → AIY and the other 342
    connections the per-neuron rule gets backwards carry their measured or predicted sign. It is the
    nearest fidelity step, and the sign-grounded pilot shows grounding changes what is learned.
  - **A cell where states must switch.** Knockouts can only shift a balance that exists. D.1b found
    that the lawn cells' economics favour one state per episode. A cost of time, recorded at D.1b,
    may be the lever.
  - **Freeze PDF's inhibition of NSM at the worm's strength.** This asks whether the worm's mutual
    inhibition, imposed rather than learned, gives the mutants' directions.
- **S8b** is next on the 8b order.

## Artefacts

**The registration and the panel:**

- [launch.md](supporting/090-modulators/launch.md) and [preflight.json](supporting/090-modulators/preflight.json): the registration and its gate preflight on the eight sign-grounded pilot seeds.
- [panel.json](supporting/090-modulators/panel.json): the gates, the two readings and their verdicts, the readings beside, the pairing with stage 0, and each run's readings.
- [per-run.csv](supporting/090-modulators/per-run.csv): one row per evaluated run, with concentrations by state and learned strengths.

**The pilots:**

- [pilot-states.json](supporting/090-modulators/pilot-states.json) and [pilot-per-run.csv](supporting/090-modulators/pilot-per-run.csv): random signs, seeds 9401–9404.
- [pilot-atlas-states.json](supporting/090-modulators/pilot-atlas-states.json) and [pilot-atlas-per-run.csv](supporting/090-modulators/pilot-atlas-per-run.csv): sign-grounded, seeds 9411–9418.

**To reproduce:**

```bash
uv run python scripts/analysis/lawn_states.py --logs campaigns/b3-panel/logs \
  --arms modulated_atlas modulated_atlas_frozen modulated_atlas+tph1 modulated_atlas+pdfr1 \
    modulated_atlas_tph1 modulated_atlas_pdfr1 modulated_atlas_satiety \
  --learner-arm modulated_atlas --floor-arm modulated_atlas_frozen --minimum 0.278 \
  --knockout-minimum 0.009 --knockout-measure speed --seeds $(seq 2001 2032) --episodes 30 \
  --compare-stage0 docs/experiments/logbooks/supporting/089-connectome-lawns/panel.json \
  --out panel.json --csv per-run.csv
```

The campaign directories and session records are archived off-repo.
