# Identifiability: what the data can and cannot pin down

Written 2026-11-14 from LANDSCAPE.md, DATA_INVENTORY.md and ORIENTATION.md. Sources are cited by
the short names used there.

## How to read this

**The marks.** For each parameter class:

- **well constrained:** the data alone narrow it to a range small enough that a fit's answer would
  not change a prediction that matters;
- **constrained with priors:** the data narrow it only once a named prior (anatomy, expression,
  literature values) fixes part of it;
- **not constrained:** no available dataset narrows it meaningfully, with or without priors.

**Two levels, which the literature keeps apart.** A fit can identify a *prediction* without
identifying the *parameters* behind it. Beiran and Litwin-Kumar find connectome-constrained networks
predict unrecorded neurons once enough are recorded, while "the unknown parameters are never
precisely recovered". Pospisil et al. show stimulation identifies the network's effective causal map
(the Jacobian of its dynamics), not its biophysical parameters. Degeneracy at conductance level is
the norm (Prinz; Gonçalves). Each class below is therefore marked at both levels where they differ.

**Two kinds of data.** Each class is judged against the neural datasets fit-first would use, and,
for a fair comparison, against the behaviour statistics the current approach fits.

**The budget of each.**

- *Neural perturbation data* (Randi's atlas): 23,433 head pairs measured, at least 1,310 of them (6%)
  significantly connected, each a response time course, from 113 animals. The data agree with
  themselves at r = 0.16 to 0.17 between halves (Creamer), so most of each response is noise or
  animal-to-animal variation. Creamer's mask leaves about 2,000 to 2,200 free weights over 154 head
  neurons (9.4% of pairs; the vendored table holds 2,011).
- *Behaviour statistics* (benchmark v4): 24 wild-type statistics plus one or two per mutant, about 28
  numbers, against C.2's 137 parameters. In C.1, single parameters missed their true value by up to
  0.86 of their range while behaviour matched (LB 121). Wild-type behaviour did not identify C.2's
  perturbation responses (LB 128).

Both are under-determined at the parameter level. They under-determine different things.

## 1. Synaptic weights and signs (chemical)

**What constrains them.**

- *Neural:* Randi's atlas gives effective, multi-path responses to single-neuron stimulation in the
  head. Creamer's connectome-masked linear model, fitted to it, predicts held-out animals' responses
  at 82% of the data's own reproducibility, while a relabelled connectome does poorly. Dunn 2025 adds
  stimulation of AWA, SMD, RIV and RIA. The silencing sets (Kato's AVA, Uzel's hubs, Kaplan's SMD and
  motor neurons) constrain the outgoing influence of the silenced nodes through correlations.
- *Behaviour:* in C.2, 3,709 edges share two fitted scales and four group gains; per-edge magnitudes
  are synapse counts under a hand-chosen normaliser. Unconstrained fits switched inhibition off
  (LB 127). Behaviour constrains essentially no individual edge.
- *Signs:* expression-based polarity (Fenyves 2020) covers about 73% of synapses and disagreed with
  measured polarity at 1 of 12 interneuronal connections; the transmitter atlas (Wang 2024) and 67
  physiology overrides cover more. Randi measured AFD to AIY excitatory where expression predicts
  inhibitory, and about 11% of significant connections inhibitory, against about 20% predicted.

**Limits.** Head neurons only: Creamer's weights cover 1,049 of 3,709 chemical edges and none onto a
body motor neuron. Chemical and gap paths are not separable in these data. Fitted weights correlate
poorly with synapse counts, and a fully connected model does as well as the masked one, so the data
do not show that the anatomical mask is the right one, only that a relabelled mask is worse. The
model is linear, under levamisole (which acts on acetylcholine receptors, the main excitatory
transmitter), at 2 Hz in calcium. Extrasynaptic signalling contributes at least 53 connections.

**Marks.**

| Level | Mark |
|---|---|
| Effective coupling among head neurons, as a predictor of perturbation responses | **constrained with priors** (the connectome mask and sign predictions as priors; the data's reproducibility as the ceiling) |
| Individual synaptic strengths, as mechanistic parameters | **not constrained** |
| Signs of edges on short paths | **constrained with priors** (expression polarity, transmitter identity, physiology overrides) |
| Anything in the body (ventral cord, motor neurons) | **not constrained** by perturbation data; Kaplan's whole-nervous-system recordings give correlations only |

**Priors that could help.** Expression polarity and transmitter identity already fix most signs; a
prior centred on signed synapse counts (Pospisil's IV-Bayes form) reduced error by an order of
magnitude in their linear simulations. Neither narrows magnitudes much in the worm, where counts and
fitted weights correlate weakly (Creamer).

## 2. Gap-junction conductances

**What constrains them.**

- *Neural:* nothing separates electrical from chemical coupling in vivo. Creamer merges them in one
  weight; electrical synapse counts correlate with fitted weight magnitudes at r = 0.107. Toyoshima's
  inferred functional network resembled the gap-junction network more than the chemical one.
  Kaplan's whole-nervous-system recordings constrain head to ventral-cord coupling, which is
  gap-dominated (AVA and AVB onto A- and B-class motor neurons), through correlations.
- *Electrophysiology:* rectifying AVA to VA5 gap junctions (Liu 2017), as figures.
- *Behaviour:* one fitted global scale in C.2. In Phase 9, the gap normaliser, a hand choice, was the
  source of a wiring effect the project had carried for two phases (LB 103).

**Mark: not constrained** as conductances separate from chemical coupling. At the level of one or a
few class-wide scales, **constrained with priors** if fitted jointly with perturbation data.

**Priors that could help.** EM gap counts (White, Varshney, Cook 2019; Witvliet, Cook 2025 and Yim omit
gap junctions) and the innexin expression atlas (Bhattacharya 2019: 14 innexins across 118 classes,
L1 to adult), which says which pairs can couple and which may rectify. Together they narrow the
*structure* (which pairs, which direction) but barely the *magnitudes*, given r = 0.107 between counts
and fitted weights. unc-7 and unc-9 mutants would separate gap from chemical paths, but no
stimulate-and-record data exist in them.

## 3. Time constants

**What constrains them.**

- *Neural:* whole-brain imaging resolves dynamics slower than about 1 s. Randi's per-pair kernels and
  Creamer's 22.5-s input filters give response timescales; CePNEM's per-neuron encoding timescales
  run from 0.9 to 31.7 s in freely moving animals; Ji 2021 shows NSM rising about 23 s before a state
  transition.
- *Confound:* indicator kinetics are of the same order in worm neurons: decay half-times of 2.7 s for
  GCaMP6f and 18 s for jGCaMP7s in ALM, mixing the neuron's response with the sensor's (Ding 2023). In
  AWA, depolarisation appeared in fluorescence after about 1 s (Liu 2018).
- *Electrophysiology:* membrane properties (capacitance, input resistance, I-V curves) for about 15
  classes.
- *Behaviour:* C.2's neurons have no time constants; its two modulator timescales are fitted to bout
  statistics (Flavell's dwelling τ ≈ 482 s, roaming 75 and 520 s, are the targets).

**Marks.**

| Level | Mark |
|---|---|
| Effective timescales of seconds to minutes, per neuron or class | **constrained with priors** (an indicator model with literature kinetics as the prior; with it, imaging pins these down well) |
| Membrane time constants below about 0.5 s | **not constrained** by imaging (below the 2 Hz sampling and the indicator); constrained for about 15 patched classes |

**Priors that could help.** Indicator kinetics from the indicator papers (mammalian and fly half-decays
of 84 to 280 ms for jGCaMP7f and jGCaMP8, Zhang 2023) and from the one worm measurement (Ding 2023);
membrane capacitance scaling with cell size (Goodman 1998). Tying by class, as flyvis did with one τ
per cell type, cuts the unknowns from 302 to 118.

## 4. Intrinsic currents

**What constrains them.**

- *Electrophysiology:* patch-clamp recordings for about 17 classes (ASER, RIM, AIY, AFD, AWA, AWCon,
  RMD, AVA, AVB, AVL, DVB, RIS, DVC, VA5, VB6, VD5, DD3), all dissected. BAAIWorm states "less than 20%" of
  neurons have been recorded. Conductance models exist for about ten of them (Nicoletti 2019 and
  2024, Naudin 2022, Liu 2018, Jiang 2022).
- *Neural imaging:* indirect evidence only: AFD's bistability, AWA's calcium spikes, RMD's plateaus,
  AVA's persistent states.
- *Behaviour:* nothing in C.2 (no intrinsic currents); C.1 has none either.

**Mark: not constrained** for the roughly 285 neurons without recordings. For the patched classes,
**constrained with priors**, and even there the fits are not unique: fitting traces and I-V curves
together gives a Pareto set (Naudin 2022), and conductances stay degenerate with the channel set fixed
(Gonçalves 2020).

**Priors that could help.** CeNGEN's expression per class (true-positive rate 0.81, false-discovery
rate 0.14 at threshold 2) fixes *which* channels a neuron carries, which is how Nicoletti 2024 used
it. That is a discrete narrowing, from any current to a named set of about 5 or 6. Gating kinetics
then come from heterologous expression or other species, and maximal conductances remain free. No
paper found quantifies how far expression narrows a conductance posterior.

**A point on fit-first's own rule.** Fit-first adds biophysical detail only where it improves
prediction. Against 2 Hz nuclear calcium, fast intrinsic currents rarely will, except where they make
slow phenomena (bistability, plateaus, calcium spikes). So fit-first would leave most intrinsic
currents out by design. That is consistent with the rule, and leaves this part of the long-run goal
unaddressed.

## 5. Neuromodulator release and effects

**What constrains them.**

- *Release and diffusion:* nothing. No release or diffusion time course for any worm peptide was
  found; serotonin and PDF release were not measured in Flavell 2013, Ji 2021 or Dag 2023; in vivo
  peptide sensors are described for mouse and fly only (Beets and Watteyne 2025).
- *Effects, neural:* Randi's unc-31 arm (18 animals, processed data only, no licence, 0.3-s pulses
  against 0.5 s) gives the lumped contribution of dense-core-vesicle release to head propagation.
  Whole-brain imaging during NSM activity (33% of neurons associated, Dag 2023), in a tdc-1 background
  (Kramer 2026), in flp-13 (Pradhan 2025), under lethargus and npr-1 (Nichols 2017), and starvation
  and daf-2 (Skora 2018) constrain state-level effects through correlations. Nothing has stimulation
  in tph-1, pdfr-1 or tax-4 backgrounds.
- *Effects, behaviour:* state durations (Flavell 2013), mutant directions (Scheer: tph-1 roams more,
  tax-4 less; pdfr-1 unresolved) and Ji 2021's patch assay. C.2 uses these, with the mutants inside
  the objective since LB 129.
- *Placement priors:* Bentley 2016 and Ripoll-Sánchez 2023 (expression, in vitro potencies, assumed
  ranges), Beets 2023 (459 validated pairs, EC50 0.1 pM to 22 µM), Dag 2023's knock-in receptor map.

**Marks.**

| Level | Mark |
|---|---|
| Release kinetics, diffusion, concentrations | **not constrained** |
| Where each modulator can act, and its sign | **constrained with priors** (receptor maps; G-protein coupling for some receptors: 16 monoamine receptors fixed in the project's layer, and a draft literature table resolving 24 of 54 free signs, LB 128) |
| Effect magnitudes and timescales | **not constrained** by neural data. By behaviour and mutants, an effective state-level timescale and a few directions are fitted in-sample (C.2), which is the most any approach currently reaches |

**Priors that could help, and how much they narrow.** Little. The peptide network admits more than ten
times the synaptic network's density (0.344 against 0.025 at short range), so placement priors leave
most pairs possible. Potencies span eight orders of magnitude. Distance models change density only
modestly (0.344 short range against 0.443 mid range). Receptor coupling fixes signs where it is known.
In the one fitted whole-brain modulator model found (Kringelbach 2020, human), even two gains could
not be separated.

**For the comparison.** This is the one class where behaviour and genetics currently carry more
information than neural recordings do. Fit-first does not reach it any better than the current
approach.

## 6. Plasticity rules

**What constrains them.**

- *Neural:* learning paradigms with calcium correlates, all comparing separate groups of animals: PA14
  aversive learning (Ha 2010: sensory responses unchanged; Liang 2025: responses changed in about 35%
  of sensory types, 856 worms), butanone conditioning (Cho 2016; Pritz 2023), salt memory (Hiroki
  2022: basal glutamate release and a receptor-dependent sign flip), thermotaxis (Hawk 2018: AFD
  adaptation at τ ≈ 45 s warming, about 14 min cooling), tap habituation (reviewed by Bozorgmehr
  2013). No whole-brain imaging of the same animals before and after learning was found.
- *Behaviour:* the worm learning model found (Hironaka and Sumi 2025) imposes a sign switch at one
  synapse; a network with the opposite sign fits equally well. Phase 7 found no biologically
  plausible rule that learns this substrate (LB 069).
- *Elsewhere:* inferring a rule by gradient through whole trajectories "fails to identify a unique,
  sparse solution even with extensive data" (Mehta 2023, on 18 flies' behaviour). Fly mushroom-body
  models take their rule from experiment (Jiang and Litwin-Kumar 2021).

**Mark: not constrained.** For sensory adaptation specifically (AFD, AWC), the time constants and
direction are measured, so that narrow case is **constrained with priors**.

**Priors that could help.** The molecular mechanism per paradigm locates the plasticity: ASER to AIB
for salt memory, AFD's threshold for thermotaxis, ADF serotonin and RIA, SMD and RIM for PA14,
GLR-1 and CREB for long-term habituation. That narrows *where* a rule acts to a handful of sites per
paradigm, which is a large narrowing of structure. It does not narrow the rule's functional form or
parameters.

## Summary table

| Class | Neural data (fit-first) | Behaviour data (current) | Mark | Priors that help | Narrowing |
|---|---|---|---|---|---|
| Chemical weights and signs | Randi atlas, Dunn 2025, silencing sets (head only) | almost nothing per edge (two scales) | **constrained with priors** for head effective coupling; **not constrained** per synapse and in the body | connectome mask, expression polarity, transmitter atlas, signed-count prior | signs: most; magnitudes: little (counts predict weights weakly) |
| Gap conductances | none separable from chemical; correlational head-to-cord | one global scale | **not constrained**; a few class scales **with priors** | EM counts, innexin atlas, unc-7/unc-9 (no data) | structure yes, magnitudes barely |
| Time constants | slow effective timescales from imaging | none in neurons; two modulator timescales | **constrained with priors** (slow, given an indicator model); **not constrained** (fast, membrane) | indicator kinetics; class tying (302 to 118) | large for slow timescales |
| Intrinsic currents | about 15 patched classes; indirect signs in imaging | none | **not constrained** (about 285 neurons); **with priors** for patched classes | CeNGEN channel sets; heterologous kinetics | channel set yes; conductances unquantified |
| Neuromodulator release and effects | unc-31 contrast (18, unlicensed); state correlations | state durations and mutant directions, in-sample | **not constrained** (release, magnitudes); **with priors** (placement, sign) | receptor maps, G-protein coupling | small: dense placement, potencies over eight orders |
| Plasticity rules | separate-group learning correlates | Phase 7 negative; behaviour does not discriminate rules | **not constrained**; sensory adaptation **with priors** | molecular mechanisms per paradigm | location yes, form no |

## What this says about the hypothesis (without a verdict)

- **No parameter class is well constrained by the data alone.** What is well constrained is a
  prediction: the head network's effective response to single-neuron stimulation, up to a ceiling the
  data set themselves (r = 0.17 between halves).
- **Neural data and behaviour data identify different layers.** Perturbation recordings reach
  effective head coupling and slow timescales, which behaviour statistics reach not at all. Behaviour
  and mutants reach state-level neuromodulator effects, which neural recordings currently reach no
  better. Neither reaches gap conductances separately, intrinsic currents broadly, modulator release,
  or plasticity rules.
- **The brief's worry is partly right.** Fit-first would move the bottleneck from "behaviour does not
  identify the internals" (LB 127, 128) to "neural data identify the head's effective coupling and
  little else". It does not move it for neuromodulator dynamics or plasticity, the two layers the
  long-run goal names that no data constrain.
- **The closed-loop bridge is the weakest link.** The perturbation data cover the head; the closed
  loop needs motor neurons and a readout to locomotion, which no perturbation dataset reaches.
  Freely moving recordings (Atanas 2023, Hallinen 2021) constrain the readout, but only
  correlationally.
