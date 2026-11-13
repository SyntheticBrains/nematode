# Reconciliation: your seeds and claims against Part A

Written 2026-11-14. Part A was committed before your list arrived, so the comparison is fair in
both directions. Every source on your list was re-opened; "checked" below means I read the source
myself today.

## Summary

- **Your ten sources were all already in Part A.** None was missing. Your list sharpened three
  details, and it prompted a full-text read of Mi et al. 2022, which Part A had only at abstract
  level. That read changed it from a footnote to the second key worm result, and LANDSCAPE.md,
  IDENTIFIABILITY.md and SPIKE_PLAN.md are updated.

- **Part A has about twenty items your list lacks.** The ones that weigh most: the Randi atlas is
  CC-BY-4.0 on DANDI; a second stimulate-and-record dataset exists (Dunn 2025); Creamer's absolute
  agreement is 0.14 against a ceiling of 0.17; flyvis was task-trained, not fitted to recordings; theory
  says predictions become identifiable before parameters do; and the project's own record (LB 126 to
  129\) bears directly on your claims.

- **Your five claims:**

  | # | Claim | Verdict |
  |---|---|---|
  | 1 | Behaviour is low-bandwidth; benchmarks give better tests but no faster search | holds in part |
  | 2 | Perturbation data carry causal information behaviour cannot; the strongest constraint | holds in part |
  | 3 | A differentiable simulator is likely the single biggest speed-up | does not hold as stated |
  | 4 | Add fidelity only where it improves prediction | holds as a principle; under current data it excludes most of the goal |
  | 5 | Plasticity last, because its data are thinnest | holds for long-term learning, not for short-term adaptation |

## 1. Your sources, verified

| Source | Your summary | Verified | Correction or nuance |
|---|---|---|---|
| Creamer, Leifer and Pillow, bioRxiv 2024.09.22.614271 | a shuffled connectome did much worse; extra connections did not help | yes, v4 full text, checked | "Shuffled" is **relabelled**: rows and columns permuted, topology kept, not degree-preserving. Fully connected "performed no better" on perturbation responses but "slightly outperformed" on correlation structure, and its edge on putative extrasynaptic pairs moved from p = 0.011 (v3) to p = 0.1 (v4). Absolute agreement is r = 0.14 against the data's own 0.17; v4 says 82%, v3 said 92%. Fitted by EM, not gradients |
| Randi et al. 2023, Nature (from memory) | the atlas | yes, arXiv full text, checked | 23,427 pairs (23,433 in the published version and on DANDI), at least 1,314 (6%) functionally connected; anatomy predicts propagation with R² < 0; 53 connections lost in unc-31. **CC-BY-4.0 on DANDI 001075**, checked |
| Mi et al. 2022, ICLR | withheld neurons predicted better with the connectome | yes, full text from the Janelia-hosted PDF, checked | Your plural title is the paper's own ("models"); the ICLR listing has it singular. Data: Yemini 2021, 21 immobilised worms, 170 head neurons, 5 min at 4 Hz. Withheld neurons: r = 0.319 with connectome counts against 0.008 dense. **Held-out worms: every variant r = 0.44 to 0.47**, so the connectome did not help there |
| arXiv 2501.00278 (Morrison and Young) | premotor model | yes, full text | Published as PLoS Comput Biol 21:e1013818 (Dec 2025), retitled "data-driven biology-based". No held-out split; "identifiability has not been established" |
| Warrington et al. 2019, arXiv 1907.11075 | virtual patch clamp | yes, full text | synthetic data only; two coupling parameters recovered |
| WormID (wormid.org) | 118 datasets from 5 labs | yes, from the paper's abstract (Sprague et al., bioRxiv 2024.04.28.591397), checked | wormid.org itself gives no count, only "a growing corpus", in NWB on DANDI |
| arXiv 2411.12091 (Simeon et al.) | 12 studies plus connectome data from 3 sources | yes, abstract checked | "9 connectome annotations based on 3 primary electron microscopy studies", plus one signal-propagation study. Data only, no models |
| Atanas et al. 2023, Cell | freely moving with behaviour | yes (preprint full text; Cell abstract) | 68 recordings on WormWideWeb, 38 on DANDI 000776 (CC-BY-4.0) |
| Dryad doi:10.5061/dryad.w9ghx3g4v | 2026 copper boundary dataset | yes, landing page checked | **Baskoylu and Flavell**, published 2026-04-15: 27 freely moving NeuroPAL animals, NLS-GCaMP7f, about 16 min each, 320 MB, with velocity, reversals, distance to copper, pumping and head angle. It reuses Atanas's methods but is a separate dataset. Dryad publishes under CC0 (the inventory recorded CC0-1.0; the landing page I fetched did not print the licence name) |
| CeNGEN, Taylor et al. 2021 (from memory) | per-neuron expression | yes | 118 classes, 128 types; at threshold 2, true-positive rate 0.81 and false-discovery rate 0.14. It fixes channel sets, not conductances |
| Ripoll-Sánchez 2023; Beets 2023 (from memory) | peptide receptor maps | yes | Expression and in vitro potencies; edge weights count possible pairings, not strength. Beets' full EC50 data are "upon request" |
| Lappalainen et al. 2024, Nature (from memory) | connectome-constrained fly visual model | yes, checked | Built "without any measurements of neural activity": trained on a motion task, then compared with recordings. It is precedent for connectome-plus-task, not for fit-first |

## 2. What you had that I missed

Nothing of substance. Three refinements:

- **Mi et al. 2022 in full.** Part A had it at abstract level because the agents could not parse the
  PDF. Your seed prompted another attempt, which worked. Its held-out-worm result (the connectome
  makes no difference to whole held-out animals) qualifies its held-out-neuron result, and it is now
  the spike's alternative target.
- **The plural title** of Mi et al.
- **The "3 primary EM studies" framing** of arXiv 2411.12091, more precise than Part A's file count.

## 3. What I found that you missed

In order of weight for the decision. References are to Part A's documents.

01. **The Randi atlas is CC-BY-4.0 on DANDI 001075.** The project's own record
    (`data/connectome/PROVENANCE.md`) says it cannot be vendored. DATA_INVENTORY.md.
02. **A second stimulate-and-record dataset:** Dunn et al. 2025 (Kato lab; AWA, SMD, RIV, RIA
    stimulation; DANDI 001623, CC-BY-4.0). DATA_INVENTORY.md § 1.
03. **Creamer's ceiling and revision:** absolute r = 0.14 against 0.17, 82% in v4 against 92% in v3, and
    a fully connected model no worse. LANDSCAPE.md § 1.1.
04. **Lappalainen is task-first** (your source, my reading).
05. **Identifiability theory:** predictions become identifiable before parameters, which are "never
    precisely recovered" (Beiran and Litwin-Kumar); stimulation identifies the effective Jacobian
    (Pospisil 2024); spontaneous-data bias does not shrink with data (Das and Fiete 2020); partial
    observation yields the wrong mechanism (Qian 2024). LANDSCAPE.md § 3.
06. **The project's own evidence:** gradient imitation of C.1 took 3.3 hours and gave the wrong
    behaviour (LB 126); Creamer's weights as a prior changed nothing (LB 072, 073); behaviour does not
    identify C.2's internals (LB 127 to 129); and the October direction review already weighed and
    deferred a version of this idea. ORIENTATION.md.
07. **Toyoshima 2024:** per-animal connectome models' virtual optogenetics matched Randi at
    r = 0.25 to 0.49.
08. **Simeon 2024:** pooled models on eight open datasets barely beat copying the last value.
09. **nih-niddk-mbs/r-elegans:** a differentiable connectome-constrained JAX worm, active November
    2026, reporting no calcium gain yet. The nearest parallel effort.
10. **Li et al. 2026 (fly):** fit-first at whole-brain scale on one workstation GPU, held-out r = 0.42
    against a 0.47 ceiling, no wiring null.
11. **The calcium link in worm neurons:** decay half-times of 2.7 to 18 s by indicator (Ding 2023);
    about 1 s from voltage to fluorescence in AWA (Liu 2018); brief RIM voltage events absent from
    calcium (Azimi Hashemi 2019).
12. **The neuromodulator gap:** no release measurements; the unc-31 arm is 18 animals, processed only
    and unlicensed; no stimulation data in tph-1, pdfr-1 or tax-4.
13. **Plasticity data:** Liang 2025 (856 worms, separate groups); rule inference is sloppy even with
    extensive data (Mehta 2023); worm learning models impose rather than infer plasticity (Hironaka and
    Sumi 2025).
14. **Variability:** 43% of connections differ between isogenic animals (Witvliet); averaging neurons
    across animals destroys behavioural information (Brennan and Proekt).
15. **Tooling limits:** long horizons are Jaxley's stated open problem; jax-metal is abandoned; PyTorch
    on Apple's GPU has no float64.

## 4. Your five claims

### Claim 1. Behaviour is low-bandwidth, so behaviour benchmarks give better tests but do not speed up parameter search

**Verdict: holds in part.** True for the benchmark's summary statistics and for a connectome model's
internal parameters; not true of behaviour as such, and the "better tests" half is weaker than it
sounds.

- **For.** Benchmark v4 gives about 28 numbers against C.2's 137 parameters. Wild-type behaviour did not
  identify C.2's perturbation responses (LB 128). C.1's single parameters missed their truth by up to
  0.86 of their range with behaviour matched (LB 121). In a worm learning model, networks of opposite
  sign fit behaviour equally well (Hironaka and Sumi 2025); behaviour-only rule inference is sloppy
  (Mehta 2023).
- **Against.**
  - *Bandwidth is a choice, not a property.* The benchmark compresses Scheer's 2,847 animals of 40
    minutes at 3 frames/s into 24 statistics. The primitives' bout starts are competing Poisson
    hazards, so the exact log-likelihood of a real track given its senses is differentiable and uses
    far more of the data (ORIENTATION.md § 6.2, untested).
  - *Behaviour under perturbation identifies the right layer.* Ablation behaviour resolved the signs
    of 7 of 9 classes (Wicks 1996); knockout training aligned model units with neuron types (Cowley
    2024); fitting C.2 with its mutants made serotonin and PDF go the real way (LB 128, 129).
  - *As tests, behaviour benchmarks discriminate wiring poorly.* An optimised controller walks a fly
    body with a worm's wiring (the digital sphinx); a random network with the same neurons and edge
    count also does klinotaxis (Chen 2022, second-hand via the project's October direction review).
  - *Behaviour fits are slow mostly because each evaluation is costly:* a closed-loop simulation of
    200 animals over 2,400 to 5,400 steps, about 28 hours per C.2 fit.
- **Restated:** behaviour summary statistics are low-bandwidth for a connectome model's internals.
  Behaviour likelihoods and mutant behaviour carry more, and identify state-level and neuromodulator
  effects that neural recordings do not yet reach.

### Claim 2. Perturbation data carry causal information that behaviour cannot, and are the strongest constraint available

**Verdict: holds in part.** True for the head network's effective coupling; not as "the strongest
constraint available".

- **For.** Stimulation acts as an instrument that breaks confounding by common input (Pospisil 2024);
  the bias of spontaneous-data inference does not shrink with more data (Das and Fiete 2020);
  partially observed fits can get the activity right and the mechanism wrong, and perturbation is named
  as the strongest check (Qian 2024); a relabelled connectome fails on perturbation responses
  (Creamer).
- **Against.**
  - *Behaviour carries causal information too.* Mutant and ablation phenotypes are interventions
    (Scheer's mutants; Wicks; Cowley). What behaviour lacks is the internal map, not causality.
  - *The worm's perturbation data are weak in practice.* They agree with themselves at r = 0.16 to
    0.17; 6% of pairs are significant; animals are immobilised under levamisole, which acts on
    acetylcholine receptors; the head only, with no body motor neurons; a strain "significantly less
    active than WT" that over-expresses a light-sensitive receptor.
  - *They identify what they perturb:* the effective Jacobian, not parameters (Pospisil; Beiran), with
    gap and chemical paths merged, and confounds remain when stimulation is not cell-specific
    (Lepperød 2023).
  - *Other constraints are stronger for some layers:* the EM connectome itself (relabelling breaks
    the fit), expression polarity for signs (1 discordant of 12, Fenyves 2020), and behaviour with
    mutants for neuromodulator effects (IDENTIFIABILITY.md § 5).
- **Restated:** perturbation recordings are the only data that identify the head network's effective
  causal coupling. They are not the strongest constraint on signs, on neuromodulator effects, or on
  anything outside the head.

### Claim 3. Making the simulator differentiable is likely the single biggest speed-up

**Verdict: does not hold as stated.**

- **For.** Gradient fitting scales when the objective is a likelihood on traces: about 15 million
  parameters on one GPU (Li 2026); a 1,390-parameter model in about 100 iterations, where a genetic
  algorithm stayed two orders of magnitude worse (Jaxley). Each sep-CMA fit of C.2 takes about 28
  hours.
- **Against.**
  - *Fit-first does not need the simulator differentiable.* A neural fit runs open-loop on recorded
    stimuli; the environment, primitives and grading can stay as they are. Making those
    differentiable is 6 weeks to 6 months of work, high risk, and would optimise a surrogate that is not
    the benchmark (ORIENTATION.md § 6).
  - *The result to reproduce did not use gradients:* Creamer fits by EM in about 10 hours on 10 CPUs.
    Its result came from the data and the model class.
  - *The project already took a large compute speed-up* (the batched evaluator, 48 to 98 times faster
    per core, LB 104; then 1.35 to 1.54 times more, LB 118). Phase 10 still revised its benchmark four
    times, and C.1's verdict changed with each version. Compute was not the binding constraint.
  - *The brain is already differentiable,* and was fitted by gradient in 3.3 hours, quickly, to the
    wrong behaviour (LB 126).
  - *Gradients bring their own costs:* long horizons, chaotic dynamics, and degenerate solution sets
    where one point estimate misleads (Gonçalves 2020).
- **Restated:** the biggest speed-up is changing the objective: from 28 summary statistics searched
  black-box, to a likelihood on data that carry internal information (neural traces, or behaviour
  tracks). Gradients come with that choice. They are its consequence, not the lever.

### Claim 4. Fidelity should be added only where it improves prediction; full biophysics everywhere is slow by construction

**Verdict: holds as a principle. Under the current data it excludes most of the long-run goal by
default.**

- **For.** Full biophysics with parameters shared by group is "one of many solutions" (BAAIWorm);
  conductance-level parameters are degenerate (Prinz; Gonçalves); the successes tie parameters hard
  (flyvis: 734 for 45,669 neurons; Vishwanathan: two or three scalars).
- **Against.**
  - *What a fit can reward is capped by what the data can see.* Against 2 Hz nuclear calcium that
    reproduces itself at 0.17, and where a fully connected linear model does as well as a masked one,
    nonlinearity, intrinsic currents, rectifying gap junctions and modulator kinetics will rarely
    improve prediction measurably. The rule then rejects them because the data cannot see them, not
    because they are absent.
  - *Prediction alone can choose the wrong mechanism* (Qian 2024; Mi's dense model, "equally accurate
    but mechanistically incorrect").
  - *The closed loop needs fidelity the immobilised data cannot select:* persistent states (roaming
    and dwelling, AVA's states, AFD's bistability) need nonlinearity or slow variables, and Creamer's
    model has one fixed point.
  - *Minimal fidelity is not automatically fast.* C.2 has no neuron dynamics at all, and it compensated
    with about 130 interface parameters and a dozen post-hoc options.
- **Restated:** add fidelity where it improves prediction of held-out perturbations, held-out neurons
  or held-out behaviour, or where independent data (electrophysiology, expression) require it. Record
  "does not improve prediction" as "not visible in these data", and list what that excludes.

### Claim 5. Plasticity should come last, because its data are thinnest

**Verdict: holds for long-term associative learning; not for short-term adaptation.**

- **For.** No plasticity rule is identified anywhere (Mehta 2023); worm learning models impose
  plasticity rather than infer it (Hironaka and Sumi 2025); no same-animal before-and-after
  whole-brain imaging exists; Phase 7 found no plausible rule that learns an ungrounded substrate.
- **Against.**
  - *Short-term plasticity is inside the data a static fit would use.* AFD adapts at τ ≈ 45 s (Hawk
    2018); AWC's dynamic range shifts and recovers within 10 minutes (Cho 2016); a fifth to a third of
    encoding neurons change encoding within one recording (Atanas 2023, by version read); Randi
    stimulates every 30 s for up to 40 minutes and names plasticity as a possible source of its
    trial-to-trial variability. A model without adaptation absorbs it into weights and noise.
  - *The data are thin on rules, not on correlates.* Liang 2025 imaged 856 worms across about 78% of
    head classes after aversive learning; molecular work locates the plastic site in each paradigm
    (Hiroki 2022; Ha 2010).
  - *Last with no plan repeats Phase 7.* The goal names local learning; it needs a bridge planned in
    advance, possibly in Wormturn.
- **Restated:** fit sensory adaptation with the dynamics, because it is in the data. Leave long-term
  associative rules for last, located by molecular mechanism and tested on separate-group learning
  data.

## 5. What changed in Part A because of this

- LANDSCAPE.md § 1.2, § 1.8, § 4.1 and the source table: Mi et al. upgraded to full text, with its
  numbers.
- IDENTIFIABILITY.md § 1: Mi's held-out-worm result.
- SPIKE_PLAN.md § 3: Mi et al. on Yemini 2021 (DANDI 000541, CC-BY-4.0) as the alternative target if
  neither data route for the Randi atlas opens, with its own bars.
- DECISIONS.md A13.
