# 134: The Connectome's Specific Wiring Predicts Measured Neural Responses Unfitted, and Behaviour Fitting Adds Nothing to It (Phase 10 / the fit-first investigation)

**Status**: `completed`. **H0 passes; M-a is above chance, not above its unfitted floor.**
Established on Creamer et al.'s measured responses to single-neuron stimulation:

- C.2's encoder at the box's centre, on the true wiring, beats all 19 degree-preserving rewirings by
  about 5.5 times the registered minimum;
- C.2's behaviour-fitted answer is not distinguishable from that unfitted network.

Unresolved:

- whether a neural fit would raise the score;
- whether any of this holds on a licensed, complete stimulation record.

No earlier verdict is conditioned.

**Date**: 2026-11-14.

**OpenSpec change**: none. These are two descriptive readings from a side investigation into
fitting connectome models to neural recordings. No package code changed. The investigation's record
is [investigation/](../../../investigation/), and its conclusion is
[RECOMMENDATION.md](../../../investigation/RECOMMENDATION.md).

**Pre-registration**:
[supporting/134-fit-first-neural-readings/launch.md](supporting/134-fit-first-neural-readings/launch.md).
Both readings were registered and committed before they were computed, and each was computed once.

## Objective

The investigation asked whether fitting a connectome-constrained model to neural recordings would
reach the project's long-run goal faster than fitting to behaviour. Its spike stopped at
fact-finding: neither published fit can be reproduced as specified on licensed materials
([SPIKE_REPORT.md](../../../investigation/SPIKE_REPORT.md)). Two cheap readings remained possible
without a fit, and both bear on Phase 10's own question of what the connectome contributes to
fidelity:

- **M-a:** do C.2's head responses to in-silico single-neuron stimulation resemble the measured
  responses?
- **H0:** proposed by the investigation's independent check, and adopted by the maintainer. Is any
  resemblance the specific wiring, or only the degree statistics that have explained two earlier
  wiring effects here (Logbooks 034 and 103)?

## Method

**Measured responses.** Creamer, Leifer and Pillow's `measured_stams.pkl` (MIT repository,
commit bba43302):

- two 154 × 154 matrices of stimulus-triggered response magnitudes [responder, stimulated], from 80
  training and 30 test animals of the Randi et al. 2023 atlas;
- the test matrix is primary;
- the diagonal is masked.

**The metric.** Creamer's `nan_corr`, ported verbatim: one Pearson correlation over every entry
finite in both matrices.

**The networks.** C.2's encoded brain, built under the probe module's `PROCEDURE_SETTINGS`:

- at the box's centre, `[0.5] × 137`, which carries no fitting;
- at Logbook 131's answer, the `z` vector of `hinge4.json`.

All 154 measured neurons match by name. The network has no biases, so rest is zero.

**The stimulation.** One unit added to neuron s's starting state for the settling, the way C.2's
senses enter. The response of neuron j is its settled activity after the forward's four
iterations. A spike-local settling copy matched the package's forward at 100 random nonzero inputs
(worst output difference 4.8 × 10⁻⁷, against a tolerance of 10⁻⁵), and both input paths passed
their unit tests. All of this ran before any measured matrix was loaded.

**M-a's null.** 1,000 joint relabellings of the model's response matrix, rows and columns together
(`default_rng(9731)`).

- **at chance:** below the null's 97.5th percentile;
- **above chance, not above the floor:** above it, but not above the box centre;
- **above both:** above the null and the box centre.

**H0's null.** 19 brains built by the package's own path, with `wiring: rewired_degree_preserving`
and `rewire_seed` 9741 to 9759. Every neuron's chemical in- and out-degree is preserved, and the gap
junctions are rewired with their counts.

- **The minimum:** m0 = 0.25 × (r_true − 0.0216), where 0.0216 is the box centre's own relabelling
  null's 97.5th percentile.
- **PASS:** r_true is above all 19 draws, and above their median by at least m0. By chance the true
  wiring is highest of 20 with probability 0.05.
- **SMALL:** above all 19, but by less than m0.
- **FAIL:** not above all 19.
- **The gate:** the true wiring had to reproduce M-a's box-centre value to within 10⁻⁴ before any
  null was read.

## Results

### The gates

| Gate | Result |
|---|---|
| Parity of the settling copy (100 inputs, `default_rng(9732)`) | worst output difference 3.6 × 10⁻⁷ for C.2, 0 for the box centre |
| Both unit tests (starting-state and held input paths) | passed |
| H0: every null brain builds and differs from the true wiring | 19 of 19 |
| H0's positive control: the true wiring against M-a's value | 0.06527048652383569 both times; difference 0 |

### M-a, the registered reading: above chance, not above the floor

| Network | r against the test matrix (95% CI) | Share of the ceiling |
|---|---|---|
| Data ceiling, training against test | 0.173 (0.155 to 0.192) | 100% |
| Box centre, unfitted | 0.065 | 38% |
| C.2, Logbook 131's answer | 0.053 (0.035 to 0.072) | 31% |
| Relabelling null, 97.5th percentile | 0.022 | |

C.2 sits above all 1,000 relabellings, but below the box centre, and the two intervals overlap
widely. The ordering holds at inputs of 0.5 and 2, under a held input, and against the training
matrix. Sign agreement on the test entries whose measured size is above the median is 0.47 for C.2
and 0.48 for the box centre, which is chance.

### H0, the registered reading: PASS

| | r against the test matrix |
|---|---|
| True wiring, box centre | 0.0653, first of 20 |
| 19 degree-preserving rewirings | median 0.0050, range −0.0056 to 0.0219 |
| m0 | 0.0109 |
| True wiring minus the null median | 0.0603, about 5.5 × m0 |

Each draw moved 89.7% to 91.2% of the 3,709 chemical edges and 91.2% to 95.3% of the 1,095 gap
pairs. Reported beside the reading:

- **C.2's answer on each wiring:** 0.0534 on the true wiring, also first of 20; the nulls' median is
  0.0047.
- **The training matrix:** the same ordering, with the true wiring at 0.0898 against a null median
  of 0.0003.

## Corrections made in the open

M-a's registration had two errors about the code, both found by the computing agent before any
measured data were read. Amendment 1 (d9f928ac) corrected them:

- **The answer vector.** The registration said to convert `hinge4.json`'s `z` with
  `package_vector`, but `z` is already in the package's layout. Converting it would have moved 9 of
  its 137 entries and scored a brain that was never fitted.
- **The stimulation.** The registration defined stimulation as an input held at every settling
  iteration, but sensory drive enters the encoded forward only as the starting state. The primary
  definition became the starting state, and the held form became description. The held form gives
  the same ordering (C.2 0.044, box centre 0.069 to 0.071).

A third error, in the investigation's text rather than a registration: before the reading, the
spike plan named `best` as the answer vector. It is the list of per-generation scores, and the
answer is `z`.

## What this establishes, and what it does not

**Establishes.**

- The connectome's specific wiring predicts measured whole-brain responses to single-neuron
  stimulation with no fitting at all. It reaches about 38% of the data's own ceiling, against
  essentially nothing for 19 degree-preserving rewirings of it.
- Behaviour fitting, at Logbook 131's answer, neither adds to that nor detectably removes it.
- Unlike the two earlier wiring effects in this project, this one is not degree statistics.

**Leaves unresolved.**

- Whether C.2's registered fit, as opposed to a probe's answer, behaves the same.
- Whether a model fitted to neural data would raise the score toward Creamer's published 82% of
  the ceiling.
- Whether the result holds on a licensed, complete stimulation record. The measured arrays were
  built from the fuller, unlicensed record, while the CC-BY deposit on DANDI names only half its
  stimulation targets.

**Does not establish.**

- **Fidelity in signs or dynamics.** Sign agreement is at chance, and the networks settle within one
  step against responses integrated over 30 s.
- **That behaviour fitting harms neural fidelity:** the intervals overlap.
- **Anything about closed-loop behaviour.**

## Registered consequences

- **H0's map: PASS, so the slimmed hybrid.** Met: the investigation's
  [RECOMMENDATION.md](../../../investigation/RECOMMENDATION.md) was finalised under it. H1 is a
  standing neural readout floored by this degree-preserving null; H2 needs a causal bar; H3 is
  gated.
- **H1's adoption into Phase 10** (the readout beside C.2's registered answer, C.1, and W's arms):
  deferred-with-destination. It goes to the Phase 10 session, through
  [investigation/HANDOVER_PHASE10.md](../../../investigation/HANDOVER_PHASE10.md), for the tracker and
  the roadmap.
- **M-a's reading on C.2's registered answer:** deferred-with-destination, to C.2's registered fit
  (launch record 133), once it exists.

## Artefacts

Under [supporting/134-fit-first-neural-readings/](supporting/134-fit-first-neural-readings/):

- `launch.md`: the registrations and the commits that order them.
- `ma_c2_analysis.json`: every M-a figure above (`primary`, `readings`, `null`,
  `data_ceiling_train_vs_test`, `checks`).
- `h0_analysis.json`: every H0 figure above (`primary`, `description`, `check1`, `check2`).
- `h0_per_draw.csv`: one row per wiring (the true one and the 19 draws): the box centre's and C.2's
  r against the test matrix, and the fractions of edges moved.

Drivers: `investigation/spike/ma_c2/reading.py.txt` and `investigation/spike/h0/h0_reading.py.txt`
(probe scripts, kept as `.py.txt`). Inputs: Creamer et al.'s measured arrays, exported with
checksums in `investigation/spike/session0/exports/`, and `hinge4.json` from Logbook 131.
