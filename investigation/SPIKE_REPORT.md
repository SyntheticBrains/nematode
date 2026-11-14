# Spike report

Written 2026-11-14. **The spike stopped at fact-finding, under the brief's stop rule.** Neither
published connectome-constrained fit to worm neural data can be reproduced as specified on licensed
materials within a spike's budget. No fit was run, so no bar was read. One registered reading was
made: the project's current model, C.2, scored against measured responses to single-neuron
stimulation. The evidence is in `spike/SESSION_0.md`, `spike/SESSION_0B.md` and
`spike/MA_C2_RESULT.md`. The decisions are A24 to A31 in DECISIONS.md.

## 1. What was attempted

The brief asked for a reproduction of one published connectome-constrained fit, in a differentiable
framework, scored on held-out perturbations or held-out neurons against a degree-preserving null,
with a closed-loop bridge. Two targets were examined in turn:

1. **Creamer, Leifer and Pillow** (bioRxiv 2024.09.22.614271): a connectome-masked latent LDS fitted
   to the Randi et al. 2023 stimulate-and-record atlas.
2. **Mi et al. 2022** (ICLR): a connectome-constrained latent-voltage model fitted to Yemini et al.
   2021's whole-brain recordings. This was the maintainer's pre-approved fallback (A20, A27).

The plan for the first target went through three independent reviews before any data were opened
(REVIEW_PART_A.md § 3, § 6 and § 7). The third found three blockers and recommended fact-finding
first. Two fact-finding sessions followed, one for each target.

## 2. Creamer et al.: blocked on licensed data, and costly

| Fact | Evidence | Consequence |
|---|---|---|
| The CC-BY DANDI deposit (001075) names the stimulated neuron for 2,815 of 5,614 stimulations. In 52 of 110 files fewer than half are named, and in 2 none are. Matching the stored coordinates to neurons recovers known identities only 56 to 65% of the time. | session 0, C.2; re-counted independently over all 110 files | Half the perturbation signal is missing from the licensed data. |
| The NWB labels do not reproduce Creamer's 154 neurons under his own selection rule: 35 to 42 are missed. | session 0, C.4 | His neuron set must be imposed by name, and some neurons are thinly covered. |
| His measured response arrays hold 2,373 stimulated pairs that the licensed data cannot produce. | session 0, item 1 | He worked from a fuller record, probably the OSF exports, which carry no licence. |
| The traces are exactly his processing stage: his preparation, re-run on the NWB traces, reproduces his example recording at a median per-neuron r of 1.0000000. | session 0, C.3 | The licensed traces are sound. The gap is the stimulation and label metadata. |
| Missing values run through every animal: the median neuron-recording is 31% missing, and every animal has a gap inside a recording. | session 0, E | The exact Kalman filter is needed, and the fast steady-state filter never applies. |
| One gradient fit takes 44 to 154 h on this machine. The planned controlled panel is about 1,000 to 2,500 machine-hours, roughly W's size. | session 0, F; measured, then extrapolated | A proper test with nulls is not a spike-sized job here. A float64 GPU might bring it to $100 to $1,000, but that figure is unmeasured. |
| The released models use no ridge and no pruning. His EM is not converged (relative change 6 to 9 × 10⁻⁴ over its last 100 of 200 steps). His likelihood counts each missing entry as an observation with variance 10⁸. | session 0, A and B | "Reproducing EM's optimum" is not a well-defined target. |

## 3. Mi et al.: unlicensed code with a wiring bug and no recorded protocol

| Fact | Evidence | Consequence |
|---|---|---|
| TuragaLab/wormvae has no licence; GitHub reports none. | session 0b, A; re-checked | The model must be rewritten from the paper. |
| Its parser stores polyadic sends (`Sp`, 1,625 rows) backwards. The line tests `'SP'` (`model/connectome_preprocess.py` line 100), so 1,270 of the 3,464 chemical entries are reversed copies, and true reciprocal pairs rise from about 233 to 1,503. The paper's "3464 chemical synapses" is this count. | session 0b, A; I read the line and the file's type counts myself | Mi's headline (the connectome predicts withheld neurons at r 0.32, against 0.08 without it) was obtained on partly wrong wiring. |
| No scoring code exists, in the current code or its history. The held-out worms, the scored frames and the seeds are not recorded. Only 5 of the 9 variants exist in the code, and it departs from the paper in its nonlinearity, its synapse formula and its handling of unrecorded neurons. | session 0b, A and C | A reproduction cannot be checked against the published numbers. |
| Mi's training traces are bitwise the MIT-licensed Zenodo 3906530 head-activity file. DANDI 000541's CC-BY traces are a different extraction, matching it at a median r of only 0.69. | session 0b, B | Licensed data exist for a re-implementation. |
| One held-out-neuron run takes about 1.5 h on CPU, and a 20-run panel about 9 machine-hours. | session 0b, E; measured | Cheap enough to run, if the model is rewritten. |

## 4. The one reading made: C.2 against measured neural responses

Registered before computing (`spike/MA_C2_READING.md`). Amendment 1 corrected two rules before any
measured data were read. The reading was made once, and I recomputed its headline numbers
independently.

| | r against the measured test responses | Share of the data ceiling |
|---|---|---|
| Data ceiling (training animals against test animals) | 0.173 | 100% |
| C.2, behaviour-fitted (Logbook 131's answer) | 0.053 (95% CI 0.035 to 0.072) | 31% |
| The same network unfitted, at the box's centre | 0.065 (0.047 to 0.084) | 38% |
| Chance, the 97.5th percentile of 1,000 relabellings | 0.022 | |
| Creamer's LDS fitted to these data, as published | 0.14 | 82% |

**The registered reading:** above chance, not above the floor.

- The connectome's structure alone carries some signal about who responds to whom.
- Fitting to behaviour did not move C.2's head responses toward the measured ones.
- On the strongest measured responses, the sign agrees only about half the time, which is chance.

The limits stated before reading still apply: this is a probe's answer, the network has no slow
dynamics, and the measured arrays come from the fuller, unlicensed record.

## 5. The bridge

The bridge was not run, under the stop rule. Fact-finding still showed what it would need:

- **Creamer's model:**
  - his loader blacklists AWC by default, so the odour's fall has no input neuron;
  - his model's inputs are single-frame impulses, while food contact is sustained.
- **Mi's model:**
  - it has three binary stimulus channels (butanone, pentanedione, NaCl) onto 79 "sensory" neurons;
  - CEP and ADE get no input;
  - the code never runs the model free of its inference network;
  - one 1-s environment step would be 160 simulation steps.

Neither model is close to driving the closed loop unchanged.

## 6. The effort ledger

From `spike/LEDGER.md`:

| Item | Effort |
|---|---|
| Plan drafts | 3 |
| Independent reviews of the plan | 3, about 40 minutes of reviewer time. Each found problems the last missed; the third found three blockers in a plan the second had called nearly ready. |
| Fact-finding | 2 sessions, about 1 h of agent wall-clock, including about 0.75 h of data ingestion |
| The C.2 reading | about 11 minutes of agent time, plus one amendment of my registration's errors before any data were read |
| Maintainer decision rounds | 6 (A15 to A21, A24 to A26, A27 to A29, A30 and A31, the C.2 amendment, the stop) |

**What the ledger says.** The engineering per step was fast. What took the effort was checking
published work and specifying a fair test. Every target and every plan draft hid a problem that only
a close reading of code or data revealed:

- the incomplete metadata on DANDI;
- the unconverged EM;
- a wiring bug in the field's second published fit;
- two errors in my own registration.

That is the real cost profile of a fit-first route here: verification dominates, and compute
dominates once a fair control panel is attempted.

## 7. What this means for the brief's question

The brief asks whether fitting a connectome-constrained model to neural recordings would reach the
long-run goal *faster and more reliably* than the current approach.

- **Reliability: not now.** Both published fits fail a reproduction check on licensed materials, for
  different reasons. Neither has been tested against a degree-preserving null. The project has twice
  seen an apparent wiring effect turn out to be degree statistics (Logbooks 034 and 103).
- **Speed: not as a switch.** A fair control panel on the perturbation data costs about as much
  machine time as W, before any closed-loop work. The closed loop needs an interface that neither
  model provides.
- **But behaviour fitting does not buy neural fidelity either.** C.2's fitted head is no closer to
  the measured responses than its unfitted wiring. If neural fidelity is part of the long-run goal,
  neural data have to supply it.

## 8. What stays unknown

- Whether a gradient fitter on licensed data reproduces Creamer's held-out score.
- Whether either target's connectome advantage survives a degree-preserving null, and, for Mi's, a
  correction of the wiring bug.
- What a float64 GPU fit costs; the $100 to $1,000 range is unmeasured.
- Whether the atlas authors will supply the missing target identities, or a licence for the OSF
  exports (the draft note is `spike/NOTE_TO_AUTHORS_DRAFT.md`).
