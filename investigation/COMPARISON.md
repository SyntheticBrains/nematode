# Comparison: stay, fit-first, or a hybrid

Written 2026-11-14. Three routes are compared against the brief's five milestones.

## The routes

- **Stay:** the current roadmap. Phase 10 fits controllers to behaviour statistics and grades them
  on benchmark v4. Phase 11 adds a body. Nothing on it grades neural activity.
- **Fit-first:** fit a connectome-constrained model to neural recordings first, and use behaviour
  as a held-out test.
- **Hybrid:** Phase 10 continues, and a small neural track is added beside it. Measured neural
  responses become a standing held-out readout for every connectome model. A connectome-constrained
  neural fit, on licensed data, supplies what behaviour cannot constrain.

## The basis for the estimates

- **The pacing history** (ORIENTATION.md § 5):
  - a phase-sized step takes about three to four weeks at the maintainer's pace;
  - registered steps are amended about twice before they run;
  - Phases 6a to 9 each closed without a positive on their primary question that held;
  - Phase 10 has 12 to 27 weeks left if W runs at its registered size, or 7 to 14 weeks with W cut
    to a quarter.
- **The spike** (SPIKE_REPORT.md):
  - checking one published target takes about a day of agent work;
  - specifying a fair test took three review rounds;
  - every target hid a problem found only by reading its code or data;
  - a controlled panel at the Randi scale costs about 1,000 to 2,500 machine-hours on this machine
    (a float64 GPU would cost an unmeasured $100 to $1,000);
  - a panel at Mi's scale costs about 9 to 20 machine-hours.
- **Times are at part-time pace and include amendment rounds.** They are judgement on that basis,
  not measurement, and the negative-rate prior applies to every route equally. The hybrid's
  estimates are the neural track's own, on top of whatever Phase 10 takes.

## M-a: predicts responses to held-out perturbations

| | Stay | Fit-first | Hybrid |
|---|---|---|---|
| Time | Not on the roadmap, so never reached by it. Adding the readout costs about one session, since it now exists. | 6 to 12 weeks if the atlas authors supply complete licensed metadata. Without it, blocked: only half the stimulations name their target. | The same as fit-first for the fit, plus about one session for the standing readout, which can come first. |
| Fidelity | Measured: C.2 at 31% of the data ceiling, not above its unfitted wiring (38%). | Up to Creamer's published 82% of the ceiling (absolute r 0.14 against 0.17). Whether that survives a degree-preserving null is untested. | As fit-first for the head; the closed-loop model's readout tracks how much of it carries over. |
| Main risks | Neural fidelity is never measured as the model changes. | Licence and metadata; a reproduction that misses; a connectome effect that turns out to be degree statistics. | As fit-first, but nothing else waits on it. |
| Compute | None. | About 1,000 to 2,500 machine-hours for a controlled panel here, or a rented float64 GPU. | As fit-first, run on rented capacity so that Phase 10 keeps the machine. |
| Reused | The batched evaluator, C.2's encoder, the session's settling copy. | The connectome loaders, neuron classes, NWB pipeline and scorer port from session 0. | Both. |
| Credibility | A single in-sample answer, scored against data it never saw: a fair but weak test. | Direct, if reproduced with a proper null. Today neither published fit has one. | Two independent graders, behaviour and neural, with neither fitted to the other. |

## M-b: predicts freely moving whole-brain activity on held-out worms

| | Stay | Fit-first | Hybrid |
|---|---|---|---|
| Time | Not on the roadmap. | 2 to 4 months after M-a: Atanas 2023 is about 1 TB on DANDI, 40 of 68 recordings carry NeuroPAL identities, and it uses a different indicator. | Deferred until M-a and the import show value. |
| Fidelity | None. | Unknown for this project. Mi's held-out-worm scores (0.44 to 0.47) did not separate connectome from dense models, and were obtained on mis-parsed wiring. | As fit-first, later. |
| Main risks | | Immobilised-animal fits may not transfer; head-only coverage; the connectome may add nothing on whole animals. | |
| Compute | | Ingestion dominates: byte-range reads of 26 GB files; fits at least at the Randi scale. | |
| Reused | | M-a's fitting code. | |
| Credibility | | High if reached, since few models attempt it. | |

## M-c: passes the existing behaviour benchmarks in closed loop, with no fitting to them

| | Stay | Fit-first | Hybrid |
|---|---|---|---|
| Time | Unreachable by construction, because C.2 is fitted to the benchmark. Its counterpart, held-out recording dates and held-out families, is inside Phase 10 (C.2's registered fit reads held-out dates). | Many months. Neither published model can drive the closed loop unchanged: Creamer's has no AWC and impulse inputs, and Mi's has three stimulus channels and never runs free. | The behaviour-fitted closed loop keeps the fitted families. M-c is read on the families and dates held out from the fit, as in Stay. |
| Fidelity | In-sample passes at probe level (Logbook 131: every statistic and mutant in 15 of 15 populations); the registered fit is pending. Held-out families failed in Logbook 126's fit. | A pass would be the strongest result available. The spike's interface analysis says it is unlikely soon. | As Stay. |
| Main risks | The benchmark and the controllers evolved together (v1 to v4), so in-sample passes overstate fidelity. | The interface problem may be larger than the fit. | As Stay. |
| Compute | C.2's sequence is about 117 machine-hours plus amendments; W is 1,200 to 2,900 at its registered size. | Unknown. | As Stay, with W cut. |
| Reused | Everything. | The benchmark, the evaluator and the primitives body. | Everything. |
| Credibility | Protocol strong, claim limited to in-sample behaviour statistics. | Highest, if reached. | As Stay, plus the neural readout beside it. |

## M-d: reproduces a neuromodulator-driven state switch (roaming and dwelling)

| | Stay | Fit-first | Hybrid |
|---|---|---|---|
| Time | Within Phase 10. C.2 roams and dwells in bouts in-sample, through its data-derived modulator layer. | Long. Neural data do not constrain modulator release (IDENTIFIABILITY.md), and a linear model cannot switch states. | As Stay. |
| Fidelity | Probe-level in-sample, with the serotonin and PDF mutants passing (Logbooks 129 to 131); registered fit pending. | None yet. | As Stay. |
| Main risks | In-sample only; the fit's perturbation responses were not pinned by wild-type data (Logbook 128). | Needs nonlinear models with slow states fitted to freely moving data: research frontier. | As Stay. |
| Compute | Part of Phase 10's. | Large. | Part of Phase 10's. |
| Reused | The modulator layer and the procedure. | The modulator layer's placement data. | Everything. |
| Credibility | Moderate: the mutants are causal tests, in-sample. | | As Stay. |

## M-e: shows an experience-dependent change in behaviour (Wormturn)

| | Stay | Fit-first | Hybrid |
|---|---|---|---|
| Time | Not planned. Phase 7's plasticity work closed SPLIT, and the rule had no positive control. | Long-term learning is not constrained by any recordings available. Short-term adaptation is in the recordings (AFD adapts over about 45 s) and could be fitted with the dynamics. | As Stay for learning; short-term adaptation could ride on the neural track. |
| Fidelity | None. | None yet. | None yet. |
| Main risks | No positive control for plasticity rules. | The thinnest data of any milestone. | As both. |
| Compute | | | |
| Reused | Phase 7's plasticity machinery. | | |
| Credibility | | | |

## Summary

- **Stay** reaches M-d, and M-c's held-out counterpart, inside Phase 10, in 7 to 27 weeks depending
  on W. It never measures neural fidelity. The spike's one reading shows behaviour fitting does not
  supply it.
- **Fit-first** has a direct path only to M-a and M-b. M-a is blocked on licensed data today, and a
  fair control panel costs about as much as W. Both published fits fail a reproduction check on
  licensed materials. M-c and M-d are far, and switching would set aside Phase 10's working machinery
  for milestones that machinery already approaches.
- **Hybrid** keeps Stay's path to M-c and M-d and adds a measured neural readout at once. It adds a
  neural fit only where neural data constrain anything (the head's effective coupling, held-out
  neurons). The neural fit runs on rented capacity, so Phase 10 keeps the machine. Its risk is
  divided attention at part-time pace.
