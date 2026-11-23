# The serotonin-off probe: design and decision rule

*Fixed 2026-11-24, before any part of the probe ran. The maintainer approved this design and rule on 2026-11-24, with
the refit at 60 generations. It is committed before the probe launches. The script is `serotonin_off.py.txt`. The
record will be `serotonin_off.json`, as description beside the launch record.*

## The question

Can the wild type pass, at the registered pass level, while tph-1 is inert? If it can, close to the one answer known
to pass, then wild-type behaviour does not tie the tph-1 response down even at a pass. The registered fit's tph-1
reading would then rest on where 2a happens to leave serotonin.

The maintainer's rule of 2026-11-23 is: if the probe finds wild-type-passing answers with tph-1 inert, the fit does
not run. Otherwise one capped fit runs.

## What "serotonin off" means here

In this model, serotonin acts only through five receptor couples (the layer's rows 110–114). Two encoder entries hold
their strengths:

- `modulator.fixed.serotonin` holds mod-1, ser-1, ser-4 and ser-7 together. It is 0.0027 in the sixth pilot's
  answer.
- `modulator.fixed.serotonin.ser-5` is 2.98 there, at the top of its box.

**Serotonin off is defined as both entries at the bottom of their box, 0.001.** This removes what the model's tph-1
knockout removes, which holds the serotonin pool at zero, leaving at most 1/3000 of the answer's ser-5 strength.

The strengths were chosen over the producer for three reasons:

- The vector goes through the registered grade unchanged.
- pdfr-1 and tax-4 are graded on the same serotonin-off network without needing double knockouts.
- tph-1's inertness is measured, not assumed.

`wiring.unsigned.serotonin` is left alone. It scales chemical synapses, which the model's tph-1 does not touch.

## Why the probe refits, not just ablates

Switching serotonin off in the sixth pilot's answer, and nothing else, cannot find what the question asks for.

- That answer passes tph-1, so tph-1 differs from its wild type by about real tph-1's +0.27 in roaming.
- Switching serotonin off makes the wild type behave like that tph-1, so it moves the wild type by the same 0.27.
- The wild type's band at 200 animals is 0.169 ± 0.036, so the move is about 8 band scales.

**Disclosure:** this was already seen. The swap check of 2026-11-23 (small lawn only, in scratch) put ser-5 at 0.004
in this answer, and the wild type then roamed 0.47 instead of 0.20.

So the probe gives the rest of the network a chance to compensate.

## Design

| Step | What | Cost (16 workers) |
|---|---|---|
| P0 | The sixth pilot's answer (`pilot.json`) with serotonin off, graded on 5 discarded populations by the registered grade: all 24 statistics, both presences, and the 3 mutants. **Description only**, since its outcome is predicted below | in the grade step |
| P1 | A wild-type-only refit from P0's vector. The two serotonin entries stay frozen at 0.001, and the other 135 parameters are free. It uses 2a's objective (the four fitted families and the guards, against the bands' centres), a fresh sep-CMA at σ0 0.05 (2b's), population 48, 200 animals a candidate, for **60 generations** | ~81 min (27 s a candidate, measured in the recovery's 2a) |
| Readouts | The search's mean at generation 0 and every 10th generation to 60, on 200 fixed animals | ~13 min |
| Grade | P1's answer is the mean with the lowest readout D_fit among generations 10–60. It is graded with P0 on the same 5 populations | ~3 min |

In total about 1¾ hours. The review session's likelihood check has finished, so the probe uses all 16 workers.

**Seeds**, discarded, and disjoint from every registered block and from the swap check's:

- candidates: 1,995,000,000 + 200·(48·(g − 1) + m), up to 1,995,576,000;
- readouts: 1,996,000,000–1,996,000,199;
- grade population k: 1,996,100,000 + 200·k, for k = 0–4;
- CMA seed: 7311.

## Decision rule

- **Found**, so do not run the fit: P1's graded answer passes the wild type in **at least 3 of 5** populations under
  the registered rule (all 24 statistics at z\* 3.0781, with both presences), **and** tph-1 fails in at least 3 of
  5\. Under the maintainer's rule C.2 is banked, and #541 merges as "registered, piloted, recovery failed, fit not
  run".
- **Not found**, so the probe does not argue against the fit: anything else. Part C follows, with the cap the
  maintainer approved on 2026-11-24.
- **The rule is asymmetric.** "Not found" is weak evidence: a 60-generation search can miss an answer that exists. It
  means only that this probe gives no reason to stop.
- Reported beside the rule, not read by it:
  - every statistic's value and z, and pdfr-1 and tax-4;
  - P1's readout trend;
  - P0.

## Predictions, stated before the run

- **P0** fails the wild type in 5 of 5 populations, on small-lawn roaming at about z 8.
- **P1** is found with a chance of perhaps 15–25%. The original pass took 2b's 200 generations and 50 final
  generations to reach 3 of 5, and it passed at the threshold, failing on the onset rate. 2a's objective does not
  score the held-out families or the hinge, so the marginal statistics may drift.

## What the probe does not do

- It changes no registered rule, and it does not touch the recovery's gate or reading.
- It adds no code to `c2_fit.py`: the script calls the fit's own functions read-only.
