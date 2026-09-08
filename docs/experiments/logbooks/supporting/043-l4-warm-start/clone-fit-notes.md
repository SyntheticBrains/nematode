# Clone-fit notes, written before any panel arm ran

Recorded 2026-09-08 after the teacher recording and the first nineteen clones, before the panel
launch, so the reading of the clone fits is on file ahead of the outcome.

## The teacher's policy is bang-bang

Over the 300 recorded episodes (103,827 steps) the teacher's action mean has speed = 1.000 on
every step (sd 0.000) and turn = −0.15 ± 0.95, with |turn| > 0.95 on 87% of steps. The policy is a
saturated switch: full speed always, hard left or hard right almost always. The constant-predictor
baseline (the per-dimension variance of the targets, averaged over the two dimensions) is
**0.454**; all of it sits in the turn dimension. The teacher's own sampled actions sit **0.311**
(mean squared) from its means, which is the noise the teacher itself acts with.

## Every clone plateaus near 0.30

| clone set | held-out loss, initial → final (seeds 1–8 or those done) | weight norm |
|---|---|---|
| plastic, wild-type | 0.52–0.72 → 0.297–0.329 | 17 → 91–137 |
| plastic, rewired | 0.52–0.68 → 0.293–0.328 | 17 → 107–133 |
| full, wild-type (seeds 1–3 at writing) | 0.55–0.65 → 0.264–0.271 | 18 → 74–83 |

Read against the baselines: a clone at 0.30 explains about a third of the targets' variance and
sits about as far from the teacher's mean as the teacher's own noisy actions do. The full set,
with gains and readout free, gains only a little over the chemical weights alone, and both
inflate the weight norm five- to seven-fold — the optimiser driving the tanh units toward
saturation to reproduce a switch. By the registered criterion (held-out loss not below half
its initial) 11 of the 16 plastic-set clones are flagged weak — 6 wild-type and 5 rewired — and
1 of the 16 full-set clones (counts corrected once all 32 had run; the reading was written at
nineteen).

## What this does and does not say

It says the connectome's forward, a fixed number of tanh hops from a nineteen-feature sensory
injection through a fixed readout, does not fit the champion's switch policy closely by
regression on the action mean, whichever parameters are free. It does not say whether a clone
at this loss is a competent policy: a switch matched in sign on most steps could forage and
evade while missing the mean by a lot in the action space. That is what the frozen clone arms
measure, and W1 is the registered gate. Nothing about the registration changes.
