# Clone-destruction diagnostic (probe, 2026-09-08)

Ratified with Chris after Logbook 043 as the item before 7a-ii: what the rule actually does to a
cloned competent policy, read in weight space from the panel's own auto-saved endpoints and from
a few extra runs. A probe, not a registered panel: every number here is descriptive.

## Inputs

- Start points: the plastic-set clones (`campaigns/l4-warm-start/clones/plastic_{wt,rn}_seed{1..8}.pt`).
- Endpoints: every panel run's auto-saved `final.pt` (the manifest's logs name the experiment;
  `exports/<id>/weights/final.pt`), 2000 episodes for the Hebbian and three-factor arms.
- Extra runs, seeds 3 (held under the three-factor rule: clone 40% → 42%) and 5 (destroyed:
  47% → 5%): the three-factor clone arm at 250, 500 and 1000 episodes; the wild-type Hebbian arm
  from random weights at 2000; and three one-key variants of the three-factor clone arm at 1000
  episodes — `plasticity_rate` 1e-4, 1e-5, and both normalisation switches off at 1e-3 (derived
  configs under `campaigns/l4-clone-diag-variants/configs/`).
- Comparisons on the wiring's own entries (`m_chem`): cosine similarity, mean |Δw| relative to
  mean |w|, the fraction of synapses whose sign flipped, per-unit incoming-norm ratio, and change
  by post-synaptic neuron class.

## 1. Both rules rewrite the clone, on every seed

| wiring | arm | cos(clone, endpoint) | mean |Δw| / mean |w| | sign flips | incoming-norm ratio |
|---|---|---|---|---|---|
| wild-type | three-factor | 0.29–0.45 (median 0.38) | 1.35–1.55 (median 1.43) | 0.33–0.43 (median 0.37) | 1.000 |
| wild-type | Hebbian | 0.22–0.36 (median 0.26) | 1.47–1.58 (median 1.52) | 0.37–0.43 (median 0.41) | 1.000 |
| rewired | three-factor | 0.27–0.41 (median 0.33) | 1.36–1.53 (median 1.45) | 0.38–0.43 (median 0.41) | 1.000 |
| rewired | Hebbian | 0.18–0.28 (median 0.24) | 1.49–1.57 (median 1.53) | 0.41–0.46 (median 0.43) | 1.000 |

Homeostasis holds every unit's incoming norm exactly. The Hebbian endpoint is *further* from the
clone than the three-factor endpoint on every seed, yet Hebbian behaviour stays competent on the
wild-type (39% mean) while the three-factor arm's collapses (13%). Interneuron inputs change
most, motor inputs least, under both rules. The three-factor and Hebbian endpoints at the same
seed share cosine 0.13–0.50 (median 0.31) on the wild-type. Seeds the
three-factor rule held (3, 4) show no signature in any of these statistics.

## 2. The drift is steady and the good policies are transient

Wild-type three-factor arm from the clone; cosine to the clone, cosine to the 2000-episode
endpoint, and that run's plateau tail:

| seed | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| 3 (held) | 0.73 / 0.70, **11%** | 0.67 / 0.80, **66%** | 0.60 / 0.89, **68%** | 0.40 / 1.00, 42% |
| 5 (destroyed) | 0.71 / 0.65, **76%** | 0.66 / 0.73, **30%** | 0.56 / 0.88, **6%** | 0.41 / 1.00, 5% |

The weights decorrelate from the clone at a near-constant rate. Behaviour is not monotone: the
rule found a 76% policy from a 47% clone within 250 episodes on seed 5 and walked away from it;
on seed 3 it dipped, rose to 66–68% for a thousand episodes, and drifted back to 42%.

## 3. There is no start-independent Hebbian fixed point

Wild-type Hebbian arm, 2000 episodes, from the clone (the panel run) and from random weights (a
fresh run), same seed:

| seed | cos(random init, clone) | cos(Hebbian from random, Hebbian from clone) | tail from random | tail from clone |
|---|---|---|---|---|
| 3 | 0.22 | **0.04** | 4% | 58% |
| 5 | 0.20 | **0.06** | 0% | 50% |

The two Hebbian endpoints are orthogonal. Both runs drift far from their starts (cosine 0.13–0.20
to the random init, 0.26–0.30 to the clone). What the Hebbian rule preserves on the wild-type is
not the clone's weights but its *competence*: from a competent start it drifts within a competent
region; from a random start it wanders in a dead one. On the rewired null it does neither (18%
from the clone in the panel). Logbook 043's reading that the Hebbian rule "holds the clone" is
right behaviourally and wrong in weight space, and an earlier inference during this probe — that
it replaces the clone with a start-independent fixed point of the wiring — is wrong on both
counts.

## 4. No rate and no modulator form anchors the policy

Wild-type three-factor clone arm, 1000 episodes, cosine to the clone and plateau tail (frozen
clone: seed 3 40%, seed 5 47%):

| seed | rate 1e-3 (normalised) | rate 1e-4 | rate 1e-5 | rate 1e-3, raw modulator |
|---|---|---|---|---|
| 3 | 0.60, 68% | 0.77, 21% | 0.80, 10% | 0.37, 53% |
| 5 | 0.56, 6% | 0.73, 19% | 0.75, 32% | 0.30, 29% |

A hundredfold lower rate leaves the weights at cosine 0.75–0.80 after roughly 250,000 updates —
far more drift than a diffusion at that step size could produce — and the policy is already below
the frozen clone. The update direction is consistent: the rule is a biased drift toward the
correlation structure the trace encodes, and the bias accumulates in proportion to rate × time
whatever the rate. At 1e-3 the reward signal steers the drift through good policies (68% on
seed 3); at lower rates it only degrades the clone more slowly; the raw modulator drifts furthest.

## Reading

- The rule does not converge. It is a constant-speed drift on the norm sphere: the centred,
  RMS-normalised modulator and the RMS-normalised trace give a fixed step whether the policy is
  good or bad, homeostasis removes the decay term as a brake, and nothing shrinks the update when
  reward is high. Good policies are places the drift passes through.
- "Destruction" is therefore not a credit-assignment sign error to be routed away; it is the
  absence of consolidation. Structured instruction (7a-ii's B.4b) addresses which synapses get
  credit; without a mechanism that stops or slows updating once a policy is good, even perfect
  credit assignment drifts away from what it found.
- The wild-type wiring under reward-free Hebbian drift preserves competence from a competent
  start and the rewired null does not — the same wiring-specific signal panels 1–3 saw, now with
  a mechanism-level reading: the wild-type's correlation structure keeps Hebbian drift inside a
  competent region. It remains descriptive.
- The one-hour clone assay (load a competent policy, run a rule, does it hold or improve) is the
  test bed any new rule mechanism should clear before a registered panel.

## Implications carried into 7a-ii's design

1. A consolidation mechanism is the first candidate, ahead of structured routing: an update
   magnitude that falls with the prediction error's magnitude (un-normalised, or normalised
   with an annealed scale), or a slow protective variable — biologically, dopamine-gated
   consolidation rather than dopamine-gated change.
2. The clone assay gates 7a-ii's panel (B.5): a rule variant must hold or improve the clone on
   the wild-type before the expensive 2×2 is re-run.
3. The atlas (B.1) is a substrate deliverable in its own right and proceeds regardless.
