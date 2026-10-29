## Context

Attempt 1 (launch record 116) left four registered truths with committed observations, floors and priors, a pilot cap of 272, and two diagnoses (`diagnosis.json`):

- **the capped objective's flatness:** the pilot's roaming speed sat 30 scales off and γ was undefined, both at the cap;
- **the uncapped objective's escape:**
  - truths 1164 and 6985 are noisy on the wild type's n = 200 scales, with their own repeats differing by |z| 13–15 on exits;
  - their fits drifted to leave statistics undefined (8–10 of 10 re-simulations), since that cost a flat 10 while staying defined cost much more.

The re-investigation (description, on stored simulations):

- **The prior's references** (201 simulations across the box): 80% have at least one undefined statistic. γ is undefined in 64%, the small lawn's roaming fraction and exits in 19%, and kinematics in about 20%.
- **The box centre:** all 19 statistics are defined in all 10 re-simulations (it leaves the lawn about 59 times an hour).
- **Floor against exit rate:** Spearman 0.65 over the 16 candidates with repeats. Candidates leaving 15–30 times an hour have floors of 1.6–3.0; those leaving 0.4–10 times an hour have floors of 0.3–1.0.

## Goals / Non-Goals

**Goals:**

- an objective with no flat region beyond any cap, and no reward for leaving a statistic undefined;
- recovery evidence for it on the same truths that failed, and on the ones that passed;
- the same objective for C.1b.

**Non-Goals:**

- re-selecting truths, or restricting them to low floors. That is held as the fallback if this objective fails the noisy truths;
- noise-aware (heteroscedastic) scales, which would change what recovery means: C.1b's target is graded on the band scales;
- presence in the objective. Recovery's gain is on D, and C.1b's plan must handle presence, since its wild-type pass needs it.

## Decisions

1. **The ranking objective,** `predictive_distance_v2(..., compressed=True)`:

   - **A defined statistic:** with z = Δ / scale uncapped, the term is 10·asinh(z/10). It is about z near the truth (8.8 at z = 10) and grows slowly far from it (18.2 at 30, 30.0 at 100, 58.3 at 1,700).
   - **Undefined in only one record:** 100, more than any defined term below about 110,000 scales.
   - **Undefined in both:** 0.
   - **Weights:** B v2's five families equal, split equally within each.

   This replaces amendment 1's uncapped objective for every attempt-2 fit. The registered D, the floor, the oracle, the prior, every gain and the gate stay on the capped distance.

2. **The truths are reused:** attempt 1's registered four (1164, 1354, 6985, 9105), with their committed v2 observations (base 210 M), floors and priors (`preflight.json`, bound). The only change is the objective, so the result isolates it.

3. **The cap is 272,** amendment 1's pilot cap. Every attempt-1 fit plateaued well before it (generations 71–179), and the guarded plateau rule stops fits sooner. There is no re-pilot.

4. **The probe** (description; discarded seeds), registered as the gates' oracle before Part A:

   - truths 1164 and 6985, each one fit to the guarded plateau with a cap of 200;
   - truth 9105, one fit for 80 generations;
   - each answer re-simulated 10 times, with its gain read on the capped D;
   - **if any of the three gains is below 0.75, the change returns to the maintainer before registering**;
   - seeds: base 300,000,000 with attempt 1's fit scheme (k = 10,000·f + 16·(n − 1) + p, answers at 10,000·f + 5,000 + q), for f = 0–2, on CMA seeds 5995–5997.

5. **The registered fits:**

   - 4 truths × 2 restarts, with fit f = 2j + r;
   - CMA seeds 5430 + f, inside F.0's band;
   - animals at base 230,000,000 on attempt 1's fit scheme (spanning 230.0–245.0 M);
   - the guarded plateau, the cap of 272, and each answer re-simulated 10 times.

6. **The gate and the verdict map are attempt 1's.** The median best-restart gain must be at least 0.75. On a pass, F.0 is cleared for C.1 on v2 and C.1b proceeds, fitting with the same objective. On a fail, C.1b does not run, and the fallback (truths at floor ≤ 1) goes to the maintainer.

7. **The registration:** launch record 117, Part A committed after the probe and before any registered fit. It binds attempt 1's `preflight.json` and `pilot_amendment1.json`. There is no Part B, because nothing between the probe and the fits changes a rule.

## Risks / Trade-offs

- **[The noisy truths stay hard]** → their observations carry noise of 13–15 scales on exits that no fit can remove, though the floor normalises the gain for it. The probe shows whether the fix suffices, and if not, the fallback is registered as the next step.
- **[A large penalty dominates early]** → undefined γ (weight 0.1) at 100 swamps the rest until γ is defined. That is intended, since recovery needs it, and the box centre starts with γ defined.
- **[Reusing truths after seeing them fail]** → this tests the objective on exactly the cases that broke it. The truths were fixed by attempt 1's registered rule, not chosen after the fact.
- **[Cost]** → about 4–6 h for the probe and about 15 h for the fits.
