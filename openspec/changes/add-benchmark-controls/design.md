## Context

- **Held-out data** (`scheer2023_split.json`, `lopezcruz2019_split.json`):
  - the OD 2 wild type: 24 dates, 573 animals;
  - the uniform well: 1 date (20201014), 31 animals;
  - López-Cruz: 10 batches;
  - tph-1: 2 dates, both paired with N2;
  - pdfr-1: 6 held-out dates, 3 of them paired with PD1074;
  - tax-4: 7 held-out dates, 6 paired with PD1074.
- **The grader** (`benchmark_grade.py`) builds bands at any n, one generator per band. `grading.batch_search_presence` reads a real batch's presence from its bin means and SEMs.
- **The rule's own checks** (B.3): a 200-animal small-lawn population passes 0.925 of half-splits, and a López-Cruz batch 0.962.
- **The authors' HMM** (`data/roaming_dwelling/reference_hmm.json`): per 10-s bin, dwelling → roaming 0.021 and roaming → dwelling 0.102. Its stationary roaming fraction, 0.17, matches our classifier's wild type (0.169).
- **Batched controllers** (`evaluation/controllers.py`) implement `reset(n)` and `act(senses, noise) -> raw (n, 5)`. Speed and hazards map to the primitives' raw range, and curvature is raw[:, 1] ∈ [−1, 1] of ω_max = 50°/s.
- **Costs** (Logbook 104, 200 animals): hazard controllers take 1–3 s per assay, the MLP 2–4 s and the connectome 5–8 s.

## Goals / Non-Goals

**Goals**

- One registered reading of the held-out data, under rules fixed before it.
- Negatives that are hard for the right reasons: matched to the wild type's own marginal rates.

**Non-goals**

- Revising the bands. On VOID that is a new change.
- Fitted negatives (minimising D). The maintainer chose matched ones. A fitted reading may be run later as description.

## Decisions

1. **Control units and their failure rates.**
   - The 15 units are:
     - the small lawn, failing with 0.075 (the rule's check);
     - the uniform well, 0.05 (nominal: two band dates cannot be halved);
     - 10 López-Cruz batches, 0.038 each (the check);
     - 3 mutants, 0.05 each (nominal Bonferroni).
   - A López-Cruz batch fails when either statistic fails or search is absent. Every band batch was present, so absence adds little to the 0.038.
   - The failure count's Poisson-binomial distribution gives P(> 2) = 0.025 and P(> 1) = 0.137, so **the allowance is 2**.
   - Which units fail is reported in every case.
2. **The wild type's controls** are graded at their own n: 573 on the small lawn and 31 on the uniform well. López-Cruz batches are graded one at a time, as in the rule's check. All use the wild type's bar of 2.84.
3. **The mutants' controls.**
   - Each held-out paired date gives the mutant's and the wild type's tallies. Δ is pooled mutant minus pooled wild type over those dates.
   - It is graded on the mutant's bands at n = the held-out mutant count, at 2.24, with the bands' resolved directions: tph-1 roaming more, tax-4 roaming less.
4. **Matched parameters, from the OD 2 band cohorts only** (`benchmark_negatives_params.py`):
   - **The state** of each second is the authors' label on its 10-s bin (roaming or dwelling). Seconds are on the lawn by the geometric rule, outside every reversal and omega flag.
   - **Speed** is the mean 1-s displacement. **Curvature noise** is the SD of the 1-s heading change, clipped to the primitives' ±50°/s.
   - **Reversal and omega hazards** are bout onsets per state-second, by B.1's bout rule.
   - **The delta hazard** is set to the primitives' minimum: Scheer has no delta flag.
   - **The Poisson worm** uses the same quantities pooled over both states.
   - **Heading persistence,** for the correlated walk: the lag-1 autocorrelation of 1-s heading changes in run seconds, pooled.
   - Every value is written to the launch record before any negative runs.
5. **The negatives' controllers** (batched, each animal's randomness from its own seeded generator, `default_rng([seed, 2036])`):
   - `BatchedStateHazards`: a two-state chain stepped every 10 s by the given transitions, starting from the chain's stationary distribution. Each state has its own speed, curvature SD and hazards.
   - `BatchedCurvatureWalk`: a fixed speed and minimum hazards. Curvature is drawn uniformly over [−1, 1] each second (the random walk), or as an AR(1) with the wild type's persistence and SD (the correlated walk).
   - The Poisson worm is `BatchedCurvatureWalk` with the pooled hazards and AR(1) curvature. It differs from `BatchedConstantHazards`, which runs straight.
6. **The untrained policy.** Arm configs `mlpppo_..._internal_s1_primitives` and `connectomeppo_..._modulated_s1_primitives`, brains built by `body_kinematics_eval.build_brain(config, seed)` with seeds 5301–5308 (MLP) and 5309–5316 (connectome). Network j is evaluated on animal seeds 5101 + 25j to 5125 + 25j, and the 200 animals are graded pooled.
7. **W's minimum.**
   - Split the held-out OD 2 dates into halves by date, 1,000 times (seed 2037). For each split, take z = (s_A − s_B)/σ, σ the band's scale at a half's size, on the six small-lawn statistics.
   - Local search does the same over halves of the 10 held-out batches, on each half's mean of the per-batch statistics, σ the band's scale.
   - The uniform well has one held-out date and is left out. D uses the remaining families' weights, renormalised.
   - W's minimum is the 95th percentile.
8. **The verdict.** VALID when at most 2 control units fail and all six negatives fail the wild type. VOID otherwise, with every unit's and negative's reading recorded. Each negative's D and D_fit are reported beside its verdict.

## Risks / Trade-offs

- **[The allowance lets a main control fail]** → a failed small-lawn wild type, the benchmark's central control, would count as one of the two allowed failures. It is reported prominently. The verdict map does not single it out: the maintainer chose one allowance.
- **[Nominal rates for unchecked units]** → the uniform well and the mutants have no measured pass rate, so 0.05 is used. That is stated, and the allowance is not sensitive to it: at 0.10 each, P(more than 2) is still about 0.05.
- **[A matched negative may pass]** → then the benchmark is VOID by D34. That is the result B.4 exists to find, before any long fit.
- **[Held-out data read once]** → the scripts read them only after `launch.md`'s commit. A smoke on held-out data is not allowed; smokes use band data.
