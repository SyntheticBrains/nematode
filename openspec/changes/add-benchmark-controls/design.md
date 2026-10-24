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
3. **The mutants' controls.** Only held-out dates on which the mutant and its control were recorded together are used. pdfr-1's three unpaired held-out dates (20180202, 20180212, 20180601: two recorded beside N2, one beside neither) are left out, and the record says so.
   - Each held-out paired date gives the mutant's and the wild type's tallies. Δ is pooled mutant minus pooled wild type over those dates.
   - It is graded on the mutant's bands at n = the held-out mutant count, at 2.24, with the bands' resolved directions: tph-1 roaming more, tax-4 roaming less.
4. **Matched parameters, from the OD 2 band cohorts only** (`benchmark_negatives_params.py`):
   - **The state** of each second is the authors' label on its 10-s bin (roaming or dwelling). Seconds are on the lawn by the geometric rule, outside every reversal and omega flag.
   - **Speed** is the mean 1-s displacement.
   - **Curvature noise** is the SD of the 1-s heading change, clipped to the primitives' ±50°/s. It is read only on seconds whose displacement, and the previous second's, is at least `FLOOR_MM` (0.01 mm, the readouts' heading floor). On a lawn a dwelling worm moves a few tens of µm a second against a pixel of about 8.6 µm, so slower seconds' headings are tracking noise (Logbook 105). Dwelling's slowest seconds are therefore left out of its curvature, and the record says how many.
   - **Reversal and omega hazards** are bout onsets, by B.1's bout rule, divided by the state's seconds outside every bout. The primitives draw a start only while no bout runs, so dividing by all of a state's seconds would bias the rates low by the time spent in bouts.
   - **The delta hazard** is set to the primitives' minimum: Scheer has no delta flag.
   - **The Poisson worm** uses the same quantities pooled over both states.
   - **Heading persistence,** for the correlated walk: the lag-1 autocorrelation of 1-s heading changes over consecutive run seconds above the floor, pooled.
   - Every value is written to the launch record before any negative runs.
5. **The negatives' controllers** (batched). Each animal's randomness comes from the per-animal generator the evaluator already passes to `act(senses, noise)`. That generator is seeded by the animal's seed, as the MLP and connectome use it, so an animal behaves the same alone or in any batch by the evaluator's existing guarantee. `reset(n)` receives only the count, so a controller cannot seed its own streams. The new controllers refuse `noise=None`, a mean-action run.
   - `BatchedStateHazards`: a two-state chain stepped every 10 s by the given transitions, starting from the chain's stationary distribution. Each state has its own speed, curvature SD and hazards.
   - `BatchedCurvatureWalk`: a fixed speed and minimum hazards. Curvature is drawn uniformly over [−1, 1] each second (the random walk), or as an AR(1) with the wild type's persistence and SD (the correlated walk).
   - The Poisson worm is `BatchedCurvatureWalk` with the pooled hazards and AR(1) curvature. It differs from `BatchedConstantHazards`, which runs straight.
6. **The untrained policy.** Each network is built by `body_kinematics_eval.build_brain(assay_config, seed)` from each assay's own config for the arm (`mlpppo_..._internal_s1_primitives_assay_*` and `connectomeppo_..._modulated_s1_primitives_assay_*`), as the parity suite builds them, with seeds 5301–5308 (MLP) and 5309–5316 (connectome). The arm's brain config is the same across its assays, so one seed gives one network on all three. A test checks it. Network j is evaluated on animal seeds 5101 + 25j to 5125 + 25j, and the 200 animals are graded pooled.
7. **W's minimum.**
   - Split the held-out OD 2 dates into halves by date, 1,000 times (seed 2037). For each split, take z = (s_A − s_B)/σ, σ the band's scale at a half's size, on the six small-lawn statistics.
   - Local search does the same over halves of the 10 held-out batches, on each half's mean of the per-batch statistics. σ is the band batches' SD ÷ √(half size): the band's scale for a mean of that many batches, which is "the band's scale at a half's size". The single-batch scale would be too large by about √5. That would shrink the z's and D, and lower W's minimum, so a wiring effect would be easier to claim.
   - The uniform well has one held-out date and is left out. D uses the remaining families' weights, renormalised.
   - W's minimum is the 95th percentile.
8. **The verdict.** VALID when at most 2 control units fail and all six negatives fail the wild type. VOID otherwise, with every unit's and negative's reading recorded. Each negative's D and D_fit are reported beside its verdict.

## Risks / Trade-offs

- **[The allowance lets a main control fail]** → a failed small-lawn wild type, the benchmark's central control, would count as one of the two allowed failures. It is reported prominently. The verdict map does not single it out: the maintainer chose one allowance.
- **[The units are not independent]** → the mutant controls reuse held-out wild-type animals that also form the small-lawn control. The Poisson-binomial assumes independence, so real failures may cluster slightly more than it allows. Stated.
- **[Nominal rates for unchecked units]** → the uniform well and the mutants have no measured pass rate, so 0.05 is used. That is stated, and the allowance is not sensitive to it: at 0.10 each, P(more than 2) is still about 0.05.
- **[A matched negative may pass]** → then the benchmark is VOID by D34. That is the result B.4 exists to find, before any long fit.
- **[Held-out data read once]** → the scripts read them only after `launch.md`'s commit. A smoke on held-out data is not allowed; smokes use band data.
