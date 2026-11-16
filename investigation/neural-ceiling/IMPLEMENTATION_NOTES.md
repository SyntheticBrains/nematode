# Implementation notes for launch record 135

Written 2026-11-15, with amendment 1, before any fit to real data and before any read of S. The
code (`ceiling/`, committed at e771846a) was built from the registration and tested on synthetic
data. Where the registration was ambiguous, the build took the most literal reading and named it as
a constant in `ceiling/config.py`. Amendment 1 registers these readings as the rules. The numbering
follows the build report.

| # | Rule | The reading registered |
|---|---|---|
| 1 | Preflight and S | Preflight makes one guarded, logged, pattern-only read of S to build the entry sets, cached in `entry_sets.npz` for the oracles. |
| 2 | Classes before the 9771 permutation | Sorted by name; `permutation(74)` of their indices; fold = position mod 5. |
| 3 | Inner class folds | The visible classes, in the outer permutation's order, dealt round-robin into 3. Validation excludes pairs within an inner fold, and the responders in the outer fold stay. |
| 4 | Held-out-worm inner folds | `default_rng(9770).permutation` of T's finite off-diagonal entries in row-major order; fold = position mod 3. |
| 5 | floor-T on held-out worms | Absent, since the registration gives it "in the held-out splits". There, floor-C is a + b·floor-ME, and H0-cal and C.2-cal drop their floor-T term. |
| 6 | The held-out-neuron entry set | Superseded by amendment 1, item 5. |
| 7 | The ridge | The mean of the squared parameters, taken jointly over an arm's penalised tensors (U and R together); for **dense**, over off-diagonal entries only. |
| 8 | Refits and inner-CV scoring | The refit rule also applies to inner-CV fits. An unreadable inner fit scores +∞. The summed inner loss is the sum of the per-fit validation mean squared errors. |
| 9 | Initialisation | **dense** starts at zero. **untied** draws U then V from N(0, 0.01²) with `default_rng(9774)`. **tied** draws U then R. |
| 10 | L-BFGS stops | Torch's gradient tolerance is set to 0, because otherwise **tied**'s small starting gradient stops it at iteration 0. Its parameter-change test runs at 10⁻¹⁰. Function evaluations are capped at 25 × the iterations. |
| 11 | Per-hop shares | Cov(Wᵏ, M) / Var(M), which sum to 1; Var(Wᵏ) / Var(M) is reported beside them. |
| 12 | null-R's swaps | Undirected swaps first, then directed. "10 swaps per entry" counts each part's directed entries. The extension runs in rounds of one attempt per entry, up to 30. Counts travel with their entries. |
| 13 | sparse-ER | `rng.choice` over the 302 × 301 off-diagonal positions, without replacement. |
| 14 | Oracle K | A fixed λ of 10⁻⁴ (the budget's "masked only" rules out selection): 10 fits. "Near-zero loss" means at most 10⁻³ in standardised units. K's matrices sit on T's and S's finite patterns. |
| 15 | Residual blocks | Each target column draws one source column. Entries are copied by responder, and gaps are filled from that column's finite residuals. One generator draws T's noise, then S's. The noise scale solves the reliability ratio exactly, keeping Cov(M\*, e). |
| 16 | The ceiling's fallback | The count-model value replaces only the estimate that falls outside (0, 1). |
| 17 | Oracle N's selection | Its λ selection set includes the generating draw; only the rank leaves it out. If cut 3 applies (two truths), "at least 2 of 3" becomes "both". |
| 18 | Gates | Superseded by amendment 1, item 6. |
| 19 | The bootstrap | One 2,000 × 74 draw from `default_rng(9772)` is reused everywhere, with classes in sorted order and percentile intervals. r_ref's arm is chosen on the point estimate. "First of 20" means strictly greater than every readable draw. |
| 20 | Readability | An arm, or a null draw, is readable on a split only if all 5 of its fold fits are readable. |
| 21 | Unnamed outcomes | The map's unnamed cells are labelled, never silently dropped:<br>• R-B with one family unreadable: "*family* read alone: passes" or "fails";<br>• R-D above r_att by more than m, with its interval including 0: "above, unresolved";<br>• denoising not met: "does not denoise beyond copying". |
| 22 | Cuts and preflight details | Cuts are applied one at a time until the projection fits. The timed preflight fits use held-out-stimulation fold 0 at λ = 10⁻². C.2 − H0's width is computed for each split, on the entry set where T is finite. Predictions are returned to T's units before pooling. Fold heterogeneity is a per-fold least-squares scale and offset. |

## Readings added when amendment 1 was implemented

| # | Rule | The reading registered |
|---|---|---|
| 23 | The blob pin (amendment 1, item 12) | It applies to every read after the first, not only to repeats. "Accounts for the difference" means the named amendment itself changed the registration to its current blob. |
| 24 | The convergence cap (item 2) | The long cap covers **tied** at rank 28 and the sparse-support families. It does not cover the tied sweep, **dense** or **untied**. |
| 25 | Unnamed cells | R-C with one comparator unreadable reads "*tied or sparse-ER* read alone: wins" or "fails". **tied**'s R-A while **masked** is unreadable reads "Unreadable: m is undefined while masked is unreadable". |
| 26 | Checksum before logging | The checksum is verified before a read is logged, so a wrong file spends no once-only reading. |
| 27 | Equivalent mutants | Once amendment 1, item 5 applies, the within-fold exclusion is implied by the requirement that the transpose be defined. Removing either rule alone changes nothing. Removing both is caught by the tests. |

## Readings added with amendment 2

| # | Rule | The reading registered |
|---|---|---|
| 28 | A constant prediction (amendment 2) | `readings.is_constant`: standard deviation at most `CONSTANT_RTOL` (10⁻¹⁰) of the largest absolute value, over the split's entry set and over each fold's entries. One collapsed fold makes the arm unreadable on the split, with `why` set to `CONSTANT_PREDICTION`. The target is never consulted. |
