# Launch record 135: the neural readout's ceiling, and what learned models reach under it

Drafted 2026-11-15 on branch `investigation/neural-ceiling`. It was revised twice that day after an
independent review:

- the first review found that the first draft's unconstrained comparator could not predict an unseen
  column by construction, and that several floors and rules were missing;
- the second review found four text-level gaps: a vacuous negative oracle, the held-out-worm
  split's selection, the outcome map and the budget. All four are fixed here.

**Registered 2026-11-15, on the maintainer's approval,** before any code was written, any fit was
run or S was read. It changes only by committed amendment before any score, never after. A registered verdict is never re-read under a different rule.
Anything else is reported beside it as description.

## 1. What this asks, and what it cannot answer

Every neural reading so far sits near r 0.06 against Creamer et al.'s measured test responses:

- the unfitted encoder on the true wiring, 0.0653;
- C.2, 0.0534;
- 19 degree-preserving rewirings, median 0.0050 (Logbook 134).

Three questions, on the readout's own targets:

1. **How high could a model go?** Is the limit the data, or the model class?
2. **Does a learned model constrained to the anatomy predict better** than the same model on
   degree-preserving rewirings, and better than an unconstrained model of the same size?
3. **Do C.2 and H0 differ, paired?** Logbook 134 left this untested.

**Scope, stated before running.** This is a test of linear-response models fitted to aggregated
response matrices.

- It **cannot satisfy H2's causal bar.** That bar is registered for Mi et al.'s latent-voltage model,
  fitted to activity, in the fit-first recommendation.
- It can show **whether anatomy helps a learned linear model predict responses to stimulations it
  never saw.**
- It bears on the neural readout, and on what neural readings of W's arms could show. It does not
  bear on W's own question, which is about behaviour, or on W's cost.

## 2. Data

- **Targets:** `investigation/spike/session0/exports/measured_stams.npz`
  (SHA-256 `1c98ae81f5422ac5f9cc35bbd375ac7ea7815c59e3cf0d5545d08e3898237b28`, MIT, Creamer et al.
  at bba43302). It holds two 154 × 154 matrices, \[responder, stimulated\]:
  - **T,** from 80 training animals, with 21,754 finite entries;
  - **S,** from 30 test animals, with 11,010 finite entries.
- **Names:** `investigation/spike/session0/creamer_154_names.json`. The diagonal is masked
  everywhere.
- **No per-animal, per-trial or time-series data exist for these targets,** so neither a
  time-series forecaster nor a direct split-half ceiling is possible (SESSION_0.md, fact 13).
- **S is read only by the scoring script.** That script refuses to run unless this record is
  committed and unchanged. It logs every reading to `READS.md` in this folder, and each registered
  reading is made once.
  - S's values enter nothing else, with one stated exception: the oracle's noise calibration (§ 7)
    uses two summary numbers, Var(S) and Cov(T, S) on their overlap.
- **The scorer:** Creamer's `nan_corr` as ported for Logbook 134. It is one Pearson correlation over
  every entry in a split's registered entry set.

## 3. The ceiling

The ceiling is reported as description and is not a bar on any arm. Its two estimates do, however,
set the oracle's noise and so its attainable level (§ 7), so they are load-bearing there.

- **Where it is read:** the scoring script reads them and logs them in `READS.md`.

- **If an estimate falls outside (0, 1):** the oracle uses the count-model value at k = 80/30 instead,
  and the report says so.

- **Estimator.** Take T and S to measure the same truth, on the same scale, with independent noise.
  On the entries finite in both:

  - rel_S = Cov(T, S) / Var(S);
  - rel_T = Cov(T, S) / Var(T);
  - the ceiling for predicting S is **√rel_S**.

  Its 95% interval comes from 2,000 resamples of the 74 stimulated-neuron classes
  (`default_rng(9772)`).

- **Beside it:** the count-model value at k = 80/30 (0.34), with its bracket from r(T, S) to √r(T, S).

- **Biases, both stated:**

  - true differences between the training and test cohorts make the estimate too low;
  - artefacts shared between them make it too high.

- **The attainable reference.** The ceiling bounds knowing the truth, not generalising to unseen
  stimulations. That attainable level comes from the oracle (§ 7).

## 4. Splits and entry sets

**Classes.** The 302-neuron classes of `quantumnematode.connectome.neuron_classes`, restricted to
the 154: 74 classes. They are permuted by `default_rng(9771)` and dealt round-robin into 5 folds.

| Split | What is hidden from fitting | Scored entries | Status |
|---|---|---|---|
| **Held-out stimulations** | every column of T whose stimulated neuron is in the fold | S's finite entries in those columns, where the transpose floor and every reference are defined | primary |
| **Held-out neurons** | every row of T whose responder is in the fold | S's finite entries in those rows, excluding columns with no finite T entry (DB1) | secondary |
| **Held-out worms** | nothing: fit to all of T | every finite S entry with a finite T counterpart | the home of the matched-size reading (R-C) and the denoising reading |

- **Pooling:** each held-out split's predictions are pooled over its 5 folds. Per-fold r is reported
  as description.
- **Exclusions:** each split's entry set is fixed by the rule above before any arm runs, and its size
  and exclusions are recorded.
- **Pairs within one fold** are excluded from every arm's scored set. On the held-out-stimulation
  split, an entry (i, j) whose responder i is also in the fold would let **floor-T** read T[j, i]
  from a hidden column. The same rule applies to rows on the held-out-neuron split. The calibration
  fits use the same rule.

## 5. Arms

**The family.** Every fitted arm is a truncated linear response, matching H0's four settling
iterations: **M = W + W² + W³ + W⁴**, read on the 154.

- W has a zero diagonal (no autapses) in every arm.
- The series is always defined, with no inverse and no free scale.
- T is mean-centred and standardised by its visible entries.
- Each fit reports how much of M's variance each of W, W², W³ and W⁴ carries.
- The latent neurons carry an exact gauge, D·W·D⁻¹ with D = 1 on the 154. So predictions are
  identified but the fitted latent weights are not, and are not interpreted.

| Arm | Coupling W | Parameters |
|---|---|---|
| **floor-ME** | none. Held-out column: each responder's mean visible response. Held-out row: each stimulated neuron's mean visible effect. Held-out worms: row + column − grand mean. | 0 |
| **floor-T** | none. Predicts entry (i, j) by T[j, i] in the held-out splits. | 0 |
| **floor-C** | a + b·floor-ME + c·floor-T, the coefficients fitted on visible entries | 3 |
| **H0, H0-cal** | the unfitted encoder at the box centre, zero-shot (Logbook 134); and a + b·H0 + c·floor-ME + d·floor-T, fitted on visible entries | 0; 4 |
| **C.2, C.2-cal** | Logbook 131's answer, zero-shot; and its calibrated form, as for H0 | 0; 4 |
| **masked** | free on the anatomical support over 302 neurons: the union of Emmons 2024's directed chemical edges and its gap junctions both ways, off the diagonal | one per union entry |
| **null-D** | **masked** on each of H0's 19 degree-preserving rewirings, taken from the brains H0 built (`rewire_seed` 9741 to 9759), with the diagonal removed | as each support |
| **null-R** | **masked** on 19 reciprocity-preserving rewirings of the anatomical union support | as **masked** |
| **sparse-ER** | **masked** on 5 random supports of the same size over 302 neurons, off the diagonal (`default_rng(9801)` to `default_rng(9805)`) | as **masked** |
| **tied** | W = U R Uᵀ over the 154: one embedding per neuron, used in both roles. The rank is set so its parameter count is closest to **masked**'s. | 154·r + r² |
| **masked-prior** (description) | **masked** with W = s_c·log(1 + chemical count) + s_g·log(1 + gap count) + Δ, the ridge on Δ only; the nulls likewise, with their moved counts | union + 2 |
| **dense, tied-sweep, untied** (description) | W dense over the 154 with a ridge; **tied** at ranks 2, 5, 10, 20, 40; on held-out worms only, an untied UVᵀ at rank 17 | |

**What tied does.** Its route to an unseen column is a learned transpose map, so it generalises
**floor-T**. At rank 28 (5,096 parameters, against **masked**'s 5,103) it contains every untied
matrix of rank 14.

**Supports.** null-D's supports are about 12% larger than **masked**'s (5,729 to 5,749 entries,
against 5,103), because the package's null does not preserve the overlap of chemical and gap
edges. null-R and sparse-ER match **masked** exactly.

**Counts for masked-prior.** null-D's counts come from `rewire_degree_preserving` at the same seed,
and its supports are asserted equal to the built brains'.

**null-R's rewiring.** Entries whose reverse is also in the support are rewired among themselves by
undirected double-edge swaps. The rest are rewired by directed double-edge swaps.

- A swap that creates a duplicate, a self-connection or a new reciprocal pair is rejected.
- Each draw attempts 10 swaps per entry, with `default_rng(9781)` to `default_rng(9799)`.
- This preserves exactly the number of entries, every in- and out-degree, and the number of
  reciprocal pairs. Each draw asserts all three.
- Each draw reports its accepted swaps and the fraction of entries moved. A draw moving under 80%
  continues, up to 30 attempted swaps per entry.
- The code is spike-local, and tested on small graphs before use.

**Why the comparator is tied.** An untied low-rank linear response, (I − UVᵀ)⁻¹ − I, rewrites
exactly as a factorisation BVᵀ. A held-out column's factor is then left unidentified by the visible
data, so it could only be decided by the regulariser. **tied** shares one embedding per neuron
between rows and columns. A held-out column is then identified through its visible row, as it is for
the sparse arms through the paths that pass through it. Every arm's support size and reciprocity are
reported.

## 6. Fitting and selection

**The fit.**

- Loss: the mean squared error over visible entries, plus λ × the mean of the squared parameters.
- Optimiser: L-BFGS with a strong-Wolfe line search, history 20, at most 500 iterations, and an
  absolute loss change below 10⁻¹⁰.
- Initialisation:
  - the sparse arms' W at 0, which is not a stationary point of the series;
  - **tied**'s U and R from N(0, 0.01²) with `default_rng(9774)`. R starts nonzero because at
    R = 0 the gradient on U is exactly zero. The first timed fit checks that both move.
- A fit that ends at the iteration cap is flagged and kept. A fit whose loss is not finite is refit
  once at the next larger λ. If it is still not finite, it is unreadable.

**λ** is chosen from {10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1}.

- The grid is extended once, by a decade, if the choice lands on an edge.
- **On the held-out splits,** the choice is by 3-fold inner cross-validation over held-out classes
  within the visible set, so inner folds mimic outer folds.
- **On the held-out-worm split,** whose task is denoising seen entries, the choice is by 3-fold inner
  cross-validation over random held-out entries of T (`default_rng(9770)`).
- It is made per outer fold and split.
- **One λ serves every sparse-support arm** (**masked**, **null-D**, **null-R**, **sparse-ER**,
  **masked-prior**). It is chosen on the summed inner-CV loss of a fixed selection set:
  **masked**, null-D draws 9741 to 9744 and null-R draws 9781 to 9784. Masked is one of nine, and
  nothing is tuned on it alone.
- **tied** gets its own λ.
- The calibration coefficients (floor-C, H0-cal, C.2-cal) are fitted by least squares on visible
  entries.

**Readability.** A null family is readable if at least 17 of its 19 draws are readable. Unreadable
draws are reported and left out of the rank.

## 7. Oracles, before any real score

Every oracle uses the same splits, selection and arms as the real run, except H0 and C.2.

**The synthetic truth M\*:** **masked** fitted to all of T, at the λ its own inner cross-validation
chooses on T.

**Noise.** Residuals R = T − M\* on T's finite entries are resampled in column blocks, which keeps
their heavy tails. They are scaled so that Var(M\*) / Var(T_syn) = rel_T and
Var(M\*) / Var(S_syn) = rel_S, using the § 3 estimates. They are placed on exactly T's and S's
finite-entry patterns. There are three independent noise draws: `default_rng(9773)`, `9775` and
`9776`.

| Oracle | Truth | Must |
|---|---|---|
| **K, known answer** | M\*, with no noise (T_syn = S_syn = M\*) | **masked** recovers held-out columns, and separately held-out rows, at r ≥ 0.8. K is allowed 2,000 iterations, and records its visible-entry loss, which should be near 0. A failure with a near-zero loss means the channel is not identified for this family. A failure with a high loss means the optimiser stopped, and the report says which. |
| **P, positive** | M\* plus noise, 3 draws | on each held-out split separately, **masked** meets R-B's bar (§ 8) against both null families in at least 2 of 3 draws. On held-out worms, **masked** exceeds r(T_syn, S_syn) with its interval above 0 in at least 2 of 3 draws. |
| **N, negative** | the fits to T of null-D draws 9741, 9742 and 9743, one truth each, each with one noise draw from its own residuals | the generating draw is left out, and **masked** is ranked among the other 18 null-D draws. **N fails, and the pipeline favours the anatomical support for structural reasons, if** in at least 2 of 3 truths **masked** is first of 19, or exceeds those draws' median by at least m. |

- **Each oracle uses its own truth's residuals.** Each truth is fitted at the λ chosen by its own
  inner cross-validation on T.

- **The reference in every oracle is floor-C,** since H0 and C.2 are not part of the oracles.

- **Each split is gated by its own result:**

  - K and P gate the held-out-stimulation and held-out-neuron readings separately;
  - P's held-out-worm line gates the denoising reading.

- **The attainable level, r_att:** **masked**'s held-out r in P, as the median over its three draws,
  for each split. Its spread over the three draws is reported, beside √rel_S computed on the
  synthetic data.

- **Fold heterogeneity:** per-fold scale and offset in P are reported, to check that pooling is sound.

- **The oracles fit only the arms their gates read:**

  - **masked**, null-D, null-R and floor-C on the held-out splits;
  - **tied** and sparse-ER as well on held-out worms.

  masked-prior, dense and the sweeps are not fitted in the oracles.

## 8. Readings

Each is read once.

**The paired intervals** are 95% bootstraps over classes: 2,000 resamples of the 74 classes,
`default_rng(9772)`. The unit is stimulated-neuron classes for the held-out-stimulation and
held-out-worm splits, and responder classes for the held-out-neuron split. Within each resample,
both arms of a pair are recomputed, and so is the nulls' median.

**The minimum effect** is m = max(0.25 × (r_masked − r_ref), 0.0109). The 0.0109 is H0's committed
m0. r_ref, the best reference, is the highest of floor-C, H0-cal and C.2-cal, judged on the split.

| Reading | Content | Splits |
|---|---|---|
| **R-0** | every arm's and reference's r; and C.2 − H0, paired, which closes Logbook 134's open point | all three |
| **R-A** | **masked** − r_ref, and **tied** − r_ref, paired | held-out stimulations (primary), held-out neurons |
| **R-B** | **masked** against each null family: its rank among 20; **masked** minus the family's median, paired | held-out stimulations (primary), held-out neurons |
| **R-C** | **masked** − **tied** at matched size, and **masked** − the median of **sparse-ER**, paired; and the denoising reading, **masked** − r(T, S) | held-out worms |
| **R-D** | real **masked** r against r_att | held-out stimulations, held-out neurons |

## 9. What each outcome means, with precedence

The held-out-stimulation split governs. The others are reported, and a disagreement is named but not
resolved.

**Gate.** If oracle K fails, R-A, R-B and R-D on held-out stimulations are **unreadable: the channel
is not identified for this family**. Only the held-out-worm split is read.

**Learned against the best reference (R-A),** for each of **masked** and **tied**:

- **below the reference:** its interval lies entirely below 0;
- **not above the reference:** its interval includes 0;
- **above, under the minimum:** the interval is above 0, but the difference is less than m;
- **above the reference:** at least m, with the interval above 0. This means there is structure
  beyond the floors and the calibrated mechanistic models.

**Anatomy in a learned model (R-B).** A family "passes" when **masked** is first of 20 against it,
at least m above its median, with the paired interval above 0. Read in this order:

1. **Unreadable,** if **masked** itself is unreadable, or oracle K or P fails on the split.
   **Structurally biased,** if oracle N fails. Nothing below is read.
2. **A family is unreadable** if fewer than 17 of its 19 draws are readable. Its cell is then
   reported as unreadable, and the other family's cell is read alone.
3. **Reversed,** if a paired interval against either family's median lies entirely below −m.
4. **Otherwise, by which families pass:**

| null-D | null-R | Reading |
|---|---|---|
| passes | passes | **Anatomy helps beyond degree and reciprocity statistics.** |
| passes | fails | **The advantage is reciprocity and overlap,** which null-R keeps and null-D does not. |
| fails | passes | **Anatomy beats reciprocity-matched rewirings, but not degree-matched ones.** null-D's supports are about 12% larger, which may let it compensate. Reported as found, not resolved. |
| fails | fails | **Anatomy does not help a learned model here.** |

**Size against structure (R-C, held-out worms).** Two contrasts, each "won" when **masked** beats
the comparator by at least m with the interval above 0:

- against **tied**: equal size, no sparsity;
- against the median of **sparse-ER**: equal sparsity, no anatomy.

Read in this order:

1. **Unreadable,** if P's held-out-worm line fails.
2. **Reversed,** if either interval lies entirely below −m.
3. **Otherwise:**

| Against tied | Against sparse-ER | Reading |
|---|---|---|
| wins | wins | **Anatomy helps at equal size, beyond random sparsity.** |
| fails | wins | **Anatomy beats random sparsity,** but not an equal-size dense model. |
| wins | fails | **Sparsity helps, not anatomy:** random sparsity does as well. |
| fails | fails | **Neither the constraint nor sparsity helps here.** |

- **Denoising, read beside R-C:** **masked** − r(T, S) is at least m, with the interval above 0.
  Then it denoises beyond copying the training matrix. Gated by P's held-out-worm line.

**Data against model class (R-D).** The interval is the class bootstrap of real r, minus a fixed
r_att (the median of P's three draws). The spread of r_att over the draws is reported beside it.

- **Above the attainable level:** above r_att by more than m, with the interval above 0. The real
  truth differs from the oracle's model, or the oracle understates what is attainable.
- **Data-limited:** within m of r_att. A model of the right class could not do better at this data
  size.
- **Model- or truth-limited:** below r_att by more than m, with the interval below 0.
- **Unresolved:** below r_att by more than m, but the interval includes 0.

## 10. Preflight, budget, stop rules, seeds

**Preflight**, before the oracles. It reads no S.

- Each split's entry set and its size.
- The paired-bootstrap width of C.2 − H0 computed against T, to show what differences are
  resolvable.
- One fit of each kind, timed. The full cost is projected from that timing and recorded.

**Budget.** The sparse arms use sparse matrix products.

| Data set | Fits | Breakdown |
|---|---|---|
| Real | about 3,600 | about 330 per fold-unit: 135 to select λ, 83 sparse final fits, 16 for **tied**, 16 for **dense**, up to 80 for the sweep; 11 fold-units |
| P, 3 draws | about 5,800 | about 1,935 per draw: the gated arms only |
| N, 3 truths | about 4,600 | the held-out splits, null-D only |
| K | about 20 | **masked** only |
| **All** | **about 14,000** | |

- **Time:** at 1 to 3 s a fit with sparse products, that is about 4 to 12 h on one process, or about
  0.5 to 1.5 h on 8.
- **Sessions:** 3 part-time sessions.

**Compute.** It runs only within the 16-worker core budget. The maintainer decides whether it waits
for C.2's pilot or runs beside it at reduced width.

**If the timed preflight projects more than 6 h on 8 processes,** cut in this order, and record each
cut before any score:

1. masked-prior, dense, the sweeps and the untied arm (all description);
2. the held-out-neuron split's oracles, making that split description only;
3. N to two truths.

If the projection is still over 6 h after those cuts, stop and amend.

**Stop rules.**

- Stop if any data step needs more than one session of new pipeline work.
- Stop if oracle K fails on the held-out-stimulation split: the held-out-worm split is then the only
  reading.

**Seeds.**

| Seed | Use |
|---|---|
| 9770 | the held-out-worm split's inner folds |
| 9771 | the folds |
| 9772 | the bootstraps |
| 9773, 9775, 9776 | oracle P's noise |
| 9777, 9778, 9779 | oracle N's noise, one per truth |
| 9774 | **tied**'s initialisation |
| 9781 to 9799 | null-R |
| 9801 to 9805 | sparse-ER |
| 9741 to 9759 | null-D, reused from H0 |

They enter the phase's seed table with the logbook.

**Report.** Logbook 135, with:

- the ceiling and its interval;
- every arm's r with intervals;
- the oracles;
- every registered reading, with its branch;
- one paragraph on what the result implies for the neural track and for W's neural readings;
- a dated note on Logbook 134 that 0.173 was a reference, not a ceiling.
