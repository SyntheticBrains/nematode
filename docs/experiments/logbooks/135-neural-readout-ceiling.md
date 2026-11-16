# 135: The Neural Readout's Ceiling Is About 0.42, and a Linear Model Learned on the Connectome Does Not Beat the Calibrated Unfitted Network Under It (Phase 10 / the fit-first investigation)

**Status**: `completed`. **Ceiling 0.42; R-A not above the reference; R-B and R-D unreadable
(oracle P failed); R-C: anatomy beats random sparsity; no denoising beyond copying.** Established
on Creamer et al.'s two measured response matrices:

- the reliability ceiling for predicting the test matrix is **0.423** (95% interval 0.369 to 0.471);
- on held-out stimulations, the best predictor is the calibrated unfitted network (H0-cal, 0.132,
  31% of the ceiling), and a linear response learned on the anatomical support (0.099) is not above
  it; on held-out neurons it is below it;
- on held-out worms, the anatomical support beats random supports of the same size by 0.106, but the
  learned model does not improve on copying the training matrix;
- paired, C.2's behaviour-fitted answer is not above its unfitted network on any split: C.2 − H0 is
  −0.012 to −0.013, with the interval including 0 on the governing split and just below 0 on the
  other two.

Unresolved: whether anatomy helps a learned model beyond degree- and reciprocity-matched rewirings.
The positive oracle showed that this test, as registered (every ridge selection at the grid's
extended edge), cannot detect a true anatomical model at the data's reliability, so R-B is
unreadable, not null. **Condition on an earlier result:** Logbook 134's
"data ceiling" of 0.173 was r(training, test), a reference, not a ceiling; its shares of the
ceiling become about 15% and 13%, and its "no detectable gain" from behaviour fitting holds
paired.

**Date**: 2026-11-16. Registered 2026-11-15, amended twice before the registered reading (see
[Corrections](#corrections-made-in-the-open)). Before it, S's values entered only § 3's summaries.

**OpenSpec change**: none. A side spike of the fit-first investigation, local to
[investigation/neural-ceiling/](../../../investigation/neural-ceiling/). No package code changed.

**Pre-registration**:
[supporting/135-neural-readout-ceiling/launch.md](supporting/135-neural-readout-ceiling/launch.md),
registered (4ab1aaa7) before any code was written or any fit run, with amendment 1 (ac3e535e)
before any fit to real data and amendment 2 (869c5659) before the registered reading. Every access
to S is logged in [READS.md](supporting/135-neural-readout-ceiling/READS.md): its finite-entry
pattern, then § 3's summaries, then the registered reading, each once.

## Objective

Every neural reading so far sat near r 0.06 against the measured test responses (Logbook 134: the
unfitted network 0.065, C.2 0.053, 19 degree-preserving rewirings 0.005). The launch record asked
three questions of the readout's own targets:

1. **How high could a model go?** Is the limit the data, or the model class?
2. **Does a learned model constrained to the anatomy predict better** than the same model on
   degree-preserving rewirings, and better than an unconstrained model of the same size?
3. **Do C.2 and H0 differ, paired?** Logbook 134 left this untested.

It is a test of linear-response models fitted to two aggregated matrices. It cannot meet H2's causal
bar, which the fit-first recommendation registers for a model fitted to activity.

## Method

The launch record holds the full design; in brief:

- **Targets.** T (80 training animals, 21,754 finite entries) and S (30 test animals, 11,010), both
  154 × 154 [responder, stimulated], from Creamer et al.'s `measured_stams` (MIT, bba43302). No
  per-animal data exist for them. S was read only through a guarded scorer.
- **The ceiling.** √rel_S, with rel_S = Cov(T, S) / Var(S) on the 10,693 entries finite in both: T
  and S measure one truth with independent noise. Its interval resamples the 74 stimulated-neuron
  classes 2,000 times.
- **Splits.** The 74 classes of the 154, permuted (`default_rng(9771)`) into 5 folds. Held-out
  stimulations (primary, 8,588 scored entries) hides the fold's columns; held-out neurons (8,541)
  its rows; held-out worms (10,693) fits all of T and scores S.
- **The model family.** A truncated linear response, M = W + W² + W³ + W⁴, W zero on the diagonal,
  read on the 154. Arms: **masked** (W free on Emmons 2024's union of chemical and gap edges, 5,103
  entries over 302 neurons); **null-D** (H0's 19 degree-preserving rewirings); **null-R** (19
  reciprocity-preserving rewirings); **sparse-ER** (5 random supports of 5,103); **tied** (W = URUᵀ
  at rank 28, 5,096 parameters). Floors and references: **floor-ME** (row and column means),
  **floor-T** (the transpose), **floor-C** (their least-squares combination), H0 and C.2 zero-shot,
  and H0-cal and C.2-cal (each calibrated with the floors on visible entries). Description arms:
  masked-prior, dense, a tied sweep (ranks 2 to 40) and an untied rank-17 model.
- **Fitting.** L-BFGS, a ridge λ chosen per fold by 3-fold inner cross-validation from {10⁻⁴ … 1},
  extended once by a decade at an edge, one λ shared by every sparse-support arm.
- **Oracles, before any real score.** K (a known masked model, no noise), P (a masked fit to T plus
  noise at the estimated reliabilities, 3 draws) and N (null-D fits to T as truths, 3 truths). They
  gate the readings: R-A by K; R-B by K, P and N; R-D by K and P; R-C and the denoising reading by
  P's held-out-worm line.
- **Readings.** R-A: masked and tied against the best reference. R-B: masked against each null
  family (first of 20, at least m above the median, interval above 0). R-C (held-out worms): masked
  against tied and against sparse-ER's median, with the denoising reading (masked against
  r(T, S)). R-D: masked against the oracle's attainable level. R-0: C.2 − H0, paired. The minimum
  is m = max(0.25 × (r_masked − r_ref), 0.0109); intervals are paired class bootstraps
  (`default_rng(9772)`).
- **Compute.** 16,326 fits on 8 single-thread workers: 6,705 for P, 5,430 for N, 10 for K, 76 truth
  fits and 4,105 real. The oracle stage ran from its § 3 read at 06:41 to 10:48 on 2026-11-16 (its 76 truth
  fits first, then K, P and N from 06:44), and the real stage 10:51 to 11:49; the reading was made
  at 11:55.

## Results

### Preflight

- **Convergence.** At λ = 10⁻², masked's visible loss at 500 iterations exceeded its loss at 2,000
  by 2.7% of the latter, above the 1% bar, so every sparse-support and tied fit ran to a cap of 2,000.
- **Projection.** 14,145 fits, 3.8 h on 8 workers, inside the 6-h budget: no cut.
- **C.2 − H0 against T** (preflight's resolution check, not a reading): about −0.021 on every split,
  interval width 0.029 to 0.042.

### The ceiling

| Estimate | Value |
|---|---|
| **√rel_S, the ceiling for predicting S** | **0.423** (0.369 to 0.471) |
| rel_S; rel_T | 0.179; 0.168 |
| r(T, S), Logbook 134's "data ceiling" | 0.173 |
| Count model at k = 80/30, and its bracket | 0.341 (0.173 to 0.416) |

The estimate lies inside (0, 1), so no fallback was used. Two biases are stated in the registration:
true differences between the cohorts make it too low; artefacts shared between them make it too
high.

### The oracles

| Oracle | Held-out stimulations | Held-out neurons | Held-out worms |
|---|---|---|---|
| **K**, known answer | **passes**: r 0.992, visible loss 3.0 × 10⁻⁸ | **passes**: r 0.996 | not run |
| **P**, positive (3 draws) | **fails** in 3 of 3 | **fails** in 3 of 3 | **passes** in 3 of 3 |
| **N**, negative (3 truths) | **passes**: masked ranks 10, 11 and 5 of 19 | **passes**: biased in 1 truth of 3 | not run |

**Why P fails.** When the truth is a masked model, masked does not clear either null family with an
interval above 0:

| P draw | Held-out stimulations: masked r; against null-D (rank, margin) | Held-out neurons: masked r; against null-D |
|---|---|---|
| 9773 | 0.127; 4th of 20, +0.005 (−0.053 to 0.103) | 0.151; 1st, +0.057 (−0.020 to 0.140) |
| 9775 | 0.145; 6th, +0.017 (−0.060 to 0.111) | 0.144; 1st, +0.029 (−0.060 to 0.130) |
| 9776 | 0.132; 5th, +0.015 (−0.047 to 0.095) | 0.182; 1st, +0.071 (−0.035 to 0.188) |

null-R gives the same picture (ranks 3 to 4 on held-out stimulations, 1 to 2 on held-out neurons,
every interval including 0). On held-out worms, masked beats r(T_syn, S_syn) by 0.132 to 0.153, every
interval above 0, which gates the denoising reading. The attainable level, r_att, is 0.132
(0.127 to 0.145) on held-out stimulations and 0.151 (0.144 to 0.182) on held-out neurons. The
synthetic ceilings are 0.438 to 0.470.

Pooling is uneven in P. In five of the six draw-and-split cells, fold 1's least-squares scale of
target on prediction is 54 to 123, against −0.4 to 20 in every other fold: that fold's predictions
vary far less than its target does.

### The registered readings

| Reading | Result | Branch |
|---|---|---|
| **R-A**, held-out stimulations (governs) | masked 0.099 − H0-cal 0.132 = −0.033 (−0.068 to 0.003); m 0.0109 | **not above the reference** |
| R-A, held-out neurons | masked 0.031 − H0-cal 0.120 = −0.089 (−0.126 to −0.043) | **below the reference**, a named disagreement |
| R-A, tied | collapsed on every split (amendment 2) | **unreadable** |
| **R-B**, both held-out splits | | **unreadable**: oracle P failed |
| **R-C**, held-out worms | masked 0.140 − sparse-ER median 0.034 = +0.106 (0.072 to 0.148); m 0.0120; tied unreadable | **sparse-ER read alone: wins** |
| Denoising, held-out worms | masked 0.140 − r(T, S) 0.173 = −0.033 (−0.074 to 0.007) | **does not denoise beyond copying** |
| **R-D**, both held-out splits | | **unreadable**: oracle P failed |
| **R-0**, C.2 − H0 | held-out stimulations −0.012 (−0.030 to 0.001); held-out neurons −0.013 (−0.025 to −0.001); held-out worms −0.013 (−0.029 to −0.0004) | reported |

### Every arm against S

r with its class-bootstrap interval, and as a share of the ceiling. The null families are given by
their medians.

| Arm | Held-out stimulations | Held-out neurons | Held-out worms |
|---|---|---|---|
| **masked** | 0.099 (0.062 to 0.134), 23% | 0.031 (0.007 to 0.072), 7% | 0.140 (0.106 to 0.178), 33% |
| null-D, median of 19 | 0.021 | 0.013 | 0.037 |
| null-R, median of 19 | −0.004 | 0.009 | 0.054 |
| sparse-ER, median of 5 | −0.032 | −0.023 | 0.034 |
| tied | unreadable | unreadable | unreadable |
| floor-ME | 0.035 | 0.039 | 0.067 |
| floor-T | 0.120 (0.080 to 0.166), 28% | 0.123 (0.085 to 0.164), 29% | |
| floor-C | 0.119, 28% | 0.106, 25% | 0.067 |
| H0, unfitted | 0.072 (0.038 to 0.105), 17% | 0.072, 17% | 0.069, 16% |
| **H0-cal** | **0.132 (0.088 to 0.176), 31%** | **0.120 (0.081 to 0.158), 28%** | 0.092, 22% |
| C.2, behaviour-fitted | 0.059 (0.026 to 0.093), 14% | 0.060, 14% | 0.056, 13% |
| C.2-cal | 0.128, 30% | 0.116, 27% | 0.084, 20% |
| r(T, S), copying T | | | 0.173 (0.133 to 0.214), 41% |
| masked-prior (description) | 0.063, 15% | 0.063, 15% | 0.123, 29% |
| dense (description) | 0.012 | 0.013 | 0.173, 41% |
| untied rank 17; tied rank 40 (description) | | | 0.159; 0.163 |

### Reported beside the readings, as description

- **masked against the rewirings, on real data.** On held-out stimulations, masked is first of 20
  against both families: +0.078 (0.043 to 0.112) over null-D's median and +0.103 (0.067 to 0.144)
  over null-R's. On held-out neurons it is first and second, with intervals including 0. These
  margins on real data are larger than any P draw's on held-out stimulations, but P is the gate that
  says whether this test can see anatomy at all, and it says no. They are not a reading.
  masked-prior is likewise first of 20 against its prior nulls on every split.
- **λ sat at the extended edge.** Every sparse-support arm (masked, both null families,
  sparse-ER, masked-prior) chose λ = 10, the one-time extension, in every fold of every split, on
  real data and in every P and N oracle (K's λ is fixed at 10⁻⁴). The inner loss was still falling
  there. On held-out stimulations,
  masked's summed inner loss at λ = 10 was 14.59, against 14.31 for predicting the fold's mean, so
  the learned sparse models are as shrunk as the registered grid allows.
- **tied collapsed.** On every split of the real data, and on held-out worms in all three P draws,
  tied's inner cross-validation chose λ = 1 or 10, where W = URUᵀ falls to zero within a few
  iterations and predicts a constant. At λ ≤ 0.1 its summed inner loss was 147 to 388 on the held-out
  splits, against 14.3 for the constant. The tied sweep collapsed in some or all folds at every rank
  and split, except rank 40 on held-out worms. No fold of masked, the nulls, sparse-ER, dense, untied or
  masked-prior collapsed.
- **dense copies T.** On held-out worms, dense (0.173) equals copying the training matrix; it chose
  λ = 10⁻⁴.
- **Cap.** 8,660 of the 16,326 fits ended at their iteration cap, flagged and kept as registered.

## Corrections made in the open

- **Amendment 2: a constant prediction is unreadable.** After the oracles, P's tied fits on held-out
  worms were found to predict the training mean exactly, and Creamer's `nan_corr` returned −∞, +∞
  and −0.094 for them. In those draws the infinite values would have read R-C's contrast against
  tied as "fails", and the finite rounding noise as a spurious win of about +0.47; amendment 2's
  text names only the first. On real data, an infinite tied score would have read R-A's tied cell
  as "above the reference" or "above, under the minimum". The maintainer chose, before the
  registered reading, to make such an arm unreadable. When the real fits to T then showed tied collapsing fold by fold (five
  folds, five different constants, which the first rule missed), the maintainer extended the rule to
  any collapsed fold, again before the read. Those real fits were seen before the extension was
  chosen; they are fits to T and carry nothing of S. No gate depends on the rule: no masked, null or
  sparse-ER fit collapses in K, P or N. Its effect on the readings is that tied's R-A cells and R-C's
  contrast against tied read unreadable, and R-C is read on sparse-ER alone.
- **The cost was under-projected.** Preflight projected 14,145 fits and 3.8 h. It could not count
  the λ-grid extensions, which depend on the data, and every sparse selection extended. The run made
  16,326 fits in about 5 h, inside the 6-h budget, so no cut applied. The registration's own
  estimate before the build (0.5 to 1.5 h) assumed 1 to 3 s a fit; at the 2,000 cap a sparse fit
  takes about 8 s.
- **Oracle P's tied values.** The oracle stage recorded tied's held-out-worm r as −∞, +∞ and the
  finite −0.094 before amendment 2, because it ran on the earlier scoring code. `oracle_per_arm.csv` keeps the recorded values and flags those rows `collapsed`;
  `analysis.json` labels them unreadable. Neither enters a gate.
- **Amendment 2's tests are weaker than they look.** One documents `nan_corr`'s behaviour with an
  assertion any nonzero r passes, and none pins the threshold's relative form: an absolute
  threshold of 10⁻¹⁰ passes them all. The code uses the relative form, as registered.
- **The governing split disagrees with the secondary one on R-A** (not above, against below). The
  registration names a disagreement and does not resolve it; held-out stimulations governs.

## What this establishes, and what it does not

**Establishes.**

- Under the reliability model, the measured test matrix can be predicted at up to about r 0.42
  (0.37 to 0.47). Every model here reaches at most 31% of that on unseen stimulations.
- On unseen stimulations, a linear response learned on the anatomical support, under this fitting,
  is not above the best reference, the unfitted network calibrated with the floors (H0-cal); on
  unseen responders it is below it. As description, the highest point estimates are H0-cal (0.132),
  C.2-cal (0.128) and the transpose floor (0.120), which no registered contrast separates; H0-cal
  itself contains the floors.
- On held-out worms, the anatomical support beats random supports of the same size by 0.106, so it
  is not just sparsity. But the learned model does not denoise: it is not above copying T (0.140
  against 0.173, difference −0.033, −0.074 to 0.007), and a dense model does as well as copying.
- Paired, behaviour fitting shows no neural gain: C.2's answer is not above its unfitted network on
  any split (C.2 − H0 is −0.012 to −0.013).
- The pipeline recovers a known model (K), and passes the negative oracle (N): when the truth is a
  rewiring, masked was favoured in 1 of 6 truth-and-split cells, under the gate's 2-of-3 rule.

**Leaves unresolved.**

- **Whether anatomy helps a learned linear model beyond degree and reciprocity.** Unresolved at this
  data's reliability: the positive oracle could not detect a true masked model against the nulls, so
  R-B is unreadable. On the real data, masked is first of 20 against both families on held-out
  stimulations, by margins larger than P's; that is description the gate does not license.
- **Whether the limit is the data or the model class** (R-D unreadable for the same reason).
- **Whether behaviour fitting costs neural fidelity.** C.2 − H0's interval lies just below 0 on
  held-out neurons and worms but includes 0 on the governing split, and the three splits score the
  same two fixed predictions against overlapping entries of one S, so they are not replications.
- **Whether a matched-size unconstrained model would do as well as the anatomy.** tied collapsed
  under the registered grid, so that comparison was not made.
- **Whether a larger ridge would change the sparse arms.** Every selection sat at the extended
  edge, λ = 10.

**Does not establish.**

- **That anatomy does not help.** An unreadable R-B is not a null.
- **Fidelity in signs, dynamics or closed-loop behaviour**, or anything at H2's causal bar.
- **Anything on a licensed, complete record.** As in Logbook 134, the measured arrays come from a
  fuller record than the CC-BY deposit, probably the OSF exports, which state no licence.

## What it implies for the neural track and for W's neural readings

The standing readout has more room than Logbook 134 implied: its ceiling is about 0.42, not 0.17,
and the best model reaches about a third of it. That room is not cheaply taken by a learned model on
these targets. With only two aggregated matrices, a learned linear model's held-out score is
dominated by shrinkage (every selection at the grid's edge), and the positive control shows that,
so fitted, such a model cannot tell a true anatomical support from rewirings at this reliability. So W's neural
readings should keep using the unfitted network's score against H0's degree-preserving null, as in
Logbook 134, which does separate the true wiring from its rewirings, and should report the
calibrated network (H0-cal) as the reference to beat. Paired, C.2 shows no neural gain over its
unfitted wiring, and its small deficit is below 0 on two splits but not on the governing one; W's
arms can be read for the same paired difference. A learned test of anatomy would need more independent data than two aggregated
matrices; [DATA_CHECK.md](../../../investigation/neural-ceiling/DATA_CHECK.md) lists the datasets that
pair controlled inputs with neural activity and behaviour, without building anything.

## Registered consequences

- **The report's contents** (the ceiling, every arm, the oracles, every reading with its branch, the
  implications paragraph): met, above.
- **The dated note on Logbook 134** that 0.173 was a reference, not a ceiling: met, in this change,
  at every site that cites it as the ceiling (Logbook 134, its index row, and the investigation's
  RECOMMENDATION.md, SPIKE_REPORT.md, COMPARISON.md, IDENTIFIABILITY.md, RECONCILE.md,
  REVIEW_PART_A.md and spike/MA_C2_RESULT.md). The same note records R-0's paired answer to 134's
  open point wherever 134 says no paired test was run.
- **The seeds enter the phase's seed table:** met, in this change.
- **The data check:** met, as desk work
  ([DATA_CHECK.md](../../../investigation/neural-ceiling/DATA_CHECK.md)); nothing was built.
- **R-B, anatomy in a learned model:** unreachable-with-reason. The positive oracle shows the test,
  as registered, cannot detect a true anatomical model at the data's reliability; it would need
  more independent data, or a fitting that is not held at the ridge grid's edge.
- **R-D, data against model class:** unreachable-with-reason, for the same oracle.
- **R-C against tied:** unreachable-with-reason. tied collapses under the registered grid on every
  split.
- **The stop rules:** neither triggered. No data step needed new pipeline work beyond the spike, and
  K passed on held-out stimulations.
- **The readout's use in Phase 10** (the implications above for H1 and W's neural readings):
  deferred-with-destination, to the Phase 10 session through an addendum to
  [investigation/HANDOVER_PHASE10.md](../../../investigation/HANDOVER_PHASE10.md). As for Logbook 134,
  this change does not write tracker or roadmap entries, which the Phase 10 session owns.

## Artefacts

Under [supporting/135-neural-readout-ceiling/](supporting/135-neural-readout-ceiling/):

- `launch.md`: the registration and both amendments.
- `READS.md`: every access to S, logged before it was decoded and committed after it (the
  end-of-file hook later removed its trailing blank line).
- `registered_reading.json` and `real_per_arm.csv`: the score stage's outputs, verbatim (the JSON
  files here gain only the final newline the repository's end-of-file hook adds). Every real
  arm's r, interval, entry count and per-fold r on each split; every registered contrast; the
  branches.
- `section3.json`: the ceiling and its inputs. `preflight.json`: the entry sets, the convergence
  check and the projection.
- `oracle_per_arm.csv`: every oracle P arm's r on each split, as recorded, with the collapsed rows
  flagged.
- `analysis.json`: everything else stated above: the oracles' gates and per-draw readings, r_att,
  shares of the ceiling, Logbook 134's values against this ceiling, the λ choices, the inner losses,
  the collapse audit, the cap counts, the fit counts and the stage times.

Drivers: `investigation/neural-ceiling/run.py.txt` (setup, preflight, oracles, real, score), its
package `ceiling/` and tests (176), and `summarise.py.txt`, which wrote the supporting data from the
run directory. They are kept as `.py.txt`. The score stage ran at 869c5659; the oracle and real
stages ran at 15aa524f, whose fitting code is the same, while amendment 2 changed only the scoring
(`readings`, `config`, `report`), so the oracle stage's per-draw readings were scored by the
earlier code. Apart from one trailing blank line the end-of-file hook removed from
`pipeline.py.txt`, the files are identical to 869c5659. The folder's README says how to run them. The run directory itself (fits, predictions, checkpoints) is not committed.
