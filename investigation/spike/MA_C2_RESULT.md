# Reading M-a for C.2: result

Computed once on 2026-11-14, under `MA_C2_READING.md` with Amendment 1 (d9f928ac). Every number is
in `ma_c2/result.json`.

## The named reading

**Above chance, not above the floor.** C.2's response matrix correlates with the measured test
matrix at r = 0.0534. That is above the null's 97.5th percentile (0.0224), but below the box
centre's r = 0.0653. The unfitted network matches the measured responses slightly better than the
behaviour-fitted one. The two 95% intervals overlap widely (0.035 to 0.072 for C.2, 0.047 to 0.084
for the box centre), so the two are not told apart here. Both sit near a third of the data ceiling.

## Numbers

Primary definition: one unit added to neuron s's starting state, from rest, settled for the
forward's 4 iterations. The diagonal is masked everywhere.

| Quantity | Test (primary) | Train (description) |
|---|---|---|
| C.2, r | 0.0534 (95% CI 0.035 to 0.072) | 0.0725 |
| Box centre, r | 0.0653 (95% CI 0.047 to 0.084) | 0.0898 |
| Data ceiling, r(train, test) | 0.1734 (95% CI 0.155 to 0.192), 10,693 entries | |
| C.2 relative to the ceiling | 0.308 | |
| Box centre relative to the ceiling | 0.376 | |
| Null on C.2's matrix: 2.5th, 50th, 97.5th, 99th percentiles | -0.0195, -0.0001, 0.0224, 0.0258 | -0.0187, -0.0005, 0.0158, 0.0178 |
| C.2's position in its null | above all 1,000 | above all 1,000 |
| Box centre's position in C.2's null | above all 1,000 | |
| Null on the box centre's own matrix, 97.5th percentile | 0.0216 (box centre above all 1,000) | |

Descriptive extras (test matrix; train in `result.json`):

| Variant | C.2, r | Box centre, r |
|---|---|---|
| Starting state, input 0.5 | 0.0524 | 0.0647 |
| Starting state, input 2 | 0.0545 | 0.0669 |
| Held at every iteration, input 0.5 | 0.0444 | 0.0690 |
| Held at every iteration, input 1 | 0.0442 | 0.0704 |
| Held at every iteration, input 2 | 0.0433 | 0.0712 |

Sign agreement, on the 5,505 test entries whose measured magnitude is above the median (0.890):

| | C.2 | Box centre |
|---|---|---|
| Model zero counted as disagreeing | 0.471 | 0.483 |
| Model zeros left out | 0.509 (2,594 of 5,094) | 0.529 (2,660 of 5,025) |

C.2 leaves 411 of those entries at exactly zero, and the box centre 480. These are neurons that
four settling steps from one neuron do not reach. Over all off-diagonal entries, C.2 has 2,063
exact zeros. The held form has 4,575. Both orderings of the two brains hold at every amplitude and
in both forms.

## Checks before the reading

- **Parity:** the spike-local settling copy against the package's `forward_with_hidden_batched` at
  100 random nonzero sensory inputs (`default_rng(9732)`). The largest difference was 3.6e-7 for
  C.2 and 0 for the box centre, below the 1e-5 tolerance.
- **Unit test 1:** a starting-state input equal to a sensory drive reproduces the sensory forward.
  It passed: 3.6e-7 for C.2, 0 for the box centre.
- **Unit test 2:** a held input equal to the modulators' current reproduces the forward with those
  modulators. It passed: 4.8e-7 for C.2 (held currents up to 19.4), 0 for the box centre.
- **Rest:** with modulators at rest, the held current is zero and the settled activity is zero.
- **Names:** all 154 of Creamer's names match neurons in C.2 by exact string; none is unmatched.
- **Checksum:** the measured file's SHA-256 matches session 0's MANIFEST (`1c98ae81...`).
- **Gate:** both unit tests ran before the measured matrices were loaded. The script would have
  stopped there had either failed.
- **A failed check before Amendment 1, recorded here for completeness:** the first run tested the
  registered added-input path (an input held at every iteration, equal to a sensory drive), and it
  failed, with a largest hidden-activity difference of about 1.57 against 1e-5 (`parity.json`,
  `unit_test_passes: false`). That exposed the stimulation-path error which Amendment 1 corrected.
  The verdict is unaffected: the held form, run as description, gives the same ordering.

## Interpretations made

1. **Parity inputs.** Food strength and food contact were drawn from U(0.05, 1), and the lateral
   gradient and dC/dt from U(-1, 1). For unit test 2, all 56 modulator inputs (the concentrations
   and the drive) were drawn from U(0.05, 1), from the same random stream, after the features.
2. **Row and column order.** The measured matrices are indexed in the order of
   `creamer_154_names.json`, the released models' `cell_ids`. This is how his `predict_stams.py`
   scores model against measured, index by index.
3. **Rest.** Every modulator concentration is zero, as the batched controller's `reset` sets it, and
   the ingestion drive is zero, since nothing is eaten.
4. **Settled activity.** The response is the hidden state after the forward's 4 settling
   iterations, the `h` the forward returns.
5. **Diagonal.** The model matrix's diagonal is masked as well, as his scorer masks it. A joint
   relabelling keeps the diagonal on the diagonal, so it stays masked in the null.
6. **The null's draws.** There are 1,000 sequential `rng.permutation(154)` draws from
   `default_rng(9731)`, each applied as `M[q][:, q]`. "Exceeds" means strictly greater.
7. **"The box centre's own position in the null."** This is read as its percentile in C.2's null.
   Its percentile in a null built the same way (same permutations) from its own matrix is also
   reported.
8. **Sign agreement.** It is computed over off-diagonal entries finite in the measured matrix.
   "Above the median" means strictly above the median of |measured| over those entries. The primary
   figure counts a model zero as disagreeing. The figure with zeros left out is added beside it.
9. **The held form.** This description was run at 0.5, 1 and 2, like the starting-state form.

## Limits, from the registration

- C.2 settles within one step and has no slow dynamics. Only the pattern of signs and sizes is
  compared with a response integrated over 30 s.
- Creamer's arrays were built from a fuller stimulation record than the licensed deposit holds
  (SESSION_0.md). They are used as published in his MIT repository.
- This is a probe's answer (Logbook 131's `hinge4.json`), not C.2's registered fit. When that fit
  exists, it is the proper row.

## Scripts and outputs (`investigation/spike/ma_c2/`)

- `reading.py.txt`: builds both brains, runs parity and both unit tests, and only then loads the
  measured matrices and scores.
- `parity_check.py.txt`: the first parity run, before Amendment 1, which wrote `parity.json`.
- `layout_check.py.txt`: the check that `hinge4.json`'s `z` is in the package's layout.
- `result.json`: every number.
- `responses.npz`: the 12 response matrices. Gitignored (over the 500 KB limit), and regenerated by
  `reading.py.txt`.
- `parity.json`: the pre-amendment parity record, including the failed unit test.
