# Reading M-a for C.2: the current model against measured neural responses

Registered 2026-11-14, before anything below is computed (decision A28). It is description, not a
bar. Its rules are fixed here and the reading is made once.

## What it asks

C.2 was fitted only to behaviour. Do its fitted head responses to single-neuron stimulation resemble
the measured ones more than an unfitted network's do, and more than chance?

## Inputs

- **Measured responses:** Creamer et al.'s `measured_stams.pkl` (his MIT repository at bba43302),
  exported in session 0 to `session0/exports/measured_stams.npz` with its checksum. It holds two
  154 × 154 matrices [responder, stimulated], one from 80 training animals and one from 30 test
  animals. C.2 was fitted to neither, so both are held out from it. The test matrix is primary.
- **C.2:** Logbook 131's answer, the `z` vector (137 entries) in
  `docs/experiments/logbooks/supporting/131-c2-clean-start/probes/hinge4.json`. It is converted with
  `c2_encoder_probe.package_vector(z, entries)` under the encoder options that run used.
- **The floor:** the same encoder at the box's centre.

## The response matrix

From rest (no sensory input, modulators at their initial values), one step of the encoder's settling
is run with an input of +1 added to neuron s's pre-activation at every settling iteration. The
response of neuron j is its settled activity. The network has no biases, so rest is zero. This is
done for each of the 154 neurons that C.2 also contains (matched by name). Inputs of 0.5 and 2 are
description only.

The settling is a spike-local copy of the package's encoded forward. Before any reading, it must
match the package's forward at 100 random nonzero sensory inputs (`default_rng(9732)`) to 10⁻⁵.
A unit test checks the added-input path: an input equal to a sensory neuron's own drive must
reproduce the sensory forward.

## The metric and the null

- **The metric** is Creamer's `nan_corr`: one Pearson correlation over every entry finite in both
  matrices, with the diagonal masked as his scorer masks it. It is read against the test matrix
  (primary) and the training matrix (description). The data ceiling is the correlation between the
  training and test matrices, by the same function, and C.2's score is also reported relative to it.
- **The null:** 1,000 joint relabellings of C.2's response matrix, with the same permutation applied
  to its rows and columns (`default_rng(9731)`). This gives the correlation's chance distribution
  for a matrix with C.2's structure.

## The readings, named in advance

- **At chance:** C.2's correlation with the test matrix is below the null's 97.5th percentile.
- **Above chance, not above the floor:** it exceeds the 97.5th percentile but not the box centre's
  correlation.
- **Above both:** it exceeds both.

Reported beside them, as description: the box centre's own position in the null; the training
matrix readings; sign agreement over entries whose measured magnitude is above the median; and the
inputs 0.5 and 2.

## Limits, stated before reading

- C.2 settles within one step and has no slow dynamics, so only the pattern of signs and sizes is
  compared with a response integrated over 30 s.
- Creamer's arrays were built from a fuller stimulation record than the licensed deposit holds
  (SESSION_0.md). They are used as published in his MIT repository.
- This is a probe's answer, not C.2's registered fit. When that fit exists, it is the proper row.

## Amendment 1 (2026-11-14, before any measured matrix was loaded)

The parity check ran and passed (`ma_c2/parity.json`): the spike-local settling copy matches the
package's forward to 4.8 × 10⁻⁷ at most, and all 154 names match. No measured matrix was loaded and
no reading was computed. Two rules above were wrong about the code, and are corrected here. Neither
correction can be informed by the outcome.

1. **The vector.** `hinge4.json`'s `z` is already in the package's layout. Logbook 131's run wrote
   it directly with `encoder.write_encoder(brain, z)` on the brain built with the probe module's
   `PROCEDURE_SETTINGS`. `package_vector` converts an older probe layout: applied to this `z`, it
   would move 9 of the 137 entries and score a brain that was never fitted. The rule becomes: `z` is
   written exactly as the run wrote it. The box centre is `[0.5] × 137` in the same layout.
2. **The stimulation.** In the package's encoded forward, sensory drive enters only as the settling's
   starting state (`h0 = features @ injection`), and each iteration is `h = tanh(h @ R + held)`,
   where `held` is the modulators' current. An input added at every iteration therefore does not
   correspond to how the network receives drive.
   - **Primary:** the stimulation enters as the starting state, the way C.2's senses do. One unit is
     added to neuron s's starting state, with every other entry at rest.
   - **Description:** the stimulation is held at every iteration, as the modulators' current is.
   - **Unit tests:** a starting-state input equal to a sensory drive must reproduce the sensory
     forward, and a held input equal to the modulators' current must reproduce the forward with
     those modulators. Both to 10⁻⁵.
     Everything else (metric, null, seeds, named readings) is unchanged and applies to the primary
     definition.

**Note added 2026-11-14 for the record.** The paragraph above says only that the parity check
passed. The first run also ran the registered unit test for the added-input path, and that test
failed: `ma_c2/parity.json` records `unit_test_passes: false`, with a largest hidden-activity
difference of about 1.57 against the tolerance of 10⁻⁵. That failure is what exposed the second error
above: the registered stimulation was held at every iteration, while the package's senses enter
only as the settling's starting state. The verdict is unaffected, because the held form, run as
description, gives the same ordering. The amendment's text is otherwise unchanged.
