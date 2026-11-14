# Reading H0: result

Computed once on 2026-11-14, under `H0_READING.md` (edfafa94). Every number is in `h0/result.json`.

## The named reading

**PASS.** The box-centre network on the true wiring correlates with the measured test matrix at
r_true = 0.0653. That is higher than all 19 degree-preserving rewirings, whose highest is 0.0219.
It is 0.0603 above their median (0.0050), more than five times the minimum effect m0 = 0.0109.

Under the registration's map, the recommendation is the slimmed hybrid:

- H1 is the standing readout, with this degree-preserving null as its floor.
- H2 runs only with the causal bar.
- H3 is gated as before.

## Numbers

| Quantity | Test (primary) | Train (description) |
|---|---|---|
| True wiring, box centre | 0.0653 | 0.0898 |
| 19 nulls, box centre: median | 0.0050 | 0.0003 |
| 19 nulls, box centre: range | -0.0056 to 0.0219 | -0.0183 to 0.0134 |
| True wiring's rank among 20 | 1 | 1 |
| m0 = 0.25 × (r_true - 0.02159) | 0.0109 | |
| r_true minus the null median | 0.0603 | |

The 19 null values against the test matrix, sorted, are:

> -0.0056, -0.0033, -0.0019, -0.0003, 0.0015, 0.0023, 0.0028, 0.0031, 0.0035, 0.0050, 0.0073,
> 0.0084, 0.0087, 0.0096, 0.0104, 0.0140, 0.0163, 0.0207, 0.0219

Description: C.2's fitted answer (`hinge4.json`'s `z`) written onto each wiring.

| Quantity | Test | Train |
|---|---|---|
| On the true wiring | 0.0534 | 0.0725 |
| On the 19 nulls: median | 0.0047 | 0.0049 |
| On the 19 nulls: range | -0.0057 to 0.0194 | -0.0097 to 0.0119 |
| Rank of the true wiring among 20 | 1 | 1 |

The edges moved per draw:

- **Chemical edges:** 3,709 in every draw, of which 89.7% to 91.2% were moved.
- **Gap pairs:** 1,095 in every draw, of which 91.2% to 95.3% were moved.
- **Degrees:** every neuron's chemical in-degree and out-degree is preserved in every draw.

## Checks

1. **Every null brain builds, and each differs from the true wiring.** All 19 built through the
   package's own path (`wiring: rewired_degree_preserving`, `rewire_seed` 9741 to 9759). Each has
   a chemical mask and gap pattern unlike the true wiring's, and the same 137 encoder entry names.
2. **The positive control.** The true wiring at the box centre gives 0.06527048652383569 against
   the test matrix. That is identical to the M-a reading's value (difference 0, tolerance 1e-4).
   This was checked before any null was scored.
3. **Parity on each wiring (an added check, not registered).** The spike-local settling copy
   matches the package's forward on every null wiring and on the true one, at the box centre and at
   C.2's `z`. The settled-activity difference was 0 at M-a's 100 parity inputs.

## Interpretations made

1. **"Differs from the true one."** The check compares the built brain's wiring itself: the
   topology's chemical mask `m_chem` (directed) and the nonzero pattern of its gap matrix `g_gap`
   (unordered pairs). The connectome object is not kept on the brain.
2. **Fraction of edges moved.** This is the share of the true wiring's chemical edges (or gap pairs)
   absent from the null's, by presence only, not weighted by synapse count.
3. **The true wiring.** It was built with `wiring: wild_type` stated explicitly. That is the default,
   and it is the M-a brain, as check 2 confirms.
4. **m0.** It uses the M-a record's unrounded 97.5th percentile, 0.02159225789969175, where the
   registration writes 0.0216. With 0.0216, m0 is 0.010918 rather than 0.010920. The reading is
   the same.
5. **"Exceeds their median by at least m0."** Read as r_true minus the null median being at least m0.
6. **C.2's rank.** This is the rank of C.2's `z` on the true wiring among the 20 values: the true
   wiring and the 19 nulls, each with the same `z` written on it.

## Limits, from the registration

- These are the M-a reading's limits:
  - The network is static: it settles in one step with no slow dynamics.
  - Creamer's arrays come from a fuller stimulation record, probably the OSF exports, which state no
    licence.
- The package's null rewires the chemical and gap graphs separately. It does not preserve the
  overlap of the two graphs, or reciprocity.

## Scripts and outputs (`investigation/spike/h0/`)

- `h0_reading.py.txt`: builds the 20 brains, runs both checks in order, then scores.
- `result.json`: every number, including each seed's value, its edges moved and its parity.
