## 1. Baselines

- [x] 1.1 `connectome_identity_check.py baseline --build build/c2-encoder-procedure/identity` on the code before any change.

## 2. The receptor-coupling table

- [x] 2.1 `data/neuromodulators/receptor_couplings.json` from the literature pass's draft (`supporting/128-c2-identifiability/probes/receptor_couplings_draft.json`). Check every signed entry's quote against its source, and record `verified` with its date. An entry that cannot be checked is kept with its sign set to null and the reason in its note. PDFR-1's note records that isoform c couples to Gαi/o (Janssen et al. 2008), so the +1 is isoforms a and b's, with Flavell et al. 2013's in vivo cAMP support.
- [x] 2.2 `PROVENANCE.md`: the table's rule, its sources and the check.
- [x] 2.3 `Row.sign_source: str | None = None`, and `load_data_layer(..., receptor_signs=())` setting it with the sign, with its refusals (D5, D6). The receptor-placement null carries both through `dataclasses.replace`.
- [x] 2.4 Tests:
  - an empty list leaves the rows unchanged;
  - PDFR-1 and SER-5 fix their three couples;
  - an unresolved or unknown receptor is refused;
  - every signed table entry is verified;
  - under the receptor-placement null, PDFR-1's moved couples keep sign +1, their own group and the wild type's names.

## 3. The encoder

- [x] 3.1 `ConnectomePPOBrainConfig` gains `encoder_readout`, `encoder_pool_centres`, `encoder_box`, `encoder_inhibition_floor` and `modulator_receptor_signs`. Their refusals go into the validator loop and the constructor (D9), and their classification into `utils/step_duration.py`.
- [x] 3.2 `encoder_entries(..., box=, inhibition_floor=, pool_centres=)`:
  - the graded ranges (D3) and the floor (D4);
  - the centre entries appended after the speed noise's (D7);
  - the table-signed groups under per-ligand tying (D6);
  - threaded through `encoder_parameters`, `box_centre`, `tax4_vector` and `CompactEncoder`.
- [x] 3.3 `_forward_encoded`: the centres subtracted before the readout, and the cap after it (D1, D2), then the optional forward observer (D10). `write_encoder` and `read_encoder` carry the centres.
- [x] 3.4 Tests:
  - off is byte-identical;
  - the cap's bound;
  - zero centres leave the readout unchanged;
  - the graded centre, and the floor at unit 0;
  - an out-of-box floor is refused;
  - the counts (132, 133, 137) and the prefix property;
  - a vector round-trips;
  - the centres are learnable under PPO;
  - the observer is called once per settling, and changes nothing.

## 4. Parity and identity

- [x] 4.1 A parity test between the per-step runner and `BatchedConnectome` with every option on.
- [x] 4.2 `connectome_identity_check.py check --build build/c2-encoder-procedure/identity --out <record>`: every job byte-identical.

## 5. The reproduction check

- [x] 5.1 `c2_encoder_probe.package_vector(probe_z)` (D8), mapping by entry name, with a unit test on a known vector.
- [x] 5.2 The forward parity (a `reproduce` stage): the package brain on the converted answer against the probe's patched brain on the probe's, over the recorded states of one reference episode on the small lawn and one on food removal. Every output within 1e-5, and every step's primitive the same.
- [x] 5.3 The five-population grade of the converted answer (795,000,000 + 200·g + i, with presence), reported beside the probe's as description, and recorded with the identity record.

## 6. Close-out

- [x] 6.1 AGENTS.md, CHANGELOG and the tracker (C.2: the procedure built; Change B next).
- [x] 6.2 The full fast suite, `pre-commit run --all-files`, archive, then the PR.
