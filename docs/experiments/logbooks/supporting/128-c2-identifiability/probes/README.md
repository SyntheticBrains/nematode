# Logbook 128's probes

Each `*.py.txt` is a probe script as it was run, beside the record it wrote. They are description on discarded seeds. They run from the repository root as `build/c2-run-mode/<name>.py`, alongside Logbooks 126's and 127's probes (see [127's README](../../127-c2-readability/probes/README.md)).

## The order, and the inputs to copy into `build/c2-run-mode/`

1. **`pdf_sign_check`** reads `readability.json`: Logbook 127's committed record.
2. **`prototypes V1` and `V2`** ran in parallel. They read Logbook 126's `imitation_d.json` from that logbook's `probes/` directly, and write `prototypes/V1.json` and `V2.json`, committed here as `prototypes_V1.json` and `prototypes_V2.json`.
3. **`leave_one_out`** starts from `prototypes/V2.json`, and writes `leave_one_out.json`.
4. **`tph1_capacity`** reads `leave_one_out.json`.
5. **`fit_mutants`** starts from `tph1_capacity`'s point, and writes `fit_mutants.json`.

`receptor_couplings_draft.json` is the literature pass's output. It is a draft: two of its quotes have been checked against their sources.

## On the record

- **V1 and V2 were stopped by hand,** on the maintainer's word: V2 at generation 91 and V1 at 98. Their records end at their generation-90 readouts, with `stopped` null.
