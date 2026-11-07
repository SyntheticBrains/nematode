# Logbook 127's probes

Each `*.py.txt` is a probe script as it was run, beside the record it wrote. They are description on discarded seeds, not a registered procedure.

## Running one again

- **Where the scripts expect to be.** They run from the repository root as `build/c2-run-mode/<name>.py`. They import each other, and Logbook 126's probes, from there: `readability` imports `staged_bc` and `guarded_wide`, and every later script imports `readability`. Create `build/c2-run-mode/` first. Copy every `<name>.py.txt` here, and Logbook 126's `staged_bc`, `stage2_v4` and `guarded_wide` (see [its README](../../126-c2-run-mode/probes/README.md)), there as `<name>.py`.
- **Inputs that are committed records.** `readability.py` starts from Logbook 126's `guarded_wide.json`, which it reads from that logbook's `probes/` directly. The later scripts read the readability probe's answer from `build/c2-run-mode/readability.json`. The committed `readability.json` here is trimmed, but keeps its `answer`, so copy it there. `inhibition_check2.py`, `wiring_scales.py` and `screens.py` also read Logbook 126's `imitation_d.json`, `guarded_wide.json` and `guarded_stage1.json` from that logbook's `probes/`.
- **The order they ran in:** `readability`, `mgl1_diagnosis`, `inhibition_check`, `inhibition_check2`, `wiring_scales`, `serotonin_check`, then `screens.py S1`, `S2` and `S3` in parallel.
- **A trimmed record.** `readability.json` keeps each generation's scores, but only the final and checkpoint answers (its `note` says so).

## Two corrections on the record

- **`mgl1_diagnosis.py` and `wiring_scales.py`.** The figures for the wiring scales and the MGL-1 edges' weights were first computed in a terminal. They were then recomputed by `wiring_scales.py`, which writes `wiring_scales.json`, and `mgl1_diagnosis.py` was re-run so its record holds only what it computes.
- **The screens' first launch** stopped at its generation-0 readout: with presence off, the grader returns no family passes, and the readout's printout read them. The committed `screens.py.txt` computes them from the readings. It was relaunched on the same seeds, and no generation had run.
