# Logbook 126's probes

Each `*.py.txt` is a probe script as it was run, beside the record it wrote. They are description on discarded seeds, not a registered procedure.

## Running one again

- **Where the scripts expect to be.** They ran from the repository root as `build/c2-run-mode/<name>.py`, and import each other from there (`staged_bc` imports `stage2_v4`; `imitation_d`, `guarded_wide`, `guarded_stage1` and `full_grade` import `staged_bc`). Create `build/c2-run-mode/` first, then copy every `<name>.py.txt` there as `<name>.py`.
- **Inputs that are committed records.** Two scripts read another probe's record from `build/c2-run-mode/`. Copy it there from this directory before running:
  - `guarded_wide.py` starts from `imitation_d.json`'s `refine.answer`: copy `imitation_d.json` here to `build/c2-run-mode/imitation_d.json`;
  - `full_grade.py` grades `guarded_wide.json`'s `answer`: copy `guarded_wide.json` here to `build/c2-run-mode/guarded_wide.json`.
- **Inputs that are not committed.** Each fit's working state (its `fit0.json`, `stage1_answer.json` or `imitation_answer.json` under `build/c2-run-mode/`) is written by the script that runs that fit. `readout_anatomy.py` and `network_gain.py` read the stopped fits' state, and `staged_bc.py` and `imitation_d.py` resume from their own. Run the upstream script first. The order the probes ran in: `profile_points`, `stage2_v4`, `readout_anatomy`, `bounded_v4`, `network_gain`, `graded_v4`, `staged_bc`, `imitation_d`, `guarded_wide`, `guarded_stage1`, `full_grade`.
- **Trimmed records.** `guarded_wide.json` and `guarded_stage1.json` keep each generation's scores, but only the final and checkpoint answers (their `note` says so).

## The imitation's two runs

- The first run of `imitation_d.py` trained, wrote its answer to `build/c2-run-mode/imitation/imitation_answer.json`, then stopped with the `TypeError` at the end of its log. Profiling assigned a plain tensor to the pool centres, which training had made a trained parameter.
- That log is `imitation_training.log.txt`. It is the `imitation_d.log` that `imitation_d.json`'s `training` field names, and it holds the three rounds' losses and fits to the teacher.
- The first run's script was not kept, so the line numbers the traceback cites do not match the committed script.
- `imitation_d.py.txt` is the corrected script that wrote `imitation_d.json`. Finding the answer file, it skips training and grades, profiles and refines that answer.
- The answer it used is `imitation_d.json`'s `imitation_answer.z`, identical to the file the first run wrote.
