# Logbook 131's probes

Each `*.py.txt` is a probe script as it was run, beside the record it wrote. All are description on discarded seeds. They run from the repository root as `build/c2-change-b/<name>.py`, and they import `scripts/analysis/` and each other from there. `probes.py` holds the shared pieces: the procedure brain, both guards (through `forward_observer`), the objective, and the readout every later script uses.

The search states (`*/state.pkl`) are not committed. Each continuation records the step sizes it started from in its JSON (`sigma0`, `stds0`).

## The order, and what each reads

01. **`probes.py B01`** and **`probes.py B02`** ran in parallel, then **`B03`** and **`B04`**, each writing `<run>.json`. B02 starts from Logbook 129's `grade_nudged.json`.
02. **`realworm_predictability.py`** reads the vendored Scheer tracks only.
03. **`curriculum.py`** starts from `B04.json`'s stage-1 answer, and writes `curriculum.json`. It saves its state every generation and resumes from it. Stage 2a's cap was raised from 150 to 300 at generation 33; `curriculum.log.txt` shows the stop and the resume.
04. **`realworm_imitation.py`**, then **`imitation_ceiling.py`**, read the vendored tracks. They ran beside the curriculum, on 4 workers.
05. **`onset_gap.py <state>`** read a copy of the curriculum's state after stage 2b.
06. **`serotonin_route.py <state>`** read copies of the state at stage 3's generation 25 and generation 60. The two records it wrote were renamed `serotonin_route_gen25.json` and `serotonin_route_gen60.json`. **`serotonin_nudge.py <state>`** read the generation-25 copy.
07. **`stage3_kill.py <copy>`** watched stage 3 for its kill rule. `stage3_kill.log.txt` is what it printed.
08. **`readout_snapshot.py <state>`** read the state at stage 3's generation 73, and wrote `stage3_snapshot.json`.
09. **`verify_presence.py`** reads `curriculum.json`.
10. **`hinge.py`** reads `curriculum.json` and a copy of stage 3's state at generation 132, `curriculum/state_stage3_gen132.pkl`, for its step sizes. `hinge_isotropic_aborted.log.txt` is its first launch, which started from a uniform step and was stopped after 2 generations. **`hinge_candidates.py`** and **`hinge_breakdown.py 5`** read `curriculum.json`. Both re-ran after the fact to save their records; their numbers reproduce what was printed during the run.
11. **`noise_measure.py`** reads `hinge.json`. **`hinge2.py`** reads `hinge.json`, `hinge/state.pkl` and `noise_measure.json`. **`robustness_check.py`** reads `hinge2.json`.
12. **`penalty_noise.py`** reads nothing; its means and SDs are typed in from `robustness_check.json`. It reproduces two comparisons first run inline.
13. **`hinge3.py`** reads `hinge2.json`, `hinge2/state.pkl`, `noise_measure.json` and `robustness_check.json`. After it, `robustness_check.py build/c2-change-b/hinge3.json 871000000 build/c2-change-b/robustness_check_hinge3.json` ran, chained in the same command.
14. **`dwelling_windows.py`** reads `hinge3.json`. **`reversal_by_state.py`**, **`reversal_coupling_fit.py`** and **`reversal_coupling_fit_held_out.py`** read the vendored tracks. **`reversal_coupling_probe.py`** reads `hinge3.json` and `robustness_check_hinge3.json`.
15. **`hinge4.py`** reads `hinge3.json`, `hinge3/state.pkl` and the four noise records: `noise_measure.json`, `robustness_check.json`, `robustness_check_hinge3.json` and `reversal_coupling_probe.json`. It patches the evaluator in every worker through `reversal_coupling_probe.patch`.

`probes.py`'s readout began recording every statistic's value at stage 3's generation 73, after the curriculum had started. The curriculum's own readouts record the failing statistics only. Every later script's readouts record them all.
