# Probe 4 (robustness mechanisms): selecting the initial action noise

Run 2026-09-06 from the panel branch (rules recorded in the design at commit `f354580c` before
any result was read) on the rule as merged in PR #316: every arm with `plasticity_homeostasis: true`, both scaling switches on, rate `1e-3`, the MLP arm with `activation: tanh`, and
`initial_log_std ∈ {0, −0.5, −1.0, −1.5}` on the three three-factor arms and both frozen floors.
Seed 101, 600 episodes, 20 runs, 8 min on 16 workers. Diagnostic, not part of the registration.

## Full-clear successes per 100-episode block and plateau tail (last 150), seed 101

| arm | log_std | blocks 1–6 | tail % | saturated | trace scale ρ | norm drift |
|---|---|---|---|---|---|---|
| wt_plastic | 0 | 3 12 20 17 20 21 | 20.0 | 0.000 | 8.1 | 2.7e-3 |
| wt_plastic | −0.5 | 14 30 29 22 22 21 | 22.7 | 0.000 | 7.7 | 2.6e-3 |
| wt_plastic | **−1.0** | **29 43 56 61 63 64** | **64.7** | 0.000 | 7.7 | 2.4e-3 |
| wt_plastic | −1.5 | 39 63 61 29 11 8 | 9.3 | 0.000 | 8.1 | 2.7e-3 |
| rn_plastic | 0 | 6 4 2 4 2 2 | 2.7 | 0.000 | 9.4 | 3.3e-3 |
| rn_plastic | −0.5 | 4 4 2 0 2 4 | 3.3 | 0.000 | 9.3 | 3.3e-3 |
| rn_plastic | **−1.0** | **3 29 44 42 43 39** | **40.7** | 0.000 | 9.5 | 2.4e-3 |
| rn_plastic | −1.5 | 14 14 1 6 8 2 | 4.0 | 0.000 | 9.0 | 3.3e-3 |
| mlp_plastic (tanh) | 0 / −0.5 / −1.0 / −1.5 | all ≈ 0 | 0.7 / 0 / 0 / 0 | 0.000 | 19–24 | 1–2e-3 |
| wt_frozen | 0 / −0.5 / −1.0 / −1.5 | flat | 2.0 / 2.0 / 1.3 / 3.3 | — | — | 0 |
| rn_frozen | 0 / −0.5 / −1.0 / −1.5 | flat | 5.3 / 8.0 / 7.3 / 6.7 | — | — | 0 |

Pooled plateau-tail mean of the three three-factor arms: `0 → 7.8`, `−0.5 → 8.7`, `−1.0 → 35.1`,
`−1.5 → 4.4`. **The registered rule selects `initial_log_std = −1.0`** (action std ≈ 0.37).

## Reading

- **The caps were ours.** With the runaway held (saturation zero at every rate, drift a few
  thousandths per step) and the noise at 0.37 instead of 1.0, the wild-type plastic connectome
  learns cleanly and monotonically to a plateau thirty times its frozen floor within 600 episodes.
  Pilot 2 had the same arm at 6–10% under unit noise and a fifth of its synapses clamped.
- **Noise has a window.** At `−1.5` (std 0.22) both connectome arms rise fast and then collapse
  by episode 400; at `0` and `−0.5` they plateau near 20%. Too little exploration is as
  damaging as too much, and the selected value sits between them.
- **The yardstick still does not learn.** With bounded tanh units the MLP no longer explodes
  (trace scale stays near 20) and no longer dies, and it still clears the task on no run at any
  noise. Under this local three-factor rule with a frozen readout, a dense feedforward MLP does
  not learn this task at 600 episodes; whether it does at the pilot's 3000 is what pilot 3 reads.
  If it does not, D2 test (ii) passes by construction and the logbook must say so (panel design D4).
- **One seed.** The wild-type arm above the rewired arm at `−1.0` (64.7 against 40.7) is the
  direction the primary hypothesis predicts, on one seed, at 600 episodes. It is recorded as such
  and weighs nothing until the panel.

## Pinned into the arm configs (dated amendment in the panel design)

`plasticity_homeostasis: true` and `initial_log_std: -1.0` on all seven arms; `activation: tanh`
on the MLP arm. The rate and the budget remain the pilot's to select.
