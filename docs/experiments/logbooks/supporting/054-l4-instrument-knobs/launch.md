# I.3 — the unexamined knobs: the registered protocol

Registered in `openspec/changes/add-l4-instrument-knobs`, reviewed and committed **before** the run.
Nothing below was chosen after a result existed.

## The question

Three settings have been pinned since A.3 and never examined: the eligibility horizon
(`trace_decay 0.9`), the homeostatic norm sphere, and the exploration noise (`initial_log_std −1.0`,
std ≈ 0.37). I.3 registered them as "each examined on the yardstick under I.1's rule".

**That platform does not work**, and the change records the pilot that says so: on disjoint seeds
101–102 at 3000 episodes the MLP yardstick under the node-perturbation rule ends at 0.00% full
clears and 0.11 / 0.00 mean foods, **below its own frozen control** at 0.90 / 0.35. Two seeds is a
pilot, not a result. It agrees with Logbook 040 under the original rule.

**And the horizon is unmeasurable on the undelayed control**: `trace_decay` at 0.0, 0.9 and 0.99
gives results identical to four decimals, because the control resets the trace every trial — the
confound it was built to remove.

## The platforms

| knob | platform |
|---|---|
| homeostatic norm sphere | the committed control |
| exploration noise | the committed control |
| eligibility horizon | the committed control **with the reward delayed** |

The delay shows the cue, takes the scored action, runs `D` steps against a constant filler
observation, then delivers the reward once. **The bounds do not move** — they depend on the targets
and the exploration noise alone — so every cell is scored by the control's own registered pass rule,
and `D = 0` reproduces the committed arm. Verified before the run: the `σ = 0.2` arm's mean is
bit-identical to the committed `−0.19623935494991745`.

**The delay measures dilution, not decay.** The credited step is one term among `D + 1` when the
modulator arrives, and the recipe's trace normalisation rescales the sum while leaving that share
alone. A filler that added nothing would be a pure scalar decay and would be divided straight out —
checked against the rule's running scale, a zero-filler delay reads 1.000 at `D = 20`, identical to
`D = 0`. The filler is therefore the uniform vector over the cue channels: nonzero so it drives the
plastic layer, constant so it leaks no cue, and the same width so `D = 0` stays the committed arm.

## The grids

| grid | values | baseline |
|---|---|---|
| homeostasis | on, off | on |
| exploration noise | 0.22, 0.37, 0.61, 1.0 | 0.37 |
| horizon | `trace_decay` ∈ {0.9, 0.99, 0.999} × `D` ∈ {0, 2, 5, 10, 20} | 0.9 at `D = 0` |

The noise grid is the panels' history: std 1.0 capped every plastic arm near its floor, 0.22 rose
and collapsed, 0.37 was selected by a probe. The delays bracket the ten-step scale `0.9` implies
(0.9¹⁰ = 0.35, 0.9²⁰ = 0.12) and run past it.

Arm: `node_perturbation` at σ = 0.2 — the eligibility that passed the control, at the scale that
passed it. Seeds 1–8, 20,000 trials, the registered pass rule (≥ 7 of 8 seeds above floor **and** a
mean at or above half the floor-to-optimum gap).

## What each outcome licenses — fixed before the run

- **A knob passes where the baseline fails, or fails where the baseline passes.** That setting was
  holding the rule back or propping it up, and I.4 re-reads the panels that pinned it in that light.
- **The horizon degrades with `D`.** The rule's one-step success and its multi-step failures have a
  mechanical explanation, and the horizon becomes the next registered target.
- **The horizon does not degrade with `D`.** The trace is not what separates them; the suspicion
  closes, which is worth as much as confirming it.
- **Nothing moves.** The three settings are exonerated on the platform where the rule works, and
  I.4 records that the failures are not attributable to them.

## Reproduce

```bash
uv run python scripts/analysis/l4_instrument_knobs.py \
  --out docs/experiments/logbooks/supporting/054-l4-instrument-knobs/knobs.json \
  --csv docs/experiments/logbooks/supporting/054-l4-instrument-knobs/per-cell.csv
```
