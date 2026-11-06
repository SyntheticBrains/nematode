# Plan review: how C.2 is fitted (2026-11-06)

*Committed with [Logbook 126](../../126-c2-run-mode.md) as written, mid-way through the probes. Its records are now in `probes/` beside this file, not `build/c2-run-mode/`. Its recommendation (B and C) was tested next and failed. D (the imitation start) then gave a responsive start, and a responsiveness guard with a larger population fitted D_fit from it. Logbook 126 has the whole sequence.*

For the maintainer, to decide C.2's next change. Nothing here is registered; every figure comes from a probe record in `build/c2-run-mode/` or from Logbooks 122 and 123.

## The question

C.2 (the encoded connectome, 196 parameters) must reach benchmark v4's fitted families (small lawn, uniform well, kinematics, activity) by sep-CMA from a registered start, at a cost W can afford across 6 arms × 8–16 draws. No fit has yet produced a worm that runs, reverses and dwells like a real one.

## What has been tried

| Design | Objective | Start | Generations | Score at the end | What it found |
|---|---|---|---|---|---|
| one-stage fits (Logbook 122) | v3 D_fit | box centre | 150–325 | ~10.6 | a stall, off the lawn (0.22–0.26) |
| staged pilot (Logbooks 122, 123) | v3 D_fit | stage 1, then stage 2 | 500 | 3.43 (v3) | holds the lawn by not running: speed and curvature pinned off the squash; fails v4's activity 5 of 5 |
| stage 2, unbounded | v4 D_fit | stage 1's answer | 155 (stopped) | ~9 | curvature pinned, reversals suppressed (0.1%) |
| stage 2, soft cap (±2) | v4 D_fit | stage 1's answer | 81 (stopped) | ~9.4 | outputs parked at the cap; 97% of motor neurons saturated |
| stage 2, cap + centred pools | v4 D_fit | stage 1's answer | 80 (stopped) | ~9.6 | 97% of motor neurons saturated |
| stage 2, graded box + cap + centres + guard | v4 D_fit | stage 1's answer | 200 | 8.6 (min 6.35); the final answer's D_fit 4.41 | responsive and unsaturated; fails 3 of 4 fitted families; kinematics z −7.8 to +10 |

## The diagnosis

1. **Every fit starts open-loop.** At the box centre every sensory gain is 0 (their box is symmetric), so the network gets no input: its pools' SDs are about 1e-4.

2. **Stage 1 never switches the network on.** Its 18 parameters are the three wiring scales, the three food-contact gains, the speed and reversal readouts, and two noise terms, but none of the 33 amphid sensory gains. Its answer holds the lawn through readout biases, with 63% of sensory neurons silent and pool spreads of 0.02.

3. **Stage 2 finds bias-driven optima first, through whichever route is open:**

   - the readout driven off the squash, when it is unbounded;
   - the motor layer saturated, when the readout is capped;
   - wiring scales of 1.26–1.34 (the encoder's graded range is 0.5–0.75) and sensory gains at their ±5 bound, in every fit that switched the network on.

   Closing all three routes (the graded box, the cap and the saturation guard) keeps the network responsive (sense-to-pool gains up to 0.58, against about 0.001 when saturated). But progress is slow: the median score fell from about 9.5 at generation 40 to 8.6 at 200, with the kinematics still far off.

4. **The search may be under-powered.** Diagonal sep-CMA with population 16 in 196–200 dimensions, on an objective with each candidate's 200-animal noise, learns one scale per coordinate and no coupled directions. Wiring scale, sensory gain and readout weight act together.

5. **The cost.** About 25 s per generation. The staged fit that held the lawn took 500 generations (about 4.8 h), the figure W's 232–815 h projection was built on, and no v4 fit is yet near a pass at 200.

## Options

**A. Longer graded fits, unchanged.** Pilot the graded design to 500–1,000 generations (3.5–7 h). It is cheapest to set up. But its curve is shallow, it does not address the start or the search, and W's cost grows with it.

**B. A stage 1 that switches the network on.** Stage 1 fits the subset that carries the senses to the outputs (the wiring scales, all 36 sensory gains, the 25 readout weights and biases, the noise; about 70 parameters), in the graded box with the guard and the cap, on v4's D_fit. Stage 2 then frees the rest. This addresses the diagnosed start problem directly, with fewer dimensions early.

**C. Fewer parameters throughout.** Per-ligand tying already gives 132. Tying further would bring stage 2 to about 70–100 parameters, which sep-CMA handles much better, and every W arm gets cheaper. One way: fix the fitted modulator strengths at the box centre for stage 1 and tie the sensory gains by modality. The cost is that it changes C.2's model: the data layer's strengths would be fitted coarsely or not at all.

**D. An imitation start.** Train the encoded network by gradient descent to reproduce C.1's actions given the same senses, then refine by sep-CMA on D_fit. It is fast, and gives a responsive worm that holds the lawn. But C.2 then starts from C.1's strategy, which entangles the C.2-against-C.1 contrast. Every W arm imitating one teacher is equal treatment, but the question becomes how well each wiring imitates and then refines.

**E. A stronger search in a reduced space.** Full-covariance CMA or a larger population (32–64) on B's or C's subset. It costs more per generation and may escape the degenerate optima.

**F. The roadmap's fallback.** "The connectome cannot be fitted": report the fidelity table (a mechanistic model beats the connectome), and read W on graded distance. W on arms that never fit reads little about wiring.

## Recommendation

**B and C together, piloted once (about 3–4 h):** a stage 1 that switches the network on, in the graded box with the guard and cap, and tying that brings stage 2 to roughly 70–100 parameters. Read it against v4's D_fit with checkpoints. B attacks the start problem the diagnosis found, and C attacks the search's scaling and W's cost. Both keep C.2's fit independent of C.1.

- **If the pilot holds the lawn families** (small lawn and uniform well passing at a checkpoint), the next change registers that procedure, the readout bound and the guard. The slow-noise control then follows on that fit.
- **If not,** D is the next step. If D also fails, F, and a re-scoping of W goes to the phase's plan.

## What W needs from this

A fitted arm that passes D_fit's families at an affordable cost. If no design gets there, W as planned (6 arms × 8–16 draws) is not affordable, and its scope (fewer draws, fewer arms, or W on a reduced model) is a phase decision, not a C.2 one.

## Records to commit with the next change

`build/c2-run-mode/`:

- `profile_points.json`, `readout_anatomy.json`, `network_gain.json`;
- `stage2_v4_stopped.json`, `bounded_cap_stopped.json`, `bounded_centred_stopped.json`, `graded.json`;
- their scripts.

`scripts/analysis/connectome_identity_check.py` now also digests the encoded brain's forward path, with its baseline recorded in `build/c2-run-mode/identity` on main's code.
