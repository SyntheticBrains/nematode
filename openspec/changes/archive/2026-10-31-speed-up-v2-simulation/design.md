## Context

A v2 simulation (`cma_recovery_v2.simulate_v2`) runs `BatchedEvaluator` on four assays, converts each animal to `BehaviourStep` records, passes Scheer's families through the measurement model, and grades. The profile of one simulation (26 s, one core): the evaluator 12.8 s (C.1's uniforms 3.2 s of it), the conversion 5.1 s, the measurement model 6.3 s (`NoiseLibrary.draw` and `_splice` 3.1 s, rebuilding the steps 2.0 s), grading 1.8 s.

The earlier 1.76× speed-up (Logbook 116) set the bar: identical output, shown on a discarded population.

## Goals / Non-Goals

**Goals:** about 1.6–1.8× per simulation, with every statistic and D byte-identical to today's.

**Non-Goals:**

- the simulation kernels and whole-generation batching (the compute plan, at C.2);
- the other controllers' draws (`BatchedStateHazards` mixes uniforms and normals from one generator, and the MLP and connectome draw normals), which keep their current path;
- grading's 1.8 s.

## Decisions

1. **The conversion.**
   - Each column of the animal's row is read once with `.tolist()`, which yields the same Python floats as `float(array[i, t])`.
   - The signed speed is computed exactly as now, in a loop over those floats with `math.cos` and `math.sin`, since numpy's vectorised `cos` need not match libm to the last bit.
   - Steps are built positionally, as `measure` already does.
2. **The measurement model.**
   - `measure` reads `x`, `y` and the other init fields as columns from the input steps, adds the noise as an array, and builds each step once.
   - A lost second's NaN positions are set exactly as now.
   - `_splice` keeps its `rng.integers(..., size=SPLICE_CANDIDATES)` draw and its `argmin` over the same (candidate, offset) grid in the same row-major order, so the first minimum wins as now. Only per-call overhead (repeated `np.arange`, temporary masks) is reduced, and only where the arithmetic is unchanged.
3. **C.1's uniforms.**
   - `BatchedMechanistic` keeps a per-animal buffer of uniforms, refilled by one `torch.rand(3·B, dtype=float64)` per animal when empty, with B = 4,096 steps, and consumed three per step in order.
   - The buffer is tied to the identity of the generator list it was filled from. A new list starts a new buffer, and `reset` does not discard it, so no draw is skipped or repeated.
   - Draws left in the buffer at the end of a run advance the generators past what was used. Nothing reads them afterwards: `_build_animals` builds fresh generators every run, and the controller is their only consumer.
   - Only `BatchedMechanistic` changes. It is the only consumer of its torch generators, and it draws only uniforms. Its docstring says the numbers drawn are the same as one call per second, now taken in blocks.
4. **The identity checks.**
   - **In the suite:** each new path is compared, field by field and exactly, with a reference copy of the old code kept in the test, on a short evaluator run (Scheer's small lawn and Chen's chemotaxis), on measured tracks with splices and lost seconds, and on uniform draws across a run boundary. The torch property (block equals sequence, and the same generator state after) is its own test.
   - **Once, before the change is merged**, `scripts/analysis/speedup_identity_check.py`, in two stages:
     - `baseline`, run on `main`'s code before any edit, writes `build/v2-speedup/baseline.jsonl`. It holds v2 simulations without presence and with presence, and v1 simulations (`npe_recovery.simulate`), all at the box centre and at two F.0a parameter vectors, on discarded seeds (base 99,700,000).
     - `check`, run after the change, re-simulates the baseline's jobs and these stored records, and compares every statistic and D exactly:
       - attempt 2's probe answers (`build/cma-recovery-v2/attempt2/probe/answers.jsonl`);
       - three observations graded with presence from `build/cma-recovery-v2/candidates.jsonl`, with their presence readings;
       - three of F.0a's stored v1 simulations (`npe_recovery.load_simulations`).
     - Its summary is committed as `supporting/118-v2-speedup/identity.json`.

## Risks / Trade-offs

- **[A torch upgrade changes block draws]** → the property test fails in CI, and the block draw is reverted to one draw per step before any further simulation.
- **[A floating-point reordering slips into the measurement model]** → the exact field-by-field tests and the stored-record check catch it before merge.
- **[Gains smaller than expected, or less on 16 workers]** → re-timed with the same bench on 16 workers in High Power mode, reported as measured.
