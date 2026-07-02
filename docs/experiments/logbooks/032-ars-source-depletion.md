# 032: Area-Restricted Search via Source-Depletion — the Biological Twin of the Bit-Memory Control (T7)

**Status**: completed — **null with a mechanism**. Environmental source-depletion does **not** induce a
within-episode-memory demand that separates the memory arms from the memoryless MLP at current
fidelity. The demand it does create is **short-horizon** (covered by the biological adaptive
chemosensor, or supplantable by a small attention window) and does **not** scale to an episode-scale
memory requirement — the strongly-long-horizon regime is an unlearnable blind search. The result is
biologically consistent: real ARS is driven by current food sensation + slow neuromodulatory state,
not a remembered map.

**Branch**: `openspec/add-ars-depletion`.

**Date**: 2026-07-02.

**OpenSpec change**: `add-ars-depletion` (source-depletion mechanism + cells; committed §1–§6, green).

## Objective

The bit-memory positive control ([Logbook 030](030-bit-memory-positive-control.md)) proved the
architecture comparison **can** separate working memory in principle (CfC / Transformer / LSTM /
minGRU / minLSTM ≫ MLP on a delayed-match-to-cue). ARS-via-depletion is its **biological twin** — the
source-dynamics half of the deferred `T6.gradients.2` stretch. Question: does an *environmental*
within-episode-memory demand (food depletes as it is eaten → the static Fick field changes in-episode
→ area-restricted search) reproduce that separation on a biologically-valid foraging task?

## Background

Logbook 025 / [029](029-continuous-architecture-ranking.md): on the reactive foraging cell,
architecture does not discriminate — a memoryless MLP ties or wins because a one-step temporal
derivative makes the gradient locally readable. The standing hypothesis (Logbook 025 § Limitations)
was that a within-episode-memory demand would separate the arms, and depletion-driven ARS was the
biologically-valid route to create one. This experiment builds + evaluates that mechanism.

## Hypothesis

Depletion flattens a food patch **in place** as it is grazed; with `no_respawn` the patches are
finite. After exhausting a patch the worm stands in a gradient-poor region and must relocate by
memory/search rather than hill-climbing — a within-episode-memory demand on which the recurrent /
attention arms should separate from the MLP.

## Method

### Mechanism (committed §1–§6, 11 tests, byte-identical when off)

Per-source remaining-amount store (`food_amounts`, index-aligned via `_add_food`/`_remove_food`,
carried by `copy()`); the food-field contribution scales with the amount (`_food_field_magnitude`);
feeding decrements the matched source by index via a shared `_deplete_or_remove` (deplete-in-place,
remove + respawn at exhaustion subject to `no_respawn`); off by default = byte-identical (216 env
tests unchanged). Distance-shaping shown perverse under depletion (exhausting a patch spikes
nearest-food distance) ⇒ `reward_distance_scale: 0` in the ARS cell.

### Calibration (before any panel)

- **Field geometry `gradient_decay_constant` L=8→4.** The recipe's L=8 mm merges the 5 patches into a
  single broad blob (peak/valley ~2.3) — a depleted patch hands straight off to neighbours (purely
  reactive). L=4 isolates the patches (peak/valley ~270) so depleting one opens a genuine low-field
  valley.
- **Target is the demand lever.** At the recipe target (10 of ~20 collectable feeds) the *untrained*
  MLP solves the cell (10/10) — no demand. Near-total targets (14–18) force patch exhaustion + the
  stranded endgame.
- **Entropy 0.2 stabilises the memoryless baseline** (0.05/0.1 collapse; lr-decay/larger-buffer
  backfire). The MLP is intrinsically oscillatory on this task at every target (sd ~14).

### Panels (continuous-2D, klinotaxis, plateau-tail full-clear success)

- **Full-sensing panel** — 5 arms × {ARS cell, `no_respawn` control} × 6 seeds × 1500 runs, matched
  entropy 0.2, L4, target 14.
- **Sensory-memory ablation** — the same ARS cell with the stateful sensory front-end stripped. Note
  the dC/dt derivative is computed inside the STAM block, so `adaptive_chemosensor_enabled: false` +
  `stam_enabled: false` yields a purely **instantaneous** (spatial klinotaxis lateral gradient only)
  observation with **no code change** — the architecture becomes the only within-episode memory. MLP
  vs Transformer × 4 seeds, forming a 2×2 (aid × architecture).
- **Episode-scale test** — instantaneous sensing on progressively larger/sparser arenas (learnability
  sweep), then a memory panel on the largest *learnable* cell (`es_b`: 30 mm, 8 patches).

## Results

### Full biological sensing → NULL (6 seeds, 1500 runs)

| arm | ARS-cell plateau-tail | sd | control cell | marginal (ARS−ctrl) |
|-----|-----|-----|-----|-----|
| **MLP** | **41%** | 21.7 | 100% | −59 |
| Transformer | 33% | 17.9 | 96% | −62 |
| CfC | 20% | 16.3 | 99% | −79 |
| minGRU | 8% | 7.2 | 99% | −91 |
| minLSTM | 6% | 7.4 | 99% | −93 |

The memoryless MLP is the **best** arm; every memory arm underperforms it, and the recurrent arms
collapse hardest. The `no_respawn` control is ~100% for all arms (trivially solvable → non-stationary
food alone is not a memory demand). An earlier 800-run read showed a Transformer "separation" (39% vs
MLP 20%) — an **MLP-undertraining artefact**: MLP 20%→41% from 800→1500 runs while the Transformer
stayed flat.

### Strip the sensory aid (instantaneous, 20 mm) → a SHORT-HORIZON separation appears

2×2 (aid × architecture), ARS cell, seeds 1–4:

| condition | MLP | Transformer |
|-----------|-----|-------------|
| aids **ON** (adaptive + STAM) | 49% | 30% |
| aids **OFF** (instantaneous) | 20% | 36% |

Stripping the adaptive aid **crashes the MLP (49→20)** but the Transformer is **robust (30→36)** and
wins on 3/4 seeds. The asymmetry is the proof: were the adaptive sensor merely needed for chemotaxis,
the Transformer would crash too — it self-supplies the temporal integration the MLP was reading off
the sensor. So the full-sensing null is partly a **sensory-aid artefact**; the demand it masks is
real but short-horizon (~10-step sensor ↔ ~16-step attention window).

### Scale the horizon up → the separation VANISHES; strongly episode-scale is unlearnable

Learnability sweep (MLP, instantaneous, 150 runs): large/sparse cells floor (w45/f6 → 0%; w40/f8 →
5%); only the findable/dense `es_b` (w30/f8/d5) learns (54%). Memory panel on `es_b` (4 seeds, 1200):

| cell (instantaneous) | MLP | Transformer | minGRU | minLSTM | Transformer Δ |
|-----|-----|-----|-----|-----|-----|
| 20 mm ablation (target 14) | 20% | 36% | — | — | **+16** |
| 30 mm `es_b` (target 16) | 25% | 16% | 2% | 5% | **−9** |

The Transformer's short-horizon advantage **does not grow with arena size — it vanishes**. Making the
cell more episode-scale turns it into a harder blind search that destabilises the memory arms, not a
long-horizon-memory task they exploit.

## Analysis

A clean three-step story: (1) with full biological sensing the task is reactively solvable and the
adaptive chemosensor (a ~10-step leaky integrator, `B_t=(1−α)B_{t−1}+αC_t`) supplies the temporal
integration → null; (2) strip that aid and a short-horizon demand appears that an attention window
fills → the Transformer separates at 20 mm; (3) push the horizon up and the demand does not scale —
findability requires spatial-gradient coverage, so removing the signal to force long-horizon memory
produces unlearnable blind search (the **findability-vs-horizon wall**).

**This is biologically consistent, not a sim artefact.** Real *C. elegans* ARS (the dwelling→roaming
switch) is driven by *current* food sensation plus slow internal / neuromodulatory state (serotonin,
dopamine, TGF-β, satiety) — **not** a remembered allocentric map of visited patches. The worm is a
largely reactive/reflexive animal whose genuine "memories" are slow adaptation/integration
(leaky-integrator-like — which the adaptive sensor already models), associative plasticity, and slow
reference set-points, not delay-bridging working memory. So the ARS null is the *expected* biological
answer: environmental depletion does not demand episode-scale working memory of the worm.

## Conclusions

- **Null for the ARS separation hypothesis at current fidelity.** The biological twin does **not**
  reproduce the artificial bit-memory control's working-memory separation.
- The depletion demand is **short-horizon** — masked by the biological adaptive sensor, supplantable
  by a small attention window, and it does not scale to an episode-scale memory requirement.
- The result is **biologically correct**: real ARS is sensory + slow-neuromodulatory-state, not
  spatial working memory; *C. elegans* foraging is reactive-dominated.
- The **source-depletion mechanism is committed, tested, and byte-identical-when-off** — a reusable
  env capability independent of this cell's null (available for phase-7 spatiotemporal work).

## Limitations

- Continuous-2D **static-Fick amplitude** depletion (the field amplitude drops); the full
  `∂C/∂t = D∇²C` PDE (traveling depletion fronts → real spatiotemporal integration) is phase-7.
- **Recurrent-PPO training instability** is a persistent confound — LSTM/minGRU/minLSTM collapse
  repeatedly, so the Transformer is currently the only reliable memory detector; "memory doesn't
  help" is partly entangled with "recurrent arms don't train stably here."
- The 30 mm MLP plateau is seed-noisy (one high seed); some reads are 4-seed.
- Evaluation used investigation-scratchpad per-arm configs (calibration, not a shipped panel); the
  committed cells are the `mlpppo` reference ARS + control.

## Next Steps

- [ ] **Chemosensory associative-memory DMTS** — the next T7 naturalistic memory probe and the more
  faithful "remember-and-use" biological analogue of bit-memory (real gustatory / odor↔food
  plasticity). See [docs/research/associative-memory-probe.md](../../research/associative-memory-probe.md);
  tracked as `T7.separation.associative_memory`.
- [ ] Thermal reference memory (cultivation-temperature DMTS) as a backup naturalistic probe.
- [ ] **Faithful slow-forming memory** (memory that forms over hours of cultivation / repeated trials
  / across episodes, with neuromodulatory dynamics) is **phase-7** — recorded in
  `docs/roadmap.md` § Known Gaps + Phase 7 (it ties to the L4 plasticity/neuromodulation deliverable).

## Data References

- Mechanism + cells: `packages/quantum-nematode/quantumnematode/env/env.py` (source-depletion),
  `configs/scenarios/foraging/mlpppo_small_continuous2d_fick_adaptive_klinotaxis_ars_depletion.yml`
  (+ `_no_respawn_control.yml`), tests
  `packages/quantum-nematode/tests/quantumnematode_tests/env/test_source_depletion.py`.
- Eval regime: L4, `reward_distance_scale 0`, entropy 0.2, targets 14/16, `no_respawn`; instantaneous
  sensing = `adaptive_chemosensor_enabled: false` + `stam_enabled: false`.
