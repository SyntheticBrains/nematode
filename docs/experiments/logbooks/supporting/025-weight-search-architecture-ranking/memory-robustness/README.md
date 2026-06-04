# Memory-robustness arc — supporting configs + results

Reproducibility artifacts for the **`## Limitations`** section of
[Logbook 025](../../025-weight-search-architecture-ranking.md). The arc tests whether the four-way
architecture tie reflects genuine architectural equivalence or a reactive-task regime that under-tests
memory. Three experiments, MLP vs LSTM/GRU, n=4 (seeds 42–45), `--theme headless --track-experiment`.

Primary metric for the sparse cells is **`avg_foods_collected`** (a graded search-efficiency measure;
all-clear success is too rare/noisy on a hard sparse env). Near-converged read = **last-500-episode mean**.

## EXP 1 — large-grid dense (does grid size flip the ranking?)

Configs: [`configs/scenarios/foraging_predator_thermal/mlpppo_large_combined_klinotaxis.yml`](../../../../../configs/scenarios/foraging_predator_thermal/mlpppo_large_combined_klinotaxis.yml)

- the `lstmppo_` sibling (committed under `configs/scenarios/`, not here — they are reusable combined cells).
  Env: 100×100, 4 pursuit predators, spot-based thermal, 45 foods / 25 target, max_steps 2000, klinotaxis +
  new two-channel predator biology. 2000 episodes.

```bash
uv run ./scripts/run_simulation.py --runs 2000 --theme headless --track-experiment --seed <42..45> \
  --config configs/scenarios/foraging_predator_thermal/{mlpppo,lstmppo}_large_combined_klinotaxis.yml
```

| arch | post-conv full-clear |
|---|---|
| MLP | 81.8 ± 2.6 |
| LSTM | 84.9 ± 1.1 |

Gap **narrows** to 3.1pp (small grid was 10.5pp); the memoryless arm becomes *more* competitive. Grid size
is not the lever. (`cfcppo` + `connectomeppo` large configs exist but were not run — batch 2 deprioritised.)

## EXP 2 — sparse depleting search (the deliberate memory-stressor)

Configs (here): `mlpppo_sparse_g100_d3.yml`, `lstmppo_sparse_g100_d3.yml`.
Env: 100×100, **6 foods, `no_respawn: true`, collect-all**, foraging-only, `reward_exploration: 0`
(food is the only reward), short-range gradient (`gradient_decay_constant: 3.0`). 2000 episodes.

```bash
uv run ./scripts/run_simulation.py --runs 2000 --theme headless --track-experiment --seed <42..45> \
  --config docs/experiments/logbooks/supporting/025-weight-search-architecture-ranking/memory-robustness/{mlpppo,lstmppo}_sparse_g100_d3.yml
```

| arch | foods/6 (full-run) | foods/6 (last-500ep) |
|---|---|---|
| MLP | 4.36 ± 0.22 | 4.26 ± 0.44 |
| LSTM | 3.71 ± 0.09 | 4.06 ± 0.04 |

No memory advantage: the arms converge to ~the same level; the GRU is just slower (it is still rising at
ep 2000 — trajectory 3.38 → 3.34 → 3.66 → 4.12 per 500-ep window — while the MLP is near-flat 4.35 → 4.58).

**Goldilocks tuning** (300-ep MLP probes that picked `100/decay 3.0`): `60/d4` too easy (83% all-clear);
`60/d2` 4.92/6; **`100/d3` 3.94/6 (chosen — sub-optimal + learnable)**; `100/d2` 2.91/6.

## EXP 3 — STAM memory-depth ablation (the cleanest evidence)

Klinotaxis *requires* STAM (auto-enabled; `dC/dt` computed from its buffer — `config_loader.py:891`), so a
zero-memory brain is not testable; instead vary `stam_buffer_size` on the **same MLP brain**. Configs (here):
`mlpppo_sparse_g100_d3_stam2.yml`, `_stam8.yml`, and the buffer-30 base is `mlpppo_sparse_g100_d3.yml`.
1500 episodes.

| STAM buffer | foods/6 (last-500ep) |
|---|---|
| 2 (one-step derivative) | 4.36 ± 0.25 |
| 8 | 4.25 ± 0.24 |
| 30 (default) | 4.26 ± 0.44 |

Flat — memory depth **beyond a single step** is irrelevant (same brain, only the memory knob changed → no
tuning confound). This shows depth beyond one step adds nothing, **not** that memory is irrelevant (buffer-2
still carries the one-step derivative; a true zero-memory baseline would require non-biological oracle sensing).

## Conclusion

Across grid size, food sparsity, and memory depth, architecture does not affect performance on these
klinotaxis-foraging tasks — they are solvable with the local gradient + a one-step temporal derivative,
shared by every arm via STAM. The broader "*C. elegans* has no within-episode working-memory niche" reading
is left as an **open hypothesis** (untested counterexamples) in the logbook, not a conclusion. See
[Logbook 025 § Limitations](../../025-weight-search-architecture-ranking.md#limitations) for the full framing.
