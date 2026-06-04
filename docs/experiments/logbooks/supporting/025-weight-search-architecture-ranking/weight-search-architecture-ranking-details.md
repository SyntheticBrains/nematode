# 025 — Supporting Detail: Cross-Architecture C3 Ranking + Quantum Attribution

Detailed per-seed data and the full quantum attribution behind
[Logbook 025](../025-weight-search-architecture-ranking.md). Phase 0 (predator-sensing
canonicalisation) is in [phase-0/](phase-0/README.md). Raw per-seed CSVs:
[phase-4/analysis/ranking-per-seed.csv](phase-4/analysis/ranking-per-seed.csv),
[phase-4/analysis/quantum-attribution-per-seed.csv](phase-4/analysis/quantum-attribution-per-seed.csv).

## Primary metric

`post_convergence_success_rate` — the full-clear (10/10-food) rate averaged over the post-convergence
plateau. `detect_convergence` finds the earliest window where success-rate variance < 0.05 and mean
success > 0.5, then averages success from there to the end. This is the project's convergence metric for
the weight-search comparison (there is no separate fixed-window "L100" field); it is used in place of a
fixed last-N window because the arms have very different warm-up lengths — the spiking and quantum arms
have long from-scratch dead-exploration warm-ups on this lethal cell, so ranking on the converged plateau
is the fair comparison. (Overall `success_rate`, which includes the warm-up, is reported alongside in the
per-seed CSV.)

## Per-seed post-convergence (%) — 7-arch ranking (n=8, seeds 42–49)

| arch | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| equivariantquantum | 88.8 | 88.5 | 86.3 | 85.2 | 87.1 | 82.7 | 88.8 | 80.3 | **86.0 ± 2.9** |
| cfcppo | 89.8 | 87.1 | 86.0 | 91.5 | 89.7 | 60.0 | 83.9 | 87.6 | 84.4 ± 9.5 |
| spikingppo | 83.0 | 83.8 | 84.2 | 84.2 | 83.3 | 85.0 | 86.5 | 83.2 | 84.2 ± 1.1 |
| lstmppo | 85.2 | 84.5 | 73.1 | 81.2 | 85.5 | 82.8 | 88.1 | 88.2 | 83.6 ± 4.5 |
| connectomeppo | 79.2 | 71.8 | 73.5 | 78.8 | 76.8 | 71.4 | 75.6 | 77.2 | 75.6 ± 2.8 |
| mlpppo | 76.6 | 79.4 | 78.4 | 70.2 | 79.4 | 55.3 | 73.1 | 72.4 | 73.1 ± 7.5 |
| feedforwardga | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 ± 0.0 |

Note: mlpppo + connectomeppo seeds 42–45 are reused byte-identical from the n=4 `c3/` location (config
unchanged); 46–49 + all of lstm/cfc/spiking/quantum are fresh `c3-n8/` runs.

## Pairwise (paired-seed one-sided Wilcoxon, 80% bootstrap CI, BH-FDR q) — quantum row

| a vs b | Δ | 80% CI | p | q | verdict |
|---|---|---|---|---|---|
| equivariantquantum vs cfcppo | +1.5 | [−2.2, +5.4] | 0.578 | 0.578 | ns |
| equivariantquantum vs spikingppo | +1.8 | [+0.6, +3.1] | 0.125 | 0.164 | ns |
| equivariantquantum vs lstmppo | +2.4 | [−0.0, +5.2] | 0.098 | 0.137 | ns |
| equivariantquantum vs connectomeppo | +10.4 | [+8.8, +12.2] | 0.004 | 0.007 | \*\*\* |
| equivariantquantum vs mlpppo | +12.9 | [+10.0, +15.7] | 0.004 | 0.007 | \*\*\* |
| equivariantquantum vs feedforwardga | +86.0 | [+84.8, +87.3] | 0.004 | 0.007 | \*\*\* |

Full pairwise matrix (all 21 pairs): [phase-4/analysis/cross_arch_pairwise.json](phase-4/analysis/cross_arch_pairwise.json).

## Quantum attribution (per-seed, n=8)

| control | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| equivariant-quantum (main) | 88.8 | 88.5 | 86.3 | 85.2 | 87.1 | 82.7 | 88.8 | 80.3 | 86.0 ± 2.9 |
| rich classical-equivariant (fair) | 85.4 | 88.5 | 90.1 | 91.0 | 87.9 | 87.8 | 86.8 | 85.5 | **87.9 ± 1.9** |
| rich classical non-equivariant | 88.4 | 83.6 | 84.8 | 83.1 | 88.1 | 84.6 | 89.0 | 89.1 | 86.3 ± 2.4 |
| unstructured-quantum | 82.7 | 85.7 | 83.9 | 86.9 | 85.7 | 87.5 | 76.8 | 79.3 | 83.6 ± 3.6 |
| thin classical-equivariant | 61.9 | 63.8 | 57.6 | 66.6 | 63.0 | 56.3 | 59.2 | 62.8 | 61.4 ± 3.2 |

**Deltas (paired seeds):**

| delta | isolates | value | p | verdict |
|---|---|---|---|---|
| main − rich-classical-equivariant | quantum circuit (FAIR) | −1.9 | 0.926 | ns — **no quantum advantage** |
| main − thin-classical-equivariant | quantum circuit (naive) | +24.6 | 0.004 | \*\*\* — **artifact** (thin control is sub-MLP) |
| rich-equivariant − rich-non-equivariant | symmetry prior (classical, matched) | +1.5 | 0.191 | ns |
| equivariant-quantum − unstructured-quantum | symmetry prior (quantum, matched) | +2.4 | 0.125 | ns |

**Why the thin control was an artifact.** The original classical-equivariant ablation's left/right (odd)
output was `tanh(linear(odd-latents-only))` — a single scalar fed only by the odd latents, which starves
the klinotaxis left/right decisions that dominate this task. It scored 61.4%, *below* even plain MLP-PPO
(73.1%). The fair control replaces it with a Z₂-symmetrised full MLP (`even=½(f(x)+f(Rx))`,
`odd=½(g(x)−g(Rx))`, exactly equivariant, ~9.9k params) whose odd path depends richly on the full latent;
it scores 87.9%, matching the quantum. The matched-capacity non-equivariant control (same 9.9k-param MLP,
symmetrisation removed) scores 86.3%, isolating the symmetry as ns.

## Quantum-arm convergence trajectory (single seed, from-scratch C3)

Per-500-ep full-clear rate, illustrating the long warm-up then ignition (why post-convergence, not a
fixed window, is the fair metric):

| ep | 1–500 | 501–1000 | 1001–1500 | 1501–2000 | 2001–2500 | 2501–3000 |
|---|---|---|---|---|---|---|
| clears | 0.0% | 7.2% | 65.6% | 83.0% | 89.8% | 91.8% |
| foods | 0.37 | 3.39 | 8.50 | 9.34 | 9.64 | 9.69 |

C2 (food+predator, no thermal) by contrast reaches 99.8% by ~ep 1500 — the lethal thermal gradient is
the sole source of the C3 warm-up.

## Quantum arm — key hyperparameters

| field | value |
|---|---|
| num_qubits / k_odd / num_layers | 8 / 3 / 3 |
| entangler set | RX (all), RZ (even), IsingXX (any pair), IsingZZ (same-parity) |
| readout | even Pauli → FORWARD/STAY; even±odd → LEFT/RIGHT |
| differentiation | in-repo torch statevector simulator (backprop, no parameter-shift) |
| actor_lr / critic_lr | 3e-4 / 3e-4 |
| entropy | 0.08 → 0.02 over 2000 ep (sustained exploration for the warm-up) |
| rollout / epochs / clip / γ / λ | 512 / 4 / 0.2 / 0.99 / 0.95 |
| budget | 4000 ep (plateau by ~ep 2500) |
