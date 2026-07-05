# Design — connectome-structure controls

## Context

The connectome brain (`packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py`) loads
the Cook-2019 hermaphrodite wiring into an in-memory `Connectome` object
(`connectome/model.py`: `neurons`, `chemical_synapses` directed, `gap_junctions` undirected) via
`load_cook_2019_hermaphrodite()`. That object is the **sole** source of adjacency: `ConnectomeTopology`
reads only `connectome.chemical_synapses`, `connectome.gap_junctions`, and `sorted(connectome.neurons)`
to build the boolean strict-mask `m_chem`, the learnable `w_chem` (init `N(0, 1/√fan_in)` on edges,
zero elsewhere), and the fixed gap-junction buffer `g_gap` (normalised by `√(dᵢdⱼ)`). Chemical-synapse
*weights* from Cook are used only for presence — the trainable weight init depends on the **degree**
(fan-in), not the EM counts. Gap junctions are fixed (a `register_buffer`, excluded from
`learnable_parameters`).

## Goals

1. A **degree-preserving rewired-null** connectome that keeps every neuron's in/out degree (chemical)
   and degree (gap) exactly, scrambling only which neurons connect, trained under the **identical**
   PPO recipe + substrate as the wild-type `T7.connectome.c3_integrated` cell.
2. Isolate the topology effect: wild-type vs rewired-null differ in **nothing** but the wiring.
3. Zero behaviour change to existing runs (byte-identical when `wiring: wild_type`).

(The learnable-gap-junction control is a tracked fast-follow, out of scope here — see § Deferred.)

## Decisions

### D1 — Config options on the existing brain, not a new brain class

Add fields to `ConnectomePPOBrainConfig`, apply the transform at the load→topology seam
(connectome_ppo.py, between `connectome = load_cook_2019_hermaphrodite()` and
`ConnectomeTopology(connectome, …)`):

```python
wiring: Literal["wild_type", "rewired_degree_preserving"] = "wild_type"
rewire_seed: int | None = None          # None -> self.seed (= ensure_seed(config.seed)); per-run draw
```

(The brain already resolves `self.seed = ensure_seed(config.seed)` at construction, so `rewire_seed`
needs no new seed plumbing — it defaults to that per-run seed.)

Rationale: the name→config-class map is registry-derived (`_build_brain_config_map`), so config fields
need **zero** loader / factory / enum / `__init__` changes — contrast the 5-file new-brain path. It
guarantees "everything else identical" **by construction** (same class, same `ConnectomeTopology`,
same dims/projections/critic — only `connectome.chemical_synapses` / `gap_junctions` are swapped). It
matches the existing precedent of structural config toggles on this brain (`chemical_mask_mode`,
`enable_gap_junctions`, `freeze_updates`).

### D2 — Hand-rolled degree-preserving double-edge-swap (no new dependency)

`networkx` is not a dependency and would be a heavy add for one function; the repo already threads a
seeded `np.random.Generator`. New `connectome/rewiring.py`:

- **Chemical (directed):** repeated **directed double-edge-swap** — pick two directed edges
  `(a→b), (c→d)`, rewire to `(a→d), (c→b)`; reject if it creates a self-loop or a duplicate edge.
  This preserves every neuron's **out-degree and in-degree exactly** (a's out, d's/b's in, …). Run
  `≈ swaps_per_edge × |E|` accepted swaps (default `swaps_per_edge = 10`) for mixing.
- **Gap junctions (undirected):** undirected double-edge-swap on canonical `(a<b)` pairs — pick
  `{a,b}, {c,d}`, rewire to `{a,d}, {c,b}` (re-canonicalise), reject self-loops / duplicates. Preserves
  each neuron's gap **degree** exactly.
- Both graphs rewired **independently**, from a **dedicated `np.random.default_rng(rewire_seed)` that
  is separate from the weight-init RNG** (`self.rng` / the global torch seed set at brain construction).
  The rewiring runs on the `Connectome` **before** `ConnectomeTopology` is built, so the topology's
  `w_chem` init draws land at the **same RNG state as the wild-type run** for the same seed — the
  pairing then isolates *topology*, not an init-RNG offset (this is what "matched initialisation"
  means: identical random draws applied to a rewired mask). The neuron set and ordering are untouched
  (so index stability at `sorted(connectome.neurons)` holds, and per-post fan-in — hence `w_chem` init
  scale and `g_gap` normalisation — is preserved).
- Output is a **new `Connectome`** (same `neurons`, rewired `chemical_synapses` / `gap_junctions`,
  `source` annotated e.g. `cook_2019_hermaphrodite+rewired(seed=…)`). Edge weights are carried with the
  edges (irrelevant to training beyond presence, but kept for provenance).

*Alternative considered:* add `networkx` and use `directed_configuration_model` / `double_edge_swap`.
Rejected — a new dependency for one seeded routine we can write in ~40 lines, and networkx's
`double_edge_swap` is undirected-only (the directed case would still be hand-rolled).

### D3 — Per-run rewiring draw + paired-seed comparison

`rewire_seed = None` derives the rewiring seed from the run's global seed, so each of the n≥8 runs
draws its **own** degree-preserving rewiring **and** its own PPO weight init. This tests the wild-type
against the **null distribution** of degree-matched graphs, not one arbitrary rewiring (the robustness
Dhiman's critique demands). The comparison is **paired by seed** (same PPO init seed, wild-type vs
rewired topology), analysed with the committed paired-seed Wilcoxon + 80% bootstrap CI + BH-FDR layer
(`weight_search_architecture_ranking`), on the same **C3 plateau-tail ranked-success** metric as the
029 ranking (reuse `t7_continuous_ranking` / the level-agnostic ranked metric — no new metric).

Verdict rule (pre-registered): **wild-type significantly > rewired (BH-FDR q\<0.05, positive delta)** →
"specific wiring matters"; **indistinguishable (CI spans 0)** → "degree statistics, not wiring".

### D4 — Configs, analysis, and the byte-identical invariant

- **Config:** copy `connectomeppo_small_continuous2d_combined_klinotaxis.yml` verbatim into a
  `…_rewired_null.yml` (add `wiring: rewired_degree_preserving`). No other field changes — the recipe
  match is the whole point. Both arms (wild-type + rewired) are re-run in one fresh panel for an exact
  seed-paired comparison at the 029 connectome budget.
- **Analysis:** `scripts/analysis/connectome_structure_controls.py` mirrors the associative/bit-memory
  separation harnesses — load wild-type + rewired run `.out`s by a manifest, compute the C3 ranked
  metric per seed, print the paired deltas + BH-FDR + the verdict, write a summary JSON.
- **Byte-identical invariant:** with `wiring: wild_type` (the default) the transform is a no-op; a test
  asserts the built `m_chem` / `w_chem` init / `g_gap` are identical to the unmodified load path
  (defence against an accidental behaviour shift for the wild-type cell that the 029 ranking reported).

### D5 — Deferred: learnable-gap-junction control

`T7.controls.learnable_gj` (promote the fixed `g_gap` buffer to a trainable, symmetry-preserving
`nn.Parameter` in the optimiser — the frozen-electrical-synapse fairness question) is a tracked
**fast-follow**, out of scope here. Rationale: lower expected signal (gap junctions are a minority of
the wiring; the connectome trails the MLP by a wide margin), lower biological fidelity (worm
gap-junction conductances are far less plastic than chemical synapses), and it carries the only
non-trivial correctness risk in this space (the batched forward's `h @ gap_mat == gap_mat.T @ h`
equivalence assumes symmetry, so a learnable `g_gap` must be re-symmetrised each forward). Revisited if
the rewired-null result or the 6a-synthesis review raises the frozen-gj objection.

## Risks / open questions

- **Swap mixing / connectivity:** a degree-preserving swap can in principle fragment the graph; we do
  **not** enforce connectivity (a fragmented degree-matched graph is a valid null and the fan-in scales
  still hold). If a rewired draw produces a pathological isolate that fails to train at all, it is
  reported, not silently reseeded (silent reseeding would bias the null).
- **Gap-junction rewiring scope:** we rewire both edge types (the tracker's "gap-junction graph handled
  consistently"). An alternative — rewire chemical only, keep gap wild-type — is recorded as a
  sensitivity variant but not the primary null (the primary null scrambles all specific wiring).
- **Directed swap acceptance rate:** on a sparse directed graph the reject-on-multi-edge rate is low;
  if mixing is poor we raise `swaps_per_edge`, recorded with the run.
