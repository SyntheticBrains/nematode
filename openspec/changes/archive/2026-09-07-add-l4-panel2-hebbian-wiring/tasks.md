# Tasks: panel 2 — the Hebbian wiring contrast

## 1. Count-scaled initialisation

- [x] 1.1 `weight_init: Literal["degree_scaled", "count_scaled"] = "degree_scaled"` on the connectome
  config, passed to the topology; under `count_scaled` each chemical weight is
  `z · n / sqrt(Σ n²)` over the post-synaptic neuron's incoming counts, same draw order and
  generator as the degree-scaled path.
- [x] 1.2 Tests: the default is bit-identical (frozen-reference and wiring-arms tests keep passing);
  under `count_scaled` the scale factors `n / sqrt(Σ n²)` of every neuron's inputs have unit sum of
  squares (exact, per neuron) and magnitudes within a neuron proportional to counts; gap junctions
  are untouched;
  the rewired arm's counts travel with their pre-synaptic endpoints and its normalisation is
  recomputed; the two wirings stay paired at one seed (readout, gains, `log_std` identical).

## 2. Configs

- [x] 2.1 Four count-initialised configs (`_countinit` suffix on the frozen and Hebbian arms, wild-type
  and rewired), each one `weight_init` key off its parent; variant tests and smoke entries.

## 3. The harness

- [x] 3.1 `scripts/analysis/l4_panel2.py`: eight-arm registry; seed ranges enforced (Hebbian 1–16,
  frozen 1–64); panel 1's readers and the statistics layer imported; P1–P4 as one BH-FDR family;
  the verdict map; the prior-sweep analysis (distributions, competent fraction at 20%, P4); the
  descriptive pairs and learning gains; per-seed CSV and curves.
- [x] 3.2 Tests on synthetic logs: registry and seed ranges, each direction, family size, every
  verdict row, the reverse case, the competent fraction, the P4 annotation, the CSV shape.

## 4. Launch and run

- [x] 4.1 Launch record under `supporting/041-l4-panel2/` (commit, commands, seeds, budgets) committed
  before the campaigns run.
- [x] 4.2 Hebbian panel: four arms × seeds 1–16 × 1000 episodes; the single registered extension for
  any seed without a plateau (fresh run at 1500).
- [x] 4.3 Prior sweep: four frozen arms × seeds 1–64 × 600 episodes.
- [x] 4.4 Check: on seeds 1–8 the per-episode outcomes of the Hebbian arms' first 1000 and the
  frozen arms' first 600 episodes equal panel 1's logs' prefix.

## 5. Analysis and records

- [x] 5.1 Analyse; promote `panel2.json`, `per-seed.csv`, `curves.csv`, the manifest and a
  `details.md` to the supporting directory.

## 6. Close-out

- [x] 6.1 `docs/architectures.md`, `configs/README.md`, `CHANGELOG.md`; tracker A.9 ticked with the
  verdict.
- [x] 6.2 Pre-commit gate on all files exit 0; full suite green.
- [x] 6.3 No implementation code or docstring references a planning document.
- [x] 6.4 Re-review for drift, archive, review the branch, open the PR.
