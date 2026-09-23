## Why

Phase 8's **B.1a**, the data sub-deliverable of B.1 — the last MUST of shipment 8a, which ships on
A.1, A.2 and B.1.

Every connectome brain in this project has been anatomically constrained in **topology** and
randomly initialised in **weight**. B.1 replaces the random draw with a measured one and asks, in
B.1c's 2×3, whether the animal's own weights make its wiring legible where random ones did not.
Decision **D17** fixes the design: wiring {wild type, rewired null} × prior {random, measured,
measured-shuffled}, head-only coverage reported at two scopes, the null's measured arm assigned by
A.1's per-neuron fan-in rule, and the unit scale treated as a pin. B.1a builds the data and the
switch; no campaign runs here.

**The licence check ran before anything was vendored, and it changed the deliverable.** The roadmap
and tracker both name the Randi 2023 signal-propagation atlas as a second vendored source. Its
primary deposit (OSF `e2syt`) states **no licence**, and its only licensed copy is a GPL-3.0 file
inside the `wormneuroatlas` package; this repository is Apache-2.0. **Randi is therefore cited, not
vendored**, as the raw measurement Creamer et al. fitted. The Creamer–Leifer–Pillow fitted weights are
released under **MIT** as `quick_start_examples/model_weights.csv` in
`Nondairy-Creamer/Creamer_LDS_2026`, and that file is what lands.

**Reading the data before designing for it narrowed several claims in the plan.**

- The released table covers **125 neurons** (2,011 signed edges); the fitted model holds 154. The
  "156 head neurons" in the roadmap is the paper's recording figure.
- The values are coefficients of a **2 Hz linear dynamical system** on calcium signals, fitted on a
  **White 1986 + Witvliet 2020** mask that is the **union of chemical and gap-junction** edges — a
  different connectome from this substrate's Cook 2019, and not typed by connection.
- Joined to Cook 2019's 3,709 chemical edges, **1,049 are covered**: 28.3% at full scope and 77.0%
  of the 1,363 coverable edges whose endpoints both lie in the table's neuron set (a further 23 there
  are self-loops, which a table with no diagonal cannot cover). **None** falls on the 39
  body motor neurons, as D17 anticipated. 265 table edges sit on Cook gap junctions only and 697 on
  no Cook connection; both are reported and neither is applied.
- Creamer et al. is still a **preprint** (bioRxiv 2024.09.22.614271 v3). It never stands alone.

## What Changes

### 1. Vendored data

`data/connectome/creamer_lds_2026_model_weights.csv`, the upstream file byte-for-byte, pinned by SHA256
and upstream commit; the MIT notice beside it; a `PROVENANCE.md` entry in the house shape; and a
"What is NOT vendored" entry recording why Randi, the model pickles and both OSF deposits stay out.

### 2. A loader for the measured table

Read at load time with its SHA256 checked — a 42 KB plain-git text file that is always present,
unlike the LFS spreadsheet the transmitter atlas copies into a committed literal. Every name is
validated against the canonical classification, and a coverage report joins it to any connectome.

### 3. A `weight_prior` switch on the connectome brain

`random` (default, bit-identical), `measured`, `measured_signs` and `measured_shuffled`, plus
`measured_weight_scale` (default 1.0), the pin B.1b sweeps.

- **Measured values are variance-matched.** Covered values sit on the same per-neuron
  `1/sqrt(in-degree)` the random draw uses, times one constant chosen so that their RMS over the
  wild type's covered edges equals the draw's expected RMS there. At the default multiplier a
  measured arm has the random arm's magnitude, so B.1c compares structure rather than size.
- **`measured_signs`** keeps the random draw's magnitude and takes the measured sign — B.1b's
  sign-only arm, built now so B.1b is configs only.
- **`measured_shuffled`** permutes the measured values among the wild type's own covered edges,
  which is what makes a positive interpretable.
- **On the rewired null** each post-synaptic neuron receives the values its wild-type edges carried
  under the same prior, in pre-synaptic-index order, as A.1's `per_neuron_fanin` assigns a drawn
  multiset — defined for all three measured priors, since B.1c's 2×3 needs every one on both wirings.
  The wild-type values are computed before rewiring, because the brain rewires before it builds.
- **`measured_weight_scale` is refused where nothing reads it** (`random`, `measured_signs`) — the
  failure the requirement A.2 added names.
- Uncovered edges keep the random draw in every mode.

The edge loop still takes exactly one value per edge from the generator the rollout buffer shares,
whatever the prior — A.1's defect was a mode that changed how many values that generator yielded.

## Capabilities

**Modified**: `connectome-substrate` (the vendored measured table and its loader) and
`connectome-ppo-brain` (the `weight_prior` switch). No methodology requirement: the rules governing
B.1c's contrast already exist and are cited where B.1c registers.

## Impact

- `data/connectome/` — the CSV, its licence notice, `PROVENANCE.md`
- `packages/quantum-nematode/quantumnematode/connectome/measured_weights.py` — new
- `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py` — two config fields, the
  prior in the edge loop, validation in both places, `training_state`
- Tests: a loader test, a brain test modelled on `test_connectome_weight_draw.py`, a persistence check
- Docs: the architectures catalogue row, the CHANGELOG, two stale docstrings fixed in passing
- Tracker and roadmap: B.1a ticked, with dated corrections for Randi and the neuron count

## Breaking Changes

None. `weight_prior` defaults to `random`, bit-identical to the brain without it.
