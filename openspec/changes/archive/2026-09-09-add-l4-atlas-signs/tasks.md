# Tasks: grounding synapse signs in the neurotransmitter atlas

## 1. The vendored atlas

- [x] 1.1 Vendor `data/connectome/elife-95402-supp2-v1.xlsx` (LFS) and add its `PROVENANCE.md`
  entry in the existing format: description, upstream filename, size, SHA256, source URL,
  mirror licence, retrieval date, citation, redistribution rationale.
- [x] 1.2 Test: the file is present and its SHA256 matches the recorded digest.

## 2. Transmitter identities and the sign table

- [x] 2.1 An atlas loader that reads `Supp File 2`, normalises the transmitter column (leading
  `*`, trailing `- NEW`, case, parenthetical qualifiers), maps `DB1/3`→`DB1` and
  `DB3/1`→`DB3`, and returns a per-neuron release identity; uptake-only entries give none.
- [x] 2.2 A generation script that writes the atlas's transmitter values into the checked-in
  `NEURON_CLASSIFICATION` literal (committed, reviewable, no import-time spreadsheet read), with
  the derived sign table (ACh/Glu excitatory, GABA inhibitory,
  monoamines and orphans unknown) and the receptor caveat stated in the module docstring.
- [x] 2.3 Tests: the committed table's transmitters equal what the atlas loader re-derives, entry
  for entry; 302 names covered exactly and the two renamed entries asserted; the normalisation
  vocabulary; uptake grounds no sign; the coverage numbers hold (268 neurons signed, 3,176 of
  3,709 synapses, 2,962 excitatory, 214 inhibitory); `Neuron.neurotransmitter` is populated
  through the loader.

## 3. Sign grounding and Dale's law

- [x] 3.1 `synapse_signs: Literal["random","atlas"] = "random"` on the connectome config; under
  `atlas`, `w = |draw| · sign(pre)` with the same draw order and generator, unknown sources
  keeping the drawn sign.
- [x] 3.2 `enforce_synapse_signs: bool = False` on `ConnectomePPOBrainConfig` (not the shared
  mixin, which the MLP brain inherits); the rule projects each plastic weight onto its sign in the
  order the rule already uses — update → decay → projection → homeostasis → clamp; refused at
  construction unless signs are grounded.
- [x] 3.3 Tests: the default is bit-identical; under `atlas` magnitudes and per-neuron incoming
  norms are unchanged and only signs differ, with the grounded fraction as computed; unknown
  sources keep their drawn sign; enforcement holds every grounded sign across many updates while
  unknown ones stay free; enforcement without grounding raises; no forbidden sign survives the
  homeostatic rescale and the clamp; the MLP brain has no such field.

## 4. Configs

- [x] 4.1 Six configs, each one or two keys off a committed parent, naming keeping the parent as a
  prefix: `…_plastic_frozen` and `…_plastic_frozen_rewired_null` plus `synapse_signs: atlas`;
  `…_plastic_hebbian` and `…_plastic_hebbian_rewired_null` plus `synapse_signs: atlas`; and those
  two Hebbian arms again plus `enforce_synapse_signs: true`. Variant tests asserting the exact key
  delta, and smoke entries — the vendored atlas is under `data/**`, which `.lfsconfig` fetches on a
  default clone, so a fresh checkout can run them.

## 5. The harness

- [x] 5.1 `scripts/analysis/l4_atlas_signs.py`: the arm registry and per-arm seed ranges, panel 2's
  committed per-seed table as the random-sign comparator, G1–G4 as one BH-FDR family, the
  verdict map with `substrate_fail` first, the descriptive layer, sign-flip and saturation
  telemetry, per-seed CSV and curves.
- [x] 5.2 Tests on synthetic values: registry and ranges, each direction, family size, every
  verdict row in order including `substrate_fail`, the annotations, the CSV shape.

## 6. Launch and run

- [x] 6.1 Launch record under `supporting/044-l4-atlas-signs/` committed before any run.
- [x] 6.2 Prior sweep: the two grounded frozen arms × seeds 1–64 × 600 episodes.
- [x] 6.3 Hebbian contrast: the four grounded Hebbian arms × seeds 1–16 × 1000 episodes; the single
  registered extension at 1.5× for any run the plateau detector marks non-converged.

## 7. Analysis and records

- [x] 7.1 Analyse; promote `panel.json`, `per-seed.csv`, `curves.csv`, the manifest and a
  `details.md` to the supporting directory.

## 8. Close-out

- [x] 8.1 `docs/architectures.md`, `configs/README.md`, `CHANGELOG.md`; tracker B.1 ticked with the
  verdict.
- [x] 8.2 Pre-commit gate on all files exit 0; full suite green.
- [x] 8.3 No implementation code or docstring references a planning document.
- [x] 8.4 Re-review for drift, archive, review the branch, open the PR.
