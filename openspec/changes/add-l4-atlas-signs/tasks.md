# Tasks: grounding synapse signs in the neurotransmitter atlas

## 1. The vendored atlas

- [ ] 1.1 Vendor `data/connectome/elife-95402-supp2-v1.xlsx` (LFS) and add its `PROVENANCE.md`
  entry in the existing format: description, upstream filename, size, SHA256, source URL,
  mirror licence, retrieval date, citation, redistribution rationale.
- [ ] 1.2 Test: the file is present and its SHA256 matches the recorded digest.

## 2. Transmitter identities and the sign table

- [ ] 2.1 An atlas loader that reads `Supp File 2`, normalises the transmitter column (leading
  `*`, trailing `- NEW`, case, parenthetical qualifiers), maps `DB1/3`→`DB1` and
  `DB3/1`→`DB3`, and returns a per-neuron release identity; uptake-only entries give none.
- [ ] 2.2 `NEURON_CLASSIFICATION`'s transmitter slot populated for all 302 neurons from that
  loader's output, with the derived sign table (ACh/Glu excitatory, GABA inhibitory,
  monoamines and orphans unknown) and the receptor caveat stated in the module docstring.
- [ ] 2.3 Tests: 302 names covered exactly and the two renamed entries asserted; the normalisation
  vocabulary; uptake grounds no sign; the coverage numbers hold (268 neurons signed, 3,176 of
  3,709 synapses, 2,962 excitatory, 214 inhibitory); `Neuron.neurotransmitter` is populated
  through the loader.

## 3. Sign grounding and Dale's law

- [ ] 3.1 `synapse_signs: Literal["random","atlas"] = "random"` on the connectome config; under
  `atlas`, `w = |draw| · sign(pre)` with the same draw order and generator, unknown sources
  keeping the drawn sign.
- [ ] 3.2 `enforce_synapse_signs: bool = False` on the plasticity mixin; the rule projects each
  plastic weight onto its sign after decay and before the clamp (update → decay → projection →
  clamp → homeostasis); refused at construction unless signs are grounded.
- [ ] 3.3 Tests: the default is bit-identical; under `atlas` magnitudes are unchanged and only
  signs differ, with the grounded fraction as computed; unknown sources keep their drawn sign;
  enforcement holds every grounded sign across many updates while unknown ones stay free;
  enforcement without grounding raises; the ordering leaves no forbidden sign after
  homeostasis.

## 4. Configs

- [ ] 4.1 Six configs, each one or two keys off a committed parent: the frozen arms on both
  wirings under `synapse_signs: atlas`, and the Hebbian arms on both wirings under
  `synapse_signs: atlas` — the enforced Hebbian arms adding `enforce_synapse_signs: true`
  (naming keeps each parent as a prefix). Variant tests and smoke entries.

## 5. The harness

- [ ] 5.1 `scripts/analysis/l4_atlas_signs.py`: the arm registry and per-arm seed ranges, panel 2's
  committed per-seed table as the random-sign comparator, G1–G4 as one BH-FDR family, the
  verdict map with `substrate_fail` first, the descriptive layer, sign-flip and saturation
  telemetry, per-seed CSV and curves.
- [ ] 5.2 Tests on synthetic values: registry and ranges, each direction, family size, every
  verdict row in order including `substrate_fail`, the annotations, the CSV shape.

## 6. Launch and run

- [ ] 6.1 Launch record under `supporting/044-l4-atlas-signs/` committed before any run.
- [ ] 6.2 Prior sweep: the two grounded frozen arms × seeds 1–64 × 600 episodes.
- [ ] 6.3 Hebbian contrast: the four grounded Hebbian arms × seeds 1–16 × 1000 episodes; the single
  registered extension at 1.5× for any run the plateau detector marks non-converged.

## 7. Analysis and records

- [ ] 7.1 Analyse; promote `panel.json`, `per-seed.csv`, `curves.csv`, the manifest and a
  `details.md` to the supporting directory.

## 8. Close-out

- [ ] 8.1 `docs/architectures.md`, `configs/README.md`, `CHANGELOG.md`; tracker B.1 ticked with the
  verdict.
- [ ] 8.2 Pre-commit gate on all files exit 0; full suite green.
- [ ] 8.3 No implementation code or docstring references a planning document.
- [ ] 8.4 Re-review for drift, archive, review the branch, open the PR.
