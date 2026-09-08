## ADDED Requirements

### Requirement: Neurotransmitter Identities From the Vendored Atlas

The substrate SHALL vendor the Wang et al. 2024 neurotransmitter atlas (Supplementary File 2)
under `data/connectome/` with a `PROVENANCE.md` entry recording its description, upstream
filename, size, SHA256, source URL, mirror licence, retrieval date, citation and redistribution
rationale, in the format the directory's existing entries use. A loader SHALL read that file and
return a release identity for each of the 302 hermaphrodite neurons, normalising the atlas's
editorial annotation (a leading `*`, a trailing `- NEW`, letter case, and a parenthetical
qualifier separated from the base label) and mapping the atlas's `DB1/3` and `DB3/1` to the
project's `DB1` and `DB3`. An entry qualified as uptake SHALL NOT yield a release identity. The
302-entry neuron classification table's transmitter slot SHALL be populated from that loader's
output as committed literal values, not read from the vendored file at import, and a test SHALL
assert that the committed values equal what the loader re-derives; `Neuron.neurotransmitter` SHALL
carry them through the connectome loader.

The substrate SHALL derive a per-transmitter sign — acetylcholine and glutamate excitatory, GABA
inhibitory, monoamines and orphan or uptake-only identities unknown — and SHALL document, where
that table is defined, that a transmitter does not determine a synapse's sign because the
post-synaptic receptor does, so the table is a per-neuron approximation pending receptor classes.

#### Scenario: Every neuron has an identity from the atlas

- **WHEN** the atlas loader runs
- **THEN** it SHALL return an entry for each of the 302 canonical neuron names, with the atlas's
  two differing names mapped
- **AND** the classification table's transmitter slot SHALL be populated for all 302

#### Scenario: Uptake is not release

- **WHEN** the atlas records a transmitter qualified as uptake for a neuron
- **THEN** that neuron SHALL have no release identity and no derived sign

#### Scenario: The vendored file is the one recorded

- **WHEN** the vendored atlas is read
- **THEN** its SHA256 SHALL equal the digest recorded in `PROVENANCE.md`

#### Scenario: Sign coverage is the documented one

- **WHEN** the derived sign table is applied to the Cook 2019 chemical synapses
- **THEN** the neurons with a sign, the synapses grounded, and the excitatory and inhibitory
  counts SHALL equal the figures the change records
