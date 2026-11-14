# Placement: where a neural track would live

Written 2026-11-14. RECOMMENDATION.md recommends a hybrid, so the question is where its neural track
lives: in Quantum Nematode, in a new repository that depends on it, or in a standalone repository.

**Verdict: a module inside Quantum Nematode, behind an optional extra, with a versioned export
format from its first fitted model.** It should be extracted to its own repository only if a kill
criterion has not fired and the track outgrows the hybrid. "Outgrowing" means it needs a dependency
or a CI load that Quantum Nematode should not carry, or Wormturn needs fitted models without Quantum
Nematode.

## The criteria

**Architecture.** The simulation core does not need a differentiable backend. The spike showed why:

- fit-first's lever is the objective, a likelihood on neural data;
- the neural model is fitted offline, on recordings, and needs no simulator;
- in the closed loop, it plugs in as a controller through the batched evaluator's existing
  `BatchedController` protocol (`reset`, `act`), as the bridge design did.

So the environment, the primitives body and the benchmark stay as they are. Nothing in the core is
distorted, and JAX is not needed. The neural code is a separate layer: data ingestion, a likelihood,
a fitter, and an adapter into the controller protocol.

**Identity.** Since roadmap v5.0, the project's stated question is how faithfully a simulated worm
reproduces C. elegans behaviour, graded against real-worm data, and what the connectome contributes
to that fidelity. Two parts of a neural track fit that question directly:

- a neural readout is a second real-worm grader;
- a degree-preserving null on neural prediction is a test of what the connectome contributes.

A switch to fit-first would change the identity to a neural-fitting project, and a separate
repository would suit that better. The name already understates the scope, since quantum circuits
became one controller family at v5.0. Publicly, the hybrid is positioned as "behaviour and neural
fidelity, graded separately, on one connectome".

**Dependencies.**

| Need | Weight | Where |
|---|---|---|
| PyTorch | none new; already a dependency | core |
| `pynwb`, `dandi`, `h5py` | moderate; only for ingesting recordings | an optional extra, `neural` |
| JAX | not needed | none |
| A float64 GPU | only for Randi-scale panels; rented, not installed | outside the repository |

**Licences.**

- **Code:** the repository is Apache-2.0.
  - Creamer's code is MIT, and can be ported with its notice.
  - Mi's code has no licence, so nothing from it can be reused; it is rewritten from the paper.
  - `wormneuroatlas` is GPL-3.0 and stays out.
- **Data:** CC-BY or CC0 deposits are vendored as derived arrays with their provenance, as `data/`
  already does: DANDI 001075 traces, Yemini via Zenodo 3906530 (MIT), Atanas 2023 on DANDI 000776.
  Unlicensed deposits (the OSF exports) are not used until their authors state a licence. Two drafted
  notes ask them to.

**Maintenance at part-time pace.**

- **One repository has one CI, one protocol and one set of hooks.** The phase protocol, the
  registration rules and the review skills apply without duplication.
- **A second repository would double the release, CI and dependency upkeep** for a track that, under
  the hybrid, is a few milestones long.
- **The cost of staying inside:** the neural module's tests join a suite already split over five CI
  shards. They should be small, with heavy ingestion behind the optional extra and marked slow.

**Integration with Wormlight and Wormturn.**

- **Wormlight** (the browser worm) already consumes a versioned export from this repository:
  `scripts/export_wormlight.py` writes `wormlight.connectome/1`, one deterministic JSON file whose
  provenance records the commit and every source file's SHA-256. Wormlight runs a graded,
  conductance-based network that drives a body, so it can take fitted weights for the neurons a fit
  covers, and keep its own defaults elsewhere.
- **Wormturn** (plasticity research) also consumes a versioned export now. `main` gained
  `scripts/export_wormturn.py` on 2026-11-14 (PR #532), after this branch was cut. It writes
  `wormturn.connectome/1`: the signed chemical wiring, with each sign's source and citation, and the
  data-derived neuromodulator layer, under the same provenance rules. On top of that, Wormturn needs
  to know which parameters a fit constrains and which it leaves free. Plasticity rules can then act
  on the unconstrained ones without overwriting what the data pinned.

Both want the same thing: a fitted model with its neuron names, the wiring it was fitted on, its
parameters, and an honest statement of what the data constrained.

## The export format: `nematode.fit/1`

One directory per fitted model, following the pattern of `wormlight.connectome/1` and
`wormturn.connectome/1`: deterministic, versioned and provenance-first. A fit does not carry its own
copy of the wiring. It names the connectome export it was fitted on, by schema, commit and SHA-256,
so consumers already holding that export can join the two.

- **`manifest.json`:**
  - **`schema`:** `"nematode.fit/1"`. The major version changes for any change a consumer must
    handle; the minor version adds optional fields only.
  - **`model`:**
    - the family (for example `lds_masked`, `latent_voltage`, `rate_settling`);
    - the time step in seconds;
    - the state, input and output definitions, each by canonical neuron name and class;
    - the input channels, with their neuron, encoding and units.
  - **`parameters`:** an index into `arrays.npz`, giving each array's name, shape, dtype, units,
    and whether the data constrained it (`constrained`, `prior`, `free`, following
    IDENTIFIABILITY.md's classes).
  - **`wiring`:**
    - the connectome source and release;
    - the mask rule;
    - the SHA-256 of every source file;
    - for a null, the null's kind and seed.
  - **`training`:**
    - every dataset, by identifier, version and asset;
    - its licence;
    - the split, the fit configuration, and the fitting code's commit.
  - **`scores`:**
    - each registered reading, with its value, its floor, its null and the commit of its
      registration (for example the M-a reading, benchmark v4's D);
    - a missing score is absent, never zero.
  - **`licence`:** the export's own licence, which must allow everything its sources allow and no
    more.
- **`arrays.npz`:** the parameters, float64.
- **`checksums.txt`:** the SHA-256 of both files.

**The consumers:**

- **Quantum Nematode** reads an export through one adapter class that implements `BatchedController`.
- **Wormlight** gets a JSON-only projection, the fitted weights mapped onto its connectome's
  neuron and synapse records, written by an extension of the existing export script.
- **Wormturn** joins a fit to the `wormturn.connectome/1` export it names, and reads the
  `constrained` flags.

## What carries over, and what does not

**Carries over:**

- the connectome loaders and the 118 neuron classes;
- the batched evaluator and its controller protocol;
- benchmark v4 and its grading;
- the data-derived modulator layer;
- the Wormlight export's provenance pattern;
- from the spike: the NWB ingestion and preparation re-implemented in session 0, the port of
  Creamer's STAM scorer, the settling copy and parity tests of the M-a reading, and the mask-level
  degree-preserving null's design.

**Does not carry over:**

- C.2's CMA-ES procedure, since neural fits use likelihood gradients instead;
- Mi's code, which has no licence;
- Creamer's EM, which is not needed;
- the 4-TB imaging files.
