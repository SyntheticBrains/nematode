# wormbench

Grade simulated or real *Caenorhabditis elegans* tracks against real-worm behaviour.

`wormbench` reads 1-s positions and grades a population on standard assays:

- **Scheer & Bargmann's small lawn:** roaming fraction, roaming and dwelling bout lengths, lawn
  exits per on-lawn hour, roaming enrichment before an exit, and time on the lawn;
- **Scheer & Bargmann's uniform well:** roaming fraction and both bout lengths;
- **kinematics on the lawn:** 10-s window speeds while dwelling and roaming, and roaming-bout
  survival;
- **activity on the lawn:** the pause share, the direction-flip fraction and the reversal-onset rate;
- **López-Cruz et al.'s local search** after food removal: its decay rate and early/late ratio.

Each statistic is graded against a band built from the spread between real recording dates, at the
population's own size. A population passes a statistic when its z lies within the benchmark's
Bonferroni bar, and it passes the wild type when every family passes. Two distances summarise the
grading: D over every family and D_fit over the families a fit reads. A mutant is graded on its
difference from its own wild type.

It depends on numpy and scipy only, on Python 3.10 or later.

## Install

```bash
pip install wormbench
```

## A first grading

The package ships 64 real wild-type animals on the small lawn, so a grading runs with no download:

```bash
wormbench example --out example
wormbench grade --manifest example/manifest.txt --out example/grade.json
```

These animals come from the recording dates the bands are built from, so the example shows how the
grader reads tracks and validates nothing.

## Your own tracks

A manifest lists one file per line, as `<family> <file>`, with `family` one of `small_lawn`,
`uniform_lawn` or `local_search`. Each animal in a file is one member of its family's population.
Relative files are read against the manifest's directory.

**CSV**, with a sidecar `<file>.json`:

- columns `animal`, `t_s`, `x`, `y`, and an optional `on_lawn` (0 or 1);
- the sidecar names the `family`, the `analysis_start_s` at which the family's window begins, the
  `units` (`mm`, `um`, or in a CSV pixels with `mm_per_px`) and the `lawn`. A lawn is
  `{"centre": [x, y], "radius": r}`, `{"centre": [x, y], "radii": [...]}` at equally spaced angles
  from +x, or `"uniform"`.

**WCON 1.0**: its core (`units`, `data`), with the same metadata in a top-level
`@nematode_benchmark` object. Records sharing an `id` are merged in time order. A skeleton is reduced
to its middle point, and `null` is a missing sample.

Each track is resampled to 1 s and cut to its family's window: the small lawn's and the uniform
well's last 40 minutes, and local search's 45 minutes from food removal.

### Simulated tracks

A real tracker loses and blurs a worm's position; a simulator does not. To grade a simulated worm the
way the benchmark grades its own, pass `--simulated`. Each animal on Scheer's rig is then measured
first: tracker noise resampled from real paused worms, and lost seconds.

```bash
wormbench grade --manifest sim.txt --out sim.json --simulated
```

That needs each second's primitive (`run`, `reversal`, `omega` or `delta`) and one integer seed per
animal: `primitive` and `seed` CSV columns, or a WCON record's `@primitive` (one per time) and
`@seed`. A simulated track must be sampled every second. A seed used twice is refused, since two
animals would draw the same noise. Local search is never measured.

### What is not graded for imported tracks

Chemotaxis is graded on Chen et al.'s measured butanone field, which an imported track does not
carry, so an imported population is graded on the other families, and its D is undefined.

## Versions

`--bands v1|v2|v3|v4` picks the benchmark's version. Every 1.x release grades on v4 by default;
only a major release changes the default. v1 to v3 are frozen, so a past grading can be reproduced.
Every grading records the package version, the bands version and the bands' SHA-256.

## Data

The bands, the classifier, the tracker-noise library and the local-search reference ship inside the
package. Grading at a population size with no vendored bands (anything but 64 or 200 animals)
rebuilds the bands from the real tracks, which are fetched once:

```bash
wormbench data fetch
wormbench data verify
```

**Not yet available in this revision.** The tracks' data deposit is not published, so the package
records no download addresses, and `wormbench data fetch` refuses with a message saying so. Until it
is, grade populations of 64 or 200 animals, or point `WORMBENCH_DATA` at a directory holding the
tracks, laid out as `benchmark/<file>`. The example in *A first grading* needs no download.

Nothing is downloaded otherwise. Every file read from `WORMBENCH_DATA`, the package or the fetched
cache is checked against the SHA-256 the package records. A program that sets its own data root
(as the project this package comes from does for its repository's `data/`) has those files read as
given. `wormbench info` shows where each file is read from.

## Citation

Cite the package (see `CITATION.cff`) and the data it grades against:

- Scheer, E. & Bargmann, C. I. (2023). *eLife* 12:RP88657. Dryad doi:10.5061/dryad.47d7wm3jf.
- López-Cruz, A., et al. (2019). *Neuron* 102:407–419. Mendeley Data doi:10.17632/prdtw5jgcd.1.
- Chen, K. S., et al. (2025). *PLoS Biology*, doi:10.1371/journal.pbio.3003005.
- Calhoun, A. J., et al. (2014). *eLife* 3:e04220; Roberts, W. M., et al. (2016). *eLife* 5:e12572.

## Licence

The code is licensed under Apache-2.0. The reference data keep their deposits' licences, CC0-1.0
and CC BY 4.0; see `NOTICE` and each data namespace's `PROVENANCE.md`.
