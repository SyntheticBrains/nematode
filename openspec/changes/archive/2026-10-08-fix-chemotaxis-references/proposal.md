## Why

`data/chemotaxis/literature_ci_values.json` was flagged in the Wormlight review (tracker M.7) for two
misattributions. A source-by-source check found the whole file unreliable:

- **None of its five CI values comes from its cited paper.**
- **Three citations point to unrelated articles.** *Cell* 65:837 (1991) is Hall & Hedgecock's kinesin
  paper; the Bargmann & Horvitz 1991 chemotaxis paper is *Neuron* 7:729, on water-soluble attractants.
  "Saeki 2001, *Neuron* 32:249" is *J Exp Biol* 204:1757. "Ferkey 2007, *Genetics* 175:43" does not
  exist; the paper is *Neuron* 53:39 and never assays benzaldehyde.
- **Two entries describe assays their papers did not run.** Bargmann et al. 1993 assayed volatile
  odorants, never bacteria. Pierce-Shimomura et al. 1999 used ammonium chloride and biotin gradients,
  with a different index.

The experiment tracker reads the file's default source, the "bacteria" entry, and writes its citation,
range and a `matches_biology` verdict into every tracked run. The comparison is also invalid in kind:
the simulated index is a time-in-zone fraction, and every published chemotaxis index is an endpoint
count of worms. No logbook or spec relies on those fields. The project's behavioural validation is the
bias-curve harness (Logbooks 035, 036), which does not use this file.

## What Changes

- **The reference file is rebuilt from verified entries only.** Each carries its correct citation,
  what was assayed, the reported index, and whether the value was stated in the text or read from a
  figure:

  | source | attractant | wild-type CI | read from |
  |---|---|---|---|
  | Bargmann, Hartwieg & Horvitz 1993, *Cell* 74:515 | diacetyl, 10⁻³ | about 0.92 | Fig. 2 |
  | Bargmann, Hartwieg & Horvitz 1993 | diacetyl, 10⁻² | about 0.97 | Fig. 2 |
  | Bargmann, Hartwieg & Horvitz 1993 | benzaldehyde, 10⁻² | about 0.88 | Fig. 2 |
  | Bargmann, Hartwieg & Horvitz 1993 | benzaldehyde, 10⁻³ | about 0.65 | Fig. 2 |
  | Rodriguez et al. 2025, *PNAS* 122:e2513137122 | OP50 bacteria | 0.9 | text |

  The rest are dropped, each with its reason recorded in the file.

- **The tracker stops recording a literature verdict.** It keeps the simulated index and its
  validation level, and leaves the literature range, typical value, citation and `matches_biology`
  as `None`. The fields stay in the record schema, so older records load unchanged.

- **The validation level is described for what it is**: a banding of the simulated index at 0.4, 0.6
  and 0.75, not a biological match.

- **The literature comparison API is retired**: `ChemotaxisValidationBenchmark`, the built-in
  fallback dataset that repeated the same errors, and `validate_agent`'s range verdict. Loading the
  reference set stays.

## Capabilities

**Modified**: `experiment-tracking`, with one modified requirement (the per-episode chemotaxis summary)
and two added (no literature verdict on the simulated index; a verified reference set).

## Impact

- `data/chemotaxis/literature_ci_values.json`: rebuilt.
- `validation/datasets.py`: the loader reads the rebuilt schema; the benchmark and the fallback dataset
  are removed.
- `experiment/tracker.py`, `experiment/metadata.py`, `scripts/run_simulation.py`: the verdict is no
  longer computed or printed.
- Tests, the tracker M.7 entry, and the CHANGELOG.

## Breaking Changes

`ChemotaxisValidationBenchmark` and the built-in fallback dataset are removed, and the reference file's
schema changes. Tracked runs no longer populate four metadata fields. Older experiment records keep and
load their values.
