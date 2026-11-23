# wormbench

Grade simulated or real *Caenorhabditis elegans* tracks against real-worm behaviour.

`wormbench` reads 1-s positions and grades them on standard assays:

- Scheer & Bargmann's small lawn and uniform well (roaming, dwelling, lawn leaving);
- the kinematics and activity of worms on a lawn;
- López-Cruz et al.'s local search after food removal;
- Chen et al.'s butanone chemotaxis.

Each statistic is graded against bands built from the spread between real recording dates. A
population passes a statistic when its z lies within the benchmark's Bonferroni bar.

It depends on numpy and scipy only, on Python 3.10 or later.

> This package is under construction. The commands, the data and the worked example are described
> here as each lands.

## Licence

The code is licensed under Apache-2.0. The reference data keep their deposits' licences (CC0-1.0
and CC BY 4.0); see `NOTICE` and each data namespace's `PROVENANCE.md`.
