# Add the L4 2×2 panel (plastic wild-type vs plastic rewired-null)

## Why

Phase 7's headline question is now fully instrumented and entirely unanswered. Every arm the
pre-registered design (roadmap D2/D10) calls for exists as a one-key config on the frozen C3
substrate: the plastic wild-type connectome, its plastic degree-preserving rewired-null, the
frozen-weights and unmodulated-Hebbian sanity floors, and the matched-rule MLP yardstick. What does
not exist is the experiment that puts them side by side — and, critically, **no plastic arm has ever
run past the smoke test**. Nothing in the repository shows whether the three-factor rule learns on
this cell at all, how fast it converges, or whether the shipped default recipe is anywhere near a
sensible operating point.

The Phase 7 tracker deliberately deferred two numbers to this change: the uniform episode budget and
the concrete matched-rule-MLP band, "pinned in the panel milestone change after the minimal rule's
convergence behaviour is measured". The whole value of the panel rests on the order in which things
happen: the **tests are fixed before any panel run**, the **recipe and budget are pinned by a
pre-registered pilot on seeds the panel never uses**, and only then does the panel launch. A panel
whose success criteria could still move after its data existed would be worthless to a hostile
referee — and the roadmap's own risk table says the likeliest outcome is the robustness branch, the
outcome that most tempts post-hoc adjustment.

Two design choices settled with Chris before this proposal was written:

- **The panel is the full 2×3 factorial** — wiring {wild-type, rewired-null} × rule {frozen,
  unmodulated Hebbian, three-factor} — plus the matched-rule MLP: seven arms. The two rewired floors
  are new one-key configs. They earn their place by making a *learning-gain* read available: plastic
  minus own-frozen within each wiring removes whatever initial-policy offset the random wiring carries,
  so the primary contrast can be read both on raw plateau and on what the rule added.
- **The pilot sets one shared recipe, then the panel runs it.** A pre-registered three-point
  `plasticity_rate` grid on disjoint pilot seeds, selected by a neutral pooled criterion across the
  three-factor arms — never by the arm the hypothesis favours — and then applied identically to every
  arm. This mirrors how Logbook 029 tuned recipes on the cell before ranking, and keeps "matched rule"
  literally true.

## What Changes

- **Two configs**: rewired-null versions of the frozen and Hebbian floors, each one `wiring` key off
  its wild-type parent (and one key off the plastic rewired-null arm), with variant tests pinning the
  one-key property.
- **A panel harness** (`scripts/analysis/l4_panel.py`) reusing the committed plateau-tail metric and
  the paired-seed Wilcoxon / 80% bootstrap CI / BH-FDR layer verbatim. It reads a campaign directory
  or a manifest, computes the four pre-registered confirmatory tests as one BH-FDR family, the
  CI-based MLP band test, the verdict under the pre-registered map, the ensemble-invariance read the
  roadmap's bar (a) needs, and every other pair descriptively. A `--pilot` mode summarises the recipe
  grid and prints the recipe and budget the pre-registered rules select.
- **A pilot runner** (`scripts/campaigns/l4_panel_pilot.py`) that derives the grid configs from the
  committed arms, records them beside the results, and drives the parallel campaign runner.
- **The pre-registration itself**, as capability requirements: arms, seeds, metric, tests, family,
  verdict map, the pilot's selection rules, and what may and may not change after the pilot. The
  budget and recipe are pinned by a dated amendment to this change's design before launch, and the
  launch is recorded (commit, command, seeds, budget, recipe) before any result is read.
- **The runs**: the pilot, then the panel (seven arms × eight paired seeds), with the harness output
  and per-seed data promoted to `docs/experiments/logbooks/supporting/040-l4-panel/` for the
  logbook that follows.

Out of scope: the logbook and the roadmap status sync (the next tracker item), the imitation
warm-start arm, any change to the substrate, the rule, or the existing five configs beyond making the
pinned recipe explicit.

## Capabilities

**New**: `l4-plasticity-panel` — the pre-registered protocol for the L4 panel: its arms, seed
discipline, ranked metric, confirmatory family and verdict map, the pilot that pins recipe and
budget, the harness that computes it all, and the persistence of its data.

## Impact

- New: two configs, one analysis script, one campaign script, their tests, the supporting-data
  directory.
- Edited: all seven plastic-family configs (six connectome arms and the MLP) gain an explicit
  `plasticity_rate` at the pin step (the
  value the pilot selects, stated even when it equals the default so the recipe is visible in the
  file); `configs/README.md`, `docs/architectures.md`, `CHANGELOG.md`, the Phase 7 tracker.
- No package code changes. Every run goes through the standard single-run entry point via the
  campaign runner, so results remain comparable with Logbooks 029 and 034.
