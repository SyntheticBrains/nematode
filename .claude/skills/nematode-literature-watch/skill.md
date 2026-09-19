---
name: nematode-literature-watch
description: Run the literature watch locally (fetch, score with Haiku, digest with Opus), or refresh its seed set and project brief. Use when the user wants to sweep for new relevant research now rather than waiting for Monday's workflow, or when the phase plan has moved and the watch should be re-aimed.
metadata:
  author: nematode
  version: '1.0'
---

The weekly watch sweeps three sources, scores every candidate against a hand-maintained brief with
Claude Haiku, and has Claude Opus write the digest for the survivors. It runs on Mondays via
[`.github/workflows/literature-watch.yml`](../../../.github/workflows/literature-watch.yml) and
opens an issue. This skill does the same thing locally, and maintains the two hand-written inputs
that decide what it finds.

Read [`docs/research/literature-watch/README.md`](../../../docs/research/literature-watch/README.md)
first if you have not — it explains the sources and why they are ordered as they are.

**Requires `ANTHROPIC_API_KEY` in the environment for anything past the fetch.** The fetch itself
needs no key.

## Which workflow

- "run the literature watch", "what's new", "sweep for papers" → **Workflow A**.
- "update the seeds", "the digest is finding the wrong things", a phase plan that has moved on,
  or a phase-plan review in progress → **Workflow B**.

Working files go in `build/literature/` (gitignored). Create it if it does not exist.

______________________________________________________________________

## Workflow A — run the watch now

**1. Fetch, and cache what you fetched.**

The sweep takes five to ten minutes and is the slow part; cache it so scoring can be repeated
without paying for it again.

```bash
mkdir -p build/literature
uv run python scripts/literature_watch.py \
    --dry-run --candidates-out build/literature/candidates.json
```

Report the counts from stderr to the user — records per source, candidates after dedupe. A source
that failed is reported there too, and a digest built on a partial sweep says so in its header.

**2. Carry the already-reported ids, if the repository has past digests.**

Skip this when running ad hoc; it only matters if the user wants the local run to behave like the
scheduled one.

```bash
gh issue list --label literature-watch --state all --limit 60 --json body --jq '.[].body' \
  | grep -o '<!-- literature-watch-reported:[^>]*-->' \
  | sed -e 's/<!-- literature-watch-reported://' -e 's/-->//' \
  | tr ' ' '\n' | sed '/^$/d' | sort -u > build/literature/seen.txt
```

**3. Score and write the digest.**

```bash
uv run --no-project --with anthropic python scripts/literature_watch.py \
    --candidates-in build/literature/candidates.json \
    --out build/literature/digest.md
```

Add `--seen-file build/literature/seen.txt` if step 2 ran. Costs a few cents.

**4. Read it with the user.**

Show the digest. Then say which entries, if any, actually warrant an action, and what that action
is — none of these are automatic:

- A paper that bears on an open question → it belongs in the roadmap or a logbook, cited where the
  question is stated. Offer to draft that.
- A paper the project builds on from here → it belongs in `seeds.toml` (Workflow B, step 3).
- A result that contradicts or pre-empts a registered reading → say so plainly; that is the
  finding, not the paper.

Do not open an issue from a local run. The scheduled workflow owns the issue series, and a
hand-made one pollutes the seen-list it reads back.

______________________________________________________________________

## Workflow B — re-aim the watch

The brief and the seed list are the only things deciding what the watch finds, and neither can
notice it has gone stale. This is the maintenance pass, and it is principle 13 of the
[phase protocol](../../../docs/research/phase-protocol.md) — run it at each phase close.

**1. Read what the project is actually asking now.**

- `docs/roadmap.md` — the current phase's section, and § what the next phase opens on.
- The two or three most recent logbooks in `docs/experiments/logbooks/`.
- `docs/research/literature-watch/context.md` as it stands.

**2. Rewrite the brief, not the whole file.**

Edit `context.md`. Its § *What is open right now* is the part that decays; the project description
and the § *What is not relevant* list rarely change. Each open question should be phrased so that a
model reading only this file can tell whether an abstract bears on it — name the mechanism, not the
decision ID. **Never cite roadmap sections, decision IDs, logbooks or phase names in it**: the
scorer has no access to those and a pointer it cannot follow is a wasted line.

Keep it roughly its current length. It is read once per scoring batch; detail that does not change
a score is detail that is paid for every week.

Update the `Last reviewed:` line.

**3. Add seed papers.**

A seed is a paper this project builds on, such that new work citing it is likely to matter here.
Resolve each to an OpenAlex work id:

```bash
curl -s --get "https://api.openalex.org/works" \
  --data-urlencode "search=<title>" \
  --data-urlencode "select=id,display_name,publication_year" \
  --data-urlencode "per-page=3" | python3 -m json.tool
```

Add an `[[openalex.seeds]]` block with the `W…` id and a label naming author, year and subject.
Add the preprint as its own seed where one exists — citations split between the two versions.

**4. Check the change did something.**

Re-score a cached sweep against the edited brief and compare. This is the only cheap way to see
whether an edit moved anything:

```bash
uv run --no-project --with anthropic python scripts/literature_watch.py \
    --candidates-in build/literature/candidates.json \
    --out build/literature/digest-after.md
```

Diff it against the digest from before the edit. If nothing moved, the edit was decorative —
say so rather than claiming an improvement.

**5. Verify and commit.**

```bash
uv run pytest packages/quantum-nematode/tests/quantumnematode_tests/entrypoints/ -q --no-cov
uv run pre-commit run -a
```

The tests check that the config parses, the context file resolves, and every seed is a well-formed
OpenAlex id — which is what catches a typo before Monday does.
