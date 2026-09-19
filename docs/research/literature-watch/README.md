# Literature watch

A weekly sweep for new work bearing on this project, delivered as a GitHub issue.

The problem it solves is precision, not retrieval. A keyword alert for "C. elegans" returns mostly
molecular biology; one for "plasticity" returns mostly deep learning. So the watch sources narrowly,
then reads everything it finds against a written brief.

## How it works

1. **Citation chaining** — [`seeds.toml`](seeds.toml) lists the papers this project builds on, and
   OpenAlex is asked for new work citing any of them. Almost everything it returns is on-topic; this
   is the channel worth reading first.
2. **Preprint sweeps** — arXiv (`q-bio.NC`, `cs.NE`) and bioRxiv subject collections for the same
   window. Low precision, but they see work weeks before it is citable.
3. **Triage** — Claude Haiku scores every candidate 0–3 against [`context.md`](context.md), the
   hand-maintained brief. Anything at 2 or above survives.
4. **Digest** — Claude Opus writes two or three sentences per survivor: what it found, and what it
   would change here. The digest is opened as an issue labelled `literature-watch`.

Roughly 600–900 records a week reach the triage pass, at single-digit cents. The GitHub Actions
minutes are free on a public repository.

## Running it by hand

The `nematode-literature-watch` skill drives both of the flows below; these are the commands under
it.

```bash
# Fetch and list candidates — no model calls, no API key. Takes five to ten minutes.
uv run python scripts/literature_watch.py --dry-run --candidates-out candidates.json

# Score that cached sweep and write the digest, without fetching again.
ANTHROPIC_API_KEY=... uv run --no-project --with anthropic \
    python scripts/literature_watch.py --candidates-in candidates.json --out digest.md
```

`--candidates-out` / `--candidates-in` exist because the fetch is the slow part and the scoring is
the part worth repeating — after editing the brief, re-score the same sweep and diff the digests to
see whether the edit changed anything.

The workflow ([`.github/workflows/literature-watch.yml`](../../../.github/workflows/literature-watch.yml))
runs Mondays at 08:00 UTC and can be dispatched by hand, with a `dry_run` input that lists
candidates without calling a model or opening an issue.

**Setup:** the repository needs an `ANTHROPIC_API_KEY` secret. `OPENALEX_MAILTO` is optional — an
address there moves OpenAlex requests into a faster pool, and it is kept in a secret rather than in
`seeds.toml` because this repository is public.

## What has to be maintained

Two things, both judgement calls, neither of which the machinery can do for itself.

**The brief** ([`context.md`](context.md)) defines relevance. Review it at each phase close —
principle 13 of the [phase protocol](../phase-protocol.md): a brief describing questions that closed
six months ago will keep producing a confident-looking digest about the wrong things.

**The seed list** ([`seeds.toml`](seeds.toml)) defines the high-precision channel. Add a paper when
it becomes one this work builds on. A seed set that stops matching the frontier is how a watch like
this quietly stops finding anything.

## Deduplication

The digest ends with an HTML comment listing the ids it reported. The workflow reads those comments
back out of previous issues and passes them to the next run, so state lives in the issues a human
already reads — no committed state file, nothing to expire. The citation-chaining window is
deliberately wider than the run cadence (three weeks against seven days), because OpenAlex indexes a
DOI some days after publication and only publication-date filters are available on the free tier; the
seen-list is what makes that overlap free.
