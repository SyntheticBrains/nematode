"""Weekly literature watch: gather new work, score it for relevance, write a digest.

Three sources, in descending order of precision:

1. **Citation chaining** (OpenAlex) — new work citing a curated seed set. Almost everything that
   cites the *C. elegans* connectome papers is on-topic, whereas a keyword search for "C. elegans"
   returns mostly wet-lab molecular biology. This is the channel worth reading first.
2. **arXiv** — `q-bio.NC` and `cs.NE` submissions in the window.
3. **bioRxiv** — the configured subject collections in the window.

The raw sweep is a few hundred abstracts a week, which is too many to read and mostly irrelevant.
A cheap model scores every candidate against a hand-maintained project brief, and a capable model
writes the digest for the survivors. The brief is what makes the scores worth anything: it names
the open questions, so relevance is judged against the work actually in front of the project rather
than against a fixed keyword list.

Sources fail independently. One source erroring out degrades the digest rather than losing the week
— the run only fails when every source fails.

Two things are deliberately *not* automatic. The seed list and the brief are hand-maintained: a
stale seed set is how a watch like this quietly stops finding anything, and that is a judgement
call, not a scrape. And deduplication is caller-supplied (``--seen-file``), because what counts as
already-reported depends on where the digest is delivered.

Usage::

    # Offline: fetch and list candidates, no model calls, no API key needed.
    uv run python scripts/literature_watch.py --dry-run

    # Full run, writing a Markdown digest.
    ANTHROPIC_API_KEY=... uv run --no-project --with anthropic python
        scripts/literature_watch.py --out digest.md --seen-file seen.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "docs" / "research" / "literature-watch" / "seeds.toml"

# The API asks for a contact address so it can reach heavy users; supplying one also moves the
# request into a faster pool. Kept in the environment rather than the config because this is a
# public repository and an address in a tracked file is an address in a scraper's next crawl.
OPENALEX_MAILTO_ENV = "OPENALEX_MAILTO"

USER_AGENT = "nematode-literature-watch/1.0 (+https://github.com/SyntheticBrains/nematode)"
HTTP_TIMEOUT = 60.0
HTTP_RETRIES = 3
HTTP_BACKOFF = 4.0

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "os": "http://a9.com/-/spec/opensearch/1.1/"}
# arXiv asks callers to leave three seconds between requests.
ARXIV_DELAY = 3.0
ARXIV_PAGE = 100
BIORXIV_PAGE = 30
# bioRxiv pages thirty records at a time, so a week of one collection is a dozen-odd requests
# back to back; a short pause between them keeps the API from timing out partway through.
BIORXIV_DELAY = 1.0
# Guards against an unbounded crawl if a window is opened far too wide by hand.
MAX_PAGES = 60

ABSTRACT_CHARS = 1200


@dataclass
class Candidate:
    """One paper found by one source, in the shape the scorer reads."""

    uid: str
    source: str
    title: str
    abstract: str
    authors: str
    published: str
    url: str
    venue: str = ""
    # Which seed papers this work cites, for the citation-chaining source.
    cites: list[str] = field(default_factory=list)

    def prompt_block(self, index: int) -> str:
        """Render this candidate for the scoring prompt."""
        lines = [
            f"[{index}] {self.title}",
            f"    source: {self.source}  date: {self.published}  authors: {self.authors}",
        ]
        if self.venue:
            lines.append(f"    venue: {self.venue}")
        if self.cites:
            lines.append(f"    cites seed work: {'; '.join(self.cites)}")
        lines.append(f"    abstract: {self.abstract[:ABSTRACT_CHARS] or '(none available)'}")
        return "\n".join(lines)


@dataclass
class Sweep:
    """One window's fetch: what was found, what failed, and the window it covers."""

    candidates: list[Candidate]
    failures: list[str]
    earliest: date
    until: date


class SourceError(RuntimeError):
    """A source could not be fetched; the run continues without it."""


def _http_json(url: str) -> dict[str, Any]:
    """GET a URL and parse the response as JSON, retrying transient failures."""
    return json.loads(_http_bytes(url))


def _http_bytes(url: str) -> bytes:
    """GET a URL, retrying transient failures with a widening delay."""
    last: Exception | None = None
    for attempt in range(HTTP_RETRIES):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 — https URLs built from the config, not from user input
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:  # noqa: S310 — same
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt < HTTP_RETRIES - 1:
                time.sleep(HTTP_BACKOFF * (attempt + 1))
    msg = f"{url} failed after {HTTP_RETRIES} attempts: {last}"
    raise SourceError(msg)


def _partial(label: str, page: int, exc: Exception, kept: int) -> None:
    """Report a page that could not be fetched, keeping the pages already collected.

    An unattended weekly job that drops a whole source because its ninth page timed out has
    thrown away eight good pages. Whatever arrived is worth more than the shortfall costs.
    """
    print(
        f"warning: {label} stopped at page {page} ({exc}); keeping {kept} records already fetched",
        file=sys.stderr,
    )


def _invert_abstract(inverted: dict[str, list[int]] | None) -> str:
    """Rebuild plain text from OpenAlex's inverted abstract index."""
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = [
        (position, word) for word, spots in inverted.items() for position in spots
    ]
    positions.sort()
    return " ".join(word for _, word in positions)


def fetch_openalex(config: dict[str, Any], until: date) -> list[Candidate]:
    """Fetch new work citing any of the seed papers."""
    seeds: list[dict[str, str]] = config.get("seeds", [])
    if not seeds:
        return []
    labels = {seed["id"]: seed.get("label", seed["id"]) for seed in seeds}
    since = until - timedelta(days=int(config.get("lookback_days", 21)))

    params = {
        "filter": (
            f"cites:{'|'.join(labels)},"
            f"from_publication_date:{since.isoformat()},"
            f"to_publication_date:{until.isoformat()}"
        ),
        "select": "id,doi,title,publication_date,primary_location,abstract_inverted_index,"
        "authorships,referenced_works,type",
        "per-page": "200",
    }
    mailto = os.environ.get(OPENALEX_MAILTO_ENV)
    if mailto:
        params["mailto"] = mailto

    found: list[Candidate] = []
    cursor = "*"
    for page_number in range(MAX_PAGES):
        try:
            page = _http_json(
                "https://api.openalex.org/works?"
                + urllib.parse.urlencode({**params, "cursor": cursor}),
            )
        except SourceError as exc:
            if not found:
                raise
            _partial("citation chaining", page_number, exc, len(found))
            break
        if "results" not in page:
            msg = f"OpenAlex returned no results block: {str(page)[:200]}"
            raise SourceError(msg)
        found.extend(_openalex_candidate(work, labels) for work in page["results"])
        cursor = page.get("meta", {}).get("next_cursor")
        if not cursor or not page["results"]:
            break
    return found


def _openalex_candidate(work: dict[str, Any], labels: dict[str, str]) -> Candidate:
    """Convert one OpenAlex work record into a candidate."""
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    authors = [
        (authorship.get("author") or {}).get("display_name", "")
        for authorship in (work.get("authorships") or [])[:6]
    ]
    referenced = work.get("referenced_works") or []
    cited_seeds = [
        labels[ref.rsplit("/", 1)[-1]] for ref in referenced if ref.rsplit("/", 1)[-1] in labels
    ]
    doi = work.get("doi") or ""
    return Candidate(
        uid=doi.removeprefix("https://doi.org/") or work["id"].rsplit("/", 1)[-1],
        source="openalex-cites",
        title=work.get("title") or "(untitled)",
        abstract=_invert_abstract(work.get("abstract_inverted_index")),
        authors=", ".join(name for name in authors if name),
        published=work.get("publication_date", ""),
        url=doi or work["id"],
        venue=source.get("display_name", ""),
        cites=cited_seeds,
    )


def fetch_arxiv(config: dict[str, Any], until: date) -> list[Candidate]:
    """Fetch arXiv submissions in the configured categories and window."""
    categories: list[str] = config.get("categories", [])
    if not categories:
        return []
    since = until - timedelta(days=int(config.get("lookback_days", 8)))
    category_clause = " OR ".join(f"cat:{category}" for category in categories)
    query = (
        f"({category_clause}) AND "
        f"submittedDate:[{since.strftime('%Y%m%d')}0000 TO {until.strftime('%Y%m%d')}2359]"
    )

    found: list[Candidate] = []
    for page in range(MAX_PAGES):
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {
                "search_query": query,
                "start": page * ARXIV_PAGE,
                "max_results": ARXIV_PAGE,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        try:
            feed = _http_bytes(url).decode("utf-8")
        except SourceError as exc:
            if not found:
                raise
            _partial("arXiv", page, exc, len(found))
            break
        entries = ET.fromstring(feed).findall("atom:entry", ARXIV_NS)  # noqa: S314 — arXiv's own Atom feed over https
        found.extend(_arxiv_candidate(entry) for entry in entries)
        if len(entries) < ARXIV_PAGE:
            break
        time.sleep(ARXIV_DELAY)
    return found


def _arxiv_candidate(entry: ET.Element) -> Candidate:
    """Convert one arXiv Atom entry into a candidate."""

    def text(tag: str) -> str:
        return " ".join((entry.findtext(tag, default="", namespaces=ARXIV_NS) or "").split())

    authors = [
        " ".join((author.findtext("atom:name", default="", namespaces=ARXIV_NS) or "").split())
        for author in entry.findall("atom:author", ARXIV_NS)[:6]
    ]
    url = text("atom:id")
    return Candidate(
        uid=url.rsplit("/", 1)[-1],
        source="arxiv",
        title=text("atom:title"),
        abstract=text("atom:summary"),
        authors=", ".join(name for name in authors if name),
        published=text("atom:published")[:10],
        url=url,
        venue="arXiv preprint",
    )


def fetch_biorxiv(config: dict[str, Any], until: date) -> list[Candidate]:
    """Fetch preprints posted to the configured servers and collections in the window."""
    servers: list[str] = config.get("servers", [])
    categories: list[str] = config.get("categories", [])
    if not servers or not categories:
        return []
    since = until - timedelta(days=int(config.get("lookback_days", 8)))

    # Thirty records a page over several collections is a lot of requests, and the API starts
    # timing out under sustained paging — so each collection is allowed to fail on its own.
    found: list[Candidate] = []
    failed = 0
    collections = [(server, category) for server in servers for category in categories]
    for server, category in collections:
        try:
            found.extend(_biorxiv_collection(server, category, since, until))
        except SourceError as exc:
            failed += 1
            print(f"warning: bioRxiv {category} failed: {exc}", file=sys.stderr)
    if failed == len(collections):
        msg = f"every bioRxiv collection failed ({failed})"
        raise SourceError(msg)
    return found


def _biorxiv_collection(server: str, category: str, since: date, until: date) -> list[Candidate]:
    """Page through one server's subject collection for the window."""
    found: list[Candidate] = []
    for page in range(MAX_PAGES):
        url = (
            f"https://api.biorxiv.org/details/{server}/"
            f"{since.isoformat()}/{until.isoformat()}/{page * BIORXIV_PAGE}"
            f"?{urllib.parse.urlencode({'category': category})}"
        )
        try:
            collection = _http_json(url).get("collection") or []
        except SourceError as exc:
            if not found:
                raise
            _partial(f"bioRxiv {category}", page, exc, len(found))
            break
        found.extend(_biorxiv_candidate(item, server) for item in collection)
        if len(collection) < BIORXIV_PAGE:
            break
        time.sleep(BIORXIV_DELAY)
    return found


def _biorxiv_candidate(item: dict[str, Any], server: str) -> Candidate:
    """Convert one bioRxiv collection record into a candidate."""
    doi = item.get("doi", "")
    return Candidate(
        uid=doi,
        source=server,
        title=" ".join((item.get("title") or "").split()),
        abstract=" ".join((item.get("abstract") or "").split()),
        authors=item.get("authors", ""),
        published=item.get("date", ""),
        url=f"https://doi.org/{doi}" if doi else "",
        venue=f"{item.get('server', server)} preprint ({item.get('category', '')})",
    )


def _title_key(title: str) -> str:
    """Build a loose title key, so a preprint and its published version collapse into one."""
    return re.sub(r"[^a-z0-9]+", "", title.lower())[:80]


def dedupe(candidates: list[Candidate], seen: set[str]) -> list[Candidate]:
    """Drop already-reported work, exact repeats, and preprint/publication pairs.

    Sources overlap by design — a preprint reached through citation chaining is usually also in
    that week's bioRxiv sweep. The first occurrence wins, and sources are fetched in precision
    order, so the surviving copy carries the citation-chaining metadata when there is any.
    """
    normalised_seen = {value.lower().removeprefix("https://doi.org/") for value in seen}
    kept: list[Candidate] = []
    uids: set[str] = set()
    titles: set[str] = set()
    for candidate in candidates:
        uid = candidate.uid.lower()
        title_key = _title_key(candidate.title)
        if not uid or uid in uids or uid in normalised_seen:
            continue
        if title_key and title_key in titles:
            continue
        uids.add(uid)
        if title_key:
            titles.add(title_key)
        kept.append(candidate)
    return kept


SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "integer", "enum": [0, 1, 2, 3]},
                    "why": {"type": "string"},
                },
                "required": ["index", "score", "why"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scores"],
    "additionalProperties": False,
}

SCORE_SYSTEM = """You triage new research papers for one specific research project.

The project brief below states what the project is and which questions are open in front of it.
Score each candidate on how much it bears on THAT work — not on how good or how interesting the
paper is in general.

- 3: directly bears on an open question named in the brief. A result the project would have to cite,
     replicate, or answer to. Rare — most weeks have none.
- 2: relevant method, dataset, negative result, or organism the project could use or would want to
     know about, without settling anything named in the brief.
- 1: adjacent. Same broad field, no usable bearing.
- 0: unrelated, or related only by a keyword coincidence (a C. elegans molecular-biology paper with
     no circuit, behaviour, or learning content scores 0).

Be strict. A digest of four real items is worth more than twenty hedged ones. Score every candidate
you are given exactly once, by its index.

--- PROJECT BRIEF ---
{brief}
--- END BRIEF ---"""

DIGEST_SYSTEM = """You write a weekly research digest for one specific research project.

The project brief below states what the project is and which questions are open. For each paper you
are given, write an entry of at most three sentences: what the paper found, and what it would change
for THIS project — a rung in the plan it informs, a prediction it tests, a result worth citing, a
claim it undercuts. If a paper turns out not to bear on the work once you read the abstract
properly, say so in one line rather than inflating it; the triage pass is fallible and saying "this
one is thinner than its score suggests" is useful.

Write in Markdown. Use `### <title>` for each entry, followed by an italic line with the authors,
date, venue and link, then the prose. Order by importance to the project, most important first. Do
not add a preamble, a conclusion, or a summary section — entries only.

--- PROJECT BRIEF ---
{brief}
--- END BRIEF ---"""


def _client() -> Any:  # noqa: ANN401 — the SDK client type is not imported at module scope
    """Build an Anthropic client, importing the SDK only when a run actually calls the API."""
    import anthropic

    return anthropic.Anthropic()


def score(candidates: list[Candidate], brief: str, config: dict[str, Any]) -> dict[str, int]:
    """Score every candidate for relevance, returning uid -> score."""
    client = _client()
    model = config.get("triage_model", "claude-haiku-4-5")
    batch_size = int(config.get("batch_size", 40))
    scores: dict[str, int] = {}

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        listing = "\n\n".join(item.prompt_block(index) for index, item in enumerate(batch))
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            system=SCORE_SYSTEM.format(brief=brief),
            messages=[{"role": "user", "content": listing}],
            output_config={"format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
        )
        text = next((block.text for block in response.content if block.type == "text"), "")
        for entry in json.loads(text).get("scores", []):
            index = entry.get("index")
            if isinstance(index, int) and 0 <= index < len(batch):
                scores[batch[index].uid] = int(entry.get("score", 0))

    # A candidate the model skipped is not silently dropped: it keeps a score of 0 and shows up in
    # the run's counts, so a systematically failing batch is visible rather than invisible.
    return scores


def write_digest(selected: list[Candidate], brief: str, config: dict[str, Any]) -> str:
    """Write the prose digest for the selected papers."""
    if not selected:
        return "_Nothing this week cleared the relevance threshold._\n"
    client = _client()
    listing = "\n\n".join(
        f"{item.prompt_block(index)}\n    link: {item.url}" for index, item in enumerate(selected)
    )
    response = client.messages.create(
        model=config.get("digest_model", "claude-opus-5"),
        max_tokens=16000,
        system=DIGEST_SYSTEM.format(brief=brief),
        messages=[{"role": "user", "content": listing}],
    )
    return "\n".join(block.text for block in response.content if block.type == "text")


def render(
    body: str,
    selected: list[Candidate],
    counts: dict[str, int],
    failures: list[str],
    window: tuple[date, date],
) -> str:
    """Assemble the final Markdown digest."""
    since, until = window
    lines = [
        f"# Literature watch — {until.isocalendar().year}-W{until.isocalendar().week:02d}",
        "",
        (
            f"Window ending {until.isoformat()} (source lookbacks vary; earliest "
            f"{since.isoformat()}). Swept {counts.get('fetched', 0)} records, "
            f"{counts.get('candidates', 0)} after dedupe, "
            f"{counts.get('selected', 0)} above threshold."
        ),
        "",
    ]
    if failures:
        lines += ["> **Partial sweep.** " + " ".join(failures), ""]
    lines += [body.strip(), ""]
    if selected:
        lines += [
            "---",
            "",
            "<details><summary>Everything above threshold, as links</summary>",
            "",
            *[
                f"- [{item.title}]({item.url}) — {item.source}, {item.published}"
                for item in selected
            ],
            "",
            "</details>",
            "",
        ]
    # Machine-readable, so the next run can skip what this one already reported.
    lines.append(f"<!-- literature-watch-reported: {' '.join(item.uid for item in selected)} -->")
    return "\n".join(lines) + "\n"


def load_brief(config: dict[str, Any], config_path: Path) -> str:
    """Concatenate the hand-maintained context files that define relevance."""
    parts: list[str] = []
    for name in config.get("context_files", []):
        path = (config_path.parent / name).resolve() if not Path(name).is_absolute() else Path(name)
        if not path.is_file():
            msg = f"context file not found: {path}"
            raise FileNotFoundError(msg)
        parts.append(path.read_text(encoding="utf-8"))
    if not parts:
        msg = "no context files configured; the scorer has nothing to judge relevance against"
        raise ValueError(msg)
    return "\n\n".join(parts)


def gather(config: dict[str, Any], until: date) -> tuple[list[Candidate], list[str], date]:
    """Fetch every enabled source, tolerating individual failures."""
    fetchers = (
        ("citation chaining", fetch_openalex, config.get("openalex", {})),
        ("arXiv", fetch_arxiv, config.get("arxiv", {})),
        ("bioRxiv", fetch_biorxiv, config.get("biorxiv", {})),
    )
    found: list[Candidate] = []
    failures: list[str] = []
    attempted = 0
    earliest = until
    for label, fetcher, source_config in fetchers:
        if not source_config.get("enabled", False):
            continue
        attempted += 1
        earliest = min(earliest, until - timedelta(days=int(source_config.get("lookback_days", 8))))
        try:
            got = fetcher(source_config, until)
        except (SourceError, ET.ParseError, json.JSONDecodeError, KeyError) as exc:
            failures.append(f"The {label} source failed ({exc}); this digest is missing it.")
            print(f"warning: {label} failed: {exc}", file=sys.stderr)
            continue
        print(f"{label}: {len(got)} records", file=sys.stderr)
        found.extend(got)
    if attempted and len(failures) == attempted:
        msg = "every configured source failed"
        raise SourceError(msg)
    return found, failures, earliest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML config path")
    parser.add_argument("--until", type=date.fromisoformat, help="window end (default: today, UTC)")
    parser.add_argument("--out", type=Path, help="write the digest here (default: stdout)")
    parser.add_argument(
        "--seen-file",
        type=Path,
        help="newline- or space-separated ids already reported; they are excluded",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and list candidates without calling the model (no API key needed)",
    )
    parser.add_argument(
        "--candidates-out",
        type=Path,
        help="save the fetched candidates here, for a later --candidates-in run",
    )
    parser.add_argument(
        "--candidates-in",
        type=Path,
        help="score these saved candidates instead of fetching (the sweep takes minutes)",
    )
    return parser.parse_args(argv)


def save_candidates(path: Path, sweep: Sweep) -> None:
    """Save a sweep so it can be scored later without fetching again."""
    path.write_text(
        json.dumps(
            {
                "until": sweep.until.isoformat(),
                "earliest": sweep.earliest.isoformat(),
                "failures": sweep.failures,
                "candidates": [asdict(item) for item in sweep.candidates],
            },
            indent=1,
        ),
        encoding="utf-8",
    )


def load_candidates(path: Path) -> Sweep:
    """Reload a saved sweep, window and all, so the digest header stays honest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Sweep(
        candidates=[Candidate(**item) for item in payload["candidates"]],
        failures=payload.get("failures", []),
        earliest=date.fromisoformat(payload["earliest"]),
        until=date.fromisoformat(payload["until"]),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the watch end to end."""
    args = parse_args(argv)
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    until = args.until or datetime.now(tz=UTC).date()

    seen: set[str] = set()
    if args.seen_file and args.seen_file.is_file():
        seen = set(args.seen_file.read_text(encoding="utf-8").split())

    if args.candidates_in:
        sweep = load_candidates(args.candidates_in)
        found, failures, earliest, until = (
            sweep.candidates,
            sweep.failures,
            sweep.earliest,
            sweep.until,
        )
        print(f"loaded {len(found)} candidates from {args.candidates_in}", file=sys.stderr)
    else:
        found, failures, earliest = gather(config, until)
    # Applied on both paths: a saved sweep re-scored later must still honour a grown seen-list.
    candidates = dedupe(found, seen)
    counts = {"fetched": len(found), "candidates": len(candidates)}
    print(
        f"{len(candidates)} candidates after dedupe ({len(seen)} ids already reported)",
        file=sys.stderr,
    )

    scoring = config.get("scoring", {})
    limit = int(scoring.get("max_candidates", 600))
    if len(candidates) > limit:
        print(f"capping at {limit} candidates", file=sys.stderr)
        candidates = candidates[:limit]

    if args.candidates_out:
        save_candidates(args.candidates_out, Sweep(candidates, failures, earliest, until))
        print(f"saved {len(candidates)} candidates to {args.candidates_out}", file=sys.stderr)

    if args.dry_run:
        for item in candidates:
            print(f"{item.source}\t{item.published}\t{item.title}")
        return 0

    brief = load_brief(scoring, args.config)
    scores = score(candidates, brief, scoring)
    threshold = int(scoring.get("keep_threshold", 2))
    selected = sorted(
        (item for item in candidates if scores.get(item.uid, 0) >= threshold),
        key=lambda item: -scores.get(item.uid, 0),
    )[: int(scoring.get("max_digest_items", 8))]
    counts["selected"] = len(selected)
    print(f"{len(selected)} above threshold {threshold}", file=sys.stderr)

    digest = render(
        write_digest(selected, brief, scoring),
        selected,
        counts,
        failures,
        (earliest, until),
    )
    if args.out:
        args.out.write_text(digest, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
