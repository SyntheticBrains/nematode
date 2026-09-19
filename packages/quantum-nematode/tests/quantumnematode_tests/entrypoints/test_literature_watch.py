"""The literature watch's parsing, deduplication, and failure tolerance.

Everything here is offline: no source is contacted and no model is called. The network-facing
halves are exercised by `--dry-run` against the live APIs, which is not a test's job.
"""

from __future__ import annotations

import json
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import ClassVar

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4].parent
_scripts = _REPO_ROOT / "scripts"
if not _scripts.is_dir():
    msg = f"could not locate scripts/ from {Path(__file__).resolve()}"
    raise RuntimeError(msg)
sys.path.insert(0, str(_scripts))

import literature_watch as lw  # noqa: E402  # pyright: ignore[reportMissingImports]


def _candidate(uid: str, title: str = "A title", source: str = "arxiv") -> lw.Candidate:
    return lw.Candidate(
        uid=uid,
        source=source,
        title=title,
        abstract="abstract",
        authors="Author, A.",
        published="2026-09-01",
        url=f"https://example.invalid/{uid}",
    )


class TestInvertAbstract:
    def test_rebuilds_word_order(self):
        inverted = {"learning": [1], "Connectome": [0], "rules": [2]}
        assert lw._invert_abstract(inverted) == "Connectome learning rules"

    def test_repeated_word_appears_at_each_position(self):
        assert lw._invert_abstract({"a": [0, 2], "b": [1]}) == "a b a"

    def test_missing_index_is_empty(self):
        assert lw._invert_abstract(None) == ""
        assert lw._invert_abstract({}) == ""


class TestDedupe:
    def test_drops_repeated_uid_keeping_the_first(self):
        first = _candidate("10.1/x", source="openalex-cites")
        second = _candidate("10.1/X", source="biorxiv")
        assert lw.dedupe([first, second], set()) == [first]

    def test_drops_ids_already_reported(self):
        # The seen-list comes from previous digests, where ids may carry a DOI prefix.
        seen = {"https://doi.org/10.1/X"}
        assert lw.dedupe([_candidate("10.1/x")], seen) == []

    def test_collapses_preprint_and_published_versions_by_title(self):
        preprint = _candidate("10.1101/abc", "Connectome Learning Rules!")
        published = _candidate("10.1038/xyz", "connectome learning rules")
        assert lw.dedupe([preprint, published], set()) == [preprint]

    def test_distinct_titles_both_survive(self):
        one = _candidate("10.1/a", "First paper")
        two = _candidate("10.1/b", "Second paper")
        assert lw.dedupe([one, two], set()) == [one, two]

    def test_candidate_without_an_id_is_dropped(self):
        assert lw.dedupe([_candidate("")], set()) == []


class TestOpenAlexCandidate:
    WORK: ClassVar[dict] = {
        "id": "https://openalex.org/W999",
        "doi": "https://doi.org/10.1038/test",
        "title": "A connectome result",
        "publication_date": "2026-09-02",
        "abstract_inverted_index": {"Two": [0], "words": [1]},
        "authorships": [{"author": {"display_name": "Perks, K."}}],
        "primary_location": {"source": {"display_name": "Nature"}},
        "referenced_works": [
            "https://openalex.org/W2954731284",
            "https://openalex.org/W404",
        ],
    }

    def test_maps_the_fields_the_scorer_reads(self):
        got = lw._openalex_candidate(self.WORK, {"W2954731284": "Cook et al. 2019"})
        assert got.uid == "10.1038/test"
        assert got.title == "A connectome result"
        assert got.abstract == "Two words"
        assert got.authors == "Perks, K."
        assert got.venue == "Nature"
        assert got.published == "2026-09-02"

    def test_reports_which_seed_papers_it_cites(self):
        got = lw._openalex_candidate(self.WORK, {"W2954731284": "Cook et al. 2019"})
        assert got.cites == ["Cook et al. 2019"]

    def test_falls_back_to_the_openalex_id_without_a_doi(self):
        got = lw._openalex_candidate({**self.WORK, "doi": None}, {})
        assert got.uid == "W999"

    def test_tolerates_null_nested_records(self):
        sparse = {"id": "https://openalex.org/W1", "primary_location": None, "authorships": None}
        got = lw._openalex_candidate(sparse, {})
        assert got.uid == "W1"
        assert got.venue == ""
        assert got.title == "(untitled)"


class TestArxivCandidate:
    ENTRY = """<entry xmlns="http://www.w3.org/2005/Atom">
      <id>http://arxiv.org/abs/2609.01234v1</id>
      <title>A learning
        rule</title>
      <summary>An   abstract.</summary>
      <published>2026-09-16T02:31:27Z</published>
      <author><name>Beer, R.</name></author>
      <author><name>Izquierdo, E.</name></author>
    </entry>"""

    def test_parses_and_collapses_wrapped_whitespace(self):
        got = lw._arxiv_candidate(ET.fromstring(self.ENTRY))  # noqa: S314 — fixed literal
        assert got.uid == "2609.01234v1"
        assert got.title == "A learning rule"
        assert got.abstract == "An abstract."
        assert got.authors == "Beer, R., Izquierdo, E."
        assert got.published == "2026-09-16"


class TestBiorxivCandidate:
    def test_maps_the_collection_record(self):
        got = lw._biorxiv_candidate(
            {
                "doi": "10.1101/2026.09.01.123456",
                "title": "A  preprint",
                "abstract": "Body.",
                "authors": "Cook, S.",
                "date": "2026-09-15",
                "category": "neuroscience",
                "server": "bioRxiv",
            },
            "biorxiv",
        )
        assert got.uid == "10.1101/2026.09.01.123456"
        assert got.title == "A preprint"
        assert got.url == "https://doi.org/10.1101/2026.09.01.123456"
        assert "neuroscience" in got.venue


class TestPartialPaging:
    """A page failing late must not discard the pages that already arrived."""

    def _pages(self, monkeypatch, responses):
        calls = {"n": 0}

        def fake(url):
            index = calls["n"]
            calls["n"] += 1
            response = responses[min(index, len(responses) - 1)]
            if isinstance(response, Exception):
                raise response
            return response

        monkeypatch.setattr(lw, "_http_json", fake)
        monkeypatch.setattr(lw.time, "sleep", lambda _seconds: None)
        return calls

    def test_biorxiv_keeps_the_pages_it_already_fetched(self, monkeypatch):
        full = {"collection": [{"doi": f"10.1101/{i}", "title": f"T{i}"} for i in range(30)]}
        self._pages(monkeypatch, [full, lw.SourceError("read timed out")])

        got = lw._biorxiv_collection(
            "biorxiv",
            "neuroscience",
            lw.date(2026, 9, 11),
            lw.date(2026, 9, 19),
        )
        assert len(got.candidates) == 30
        # And the shortfall is carried, not merely logged.
        assert len(got.partials) == 1
        assert "stopped at page 1" in got.partials[0]

    def test_a_first_page_failure_is_still_a_source_failure(self, monkeypatch):
        self._pages(monkeypatch, [lw.SourceError("read timed out")])
        with pytest.raises(lw.SourceError):
            lw._biorxiv_collection(
                "biorxiv",
                "neuroscience",
                lw.date(2026, 9, 11),
                lw.date(2026, 9, 19),
            )

    def test_one_collection_failing_does_not_lose_the_others(self, monkeypatch):
        def one_bad(server, category, since, until):
            if category == "neuroscience":
                msg = "read timed out"
                raise lw.SourceError(msg)
            return lw.Fetched([_candidate("10.1101/ok")])

        monkeypatch.setattr(lw, "_biorxiv_collection", one_bad)
        got = lw.fetch_biorxiv(
            {"servers": ["biorxiv"], "categories": ["neuroscience", "systems biology"]},
            lw.date(2026, 9, 19),
        )
        assert [item.uid for item in got.candidates] == ["10.1101/ok"]
        assert "neuroscience collection failed" in got.partials[0]

    def test_every_collection_failing_fails_the_source(self, monkeypatch):
        def all_bad(server, category, since, until):
            msg = "read timed out"
            raise lw.SourceError(msg)

        monkeypatch.setattr(lw, "_biorxiv_collection", all_bad)
        with pytest.raises(lw.SourceError, match="every bioRxiv collection failed"):
            lw.fetch_biorxiv(
                {"servers": ["biorxiv"], "categories": ["neuroscience", "systems biology"]},
                lw.date(2026, 9, 19),
            )


class TestPartialsReachTheReader:
    """A shortfall that only reached stderr would leave a complete-looking digest."""

    def test_openalex_treats_a_malformed_response_as_a_partial_page(self, monkeypatch):
        # Not a transport failure: the request succeeded and the body was wrong. Losing the
        # pages already fetched over it is the same waste as losing them to a timeout.
        pages = iter(
            [
                {"results": [{"id": "https://openalex.org/W1"}], "meta": {"next_cursor": "c2"}},
                {"error": "Plan upgrade required"},
            ],
        )
        monkeypatch.setattr(lw, "_http_json", lambda url: next(pages))

        got = lw.fetch_openalex({"seeds": [{"id": "W9"}]}, lw.date(2026, 9, 21))
        assert [item.uid for item in got.candidates] == ["W1"]
        assert "no results block" in got.partials[0]

    def test_arxiv_treats_a_truncated_feed_as_a_partial_page(self, monkeypatch):
        full = (
            '<feed xmlns="http://www.w3.org/2005/Atom">'
            + "".join(
                f"<entry><id>http://arxiv.org/abs/{i}</id><title>T</title></entry>"
                for i in range(lw.ARXIV_PAGE)
            )
            + "</feed>"
        )
        feeds = iter([full.encode(), b"<feed><entry>truncated"])
        monkeypatch.setattr(lw, "_http_bytes", lambda url: next(feeds))
        monkeypatch.setattr(lw.time, "sleep", lambda _seconds: None)

        got = lw.fetch_arxiv({"categories": ["q-bio.NC"]}, lw.date(2026, 9, 21))
        assert len(got.candidates) == lw.ARXIV_PAGE
        assert got.partials

    def test_a_partial_sweep_is_declared_in_the_digest(self, monkeypatch):
        monkeypatch.setattr(
            lw,
            "fetch_openalex",
            lambda config, until: lw.Fetched([_candidate("10.1/a")], ["The arXiv sweep stopped."]),
        )
        config = {"openalex": {"enabled": True, "lookback_days": 21}}

        found, failures, _earliest = lw.gather(config, lw.date(2026, 9, 21))
        assert [item.uid for item in found] == ["10.1/a"]
        assert failures == ["The arXiv sweep stopped."]
        assert "**Partial sweep.**" in lw.render(
            "body",
            [],
            {},
            failures,
            (lw.date(2026, 8, 31), lw.date(2026, 9, 21)),
        )

    def test_partials_alone_are_not_a_total_failure(self, monkeypatch):
        # `failures` now carries partial notes too, so counting it would call a run where every
        # source returned something incomplete a run where every source died.
        monkeypatch.setattr(
            lw,
            "fetch_openalex",
            lambda config, until: lw.Fetched([_candidate("10.1/a")], ["stopped early"]),
        )
        found, failures, _earliest = lw.gather(
            {"openalex": {"enabled": True, "lookback_days": 21}},
            lw.date(2026, 9, 21),
        )
        assert found
        assert failures == ["stopped early"]


class TestGather:
    """One source failing must degrade the digest, not lose the week."""

    CONFIG: ClassVar[dict] = {
        "openalex": {"enabled": True, "lookback_days": 21, "seeds": [{"id": "W1"}]},
        "arxiv": {"enabled": True, "lookback_days": 8, "categories": ["q-bio.NC"]},
        "biorxiv": {"enabled": False},
    }

    def test_surviving_source_still_produces_candidates(self, monkeypatch):
        def boom(config, until):
            msg = "upstream is down"
            raise lw.SourceError(msg)

        monkeypatch.setattr(lw, "fetch_openalex", boom)
        monkeypatch.setattr(
            lw,
            "fetch_arxiv",
            lambda config, until: lw.Fetched([_candidate("10.1/a")]),
        )

        found, failures, earliest = lw.gather(self.CONFIG, lw.date(2026, 9, 21))
        assert [item.uid for item in found] == ["10.1/a"]
        assert len(failures) == 1
        assert "citation chaining" in failures[0]
        # The window reported to the reader spans the widest source lookback.
        assert earliest == lw.date(2026, 8, 31)

    def test_every_source_failing_is_a_run_failure(self, monkeypatch):
        def boom(config, until):
            msg = "upstream is down"
            raise lw.SourceError(msg)

        monkeypatch.setattr(lw, "fetch_openalex", boom)
        monkeypatch.setattr(lw, "fetch_arxiv", boom)
        with pytest.raises(lw.SourceError, match="every configured source failed"):
            lw.gather(self.CONFIG, lw.date(2026, 9, 21))

    def test_disabled_sources_are_not_fetched(self, monkeypatch):
        called: list[str] = []
        monkeypatch.setattr(
            lw,
            "fetch_biorxiv",
            lambda config, until: called.append("biorxiv") or lw.Fetched([]),
        )
        monkeypatch.setattr(lw, "fetch_openalex", lambda config, until: lw.Fetched([]))
        monkeypatch.setattr(lw, "fetch_arxiv", lambda config, until: lw.Fetched([]))
        lw.gather(self.CONFIG, lw.date(2026, 9, 21))
        assert called == []


class TestSweepCache:
    """A saved sweep lets a local run score without paying the several-minute fetch again."""

    def test_round_trips_candidates_and_window(self, tmp_path):
        original = lw.Sweep(
            candidates=[
                lw.Candidate(
                    uid="10.1/a",
                    source="openalex-cites",
                    title="A paper",
                    abstract="Body.",
                    authors="Cook, S.",
                    published="2026-09-02",
                    url="https://doi.org/10.1/a",
                    venue="Nature",
                    cites=["Cook et al. 2019"],
                ),
            ],
            failures=["The arXiv source failed (boom)."],
            earliest=lw.date(2026, 8, 31),
            until=lw.date(2026, 9, 21),
        )
        path = tmp_path / "candidates.json"
        lw.save_candidates(path, original)
        assert lw.load_candidates(path) == original

    def test_a_cached_sweep_still_honours_a_grown_seen_list(self, tmp_path):
        # The point of re-scoring a cache is usually that something changed since the fetch.
        sweep = lw.Sweep(
            [_candidate("10.1/a"), _candidate("10.1/b", "Other")],
            [],
            lw.date(2026, 9, 1),
            lw.date(2026, 9, 21),
        )
        path = tmp_path / "candidates.json"
        lw.save_candidates(path, sweep)
        reloaded = lw.load_candidates(path)
        assert [item.uid for item in lw.dedupe(reloaded.candidates, {"10.1/a"})] == ["10.1/b"]


class TestRender:
    WINDOW = (lw.date(2026, 8, 31), lw.date(2026, 9, 21))

    def test_embeds_the_reported_ids_for_the_next_run(self):
        selected = [_candidate("10.1/a"), _candidate("W7", "Another")]
        out = lw.render("body", selected, {"fetched": 9, "candidates": 5}, [], self.WINDOW)
        assert "<!-- literature-watch-reported: 10.1/a W7 -->" in out

    def test_titles_the_digest_by_iso_week(self):
        out = lw.render("body", [], {}, [], self.WINDOW)
        assert out.startswith("# Literature watch — 2026-W39")

    def test_a_partial_sweep_says_so(self):
        out = lw.render("body", [], {}, ["The arXiv source failed (boom)."], self.WINDOW)
        assert "**Partial sweep.**" in out
        assert "The arXiv source failed" in out

    def test_an_empty_week_still_emits_a_marker(self):
        out = lw.render("nothing", [], {}, [], self.WINDOW)
        assert "<!-- literature-watch-reported:  -->" in out


class TestLoadBrief:
    def test_resolves_context_files_relative_to_the_config(self, tmp_path):
        (tmp_path / "context.md").write_text("the brief", encoding="utf-8")
        got = lw.load_brief({"context_files": ["context.md"]}, tmp_path / "seeds.toml")
        assert got == "the brief"

    def test_a_missing_context_file_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            lw.load_brief({"context_files": ["gone.md"]}, tmp_path / "seeds.toml")

    def test_no_context_at_all_is_an_error(self, tmp_path):
        # Without a brief the scorer has no definition of relevance and would score noise.
        with pytest.raises(ValueError, match="no context files"):
            lw.load_brief({}, tmp_path / "seeds.toml")


class TestShippedConfig:
    """The committed config must stay loadable and consistent with the script's expectations."""

    CONFIG = lw.DEFAULT_CONFIG

    def test_config_exists_and_parses(self):
        assert self.CONFIG.is_file()
        tomllib.loads(self.CONFIG.read_text(encoding="utf-8"))

    def test_context_files_resolve(self):
        config = tomllib.loads(self.CONFIG.read_text(encoding="utf-8"))
        assert lw.load_brief(config.get("scoring", {}), self.CONFIG).strip()

    def test_every_seed_has_an_openalex_work_id(self):
        config = tomllib.loads(self.CONFIG.read_text(encoding="utf-8"))
        seeds = config["openalex"]["seeds"]
        assert seeds
        for seed in seeds:
            assert seed["id"].startswith("W"), seed
            assert seed["id"][1:].isdigit(), seed
            assert seed["label"]

    def test_models_are_current_ids(self):
        config = tomllib.loads(self.CONFIG.read_text(encoding="utf-8"))
        scoring = config["scoring"]
        assert scoring["triage_model"] == "claude-haiku-4-5"
        assert scoring["digest_model"] == "claude-opus-5"


class TestMainWiring:
    """`main` against the committed config, with the two model calls stubbed.

    The unit tests above each exercise one function with arguments chosen by hand, which cannot
    catch a caller passing the wrong slice of the config — the failure this covers.
    """

    def test_a_cached_sweep_produces_a_digest(self, tmp_path, monkeypatch, capsys):
        cache = tmp_path / "candidates.json"
        lw.save_candidates(
            cache,
            lw.Sweep(
                [
                    _candidate("10.1/a", "A relevant paper"),
                    _candidate("10.1/b", "An irrelevant one"),
                ],
                [],
                lw.date(2026, 9, 1),
                lw.date(2026, 9, 21),
            ),
        )
        seen_brief: list[str] = []

        def fake_score(candidates, brief, config):
            seen_brief.append(brief)
            return {"10.1/a": 3, "10.1/b": 0}

        def fake_digest(selected, brief, config):
            return "### A relevant paper\n\nIt bears on the open question."

        monkeypatch.setattr(lw, "score", fake_score)
        monkeypatch.setattr(lw, "write_digest", fake_digest)

        out = tmp_path / "digest.md"
        assert lw.main(["--candidates-in", str(cache), "--out", str(out)]) == 0

        digest = out.read_text(encoding="utf-8")
        assert digest.startswith("# Literature watch — 2026-W39")
        assert "A relevant paper" in digest
        # Only the item above threshold is reported, and the marker carries it for the next run.
        assert "<!-- literature-watch-reported: 10.1/a -->" in digest
        # The committed brief reached the scorer, rather than an empty string from a wrong lookup.
        assert "What is open right now" in seen_brief[0]


class TestScoreSchema:
    def test_schema_is_strict_enough_to_parse_blind(self):
        # The scorer reads the response without a validation pass of its own, so the schema has to
        # pin the shape: closed objects, required keys, and a bounded score.
        item = lw.SCORE_SCHEMA["properties"]["scores"]["items"]
        assert item["additionalProperties"] is False
        assert item["required"] == ["index", "score", "why"]
        assert item["properties"]["score"]["enum"] == [0, 1, 2, 3]

    def test_schema_is_serialisable(self):
        json.dumps(lw.SCORE_SCHEMA)
