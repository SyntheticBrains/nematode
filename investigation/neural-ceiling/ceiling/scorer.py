"""The guarded scorer: the only module that loads S (`measured_stams.test`).

Every access goes through `authorise`, which refuses unless --confirm-registered-read was passed,
the registration file is tracked, unmodified and in the git log, and the quantity has not been read
before (a repeat needs a named, committed amendment). Every access is appended to READS.md with its
time, git HEAD and what was read. S itself never leaves this module: callers get the finite-entry
pattern, the section 3 summaries, or the registered scores.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from ceiling import config as C


class Refusal(RuntimeError):
    """The scorer will not read S."""


# ---- The guard -----------------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


def guard(confirm: bool, repo: Path = C.REPO, registration: Path = C.REGISTRATION) -> dict:
    """Refuse unless the flag is passed and the registration is tracked, unchanged and committed."""
    if not confirm:
        raise Refusal("refused: --confirm-registered-read was not passed")
    repo, registration = Path(repo).resolve(), Path(registration).resolve()
    rel = str(registration.relative_to(repo))
    if not registration.is_file():
        raise Refusal(f"refused: {rel} does not exist")
    if _git(repo, "ls-files", "--error-unmatch", "--", rel).returncode != 0:
        raise Refusal(f"refused: {rel} is not tracked")
    status = _git(repo, "status", "--porcelain", "--", rel)
    if status.returncode != 0 or status.stdout.strip():
        raise Refusal(f"refused: {rel} differs from its committed version: {status.stdout.strip()!r}")
    log = _git(repo, "log", "-1", "--format=%H", "--", rel)
    if log.returncode != 0 or not log.stdout.strip():
        raise Refusal(f"refused: git log shows no commit of {rel}")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    blob = _git(repo, "rev-parse", f"HEAD:{rel}").stdout.strip()
    return {"head": head, "registration": rel, "registration_commit": log.stdout.strip(),
            "registration_blob": blob, "code": _code_state(repo)}


def _code_state(repo: Path) -> str:
    """Whether this spike's code differs from HEAD (HEAD alone would not show it)."""
    try:
        rel = str(C.HERE.relative_to(repo))
    except ValueError:
        return "not in this repository"
    changed = _git(repo, "status", "--porcelain", "--", rel).stdout.splitlines()
    return "clean at HEAD" if not changed else f"{len(changed)} uncommitted change(s) under {rel}"


def _amendment_ok(repo: Path, sha: str, registration: Path) -> bool:
    """A committed amendment: a commit in HEAD's history that touches the registration's folder."""
    repo = Path(repo).resolve()
    folder = str(Path(registration).resolve().parent.relative_to(repo))
    if _git(repo, "cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        return False
    if _git(repo, "merge-base", "--is-ancestor", sha, "HEAD").returncode != 0:
        return False
    touched = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", sha).stdout
    return any(line.startswith(folder + "/") for line in touched.splitlines())


_ENTRY = re.compile(r"^- quantity: `([^`]+)`$")
_AMEND = re.compile(r"^- amendment: `([0-9a-f]+)`$")


def prior_reads(reads: Path) -> list[dict]:
    """The quantities (and their amendments) already logged in READS.md."""
    if not Path(reads).exists():
        return []
    out: list[dict] = []
    for line in Path(reads).read_text().splitlines():
        if match := _ENTRY.match(line):
            out.append({"quantity": match.group(1), "amendment": None})
        elif (match := _AMEND.match(line)) and out:
            out[-1]["amendment"] = match.group(1)
    return out


def authorise(quantity: str, confirm: bool, amendment: str | None = None, repo: Path = C.REPO,
              registration: Path = C.REGISTRATION, reads: Path = C.READS_LOG) -> dict:
    info = guard(confirm, repo, registration)
    prior = [p for p in prior_reads(reads) if p["quantity"] == quantity]
    if prior:
        if amendment is None:
            raise Refusal(f"refused: {quantity!r} was already read; name a committed amendment")
        if not _amendment_ok(repo, amendment, registration):
            raise Refusal(f"refused: {amendment} is not a committed amendment of the registration")
        if any(p["amendment"] == amendment for p in prior):
            raise Refusal(f"refused: amendment {amendment} was already used for {quantity!r}")
    info["amendment"] = amendment
    return info


def log_read(quantity: str, what: str, info: dict, reads: Path = C.READS_LOG) -> None:
    reads = Path(reads)
    now = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [] if reads.exists() else [
        "# Reads of S for launch record 135", "",
        "Appended by `investigation/neural-ceiling/ceiling/scorer.py` on every access to S. Each",
        "registered reading is made once; a repeat names a committed amendment.", ""]
    lines += [f"## {quantity}, {now}", "",
              f"- quantity: `{quantity}`",
              f"- time: {now}",
              f"- HEAD: `{info['head']}`",
              f"- registration: `{info['registration']}` at commit `{info['registration_commit']}`,"
              f" blob `{info['registration_blob']}`"]
    lines.append(f"- amendment: `{info['amendment']}`" if info.get("amendment") else "- amendment: none")
    lines.append(f"- code: {info.get('code', 'unknown')}")
    lines += [f"- read: {what}", ""]
    with open(reads, "a") as handle:
        handle.write("\n".join(lines) + "\n")


def _load_S(path: Path, expected_sha256: str) -> np.ndarray:
    from ceiling.inputs import file_sha256

    found = file_sha256(path)
    if found != expected_sha256:
        raise Refusal(f"refused: {path} has SHA-256 {found}, expected {expected_sha256}")
    with zipfile.ZipFile(path) as archive:
        S = np.array(np.lib.format.read_array(io.BytesIO(archive.read(C.TEST_MEMBER))), dtype=np.float64)
    np.fill_diagonal(S, np.nan)
    return S


def check_scoring_inputs(pooled: dict, oracle: dict, out_dir: Path) -> None:
    """Everything the registered reading needs, checked before S is read (a failure after the
    read would spend the once-only reading)."""
    problems = []
    if not (out_dir / "entry_sets.npz").exists():
        problems.append("entry_sets.npz is missing")
    if "worms" not in pooled:
        problems.append("no held-out-worm predictions")
    for split, arms in pooled.items():
        for arm in ("masked", "floor-C"):
            if arm not in arms:
                problems.append(f"{split}: no {arm!r} arm")
    gates = oracle.get("gates", {})
    for name, splits in (("K", ("stimulations", "neurons")), ("P", ("stimulations", "neurons", "worms")),
                         ("N", ("stimulations", "neurons"))):
        missing = [s for s in splits if s not in gates.get(name, {})]
        if missing:
            problems.append(f"oracle {name} has no gate for {missing}")
    if problems:
        raise Refusal("refused before reading S: " + "; ".join(problems))


class Scorer:
    """One guarded session. Paths are parameters so tests can run on a temporary repository and a
    synthetic matrix file."""

    def __init__(self, confirm: bool, amendment: str | None = None, repo: Path = C.REPO,
                 registration: Path = C.REGISTRATION, reads: Path = C.READS_LOG,
                 stams: Path = C.STAMS_PATH, stams_sha256: str = C.STAMS_SHA256) -> None:
        self.confirm, self.amendment = confirm, amendment
        self.repo, self.registration, self.reads = repo, registration, reads
        self.stams, self.stams_sha256 = stams, stams_sha256

    def _read(self, quantity: str, what: str) -> tuple[np.ndarray, dict]:
        info = authorise(quantity, self.confirm, self.amendment, self.repo, self.registration, self.reads)
        S = _load_S(self.stams, self.stams_sha256)
        log_read(quantity, what, info, self.reads)
        return S, info

    # ---- the three registered quantities ----

    def entry_sets(self, T: np.ndarray, layout, out_dir: Path) -> dict:
        """S's finite-entry pattern and every split's entry set; no S value leaves."""
        from ceiling import splits

        S, info = self._read("entry_sets", "S's finite-entry pattern only (no values): the entry "
                             "sets of all three splits and the pattern for the oracles' S_syn")
        pattern = np.isfinite(S)
        del S
        sets = splits.entry_sets(np.isfinite(T), pattern, layout)
        report = splits.entry_set_report(np.isfinite(T), pattern, layout)
        out_dir = Path(out_dir)
        arrays = {"s_finite": pattern}
        for split, masks in sets.items():
            for f, mask in enumerate(masks):
                arrays[f"{split}/{f}"] = mask
        np.savez_compressed(out_dir / "entry_sets.npz", **arrays)
        (out_dir / "entry_sets.json").write_text(json.dumps({"read": info, "report": report}, indent=1))
        return report

    def section3(self, T: np.ndarray, layout, out_dir: Path) -> dict:
        """Section 3: rel_S, rel_T, the ceiling sqrt(rel_S) with its class interval, r(T, S),
        the count-model value and its bracket, and Var(S) and Cov(T, S) on the overlap."""
        from ceiling import bootstrap
        from ceiling.oracles import count_model

        S, info = self._read("section3", "section 3's summaries on the overlap of T and S: Var(T), "
                             "Var(S), Cov(T, S), r(T, S) and the ceiling's class-bootstrap interval")
        both = np.isfinite(T) & np.isfinite(S)
        t, s = T[both], S[both]
        cov = float(np.mean((t - t.mean()) * (s - s.mean())))
        var_t, var_s = float(np.var(t)), float(np.var(s))
        rows, cols = np.nonzero(both)
        counts = bootstrap.resample_counts(layout.n_classes)
        moments = bootstrap.resampled_moments(T[rows, cols], S[rows, cols], layout.class_index[cols], counts)
        del S
        with np.errstate(invalid="ignore"):
            ceiling_boot = np.sqrt(moments["cov"] / moments["var_y"])
        r_ts = cov / np.sqrt(var_t * var_s)
        out = {"n_overlap": int(both.sum()), "var_T": var_t, "var_S": var_s, "cov_TS": cov,
               "rel_S": cov / var_s, "rel_T": cov / var_t, "r_TS": float(r_ts),
               "ceiling": float(np.sqrt(cov / var_s)) if cov > 0 else float("nan"),
               "ceiling_ci": bootstrap.interval(ceiling_boot),
               "ceiling_resamples_undefined": int(np.sum(~np.isfinite(ceiling_boot))),
               "count_model": count_model(float(r_ts)), "read": info}
        Path(out_dir, "section3.json").write_text(json.dumps(out, indent=1))
        return out

    def registered_scores(self, T: np.ndarray, layout, pooled: dict, plan: dict, oracle: dict,
                          out_dir: Path) -> dict:
        """The registered readings on S, read once: every arm's r, R-0 to R-D and the branches."""
        from ceiling import report

        check_scoring_inputs(pooled, oracle, Path(out_dir))  # before S is read, so a bad input
        S, info = self._read("registered_scores", "the registered readings: every arm's r on S over "
                             "each split's entry set, the paired class bootstraps, R-0 to R-D")
        entry = np.load(Path(out_dir) / "entry_sets.npz")
        if not np.array_equal(entry["s_finite"], np.isfinite(S)):
            raise Refusal("refused: S's pattern differs from the one the entry sets were built on")
        results = report.readings_on_target(S, T, layout, pooled, plan, oracle, entry, is_real=True)
        del S
        results["read"] = info
        report.write_results(results, Path(out_dir))
        return results
