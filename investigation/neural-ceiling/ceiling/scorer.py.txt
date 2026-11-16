"""The guarded scorer: the only module that loads S (`measured_stams.test`).

Every access goes through `authorise`, which refuses unless --confirm-registered-read was passed,
the registration file is tracked, unmodified and in the git log, this spike's code under
`ceiling/` has no uncommitted change, and the once-only rule holds (amendment 1, item 12). The
file's checksum is verified, then the access is appended to READS.md with its time, git HEAD,
registration blob and what is read, and only then is S decoded; READS.md is committed, alone,
after every read (item 11). S itself never leaves this module: callers get the finite-entry
pattern, the section 3 summaries, or the registered scores.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from ceiling import config as C


class Refusal(RuntimeError):
    """The scorer will not read S."""


# ---- The guard -----------------------------------------------------------------------------------


def _git(repo: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False,
                          env=env)


def _relative(path: Path, repo: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(repo).resolve()))


def guard(confirm: bool, repo: Path = C.REPO, registration: Path = C.REGISTRATION,
          code_dir: Path = C.CODE_DIR) -> dict:
    """Refuse unless the flag is passed, the registration is tracked, unchanged and committed, and
    the scorer's code has no uncommitted change."""
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
    code = _code_state(repo, code_dir)
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    blob = _git(repo, "rev-parse", f"HEAD:{rel}").stdout.strip()
    return {"head": head, "registration": rel, "registration_commit": log.stdout.strip(),
            "registration_blob": blob, "code": code}


def _code_state(repo: Path, code_dir: Path) -> str:
    """Refuse while the scorer's code has an uncommitted (or untracked) change: HEAD, which every
    read logs, must be the code that made the read."""
    try:
        rel = _relative(code_dir, repo)
    except ValueError:
        raise Refusal(f"refused: the scorer's code {code_dir} is not in {repo}") from None
    if not Path(code_dir).is_dir():
        raise Refusal(f"refused: {rel} does not exist")
    changed = _git(repo, "status", "--porcelain", "--", rel).stdout.splitlines()
    if changed:
        raise Refusal(f"refused: uncommitted code under {rel}: {changed[:5]}")
    if not _git(repo, "ls-files", "--", rel).stdout.strip():
        raise Refusal(f"refused: no committed code under {rel}")
    return "clean at HEAD"


# ---- READS.md -----------------------------------------------------------------------------------

_FIELDS = {
    "quantity": re.compile(r"^- quantity: `([^`]+)`$"),
    "head": re.compile(r"^- HEAD: `([0-9a-f]+)`$"),
    "blob": re.compile(r"^- registration: `[^`]*` at commit `[0-9a-f]*`, blob `([0-9a-f]*)`$"),
    "amendment": re.compile(r"^- amendment: `([0-9a-f]+)`$"),
}


def prior_reads(reads: Path) -> list[dict]:
    """Every read logged in READS.md, in order: quantity, HEAD, registration blob, amendment."""
    if not Path(reads).exists():
        return []
    out: list[dict] = []
    for line in Path(reads).read_text().splitlines():
        if match := _FIELDS["quantity"].match(line):
            out.append({"quantity": match.group(1), "head": None, "blob": None, "amendment": None})
            continue
        for key in ("head", "blob", "amendment"):
            if out and (match := _FIELDS[key].match(line)):
                out[-1][key] = match.group(1)
    return out


def _amendment_problem(repo: Path, sha: str, registration: Path, earlier_head: str | None) -> str | None:
    """Why `sha` is not a committed amendment for a repeat read, or None if it is: a commit in
    HEAD's history, not an ancestor of the commit logged at the earlier read, that changes the
    registration or adds an amendment file in its folder."""
    repo = Path(repo).resolve()
    rel = _relative(registration, repo)
    folder = str(Path(rel).parent)
    if _git(repo, "cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        return f"{sha} is not a commit"
    if _git(repo, "merge-base", "--is-ancestor", sha, "HEAD").returncode != 0:
        return f"{sha} is not in HEAD's history"
    if earlier_head and _git(repo, "merge-base", "--is-ancestor", sha, earlier_head).returncode == 0:
        return f"{sha} is an ancestor of {earlier_head}, the commit logged at the earlier read"
    if _changes_registration(repo, sha, rel):
        return None
    changes = _git(repo, "diff-tree", "--no-commit-id", "--name-status", "-r", "--root", sha).stdout
    for line in changes.splitlines():
        status, _, path = line.partition("\t")
        name = Path(path).name.lower()
        if (status[:1] == "A" and str(Path(path).parent) == folder
                and name.startswith(C.AMENDMENT_FILE_PREFIX)):
            return None
    return f"{sha} neither changes the registration nor adds an amendment file"


def _changes_registration(repo: Path, sha: str, rel: str) -> bool:
    changes = _git(repo, "diff-tree", "--no-commit-id", "--name-status", "-r", "--root", sha).stdout
    return any(line.partition("\t")[2] == rel and line[:1] in ("M", "A") for line in changes.splitlines())


def authorise(quantity: str, confirm: bool, amendment: str | None = None, repo: Path = C.REPO,
              registration: Path = C.REGISTRATION, reads: Path = C.READS_LOG,
              run_dir: Path | None = None, code_dir: Path = C.CODE_DIR) -> dict:
    """The guard and the once-only rule (amendment 1, item 12)."""
    info = guard(confirm, repo, registration, code_dir)
    reads = Path(reads)
    if not reads.exists() and run_dir is not None:
        products = [name for name in C.READ_PRODUCTS if (Path(run_dir) / name).exists()]
        if products:
            raise Refusal(f"refused: READS.md is missing while {run_dir} holds {products}")
    prior = prior_reads(reads)
    repeats = [r for r in prior if r["quantity"] == quantity]
    blob_changed = bool(prior) and info["registration_blob"] != prior[0]["blob"]
    if repeats or blob_changed:
        why = (f"{quantity!r} was already read" if repeats else
               "the registration's blob differs from the one logged at the first read")
        if amendment is None:
            raise Refusal(f"refused: {why}; name a committed amendment")
        earlier = (repeats or prior)[-1 if repeats else 0]["head"]
        problem = _amendment_problem(repo, amendment, registration, earlier)
        if problem:
            raise Refusal(f"refused: {why}, and {problem}")
        if any(r["amendment"] == amendment for r in repeats):
            raise Refusal(f"refused: amendment {amendment} was already used for {quantity!r}")
        if blob_changed:  # the amendment itself must have made the registration what it now is
            at_amendment = _git(repo, "rev-parse", f"{amendment}:{info['registration']}").stdout.strip()
            if (at_amendment != info["registration_blob"]
                    or not _changes_registration(repo, amendment, info["registration"])):
                raise Refusal(f"refused: the registration changed since the first read, and amendment "
                              f"{amendment} does not account for the change")
    info["amendment"] = amendment
    return info


def log_read(quantity: str, what: str, info: dict, reads: Path = C.READS_LOG) -> None:
    reads = Path(reads)
    now = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [] if reads.exists() else [
        "# Reads of S for launch record 135", "",
        "Appended by `investigation/neural-ceiling/ceiling/scorer.py` before every access to S, and",
        "committed after it. Each registered reading is made once; a repeat names a committed",
        "amendment.", ""]
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
        handle.flush()
        os.fsync(handle.fileno())


def commit_reads(repo: Path, reads: Path, quantity: str) -> str:
    """Commit READS.md alone (git add, then a commit of that one path)."""
    rel = _relative(reads, repo)
    env = {**os.environ, "SKIP": "pytest"}  # a pre-commit pytest hook would use every core
    add = _git(repo, "add", "--", rel, env=env)
    done = _git(repo, "commit", "-q", "-m", C.READS_COMMIT_MESSAGE.format(quantity=quantity), "--", rel,
                env=env)
    if add.returncode != 0 or done.returncode != 0:
        raise RuntimeError(f"READS.md records the {quantity} read but was not committed "
                           f"({(add.stderr + done.stderr).strip()}); commit it by hand before going on")
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


# ---- S ------------------------------------------------------------------------------------------


def _check_sha(path: Path, expected_sha256: str) -> None:
    """The file's checksum, over its bytes (nothing is decoded)."""
    from ceiling.inputs import file_sha256

    found = file_sha256(path)
    if found != expected_sha256:
        raise Refusal(f"refused: {path} has SHA-256 {found}, expected {expected_sha256}")


def _load_S(path: Path, expected_sha256: str) -> np.ndarray:
    """Decode S. Called only inside a logged read."""
    _check_sha(path, expected_sha256)
    with zipfile.ZipFile(path) as archive:
        S = np.array(np.lib.format.read_array(io.BytesIO(archive.read(C.TEST_MEMBER))), dtype=np.float64)
    np.fill_diagonal(S, np.nan)
    return S


REFERENCE_ARMS = ("floor-ME", "floor-C", "H0", "H0-cal", "C.2", "C.2-cal")


def check_scoring_inputs(pooled: dict, oracle: dict, out_dir: Path) -> None:
    """Everything the registered reading needs, checked before S is read (a failure after the
    read would spend the once-only reading). The masked arm may be unreadable or not run: its split
    is then reported unreadable, not refused (amendment 1, item 6). Every readable arm must be
    defined on every entry of its split's set (item 5)."""
    from ceiling import report

    problems = []
    entry_path = Path(out_dir) / "entry_sets.npz"
    if not entry_path.exists():
        raise Refusal("refused before reading S: entry_sets.npz is missing")
    entry = np.load(entry_path)
    for split in C.SPLITS:
        if split not in pooled:
            problems.append(f"no predictions on {split} (R-0 is reported on every split)")
            continue
        refs = REFERENCE_ARMS + (("floor-T",) if split in C.HELD_OUT_SPLITS else ())
        problems += [f"{split}: no {arm!r}" for arm in refs if pooled[split].get(arm) is None]
        entries = np.any(np.stack(report.fold_entry_masks(entry, split)), axis=0)
        for arm, pred in pooled[split].items():
            if pred is not None and not np.isfinite(pred[entries]).all():
                problems.append(f"{split}: {arm} is undefined on {int((~np.isfinite(pred[entries])).sum())}"
                                f" of {int(entries.sum())} entries")
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
                 stams: Path = C.STAMS_PATH, stams_sha256: str = C.STAMS_SHA256,
                 code_dir: Path = C.CODE_DIR) -> None:
        self.confirm, self.amendment = confirm, amendment
        self.repo, self.registration, self.reads = repo, registration, reads
        self.stams, self.stams_sha256, self.code_dir = stams, stams_sha256, code_dir

    @contextlib.contextmanager
    def _reading(self, quantity: str, what: str, out_dir: Path):
        """Authorise, verify the checksum, log the read, then let the caller decode S; commit
        READS.md after the read, even if the caller fails."""
        info = authorise(quantity, self.confirm, self.amendment, self.repo, self.registration,
                         self.reads, Path(out_dir), self.code_dir)
        _check_sha(self.stams, self.stams_sha256)
        log_read(quantity, what, info, self.reads)
        try:
            yield info
        finally:
            info["reads_commit"] = commit_reads(self.repo, self.reads, quantity)

    # ---- the three registered quantities ----

    def entry_sets(self, T: np.ndarray, layout, out_dir: Path) -> dict:
        """S's finite-entry pattern and every split's entry set; no S value leaves."""
        from ceiling import splits

        out_dir = Path(out_dir)
        with self._reading("entry_sets", "S's finite-entry pattern only (no values): the entry "
                           "sets of all three splits and the pattern for the oracles' S_syn",
                           out_dir) as info:
            S = _load_S(self.stams, self.stams_sha256)
            pattern = np.isfinite(S)
            del S
            sets = splits.entry_sets(np.isfinite(T), pattern, layout)
            report = splits.entry_set_report(np.isfinite(T), pattern, layout)
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

        with self._reading("section3", "section 3's summaries on the overlap of T and S: Var(T), "
                           "Var(S), Cov(T, S), r(T, S) and the ceiling's class-bootstrap interval",
                           out_dir) as info:
            S = _load_S(self.stams, self.stams_sha256)
            both = np.isfinite(T) & np.isfinite(S)
            t, s = T[both], S[both]
            cov = float(np.mean((t - t.mean()) * (s - s.mean())))
            var_t, var_s = float(np.var(t)), float(np.var(s))
            rows, cols = np.nonzero(both)
            counts = bootstrap.resample_counts(layout.n_classes)
            moments = bootstrap.resampled_moments(T[rows, cols], S[rows, cols], layout.class_index[cols],
                                                  counts)
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
        entry = np.load(Path(out_dir) / "entry_sets.npz")  # cannot spend the once-only reading
        with self._reading("registered_scores", "the registered readings: every arm's r on S over "
                           "each split's entry set, the paired class bootstraps, R-0 to R-D",
                           out_dir) as info:
            S = _load_S(self.stams, self.stams_sha256)
            if not np.array_equal(entry["s_finite"], np.isfinite(S)):
                raise Refusal("refused: S's pattern differs from the one the entry sets were built on")
            results = report.readings_on_target(S, T, layout, pooled, plan, oracle, entry, is_real=True)
            del S
            results["read"] = info
            report.write_results(results, Path(out_dir))
        return results
