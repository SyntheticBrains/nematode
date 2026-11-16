"""The guarded scorer, on a temporary repository holding a copy of the registration, a committed
copy of the scorer's code folder and a synthetic matrix file. The real registration is never
modified and the real S is never opened."""

import io
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ceiling import config as C
from ceiling import inputs, scorer, splits

REL = Path("docs/experiments/logbooks/supporting/135-neural-readout-ceiling/launch.md")
CODE_REL = Path("investigation/neural-ceiling/ceiling")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / REL.parent).mkdir(parents=True)
    shutil.copy(C.REGISTRATION, root / REL)
    (root / CODE_REL).mkdir(parents=True)
    (root / CODE_REL / "__init__.py").write_text('"""A stand-in for the scorer\'s code."""\n')
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "test")
    git(root, "add", str(REL), str(CODE_REL))
    git(root, "commit", "-q", "-m", "registration")
    return root


def write_stams(path, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.normal(size=(C.N_MEASURED, C.N_MEASURED))
    S = 0.5 * T + rng.normal(size=T.shape)
    S[rng.random(S.shape) < 0.4] = np.nan
    with zipfile.ZipFile(path, "w") as archive:
        for member, array in ((C.TRAIN_MEMBER, T), (C.TEST_MEMBER, S)):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array)
            archive.writestr(member, buffer.getvalue())
    return inputs.file_sha256(path)


def scorer_for(repo, stams, sha, confirm=True, amendment=None):
    return scorer.Scorer(confirm, amendment, repo=repo, registration=repo / REL,
                         reads=repo / REL.parent / "READS.md", stams=stams, stams_sha256=sha,
                         code_dir=repo / CODE_REL)


def make_scorer(repo, tmp_path, confirm=True, amendment=None):
    stams = tmp_path / "stams.npz"
    sha = write_stams(stams)
    return scorer_for(repo, stams, sha, confirm, amendment), stams, sha


def layout():
    return splits.make_layout([f"c{k % 74:02d}" for k in range(C.N_MEASURED)])


def guard(repo, confirm=True):
    return scorer.guard(confirm, repo, repo / REL, repo / CODE_REL)


def reads_md(repo):
    return repo / REL.parent / "READS.md"


def first_read(repo, tmp_path):
    s, stams, sha = make_scorer(repo, tmp_path)
    T = inputs.load_T(stams, sha)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    s.entry_sets(T, layout(), out)
    return s, stams, sha, T, out


# ---- the guard ------------------------------------------------------------------------------------


def test_guard_refuses_without_the_flag(repo):
    with pytest.raises(scorer.Refusal, match="confirm-registered-read"):
        guard(repo, confirm=False)


def test_guard_passes_on_a_committed_unchanged_registration(repo):
    info = guard(repo)
    assert info["head"] == git(repo, "rev-parse", "HEAD").strip()
    assert info["registration_commit"] == info["head"] and info["code"] == "clean at HEAD"


def test_guard_refuses_a_modified_registration(repo):
    with open(repo / REL, "a") as handle:
        handle.write("\nedited\n")
    with pytest.raises(scorer.Refusal, match="differs"):
        guard(repo)
    git(repo, "checkout", "--", str(REL))
    guard(repo)


def test_guard_refuses_a_staged_change(repo):
    with open(repo / REL, "a") as handle:
        handle.write("\nstaged\n")
    git(repo, "add", str(REL))
    with pytest.raises(scorer.Refusal, match="differs"):
        guard(repo)


def test_guard_refuses_an_untracked_registration(tmp_path):
    root = tmp_path / "bare"
    (root / REL.parent).mkdir(parents=True)
    shutil.copy(C.REGISTRATION, root / REL)
    git(root, "init", "-q")
    with pytest.raises(scorer.Refusal, match="not tracked"):
        guard(root)


@pytest.mark.parametrize("change", ["edit", "untracked", "staged"])
def test_guard_refuses_uncommitted_scorer_code(repo, change):
    """Amendment 1, item 11: uncommitted code under ceiling/ is refused, not just logged."""
    if change == "edit":
        (repo / CODE_REL / "__init__.py").write_text("changed = True\n")
    elif change == "untracked":
        (repo / CODE_REL / "new.py").write_text("x = 1\n")
    else:
        (repo / CODE_REL / "__init__.py").write_text("staged = True\n")
        git(repo, "add", str(CODE_REL))
    with pytest.raises(scorer.Refusal, match="uncommitted code"):
        guard(repo)


def test_guard_refuses_code_outside_the_repository(repo, tmp_path):
    with pytest.raises(scorer.Refusal, match="not in"):
        scorer.guard(True, repo, repo / REL, tmp_path / "elsewhere")


# ---- reading S ------------------------------------------------------------------------------------


def test_scorer_does_not_load_s_when_refused(repo, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(scorer, "_load_S", lambda *a: calls.append(a))
    s, _, _ = make_scorer(repo, tmp_path, confirm=False)
    with pytest.raises(scorer.Refusal):
        s.entry_sets(np.zeros((C.N_MEASURED, C.N_MEASURED)), layout(), tmp_path)
    assert calls == []
    assert not reads_md(repo).exists()


def test_the_read_is_logged_before_s_is_decoded(repo, tmp_path, monkeypatch):
    """Amendment 1, item 11: READS.md holds the read before the data are decoded."""
    real = scorer._load_S
    seen = []

    def decode(path, sha):
        seen.append(reads_md(repo).read_text())
        return real(path, sha)

    monkeypatch.setattr(scorer, "_load_S", decode)
    first_read(repo, tmp_path)
    assert len(seen) == 1 and "- quantity: `entry_sets`" in seen[0]


def test_reads_md_is_committed_alone_after_every_read(repo, tmp_path):
    """Amendment 1, item 11: after each read READS.md is committed, by itself, with the attribution."""
    (repo / "unrelated.txt").write_text("staged elsewhere\n")
    git(repo, "add", "unrelated.txt")
    s, stams, sha, T, out = first_read(repo, tmp_path)
    s.section3(T, layout(), out)
    log = git(repo, "log", "--format=%H %s", "-3").splitlines()
    assert len(log) == 3 and "section3" in log[0] and "entry_sets" in log[1]
    for line in log[:2]:
        commit = line.split()[0]
        files = git(repo, "show", "--name-only", "--format=", commit).split()
        assert files == [str(REL.parent / "READS.md")]
        body = git(repo, "show", "-s", "--format=%B", commit)
        assert "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" in body
    assert git(repo, "status", "--porcelain", "--", str(REL.parent / "READS.md")) == ""
    assert "A  unrelated.txt" in git(repo, "status", "--porcelain")  # left staged, not committed


def test_reads_md_is_committed_even_when_the_computation_fails(repo, tmp_path, monkeypatch):
    s, stams, sha = make_scorer(repo, tmp_path)
    from ceiling import splits as sp

    monkeypatch.setattr(sp, "entry_sets", lambda *a: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        s.entry_sets(inputs.load_T(stams, sha), layout(), tmp_path)
    assert git(repo, "status", "--porcelain", "--", str(REL.parent / "READS.md")) == ""
    assert "entry_sets" in git(repo, "log", "-1", "--format=%s")


def test_reads_are_logged_and_a_second_reading_is_refused(repo, tmp_path):
    s, stams, sha, T, out = first_read(repo, tmp_path)
    assert (out / "entry_sets.npz").exists()
    log = reads_md(repo).read_text()
    assert "- quantity: `entry_sets`" in log
    assert re.search(r"- HEAD: `[0-9a-f]{40}`", log) and "- time: " in log and "- code: clean at HEAD" in log
    with pytest.raises(scorer.Refusal, match="already read"):
        s.entry_sets(T, layout(), out)
    sec3 = s.section3(T, layout(), out)  # a different quantity is allowed once
    assert 0 < sec3["rel_S"] < 1 and sec3["n_overlap"] > 0
    assert sec3["ceiling_ci"][0] <= sec3["ceiling"] <= sec3["ceiling_ci"][1]


def test_a_wrong_checksum_is_refused_before_anything_is_logged(repo, tmp_path):
    s, _, _ = make_scorer(repo, tmp_path)
    s.stams_sha256 = "0" * 64
    with pytest.raises(scorer.Refusal, match="SHA-256"):
        s.entry_sets(np.zeros((C.N_MEASURED, C.N_MEASURED)), layout(), tmp_path)
    assert not reads_md(repo).exists()


# ---- the once-only rule (amendment 1, item 12) ----------------------------------------------------


def add_amendment(repo, name="amendment-1.md", edit_registration=False):
    if edit_registration:
        with open(repo / REL, "a") as handle:
            handle.write("\n## Amendment 2\n\nA change.\n")
        git(repo, "add", str(REL))
    else:
        (repo / REL.parent / name).write_text("Amendment.\n")
        git(repo, "add", str(REL.parent / name))
    git(repo, "commit", "-q", "-m", "amendment")
    return git(repo, "rev-parse", "HEAD").strip()


def test_a_repeat_needs_a_committed_amendment(repo, tmp_path):
    s, stams, sha, T, out = first_read(repo, tmp_path)
    with pytest.raises(scorer.Refusal, match="is not a commit"):
        scorer_for(repo, stams, sha, amendment="0" * 40).entry_sets(T, layout(), out)
    amendment = add_amendment(repo)
    good = scorer_for(repo, stams, sha, amendment=amendment)
    good.entry_sets(T, layout(), out)
    assert f"- amendment: `{amendment}`" in reads_md(repo).read_text()
    with pytest.raises(scorer.Refusal, match="already used|ancestor"):
        good.entry_sets(T, layout(), out)  # once used, it is an ancestor of the HEAD that read logged


def test_an_amendment_editing_the_registration_authorises_a_repeat(repo, tmp_path):
    s, stams, sha, T, out = first_read(repo, tmp_path)
    amendment = add_amendment(repo, edit_registration=True)
    scorer_for(repo, stams, sha, amendment=amendment).entry_sets(T, layout(), out)
    assert reads_md(repo).read_text().count("- quantity: `entry_sets`") == 2


def test_the_registration_commit_itself_is_not_an_amendment(repo, tmp_path):
    """The reviewer's probe: the registration's own commit is an ancestor of the earlier read."""
    s, stams, sha, T, out = first_read(repo, tmp_path)
    registration_commit = git(repo, "log", "-1", "--format=%H", "--", str(REL)).strip()
    with pytest.raises(scorer.Refusal, match="ancestor"):
        scorer_for(repo, stams, sha, amendment=registration_commit).entry_sets(T, layout(), out)
    assert reads_md(repo).read_text().count("- quantity: `entry_sets`") == 1


def test_committing_reads_md_does_not_authorise_a_repeat(repo, tmp_path):
    """The reviewer's probe: the READS.md commit (in the folder) is not an amendment."""
    s, stams, sha, T, out = first_read(repo, tmp_path)
    log_commit = git(repo, "log", "-1", "--format=%H", "--", str(REL.parent / "READS.md")).strip()
    with pytest.raises(scorer.Refusal, match="neither changes"):
        scorer_for(repo, stams, sha, amendment=log_commit).entry_sets(T, layout(), out)
    (repo / REL.parent / "notes.md").write_text("not an amendment file\n")
    git(repo, "add", str(REL.parent / "notes.md"))
    git(repo, "commit", "-q", "-m", "notes")
    notes = git(repo, "rev-parse", "HEAD").strip()
    with pytest.raises(scorer.Refusal, match="neither changes"):
        scorer_for(repo, stams, sha, amendment=notes).entry_sets(T, layout(), out)


def test_a_committed_edit_of_the_registration_needs_naming_after_the_first_read(repo, tmp_path):
    """The reviewer's probe: the guard passes a committed edit, but the blob logged at the first
    read pins the registration; a later read must name the amendment that accounts for it."""
    s, stams, sha, T, out = first_read(repo, tmp_path)
    with open(repo / REL, "a") as handle:
        handle.write("\nThe bar is now r >= 0.1.\n")
    git(repo, "commit", "-q", "-am", "edit")
    edit = git(repo, "rev-parse", "HEAD").strip()
    guard(repo)  # the guard alone does not refuse a committed edit
    with pytest.raises(scorer.Refusal, match="blob differs"):
        s.section3(T, layout(), out)
    # a later commit that only adds an amendment file does not account for the edit
    other = add_amendment(repo, "amendment-x.md")
    with pytest.raises(scorer.Refusal, match="does not account"):
        scorer_for(repo, stams, sha, amendment=other).section3(T, layout(), out)
    scorer_for(repo, stams, sha, amendment=edit).section3(T, layout(), out)


def test_deleting_reads_md_does_not_allow_a_repeat(repo, tmp_path):
    """The reviewer's probe: products of an earlier read with no READS.md are refused."""
    s, stams, sha, T, out = first_read(repo, tmp_path)
    reads_md(repo).unlink()
    with pytest.raises(scorer.Refusal, match="READS.md is missing"):
        s.entry_sets(T, layout(), out)
    with pytest.raises(scorer.Refusal, match="READS.md is missing"):
        s.section3(T, layout(), out)


def test_registered_reading_checks_its_inputs_before_reading_s(repo, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(scorer, "_load_S", lambda *a: calls.append(a))
    s, _, _ = make_scorer(repo, tmp_path)
    with pytest.raises(scorer.Refusal, match="before reading S"):
        s.registered_scores(np.zeros((C.N_MEASURED, C.N_MEASURED)), layout(), {"worms": {}}, {}, {}, tmp_path)
    assert calls == [] and not reads_md(repo).exists()


def test_only_the_scorer_names_the_test_member():
    """No module but the scorer (and the constant's definition) refers to S's zip member."""
    package = Path(scorer.__file__).parent
    for path in package.glob("*.py"):
        text = path.read_text()
        if path.name in ("scorer.py", "config.py"):
            continue
        assert "TEST_MEMBER" not in text and "measured_stams.test" not in text, path.name
