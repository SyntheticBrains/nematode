"""The guarded scorer, on a temporary repository holding a copy of the registration and a synthetic
matrix file. The real registration is never modified and the real S is never opened."""

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


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / REL.parent).mkdir(parents=True)
    shutil.copy(C.REGISTRATION, root / REL)
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "test")
    git(root, "add", str(REL))
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


def make_scorer(repo, tmp_path, confirm=True, amendment=None):
    stams = tmp_path / "stams.npz"
    sha = write_stams(stams)
    return scorer.Scorer(confirm, amendment, repo=repo, registration=repo / REL,
                         reads=repo / REL.parent / "READS.md", stams=stams, stams_sha256=sha), stams, sha


def test_guard_refuses_without_the_flag(repo):
    with pytest.raises(scorer.Refusal, match="confirm-registered-read"):
        scorer.guard(False, repo, repo / REL)


def test_guard_passes_on_a_committed_unchanged_registration(repo):
    info = scorer.guard(True, repo, repo / REL)
    assert info["head"] == git(repo, "rev-parse", "HEAD").strip()
    assert info["registration_commit"] == info["head"]


def test_guard_refuses_a_modified_registration(repo):
    with open(repo / REL, "a") as handle:
        handle.write("\nedited\n")
    with pytest.raises(scorer.Refusal, match="differs"):
        scorer.guard(True, repo, repo / REL)
    git(repo, "checkout", "--", str(REL))
    scorer.guard(True, repo, repo / REL)


def test_guard_refuses_a_staged_change(repo):
    with open(repo / REL, "a") as handle:
        handle.write("\nstaged\n")
    git(repo, "add", str(REL))
    with pytest.raises(scorer.Refusal, match="differs"):
        scorer.guard(True, repo, repo / REL)


def test_guard_refuses_an_untracked_registration(tmp_path):
    root = tmp_path / "bare"
    (root / REL.parent).mkdir(parents=True)
    shutil.copy(C.REGISTRATION, root / REL)
    git(root, "init", "-q")
    with pytest.raises(scorer.Refusal, match="not tracked"):
        scorer.guard(True, root, root / REL)


def test_scorer_does_not_load_s_when_refused(repo, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(scorer, "_load_S", lambda *a: calls.append(a))
    s, _, _ = make_scorer(repo, tmp_path, confirm=False)
    layout = splits.make_layout([f"c{k % 74:02d}" for k in range(C.N_MEASURED)])
    with pytest.raises(scorer.Refusal):
        s.entry_sets(np.zeros((C.N_MEASURED, C.N_MEASURED)), layout, tmp_path)
    assert calls == []


def test_reads_are_logged_and_a_second_reading_is_refused(repo, tmp_path):
    s, stams, _ = make_scorer(repo, tmp_path)
    layout = splits.make_layout([f"c{k % 74:02d}" for k in range(C.N_MEASURED)])
    T = inputs.load_T(stams, inputs.file_sha256(stams))
    out = tmp_path / "out"
    out.mkdir()
    report = s.entry_sets(T, layout, out)
    assert report["worms"]["size"] > 0
    assert (out / "entry_sets.npz").exists()
    log = (repo / REL.parent / "READS.md").read_text()
    assert "- quantity: `entry_sets`" in log
    assert re.search(r"- HEAD: `[0-9a-f]{40}`", log) and "- time: " in log
    with pytest.raises(scorer.Refusal, match="already read"):
        s.entry_sets(T, layout, out)
    sec3 = s.section3(T, layout, out)  # a different quantity is allowed once
    assert 0 < sec3["rel_S"] < 1 and sec3["n_overlap"] > 0
    assert sec3["ceiling_ci"][0] <= sec3["ceiling"] <= sec3["ceiling_ci"][1]


def test_a_repeat_needs_a_committed_amendment(repo, tmp_path):
    s, stams, _ = make_scorer(repo, tmp_path)
    layout = splits.make_layout([f"c{k % 74:02d}" for k in range(C.N_MEASURED)])
    T = inputs.load_T(stams, inputs.file_sha256(stams))
    s.entry_sets(T, layout, tmp_path)
    # an unknown commit is refused
    bad = scorer.Scorer(True, "0" * 40, repo=repo, registration=repo / REL,
                        reads=repo / REL.parent / "READS.md", stams=stams, stams_sha256=s.stams_sha256)
    with pytest.raises(scorer.Refusal, match="not a committed amendment"):
        bad.entry_sets(T, layout, tmp_path)
    # a commit that amends the registration's folder is accepted, once
    (repo / REL.parent / "amendment-1.md").write_text("Amendment 1.\n")
    git(repo, "add", str(REL.parent / "amendment-1.md"))
    git(repo, "commit", "-q", "-m", "amendment 1")
    sha = git(repo, "rev-parse", "HEAD").strip()
    good = scorer.Scorer(True, sha, repo=repo, registration=repo / REL,
                         reads=repo / REL.parent / "READS.md", stams=stams, stams_sha256=s.stams_sha256)
    good.entry_sets(T, layout, tmp_path)
    assert f"- amendment: `{sha}`" in (repo / REL.parent / "READS.md").read_text()
    with pytest.raises(scorer.Refusal, match="already used"):
        good.entry_sets(T, layout, tmp_path)


def test_a_wrong_checksum_is_refused(repo, tmp_path):
    s, _, _ = make_scorer(repo, tmp_path)
    s.stams_sha256 = "0" * 64
    layout = splits.make_layout([f"c{k % 74:02d}" for k in range(C.N_MEASURED)])
    with pytest.raises(scorer.Refusal, match="SHA-256"):
        s.entry_sets(np.zeros((C.N_MEASURED, C.N_MEASURED)), layout, tmp_path)


def test_only_the_scorer_names_the_test_member():
    """No module but the scorer (and the constant's definition) refers to S's zip member."""
    package = Path(scorer.__file__).parent
    for path in package.glob("*.py"):
        text = path.read_text()
        if path.name in ("scorer.py", "config.py"):
            continue
        assert "TEST_MEMBER" not in text and "measured_stams.test" not in text, path.name


def test_registered_reading_checks_its_inputs_before_reading_s(repo, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(scorer, "_load_S", lambda *a: calls.append(a))
    s, _, _ = make_scorer(repo, tmp_path)
    layout = splits.make_layout([f"c{k % 74:02d}" for k in range(C.N_MEASURED)])
    with pytest.raises(scorer.Refusal, match="before reading S"):
        s.registered_scores(np.zeros((C.N_MEASURED, C.N_MEASURED)), layout, {"worms": {}}, {}, {}, tmp_path)
    assert calls == [] and not (repo / REL.parent / "READS.md").exists()
