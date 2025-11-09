"""Tests for git utility functions."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from quantumnematode.experiment.git_utils import (
    capture_git_context,
    get_git_branch_name,
    get_git_commit_hash,
    get_relative_config_path,
    is_git_dirty,
    is_git_repository,
)


class TestGitRepository:
    """Test git repository detection."""

    def test_is_git_repository_true(self):
        """Test detecting a valid git repository."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = is_git_repository()
            assert result is True

    def test_is_git_repository_false(self):
        """Test detecting non-git directory."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            result = is_git_repository()
            assert result is False

    def test_is_git_repository_no_git_command(self):
        """Test handling missing git command."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = is_git_repository()
            assert result is False


class TestGitCommitHash:
    """Test git commit hash retrieval."""

    def test_get_commit_hash_success(self):
        """Test getting commit hash successfully."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="abc123def456\n",
                returncode=0,
            )
            commit = get_git_commit_hash()
            assert commit == "abc123def456"

    def test_get_commit_hash_failure(self):
        """Test handling git command failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            commit = get_git_commit_hash()
            assert commit is None

    def test_get_commit_hash_no_git(self):
        """Test handling missing git."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            commit = get_git_commit_hash()
            assert commit is None


class TestGitBranchName:
    """Test git branch name retrieval."""

    def test_get_branch_name_success(self):
        """Test getting branch name successfully."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="main\n",
                returncode=0,
            )
            branch = get_git_branch_name()
            assert branch == "main"

    def test_get_branch_name_detached_head(self):
        """Test handling detached HEAD state."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="HEAD\n",
                returncode=0,
            )
            branch = get_git_branch_name()
            assert branch is None

    def test_get_branch_name_failure(self):
        """Test handling git command failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            branch = get_git_branch_name()
            assert branch is None


class TestGitDirty:
    """Test git dirty state detection."""

    def test_is_git_dirty_clean(self):
        """Test detecting clean repository."""
        with patch("subprocess.run") as mock_run:
            # Both staged and unstaged return 0 (clean)
            mock_run.return_value = MagicMock(returncode=0)
            dirty = is_git_dirty()
            assert dirty is False

    def test_is_git_dirty_staged_changes(self):
        """Test detecting staged changes."""
        with patch("subprocess.run") as mock_run:
            # Staged check returns 1 (changes present)
            mock_run.side_effect = [
                MagicMock(returncode=1),  # staged check
                MagicMock(returncode=0),  # unstaged check
            ]
            dirty = is_git_dirty()
            assert dirty is True

    def test_is_git_dirty_unstaged_changes(self):
        """Test detecting unstaged changes."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0),  # staged check
                MagicMock(returncode=1),  # unstaged check
            ]
            dirty = is_git_dirty()
            assert dirty is True

    def test_is_git_dirty_both_changes(self):
        """Test detecting both staged and unstaged changes."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=1),  # staged check
                MagicMock(returncode=1),  # unstaged check
            ]
            dirty = is_git_dirty()
            assert dirty is True

    def test_is_git_dirty_error(self):
        """Test handling git errors."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            dirty = is_git_dirty()
            assert dirty is False


class TestCaptureGitContext:
    """Test complete git context capture."""

    def test_capture_git_context_full(self):
        """Test capturing complete git context."""
        with (
            patch("quantumnematode.experiment.git_utils.is_git_repository", return_value=True),
            patch(
                "quantumnematode.experiment.git_utils.get_git_commit_hash",
                return_value="abc123",
            ),
            patch("quantumnematode.experiment.git_utils.get_git_branch_name", return_value="main"),
            patch("quantumnematode.experiment.git_utils.is_git_dirty", return_value=False),
        ):
            context = capture_git_context()

            assert context["git_commit"] == "abc123"
            assert context["git_branch"] == "main"
            assert context["git_dirty"] is False

    def test_capture_git_context_dirty(self):
        """Test capturing context with dirty repository."""
        with (
            patch("quantumnematode.experiment.git_utils.is_git_repository", return_value=True),
            patch(
                "quantumnematode.experiment.git_utils.get_git_commit_hash",
                return_value="def456",
            ),
            patch(
                "quantumnematode.experiment.git_utils.get_git_branch_name",
                return_value="feature",
            ),
            patch("quantumnematode.experiment.git_utils.is_git_dirty", return_value=True),
        ):
            context = capture_git_context()

            assert context["git_commit"] == "def456"
            assert context["git_branch"] == "feature"
            assert context["git_dirty"] is True

    def test_capture_git_context_not_git_repo(self):
        """Test capturing context outside git repository."""
        with patch("quantumnematode.experiment.git_utils.is_git_repository", return_value=False):
            context = capture_git_context()

            assert context["git_commit"] is None
            assert context["git_branch"] is None
            assert context["git_dirty"] is False

    def test_capture_git_context_detached_head(self):
        """Test capturing context in detached HEAD state."""
        with (
            patch("quantumnematode.experiment.git_utils.is_git_repository", return_value=True),
            patch(
                "quantumnematode.experiment.git_utils.get_git_commit_hash",
                return_value="xyz789",
            ),
            patch("quantumnematode.experiment.git_utils.get_git_branch_name", return_value=None),
            patch("quantumnematode.experiment.git_utils.is_git_dirty", return_value=False),
        ):
            context = capture_git_context()

            assert context["git_commit"] == "xyz789"
            assert context["git_branch"] is None
            assert context["git_dirty"] is False


class TestGetRelativeConfigPath:
    """Test config path relativization."""

    def test_get_relative_config_path_absolute(self):
        """Test converting absolute path to relative."""
        config_path = Path("/Users/test/project/configs/test.yml")

        with (
            patch("quantumnematode.experiment.git_utils.is_git_repository", return_value=True),
            patch("subprocess.run") as mock_run,
        ):
            # Mock git rev-parse --show-toplevel
            mock_run.return_value = MagicMock(
                stdout="/Users/test/project\n",
                returncode=0,
            )
            relative = get_relative_config_path(config_path)
            assert relative == "configs/test.yml"

    def test_get_relative_config_path_already_relative(self):
        """Test handling already relative path."""
        config_path = Path("configs/test.yml")

        with patch("pathlib.Path.cwd", return_value=Path("/Users/test/project")):
            relative = get_relative_config_path(config_path)
            assert relative == "configs/test.yml"

    def test_get_relative_config_path_nested(self):
        """Test nested config path."""
        config_path = Path("/Users/test/project/configs/examples/test.yml")

        with (
            patch("quantumnematode.experiment.git_utils.is_git_repository", return_value=True),
            patch("subprocess.run") as mock_run,
        ):
            # Mock git rev-parse --show-toplevel
            mock_run.return_value = MagicMock(
                stdout="/Users/test/project\n",
                returncode=0,
            )
            relative = get_relative_config_path(config_path)
            assert relative == "configs/examples/test.yml"

    def test_get_relative_config_path_outside_project(self):
        """Test config path outside project (should return absolute)."""
        config_path = Path("/opt/configs/test.yml")

        with patch("pathlib.Path.cwd", return_value=Path("/Users/test/project")):
            relative = get_relative_config_path(config_path)
            # Should return the original path as string since it's outside project
            assert str(config_path) in relative or relative.startswith("..")
