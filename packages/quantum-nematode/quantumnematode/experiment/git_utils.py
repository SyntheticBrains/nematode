"""Git repository context capture utilities."""

import subprocess
from pathlib import Path

from quantumnematode.logging_config import logger


def is_git_repository() -> bool:
    """Check if the current directory is in a git repository.

    Returns
    -------
    bool
        True if in a git repository, False otherwise.
    """
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_git_commit_hash() -> str | None:
    """Get the current git commit hash.

    Returns
    -------
    str | None
        Commit hash or None if not in a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch_name() -> str | None:
    """Get the current git branch name.

    Returns
    -------
    str | None
        Branch name or None if not in a git repository or in detached HEAD.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        branch = result.stdout.strip()
        # Return None for detached HEAD state
        return branch if branch != "HEAD" else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_git_dirty() -> bool:
    """Check if there are uncommitted changes in the repository.

    Returns
    -------
    bool
        True if there are uncommitted changes, False otherwise.
    """
    try:
        # Check for staged changes
        result = subprocess.run(
            ["git", "diff", "--staged", "--quiet"],
            check=False,
            capture_output=True,
        )
        staged_changes = result.returncode != 0

        # Check for unstaged changes
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            check=False,
            capture_output=True,
        )
        unstaged_changes = result.returncode != 0

        return staged_changes or unstaged_changes
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def capture_git_context() -> dict[str, str | bool | None]:
    """Capture complete git repository context.

    Returns
    -------
    dict[str, str | bool | None]
        Dictionary with git_commit, git_branch, and git_dirty.
    """
    if not is_git_repository():
        logger.warning("Not in a git repository - git context will not be captured")
        return {
            "git_commit": None,
            "git_branch": None,
            "git_dirty": False,
        }

    commit = get_git_commit_hash()
    branch = get_git_branch_name()
    dirty = is_git_dirty()

    if dirty:
        logger.warning(
            "Repository has uncommitted changes - consider committing changes for reproducibility",
        )

    if branch is None:
        logger.warning("Repository is in detached HEAD state")

    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
    }


def get_relative_config_path(config_path: Path) -> str:
    """Get config file path relative to repository root.

    Parameters
    ----------
    config_path : Path
        Absolute path to config file.

    Returns
    -------
    str
        Path relative to repository root, or absolute path if not in repo.
    """
    if not is_git_repository():
        return str(config_path)

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        repo_root = Path(result.stdout.strip())
        return str(config_path.relative_to(repo_root))
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return str(config_path)
