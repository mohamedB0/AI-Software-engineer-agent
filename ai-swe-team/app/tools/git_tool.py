"""
Git utility wrappers used by the Backend Developer and Code Reviewer agents.

These wrap gitpython for common operations. Agents can call these directly
or via the MCP Git server (mcp_config.py); the MCP server is preferred for
graph-wide uniformity, but these utilities are useful for scripting and tests.
"""


import git

from app.config import settings


def _get_repo(repo_path: str | None = None) -> git.Repo:
    """Return a gitpython Repo instance for the configured repository."""
    path = repo_path or settings.repo_path
    return git.Repo(path)


def get_diff(repo_path: str | None = None) -> str:
    """Return the current working-tree diff (unstaged + staged changes)."""
    repo = _get_repo(repo_path)
    return repo.git.diff("HEAD")


def commit_files(
    files: list[str],
    message: str,
    author_name: str = "AI Backend Dev",
    author_email: str = "agent@ai-swe-team.local",
    repo_path: str | None = None,
) -> str:
    """
    Stage the given file paths and create a commit.

    Args:
        files: List of file paths to stage (relative to repo root).
        message: Commit message.
        author_name: Git author name.
        author_email: Git author email.
        repo_path: Path to the repository (defaults to REPO_PATH env var).

    Returns:
        The hexsha of the new commit.
    """
    repo = _get_repo(repo_path)
    repo.index.add(files)
    actor = git.Actor(author_name, author_email)
    commit = repo.index.commit(message, author=actor, committer=actor)
    return commit.hexsha


def create_branch(branch_name: str, repo_path: str | None = None) -> str:
    """Create and check out a new branch. Returns the branch name."""
    repo = _get_repo(repo_path)
    branch = repo.create_head(branch_name)
    branch.checkout()
    return branch_name


def get_current_branch(repo_path: str | None = None) -> str:
    """Return the name of the currently active branch."""
    repo = _get_repo(repo_path)
    return repo.active_branch.name
