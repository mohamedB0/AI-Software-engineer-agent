"""
LangChain tool wrapper around the Docker sandbox runner.

Agents call this tool via bind_tools / tool-calling. The underlying runner
(sandbox_runner.py) handles all Docker lifecycle management and security.
"""

from langchain_core.tools import tool

from app.tools.sandbox_runner import run_in_sandbox


@tool
def run_python_tests(
    code_files: dict[str, str],
    test_command: str = "pytest -q",
) -> dict:
    """
    Run the given source files and test command inside an isolated,
    network-disabled Docker container.

    Args:
        code_files: Mapping of relative file path to source code content.
                    Include both implementation files and test files.
        test_command: Shell command to execute (default: 'pytest -q').

    Returns:
        dict with keys:
            passed (bool): True if exit code was 0.
            total (int): Total number of tests discovered.
            failed (int): Number of failed tests.
            failures (list[str]): Names of failing test items.
            logs (str): Captured stdout/stderr (capped at 4000 chars).
    """
    return run_in_sandbox(code_files, test_command)
