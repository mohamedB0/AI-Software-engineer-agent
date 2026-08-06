"""
Graph unit and integration tests.

Covers:
1. Routing function logic (route_after_qa, route_after_review) in isolation.
2. Revision loop termination: QA failures must cycle back to developers up to
   max_revisions times and then end with status="failed".
3. Sandbox runner: verify a passing and a failing test suite behave correctly.
4. Sandbox isolation: verify network_disabled=True blocks outbound calls.

Run with:
    pytest tests/test_graph.py -v

Prerequisites for sandbox tests:
    - Docker running
    - Sandbox image built: docker build -t ai-swe-sandbox:latest -f Dockerfile.sandbox .
"""

import pytest

from app.graph.build_graph import route_after_qa, route_after_review
from app.graph.state import ProjectState
from app.graph.state import TestResult as StateTestResult


def _docker_available() -> bool:
    """True only when the Docker daemon is actually reachable (not just the CLI)."""
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> ProjectState:
    """Return a minimal ProjectState dict with sensible defaults."""
    base: ProjectState = {
        "spec": "test spec",
        "requirements": "",
        "user_stories": [],
        "architecture": {},
        "backend_files": {},
        "frontend_files": {},
        "test_results": None,
        "review_comments": [],
        "review_approved": False,
        "revision_count": 0,
        "max_revisions": 3,
        "status": "testing",
        "messages": [],
    }
    base.update(overrides)
    return base


def _passing_test_result() -> StateTestResult:
    return StateTestResult(passed=True, total=5, failed=0, failures=[], logs="5 passed")


def _failing_test_result() -> StateTestResult:
    return StateTestResult(
        passed=False,
        total=5,
        failed=2,
        failures=["test_foo", "test_bar"],
        logs="2 failed",
    )


# ---------------------------------------------------------------------------
# route_after_qa tests
# ---------------------------------------------------------------------------


class TestRouteAfterQa:
    def test_routes_to_reviewer_when_tests_pass(self):
        state = _make_state(test_results=_passing_test_result())
        assert route_after_qa(state) == "code_reviewer"

    def test_routes_to_revise_when_tests_fail_and_revisions_remain(self):
        state = _make_state(
            test_results=_failing_test_result(),
            revision_count=1,
            max_revisions=3,
        )
        assert route_after_qa(state) == "revise"

    def test_routes_to_failed_when_tests_fail_and_revisions_exhausted(self):
        state = _make_state(
            test_results=_failing_test_result(),
            revision_count=3,
            max_revisions=3,
        )
        assert route_after_qa(state) == "failed"

    def test_routes_to_failed_at_exactly_max_revisions(self):
        """revision_count == max_revisions means we are out of retries."""
        state = _make_state(
            test_results=_failing_test_result(),
            revision_count=3,
            max_revisions=3,
        )
        assert route_after_qa(state) == "failed"


# ---------------------------------------------------------------------------
# route_after_review tests
# ---------------------------------------------------------------------------


class TestRouteAfterReview:
    def test_routes_to_done_when_approved(self):
        state = _make_state(review_approved=True, test_results=_passing_test_result())
        assert route_after_review(state) == "done"

    def test_routes_to_revise_when_not_approved_and_revisions_remain(self):
        state = _make_state(
            review_approved=False,
            revision_count=0,
            max_revisions=3,
            test_results=_passing_test_result(),
        )
        assert route_after_review(state) == "revise"

    def test_routes_to_failed_when_not_approved_and_revisions_exhausted(self):
        state = _make_state(
            review_approved=False,
            revision_count=3,
            max_revisions=3,
            test_results=_passing_test_result(),
        )
        assert route_after_review(state) == "failed"


# ---------------------------------------------------------------------------
# Merge node tests
# ---------------------------------------------------------------------------


class TestMergeNode:
    def test_merge_sets_dev_done_status(self):
        from app.graph.build_graph import merge_dev_outputs

        state = _make_state()
        assert merge_dev_outputs(state) == {"status": "dev_done"}


# ---------------------------------------------------------------------------
# Sandbox helper tests (pure functions, no Docker required)
# ---------------------------------------------------------------------------


class TestSandboxHelpers:
    def test_safe_relative_path_rejects_unsafe_paths(self):
        from app.tools.sandbox_runner import _safe_relative_path

        assert _safe_relative_path("../evil.py") is None
        assert _safe_relative_path("a/../../evil.py") is None
        assert _safe_relative_path("/etc/passwd") is None
        assert _safe_relative_path("/workspace/main.py") is None
        assert _safe_relative_path("./x.py") is None
        assert _safe_relative_path("") is None
        assert _safe_relative_path("a//b.py") is None

    def test_safe_relative_path_normalizes_common_paths(self):
        from app.tools.sandbox_runner import _safe_relative_path

        assert _safe_relative_path("src/main.py") == "src/main.py"
        assert _safe_relative_path("app\\main.py") == "app/main.py"

    def test_build_tar_skips_unsafe_paths(self):
        import tarfile

        from app.tools.sandbox_runner import _build_tar

        buf = _build_tar({"../evil.py": "x", "good.py": "y"})
        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r") as tar:
            names = tar.getnames()
        assert names == ["good.py"]

    def test_parse_pytest_output_passed(self):
        from app.tools.sandbox_runner import _parse_pytest_output

        res = _parse_pytest_output(0, "3 passed, 1 failed in 1.00s")
        assert res["passed"] is True
        assert res["total"] == 4
        assert res["failed"] == 1

    def test_parse_pytest_output_failed(self):
        from app.tools.sandbox_runner import _parse_pytest_output

        logs = "FAILED test_bad.py::test_foo - AssertionError\n1 failed"
        res = _parse_pytest_output(1, logs)
        assert res["passed"] is False
        assert res["total"] == 1
        assert res["failures"] == ["test_bad.py::test_foo"]


# ---------------------------------------------------------------------------
# Sandbox runner tests (require Docker)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon is not reachable in this environment",
)
class TestSandboxRunner:
    def test_passing_pytest_suite(self):
        from app.tools.sandbox_runner import run_in_sandbox

        result = run_in_sandbox(
            code_files={
                "main.py": "def add(a, b):\n    return a + b\n",
                "test_main.py": (
                    "from main import add\n\n"
                    "def test_add():\n"
                    "    assert add(2, 3) == 5\n"
                ),
            },
            test_command="pytest -q",
        )
        assert result["passed"] is True
        assert result["total"] == 1
        assert result["failed"] == 0
        assert result["failures"] == []

    def test_failing_pytest_suite(self):
        from app.tools.sandbox_runner import run_in_sandbox

        result = run_in_sandbox(
            code_files={
                "test_fail.py": (
                    "def test_always_fails():\n"
                    "    assert False, 'expected failure'\n"
                )
            },
            test_command="pytest -q",
        )
        assert result["passed"] is False
        assert result["failed"] == 1
        assert "test_always_fails" in result["failures"][0]

    def test_network_is_disabled(self):
        """
        Confirm that network_disabled=True actually blocks outbound connections.
        The test script tries to import urllib and open a connection; it should
        raise an exception (OSError/socket.gaierror) rather than succeed.
        """
        from app.tools.sandbox_runner import run_in_sandbox

        result = run_in_sandbox(
            code_files={
                "test_no_network.py": (
                    "import socket\n\n"
                    "def test_network_blocked():\n"
                    "    try:\n"
                    "        socket.create_connection(('example.com', 80), timeout=2)\n"
                    "        assert False, 'Network should be disabled'\n"
                    "    except OSError:\n"
                    "        pass  # expected - network is disabled\n"
                )
            },
            test_command="pytest -q",
        )
        assert result["passed"] is True, (
            f"Network isolation test failed. Logs:\n{result['logs']}"
        )
