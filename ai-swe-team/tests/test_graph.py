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
from app.graph.state import ProjectState, TestResult


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


def _passing_test_result() -> TestResult:
    return TestResult(passed=True, total=5, failed=0, failures=[], logs="5 passed")


def _failing_test_result() -> TestResult:
    return TestResult(
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
# Sandbox runner tests (require Docker)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    __import__("shutil").which("docker") is None,
    reason="Docker is not available in this environment",
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
