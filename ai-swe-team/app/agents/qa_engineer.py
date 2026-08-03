from app.agents._qa_test_writer import generate_tests
from app.graph.state import ProjectState, TestResult
from app.tools.sandbox import run_python_tests


def qa_engineer_node(state: ProjectState) -> dict:
    """
    QA Engineer agent:
    1. Asks _qa_test_writer to generate a pytest test suite targeting the API
       contracts and user stories.
    2. Merges implementation files and test files.
    3. Executes the combined file set inside the locked-down Docker sandbox.
    4. Returns a TestResult and sets status to 'qa_passed' or 'qa_failed'.

    The sandbox is network-disabled, memory-limited, and always removed after
    execution - see app/tools/sandbox_runner.py for security details.
    """
    # Generate test files using a separate LLM call with structured output.
    test_files = generate_tests(state)

    # Merge implementation + test files; test files may override implementation
    # paths only if they share a name, which indicates a test discovery conflict
    # - the QA writer is prompted to avoid this.
    all_files: dict[str, str] = {
        path: artifact["content"]
        for path, artifact in state["backend_files"].items()
    }
    all_files.update(test_files)

    raw_result = run_python_tests.invoke(
        {"code_files": all_files, "test_command": "pytest -q"}
    )

    result = TestResult(
        passed=raw_result["passed"],
        total=raw_result["total"],
        failed=raw_result["failed"],
        failures=raw_result["failures"],
        logs=raw_result["logs"],
    )

    tests_run = result["total"]
    tests_passed = result["total"] - result["failed"]
    return {
        "test_results": result,
        "status": "qa_passed" if result["passed"] else "qa_failed",
        "messages": [f"[QA] {tests_passed}/{tests_run} tests passed."],
    }
