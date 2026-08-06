import operator
from typing import Annotated, TypedDict


class FileArtifact(TypedDict):
    """A single source file produced by a developer agent."""

    path: str
    content: str
    language: str


class TestResult(TypedDict):
    """Output of the QA sandbox execution step."""

    passed: bool
    total: int
    failed: int
    failures: list[str]
    logs: str


class ReviewComment(TypedDict):
    """A single comment produced by the Code Reviewer agent."""

    file: str
    line: int | None
    # "blocking" | "suggestion" | "nit"
    severity: str
    comment: str


class ProjectState(TypedDict):
    """
    The single shared state object threaded through every node in the graph.

    Design notes:
    - All fields except `messages` use last-write-wins semantics (default for LangGraph).
    - `messages` uses Annotated[list, operator.add] so each node appends its own
      entries without overwriting prior ones, giving a full audit trail.
    - Structured dicts/TypedDicts are used for all artifacts so routing functions
      can rely on typed fields rather than parsing prose.
    - revision_count / max_revisions together guarantee the QA/review loops
      terminate; the graph routes to END with status="failed" once exhausted.
    """

    # --- Input ---
    spec: str

    # --- Product Manager output ---
    requirements: str
    user_stories: list[str]

    # --- Software Architect output ---
    # Keys: "stack", "file_tree", "api_contracts", "db_schema", "rationale"
    architecture: dict

    # --- Developer outputs ---
    backend_files: dict[str, FileArtifact]
    frontend_files: dict[str, FileArtifact]

    # --- QA output ---
    test_results: TestResult | None

    # --- Code Reviewer output ---
    review_comments: list[ReviewComment]
    review_approved: bool

    # --- Control flow ---
    revision_count: int
    max_revisions: int
    # "planning" | "requirements_done" | "architecture_done" | "dev_done" |
    # "qa_passed" | "qa_failed" | "approved" |
    # "changes_requested" | "done" | "failed"
    status: Annotated[str, lambda old, new: new]

    # --- Observability: append-only message log across all agents ---
    messages: Annotated[list, operator.add]
