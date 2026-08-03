"""
LangGraph StateGraph assembly.

Graph topology:

    pm --> architect --> backend_dev  --|
                    `--> frontend_dev --|
                                       v
                                      qa --> (conditional) --> reviewer --> (conditional) --> END
                                              |                              |
                                              v                              v
                                         bump_revision <--------------------'
                                              |
                                              +--> backend_dev
                                              `--> frontend_dev

Revision loop safety:
- revision_count is incremented by the dedicated bump_revision node.
- max_revisions is set once at invocation and never modified inside the loop.
- When revision_count >= max_revisions, routing functions return "failed" and
  the graph ends rather than looping forever. Treat "failed" as a signal for
  human escalation (surfaced in the FastAPI layer).

Fan-out / fan-in:
- architect has two outgoing edges (backend_dev and frontend_dev), so LangGraph
  schedules both in the same superstep (parallel execution).
- qa has two incoming edges; LangGraph only executes it once both branches
  have completed for the current superstep (automatic fan-in).
- backend_dev and frontend_dev write to disjoint state keys, so there is no
  concurrent write conflict.
"""

from langgraph.graph import END, StateGraph

from app.agents.architect import architect_node
from app.agents.backend_dev import backend_dev_node
from app.agents.code_reviewer import code_reviewer_node
from app.agents.frontend_dev import frontend_dev_node
from app.agents.product_manager import product_manager_node
from app.agents.qa_engineer import qa_engineer_node
# get_checkpointer is imported lazily inside build_graph() to avoid
# establishing a Postgres connection at module import time.
from app.graph.state import ProjectState


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_after_qa(state: ProjectState) -> str:
    """
    After the QA node:
    - Tests passed  -> send to code reviewer.
    - Tests failed, revisions remaining -> bump revision counter and re-develop.
    - Tests failed, revisions exhausted -> end graph with status "failed".
    """
    if state["test_results"]["passed"]:
        return "code_reviewer"
    if state["revision_count"] < state["max_revisions"]:
        return "revise"
    return "failed"


def route_after_review(state: ProjectState) -> str:
    """
    After the Code Reviewer node:
    - Approved -> end graph successfully.
    - Changes requested, revisions remaining -> bump revision counter and re-develop.
    - Changes requested, revisions exhausted -> end graph with status "failed".
    """
    if state["review_approved"]:
        return "done"
    if state["revision_count"] < state["max_revisions"]:
        return "revise"
    return "failed"


# ---------------------------------------------------------------------------
# Utility nodes
# ---------------------------------------------------------------------------


def increment_revision(state: ProjectState) -> dict:
    """Increment the revision counter. Always routes to backend_dev + frontend_dev."""
    return {"revision_count": state["revision_count"] + 1}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph():
    """
    Build and compile the StateGraph with the Postgres checkpointer.

    Call this once at application startup; the returned compiled graph is
    thread-safe and can be shared across concurrent requests.
    """
    # Lazy import to avoid establishing a Postgres connection at module import time.
    from app.graph.checkpointer import get_checkpointer
    graph = StateGraph(ProjectState)

    # --- Register nodes ---
    graph.add_node("pm", product_manager_node)
    graph.add_node("architect", architect_node)
    graph.add_node("backend_dev", backend_dev_node)
    graph.add_node("frontend_dev", frontend_dev_node)
    graph.add_node("qa", qa_engineer_node)
    graph.add_node("reviewer", code_reviewer_node)
    graph.add_node("bump_revision", increment_revision)

    # --- Entry point ---
    graph.set_entry_point("pm")

    # --- Fixed edges ---
    graph.add_edge("pm", "architect")

    # Parallel fan-out: both developer agents start after the architect.
    graph.add_edge("architect", "backend_dev")
    graph.add_edge("architect", "frontend_dev")

    # Fan-in: QA waits for both developer branches to complete.
    graph.add_edge("backend_dev", "qa")
    graph.add_edge("frontend_dev", "qa")

    # --- Conditional edges (routing logic) ---
    graph.add_conditional_edges(
        "qa",
        route_after_qa,
        {
            "code_reviewer": "reviewer",
            "revise": "bump_revision",
            "failed": END,
        },
    )
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "done": END,
            "revise": "bump_revision",
            "failed": END,
        },
    )

    # After bumping the revision counter, send work back to both developers.
    graph.add_edge("bump_revision", "backend_dev")
    graph.add_edge("bump_revision", "frontend_dev")

    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)
