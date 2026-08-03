"""
FastAPI route handlers for the /projects resource.

Endpoints:
    POST   /projects              - Submit a new project spec, start graph execution.
    GET    /projects/{id}         - Get project status and metadata.
    GET    /projects/{id}/state   - Get the full LangGraph state (files, test results,
                                    review comments, message trace) for debugging.

Important: graph execution is run as a FastAPI BackgroundTask here, which is
fine for prototyping. For production use, move run_graph_and_update_db into a
real task queue (Celery, RQ, or Dramatiq) so runs survive process restarts and
workers can be scaled independently of the API.
"""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.schemas import CreateProjectRequest, ProjectResponse
from app.db.models import Project
from app.db.session import SessionLocal, get_db
from app.graph.build_graph import build_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["projects"])

# Build the graph once at module load; it is thread-safe and reused across requests.
graph = build_graph()


# ---------------------------------------------------------------------------
# Background execution helper
# ---------------------------------------------------------------------------


async def run_graph_and_update_db(
    thread_id: str,
    spec: str,
    project_db_id: str,
) -> None:
    """
    Run the LangGraph graph asynchronously and persist the final status and
    revision count back to the projects table.

    Uses its own database session (not the request-scoped one) since this runs
    after the HTTP response has already been sent.
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "spec": spec,
        "revision_count": 0,
        "max_revisions": 3,
        "status": "planning",
        "review_comments": [],
        "messages": [],
    }

    final_state: dict = {}
    try:
        final_state = await graph.ainvoke(initial_state, config=config)
        status = final_state.get("status", "done")
    except Exception:
        logger.exception(
            "Graph execution failed for project %s (thread %s)",
            project_db_id,
            thread_id,
        )
        status = "failed"

    db = SessionLocal()
    try:
        project = db.query(Project).get(project_db_id)
        if project:
            project.status = status
            project.revision_count = final_state.get("revision_count", 0)
            db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post("", response_model=ProjectResponse, status_code=202)
async def create_project(
    req: CreateProjectRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Create a new project and start the AI agent graph in the background.

    Returns 202 Accepted immediately with the project ID and thread_id that
    can be used to poll status or retrieve the full graph state.
    """
    thread_id = str(uuid.uuid4())
    project = Project(
        thread_id=thread_id,
        name=req.name,
        spec=req.spec,
        status="running",
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    background_tasks.add_task(
        run_graph_and_update_db,
        thread_id,
        req.spec,
        project.id,
    )

    return ProjectResponse(
        id=project.id,
        thread_id=thread_id,
        name=project.name,
        status=project.status,
        revision_count=0,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: Session = Depends(get_db)):
    """Get the current status and metadata for a project."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectResponse(
        id=project.id,
        thread_id=project.thread_id,
        name=project.name,
        status=project.status,
        revision_count=project.revision_count,
        repo_url=project.repo_url,
    )


@router.get("/{project_id}/state")
async def get_project_full_state(project_id: str, db: Session = Depends(get_db)):
    """
    Return the full LangGraph ProjectState for a project thread.

    Includes: generated backend_files, frontend_files, test_results,
    review_comments, review_approved, revision_count, status, and the
    full message trace from all agents.

    Useful for debugging, human review, and building a replay UI.
    """
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    config = {"configurable": {"thread_id": project.thread_id}}
    snapshot = graph.get_state(config)
    return snapshot.values
