from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.api.routes_projects import router as projects_router

app = FastAPI(
    title="AI Software Engineering Team",
    description=(
        "A multi-agent system where a Product Manager, Software Architect, "
        "Backend Developer, Frontend Developer, QA Engineer, and Code Reviewer "
        "collaborate through a stateful LangGraph to turn a natural-language "
        "specification into a working, tested, reviewed codebase."
    ),
    version="1.0.0",
)

app.include_router(projects_router)


@app.get("/health", tags=["health"])
async def health():
    """Liveness check endpoint."""
    return {"status": "ok"}
