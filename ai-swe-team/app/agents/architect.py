
from pydantic import BaseModel, Field

from app.agents.llm_factory import get_reasoning_llm, structured_llm
from app.graph.state import ProjectState


class ArchitecturePlan(BaseModel):
    stack: list[str] = Field(
        description="Concrete technology choices, e.g. 'FastAPI', 'PostgreSQL', 'React+Vite'"
    )
    file_tree: list[str] = Field(
        description="Planned file/module paths, e.g. 'backend/app/main.py'"
    )
    api_contracts: list[dict] = Field(
        description="List of {method, path, request_schema, response_schema} dicts"
    )
    db_schema: dict = Field(
        description="Table/collection definitions keyed by name"
    )
    rationale: str = Field(
        description="Short justification of key design decisions"
    )


ARCHITECT_SYSTEM_PROMPT = """You are a senior Software Architect.

Given product requirements and user stories, design a concrete, buildable technical plan:
- Choose a specific tech stack appropriate to the requirements' scale and constraints.
- Define the file/module structure the developers will implement against.
- Specify REST API contracts precisely enough that backend and frontend developers
  can work in parallel without conflicting on endpoints or schemas.
- Define the database schema.

Favor simple, well-supported technology choices over novel ones unless the
requirements specifically demand otherwise.
Output structured data only; do not include conversational prose outside the schema.
"""



def architect_node(state: ProjectState) -> dict:
    """
    Translates requirements into a concrete technical plan: stack, file tree,
    API contracts, and DB schema. Backend and Frontend developers run in parallel
    off this node's output.
    """
    llm = structured_llm(get_reasoning_llm(), ArchitecturePlan)
    result: ArchitecturePlan = llm.invoke(
        [
            ("system", ARCHITECT_SYSTEM_PROMPT),
            (
                "human",
                f"Requirements:\n{state['requirements']}\n\n"
                f"User stories:\n{chr(10).join(state['user_stories'])}",
            ),
        ]
    )
    return {
        "architecture": result.model_dump(),
        "status": "architecture_done",
        "messages": [f"[Architect] Chose stack: {', '.join(result.stack)}"],
    }
