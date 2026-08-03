from typing import Optional

from pydantic import BaseModel, Field


class CreateProjectRequest(BaseModel):
    name: str = Field(description="Human-readable name for this project run")
    spec: str = Field(description="Natural-language project specification")


class ProjectResponse(BaseModel):
    id: str
    thread_id: str
    name: str
    # pending | running | awaiting_review | done | failed
    status: str
    revision_count: int
    repo_url: Optional[str] = None
