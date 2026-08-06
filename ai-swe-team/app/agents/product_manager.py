
from pydantic import BaseModel, Field

from app.agents.llm_factory import get_reasoning_llm, structured_llm
from app.graph.state import ProjectState


class Requirements(BaseModel):
    requirements: str = Field(
        description="Clear, structured functional and non-functional requirements"
    )
    user_stories: list[str] = Field(
        description="User stories in 'As a ... I want ... so that ...' form"
    )


PM_SYSTEM_PROMPT = """You are a senior Product Manager on a software team.

Given a raw, possibly vague natural-language project specification, produce:
1. Clear functional requirements (what the system must do)
2. Clear non-functional requirements (performance, security, scale, if implied)
3. A set of concrete, testable user stories

Be concrete. Resolve ambiguity by making explicit, reasonable assumptions and
stating them. Do not design the technical solution - that is the Architect's job.
Output structured data only; do not include conversational prose outside the schema.
"""



def product_manager_node(state: ProjectState) -> dict:
    """
    Converts the raw project specification into structured requirements and
    user stories. This is the graph's entry point.
    """
    llm = structured_llm(get_reasoning_llm(), Requirements)
    result: Requirements = llm.invoke(
        [
            ("system", PM_SYSTEM_PROMPT),
            ("human", f"Project specification:\n\n{state['spec']}"),
        ]
    )
    return {
        "requirements": result.requirements,
        "user_stories": result.user_stories,
        "status": "requirements_done",
        "messages": [f"[PM] Produced {len(result.user_stories)} user stories."],
    }
