
from pydantic import BaseModel, Field

from app.agents.llm_factory import get_coding_llm, structured_llm
from app.graph.state import FileArtifact, ProjectState


class CodeFile(BaseModel):
    path: str
    content: str
    language: str


class FrontendOutput(BaseModel):
    files: list[CodeFile]
    notes: str = Field(
        description="Anything the QA engineer or code reviewer should know about this implementation"
    )


FRONTEND_SYSTEM_PROMPT = """You are a senior Frontend Developer.

Implement the client application against the architecture's API contracts.
Write complete, runnable code - components, routing, API client, styles.
Match each of the PM's user stories to a concrete UI flow.

If reviewer blocking comments are included below, address every point before
returning new files. Do not omit any previously correct files - always return
the full set.
Output structured data only; do not include conversational prose outside the schema.
"""



def frontend_dev_node(state: ProjectState) -> dict:
    """
    Implements the client-side code against the architecture's API contracts.
    Runs in parallel with backend_dev_node. On revision passes, incorporates
    blocking review comments.
    """
    llm = structured_llm(get_coding_llm(), FrontendOutput)
    feedback_parts: list[str] = []
    if state.get("review_comments"):
        blocking = [
            c for c in state["review_comments"] if c["severity"] == "blocking"
        ]
        if blocking:
            feedback_parts.append(f"Reviewer blocking comments to address:\n{blocking}")

    feedback = "\n\n".join(feedback_parts) if feedback_parts else "None (first pass)."

    result: FrontendOutput = llm.invoke(
        [
            ("system", FRONTEND_SYSTEM_PROMPT),
            (
                "human",
                f"Architecture / API contracts:\n{state['architecture']}\n\n"
                f"User stories:\n{chr(10).join(state['user_stories'])}\n\n"
                f"Feedback to address:\n{feedback}",
            ),
        ]
    )

    files = {
        f.path: FileArtifact(path=f.path, content=f.content, language=f.language)
        for f in result.files
    }
    return {
        "frontend_files": files,
        "messages": [f"[Frontend Dev] Wrote {len(files)} files. {result.notes}"],
    }
