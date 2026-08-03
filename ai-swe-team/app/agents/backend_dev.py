from typing import List

from pydantic import BaseModel, Field

from app.agents.llm_factory import get_coding_llm
from app.graph.state import FileArtifact, ProjectState
from app.tools.docs_retrieval import search_docs


class CodeFile(BaseModel):
    path: str
    content: str
    language: str


class BackendOutput(BaseModel):
    files: List[CodeFile]
    notes: str = Field(
        description="Anything the QA engineer or code reviewer should know about this implementation"
    )


BACKEND_SYSTEM_PROMPT = """You are a senior Backend Developer.

Implement the backend exactly according to the given architecture plan:
API contracts, database schema, and file structure. Write complete, runnable
code - not snippets or pseudocode. Include:
- All route handlers matching the specified API contracts
- Database models and migrations matching the schema
- A requirements.txt or equivalent dependency list
- Basic input validation and error handling on every endpoint

If QA failure logs or reviewer blocking comments are included below, address
every point before returning new files. Do not omit any previously correct
files - always return the full set.
Output structured data only; do not include conversational prose outside the schema.
"""



def backend_dev_node(state: ProjectState) -> dict:
    """
    Implements the server-side code according to the architecture plan.
    On revision passes, incorporates QA failures and blocking review comments.
    Retrieves relevant library docs via RAG to reduce API hallucination.
    """
    llm = get_coding_llm().with_structured_output(BackendOutput)
    # Pull relevant docs for the chosen stack to reduce API hallucination.
    doc_snippets: list[str] = []
    for tech in state["architecture"].get("stack", []):
        doc_snippets.extend(search_docs.invoke({"query": tech, "n_results": 3}))

    # Build feedback section for revision passes.
    feedback_parts: list[str] = []
    if state.get("test_results") and not state["test_results"]["passed"]:
        feedback_parts.append(
            f"QA failures to fix:\n{state['test_results']['failures']}\n"
            f"QA logs:\n{state['test_results']['logs']}"
        )
    if state.get("review_comments"):
        blocking = [
            c for c in state["review_comments"] if c["severity"] == "blocking"
        ]
        if blocking:
            feedback_parts.append(f"Reviewer blocking comments to address:\n{blocking}")

    feedback = "\n\n".join(feedback_parts) if feedback_parts else "None (first pass)."

    result: BackendOutput = llm.invoke(
        [
            ("system", BACKEND_SYSTEM_PROMPT),
            (
                "human",
                f"Architecture plan:\n{state['architecture']}\n\n"
                f"Relevant docs:\n{doc_snippets}\n\n"
                f"Feedback to address:\n{feedback}",
            ),
        ]
    )

    files = {
        f.path: FileArtifact(path=f.path, content=f.content, language=f.language)
        for f in result.files
    }
    return {
        "backend_files": files,
        "status": "backend_done",
        "messages": [f"[Backend Dev] Wrote {len(files)} files. {result.notes}"],
    }
