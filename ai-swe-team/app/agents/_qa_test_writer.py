"""
Internal helper used by qa_engineer_node to generate a pytest test suite
targeting the project's API contracts and user stories.
"""

from typing import Dict, List

from pydantic import BaseModel, Field

from app.agents.llm_factory import get_fast_llm
from app.graph.state import ProjectState


class TestFile(BaseModel):
    path: str
    content: str


class TestSuite(BaseModel):
    files: List[TestFile] = Field(
        description="List of pytest test files to execute against the implementation"
    )
    notes: str = Field(
        description="Brief description of what is covered and any assumptions"
    )


QA_WRITER_SYSTEM_PROMPT = """You are a QA Engineer specialising in API testing.

Given backend source files, the architecture's API contracts, and the PM's user
stories, write a comprehensive pytest test suite. Requirements:
- Use httpx or requests to test HTTP endpoints; do not rely on a running server -
  use FastAPI's TestClient if FastAPI is detected, otherwise mock as needed.
- Cover every API contract endpoint (happy path + key error cases).
- Cover each user story with at least one integration test.
- Tests must be self-contained and runnable with 'pytest -q'.
- Do not import packages that are not in the standard library or in:
  pytest, httpx, requests, fastapi, sqlalchemy, pydantic.
Output structured data only.
"""



def generate_tests(state: ProjectState) -> Dict[str, str]:
    """
    Ask the LLM to generate a pytest test suite for the current backend
    implementation. Returns a dict of {path: content} ready for the sandbox.
    """
    llm = get_fast_llm().with_structured_output(TestSuite)
    backend_content = {
        path: artifact["content"]
        for path, artifact in state["backend_files"].items()
    }

    result: TestSuite = llm.invoke(
        [
            ("system", QA_WRITER_SYSTEM_PROMPT),
            (
                "human",
                f"Backend source files:\n{backend_content}\n\n"
                f"API contracts:\n{state['architecture'].get('api_contracts', [])}\n\n"
                f"User stories:\n{chr(10).join(state['user_stories'])}",
            ),
        ]
    )
    return {f.path: f.content for f in result.files}
