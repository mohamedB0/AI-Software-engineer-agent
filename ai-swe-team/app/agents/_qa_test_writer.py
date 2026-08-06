"""
Internal helper used by qa_engineer_node to generate a pytest test suite
targeting the project's API contracts and user stories.
"""

import json as _json
import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.agents.llm_factory import get_fast_llm, structured_llm
from app.graph.state import ProjectState

logger = logging.getLogger(__name__)


class TestFile(BaseModel):
    path: str
    content: str

    @field_validator("content", mode="before")
    @classmethod
    def coerce_content_to_str(cls, v: Any) -> str:
        """If the LLM returns a dict/object instead of a code string, serialise it."""
        if isinstance(v, str):
            return v
        return _json.dumps(v, indent=2)


class TestSuite(BaseModel):
    files: list[TestFile] = Field(
        description="List of pytest test files to execute against the implementation"
    )
    notes: str = Field(
        default="",
        description="Brief description of what is covered and any assumptions",
    )


QA_WRITER_SYSTEM_PROMPT = """You are a QA Engineer specialising in API testing.

Given backend source files, the architecture's API contracts, and the PM's user
stories, write a pytest test suite. Requirements:
- Use FastAPI's TestClient for HTTP endpoint testing.
- Cover the key API contract endpoints (happy path only).
- Tests must be self-contained and runnable with 'pytest -q'.
- Only import from: pytest, fastapi.testclient, the backend app modules.
- Keep each test file SHORT (under 30 lines) to stay within token limits.

CRITICAL JSON FORMATTING RULES:
- The `content` field MUST be a plain JSON string.
- Escape ALL newlines as \\n, ALL double-quotes as \\".
- Do NOT use triple-quotes (\"\"\" or ''') anywhere in the JSON output.
- Do NOT use multiline strings; use \\n for line breaks inside the string value.
Output structured JSON only.
"""



def generate_tests(state: ProjectState) -> dict[str, str]:
    """
    Ask the LLM to generate a pytest test suite for the current backend
    implementation. Returns a dict of {path: content} ready for the sandbox.
    """
    llm = structured_llm(get_fast_llm(), TestSuite)
    backend_content = {
        path: artifact["content"]
        for path, artifact in state["backend_files"].items()
    }

    try:
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
    except Exception as e:
        logger.warning(
            "Test generation failed (pipeline continues without tests): %s", e
        )
        return {}
