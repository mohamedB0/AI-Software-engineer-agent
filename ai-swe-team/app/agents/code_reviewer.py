from typing import List

from pydantic import BaseModel, Field

from app.agents.llm_factory import get_reasoning_llm
from app.graph.state import ProjectState, ReviewComment


class ReviewOutput(BaseModel):
    approved: bool = Field(
        description="True only if there are zero blocking issues"
    )
    comments: List[dict] = Field(
        description=(
            "List of {file, line, severity, comment} dicts. "
            "severity must be one of: 'blocking', 'suggestion', 'nit'."
        )
    )


REVIEWER_SYSTEM_PROMPT = """You are a senior Code Reviewer.

Review the given backend and frontend source files for:
1. Correctness against the architecture plan (endpoints, schema, data types)
2. Security issues: SQL injection, secrets hard-coded in code, missing auth checks,
   missing input validation
3. Code quality: readability, consistency, dead code, obvious performance issues
4. Consistency with the test results provided

Classify every issue's severity:
- blocking: must be fixed before the code can be approved
- suggestion: recommended improvement, not a blocker
- nit: minor style or formatting point

Set approved=true only if there are zero blocking issues.
Output structured data only; do not include conversational prose outside the schema.
"""



def code_reviewer_node(state: ProjectState) -> dict:
    """
    Code Reviewer agent: performs a static review of all backend and frontend files
    against the architecture plan and test results. Approves or requests changes.
    """
    llm = get_reasoning_llm().with_structured_output(ReviewOutput)
    all_files = {**state["backend_files"], **state["frontend_files"]}

    result: ReviewOutput = llm.invoke(
        [
            ("system", REVIEWER_SYSTEM_PROMPT),
            (
                "human",
                f"Architecture plan:\n{state['architecture']}\n\n"
                f"Files under review:\n{all_files}\n\n"
                f"Test results:\n{state['test_results']}",
            ),
        ]
    )

    comments = [ReviewComment(**c) for c in result.comments]
    blocking_count = sum(1 for c in comments if c["severity"] == "blocking")

    return {
        "review_comments": comments,
        "review_approved": result.approved,
        "status": "approved" if result.approved else "changes_requested",
        "messages": [
            f"[Reviewer] {'Approved.' if result.approved else f'Changes requested: {blocking_count} blocking, {len(comments) - blocking_count} other.'}"
        ],
    }
