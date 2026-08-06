"""
End-to-end smoke test: runs the full agent graph against a sample spec and
streams progress node-by-node to the terminal.

Run from the project root:
    python scripts/run_sample_project.py

Prerequisites:
    - .env file populated with at least ANTHROPIC_API_KEY and DATABASE_URL
    - PostgreSQL running and init_db.py executed at least once
    - Docker running with the sandbox image built:
        docker build -t ai-swe-sandbox:latest -f Dockerfile.sandbox .
"""

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.config import settings
from app.graph.build_graph import build_graph

graph = build_graph()

SAMPLE_SPEC = """
Build a small task-tracking API: users can create projects, add tasks to a
project with a title, description, and due date, mark tasks as complete, and
list tasks filtered by status. Include a minimal web UI to view and manage tasks.
"""

config = {"configurable": {"thread_id": str(uuid.uuid4())}}
initial_state = {
    "spec": SAMPLE_SPEC,
    "revision_count": 0,
    "max_revisions": settings.max_revisions,
    "status": "planning",
    "review_comments": [],
    "messages": [],
}

print("Starting graph run...")
print("-" * 60)

for step in graph.stream(initial_state, config=config, stream_mode="values"):
    status = step.get("status", "unknown")
    print(f"[{status}]")
    # Print only the most recent message from each step to avoid flooding output.
    messages = step.get("messages", [])
    if messages:
        print(f"  {messages[-1]}")

final = graph.get_state(config).values
print("-" * 60)
print("Final state summary:")
print(
    json.dumps(
        {
            "status": final.get("status"),
            "revision_count": final.get("revision_count"),
            "backend_files": list(final.get("backend_files", {}).keys()),
            "frontend_files": list(final.get("frontend_files", {}).keys()),
            "test_passed": final.get("test_results", {}).get("passed"),
            "review_approved": final.get("review_approved"),
        },
        indent=2,
    )
)
