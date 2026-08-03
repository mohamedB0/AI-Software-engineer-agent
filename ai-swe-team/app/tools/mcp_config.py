"""
MCP (Model Context Protocol) client configuration.

Each MCP server is defined here once. Agents call get_tools_for() to receive
only the tools relevant to their role. Tool names follow the convention:
    <server_name>__<tool_name>
"""

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings


def build_mcp_client() -> MultiServerMCPClient:
    """Build a MultiServerMCPClient covering Git, GitHub, and the docs RAG server."""
    return MultiServerMCPClient(
        {
            # Git MCP server (local, installed via pip install mcp-server-git).
            # Operates on the repository at REPO_PATH.
            "git": {
                "command": "mcp-server-git",
                "args": ["--repository", settings.repo_path],
                "transport": "stdio",
            },
            # GitHub's remotely hosted MCP server (no local install needed).
            # Requires a GitHub Personal Access Token with 'repo' scope.
            "github": {
                "url": "https://api.githubcopilot.com/mcp/",
                "transport": "sse",
                "headers": {"Authorization": f"Bearer {settings.github_token}"},
            },
            # Local documentation RAG MCP server (see docs_mcp_server.py).
            "docs": {
                "command": "python",
                "args": ["-m", "app.tools.docs_mcp_server"],
                "transport": "stdio",
            },
        }
    )


async def get_tools_for(agent_names: list[str]) -> list:
    """
    Return only the MCP tools whose server name prefix matches one of the
    supplied agent_names. This keeps each agent's tool set minimal and avoids
    leaking capabilities across agents.

    Example:
        tools = await get_tools_for(["git", "github"])
    """
    client = build_mcp_client()
    all_tools = await client.get_tools()
    return [
        t
        for t in all_tools
        if any(t.name.startswith(f"{name}__") for name in agent_names)
    ]
