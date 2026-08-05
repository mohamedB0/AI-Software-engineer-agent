"""
MCP server wrapper that exposes search_docs over stdio.

Run as:
    python -m app.tools.docs_mcp_server

The MCP client in mcp_config.py launches this as a subprocess and communicates
with it over stdin/stdout using the MCP stdio transport.
"""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from app.tools.docs_retrieval import _get_collection

server = Server("docs")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_docs",
            description=(
                "Search indexed library/framework documentation for relevant snippets. "
                "Use this before writing code to look up real API signatures."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the API or concept",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Maximum number of chunks to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "search_docs":
        raise ValueError(f"Unknown tool: {name}")
    query = arguments["query"]
    n_results = arguments.get("n_results", 5)
    collection = _get_collection()
    if collection is None:
        return [TextContent(type="text", text="(RAG disabled — no embedding API key configured)")]
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0] if results["documents"] else []
    return [TextContent(type="text", text="\n---\n".join(docs))]


if __name__ == "__main__":
    import asyncio

    asyncio.run(stdio_server(server))
