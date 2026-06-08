"""Tool registry mapping agent roles to their allowed tools."""
from __future__ import annotations

import asyncio
import json
import os

from langchain_core.tools import tool

AGENT_TOOL_SCOPES: dict[str, list[str]] = {
    "orchestrator": ["web_search"],
    "researcher":   ["web_search", "rag_retrieval", "read_file"],
    "coder":        ["execute_python", "read_file", "write_file", "read_log_file", "list_log_files", "query_git_log"],
    "analyst":      ["run_sql_query", "execute_python", "rag_retrieval", "read_log_file"],
    "critic":       [],
}


@tool
async def web_search(query: str) -> str:
    """Search the web for information using Tavily."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
        results = client.search(query, max_results=3)
        return json.dumps([r["content"] for r in results.get("results", [])])
    except Exception as e:
        return f"Search unavailable: {e}"


@tool
async def execute_python(code: str) -> str:
    """Execute Python code in a sandboxed subprocess."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            return f"Error:\n{stderr.decode()}"
        return stdout.decode()
    except asyncio.TimeoutError:
        return "Execution timed out (30s limit)"
    except Exception as e:
        return f"Execution failed: {e}"


@tool
async def read_file(path: str) -> str:
    """Read a file from the agent workspace."""
    workspace = os.getenv("AGENT_WORKSPACE", "/tmp/agent_workspace")
    safe_path = os.path.join(workspace, os.path.basename(path))
    try:
        with open(safe_path) as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {path}"


@tool
async def write_file(path: str, content: str) -> str:
    """Write content to a file in the agent workspace."""
    workspace = os.getenv("AGENT_WORKSPACE", "/tmp/agent_workspace")
    os.makedirs(workspace, exist_ok=True)
    safe_path = os.path.join(workspace, os.path.basename(path))
    with open(safe_path, "w") as f:
        f.write(content)
    return f"Written {len(content)} bytes to {safe_path}"


@tool
async def read_log_file(path: str, tail_lines: int = 500, pattern: str = "") -> str:
    """Read a log file from an absolute path, optionally filtered by a regex pattern. Returns last N lines."""
    import re as _re
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        if pattern:
            regex = _re.compile(pattern, _re.IGNORECASE)
            lines = [l for l in lines if regex.search(l)]
        lines = lines[-tail_lines:]
        return f"Showing {len(lines)} lines from {path}:\n" + "".join(lines)
    except FileNotFoundError:
        return f"Log file not found: {path}"
    except Exception as e:
        return f"Error reading log file: {e}"


@tool
async def list_log_files(path: str) -> str:
    """Recursively list log and config files in a directory (.log, .txt, .json, .yaml)."""
    import os as _os
    exts = (".log", ".txt", ".json", ".yaml", ".yml")
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv"}
    files = []
    try:
        if not _os.path.exists(path):
            return f"Error: path does not exist: {path}"
        for root, dirs, filenames in _os.walk(path):
            dirs[:] = [d for d in dirs if d not in skip]
            for fn in filenames:
                if fn.endswith(exts):
                    files.append(_os.path.join(root, fn))
        if not files:
            return f"No log/config files found in {path}"
        return f"Found {len(files)} files:\n" + "\n".join(files[:100])
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
async def query_git_log(repo_path: str, since_hours: int = 24) -> str:
    """Get recent git commits for a repository to correlate with incident timing."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", repo_path, "log",
            f"--since={since_hours} hours ago",
            "--format=%h|%ae|%ad|%s",
            "--date=iso-strict",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode().strip()
        if not output:
            return f"No commits in the last {since_hours} hours in {repo_path}"
        return f"Recent commits (last {since_hours}h):\n{output[:3000]}"
    except asyncio.TimeoutError:
        return "git log timed out"
    except Exception as e:
        return f"git log failed: {e}"


@tool
async def rag_retrieval(query: str) -> str:
    """Retrieve relevant documents from the RAG service."""
    rag_url = os.getenv("RAG_SERVICE_URL", "")
    if not rag_url:
        return "RAG service not configured"
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{rag_url}/retrieve",
                json={"query": query, "top_k": 5},
                timeout=10,
            )
            resp.raise_for_status()
            return json.dumps(resp.json())
    except Exception as e:
        return f"RAG retrieval failed: {e}"


@tool
async def run_sql_query(query: str) -> str:
    """Execute a read-only SQL query against the analytics database."""
    db_url = os.getenv("ANALYTICS_DB_URL", "")
    if not db_url:
        return "Analytics database not configured"
    try:
        import asyncpg
        conn = await asyncpg.connect(db_url)
        try:
            rows = await conn.fetch(query)
            return json.dumps([dict(r) for r in rows])
        finally:
            await conn.close()
    except Exception as e:
        return f"Query failed: {e}"


_ALL_TOOLS = {
    "web_search":     web_search,
    "execute_python": execute_python,
    "read_file":      read_file,
    "write_file":     write_file,
    "read_log_file":  read_log_file,
    "list_log_files": list_log_files,
    "query_git_log":  query_git_log,
    "rag_retrieval":  rag_retrieval,
    "run_sql_query":  run_sql_query,
}


class ToolRegistry:
    """Maps agent roles to their permitted tools."""

    def __init__(self, tools: dict):
        """Initialise with a dict of name→tool."""
        self._tools = tools

    def get_tools(self, agent_role: str) -> list:
        """Return the list of tools available to the given agent role."""
        names = AGENT_TOOL_SCOPES.get(agent_role, [])
        return [self._tools[n] for n in names if n in self._tools]

    def get_tool_names(self, agent_role: str) -> list[str]:
        """Return the names of tools available to the given agent role."""
        return AGENT_TOOL_SCOPES.get(agent_role, [])


async def create_tool_registry() -> ToolRegistry:
    """Build and return the global tool registry."""
    return ToolRegistry(_ALL_TOOLS)
