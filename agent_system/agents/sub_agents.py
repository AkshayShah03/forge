"""Researcher, coder, and analyst sub-agents."""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agent_system.agents.orchestrator import route_from_orchestrator
from agent_system.llm import get_llm
from agent_system.state.schema import AgentState, SubtaskResult
from agent_system.tools.registry import create_tool_registry

logger = logging.getLogger(__name__)

ROLE_PROMPTS = {
    "researcher": """You are an incident response researcher. Given an alert or service degradation description:
1. Use web_search to find: (a) common root causes for this alert type, (b) known failure patterns for the affected service/component, (c) relevant runbook or postmortem patterns from engineering blogs
2. Return a ranked list of likely root causes with brief reasoning for each.
Be specific to the technology mentioned (e.g. if a Python service, mention GIL contention, GC pauses, asyncio event loop blocking).""",

    "coder": """You are an incident investigator. Given an alert with log paths and/or a repository path:
1. Use list_log_files to discover log files in the incident directory
2. Use read_log_file to read relevant logs (filter by pattern if the alert mentions a specific error or endpoint)
3. Use execute_python to parse log lines and compute: error rates, P50/P95/P99 latency, request volume, and timestamp of the anomaly onset
4. Use query_git_log on the repository path to find deploys in the 2 hours before the anomaly
5. Return structured findings: {anomaly_start: ISO timestamp, affected_endpoints: [...], stats: {...}, recent_commits: [...]}
Be precise. Return actual numbers from the logs.""",

    "analyst": """You are a reliability engineer. Given researcher findings (likely causes) and coder findings (evidence from logs and git):
1. Correlate the anomaly timestamp with recent commits — identify the most likely causal commit
2. Rank root cause hypotheses by confidence (0–100%) based on the evidence
3. Draft a complete postmortem with these sections:

## Incident Summary
[one paragraph: what happened, when, impact]

## Timeline
[bullet list with ISO timestamps: alert fired, anomaly onset, deploy that preceded it, etc.]

## Root Cause (confidence: N%)
[specific: name the commit hash, file, function, or query responsible — not "increased load"]

## Evidence
[bullet list: log statistics, query counts, error rates that support the root cause]

## Action Items
[3–5 concrete tasks: what to fix, who owns it, when]

## Prevention
[how to detect this pattern earlier]

Be specific. Reference exact log statistics and commit hashes from the coder's output.""",
}

SUB_AGENT_PROMPT = """You are a specialist agent. Complete your assigned subtask thoroughly and concisely.
Return your findings or results as plain text. Be specific and actionable."""


async def _run_sub_agent(state: AgentState, role: str) -> dict:
    """Find the next pending subtask for this role and execute it."""
    completed = {r["subtask"] for r in state.get("subtask_results", [])}
    plan = state.get("task_plan", [])

    target: Optional[dict] = None
    for subtask in plan:
        if subtask.get("agent") != role:
            continue
        if subtask["subtask"] in completed:
            continue
        deps = subtask.get("depends_on", [])
        if all(dep in completed for dep in deps):
            target = subtask
            break

    if not target:
        logger.warning("No pending %s subtask found", role)
        return {}

    registry = await create_tool_registry()
    tools = registry.get_tools(role)

    llm = get_llm(role)

    prior = "\n".join(
        f"- {r['subtask']}: {r['result'][:200]}"
        for r in state.get("subtask_results", [])
    )

    content = (
        f"{ROLE_PROMPTS.get(role, SUB_AGENT_PROMPT)}\n\n"
        f"Original task: {state['user_input']}\n\n"
        f"Your subtask: {target['subtask']}\n\n"
        + (f"Prior results:\n{prior}\n\n" if prior else "")
    )

    agent = create_react_agent(llm, tools)
    result = await agent.ainvoke({"messages": [HumanMessage(content=content)]})

    last_msg = result["messages"][-1]
    result_text = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    tokens_used = 0
    for msg in result["messages"]:
        meta = getattr(msg, "usage_metadata", None)
        if meta:
            tokens_used += meta.get("total_tokens", 0)

    updated_plan = []
    for s in plan:
        if s["subtask"] == target["subtask"]:
            updated_plan.append({**s, "status": "complete"})
        else:
            updated_plan.append(s)

    new_result: SubtaskResult = {
        "subtask": target["subtask"],
        "agent": role,
        "result": result_text,
        "tokens_used": tokens_used,
    }

    logger.info("%s completed subtask: %s", role, target["subtask"])
    return {
        "task_plan": updated_plan,
        "subtask_results": [new_result],
        "messages": result["messages"],
        "total_tokens_used": state["total_tokens_used"] + tokens_used,
    }


async def researcher_node(state: AgentState) -> dict:
    """Execute the next pending researcher subtask."""
    return await _run_sub_agent(state, "researcher")


async def coder_node(state: AgentState) -> dict:
    """Execute the next pending coder subtask."""
    return await _run_sub_agent(state, "coder")


async def analyst_node(state: AgentState) -> dict:
    """Execute the next pending analyst subtask."""
    return await _run_sub_agent(state, "analyst")


def route_after_subagent(state: AgentState) -> str:
    """Route to next unblocked subtask, critic, or end after a sub-agent completes."""
    return route_from_orchestrator(state)
