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
        f"{SUB_AGENT_PROMPT}\n\n"
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
