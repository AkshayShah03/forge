"""Orchestrator agent — plans and dispatches subtasks."""
from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agent_system.llm import get_llm
from agent_system.state.schema import AgentState
from agent_system.tools.registry import create_tool_registry

logger = logging.getLogger(__name__)

ORCHESTRATOR_SYSTEM_PROMPT = """You are an orchestrator agent. Your job is to break a task into subtasks and assign each to a specialist agent.

Available agent types:
- researcher: web search, document retrieval, fact-finding
- coder: write and execute Python code
- analyst: SQL queries, data analysis, statistics

Respond ONLY with a JSON object in this exact format:
{
  "task_plan": [
    {
      "subtask": "description of the subtask",
      "agent": "researcher|coder|analyst",
      "depends_on": [],
      "status": "pending"
    }
  ]
}

Keep plans simple: 1-3 subtasks. Each depends_on entry is the exact "subtask" string of a prerequisite."""


def _extract_task_plan(text: str) -> list[dict]:
    """Parse task_plan JSON from an LLM response text, handling markdown code fences."""
    # Strip markdown code fences
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    for candidate in (stripped, text):
        # Try parsing the whole candidate
        try:
            data = json.loads(candidate)
            plan = data.get("task_plan", [])
            if isinstance(plan, list) and plan:
                return plan
        except json.JSONDecodeError:
            pass
        # Find outermost {...} block
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                plan = data.get("task_plan", [])
                if isinstance(plan, list) and plan:
                    return plan
            except json.JSONDecodeError:
                pass

    try:
        data = json.loads(text)
        return data.get("task_plan", [])
    except json.JSONDecodeError:
        pass
    return [{"subtask": "Complete the task", "agent": "researcher", "depends_on": [], "status": "pending"}]


async def orchestrator_node(state: AgentState) -> dict:
    """Plan subtasks on first iteration; revise plan on retry."""
    registry = await create_tool_registry()
    tools = registry.get_tools("orchestrator")

    llm = get_llm("orchestrator")

    if state["iteration_count"] == 0:
        content = (
            f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n"
            f"Task: {state['user_input']}"
        )
    else:
        results_str = "\n".join(
            f"- {r['subtask']} ({r['agent']}): {r['result'][:200]}"
            for r in state["subtask_results"]
        )
        feedback = state.get("critic_feedback") or {}
        content = (
            f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n"
            f"Task: {state['user_input']}\n\n"
            f"Previous iteration results:\n{results_str}\n\n"
            f"Critic feedback: {feedback.get('reasoning', 'No feedback')}\n"
            f"Suggestions: {', '.join(feedback.get('suggestions', []))}\n\n"
            f"Revise or create a new task plan to address the feedback."
        )

    agent = create_react_agent(llm, tools)
    result = await agent.ainvoke({"messages": [HumanMessage(content=content)]})

    last_msg = result["messages"][-1]
    text = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
    task_plan = _extract_task_plan(text)

    tokens_used = 0
    for msg in result["messages"]:
        meta = getattr(msg, "usage_metadata", None)
        if meta:
            tokens_used += meta.get("total_tokens", 0)

    logger.info("Orchestrator planned %d subtasks (iter=%d)", len(task_plan), state["iteration_count"])
    return {
        "task_plan": task_plan,
        "messages": result["messages"],
        "total_tokens_used": state["total_tokens_used"] + tokens_used,
    }


def route_from_orchestrator(state: AgentState) -> str:
    """Route to the first unblocked subtask, critic, or end."""
    if state["total_tokens_used"] >= state["token_budget"]:
        logger.warning("Token budget exhausted — routing to end")
        return "end"

    completed = {r["subtask"] for r in state.get("subtask_results", [])}
    plan = state.get("task_plan", [])

    if not plan:
        return "critic"

    for subtask in plan:
        if subtask.get("status") == "complete":
            continue
        if subtask["subtask"] in completed:
            continue
        deps = subtask.get("depends_on", [])
        if all(dep in completed for dep in deps):
            agent = subtask.get("agent", "researcher")
            logger.info("Routing to %s for subtask: %s", agent, subtask["subtask"])
            return agent

    return "critic"
