"""Orchestrator agent — plans and dispatches subtasks."""
from __future__ import annotations

import asyncio
import json
import logging
import re

from langchain_core.messages import HumanMessage

from agent_system.llm import get_llm
from agent_system.state.schema import AgentState

logger = logging.getLogger(__name__)

ORCHESTRATOR_SYSTEM_PROMPT = """You are an incident response orchestrator. When given an alert or service degradation report, break the investigation into exactly 3 subtasks.

Available agent types:
- researcher: searches for known causes of this alert type, similar historical incidents, and relevant runbook patterns
- coder: reads log files, queries git history, and computes statistics to find the evidence trail
- analyst: correlates coder evidence with researcher context, ranks root cause hypotheses, and drafts a postmortem

ALWAYS use this 3-subtask plan:
1. researcher subtask: "Research known causes and patterns for this alert type" (no depends_on)
2. coder subtask: "Parse logs and query git history to find the evidence trail" (no depends_on)
3. analyst subtask: "Correlate findings, identify root cause, and draft postmortem" (depends_on both above)

Respond ONLY with a JSON object — no markdown, no explanation, no code fences:
{
  "task_plan": [
    {"subtask": "Research known causes and patterns for this alert type", "agent": "researcher", "depends_on": [], "status": "pending"},
    {"subtask": "Parse logs and query git history to find the evidence trail", "agent": "coder", "depends_on": [], "status": "pending"},
    {"subtask": "Correlate findings, identify root cause, and draft postmortem", "agent": "analyst", "depends_on": ["Research known causes and patterns for this alert type", "Parse logs and query git history to find the evidence trail"], "status": "pending"}
  ]
}"""


def _default_plan() -> list[dict]:
    return [
        {"subtask": "Research known causes and patterns for this alert type",   "agent": "researcher", "depends_on": [], "status": "pending"},
        {"subtask": "Parse logs and query git history to find the evidence trail", "agent": "coder",      "depends_on": [], "status": "pending"},
        {"subtask": "Correlate findings, identify root cause, and draft postmortem", "agent": "analyst",   "depends_on": [
            "Research known causes and patterns for this alert type",
            "Parse logs and query git history to find the evidence trail",
        ], "status": "pending"},
    ]


def _extract_task_plan(text: str) -> list[dict]:
    """Parse task_plan JSON from an LLM response, handling markdown code fences."""
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    for candidate in (stripped, text):
        try:
            data = json.loads(candidate)
            plan = data.get("task_plan", [])
            if isinstance(plan, list) and plan:
                return plan
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                plan = data.get("task_plan", [])
                if isinstance(plan, list) and plan:
                    return plan
            except json.JSONDecodeError:
                pass

    return _default_plan()


async def orchestrator_node(state: AgentState) -> dict:
    """Plan subtasks on first iteration; revise plan on retry.

    Uses a direct LLM call (no ReAct loop) — the orchestrator's only job
    is to output a JSON plan, so tool-calling overhead is wasted here.
    """
    llm = get_llm("orchestrator")

    if state["iteration_count"] == 0:
        content = f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\nTask: {state['user_input']}"
    else:
        results_str = "\n".join(
            f"- {r['subtask']} ({r['agent']}): {r['result'][:300]}"
            for r in state["subtask_results"]
        )
        feedback = state.get("critic_feedback") or {}
        content = (
            f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n"
            f"Task: {state['user_input']}\n\n"
            f"Previous results:\n{results_str}\n\n"
            f"Critic feedback: {feedback.get('reasoning', '')}\n"
            f"Suggestions: {', '.join(feedback.get('suggestions', []))}\n\n"
            "Output a revised JSON plan addressing the feedback."
        )

    task_plan = _default_plan()
    tokens_used = 0

    try:
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=content)]),
            timeout=60,
        )
        text = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _extract_task_plan(text)
        if parsed:
            task_plan = parsed
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            tokens_used = usage.get("total_tokens", 0)
    except asyncio.TimeoutError:
        logger.error("Orchestrator LLM call timed out — using default 3-subtask plan")
    except Exception as e:
        logger.exception("Orchestrator failed: %s — using default plan", e)

    logger.info("Orchestrator planned %d subtasks (iter=%d)", len(task_plan), state["iteration_count"])
    return {
        "task_plan": task_plan,
        "messages": [],
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
