"""Researcher, coder, and analyst sub-agents."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agent_system.agents.orchestrator import route_from_orchestrator
from agent_system.llm import get_llm
from agent_system.state.schema import AgentState, SubtaskResult
from agent_system.tools.registry import create_tool_registry

logger = logging.getLogger(__name__)

# Maximum steps the ReAct loop can take (LLM call + tool call = 2 steps).
# Prevents infinite retries when tools are unavailable.
_REACT_RECURSION_LIMIT = 12

# Per-agent wall-clock timeout in seconds.
_AGENT_TIMEOUT = 120

ROLE_PROMPTS = {
    "researcher": """\
You are an incident response researcher. Your only job is to produce a ranked list of root cause hypotheses.

STRICT RULES — follow exactly:
1. Call web_search at most ONCE. If it returns any error or "unavailable", do NOT call it again.
2. Do NOT call rag_retrieval or read_file unless you have a specific, known path.
3. After at most one tool attempt, write your findings using your training knowledge.
4. Stop immediately after providing your findings — do not add commentary or ask follow-up questions.

Given the incident description, output:
## Likely Root Causes (ranked)
For each cause: name, why it fits this incident, key diagnostic signals, and immediate fix.

Use technology-specific knowledge:
- Python services: GIL contention, asyncio event loop blocking, GC pauses, connection pool exhaustion
- Redis: connection pool exhaustion, eviction policy, slow commands (KEYS, SORT), replication lag
- Databases: N+1 queries, missing indexes, lock contention, autovacuum, connection limits
- Memory: object retention, circular references, unbounded caches, memory-mapped files
- Latency: serialization overhead, DNS resolution, TLS handshake, cold starts""",

    "coder": """\
You are an incident investigator. Extract hard evidence from logs and git history.

STRICT RULES — follow exactly:
1. Call each tool at most ONCE with a given set of arguments. Never retry the same call.
2. If list_log_files returns "path does not exist", note it and move to the next step immediately.
3. After checking all provided paths, write your findings and STOP — do not loop.

INVESTIGATION STEPS (in order):
1. If a log directory path is mentioned: call list_log_files(path) ONCE.
   - If files exist: call read_log_file for the most relevant 1-2 files (checkout/service/app logs first).
   - If path missing: write "Log path <path> not accessible on this host."
2. If log content was retrieved: call execute_python ONCE to compute:
   - Error rate (errors / total requests × 100)
   - P50, P95, P99 latency
   - DB query count per request (if present)
   - Exact timestamp where metrics first crossed threshold
3. If a repo path is mentioned: call query_git_log(repo_path, since_hours=4) ONCE.
4. STOP and write your structured findings.

Output format:
- Log status: [accessible with N lines | not accessible]
- Anomaly onset: [ISO timestamp or "not determinable"]
- Key statistics: [actual numbers or "logs unavailable"]
- Recent commits: [list or "no repo provided"]
- Evidence summary: [2-3 sentences connecting timing to commits/metrics]""",

    "analyst": """\
You are a reliability engineer writing an incident postmortem.

You receive two sets of findings:
- Researcher findings: known root cause patterns for this alert type
- Coder findings: evidence from logs and git history (may be limited if logs were inaccessible)

Your job: correlate the evidence with the patterns and write a complete postmortem.

If coder found real log data: anchor every claim to specific numbers (e.g. "P95 rose from 45ms to 385ms").
If logs were inaccessible: write the postmortem based on the pattern match, note the data gap, and recommend what logs to check.

OUTPUT — write exactly these sections, in order:

## Incident Summary
One paragraph: what failed, when it started, what the user impact was.

## Timeline
Bullet list with timestamps. Include: alert time, anomaly onset, any deploy or config change that preceded it.

## Root Cause (confidence: N%)
One specific sentence naming the cause: a commit hash, a missing index, an exhausted connection pool, a removed ORM clause — not "increased traffic" or "high load".

## Evidence
Bullet list of supporting facts: actual log statistics, error rates, latency numbers, query counts, commit message. Quote numbers directly.

## Action Items
3-5 items, each with: what to do, who owns it (team/role), by when.

## Prevention
2-3 specific detection or prevention measures: alerting thresholds, code review checks, load tests.

Be concise and specific. A recruiter or senior engineer reading this should learn exactly what broke and how to prevent it.""",
}

SUB_AGENT_PROMPT = """\
You are a specialist agent. Complete your assigned subtask thoroughly.
Return your findings as plain text. Be specific and actionable.
Do not retry tools that return errors — use your training knowledge instead."""


async def _direct_llm_summary(role: str, subtask: str, user_input: str) -> str:
    """Fallback: call the LLM directly without tools when the ReAct agent times out."""
    llm = get_llm(role)
    prompt = (
        f"{ROLE_PROMPTS.get(role, SUB_AGENT_PROMPT)}\n\n"
        f"Incident: {user_input[:800]}\n\n"
        f"Task: {subtask}\n\n"
        "NOTE: Tool access is unavailable. Answer using your training knowledge only. "
        "Be specific about what you would find and why."
    )
    try:
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=45,
        )
        text = response.content if isinstance(response.content, str) else str(response.content)
        return f"[Tool access unavailable — knowledge-based analysis]\n\n{text}"
    except Exception as e:
        logger.exception("Fallback LLM call failed for %s: %s", role, e)
        return f"[{role} could not complete analysis: {e}]"


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
        f"- {r['subtask']} ({r['agent']}): {r['result'][:400]}"
        for r in state.get("subtask_results", [])
    )

    content = (
        f"{ROLE_PROMPTS.get(role, SUB_AGENT_PROMPT)}\n\n"
        f"Incident description: {state['user_input']}\n\n"
        f"Your subtask: {target['subtask']}\n\n"
        + (f"Prior agent findings:\n{prior}\n\n" if prior else "")
    )

    result_text = ""
    result_messages: list = []
    tokens_used = 0

    if tools:
        agent = create_react_agent(llm, tools)
        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [HumanMessage(content=content)]},
                    config={"recursion_limit": _REACT_RECURSION_LIMIT},
                ),
                timeout=_AGENT_TIMEOUT,
            )
            last_msg = result["messages"][-1]
            result_text = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
            result_messages = result["messages"]
            for msg in result_messages:
                meta = getattr(msg, "usage_metadata", None)
                if isinstance(meta, dict):
                    tokens_used += meta.get("total_tokens", 0)
        except asyncio.TimeoutError:
            logger.warning("%s timed out after %ds — falling back to direct LLM", role, _AGENT_TIMEOUT)
            result_text = await _direct_llm_summary(role, target["subtask"], state["user_input"])
        except Exception as e:
            logger.exception("%s agent raised an exception: %s", role, e)
            result_text = await _direct_llm_summary(role, target["subtask"], state["user_input"])
    else:
        # No tools (e.g. analyst in some configurations) — direct LLM call
        try:
            response = await asyncio.wait_for(
                llm.ainvoke([HumanMessage(content=content)]),
                timeout=_AGENT_TIMEOUT,
            )
            result_text = response.content if isinstance(response.content, str) else str(response.content)
            result_messages = [response]
            usage = getattr(response, "usage_metadata", None)
            if isinstance(usage, dict):
                tokens_used = usage.get("total_tokens", 0)
        except asyncio.TimeoutError:
            logger.warning("%s direct call timed out", role)
            result_text = f"[{role} timed out]"
        except Exception as e:
            logger.exception("%s direct call failed: %s", role, e)
            result_text = f"[{role} error: {e}]"

    updated_plan = [
        {**s, "status": "complete"} if s["subtask"] == target["subtask"] else s
        for s in plan
    ]

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
        "messages": result_messages,
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
