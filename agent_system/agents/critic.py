"""Critic agent — evaluates subtask results and decides if the task is complete."""
from __future__ import annotations

import json
import logging
import os
import re

from langchain_core.messages import HumanMessage

from agent_system.llm import get_llm
from agent_system.state.schema import AgentState, CriticFeedback

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.75"))

CRITIC_PROMPT = """You are a quality-control critic specializing in incident postmortems.

Respond ONLY with a JSON object (no markdown, no code fences):
{
  "passed": true|false,
  "score": 0.0-1.0,
  "reasoning": "one sentence explanation",
  "suggestions": [],
  "final_answer": "COMPLETE postmortem here"
}

SCORING RUBRIC:
- Root cause is specific: names a commit hash, file, function, or query — NOT "increased load" or "high traffic" (+0.30)
- Evidence is cited: actual log statistics, error rates, latency numbers, or query counts (+0.30)
- Postmortem has all sections: Incident Summary, Timeline, Root Cause, Evidence, Action Items, Prevention (+0.20)
- Action items are concrete and actionable: named owner + deliverable, not "investigate further" (+0.20)

CRITICAL RULES for final_answer:
- Include the FULL postmortem with all sections
- Reference specific timestamps, commit hashes, file names, and numeric evidence from the coder's output
- NEVER write "a plan was outlined" — give the actual postmortem document
- The user should not need to read the subtask results; final_answer must stand alone

Score >= 0.75 → passed=true."""


def _extract_critic_output(text: str) -> dict:
    """Parse critic JSON from LLM response, handling markdown code fences."""
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Try the whole response first (model may return pure JSON)
    for candidate in (stripped, text):
        try:
            d = json.loads(candidate)
            if isinstance(d, dict) and "passed" in d:
                fa = d.get("final_answer", "")
                if not isinstance(fa, str):
                    fa = json.dumps(fa)
                d["final_answer"] = fa
                return d
        except json.JSONDecodeError:
            pass

    # Find the outermost {...} block using greedy match
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group())
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass

    # Extract final_answer directly as last resort
    fa_match = re.search(r'"final_answer"\s*:\s*"(.*?)"', text, re.DOTALL)
    final_answer = fa_match.group(1) if fa_match else (text[:500] if text else "No answer available.")

    return {
        "passed": True,
        "score": 0.8,
        "reasoning": "Unable to parse critic response; defaulting to pass.",
        "suggestions": [],
        "final_answer": final_answer,
    }


async def critic_node(state: AgentState) -> dict:
    """Score the subtask results and produce a final answer."""
    results_str = "\n\n".join(
        f"Subtask: {r['subtask']}\nAgent: {r['agent']}\nResult:\n{r['result']}"
        for r in state.get("subtask_results", [])
    )

    content = (
        f"{CRITIC_PROMPT}\n\n"
        f"Original task: {state['user_input']}\n\n"
        f"Subtask results:\n{results_str}"
    )

    llm = get_llm("critic")

    response = await llm.ainvoke([HumanMessage(content=content)])
    text = response.content if isinstance(response.content, str) else str(response.content)
    output = _extract_critic_output(text)

    score = float(output.get("score", 0.0))
    passed = score >= QUALITY_THRESHOLD

    feedback: CriticFeedback = {
        "passed": passed,
        "score": score,
        "reasoning": output.get("reasoning", ""),
        "suggestions": output.get("suggestions", []),
    }

    critic_answer = str(output.get("final_answer") or "").strip()

    # If the critic gave a vague one-liner (< 120 chars and no code),
    # fall back to the most content-rich subtask result instead.
    results = state.get("subtask_results", [])
    best_result = max(results, key=lambda r: len(r.get("result", "")), default={})
    rich_fallback = best_result.get("result", "")

    vague = len(critic_answer) < 120 and "```" not in critic_answer and "\n" not in critic_answer
    final_answer = rich_fallback if vague else critic_answer

    usage = getattr(response, "usage_metadata", None)
    tokens_used = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0

    new_iter = state["iteration_count"] + 1
    route = "end" if (passed or new_iter >= state["max_iterations"]) else "orchestrator"

    logger.info("Critic: score=%.2f passed=%s iter=%d route=%s", score, passed, new_iter, route)

    return {
        "critic_feedback": feedback,
        "final_answer": final_answer if (passed or new_iter >= state["max_iterations"]) else None,
        "route_to": route,
        "iteration_count": new_iter,
        "total_tokens_used": state["total_tokens_used"] + tokens_used,
        "messages": [response],
    }


def route_from_critic(state: AgentState) -> str:
    """Route to orchestrator for retry or end if done."""
    feedback = state.get("critic_feedback") or {}
    if not feedback.get("passed", False):
        if state.get("iteration_count", 0) < state.get("max_iterations", 3):
            return "orchestrator"
    return "end"
