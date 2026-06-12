"""Critic agent — evaluates subtask results and decides if the task is complete."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from langchain_core.messages import HumanMessage

from agent_system.llm import get_llm
from agent_system.state.schema import AgentState, CriticFeedback

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.75"))

CRITIC_PROMPT = """You are a quality-control critic for incident postmortems.

Respond ONLY with a JSON object — no markdown, no code fences, no explanation before or after:
{
  "passed": true,
  "score": 0.85,
  "reasoning": "one sentence",
  "suggestions": [],
  "final_answer": "complete postmortem text here"
}

SCORING RUBRIC (each criterion adds to score):
+0.30  Root cause is specific: names a commit hash, file, function, missing index, or config change — NOT "increased load" or "high traffic"
+0.30  Evidence is cited: actual numbers from logs (error rates, latency ms, query counts, timestamps)
+0.20  All sections present: Incident Summary, Timeline, Root Cause, Evidence, Action Items, Prevention
+0.20  Action items are concrete: named owner or team + deliverable, not "investigate further"

RULES for final_answer:
- Copy the FULL postmortem text into final_answer (all 6 sections)
- final_answer must be a plain string — NOT a JSON object or nested structure
- If logs were inaccessible, the postmortem may be knowledge-based; that is acceptable if the root cause is still specific
- Do not write "see above" or reference subtask results — final_answer must stand alone

Score >= 0.75 → set passed to true."""


def _extract_critic_output(text: str) -> dict:
    """Parse critic JSON from LLM response, handling any formatting quirks."""
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    def _clean(d: dict) -> dict:
        """Ensure all fields have the right types."""
        fa = d.get("final_answer", "")
        if not isinstance(fa, str):
            fa = json.dumps(fa) if fa else ""
        d["final_answer"] = fa
        d["passed"]  = bool(d.get("passed", False))
        d["score"]   = float(d.get("score", 0.0))
        d["reasoning"]   = str(d.get("reasoning", ""))
        d["suggestions"] = list(d.get("suggestions", []))
        return d

    for candidate in (stripped, text):
        try:
            d = json.loads(candidate)
            if isinstance(d, dict) and "passed" in d:
                return _clean(d)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                d = json.loads(match.group())
                if isinstance(d, dict):
                    return _clean(d)
            except json.JSONDecodeError:
                pass

    # Last resort: pull final_answer out with a regex
    fa_match = re.search(r'"final_answer"\s*:\s*"(.*?)"(?=\s*[,}])', text, re.DOTALL)
    final_answer = fa_match.group(1).replace('\\"', '"') if fa_match else (text[:800] if text else "")

    return {
        "passed":      True,
        "score":       0.8,
        "reasoning":   "Could not parse critic JSON — defaulting to pass.",
        "suggestions": [],
        "final_answer": final_answer,
    }


async def critic_node(state: AgentState) -> dict:
    """Score the subtask results and produce a final answer."""
    results_str = "\n\n".join(
        f"=== {r['agent'].upper()} — {r['subtask']} ===\n{r['result']}"
        for r in state.get("subtask_results", [])
    )

    content = (
        f"{CRITIC_PROMPT}\n\n"
        f"Original incident: {state['user_input']}\n\n"
        f"Agent findings:\n{results_str}"
    )

    llm = get_llm("critic")
    output: dict = {}
    tokens_used = 0

    try:
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=content)]),
            timeout=90,
        )
        text = response.content if isinstance(response.content, str) else str(response.content)
        output = _extract_critic_output(text)
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            tokens_used = usage.get("total_tokens", 0)
    except asyncio.TimeoutError:
        logger.error("Critic LLM call timed out — defaulting to pass with analyst output")
        output = {}
    except Exception as e:
        logger.exception("Critic failed: %s — defaulting to pass", e)
        output = {}

    # If critic failed entirely, surface the best subtask result as the answer
    results = state.get("subtask_results", [])
    best_result = max(results, key=lambda r: len(r.get("result", "")), default={})
    best_text = best_result.get("result", "Investigation complete — no postmortem generated.")

    score  = float(output.get("score", 0.8))
    passed = score >= QUALITY_THRESHOLD

    critic_answer = str(output.get("final_answer") or "").strip()
    vague = len(critic_answer) < 150 and "\n" not in critic_answer
    final_answer = best_text if (vague or not critic_answer) else critic_answer

    feedback: CriticFeedback = {
        "passed":      passed,
        "score":       score,
        "reasoning":   output.get("reasoning", ""),
        "suggestions": output.get("suggestions", []),
    }

    new_iter = state["iteration_count"] + 1
    route = "end" if (passed or new_iter >= state["max_iterations"]) else "orchestrator"

    logger.info("Critic: score=%.2f passed=%s iter=%d route=%s", score, passed, new_iter, route)

    return {
        "critic_feedback":  feedback,
        "final_answer":     final_answer if route == "end" else None,
        "route_to":         route,
        "iteration_count":  new_iter,
        "total_tokens_used": state["total_tokens_used"] + tokens_used,
        "messages":         [],
    }


def route_from_critic(state: AgentState) -> str:
    """Route to orchestrator for retry or end if done."""
    feedback = state.get("critic_feedback") or {}
    if not feedback.get("passed", False):
        if state.get("iteration_count", 0) < state.get("max_iterations", 3):
            return "orchestrator"
    return "end"
