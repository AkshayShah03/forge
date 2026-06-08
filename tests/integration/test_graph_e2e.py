"""End-to-end integration tests for the LangGraph agent system."""
from __future__ import annotations

import json
import os
import sys
import uuid
from contextlib import contextmanager
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver

from agent_system.graph.builder import build_graph
from agent_system.state.schema import initial_state

# FakeChatModel is defined in tests/conftest.py — import it from the package root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import FakeChatModel


PLAN_TEXT = json.dumps({
    "task_plan": [
        {"subtask": "gather facts", "agent": "researcher", "depends_on": [], "status": "pending"}
    ]
})

RESEARCH_TEXT = "Gathered facts: AI is advancing rapidly."

CRITIC_TEXT = json.dumps({
    "passed": True,
    "score": 0.85,
    "reasoning": "Adequate coverage.",
    "suggestions": [],
    "final_answer": "AI is advancing rapidly.",
})


@contextmanager
def _patch_llm(responses: list[str]):
    """Patch get_llm at every import site to return a FakeChatModel with preset responses."""
    fake = FakeChatModel(responses=responses)
    with (
        patch("agent_system.agents.orchestrator.get_llm", return_value=fake),
        patch("agent_system.agents.sub_agents.get_llm",   return_value=fake),
        patch("agent_system.agents.critic.get_llm",       return_value=fake),
    ):
        yield


class TestSimpleTaskRunToCompletion:
    async def test_simple_task_runs_to_completion(self):
        """A single-subtask run produces final_answer and a passing critic."""
        with _patch_llm([PLAN_TEXT, RESEARCH_TEXT, CRITIC_TEXT]):
            graph = build_graph(MemorySaver())
            tid = str(uuid.uuid4())
            state = initial_state(tid, "Tell me about AI", token_budget=50_000)
            result = await graph.ainvoke(state, config={"configurable": {"thread_id": tid}})

        assert result.get("final_answer") is not None
        assert result.get("critic_feedback", {}).get("passed") is True
        assert result.get("iteration_count") == 1


class TestCheckpointing:
    async def test_checkpointing_persists_state(self):
        """State persisted to MemorySaver can be read back via aget_state."""
        with _patch_llm([PLAN_TEXT, RESEARCH_TEXT, CRITIC_TEXT]):
            checkpointer = MemorySaver()
            graph = build_graph(checkpointer)
            tid = str(uuid.uuid4())
            state = initial_state(tid, "Tell me about AI")
            await graph.ainvoke(state, config={"configurable": {"thread_id": tid}})

            saved = await graph.aget_state({"configurable": {"thread_id": tid}})

        assert saved is not None
        assert saved.values.get("task_id") == tid


class TestIncidentWorkflow:
    """Integration tests for the 3-agent incident response workflow."""

    async def test_three_agent_incident_run(self):
        """Researcher + coder (parallel) → analyst (depends on both) → critic passes."""
        researcher_plan = json.dumps({
            "task_plan": [
                {"subtask": "Research known causes and patterns for this alert type", "agent": "researcher", "depends_on": [], "status": "pending"},
                {"subtask": "Parse logs and query git history to find the evidence trail", "agent": "coder", "depends_on": [], "status": "pending"},
                {"subtask": "Correlate findings, identify root cause, and draft postmortem", "agent": "analyst", "depends_on": [
                    "Research known causes and patterns for this alert type",
                    "Parse logs and query git history to find the evidence trail",
                ], "status": "pending"},
            ]
        })
        postmortem = (
            "## Incident Summary\nP95 latency spiked from 45ms to 385ms.\n\n"
            "## Timeline\n- 14:02 Deploy v2.4.1\n- 14:32 Latency spike detected\n\n"
            "## Root Cause (confidence: 92%)\nCommit a3f9c12 introduced missing select_related()\n\n"
            "## Evidence\n- 23 DB queries/req vs 2 before deploy\n\n"
            "## Action Items\n1. Revert commit a3f9c12\n\n"
            "## Prevention\nAdd DB query count alerting."
        )
        critic_response = json.dumps({
            "passed": True,
            "score": 0.88,
            "reasoning": "Specific root cause with commit hash and evidence.",
            "suggestions": [],
            "final_answer": postmortem,
        })

        # Responses consumed in order: orchestrator → researcher → coder → analyst → critic
        responses = [researcher_plan, "Common causes: N+1 queries, missing indexes.", "log stats: P95=385ms, db_queries=23", postmortem, critic_response]
        with _patch_llm(responses):
            graph = build_graph(MemorySaver())
            tid = str(uuid.uuid4())
            state = initial_state(tid, "P95 latency spike at 14:32. Logs at /tmp/demo/. Repo at /repo.")
            result = await graph.ainvoke(state, config={"configurable": {"thread_id": tid}})

        assert result["final_answer"] is not None
        assert result["critic_feedback"]["passed"] is True
        assert result["iteration_count"] == 1
        # All three agent subtasks should appear in results
        agents_used = {r["agent"] for r in result["subtask_results"]}
        assert "researcher" in agents_used
        assert "coder" in agents_used
        assert "analyst" in agents_used

    async def test_dependency_blocks_analyst_until_both_complete(self):
        """Analyst subtask stays blocked while researcher or coder result is missing."""
        from agent_system.agents.orchestrator import route_from_orchestrator
        from agent_system.state.schema import initial_state

        plan = [
            {"subtask": "Research known causes", "agent": "researcher", "depends_on": [], "status": "complete"},
            {"subtask": "Parse logs", "agent": "coder", "depends_on": [], "status": "pending"},
            {"subtask": "Correlate and postmortem", "agent": "analyst", "depends_on": ["Research known causes", "Parse logs"], "status": "pending"},
        ]
        s = initial_state(str(uuid.uuid4()), "incident")
        s.update({
            "task_plan": plan,
            "subtask_results": [
                {"subtask": "Research known causes", "agent": "researcher", "result": "N+1 queries", "tokens_used": 100}
            ],
            "total_tokens_used": 100,
            "token_budget": 50_000,
        })
        # Coder still pending — analyst must be blocked, coder should run next
        route = route_from_orchestrator(s)
        assert route == "coder"

    async def test_analyst_unblocked_after_both_complete(self):
        """Analyst runs once both researcher and coder have results."""
        from agent_system.agents.orchestrator import route_from_orchestrator
        from agent_system.state.schema import initial_state

        plan = [
            {"subtask": "Research known causes", "agent": "researcher", "depends_on": [], "status": "complete"},
            {"subtask": "Parse logs", "agent": "coder", "depends_on": [], "status": "complete"},
            {"subtask": "Correlate and postmortem", "agent": "analyst", "depends_on": ["Research known causes", "Parse logs"], "status": "pending"},
        ]
        s = initial_state(str(uuid.uuid4()), "incident")
        s.update({
            "task_plan": plan,
            "subtask_results": [
                {"subtask": "Research known causes", "agent": "researcher", "result": "N+1", "tokens_used": 50},
                {"subtask": "Parse logs", "agent": "coder", "result": "P95=385ms", "tokens_used": 80},
            ],
            "total_tokens_used": 130,
            "token_budget": 50_000,
        })
        route = route_from_orchestrator(s)
        assert route == "analyst"

    async def test_incident_retry_with_vague_postmortem(self):
        """Critic fails a vague postmortem; orchestrator replans and produces a better one."""
        vague_plan = json.dumps({
            "task_plan": [
                {"subtask": "Research alert", "agent": "researcher", "depends_on": [], "status": "pending"},
            ]
        })
        better_plan = json.dumps({
            "task_plan": [
                {"subtask": "Recheck logs carefully", "agent": "coder", "depends_on": [], "status": "pending"},
            ]
        })
        vague_postmortem = "The system experienced high load."
        specific_postmortem = (
            "## Root Cause (confidence: 89%)\nCommit abc123 — missing select_related() in checkout/views.py:L42"
        )
        fail_critic = json.dumps({
            "passed": False, "score": 0.3,
            "reasoning": "Root cause too vague, no commit hash or file cited.",
            "suggestions": ["Identify the specific commit and file responsible."],
            "final_answer": vague_postmortem,
        })
        pass_critic = json.dumps({
            "passed": True, "score": 0.88,
            "reasoning": "Specific commit hash and file cited with evidence.",
            "suggestions": [],
            "final_answer": specific_postmortem,
        })

        responses = [
            vague_plan,         # iteration 1 orchestrator
            "common causes",    # researcher
            fail_critic,        # critic fails
            better_plan,        # iteration 2 orchestrator
            "P95=385ms, db=23", # coder
            pass_critic,        # critic passes
        ]
        with _patch_llm(responses):
            graph = build_graph(MemorySaver())
            tid = str(uuid.uuid4())
            state = initial_state(tid, "P95 spike — investigate.", max_iterations=3)
            result = await graph.ainvoke(state, config={"configurable": {"thread_id": tid}})

        assert result["iteration_count"] == 2
        assert result["critic_feedback"]["passed"] is True
        assert "commit" in result["final_answer"].lower() or "89" in result["final_answer"]


class TestTokenBudget:
    async def test_token_budget_stops_graph(self):
        """Exhausting the token budget terminates the graph without running sub-agents."""
        call_count = {"n": 0}

        async def mock_orchestrator(state):
            """Mock orchestrator that immediately exhausts the token budget."""
            call_count["n"] += 1
            return {
                "task_plan": [
                    {"subtask": "huge task", "agent": "researcher", "depends_on": [], "status": "pending"}
                ],
                "messages": [],
                "total_tokens_used": state["token_budget"] + 1,
            }

        with patch("agent_system.graph.builder.orchestrator_node", mock_orchestrator):
            graph = build_graph(MemorySaver())
            tid = str(uuid.uuid4())
            state = initial_state(tid, "test", token_budget=100)
            result = await graph.ainvoke(state, config={"configurable": {"thread_id": tid}})

        assert result["total_tokens_used"] > result["token_budget"]
        assert call_count["n"] == 1
