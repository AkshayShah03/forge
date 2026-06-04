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
