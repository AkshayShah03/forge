"""Unit tests for graph structure and routing correctness."""
from __future__ import annotations

import uuid
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver

from agent_system.agents.orchestrator import route_from_orchestrator
from agent_system.graph.builder import build_graph
from agent_system.state.schema import AgentState, initial_state


def _tid() -> str:
    """Generate a random task ID."""
    return str(uuid.uuid4())


class TestGraphCompilation:
    def test_graph_compiles_with_memory_saver(self):
        """Graph compiles without raising."""
        graph = build_graph(MemorySaver())
        assert graph is not None

    def test_graph_node_names(self):
        """All five agent nodes are registered."""
        graph = build_graph(MemorySaver())
        expected = {"orchestrator", "researcher", "coder", "analyst", "critic"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_graph_edges_orchestrator_to_all_agents(self):
        """Orchestrator router returns the correct agent name for each subtask type."""
        def _s(agent: str) -> AgentState:
            s = initial_state(_tid(), "t")
            s.update({
                "task_plan": [{"subtask": "x", "agent": agent, "depends_on": [], "status": "pending"}],
                "subtask_results": [],
                "total_tokens_used": 0,
                "token_budget": 50_000,
            })
            return s

        assert route_from_orchestrator(_s("researcher")) == "researcher"
        assert route_from_orchestrator(_s("coder")) == "coder"
        assert route_from_orchestrator(_s("analyst")) == "analyst"


class TestOrchestratorBudget:
    def test_orchestrator_routes_to_budget_end_when_exhausted(self):
        """Token budget exhaustion routes to 'end'."""
        s = initial_state(_tid(), "t")
        s.update({
            "task_plan": [{"subtask": "x", "agent": "researcher", "depends_on": [], "status": "pending"}],
            "subtask_results": [],
            "total_tokens_used": 999_999,
            "token_budget": 50_000,
        })
        assert route_from_orchestrator(s) == "end"


class TestMockGraphRuns:
    async def test_full_mock_run_single_subtask(self):
        """Full graph run with mocked nodes completes in one iteration."""
        task_plan = [{"subtask": "find info", "agent": "researcher", "depends_on": [], "status": "pending"}]

        async def mock_orchestrator(state):
            """Mock orchestrator returns a single researcher subtask."""
            return {
                "task_plan": task_plan,
                "messages": [],
                "total_tokens_used": state["total_tokens_used"] + 100,
            }

        async def mock_researcher(state):
            """Mock researcher completes its subtask."""
            return {
                "task_plan": [{**task_plan[0], "status": "complete"}],
                "subtask_results": [{"subtask": "find info", "agent": "researcher", "result": "result", "tokens_used": 50}],
                "messages": [],
                "total_tokens_used": state["total_tokens_used"] + 50,
            }

        async def mock_critic(state):
            """Mock critic passes the result."""
            return {
                "critic_feedback": {"passed": True, "score": 0.9, "reasoning": "good", "suggestions": []},
                "final_answer": "The answer is 42.",
                "route_to": "end",
                "iteration_count": 1,
                "total_tokens_used": state["total_tokens_used"] + 30,
                "messages": [],
            }

        with (
            patch("agent_system.graph.builder.orchestrator_node", mock_orchestrator),
            patch("agent_system.graph.builder.researcher_node", mock_researcher),
            patch("agent_system.graph.builder.critic_node", mock_critic),
        ):
            graph = build_graph(MemorySaver())
            tid = _tid()
            state = initial_state(tid, "test")
            result = await graph.ainvoke(state, config={"configurable": {"thread_id": tid}})

        assert result["final_answer"] == "The answer is 42."
        assert result["iteration_count"] == 1

    async def test_retry_loop_increments_iteration(self):
        """Critic failure causes retry; iteration_count increments."""
        task_plan = [{"subtask": "find info", "agent": "researcher", "depends_on": [], "status": "pending"}]
        call_count = {"critic": 0}

        async def mock_orchestrator(state):
            """Mock orchestrator returns a researcher subtask."""
            return {
                "task_plan": task_plan,
                "messages": [],
                "total_tokens_used": state["total_tokens_used"] + 10,
            }

        async def mock_researcher(state):
            """Mock researcher completes its subtask."""
            return {
                "task_plan": [{**task_plan[0], "status": "complete"}],
                "subtask_results": [{"subtask": "find info", "agent": "researcher", "result": "ok", "tokens_used": 5}],
                "messages": [],
                "total_tokens_used": state["total_tokens_used"] + 5,
            }

        async def mock_critic(state):
            """Mock critic fails on first call, passes on second."""
            call_count["critic"] += 1
            passed = call_count["critic"] >= 2
            return {
                "critic_feedback": {"passed": passed, "score": 0.9 if passed else 0.5, "reasoning": "ok", "suggestions": []},
                "final_answer": "Done" if passed else None,
                "route_to": "end" if passed else "orchestrator",
                "iteration_count": state["iteration_count"] + 1,
                "total_tokens_used": state["total_tokens_used"] + 10,
                "messages": [],
            }

        with (
            patch("agent_system.graph.builder.orchestrator_node", mock_orchestrator),
            patch("agent_system.graph.builder.researcher_node", mock_researcher),
            patch("agent_system.graph.builder.critic_node", mock_critic),
        ):
            graph = build_graph(MemorySaver())
            tid = _tid()
            state = initial_state(tid, "test", max_iterations=3)
            result = await graph.ainvoke(state, config={"configurable": {"thread_id": tid}})

        assert result["iteration_count"] == 2
        assert call_count["critic"] == 2

    async def test_max_iterations_enforced(self):
        """Graph terminates after max_iterations even if critic never passes."""
        task_plan = [{"subtask": "find info", "agent": "researcher", "depends_on": [], "status": "pending"}]

        async def mock_orchestrator(state):
            """Mock orchestrator always returns a fresh researcher subtask."""
            return {
                "task_plan": task_plan,
                "messages": [],
                "total_tokens_used": state["total_tokens_used"] + 10,
            }

        async def mock_researcher(state):
            """Mock researcher always returns partial results."""
            return {
                "task_plan": [{**task_plan[0], "status": "complete"}],
                "subtask_results": [{"subtask": "find info", "agent": "researcher", "result": "partial", "tokens_used": 5}],
                "messages": [],
                "total_tokens_used": state["total_tokens_used"] + 5,
            }

        async def mock_critic(state):
            """Mock critic always fails."""
            new_iter = state["iteration_count"] + 1
            return {
                "critic_feedback": {"passed": False, "score": 0.3, "reasoning": "poor", "suggestions": []},
                "final_answer": None,
                "route_to": "end" if new_iter >= state["max_iterations"] else "orchestrator",
                "iteration_count": new_iter,
                "total_tokens_used": state["total_tokens_used"] + 10,
                "messages": [],
            }

        with (
            patch("agent_system.graph.builder.orchestrator_node", mock_orchestrator),
            patch("agent_system.graph.builder.researcher_node", mock_researcher),
            patch("agent_system.graph.builder.critic_node", mock_critic),
        ):
            graph = build_graph(MemorySaver())
            tid = _tid()
            state = initial_state(tid, "test", max_iterations=2)
            result = await graph.ainvoke(state, config={"configurable": {"thread_id": tid}})

        assert result["iteration_count"] == 2
