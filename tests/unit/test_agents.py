"""Unit tests for agent routing logic and state schema."""
from __future__ import annotations

import uuid

from agent_system.agents.critic import route_from_critic
from agent_system.agents.orchestrator import route_from_orchestrator
from agent_system.agents.sub_agents import route_after_subagent
from agent_system.graph.builder import build_graph
from agent_system.state.schema import AgentState, initial_state
from agent_system.tools.registry import AGENT_TOOL_SCOPES


def _state(**kwargs) -> AgentState:
    """Build a minimal AgentState for routing tests."""
    base = initial_state(task_id=str(uuid.uuid4()), user_input="test")
    base.update(kwargs)
    return base


class TestAgentState:
    def test_initial_state_defaults(self):
        """initial_state fills in all required fields with sensible defaults."""
        s = initial_state("tid", "hello")
        assert s["task_id"] == "tid"
        assert s["user_input"] == "hello"
        assert s["iteration_count"] == 0
        assert s["subtask_results"] == []
        assert s["task_plan"] == []
        assert s["final_answer"] is None

    def test_initial_state_custom_budget(self):
        """Token budget and max_iterations are configurable."""
        s = initial_state("tid", "hello", token_budget=10_000, max_iterations=5)
        assert s["token_budget"] == 10_000
        assert s["max_iterations"] == 5


class TestOrchestratorRouting:
    def test_routes_to_researcher_when_researcher_subtask_pending(self):
        """Pending researcher subtask routes to researcher."""
        s = _state(
            task_plan=[{"subtask": "do research", "agent": "researcher", "depends_on": [], "status": "pending"}],
            subtask_results=[],
            total_tokens_used=0,
            token_budget=50_000,
        )
        assert route_from_orchestrator(s) == "researcher"

    def test_routes_to_coder(self):
        """Pending coder subtask routes to coder."""
        s = _state(
            task_plan=[{"subtask": "write code", "agent": "coder", "depends_on": [], "status": "pending"}],
            subtask_results=[],
            total_tokens_used=0,
            token_budget=50_000,
        )
        assert route_from_orchestrator(s) == "coder"

    def test_routes_to_analyst(self):
        """Pending analyst subtask routes to analyst."""
        s = _state(
            task_plan=[{"subtask": "analyse data", "agent": "analyst", "depends_on": [], "status": "pending"}],
            subtask_results=[],
            total_tokens_used=0,
            token_budget=50_000,
        )
        assert route_from_orchestrator(s) == "analyst"

    def test_routes_to_critic_when_all_complete(self):
        """All subtasks complete routes to critic."""
        s = _state(
            task_plan=[{"subtask": "do research", "agent": "researcher", "depends_on": [], "status": "complete"}],
            subtask_results=[{"subtask": "do research", "agent": "researcher", "result": "done", "tokens_used": 10}],
            total_tokens_used=10,
            token_budget=50_000,
        )
        assert route_from_orchestrator(s) == "critic"

    def test_routes_to_end_on_budget_exhaustion(self):
        """Exhausted token budget routes to end."""
        s = _state(
            task_plan=[{"subtask": "do research", "agent": "researcher", "depends_on": [], "status": "pending"}],
            subtask_results=[],
            total_tokens_used=50_001,
            token_budget=50_000,
        )
        assert route_from_orchestrator(s) == "end"

    def test_blocks_subtask_with_unmet_dependency(self):
        """A subtask whose dependency is not yet in subtask_results is skipped."""
        s = _state(
            task_plan=[
                {"subtask": "step A", "agent": "researcher", "depends_on": [], "status": "pending"},
                {"subtask": "step B", "agent": "coder", "depends_on": ["step A"], "status": "pending"},
            ],
            subtask_results=[],
            total_tokens_used=0,
            token_budget=50_000,
        )
        assert route_from_orchestrator(s) == "researcher"

    def test_unblocks_subtask_when_dependency_met(self):
        """Completed dependency allows the blocked subtask to run."""
        s = _state(
            task_plan=[
                {"subtask": "step A", "agent": "researcher", "depends_on": [], "status": "complete"},
                {"subtask": "step B", "agent": "coder", "depends_on": ["step A"], "status": "pending"},
            ],
            subtask_results=[{"subtask": "step A", "agent": "researcher", "result": "done", "tokens_used": 5}],
            total_tokens_used=5,
            token_budget=50_000,
        )
        assert route_from_orchestrator(s) == "coder"


class TestSubAgentRouting:
    def test_routes_to_next_subtask_after_first_completes(self):
        """After researcher finishes, routes to next unblocked coder subtask."""
        s = _state(
            task_plan=[
                {"subtask": "A", "agent": "researcher", "depends_on": [], "status": "complete"},
                {"subtask": "B", "agent": "coder", "depends_on": [], "status": "pending"},
            ],
            subtask_results=[{"subtask": "A", "agent": "researcher", "result": "ok", "tokens_used": 5}],
            total_tokens_used=5,
            token_budget=50_000,
        )
        assert route_after_subagent(s) == "coder"

    def test_routes_to_critic_when_all_done(self):
        """All subtasks complete routes to critic."""
        s = _state(
            task_plan=[{"subtask": "A", "agent": "researcher", "depends_on": [], "status": "complete"}],
            subtask_results=[{"subtask": "A", "agent": "researcher", "result": "ok", "tokens_used": 5}],
            total_tokens_used=5,
            token_budget=50_000,
        )
        assert route_after_subagent(s) == "critic"

    def test_routes_to_end_on_budget(self):
        """Budget exhaustion routes to end."""
        s = _state(
            task_plan=[{"subtask": "A", "agent": "researcher", "depends_on": [], "status": "pending"}],
            subtask_results=[],
            total_tokens_used=100_000,
            token_budget=50_000,
        )
        assert route_after_subagent(s) == "end"


class TestCriticRouting:
    def test_routes_to_end_when_passed(self):
        """Passing critic score routes to end."""
        s = _state(
            critic_feedback={"passed": True, "score": 0.9, "reasoning": "good", "suggestions": []},
            iteration_count=1,
            max_iterations=3,
        )
        assert route_from_critic(s) == "end"

    def test_routes_to_orchestrator_when_failed_and_iterations_remain(self):
        """Failing critic with remaining iterations routes to orchestrator."""
        s = _state(
            critic_feedback={"passed": False, "score": 0.5, "reasoning": "needs work", "suggestions": []},
            iteration_count=1,
            max_iterations=3,
        )
        assert route_from_critic(s) == "orchestrator"

    def test_routes_to_end_when_max_iterations_reached(self):
        """Max iterations reached forces end even with failing score."""
        s = _state(
            critic_feedback={"passed": False, "score": 0.5, "reasoning": "still failing", "suggestions": []},
            iteration_count=3,
            max_iterations=3,
        )
        assert route_from_critic(s) == "end"


class TestToolRegistry:
    def test_orchestrator_gets_web_search_only(self):
        """Orchestrator scope contains only web_search."""
        assert AGENT_TOOL_SCOPES["orchestrator"] == ["web_search"]

    def test_researcher_scope(self):
        """Researcher scope contains web_search, rag_retrieval, read_file."""
        assert set(AGENT_TOOL_SCOPES["researcher"]) == {"web_search", "rag_retrieval", "read_file"}

    def test_coder_scope(self):
        """Coder scope contains execute_python, read_file, write_file."""
        assert set(AGENT_TOOL_SCOPES["coder"]) == {"execute_python", "read_file", "write_file"}

    def test_critic_has_no_tools(self):
        """Critic has an empty tool scope."""
        assert AGENT_TOOL_SCOPES["critic"] == []


class TestGraphTopology:
    def test_graph_compiles(self):
        """Graph compiles without error using MemorySaver."""
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(MemorySaver())
        assert graph is not None

    def test_graph_has_required_nodes(self):
        """All five agent nodes are present."""
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(MemorySaver())
        nodes = set(graph.nodes.keys())
        assert {"orchestrator", "researcher", "coder", "analyst", "critic"}.issubset(nodes)
