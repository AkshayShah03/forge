"""LangGraph graph assembly and factory."""
from __future__ import annotations

import logging
import os
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent_system.agents.critic import critic_node, route_from_critic
from agent_system.agents.orchestrator import orchestrator_node, route_from_orchestrator
from agent_system.agents.sub_agents import (
    analyst_node,
    coder_node,
    researcher_node,
    route_after_subagent,
)
from agent_system.state.schema import AgentState

logger = logging.getLogger(__name__)

_SUBAGENT_ROUTES = {
    "researcher": "researcher",
    "coder":      "coder",
    "analyst":    "analyst",
    "critic":     "critic",
    "end":        END,
}


def build_graph(checkpointer=None):
    """Compile the full multi-agent StateGraph with the given checkpointer."""
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("researcher",   researcher_node)
    graph.add_node("coder",        coder_node)
    graph.add_node("analyst",      analyst_node)
    graph.add_node("critic",       critic_node)

    graph.add_edge(START, "orchestrator")

    graph.add_conditional_edges("orchestrator", route_from_orchestrator, _SUBAGENT_ROUTES)
    graph.add_conditional_edges("researcher",   route_after_subagent,    _SUBAGENT_ROUTES)
    graph.add_conditional_edges("coder",        route_after_subagent,    _SUBAGENT_ROUTES)
    graph.add_conditional_edges("analyst",      route_after_subagent,    _SUBAGENT_ROUTES)

    graph.add_conditional_edges(
        "critic",
        route_from_critic,
        {"orchestrator": "orchestrator", "end": END},
    )

    return graph.compile(checkpointer=checkpointer)


class AgentGraphFactory:
    """Async context manager that owns the compiled graph and its checkpointer."""

    def __init__(self):
        """Initialise without connecting — call __aenter__ to connect."""
        self._checkpointer = None
        self._graph = None

    async def __aenter__(self) -> "AgentGraphFactory":
        """Set up checkpointer and compile graph."""
        postgres_url = os.getenv("POSTGRES_URL", "")
        if postgres_url:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                conn_str = postgres_url.replace("postgresql://", "postgresql+psycopg://", 1)
                self._checkpointer = await AsyncPostgresSaver.afrom_conn_string(conn_str)
                await self._checkpointer.setup()
                logger.info("Using AsyncPostgresSaver for checkpointing")
            except Exception as e:
                logger.warning("PostgreSQL checkpointer failed (%s) — using MemorySaver", e)
                self._checkpointer = MemorySaver()
        else:
            logger.info("No POSTGRES_URL — using MemorySaver")
            self._checkpointer = MemorySaver()

        self._graph = build_graph(self._checkpointer)
        return self

    async def __aexit__(self, *args) -> None:
        """Close the checkpointer connection if applicable."""
        if hasattr(self._checkpointer, "conn") and self._checkpointer.conn:
            try:
                await self._checkpointer.conn.close()
            except Exception:
                pass

    async def run_task(self, state: AgentState) -> AgentState:
        """Run a new task from initial state to completion."""
        config = {"configurable": {"thread_id": state["task_id"]}}
        result = await self._graph.ainvoke(state, config=config)
        return result

    async def resume_task(self, task_id: str) -> AgentState:
        """Resume a previously checkpointed task."""
        config = {"configurable": {"thread_id": task_id}}
        result = await self._graph.ainvoke(None, config=config)
        return result

    async def get_task_state(self, task_id: str) -> Optional[AgentState]:
        """Read the latest checkpoint for a task."""
        config = {"configurable": {"thread_id": task_id}}
        checkpoint = await self._graph.aget_state(config)
        return checkpoint.values if checkpoint and checkpoint.values else None

    async def list_task_history(self, task_id: str) -> list[dict]:
        """Return all checkpoint snapshots for a task."""
        config = {"configurable": {"thread_id": task_id}}
        history = []
        async for snapshot in self._graph.aget_state_history(config):
            history.append({
                "step":              snapshot.metadata.get("step", 0),
                "node":              snapshot.metadata.get("source", ""),
                "iteration_count":   snapshot.values.get("iteration_count", 0),
                "total_tokens_used": snapshot.values.get("total_tokens_used", 0),
            })
        return history
