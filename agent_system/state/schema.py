"""State schema for the multi-agent orchestration system."""
from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class SubtaskResult(TypedDict):
    """Result of a completed subtask."""

    subtask: str
    agent: str
    result: str
    tokens_used: int


class CriticFeedback(TypedDict):
    """Feedback from the critic agent."""

    passed: bool
    score: float
    reasoning: str
    suggestions: list[str]


class AgentState(TypedDict):
    """Central state shared by all nodes in the agent graph."""

    task_id: str
    user_input: str
    task_plan: list[dict]
    final_answer: Optional[str]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    subtask_results: Annotated[list[SubtaskResult], operator.add]
    critic_feedback: Optional[CriticFeedback]
    iteration_count: int
    max_iterations: int
    next_agent: Optional[str]
    route_to: Optional[str]
    total_tokens_used: int
    token_budget: int
    trace_id: str
    metadata: dict[str, Any]


def initial_state(
    task_id: str,
    user_input: str,
    token_budget: int = 50_000,
    max_iterations: int = 3,
    metadata: Optional[dict] = None,
) -> AgentState:
    """Create a fresh initial state for a new task."""
    return AgentState(
        task_id=task_id,
        user_input=user_input,
        task_plan=[],
        final_answer=None,
        messages=[],
        subtask_results=[],
        critic_feedback=None,
        iteration_count=0,
        max_iterations=max_iterations,
        next_agent=None,
        route_to=None,
        total_tokens_used=0,
        token_budget=token_budget,
        trace_id=str(uuid.uuid4()),
        metadata=metadata or {},
    )
