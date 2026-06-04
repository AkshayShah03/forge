"""End-to-end smoke test using MemorySaver and a FakeChatModel (no API key needed)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import patch

# Ensure project root is on the path when run directly
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tests.conftest import FakeChatModel  # noqa: E402  (after sys.path insert)


ORCHESTRATOR_RESPONSE = json.dumps({
    "task_plan": [
        {
            "subtask": "Research the topic",
            "agent": "researcher",
            "depends_on": [],
            "status": "pending",
        }
    ]
})

RESEARCHER_RESPONSE = "Research complete. The topic involves important findings about artificial intelligence."

CRITIC_RESPONSE = json.dumps({
    "passed": True,
    "score": 0.9,
    "reasoning": "The research covers the topic adequately.",
    "suggestions": [],
    "final_answer": "Artificial intelligence is a broad field covering machine learning and more.",
})


async def run_smoke_test() -> None:
    """Run a full graph with FakeChatModel and assert expected state."""
    import uuid
    from langgraph.checkpoint.memory import MemorySaver

    from agent_system.graph.builder import build_graph
    from agent_system.state.schema import initial_state

    responses = [ORCHESTRATOR_RESPONSE, RESEARCHER_RESPONSE, CRITIC_RESPONSE]
    fake = FakeChatModel(responses=responses)

    with (
        patch("agent_system.agents.orchestrator.get_llm", return_value=fake),
        patch("agent_system.agents.sub_agents.get_llm",   return_value=fake),
        patch("agent_system.agents.critic.get_llm",       return_value=fake),
    ):
        checkpointer = MemorySaver()
        graph = build_graph(checkpointer)

        task_id = str(uuid.uuid4())
        state = initial_state(
            task_id=task_id,
            user_input="Summarise recent advances in AI",
            token_budget=50_000,
            max_iterations=3,
        )

        result = await graph.ainvoke(state, config={"configurable": {"thread_id": task_id}})

    assert result.get("final_answer") is not None, "final_answer should not be None"
    assert result.get("iteration_count") == 1, f"Expected 1 iteration, got {result.get('iteration_count')}"
    assert len(result.get("subtask_results", [])) >= 1, "Should have at least 1 subtask result"
    feedback = result.get("critic_feedback") or {}
    assert feedback.get("passed") is True, f"Critic should pass, got {feedback}"

    print("PASS")


if __name__ == "__main__":
    try:
        asyncio.run(run_smoke_test())
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
