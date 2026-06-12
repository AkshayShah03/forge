"""Unit tests for JSON parsing helpers and role prompt selection."""
from __future__ import annotations

import json

from agent_system.agents.critic import _extract_critic_output
from agent_system.agents.orchestrator import _extract_task_plan
from agent_system.agents.sub_agents import ROLE_PROMPTS, SUB_AGENT_PROMPT


# ---------------------------------------------------------------------------
# _extract_task_plan
# ---------------------------------------------------------------------------

class TestExtractTaskPlan:
    def _valid_plan(self, agent: str = "researcher") -> list[dict]:
        return [{"subtask": "do work", "agent": agent, "depends_on": [], "status": "pending"}]

    def test_parses_plain_json(self):
        """Parses a clean JSON string with task_plan key."""
        text = json.dumps({"task_plan": self._valid_plan()})
        result = _extract_task_plan(text)
        assert len(result) == 1
        assert result[0]["agent"] == "researcher"

    def test_parses_markdown_fenced_json(self):
        """Strips ```json fences before parsing."""
        plan = self._valid_plan("coder")
        text = f"```json\n{json.dumps({'task_plan': plan})}\n```"
        result = _extract_task_plan(text)
        assert result[0]["agent"] == "coder"

    def test_parses_plain_code_fence(self):
        """Strips plain ``` fences."""
        plan = self._valid_plan("analyst")
        text = f"```\n{json.dumps({'task_plan': plan})}\n```"
        result = _extract_task_plan(text)
        assert result[0]["agent"] == "analyst"

    def test_parses_nested_json_in_prose(self):
        """Extracts JSON object embedded in surrounding prose."""
        plan = self._valid_plan()
        text = f"Sure, here is my plan:\n{json.dumps({'task_plan': plan})}\nLet me know if that works."
        result = _extract_task_plan(text)
        assert len(result) == 1

    def test_multi_subtask_plan_preserved(self):
        """All subtasks in a multi-task plan are returned."""
        plan = [
            {"subtask": "research", "agent": "researcher", "depends_on": [], "status": "pending"},
            {"subtask": "code", "agent": "coder", "depends_on": [], "status": "pending"},
            {"subtask": "analyse", "agent": "analyst", "depends_on": ["research", "code"], "status": "pending"},
        ]
        text = json.dumps({"task_plan": plan})
        result = _extract_task_plan(text)
        assert len(result) == 3
        assert result[2]["depends_on"] == ["research", "code"]

    def test_fallback_on_empty_string(self):
        """Empty string produces the default 3-subtask plan."""
        result = _extract_task_plan("")
        assert len(result) == 3
        assert result[0]["agent"] == "researcher"
        assert result[1]["agent"] == "coder"
        assert result[2]["agent"] == "analyst"

    def test_fallback_on_garbage_text(self):
        """Non-JSON text falls back to default plan."""
        result = _extract_task_plan("I cannot complete this task for ethical reasons.")
        assert len(result) >= 1
        assert "agent" in result[0]

    def test_preserves_depends_on_list(self):
        """depends_on is preserved as a list, not stringified."""
        plan = [{"subtask": "B", "agent": "coder", "depends_on": ["A"], "status": "pending"}]
        text = json.dumps({"task_plan": plan})
        result = _extract_task_plan(text)
        assert result[0]["depends_on"] == ["A"]

    def test_handles_extra_whitespace_in_json(self):
        """Pretty-printed JSON with extra whitespace parses correctly."""
        plan = self._valid_plan()
        text = json.dumps({"task_plan": plan}, indent=4)
        result = _extract_task_plan(text)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _extract_critic_output
# ---------------------------------------------------------------------------

class TestExtractCriticOutput:
    def _valid_critic(self, passed: bool = True, score: float = 0.9) -> dict:
        return {
            "passed": passed,
            "score": score,
            "reasoning": "looks good",
            "suggestions": [],
            "final_answer": "The root cause is X.",
        }

    def test_parses_plain_json(self):
        """Parses a clean critic JSON response."""
        text = json.dumps(self._valid_critic())
        result = _extract_critic_output(text)
        assert result["passed"] is True
        assert result["score"] == 0.9
        assert result["final_answer"] == "The root cause is X."

    def test_parses_markdown_fenced_json(self):
        """Strips ```json fences before parsing."""
        text = f"```json\n{json.dumps(self._valid_critic(passed=False, score=0.4))}\n```"
        result = _extract_critic_output(text)
        assert result["passed"] is False
        assert result["score"] == 0.4

    def test_parses_plain_code_fence(self):
        """Strips plain ``` fences."""
        text = f"```\n{json.dumps(self._valid_critic())}\n```"
        result = _extract_critic_output(text)
        assert "passed" in result

    def test_extracts_json_from_prose(self):
        """Finds JSON embedded in surrounding prose."""
        data = self._valid_critic()
        text = f"Here is my evaluation:\n{json.dumps(data)}\nEnd of review."
        result = _extract_critic_output(text)
        assert result["passed"] is True

    def test_fallback_on_unparseable_response(self):
        """Garbage text produces a default pass response instead of crashing."""
        result = _extract_critic_output("I cannot evaluate this properly.")
        assert isinstance(result, dict)
        assert "passed" in result
        assert "score" in result

    def test_passing_score_preserved(self):
        """Score of 0.95 is returned as float, not string."""
        text = json.dumps(self._valid_critic(score=0.95))
        result = _extract_critic_output(text)
        assert isinstance(result["score"], float)
        assert abs(result["score"] - 0.95) < 1e-9

    def test_suggestions_list_preserved(self):
        """Non-empty suggestions list is returned intact."""
        data = self._valid_critic()
        data["suggestions"] = ["add more detail", "cite log lines"]
        result = _extract_critic_output(json.dumps(data))
        assert result["suggestions"] == ["add more detail", "cite log lines"]

    def test_final_answer_with_newlines(self):
        """final_answer containing newlines (a multiline postmortem) is preserved."""
        data = self._valid_critic()
        data["final_answer"] = "## Incident Summary\nP95 latency spiked.\n\n## Root Cause\nN+1 query."
        result = _extract_critic_output(json.dumps(data))
        assert "## Incident Summary" in result["final_answer"]


# ---------------------------------------------------------------------------
# Role prompt selection
# ---------------------------------------------------------------------------

class TestRolePrompts:
    def test_researcher_prompt_defined(self):
        """ROLE_PROMPTS contains a non-empty researcher prompt."""
        assert "researcher" in ROLE_PROMPTS
        assert len(ROLE_PROMPTS["researcher"]) > 50

    def test_coder_prompt_defined(self):
        """ROLE_PROMPTS contains a non-empty coder prompt."""
        assert "coder" in ROLE_PROMPTS
        assert len(ROLE_PROMPTS["coder"]) > 50

    def test_analyst_prompt_defined(self):
        """ROLE_PROMPTS contains a non-empty analyst prompt."""
        assert "analyst" in ROLE_PROMPTS
        assert len(ROLE_PROMPTS["analyst"]) > 50

    def test_unknown_role_falls_back_to_sub_agent_prompt(self):
        """An unknown role key falls back to SUB_AGENT_PROMPT."""
        result = ROLE_PROMPTS.get("unknown_role", SUB_AGENT_PROMPT)
        assert result is SUB_AGENT_PROMPT

    def test_researcher_prompt_mentions_incident_keywords(self):
        """Researcher prompt is incident-investigation scoped."""
        prompt = ROLE_PROMPTS["researcher"].lower()
        assert any(kw in prompt for kw in ["incident", "alert", "root cause", "runbook"])

    def test_coder_prompt_mentions_log_tools(self):
        """Coder prompt references log-reading tools."""
        prompt = ROLE_PROMPTS["coder"].lower()
        assert "list_log_files" in prompt or "read_log_file" in prompt

    def test_analyst_prompt_mentions_postmortem(self):
        """Analyst prompt instructs the agent to produce a postmortem."""
        prompt = ROLE_PROMPTS["analyst"].lower()
        assert "postmortem" in prompt

    def test_analyst_prompt_includes_all_sections(self):
        """Analyst prompt specifies all required postmortem sections."""
        prompt = ROLE_PROMPTS["analyst"]
        for section in ["Incident Summary", "Timeline", "Root Cause", "Evidence", "Action Items", "Prevention"]:
            assert section in prompt, f"Missing postmortem section: {section}"
