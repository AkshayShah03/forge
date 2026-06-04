"""Shared pytest fixtures and test helpers."""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class FakeChatModel(BaseChatModel):
    """Fake ChatModel that returns predefined responses and supports bind_tools."""

    responses: list[str]
    _idx: int = PrivateAttr(default=0)

    def _generate(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Return the next predefined response."""
        text = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of _generate."""
        return self._generate(messages, stop=stop, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return "fake-chat"

    def bind_tools(self, tools: list, **kwargs: Any) -> "FakeChatModel":
        """Ignore tools and return self — sufficient for ReAct testing."""
        return self
