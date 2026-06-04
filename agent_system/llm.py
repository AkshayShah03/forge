"""LLM factory — Groq (cloud) with Ollama fallback when no GROQ_API_KEY is set."""
from __future__ import annotations

import os

_GROQ_KEY    = os.getenv("GROQ_API_KEY", "")
_GROQ_MODEL  = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")

_OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL",   "qwen2.5:3b")

_PER_ROLE = {
    "orchestrator": "ORCHESTRATOR_MODEL",
    "researcher":   "SUB_AGENT_MODEL",
    "coder":        "CODER_MODEL",
    "analyst":      "SUB_AGENT_MODEL",
    "critic":       "CRITIC_MODEL",
}


def get_llm(role: str = "default"):
    """Return a Groq ChatModel if GROQ_API_KEY is set, otherwise ChatOllama."""
    if _GROQ_KEY:
        from langchain_groq import ChatGroq
        model = os.getenv(_PER_ROLE.get(role, "GROQ_MODEL"), _GROQ_MODEL)
        return ChatGroq(model=model, api_key=_GROQ_KEY)

    from langchain_ollama import ChatOllama
    model = os.getenv(_PER_ROLE.get(role, "OLLAMA_MODEL"), _OLLAMA_MODEL)
    return ChatOllama(model=model, base_url=_OLLAMA_URL)
