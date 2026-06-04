"""Integration tests for the FastAPI application."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from api.azure_main import app


@pytest.fixture()
async def client():
    """HTTP test client with mocked AgentGraphFactory and Azure SDK."""
    mock_factory = AsyncMock()
    mock_factory.get_task_state = AsyncMock(return_value=None)
    mock_factory.run_task = AsyncMock(return_value={})
    mock_factory.list_task_history = AsyncMock(return_value=[])

    mock_sender = AsyncMock()
    mock_sender.send_messages = AsyncMock()

    mock_sb_client = AsyncMock()
    mock_sb_client.get_queue_sender = MagicMock(return_value=mock_sender)

    with (
        patch("api.azure_main.AgentGraphFactory", return_value=mock_factory),
        patch("api.azure_main.ServiceBusClient", return_value=mock_sb_client),
        patch("api.azure_main.DefaultAzureCredential", return_value=AsyncMock()),
        patch("api.azure_main.SERVICE_BUS_NAMESPACE", "test.servicebus.windows.net"),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            # Override app.state after lifespan startup (Section 5F)
            app.state.factory = mock_factory
            app.state.sb_client = mock_sb_client
            yield c, mock_factory


class TestHealthEndpoint:
    async def test_health_returns_ok(self, client):
        """GET /health returns 200 with status ok."""
        c, _ = client
        resp = await c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestSubmitTask:
    async def test_submit_task_returns_202(self, client):
        """POST /tasks with valid body returns 202 and a task_id."""
        c, _ = client
        resp = await c.post("/tasks", json={
            "user_input": "Hello world",
            "user_id": "user-1",
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "queued"

    async def test_submit_task_validates_empty_input(self, client):
        """POST /tasks with empty user_input returns 422."""
        c, _ = client
        resp = await c.post("/tasks", json={
            "user_input": "",
            "user_id": "user-1",
        })
        assert resp.status_code == 422


class TestGetTask:
    async def test_get_task_not_found(self, client):
        """GET /tasks/{id} for unknown task returns 404."""
        c, mock_factory = client
        mock_factory.get_task_state = AsyncMock(return_value=None)
        resp = await c.get(f"/tasks/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_get_task_running(self, client):
        """GET /tasks/{id} for an in-progress task returns status running."""
        c, mock_factory = client
        tid = str(uuid.uuid4())
        mock_factory.get_task_state = AsyncMock(return_value={
            "task_id": tid,
            "final_answer": None,
            "route_to": None,
            "critic_feedback": None,
            "iteration_count": 1,
            "total_tokens_used": 500,
        })
        resp = await c.get(f"/tasks/{tid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    async def test_get_task_complete(self, client):
        """GET /tasks/{id} for a finished task returns status complete and final_answer."""
        c, mock_factory = client
        tid = str(uuid.uuid4())
        mock_factory.get_task_state = AsyncMock(return_value={
            "task_id": tid,
            "final_answer": "The answer",
            "route_to": "end",
            "critic_feedback": {"score": 0.9},
            "iteration_count": 1,
            "total_tokens_used": 1000,
        })
        resp = await c.get(f"/tasks/{tid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "complete"
        assert data["final_answer"] == "The answer"


class TestGetHistory:
    async def test_get_history(self, client):
        """GET /tasks/{id}/history returns checkpoint list."""
        c, mock_factory = client
        tid = str(uuid.uuid4())
        mock_factory.list_task_history = AsyncMock(return_value=[
            {"step": 0, "node": "orchestrator", "iteration_count": 0, "total_tokens_used": 0},
            {"step": 1, "node": "researcher", "iteration_count": 0, "total_tokens_used": 100},
        ])
        resp = await c.get(f"/tasks/{tid}/history")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


class TestSSEStream:
    async def test_sse_stream_emits_complete_event(self, client):
        """GET /tasks/{id}/stream emits a complete SSE event when task is done."""
        c, mock_factory = client
        tid = str(uuid.uuid4())
        mock_factory.get_task_state = AsyncMock(return_value={
            "task_id": tid,
            "final_answer": "Done",
            "route_to": "end",
            "critic_feedback": {"score": 0.95},
            "iteration_count": 1,
            "total_tokens_used": 500,
        })
        async with c.stream("GET", f"/tasks/{tid}/stream") as resp:
            assert resp.status_code == 200
            body = ""
            async for chunk in resp.aiter_text():
                body += chunk
                if "complete" in body:
                    break
        assert "complete" in body
