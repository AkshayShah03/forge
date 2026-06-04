"""Unit tests for the Azure Service Bus worker."""
from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch


def _make_raw_message(body: dict, delivery_count: int = 1) -> MagicMock:
    """Create a mock Service Bus raw message."""
    raw = MagicMock()
    raw.__str__ = MagicMock(return_value=json.dumps(body))
    raw.delivery_count = delivery_count
    raw.session_id = body.get("user_id", "user-1")
    return raw


def _valid_body(task_id: str | None = None) -> dict:
    """Return a valid task message body dict."""
    return {
        "task_id":        task_id or str(uuid.uuid4()),
        "user_id":        "user-1",
        "user_input":     "test task",
        "token_budget":   10_000,
        "max_iterations": 2,
        "metadata":       {"env": "test"},
    }


class TestTaskMessageParsing:
    def test_task_message_parsing_valid_body(self):
        """TaskMessage correctly parses all fields from the raw message."""
        from worker.servicebus_worker import TaskMessage
        body = _valid_body("tid-123")
        raw = _make_raw_message(body)
        msg = TaskMessage(raw)
        assert msg.task_id == "tid-123"
        assert msg.user_id == "user-1"
        assert msg.input == "test task"
        assert msg.budget == 10_000
        assert msg.max_iter == 2
        assert msg.metadata == {"env": "test"}

    def test_task_message_default_budget(self):
        """Missing token_budget falls back to 50_000."""
        from worker.servicebus_worker import TaskMessage
        body = _valid_body()
        del body["token_budget"]
        raw = _make_raw_message(body)
        msg = TaskMessage(raw)
        assert msg.budget == 50_000

    def test_task_message_to_initial_state(self):
        """to_initial_state() returns an AgentState with matching fields."""
        from worker.servicebus_worker import TaskMessage
        body = _valid_body("state-tid")
        raw = _make_raw_message(body)
        msg = TaskMessage(raw)
        state = msg.to_initial_state()
        assert state["task_id"] == "state-tid"
        assert state["user_input"] == "test task"
        assert state["token_budget"] == 10_000
        assert state["metadata"]["user_id"] == "user-1"


class TestLockRenewal:
    async def test_lock_renewal_stops_on_event(self):
        """lock_renewal_loop exits promptly when stop_event is set."""
        from worker.servicebus_worker import lock_renewal_loop

        receiver = MagicMock()
        receiver.renew_message_lock = AsyncMock()
        message = MagicMock()
        stop_event = asyncio.Event()
        stop_event.set()

        with patch("worker.servicebus_worker.LOCK_RENEWAL_INTERVAL", 0):
            await asyncio.wait_for(
                lock_renewal_loop(receiver, message, stop_event),
                timeout=2.0,
            )

        receiver.renew_message_lock.assert_not_called()


class TestPublishResult:
    async def test_publish_result_uses_session_id(self):
        """publish_result sends a message with session_id = user_id."""
        from worker.servicebus_worker import TaskMessage, publish_result

        body = _valid_body("pub-tid")
        raw = _make_raw_message(body)
        task = TaskMessage(raw)

        mock_sender = AsyncMock()
        mock_sender.__aenter__ = AsyncMock(return_value=mock_sender)
        mock_sender.__aexit__ = AsyncMock(return_value=False)
        mock_sender.send_messages = AsyncMock()

        mock_client = MagicMock()
        mock_client.get_queue_sender = MagicMock(return_value=mock_sender)

        result_state = {
            "final_answer": "done",
            "critic_feedback": {"score": 0.9},
            "iteration_count": 1,
            "total_tokens_used": 100,
        }

        await publish_result(mock_client, task, result_state, success=True)

        mock_sender.send_messages.assert_called_once()
        sent_msg = mock_sender.send_messages.call_args[0][0]
        assert sent_msg.session_id == "user-1"


class TestDeadLetter:
    async def test_delivery_count_triggers_dead_letter(self):
        """A message at MAX_DELIVERY_COUNT is dead-lettered on failure."""
        from worker.servicebus_worker import MAX_DELIVERY_COUNT, TaskMessage, process_task

        body = _valid_body("dl-tid")
        raw = _make_raw_message(body, delivery_count=MAX_DELIVERY_COUNT)
        task = TaskMessage(raw)

        receiver = MagicMock()
        receiver.renew_message_lock = AsyncMock()
        receiver.complete_message = AsyncMock()
        receiver.dead_letter_message = AsyncMock()

        mock_factory = MagicMock()
        mock_factory.get_task_state = AsyncMock(return_value=None)
        mock_factory.run_task = AsyncMock(side_effect=RuntimeError("boom"))

        mock_client = MagicMock()

        await process_task(receiver, raw, task, mock_factory, mock_client)

        receiver.dead_letter_message.assert_called_once()
        args = receiver.dead_letter_message.call_args
        assert args[1].get("reason") == "MaxDeliveryCountExceeded"
