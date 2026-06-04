"""
Azure Service Bus Worker

Drop-in replacement for worker/sqs_worker.py.
The LangGraph graph, agents, tools, and state schema are completely unchanged.
Only this file and api/main.py change — everything else is identical.

AWS SQS  →  Azure Service Bus mapping:
  SQS FIFO queue               →  Service Bus queue (sessions enabled)
  MessageGroupId = user_id     →  SessionId = user_id   (FIFO per user)
  MessageDeduplicationId       →  MessageId             (exactly-once)
  Visibility timeout           →  Lock duration         (renewable lease)
  extend_visibility (heartbeat)→  renew_message_lock()  (same pattern)
  Dead-letter queue            →  $DeadLetterQueue      (automatic, built-in)
  MaxReceiveCount → DLQ        →  maxDeliveryCount      (set on queue)
  Long-polling (WaitTime=20s)  →  receive_messages(max_wait_time=20)

WHY Service Bus sessions (not standard queues):
  Sessions are Azure's answer to SQS FIFO MessageGroupId.
  When sessions are enabled, each SessionId gets its own ordered stream.
  user_id as SessionId means all tasks from one user are processed in order,
  exactly as SQS FIFO guaranteed with MessageGroupId = user_id.

WHY Service Bus Premium (not Standard):
  Sessions require Premium tier. It also gives you:
    - VNet integration (same as SQS VPC endpoints)
    - 1 GB message size (vs 256 KB Standard)
    - Dedicated capacity (no noisy neighbours)
  Cost: ~$677/month for a single messaging unit — comparable to SQS at scale.

Lock renewal (heartbeat):
  Service Bus message locks expire after the configured lock duration (default 60s).
  For 10-minute agent tasks, we must renew the lock every 50s.
  This is identical to SQS's change_message_visibility heartbeat.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from typing import Optional

from azure.identity.aio import DefaultAzureCredential
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage, ServiceBusReceiveMode
from azure.servicebus.exceptions import ServiceBusError, MessageLockLostError

from agent_system.graph.builder import AgentGraphFactory
from agent_system.state.schema import AgentState, initial_state

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# Service Bus configuration — from env vars (populated by Key Vault references in Container Apps)
SERVICE_BUS_NAMESPACE  = os.getenv("AZURE_SERVICE_BUS_NAMESPACE", "")   # e.g. myagents.servicebus.windows.net
TASK_QUEUE_NAME        = os.getenv("AZURE_TASK_QUEUE_NAME",   "task-inbox")
RESULT_QUEUE_NAME      = os.getenv("AZURE_RESULT_QUEUE_NAME", "result-outbox")

LOCK_RENEWAL_INTERVAL  = int(os.getenv("SB_LOCK_RENEWAL_INTERVAL", "50"))  # renew lock every 50s
MAX_DELIVERY_COUNT     = int(os.getenv("SB_MAX_DELIVERY_COUNT",     "3"))   # after 3 → DLQ
MAX_WAIT_SECONDS       = int(os.getenv("SB_MAX_WAIT_SECONDS",       "20"))  # long-poll timeout


# ---------------------------------------------------------------------------
# Message schema — identical structure to SQS worker
# ---------------------------------------------------------------------------

class TaskMessage:
    """A task message received from Azure Service Bus."""

    def __init__(self, raw):
        # Service Bus message body is bytes
        body = json.loads(str(raw))
        self.raw          = raw                        # keep for lock renewal / complete / dead-letter
        self.task_id      = body["task_id"]
        self.user_id      = body["user_id"]            # also the session_id
        self.input        = body["user_input"]
        self.metadata     = body.get("metadata", {})
        self.budget       = body.get("token_budget", 50_000)
        self.max_iter     = body.get("max_iterations", 3)
        # Service Bus: delivery count tells us how many times this was attempted
        self.delivery_count = raw.delivery_count

    def to_initial_state(self) -> AgentState:
        return initial_state(
            task_id=self.task_id,
            user_input=self.input,
            token_budget=self.budget,
            max_iterations=self.max_iter,
            metadata={"user_id": self.user_id, **self.metadata},
        )


# ---------------------------------------------------------------------------
# Lock renewal heartbeat — equivalent to SQS change_message_visibility
# ---------------------------------------------------------------------------

async def lock_renewal_loop(
    receiver,
    message,
    stop_event: asyncio.Event,
) -> None:
    """
    Renew the Service Bus message lock every LOCK_RENEWAL_INTERVAL seconds.

    Without this, the lock expires after the configured duration (default 60s)
    and the message becomes available to other consumers — causing duplicate processing.
    This is the Azure equivalent of extending SQS visibility timeout.
    """
    while not stop_event.is_set():
        try:
            await asyncio.sleep(LOCK_RENEWAL_INTERVAL)
            if stop_event.is_set():
                break
            await receiver.renew_message_lock(message)
            logger.debug("Lock renewed for message session=%s", message.session_id)
        except MessageLockLostError:
            logger.warning("Lock lost — message may be redelivered to another worker")
            break
        except Exception as e:
            logger.warning("Lock renewal error: %s", e)


# ---------------------------------------------------------------------------
# Result publishing
# ---------------------------------------------------------------------------

async def publish_result(
    client: ServiceBusClient,
    task: TaskMessage,
    result: AgentState,
    success: bool,
) -> None:
    """Send result to the result queue. SessionId = user_id for ordered delivery."""
    critic = result.get("critic_feedback") or {}
    body = {
        "task_id":      task.task_id,
        "user_id":      task.user_id,
        "success":      success,
        "final_answer": result.get("final_answer", ""),
        "score":        critic.get("score", 0.0),
        "iterations":   result.get("iteration_count", 0),
        "tokens_used":  result.get("total_tokens_used", 0),
        "completed_at": time.time(),
    }
    async with client.get_queue_sender(RESULT_QUEUE_NAME) as sender:
        msg = ServiceBusMessage(
            body=json.dumps(body),
            message_id=f"{task.task_id}-result",   # deduplication
            session_id=task.user_id,               # FIFO per user in result queue too
        )
        await sender.send_messages(msg)
    logger.info("Result published task_id=%s score=%.2f", task.task_id, body["score"])


# ---------------------------------------------------------------------------
# Core task processor
# ---------------------------------------------------------------------------

async def process_task(
    receiver,
    message,
    task: TaskMessage,
    factory: AgentGraphFactory,
    client: ServiceBusClient,
) -> None:
    """
    Process one task end-to-end with lock renewal and crash recovery.
    Identical semantics to the SQS version — only the SDK calls differ.
    """
    stop_event = asyncio.Event()
    renewal_task = asyncio.create_task(
        lock_renewal_loop(receiver, message, stop_event)
    )

    try:
        # Crash recovery: check if task already has a checkpoint
        existing = await factory.get_task_state(task.task_id)
        if existing and existing.get("final_answer"):
            logger.info("Task %s already complete — skipping (idempotent)", task.task_id)
            await receiver.complete_message(message)
            return

        state = task.to_initial_state()

        if existing:
            logger.info("Resuming crashed task %s from LangGraph checkpoint", task.task_id)
            result = await factory.resume_task(task.task_id)
        else:
            logger.info("Starting new task %s session=%s", task.task_id, task.user_id)
            result = await factory.run_task(state)

        await publish_result(client, task, result, success=True)

        # Complete = acknowledge + remove from queue (equivalent to SQS delete_message)
        await receiver.complete_message(message)
        logger.info("Task %s complete — message completed", task.task_id)

    except Exception as e:
        logger.error("Task %s failed (delivery %d): %s", task.task_id, task.delivery_count, e, exc_info=True)
        if task.delivery_count >= MAX_DELIVERY_COUNT:
            # Explicitly dead-letter after max attempts (Service Bus does this automatically
            # via maxDeliveryCount, but we can add a reason for diagnostics)
            await receiver.dead_letter_message(
                message,
                reason="MaxDeliveryCountExceeded",
                error_description=str(e)[:500],
            )
        # Otherwise: don't complete → lock will expire → requeued automatically

    finally:
        stop_event.set()
        renewal_task.cancel()
        try:
            await renewal_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Session-based consumer loop
# ---------------------------------------------------------------------------

async def session_worker_loop(
    client: ServiceBusClient,
    factory: AgentGraphFactory,
) -> None:
    """
    Accept one session at a time (= one user's ordered task stream).

    Service Bus sessions work differently from SQS:
    - accept_next_session() claims an available session (blocks until one arrives)
    - Within the session, receive_messages() gets queued messages in order
    - Complete the session receiver when the session is drained

    This gives you true FIFO per user — no separate ordering logic needed.
    """
    async with client.get_queue_receiver(
        TASK_QUEUE_NAME,
        receive_mode=ServiceBusReceiveMode.PEEK_LOCK,   # don't auto-complete
        max_wait_time=MAX_WAIT_SECONDS,
    ) as receiver:
        # Accept a session (blocks for up to MAX_WAIT_SECONDS)
        session_receiver = await client.accept_next_session(
            TASK_QUEUE_NAME,
            receive_mode=ServiceBusReceiveMode.PEEK_LOCK,
            max_wait_time=MAX_WAIT_SECONDS,
        )

        async with session_receiver:
            logger.info("Accepted session: %s", session_receiver.session_id)

            async for message in session_receiver:
                task = TaskMessage(message)
                logger.info(
                    "Received task task_id=%s session=%s delivery=%d",
                    task.task_id, task.user_id, task.delivery_count,
                )
                await process_task(session_receiver, message, task, factory, client)


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------

async def worker_loop(factory: AgentGraphFactory) -> None:
    """
    Outer loop: keep accepting sessions until shutdown signal.
    Each iteration handles one complete session (one user's queue).
    Multiple Container App replicas run this concurrently — each accepts a different session.
    """
    logger.info(
        "Azure Service Bus worker started — namespace=%s queue=%s",
        SERVICE_BUS_NAMESPACE,
        TASK_QUEUE_NAME,
    )

    running = True

    def handle_shutdown(sig, frame):
        nonlocal running
        logger.info("Received %s — shutting down", sig)
        running = False

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # DefaultAzureCredential: uses Managed Identity in Container Apps (no key needed),
    # falls back to CLI / environment for local dev.
    credential = DefaultAzureCredential()

    async with ServiceBusClient(
        fully_qualified_namespace=SERVICE_BUS_NAMESPACE,
        credential=credential,
        logging_enable=False,
    ) as client:
        while running:
            try:
                await session_worker_loop(client, factory)
            except ServiceBusError as e:
                logger.error("Service Bus error: %s — retrying in 5s", e)
                await asyncio.sleep(5)
            except Exception as e:
                logger.error("Worker error: %s — retrying in 5s", e, exc_info=True)
                await asyncio.sleep(5)


async def main() -> None:
    async with AgentGraphFactory() as factory:
        await worker_loop(factory)


if __name__ == "__main__":
    asyncio.run(main())
