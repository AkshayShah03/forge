"""
FastAPI API — Azure Service Bus edition.

Identical endpoints and contracts to api/main.py.
Only the queue client changes: boto3 SQS → azure-servicebus SDK.

Authentication:
  Uses DefaultAzureCredential — when running in Container Apps with a
  managed identity assigned, this automatically authenticates to Service Bus
  and Key Vault with zero credentials in code or environment variables.
  Locally: uses 'az login' credentials transparently.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from azure.identity.aio import DefaultAzureCredential
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent_system.graph.builder import AgentGraphFactory
from agent_system.state.schema import initial_state

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SERVICE_BUS_NAMESPACE = os.getenv("AZURE_SERVICE_BUS_NAMESPACE", "")
TASK_QUEUE_NAME       = os.getenv("AZURE_TASK_QUEUE_NAME",   "task-inbox")


def _setup_demo_logs() -> None:
    """Generate synthetic logs and a git repo so demo examples are fully grounded in real data."""
    import json as _json
    import random as _random
    import subprocess as _sp
    from datetime import datetime, timezone, timedelta
    from pathlib import Path

    log_dir  = Path("/tmp/forge-demo/logs")
    repo_dir = Path("/tmp/forge-demo/repo")
    log_dir.mkdir(parents=True, exist_ok=True)
    repo_dir.mkdir(parents=True, exist_ok=True)

    incident_start = datetime.now(timezone.utc) - timedelta(minutes=45)
    deploy_time    = incident_start - timedelta(minutes=30)

    # --- logs ---
    lines: list[str] = []
    t = datetime.now(timezone.utc) - timedelta(minutes=120)
    endpoints = ["/api/checkout", "/api/cart", "/api/payment"]
    while t < datetime.now(timezone.utc):
        post = t >= incident_start
        latency = max(1, int(_random.gauss(385 if post else 45, 60 if post else 8)))
        db_q    = _random.randint(18, 28) if post else _random.randint(1, 3)
        lines.append(_json.dumps({
            "ts": t.isoformat(), "method": "POST",
            "path": _random.choice(endpoints),
            "status": 200 if latency < 800 else _random.choice([200, 504]),
            "latency_ms": latency, "db_queries": db_q,
        }))
        t += timedelta(seconds=_random.uniform(0.5, 3.0))

    (log_dir / "checkout-service.log").write_text("\n".join(lines))
    (log_dir / "deploy.log").write_text("\n".join([
        _json.dumps({"ts": deploy_time.isoformat(), "event": "deploy_started",
                     "service": "checkout-service", "version": "v2.4.1",
                     "commit": "a3f9c12", "author": "eng-team"}),
        _json.dumps({"ts": (deploy_time + timedelta(seconds=45)).isoformat(),
                     "event": "deploy_completed", "service": "checkout-service",
                     "version": "v2.4.1", "rollout": "100%"}),
    ]))

    # --- git repo with real commits the coder can inspect ---
    def _git(*args: str, env: dict | None = None) -> None:
        _sp.run(["git", "-C", str(repo_dir)] + list(args),
                check=False, capture_output=True,
                env={**os.environ, **(env or {})})

    _git("init")
    _git("config", "user.email", "eng@checkout-service.internal")
    _git("config", "user.name",  "Checkout Engineering")

    # v2.4.0 — healthy baseline committed 2 hours before deploy
    views_before = '''\
from django.db import models

def get_order_items(order_id):
    """Return all items for an order with their product details."""
    return (
        models.OrderItem.objects
        .filter(order_id=order_id)
        .select_related("product", "product__category")
    )
'''
    (repo_dir / "checkout").mkdir(exist_ok=True)
    (repo_dir / "checkout" / "views.py").write_text(views_before)
    (repo_dir / "checkout" / "__init__.py").write_text("")
    _git("add", ".")
    baseline_time = (deploy_time - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
    _git("commit", "--allow-empty", "-m", "v2.4.0: stable release",
         env={"GIT_AUTHOR_DATE": baseline_time, "GIT_COMMITTER_DATE": baseline_time,
              **os.environ})

    # v2.4.1 — the regression: select_related removed during cart refactor
    views_after = '''\
from django.db import models

def get_order_items(order_id):
    """Return all items for an order with their product details."""
    # Refactored for new cart pipeline — removed select_related to support lazy loading
    return (
        models.OrderItem.objects
        .filter(order_id=order_id)
    )
'''
    (repo_dir / "checkout" / "views.py").write_text(views_after)
    _git("add", ".")
    deploy_ts = deploy_time.strftime("%Y-%m-%dT%H:%M:%S")
    _git("commit", "-m",
         "v2.4.1: Refactor checkout item loading for new cart pipeline\n\n"
         "Removes select_related() to support the new lazy-loading cart feature.\n"
         "Refs: CART-412",
         env={"GIT_AUTHOR_DATE": deploy_ts, "GIT_COMMITTER_DATE": deploy_ts,
              **os.environ})

    logger.info(
        "Demo ready — logs: %s  repo: %s  incident_start: %s",
        log_dir, repo_dir, incident_start.isoformat(),
    )


# ---------------------------------------------------------------------------
# Request / response models — unchanged from AWS version
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    user_input:     str  = Field(..., min_length=1, max_length=10_000)
    user_id:        str  = Field(..., min_length=1)
    token_budget:   int  = Field(default=50_000, ge=1_000, le=200_000)
    max_iterations: int  = Field(default=3, ge=1, le=5)
    metadata:       dict = Field(default_factory=dict)

class TaskResponse(BaseModel):
    task_id:   str
    status:    str
    queued_at: float

class TaskStatus(BaseModel):
    task_id:         str
    status:          str
    final_answer:    Optional[str]
    score:           Optional[float]
    iterations:      Optional[int]
    tokens_used:     Optional[int]
    error:           Optional[str]
    subtask_results: Optional[list]


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_demo_logs()

    graph_factory = AgentGraphFactory()
    await graph_factory.__aenter__()
    app.state.factory = graph_factory

    if SERVICE_BUS_NAMESPACE:
        credential = DefaultAzureCredential()
        sb_client  = ServiceBusClient(
            fully_qualified_namespace=SERVICE_BUS_NAMESPACE,
            credential=credential,
        )
        app.state.sb_client  = sb_client
        app.state.credential = credential
        logger.info("API started — Service Bus namespace: %s", SERVICE_BUS_NAMESPACE)
    else:
        logger.info("API started — local mode (no Service Bus configured)")

    yield

    await graph_factory.__aexit__(None, None, None)
    if SERVICE_BUS_NAMESPACE and hasattr(app.state, "sb_client"):
        await app.state.sb_client.close()
        await app.state.credential.close()


app = FastAPI(
    title="Multi-Agent Orchestration API (Azure)",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints — identical contracts to the AWS version
# ---------------------------------------------------------------------------

@app.post("/tasks", response_model=TaskResponse, status_code=202)
async def submit_task(req: TaskRequest, request: Request) -> TaskResponse:
    """
    Submit a task to Azure Service Bus.

    SessionId = user_id  →  FIFO ordering per user (equiv. to SQS MessageGroupId)
    MessageId = task_id  →  duplicate detection (equiv. to SQS MessageDeduplicationId)
    """
    task_id   = str(uuid.uuid4())
    queued_at = time.time()

    body = {
        "task_id":        task_id,
        "user_id":        req.user_id,
        "user_input":     req.user_input,
        "token_budget":   req.token_budget,
        "max_iterations": req.max_iterations,
        "metadata":       req.metadata,
        "queued_at":      queued_at,
    }

    if SERVICE_BUS_NAMESPACE:
        client: ServiceBusClient = request.app.state.sb_client
        async with client.get_queue_sender(TASK_QUEUE_NAME) as sender:
            msg = ServiceBusMessage(
                body=json.dumps(body),
                message_id=task_id,         # deduplication ID
                session_id=req.user_id,     # FIFO session key
            )
            await sender.send_messages(msg)
        logger.info("Task %s enqueued to Service Bus session=%s", task_id, req.user_id)
    else:
        # Local dev: run synchronously without a real queue
        logger.warning("AZURE_SERVICE_BUS_NAMESPACE not set — running inline (dev mode)")
        factory: AgentGraphFactory = request.app.state.factory
        state = initial_state(
            task_id=task_id,
            user_input=req.user_input,
            token_budget=req.token_budget,
            max_iterations=req.max_iterations,
            metadata={"user_id": req.user_id, **req.metadata},
        )
        import asyncio

        async def _run_and_log():
            try:
                await factory.run_task(state)
            except Exception:
                logger.exception("Task %s failed", task_id)

        asyncio.create_task(_run_and_log())

    return TaskResponse(task_id=task_id, status="queued", queued_at=queued_at)


@app.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str, request: Request) -> TaskStatus:
    """Poll status — reads directly from LangGraph PostgreSQL checkpoint. Unchanged."""
    factory: AgentGraphFactory = request.app.state.factory
    state = await factory.get_task_state(task_id)

    if not state:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    final_answer = state.get("final_answer")
    critic       = state.get("critic_feedback") or {}
    status       = "complete" if (final_answer or state.get("route_to") == "end") else "running"

    return TaskStatus(
        task_id=task_id,
        status=status,
        final_answer=final_answer,
        score=critic.get("score"),
        iterations=state.get("iteration_count"),
        tokens_used=state.get("total_tokens_used"),
        error=None,
        subtask_results=state.get("subtask_results"),
    )


@app.get("/tasks/{task_id}/stream")
async def stream_task_updates(task_id: str, request: Request) -> StreamingResponse:
    """SSE stream — unchanged logic, polls LangGraph checkpoint."""
    factory: AgentGraphFactory = request.app.state.factory

    async def event_generator() -> AsyncIterator[str]:
        import asyncio
        last_iter = -1
        while True:
            if await request.is_disconnected():
                break
            state = await factory.get_task_state(task_id)
            if not state:
                yield f"event: error\ndata: {{\"message\": \"Task not found\"}}\n\n"
                break
            cur_iter     = state.get("iteration_count", 0)
            final_answer = state.get("final_answer")
            critic       = state.get("critic_feedback") or {}

            if cur_iter != last_iter:
                last_iter = cur_iter
                yield f"event: update\ndata: {json.dumps({'task_id': task_id, 'iteration': cur_iter, 'status': 'running'})}\n\n"

            if final_answer or state.get("route_to") == "end":
                yield f"event: complete\ndata: {json.dumps({'task_id': task_id, 'status': 'complete', 'final_answer': final_answer, 'score': critic.get('score')})}\n\n"
                break
            await asyncio.sleep(2.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/tasks/{task_id}/history")
async def get_task_history(task_id: str, request: Request) -> list[dict]:
    factory: AgentGraphFactory = request.app.state.factory
    history = await factory.list_task_history(task_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return history


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "multi-agent-api-azure"}


@app.get("/demo")
async def demo_info() -> dict:
    """Return metadata about the demo data so the UI can show exactly what is being analyzed."""
    import subprocess as _sp
    from pathlib import Path

    log_path  = Path("/tmp/forge-demo/logs/checkout-service.log")
    repo_path = Path("/tmp/forge-demo/repo")

    line_count = len(log_path.read_text().splitlines()) if log_path.exists() else 0

    git_log = ""
    diff    = ""
    if repo_path.exists():
        git_log = _sp.run(
            ["git", "-C", str(repo_path), "log", "--format=%h %ad %s", "--date=short"],
            capture_output=True, text=True,
        ).stdout.strip()
        diff = _sp.run(
            ["git", "-C", str(repo_path), "show", "HEAD", "-p", "--", "checkout/views.py"],
            capture_output=True, text=True,
        ).stdout.strip()

    return {
        "log_dir":        "/tmp/forge-demo/logs",
        "repo_dir":       "/tmp/forge-demo/repo",
        "log_line_count": line_count,
        "git_log":        git_log,
        "diff":           diff,
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui() -> str:
    """Browser UI for submitting incident investigations and watching live agent progress."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Forge / Incident Analyzer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 2rem; max-width: 900px; margin: 0 auto; }
  h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.2rem; color: #f8fafc; letter-spacing: -0.01em; }
  .subtitle { color: #64748b; font-size: 0.85rem; margin-bottom: 1.25rem; }
  .subtitle a { color: #6366f1; text-decoration: none; }
  .subtitle a:hover { text-decoration: underline; }

  .card { background: #1e2230; border: 1px solid #2d3348; border-radius: 0.75rem;
          padding: 1.25rem 1.5rem; margin-bottom: 1.1rem; }

  /* demo data panel */
  .demo-panel { background: #161b27; border: 1px solid #1e2d1e; border-radius: 0.75rem;
                margin-bottom: 1.1rem; overflow: hidden; }
  .demo-panel-header { display: flex; align-items: center; justify-content: space-between;
                       padding: 0.7rem 1.1rem; cursor: pointer; user-select: none; }
  .demo-panel-header:hover { background: #1a2230; }
  .demo-panel-title { font-size: 0.82rem; font-weight: 600; color: #4ade80; display: flex; align-items: center; gap: 0.5rem; }
  .demo-panel-title::before { content: ""; display: inline-block; width: 7px; height: 7px;
                               background: #4ade80; border-radius: 50%; }
  .demo-chevron { color: #475569; font-size: 0.75rem; transition: transform .2s; }
  .demo-panel-body { padding: 0 1.1rem 1rem; display: none; }
  .demo-panel-body.open { display: block; }

  .demo-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.85rem; }
  .demo-box { background: #0f1117; border: 1px solid #1e2535; border-radius: 0.5rem; padding: 0.65rem 0.85rem; }
  .demo-box-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: .08em;
                    color: #475569; margin-bottom: 0.3rem; }
  .demo-box-val { font-family: "SF Mono","Fira Code",monospace; font-size: 0.78rem; color: #7dd3fc; }
  .demo-box-sub { font-size: 0.72rem; color: #475569; margin-top: 0.2rem; }

  .diff-block { background: #0a0c12; border: 1px solid #1e2535; border-radius: 0.5rem;
                padding: 0.65rem 0.85rem; font-family: "SF Mono","Fira Code",monospace;
                font-size: 0.76rem; line-height: 1.7; overflow-x: auto; }
  .diff-filename { color: #64748b; margin-bottom: 0.4rem; font-size: 0.72rem; }
  .diff-add { color: #4ade80; }
  .diff-del { color: #f87171; }
  .diff-ctx { color: #475569; }
  .diff-loading { color: #475569; font-style: italic; font-size: 0.78rem; }

  /* live pipeline */
  .pipeline-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .pipe-step { display: flex; align-items: center; gap: 0.4rem; background: #151b2a;
               border: 1px solid #2a3450; border-radius: 0.4rem; padding: 0.3rem 0.75rem;
               font-size: 0.78rem; font-weight: 500; color: #475569; transition: all .3s; }
  .pipe-step.active { border-color: #6366f1; color: #a5b4fc; background: #1a1f3a; }
  .pipe-step.done   { border-color: #4ade80; color: #4ade80; background: #0f1f14; }
  .pipe-arrow { color: #2d3348; font-size: 0.8rem; }
  .pipe-dot { width: 7px; height: 7px; border-radius: 50%; background: #2d3348; flex-shrink: 0; }
  .pipe-step.active .pipe-dot { background: #6366f1; animation: pulse 1.2s ease-in-out infinite; }
  .pipe-step.done   .pipe-dot { background: #4ade80; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  .examples-label { color: #64748b; font-size: 0.78rem; margin-bottom: 0.5rem; }
  .examples { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.9rem; }
  .example-btn { background: #151b2a; border: 1px solid #2a3450; border-radius: 0.4rem;
                 color: #94a3b8; font-size: 0.75rem; padding: 0.3rem 0.7rem;
                 cursor: pointer; transition: border-color .15s, color .15s; }
  .example-btn:hover { border-color: #6366f1; color: #e2e8f0; }

  textarea { width: 100%; background: #0f1117; border: 1px solid #2d3348;
             border-radius: 0.5rem; color: #e2e8f0; font-size: 0.88rem;
             padding: 0.75rem; resize: vertical; min-height: 95px;
             outline: none; transition: border-color .2s; line-height: 1.55; }
  textarea:focus { border-color: #6366f1; }
  .hint { color: #475569; font-size: 0.73rem; margin-top: 0.35rem; }

  .row { display: flex; gap: 0.75rem; align-items: center; margin-top: 0.85rem; }
  label { color: #94a3b8; font-size: 0.82rem; white-space: nowrap; }
  select { background: #0f1117; border: 1px solid #2d3348; border-radius: 0.45rem;
           color: #e2e8f0; padding: 0.4rem 0.6rem; font-size: 0.82rem; }
  button#submit-btn { background: #6366f1; color: #fff; border: none; border-radius: 0.5rem;
           padding: 0.55rem 1.4rem; font-size: 0.88rem; font-weight: 500;
           cursor: pointer; transition: background .15s; margin-left: auto; }
  button#submit-btn:hover:not(:disabled) { background: #4f46e5; }
  button#submit-btn:disabled { opacity: .5; cursor: not-allowed; }

  /* agent outputs — appear live as each agent finishes */
  #subtasks { margin-bottom: 1.1rem; }
  .subtasks-label { color: #64748b; font-size: 0.75rem; margin-bottom: 0.5rem; padding-left: 0.1rem; }
  .subtask { border: 1px solid #2d3348; border-radius: 0.5rem; margin-bottom: 0.6rem; overflow: hidden; }
  .subtask-header { background: #1a1f2e; padding: 0.4rem 0.85rem;
                    font-size: 0.76rem; display: flex; gap: 0.75rem; align-items: center; }
  .subtask-role { color: #7dd3fc; font-weight: 600; text-transform: uppercase;
                  font-size: 0.67rem; letter-spacing: .07em; }
  .subtask-name { color: #94a3b8; flex: 1; }
  .subtask-body { padding: 0.65rem 0.85rem; font-size: 0.78rem; line-height: 1.65;
                  white-space: pre-wrap; color: #cbd5e1;
                  font-family: "SF Mono","Fira Code",monospace;
                  max-height: 260px; overflow-y: auto; background: #0a0c12; }

  /* postmortem */
  #answer { display: none; }
  #answer h2 { font-size: 0.88rem; color: #94a3b8; margin-bottom: 0.6rem; font-weight: 500; }
  #answer-text { color: #f1f5f9; line-height: 1.8; white-space: pre-wrap; font-size: 0.88rem; }
  .meta { display: flex; gap: 1.5rem; margin-top: 1rem; font-size: 0.76rem; flex-wrap: wrap; }
  .meta span { color: #64748b; }
  .meta b { color: #94a3b8; }

  /* activity log */
  #log { font-family: "SF Mono","Fira Code",monospace; font-size: 0.76rem;
         line-height: 1.7; max-height: 180px; overflow-y: auto;
         background: #0a0c12; border-radius: 0.5rem; padding: 0.75rem;
         border: 1px solid #1a1f2e; display: none; }
  .log-line { display: flex; gap: 0.5rem; }
  .log-time { color: #475569; flex-shrink: 0; }
  .log-info { color: #38bdf8; }
  .log-done { color: #34d399; }
  .log-warn { color: #fb923c; }
  .log-err  { color: #f87171; }

  .badge { display: inline-flex; align-items: center; gap: 0.35rem;
           background: #1a2332; border: 1px solid #2d3f55;
           border-radius: 9999px; padding: 0.2rem 0.65rem;
           font-size: 0.72rem; color: #7dd3fc; }
  .spinner { display: inline-block; width: 10px; height: 10px;
             border: 2px solid #334155; border-top-color: #6366f1;
             border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<h1>Forge</h1>
<p class="subtitle">Incident root cause analyzer &nbsp;&middot;&nbsp; <a href="/docs" target="_blank">API docs</a> &nbsp;&middot;&nbsp; <a href="https://github.com/AkshayShah03/forge" target="_blank">GitHub</a></p>

<!-- demo data panel -->
<div class="demo-panel" id="demo-panel">
  <div class="demo-panel-header" onclick="toggleDemo()">
    <span class="demo-panel-title">Demo data — what the agents actually read</span>
    <span class="demo-chevron" id="demo-chevron">&#9660;</span>
  </div>
  <div class="demo-panel-body open" id="demo-body">
    <div class="demo-grid">
      <div class="demo-box">
        <div class="demo-box-label">Log files</div>
        <div class="demo-box-val" id="demo-log-path">/tmp/forge-demo/logs/</div>
        <div class="demo-box-sub" id="demo-log-lines">loading...</div>
      </div>
      <div class="demo-box">
        <div class="demo-box-label">Git repository</div>
        <div class="demo-box-val" id="demo-repo-path">/tmp/forge-demo/repo</div>
        <div class="demo-box-sub" id="demo-git-log">loading...</div>
      </div>
    </div>
    <div class="diff-filename">checkout/views.py &mdash; what changed in v2.4.1 (the coder reads this file)</div>
    <div class="diff-block" id="demo-diff"><span class="diff-loading">loading diff...</span></div>
  </div>
</div>

<div class="card">
  <div class="examples-label">Try an example</div>
  <div class="examples">
    <button class="example-btn" onclick="loadExample('latency')">P95 latency spike</button>
    <button class="example-btn" onclick="loadExample('error')">Error rate jump</button>
    <button class="example-btn" onclick="loadExample('memory')">OOM / memory leak</button>
    <button class="example-btn" onclick="loadExample('db')">Slow database queries</button>
  </div>
  <textarea id="input" placeholder="Describe the incident: what broke, when it started, where the logs are, and which service is affected..."></textarea>
  <p class="hint">The P95 latency example uses the real logs and git repo above. Cmd+Enter to submit.</p>
  <div class="row">
    <label>Max retries</label>
    <select id="max-iter">
      <option value="1">1</option>
      <option value="2" selected>2</option>
      <option value="3">3</option>
    </select>
    <label>Token budget</label>
    <select id="budget">
      <option value="20000">20k</option>
      <option value="50000" selected>50k</option>
      <option value="100000">100k</option>
    </select>
    <button id="submit-btn" onclick="submitTask()">Investigate</button>
  </div>
</div>

<!-- live pipeline — updates as agents complete -->
<div class="pipeline-row" id="pipeline" style="display:none">
  <div class="pipe-step" id="ps-orchestrator"><span class="pipe-dot"></span>Orchestrator</div>
  <span class="pipe-arrow">&#8594;</span>
  <div class="pipe-step" id="ps-researcher"><span class="pipe-dot"></span>Researcher</div>
  <span class="pipe-arrow">&#8594;</span>
  <div class="pipe-step" id="ps-coder"><span class="pipe-dot"></span>Coder</div>
  <span class="pipe-arrow">&#8594;</span>
  <div class="pipe-step" id="ps-analyst"><span class="pipe-dot"></span>Analyst</div>
  <span class="pipe-arrow">&#8594;</span>
  <div class="pipe-step" id="ps-critic"><span class="pipe-dot"></span>Critic</div>
</div>

<!-- agent outputs appear here live as each agent finishes -->
<div id="subtasks">
  <div class="subtasks-label" id="subtasks-label" style="display:none">Agent outputs (live)</div>
  <div id="subtasks-list"></div>
</div>

<div class="card" id="answer">
  <h2>Postmortem</h2>
  <div id="answer-text"></div>
  <div class="meta">
    <span><b>Quality score</b> <span id="meta-score">&#8212;</span></span>
    <span><b>Iterations</b> <span id="meta-iter">&#8212;</span></span>
    <span><b>Tokens used</b> <span id="meta-tokens">&#8212;</span></span>
  </div>
</div>

<div class="card" style="padding:0.65rem 1rem;">
  <div id="log"></div>
  <div id="log-empty" style="color:#475569;font-size:0.8rem;">Activity log appears here when an investigation is running.</div>
</div>

<script>
// ── demo data panel ──────────────────────────────────────────────────────────
async function loadDemoInfo() {
  try {
    const d = await fetch('/demo').then(r => r.json());
    document.getElementById('demo-log-path').textContent = d.log_dir + '/';
    document.getElementById('demo-log-lines').textContent =
      d.log_line_count.toLocaleString() + ' log lines — generated at startup';
    document.getElementById('demo-repo-path').textContent = d.repo_dir;
    document.getElementById('demo-git-log').textContent =
      d.git_log ? d.git_log.split('\\n').join('  |  ') : 'no commits';
    if (d.diff) {
      const html = d.diff.split('\\n').map(line => {
        if (line.startsWith('+++') || line.startsWith('---')) return `<span class="diff-ctx">${escHtml(line)}</span>`;
        if (line.startsWith('+')) return `<span class="diff-add">${escHtml(line)}</span>`;
        if (line.startsWith('-')) return `<span class="diff-del">${escHtml(line)}</span>`;
        if (line.startsWith('@@')) return `<span class="diff-ctx" style="color:#7dd3fc">${escHtml(line)}</span>`;
        return `<span class="diff-ctx">${escHtml(line)}</span>`;
      }).join('\\n');
      document.getElementById('demo-diff').innerHTML = html;
    }
  } catch(e) {
    document.getElementById('demo-diff').innerHTML = '<span class="diff-loading">Could not load demo info</span>';
  }
}

function toggleDemo() {
  const body = document.getElementById('demo-body');
  const chev = document.getElementById('demo-chevron');
  body.classList.toggle('open');
  chev.style.transform = body.classList.contains('open') ? '' : 'rotate(-90deg)';
}

// ── live pipeline state ───────────────────────────────────────────────────────
const AGENT_ORDER = ['orchestrator','researcher','coder','analyst','critic'];

function setPipeState(doneAgents, activeAgent) {
  AGENT_ORDER.forEach(a => {
    const el = document.getElementById('ps-' + a);
    el.classList.remove('active','done');
    if (doneAgents.includes(a)) el.classList.add('done');
    else if (a === activeAgent)  el.classList.add('active');
  });
}

function inferPipeState(subtaskResults, isDone) {
  const doneAgents = ['orchestrator']; // orchestrator always done once we have a plan
  const roles = (subtaskResults || []).map(r => r.agent);
  roles.forEach(r => { if (!doneAgents.includes(r)) doneAgents.push(r); });

  if (isDone) {
    setPipeState([...doneAgents, 'critic'], null);
    return;
  }
  // infer what's currently running based on what's done
  const remaining = AGENT_ORDER.filter(a => !doneAgents.includes(a));
  const active = remaining[0] || null;
  setPipeState(doneAgents, active);
}

// ── examples ─────────────────────────────────────────────────────────────────
const _now = new Date();
const _incident = new Date(_now - 45 * 60 * 1000);
const _deploy   = new Date(_now - 75 * 60 * 1000);
const EXAMPLES = {
  latency: `P95 latency in checkout-service jumped from 45ms to 385ms starting at ${_incident.toISOString()}. A deploy (v2.4.1) finished at ${_deploy.toISOString()}. Logs are at /tmp/forge-demo/logs/. The repo is at /tmp/forge-demo/repo. Investigate the root cause and write a postmortem.`,
  error: `Error rate on the payments API spiked from 0.1% to 8.3% at 03:17 UTC. Errors are all 500s with "connection refused". The service connects to Redis for session data. Investigate and draft a postmortem.`,
  memory: `The recommendation-service pod has been OOMKilled three times in the past two hours. Memory climbs from 400MB to 2GB over 45 minutes. A new feature flag was enabled yesterday. Investigate the root cause.`,
  db: `Slow query alerts firing for user-service since 09:40. Average query time went from 8ms to 340ms. The user table has 12 million rows. A migration ran this morning that added a new index. Find the root cause.`,
};

function loadExample(key) {
  document.getElementById('input').value = EXAMPLES[key];
}

// ── logging ──────────────────────────────────────────────────────────────────
let evtSource = null;

function ts() {
  return new Date().toLocaleTimeString('en-US',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
}

function addLog(msg, cls='log-info') {
  const log = document.getElementById('log');
  document.getElementById('log-empty').style.display = 'none';
  log.style.display = 'block';
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="log-time">${ts()}</span><span class="${cls}">${msg}</span>`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

// ── subtask rendering (live, called during polling) ───────────────────────────
let _renderedCount = 0;

function renderSubtasksLive(results) {
  if (!results || results.length === 0) return;
  const list = document.getElementById('subtasks-list');
  document.getElementById('subtasks-label').style.display = '';
  for (let i = _renderedCount; i < results.length; i++) {
    const r = results[i];
    const el = document.createElement('div');
    el.className = 'subtask';
    el.innerHTML = `
      <div class="subtask-header">
        <span class="subtask-role">${r.agent}</span>
        <span class="subtask-name">${escHtml(r.subtask)}</span>
      </div>
      <div class="subtask-body">${escHtml(r.result || '')}</div>`;
    list.appendChild(el);
    addLog(`${r.agent} finished: ${r.subtask.slice(0,60)}...`, 'log-done');
  }
  _renderedCount = results.length;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── main submit ──────────────────────────────────────────────────────────────
async function submitTask() {
  const input = document.getElementById('input').value.trim();
  if (!input) return;

  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';

  document.getElementById('log').innerHTML = '';
  document.getElementById('log').style.display = 'none';
  document.getElementById('log-empty').style.display = '';
  document.getElementById('answer').style.display = 'none';
  document.getElementById('subtasks-list').innerHTML = '';
  document.getElementById('subtasks-label').style.display = 'none';
  document.getElementById('pipeline').style.display = 'flex';
  _renderedCount = 0;
  if (evtSource) { evtSource.close(); evtSource = null; }

  setPipeState([], 'orchestrator');
  addLog('Submitting incident to Forge...');

  const resp = await fetch('/tasks', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      user_input: input,
      user_id: 'ui-user',
      max_iterations: parseInt(document.getElementById('max-iter').value),
      token_budget: parseInt(document.getElementById('budget').value),
    })
  });

  if (!resp.ok) {
    addLog('Submission failed: ' + resp.status, 'log-err');
    btn.disabled = false; btn.innerHTML = 'Investigate';
    return;
  }

  const {task_id} = await resp.json();
  addLog(`Task <span class="badge">${task_id.slice(0,8)}&hellip;</span> queued — orchestrator planning subtasks...`);

  // SSE for completion signal
  evtSource = new EventSource(`/tasks/${task_id}/stream`);
  evtSource.addEventListener('complete', async e => {
    evtSource.close();
    const s = await fetch(`/tasks/${task_id}`).then(r => r.json());
    renderSubtasksLive(s.subtask_results || []);
    inferPipeState(s.subtask_results, true);
    addLog('Investigation complete', 'log-done');
    btn.disabled = false; btn.innerHTML = 'Investigate';
    document.getElementById('answer').style.display = 'block';
    document.getElementById('answer-text').textContent = s.final_answer || '(no answer)';
    const score = s.score ?? JSON.parse(e.data).score;
    document.getElementById('meta-score').textContent = score != null ? Number(score).toFixed(2) : '—';
    document.getElementById('meta-iter').textContent = s.iterations ?? '—';
    document.getElementById('meta-tokens').textContent = s.tokens_used != null ? s.tokens_used.toLocaleString() : '—';
  });
  evtSource.addEventListener('error', () => {
    addLog('Stream dropped, switching to polling...', 'log-warn');
    evtSource.close();
    pollUntilDone(task_id, btn);
  });

  // live polling every 3s to show subtask results as they appear
  const pollInterval = setInterval(async () => {
    try {
      const s = await fetch(`/tasks/${task_id}`).then(r => r.json());
      renderSubtasksLive(s.subtask_results || []);
      inferPipeState(s.subtask_results, s.status === 'complete');
      if (s.status === 'complete') clearInterval(pollInterval);
    } catch(_) {}
  }, 3000);

  // clear poll interval on completion too
  evtSource.addEventListener('complete', () => clearInterval(pollInterval));
}

async function pollUntilDone(task_id, btn) {
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 4000));
    try {
      const s = await fetch(`/tasks/${task_id}`).then(r => r.json());
      renderSubtasksLive(s.subtask_results || []);
      inferPipeState(s.subtask_results, s.status === 'complete');
      if (s.status === 'complete') {
        addLog('Investigation complete', 'log-done');
        btn.disabled = false; btn.innerHTML = 'Investigate';
        document.getElementById('answer').style.display = 'block';
        document.getElementById('answer-text').textContent = s.final_answer || '(no answer)';
        document.getElementById('meta-score').textContent = s.score?.toFixed(2) ?? '—';
        document.getElementById('meta-iter').textContent = s.iterations ?? '—';
        document.getElementById('meta-tokens').textContent = s.tokens_used?.toLocaleString() ?? '—';
        return;
      }
    } catch(_) {}
    if (i % 3 === 2) addLog(`Still running... (${(i+1)*4}s elapsed)`);
  }
  addLog('Timed out after 4 minutes', 'log-err');
  btn.disabled = false; btn.innerHTML = 'Investigate';
}

document.getElementById('input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submitTask();
});

loadDemoInfo();
</script>
</body>
</html>"""
