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
    """Generate synthetic checkout-service logs so the built-in demo examples work immediately."""
    import json as _json
    import random as _random
    from datetime import datetime, timezone, timedelta
    from pathlib import Path

    log_dir = Path("/tmp/forge-demo/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    incident_start = datetime.now(timezone.utc) - timedelta(minutes=45)
    deploy_time    = incident_start - timedelta(minutes=30)

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
    logger.info("Demo logs ready at %s  incident_start=%s", log_dir, incident_start.isoformat())


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
         background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 2rem; max-width: 860px; margin: 0 auto; }
  h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.2rem; color: #f8fafc; letter-spacing: -0.01em; }
  .subtitle { color: #64748b; font-size: 0.85rem; margin-bottom: 1.75rem; }
  .subtitle a { color: #6366f1; text-decoration: none; }
  .subtitle a:hover { text-decoration: underline; }

  .how-it-works { background: #161b27; border: 1px solid #1e2535; border-radius: 0.6rem;
                  padding: 1rem 1.25rem; margin-bottom: 1.25rem; font-size: 0.82rem; color: #64748b; line-height: 1.6; }
  .how-it-works strong { color: #94a3b8; }
  .pipeline { display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap;
              margin-top: 0.6rem; font-size: 0.78rem; }
  .agent-chip { background: #1a2235; border: 1px solid #2a3450;
                border-radius: 0.35rem; padding: 0.25rem 0.6rem; color: #7dd3fc; font-weight: 500; }
  .arrow { color: #334155; }

  .card { background: #1e2230; border: 1px solid #2d3348; border-radius: 0.75rem;
          padding: 1.5rem; margin-bottom: 1.25rem; }
  .examples-label { color: #64748b; font-size: 0.78rem; margin-bottom: 0.5rem; }
  .examples { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.9rem; }
  .example-btn { background: #151b2a; border: 1px solid #2a3450; border-radius: 0.4rem;
                 color: #94a3b8; font-size: 0.75rem; padding: 0.3rem 0.7rem;
                 cursor: pointer; transition: border-color .15s, color .15s; }
  .example-btn:hover { border-color: #6366f1; color: #e2e8f0; }

  textarea { width: 100%; background: #0f1117; border: 1px solid #2d3348;
             border-radius: 0.5rem; color: #e2e8f0; font-size: 0.9rem;
             padding: 0.75rem; resize: vertical; min-height: 100px;
             outline: none; transition: border-color .2s; line-height: 1.55; }
  textarea:focus { border-color: #6366f1; }
  .hint { color: #475569; font-size: 0.75rem; margin-top: 0.4rem; }

  .row { display: flex; gap: 0.75rem; align-items: center; margin-top: 0.85rem; }
  label { color: #94a3b8; font-size: 0.82rem; white-space: nowrap; }
  select { background: #0f1117; border: 1px solid #2d3348; border-radius: 0.45rem;
           color: #e2e8f0; padding: 0.4rem 0.6rem; font-size: 0.82rem; }
  button#submit-btn { background: #6366f1; color: #fff; border: none; border-radius: 0.5rem;
           padding: 0.55rem 1.4rem; font-size: 0.9rem; font-weight: 500;
           cursor: pointer; transition: background .15s; margin-left: auto; }
  button#submit-btn:hover:not(:disabled) { background: #4f46e5; }
  button#submit-btn:disabled { opacity: .5; cursor: not-allowed; }

  #log { font-family: "SF Mono", "Fira Code", monospace; font-size: 0.78rem;
         line-height: 1.7; max-height: 220px; overflow-y: auto;
         background: #0a0c12; border-radius: 0.5rem; padding: 0.85rem;
         border: 1px solid #1a1f2e; display: none; }
  .log-line { display: flex; gap: 0.5rem; }
  .log-time { color: #475569; flex-shrink: 0; }
  .log-info  { color: #38bdf8; }
  .log-done  { color: #34d399; }
  .log-warn  { color: #fb923c; }
  .log-err   { color: #f87171; }

  #answer { display: none; }
  #answer h2 { font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.6rem; font-weight: 500; }
  #answer-text { color: #f1f5f9; line-height: 1.75; white-space: pre-wrap; font-family: inherit; font-size: 0.9rem; }
  .meta { display: flex; gap: 1.5rem; margin-top: 1rem; font-size: 0.78rem; }
  .meta span { color: #64748b; }
  .meta b { color: #94a3b8; }

  #subtasks { display: none; margin-top: 1.25rem; }
  .subtasks-label { color: #64748b; font-size: 0.78rem; margin-bottom: 0.5rem; padding-left: 0.1rem; }
  .subtask { border: 1px solid #2d3348; border-radius: 0.5rem; margin-bottom: 0.75rem; overflow: hidden; }
  .subtask-header { background: #1a1f2e; padding: 0.45rem 0.85rem;
                    font-size: 0.78rem; display: flex; gap: 0.75rem; align-items: center; }
  .subtask-role { color: #7dd3fc; font-weight: 600; text-transform: uppercase;
                  font-size: 0.68rem; letter-spacing: .06em; }
  .subtask-name { color: #94a3b8; flex: 1; }
  .subtask-body { padding: 0.75rem 0.85rem; font-size: 0.8rem; line-height: 1.65;
                  white-space: pre-wrap; color: #cbd5e1;
                  font-family: "SF Mono","Fira Code",monospace;
                  max-height: 280px; overflow-y: auto; background: #0a0c12; }

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
<p class="subtitle">Incident root cause analyzer &nbsp;&middot;&nbsp; <a href="/docs" target="_blank">API docs</a></p>

<div class="how-it-works">
  <strong>How it works:</strong> describe an incident (what broke, when, where the logs are) and Forge spins up a team of agents to investigate it.
  <div class="pipeline">
    <span class="agent-chip">Researcher</span><span class="arrow">&#8594;</span>
    <span class="agent-chip">Coder</span><span class="arrow">&#8594;</span>
    <span class="agent-chip">Analyst</span><span class="arrow">&#8594;</span>
    <span class="agent-chip">Critic</span>
  </div>
  The researcher looks up known failure patterns. The coder reads the logs and queries git history. The analyst correlates both and writes a postmortem. The critic scores it and re-runs if it's too vague.
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
  <p class="hint">Tip: include the log directory path and repo path for the best results. Cmd+Enter to submit.</p>
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

<div class="card" id="answer">
  <h2>Postmortem</h2>
  <div id="answer-text"></div>
  <div class="meta">
    <span><b>Quality score</b> <span id="meta-score">&#8212;</span></span>
    <span><b>Iterations</b> <span id="meta-iter">&#8212;</span></span>
    <span><b>Tokens used</b> <span id="meta-tokens">&#8212;</span></span>
  </div>
</div>

<div id="subtasks">
  <div class="subtasks-label">Agent outputs</div>
  <div id="subtasks-list"></div>
</div>

<div class="card" style="padding:0.75rem 1rem;">
  <div id="log"></div>
  <div id="log-empty" style="color:#475569;font-size:0.82rem;">Agent steps will appear here once you start an investigation.</div>
</div>

<script>
const _now = new Date();
const _incident = new Date(_now - 45 * 60 * 1000);
const _deploy   = new Date(_now - 75 * 60 * 1000);
const EXAMPLES = {
  latency: `P95 latency in checkout-service jumped from 45ms to 385ms starting at ${_incident.toISOString()}. A deploy (v2.4.1) finished at ${_deploy.toISOString()}. Logs are at /tmp/forge-demo/logs/. Investigate the root cause and write a postmortem.`,
  error: `Error rate on the payments API spiked from 0.1% to 8.3% at 03:17 UTC. The errors are all 500s with "connection refused" in the logs. The service connects to Redis for session data. Logs are at /var/log/payments/. Investigate and draft a postmortem.`,
  memory: `The recommendation-service pod has been OOMKilled three times in the past two hours. Memory usage climbs steadily from 400MB to 2GB over about 45 minutes before the process is killed. A new feature flag was enabled yesterday. Logs are at /var/log/reco/. Investigate the root cause.`,
  db: `Slow query alerts firing for the user-service database since 09:40. Average query time went from 8ms to 340ms. The user table has 12 million rows. A migration ran this morning that added a new index. Logs are at /var/log/userservice/ and the DB slow query log is at /var/log/postgres/slow.log. Find the root cause.`,
};

function loadExample(key) {
  document.getElementById('input').value = EXAMPLES[key];
}

let evtSource = null;

function ts() {
  return new Date().toLocaleTimeString('en-US',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
}

function addLog(msg, cls='log-info') {
  const log = document.getElementById('log');
  const empty = document.getElementById('log-empty');
  log.style.display = 'block';
  empty.style.display = 'none';
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="log-time">${ts()}</span><span class="${cls}">${msg}</span>`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

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
  document.getElementById('subtasks').style.display = 'none';
  document.getElementById('subtasks-list').innerHTML = '';
  if (evtSource) { evtSource.close(); evtSource = null; }

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
  addLog(`Task <span class="badge">${task_id.slice(0,8)}&hellip;</span> queued &mdash; agents starting...`);

  evtSource = new EventSource(`/tasks/${task_id}/stream`);

  evtSource.addEventListener('update', e => {
    const d = JSON.parse(e.data);
    addLog(`Iteration ${d.iteration} running...`);
  });

  evtSource.addEventListener('complete', e => {
    const d = JSON.parse(e.data);
    evtSource.close();
    addLog('Investigation complete', 'log-done');
    btn.disabled = false; btn.innerHTML = 'Investigate';

    fetch(`/tasks/${task_id}`)
      .then(r => r.json())
      .then(s => {
        document.getElementById('answer').style.display = 'block';
        document.getElementById('answer-text').textContent = s.final_answer || d.final_answer || '(no answer)';
        const score = s.score ?? d.score;
        document.getElementById('meta-score').textContent = score != null ? score.toFixed(2) : '—';
        document.getElementById('meta-iter').textContent = s.iterations ?? '—';
        document.getElementById('meta-tokens').textContent = s.tokens_used != null ? s.tokens_used.toLocaleString() : '—';
        renderSubtasks(s.subtask_results || []);
      });
  });

  evtSource.addEventListener('error', () => {
    addLog('Stream dropped, switching to polling...', 'log-warn');
    evtSource.close();
    pollUntilDone(task_id, btn);
  });
}

function renderSubtasks(results) {
  if (!results || results.length === 0) return;
  const list = document.getElementById('subtasks-list');
  list.innerHTML = '';
  results.forEach(r => {
    const el = document.createElement('div');
    el.className = 'subtask';
    el.innerHTML = `
      <div class="subtask-header">
        <span class="subtask-role">${r.agent}</span>
        <span class="subtask-name">${r.subtask}</span>
      </div>
      <div class="subtask-body">${escHtml(r.result || '')}</div>`;
    list.appendChild(el);
  });
  document.getElementById('subtasks').style.display = 'block';
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function pollUntilDone(task_id, btn) {
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 4000));
    const r = await fetch(`/tasks/${task_id}`);
    const d = await r.json();
    if (d.status === 'complete') {
      addLog('Investigation complete', 'log-done');
      btn.disabled = false; btn.innerHTML = 'Investigate';
      document.getElementById('answer').style.display = 'block';
      document.getElementById('answer-text').textContent = d.final_answer || '(no answer)';
      document.getElementById('meta-score').textContent = d.score?.toFixed(2) ?? '—';
      document.getElementById('meta-iter').textContent = d.iterations ?? '—';
      document.getElementById('meta-tokens').textContent = d.tokens_used?.toLocaleString() ?? '—';
      renderSubtasks(d.subtask_results || []);
      return;
    }
    if (i % 3 === 2) addLog(`Still running... (${(i+1)*4}s elapsed)`);
  }
  addLog('Timed out after 4 minutes', 'log-err');
  btn.disabled = false; btn.innerHTML = 'Investigate';
}

document.getElementById('input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submitTask();
});
</script>
</body>
</html>"""
