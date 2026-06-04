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
    credential = DefaultAzureCredential()
    sb_client  = ServiceBusClient(
        fully_qualified_namespace=SERVICE_BUS_NAMESPACE,
        credential=credential,
    )
    graph_factory = AgentGraphFactory()
    await graph_factory.__aenter__()

    app.state.sb_client    = sb_client
    app.state.credential   = credential
    app.state.factory      = graph_factory
    logger.info("API started — Service Bus namespace: %s", SERVICE_BUS_NAMESPACE)
    yield

    await graph_factory.__aexit__(None, None, None)
    await sb_client.close()
    await credential.close()


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
        asyncio.create_task(factory.run_task(state))

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
    """Simple browser UI for submitting tasks and watching live agent progress."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Agent System</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 2rem; }
  h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; color: #f8fafc; }
  .subtitle { color: #64748b; font-size: 0.875rem; margin-bottom: 2rem; }
  .card { background: #1e2230; border: 1px solid #2d3348; border-radius: 0.75rem;
          padding: 1.5rem; margin-bottom: 1.25rem; }
  textarea { width: 100%; background: #0f1117; border: 1px solid #2d3348;
             border-radius: 0.5rem; color: #e2e8f0; font-size: 1rem;
             padding: 0.75rem; resize: vertical; min-height: 80px;
             outline: none; transition: border-color .2s; }
  textarea:focus { border-color: #6366f1; }
  .row { display: flex; gap: 0.75rem; align-items: center; margin-top: 0.75rem; }
  label { color: #94a3b8; font-size: 0.85rem; white-space: nowrap; }
  select { background: #0f1117; border: 1px solid #2d3348; border-radius: 0.5rem;
           color: #e2e8f0; padding: 0.4rem 0.6rem; font-size: 0.85rem; }
  button { background: #6366f1; color: #fff; border: none; border-radius: 0.5rem;
           padding: 0.55rem 1.4rem; font-size: 0.95rem; font-weight: 500;
           cursor: pointer; transition: background .15s; margin-left: auto; }
  button:hover:not(:disabled) { background: #4f46e5; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  #log { font-family: "SF Mono", "Fira Code", monospace; font-size: 0.8rem;
         line-height: 1.7; max-height: 320px; overflow-y: auto;
         background: #0a0c12; border-radius: 0.5rem; padding: 1rem;
         border: 1px solid #1a1f2e; display: none; }
  .log-line { display: flex; gap: 0.5rem; }
  .log-time { color: #475569; flex-shrink: 0; }
  .log-info  { color: #38bdf8; }
  .log-done  { color: #34d399; }
  .log-warn  { color: #fb923c; }
  .log-err   { color: #f87171; }
  #answer { display: none; }
  #answer h2 { font-size: 1rem; color: #94a3b8; margin-bottom: 0.6rem; }
  #answer-text { color: #f1f5f9; line-height: 1.7; white-space: pre-wrap; font-family: inherit; }
  .meta { display: flex; gap: 1.5rem; margin-top: 1rem; font-size: 0.8rem; }
  .meta span { color: #64748b; }
  #subtasks { display: none; margin-top: 1.25rem; }
  .subtask { border: 1px solid #2d3348; border-radius: 0.5rem; margin-bottom: 0.75rem; overflow: hidden; }
  .subtask-header { background: #1a1f2e; padding: 0.5rem 0.85rem;
                    font-size: 0.8rem; display: flex; gap: 0.75rem; align-items: center; }
  .subtask-role { color: #7dd3fc; font-weight: 600; text-transform: uppercase;
                  font-size: 0.7rem; letter-spacing: .05em; }
  .subtask-name { color: #94a3b8; flex: 1; }
  .subtask-body { padding: 0.75rem 0.85rem; font-size: 0.82rem; line-height: 1.65;
                  white-space: pre-wrap; color: #cbd5e1;
                  font-family: "SF Mono","Fira Code",monospace;
                  max-height: 300px; overflow-y: auto; background: #0a0c12; }
  .meta b { color: #94a3b8; }
  .badge { display: inline-flex; align-items: center; gap: 0.35rem;
           background: #1a2332; border: 1px solid #2d3f55;
           border-radius: 9999px; padding: 0.2rem 0.65rem;
           font-size: 0.75rem; color: #7dd3fc; }
  .spinner { display: inline-block; width: 10px; height: 10px;
             border: 2px solid #334155; border-top-color: #6366f1;
             border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  a.docs-link { color: #6366f1; font-size: 0.8rem; text-decoration: none; }
  a.docs-link:hover { text-decoration: underline; }
</style>
</head>
<body>
<h1>Multi-Agent System</h1>
<p class="subtitle">Powered by LangGraph + Ollama &nbsp;·&nbsp; <a class="docs-link" href="/docs" target="_blank">API docs →</a></p>

<div class="card">
  <textarea id="input" placeholder="Ask anything… e.g. 'What are the 5 largest cities in Europe by population?'"></textarea>
  <div class="row">
    <label>Iterations</label>
    <select id="max-iter">
      <option value="1">1</option>
      <option value="2" selected>2</option>
      <option value="3">3</option>
    </select>
    <label>Budget</label>
    <select id="budget">
      <option value="20000">20k tokens</option>
      <option value="50000" selected>50k tokens</option>
      <option value="100000">100k tokens</option>
    </select>
    <button id="submit-btn" onclick="submitTask()">Run</button>
  </div>
</div>

<div class="card" id="answer">
  <h2>Answer</h2>
  <div id="answer-text"></div>
  <div class="meta">
    <span><b>Score</b> <span id="meta-score">—</span></span>
    <span><b>Iterations</b> <span id="meta-iter">—</span></span>
    <span><b>Tokens</b> <span id="meta-tokens">—</span></span>
  </div>
</div>

<div id="subtasks">
  <div style="color:#64748b;font-size:0.8rem;margin-bottom:0.5rem;padding-left:0.25rem;">Agent outputs</div>
  <div id="subtasks-list"></div>
</div>

<div class="card" style="padding:0.75rem 1rem;">
  <div id="log"></div>
  <div id="log-empty" style="color:#475569;font-size:0.85rem;">Agent steps will appear here as they run.</div>
</div>

<script>
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

  // Reset UI
  document.getElementById('log').innerHTML = '';
  document.getElementById('log').style.display = 'none';
  document.getElementById('log-empty').style.display = '';
  document.getElementById('answer').style.display = 'none';
  document.getElementById('subtasks').style.display = 'none';
  document.getElementById('subtasks-list').innerHTML = '';
  if (evtSource) { evtSource.close(); evtSource = null; }

  addLog('Submitting task…');

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
    addLog('Submit failed: ' + resp.status, 'log-err');
    btn.disabled = false; btn.innerHTML = 'Run';
    return;
  }

  const {task_id} = await resp.json();
  addLog(`Task <span class="badge">${task_id.slice(0,8)}…</span> queued — watching stream…`);

  evtSource = new EventSource(`/tasks/${task_id}/stream`);

  evtSource.addEventListener('update', e => {
    const d = JSON.parse(e.data);
    addLog(`Iteration ${d.iteration} — running agents…`);
  });

  evtSource.addEventListener('complete', e => {
    const d = JSON.parse(e.data);
    evtSource.close();
    addLog('Done ✓', 'log-done');
    btn.disabled = false; btn.innerHTML = 'Run';

    // Fetch full status (includes subtask_results)
    fetch(`/tasks/${task_id}`)
      .then(r => r.json())
      .then(s => {
        document.getElementById('answer').style.display = 'block';
        document.getElementById('answer-text').textContent = s.final_answer || d.final_answer || '(no answer)';
        document.getElementById('meta-score').textContent = (s.score ?? d.score) != null ? (s.score ?? d.score).toFixed(2) : '—';
        document.getElementById('meta-iter').textContent = s.iterations ?? '—';
        document.getElementById('meta-tokens').textContent = s.tokens_used ?? '—';
        renderSubtasks(s.subtask_results || []);
      });
  });

  evtSource.addEventListener('error', e => {
    addLog('Stream error — polling for result…', 'log-warn');
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
      addLog('Done ✓', 'log-done');
      btn.disabled = false; btn.innerHTML = 'Run';
      document.getElementById('answer').style.display = 'block';
      document.getElementById('answer-text').textContent = d.final_answer || '(no answer)';
      document.getElementById('meta-score').textContent = d.score?.toFixed(2) ?? '—';
      document.getElementById('meta-iter').textContent = d.iterations ?? '—';
      document.getElementById('meta-tokens').textContent = d.tokens_used ?? '—';
      renderSubtasks(d.subtask_results || []);
      return;
    }
    if (i % 3 === 2) addLog(`Still running… (${(i+1)*4}s)`);
  }
  addLog('Timed out after 4 minutes', 'log-err');
  btn.disabled = false; btn.innerHTML = 'Run';
}

document.getElementById('input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submitTask();
});
</script>
</body>
</html>"""
