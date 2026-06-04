# Forge

**Multi-agent task orchestration. Give it a goal — a team of AI specialists plans, executes, and retries until the quality passes.**

---

## Why I built this

Every complex task I tried to automate with a single LLM call failed the same way: the model would either skip steps, hallucinate a detail it couldn't verify, or produce code that looked right but didn't run. The problem isn't the model — it's that complex tasks need different skills at different stages, sequentially, with each stage building on the last.

I wanted something where I could say *"find the top 5 cloud providers by revenue, compare their AI services, and write a Python script that fetches their current stock prices"* and have it actually work end-to-end — not with hand-holding, not with retrying manually, but with the system knowing when its own output isn't good enough.

That meant building routing (which specialist handles which step), dependency tracking (step B can't start until step A is verified), quality scoring (is this actually correct?), and automatic retry (if it isn't, re-plan with the feedback). Not a chatbot. Infrastructure.

---

## What it does

You submit a task. Forge breaks it down and runs it through a pipeline:

```
Your goal
    │
    ▼
┌─────────────────┐
│  Orchestrator   │  Plans 1–3 subtasks, assigns each to a specialist
└────────┬────────┘
         │
    ┌────┴─────────────────────┐
    │                          │
    ▼                          ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Researcher│  │  Coder   │  │ Analyst  │
│web search │  │ executes │  │SQL/stats │
│doc lookup │  │  Python  │  │ execute  │
└────┬──────┘  └────┬─────┘  └────┬─────┘
     └──────────────┴──────────────┘
                    │
                    ▼
           ┌────────────────┐
           │     Critic     │  Scores 0–1. Below 0.75? Orchestrator
           │  quality gate  │  re-plans with the feedback.
           └────────┬───────┘
                    │
              passed? ──yes──▶  Final answer
                    │
                   no
                    │
              max iterations? ──yes──▶  Best attempt
                    │
                   no
                    │
              back to Orchestrator
```

Every node writes its state to a checkpoint before moving on. If the process crashes mid-task, it resumes from where it stopped — not from the beginning.

---

## Architecture decisions worth explaining

**Why LangGraph instead of a simple loop?**

A loop would work for the happy path. LangGraph gives me a compiled state machine with typed reducers — the graph topology is explicit, routing is pure functions I can unit test with plain dicts, and checkpointing is built in. When I add a new agent type, I add a node and an edge. I don't touch the routing logic for existing agents.

**Why Azure Service Bus with sessions?**

Sessions enforce FIFO per `session_id`. I use `user_id` as the session key. That means all tasks from one user run in submission order — user A's long-running task doesn't block user B, but user A's tasks always complete in sequence. SQS FIFO queues do the same thing with `MessageGroupId`. The semantics are identical; I'm on Azure.

**Why a critic agent instead of just checking the output in code?**

A rule-based quality check would need to know what "good" looks like for every possible task type. The critic doesn't — it reads the task, reads the outputs, and scores them the same way a human would review work. It also generates the feedback that drives the retry: *"the code runs but doesn't handle edge cases"* is more useful than *"score: 0.4"*.

**Why not just use Claude / GPT-4 and skip the complexity?**

The whole system runs on open weights models via Ollama (local) or Groq (cloud free tier). No per-token cost at inference time, no vendor lock-in on the model layer. The orchestration patterns — checkpointing, retry loops, quality gates, message queues — work identically regardless of which model is behind `get_llm()`. Swapping providers is one env var.

**Why does the lock renewal loop exist?**

Service Bus message locks expire after 60 seconds by default. Agent tasks can take 10+ minutes. Without the heartbeat, the lock expires mid-task, the message becomes available again, a second worker picks it up, and you get duplicate processing. The lock renewal loop wakes every 50 seconds and extends the lease. It's the Azure equivalent of SQS's `change_message_visibility`.

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| Agent graph | LangGraph 0.6 | State machines, typed reducers, built-in checkpointing |
| LLM (cloud) | Groq + Llama 3.3 70B | 300+ tokens/sec, free tier, tool calling that works |
| LLM (local) | Ollama + qwen2.5:3b | No API key needed for dev, full offline capability |
| Checkpoint store | PostgreSQL (AsyncPostgresSaver) | Crash recovery, task history, resume from any node |
| Task queue | Azure Service Bus (sessions) | FIFO per user, at-least-once delivery, dead-letter on failure |
| API | FastAPI + SSE | Real-time agent progress without websocket complexity |
| Infra | Terraform + Azure Container Apps | Serverless scaling, managed identity (no credential files) |

---

## What it actually looks like

The browser UI at `localhost:8000` shows agent steps as they happen via Server-Sent Events:

- Submit a task
- Watch the orchestrator plan subtasks in real time
- See each specialist complete its work
- Critic score + final answer when done

The "Agent outputs" section shows what each agent actually produced — not a summary, the raw output. If the coder wrote Python, you see the code. If the researcher found data, you see the data.

---

## Quick start

**With Groq (cloud, fastest):**
```bash
git clone https://github.com/AkshayShah03/forge
cd forge
uv venv .venv --python 3.11 && uv pip install -r requirements-azure.txt
GROQ_API_KEY=your_key PYTHONPATH=. .venv/bin/uvicorn api.azure_main:app --reload
# open http://localhost:8000
```

**With Ollama (local, no API key):**
```bash
ollama pull qwen2.5:3b
OLLAMA_MODEL=qwen2.5:3b PYTHONPATH=. .venv/bin/uvicorn api.azure_main:app --reload
```

---

## Running the tests

```bash
pytest tests/ -v --tb=short   # 45 tests, ~4 seconds
python scripts/smoke_test.py  # end-to-end with mocked LLM
```

The routing functions are pure — they take a state dict and return a string. No mocks needed, no running LLM. The integration tests use a `FakeChatModel` that returns preset responses so the full graph runs without hitting any API.

---

## Production deployment

Terraform provisions everything on Azure:

```bash
cd infra/azure/terraform
terraform init && terraform apply
```

This creates: Container Apps (API + worker), Service Bus Premium namespace with sessions, PostgreSQL Flexible Server, Key Vault for secrets. GitHub Actions deploys on push to `main`.

The worker scales horizontally — each replica accepts a different user session, so concurrent users don't block each other. The API is stateless; it just enqueues to Service Bus and reads checkpoints from Postgres.

Cost on Azure: ~$700/month for one messaging unit (Service Bus Premium requires it for sessions) + Container Apps + Postgres. Not for hobby use.

---

## Honest limitations

- **No real web search without a Tavily key** — the `web_search` tool returns an error if `TAVILY_API_KEY` isn't set. Agents fall back to model knowledge, which works for well-known facts and fails on recent events.
- **3B models struggle with tool calling** — `qwen2.5:3b` works for straightforward tasks; anything with complex tool use needs at least 7B. `llama-3.1-8b-instant` on Groq hallucinates tool names. `llama-3.3-70b-versatile` is reliable.
- **Critic adds ~30% latency** — it's an extra LLM call per iteration. For tasks where you trust the first answer, set `max_iterations=1`.
- **MemorySaver is not durable** — the default (no `POSTGRES_URL`) keeps checkpoints in RAM. Restart the server and task history is gone.

---

## What I'd build next

- **Streaming agent outputs to the UI in real time** — right now you see the log steps but the actual text appears only when the subtask finishes. SSE from within the ReAct loop would show token-by-token output.
- **Tool result caching** — if the researcher looks up the same fact twice across retries, it shouldn't hit the API twice.
- **A proper task queue UI** — the current UI is for submitting and watching one task. A dashboard showing all tasks, their statuses, and the ability to kill a runaway task.

---

## Project structure

```
forge/
├── agent_system/
│   ├── agents/
│   │   ├── orchestrator.py   # plans subtasks, routes to specialists
│   │   ├── sub_agents.py     # researcher, coder, analyst
│   │   └── critic.py         # quality gate, drives retry
│   ├── graph/
│   │   └── builder.py        # StateGraph assembly, AgentGraphFactory
│   ├── state/
│   │   └── schema.py         # AgentState TypedDict, reducers
│   ├── tools/
│   │   └── registry.py       # tool scopes per agent role
│   └── llm.py                # Groq / Ollama factory
├── api/
│   └── azure_main.py         # FastAPI: /tasks, /stream, browser UI
├── worker/
│   └── servicebus_worker.py  # Service Bus consumer, lock renewal
├── tests/
│   ├── unit/                 # routing logic, state schema, worker parsing
│   └── integration/          # API endpoints, full graph e2e
├── scripts/
│   ├── init_db.py            # create Postgres schema
│   └── smoke_test.py         # mocked end-to-end run
└── infra/azure/terraform/    # Container Apps, Service Bus, Postgres
```
