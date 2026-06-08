# Forge

**An incident response system powered by a team of AI agents. Give it an alert and a log path. It investigates, finds the root cause, and drafts the postmortem.**

---

## What it actually does

When a production alert fires, the usual process is: someone wakes up, SSHes into a box, scrolls through logs, checks recent deploys, forms a hypothesis, validates it, then spends another hour writing a postmortem. That process is mostly mechanical. The interesting part is the judgment call at the end.

Forge handles the mechanical part. You describe the incident (what broke, when, where the logs are, what service is affected) and the system spins up three agents that run in parallel:

* A **researcher** searches for known failure patterns and runbook guidance for the alert type
* A **coder** reads the log files, computes latency percentiles, counts DB queries per request, and queries the git history for deploys that correlate with the anomaly onset
* An **analyst** receives both sets of findings, ranks root cause hypotheses by confidence, and drafts a structured postmortem

The postmortem goes through a critic agent that scores it against a rubric: is the root cause specific (a commit hash, a file, a query)? Is the evidence cited (actual numbers from the logs)? If the score is below the threshold, the orchestrator re-plans with the feedback and runs again.

The output is a postmortem you can actually use, not a summary that says "increased latency was observed."

---

## Architecture

```
Incident description (what broke, log path, repo path)
    |
    v
Orchestrator  —  plans the investigation into 3 subtasks
    |
    +——— Researcher  (web search for known causes + runbook patterns)
    |
    +——— Coder  (reads logs, parses latency stats, queries git history)
    |
    v  (analyst waits for both above to finish)
    Analyst  (correlates findings, ranks hypotheses, writes postmortem)
    |
    v
Critic  (scores on specificity + evidence + completeness)
    |
    +——— score < 0.75: re-plan and retry (up to 3 iterations)
    |
    +——— score >= 0.75: return final postmortem
```

Every agent node writes its state to a checkpoint before it hands off to the next. If the process crashes mid-investigation, it resumes from where it stopped.

---

## Quick start

You need either a Groq API key (free at console.groq.com) or Ollama running locally. No other external services are required.

**With Groq:**

```bash
git clone https://github.com/AkshayShah03/forge
cd forge
uv venv .venv --python 3.11 && uv pip install -r requirements-azure.txt
GROQ_API_KEY=your_key PYTHONPATH=. .venv/bin/uvicorn api.azure_main:app --reload
```

Open `http://localhost:8000`. The UI has pre-filled example incidents to try immediately.

**With Ollama (no API key needed):**

```bash
ollama pull qwen2.5:3b
PYTHONPATH=. .venv/bin/uvicorn api.azure_main:app --reload
```

Note: smaller models (3B) work for straightforward incidents but struggle with multi-step tool use. `llama3.1:8b` or larger gives much better results.

---

## How to use it

The system works best when you give it specifics. Instead of "our API is slow," try:

```
P95 latency in checkout-service jumped from 45ms to 385ms at 2026-06-08T14:32Z.
A deploy (v2.4.1) finished at 14:02. Logs are at /var/log/checkout/.
The repo is at /home/app/checkout-service. Investigate and write a postmortem.
```

The coder agent will find the log files, compute the latency stats, and check what changed in the repo around that time. The analyst matches the anomaly timestamp to the deploy window and tries to identify the specific commit or code change responsible.

You can also run the pre-built demo that generates synthetic logs and submits a realistic incident:

```bash
python scripts/demo_incident.py
```

This creates log files in `/tmp/forge-demo/logs/` with a simulated N+1 query regression (DB queries per request jumping from 2 to 23 at a known timestamp) and an accompanying deploy log. The system investigates it without any further setup.

---

## Tech decisions

**LangGraph instead of a loop.** A plain async loop would handle the happy path fine. LangGraph gives compiled state machines with typed reducers, which means routing functions are pure and testable with plain dicts. When the critic fails a result and routes back to the orchestrator, that path is explicit in the graph, not an `if/else` buried in a loop body.

**Dependency tracking.** The analyst depends on both the researcher and the coder. The orchestrator tracks this by checking which subtasks are in `subtask_results` before it routes to the next agent. The researcher and coder run sequentially in the current implementation (parallel execution is on the list), but the dependency contract means they can be parallelized without changing the analyst or critic code.

**Critic as quality gate.** A rule-based check can tell you whether the output is well-formed JSON. It cannot tell you whether "the root cause was high traffic" is a useful postmortem or not. The critic reads the original incident description and the full agent output and scores it the same way a senior engineer would review a draft. The feedback it generates ("root cause too vague, no commit hash cited") drives the retry in a way that a numeric score alone cannot.

**Azure Service Bus with sessions.** FIFO per `user_id`. If you submit multiple incidents, they run in submission order without blocking other users. The worker renews the Service Bus message lock every 50 seconds because agent runs take longer than the default 60-second lock timeout. Without this, the message re-enters the queue and a second worker picks it up mid-run.

---

## Tools available to each agent

| Agent | Tools |
|---|---|
| Orchestrator | web search |
| Researcher | web search, RAG retrieval, file read |
| Coder | Python execution, file read/write, log file reader, directory listing, git log query |
| Analyst | SQL query, Python execution, RAG retrieval, log file reader |
| Critic | none (pure LLM evaluation) |

The tool scopes are enforced in `agent_system/tools/registry.py`. Adding a new tool means adding a function and listing it in the scope for the roles that should have it.

---

## Running the tests

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```

93 tests, all under a second. Routing functions are pure so no mocks needed for unit tests. Integration tests use a `FakeChatModel` with preset responses, so the full graph runs without hitting any API.

The test suite covers:

* All routing logic (orchestrator, subagents, critic, retry loop)
* The three new tools (`read_log_file`, `list_log_files`, `query_git_log`) including edge cases like nonexistent paths and zero-hour git windows
* JSON parsing resilience for orchestrator and critic outputs (markdown fences, embedded JSON, malformed responses)
* The full 3-agent incident workflow end to end
* Dependency blocking and unblocking (analyst waits for both researcher and coder)
* API endpoints (submit, poll, SSE stream, history)
* Service Bus worker (message parsing, lock renewal, dead-letter on max delivery count)

---

## Limitations

The web search tool requires a Tavily API key. Without one, the researcher falls back to the model's training knowledge, which works well for common failure patterns (N+1 queries, GC pressure, lock contention) but not for service-specific or recent issues.

The critic adds roughly one extra LLM call per iteration. For incidents where you trust the first pass, set `max_iterations=1`.

By default (no `POSTGRES_URL`), the system uses in-memory checkpointing. Task history is lost on restart. Set `POSTGRES_URL` to a PostgreSQL instance to persist across restarts.

The `query_git_log` tool reads from a local git repository. If the service repo is on a remote or only accessible via an API, you would need to add a tool that fetches commit history from GitHub or GitLab instead.

---

## Production deployment

Terraform provisions everything on Azure:

```bash
cd infra/azure/terraform
terraform init && terraform apply
```

This creates Container Apps for the API and worker, a Service Bus Premium namespace (required for session support), PostgreSQL Flexible Server for checkpoints, and Key Vault for secrets. GitHub Actions deploys on push to main.

The worker scales horizontally. Each replica takes a different user session so concurrent incidents do not block each other. Cost is around $700/month primarily because Service Bus Premium requires one messaging unit to enable sessions.

---

## Project structure

```
forge/
├── agent_system/
│   ├── agents/
│   │   ├── orchestrator.py     plans subtasks, tracks dependencies
│   │   ├── sub_agents.py       researcher, coder, analyst with role-specific prompts
│   │   └── critic.py           quality scoring, drives retry
│   ├── graph/
│   │   └── builder.py          LangGraph StateGraph, AgentGraphFactory
│   ├── state/
│   │   └── schema.py           AgentState TypedDict, initial_state factory
│   ├── tools/
│   │   └── registry.py         tool definitions, scopes per agent role
│   └── llm.py                  Groq / Ollama factory
├── api/
│   └── azure_main.py           FastAPI endpoints + browser UI
├── worker/
│   └── servicebus_worker.py    Service Bus consumer, lock renewal, dead-letter
├── tests/
│   ├── unit/                   routing, state schema, tools, parsing, worker
│   └── integration/            API endpoints, full graph end to end
├── scripts/
│   ├── demo_incident.py        generates synthetic logs and runs a full investigation
│   ├── init_db.py              creates the Postgres checkpoint schema
│   └── smoke_test.py           mocked end-to-end graph run
└── infra/azure/terraform/      Container Apps, Service Bus, Postgres, Key Vault
```
