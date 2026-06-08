"""
Demo: Forge Incident Root Cause Analyzer

Generates synthetic checkout-service logs with a realistic latency spike,
then submits an incident investigation task to the Forge API and streams
the postmortem result.

Usage:
    python scripts/demo_incident.py [--api http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx


# ---------------------------------------------------------------------------
# Synthetic log generation
# ---------------------------------------------------------------------------

LOG_DIR = Path("/tmp/forge-demo/logs")
REPO_PATH = str(Path(__file__).resolve().parent.parent)

# Incident timeline: normal traffic for 120 min, then N+1 regression kicks in
INCIDENT_START = datetime.now(timezone.utc) - timedelta(minutes=45)
DEPLOY_TIME = INCIDENT_START - timedelta(minutes=30)


def _generate_checkout_log() -> str:
    lines = []
    t = datetime.now(timezone.utc) - timedelta(minutes=120)
    endpoints = ["/api/checkout", "/api/cart", "/api/payment"]

    while t < datetime.now(timezone.utc):
        latency_ms = _sample_latency(t)
        db_queries = _sample_db_queries(t)
        endpoint = random.choice(endpoints)
        status = 200 if latency_ms < 800 else random.choice([200, 504])
        entry = {
            "ts": t.isoformat(),
            "method": "POST",
            "path": endpoint,
            "status": status,
            "latency_ms": latency_ms,
            "db_queries": db_queries,
            "user_id": f"u{random.randint(1000, 9999)}",
        }
        lines.append(json.dumps(entry))
        t += timedelta(seconds=random.uniform(0.5, 3.0))

    return "\n".join(lines)


def _sample_latency(t: datetime) -> int:
    if t >= INCIDENT_START:
        # Post-deploy regression: N+1 query → ~20x DB roundtrip overhead
        base = random.gauss(385, 60)
    else:
        base = random.gauss(45, 8)
    return max(1, int(base))


def _sample_db_queries(t: datetime) -> int:
    if t >= INCIDENT_START:
        # Missing select_related() → one query per related object (avg 23)
        return random.randint(18, 28)
    return random.randint(1, 3)


def _generate_deploy_log() -> str:
    lines = [
        f'{{"ts": "{DEPLOY_TIME.isoformat()}", "event": "deploy_started", "service": "checkout-service", "version": "v2.4.1", "commit": "a3f9c12", "author": "eng-team"}}',
        f'{{"ts": "{(DEPLOY_TIME + timedelta(seconds=45)).isoformat()}", "event": "deploy_completed", "service": "checkout-service", "version": "v2.4.1", "rollout": "100%"}}',
        f'{{"ts": "{(DEPLOY_TIME + timedelta(minutes=2)).isoformat()}", "event": "health_check_passed", "service": "checkout-service"}}',
    ]
    return "\n".join(lines)


def setup_demo_logs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "checkout-service.log").write_text(_generate_checkout_log())
    (LOG_DIR / "deploy.log").write_text(_generate_deploy_log())
    print(f"Demo logs written to {LOG_DIR}/")
    print(f"  Incident start: {INCIDENT_START.isoformat()}")
    print(f"  Deploy time:    {DEPLOY_TIME.isoformat()}")
    print(f"  Repository:     {REPO_PATH}")


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

def submit_incident(api_base: str) -> str:
    task_input = (
        f"P95 latency in checkout-service increased from ~45ms to ~385ms starting at "
        f"{INCIDENT_START.isoformat()}. A deploy (v2.4.1) was completed at "
        f"{DEPLOY_TIME.isoformat()}. "
        f"Logs are at {LOG_DIR}/. "
        f"The repository is at {REPO_PATH}. "
        "Investigate the root cause and draft a complete postmortem."
    )

    payload = {
        "user_input": task_input,
        "user_id": "demo",
        "token_budget": 80000,
        "max_iterations": 2,
    }

    resp = httpx.post(f"{api_base}/tasks", json=payload, timeout=10)
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    print(f"\nTask submitted: {task_id}")
    return task_id


def poll_until_done(api_base: str, task_id: str, timeout_s: int = 300) -> dict:
    deadline = time.time() + timeout_s
    interval = 3
    while time.time() < deadline:
        resp = httpx.get(f"{api_base}/tasks/{task_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "unknown")
        iteration = data.get("iteration_count", 0)
        print(f"  [{status}] iteration={iteration}", end="\r", flush=True)
        if status in ("complete", "failed"):
            print()
            return data
        time.sleep(interval)
    raise TimeoutError(f"Task {task_id} did not complete within {timeout_s}s")


def print_result(result: dict) -> None:
    print("\n" + "=" * 70)
    print("FORGE INCIDENT ANALYSIS — POSTMORTEM")
    print("=" * 70)

    final = result.get("final_answer") or result.get("result", "No output")
    print(final)

    feedback = result.get("critic_feedback") or {}
    if feedback:
        score = feedback.get("score", 0)
        print(f"\n{'=' * 70}")
        print(f"Critic score: {score:.2f}  |  passed: {feedback.get('passed')}")
        print(f"Reasoning: {feedback.get('reasoning', '')}")

    tokens = result.get("total_tokens_used", 0)
    iters = result.get("iteration_count", 0)
    print(f"Tokens used: {tokens:,}  |  Iterations: {iters}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Forge incident analysis demo")
    parser.add_argument("--api", default="http://localhost:8000", help="Forge API base URL")
    args = parser.parse_args()

    print("Setting up synthetic incident logs...")
    setup_demo_logs()

    print(f"\nConnecting to Forge API at {args.api}...")
    try:
        httpx.get(f"{args.api}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach API — {e}")
        print("Start the API first: uvicorn api.azure_main:app --reload")
        return

    task_id = submit_incident(args.api)
    print("Waiting for agents to investigate...")
    result = poll_until_done(args.api, task_id)
    print_result(result)


if __name__ == "__main__":
    main()
