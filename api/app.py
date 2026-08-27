"""The public API.

Three endpoints and one rule: **nothing that costs money runs before the
protection layer says it may.**

`/api/scenarios` and `/api/scenario/{key}` serve pre-computed runs and never
touch a model, so they are exempt from every limit. `/api/ask` is the only path
that can spend, and it is gated.

**The graph is built once at startup, not per request.** Building it per request
would reconstruct the Qdrant client and the model client on every call, which is
latency for nothing and a connection leak under load.

**Tracing happens here**, at the request boundary, because one request is one
trace. See `agent/tracing.py` for why it is not inside `build_agent`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent.build import build_agent, build_baseline
from agent.models import StructuredCaller, Usage
from agent.state import initial_state
from agent.tracing import configured as tracing_configured
from agent.tracing import run_traced
from api import precomputed
from api.limits import Protection
from domain import EmployeeContext
from eval.run_precedence import ADOPTED_MODEL, ADOPTED_STRATEGY
from eval.run_retrieval import CORPUS_SNAPSHOT
from retrieval.embed import get_provider
from retrieval.store import ChunkStore

router = APIRouter(prefix="/api")


class AskRequest(BaseModel):
    question: str
    session_id: str = Field(default="anonymous", max_length=64)
    employee_context: EmployeeContext = Field(default_factory=EmployeeContext)
    as_of: date | None = None
    # Phase 9's argument in one flag: the same question through the same graph
    # with precedence swapped for "trust the top-ranked passage".
    baseline: bool = False


def client_ip(request: Request, forwarded: str | None) -> str:
    """Behind Fly.io the socket peer is the proxy, so every caller would share
    one bucket and the per-IP limit would be a global one under another name.

    Only the FIRST entry of X-Forwarded-For is used. Later entries are attacker
    controlled: appending a fake hop is how a client gives itself a fresh bucket.
    """
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def build_state(payload: AskRequest) -> dict[str, Any]:
    return initial_state(
        payload.question.strip(),
        payload.employee_context,
        payload.as_of or CORPUS_SNAPSHOT,
    )


def serialise(state: dict[str, Any], usage: Usage, trace_url: str | None) -> dict:
    resolution = state.get("resolution")
    verification = state.get("verification")
    return {
        "route": state.get("route"),
        "answer": state.get("answer"),
        "citations": state.get("citations", []),
        "controlling_authority": resolution.controlling if resolution else None,
        "defensible_authorities": list(resolution.defensible) if resolution else [],
        "precedence_rule": resolution.rule if resolution else None,
        "verification": {
            "passed": bool(verification and verification.passed),
            "checks": dict(verification.checks) if verification else {},
            "failures": list(verification.failures) if verification else [],
        },
        "trace": [
            {"node": e.node, "summary": e.summary, "detail": e.detail}
            for e in state.get("trace", [])
        ],
        "precomputed": False,
        "cost_usd": round(usage.usd, 6),
        "tokens": {"input": usage.input_tokens, "output": usage.output_tokens},
        "trace_url": trace_url,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the graph once, at startup.

    Per-request construction would rebuild the Qdrant and model clients on every
    call: latency for nothing and a connection leak under load.

    `on_event` is deprecated in this FastAPI version. Two unbounded dependency
    floors have already been silently wrong in this project (langgraph and
    langfuse), so a deprecation warning gets acted on rather than tolerated.
    """
    if not app.state.agents:
        store = ChunkStore(get_provider(ADOPTED_MODEL), strategy=ADOPTED_STRATEGY)
        caller = StructuredCaller()
        app.state.agents = {
            "agent": build_agent(store, caller),
            "baseline": build_baseline(store, caller),
        }
        app.state.caller = caller
    yield


def create_app(protection: Protection | None = None) -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
        title="Controlling Authority",
        description=(
            "Determines which authority controls an employee leave question, "
            "then answers from it with citations."
        ),
    )
    app.state.protection = protection or Protection()
    app.state.agents = {}
    app.include_router(router)
    return app


@router.get("/health")
def health(request: Request) -> dict:
    protection: Protection = request.app.state.protection
    return {
        "status": "ok",
        "limits": protection.snapshot(),
        "precomputed_available": precomputed.available(),
        "precomputed_stale": precomputed.stale(
            precomputed.current_provenance(CORPUS_SNAPSHOT)
        ),
        "tracing": tracing_configured(),
    }


@router.get("/scenarios")
def scenarios() -> dict:
    """The curated set. Free, unlimited, and the path most reviewers take."""
    records = [precomputed.load(k) for k in precomputed.available()]
    return {
        "scenarios": [
            {
                "key": r.key,
                "scenario_id": r.scenario_id,
                "question": r.question,
                "as_of": r.as_of,
                "employee_context": r.employee_context,
            }
            for r in records
            if r
        ]
    }


@router.get("/scenario/{key}")
def scenario(key: str) -> JSONResponse:
    record = precomputed.load(key)
    if record is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"no pre-computed scenario named {key!r}"},
        )
    # No limit check and no budget charged: this costs nothing to serve, so
    # rate-limiting it would degrade the common path to protect a budget it
    # never touches.
    return JSONResponse(content=record.payload)


@router.post("/ask")
def ask(
    payload: AskRequest,
    request: Request,
    x_forwarded_for: str | None = Header(default=None),
) -> JSONResponse:
    protection: Protection = request.app.state.protection
    ip = client_ip(request, x_forwarded_for)
    question = payload.question.strip()

    if not question:
        return JSONResponse(
            status_code=400, content={"error": "question must not be empty"}
        )

    verdict = protection.check(ip, payload.session_id, question)
    if not verdict.allowed:
        headers = (
            {"Retry-After": str(verdict.retry_after_seconds)}
            if verdict.retry_after_seconds
            else None
        )
        return JSONResponse(
            status_code=429 if verdict.limit_hit != "input_length" else 413,
            headers=headers,
            content={
                "error": verdict.reason,
                "limit": verdict.limit_hit,
                "retry_after_seconds": verdict.retry_after_seconds,
                "precomputed_scenarios_are_always_available": True,
            },
        )

    graph = request.app.state.agents["baseline" if payload.baseline else "agent"]
    usage = Usage()
    caller = request.app.state.caller
    before = (caller.usage.input_tokens, caller.usage.output_tokens, caller.usage.calls)

    final, trace_url = run_traced(
        graph, build_state(payload), session_id=payload.session_id
    )

    # Budget is charged only now, after the work actually ran. Charging on check
    # would drift the global cap down against requests that failed downstream.
    protection.record(ip, payload.session_id)

    usage.input_tokens = caller.usage.input_tokens - before[0]
    usage.output_tokens = caller.usage.output_tokens - before[1]
    usage.calls = caller.usage.calls - before[2]
    usage.by_model = dict(caller.usage.by_model)

    return JSONResponse(content=serialise(final, usage, trace_url))


app = create_app()
