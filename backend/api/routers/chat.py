"""
POST /chat  — main RAG endpoint.
GET  /chat/history/{session_id}  — conversation history.

Pipeline (each step is a stub until the downstream service is wired in):
    1. Validate & attach user_id / session_id          ← done here
    2. Guardrail inbound check                         ← stub
    3. Route query (legal / trend / out_of_scope)      ← stub
    4. Retrieve context or call FRED tool              ← stub
    5. Assemble prompt + call LLM                      ← stub
    6. Guardrail outbound check                        ← stub
    7. Persist turn to SQLite                          ← stub
    8. Return ChatResponse (SSE streaming optional)    ← returns JSON for now
"""
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.core.config import Settings, get_settings
from guardrails.rules import check_inbound, check_outbound
from guardrails.models import GuardrailResult
from api.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    Citation,
    GuardrailVerdict,
    MessageRecord,
    SessionCreateRequest,
    SessionCreateResponse,
)

import yaml
from pathlib import Path as _Path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# Load human-readable block messages from policy.yaml once at import time.
_POLICY_PATH = _Path(__file__).resolve().parents[2] / "guardrails" / "policy.yaml"
with _POLICY_PATH.open() as _f:
    _POLICY = yaml.safe_load(_f)
_BLOCK_MESSAGES: dict[str, str] = _POLICY.get("messages", {})


# ---------------------------------------------------------------------------
# Helper — build a correlation / request ID
# ---------------------------------------------------------------------------

def _request_id(request: Request) -> str:
    """Return X-Request-ID header if present, else generate one."""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ChatResponse,
    summary="Submit a user query and receive a grounded RAG answer",
    status_code=status.HTTP_200_OK,
)
async def chat(
    body: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """
    Accepts a user query together with user_id and session_id.
    Returns a structured response that includes the answer, source citations,
    guardrail verdict, query route, and token usage.

    Downstream services (guardrails, retrieval, LLM, memory) are stubbed and
    will be replaced incrementally as each layer is built.
    """
    req_id = _request_id(request)
    logger.info("chat request user=%s session=%s req_id=%s query=%r",
                body.user_id, body.session_id, req_id, body.query[:120])

    # ------------------------------------------------------------------
    # Step 1 — basic validation already done by Pydantic.
    # Attach correlation ID to response headers (done in middleware).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 2 — INBOUND guardrail checks
    #   • Prompt injection detection
    #   • PII detection
    #   • Domain classifier (out-of-scope)
    # ------------------------------------------------------------------
    inbound: GuardrailResult = check_inbound(body.query)

    if inbound.blocked:
        rule = inbound.rule or "unknown"
        # Map rule name to a friendly message; fall back to the raw reason.
        friendly_msg = _BLOCK_MESSAGES.get(
            "out_of_scope" if rule == "domain_classifier" else rule,
            inbound.reason or "Request blocked by content policy.",
        )
        # Domain out-of-scope → polite refusal as a normal response (not an HTTP error)
        if rule == "domain_classifier":
            return ChatResponse(
                user_id=body.user_id,
                session_id=body.session_id,
                answer=friendly_msg,
                citations=[],
                guardrail=GuardrailVerdict(
                    action="block",
                    reason=inbound.reason,
                    violations=inbound.violations,
                ),
                route="out_of_scope",
                token_usage={},
            )
        # PII / prompt-injection → HTTP 400
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": friendly_msg,
                "guardrail": {
                    "rule": rule,
                    "reason": inbound.reason,
                    "violations": inbound.violations,
                },
            },
        )

    # ------------------------------------------------------------------
    # Steps 3–5 — Assemble memory context, retrieve, ground, invoke LLM
    # ------------------------------------------------------------------
    from memory.memory_service import MemoryService
    from memory.token_budget import get_context_window
    from orchestration.chains import OrchestrationService

    memory = MemoryService()

    # Long-term memory: user intent/profile persisted across sessions
    user_intent = memory.get_user_intent(body.user_id)

    # Short-term memory: token-budget-aware context window
    # Returns (summary_of_older_turns, last_N_recent_turns)
    summary, recent_turns = get_context_window(
        body.session_id,
        T=settings.token_budget_t,
        N=settings.token_budget_n,
        settings=settings,
    )
    history_payload = [
        {"role": t["role"], "content": t["message"]}
        for t in recent_turns
    ]

    orchestrator = OrchestrationService(settings=settings)
    grounded_answer = await orchestrator.generate_grounded_answer(
        question=body.query,
        history=history_payload,
        top_k=settings.rag_top_k,
        user_intent=user_intent if user_intent else None,
        summary=summary if summary else None,
    )
    answer = grounded_answer.answer
    token_usage = grounded_answer.token_usage
    citations = [
        Citation(
            source_doc=citation.source_doc,
            law=citation.law,
            section=citation.section,
            chunk_id=citation.chunk_id,
        )
        for citation in grounded_answer.context.citations
    ]

    # ------------------------------------------------------------------
    # Step 6 — OUTBOUND guardrail check (PII scan on LLM answer)
    # ------------------------------------------------------------------
    outbound: GuardrailResult = check_outbound(answer)
    if outbound.blocked:
        answer = _BLOCK_MESSAGES.get(
            "outbound_pii",
            "The response contained sensitive information and has been withheld.",
        )
    outbound_verdict = GuardrailVerdict(
        action=outbound.action,
        reason=outbound.reason,
        citations_present=bool(citations),
        violations=outbound.violations,
    )

    # ------------------------------------------------------------------
    # Step 7 — Persist turn to SQLite via MemoryService
    # ------------------------------------------------------------------
    memory.save_message(session_id=body.session_id, role=body.role, message=body.query, user_id=body.user_id)
    memory.save_message(session_id=body.session_id, role="assistant", message=answer, user_id=body.user_id)

    # ------------------------------------------------------------------
    # Step 7b — Extract user intent (async, fire-and-forget)
    # Analyses the user's query to build long-term memory profile.
    # Runs in background — never blocks the response.
    # ------------------------------------------------------------------
    import asyncio
    from memory.intent_extractor import extract_and_update_intent

    asyncio.create_task(
        extract_and_update_intent(
            user_id=body.user_id,
            user_query=body.query,
            settings=settings,
        )
    )

    logger.info("chat response user=%s session=%s req_id=%s",
                body.user_id, body.session_id, req_id)

    return ChatResponse(
        user_id=body.user_id,
        session_id=body.session_id,
        answer=answer,
        citations=citations,
        guardrail=outbound_verdict,
        route="legal",  # single pipeline for now; extend when retrieval routing is wired
        token_usage=token_usage,
    )


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------

@router.post(
    "/sessions",
    response_model=SessionCreateResponse,
    summary="Create a new chat session",
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    body: SessionCreateRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> SessionCreateResponse:
    """
    Creates a new chat session. The session_id is generated server-side by
    MemoryService and returned in the response — the UI does not need to supply one.
    """
    req_id = _request_id(request)

    from memory.memory_service import MemoryService
    memory = MemoryService()
    session_id = memory.create_session(title=body.title)

    logger.info("create_session req_id=%s session_id=%s title=%r",
                req_id, session_id, body.title)

    return SessionCreateResponse(
        session_id=session_id,
        title=body.title,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# GET /chat/history/{session_id}
# ---------------------------------------------------------------------------

@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Retrieve conversation history for a session",
)
async def chat_history(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> ChatHistoryResponse:
    """
    Returns all messages for the given session_id in chronological order.
    Stub — will query the SQLite messages table once the memory layer is built.
    """
    logger.info("history request session=%s", session_id)

    from memory.memory_service import MemoryService

    memory = MemoryService()
    rows = memory.get_history(session_id)
    messages = [
        MessageRecord(
            role=row.role if row.role in {"user", "assistant"} else "user",
            content=row.message,
            created_at=row.created_at,
        )
        for row in rows
    ]

    return ChatHistoryResponse(session_id=session_id, messages=messages)
