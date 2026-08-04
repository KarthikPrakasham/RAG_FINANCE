"""
Guardrail orchestrator.

check_inbound(query)   — run all inbound checks (PII, prompt injection, domain)
check_outbound(answer) — run outbound checks (PII in answer)

Each function returns a single ``GuardrailResult``.  The first check that
produces ``action='block'`` short-circuits; remaining checks are skipped.
"""
from __future__ import annotations

import logging

from .models import GuardrailResult
from .pii_detector import detect_pii
from .prompt_injection import detect_prompt_injection
from .domain_classifier import classify_domain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inbound pipeline (runs on the user query before the LLM is called)
# ---------------------------------------------------------------------------

def check_inbound(query: str) -> GuardrailResult:
    """
    Run inbound guardrail checks in priority order:
      1. Prompt injection  — highest risk, check first
      2. PII detection     — block sensitive data from leaving the client
      3. Domain classifier — reject off-topic queries

    Returns the first blocking result, or ``allow`` if all pass.
    """
    checks = [
        detect_prompt_injection,
        detect_pii,
        classify_domain,
    ]

    for check in checks:
        result = check(query)
        if result.blocked:
            logger.warning(
                "inbound guardrail blocked | rule=%s reason=%s violations=%s",
                result.rule, result.reason, result.violations,
            )
            return result

    return GuardrailResult(action="allow", rule="inbound_pipeline")


# ---------------------------------------------------------------------------
# Outbound pipeline (runs on the LLM answer before it is returned to the UI)
# ---------------------------------------------------------------------------

def check_outbound(answer: str) -> GuardrailResult:
    """
    Run outbound guardrail checks:
      1. PII detection — ensure the model didn't echo back or hallucinate PII

    Returns the first blocking result, or ``allow`` if all pass.
    """
    result = detect_pii(answer)
    if result.blocked:
        logger.warning(
            "outbound guardrail blocked | rule=%s reason=%s violations=%s",
            result.rule, result.reason, result.violations,
        )
        return result

    return GuardrailResult(action="allow", rule="outbound_pipeline")
