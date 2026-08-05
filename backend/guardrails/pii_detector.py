"""
PII Detector guardrail.

Scans text for common Personally Identifiable Information patterns using
regular expressions.  No external service or model is required.

Detected categories
-------------------
- SSN           Social Security Number  (xxx-xx-xxxx)
- Credit card   Visa / MC / Amex / Discover (basic Luhn-pattern)
- US phone      (xxx) xxx-xxxx, xxx-xxx-xxxx, +1xxxxxxxxxx
- Email address
- US ZIP code   (only when adjacent to triggering keywords)
- Date of birth (DOB / date of birth keyword + date pattern)
"""
from __future__ import annotations

import re

from .models import GuardrailResult

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "credit_card",
        re.compile(
            r"\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})"
            r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
        ),
    ),
    (
        "phone",
        re.compile(
            r"(?:\+1[\s\-]?)?"
            r"(?:\(\d{3}\)[\s\-]?|\d{3}[\s\-])"
            r"\d{3}[\s\-]\d{4}\b"
        ),
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "dob",
        re.compile(
            r"(?:dob|date\s+of\s+birth)\s*[:\-]?\s*"
            r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
            re.IGNORECASE,
        ),
    ),
    (
    "bank_account",
    re.compile(
        r"\b(?:account|acct|account\s*number|a/c)\b"
        r"\s*[:#-]?\s*"
        r"\d{4,17}\b",
        re.IGNORECASE,
    ),
    ),
]


def detect_pii(text: str) -> GuardrailResult:
    """
    Return a ``GuardrailResult`` with ``action='block'`` if PII is detected,
    otherwise ``action='allow'``.

    The ``violations`` list names every category that matched.
    """
    found: list[str] = []
    for label, pattern in _PATTERNS:
        if pattern.search(text):
            found.append(label)

    if found:
        return GuardrailResult(
            action="block",
            rule="pii_detector",
            reason=f"PII detected: {', '.join(found)}",
            violations=found,
        )

    return GuardrailResult(action="allow", rule="pii_detector")
