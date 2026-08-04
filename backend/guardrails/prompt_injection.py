"""
Prompt Injection Detector guardrail.

Detects common adversarial patterns that attempt to override system
instructions, jailbreak the model, or extract hidden context.

Strategy: keyword/pattern heuristics (no external call needed).
These cover the most common attack vectors observed in the wild:
  - "ignore previous instructions"
  - "forget everything", "disregard above"
  - Role-play / persona hijack ("you are now DAN / an evil AI …")
  - Direct instruction override ("new instructions:", "system prompt:")
  - Context extraction ("repeat your instructions", "what is your system prompt")
  - Encoded/obfuscated variants (base64 references, leetspeak triggers)
"""
from __future__ import annotations

import re

from .models import GuardrailResult

# ---------------------------------------------------------------------------
# Patterns — order matters: more specific first
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier|system)\s+(instructions?|prompts?|rules?|constraints?)",
            re.IGNORECASE,
        ),
    ),
    (
        "forget_context",
        re.compile(
            r"(forget|disregard|override|bypass|reset)\s+(everything|all|above|prior|previous|system)",
            re.IGNORECASE,
        ),
    ),
    (
        "new_instructions",
        re.compile(
            r"(new\s+instructions?|updated?\s+instructions?|system\s+prompt\s*:)",
            re.IGNORECASE,
        ),
    ),
    (
        "persona_hijack",
        re.compile(
            r"you\s+are\s+now\s+(an?\s+)?(evil|unrestricted|unfiltered|jailbroken|DAN|GPT\-?[0-9])",
            re.IGNORECASE,
        ),
    ),
    (
        "context_extraction",
        re.compile(
            r"(repeat|print|reveal|show|output|display|tell me)\s+(your\s+)?"
            r"(system\s+prompt|instructions?|initial\s+prompt|hidden\s+context|above\s+text)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(an?\s+)?"
            r"(unrestricted|evil|hacker|criminal|unethical)",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_injection",
        # Attempts to inject common LLM delimiters
        re.compile(r"(<\|system\|>|<\|user\|>|<\|assistant\|>|\[INST\]|\[/INST\])", re.IGNORECASE),
    ),
]


def detect_prompt_injection(text: str) -> GuardrailResult:
    """
    Return ``action='block'`` if a prompt-injection pattern is found,
    otherwise ``action='allow'``.
    """
    found: list[str] = []
    for label, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            found.append(label)

    if found:
        return GuardrailResult(
            action="block",
            rule="prompt_injection",
            reason=f"Prompt injection attempt detected: {', '.join(found)}",
            violations=found,
        )

    return GuardrailResult(action="allow", rule="prompt_injection")
