"""
Shared data models for the guardrails layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Action = Literal["allow", "block", "rewrite"]


@dataclass
class GuardrailResult:
    """Unified verdict returned by every guardrail check."""

    action: Action = "allow"
    reason: str | None = None
    # Which specific rule fired, e.g. "pii", "prompt_injection", "out_of_scope"
    rule: str | None = None
    violations: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action == "block"
