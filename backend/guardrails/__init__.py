"""
Guardrails package.

Public surface:
    check_inbound(query)  -> GuardrailResult
    check_outbound(answer) -> GuardrailResult
"""
from .rules import check_inbound, check_outbound

__all__ = ["check_inbound", "check_outbound"]
