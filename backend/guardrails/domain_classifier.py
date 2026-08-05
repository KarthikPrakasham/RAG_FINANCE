"""
Domain Classifier guardrail.

Determines whether a query is within the accepted domain:
  IN-SCOPE  : U.S. financial regulations, fair lending, consumer credit rights,
              ECOA / Reg B, Fair Housing Act, FRED economic data, interest rates,
              credit scores, mortgages, loans, banking compliance.
  OUT-OF-SCOPE: Everything else (cooking, sports, general knowledge, etc.)

Strategy: keyword allowlist + blocklist heuristics.
No LLM call is made here — the classifier is deterministic and fast.
A low-confidence result (neither clear IN nor clear OUT) passes through
so that the downstream LLM can handle ambiguous edge-cases.
"""
from __future__ import annotations

import re

from .models import GuardrailResult

# ---------------------------------------------------------------------------
# In-scope keyword set (finance / legal / regulatory domain)
# ---------------------------------------------------------------------------

_IN_SCOPE_TERMS: list[str] = [
    # Regulations & laws
    "ecoa", "regulation b", "reg b", "fair housing act", "fha",
    "equal credit opportunity", "fair lending", "consumer financial",
    "cfpb", "fdic", "occ", "federal reserve", "dodd-frank",
    "truth in lending", "tila", "respa", "hmda",
    # Financial products
    "mortgage", "loan", "credit", "debt", "interest rate", "apr",
    "credit score", "fico", "credit report", "credit bureau",
    "credit card", "auto loan", "student loan", "home equity",
    "refinanc", "foreclosure", "bankruptcy",
    # Banking & compliance
    "bank", "lender", "underwriting", "redlining", "discrimination",
    "adverse action", "disparate impact", "fair housing",
    # Economic data
    "fred", "gdp", "inflation", "cpi", "unemployment", "federal funds rate",
    "consumer credit", "consumer debt", "economy", "economic",
    # General finance
    "financ", "invest", "stock", "bond", "portfolio", "asset",
    "income", "salary", "wage", "tax", "retirement", "pension",
]

_IN_SCOPE_PATTERNS = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _IN_SCOPE_TERMS) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Hard out-of-scope categories — block immediately
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_TERMS: list[str] = [
    # Food & cooking
    "recipe", "cook", "cooking", "bake", "baking", "pizza", "pasta",
    "burger", "sandwich", "salad", "soup", "dessert", "ingredient",
    "cuisine", "restaurant", "menu", "chef",
    # Sports & entertainment
    "sport", "football", "basketball", "baseball", "soccer", "tennis",
    "cricket", "golf", "olympics", "athlete",
    "movie", "film", "celebrity", "actor", "actress", "music", "song",
    "lyrics", "album", "concert", "tv show", "anime",
    # Weather / lifestyle
    "weather", "forecast", "horoscope", "astrology", "zodiac",
    # Gaming
    "video game", "gaming", "minecraft", "fortnite", "playstation", "xbox",
    # Relationships / social
    "relationship", "dating", "romance", "breakup", "marriage",
    # Academic cheating
    "homework", "essay", "write my", "assignment",
    # Security threats
    "hack", "exploit", "malware", "ransomware", "phishing",
    # Controlled substances / weapons
    "drug", "narcotic", "weapon", "firearm", "ammunition",
    # Travel / misc off-topic
    "travel", "vacation", "hotel", "flight", "tourism",
    "fitness", "workout", "exercise", "diet", "calories",
    "fashion", "clothing", "outfit",
    "pet", "dog", "cat", "animal",
]

_OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _OUT_OF_SCOPE_TERMS) + r")\b",
    re.IGNORECASE,
)


def classify_domain(text: str) -> GuardrailResult:
    """
    Returns:
      - ``action='allow'``  if the query is clearly in-scope or ambiguous.
      - ``action='block'``  if the query is clearly out-of-scope.

    The ``rule`` field is set to ``'domain_classifier'``.
    The ``reason`` is ``'in_scope'``, ``'out_of_scope'``, or ``'ambiguous'``.
    """
    has_in_scope = bool(_IN_SCOPE_PATTERNS.search(text))
    has_out_of_scope = bool(_OUT_OF_SCOPE_PATTERNS.search(text))

    # Explicit out-of-scope hit → block regardless of in-scope signals.
    if has_out_of_scope and not has_in_scope:
        return GuardrailResult(
            action="block",
            rule="domain_classifier",
            reason="out_of_scope",
            violations=["query_not_in_financial_domain"],
        )

    # No in-scope signal at all → treat as off-topic and block.
    if not has_in_scope:
        return GuardrailResult(
            action="block",
            rule="domain_classifier",
            reason="out_of_scope",
            violations=["no_financial_domain_signal"],
        )

    return GuardrailResult(
        action="allow",
        rule="domain_classifier",
        reason="in_scope",
    )
