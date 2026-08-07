"""
Automatic user-intent extraction from conversation turns.

After each chat response is sent, this module analyses the user's query
(NOT the assistant's answer) to extract structured facts about the user's
situation. These facts accumulate in User.profile_json as long-term memory.

Model choice: uses a cheap/free local model via Ollama (e.g. gemma3:4b,
phi4-mini, qwen3:4b). Falls back to Gemini Flash if Ollama is unavailable.

Periodic pruning: if profile_json exceeds INTENT_MAX_PROFILE_CHARS, the
profile is consolidated via a summarisation LLM call to keep it compact.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.core.config import Settings

from memory.user_repository import UserRepository

logger = logging.getLogger(__name__)

# Maximum character length for profile_json before pruning triggers.
INTENT_MAX_PROFILE_CHARS = 2000

_EXTRACTION_PROMPT = """\
Extract factual information about the user from their message below.
Only extract facts the user explicitly states about themselves or their situation.
Do NOT infer, assume, or include information from context or assistant responses.

User message: "{user_query}"

Current user profile: {current_profile}

Return ONLY a valid JSON object with these fields (omit any field with no new info):
- "scenario": what happened to them (e.g. "denied mortgage", "denied auto loan")
- "credit_type": type of credit (e.g. "mortgage", "credit card", "auto loan")
- "protected_basis": list of protected classes mentioned (e.g. ["age", "race"])
- "lender_mentioned": name of bank/lender if stated
- "key_facts": list of specific facts stated (e.g. ["applicant is 62", "applied last week"])
- "complaint_filed": true/false if they mention having filed or wanting to file

Output JSON only, no explanation:"""

_CONSOLIDATION_PROMPT = """\
The following user profile has grown too large. Consolidate it into a shorter
version keeping only the most important and recent facts. Remove duplicates,
merge similar items, and keep at most 8 key_facts entries.

Current profile:
{profile_json}

Return ONLY a valid consolidated JSON object with the same field names:"""


async def extract_and_update_intent(
    user_id: str,
    user_query: str,
    settings: "Settings",
) -> None:
    """
    Extract user intent from their query and merge into long-term profile.
    Runs as a fire-and-forget background task — never raises to the caller.

    Steps:
        1. Fetch current profile for user_id
        2. Call cheap LLM to extract facts from user_query only
        3. Parse JSON response, merge into profile (additive)
        4. If profile > INTENT_MAX_PROFILE_CHARS, consolidate/prune
        5. Persist updated profile
    """
    try:
        user_repo = UserRepository()
        current_profile = user_repo.get_profile(user_id)

        # Skip trivial queries (too short to contain facts)
        if len(user_query.strip()) < 15:
            return

        # 1. Extract facts via cheap LLM
        extracted = await _call_extraction_llm(user_query, current_profile, settings)
        if not extracted:
            return

        # 2. Merge into existing profile
        merged = user_repo.update_profile(user_id, extracted)

        # 3. Check if pruning is needed
        profile_size = len(json.dumps(merged))
        if profile_size > INTENT_MAX_PROFILE_CHARS:
            logger.info(
                "Profile for user=%s exceeded %d chars (%d) — consolidating",
                user_id, INTENT_MAX_PROFILE_CHARS, profile_size,
            )
            consolidated = await _consolidate_profile(merged, settings)
            if consolidated:
                user_repo.set_profile(user_id, consolidated)

        logger.debug("Intent updated for user=%s: %s", user_id, extracted)

    except Exception:
        logger.exception("Intent extraction failed for user=%s (non-blocking)", user_id)


async def _call_extraction_llm(
    user_query: str,
    current_profile: dict,
    settings: "Settings",
) -> dict | None:
    """Call the cheap extraction model and parse its JSON output."""
    prompt = _EXTRACTION_PROMPT.format(
        user_query=user_query,
        current_profile=json.dumps(current_profile) if current_profile else "{}",
    )

    answer = await _invoke_cheap_model(prompt, settings)
    if not answer:
        return None

    return _parse_json_response(answer)


async def _consolidate_profile(
    profile: dict,
    settings: "Settings",
) -> dict | None:
    """Call the cheap model to consolidate a bloated profile."""
    prompt = _CONSOLIDATION_PROMPT.format(
        profile_json=json.dumps(profile, indent=2),
    )

    answer = await _invoke_cheap_model(prompt, settings)
    if not answer:
        return None

    return _parse_json_response(answer)


async def _invoke_cheap_model(prompt: str, settings: "Settings") -> str | None:
    """
    Call a cheap/free model for intent extraction.

    Priority:
        1. Ollama (local, free) — uses settings.intent_extraction_model
        2. Gemini Flash (fallback, near-free for short outputs)
    """
    # Try Ollama first (free, local)
    try:
        answer = await _call_ollama(prompt, settings)
        if answer:
            return answer
    except Exception:
        logger.debug("Ollama unavailable for intent extraction, trying Gemini")

    # Fallback to Gemini
    if settings.gemini_api_key:
        try:
            return await _call_gemini_flash(prompt, settings)
        except Exception:
            logger.debug("Gemini also failed for intent extraction")

    # Fallback to OpenAI
    if settings.openai_api_key:
        try:
            return _call_openai_sync(prompt, settings)
        except Exception:
            logger.debug("OpenAI also failed for intent extraction")

    return None


async def _call_ollama(prompt: str, settings: "Settings") -> str | None:
    """Call local Ollama model for extraction."""
    try:
        from ollama import AsyncClient

        client = AsyncClient(host=settings.ollama_base_url)
        model_name = settings.intent_extraction_model

        response = await client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 200},
        )
        return (response.message.content or "").strip()
    except Exception as e:
        logger.debug("Ollama call failed: %s", e)
        raise


async def _call_gemini_flash(prompt: str, settings: "Settings") -> str | None:
    """Call Gemini for extraction (near-free for short outputs)."""
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.intent_fallback_model)

    response = await model.generate_content_async(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=200,
        ),
    )
    return (response.text or "").strip()


def _call_openai_sync(prompt: str, settings: "Settings") -> str | None:
    """Call OpenAI (cheap, as last resort)."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.intent_openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_json_response(text: str) -> dict | None:
    """Parse a JSON response from the LLM, handling markdown code fences."""
    if not text:
        return None

    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            # Filter out empty/null values
            return {k: v for k, v in parsed.items() if v is not None and v != "" and v != []}
        return None
    except (json.JSONDecodeError, ValueError):
        logger.debug("Failed to parse intent extraction response: %s", text[:200])
        return None
