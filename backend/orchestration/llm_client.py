"""
Multi-provider LLM wrapper (Stage 8 candidates: Gemini, Claude, OpenAI, Ollama).

Provider selection is driven by ``Settings.llm_model`` — the model name is
pattern-matched to decide which SDK to call:

    "gemini*"                 -> Google Generative AI  (google-generativeai)
    "claude*"                 -> Anthropic              (anthropic)
    "gpt*" / "o1*" / "o3*"    -> OpenAI                 (openai)
    anything else             -> Ollama                 (ollama) — local/self-hosted
                                 models such as "llama3", "mistral", "gemma3", etc.

Ollama doesn't require an API key — it talks to a local (or self-hosted)
server at ``Settings.ollama_base_url`` (default http://localhost:11434), so
it's treated as always "configured" and used as the catch-all provider.

If the matching provider's API key is not configured (Gemini/Anthropic/
OpenAI), or the API call raises for any reason (network, auth, rate limit,
model not pulled, etc.), this falls back to a deterministic canned answer so
the chat endpoint never hard-fails just because an upstream LLM is unavailable.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from api.core.config import Settings

logger = logging.getLogger(__name__)

Provider = str  # "gemini" | "anthropic" | "openai" | "ollama"


def _infer_provider(model_name: str) -> Provider:
    name = model_name.lower()
    if "gemini" in name:
        return "gemini"
    if "claude" in name:
        return "anthropic"
    if "gpt" in name or name.startswith("o1") or name.startswith("o3"):
        return "openai"
    # Catch-all: local/self-hosted models served by Ollama (llama3, mistral,
    # gemma3, qwen, phi, deepseek, etc.) — no API key needed, just a model name.
    return "ollama"


class LLMClient:
    """Generate a response from the assembled prompt.

    Routes to the configured provider's SDK based on ``settings.llm_model``.
    Falls back to a deterministic canned answer if no API key is configured
    for that provider, or if the provider call fails.
    """

    async def generate(self, prompt: str, settings: Settings) -> tuple[str, dict[str, int]]:
        provider = _infer_provider(settings.llm_model)

        try:
            if provider == "gemini" and settings.gemini_api_key:
                return await self._generate_gemini(prompt, settings)
            if provider == "anthropic" and settings.anthropic_api_key:
                return await self._generate_anthropic(prompt, settings)
            if provider == "openai" and settings.openai_api_key:
                return await self._generate_openai(prompt, settings)
            if provider == "ollama":
                return await self._generate_ollama(prompt, settings)

            logger.warning(
                "No API key configured for provider=%s (model=%s) — using fallback answer.",
                provider, settings.llm_model,
            )
        except Exception:
            logger.exception(
                "LLM call failed for provider=%s model=%s — using fallback answer.",
                provider, settings.llm_model,
            )

        return self._fallback(prompt)

    # ------------------------------------------------------------------
    # Google Gemini
    # ------------------------------------------------------------------
    async def _generate_gemini(self, prompt: str, settings: Settings) -> tuple[str, dict[str, int]]:
        import google.generativeai as genai

        model = _get_gemini_model(settings.gemini_api_key, settings.llm_model)
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_tokens,
            ),
        )
        answer = (response.text or "").strip()

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", prompt_tokens + completion_tokens) or (
            prompt_tokens + completion_tokens
        )
        return answer, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    # ------------------------------------------------------------------
    # Anthropic Claude
    # ------------------------------------------------------------------
    async def _generate_anthropic(self, prompt: str, settings: Settings) -> tuple[str, dict[str, int]]:
        client = _get_anthropic_client(settings.anthropic_api_key)
        response = await client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        input_tokens = getattr(response.usage, "input_tokens", 0) or 0
        output_tokens = getattr(response.usage, "output_tokens", 0) or 0
        return answer, {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    async def _generate_openai(self, prompt: str, settings: Settings) -> tuple[str, dict[str, int]]:
        client = _get_openai_client(settings.openai_api_key)
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        answer = (response.choices[0].message.content or "").strip()

        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (
            prompt_tokens + completion_tokens
        )
        return answer, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    # ------------------------------------------------------------------
    # Ollama (local / self-hosted)
    # ------------------------------------------------------------------
    async def _generate_ollama(self, prompt: str, settings: Settings) -> tuple[str, dict[str, int]]:
        client = _get_ollama_client(settings.ollama_base_url)
        response = await client.chat(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        )
        answer = (response.message.content or "").strip()

        prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
        completion_tokens = getattr(response, "eval_count", 0) or 0
        return answer, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    # ------------------------------------------------------------------
    # Fallback — used when no provider is configured/available, or the
    # provider call raised an exception.
    # ------------------------------------------------------------------
    def _fallback(self, prompt: str) -> tuple[str, dict[str, int]]:
        answer = self._fallback_answer(prompt)
        token_usage = {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens": len(prompt.split()) + len(answer.split()),
        }
        return answer, token_usage

    def _fallback_answer(self, prompt: str) -> str:
        question = ""
        if "User question:" in prompt:
            question = prompt.split("User question:", 1)[1].strip()
        question = question or "your question"

        context_lines = re.findall(r"^- (.+?): (.+)", prompt)
        if context_lines:
            context_snippet = context_lines[0][1]
            return (
                f"Based on the available context, I would answer: {question}. "
                f"The strongest retrieved detail appears to be: {context_snippet}"
            )

        return (
            "I can help with that. The orchestration layer assembled a prompt and "
            f"the fallback response is ready for: {question}"
        )


# ---------------------------------------------------------------------------
# Cached client/model constructors — imported lazily so this module loads
# without any of the three SDKs installed unless that provider is actually
# used, and so client construction (auth, etc.) happens once per api key.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _get_gemini_model(api_key: str, model_name: str):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


@lru_cache(maxsize=2)
def _get_anthropic_client(api_key: str):
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key=api_key)


@lru_cache(maxsize=2)
def _get_openai_client(api_key: str):
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key)


@lru_cache(maxsize=2)
def _get_ollama_client(base_url: str):
    from ollama import AsyncClient

    return AsyncClient(host=base_url)
