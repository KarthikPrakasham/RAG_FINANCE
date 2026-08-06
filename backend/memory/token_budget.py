"""
Token-budget manager for conversation memory.

Implements the §2.6 context assembly rule from the architecture doc:
    Prompt history = [latest summary (if any)] + [last N raw turns]

When the cumulative token count of unsummarised messages in a session exceeds
threshold T, all messages except the last N are summarised into the `summary`
table. The raw messages remain in SQLite for audit but are no longer injected
into the LLM prompt.

Token counting uses tiktoken's cl100k_base encoding (GPT-4/3.5 tokeniser) —
close enough for budget estimation regardless of which LLM actually generates.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from api.core.config import Settings

from memory.base_repository import BaseRepository
from memory.conversation_repository import ConversationRepository
from memory.models import Summary

logger = logging.getLogger(__name__)

# Shared tokeniser — cached after first call by tiktoken itself.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a string using cl100k_base (GPT-4 compatible)."""
    if not text:
        return 0
    return len(_ENCODING.encode(text))


class _SummaryRepository(BaseRepository):
    """Thin read/write layer for the summary table."""

    def get_latest(self, session_id: str) -> Summary | None:
        """Return the most recent summary for a session, or None."""
        with self._read() as db:
            return (
                db.query(Summary)
                .filter(Summary.session_id == session_id)
                .order_by(Summary.id.desc())
                .first()
            )

    def create(self, session_id: str, summary_text: str, covers_up_to_message_id: int) -> Summary:
        """Persist a new summary row."""
        with self._write() as db:
            row = Summary(
                session_id=session_id,
                summary_text=summary_text,
                covers_up_to_message_id=covers_up_to_message_id,
            )
            db.add(row)
            db.flush()
            return row


def get_context_window(
    session_id: str,
    T: int,
    N: int,
    settings: "Settings | None" = None,
) -> tuple[str, list[dict]]:
    """
    Assemble the memory context for the LLM prompt.

    Returns:
        (summary_text, recent_turns)
        - summary_text: rolling summary of older messages (or "" if none)
        - recent_turns: last N messages as [{"role": ..., "message": ...}, ...]

    Logic:
        1. Fetch all messages for the session (ordered by id ASC)
        2. Find the latest summary (if any) — gives us covers_up_to_message_id
        3. Unsummarised messages = those with id > covers_up_to_message_id
        4. Count tokens of all unsummarised messages
        5. If total tokens > T and there are more than N unsummarised messages:
           a. Take all unsummarised messages except the last N
           b. Summarise them (calls memory.summarizer)
           c. Persist the new summary row
        6. Return (latest summary text, last N unsummarised messages)
    """
    conversation_repo = ConversationRepository()
    summary_repo = _SummaryRepository()

    # 1. All messages for this session
    all_messages = conversation_repo.list_by_session(session_id)
    if not all_messages:
        return "", []

    # 2. Latest summary
    latest_summary = summary_repo.get_latest(session_id)
    covers_up_to = latest_summary.covers_up_to_message_id if latest_summary else 0

    # 3. Unsummarised messages (those after the summary boundary)
    unsummarised = [m for m in all_messages if m.id > covers_up_to]

    if not unsummarised:
        # Everything is already summarised — return the summary + empty recent
        summary_text = latest_summary.summary_text if latest_summary else ""
        return summary_text, []

    # 4. Count tokens of unsummarised messages
    total_tokens = sum(count_tokens(m.message or "") for m in unsummarised)

    # 5. Check if we need to summarise
    if total_tokens > T and len(unsummarised) > N:
        # Messages to summarise = all except last N
        to_summarise = unsummarised[:-N]
        to_keep = unsummarised[-N:]

        # Call the summariser
        messages_for_summary = [
            {"role": m.role, "message": m.message}
            for m in to_summarise
        ]

        try:
            from memory.summarizer import summarize_messages

            if settings is None:
                from api.core.config import get_settings
                settings = get_settings()

            new_summary_text = summarize_messages(messages_for_summary, settings)

            # If there was a previous summary, prepend it for continuity
            if latest_summary and latest_summary.summary_text:
                new_summary_text = (
                    f"{latest_summary.summary_text}\n\n"
                    f"[Continued]: {new_summary_text}"
                )

            # Persist the new summary
            highest_id = to_summarise[-1].id
            summary_repo.create(
                session_id=session_id,
                summary_text=new_summary_text,
                covers_up_to_message_id=highest_id,
            )

            logger.info(
                "Token budget triggered: session=%s tokens=%d > T=%d, "
                "summarised %d messages (up to id=%d), keeping last %d",
                session_id, total_tokens, T, len(to_summarise), highest_id, N,
            )

            # Return the new summary + last N turns
            recent_turns = [
                {"role": m.role, "message": m.message}
                for m in to_keep
            ]
            return new_summary_text, recent_turns

        except Exception:
            logger.exception(
                "Summarisation failed for session=%s — falling back to last %d turns without summary",
                session_id, N,
            )
            # Fall through to the default return below

    # 6. No summarisation needed (or it failed) — return existing summary + last N turns
    summary_text = latest_summary.summary_text if latest_summary else ""
    recent = unsummarised[-N:] if len(unsummarised) > N else unsummarised
    recent_turns = [
        {"role": m.role, "message": m.message}
        for m in recent
    ]
    return summary_text, recent_turns
