from .base_repository import BaseRepository
from .models import Conversation


class ConversationRepository(BaseRepository):
    """All SQLite reads/writes for the ``conversation`` table."""

    def add(self, session_id: str, role: str, message: str, user_id: str | None = None) -> dict:
        """Append one turn to a session and return it as a plain dict."""

        with self._write() as db:

            row = Conversation(
                session_id=session_id,
                user_id=user_id,
                role=role,
                message=message
            )

            db.add(row)

            # Flush so the autoincrement id and the created_at default are
            # populated while the row is still attached to this session.
            db.flush()

            return row.to_dict()

    def list_by_session(self, session_id: str):
        """Full history for a session, oldest message first."""

        with self._read() as db:

            return db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).order_by(
                Conversation.id.asc()
            ).all()

    def get_recent(self, session_id: str, limit: int = 4):
        """The last ``limit`` turns, still ordered oldest → newest."""

        if limit <= 0:
            return []

        with self._read() as db:

            rows = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).order_by(
                Conversation.id.desc()
            ).limit(limit).all()

        return list(reversed(rows))

    def count(self, session_id: str) -> int:

        with self._read() as db:

            return db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).count()

    def delete_by_session(self, session_id: str) -> int:
        """Remove every message of a session. Returns the number deleted."""

        with self._write() as db:

            return db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).delete()
