from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import DateTime
from sqlalchemy import JSON

from datetime import datetime

from .db import Base


class Session(Base):

    __tablename__ = "session"

    id = Column(String, primary_key=True)

    title = Column(String, default="New Chat")

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    def to_dict(self):

        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Conversation(Base):

    __tablename__ = "conversation"

    id = Column(Integer, primary_key=True)

    session_id = Column(
        String,
        ForeignKey("session.id", ondelete="CASCADE"),
        index=True
    )

    user_id = Column(String, nullable=True, index=True)

    role = Column(String)

    message = Column(Text)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    def to_dict(self):

        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "role": self.role,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class User(Base):
    """Long-term memory: persists user profile and intent across sessions.

    The profile_json column stores durable facts about the user (e.g. their
    scenario, protected basis, preferences) that should be injected into the
    system prompt regardless of which session they're in.
    """

    __tablename__ = "user"

    user_id = Column(String, primary_key=True)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    profile_json = Column(
        JSON,
        default=dict,
        nullable=False,
        doc="Durable user intent / facts that persist across sessions"
    )

    def to_dict(self):

        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "profile_json": self.profile_json or {},
        }


class Summary(Base):
    """Rolling conversation summary for token-budget management.

    When the cumulative token count of unsummarised messages in a session
    exceeds threshold T, all messages except the last N are summarised into
    this table. The summary_text is then used as compressed history in the
    LLM prompt, while raw messages remain in the conversation table for audit.
    """

    __tablename__ = "summary"

    id = Column(Integer, primary_key=True)

    session_id = Column(
        String,
        ForeignKey("session.id", ondelete="CASCADE"),
        index=True
    )

    summary_text = Column(Text, nullable=False)

    covers_up_to_message_id = Column(
        Integer,
        nullable=False,
        doc="The highest conversation.id that this summary covers"
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    def to_dict(self):

        return {
            "id": self.id,
            "session_id": self.session_id,
            "summary_text": self.summary_text,
            "covers_up_to_message_id": self.covers_up_to_message_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
