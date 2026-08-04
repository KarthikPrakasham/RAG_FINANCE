from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import DateTime

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
