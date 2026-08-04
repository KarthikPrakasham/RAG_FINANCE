"""
db.py — SQLAlchemy engine, session factory, and declarative base.

Layout
------
* ``engine``        — SQLite engine with WAL mode for safe concurrent access.
* ``SessionLocal``  — session factory used by ``BaseRepository``.
* ``Base``          — declarative base imported by ``models.py``.
* ``get_db()``      — FastAPI dependency that yields a scoped session and
                      closes it automatically after the request.

The DB file is stored next to this module so its location is stable
regardless of the working directory the server is launched from.

Application code should use ``MemoryService`` rather than importing
this module directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# Database location — always next to this file
# ---------------------------------------------------------------------------
_DB_PATH = Path(__file__).resolve().parent / "chat_memory.db"
DATABASE_URL = f"sqlite:///{_DB_PATH}"

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _connection_record) -> None:
    """Enable WAL mode and foreign-key enforcement for every new connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

# ---------------------------------------------------------------------------
# Declarative base — imported by models.py
# ---------------------------------------------------------------------------
Base = declarative_base()


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
def get_db() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy session for the duration of a request, then close it.

    Usage in a router::

        from memory.db import get_db
        from sqlalchemy.orm import Session

        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# # process's current working directory. Override with RAG_MEMORY_DB (a file path)
# # or RAG_MEMORY_DB_URL (a full SQLAlchemy URL, e.g. a Postgres DSN) if needed.
# MEMORY_DIR = Path(__file__).resolve().parent
# DEFAULT_DB_PATH = Path(os.environ.get("RAG_MEMORY_DB", MEMORY_DIR / "chat_memory.db"))
# DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# DATABASE_URL = os.environ.get(
#     "RAG_MEMORY_DB_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"
# )

# # ``check_same_thread=False`` lets the connection be shared across threads (e.g.
# # FastAPI/uvicorn workers); safety is provided by using one Session per request.
# engine = create_engine(
#     DATABASE_URL,
#     connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
#     future=True,
# )

# SessionLocal = sessionmaker(
#     autocommit=False,
#     autoflush=False,
#     bind=engine,
#     future=True,
# )

# Base = declarative_base()


# @event.listens_for(engine, "connect")
# def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
#     """Enable WAL journaling (better read/write concurrency) and FK enforcement.

#     Only relevant for SQLite; harmless to guard so a future Postgres URL is safe.
#     """
#     if not DATABASE_URL.startswith("sqlite"):
#         return
#     cursor = dbapi_connection.cursor()
#     cursor.execute("PRAGMA journal_mode=WAL;")
#     cursor.execute("PRAGMA foreign_keys=ON;")
#     cursor.close()


# def _utcnow() -> datetime:
#     return datetime.now(timezone.utc)


# # --- ORM schema -------------------------------------------------------------
# class ChatSession(Base):
#     """A single conversation thread. Groups an ordered list of messages."""

#     __tablename__ = "chat_sessions"

#     id: Mapped[str] = mapped_column(String(64), primary_key=True)
#     title: Mapped[str | None] = mapped_column(String(255), nullable=True)
#     created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
#     updated_at: Mapped[datetime] = mapped_column(
#         DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
#     )
#     # Free-form JSON for anything callers want to attach (user id, tags, …).
#     meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)

#     messages: Mapped[list["ChatMessage"]] = relationship(
#         back_populates="session",
#         cascade="all, delete-orphan",
#         order_by="ChatMessage.id",
#     )

#     def to_dict(self) -> dict:
#         return {
#             "id": self.id,
#             "title": self.title,
#             "created_at": self.created_at.isoformat() if self.created_at else None,
#             "updated_at": self.updated_at.isoformat() if self.updated_at else None,
#             "meta": self.meta or {},
#         }


# class ChatMessage(Base):
#     """One turn in a conversation (a user question or an assistant answer).

#     The token/latency columns mirror the metrics the frontend shows, and
#     ``sources`` stores the RAG citations as JSON so a conversation can be
#     replayed exactly as it was rendered.
#     """

#     __tablename__ = "chat_messages"

#     id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
#     session_id: Mapped[str] = mapped_column(
#         String(64),
#         ForeignKey("chat_sessions.id", ondelete="CASCADE"),
#         index=True,
#         nullable=False,
#     )
#     role: Mapped[str] = mapped_column(String(16), nullable=False)  # user|assistant|system
#     content: Mapped[str] = mapped_column(Text, nullable=False)
#     status: Mapped[str] = mapped_column(String(16), default="done", nullable=False)

#     # Usage + latency metrics (assistant turns).
#     input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
#     output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
#     response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

#     # RAG grounding citations, stored as a JSON array of source objects.
#     sources: Mapped[list | None] = mapped_column(JSON, nullable=True)

#     created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

#     session: Mapped["ChatSession"] = relationship(back_populates="messages")

#     def to_dict(self) -> dict:
#         return {
#             "id": self.id,
#             "session_id": self.session_id,
#             "role": self.role,
#             "content": self.content,
#             "status": self.status,
#             "input_tokens": self.input_tokens,
#             "output_tokens": self.output_tokens,
#             "response_time_ms": self.response_time_ms,
#             "sources": self.sources or [],
#             "created_at": self.created_at.isoformat() if self.created_at else None,
#         }


# def init_db() -> None:
#     """Create the schema if it doesn't exist yet. Safe to call repeatedly."""
#     Base.metadata.create_all(bind=engine)


# @contextmanager
# def session_scope() -> Iterator["SessionLocal"]:
#     """Provide a transactional scope around a series of operations.

#     Commits on success, rolls back on error, and always closes the session::

#         with session_scope() as db:
#             db.add(obj)
#     """
#     session = SessionLocal()
#     try:
#         yield session
#         session.commit()
#     except Exception:
#         session.rollback()
#         raise
#     finally:
#         session.close()
