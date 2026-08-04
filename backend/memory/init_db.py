from sqlalchemy import inspect, text

from .db import engine
# Explicit model imports ensure Session and Conversation are registered
# with Base.metadata before create_all is called.
from .models import Base, Session, Conversation  # noqa: F401


def _apply_migrations() -> None:
    """Apply any schema changes that CREATE TABLE cannot handle.

    SQLAlchemy's ``create_all`` only creates missing tables; it never alters
    existing ones.  Add a branch here for every column added after the initial
    schema was deployed.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    # ── conversation.user_id ────────────────────────────────────────────────
    # Guard: table must exist before we inspect its columns.
    if "conversation" in existing_tables:
        conv_cols = {col["name"] for col in inspector.get_columns("conversation")}
        if "user_id" not in conv_cols:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE conversation ADD COLUMN user_id VARCHAR"))
                conn.commit()


def init_database() -> None:
    # Create any tables that don't exist yet (safe to call on every startup).
    Base.metadata.create_all(bind=engine)
    # Apply incremental schema migrations for columns added post-deploy.
    _apply_migrations()
