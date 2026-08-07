"""
CRUD for the User model — long-term memory (profile/intent) that persists
across sessions for the same user_id.

profile_json stores structured facts about the user's situation, extracted
automatically from their queries by intent_extractor.py.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from .base_repository import BaseRepository
from .models import User

logger = logging.getLogger(__name__)


class UserRepository(BaseRepository):
    """All SQLite reads/writes for the ``user`` table."""

    def get_or_create(self, user_id: str) -> User:
        """Return the User row, creating it with an empty profile if it doesn't exist."""
        with self._read() as db:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user:
                return user

        # Doesn't exist — create
        with self._write() as db:
            # Double-check inside the write transaction (race condition guard)
            user = db.query(User).filter(User.user_id == user_id).first()
            if user:
                return user
            user = User(user_id=user_id, profile_json={})
            db.add(user)
            db.flush()
            return user

    def get_profile(self, user_id: str) -> dict:
        """Return the user's profile_json dict, or {} if the user doesn't exist."""
        with self._read() as db:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user is None:
                return {}
            return user.profile_json or {}

    def update_profile(self, user_id: str, updates: dict) -> dict:
        """Merge `updates` into the user's existing profile_json (additive).

        - Scalar values are overwritten (latest wins)
        - List values are extended and deduplicated
        - Dict values are recursively merged

        Returns the merged profile.
        """
        with self._write() as db:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user is None:
                user = User(user_id=user_id, profile_json={})
                db.add(user)

            current = user.profile_json or {}
            merged = _deep_merge(current, updates)
            user.profile_json = merged
            db.flush()
            return merged

    def set_profile(self, user_id: str, profile: dict) -> None:
        """Replace the entire profile_json (used after pruning/consolidation)."""
        with self._write() as db:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user is None:
                user = User(user_id=user_id, profile_json=profile)
                db.add(user)
            else:
                user.profile_json = profile

    def get_profile_size(self, user_id: str) -> int:
        """Return the character length of the serialised profile_json."""
        profile = self.get_profile(user_id)
        return len(json.dumps(profile))


def _deep_merge(base: dict, updates: dict) -> dict:
    """Recursively merge `updates` into `base`.

    - Lists are extended and deduplicated (order preserved, latest additions at end)
    - Dicts are recursively merged
    - Scalars are overwritten by the update value
    """
    merged = base.copy()
    for key, new_val in updates.items():
        if key not in merged:
            merged[key] = new_val
        elif isinstance(merged[key], list) and isinstance(new_val, list):
            # Extend and deduplicate while preserving order
            combined = merged[key].copy()
            for item in new_val:
                if item not in combined:
                    combined.append(item)
            merged[key] = combined
        elif isinstance(merged[key], dict) and isinstance(new_val, dict):
            merged[key] = _deep_merge(merged[key], new_val)
        else:
            # Scalar — latest wins
            merged[key] = new_val
    return merged
