#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sqlite3
from datetime import datetime
from typing import Optional
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "memory.db"


class RollingMemoryStorage:
    """Single rolling text memory stored in SQLite.

    Table schema:
        rolling_memory(id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, content TEXT NOT NULL)
    We use a single row with id='default'.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("MEMORY_DB_PATH") or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rolling_memory (
                    id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    content TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_text(self) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT content FROM rolling_memory WHERE id = 'default'").fetchone()
            return row[0] if row else ""

    def set_text(self, content: str) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO rolling_memory (id, updated_at, content) VALUES ('default', ?, ?)\n                 ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at, content=excluded.content",
                (now, content.strip()),
            )
            conn.commit()

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM rolling_memory WHERE id = 'default'")
            conn.commit()


