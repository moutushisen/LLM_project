#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sqlite3
import uuid
import json
from datetime import datetime
from typing import List, Optional, Dict, Any


DEFAULT_DB_PATH = os.getenv("MEMORY_DB_PATH", "/home/mihoyohb/LLM_project/data/memory.db")


class MemoryStorage:
    """Lightweight SQLite storage for short text memories."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
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
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source TEXT,
                    tags TEXT,
                    content TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add_memory(
        self,
        content: str,
        mtype: str = "interaction",
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        mem_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        tags_str = ",".join(tags) if tags else None
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories (id, type, created_at, source, tags, content) VALUES (?, ?, ?, ?, ?, ?)",
                (mem_id, mtype, created_at, source, tags_str, content.strip()),
            )
            conn.commit()
        return mem_id

    def delete_memory(self, mem_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_memories(
        self,
        query: Optional[str] = None,
        mtype: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        where = []
        params: List[Any] = []
        if query:
            where.append("(content LIKE ? OR tags LIKE ? OR source LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
        if mtype:
            where.append("type = ?")
            params.append(mtype)
        where_clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"SELECT * FROM memories {where_clause} ORDER BY datetime(created_at) DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def export_all(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM memories ORDER BY datetime(created_at) DESC").fetchall()
        return [dict(r) for r in rows]

    def to_json(self, records: Optional[List[Dict[str, Any]]] = None) -> str:
        data = records if records is not None else self.export_all()
        return json.dumps(data, ensure_ascii=False, indent=2)


