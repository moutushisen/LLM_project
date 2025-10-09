#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import sqlite3
from datetime import datetime
from pathlib import Path


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent
    DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "memory.db"
    db_path = os.getenv("MEMORY_DB_PATH") or DEFAULT_DB_PATH
    print(f"DB: {db_path}")

    if not os.path.exists(db_path):
        print("DB file not found.")
        sys.exit(1)

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        print(f"Failed to open DB: {e}")
        sys.exit(1)

    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rolling_memory'")
        tbl = cur.fetchone()
        if not tbl:
            print("Table 'rolling_memory' not found.")
            sys.exit(2)

        row = cur.execute(
            "SELECT content, updated_at FROM rolling_memory WHERE id = 'default'"
        ).fetchone()

        if not row:
            print("No rolling memory present (row id='default' not found).")
            sys.exit(3)

        content = row["content"]
        updated_at = row["updated_at"]
        print(f"Updated: {updated_at}")
        print("----- MEMORY START -----")
        print(content)
        print("----- MEMORY END -----")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


