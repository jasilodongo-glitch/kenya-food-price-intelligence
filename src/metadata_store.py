"""Optional SQLite store for dataset metadata and activity history (no raw data is stored)."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import config


def _conn(path: Path | None = None) -> sqlite3.Connection:
    p = path or config.DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.execute("""CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, sha256 TEXT, sheet TEXT,
        rows INTEGER, columns INTEGER, size_bytes INTEGER, uploaded_at TEXT)""")
    return conn


def record_dataset(filename: str, sha256: str, sheet: str | None, rows: int, columns: int, size_bytes: int, path: Path | None = None) -> None:
    with closing(_conn(path)) as c, c:
        c.execute("INSERT INTO datasets (filename, sha256, sheet, rows, columns, size_bytes, uploaded_at) VALUES (?,?,?,?,?,?,?)",
                  (filename, sha256, sheet, rows, columns, size_bytes, datetime.now(timezone.utc).isoformat(timespec="seconds")))


def history(path: Path | None = None) -> list[dict]:
    with closing(_conn(path)) as c:
        cur = c.execute("SELECT filename, sheet, rows, columns, size_bytes, uploaded_at FROM datasets ORDER BY id DESC LIMIT 50")
        keys = [d[0] for d in cur.description]
        return [dict(zip(keys, r)) for r in cur.fetchall()]


def clear_history(path: Path | None = None) -> None:
    with closing(_conn(path)) as c, c:
        c.execute("DELETE FROM datasets")
