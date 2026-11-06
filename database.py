"""Consent-gated SQLite storage for scan and chat history."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


DEFAULT_DB = Path(os.environ.get("DERMACARE_DB", Path(__file__).with_name("dermacare_history.sqlite3")))


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _managed_connection(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    connection = connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_db(db_path: str | Path | None = None) -> None:
    with _managed_connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                triage_level TEXT NOT NULL,
                symptom_summary TEXT NOT NULL,
                image_sha256 TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def save_scan(
    user_id: str,
    prediction: str,
    confidence: float,
    triage_level: str,
    symptom_summary: str,
    image_sha256: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Persist a scan only when called after explicit consent."""
    initialize_db(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    with _managed_connection(db_path) as connection:
        cursor = connection.execute(
            """INSERT INTO scans
               (user_id, created_at, prediction, confidence, triage_level, symptom_summary, image_sha256)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, timestamp, prediction, float(confidence), triage_level, symptom_summary, image_sha256),
        )
        return int(cursor.lastrowid)


def save_messages(
    scan_id: int,
    messages: Sequence[Mapping[str, Any]],
    db_path: str | Path | None = None,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    allowed = [(scan_id, str(m["role"]), str(m["content"]), timestamp)
               for m in messages if m.get("role") in {"user", "assistant"} and m.get("content")]
    if not allowed:
        return
    with _managed_connection(db_path) as connection:
        connection.execute("DELETE FROM messages WHERE scan_id = ?", (scan_id,))
        connection.executemany(
            "INSERT INTO messages (scan_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            allowed,
        )


def update_scan_summary(
    scan_id: int,
    triage_level: str,
    symptom_summary: str,
    db_path: str | Path | None = None,
) -> None:
    with _managed_connection(db_path) as connection:
        connection.execute(
            "UPDATE scans SET triage_level = ?, symptom_summary = ? WHERE id = ?",
            (triage_level, symptom_summary, scan_id),
        )


def list_scans(user_id: str, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    initialize_db(db_path)
    with _managed_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT id, created_at, prediction, confidence, triage_level, symptom_summary "
            "FROM scans WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_messages(scan_id: int, db_path: str | Path | None = None) -> list[dict[str, str]]:
    with _managed_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT role, content, created_at FROM messages WHERE scan_id = ? ORDER BY id",
            (scan_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_user_data(user_id: str, db_path: str | Path | None = None) -> int:
    """Delete a user's scans; associated chat messages cascade automatically."""
    initialize_db(db_path)
    with _managed_connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM scans WHERE user_id = ?", (user_id,))
        return int(cursor.rowcount)
