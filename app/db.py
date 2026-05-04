import json
import sqlite3
from pathlib import Path
from typing import Any

from . import DATABASE_PATH, INSTANCE_DIR


def _connection() -> sqlite3.Connection:
    db = sqlite3.connect(DATABASE_PATH)
    db.row_factory = sqlite3.Row
    return db


def ensure_database() -> None:
    Path(INSTANCE_DIR).mkdir(parents=True, exist_ok=True)
    with _connection() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS planner_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_kind TEXT NOT NULL CHECK(item_kind IN ('TASK', 'HABIT', 'EVENT')),
                title TEXT NOT NULL,
                details TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                intent_label TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                embedding_json TEXT NOT NULL,
                priority TEXT NOT NULL CHECK(priority IN ('low', 'medium', 'high')),
                is_flagged INTEGER NOT NULL DEFAULT 0,
                is_completed INTEGER NOT NULL DEFAULT 0,
                scheduled_date TEXT,
                scheduled_time TEXT,
                repeat_frequency TEXT NOT NULL DEFAULT 'none'
                    CHECK(repeat_frequency IN ('none', 'daily', 'weekly', 'monthly')),
                completed_at TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        # Indexes for common query patterns
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_scheduled_date ON planner_items (scheduled_date)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_is_completed ON planner_items (is_completed)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_category ON planner_items (category)"
        )
        db.commit()


def insert_item(payload: dict[str, Any]) -> int:
    with _connection() as db:
        cursor = db.execute(
            """
            INSERT INTO planner_items (
                item_kind,
                title,
                details,
                raw_text,
                intent_label,
                category,
                confidence,
                embedding_json,
                priority,
                is_flagged,
                is_completed,
                scheduled_date,
                scheduled_time,
                repeat_frequency,
                completed_at,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["item_kind"],
                payload["title"],
                payload["details"],
                payload["raw_text"],
                payload["intent_label"],
                payload["category"],
                payload["confidence"],
                json.dumps(payload["embedding"]),
                payload["priority"],
                int(payload["is_flagged"]),
                int(payload["is_completed"]),
                payload["scheduled_date"],
                payload["scheduled_time"],
                payload["repeat_frequency"],
                payload["completed_at"],
                payload["created_at"],
            ),
        )
        db.commit()
        return int(cursor.lastrowid)


def fetch_items() -> list[dict[str, Any]]:
    with _connection() as db:
        rows = db.execute(
            """
            SELECT
                id,
                item_kind,
                title,
                details,
                raw_text,
                intent_label,
                category,
                confidence,
                embedding_json,
                priority,
                is_flagged,
                is_completed,
                scheduled_date,
                scheduled_time,
                repeat_frequency,
                completed_at,
                created_at
            FROM planner_items
            ORDER BY datetime(COALESCE(scheduled_date || 'T' || COALESCE(scheduled_time, '23:59:59'), created_at)) ASC, id ASC
            """
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["embedding"] = json.loads(item.pop("embedding_json"))
        item["is_flagged"] = bool(item["is_flagged"])
        item["is_completed"] = bool(item["is_completed"])
        items.append(item)
    return items


def update_item(item_id: int, fields: dict[str, Any]) -> None:
    if not fields:
        return

    allowed = {
        "title",
        "details",
        "raw_text",
        "intent_label",
        "category",
        "confidence",
        "embedding_json",
        "priority",
        "is_flagged",
        "is_completed",
        "scheduled_date",
        "scheduled_time",
        "repeat_frequency",
        "completed_at",
    }
    columns = []
    values = []
    for key, value in fields.items():
        if key not in allowed:
            continue
        columns.append(f"{key} = ?")
        values.append(value)

    if not columns:
        return

    values.append(item_id)
    with _connection() as db:
        db.execute(f"UPDATE planner_items SET {', '.join(columns)} WHERE id = ?", values)
        db.commit()


def clear_items() -> None:
    with _connection() as db:
        db.execute("DELETE FROM planner_items")
        db.commit()


def delete_item(item_id: int) -> None:
    with _connection() as db:
        db.execute("DELETE FROM planner_items WHERE id = ?", (item_id,))
        db.commit()
