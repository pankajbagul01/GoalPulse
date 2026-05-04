from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import STATIC_DIR, TEMPLATES_DIR
from .db import clear_items, delete_item, ensure_database, fetch_items, insert_item, update_item
from .services.analysis import analyze_timeline
from .services.intent import extract_intent
from .services.planner import iso_now, local_today, planner_summary


def _clean_text(value: str | None) -> str:
    return (value or "").strip()


def _boolish(value) -> bool:
    return value in {True, 1, "1", "true", "True", "on", "yes"}


def _serialize_item(payload: dict, created_at: str | None = None) -> dict:
    item_kind = payload.get("item_kind", "TASK")
    title = _clean_text(payload.get("title"))
    details = _clean_text(payload.get("details"))
    raw_text = f"{title}. {details}".strip(". ").strip()
    intent = extract_intent(raw_text or title)

    priority = payload.get("priority", "medium")
    if priority not in {"low", "medium", "high"}:
        priority = "medium"

    scheduled_date = _clean_text(payload.get("scheduled_date")) or None
    scheduled_time = _clean_text(payload.get("scheduled_time")) or None
    repeat_frequency = "none"

    if item_kind == "TASK":
        scheduled_date = local_today().isoformat()
        scheduled_time = None
    elif item_kind == "EVENT":
        if not scheduled_date:
            scheduled_date = local_today().isoformat()
    elif item_kind == "HABIT":
        scheduled_date = local_today().isoformat()
        scheduled_time = None
        repeat_frequency = payload.get("repeat_frequency", "daily")
        if repeat_frequency not in {"daily", "weekly", "monthly"}:
            repeat_frequency = "daily"

    is_completed = _boolish(payload.get("is_completed"))
    completed_at = iso_now() if is_completed else None

    return {
        "item_kind": item_kind,
        "title": title,
        "details": details,
        "raw_text": raw_text or title,
        "intent_label": intent["intent_label"],
        "category": intent["category"],
        "confidence": intent["confidence"],
        "embedding": intent["embedding"],
        "priority": priority,
        "is_flagged": False,
        "is_completed": is_completed,
        "scheduled_date": scheduled_date,
        "scheduled_time": scheduled_time,
        "repeat_frequency": repeat_frequency,
        "completed_at": completed_at,
        "created_at": created_at or iso_now(),
    }


def _dashboard_payload() -> dict:
    items = fetch_items()
    return {
        "items": items,
        "planner": planner_summary(items),
        "analysis": analyze_timeline(items),
    }


class GoalPulseHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._add_cors_headers()
        self.end_headers()

    def _add_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._serve_file(TEMPLATES_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            file_path = STATIC_DIR / path.removeprefix("/static/")
            self._serve_file(file_path)
            return
        if path == "/api/items":
            self._send_json({"items": fetch_items()})
            return
        if path == "/api/dashboard":
            self._send_json(_dashboard_payload())
            return
        if path == "/api/health":
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/items":
            payload = self._read_json()
            item = _serialize_item(payload)

            if not item["title"]:
                self._send_json({"error": "title is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            if item["item_kind"] not in {"TASK", "HABIT", "EVENT"}:
                self._send_json({"error": "item_kind must be TASK, HABIT, or EVENT"}, status=HTTPStatus.BAD_REQUEST)
                return
            if item["item_kind"] == "EVENT" and not item["scheduled_time"]:
                self._send_json({"error": "event time is required"}, status=HTTPStatus.BAD_REQUEST)
                return

            item_id = insert_item(item)
            self._send_json({"id": item_id, "item": {**item, "id": item_id}, "dashboard": _dashboard_payload()}, status=HTTPStatus.CREATED)
            return

        if path == "/api/seed":
            clear_items()
            self._send_json({"ok": True, "dashboard": _dashboard_payload()}, status=HTTPStatus.CREATED)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_PATCH(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not path.startswith("/api/items/"):
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            item_id = int(path.rsplit("/", 1)[-1])
        except ValueError:
            self._send_json({"error": "Invalid item id"}, status=HTTPStatus.BAD_REQUEST)
            return

        payload = self._read_json()
        fields = {}

        if "is_completed" in payload:
            is_completed = _boolish(payload["is_completed"])
            fields["is_completed"] = int(is_completed)
            fields["completed_at"] = iso_now() if is_completed else None
        if "priority" in payload and payload["priority"] in {"low", "medium", "high"}:
            fields["priority"] = payload["priority"]

        update_item(item_id, fields)
        self._send_json({"ok": True, "dashboard": _dashboard_payload()})

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not path.startswith("/api/items/"):
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            item_id = int(path.rsplit("/", 1)[-1])
        except ValueError:
            self._send_json({"error": "Invalid item id"}, status=HTTPStatus.BAD_REQUEST)
            return

        delete_item(item_id)
        self._send_json({"ok": True, "dashboard": _dashboard_payload()})

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._add_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path: Path, content_type: str | None = None) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        data = file_path.read_bytes()
        guessed_type = content_type or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", guessed_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_server(host: str = "127.0.0.1", port: int = 5000) -> None:
    ensure_database()
    server = ThreadingHTTPServer((host, port), GoalPulseHandler)
    print(f"GoalPulse running at http://{host}:{port}")
    server.serve_forever()
