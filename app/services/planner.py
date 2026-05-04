from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta


def local_now() -> datetime:
    return datetime.now().astimezone()


def local_today() -> date:
    return local_now().date()


def iso_now() -> str:
    return local_now().isoformat(timespec="seconds")


def item_start_date(item: dict) -> date:
    source = item.get("scheduled_date") or item["created_at"][:10]
    return date.fromisoformat(source)


def occurs_on(item: dict, target_day: date) -> bool:
    start_day = item_start_date(item)

    if item["item_kind"] in {"TASK", "EVENT"}:
        return start_day == target_day

    if target_day < start_day:
        return False

    frequency = item.get("repeat_frequency", "none")
    if frequency == "daily":
        return True
    if frequency == "weekly":
        return start_day.weekday() == target_day.weekday()
    if frequency == "monthly":
        return start_day.day == target_day.day
    return False


def item_sort_key(item: dict) -> tuple:
    date_part = item.get("scheduled_date") or item["created_at"][:10]
    time_part = item.get("scheduled_time") or ("23:59" if item["item_kind"] != "HABIT" else "06:00")
    return (date_part, time_part, item["created_at"], item["id"])


def calendar_days(items: list[dict], month_date: date | None = None) -> list[dict]:
    month_date = month_date or local_today()
    year = month_date.year
    month = month_date.month
    _, days_in_month = calendar.monthrange(year, month)
    today = local_today()

    days = []
    for day_number in range(1, days_in_month + 1):
        current_day = date(year, month, day_number)
        day_items = [item for item in items if occurs_on(item, current_day)]
        completed_count = sum(1 for item in day_items if item["is_completed"])
        days.append(
            {
                "date": current_day.isoformat(),
                "day": day_number,
                "weekday": current_day.strftime("%a"),
                "count": len(day_items),
                "is_today": current_day == today,
                "completed_count": completed_count if current_day == today else 0,
            }
        )
    return days


def today_items(items: list[dict], today: date | None = None) -> list[dict]:
    today = today or local_today()
    visible = [item for item in items if item["item_kind"] != "EVENT" and occurs_on(item, today) and not item["is_completed"]]
    return sorted(visible, key=item_sort_key)


def completed_items(items: list[dict], today: date | None = None) -> list[dict]:
    today = today or local_today()
    visible = [item for item in items if item["is_completed"] and (item["item_kind"] == "HABIT" or occurs_on(item, today))]
    return sorted(visible, key=lambda item: (item["completed_at"] or item["created_at"], item["id"]), reverse=True)


def upcoming_events(items: list[dict], today: date | None = None) -> list[dict]:
    today = today or local_today()
    events = [item for item in items if item["item_kind"] == "EVENT" and item_start_date(item) >= today]
    return sorted(events, key=item_sort_key)


def habits(items: list[dict]) -> list[dict]:
    return sorted([item for item in items if item["item_kind"] == "HABIT" and not item["is_completed"]], key=item_sort_key)


def _daily_weekly_point(items: list[dict], day: date) -> dict:
    day_items = [item for item in items if item["item_kind"] == "TASK" and occurs_on(item, day)]
    entered = len(day_items)
    completed = sum(1 for item in day_items if item["is_completed"])
    rate = round((completed / entered) * 100) if entered else 0
    return {
        "date": day.isoformat(),
        "label": day.strftime("%a"),
        "entered": entered,
        "completed": completed,
        "rate": rate,
    }


def weekly_progress(items: list[dict], today: date | None = None) -> list[dict]:
    today = today or local_today()
    start = today - timedelta(days=6)
    return [_daily_weekly_point(items, start + timedelta(days=offset)) for offset in range(7)]


def planner_summary(items: list[dict], today: date | None = None) -> dict:
    today = today or local_today()
    todays = today_items(items, today)
    completed = completed_items(items, today)
    all_events = upcoming_events(items, today)
    all_habits = habits(items)
    weekly = weekly_progress(items, today)
    completed_today = sum(1 for item in completed if (item.get("completed_at") or "")[:10] == today.isoformat())
    high_priority_open = sum(1 for item in items if item["priority"] == "high" and not item["is_completed"])

    return {
        "today_items": todays,
        "completed_items": completed[:8],
        "upcoming_events": all_events[:6],
        "habits": all_habits,
        "weekly_progress": weekly,
        "stats": {
            "today_total": len(todays) + len([item for item in completed if occurs_on(item, today)]),
            "today_completed": completed_today,
            "high_priority_open": high_priority_open,
            "event_count": len(all_events),
        },
        "month": {
            "label": today.strftime("%B %Y"),
            "days": calendar_days(items, today),
        },
    }
