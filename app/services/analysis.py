from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
from statistics import mean
from typing import Any

from .ml_service import (
    EMPTY_CLUSTERS,
    EMPTY_SENTIMENT,
    cluster_productivity_days,
    generate_insights,
    get_sentiment,
    get_suggestions,
    predict_completion_probability,
    recommend_next_action,
)
from .planner import item_start_date, local_today, weekly_progress


CONFLICT_MAP = {
    "study": {"social", "other"},
    "fitness": {"other"},
    "personal": {"other"},
}

PRIORITY_WEIGHT = {
    "low": 1.0,
    "medium": 1.3,
    "high": 1.7,
}

KIND_WEIGHT = {
    "TASK": 1.3,
    "EVENT": 1.15,
    "HABIT": 1.0,
}

EMPTY_USER_DATA = {
    "primary_category": "other",
    "focus_item": None,
    "tasks": [],
    "habits": [],
    "recent_activity": [],
    "missed_tasks": [],
    "alerts": [],
    "priorities": {},
    "stats": {"entry_count": 0, "completion_rate": 0, "stability": 0, "drift": 0},
}


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _recent_items(items: list[dict], days: int = 7, today: date | None = None) -> list[dict]:
    today = today or local_today()
    cutoff = today - timedelta(days=days - 1)
    return [item for item in items if item_start_date(item) >= cutoff]


def _primary_category(items: list[dict]) -> str:
    if not items:
        return "other"

    scores: defaultdict[str, float] = defaultdict(float)
    for item in items:
        category = item["category"]
        if category == "other":
            continue
        score = PRIORITY_WEIGHT[item["priority"]] * KIND_WEIGHT[item["item_kind"]]
        if not item["is_completed"]:
            score += 0.5
        scores[category] += score

    if not scores:
        return "other"
    return max(scores.items(), key=lambda pair: pair[1])[0]


def _active_focus_item(items: list[dict], primary_category: str) -> dict | None:
    open_priority = [item for item in items if not item["is_completed"] and item["item_kind"] in {"TASK", "EVENT"}]
    if not open_priority:
        open_priority = [item for item in items if not item["is_completed"]]
    if not open_priority:
        return None

    ranked = sorted(
        open_priority,
        key=lambda item: (
            1 if item["category"] == primary_category else 0,
            PRIORITY_WEIGHT[item["priority"]],
            KIND_WEIGHT[item["item_kind"]],
            item["created_at"],
        ),
        reverse=True,
    )
    return ranked[0]


def _weekly_score_summary(items: list[dict]) -> tuple[int, int]:
    weekly = weekly_progress(items)
    rates = [point["rate"] for point in weekly if point["entered"] > 0]
    completion_stability = round(_safe_mean(rates)) if rates else 0

    week_items = _recent_items(items)
    primary = _primary_category(week_items)
    if primary == "other":
        drift = 0
    else:
        aligned = [item for item in week_items if item["category"] == primary]
        drift = round((1 - (len(aligned) / len(week_items))) * 100) if week_items else 0
    return completion_stability, drift


def _task_view(item: dict, completion_predictions: dict[str, float]) -> dict:
    return {
        "title": item["title"],
        "details": item["details"],
        "priority": item["priority"],
        "category": item["category"],
        "scheduled_date": item.get("scheduled_date"),
        "is_completed": item["is_completed"],
        "completion_probability": completion_predictions.get(str(item["id"]), 0.0),
    }


def _habit_view(item: dict) -> dict:
    return {
        "title": item["title"],
        "details": item["details"],
        "repeat_frequency": item["repeat_frequency"],
        "category": item["category"],
        "is_completed": item["is_completed"],
    }


def _activity_view(item: dict) -> dict:
    return {
        "title": item["title"],
        "details": item["details"],
        "item_kind": item["item_kind"],
        "category": item["category"],
        "priority": item["priority"],
        "is_completed": item["is_completed"],
    }


def _focus_view(focus_item: dict | None) -> dict | None:
    if not focus_item:
        return None
    return {
        "title": focus_item["title"],
        "priority": focus_item["priority"],
        "item_kind": focus_item["item_kind"],
    }


def _timeline_points(items: list[dict]) -> list[dict]:
    return [
        {
            "label": point["label"],
            "entered": point["entered"],
            "completed": point["completed"],
            "rate": point["rate"],
        }
        for point in weekly_progress(items)
    ]


def _empty_analysis_payload() -> dict:
    user_data = dict(EMPTY_USER_DATA)
    return {
        "scores": {"stability": 0, "drift": 0, "deviation": 0, "streak": 0},
        "summary": {
            "primary_category": "other",
            "stated_goal": None,
            "entry_count": 0,
            "focus_kind": None,
            "primary_intent_rule": "Primary intent appears after enough items are logged.",
            "focus_rule": "Active focus appears once an open task, event, or habit exists.",
        },
        "alerts": [],
        "timeline_points": [],
        "clusters": dict(EMPTY_CLUSTERS),
        "predictions": {},
        "sentiment": dict(EMPTY_SENTIMENT),
        "suggestions": get_suggestions(user_data),
        "insights": generate_insights(user_data),
        "recommendation": recommend_next_action(user_data),
    }


def _build_user_data(
    items: list[dict],
    *,
    primary_category: str,
    stability: int,
    drift: int,
    focus_item: dict | None,
    alerts: list[dict],
    completion_predictions: dict[str, float],
) -> dict:
    recent = sorted(items, key=lambda item: item["created_at"], reverse=True)[:8]
    tasks = [item for item in items if item["item_kind"] == "TASK" and not item["is_completed"]]
    habits = [item for item in items if item["item_kind"] == "HABIT"]
    priorities = Counter(item["priority"] for item in items if not item["is_completed"])
    completed_recently = sum(1 for item in recent if item["is_completed"])
    completion_rate = round((completed_recently / len(recent)) * 100) if recent else 0
    missed_tasks = [
        {
            "title": item["title"],
            "priority": item["priority"],
            "scheduled_date": item.get("scheduled_date"),
        }
        for item in tasks
        if item.get("scheduled_date") and item["scheduled_date"] < local_today().isoformat()
    ]

    return {
        "primary_category": primary_category,
        "focus_item": _focus_view(focus_item),
        "tasks": [_task_view(item, completion_predictions) for item in tasks[:10]],
        "habits": [_habit_view(item) for item in habits[:8]],
        "recent_activity": [_activity_view(item) for item in recent],
        "missed_tasks": missed_tasks[:6],
        "alerts": alerts[-4:],
        "priorities": dict(priorities),
        "stats": {
            "entry_count": len(items),
            "completion_rate": completion_rate,
            "stability": stability,
            "drift": drift,
        },
    }


def _completion_streak(items: list[dict], today: date | None = None) -> int:
    """Count consecutive days ending today on which at least one TASK was completed."""
    today = today or local_today()
    streak = 0
    current_day = today
    while True:
        day_tasks = [
            item
            for item in items
            if item["item_kind"] == "TASK"
            and (item.get("completed_at") or "")[:10] == current_day.isoformat()
            and item["is_completed"]
        ]
        if not day_tasks:
            break
        streak += 1
        current_day -= timedelta(days=1)
    return streak


def analyze_timeline(items: list[dict]) -> dict:
    if not items:
        return _empty_analysis_payload()

    recent = _recent_items(items)
    primary_category = _primary_category(recent or items)
    focus_item = _active_focus_item(items, primary_category)
    stability, drift = _weekly_score_summary(items)

    alerts = []
    deviation_values = []
    focus_title = focus_item["title"] if focus_item else "current plan"
    for item in sorted(items, key=lambda value: (value["created_at"], value["id"])):
        if focus_item and item["id"] == focus_item["id"]:
            continue

        category_conflict = focus_item and item["category"] in CONFLICT_MAP.get(focus_item["category"], set())
        primary_mismatch = primary_category != "other" and item["category"] not in {primary_category, "other"}
        high_impact = item["priority"] == "high" or item["item_kind"] == "EVENT"

        if category_conflict or (primary_mismatch and high_impact):
            severity = "high" if category_conflict or item["priority"] == "high" else "medium"
            alerts.append(
                {
                    "timestamp": item["created_at"],
                    "severity": severity,
                    "message": (
                        f"Logged '{item['title']}' under '{item['category']}', which conflicts with the current "
                        f"focus on '{focus_title}'."
                    ),
                }
            )
            deviation_values.append(90 if severity == "high" else 65)

    deviation = round(_safe_mean(deviation_values)) if deviation_values else 0

    timeline_points = _timeline_points(items)
    clusters = cluster_productivity_days(timeline_points)
    completion_predictions = predict_completion_probability(items)
    streak = _completion_streak(items)

    user_data = _build_user_data(
        items,
        primary_category=primary_category,
        stability=stability,
        drift=drift,
        focus_item=focus_item,
        alerts=alerts,
        completion_predictions=completion_predictions,
    )
    sentiment = get_sentiment(" ".join(item["title"] for item in user_data["recent_activity"]))

    return {
        "scores": {
            "stability": stability,
            "drift": drift,
            "deviation": deviation,
            "streak": streak,
        },
        "summary": {
            "primary_category": primary_category,
            "stated_goal": focus_item["title"] if focus_item else None,
            "entry_count": len(items),
            "focus_kind": focus_item["item_kind"] if focus_item else None,
            "primary_intent_rule": "Primary intent is learned from categorized task history and recent weighted activity.",
            "focus_rule": "Active focus is the highest-priority open task or event in the primary category.",
        },
        "alerts": alerts[-6:],
        "timeline_points": timeline_points,
        "clusters": clusters,
        "predictions": completion_predictions,
        "sentiment": sentiment,
        "suggestions": get_suggestions(user_data),
        "insights": generate_insights(user_data),
        "recommendation": recommend_next_action(user_data),
    }
