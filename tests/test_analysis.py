import unittest
from datetime import date

from app.services.analysis import analyze_timeline
from app.services.intent import extract_intent
from app.services.planner import occurs_on


def make_item(
    item_id,
    item_kind,
    title,
    details,
    created_at,
    *,
    priority="medium",
    is_completed=False,
    scheduled_date=None,
    scheduled_time=None,
    repeat_frequency="none",
    completed_at=None,
):
    raw_text = f"{title}. {details}".strip(". ").strip()
    intent = extract_intent(raw_text or title)
    return {
        "id": item_id,
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
        "created_at": created_at,
    }


class AnalysisTests(unittest.TestCase):
    def test_high_priority_open_task_becomes_active_focus(self):
        items = [
            make_item(
                1,
                "TASK",
                "Study algorithms",
                "Finish graph revision",
                "2026-04-24T08:00:00+05:30",
                priority="high",
                scheduled_date="2026-04-24",
            ),
            make_item(
                2,
                "EVENT",
                "Movie night",
                "Watch a movie with friends",
                "2026-04-25T20:00:00+05:30",
                priority="low",
                scheduled_date="2026-04-25",
                scheduled_time="20:00",
            ),
        ]

        analysis = analyze_timeline(items)

        self.assertEqual(analysis["summary"]["stated_goal"], "Study algorithms")
        self.assertEqual(analysis["summary"]["primary_category"], "study")

    def test_deviation_alert_is_generated_from_logged_item(self):
        items = [
            make_item(
                1,
                "TASK",
                "OS exam preparation",
                "Revise process scheduling",
                "2026-04-25T09:00:00+05:30",
                priority="high",
                scheduled_date="2026-04-25",
            ),
            make_item(
                2,
                "EVENT",
                "Movie outing",
                "Watch a late night show",
                "2026-04-26T19:00:00+05:30",
                priority="high",
                scheduled_date="2026-04-26",
                scheduled_time="19:00",
            ),
        ]

        analysis = analyze_timeline(items)

        self.assertEqual(len(analysis["alerts"]), 1)
        self.assertEqual(analysis["alerts"][0]["timestamp"], "2026-04-26T19:00:00+05:30")
        self.assertGreater(analysis["scores"]["deviation"], 0)

    def test_stability_uses_weekly_completion_ratio(self):
        items = [
            make_item(1, "TASK", "Task A", "Alpha", "2026-04-20T09:00:00+05:30", priority="medium", scheduled_date="2026-04-20", is_completed=True, completed_at="2026-04-20T18:00:00+05:30"),
            make_item(2, "TASK", "Task B", "Beta", "2026-04-21T09:00:00+05:30", priority="medium", scheduled_date="2026-04-21", is_completed=False),
            make_item(3, "TASK", "Task C", "Gamma", "2026-04-22T09:00:00+05:30", priority="medium", scheduled_date="2026-04-22", is_completed=True, completed_at="2026-04-22T20:00:00+05:30"),
        ]

        analysis = analyze_timeline(items)

        self.assertGreater(analysis["scores"]["stability"], 0)
        self.assertLessEqual(analysis["scores"]["stability"], 100)

    def test_daily_habit_occurs_on_future_day(self):
        habit = make_item(
            3,
            "HABIT",
            "Morning workout",
            "Strength training",
            "2026-04-25T07:00:00+05:30",
            repeat_frequency="daily",
            scheduled_date="2026-04-25",
        )

        self.assertTrue(occurs_on(habit, date(2026, 4, 28)))

    def test_empty_timeline_is_safe(self):
        analysis = analyze_timeline([])
        self.assertEqual(analysis["scores"]["stability"], 0)
        self.assertEqual(analysis["summary"]["entry_count"], 0)


if __name__ == "__main__":
    unittest.main()


class StreakTests(unittest.TestCase):
    def test_streak_is_zero_when_no_completions(self):
        items = [
            make_item(1, "TASK", "Study Python", "", "2026-04-25T09:00:00+05:30",
                      priority="medium", scheduled_date="2026-04-25"),
        ]
        analysis = analyze_timeline(items)
        self.assertEqual(analysis["scores"]["streak"], 0)

    def test_streak_counts_consecutive_completed_days(self):
        from datetime import date, timedelta
        today_str = date.today().isoformat()
        yesterday_str = (date.today() - timedelta(days=1)).isoformat()
        items = [
            make_item(1, "TASK", "Task today", "", f"{today_str}T09:00:00+05:30",
                      priority="medium", scheduled_date=today_str,
                      is_completed=True, completed_at=f"{today_str}T18:00:00+05:30"),
            make_item(2, "TASK", "Task yesterday", "", f"{yesterday_str}T09:00:00+05:30",
                      priority="medium", scheduled_date=yesterday_str,
                      is_completed=True, completed_at=f"{yesterday_str}T18:00:00+05:30"),
        ]
        analysis = analyze_timeline(items)
        self.assertGreaterEqual(analysis["scores"]["streak"], 2)
