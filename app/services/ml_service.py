from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

try:
    from textblob import TextBlob
except ImportError:  # pragma: no cover
    TextBlob = None  # type: ignore[assignment,misc]


INTENT_LABELS = {
    "study": "Focused Learning",
    "fitness": "Physical Training",
    "personal": "Personal Management",
    "social": "Social Time",
    "other": "General Activity",
}

EMPTY_SENTIMENT = {"polarity": 0.0, "subjectivity": 0.0}
EMPTY_CLUSTERS = {"clusters": [], "centers": []}
EMPTY_RECOMMENDATION = {
    "recommendation": "Review your plan and add one concrete task",
    "reason": "A clear task is needed before the system can rank next actions.",
}

INTENT_TRAINING_ROWS = [

    # ════════════════════════════════════════════════════════════════════════════
    # STUDY  — academics, learning, exams, subjects, reading, coding, research
    # ════════════════════════════════════════════════════════════════════════════

    # maths / formulas / equations
    ("learn the maths formulas", "study"),
    ("memorise all math formulas", "study"),
    ("learn trigonometry formulas", "study"),
    ("study algebra equations", "study"),
    ("practice integration and differentiation", "study"),
    ("revise calculus formulas", "study"),
    ("learn geometry theorems", "study"),
    ("practice solving quadratic equations", "study"),
    ("understand logarithm and exponential rules", "study"),
    ("study matrices and determinants", "study"),
    ("learn permutation and combination formulas", "study"),
    ("revise probability formulas", "study"),
    ("practice coordinate geometry", "study"),
    ("study binomial theorem", "study"),
    ("learn fourier transform formulas", "study"),

    # physics
    ("learn physics equations and derivations", "study"),
    ("revise newton laws of motion", "study"),
    ("study thermodynamics formulas", "study"),
    ("learn optics and wave equations", "study"),
    ("revise electromagnetic induction chapter", "study"),
    ("understand quantum mechanics concepts", "study"),
    ("study relativity and modern physics", "study"),
    ("learn kinematics equations", "study"),
    ("revise nuclear physics chapter", "study"),

    # chemistry
    ("complete chemistry homework", "study"),
    ("revise organic chemistry reactions", "study"),
    ("learn periodic table elements", "study"),
    ("study chemical bonding chapter", "study"),
    ("practice balancing chemical equations", "study"),
    ("learn electrochemistry formulas", "study"),
    ("revise thermochemistry and enthalpy", "study"),
    ("understand acid base reactions", "study"),

    # biology
    ("study biology for the test", "study"),
    ("learn cell division and mitosis", "study"),
    ("revise genetics and heredity chapter", "study"),
    ("study human anatomy diagrams", "study"),
    ("learn photosynthesis process", "study"),
    ("revise ecology and food chains", "study"),
    ("understand enzyme mechanisms", "study"),

    # computer science / coding
    ("practice python coding problems", "study"),
    ("finish machine learning assignment", "study"),
    ("study data structures and algorithms", "study"),
    ("learn about neural networks from the course", "study"),
    ("complete the online course module on databases", "study"),
    ("learn new programming concepts today", "study"),
    ("study operating systems concepts", "study"),
    ("practice competitive programming", "study"),
    ("learn javascript for the web dev course", "study"),
    ("build the project for the programming assignment", "study"),
    ("revise object oriented programming concepts", "study"),
    ("study computer networks chapter", "study"),
    ("learn sql and database queries", "study"),
    ("practice leetcode problems", "study"),
    ("complete the react js tutorial", "study"),
    ("learn about sorting and searching algorithms", "study"),
    ("study time and space complexity", "study"),

    # general academic
    ("revise calculus chapter for tomorrow exam", "study"),
    ("watch operating systems lecture and take notes", "study"),
    ("read textbook and summarize key ideas", "study"),
    ("prepare for placement test with mock questions", "study"),
    ("attend online class and review slides", "study"),
    ("solve past papers for the upcoming test", "study"),
    ("read and learn chapter 5 of the textbook", "study"),
    ("revise history notes and key dates", "study"),
    ("learn and memorise vocabulary for the exam", "study"),
    ("do practice problems from the worksheet", "study"),
    ("understand the derivation of key equations", "study"),
    ("revise all formulas before the exam", "study"),
    ("learn statistics formulas and practice", "study"),
    ("take notes for tomorrow's lecture", "study"),
    ("finish the assignment before the deadline", "study"),
    ("read research paper for the seminar", "study"),
    ("prepare for the viva and oral exam", "study"),
    ("revise semester syllabus before finals", "study"),
    ("complete tutorial sheet questions", "study"),
    ("study for mid sem exam tonight", "study"),
    ("complete the lab report write up", "study"),
    ("finish reading the chapter and make notes", "study"),
    ("go through previous year question papers", "study"),
    ("attend the extra class for doubt clearing", "study"),
    ("revise the topics from last week lectures", "study"),
    ("prepare a mind map for the chapter", "study"),
    ("watch youtube lecture on thermodynamics", "study"),
    ("complete the coursera assignment", "study"),
    ("finish the udemy course section", "study"),

    # ════════════════════════════════════════════════════════════════════════════
    # FITNESS  — exercise, sports, gym, health, diet, body
    # ════════════════════════════════════════════════════════════════════════════

    ("go for a morning run in the park", "fitness"),
    ("do a strength workout at the gym", "fitness"),
    ("practice yoga and stretching", "fitness"),
    ("finish cardio session before breakfast", "fitness"),
    ("walk 8000 steps this evening", "fitness"),
    ("play badminton with the college team", "fitness"),
    ("do home exercise and core training", "fitness"),
    ("track my water and workout routine", "fitness"),
    ("go for a cycle ride this evening", "fitness"),
    ("do a 30 minute hiit session at home", "fitness"),
    ("swim laps at the pool before dinner", "fitness"),
    ("complete leg day at the gym", "fitness"),
    ("go for a jog around the campus", "fitness"),
    ("do push ups and pull ups in the morning", "fitness"),
    ("complete chest and back workout", "fitness"),
    ("play cricket with friends in the evening", "fitness"),
    ("go to football practice after college", "fitness"),
    ("attend the zumba class today", "fitness"),
    ("do 20 minute meditation and pranayama", "fitness"),
    ("track calories and macros for the day", "fitness"),
    ("drink 3 litres of water today", "fitness"),
    ("go for a hike this weekend", "fitness"),
    ("do 100 squats as part of the challenge", "fitness"),
    ("follow the diet plan strictly today", "fitness"),
    ("complete the 5k run challenge", "fitness"),
    ("do stretching and cool down after gym", "fitness"),
    ("attend boxing class at the fitness centre", "fitness"),
    ("skip rope for 15 minutes in the morning", "fitness"),
    ("complete the plank challenge for the day", "fitness"),
    ("go to the gym for shoulder day", "fitness"),

    # ════════════════════════════════════════════════════════════════════════════
    # PERSONAL  — self, work, admin, home, money, planning, mental health
    # ════════════════════════════════════════════════════════════════════════════

    ("plan the project meeting for work", "personal"),
    ("clean the room and organize my desk", "personal"),
    ("buy groceries and cook dinner", "personal"),
    ("prepare slides for the client update", "personal"),
    ("journal for ten minutes before sleep", "personal"),
    ("pay bills and reply to important emails", "personal"),
    ("rest early and reset tomorrow schedule", "personal"),
    ("meditate and plan my weekly goals", "personal"),
    ("organise files and back up my laptop", "personal"),
    ("cook a healthy meal and prep for the week", "personal"),
    ("do laundry and clean the apartment", "personal"),
    ("update my resume and linkedin profile", "personal"),
    ("apply for internships online today", "personal"),
    ("sort finances and update budget spreadsheet", "personal"),
    ("book the doctor appointment", "personal"),
    ("renew id card and submit documents", "personal"),
    ("fix the bug in the work project", "personal"),
    ("reply to pending emails before noon", "personal"),
    ("set up the new laptop and install software", "personal"),
    ("plan the week and write todo list", "personal"),
    ("take a power nap to recharge", "personal"),
    ("practice deep breathing before sleep", "personal"),
    ("unplug and have a digital detox evening", "personal"),
    ("organise wardrobe and donate old clothes", "personal"),
    ("write in gratitude journal tonight", "personal"),
    ("read a self help book for 30 minutes", "personal"),
    ("do a weekly review of goals and progress", "personal"),
    ("schedule the dentist appointment", "personal"),
    ("complete the tax filing before the deadline", "personal"),
    ("update the project documentation", "personal"),
    ("finish freelance work for the client", "personal"),
    ("take a break and watch something light", "personal"),

    # ════════════════════════════════════════════════════════════════════════════
    # SOCIAL  — people, family, friends, relationships, events
    # ════════════════════════════════════════════════════════════════════════════

    ("call my parents after lunch", "social"),
    ("meet friends for dinner tonight", "social"),
    ("join a birthday party in the evening", "social"),
    ("chat with my roommate after class", "social"),
    ("visit family this weekend", "social"),
    ("hang out with friends at the cafe", "social"),
    ("talk to my mentor on a video call", "social"),
    ("go to a cousin wedding event", "social"),
    ("catch up with an old friend over coffee", "social"),
    ("attend the college farewell party", "social"),
    ("join the team lunch outing today", "social"),
    ("video call grandparents this evening", "social"),
    ("plan a surprise for a friend birthday", "social"),
    ("attend the networking event tonight", "social"),
    ("go to the college cultural fest", "social"),
    ("meet seniors for career guidance", "social"),
    ("attend the alumni meetup this weekend", "social"),
    ("help a classmate with their assignment", "social"),
    ("go on a date tonight", "social"),
    ("celebrate friend achievement at dinner", "social"),

    # ════════════════════════════════════════════════════════════════════════════
    # OTHER  — idle, entertainment, scrolling, procrastination
    # ════════════════════════════════════════════════════════════════════════════

    ("watch random videos and relax", "other"),
    ("scroll social media for an hour", "other"),
    ("play games late at night", "other"),
    ("listen to music and do nothing", "other"),
    ("binge a web series after dinner", "other"),
    ("browse the internet without a plan", "other"),
    ("waste time watching reels", "other"),
    ("take a long entertainment break", "other"),
    ("watch netflix for the rest of the evening", "other"),
    ("spend time on instagram and twitter", "other"),
    ("play video games with friends online", "other"),
    ("watch highlights and sports clips", "other"),
    ("doom scroll before sleeping", "other"),
    ("watch a movie just to pass time", "other"),
    ("browse memes and funny videos", "other"),
]


def preprocess_text(text: str) -> str:
    # Basic cleaning keeps the pipeline easy to understand and fully local.
    cleaned = (text or "").lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _recent_activity_text(user_data: dict[str, Any]) -> str:
    return " ".join(item.get("title", "") for item in user_data.get("recent_activity", []))


def _completed_recent_tasks(user_data: dict[str, Any]) -> list[dict[str, Any]]:
    return [task for task in user_data.get("recent_activity", []) if task.get("is_completed")]


@lru_cache(maxsize=1)
def _intent_pipeline() -> Pipeline:
    training_frame = pd.DataFrame(INTENT_TRAINING_ROWS, columns=["text", "label"])
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            ),
        ]
    )
    pipeline.fit(training_frame["text"], training_frame["label"])
    return pipeline


def clear_pipeline_cache() -> None:
    """Force the intent pipeline to retrain on next call (useful in tests)."""
    _intent_pipeline.cache_clear()


def predict_intent(text: str) -> dict[str, Any]:
    pipeline = _intent_pipeline()
    cleaned_text = preprocess_text(text)
    if not cleaned_text:
        return {"intent": "other", "confidence": 0.0}

    probabilities = pipeline.predict_proba([cleaned_text])[0]
    classes = pipeline.named_steps["classifier"].classes_
    best_index = int(np.argmax(probabilities))
    return {
        "intent": str(classes[best_index]),
        "confidence": round(float(probabilities[best_index]), 2),
    }


def vectorize_text(text: str) -> list[float]:
    vectorizer: TfidfVectorizer = _intent_pipeline().named_steps["tfidf"]
    vector = vectorizer.transform([preprocess_text(text)]).toarray()[0]
    return [round(float(value), 6) for value in vector]


def get_sentiment(text: str) -> dict[str, float]:
    cleaned_text = preprocess_text(text)
    if not cleaned_text or TextBlob is None:
        return dict(EMPTY_SENTIMENT)

    sentiment = TextBlob(cleaned_text).sentiment
    return {
        "polarity": round(float(sentiment.polarity), 3),
        "subjectivity": round(float(sentiment.subjectivity), 3),
    }


def _task_text(task: Any) -> str:
    if isinstance(task, str):
        return task
    title = str(task.get("title", ""))
    details = str(task.get("details", ""))
    return f"{title} {details}".strip()


def _task_title(task: Any) -> str:
    return task if isinstance(task, str) else task.get("title", "")


def recommend_task(new_task: str | dict[str, Any], past_tasks: list[str] | list[dict[str, Any]]) -> dict[str, Any] | None:
    productive_tasks = [task for task in past_tasks if _task_text(task).strip()]
    if not productive_tasks:
        return None

    task_frame = [preprocess_text(_task_text(new_task)), *[preprocess_text(_task_text(task)) for task in productive_tasks]]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(task_frame)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    if scores.size == 0:
        return None

    best_index = int(np.argmax(scores))
    best_task = productive_tasks[best_index]
    return {
        "task": _task_title(best_task),
        "similarity": round(float(scores[best_index]), 3),
    }


def _priority_score(priority: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(priority, 2)


def _category_score(category: str) -> int:
    return {"study": 3, "fitness": 2, "personal": 2, "social": 1, "other": 1}.get(category, 1)


def cluster_productivity_days(timeline_points: list[dict[str, Any]]) -> dict[str, Any]:
    if not timeline_points:
        return dict(EMPTY_CLUSTERS)

    feature_frame = pd.DataFrame(
        [
            {
                "entered": point["entered"],
                "completed": point["completed"],
                "rate": point["rate"],
            }
            for point in timeline_points
        ]
    )
    unique_rows = max(1, len(feature_frame.drop_duplicates()))
    cluster_count = min(3, len(feature_frame), unique_rows)
    kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
    labels = kmeans.fit_predict(feature_frame)

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_frame.columns)
    centers["score"] = centers["completed"] + (centers["rate"] / 100)
    ranked_ids = centers.sort_values("score").index.tolist()
    cluster_names = {cluster_id: name for cluster_id, name in zip(ranked_ids, ["low", "medium", "high"][-cluster_count:])}

    clustered_days = []
    for point, cluster_id in zip(timeline_points, labels, strict=False):
        clustered_days.append({**point, "cluster": cluster_names[int(cluster_id)]})

    return {
        "clusters": clustered_days,
        "centers": centers.drop(columns=["score"]).round(2).to_dict(orient="records"),
    }


def predict_completion_probability(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {}

    rows = []
    for item in items:
        rows.append(
            {
                "id": item["id"],
                "priority_score": _priority_score(item["priority"]),
                "item_kind": item["item_kind"],
                "category": item["category"],
                "text_length": len(preprocess_text(item.get("raw_text") or item["title"]).split()),
                "confidence": float(item.get("confidence", 0.0)),
                "label": int(bool(item["is_completed"])),
            }
        )

    frame = pd.DataFrame(rows)
    if frame["label"].nunique() < 2 or len(frame) < 4:
        return {
            str(row["id"]): round(min(0.95, 0.2 + (row["priority_score"] * 0.18) + (row["confidence"] * 0.25)), 2)
            for row in rows
        }

    feature_frame = pd.get_dummies(frame.drop(columns=["id", "label"]), columns=["item_kind", "category"])
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(feature_frame, frame["label"])
    probabilities = model.predict_proba(feature_frame)[:, 1]

    return {
        str(item_id): round(float(probability), 2)
        for item_id, probability in zip(frame["id"], probabilities, strict=False)
    }


def get_suggestions(user_data: dict[str, Any]) -> list[str]:
    sentiment = get_sentiment(_recent_activity_text(user_data))
    tasks = user_data.get("tasks", [])
    completed_tasks = _completed_recent_tasks(user_data)

    suggestions: list[str] = []
    if sentiment["polarity"] < -0.15:
        suggestions.append("Start with one easy 15-minute task to rebuild momentum")
    elif tasks:
        suggestions.append(f"Begin with '{tasks[0]['title']}' before switching contexts")

    if user_data.get("stats", {}).get("completion_rate", 0) < 50:
        suggestions.append("Shrink today to two must-finish tasks")

    if tasks:
        similar = recommend_task(tasks[0], completed_tasks)
        if similar and similar["similarity"] > 0.2:
            suggestions.append(f"Repeat the pattern from '{similar['task']}'")

    if not suggestions:
        suggestions.append("Keep your next step small and specific")
    return suggestions[:4]


def generate_insights(user_data: dict[str, Any]) -> dict[str, Any]:
    stats = user_data.get("stats", {})
    sentiment = get_sentiment(_recent_activity_text(user_data))

    problems = []
    improvements = []
    if stats.get("completion_rate", 0) < 50:
        problems.append("Recent completion rate is low")
        improvements.append("Reduce task load and finish one task fully before starting another")
    if sentiment["polarity"] < -0.15:
        problems.append("Recent activity language suggests low energy or frustration")
        improvements.append("Choose easier wins first, then move to harder work")
    if not improvements:
        improvements.append("Protect the routine that is already working")

    summary = (
        f"Primary focus is {user_data.get('primary_category', 'other')} with "
        f"{stats.get('completion_rate', 0)}% recent completion."
    )
    return {
        "summary": summary,
        "problems": problems[:4],
        "improvements": improvements[:4],
    }


def recommend_next_action(user_data: dict[str, Any]) -> dict[str, str]:
    tasks = user_data.get("tasks", [])
    if not tasks:
        return dict(EMPTY_RECOMMENDATION)

    ranked_tasks = sorted(
        tasks,
        key=lambda task: (
            _priority_score(task.get("priority", "medium")),
            _category_score(task.get("category", "other")),
        ),
        reverse=True,
    )
    best_task = ranked_tasks[0]
    completed_tasks = _completed_recent_tasks(user_data)
    similar = recommend_task(best_task, completed_tasks)
    reason = "It has the highest current priority."
    if similar and similar["similarity"] > 0.2:
        reason = f"It matches a past productive task: '{similar['task']}'."

    return {
        "recommendation": f"Work on '{best_task['title']}' next",
        "reason": reason,
    }