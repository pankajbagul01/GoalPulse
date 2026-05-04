# GoalPulse

> A local-first, ML-powered goal tracker that measures your **focus**, **drift**, and **consistency** over time — with zero cloud dependency.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-local--db-lightgrey?logo=sqlite)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

GoalPulse lets you log **tasks**, **habits**, and **events** in plain English. The ML pipeline automatically classifies each entry into a focus category (study, fitness, personal, social) using a TF-IDF + Logistic Regression pipeline, then tracks four behavioral scores week-over-week:

| Score | What it measures |
|---|---|
| **Stability** | How consistently you complete what you plan |
| **Drift** | How scattered your focus is across categories |
| **Deviation** | How often high-priority items conflict with your stated goal |
| **Streak** | Consecutive days with at least one completed task |

The dashboard surfaces AI-generated insights, completion predictions, productivity cluster analysis, and next-action recommendations — all running **locally** with no external API required.

---

## Features

- Log **Tasks**, **Habits**, and **Events** with priority levels in plain English
- Automatic **intent classification** via TF-IDF + Logistic Regression (scikit-learn)
- **Sentiment analysis** on recent activity text (TextBlob)
- **KMeans clustering** of daily productivity patterns into low / medium / high days
- **Completion probability prediction** for each open item
- **Conflict detection** — alerts when logged items contradict your primary focus
- **Streak tracking** — consecutive days with completed tasks
- Weekly progress chart and monthly calendar view
- Habit recurrence (daily / weekly / monthly)
- SQLite-backed persistence with indexed queries
- CORS-ready API for local tooling
- Zero external runtime dependencies — pure Python stdlib HTTP server + SQLite
- Optional OpenAI fallback for intent classification

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python stdlib (`http.server`, `sqlite3`) |
| ML / Analysis | scikit-learn, numpy, pandas, TextBlob |
| Frontend | Vanilla HTML + CSS + JavaScript |
| Database | SQLite (with performance indexes) |

---

## Project structure

```
goalpulse/
├── run.py                  # Entry point
├── requirements.txt
├── .gitignore
├── app/
│   ├── __init__.py         # Path constants
│   ├── db.py               # SQLite CRUD + indexes
│   ├── server.py           # HTTP request handler (CORS-ready)
│   ├── templates/
│   │   └── index.html      # Dashboard UI
│   ├── static/
│   │   ├── css/styles.css
│   │   └── js/app.js
│   └── services/
│       ├── intent.py       # Intent extraction wrapper
│       ├── ml_service.py   # Core ML pipeline
│       ├── analysis.py     # Timeline scoring & insights
│       └── planner.py      # Planner logic & calendar
└── tests/
    └── test_analysis.py
```

---

## Setup

**Requirements:** Python 3.11+

```bash
# 1. Clone the repo
git clone https://github.com/pankajbagul01/goalpulse.git
cd goalpulse

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python run.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## Optional: LLM features

GoalPulse works fully offline. If you want OpenAI-backed intent classification as a fallback, create a `.env` file:

```
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=12
```

---

## Running tests

```bash
pytest tests/
```

---

## How the ML pipeline works

```
User input (plain text)
        │
        ▼
  Preprocessing          lowercase, strip punctuation
        │
        ▼
  TF-IDF Vectorizer      bigrams, cached pipeline (lru_cache)
        │
        ▼
  Logistic Regression    predicts category + confidence
        │
        ▼
  Intent label           "Focused Learning", "Physical Training", etc.
        │
        ▼
  Embedding stored       TF-IDF vector saved per item in SQLite
        │
        ▼
  Timeline Analysis      stability · drift · deviation · streak · clusters · predictions
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/dashboard` | Full dashboard payload |
| GET | `/api/items` | All stored items |
| POST | `/api/items` | Add a new item |
| PATCH | `/api/items/:id` | Update completion or priority |
| DELETE | `/api/items/:id` | Delete an item |
| POST | `/api/seed` | Reset all data |

---

## Scores explained

**Stability** — mean task completion rate across the last 7 days. Days with no tasks are excluded.

**Drift** — percentage of recent items that fall outside your primary category. Lower is more focused.

**Deviation** — triggered when a high-priority logged item conflicts with your current focus (e.g. logging a "Movie night" while your goal is exam prep). Reflects average conflict severity.

**Streak** — consecutive days ending today on which at least one task was completed. Resets on any day with no completions.

---

## License

MIT
