"use strict";

/* ═══════════════════════════════════════
   GoalPulse — app.js
   Chart.js · Dark/Light toggle · Toasts
   Category colors · Confidence badge
═══════════════════════════════════════ */

// ── DOM refs ────────────────────────────
const itemForm      = document.getElementById("itemForm");
const itemKind      = document.getElementById("itemKind");
const priorityField = document.getElementById("priorityField");
const dateField     = document.getElementById("dateField");
const timeField     = document.getElementById("timeField");
const repeatField   = document.getElementById("repeatField");
const submitBtn     = document.getElementById("submitBtn");
const submitLabel   = document.getElementById("submitLabel");
const themeToggle   = document.getElementById("themeToggle");
const toastContainer= document.getElementById("toastContainer");

const calendarGrid  = document.getElementById("calendarGrid");
const alertsList    = document.getElementById("alertsList");
const todayList     = document.getElementById("todayList");
const completedList = document.getElementById("completedList");
const habitList     = document.getElementById("habitList");
const eventList     = document.getElementById("eventList");

const scoreTargets = {
  stability: document.getElementById("stabilityScore"),
  drift:     document.getElementById("driftScore"),
  deviation: document.getElementById("deviationScore"),
  streak:    document.getElementById("streakScore"),
};
const scoreRings = {
  stability: document.getElementById("stabilityRing"),
  drift:     document.getElementById("driftRing"),
  deviation: document.getElementById("deviationRing"),
  streak:    document.getElementById("streakRing"),
};
const summaryTargets = {
  primaryIntent:       document.getElementById("primaryIntent"),
  activeGoal:          document.getElementById("activeGoal"),
  entryCount:          document.getElementById("entryCount"),
  monthLabel:          document.getElementById("monthLabel"),
  todayProgress:       document.getElementById("todayProgress"),
  highPriorityOpenCount: document.getElementById("highPriorityOpenCount"),
  recommendation:      document.getElementById("recommendation"),
};

// Insights
const insightSummary   = document.getElementById("insightSummary");
const suggestionsList  = document.getElementById("suggestionsList");
const improvementsList = document.getElementById("improvementsList");
const polarityBar      = document.getElementById("polarityBar");
const subjectivityBar  = document.getElementById("subjectivityBar");
const polarityVal      = document.getElementById("polarityVal");
const subjectivityVal  = document.getElementById("subjectivityVal");

// Chart.js instance
let weeklyChartInstance = null;

// ── Theme ────────────────────────────────
const THEME_KEY = "goalpulse-theme";

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  document.getElementById("themeToggle").querySelector(".theme-icon").textContent =
    theme === "dark" ? "🌙" : "☀️";
  localStorage.setItem(THEME_KEY, theme);
  // redraw chart if exists so colors update
  if (weeklyChartInstance) {
    weeklyChartInstance.options.scales.x.ticks.color = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
    weeklyChartInstance.options.scales.y.ticks.color = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
    weeklyChartInstance.options.scales.x.grid.color  = getComputedStyle(document.documentElement).getPropertyValue("--border").trim();
    weeklyChartInstance.options.scales.y.grid.color  = getComputedStyle(document.documentElement).getPropertyValue("--border").trim();
    weeklyChartInstance.update();
  }
}

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  applyTheme(current === "dark" ? "light" : "dark");
});

// Restore saved theme
applyTheme(localStorage.getItem(THEME_KEY) || "light");

// ── Toast ────────────────────────────────
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  const icon = { success: "✓", error: "✕", info: "ℹ" }[type] || "ℹ";
  toast.innerHTML = `<span style="font-size:1rem">${icon}</span> ${escapeHtml(message)}`;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("removing");
    toast.addEventListener("animationend", () => toast.remove());
  }, 3200);
}

// ── Helpers ──────────────────────────────
function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatTimestamp(value) {
  if (!value) return "Unscheduled";
  const d = new Date(value);
  return isNaN(d.getTime()) ? value : d.toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });
}

function formatSchedule(item) {
  if (item.item_kind === "TASK")  return "Due today";
  if (item.item_kind === "HABIT") return "Habit";
  if (!item.scheduled_date)       return "No schedule";
  return item.scheduled_time ? `${item.scheduled_date} at ${item.scheduled_time}` : item.scheduled_date;
}

function formatRepeat(item) {
  if (item.item_kind !== "HABIT") return "";
  return { daily: "Daily", weekly: "Weekly", monthly: "Monthly" }[item.repeat_frequency] || "";
}

function priorityOptions(current) {
  return ["high", "medium", "low"]
    .map(v => `<option value="${v}" ${v === current ? "selected" : ""}>${v}</option>`)
    .join("");
}

function categoryClass(cat) {
  return `cat-${cat || "other"}`;
}

function confidenceBadge(confidence) {
  if (confidence === undefined || confidence === null) return "";
  const pct = Math.round(confidence * 100);
  if (confidence >= 0.5) return `<span class="pill" style="background:var(--green-soft);color:var(--green)">${pct}% sure</span>`;
  return `<span class="confidence-badge low-confidence" title="Low confidence — category may be inaccurate">⚠ ${pct}% sure</span>`;
}

// ── Item Card ────────────────────────────
function itemCard(item, options = {}) {
  const compact = options.compact || false;
  const catClass = categoryClass(item.category);
  return `
    <article class="planner-card ${item.is_completed ? "completed" : ""}">
      <div class="planner-top">
        <label class="checkbox-pill">
          <input type="checkbox" data-action="complete" data-id="${item.id}" ${item.is_completed ? "checked" : ""}>
          <span>${item.is_completed ? "Done" : "Open"}</span>
        </label>
        <div class="pill-row">
          <span class="cat-dot ${catClass}"></span>
          <span class="pill ${item.item_kind.toLowerCase()}">${item.item_kind}</span>
          ${item.item_kind !== "HABIT" ? `<span class="priority-badge ${item.priority}">${item.priority}</span>` : ""}
          ${confidenceBadge(item.confidence)}
          <button class="delete-button" data-action="delete" data-id="${item.id}" type="button">Delete</button>
        </div>
      </div>
      <div class="planner-main">
        <h3>${escapeHtml(item.title)}</h3>
        <p>${escapeHtml(item.details || "No additional notes.")}</p>
      </div>
      ${item.item_kind !== "HABIT" ? `
        <label class="priority-select" style="display:inline-grid;gap:4px;font-size:0.82rem;margin-top:10px;">
          Priority
          <select data-action="priority" data-id="${item.id}" style="width:auto">
            ${priorityOptions(item.priority)}
          </select>
        </label>` : ""}
      <div class="planner-meta ${compact ? "compact-meta" : ""}">
        <span class="${catClass}" style="color:var(--cat-${item.category || "other"})"><strong>Category:</strong> ${escapeHtml(item.category)}</span>
        <span><strong>Intent:</strong> ${escapeHtml(item.intent_label)}</span>
        <span><strong>Schedule:</strong> ${escapeHtml(formatSchedule(item))}</span>
        ${item.item_kind === "HABIT" ? `<span><strong>Repeat:</strong> ${formatRepeat(item)}</span>` : ""}
        <span><strong>Logged:</strong> ${escapeHtml(formatTimestamp(item.created_at))}</span>
      </div>
    </article>
  `;
}

// ── Renderers ────────────────────────────
function renderCollection(target, items, emptyMessage, options = {}) {
  if (!items.length) {
    target.className = "planner-list empty-state";
    target.textContent = emptyMessage;
    return;
  }
  target.className = "planner-list";
  target.innerHTML = items.map(item => itemCard(item, options)).join("");
}

function renderAlerts(alerts) {
  if (!alerts.length) {
    alertsList.className = "stack-list empty-state";
    alertsList.textContent = "No alerts yet.";
    return;
  }
  alertsList.className = "stack-list";
  alertsList.innerHTML = alerts.map(alert => `
    <article class="alert-item ${alert.severity}">
      <div class="alert-top">
        <strong>${alert.severity.toUpperCase()} deviation</strong>
        <span>${escapeHtml(formatTimestamp(alert.timestamp))}</span>
      </div>
      <div>${escapeHtml(alert.message)}</div>
    </article>
  `).join("");
}

function renderChart(points) {
  const canvas = document.getElementById("weeklyChart");
  const style = getComputedStyle(document.documentElement);
  const mutedColor = style.getPropertyValue("--muted").trim();
  const borderColor = style.getPropertyValue("--border").trim();
  const greenColor = style.getPropertyValue("--green").trim();
  const borderStrongColor = style.getPropertyValue("--border-strong").trim();

  const labels    = points.map(p => p.label);
  const completed = points.map(p => p.completed);
  const entered   = points.map(p => p.entered);
  const rates     = points.map(p => p.rate);

  if (weeklyChartInstance) weeklyChartInstance.destroy();

  weeklyChartInstance = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Completed",
          data: completed,
          backgroundColor: greenColor,
          borderRadius: 8,
          borderSkipped: false,
          barPercentage: 0.55,
          order: 2,
        },
        {
          label: "Entered",
          data: entered,
          backgroundColor: borderStrongColor,
          borderRadius: 8,
          borderSkipped: false,
          barPercentage: 0.55,
          order: 3,
        },
        {
          label: "Rate %",
          data: rates,
          type: "line",
          borderColor: greenColor,
          backgroundColor: "transparent",
          pointBackgroundColor: greenColor,
          pointRadius: 5,
          borderWidth: 2.5,
          tension: 0.35,
          yAxisID: "yRate",
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const idx = items[0]?.dataIndex;
              return idx !== undefined ? [`Rate: ${rates[idx]}%`] : [];
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: mutedColor, font: { size: 12 } },
          grid:  { color: borderColor },
        },
        y: {
          beginAtZero: true,
          ticks: { color: mutedColor, stepSize: 1, font: { size: 12 } },
          grid:  { color: borderColor },
          title: { display: true, text: "Count", color: mutedColor, font: { size: 11 } },
        },
        yRate: {
          beginAtZero: true,
          max: 100,
          position: "right",
          ticks: { color: mutedColor, callback: v => `${v}%`, font: { size: 12 } },
          grid: { display: false },
          title: { display: true, text: "Rate %", color: mutedColor, font: { size: 11 } },
        },
      },
    },
  });
}

function renderCalendar(month) {
  summaryTargets.monthLabel.textContent = month.label;
  calendarGrid.className = "calendar-grid";
  calendarGrid.innerHTML = month.days.map(day => {
    const d = new Date(day.date);
    const startCol = day.day === 1 ? `style="grid-column:${d.getDay() + 1}"` : "";
    return `
      <article class="calendar-day ${day.is_today ? "today" : ""}" ${startCol}>
        <div class="calendar-day-top">
          <strong>${day.day}</strong>
          ${day.count ? `<span class="count-dot">${day.count}</span>` : ""}
        </div>
        <div class="calendar-day-body">
          <span>${day.weekday}</span>
          ${day.count ? `<small>${day.count === 1 ? "1 item" : `${day.count} items`}</small>` : ""}
        </div>
      </article>`;
  }).join("");
}

function setRing(target, score, colorVar, fadedVar) {
  const angle = Math.max(0, Math.min(100, score)) * 3.6;
  const style = getComputedStyle(document.documentElement);
  const color = style.getPropertyValue(colorVar).trim();
  const faded = style.getPropertyValue(fadedVar).trim();
  target.style.background = `conic-gradient(${color} ${angle}deg, ${faded} ${angle}deg 360deg)`;
}

function renderInsights(analysis) {
  const { insights, suggestions, recommendation, sentiment } = analysis;

  if (insights) {
    insightSummary.textContent = insights.summary || "—";
    improvementsList.innerHTML = (insights.improvements || [])
      .map(s => `<li>${escapeHtml(s)}</li>`).join("") || "<li>—</li>";
  }

  suggestionsList.innerHTML = (suggestions || [])
    .map(s => `<li>${escapeHtml(s)}</li>`).join("") || "<li>—</li>";

  if (recommendation) {
    summaryTargets.recommendation.textContent = recommendation.recommendation || "—";
  }

  if (sentiment) {
    const polarity = Math.max(0, Math.min(1, (sentiment.polarity + 1) / 2));
    const subjectivity = Math.max(0, Math.min(1, sentiment.subjectivity));
    polarityBar.style.width = `${Math.round(polarity * 100)}%`;
    subjectivityBar.style.width = `${Math.round(subjectivity * 100)}%`;
    polarityVal.textContent = sentiment.polarity.toFixed(2);
    subjectivityVal.textContent = sentiment.subjectivity.toFixed(2);
  }
}

// ── Primary Render ────────────────────────
function renderDashboard(payload) {
  const { planner, analysis } = payload;

  renderCalendar(planner.month);
  renderCollection(todayList,     planner.today_items,     "No items for today.");
  renderCollection(completedList, planner.completed_items, "No completed items yet.");
  renderCollection(habitList,     planner.habits,          "No habits yet.",          { compact: true });
  renderCollection(eventList,     planner.upcoming_events, "No events scheduled.",    { compact: true });
  renderAlerts(analysis.alerts);
  renderChart(analysis.timeline_points);
  renderInsights(analysis);

  // Scores
  scoreTargets.stability.textContent = analysis.scores.stability;
  scoreTargets.drift.textContent     = analysis.scores.drift;
  scoreTargets.deviation.textContent = analysis.scores.deviation;
  scoreTargets.streak.textContent    = analysis.scores.streak ?? 0;
  setRing(scoreRings.stability, analysis.scores.stability, "--green", "--green-soft");
  setRing(scoreRings.drift,     analysis.scores.drift,     "--amber", "--amber-soft");
  setRing(scoreRings.deviation, analysis.scores.deviation, "--red",   "--red-soft");
  // Streak ring: cap visual at 30 days for the progress arc
  const streakCapped = Math.min((analysis.scores.streak ?? 0) * (100 / 30), 100);
  setRing(scoreRings.streak,    streakCapped,              "--blue",  "--blue-soft");

  // Summary
  const cat = analysis.summary.primary_category || "other";
  const catEl = summaryTargets.primaryIntent;
  catEl.textContent = cat;
  catEl.className = `category-badge cat-${cat}`;

  summaryTargets.activeGoal.textContent  = analysis.summary.stated_goal || "No focus item";
  summaryTargets.entryCount.textContent  = analysis.summary.entry_count;
  summaryTargets.todayProgress.textContent = `${planner.stats.today_completed} / ${planner.stats.today_total} done`;
  summaryTargets.highPriorityOpenCount.textContent = planner.stats.high_priority_open;
}

// ── Load Dashboard ───────────────────────
async function loadDashboard() {
  try {
    const res = await fetch("/api/dashboard");
    if (!res.ok) throw new Error("Server error");
    const payload = await res.json();
    renderDashboard(payload);
  } catch (err) {
    showToast("Failed to load dashboard. Is the server running?", "error");
  }
}

// ── Form Visibility ──────────────────────
function syncFormVisibility() {
  const kind    = itemKind.value;
  const isHabit = kind === "HABIT";
  const isEvent = kind === "EVENT";
  priorityField.style.display = isHabit ? "none" : "grid";
  dateField.style.display     = isEvent ? "grid" : "none";
  timeField.style.display     = isEvent ? "grid" : "none";
  repeatField.style.display   = isHabit ? "grid" : "none";
}

// ── API Actions ──────────────────────────
async function patchItem(id, body) {
  try {
    const res = await fetch(`/api/items/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await res.json();
    renderDashboard(payload.dashboard);
    if ("is_completed" in body) {
      showToast(body.is_completed ? "Marked as done ✓" : "Reverted to open", "success");
    } else {
      showToast("Priority updated", "info");
    }
  } catch {
    showToast("Failed to update item", "error");
  }
}

async function removeItem(id) {
  try {
    const res = await fetch(`/api/items/${id}`, { method: "DELETE" });
    const payload = await res.json();
    renderDashboard(payload.dashboard);
    showToast("Item deleted", "info");
  } catch {
    showToast("Failed to delete item", "error");
  }
}

// ── Event Listeners ──────────────────────
document.addEventListener("change", async (e) => {
  const t = e.target;
  if (t.dataset.action === "complete") await patchItem(t.dataset.id, { is_completed: t.checked });
  if (t.dataset.action === "priority") await patchItem(t.dataset.id, { priority: t.value });
});

document.addEventListener("click", async (e) => {
  if (e.target.dataset.action === "delete") await removeItem(e.target.dataset.id);
});

itemForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const kind = itemKind.value;
  const titleValue = document.getElementById("title").value.trim();
  if (!titleValue) {
    showToast("Title is required", "error");
    return;
  }

  submitBtn.disabled = true;
  submitLabel.textContent = "Saving…";

  const payload = {
    item_kind:        kind,
    title:            titleValue,
    details:          document.getElementById("details").value.trim(),
    priority:         kind === "HABIT" ? "medium" : document.getElementById("priority").value,
    scheduled_date:   document.getElementById("scheduledDate").value,
    scheduled_time:   document.getElementById("scheduledTime").value,
    repeat_frequency: document.getElementById("repeatFrequency").value,
  };

  try {
    const res = await fetch("/api/items", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const responsePayload = await res.json();
    if (!res.ok) {
      showToast(responsePayload.error || "Failed to save item", "error");
      return;
    }
    itemForm.reset();
    itemKind.value = "TASK";
    syncFormVisibility();
    renderDashboard(responsePayload.dashboard);
    showToast(`"${titleValue}" added successfully`, "success");
  } catch {
    showToast("Network error — could not save item", "error");
  } finally {
    submitBtn.disabled = false;
    submitLabel.textContent = "Save Item";
  }
});

itemKind.addEventListener("change", syncFormVisibility);

// ── Init ─────────────────────────────────
syncFormVisibility();
loadDashboard();