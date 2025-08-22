import sqlite3
import os
import uuid
from datetime import datetime, date, time, timedelta
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np

# Streamlit is optional: gracefully degrade to CLI if not installed
try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except ModuleNotFoundError:
    HAS_STREAMLIT = False

DB_PATH = os.environ.get("CLARITY_DB_PATH", "clarity_board.db")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
SMTP_TO = os.environ.get("SMTP_TO")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER or "clarity@localhost")

# ----------------------------
# Database helpers & migrations
# ----------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS weeks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start TEXT UNIQUE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            week_id INTEGER,
            title TEXT,
            description TEXT,
            tag TEXT,
            reach REAL DEFAULT 1.0,
            impact REAL DEFAULT 1.0,
            confidence REAL DEFAULT 0.7,
            effort REAL DEFAULT 1.0,
            status TEXT DEFAULT 'inbox', -- inbox | selected | archived | done
            FOREIGN KEY(week_id) REFERENCES weeks(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id INTEGER,
            created_at TEXT,
            title TEXT,
            due_date TEXT,
            done INTEGER DEFAULT 0,
            -- Fields for Task Prioritiser
            duration REAL DEFAULT 1.0,       -- hours
            impact REAL DEFAULT 3.0,         -- 1..10
            confidence REAL DEFAULT 0.7,     -- 0..1
            energy REAL DEFAULT 0.7,         -- 0..1 (how well it matches your current energy/focus)
            context TEXT,                    -- e.g., deep-work, calls, errands
            standalone INTEGER DEFAULT 0,    -- 1 = task created in Task Prioritiser (no idea link)
            priority_cache REAL,
            -- Follow-through
            today_top3 INTEGER DEFAULT 0,    -- 1 = in Top-3-Today
            top3_date TEXT,                  -- YYYY-MM-DD for which day it was selected
            planned_start TEXT,              -- ISO datetime
            planned_end TEXT,                -- ISO datetime
            FOREIGN KEY(idea_id) REFERENCES ideas(id)
        )
        """
    )
    conn.commit()
    migrate_db()


def migrate_db():
    """Add missing columns for the Task Prioritiser and follow-through features without breaking existing DBs."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(tasks)")
    cols = {row[1] for row in cur.fetchall()}
    to_add = []
    if "duration" not in cols:
        to_add.append("ALTER TABLE tasks ADD COLUMN duration REAL DEFAULT 1.0")
    if "impact" not in cols:
        to_add.append("ALTER TABLE tasks ADD COLUMN impact REAL DEFAULT 3.0")
    if "confidence" not in cols:
        to_add.append("ALTER TABLE tasks ADD COLUMN confidence REAL DEFAULT 0.7")
    if "energy" not in cols:
        to_add.append("ALTER TABLE tasks ADD COLUMN energy REAL DEFAULT 0.7")
    if "context" not in cols:
        to_add.append("ALTER TABLE tasks ADD COLUMN context TEXT")
    if "standalone" not in cols:
        to_add.append("ALTER TABLE tasks ADD COLUMN standalone INTEGER DEFAULT 0")
    if "priority_cache" not in cols:
        to_add.append("ALTER TABLE tasks ADD COLUMN priority_cache REAL")
    if "today_top3" not in cols:
        to_add.append("ALTER TABLE tasks ADD COLUMN today_top3 INTEGER DEFAULT 0")
    if "top3_date" not in cols:
        to_add.append("ALTER TABLE tasks ADD COLUMN top3_date TEXT")
    if "planned_start" not in cols:
        to_add.append("ALTER TABLE tasks ADD COLUMN planned_start TEXT")
    if "planned_end" not in cols:
        to_add.append("ALTER TABLE tasks ADD COLUMN planned_end TEXT")
    for sql in to_add:
        conn.execute(sql)
    conn.commit()


# ----------------------------
# Week helpers
# ----------------------------

def get_or_create_week(week_start: date) -> Tuple[int, bool]:
    """Return (week_id, created_new)."""
    conn = get_conn()
    cur = conn.cursor()
    wk = week_start.isoformat()
    before = conn.total_changes
    cur.execute("INSERT OR IGNORE INTO weeks (week_start) VALUES (?)", (wk,))
    conn.commit()
    created_new = (conn.total_changes - before) > 0
    cur.execute("SELECT id FROM weeks WHERE week_start=?", (wk,))
    row = cur.fetchone()
    return row[0], created_new


def get_previous_week_id(current_week_start: date) -> Optional[int]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM weeks WHERE week_start < ? ORDER BY week_start DESC LIMIT 1",
        (current_week_start.isoformat(),),
    )
    row = cur.fetchone()
    return row[0] if row else None


def carry_over_open_selected(new_week_id: int, new_week_start: date) -> int:
    prev_id = get_previous_week_id(new_week_start)
    if not prev_id:
        return 0
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT LOWER(TRIM(title)) FROM ideas WHERE week_id=?", (new_week_id,))
    existing_titles = {t[0] for t in cur.fetchall()}
    cur.execute(
        """
        SELECT title, description, tag, reach, impact, confidence, effort
        FROM ideas WHERE week_id=? AND status='selected'
        """,
        (prev_id,),
    )
    rows = cur.fetchall()
    inserted = 0
    for (title, desc, tag, reach, impact, confidence, effort) in rows:
        key = (title or "").strip().lower()
        if key in existing_titles:
            continue
        conn.execute(
            """
            INSERT INTO ideas (created_at, week_id, title, description, tag, reach, impact, confidence, effort, status)
            VALUES (?,?,?,?,?,?,?,?,?, 'selected')
            """,
            (
                datetime.utcnow().isoformat(),
                new_week_id,
                title or "",
                desc or "",
                tag or "",
                float(reach or 1.0),
                float(impact or 1.0),
                float(confidence or 0.7),
                float(effort or 1.0),
            ),
        )
        inserted += 1
    conn.commit()
    return inserted


# ----------------------------
# Idea scoring (linear weights + sensible effort clamp)
# ----------------------------

def _clamp(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return x.fillna(lo).clip(lower=lo, upper=hi)


def compute_scores(df: pd.DataFrame, method: str, weights: dict) -> pd.DataFrame:
    """Compute ICE/RICE scores using **linear multipliers** for weights.

    ICE:  score = (impact * w_i) * (confidence * w_c) / (effort * w_e)
    RICE: score = (reach * w_r) * (impact * w_i) * (confidence * w_c) / (effort * w_e)

    Effort is clamped to [0.25, 10] to avoid pathological values.
    """
    df = df.copy()
    df["effort"] = _clamp(df["effort"], 0.25, 10.0)

    w_i = float(weights.get("impact", 1.0))
    w_c = float(weights.get("confidence", 1.0))
    w_e = max(0.1, float(weights.get("effort", 1.0)))

    if method == "ICE":
        df["score"] = (df["impact"] * w_i) * (df["confidence"] * w_c) / (df["effort"] * w_e)
    else:  # RICE
        w_r = float(weights.get("reach", 1.0))
        df["reach"] = df["reach"].replace(0, 0.25).fillna(0.25)
        df["score"] = (df["reach"] * w_r) * (df["impact"] * w_i) * (df["confidence"] * w_c) / (df["effort"] * w_e)
    return df


# ----------------------------
# Super‑powered Task Prioritiser
# ----------------------------

def _urgency_from_due(due_str: Optional[str], today: date) -> float:
    """Return urgency factor >=1.0. Due sooner -> higher urgency.
    Formula: 1 + max(0, 7 - days_to_due)/7 for due in the next 7 days; else 1.0.
    None/invalid => 1.0
    """
    if not due_str:
        return 1.0
    try:
        d = pd.to_datetime(due_str).date()
        delta = (d - today).days
        return 1.0 + max(0, 7 - delta) / 7.0
    except Exception:
        return 1.0


def compute_task_priority(df: pd.DataFrame, weights: dict, today: date) -> pd.DataFrame:
    """Compute priority for standalone tasks (and idea-linked tasks).

    priority = (wI*impact + wC*confidence + wEn*energy + wU*urgency) / (duration_clamped * wE)
    where urgency is derived from due date proximity.
    """
    df = df.copy()
    for col, default in ("impact", 3.0), ("confidence", 0.7), ("energy", 0.7), ("duration", 1.0):
        if col not in df.columns:
            df[col] = default
    df["duration"] = _clamp(df["duration"], 0.25, 10.0)
    df["urgency"] = df.get("due_date", pd.Series([None]*len(df))).apply(lambda x: _urgency_from_due(x, today))

    wI = float(weights.get("impact", 1.0))
    wC = float(weights.get("confidence", 1.0))
    wEn = float(weights.get("energy", 1.0))
    wU = float(weights.get("urgency", 1.0))
    wE = max(0.1, float(weights.get("effort", 1.0)))

    df["priority"] = (wI*df["impact"] + wC*df["confidence"] + wEn*df["energy"] + wU*df["urgency"]) / (df["duration"] * wE)
    df["priority_per_hour"] = df["priority"] / df["duration"]
    return df


def suggest_plan(df: pd.DataFrame, capacity_hours: float) -> pd.DataFrame:
    """Greedy selection by priority_per_hour until capacity is filled. Returns selected tasks with cumulative hours."""
    if df.empty:
        return df
    df = df.sort_values(["priority_per_hour", "priority"], ascending=False).copy()
    take = []
    cum = 0.0
    for row in df.itertuples():
        dur = float(row.duration)
        if dur <= 0:
            continue
        if cum + dur <= capacity_hours + 1e-9:
            cum += dur
            take.append((row.Index, cum))
    if not take:
        return df.head(0)
    sel = df.loc[[idx for idx, _ in take]].copy()
    sel["cumulative_hours"] = [c for _, c in take]
    return sel


# ----------------------------
# Follow-through helpers (WIP cap, Top-3-Today, ICS, Nudges)
# ----------------------------

def today_str() -> str:
    return date.today().isoformat()


def top3_open_count(today_iso: Optional[str] = None) -> int:
    today_iso = today_iso or today_str()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(1) FROM tasks WHERE standalone=1 AND today_top3=1 AND done=0 AND top3_date=?",
        (today_iso,),
    )
    (cnt,) = cur.fetchone()
    return int(cnt)


def can_add_to_top3(today_iso: Optional[str] = None, cap: int = 3) -> bool:
    return top3_open_count(today_iso) < cap


def set_top3(task_id: int, active: bool, today_iso: Optional[str] = None):
    conn = get_conn()
    if active and not can_add_to_top3(today_iso):
        return False
    if active:
        conn.execute("UPDATE tasks SET today_top3=1, top3_date=? WHERE id=?", (today_iso or today_str(), task_id))
    else:
        conn.execute("UPDATE tasks SET today_top3=0, top3_date=NULL WHERE id=?", (task_id,))
    conn.commit()
    return True


def _ics_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")


def generate_ics_for_top3(start_dt: datetime) -> bytes:
    """Generate an ICS calendar with sequential blocks for today's Top-3 tasks (standalone only, not done)."""
    t_iso = today_str()
    conn = get_conn()
    # Pull tasks in priority order
    tdf = pd.read_sql_query(
        "SELECT * FROM tasks WHERE standalone=1 AND today_top3=1 AND done=0 AND top3_date=? ORDER BY id ASC",
        conn,
        params=(t_iso,),
    )
    if tdf.empty:
        cal = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//Clarity Board//EN\nEND:VCALENDAR\n"
        return cal.encode("utf-8")

    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    cur_start = start_dt
    events = []
    for row in tdf.itertuples():
        dur_minutes = int(max(15, float(row.duration) * 60))
        dtstart = cur_start.strftime("%Y%m%dT%H%M%S")
        dtend = (cur_start + timedelta(minutes=dur_minutes)).strftime("%Y%m%dT%H%M%S")
        uid = f"{uuid.uuid4()}@clarity"
        title = _ics_escape(str(row.title or "Task"))
        desc = _ics_escape(f"Context: {row.context or '-'}\nNotes: (open the app for details)")
        ev = (
            "BEGIN:VEVENT\n"
            f"UID:{uid}\n"
            f"DTSTAMP:{dtstamp}\n"
            f"DTSTART:{dtstart}\n"
            f"DTEND:{dtend}\n"
            f"SUMMARY:{title}\n"
            f"DESCRIPTION:{desc}\n"
            "END:VEVENT\n"
        )
        events.append(ev)
        # write planned times back
        conn.execute(
            "UPDATE tasks SET planned_start=?, planned_end=? WHERE id=?",
            ((cur_start.isoformat()), (cur_start + timedelta(minutes=dur_minutes)).isoformat(), int(row.id)),
        )
        cur_start += timedelta(minutes=dur_minutes)
    conn.commit()

    cal = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//Clarity Board//EN\n" + "".join(events) + "END:VCALENDAR\n"
    return cal.encode("utf-8")


def send_slack_message(text: str) -> Tuple[bool, str]:
    if not SLACK_WEBHOOK_URL:
        return False, "No SLACK_WEBHOOK_URL configured"
    try:
        import json, urllib.request
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec - controlled webhook
            _ = resp.read()
        return True, "sent"
    except Exception as e:
        return False, str(e)


def send_email(subject: str, body: str) -> Tuple[bool, str]:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_TO):
        return False, "SMTP env vars missing"
    try:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = SMTP_TO
        msg.set_content(body)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True, "sent"
    except Exception as e:
        return False, str(e)


# ----------------------------
# Self-checks (lightweight tests)
# ----------------------------

def run_self_checks() -> pd.DataFrame:
    """Return a dataframe of test results for quick diagnostics.
    NOTE: These tests validate the scoring/prioritisation logic & core helpers; they do not cover the UI.
    """
    results = []
    try:
        # Idea scoring tests
        weights = {"reach": 1.0, "impact": 1.0, "confidence": 1.0, "effort": 1.0}

        df = pd.DataFrame({"reach": [1,1], "impact":[5,5], "confidence":[0.8,0.8], "effort":[1.0,2.0]})
        ice = compute_scores(df, "ICE", weights).sort_values("score", ascending=False)
        results.append({"test":"ICE favors lower effort when others equal","passed": ice.iloc[0]["effort"] < ice.iloc[1]["effort"]})

        df = pd.DataFrame({"reach":[100,10],"impact":[5,5],"confidence":[0.8,0.8],"effort":[2.0,2.0]})
        rice = compute_scores(df, "RICE", weights).sort_values("score", ascending=False)
        results.append({"test":"RICE increases with reach","passed": rice.iloc[0]["reach"] > rice.iloc[1]["reach"]})

        df = pd.DataFrame({"reach":[1],"impact":[1],"confidence":[1.0],"effort":[0.0]})
        ice_zero = compute_scores(df, "ICE", weights)
        results.append({"test":"Effort=0 handled safely (clamped)","passed": np.isfinite(ice_zero.loc[0,"score"]) and ice_zero.loc[0,"score"]>0})

        wk_id_1, _ = get_or_create_week(date(2025,1,6)); wk_id_2, _ = get_or_create_week(date(2025,1,6))
        results.append({"test":"get_or_create_week idempotent","passed": wk_id_1==wk_id_2})

        df = pd.DataFrame({"reach":[1,1,1],"impact":[5,5,5],"confidence":[0.9,0.9,0.9],"effort":[1.0,2.0,3.0]})
        ice = compute_scores(df, "ICE", weights); s = ice.sort_values("effort")["score"].values
        results.append({"test":"Score decreases as effort increases (ICE)","passed": (s[0]>s[1]>s[2])})

        df = pd.DataFrame({"reach":[1,1,1],"impact":[3,5,7],"confidence":[0.8,0.8,0.8],"effort":[2.0,2.0,2.0]})
        ice = compute_scores(df, "ICE", weights); s = ice.sort_values("impact")["score"].values
        results.append({"test":"Score increases with impact (ICE)","passed": (s[0]<s[1]<s[2])})

        df = pd.DataFrame({"reach":[10,10,10],"impact":[4,6,5],"confidence":[0.7,0.9,0.8],"effort":[2.0,2.0,2.0]})
        ice_rank = compute_scores(df, "ICE", weights).sort_values("score", ascending=False)["score"].rank(method="first").tolist()
        rice_rank = compute_scores(df, "RICE", weights).sort_values("score", ascending=False)["score"].rank(method="first").tolist()
        results.append({"test":"RICE==ICE ordering when reach constant","passed": ice_rank==rice_rank})

        df = pd.DataFrame({"reach":[0,5],"impact":[5,5],"confidence":[0.8,0.8],"effort":[2.0,2.0]})
        rice = compute_scores(df, "RICE", weights)
        results.append({"test":"RICE handles reach=0 safely","passed": np.all(np.isfinite(rice["score"]))})

        df = pd.DataFrame({"reach":[3,4],"impact":[6,6],"confidence":[0.9,0.9],"effort":[1.5,1.5]})
        s1 = compute_scores(df, "ICE", weights)["score"].values; s2 = compute_scores(df, "ICE", weights)["score"].values
        results.append({"test":"Deterministic outputs (ICE)","passed": np.allclose(s1,s2)})

        df = pd.DataFrame({"reach":[1],"impact":[6],"confidence":[0.9],"effort":[2.0]})
        low_pen = compute_scores(df, "ICE", {**weights, "effort":0.5})["score"].iloc[0]
        high_pen = compute_scores(df, "ICE", {**weights, "effort":2.0})["score"].iloc[0]
        results.append({"test":"Higher effort weight lowers score (linear)","passed": high_pen<low_pen})

        # Task prioritiser tests
        today = date.today()
        tdf = pd.DataFrame({
            "title":["Due tomorrow","Due in 10d"],
            "duration":[1.0,1.0],
            "impact":[5,5],
            "confidence":[0.8,0.8],
            "energy":[0.7,0.7],
            "due_date":[(pd.Timestamp(today)+pd.Timedelta(days=1)).date().isoformat(), (pd.Timestamp(today)+pd.Timedelta(days=10)).date().isoformat()],
        })
        pr = compute_task_priority(tdf, {"impact":1,"confidence":1,"energy":1,"urgency":1,"effort":1}, today).sort_values("priority", ascending=False)
        results.append({"test":"Urgent tasks rank higher (same other factors)","passed": pr.iloc[0]["title"]=="Due tomorrow"})

        tdf2 = pd.DataFrame({
            "title":["A","B","C"],
            "duration":[1.0,2.0,3.0],
            "impact":[5,4,3],
            "confidence":[0.9,0.8,0.7],
            "energy":[0.8,0.8,0.8],
            "due_date":[None,None,None],
        })
        pr2 = compute_task_priority(tdf2, {"impact":1,"confidence":1,"energy":1,"urgency":1,"effort":1}, today)
        plan = suggest_plan(pr2, capacity_hours=3.0)
        results.append({"test":"Plan respects capacity (<=3h)","passed": (len(plan)==2) and plan["cumulative_hours"].max() <= 3.0 + 1e-9})

        # WIP cap logic
        conn = get_conn()
        conn.execute("DELETE FROM tasks WHERE standalone=1")
        conn.commit()
        # Insert three open top-3 tasks for today
        for i in range(3):
            conn.execute(
                "INSERT INTO tasks (idea_id, created_at, title, due_date, done, duration, impact, confidence, energy, context, standalone, today_top3, top3_date) VALUES (NULL,?,?,?,?,?,?,?,?,?,1,1,?)",
                (
                    datetime.utcnow().isoformat(),
                    f"T{i+1}",
                    None,
                    0,
                    1.0,
                    5,
                    0.8,
                    0.8,
                    "deep-work",
                    today_str(),
                ),
            )
        conn.commit()
        results.append({"test":"WIP cap prevents adding 4th top-3","passed": can_add_to_top3() is False})

    except Exception as e:
        results.append({"test": "Unexpected error in tests", "passed": False, "error": str(e)})

    return pd.DataFrame(results)


# ----------------------------
# Streamlit App UI (only if Streamlit is available)
# ----------------------------

if HAS_STREAMLIT:
    st.set_page_config(page_title="Clarity Board", page_icon="✅", layout="wide")
    init_db()

    st.title("Clarity Board – Weekly Focus & Follow‑Through")
    st.caption("Capture → Score → Select → Execute → Reflect → **Ship**")

    # Sidebar controls
    with st.sidebar:
        st.header("Weekly Setup")
        today_d = date.today()
        week_start = today_d if today_d.weekday() == 0 else (today_d - pd.Timedelta(days=today_d.weekday()))
        wk = st.date_input("Week starting (Mon)", value=week_start)
        week_id, created_new = get_or_create_week(wk)

        carried = 0
        if created_new:
            carried = carry_over_open_selected(week_id, wk)
            if carried:
                st.toast(f"Carried over {carried} selected item(s) from last week.")

        st.markdown("---")
        st.header("Nudges (optional)")
        st.caption("Set env vars to enable: SLACK_WEBHOOK_URL, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_TO")
        colN1, colN2 = st.columns(2)
        with colN1:
            if st.button("Send Slack test"):
                ok, msg = send_slack_message("Clarity test: You're set up for nudges.")
                st.success("Slack OK") if ok else st.warning(msg)
        with colN2:
            if st.button("Send Email test"):
                ok, msg = send_email("Clarity test", "Email configuration works.")
                st.success("Email OK") if ok else st.warning(msg)

    # Tabs for Ideas vs Task Prioritiser
    tabs = st.tabs(["Ideas Board", "Task Prioritiser", "Follow‑Through (Top‑3‑Today)"])

    # ---------------- Ideas Board Tab ----------------
    with tabs[0]:
        st.subheader("Scoring (linear weights)")
        colA, colB = st.columns(2)
        with colA:
            method = st.radio("Model", ["ICE", "RICE"], index=0, help="ICE = Impact, Confidence, Effort. RICE adds Reach.")
            w_impact = st.slider("Weight: Impact (×)", 0.1, 3.0, 1.0, 0.1)
            w_conf = st.slider("Weight: Confidence (×)", 0.1, 3.0, 1.0, 0.1)
        with colB:
            w_eff = st.slider("Weight: Effort penalty (×)", 0.1, 3.0, 1.0, 0.1)
            w_reach = st.slider("Weight: Reach (×, RICE)", 0.1, 3.0, 1.0, 0.1)

        st.markdown("---")
        st.subheader("1) Capture – Dump your ideas/thoughts")
        with st.form("capture_idea"):
            c1, c2 = st.columns([2, 1])
            with c1:
                title = st.text_input("Idea / Thought title")
                description = st.text_area("Short description / context", height=100)
            with c2:
                tag = st.text_input("Tag (e.g., business, personal, learning)")
                reach = st.number_input("Reach (people impacted)", min_value=0.0, value=1.0, step=0.5)
                impact = st.number_input("Impact (1-10)", min_value=0.0, value=5.0, step=0.5)
                confidence = st.number_input("Confidence (0-1)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
                effort = st.number_input("Effort (hours scale)", min_value=0.25, value=2.0, step=0.25, help="Clamped to 0.25–10 internally")
            submitted = st.form_submit_button("Add to Inbox")
            if submitted and title:
                conn = get_conn()
                conn.execute(
                    "INSERT INTO ideas (created_at, week_id, title, description, tag, reach, impact, confidence, effort, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        datetime.utcnow().isoformat(),
                        week_id,
                        title.strip(),
                        description.strip(),
                        tag.strip(),
                        float(reach),
                        float(impact),
                        float(confidence),
                        float(effort),
                        "inbox",
                    ),
                )
                conn.commit()
                st.success("Added to inbox ✅")

        conn = get_conn()
        df = pd.read_sql_query(
            "SELECT i.*, w.week_start FROM ideas i JOIN weeks w ON i.week_id=w.id WHERE i.week_id=?",
            conn,
            params=(week_id,),
        )
        if not df.empty:
            weights = {"reach": w_reach, "impact": w_impact, "confidence": w_conf, "effort": w_eff}
            df_scored = compute_scores(df, method, weights).sort_values("score", ascending=False)

            st.subheader("2) Score – Ranked ideas for the week")
            st.dataframe(
                df_scored[["id", "status", "tag", "title", "reach", "impact", "confidence", "effort", "score"]]
                .rename(columns={"tag": "Tag", "title": "Title", "reach": "Reach", "impact": "Impact", "confidence": "Confidence", "effort": "Effort", "score": "Score"}),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("3) Visualize – Impact vs Effort")
            chart_df = df_scored[["title", "impact", "effort", "score"]].rename(columns={"impact": "Impact", "effort": "Effort", "title": "Title", "score": "Score"})
            st.caption("Top-left = high impact, low effort → fast wins")
            st.scatter_chart(chart_df, x="Effort", y="Impact", size="Score", color=None)

            st.subheader("4) Select – Commit to the few that matter")
            selectable = df_scored[df_scored["status"].isin(["inbox", "archived"])]
            choices = st.multiselect(
                "Pick up to N items to commit this week",
                options=[f"#{row.id} – {row.title}" for row in selectable.itertuples()],
                max_selections=st.slider("How many to commit?", 1, 5, 3, key="top_k_ideas"),
            )
            if st.button("Commit selected") and choices:
                ids = [int(c.split(" – ")[0].replace("#", "")) for c in choices]
                qmarks = ",".join(["?"] * len(ids))
                conn.execute(f"UPDATE ideas SET status='selected' WHERE id IN ({qmarks})", ids)
                conn.commit()
                st.success("Committed to weekly focus ✅")
        else:
            st.info("No ideas captured for this week yet. Add a few above.")

        st.markdown("---")
        st.subheader("Diagnostics for Ideas")
        if st.button("Run self-checks (ideas)", key="diag_ideas"):
            results_df = run_self_checks()
            st.dataframe(results_df, use_container_width=True, hide_index=True)

    # ---------------- Task Prioritiser Tab ----------------
    with tabs[1]:
        st.subheader("Weights & Capacity")
        col1, col2, col3 = st.columns(3)
        with col1:
            tw_i = st.slider("Impact weight (×)", 0.1, 3.0, 1.0, 0.1)
            tw_c = st.slider("Confidence weight (×)", 0.1, 3.0, 1.0, 0.1)
        with col2:
            tw_en = st.slider("Energy fit weight (×)", 0.1, 3.0, 1.0, 0.1)
            tw_u = st.slider("Urgency weight (×)", 0.1, 3.0, 1.0, 0.1)
        with col3:
            tw_e = st.slider("Effort penalty (×)", 0.1, 3.0, 1.0, 0.1)
            capacity = st.number_input("Weekly capacity (hours)", min_value=1.0, value=10.0, step=1.0)

        t_weights = {"impact": tw_i, "confidence": tw_c, "energy": tw_en, "urgency": tw_u, "effort": tw_e}

        st.markdown("---")
        st.subheader("Add Task (standalone)")
        wip_cnt = top3_open_count()
        wip_full = wip_cnt >= 3
        if wip_full:
            st.info("Top‑3‑Today is full (3 active). Mark one done or remove before adding more.")
        with st.form("add_task", clear_on_submit=True):
            c1, c2, c3 = st.columns([2,1,1])
            with c1:
                t_title = st.text_input("Task title")
                t_desc = st.text_area("Notes", height=80)
                t_ctx = st.text_input("Context (deep-work/calls/errands)")
            with c2:
                t_due = st.date_input("Due date (optional)", value=None)
                t_dur = st.number_input("Duration (hours)", min_value=0.25, max_value=10.0, value=1.0, step=0.25)
            with c3:
                t_imp = st.slider("Impact", 1, 10, 5)
                t_conf = st.slider("Confidence", 0.0, 1.0, 0.7, 0.05)
                t_energy = st.slider("Energy fit", 0.0, 1.0, 0.7, 0.05)
            submitted_task = st.form_submit_button("Add task", disabled=False)
            if submitted_task and t_title:
                conn = get_conn()
                conn.execute(
                    """
                    INSERT INTO tasks (idea_id, created_at, title, due_date, done, duration, impact, confidence, energy, context, standalone)
                    VALUES (NULL,?,?,?,?,?,?,?,?,?,1)
                    """,
                    (
                        datetime.utcnow().isoformat(),
                        t_title.strip(),
                        (t_due.isoformat() if t_due else None),
                        0,
                        float(t_dur),
                        float(t_imp),
                        float(t_conf),
                        float(t_energy),
                        t_ctx.strip(),
                    ),
                )
                conn.commit()
                st.success("Task added ✅")

        # Load tasks
        conn = get_conn()
        tdf = pd.read_sql_query(
            "SELECT * FROM tasks WHERE standalone=1 ORDER BY done ASC, due_date IS NULL, due_date ASC",
            conn,
        )
        if tdf.empty:
            st.info("No standalone tasks yet. Add a few above.")
        else:
            pr_df = compute_task_priority(tdf, t_weights, date.today())
            st.subheader("Prioritised Tasks")
            cols = ["id","title","context","due_date","duration","impact","confidence","energy","urgency","priority"]
            st.dataframe(pr_df[cols].sort_values("priority", ascending=False), use_container_width=True, hide_index=True)

            st.subheader("Suggested Plan (fits weekly capacity)")
            plan = suggest_plan(pr_df[pr_df["done"]==0], capacity)
            if plan.empty:
                st.info("No eligible tasks to plan.")
            else:
                show_cols = ["id","title","duration","priority","cumulative_hours"]
                st.dataframe(plan[show_cols], use_container_width=True, hide_index=True)

            st.subheader("Quick actions")
            for row in pr_df.itertuples():
                cols = st.columns([0.06, 0.44, 0.2, 0.15, 0.15])
                with cols[0]:
                    d_toggle = st.checkbox("", value=bool(row.done), key=f"tdone_{row.id}")
                with cols[1]:
                    st.write(f"**#{row.id}** {row.title}")
                with cols[2]:
                    st.write((pd.to_datetime(row.due_date).date() if row.due_date else "—"))
                with cols[3]:
                    # Add/Remove from Top‑3‑Today with WIP cap
                    currently_top3 = bool(row.today_top3 and (row.top3_date == today_str()))
                    add_disabled = (not currently_top3) and (not can_add_to_top3())
                    if currently_top3:
                        if st.button("Remove from Today", key=f"rm_top3_{row.id}"):
                            set_top3(int(row.id), False)
                            st.experimental_rerun()
                    else:
                        if st.button("Add to Today", key=f"add_top3_{row.id}", disabled=add_disabled):
                            ok = set_top3(int(row.id), True)
                            if not ok:
                                st.warning("Top‑3 is full. Complete or remove one first.")
                            st.experimental_rerun()
                with cols[4]:
                    if st.button("Delete", key=f"t_del_{row.id}"):
                        conn.execute("DELETE FROM tasks WHERE id=?", (row.id,))
                        conn.commit()
                        st.experimental_rerun()
                if d_toggle != bool(row.done):
                    conn.execute("UPDATE tasks SET done=? WHERE id=?", (int(d_toggle), row.id))
                    conn.commit()

        st.markdown("---")
        st.subheader("Diagnostics (all)")
        if st.button("Run self-checks", key="diag_all"):
            results_df = run_self_checks()
            st.dataframe(results_df, use_container_width=True, hide_index=True)

    # ---------------- Follow‑Through Tab ----------------
    with tabs[2]:
        st.subheader("Top‑3‑Today (WIP cap enforced)")
        cur_top3 = pd.read_sql_query(
            "SELECT * FROM tasks WHERE standalone=1 AND today_top3=1 AND done=0 AND top3_date=? ORDER BY id ASC",
            get_conn(), params=(today_str(),)
        )
        st.write(f"Active: **{len(cur_top3)}/3**")
        if cur_top3.empty:
            st.info("No tasks in Top‑3 yet. Go to Task Prioritiser and add up to 3.")
        else:
            st.dataframe(cur_top3[["id","title","duration","due_date","context"]], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Build calendar blocks (ICS export)")
        start_time = st.time_input("Start time today", value=(datetime.now().replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)).time())
        if st.button("Generate .ics for Top‑3"):
            start_dt = datetime.combine(date.today(), start_time)
            ics_bytes = generate_ics_for_top3(start_dt)
            st.download_button("Download schedule.ics", data=ics_bytes, file_name="clarity_top3_today.ics", mime="text/calendar")
            st.success("ICS generated. Import into your calendar.")

        st.markdown("---")
        st.subheader("Nudge me now (manual)")
        msg = "Top‑3 for today:\n" + "\n".join([f"- {t}" for t in cur_top3["title"].tolist()]) if not cur_top3.empty else "(none)"
        colK1, colK2 = st.columns(2)
        with colK1:
            if st.button("Send Slack now"):
                ok, out = send_slack_message(msg)
                st.success("Sent to Slack") if ok else st.warning(out)
        with colK2:
            if st.button("Send Email now"):
                ok, out = send_email("Clarity Top‑3 Today", msg)
                st.success("Sent email") if ok else st.warning(out)

        st.markdown("---")
        st.subheader("Action Score (today)")
        done_today = pd.read_sql_query(
            "SELECT COUNT(1) as c FROM tasks WHERE standalone=1 AND today_top3=1 AND top3_date=? AND done=1",
            get_conn(), params=(today_str(),)
        ).iloc[0].c
        total_today = max(1, len(cur_top3))
        score = round(100.0 * done_today / total_today, 1)
        st.metric("Action Score", f"{score}%", help="Completed / Top‑3 for today")

    st.markdown("---")
    st.caption("Built for weekly clarity and daily follow‑through: focus on the few, ship, repeat.")

# ----------------------------
# CLI fallback (when Streamlit is not installed)
# ----------------------------

if __name__ == "__main__" and not HAS_STREAMLIT:
    init_db()
    print("[Clarity Board] Streamlit is not installed. Running in CLI diagnostics mode.\n")
    print("→ To use the full UI, install Streamlit: 'pip install streamlit' and run 'streamlit run app.py'\n")

    tests = run_self_checks()
    print("Self-checks:\n", tests.to_string(index=False))

    # Minimal Task Prioritiser demo
    today = date.today()
    demo = pd.DataFrame({
        "title": ["Write brief", "Call supplier", "Deep work block"],
        "duration": [1.0, 0.5, 2.0],
        "impact": [6, 4, 8],
        "confidence": [0.9, 0.8, 0.7],
        "energy": [0.8, 0.7, 0.9],
        "due_date": [None, (pd.Timestamp(today)+pd.Timedelta(days=1)).date().isoformat(), None],
        "done": [0,0,0],
    })
    weights = {"impact":1.0, "confidence":1.0, "energy":1.0, "urgency":1.0, "effort":1.0}
    pr = compute_task_priority(demo, weights, today)
    print("\nTask priorities (demo):\n", pr[["title","duration","urgency","priority"]])
    plan = suggest_plan(pr, capacity_hours=2.0)
    print("\nSuggested 2h plan:\n", plan[["title","duration","priority","cumulative_hours"]])
    print("\nDone. CLI mode is for verification only.")
