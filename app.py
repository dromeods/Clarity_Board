import sqlite3
import os
import uuid
from datetime import datetime, date, timedelta, time
from typing import Optional, Tuple

import pandas as pd
import numpy as np

# Streamlit optional fallback
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

# ----------------------------------
# DB Helpers & Migrations
# ----------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # Weeks (kept for potential future summaries)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS weeks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start TEXT UNIQUE
        )
        """
    )
    # Ideas/Notes (repurposed: notes over scoring)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            title TEXT,
            tag TEXT,
            body TEXT,         -- Markdown notes
            status TEXT DEFAULT 'active'  -- active | archived
        )
        """
    )
    # Tasks (standalone or linked to idea)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id INTEGER,
            created_at TEXT,
            title TEXT,
            notes TEXT,
            due_date TEXT,
            done INTEGER DEFAULT 0,
            duration REAL DEFAULT 1.0,
            impact REAL DEFAULT 3.0,
            confidence REAL DEFAULT 0.7,
            energy REAL DEFAULT 0.7,
            context TEXT,
            standalone INTEGER DEFAULT 1,  -- 1 standalone, 0 if derived from idea
            priority_cache REAL,
            today_top3 INTEGER DEFAULT 0,
            top3_date TEXT,
            planned_start TEXT,
            planned_end TEXT,
            FOREIGN KEY(idea_id) REFERENCES ideas(id)
        )
        """
    )
    conn.commit()
    migrate_db()


def migrate_db():
    """Add missing columns safely (idempotent)."""
    conn = get_conn()
    cur = conn.cursor()
    # ideas table columns
    cur.execute("PRAGMA table_info(ideas)")
    icol = {r[1] for r in cur.fetchall()}
    if "body" not in icol:
        conn.execute("ALTER TABLE ideas ADD COLUMN body TEXT")
    if "status" not in icol:
        conn.execute("ALTER TABLE ideas ADD COLUMN status TEXT DEFAULT 'active'")

    # tasks table columns
    cur.execute("PRAGMA table_info(tasks)")
    tcol = {r[1] for r in cur.fetchall()}
    if "notes" not in tcol:
        conn.execute("ALTER TABLE tasks ADD COLUMN notes TEXT")
    for col, sql in {
        "duration": "ALTER TABLE tasks ADD COLUMN duration REAL DEFAULT 1.0",
        "impact": "ALTER TABLE tasks ADD COLUMN impact REAL DEFAULT 3.0",
        "confidence": "ALTER TABLE tasks ADD COLUMN confidence REAL DEFAULT 0.7",
        "energy": "ALTER TABLE tasks ADD COLUMN energy REAL DEFAULT 0.7",
        "context": "ALTER TABLE tasks ADD COLUMN context TEXT",
        "standalone": "ALTER TABLE tasks ADD COLUMN standalone INTEGER DEFAULT 1",
        "priority_cache": "ALTER TABLE tasks ADD COLUMN priority_cache REAL",
        "today_top3": "ALTER TABLE tasks ADD COLUMN today_top3 INTEGER DEFAULT 0",
        "top3_date": "ALTER TABLE tasks ADD COLUMN top3_date TEXT",
        "planned_start": "ALTER TABLE tasks ADD COLUMN planned_start TEXT",
        "planned_end": "ALTER TABLE tasks ADD COLUMN planned_end TEXT",
    }.items():
        if col not in tcol:
            conn.execute(sql)
    conn.commit()


# ----------------------------------
# Prioritisation
# ----------------------------------

def _clamp(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return x.fillna(lo).clip(lower=lo, upper=hi)


def _urgency_from_due(due_str: Optional[str], today: date) -> float:
    if not due_str:
        return 1.0
    try:
        d = pd.to_datetime(due_str).date()
        delta = (d - today).days
        return 1.0 + max(0, 7 - delta) / 7.0
    except Exception:
        return 1.0


def compute_task_priority(df: pd.DataFrame, weights: dict, today: date) -> pd.DataFrame:
    df = df.copy()
    for c, default in ("impact", 3.0), ("confidence", 0.7), ("energy", 0.7), ("duration", 1.0):
        if c not in df.columns:
            df[c] = default
    df["duration"] = _clamp(df["duration"], 0.25, 10.0)
    df["urgency"] = df.get("due_date", pd.Series([None] * len(df))).apply(lambda x: _urgency_from_due(x, today))

    wI = float(weights.get("impact", 1.0))
    wC = float(weights.get("confidence", 1.0))
    wEn = float(weights.get("energy", 1.0))
    wU = float(weights.get("urgency", 1.0))
    wE = max(0.1, float(weights.get("effort", 1.0)))

    df["priority"] = (wI*df["impact"] + wC*df["confidence"] + wEn*df["energy"] + wU*df["urgency"]) / (df["duration"] * wE)
    df["priority_per_hour"] = df["priority"] / df["duration"]
    return df


def suggest_plan(df: pd.DataFrame, capacity_hours: float) -> pd.DataFrame:
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


# ----------------------------------
# Follow-through utilities
# ----------------------------------

def today_str() -> str:
    return date.today().isoformat()


def top3_open_count(today_iso: Optional[str] = None) -> int:
    today_iso = today_iso or today_str()
    cur = get_conn().cursor()
    cur.execute("SELECT COUNT(1) FROM tasks WHERE today_top3=1 AND done=0 AND top3_date=?", (today_iso,))
    (cnt,) = cur.fetchone()
    return int(cnt)


def can_add_to_top3(today_iso: Optional[str] = None, cap: int = 3) -> bool:
    return top3_open_count(today_iso) < cap


def set_top3(task_id: int, active: bool, today_iso: Optional[str] = None) -> bool:
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
    t_iso = today_str()
    conn = get_conn()
    tdf = pd.read_sql_query(
        "SELECT * FROM tasks WHERE today_top3=1 AND done=0 AND top3_date=? ORDER BY id ASC",
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
        conn.execute(
            "UPDATE tasks SET planned_start=?, planned_end=? WHERE id=?",
            (cur_start.isoformat(), (cur_start + timedelta(minutes=dur_minutes)).isoformat(), int(row.id)),
        )
        cur_start += timedelta(minutes=dur_minutes)
    conn.commit()

    cal = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//Clarity Board//EN\n" + "".join(events) + "END:VCALENDAR\n"
    return cal.encode("utf-8")


def send_slack_message(text: str):
    if not SLACK_WEBHOOK_URL:
        return False, "No SLACK_WEBHOOK_URL configured"
    try:
        import json, urllib.request
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            _ = resp.read()
        return True, "sent"
    except Exception as e:
        return False, str(e)


def send_email(subject: str, body: str):
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


# ----------------------------------
# Diagnostics
# ----------------------------------

def run_self_checks() -> pd.DataFrame:
    results = []
    try:
        # Priority urgency behaviour
        today = date.today()
        df = pd.DataFrame({
            "title": ["Due tomorrow", "Due 10d"],
            "duration": [1.0, 1.0],
            "impact": [5, 5],
            "confidence": [0.8, 0.8],
            "energy": [0.7, 0.7],
            "due_date": [ (pd.Timestamp(today)+pd.Timedelta(days=1)).date().isoformat(), (pd.Timestamp(today)+pd.Timedelta(days=10)).date().isoformat() ],
        })
        pr = compute_task_priority(df, {"impact":1,"confidence":1,"energy":1,"urgency":1,"effort":1}, today).sort_values("priority", ascending=False)
        results.append({"test":"Urgent ranks higher","passed": pr.iloc[0]["title"]=="Due tomorrow"})

        # Capacity packing
        df2 = pd.DataFrame({
            "title": ["A","B","C"],
            "duration": [1.0,2.0,3.0],
            "impact": [5,4,3],
            "confidence": [0.9,0.8,0.7],
            "energy": [0.8,0.8,0.8],
        })
        pr2 = compute_task_priority(df2, {"impact":1,"confidence":1,"energy":1,"urgency":1,"effort":1}, today)
        plan = suggest_plan(pr2, 3.0)
        results.append({"test":"Plan <= capacity","passed": plan["cumulative_hours"].max() <= 3.0 + 1e-9 if not plan.empty else True})

        # WIP cap logic quick check
        conn = get_conn(); conn.execute("DELETE FROM tasks"); conn.commit()
        for i in range(3):
            conn.execute(
                "INSERT INTO tasks (created_at, title, today_top3, top3_date) VALUES (?,?,1,?)",
                (datetime.utcnow().isoformat(), f"T{i+1}", today_str()),
            )
        conn.commit()
        results.append({"test":"WIP cap hits at 3","passed": can_add_to_top3() is False})
    except Exception as e:
        results.append({"test":"Unexpected error","passed": False, "error": str(e)})
    return pd.DataFrame(results)


# ----------------------------------
# Streamlit UI (Workflow-first)
# ----------------------------------

def do_rerun():
    # Streamlit changed API: prefer st.rerun, fallback to experimental
    fn = getattr(st, "rerun", None) if HAS_STREAMLIT else None
    if fn:
        fn()
    elif HAS_STREAMLIT and hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


if HAS_STREAMLIT:
    st.set_page_config(page_title="Clarity — Follow‑Through", page_icon="✅", layout="wide")
    init_db()

    st.title("Clarity — Task Prioritiser & Daily Follow‑Through")
    st.caption("Capture → Extract → Prioritise → Top‑3 → Calendar → Ship")

    with st.sidebar:
        st.header("Nudges (optional)")
        colN1, colN2 = st.columns(2)
        with colN1:
            if st.button("Test Slack"):
                ok, msg = send_slack_message("Clarity test: Slack OK.")
                st.success("Slack OK") if ok else st.warning(msg)
        with colN2:
            if st.button("Test Email"):
                ok, msg = send_email("Clarity test", "Email configuration works.")
                st.success("Email OK") if ok else st.warning(msg)
        st.markdown("---")
        if st.button("Run diagnostics"):
            st.dataframe(run_self_checks(), use_container_width=True, hide_index=True)

    tabs = st.tabs(["Ideas & Notes", "Task Prioritiser", "Follow‑Through"])

    # ------------- Ideas & Notes (no scoring) -------------
    with tabs[0]:
        st.subheader("Capture ideas and document them")
        with st.form("new_note", clear_on_submit=True):
            n_title = st.text_input("Title")
            n_tag = st.text_input("Tag (optional)")
            n_body = st.text_area("Notes (Markdown)", height=160, help="Write freely. You'll extract tasks on the right.")
            submitted = st.form_submit_button("Save note")
            if submitted and n_title:
                get_conn().execute(
                    "INSERT INTO ideas (created_at, title, tag, body, status) VALUES (?,?,?,?, 'active')",
                    (datetime.utcnow().isoformat(), n_title.strip(), n_tag.strip(), n_body.strip()),
                )
                get_conn().commit()
                st.success("Saved ✅")

        st.markdown("---")
        colL, colR = st.columns([1.2, 1])
        with colL:
            q = st.text_input("Search", placeholder="title, tag or text…")
            if q:
                notes = pd.read_sql_query(
                    "SELECT * FROM ideas WHERE status='active' AND (title LIKE ? OR tag LIKE ? OR body LIKE ?) ORDER BY id DESC",
                    get_conn(), params=(f"%{q}%", f"%{q}%", f"%{q}%"),
                )
            else:
                notes = pd.read_sql_query("SELECT * FROM ideas WHERE status='active' ORDER BY id DESC", get_conn())
            st.caption(f"{len(notes)} notes")
            if notes.empty:
                st.info("No notes yet. Create one above.")
            else:
                for row in notes.itertuples():
                    with st.expander(f"#{row.id} — {row.title}  ·  {row.tag or ''}"):
                        st.markdown(row.body or "_No details_")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if st.button("Archive", key=f"arc_{row.id}"):
                                get_conn().execute("UPDATE ideas SET status='archived' WHERE id=?", (row.id,))
                                get_conn().commit()
                                st.toast("Archived")
                                do_rerun()
                        with c2:
                            new_title = st.text_input("Edit title", value=row.title, key=f"edit_title_{row.id}")
                            new_tag = st.text_input("Edit tag", value=row.tag or "", key=f"edit_tag_{row.id}")
                            new_body = st.text_area("Edit notes", value=row.body or "", key=f"edit_body_{row.id}")
                            if st.button("Save edits", key=f"save_{row.id}"):
                                get_conn().execute(
                                    "UPDATE ideas SET title=?, tag=?, body=? WHERE id=?",
                                    (new_title.strip(), new_tag.strip(), new_body, row.id),
                                )
                                get_conn().commit()
                                st.toast("Updated")
                                do_rerun()
                        with c3:
                            st.write("**Extract tasks** (one per line)")
                            raw = st.text_area("", key=f"extract_{row.id}", height=120)
                            due = st.date_input("Due (optional)", value=None, key=f"due_{row.id}")
                            ctx = st.text_input("Context", placeholder="deep-work/calls/errands", key=f"ctx_{row.id}")
                            if st.button("Create tasks", key=f"ct_{row.id}"):
                                lines = [l.strip() for l in raw.split("\n") if l.strip()]
                                for line in lines:
                                    get_conn().execute(
                                        """
                                        INSERT INTO tasks (idea_id, created_at, title, notes, due_date, done, duration, impact, confidence, energy, context, standalone)
                                        VALUES (?,?,?,?,?,0,1.0,5,0.8,0.7,?,0)
                                        """,
                                        (row.id, datetime.utcnow().isoformat(), line, f"From idea #{row.id}: {row.title}", (due.isoformat() if due else None), ctx),
                                    )
                                get_conn().commit()
                                st.success(f"Created {len(lines)} task(s) ✅")
                                do_rerun()

        with colR:
            st.subheader("Archive / Restore")
            archived = pd.read_sql_query("SELECT * FROM ideas WHERE status='archived' ORDER BY id DESC", get_conn())
            for row in archived.itertuples():
                c1, c2 = st.columns([0.8, 0.2])
                with c1:
                    st.write(f"#{row.id} — {row.title}")
                with c2:
                    if st.button("Restore", key=f"res_{row.id}"):
                        get_conn().execute("UPDATE ideas SET status='active' WHERE id=?", (row.id,))
                        get_conn().commit()
                        st.toast("Restored")
                        do_rerun()

    # ------------- Task Prioritiser -------------
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
        st.subheader("Add Task")
        with st.form("add_task", clear_on_submit=True):
            c1, c2, c3 = st.columns([2,1,1])
            with c1:
                t_title = st.text_input("Task title")
                t_notes = st.text_area("Notes", height=80)
                t_ctx = st.text_input("Context (deep-work/calls/errands)")
            with c2:
                t_due = st.date_input("Due date (optional)", value=None)
                t_dur = st.number_input("Duration (hours)", min_value=0.25, max_value=10.0, value=1.0, step=0.25)
            with c3:
                t_imp = st.slider("Impact", 1, 10, 5)
                t_conf = st.slider("Confidence", 0.0, 1.0, 0.7, 0.05)
                t_energy = st.slider("Energy fit", 0.0, 1.0, 0.7, 0.05)
            sub = st.form_submit_button("Add task")
            if sub and t_title:
                get_conn().execute(
                    """
                    INSERT INTO tasks (idea_id, created_at, title, notes, due_date, done, duration, impact, confidence, energy, context, standalone)
                    VALUES (NULL,?,?,?,?,0,?,?,?,?,?,1)
                    """,
                    (
                        datetime.utcnow().isoformat(),
                        t_title.strip(),
                        (t_due.isoformat() if t_due else None),
                        float(t_dur),
                        float(t_imp),
                        float(t_conf),
                        float(t_energy),
                        t_ctx.strip(),
                    ),
                )
                get_conn().commit()
                st.success("Task added ✅")

        # List & prioritise
        tdf = pd.read_sql_query(
            "SELECT * FROM tasks ORDER BY done ASC, due_date IS NULL, due_date ASC, id DESC",
            get_conn(),
        )
        if tdf.empty:
            st.info("No tasks yet. Add one above or extract from a note.")
        else:
            pr_df = compute_task_priority(tdf, t_weights, date.today())
            st.subheader("Prioritised Tasks")
            cols = ["id","title","context","due_date","duration","impact","confidence","energy","urgency","priority","today_top3"]
            st.dataframe(pr_df[cols].sort_values(["today_top3","priority"], ascending=[False, False]), use_container_width=True, hide_index=True)

            st.subheader("Suggested Plan (fits weekly capacity)")
            plan = suggest_plan(pr_df[pr_df["done"]==0], capacity)
            if plan.empty:
                st.info("No eligible tasks to plan.")
            else:
                show_cols = ["id","title","duration","priority","cumulative_hours"]
                st.dataframe(plan[show_cols], use_container_width=True, hide_index=True)

            st.subheader("Quick actions")
            for row in pr_df.itertuples():
                cols = st.columns([0.06, 0.44, 0.18, 0.16, 0.16])
                with cols[0]:
                    d_toggle = st.checkbox("", value=bool(row.done), key=f"tdone_{row.id}")
                with cols[1]:
                    st.write(f"**#{row.id}** {row.title}")
                with cols[2]:
                    st.write((pd.to_datetime(row.due_date).date() if row.due_date else "—"))
                with cols[3]:
                    currently_top3 = bool(row.today_top3 and (row.top3_date == today_str()))
                    add_disabled = (not currently_top3) and (not can_add_to_top3())
                    if currently_top3:
                        if st.button("Remove from Today", key=f"rm_top3_{row.id}"):
                            set_top3(int(row.id), False)
                            do_rerun()
                    else:
                        if st.button("Add to Today", key=f"add_top3_{row.id}", disabled=add_disabled):
                            ok = set_top3(int(row.id), True)
                            if not ok:
                                st.warning("Top‑3 is full. Complete or remove one first.")
                            do_rerun()
                with cols[4]:
                    if st.button("Delete", key=f"t_del_{row.id}"):
                        get_conn().execute("DELETE FROM tasks WHERE id=?", (row.id,))
                        get_conn().commit()
                        st.toast("Deleted")
                        do_rerun()
                if d_toggle != bool(row.done):
                    get_conn().execute("UPDATE tasks SET done=? WHERE id=?", (int(d_toggle), row.id))
                    get_conn().commit()

    # ------------- Follow‑Through -------------
    with tabs[2]:
        st.subheader("Top‑3‑Today (WIP cap 3)")
        cur_top3 = pd.read_sql_query(
            "SELECT * FROM tasks WHERE today_top3=1 AND done=0 AND top3_date=? ORDER BY id ASC",
            get_conn(), params=(today_str(),)
        )
        st.write(f"Active: **{len(cur_top3)}/3**")
        if cur_top3.empty:
            st.info("No tasks in Top‑3 yet. Add up to 3 from the Task Prioritiser.")
        else:
            st.dataframe(cur_top3[["id","title","duration","due_date","context"]], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Build calendar blocks (ICS export)")
        default_time = (datetime.now().replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)).time()
        start_time_val = st.time_input("Start time today", value=default_time)
        if st.button("Generate .ics for Top‑3"):
            start_dt = datetime.combine(date.today(), start_time_val)
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
            "SELECT COUNT(1) as c FROM tasks WHERE today_top3=1 AND top3_date=? AND done=1",
            get_conn(), params=(today_str(),)
        ).iloc[0].c
        total_today = max(1, len(cur_top3))
        score = round(100.0 * done_today / total_today, 1)
        st.metric("Action Score", f"{score}%", help="Completed / Top‑3 for today")

else:
    # CLI fallback
    init_db()
    print("[Clarity] Streamlit is not installed. CLI diagnostics mode.\n")
    print(run_self_checks().to_string(index=False))
    demo = pd.DataFrame({
        "title": ["Write brief", "Call supplier", "Deep work block"],
        "duration": [1.0, 0.5, 2.0],
        "impact": [6, 4, 8],
        "confidence": [0.9, 0.8, 0.7],
        "energy": [0.8, 0.7, 0.9],
        "due_date": [None, (pd.Timestamp(date.today())+pd.Timedelta(days=1)).date().isoformat(), None],
        "done": [0,0,0],
    })
    weights = {"impact":1.0, "confidence":1.0, "energy":1.0, "urgency":1.0, "effort":1.0}
    pr = compute_task_priority(demo, weights, date.today())
    print("\nDemo priorities:\n", pr[["title","duration","urgency","priority"]])
    plan = suggest_plan(pr, capacity_hours=2.0)
    print("\nSuggested 2h plan:\n", plan[["title","duration","priority","cumulative_hours"]])
