import sqlite3
import os
from datetime import datetime, date
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

# ----------------------------
# Database helpers
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
            FOREIGN KEY(idea_id) REFERENCES ideas(id)
        )
        """
    )
    conn.commit()


def get_or_create_week(week_start: date) -> Tuple[int, bool]:
    """Return (week_id, created_new)."""
    conn = get_conn()
    cur = conn.cursor()
    wk = week_start.isoformat()
    # Try insert; sqlite3 total_changes increases if a row was inserted in this connection
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
    """Copy last week's *selected* (not done) ideas into the new week unless already present by title.
    Returns number of carried items.
    """
    prev_id = get_previous_week_id(new_week_start)
    if not prev_id:
        return 0
    conn = get_conn()
    cur = conn.cursor()
    # Titles already in new week
    cur.execute("SELECT LOWER(TRIM(title)) FROM ideas WHERE week_id=?", (new_week_id,))
    existing_titles = {t[0] for t in cur.fetchall()}

    # Get selected ideas from previous week
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
# Scoring (Linear weights + sensible effort clamp)
# ----------------------------

def _clamp_effort(e: pd.Series, lo: float = 0.25, hi: float = 10.0) -> pd.Series:
    e = e.fillna(lo)
    e = e.clip(lower=lo, upper=hi)
    return e

def compute_scores(df: pd.DataFrame, method: str, weights: dict) -> pd.DataFrame:
    """Compute ICE/RICE scores using **linear multipliers** for weights.

    ICE:  score = (impact * w_i) * (confidence * w_c) / (effort * w_e)
    RICE: score = (reach * w_r) * (impact * w_i) * (confidence * w_c) / (effort * w_e)

    Effort is clamped to [0.25, 10] to avoid pathological values.
    """
    df = df.copy()
    # Clamp effort to sensible range
    df["effort"] = _clamp_effort(df["effort"], 0.25, 10.0)

    w_i = float(weights.get("impact", 1.0))
    w_c = float(weights.get("confidence", 1.0))
    w_e = max(0.1, float(weights.get("effort", 1.0)))  # keep penalty reasonable

    if method == "ICE":
        df["score"] = (df["impact"] * w_i) * (df["confidence"] * w_c) / (df["effort"] * w_e)
    else:  # RICE
        w_r = float(weights.get("reach", 1.0))
        # Treat reach<=0 as small positive to avoid nullifying everything
        df["reach"] = df["reach"].replace(0, 0.25).fillna(0.25)
        df["score"] = (df["reach"] * w_r) * (df["impact"] * w_i) * (df["confidence"] * w_c) / (df["effort"] * w_e)
    return df


# ----------------------------
# Self-checks (lightweight tests)
# ----------------------------

def run_self_checks() -> pd.DataFrame:
    """Return a dataframe of test results for quick diagnostics.
    NOTE: These tests validate the scoring function & core helpers; they do not cover the UI.
    """
    results = []
    try:
        weights = {"reach": 1.0, "impact": 1.0, "confidence": 1.0, "effort": 1.0}

        # Test 1: ICE prefers lower effort if impact/confidence equal
        df = pd.DataFrame({
            "reach": [1, 1],
            "impact": [5, 5],
            "confidence": [0.8, 0.8],
            "effort": [1.0, 2.0],
        })
        ice = compute_scores(df, "ICE", weights).sort_values("score", ascending=False)
        results.append({"test": "ICE favors lower effort when others equal", "passed": ice.iloc[0]["effort"] < ice.iloc[1]["effort"]})

        # Test 2: RICE increases with reach when other factors equal
        df = pd.DataFrame({
            "reach": [100, 10],
            "impact": [5, 5],
            "confidence": [0.8, 0.8],
            "effort": [2.0, 2.0],
        })
        rice = compute_scores(df, "RICE", weights).sort_values("score", ascending=False)
        results.append({"test": "RICE increases with reach", "passed": rice.iloc[0]["reach"] > rice.iloc[1]["reach"]})

        # Test 3: Zero effort is clamped and does not crash
        df = pd.DataFrame({
            "reach": [1],
            "impact": [1],
            "confidence": [1.0],
            "effort": [0.0],
        })
        ice_zero = compute_scores(df, "ICE", weights)
        results.append({"test": "Effort=0 handled safely (clamped)", "passed": np.isfinite(ice_zero.loc[0, "score"]) and ice_zero.loc[0, "score"] > 0})

        # Test 4: Week creation stable (idempotent)
        wk_id_1, _ = get_or_create_week(date(2025, 1, 6))
        wk_id_2, _ = get_or_create_week(date(2025, 1, 6))
        results.append({"test": "get_or_create_week idempotent", "passed": wk_id_1 == wk_id_2})

        # Test 5: Increasing effort lowers score (monotonicity)
        df = pd.DataFrame({
            "reach": [1, 1, 1],
            "impact": [5, 5, 5],
            "confidence": [0.9, 0.9, 0.9],
            "effort": [1.0, 2.0, 3.0],
        })
        ice = compute_scores(df, "ICE", weights)
        s = ice.sort_values("effort")["score"].values
        results.append({"test": "Score decreases as effort increases (ICE)", "passed": (s[0] > s[1] > s[2])})

        # Test 6: Increasing impact raises score (monotonicity)
        df = pd.DataFrame({
            "reach": [1, 1, 1],
            "impact": [3, 5, 7],
            "confidence": [0.8, 0.8, 0.8],
            "effort": [2.0, 2.0, 2.0],
        })
        ice = compute_scores(df, "ICE", weights)
        s = ice.sort_values("impact")["score"].values
        results.append({"test": "Score increases with impact (ICE)", "passed": (s[0] < s[1] < s[2])})

        # Test 7: If reach is equal for all, RICE ranking equals ICE ranking
        df = pd.DataFrame({
            "reach": [10, 10, 10],
            "impact": [4, 6, 5],
            "confidence": [0.7, 0.9, 0.8],
            "effort": [2.0, 2.0, 2.0],
        })
        ice_rank = compute_scores(df, "ICE", weights).sort_values("score", ascending=False)["score"].rank(method="first").tolist()
        rice_rank = compute_scores(df, "RICE", weights).sort_values("score", ascending=False)["score"].rank(method="first").tolist()
        results.append({"test": "RICE==ICE ordering when reach constant", "passed": ice_rank == rice_rank})

        # Added tests
        # Test 8: RICE handles reach=0 safely and remains finite
        df = pd.DataFrame({
            "reach": [0, 5],
            "impact": [5, 5],
            "confidence": [0.8, 0.8],
            "effort": [2.0, 2.0],
        })
        rice = compute_scores(df, "RICE", weights)
        results.append({"test": "RICE handles reach=0 safely", "passed": np.all(np.isfinite(rice["score"]))})

        # Test 9: Determinism – same inputs produce same scores
        df = pd.DataFrame({
            "reach": [3, 4],
            "impact": [6, 6],
            "confidence": [0.9, 0.9],
            "effort": [1.5, 1.5],
        })
        s1 = compute_scores(df, "ICE", weights)["score"].values
        s2 = compute_scores(df, "ICE", weights)["score"].values
        results.append({"test": "Deterministic outputs (ICE)", "passed": np.allclose(s1, s2)})

        # Test 10: Increasing effort weight reduces scores (penalty weight, linear)
        df = pd.DataFrame({
            "reach": [1],
            "impact": [6],
            "confidence": [0.9],
            "effort": [2.0],
        })
        low_pen = compute_scores(df, "ICE", {**weights, "effort": 0.5})["score"].iloc[0]
        high_pen = compute_scores(df, "ICE", {**weights, "effort": 2.0})["score"].iloc[0]
        results.append({"test": "Higher effort weight lowers score (linear)", "passed": high_pen < low_pen})

    except Exception as e:
        results.append({"test": "Unexpected error in tests", "passed": False, "error": str(e)})

    return pd.DataFrame(results)


# ----------------------------
# Streamlit App UI (only if Streamlit is available)
# ----------------------------

if HAS_STREAMLIT:
    st.set_page_config(page_title="Clarity Board", page_icon="✅", layout="wide")
    init_db()

    st.title("Clarity Board – Weekly Focus Engine")
    st.caption("Capture → Score → Select → Execute → Reflect")

    # Sidebar controls
    with st.sidebar:
        st.header("Weekly Setup")
        today = date.today()
        week_start = today if today.weekday() == 0 else (today - pd.Timedelta(days=today.weekday()))
        wk = st.date_input("Week starting (Mon)", value=week_start)
        week_id, created_new = get_or_create_week(wk)

        # Carry over open selected items when a *new* week is first created
        carried = 0
        if created_new:
            carried = carry_over_open_selected(week_id, wk)
            if carried:
                st.toast(f"Carried over {carried} selected item(s) from last week.")

        st.header("Scoring (linear weights)")
        method = st.radio("Model", ["ICE", "RICE"], index=0, help="ICE = Impact, Confidence, Effort. RICE adds Reach.")
        colA, colB = st.columns(2)
        with colA:
            w_impact = st.slider("Weight: Impact (×)", 0.1, 3.0, 1.0, 0.1)
            w_conf = st.slider("Weight: Confidence (×)", 0.1, 3.0, 1.0, 0.1)
        with colB:
            w_eff = st.slider("Weight: Effort penalty (×)", 0.1, 3.0, 1.0, 0.1)
            w_reach = st.slider("Weight: Reach (×, RICE)", 0.1, 3.0, 1.0, 0.1)

        st.markdown("---")
        st.header("Selection")
        top_k = st.slider("How many to commit to this week?", 1, 5, 3)

        st.markdown("---")
        st.header("Diagnostics")
        if st.button("Run self-checks"):
            results_df = run_self_checks()
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            if not results_df["passed"].all():
                st.error("One or more self-checks failed. See table above.")
            else:
                st.success("All self-checks passed ✅")

    # Capture new ideas
    st.subheader("1) Capture – Dump your ideas/thoughts")
    with st.form("capture"):
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

    # Load data
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT i.*, w.week_start FROM ideas i JOIN weeks w ON i.week_id=w.id WHERE i.week_id=?",
        conn,
        params=(week_id,),
    )

    # Score + rank
    if not df.empty:
        weights = {"reach": w_reach, "impact": w_impact, "confidence": w_conf, "effort": w_eff}
        df_scored = compute_scores(df, method, weights)
        df_scored = df_scored.sort_values("score", ascending=False)

        st.subheader("2) Score – Ranked ideas for the week")
        st.dataframe(
            df_scored[["id", "status", "tag", "title", "reach", "impact", "confidence", "effort", "score"]]
            .rename(columns={"tag": "Tag", "title": "Title", "reach": "Reach", "impact": "Impact", "confidence": "Confidence", "effort": "Effort", "score": "Score"}),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("3) Visualize – Impact vs Effort")
        chart_df = df_scored[["title", "impact", "effort", "score"]]
        chart_df = chart_df.rename(columns={"impact": "Impact", "effort": "Effort", "title": "Title", "score": "Score"})
        st.caption("Top-left = high impact, low effort → fast wins")
        st.scatter_chart(chart_df, x="Effort", y="Impact", size="Score", color=None)

        st.subheader("4) Select – Commit to the few that matter")
        selectable = df_scored[df_scored["status"].isin(["inbox", "archived"])]
        choices = st.multiselect(
            "Pick up to N items to commit this week",
            options=[f"#{row.id} – {row.title}" for row in selectable.itertuples()],
            max_selections=top_k,
        )
        if st.button("Commit selected") and choices:
            ids = [int(c.split(" – ")[0].replace("#", "")) for c in choices]
            qmarks = ",".join(["?"] * len(ids))
            conn.execute(f"UPDATE ideas SET status='selected' WHERE id IN ({qmarks})", ids)
            conn.commit()
            st.success("Committed to weekly focus ✅")

        st.subheader("5) Execute – Generate tasks for selected items")
        selected_df = df_scored[df_scored["status"] == "selected"]
        if selected_df.empty:
            st.info("No selected items yet. Commit a few above.")
        else:
            for row in selected_df.itertuples():
                with st.expander(f"#{row.id} – {row.title}"):
                    st.write(row.description)
                    default_tasks = [
                        f"Define success for: {row.title}",
                        f"Break down: {row.title} into 3 steps",
                        f"Do Step 1 for: {row.title}",
                    ]
                    custom_tasks = st.text_area("Tasks (one per line)", value="\n".join(default_tasks), key=f"tasks_{row.id}")
                    due = st.date_input("Due date", value=wk + pd.Timedelta(days=6), key=f"due_{row.id}")
                    if st.button("Create tasks", key=f"create_{row.id}"):
                        for t in [t.strip() for t in custom_tasks.split("\n") if t.strip()]:
                            conn.execute(
                                "INSERT INTO tasks (idea_id, created_at, title, due_date, done) VALUES (?,?,?,?,0)",
                                (row.id, datetime.utcnow().isoformat(), t, due.isoformat()),
                            )
                        conn.commit()
                        st.success("Tasks created ✅")

        st.subheader("6) Track – Task list")
        tasks_df = pd.read_sql_query(
            "SELECT t.id, t.idea_id, i.title as idea_title, t.title, t.due_date, t.done FROM tasks t JOIN ideas i ON t.idea_id=i.id ORDER BY t.due_date ASC",
            conn,
        )
        if tasks_df.empty:
            st.info("No tasks yet.")
        else:
            for row in tasks_df.itertuples():
                cols = st.columns([0.06, 0.64, 0.15, 0.15])
                with cols[0]:
                    done_toggle = st.checkbox("", value=bool(row.done), key=f"done_{row.id}")
                with cols[1]:
                    st.write(f"**#{row.id}** {row.title} \n<small>From idea #{row.idea_id}: {row.idea_title}</small>", unsafe_allow_html=True)
                with cols[2]:
                    st.write(pd.to_datetime(row.due_date).date())
                with cols[3]:
                    if st.button("Delete", key=f"del_{row.id}"):
                        conn.execute("DELETE FROM tasks WHERE id=?", (row.id,))
                        conn.commit()
                        st.experimental_rerun()
                # Update done state
                if done_toggle != bool(row.done):
                    conn.execute("UPDATE tasks SET done=? WHERE id=?", (int(done_toggle), row.id))
                    conn.commit()

        st.subheader("7) Export / Archive")
        colx, coly, colz = st.columns(3)
        with colx:
            csv = df_scored.to_csv(index=False).encode("utf-8")
            st.download_button("Download ideas CSV", csv, file_name=f"ideas_{wk}.csv", mime="text/csv")
        with coly:
            tasks_df = pd.read_sql_query("SELECT * FROM tasks", conn)
            csv_tasks = tasks_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download tasks CSV", csv_tasks, file_name=f"tasks_{wk}.csv", mime="text/csv")
        with colz:
            if st.button("Archive unselected to Idea Bank"):
                conn.execute("UPDATE ideas SET status='archived' WHERE week_id=? AND status='inbox'", (week_id,))
                conn.commit()
                st.success("Archived inbox items ✅")
    else:
        st.info("No ideas captured for this week yet. Add a few above.")

    st.markdown("---")
    st.caption("Built for weekly clarity: focus on the few, ship, repeat.")

# ----------------------------
# CLI fallback (when Streamlit is not installed)
# ----------------------------

if __name__ == "__main__" and not HAS_STREAMLIT:
    init_db()
    print("[Clarity Board] Streamlit is not installed. Running in CLI diagnostics mode.\n")
    print("→ To use the full UI, install Streamlit: 'pip install streamlit' and run 'streamlit run app.py'\n")

    # Run tests
    tests = run_self_checks()
    print("Self-checks:\n", tests.to_string(index=False))

    # Minimal demo of scoring
    demo = pd.DataFrame({
        "title": ["Idea A", "Idea B", "Idea C"],
        "tag": ["business", "personal", "learning"],
        "reach": [10, 5, 1],
        "impact": [7, 6, 8],
        "confidence": [0.8, 0.9, 0.7],
        "effort": [2.0, 3.0, 1.0],
    })
    weights = {"reach": 1.0, "impact": 1.0, "confidence": 1.0, "effort": 1.0}
    ranked = compute_scores(demo, "RICE", weights).sort_values("score", ascending=False)

    print("\nDemo RICE ranking (higher is better):")
    print(ranked[["title", "reach", "impact", "confidence", "effort", "score"]].to_string(index=False))

    print("\nDone. This CLI mode is for verification only.")
