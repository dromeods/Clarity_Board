# 🧠 Clarity Board — Weekly Focus & Follow‑Through

A lightweight, web‑first prioritisation system that turns idea overload into **action**. Capture ideas, score them (ICE/RICE), commit to a few, and **actually ship** using a Top‑3‑Today workflow, WIP caps, calendar blocks, and optional nudges.

---

## ✨ Why this exists
Most tools help you plan. Very few help you **follow through**. Clarity Board enforces constraints (WIP caps), creates a daily action list (Top‑3), and makes your calendar the source of truth (ICS export).

---

## 🚀 Feature Overview
- **Idea Inbox → Weekly Focus**
  - Capture ideas, score with **ICE/RICE** (linear weights), pick your weekly focus.
  - Effort is clamped to **[0.25, 10]** to avoid skew.
  - Auto‑carry: last week’s **selected** (not done) ideas roll into the new week.
- **Super‑Powered Task Prioritiser**
  - Standalone tasks with **Impact, Confidence, Energy, Urgency, Duration**.
  - Priority = `(wI*Impact + wC*Confidence + wEn*Energy + wU*Urgency) / (Duration × wE)`.
  - Capacity‑aware weekly **Suggest Plan** (greedy, respects hours).
- **Follow‑Through Engine**
  - **Top‑3‑Today** with **hard WIP cap = 3**.
  - **ICS export** to block your calendar for the Top‑3.
  - **Nudges**: send Slack or Email reminders (optional env vars).
  - **Action Score**: % of Top‑3 completed today.
- **Diagnostics / Tests**
  - Built‑in self checks validate scoring, carryover, WIP cap, and planning behaviour.

---

## 🧩 Tech Stack
- **UI**: Streamlit
- **DB**: SQLite (file: `clarity_board.db`) — schema auto‑migrates safely.
- **Lang**: Python **3.9+** (recommended **3.10**)

---

## 📦 Install & Run

### 1) Clone & setup
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 2) Dependencies
Create `requirements.txt` (or copy this):
```txt
streamlit
pandas
numpy
```
Install:
```bash
pip install -r requirements.txt
```

### 3) Run the app
```bash
streamlit run app.py
```
Open the local URL (usually `http://localhost:8501`).

### CLI (no Streamlit)
```bash
python app.py
```
Runs self‑checks and a small demo to verify logic.

---

## ☁️ Deploy on Streamlit Cloud (free)
1. Push the repo to GitHub.
2. On **Streamlit Community Cloud** → **New app**.
3. Select your repo/branch, **Main file path: `app.py`**.
4. Add `requirements.txt` at repo root (see above).
5. (Optional) Pin runtime: create `runtime.txt`:
```txt
python-3.10
```
6. Deploy.

---

## 🔐 Optional: Nudges (Slack & Email)
You can trigger message tests from the sidebar. Configure via environment variables:

### Slack (Incoming Webhook)
- Create a Slack **Incoming Webhook** and copy its URL.
- Set env var:
```bash
export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/…'
```

### Email (SMTP)
Provide credentials for your SMTP provider (e.g., Postmark, SendGrid, Gmail SMTP):
```bash
export SMTP_HOST='smtp.yourprovider.com'
export SMTP_PORT='587'
export SMTP_USER='apikey-or-username'
export SMTP_PASS='secret'
export SMTP_TO='you@example.com'
export SMTP_FROM='clarity@yourdomain.com'
```
> Tip: Use a real transactional provider for deliv
