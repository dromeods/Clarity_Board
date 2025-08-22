# 🧠 Clarity Board

A lightweight weekly prioritization app built with **Streamlit** + **SQLite**.  
Capture your ideas, score them with ICE/RICE, and commit to the most important ones without drowning in overload.

---

## 🚀 Features

- **Idea Inbox** — jot down all ideas/thoughts for the week.  
- **Scoring System** — rank with ICE or RICE, using **linear weights** (Impact, Confidence, Effort, Reach).  
- **Effort Clamp** — avoids division-by-zero or extreme skew (range `[0.25, 10]`).  
- **Auto-Carry** — unfinished *selected* ideas roll over into the next week.  
- **Visualization** — bubble chart (Impact vs Effort, size=Confidence, color=Score).  
- **Task Tracking** — break ideas into tasks with due dates.  
- **Export/Archive** — send ideas/tasks to CSV; archive leftovers to an “Idea Bank.”  
- **Diagnostics Mode** — run `python app.py` (no Streamlit required) to self-check logic in a CLI sandbox.

---

## 📦 Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
