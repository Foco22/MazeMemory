import sys
import json
import subprocess
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent   # MazeMemory/
RESULTS_DIR = ROOT / "results" / "experiments"
VIDEOS_DIR  = ROOT / "results" / "videos"
RUN_SCRIPT  = ROOT / "experiments" / "run.py"

DIFFICULTY_TO_MAZE = {"easy": 1, "medium": 2, "hard": 3}
MAZE_LABELS = {"easy": "Easy (7×7)", "medium": "Medium (15×15)", "hard": "Hard (33×33)"}
SCENARIOS   = ["baseline", "shared_memory", "observer", "shared_memory_observer"]

MODELS = {
    "deepseek/deepseek-v4-flash  ($0.14/$0.28)":  ("deepseek/deepseek-v4-flash",  "deepseek"),
    "deepseek/deepseek-v4-pro    ($0.44/$0.87)":  ("deepseek/deepseek-v4-pro",    "deepseek"),
    "claude-haiku-4-5            ($1.00/$5.00)":  ("claude-haiku-4-5-20251001",   "anthropic"),
    "claude-sonnet-4-6           ($3.00/$15.00)": ("claude-sonnet-4-6",            "anthropic"),
    "gpt-4o-mini                 ($0.15/$0.60)":  ("gpt-4o-mini",                 "openai"),
    "gpt-4o                      ($2.50/$10.00)": ("gpt-4o",                      "openai"),
    "gemini-2.0-flash            ($0.08/$0.30)":  ("gemini/gemini-2.0-flash",     "google"),
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="MazeMemory", layout="wide")
st.title("MazeMemory — Experiment Runner")
st.caption("MSc Thesis · Birkbeck University · Shared Memory in LLM Multi-Agent Systems")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Configure & Run
# ─────────────────────────────────────────────────────────────────────────────
st.header("Step 1 — Configure & Run")

col1, col2, col3 = st.columns(3)

with col1:
    scenario = st.selectbox(
        "Scenario",
        SCENARIOS,
        format_func=lambda s: s.replace("_", " ").title(),
    )

with col2:
    difficulty = st.selectbox(
        "Maze Difficulty",
        list(MAZE_LABELS.keys()),
        format_func=lambda d: MAZE_LABELS[d],
    )

with col3:
    n_runs = st.slider("Number of runs", min_value=1, max_value=30, value=5)

model_label = st.selectbox("Model", list(MODELS.keys()))
selected_model, selected_provider = MODELS[model_label]

col_a, col_b, col_c = st.columns(3)
with col_a:
    live      = st.checkbox("Live output (stream logs)", value=True)
with col_b:
    save_video = st.checkbox("Generate GIF video after run", value=False)
with col_c:
    save_db    = st.checkbox("Save results to Supabase", value=True)

run_btn = st.button("▶  Run Experiment", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Execute & stream logs
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    cmd = [
        sys.executable, str(RUN_SCRIPT),
        "--model",      selected_model,
        "--provider",   selected_provider,
        "--scenarios",  scenario,
        "--difficulty", difficulty,
        "--n-runs",     str(n_runs),
    ]
    if live:
        cmd.append("--live")
    if not save_video:
        cmd.append("--no-video")
    if not save_db:
        cmd.append("--no-db")

    st.header("Step 2 — Running…")
    log_box = st.empty()
    gif_slot = st.empty()
    lines: list[str] = []

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
        bufsize=1,
    ) as proc:
        for line in proc.stdout:
            lines.append(line)
            log_box.code("".join(lines[-40:]), language=None)
        proc.wait()
        exit_ok = proc.returncode == 0

    if exit_ok:
        st.success("✅ Experiment completed!")
    else:
        st.error("❌ Experiment failed — check the logs above.")

    # Show latest GIF if video was generated
    if save_video and VIDEOS_DIR.exists():
        gifs = sorted(VIDEOS_DIR.rglob(f"{scenario}_maze{DIFFICULTY_TO_MAZE[difficulty]}_*.gif"))
        if gifs:
            gif_slot.image(str(gifs[-1]), caption="Latest run — GIF replay")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Results
    # ─────────────────────────────────────────────────────────────────────────
    if exit_ok and RESULTS_DIR.exists():
        st.header("Step 3 — Results")

        maze_id = DIFFICULTY_TO_MAZE[difficulty]
        files   = sorted(RESULTS_DIR.glob(f"{scenario}_maze{maze_id}_*.json"))

        if not files:
            st.info("No result files found yet for this combination.")
        else:
            rows = []
            for f in files:
                data = json.loads(f.read_text())
                for agent in data.get("agents", []):
                    rows.append({
                        "run":               data.get("run_number"),
                        "agent":             f"Agent {agent['agent_id']}",
                        "steps":             agent.get("steps"),
                        "optimal_steps":     agent.get("optimal_steps"),
                        "optimality_ratio":  agent.get("optimality_ratio"),
                        "tokens":            agent.get("total_tokens"),
                        "redundancy_ratio":  agent.get("redundancy_ratio"),
                        "reached_exit":      agent.get("reached_exit"),
                    })

            df = pd.DataFrame(rows)

            # Summary table
            st.subheader("Run summary")
            st.dataframe(
                df.style.format({
                    "optimality_ratio": "{:.3f}",
                    "redundancy_ratio": "{:.3f}",
                }, na_rep="—"),
                use_container_width=True,
            )

            # Charts — 3 metrics
            st.subheader("Metric charts (all saved runs for this scenario + difficulty)")
            c1, c2, c3 = st.columns(3)

            with c1:
                fig = px.box(
                    df.dropna(subset=["optimality_ratio"]),
                    y="optimality_ratio",
                    color="agent",
                    title="Path Optimality Ratio",
                    labels={"optimality_ratio": "ratio (lower = better)"},
                )
                fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                              annotation_text="optimal")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.box(
                    df.dropna(subset=["tokens"]),
                    y="tokens",
                    color="agent",
                    title="Token Consumption",
                    labels={"tokens": "tokens (lower = cheaper)"},
                )
                st.plotly_chart(fig, use_container_width=True)

            with c3:
                fig = px.box(
                    df.dropna(subset=["redundancy_ratio"]),
                    y="redundancy_ratio",
                    color="agent",
                    title="Redundant Computation Ratio",
                    labels={"redundancy_ratio": "ratio (lower = better)"},
                )
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — saved runs overview
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Saved runs")
    if RESULTS_DIR.exists():
        all_files = list(RESULTS_DIR.glob("*.json"))
        st.metric("Total runs", len(all_files))

        counts: dict[str, int] = {}
        for f in all_files:
            sc = f.stem.split("_maze")[0]
            counts[sc] = counts.get(sc, 0) + 1

        for sc, n in sorted(counts.items()):
            st.write(f"**{sc.replace('_', ' ')}**: {n} runs")
    else:
        st.info("No results yet.")