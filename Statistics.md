# Statistical Analysis Plan — MazeMemory Thesis

Jupyter notebook at `notebooks/statistical_analysis.ipynb`.  
Data source: **Supabase** — tables `experiments`, `agent_runs`, `trajectories`.

---

## Cell 0 — Configuration (only cell you edit)

```python
# ── Paste the experiment UUIDs you want to compare ──────────────────────────
# Each list becomes one group in every statistical test.
# You can add as many IDs as you have runs; they don't all need to be the same
# scenario label — the label below is just for plot/table display.

EXPERIMENT_GROUPS = {
    "Baseline": [
        "uuid-baseline-1",
        "uuid-baseline-2",
        # ...
    ],
    "Shared Memory": [
        "uuid-sm-1",
        "uuid-sm-2",
        # ...
    ],
    "Shared Memory + Observer": [
        "uuid-smo-1",
        "uuid-smo-2",
        # ...
    ],
}

# Display order for plots and tables
GROUP_ORDER = ["Baseline", "Shared Memory", "Shared Memory + Observer"]
PALETTE     = {
    "Baseline":                  "#4C72B0",
    "Shared Memory":             "#DD8452",
    "Shared Memory + Observer":  "#55A868",
}
```

---

## Cell 1 — Imports & Supabase connection

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, f_oneway, ttest_ind, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from supabase import create_client          # sync client
from dotenv import load_dotenv

# repo root must be in sys.path so src.* imports work
import sys
sys.path.insert(0, "..")

from src.maze.generator import Maze
from src.metrics.calculator import RedundantComputationReduction

load_dotenv("../.env")

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
```

---

## Cell 2 — Fetch experiment metadata + agent_runs from Supabase

```python
all_ids = [uid for ids in EXPERIMENT_GROUPS.values() for uid in ids]

# experiments table — one row per run
exp_resp = (
    sb.table("experiments")
    .select("id, scenario, maze_id, maze_seed, run_number, model, "
            "total_tokens, total_prompt_tokens, total_completion_tokens, "
            "total_cache_hit_tokens, observer_prompt_tokens, "
            "observer_completion_tokens, cost_total_usd, duration_seconds")
    .in_("id", all_ids)
    .execute()
)
df_exp = pd.DataFrame(exp_resp.data)

# agent_runs table — 3 rows per experiment (one per navigator agent)
ar_resp = (
    sb.table("agent_runs")
    .select("experiment_id, agent_id, steps, optimal_steps, "
            "optimality_ratio, total_tokens, reached_exit")
    .in_("experiment_id", all_ids)
    .execute()
)
df_ar = pd.DataFrame(ar_resp.data)

# Add group label to both DataFrames
id_to_group = {uid: grp for grp, ids in EXPERIMENT_GROUPS.items() for uid in ids}
df_exp["group"] = df_exp["id"].map(id_to_group)
df_ar["group"]  = df_ar["experiment_id"].map(id_to_group)

print(f"experiments loaded: {len(df_exp)}")
print(df_exp.groupby("group")["id"].count())
```

---

## Cell 3 — Fetch trajectories & compute RedundantComputationReduction

RCR needs the full path per agent, which lives in `trajectories`.

```python
traj_resp = (
    sb.table("trajectories")
    .select("experiment_id, agent_id, x, y, timestep")
    .in_("experiment_id", all_ids)
    .execute()
)
df_traj = pd.DataFrame(traj_resp.data)

# Reconstruct run_result dict expected by RedundantComputationReduction.compute()
rcr_rows = []
for exp_id, grp_traj in df_traj.groupby("experiment_id"):
    exp_row   = df_exp[df_exp["id"] == exp_id].iloc[0]
    maze_seed = int(exp_row["maze_seed"])
    maze      = Maze.from_seed(maze_seed)
    calc      = RedundantComputationReduction(maze)

    agents_for_run = []
    for agent_id, ag_traj in grp_traj.groupby("agent_id"):
        path = (
            ag_traj.sort_values("timestep")[["x", "y"]].values.tolist()
        )
        ar_row = df_ar[
            (df_ar["experiment_id"] == exp_id) & (df_ar["agent_id"] == agent_id)
        ]
        reached = bool(ar_row["reached_exit"].values[0]) if len(ar_row) else False
        agents_for_run.append({
            "agent_id": agent_id,
            "path": path,
            "reached_exit": reached,
            "steps": len(path) - 1,
        })

    run_result = {"agents": agents_for_run}
    for r in calc.compute(run_result):
        rcr_rows.append({
            "experiment_id": exp_id,
            "group":         id_to_group[exp_id],
            "agent_id":      r["agent_id"],
            "redundant_cells":     r["redundant_cells"],
            "total_cells_visited": r["total_cells_visited"],
            "rcr_ratio":           r["ratio"],
        })

df_rcr = pd.DataFrame(rcr_rows)
print(f"RCR rows: {len(df_rcr)}")
```

---

## Cell 4 — Build analysis DataFrames (one row per run per group)

```python
# ── Path Optimality Ratio ────────────────────────────────────────────────────
# PRIMARY  (Option B): fleet ratio  = sum(actual_steps) / sum(optimal_steps)
#   Treats the run as a unit; agents with longer optimal paths weight more,
#   which is fairer when starting positions differ across agents.
# SECONDARY (Option A): mean of individual ratios — kept for per-agent diagnosis.

finishers = df_ar[df_ar["reached_exit"]].copy()

por_run = (
    finishers
    .groupby(["group", "experiment_id"])
    .apply(lambda g: pd.Series({
        # PRIMARY — used in all hypothesis tests and plots
        "fleet_por":    g["steps"].sum() / g["optimal_steps"].sum(),
        # SECONDARY — reported in the supplementary table only
        "mean_agent_por": g["optimality_ratio"].mean(),
        "n_agents":     len(g),
    }))
    .reset_index()
)

# Token Consumption
# S3 has observer tokens ON TOP of navigator tokens — split them so comparisons are fair.
#   nav_tokens  = tokens spent by the 3 navigators only (comparable across all scenarios)
#   obs_tokens  = tokens spent by the observer (S3 only; 0 for S1/S2)
#   total_tokens = nav_tokens + obs_tokens

tok_run = df_exp[["group", "id",
                   "total_tokens",
                   "total_prompt_tokens", "total_completion_tokens",
                   "observer_prompt_tokens", "observer_completion_tokens",
                   "cost_total_usd"]].copy()
tok_run = tok_run.rename(columns={"id": "experiment_id"})

obs_prompt = tok_run["observer_prompt_tokens"].fillna(0)
obs_compl  = tok_run["observer_completion_tokens"].fillna(0)
tok_run["obs_tokens"] = obs_prompt + obs_compl
tok_run["nav_tokens"] = tok_run["total_tokens"] - tok_run["obs_tokens"]

# RCR — fleet ratio: sum(redundant_cells) / sum(total_cells_visited) per run
#   Same pooling logic as fleet_por — avoids averaging ratios across agents.
rcr_run = (
    df_rcr.dropna(subset=["rcr_ratio"])
    .groupby(["group", "experiment_id"])
    .apply(lambda g: pd.Series({
        "fleet_rcr":      g["redundant_cells"].sum() / g["total_cells_visited"].sum(),
        "mean_agent_rcr": g["rcr_ratio"].mean(),   # secondary
    }))
    .reset_index()
)

# Exit rate (how often did all 3 agents reach the exit?)
exit_rate = (
    df_ar.groupby(["group", "experiment_id"])["reached_exit"]
    .mean()           # fraction of agents that exited per run
    .reset_index(name="exit_rate")
)

print("Runs per group:")
print(por_run.groupby("group")["experiment_id"].count())
```

---

## Cell 5 — Descriptive statistics

```python
def describe(df, col, label):
    return (
        df.groupby("group")[col]
        .agg(n="count", mean="mean", std="std", median="median",
             q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reindex(GROUP_ORDER)
        .round(4)
    )

print("=== Path Optimality Ratio — fleet (PRIMARY) ===")
print(describe(por_run, "fleet_por", "POR"))

print("\n=== Path Optimality Ratio — mean of agents (secondary) ===")
print(describe(por_run, "mean_agent_por", "POR-agent"))

print("\n=== Navigator Tokens per Run — fair cross-scenario comparison ===")
print(describe(tok_run, "nav_tokens", "NavTokens"))

print("\n=== Observer Tokens per Run (S3 only; 0 for S1/S2) ===")
print(describe(tok_run, "obs_tokens", "ObsTokens"))

print("\n=== Total Tokens per Run (nav + observer) ===")
print(describe(tok_run, "total_tokens", "TotalTokens"))

print("\n=== Redundant Computation Ratio — fleet (PRIMARY) ===")
print(describe(rcr_run, "fleet_rcr", "RCR"))

print("\n=== Redundant Computation Ratio — mean of agents (secondary) ===")
print(describe(rcr_run, "mean_agent_rcr", "RCR-agent"))

print("\n=== Exit Rate (higher = better) ===")
print(describe(exit_rate, "exit_rate", "ExitRate"))

# Exit rate test — proportion comparison: does S2/S3 have a different success rate than S1?
# Mann-Whitney works on the per-run mean (0–1); higher = better so alternative="greater".
er_s1 = exit_rate.loc[exit_rate["group"] == "Baseline",                "exit_rate"].values
er_s2 = exit_rate.loc[exit_rate["group"] == "Shared Memory",           "exit_rate"].values
er_s3 = exit_rate.loc[exit_rate["group"] == "Shared Memory + Observer","exit_rate"].values

_, p_er_s2 = mannwhitneyu(er_s2, er_s1, alternative="greater")
_, p_er_s3 = mannwhitneyu(er_s3, er_s1, alternative="greater")
print(f"\nExit rate  S2 > S1:  p = {p_er_s2:.4f}  {'✓' if p_er_s2 < 0.05 else '✗'}")
print(f"Exit rate  S3 > S1:  p = {p_er_s3:.4f}  {'✓' if p_er_s3 < 0.05 else '✗'}")
# Note: if exit rates differ significantly, flag it before interpreting POR —
# POR only includes agents that reached the exit, so a lower rate biases the ratio upward.
```

---

## Cell 6 — Visualisations

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (df, col, title, ylabel) in zip(axes, [
    (por_run, "fleet_por",  "Path Optimality Ratio (fleet)",    "Σactual / Σoptimal"),
    (tok_run, "nav_tokens", "Navigator Tokens / run",           "tokens (navigators only)"),
    (rcr_run, "fleet_rcr",  "Redundant Computation (fleet)",    "Σredundant / Σvisited"),
]):
    sns.boxplot(data=df, x="group", y=col, order=GROUP_ORDER,
                palette=PALETTE, ax=ax)
    sns.stripplot(data=df, x="group", y=col, order=GROUP_ORDER,
                  color="black", size=3, alpha=0.4, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(GROUP_ORDER, rotation=15, ha="right")

plt.tight_layout()
plt.savefig("metric_comparison.png", dpi=150)
plt.show()
```

---

## Cell 6b — Token breakdown: navigator vs observer (S3)

This cell answers: did the observer help navigators be cheaper, and was that saving worth the
observer's own cost?

```python
# Mean nav_tokens and obs_tokens per group
tok_breakdown = (
    tok_run.groupby("group")[["nav_tokens", "obs_tokens", "total_tokens"]]
    .mean()
    .reindex(GROUP_ORDER)
    .round(0)
)
print(tok_breakdown)

# Stacked bar: navigator tokens (bottom) + observer tokens (top)
fig, ax = plt.subplots(figsize=(7, 5))
nav = tok_breakdown["nav_tokens"]
obs = tok_breakdown["obs_tokens"]
x   = range(len(GROUP_ORDER))
ax.bar(x, nav, color=[PALETTE[g] for g in GROUP_ORDER], label="Navigator tokens")
ax.bar(x, obs, bottom=nav, color="lightcoral", alpha=0.8, label="Observer tokens")
ax.set_xticks(x)
ax.set_xticklabels(GROUP_ORDER, rotation=15, ha="right")
ax.set_ylabel("Mean tokens / run")
ax.set_title("Token breakdown: navigators vs observer")
ax.legend()
plt.tight_layout()
plt.savefig("token_breakdown.png", dpi=150)
plt.show()

# Net efficiency question: do S3 navigators use fewer tokens than S1 navigators?
s1_nav = tok_run.loc[tok_run["group"] == "Baseline",               "nav_tokens"].values
s2_nav = tok_run.loc[tok_run["group"] == "Shared Memory",          "nav_tokens"].values
s3_nav = tok_run.loc[tok_run["group"] == "Shared Memory + Observer","nav_tokens"].values
s3_obs = tok_run.loc[tok_run["group"] == "Shared Memory + Observer","obs_tokens"].values

print(f"\nS2 navigator saving vs S1: {(s1_nav.mean() - s2_nav.mean()):.0f} tokens/run "
      f"({(1 - s2_nav.mean()/s1_nav.mean())*100:.1f}%)")
print(f"S3 navigator saving vs S1: {(s1_nav.mean() - s3_nav.mean()):.0f} tokens/run "
      f"({(1 - s3_nav.mean()/s1_nav.mean())*100:.1f}%)")
print(f"S3 observer overhead:      {s3_obs.mean():.0f} tokens/run")
print(f"S3 net vs S1:              {(tok_run[tok_run['group']=='Shared Memory + Observer']['total_tokens'].mean() - tok_run[tok_run['group']=='Baseline']['total_tokens'].mean()):+.0f} tokens/run")
```

**Hypothesis test for navigator tokens only** (S1 vs S2, S1 vs S3 — fair comparison):

```python
_, p_s2_nav = mannwhitneyu(s2_nav, s1_nav, alternative="less")
_, p_s3_nav = mannwhitneyu(s3_nav, s1_nav, alternative="less")
print(f"S2 nav < S1 nav:  p = {p_s2_nav:.4f}  {'✓' if p_s2_nav < 0.05 else '✗'}")
print(f"S3 nav < S1 nav:  p = {p_s3_nav:.4f}  {'✓' if p_s3_nav < 0.05 else '✗'}")
```

---

## Cell 7 — 95% Confidence Intervals (bootstrap)

```python
def bootstrap_ci(x, n_boot=10_000, ci=0.95):
    x = np.asarray(x, dtype=float)
    boot = np.random.choice(x, (n_boot, len(x)), replace=True).mean(axis=1)
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return lo, hi

ci_rows = []
for metric_label, df_m, col in [
    ("POR (fleet)",        por_run,  "fleet_por"),
    ("Nav Tokens",         tok_run,  "nav_tokens"),   # navigator-only, fair across scenarios
    ("RCR (fleet)",        rcr_run,  "fleet_rcr"),
]:
    for grp in GROUP_ORDER:
        vals = df_m.loc[df_m["group"] == grp, col].dropna().values
        lo, hi = bootstrap_ci(vals)
        ci_rows.append({
            "Metric":  metric_label,
            "Group":   grp,
            "Mean":    round(vals.mean(), 4),
            "CI_low":  round(lo, 4),
            "CI_high": round(hi, 4),
            "n":       len(vals),
        })

df_ci = pd.DataFrame(ci_rows)
print(df_ci.to_markdown(index=False))
```

---

## Cell 8 — Normality check (Shapiro-Wilk)

```python
print("Shapiro-Wilk  (p > 0.05 → cannot reject normality)\n")
for metric_label, df_m, col in [
    ("POR (fleet)",  por_run,  "fleet_por"),
    ("Nav Tokens",   tok_run,  "nav_tokens"),
    ("RCR (fleet)",  rcr_run,  "fleet_rcr"),
]:
    print(f"  {metric_label}")
    for grp in GROUP_ORDER:
        vals = df_m.loc[df_m["group"] == grp, col].dropna().values
        stat, p = shapiro(vals)
        flag = "✓ normal" if p > 0.05 else "✗ non-normal → use Mann-Whitney"
        print(f"    {grp:35s}  W={stat:.4f}  p={p:.4f}  {flag}")
```

**Rule**: if any group fails normality for a given metric, use **Mann-Whitney U** for that metric;
otherwise Welch t-test. Both are shown below; pick based on this output.

---

## Cell 9 — Hypothesis tests (one comparison at a time)

H₀ = no difference between group means.  
H₁ = treatment group mean **< baseline mean** (one-tailed, lower is better for all three metrics).

```python
def get_vals(df_m, col, grp):
    return df_m.loc[df_m["group"] == grp, col].dropna().values

raw_p_welch = []
raw_p_mw    = []
test_labels = []

for metric_label, df_m, col in [
    ("POR (fleet)",  por_run,  "fleet_por"),
    ("Nav Tokens",   tok_run,  "nav_tokens"),   # navigator-only; fair S1/S2/S3 comparison
    ("RCR (fleet)",  rcr_run,  "fleet_rcr"),
]:
    baseline_vals = get_vals(df_m, col, "Baseline")

    for treatment in ["Shared Memory", "Shared Memory + Observer"]:
        treat_vals = get_vals(df_m, col, treatment)

        # Welch t-test (parametric, one-tailed: treatment < baseline)
        t, p_two = ttest_ind(treat_vals, baseline_vals, equal_var=False)
        p_welch  = p_two / 2 if t < 0 else 1 - p_two / 2

        # Mann-Whitney U (non-parametric, one-tailed: treatment ranks lower)
        u, p_mw = mannwhitneyu(treat_vals, baseline_vals, alternative="less")

        label = f"{metric_label} | {treatment} vs Baseline"
        raw_p_welch.append(p_welch)
        raw_p_mw.append(p_mw)
        test_labels.append(label)

        print(f"{label}")
        print(f"  Welch t   t={t:.3f}  p(one-tail)={p_welch:.4f}")
        print(f"  Mann-Wh   U={u:.1f}  p(one-tail)={p_mw:.4f}\n")
```

---

## Cell 10 — Multiple-comparison correction (Benjamini-Hochberg)

6 tests total (2 comparisons × 3 metrics). BH controls the false discovery rate at 5%.

```python
_, p_bh_welch,  _, _ = multipletests(raw_p_welch, method="fdr_bh")
_, p_bh_mw,     _, _ = multipletests(raw_p_mw,    method="fdr_bh")

print(f"{'Test':<45}  {'raw Welch':>10}  {'BH Welch':>9}  {'raw MW':>8}  {'BH MW':>7}  sig?")
print("-" * 95)
for lbl, rw, bw, rm, bm in zip(test_labels, raw_p_welch, p_bh_welch, raw_p_mw, p_bh_mw):
    sig = "✓" if bm < 0.05 else "✗"
    print(f"{lbl:<45}  {rw:>10.4f}  {bw:>9.4f}  {rm:>8.4f}  {bm:>7.4f}  {sig}")
```

---

## Cell 11 — Effect size (Cohen's d)

Magnitude guide: |d| < 0.2 negligible, 0.5 medium, 0.8 large (Cohen 1988).

```python
def cohens_d(a, b):
    n_a, n_b = len(a), len(b)
    sp = np.sqrt(((n_a-1)*np.var(a, ddof=1) + (n_b-1)*np.var(b, ddof=1)) / (n_a + n_b - 2))
    return (np.mean(a) - np.mean(b)) / sp   # negative = treatment < baseline (improvement)

print(f"{'Metric':<8}  {'Comparison':<35}  {'d':>7}  {'magnitude'}")
print("-" * 65)
for metric_label, df_m, col in [
    ("POR (fleet)",  por_run,  "fleet_por"),
    ("Nav Tokens",   tok_run,  "nav_tokens"),
    ("RCR (fleet)",  rcr_run,  "fleet_rcr"),
]:
    bl = get_vals(df_m, col, "Baseline")
    for treatment in ["Shared Memory", "Shared Memory + Observer"]:
        tr = get_vals(df_m, col, treatment)
        d  = cohens_d(tr, bl)
        mag = ("negligible" if abs(d) < 0.2 else
               "small"      if abs(d) < 0.5 else
               "medium"     if abs(d) < 0.8 else "large")
        print(f"{metric_label:<8}  {treatment:<35}  {d:>7.3f}  {mag}")
```

---

## Cell 12 — Summary table

```python
rows = []
for metric_label, df_m, col, idx in [
    ("POR (fleet)",  por_run,  "fleet_por",  0),
    ("Nav Tokens",   tok_run,  "nav_tokens", 2),
    ("RCR (fleet)",  rcr_run,  "fleet_rcr",  4),
]:
    bl = get_vals(df_m, col, "Baseline")
    for i, treatment in enumerate(["Shared Memory", "Shared Memory + Observer"]):
        tr  = get_vals(df_m, col, treatment)
        lo, hi = bootstrap_ci(tr)
        d   = cohens_d(tr, bl)
        bm  = p_bh_mw[idx + i]
        rows.append({
            "Metric":    metric_label,
            "Group":     treatment,
            "Mean":      round(tr.mean(), 4),
            "95% CI":    f"[{lo:.4f}, {hi:.4f}]",
            "d vs Base": round(d, 3),
            "p (BH-MW)": round(bm, 4),
            "Sig?":      "✓" if bm < 0.05 else "✗",
        })

pd.DataFrame(rows).to_markdown(index=False)
```

---

## Cell 13 — Interpretation guide

| Condition | Conclusion for thesis |
|---|---|
| p(BH) < 0.05 **and** \|d\| ≥ 0.5 | Scenario is **statistically and practically** better than Baseline |
| p(BH) < 0.05 **and** \|d\| < 0.5 | Statistically significant, **small practical effect** — discuss limitations |
| p(BH) ≥ 0.05, CI overlaps baseline | **No evidence of improvement** for this metric |
| Mean higher than baseline (wrong direction) | Shared memory **may have hurt** — report as exploratory finding |

---

## Dependencies

```
pip install supabase pandas scipy matplotlib seaborn statsmodels python-dotenv
```

Place notebook at `notebooks/statistical_analysis.ipynb` and run from repo root so that
`../.env` and `../src` resolve correctly.
