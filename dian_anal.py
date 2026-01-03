import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

paths = {
    "A":  r"study_data\group_A.csv",
    "B":  r"study_data\group_B.csv",
    "BP": r"study_data\group_BP.csv",
    "C":  r"study_data\group_C.csv",
}

COLORS = {
    "A":  "tab:blue",
    "B":  "tab:orange",
    "BP": "tab:green",
    "C":  "tab:red",
}

def load_and_prepare(path):
    df = pd.read_csv(path)

    # 排除無效回合
    df = df[~df["game_id"].isin([-1, -2, -3])].copy()

    # AI 信心（優先 calibrated）
    ai_conf = df["Ai_calibrated_conf"].where(
        ~df["Ai_calibrated_conf"].isna(),
        df["AI_conf"]
    )

    # AI 建議
    df["ai_suggestion"] = np.where(ai_conf >= 50, "Red", "Black")

    # 衝突 & 是否改答案
    df["conflict"] = df["ai_suggestion"] != df["initial_decision"]
    df["switched"] = df["final_decision"] != df["initial_decision"]

    df["trial_id"] = df["trial_id"].astype(int)
    return df

dfs = {g: load_and_prepare(p) for g, p in paths.items()}

original_trials = sorted(set().union(*[set(df["trial_id"].unique()) for df in dfs.values()]))
trial_map = {old: new for new, old in enumerate(original_trials, start=1)}
n_trials = len(trial_map)

rows = []
for group, df in dfs.items():
    sub = df[df["conflict"]].copy()
    sub["trial"] = sub["trial_id"].map(trial_map)

    per_trial = (
        sub.groupby("trial")["switched"]
        .mean()
        .reindex(range(1, n_trials + 1))
    )

    for t, p in per_trial.items():
        rows.append({"group": group, "trial": t, "p_switch": p})

cond_df = pd.DataFrame(rows)

#計算斜率
slopes = {}
intercepts = {}

for group in ["A", "B", "BP", "C"]:
    sub = cond_df[(cond_df["group"] == group) & (~cond_df["p_switch"].isna())]
    x = sub["trial"].values
    y = sub["p_switch"].values
    slope, intercept = np.polyfit(x, y, 1)
    slopes[group] = slope
    intercepts[group] = intercept

print("=== Slope of P(Switch | Conflict) per round ===")
for g in ["A", "B", "BP", "C"]:
    print(f"{g}: slope = {slopes[g]:.6f}  (~{slopes[g]*100:.2f}% per round)")

#圖
plt.figure(figsize=(12, 6))

for group in ["A", "B", "BP", "C"]:
    sub = cond_df[(cond_df["group"] == group) & (~cond_df["p_switch"].isna())]
    x = sub["trial"].values
    y = sub["p_switch"].values
    plt.plot(
        x, y,
        marker="o",
        color=COLORS[group],
        label=f"{group} (slope={slopes[group]:.3f})"
    )

    y_hat = slopes[group] * x + intercepts[group]
    plt.plot(
        x, y_hat,
        linestyle="--",
        color=COLORS[group],
        alpha=0.6
    )

plt.xlabel(f"Round (reindexed 1–{n_trials})")
plt.ylabel("P(Switch | Conflict)")
plt.title("Switching Probability Given Conflict\nwith Linear Trend per Group")
plt.xticks(range(1, n_trials + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("Dian.png")