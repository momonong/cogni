import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as patches

# ==========================================
# 1. Environment and Path Settings
# ==========================================
DATA_DIR = r"human-alignment-study-1.1\study_data"

def load_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"[Loading] {filename}...")
        return pd.read_csv(path)
    else:
        print(f"[Error] {filename} not found at {path}")
        return None

# ==========================================
# 2. Data Reading and Preprocessing
# ==========================================
demo = load_file("demo_survey.csv")
group_A = load_file("group_A.csv")
group_B = load_file("group_B.csv")
group_BP = load_file("group_BP.csv")
group_C = load_file("group_C.csv")


if any(df is None for df in [demo, group_A, group_B, group_BP, group_C]):
    raise FileNotFoundError("Critical files missing.")

# Label Groups
group_A['group'] = 'A (Biased Away)'
group_B['group'] = 'B (Biased Towards)'
group_BP['group'] = 'BP (Realigned)'
group_C['group'] = 'C (No Bias)'

# Combine Data
all_games = pd.concat([group_A, group_B, group_BP, group_C], ignore_index=True)
full_df = all_games.merge(demo, on='participant_id', how='left')

# Feature Engineering
full_df['changed'] = full_df['initial_decision'] != full_df['final_decision']
full_df['initial_correct'] = (full_df['initial_decision'] == full_df['outcome']).astype(int)
full_df['final_correct'] = (full_df['final_decision'] == full_df['outcome']).astype(int)
full_df['utility_gain'] = full_df['final_correct'] - full_df['initial_correct']


# === Key Screening: Only identify moments where AI disagrees from humans. Not "is better" 

disagreement_data = full_df[
    ((full_df['human_conf'] > 50) & (full_df['AI_conf'] < 50)) |
    ((full_df['human_conf'] < 50) & (full_df['AI_conf'] > 50))
].copy()

# Calculate the switching rate for each group under these circumstances. (Compliance Rate)
compliance_stats = disagreement_data.groupby('group')['changed'].value_counts(normalize=True).unstack().fillna(0)
# compliance_stats There will be two columns: False (Not replaced), True (Changed)
# We convert it to a percentage
compliance_stats = compliance_stats * 100

sns.set_theme(style="whitegrid", font_scale=1.1)
palette = sns.color_palette("tab10")

# Define a fixed palette mapping for all subplots
group_palette = {
    'A (Biased Away)': '#d62728',   # blue
    'B (Biased Towards)': '#ff7f0e', # orange
    'BP (Realigned)': '#2ca02c',     # green
    'C (No Bias)':  '#1f77b4'      # red
}

# ==========================================
# 3. Plotting
# ==========================================
print("\nGenerating plots...")

# --- Figure 1: Trust Gap Bar Chart ---
plt.figure(figsize=(10, 6))

# Draw a stacked bar chart
# True (Green): Compliance (聽了 AI 的話)
# False (Red): Resistance (沒聽 AI 的話)
if not compliance_stats.empty:
    # 確保 True 和 False 欄位都存在
    if True not in compliance_stats.columns: compliance_stats[True] = 0
    if False not in compliance_stats.columns: compliance_stats[False] = 0
    
    #  "Resistance" (Bottom bar, Red)
    p1 = plt.bar(compliance_stats.index, compliance_stats[False], color='#e74c3c', label='Resistance (Ignored AI)', alpha=0.8)
    # "Compliance" (Top bar, Green)
    p2 = plt.bar(compliance_stats.index, compliance_stats[True], bottom=compliance_stats[False], color='#2ecc71', label='Compliance (Adapted to AI)', alpha=0.8)

    # Add value labels
    for r1, r2 in zip(p1, p2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        if h1 > 0: plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., f"{h1:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        if h2 > 0: plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., f"{h2:.1f}%", ha="center", va="center", color="white", fontweight="bold")

    plt.title('Figure 1: The Trust Deficit\n Human Behavior when AI disagrees', fontweight='bold')
    plt.ylabel('Response in %')
    plt.xlabel('Group')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add a reference line
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Fig1_Trust_Gap_Analysis.png', dpi=300)
    print(" -> Saved Fig1_Trust_Gap_Analysis.png (Clean Bar Chart)")
else:
    print("[Warning] No data found for Rational Switch Zone (sample too small?)")

plt.close()

# --- Fig 2: Temporal Trust ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=disagreement_data, x='trial_id', y='changed', hue='group', marker='o', errorbar=('ci', 95), palette=group_palette)
plt.title('Figure 2: Trust Dynamics - Probability of Compliance across Game Iterations', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Fig2_Temporal_Trust.png', dpi=300)
plt.close()


# --- Fig 3a: Silence Cost --- Whole Dataframe
plt.figure(figsize=(10, 6))
sns.histplot(data=full_df, x='AI_conf', hue='group', element='step', bins=20, kde=True, common_norm=False, palette=palette)
plt.title('Figure 3: Cost of Silence (AI Confidence Distribution)', fontweight='bold')
plt.tight_layout()
plt.savefig('Fig3_Silence_Cost_all_data.png', dpi=300)
plt.close()

# --- Fig 3b: Silence Cost --- only when disagrees
plt.figure(figsize=(10, 6))
sns.histplot(data=disagreement_data, x='AI_conf', hue='group', element='step', bins=20, kde=True, common_norm=False, palette=palette)
plt.title('Figure 3: Cost of Silence (AI Confidence Distribution)', fontweight='bold')
plt.tight_layout()
plt.savefig('Fig3_Silence_Cost_disagreement_data.png', dpi=300)
plt.close()

# --- Fig 3: Silence Cost --- Find Minima
plt.figure(figsize=(10, 6))
ax = sns.histplot(data=full_df, x='AI_conf', hue='group', element='step', bins=20, kde=True, common_norm=False, palette=palette)

# Important: First get the lines
lines = ax.get_lines()
label = lines[1].get_label() # [1] for realigned group
x_coords = lines[1].get_xdata()
y_coords = lines[1].get_ydata()

# Minima of Realigned Group: [46.73366834 64.32160804] Meaning inbetween is maximum
# Old Minima of Realigned Group: [40.20100503 64.8241206]

logical_minima = (y_coords[1:-1] < y_coords[:-2]) & (y_coords[1:-1] < y_coords[2:])
minima_indices = np.where(logical_minima)[0] + 1

found_minima_x = x_coords[minima_indices]

print(f"Minima at AI_conf: {found_minima_x}")

# plot Minima
if len(found_minima_x) > 0:
    plt.scatter(found_minima_x, y_coords[minima_indices], color='black', s=50, zorder=5)

plt.title('Figure 3: Cost of Silence (AI Confidence Distribution)', fontweight='bold')
plt.tight_layout()
plt.savefig('Fig3a_Silence_Cost_with_Minima.png', dpi=300)


# AI Confidence peak between 40.2 and 64.8 Percent

# Introduce:

# highly Certain [0,35] and [65,100]
# moderately Certain [35,45] and [55,65] 
# low Certainty [45-55]


certainty_bins = {
    'Low': lambda x: (45 <= x) & (x <= 55),
    'Moderate': lambda x: ((35 < x) & (x < 45)) | ((55 < x) & (x < 65)),
    'High': lambda x: (x <= 35) | (x >= 65)
}


# I want Figure 1 and Figure 2 for intervals: 
# human highly Certain, AI highly Certain, moderately Certain, low Certainty
# human moderately Certain, AI highly Certain, moderately Certain, low Certainty
# human low Certainty, AI highly Certain, moderately Certain, low Certainty

# This will be 3 x 3 plot. 

# ... (前面的程式碼保持不變) ...

# --- Figure 1a:
def compute_compliance_stats(df):
    """
    計算 Compliance 的統計數據，包含百分比和樣本數
    """
    total_count = len(df)
    
    # 計算 Changed (True/False) 的數量
    counts = df.groupby('group')['changed'].value_counts().unstack().fillna(0)
    
    # 確保 True 和 False 都有
    if True not in counts.columns: counts[True] = 0
    if False not in counts.columns: counts[False] = 0
    
    # 計算總數 (N)
    group_totals = counts.sum(axis=1)
    
    # 計算百分比
    percentages = counts.div(group_totals, axis=0) * 100
    
    return percentages, group_totals

human_levels = ['Low', 'Moderate', 'High']
AI_levels = ['Low', 'Moderate', 'High']

fig, axes = plt.subplots(3, 3, figsize=(20, 16), sharey=True) # 稍微加大圖片尺寸

for i, h in enumerate(human_levels):
    for j, a in enumerate(AI_levels):
        ax = axes[i, j]

        subset = disagreement_data[
            certainty_bins[h](disagreement_data['human_conf']) &
            certainty_bins[a](disagreement_data['AI_conf'])
        ]

        if subset.empty:
            ax.set_title(f"Human: {h} | AI: {a}\n(No data)", fontsize=10)
            ax.axis('off')
            continue

        # 計算統計數據
        percentages, group_totals = compute_compliance_stats(subset)

        # 繪製長條圖
        # Resistance (Bottom)
        bars_res = ax.bar(percentages.index, percentages[False],
               color='#e74c3c', alpha=0.8, label='Resistance')
        
        # Compliance (Top)
        bars_com = ax.bar(percentages.index, percentages[True],
               bottom=percentages[False],
               color='#2ecc71', alpha=0.8, label='Compliance')

        # 添加標註 (百分比和 N)
        for idx, (rect_res, rect_com) in enumerate(zip(bars_res, bars_com)):
            group_name = percentages.index[idx]
            n_val = int(group_totals[group_name])
            
            # 獲取百分比數值
            pct_res = percentages.loc[group_name, False]
            pct_com = percentages.loc[group_name, True]
            
            # 標註 Resistance 百分比 (如果在長條內部空間足夠)
            if pct_res > 15: # 避免文字擠在一起
                ax.text(rect_res.get_x() + rect_res.get_width() / 2, rect_res.get_y() + rect_res.get_height() / 2,
                        f"{pct_res:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            
            # 標註 Compliance 百分比
            if pct_com > 15:
                ax.text(rect_com.get_x() + rect_com.get_width() / 2, rect_com.get_y() + rect_com.get_height() / 2,
                        f"{pct_com:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            
            # 標註樣本數 (N) 在長條圖上方
            ax.text(rect_com.get_x() + rect_com.get_width() / 2, 102, # 稍微高於 100
                    f"N={n_val}", ha='center', va='bottom', color='black', fontsize=9, fontweight='bold')

        ax.axhline(50, linestyle='--', color='gray', alpha=0.4)
        ax.set_ylim(0, 115) # 增加 y 軸上限以容納 N 標籤
        ax.set_title(f"Human Certainty: {h} | AI Certainty: {a}", fontsize=11, fontweight='bold')

        # 設定 x 軸標籤旋轉，避免重疊
        ax.set_xticklabels(percentages.index, rotation=45, ha='right')

        if j == 0:
            ax.set_ylabel('Response in %')
        # if i == 2: # 不需要特定的 xlabel，group 名字已經在 x 軸上了

# 調整圖例
legend_patches = [
    patches.Patch(color='#2ecc71', label='Compliance (Adapted to AI)'),
    patches.Patch(color='#e74c3c', label='Resistance (Ignored AI)')
]

fig.legend(
    handles=legend_patches,
    loc='upper center',
    ncol=2,
    bbox_to_anchor=(0.5, 0.96),
    fontsize=12
)

fig.suptitle(
    'Figure 1a: The Trust Deficit (with Sample Sizes)\n Human Behavior when AI disagrees',
    fontsize=18,
    fontweight='bold',
    y=0.99
)

plt.tight_layout(rect=[0, 0, 1, 0.95]) # 留出空間給標題和圖例
plt.savefig("Fig1a_Trust_Gap_3x3_with_N.png", dpi=300, bbox_inches='tight')
plt.close()

print(" -> Saved Fig1a_Trust_Gap_3x3_with_N.png")

# Figure 2b

fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharey=True, sharex=True)


for i, h in enumerate(human_levels):
    for j, a in enumerate(AI_levels):
        ax = axes[i, j]

        subset = disagreement_data[
            certainty_bins[h](disagreement_data['human_conf']) &
            certainty_bins[a](disagreement_data['AI_conf'])
        ]

        if subset.empty:
            ax.set_title(f"Human: {h} | AI: {a}\n(No data)", fontsize=9)
            ax.axis('off')
            continue

        # Compute compliance per trial
        trial_stats = (
            subset.groupby(['trial_id', 'group'])['changed']
            .mean() * 100  # convert to percentage
        ).reset_index()
        
        sns.lineplot(
            data=trial_stats,
            x='trial_id',
            y='changed',
            hue='group',
            marker='o',
            ax=ax,
            palette=group_palette,
            errorbar=None,
            legend=False
        )

        ax.set_ylim(0, 100)
        ax.set_title(f"Human Certainty: {h} | AI Certainty: {a}", fontsize=10, fontweight='bold')

        if j == 0:
            ax.set_ylabel('Compliance in %')
        if i == 2:
            ax.set_xlabel('Trial ID')

import matplotlib.patches as mpatches
legend_patches = [mpatches.Patch(color=color, label=grp) for grp, color in group_palette.items()]

fig.legend(
    handles=legend_patches,
    loc='upper center',
    ncol=len(legend_patches),
    bbox_to_anchor=(0.5, 0.98)
)

fig.suptitle(
    "Figure 2b: Trust Dynamics by Human & AI Certainty Levels",
    fontsize=16,
    fontweight='bold',
    y=1.05
)

plt.tight_layout()
plt.savefig("Fig2b_Temporal_Trust_3x3.png", dpi=300, bbox_inches='tight')
plt.close()

print(" -> Saved Fig2b_Temporal_Trust_3x3.png")


# --- Fig 4: Education Fairness ---
plt.figure(figsize=(10, 6))
df_edu = full_df.dropna(subset=['degree'])
if not df_edu.empty:
    sns.barplot(data=df_edu, x='degree', y='utility_gain', hue='group', errorbar=('ci', 95), palette=palette, capsize=.1)
    plt.title('Figure 4: Fairness Analysis (Utility by Education)', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Fig4_Education_Fairness.png', dpi=300)
plt.close()


print("\nAll plots generated successfully.")