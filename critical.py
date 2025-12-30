import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as patches

# ==========================================
# 1. 環境與路徑設定
# ==========================================
DATA_DIR = r"D:\projects\cogni\human-alignment-study-1.1\study_data"

def load_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"[Loading] {filename}...")
        return pd.read_csv(path)
    else:
        print(f"[Error] {filename} not found at {path}")
        return None

# ==========================================
# 2. 資料讀取與前處理
# ==========================================
demo = load_file("demo_survey.csv")
group_A = load_file("group_A.csv")
group_B = load_file("group_B.csv")
group_BP = load_file("group_BP.csv")
group_C = load_file("group_C.csv")

if any(df is None for df in [demo, group_A, group_B, group_BP, group_C]):
    raise FileNotFoundError("Critical files missing.")

# Label Groups
group_A['group'] = 'A (Control)'
group_B['group'] = 'B (Misleading)'
group_BP['group'] = 'BP (Realigned)'
group_C['group'] = 'C (Aligned)'

# Combine Data
all_games = pd.concat([group_A, group_B, group_BP, group_C], ignore_index=True)
full_df = all_games.merge(demo, on='participant_id', how='left')

# Feature Engineering
full_df['switched'] = full_df['initial_decision'] != full_df['final_decision']
full_df['initial_correct'] = (full_df['initial_decision'] == full_df['outcome']).astype(int)
full_df['final_correct'] = (full_df['final_decision'] == full_df['outcome']).astype(int)
full_df['utility_gain'] = full_df['final_correct'] - full_df['initial_correct']

# AI Confidence Logic
full_df['AI_conf'] = full_df['AI_conf'].fillna(50)
full_df['ai_pred_color'] = np.where(
    full_df['AI_conf'] > 50, 'Red', 
    np.where(full_df['AI_conf'] < 50, 'Black', 'Uncertain')
)

# === 關鍵篩選：找出「AI 比人類強」但「意見不合」的時刻 ===
# 條件：AI 信心 > 人類信心 AND 初始意見不合 AND AI 不是不確定
rational_switch_zone = full_df[
    (full_df['AI_conf'] > full_df['human_conf']) & 
    (full_df['initial_decision'] != full_df['ai_pred_color']) &
    (full_df['ai_pred_color'] != 'Uncertain')
].copy()

# 計算每個組別在這種情況下的切換率 (Compliance Rate)
compliance_stats = rational_switch_zone.groupby('group')['switched'].value_counts(normalize=True).unstack().fillna(0)
# compliance_stats 會有兩欄: False (沒換), True (換了)
# 我們將其轉換為百分比
compliance_stats = compliance_stats * 100

sns.set_theme(style="whitegrid", font_scale=1.1)
palette = sns.color_palette("tab10")

# ==========================================
# 3. 繪圖
# ==========================================
print("\nGenerating plots...")

# --- Fig 1: Temporal Trust ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=full_df, x='trial_id', y='switched', hue='group', marker='o', errorbar=('ci', 95), palette=palette)
plt.title('Figure 1: Trust Dynamics - Switching Probability over Time', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Fig1_Temporal_Trust.png', dpi=300)
plt.close()

# --- Fig 2: Alignment Map ---
plt.figure(figsize=(9, 9))
sns.scatterplot(data=full_df, x='AI_conf', y='true_prob', hue='group', style='group', s=80, alpha=0.5, palette=palette)
plt.plot([0, 100], [0, 100], color='gray', linestyle='--')
rect = patches.Rectangle((75, 0), 25, 40, edgecolor='red', facecolor='red', alpha=0.1)
plt.gca().add_patch(rect)
plt.text(87.5, 20, 'DANGER ZONE', color='darkred', ha='center', fontweight='bold')
plt.title('Figure 2: Alignment Map', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Fig2_Alignment_Map.png', dpi=300)
plt.close()

# --- Fig 3: Silence Cost ---
plt.figure(figsize=(10, 6))
sns.histplot(data=full_df, x='AI_conf', hue='group', element='step', bins=20, kde=True, common_norm=False, palette=palette)
plt.title('Figure 3: Cost of Silence (AI Confidence Distribution)', fontweight='bold')
plt.tight_layout()
plt.savefig('Fig3_Silence_Cost.png', dpi=300)
plt.close()

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

# --- Figure 5 (New): Trust Gap Bar Chart ---
# 這是為了替代原本雜亂的散佈圖
plt.figure(figsize=(10, 6))

# 繪製堆疊長條圖
# True (Green): Rational Compliance (聽了 AI 的話)
# False (Red): Irrational Resistance (沒聽 AI 的話)
if not compliance_stats.empty:
    # 確保 True 和 False 欄位都存在
    if True not in compliance_stats.columns: compliance_stats[True] = 0
    if False not in compliance_stats.columns: compliance_stats[False] = 0
    
    # 繪製 "Irrational Resistance" (Bottom bar, Red)
    p1 = plt.bar(compliance_stats.index, compliance_stats[False], color='#e74c3c', label='Irrational Resistance (Ignored AI)', alpha=0.8)
    # 繪製 "Rational Compliance" (Top bar, Green)
    p2 = plt.bar(compliance_stats.index, compliance_stats[True], bottom=compliance_stats[False], color='#2ecc71', label='Rational Compliance (Listened to AI)', alpha=0.8)

    # 加入數值標籤
    for r1, r2 in zip(p1, p2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        if h1 > 0: plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., f"{h1:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        if h2 > 0: plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., f"{h2:.1f}%", ha="center", va="center", color="white", fontweight="bold")

    plt.title('Figure 5: The Trust Deficit\n(Human Behavior when AI is More Confident)', fontweight='bold')
    plt.ylabel('Percentage of Decisions (%)')
    plt.xlabel('Experimental Group')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 加入一條參考線
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Fig5_Trust_Gap_Analysis.png', dpi=300)
    print(" -> Saved Fig5_Trust_Gap_Analysis.png (Clean Bar Chart)")
else:
    print("[Warning] No data found for Rational Switch Zone (sample too small?)")

plt.close()
print("\nAll plots generated successfully.")