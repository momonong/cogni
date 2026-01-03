import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 環境與資料讀取
# ==========================================
DATA_DIR = r"human-alignment-study-1.1\study_data"

def load_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"[Loading] {filename}...")
        return pd.read_csv(path)
    else:
        # 如果找不到檔案，嘗試只用檔名讀取 (假設在當前目錄)
        if os.path.exists(filename):
             return pd.read_csv(filename)
        print(f"[Error] {filename} not found.")
        return None

# 讀取資料
demo = load_file("demo_survey.csv")
group_A = load_file("group_A.csv")
group_B = load_file("group_B.csv")
group_BP = load_file("group_BP.csv")
group_C = load_file("group_C.csv")

if any(df is None for df in [demo, group_A, group_B, group_BP, group_C]):
    raise FileNotFoundError("Critical files missing.")

# 標記組別
group_A['group'] = 'A (Biased Away)'
group_B['group'] = 'B (Biased Towards)'
group_BP['group'] = 'BP (Realigned)'
group_C['group'] = 'C (No Bias)'

# 合併資料
all_games = pd.concat([group_A, group_B, group_BP, group_C], ignore_index=True)
full_df = all_games.merge(demo, on='participant_id', how='left')

# ==========================================
# 2. 特徵工程與優化定義
# ==========================================

# 確保 AI_decision 存在 (如果資料集沒有，則根據 AI_conf 推斷)
if 'AI_decision' not in full_df.columns:
    full_df['AI_decision'] = (full_df['AI_conf'] > 50).astype(int)

# 1. 定義依從 (Compliance/Switching)
full_df['switched'] = full_df['initial_decision'] != full_df['final_decision']

# 2. 定義意見不合 (Disagreement) - **修正重點：使用決策比對**
# 只分析人類與 AI 初始意見不同的情況
disagreement_df = full_df[full_df['initial_decision'] != full_df['AI_decision']].copy()

# 3. 定義決策情境 (Decision Scenarios) - **修正重點：使用決策正確性分類**
# Scenario 1: AI is Correct (Opportunity) -> Human was initially wrong
# Scenario 2: AI is Incorrect (Trap) -> Human was initially right
disagreement_df['scenario'] = np.where(
    disagreement_df['AI_decision'] == disagreement_df['outcome'], 
    'AI Correct (Helpful)', 
    'AI Incorrect (Trap)'
)

# 設定繪圖風格
sns.set_theme(style="whitegrid", font_scale=1.1)
group_palette = {
    'A (Biased Away)': '#d62728',   # Red
    'B (Biased Towards)': '#ff7f0e', # Orange
    'BP (Realigned)': '#2ca02c',     # Green
    'C (No Bias)':  '#1f77b4'      # Blue
}

print(f"\n[Analysis Scope] Total Disagreement Events: {len(disagreement_df)}")
print(disagreement_df['scenario'].value_counts())

# ==========================================
# 3. 繪圖與分析
# ==========================================

# --- Figure 1: Compliance Rate by Scenario (Optimized Trust Gap) ---
# 分析：當 AI 是對的時候，大家聽嗎？當 AI 是錯的時候，大家能發現嗎？
plt.figure(figsize=(12, 6))

# 計算依從率
compliance_rates = disagreement_df.groupby(['group', 'scenario'])['switched'].mean().reset_index()
compliance_rates['switched'] = compliance_rates['switched'] * 100 # 轉為百分比

sns.barplot(
    data=compliance_rates,
    x='group', 
    y='switched', 
    hue='scenario',
    palette={'AI Correct (Helpful)': '#2ecc71', 'AI Incorrect (Trap)': '#e74c3c'},
    edgecolor='black',
    alpha=0.8
)

plt.title('Figure 1: Rational vs. Irrational Compliance\n(Do users distinguish between Helpful and Misleading AI?)', fontweight='bold')
plt.ylabel('Compliance Rate (%)')
plt.xlabel('Group')
plt.axhline(50, color='gray', linestyle='--', alpha=0.3)
plt.legend(title='Disagreement Scenario')
plt.tight_layout()
plt.savefig('Fig1_Optimized_Trust_Gap.png', dpi=300)
print(" -> Saved Fig1_Optimized_Trust_Gap.png")


# --- Figure 2: Temporal Trust Dynamics (Split by Scenario) ---
# 分析：隨時間推移，人們是否學會了信任「對的 AI」並拒絕「錯的 AI」？
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

scenarios = ['AI Correct (Helpful)', 'AI Incorrect (Trap)']

for i, scenario in enumerate(scenarios):
    ax = axes[i]
    subset = disagreement_df[disagreement_df['scenario'] == scenario]
    
    sns.lineplot(
        data=subset, 
        x='trial_id', 
        y='switched', 
        hue='group', 
        marker='o',
        palette=group_palette,
        ax=ax,
        errorbar=None # 為了清晰度，暫時隱藏誤差線，可視需求改為 ('ci', 95)
    )
    
    ax.set_title(f'Scenario: {scenario}', fontweight='bold')
    ax.set_xlabel('Trial ID (Time)')
    if i == 0:
        ax.set_ylabel('Compliance Rate (Switching Probability)')
    else:
        ax.set_ylabel('')
    
    ax.set_ylim(0, 1) # Switching rate 0-1
    ax.grid(True, alpha=0.3)

plt.suptitle('Figure 2: Temporal Trust Dynamics by Decision Scenario', fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('Fig2_Optimized_Temporal_Trust.png', dpi=300)
print(" -> Saved Fig2_Optimized_Temporal_Trust.png")


# --- Figure 3: Cost of Silence (AI Confidence Distribution) ---
# 這部分保持原樣，因為它分析的是 AI 的行為模式（是否傾向於給出模稜兩可的建議）
plt.figure(figsize=(10, 6))

# 只看 BP 和 B 組的對比，這最能體現 Silence Cost
focus_groups = ['B (Biased Towards)', 'BP (Realigned)']
subset_conf = full_df[full_df['group'].isin(focus_groups)]

sns.kdeplot(
    data=subset_conf, 
    x='AI_conf', 
    hue='group', 
    fill=True, 
    common_norm=False, 
    palette=group_palette,
    alpha=0.3,
    linewidth=2
)

plt.title('Figure 3: Cost of Silence - AI Confidence Distribution', fontweight='bold')
plt.xlabel('AI Confidence Score (0-100)')
plt.ylabel('Density')
plt.axvline(50, color='black', linestyle='--', alpha=0.5)

# 標註 "Silence Zone"
plt.axvspan(40, 60, color='gray', alpha=0.1, label='Silence Zone (40-60%)')

plt.legend()
plt.tight_layout()
plt.savefig('Fig3_Optimized_Silence_Cost.png', dpi=300)
print(" -> Saved Fig3_Optimized_Silence_Cost.png")

print("\nOptimization Complete.")