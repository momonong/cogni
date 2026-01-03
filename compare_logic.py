import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 設定與讀取資料
# ==========================================
# 請確認路徑是否正確
DATA_DIR = r"D:\projects\cogni\human-alignment-study-1.1\study_data"
# 如果路徑不同，請自行修改，例如:
# DATA_DIR = r"C:\Users\Luise\Documents\Courses NCKU\...\study_data"

def load_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"[Loading] {filename}...")
        return pd.read_csv(path)
    else:
        print(f"[Error] {filename} not found at {path}")
        return None

# 讀取資料
group_A = load_file("group_A.csv")
group_B = load_file("group_B.csv")
group_BP = load_file("group_BP.csv")
group_C = load_file("group_C.csv")
demo = load_file("demo_survey.csv")

if any(df is None for df in [group_A, group_B, group_BP, group_C]):
    raise FileNotFoundError("部分檔案讀取失敗，請檢查路徑。")

# 標記組別
group_A['group'] = 'A'
group_B['group'] = 'B'
group_BP['group'] = 'BP'
group_C['group'] = 'C'

# 合併
all_games = pd.concat([group_A, group_B, group_BP, group_C], ignore_index=True)
full_df = all_games.merge(demo, on='participant_id', how='left')

# 基礎前處理
full_df['switched'] = full_df['initial_decision'] != full_df['final_decision']
# 填補空值為 50 (代表沈默)
full_df['AI_conf'] = full_df['AI_conf'].fillna(50)

# ==========================================
# 2. 核心分析：50% 的影響有多大？
# ==========================================

# --- 分析 A: 各組 "AI 信心 = 50%" 的比例 (Volume of Impact) ---
full_df['is_silent'] = (full_df['AI_conf'] == 50)
silence_rate = full_df.groupby('group')['is_silent'].mean() * 100

print("\n=== 分析 1: AI 沈默 (50%) 的比例 ===")
print(silence_rate)
print("-" * 30)
# 如果某組的沈默率很高，那排除 50% 的影響就會非常巨大

# --- 分析 B: 兩種邏輯下的「切換率」對比 (Method Comparison) ---

# 邏輯 1: Dian 的邏輯 (包含 50%，強制視為 Red)
# 定義建議： >= 50 是 Red, < 50 是 Black
full_df['pred_dian'] = np.where(full_df['AI_conf'] >= 50, 'Red', 'Black')
# 定義衝突：人類初始 != AI 強制建議
mask_dian_conflict = full_df['initial_decision'] != full_df['pred_dian']
# 計算切換率
switch_rate_dian = full_df[mask_dian_conflict].groupby('group')['switched'].mean() * 100

# 邏輯 2: Luise 的邏輯 (排除 50%)
# 定義建議： > 50 是 Red, < 50 是 Black (排除 == 50)
mask_luise_valid = full_df['AI_conf'] != 50
full_df['pred_luise'] = np.where(full_df['AI_conf'] > 50, 'Red', 'Black') # 這裡其實只對非50有效
# 定義衝突：人類初始 != AI 建議 AND AI 不是 50
mask_luise_conflict = (full_df['initial_decision'] != full_df['pred_luise']) & mask_luise_valid
# 計算切換率
switch_rate_luise = full_df[mask_luise_conflict].groupby('group')['switched'].mean() * 100

# 合併數據以便繪圖
comparison_df = pd.DataFrame({
    'Dian_Method (Include 50%)': switch_rate_dian,
    'Luise_Method (Exclude 50%)': switch_rate_luise
}).reset_index()

# 計算差異
comparison_df['Diff'] = comparison_df['Luise_Method (Exclude 50%)'] - comparison_df['Dian_Method (Include 50%)']

print("\n=== 分析 2: 兩種邏輯的切換率差異 ===")
print(comparison_df)

# ==========================================
# 3. 視覺化：畫出來讓你直接看影響
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- 圖 1: 各組有多少數據受到 "50% 邏輯" 的影響？ ---
sns.barplot(x=silence_rate.index, y=silence_rate.values, ax=axes[0], palette="Blues_d")
axes[0].set_title('Impact Magnitude: % of Trials with AI Confidence = 50%', fontweight='bold')
axes[0].set_ylabel('Percentage of Silence (%)')
axes[0].set_xlabel('Group')
for i, v in enumerate(silence_rate.values):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

# --- 圖 2: 兩種邏輯算出來的結果差多少？ ---
# 轉換為長格式以便繪圖
melted_df = comparison_df.melt(id_vars='group', value_vars=['Dian_Method (Include 50%)', 'Luise_Method (Exclude 50%)'], 
                               var_name='Method', value_name='Switch_Rate')

sns.barplot(data=melted_df, x='group', y='Switch_Rate', hue='Method', ax=axes[1], palette=['#95a5a6', '#e74c3c'])
axes[1].set_title('Result Sensitivity: Switching Rate Comparison', fontweight='bold')
axes[1].set_ylabel('Switching Rate (%)')
axes[1].set_xlabel('Group')
axes[1].legend(title='Filtering Logic', bbox_to_anchor=(1, 1))

# 標示出差異
for index, row in comparison_df.iterrows():
    # 在 bar 上方標註差異
    group_idx = index
    val_dian = row['Dian_Method (Include 50%)']
    val_luise = row['Luise_Method (Exclude 50%)']
    diff = row['Diff']
    
    # 只標註差異大的
    if abs(diff) > 1:
        axes[1].text(group_idx, max(val_dian, val_luise) + 2, f"Diff: {diff:+.1f}%", 
                     ha='center', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('Impact_of_50_Percent_Logic.png', dpi=300)
plt.show()

print("\n圖表已儲存為 'Impact_of_50_Percent_Logic.png'")
print("請觀察圖 1 (BP組的柱子有多高?) 和圖 2 (BP組的紅灰柱差異有多大?)")