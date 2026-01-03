import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 設定與讀取資料
# ==========================================
DATA_DIR = r"D:\projects\cogni\human-alignment-study-1.1\study_data" # 請確認路徑

def load_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"[Loading] {filename}...")
        return pd.read_csv(path)
    else:
        print(f"[Error] {filename} not found.")
        return None

dfs = {
    "A": load_file("group_A.csv"),
    "B": load_file("group_B.csv"),
    "BP": load_file("group_BP.csv"),
    "C": load_file("group_C.csv")
}
demo = load_file("demo_survey.csv")

if any(v is None for v in dfs.values()):
    raise FileNotFoundError("Critical files missing.")

all_data = []
for group_name, df in dfs.items():
    df['group'] = group_name
    all_data.append(df)
full_df = pd.concat(all_data, ignore_index=True)

# 基礎前處理
full_df['switched'] = full_df['initial_decision'] != full_df['final_decision']
# 為了公平比較，統一填補空值，但在 Luise 邏輯中會被過濾掉
full_df['AI_conf'] = full_df['AI_conf'].fillna(50) 
full_df['human_conf'] = full_df['human_conf'].fillna(50)

# ==========================================
# 2. 定義兩種過濾邏輯
# ==========================================

# --- 邏輯 A: Dian's Logic (寬鬆/包含沈默) ---
# 1. 優先使用 calibrated conf (如果有)，否則用 AI_conf
#    (注意：原腳本有這一步，這裡還原它)
if 'Ai_calibrated_conf' in full_df.columns:
    ai_conf_dian = full_df['Ai_calibrated_conf'].fillna(full_df['AI_conf'])
else:
    ai_conf_dian = full_df['AI_conf']

# 2. 強制二分： >= 50 就是 Red (包含 50)
full_df['ai_suggestion_dian'] = np.where(ai_conf_dian >= 50, "Red", "Black")

# 3. 定義衝突
full_df['conflict_dian'] = full_df['ai_suggestion_dian'] != full_df['initial_decision']


# --- 邏輯 B: Luise's Logic (嚴格/排除沈默) ---
# 1. 嚴格對立：人 > 50 且 AI < 50，或反之。
#    這自然排除了 AI=50 或 人類=50 的情況
full_df['conflict_luise'] = (
    ((full_df['human_conf'] > 50) & (full_df['AI_conf'] < 50)) |
    ((full_df['human_conf'] < 50) & (full_df['AI_conf'] > 50))
)

# ==========================================
# 3. 分析與視覺化
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- 圖 1: 樣本數差異 (Sample Count Impact) ---
# 看看兩種邏輯分別篩出了多少筆資料？
count_data = []
for g in ['A', 'B', 'BP', 'C']:
    n_dian = full_df[(full_df['group'] == g) & (full_df['conflict_dian'])].shape[0]
    n_luise = full_df[(full_df['group'] == g) & (full_df['conflict_luise'])].shape[0]
    count_data.append({'group': g, 'Logic': 'Dian (Inc. 50%)', 'Count': n_dian})
    count_data.append({'group': g, 'Logic': 'Luise (Exc. 50%)', 'Count': n_luise})

df_count = pd.DataFrame(count_data)
sns.barplot(data=df_count, x='group', y='Count', hue='Logic', ax=axes[0], palette=['gray', '#e74c3c'])
axes[0].set_title('1. How many "Conflicts" were found?', fontweight='bold')
axes[0].set_ylabel('Number of Trials')

# --- 圖 2: 切換率分佈 (Switching Rate Distribution) ---
# 在篩選出的衝突中，人類聽從 AI 的比例
rate_data = []
for g in ['A', 'B', 'BP', 'C']:
    rate_dian = full_df[(full_df['group'] == g) & (full_df['conflict_dian'])]['switched'].mean() * 100
    rate_luise = full_df[(full_df['group'] == g) & (full_df['conflict_luise'])]['switched'].mean() * 100
    rate_data.append({'group': g, 'Logic': 'Dian (Inc. 50%)', 'Switch%': rate_dian})
    rate_data.append({'group': g, 'Logic': 'Luise (Exc. 50%)', 'Switch%': rate_luise})

df_rate = pd.DataFrame(rate_data)
sns.barplot(data=df_rate, x='group', y='Switch%', hue='Logic', ax=axes[1], palette=['gray', '#e74c3c'])
axes[1].set_title('2. Switching Rate (Trust Level)', fontweight='bold')
axes[1].set_ylabel('Switching Rate (%)')
axes[1].set_ylim(0, 60)

# --- 圖 3: BP 組的時間趨勢對比 (Temporal Trend for BP) ---
# 這就是你想看的趨勢差異
bp_df = full_df[full_df['group'] == 'BP'].copy()

# 計算每一回合的平均切換率
trend_dian = bp_df[bp_df['conflict_dian']].groupby('trial_id')['switched'].mean() * 100
trend_luise = bp_df[bp_df['conflict_luise']].groupby('trial_id')['switched'].mean() * 100

# 為了讓趨勢更明顯，我們使用移動平均 (Rolling Average)
window = 3
axes[2].plot(trend_dian.index, trend_dian.rolling(window, min_periods=1).mean(), 
             label='Dian Trend (Inc. Silence)', color='gray', linewidth=3, linestyle='--')
axes[2].plot(trend_luise.index, trend_luise.rolling(window, min_periods=1).mean(), 
             label='Luise Trend (Exc. Silence)', color='#e74c3c', linewidth=3)

axes[2].set_title('3. Temporal Trend: Group BP Only', fontweight='bold')
axes[2].set_ylabel('Switching Rate (%)')
axes[2].set_xlabel('Trial ID')
axes[2].legend()

plt.tight_layout()
plt.savefig('Logic_Distribution_Analysis.png', dpi=300)
plt.show()

print("分析完成。請查看 'Logic_Distribution_Analysis.png'")
print("重點觀察：")
print("1. 圖 1 BP 組的紅灰柱高度差：這就是被過濾掉的『沈默回合』數量。")
print("2. 圖 3 的兩條線走勢：紅色線(排除沈默)是否比灰色線(包含沈默)更高且更穩定？")