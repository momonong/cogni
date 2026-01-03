import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 設定與讀取資料
# ==========================================
# 請確認此路徑與你電腦上的資料夾結構一致
DATA_DIR = r"D:\projects\cogni\human-alignment-study-1.1\study_data"
# 如果你在不同電腦，請修改上面的路徑

def load_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"[Loading] {filename}...")
        return pd.read_csv(path)
    else:
        print(f"[Error] {filename} not found at {path}")
        return None

# 讀取檔案
dfs = {
    "A": load_file("group_A.csv"),
    "B": load_file("group_B.csv"),
    "BP": load_file("group_BP.csv"),
    "C": load_file("group_C.csv")
}
demo = load_file("demo_survey.csv")

if any(v is None for v in dfs.values()):
    raise FileNotFoundError("部分檔案讀取失敗，請檢查路徑。")

# 合併所有數據
all_data = []
for group_name, df in dfs.items():
    df['group'] = group_name
    all_data.append(df)
full_df = pd.concat(all_data, ignore_index=True)

# 基礎前處理
full_df['switched'] = full_df['initial_decision'] != full_df['final_decision']
full_df['AI_conf'] = full_df['AI_conf'].fillna(50) # 填補空值

# ==========================================
# 2. 實作兩種不同的邏輯 (The Logic Comparison)
# ==========================================

# --- 邏輯 A: Dian 的算法 (包含 50% 的強制二分法) ---
# [Dian Logic] >= 50 視為 Red, < 50 視為 Black
full_df['ai_suggestion_dian'] = np.where(full_df['AI_conf'] >= 50, "Red", "Black")
# 定義衝突：AI 建議與人類初始不同
mask_dian = full_df['ai_suggestion_dian'] != full_df['initial_decision']
# 計算切換率
dian_stats = full_df[mask_dian].groupby('group')['switched'].mean() * 100

# --- 邏輯 B: Luise 的算法 (排除 50% 的嚴格對立) ---
# [Luise Logic] 排除 AI_conf == 50 或 human_conf == 50 的情況
# 且要求兩者在 50 的對立面
mask_luise = (
    ((full_df['human_conf'] > 50) & (full_df['AI_conf'] < 50)) |
    ((full_df['human_conf'] < 50) & (full_df['AI_conf'] > 50))
)
# 計算切換率
luise_stats = full_df[mask_luise].groupby('group')['switched'].mean() * 100

# ==========================================
# 3. 視覺化對比 (The Visualization)
# ==========================================
# 整理數據以便繪圖
comparison_df = pd.DataFrame({
    'Dian Logic (Include 50%)': dian_stats,
    'Luise Logic (Exclude 50%)': luise_stats
}).reset_index()

# 轉換為長格式 (Melt)
melted_df = comparison_df.melt(id_vars='group', var_name='Logic', value_name='Compliance Rate')

# 設定繪圖風格
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 7))

# 畫出長條圖
bar_plot = sns.barplot(
    data=melted_df, 
    x='group', 
    y='Compliance Rate', 
    hue='Logic',
    palette={'Dian Logic (Include 50%)': 'gray', 'Luise Logic (Exclude 50%)': '#e74c3c'},
    alpha=0.9
)

# 加入數值標籤
for container in bar_plot.containers:
    bar_plot.bar_label(container, fmt='%.1f%%', padding=3)

# 標註差異
plt.title('Logic Comparison: How "Handling 50%" Changes the Results', fontweight='bold', fontsize=16)
plt.ylabel('Compliance Rate (Switching %)', fontsize=14)
plt.xlabel('Group', fontsize=14)
plt.ylim(0, 60)
plt.legend(title='Filtering Logic')

# 特別標註 BP 組的差異
bp_dian = dian_stats['BP']
bp_luise = luise_stats['BP']
plt.annotate(
    f'Diff: {bp_luise - bp_dian:.1f}%\n(Silence Cost)', 
    xy=(2, max(bp_dian, bp_luise)), 
    xytext=(2, max(bp_dian, bp_luise) + 5),
    ha='center', va='bottom',
    arrowprops=dict(arrowstyle='-[, widthB=1.0, lengthB=0.5', lw=1.5),
    color='red', fontweight='bold'
)

plt.tight_layout()
plt.savefig('Logic_Comparison_Chart.png', dpi=300)
plt.show()

print("圖表已生成：Logic_Comparison_Chart.png")