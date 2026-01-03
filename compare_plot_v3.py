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
# 2. 計算核心數值 (Calculations)
# ==========================================

# --- A. 計算 AI 沈默率 (Silence Rate) ---
# 每一組有多少比例的回合 AI 信心是 50%
full_df['is_silent'] = (full_df['AI_conf'] == 50)
silence_counts = full_df.groupby('group')['is_silent'].value_counts(normalize=True).unstack().fillna(0) * 100
silence_rate = silence_counts[True] # 取出 True (是沈默) 的比例

# --- B. 計算兩種邏輯的切換率 ---

# 邏輯 1: Dian 的算法 (包含 50%)
full_df['ai_suggestion_dian'] = np.where(full_df['AI_conf'] >= 50, "Red", "Black")
mask_dian = full_df['ai_suggestion_dian'] != full_df['initial_decision']
dian_stats = full_df[mask_dian].groupby('group')['switched'].mean() * 100

# 邏輯 2: Luise 的算法 (排除 50%)
mask_luise = (
    ((full_df['human_conf'] > 50) & (full_df['AI_conf'] < 50)) |
    ((full_df['human_conf'] < 50) & (full_df['AI_conf'] > 50))
)
luise_stats = full_df[mask_luise].groupby('group')['switched'].mean() * 100

# 整合為表格
comparison_df = pd.DataFrame({
    'Silence Rate (%)': silence_rate,
    'Dian Logic (Inc. 50%)': dian_stats,
    'Luise Logic (Exc. 50%)': luise_stats
})
comparison_df['Diff (Trust Gain)'] = comparison_df['Luise Logic (Exc. 50%)'] - comparison_df['Dian Logic (Inc. 50%)']

# ==========================================
# 3. 輸出數值報告 (Data Report)
# ==========================================
print("\n" + "="*60)
print("             DETAILED DATA REPORT (詳細數據報告)")
print("="*60)

print("\n[Table 1: Impact of Silence & Method Difference]")
print(comparison_df.round(2))
print("\n" + "-"*60)

# 特別針對 BP 組的詳細解讀
bp_silence = comparison_df.loc['BP', 'Silence Rate (%)']
bp_dian = comparison_df.loc['BP', 'Dian Logic (Inc. 50%)']
bp_luise = comparison_df.loc['BP', 'Luise Logic (Exc. 50%)']
bp_diff = comparison_df.loc['BP', 'Diff (Trust Gain)']

print(f"\n[Key Analysis for Group BP]")
print(f"1. Silence Rate: {bp_silence:.2f}%")
print(f"   -> Meaning: In Group BP, the AI stays silent (gives 50% conf) in {bp_silence:.2f}% of trials.")
print(f"   (BP 組有 {bp_silence:.2f}% 的回合 AI 是保持沈默的。)")

print(f"\n2. Trust Recovery: {bp_dian:.2f}% -> {bp_luise:.2f}% (+{bp_diff:.2f}%)")
print(f"   -> Meaning: When we remove the silent trials, the switching rate increases by {bp_diff:.2f}%.")
print(f"   (當我們移除沈默回合後，切換率提升了 {bp_diff:.2f}%。這證明了沈默降低了使用者的服從意願。)")

# ==========================================
# 4. 畫圖 (Visualization)
# ==========================================
melted_df = comparison_df[['Dian Logic (Inc. 50%)', 'Luise Logic (Exc. 50%)']].reset_index().melt(id_vars='group', var_name='Logic', value_name='Compliance Rate')

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 7))

bar_plot = sns.barplot(
    data=melted_df, 
    x='group', 
    y='Compliance Rate', 
    hue='Logic',
    palette={'Dian Logic (Inc. 50%)': 'gray', 'Luise Logic (Exc. 50%)': '#e74c3c'},
    alpha=0.9
)

for container in bar_plot.containers:
    bar_plot.bar_label(container, fmt='%.1f%%', padding=3)

plt.title('Logic Comparison: The Hidden Cost of Silence in Group BP', fontweight='bold', fontsize=16)
plt.ylabel('Compliance Rate (Switching %)', fontsize=14)
plt.xlabel('Group', fontsize=14)
plt.ylim(0, 65)
plt.legend(title='Filtering Logic')

# 標註 BP 組差異
plt.annotate(
    f'Gap: +{bp_diff:.1f}%\n(Silence Cost)', 
    xy=(2, max(bp_dian, bp_luise)), 
    xytext=(2, max(bp_dian, bp_luise) + 8),
    ha='center', va='bottom',
    arrowprops=dict(arrowstyle='-[, widthB=1.0, lengthB=0.5', lw=1.5),
    color='red', fontweight='bold', fontsize=12
)

plt.tight_layout()
plt.show()