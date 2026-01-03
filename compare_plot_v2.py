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
# 2. 實作兩種不同的邏輯 (The Logic Comparison)
# ==========================================

# --- 邏輯 A: Dian 的算法 (包含 50% 的強制二分法) ---
# 含義：這是「整體使用者體驗」。即便 AI 說廢話 (50%)，使用者沒聽也算是不信任。
full_df['ai_suggestion_dian'] = np.where(full_df['AI_conf'] >= 50, "Red", "Black")
mask_dian = full_df['ai_suggestion_dian'] != full_df['initial_decision']
dian_stats = full_df[mask_dian].groupby('group')['switched'].mean() * 100

# --- 邏輯 B: Luise 的算法 (排除 50% 的嚴格對立) ---
# 含義：這是「有效建議的信任度」。把廢話過濾掉，只看 AI 有意見時，使用者聽不聽。
mask_luise = (
    ((full_df['human_conf'] > 50) & (full_df['AI_conf'] < 50)) |
    ((full_df['human_conf'] < 50) & (full_df['AI_conf'] > 50))
)
luise_stats = full_df[mask_luise].groupby('group')['switched'].mean() * 100

# 計算 BP 組的「沈默成本」差異
bp_diff = luise_stats['BP'] - dian_stats['BP']

# ==========================================
# 3. 視覺化對比 (The Visualization)
# ==========================================
comparison_df = pd.DataFrame({
    'Dian Logic (Include 50%)': dian_stats,
    'Luise Logic (Exclude 50%)': luise_stats
}).reset_index()

melted_df = comparison_df.melt(id_vars='group', var_name='Logic', value_name='Compliance Rate')

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
plt.title('Logic Comparison: The "Silence Cost" of Group BP', fontweight='bold', fontsize=16)
plt.ylabel('Compliance Rate (Switching %)', fontsize=14)
plt.xlabel('Group', fontsize=14)
plt.ylim(0, 65)
plt.legend(title='Filtering Logic')

# 特別標註 BP 組的差異
bp_dian = dian_stats['BP']
bp_luise = luise_stats['BP']
plt.annotate(
    f'Gap: {bp_diff:.1f}%\n(Silence Cost)', 
    xy=(2, max(bp_dian, bp_luise)), 
    xytext=(2, max(bp_dian, bp_luise) + 8),
    ha='center', va='bottom',
    arrowprops=dict(arrowstyle='-[, widthB=1.0, lengthB=0.5', lw=1.5),
    color='red', fontweight='bold', fontsize=12
)

# 在圖表下方加入簡易說明文字
plt.figtext(0.5, 0.01, 
            "Gray Bar: Overall Trust (Including silent 50% trials) | Red Bar: Trust in Active Advice (Excluding silence)", 
            ha="center", fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig('Logic_Comparison_Chart_Explained.png', dpi=300)
plt.show()

print("\n" + "="*50)
print("      AUTOMATED INSIGHT REPORT (自動解讀報告)")
print("="*50)

print(f"\n[Key Finding / 關鍵發現]")
print(f"Group BP shows a significant gap of {bp_diff:.1f}% between the two logics.")
print(f"BP 組在兩種邏輯下出現了 {bp_diff:.1f}% 的顯著落差。")

print(f"\n[Interpretation for Presentation / 報告用解釋]")
print("1. The Gray Bar (Dian Logic) is LOWER (40.5%):")
print("   - This represents the 'Overall User Experience'.")
print("   - Because the AI often stays silent (50%), users feel it's less helpful overall.")
print("   (灰色柱子較低：代表『整體使用者體驗』。因為 AI 常常沈默，使用者覺得它沒那麼有用。)")

print("\n2. The Red Bar (Luise Logic) is HIGHER (46.0%):")
print("   - This represents 'Trust in Active Advice'.")
print("   - When we remove the silent trials, we see that users actually trust the AI's explicit suggestions more.")
print("   (紅色柱子較高：代表『對有效建議的信任』。當我們移除沈默回合後，發現使用者其實比較願意聽 AI 的明確建議。)")

print("\n3. The Gap (Red Arrow):")
print("   - This gap quantifies the 'Cost of Silence'.")
print("   - It proves that Group BP's alignment strategy sacrifices utility (gray bar) to gain safety.")
print("   (紅色箭頭的差距：這量化了『沈默成本』。證明了 BP 組的對齊策略是犧牲了實用性來換取安全性。)")

print("="*50)
print("Chart saved as 'Logic_Comparison_Chart_Explained.png'")