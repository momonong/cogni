import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# ==========================================
# 1. 資料讀取 (Data Loading)
# ==========================================
DATA_DIR = r"human-alignment-study-1.1\study_data" # 請確認你的路徑

def load_data():
    files = {
        'A': 'group_A.csv',
        'B': 'group_B.csv',
        'BP': 'group_BP.csv',
        'C': 'group_C.csv'
    }
    dfs = []
    for group, filename in files.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['group'] = group
            dfs.append(df)
        else:
            print(f"Warning: {filename} not found.")
            return None
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 讀取人口統計資料
    demo_path = os.path.join(DATA_DIR, 'demo_survey.csv')
    if os.path.exists(demo_path):
        demo = pd.read_csv(demo_path)
        full_df = full_df.merge(demo, on='participant_id', how='left')
    
    return full_df

df = load_data()

# --- 前處理 ---
df['switched'] = (df['initial_decision'] != df['final_decision']).astype(int)
df['initial_correct'] = (df['initial_decision'] == df['outcome']).astype(int)
df['final_correct'] = (df['final_decision'] == df['outcome']).astype(int)
df['utility_gain'] = df['final_correct'] - df['initial_correct']
df['AI_conf'] = df['AI_conf'].fillna(50)

print("\n" + "="*60)
print("       STATISTICAL ANALYSIS REPORT (量性分析報告)")
print("="*60)

# ==========================================
# 分析 1: 效用差異檢定 (Utility Gain ANOVA)
# ==========================================
print("\n[1. Utility Gain Analysis (ANOVA)]")
# 檢驗不同組別的 Utility Gain 平均值是否有顯著差異
model = ols('utility_gain ~ C(group)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

p_val = anova_table['PR(>F)'][0]
if p_val < 0.05:
    print(f"Result: Significant difference found (p={p_val:.4f}).")
    print("-> Group choice DOES affect utility.")
    
    # 事後檢定 (Post-hoc) 找出是哪兩組不同
    tukey = pairwise_tukeyhsd(endog=df['utility_gain'], groups=df['group'], alpha=0.05)
    print("\n[Post-hoc Tukey HSD Results]")
    print(tukey)
else:
    print(f"Result: No significant difference (p={p_val:.4f}). Groups perform similarly.")

# ==========================================
# 分析 2: 信任行為檢定 (Trust/Switching ANOVA)
# ==========================================
print("\n" + "-"*60)
print("\n[2. Trust Behavior Analysis (Switching Rate)]")
# 檢驗不同組別的 Switching Rate 是否有顯著差異
model_switch = ols('switched ~ C(group)', data=df).fit()
anova_switch = sm.stats.anova_lm(model_switch, typ=2)
print(anova_switch)

p_val_switch = anova_switch['PR(>F)'][0]
if p_val_switch < 0.05:
    print(f"Result: Significant difference in trust (p={p_val_switch:.4f}).")
else:
    print(f"Result: No significant difference in trust (p={p_val_switch:.4f}).")

# ==========================================
# 分析 3: 沈默成本顯著性 (The Silence Cost - Paired T-test for BP)
# ==========================================
print("\n" + "-"*60)
print("\n[3. The Silence Cost Analysis (Group BP Only)]")

bp_df = df[df['group'] == 'BP'].copy()

# 邏輯 A (Dian): 包含沈默
bp_df['dian_suggestion'] = np.where(bp_df['AI_conf'] >= 50, "Red", "Black")
bp_df['dian_conflict'] = bp_df['dian_suggestion'] != bp_df['initial_decision']
# 計算每個受試者的平均切換率 (Dian Logic)
user_switch_dian = bp_df[bp_df['dian_conflict']].groupby('participant_id')['switched'].mean()

# 邏輯 B (Luise): 排除沈默
mask_luise = (
    ((bp_df['human_conf'] > 50) & (bp_df['AI_conf'] < 50)) |
    ((bp_df['human_conf'] < 50) & (bp_df['AI_conf'] > 50))
)
user_switch_luise = bp_df[mask_luise].groupby('participant_id')['switched'].mean()

# 合併比較 (只取兩種邏輯下都有數據的受試者)
comparison = pd.concat([user_switch_dian, user_switch_luise], axis=1, join='inner')
comparison.columns = ['With_Silence', 'Without_Silence']

# 成對 T 檢定
t_stat, p_val_silence = stats.ttest_rel(comparison['With_Silence'], comparison['Without_Silence'])

print(f"Mean Switch Rate (With Silence):    {comparison['With_Silence'].mean()*100:.2f}%")
print(f"Mean Switch Rate (Without Silence): {comparison['Without_Silence'].mean()*100:.2f}%")
print(f"Difference: {comparison['Without_Silence'].mean()*100 - comparison['With_Silence'].mean()*100:.2f}%")
print(f"Paired T-test Result: t={t_stat:.4f}, p={p_val_silence:.5f}")

if p_val_silence < 0.05:
    print("Conclusion: Removing silence SIGNIFICANTLY increases trust.")
    print("-> The 'Silence Cost' is statistically real, not random noise.")
else:
    print("Conclusion: The difference is not statistically significant.")

# ==========================================
# 分析 4: 教育程度影響 (Education Effect)
# ==========================================
print("\n" + "-"*60)
print("\n[4. Education Impact Analysis]")

# 簡單化：比較 "Graduate degree" vs 其他
if 'degree' in df.columns:
    # 創建二分變數
    df['is_grad'] = df['degree'].apply(lambda x: 1 if x == 'Graduate degree' else 0)
    
    # 針對 BP 組分析學歷是否影響 Utility Gain
    bp_edu = df[df['group'] == 'BP'].dropna(subset=['utility_gain'])
    
    grad_util = bp_edu[bp_edu['is_grad'] == 1]['utility_gain']
    non_grad_util = bp_edu[bp_edu['is_grad'] == 0]['utility_gain']
    
    t_stat_edu, p_val_edu = stats.ttest_ind(grad_util, non_grad_util, equal_var=False)
    
    print(f"Avg Utility Gain (Graduate): {grad_util.mean():.4f}")
    print(f"Avg Utility Gain (Others):   {non_grad_util.mean():.4f}")
    print(f"T-test Result: t={t_stat_edu:.4f}, p={p_val_edu:.5f}")
    
    if p_val_edu < 0.05:
        print("Conclusion: Education level significantly affects utility gain in Group BP.")
    else:
        print("Conclusion: No significant education effect found.")
else:
    print("Column 'degree' not found for analysis.")

print("\n" + "="*60)