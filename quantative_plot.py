import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
# UPDATE THIS PATH to your actual data folder
DATA_DIR = r"human-alignment-study-1.1\study_data" 

def load_data():
    files = {'A': 'group_A.csv', 'B': 'group_B.csv', 'BP': 'group_BP.csv', 'C': 'group_C.csv'}
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
    
    # Load demographics if available
    demo_path = os.path.join(DATA_DIR, 'demo_survey.csv')
    if os.path.exists(demo_path):
        demo = pd.read_csv(demo_path)
        full_df = full_df.merge(demo, on='participant_id', how='left')
    
    return full_df

df = load_data()

# --- Feature Engineering ---
df['switched'] = (df['initial_decision'] != df['final_decision']).astype(int)
df['initial_correct'] = (df['initial_decision'] == df['outcome']).astype(int)
df['final_correct'] = (df['final_decision'] == df['outcome']).astype(int)
df['utility_gain'] = df['final_correct'] - df['initial_correct']
df['AI_conf'] = df['AI_conf'].fillna(50)

# Define "Silence" (AI is exactly 50% confident)
df['is_silent'] = (df['AI_conf'] == 50)

# Define "Rational Opportunity" (AI Conf > Human Conf AND Disagreement)
# Logic: If AI is 80% sure and I am 60% sure, I SHOULD switch.
# First, convert AI confidence to a predicted color for comparison
df['ai_pred_color'] = np.where(df['AI_conf'] > 50, 'Red', np.where(df['AI_conf'] < 50, 'Black', 'Uncertain'))
df['conflict'] = (df['initial_decision'] != df['ai_pred_color']) & (df['ai_pred_color'] != 'Uncertain')
df['rational_opp'] = df['conflict'] & (df['AI_conf'] > df['human_conf'])


# ==========================================
# 2. Statistical Analysis (The "Quantitative" Part)
# ==========================================
print("\n" + "="*60)
print("       QUANTITATIVE ANALYSIS REPORT")
print("="*60)

# --- Test 1: Utility Gain (Does Group BP actually win?) ---
print("\n[1. Utility Gain ANOVA]")
model = ols('utility_gain ~ C(group)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
p_val_util = anova_table['PR(>F)'][0]
if p_val_util > 0.05:
    print("-> Result: NO Significant Difference. (The AI didn't help much!)")
else:
    print("-> Result: Significant Difference found.")

# --- Test 2: The Silence Cost (Paired T-test for Group BP) ---
print("\n[2. Silence Cost Analysis (Group BP)]")
# Compare switch rate when 'including silence' vs 'excluding silence'
bp_data = df[df['group'] == 'BP'].copy()

# Metric A: Standard Switch Rate (Inc. Silence)
# Logic: Treat 50% as a valid suggestion that was ignored
bp_data['dian_suggestion'] = np.where(bp_data['AI_conf'] >= 50, "Red", "Black")
bp_data['dian_conflict'] = bp_data['dian_suggestion'] != bp_data['initial_decision']
rate_with_silence = bp_data[bp_data['dian_conflict']].groupby('participant_id')['switched'].mean()

# Metric B: Effective Switch Rate (Exc. Silence)
# Logic: Remove 50% trials entirely
mask_luise = (((bp_data['human_conf'] > 50) & (bp_data['AI_conf'] < 50)) | ((bp_data['human_conf'] < 50) & (bp_data['AI_conf'] > 50)))
rate_without_silence = bp_data[mask_luise].groupby('participant_id')['switched'].mean()

# Paired T-test
# Align data to ensure we compare same users
comparison = pd.concat([rate_with_silence, rate_without_silence], axis=1, join='inner')
t_stat, p_val_silence = stats.ttest_rel(comparison.iloc[:,0], comparison.iloc[:,1])

print(f"P-value for Silence Cost: {p_val_silence:.5f}")
if p_val_silence < 0.05:
    print("-> Result: HIGHLY SIGNIFICANT. Removing silence changes user behavior.")

# ==========================================
# 3. Visualization (The Clean Charts)
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.2)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- Chart 1: The Trust Gap (Stacked Bar) ---
# Goal: Show how often they listened (Green) vs Ignored (Red) in Rational Opportunities
rational_stats = df[df['rational_opp']].groupby('group')['switched'].mean() * 100
irrational_stats = 100 - rational_stats

# Plot
axes[0].bar(rational_stats.index, irrational_stats, bottom=rational_stats, color='#e74c3c', label='Irrational Resistance', alpha=0.8)
axes[0].bar(rational_stats.index, rational_stats, color='#2ecc71', label='Rational Compliance', alpha=0.8)

axes[0].set_title('1. The Trust Gap\n(Behavior when AI is smarter than Human)', fontweight='bold')
axes[0].set_ylabel('Percentage (%)')
axes[0].legend(loc='lower left')

# Add text labels
for i, v in enumerate(rational_stats):
    axes[0].text(i, v/2, f"{v:.1f}%", ha='center', color='white', fontweight='bold')
    axes[0].text(i, v + (100-v)/2, f"{100-v:.1f}%", ha='center', color='white', fontweight='bold')

# --- Chart 2: The Silence Cost (Logic Comparison) ---
# Goal: Visualize the t-test result
comp_melt = comparison.melt(var_name='Logic', value_name='Switch Rate')
# Rename for clarity
comp_melt['Logic'] = comp_melt['Logic'].replace({0: 'With Silence (Dian)', 1: 'Without Silence (Luise)'}) # Adjust if indices differ

sns.pointplot(data=comp_melt, x='Logic', y='Switch Rate', ax=axes[1], capsize=.1, color='darkred', errorbar=('ci', 95))
# Add spaghetti lines for individuals
sns.lineplot(data=comp_melt, x='Logic', y='Switch Rate', units=comp_melt.index // 2, estimator=None, ax=axes[1], color='gray', alpha=0.1)

axes[1].set_title(f'2. The Silence Effect (p={p_val_silence:.4f})', fontweight='bold')
axes[1].set_ylabel('User Switch Rate')
axes[1].text(0.5, 0.9, 'Removing silence INCREASES trust', transform=axes[1].transAxes, ha='center', color='darkred', fontweight='bold')

plt.tight_layout()
plt.savefig('Quantitative_Analysis_Charts.png', dpi=300)
print("\nCharts saved as 'Quantitative_Analysis_Charts.png'")
plt.show()