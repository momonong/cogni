import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

# Set plotting style and ignore warnings
plt.style.use('default')
sns.set_theme(style="whitegrid", font_scale=1.2)
warnings.filterwarnings('ignore')

# ==========================================
# 1. 資料準備 (Data Loading and Preprocessing)
# ==========================================

def load_data():
    """Load and merge data from all groups"""
    # Use study_data directory in current path
    DATA_DIR = "study_data"
    
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
            print(f"✓ Loaded {filename}: {len(df)} records")
        else:
            print(f"⚠ File not found: {filename}")
    
    if not dfs:
        raise FileNotFoundError("No data files found")
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Load demographic data
    demo_path = os.path.join(DATA_DIR, 'demo_survey.csv')
    if os.path.exists(demo_path):
        demo = pd.read_csv(demo_path)
        full_df = full_df.merge(demo, on='participant_id', how='left')
        print(f"✓ Loaded demographic data: {len(demo)} records")
    
    return full_df

def preprocess_data(df):
    """Data preprocessing"""
    # Calculate basic variables
    df['switched'] = (df['initial_decision'] != df['final_decision']).astype(int)
    df['initial_correct'] = (df['initial_decision'] == df['outcome']).astype(int)
    df['final_correct'] = (df['final_decision'] == df['outcome']).astype(int)
    df['utility_gain'] = df['final_correct'] - df['initial_correct']
    
    # Handle missing values - fill AI_conf with median
    df['AI_conf'] = df['AI_conf'].fillna(df['AI_conf'].median())
    
    # Filter out attention tests (game_id < 0)
    df_clean = df[df['game_id'] >= 0].copy()
    
    print(f"Data preprocessing completed:")
    print(f"  - Total records: {len(df_clean)}")
    print(f"  - Number of participants: {df_clean['participant_id'].nunique()}")
    print(f"  - Participants per group: {df_clean.groupby('group')['participant_id'].nunique().to_dict()}")
    
    return df_clean

# ==========================================
# 2. 統計分析函數
# ==========================================

def analyze_utility_gain(df):
    """Analyze utility gain differences between groups"""
    print("\n" + "="*60)
    print("1. Utility Gain Analysis (ANOVA)")
    print("="*60)
    
    # Calculate descriptive statistics for each group
    utility_stats = df.groupby('group')['utility_gain'].agg(['mean', 'std', 'count']).round(4)
    print("\nUtility gain statistics by group:")
    print(utility_stats)
    
    # ANOVA test
    model = ols('utility_gain ~ C(group)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"\nANOVA Results:")
    print(anova_table)
    
    p_val = anova_table['PR(>F)'].iloc[0]
    if p_val < 0.05:
        print(f"\n✓ Significant difference found (p={p_val:.4f})")
        # Post-hoc test
        tukey = pairwise_tukeyhsd(endog=df['utility_gain'], groups=df['group'], alpha=0.05)
        print("\nPost-hoc Tukey HSD Results:")
        print(tukey)
    else:
        print(f"\n✗ No significant difference (p={p_val:.4f}) - Groups perform similarly")
    
    return p_val

def analyze_switching_behavior(df):
    """Analyze switching behavior differences between groups"""
    print("\n" + "-"*60)
    print("2. Switching Behavior Analysis")
    print("-"*60)
    
    # Calculate switching rates by group
    switch_stats = df.groupby('group')['switched'].agg(['mean', 'std', 'count']).round(4)
    print("\nSwitching rate statistics by group:")
    print(switch_stats)
    
    # ANOVA test
    model_switch = ols('switched ~ C(group)', data=df).fit()
    anova_switch = sm.stats.anova_lm(model_switch, typ=2)
    print(f"\nANOVA Results:")
    print(anova_switch)
    
    p_val_switch = anova_switch['PR(>F)'].iloc[0]
    if p_val_switch < 0.05:
        print(f"\n✓ Significant difference in switching behavior (p={p_val_switch:.4f})")
    else:
        print(f"\n✗ No significant difference in switching behavior (p={p_val_switch:.4f})")
    
    return p_val_switch

def analyze_silence_cost(df):
    """Analyze silence cost effect (Group BP only)"""
    print("\n" + "-"*60)
    print("3. Silence Cost Analysis (Group BP Only)")
    print("-"*60)
    
    bp_df = df[df['group'] == 'BP'].copy()
    
    if len(bp_df) == 0:
        print("⚠ No BP group data found")
        return None
    
    # Logic A (Dian): Include silence cases
    bp_df['dian_suggestion'] = np.where(bp_df['AI_conf'] >= 50, "Red", "Black")
    bp_df['dian_conflict'] = bp_df['dian_suggestion'] != bp_df['initial_decision']
    
    # Calculate average switching rate per participant in conflict situations
    user_switch_dian = bp_df[bp_df['dian_conflict']].groupby('participant_id')['switched'].mean()
    
    # Logic B (Luise): Exclude silence cases - only clear confidence differences
    mask_luise = (
        ((bp_df['human_conf'] > 50) & (bp_df['AI_conf'] < 50)) |
        ((bp_df['human_conf'] < 50) & (bp_df['AI_conf'] > 50))
    )
    user_switch_luise = bp_df[mask_luise].groupby('participant_id')['switched'].mean()
    
    # Merge for comparison (only participants with data in both logics)
    comparison = pd.concat([user_switch_dian, user_switch_luise], axis=1, join='inner')
    comparison.columns = ['With_Silence', 'Without_Silence']
    comparison = comparison.dropna()
    
    if len(comparison) == 0:
        print("⚠ Insufficient paired data for analysis")
        return None
    
    print(f"\nPaired sample size: {len(comparison)}")
    print(f"Mean switching rate (with silence): {comparison['With_Silence'].mean()*100:.2f}%")
    print(f"Mean switching rate (without silence): {comparison['Without_Silence'].mean()*100:.2f}%")
    print(f"Difference: {(comparison['Without_Silence'].mean() - comparison['With_Silence'].mean())*100:.2f}%")
    
    # Paired t-test
    t_stat, p_val_silence = stats.ttest_rel(comparison['With_Silence'], comparison['Without_Silence'])
    
    print(f"\nPaired t-test results: t={t_stat:.4f}, p={p_val_silence:.5f}")
    
    if p_val_silence < 0.05:
        if comparison['Without_Silence'].mean() > comparison['With_Silence'].mean():
            print("✓ Significant silence cost effect - Excluding silence significantly increases switching (decreases trust)")
        else:
            print("✓ Significant silence cost effect - Excluding silence significantly decreases switching (increases trust)")
    else:
        print("✗ No significant silence cost effect")
    
    return p_val_silence, comparison

def analyze_education_effect(df):
    """Analyze education level impact"""
    print("\n" + "-"*60)
    print("4. Education Impact Analysis")
    print("-"*60)
    
    if 'degree' not in df.columns:
        print("⚠ Education level column not found")
        return None
    
    # Create graduate degree binary variable
    df['is_grad'] = df['degree'].apply(lambda x: 1 if 'Graduate' in str(x) else 0)
    
    # Analyze BP group
    bp_edu = df[df['group'] == 'BP'].dropna(subset=['utility_gain', 'degree'])
    
    if len(bp_edu) == 0:
        print("⚠ No education data for BP group")
        return None
    
    grad_util = bp_edu[bp_edu['is_grad'] == 1]['utility_gain']
    non_grad_util = bp_edu[bp_edu['is_grad'] == 0]['utility_gain']
    
    print(f"\nGraduate degree utility gain: {grad_util.mean():.4f} (n={len(grad_util)})")
    print(f"Other degrees utility gain: {non_grad_util.mean():.4f} (n={len(non_grad_util)})")
    
    if len(grad_util) > 0 and len(non_grad_util) > 0:
        t_stat_edu, p_val_edu = stats.ttest_ind(grad_util, non_grad_util, equal_var=False)
        print(f"\nt-test results: t={t_stat_edu:.4f}, p={p_val_edu:.5f}")
        
        if p_val_edu < 0.05:
            print("✓ Education level significantly affects utility gain")
        else:
            print("✗ No significant education effect on utility gain")
        
        return p_val_edu
    else:
        print("⚠ Insufficient sample size for statistical test")
        return None

# ==========================================
# 3. Visualization Functions
# ==========================================

def plot_utility_gain_distribution(df):
    """Create utility gain distribution plot"""
    plt.figure(figsize=(10, 6))
    
    # Box plot with strip plot overlay
    sns.boxplot(x='group', y='utility_gain', data=df, palette='Set2', showmeans=True)
    sns.stripplot(x='group', y='utility_gain', data=df, color='black', alpha=0.3, jitter=True, size=3)
    
    plt.title('Utility Gain Distribution by Group', fontweight='bold', fontsize=14)
    plt.ylabel('Utility Gain per Trial')
    plt.xlabel('Experimental Group')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = []
    for group in df['group'].unique():
        group_data = df[df['group'] == group]['utility_gain']
        mean_val = group_data.mean()
        std_val = group_data.std()
        stats_text.append(f"Group {group}: μ={mean_val:.3f}, σ={std_val:.3f}")
    
    plt.figtext(0.02, 0.02, '\n'.join(stats_text), fontsize=9, ha='left')
    
    plt.tight_layout()
    plt.savefig('Fig1_Utility_Gain_Distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: Fig1_Utility_Gain_Distribution.png")

def plot_switching_rate_distribution(df):
    """Create switching rate distribution plot"""
    plt.figure(figsize=(10, 6))
    
    # Calculate switching rates per participant
    switch_rates = df.groupby(['group', 'participant_id'])['switched'].mean().reset_index()
    
    # Box plot with strip plot overlay
    sns.boxplot(x='group', y='switched', data=switch_rates, palette='Set3', showmeans=True)
    sns.stripplot(x='group', y='switched', data=switch_rates, color='black', alpha=0.3, jitter=True, size=3)
    
    plt.title('Switching Rate Distribution by Group', fontweight='bold', fontsize=14)
    plt.ylabel('Average Switching Rate')
    plt.xlabel('Experimental Group')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = []
    for group in switch_rates['group'].unique():
        group_data = switch_rates[switch_rates['group'] == group]['switched']
        mean_val = group_data.mean()
        std_val = group_data.std()
        stats_text.append(f"Group {group}: μ={mean_val:.3f}, σ={std_val:.3f}")
    
    plt.figtext(0.02, 0.02, '\n'.join(stats_text), fontsize=9, ha='left')
    
    plt.tight_layout()
    plt.savefig('Fig2_Switching_Rate_Distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: Fig2_Switching_Rate_Distribution.png")

def plot_silence_cost_effect(silence_result):
    """Create silence cost effect plot"""
    if silence_result is None:
        print("⚠ No silence cost data to plot")
        return
    
    p_val_silence, comparison = silence_result
    
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    plot_data = comparison.melt(var_name='Analysis_Logic', value_name='Switch_Rate')
    
    # Point plot with confidence intervals
    sns.pointplot(x='Analysis_Logic', y='Switch_Rate', data=plot_data, 
                 capsize=0.1, color='darkred', errorbar=('ci', 95))
    
    # Add individual participant lines
    for i in range(len(comparison)):
        plt.plot(['With_Silence', 'Without_Silence'], 
                [comparison.iloc[i]['With_Silence'], comparison.iloc[i]['Without_Silence']], 
                'gray', alpha=0.3, linewidth=0.8)
    
    plt.title(f'Silence Cost Effect (Paired t-test: p={p_val_silence:.4f})', 
              fontweight='bold', fontsize=14)
    plt.ylabel('Average Switching Rate')
    plt.xlabel('Analysis Logic')
    plt.grid(True, alpha=0.3)
    
    # Add significance annotation
    if p_val_silence < 0.05:
        if comparison['Without_Silence'].mean() > comparison['With_Silence'].mean():
            effect_text = 'Significant: Excluding Silence\nIncreases Switching'
        else:
            effect_text = 'Significant: Excluding Silence\nDecreases Switching'
        plt.text(0.5, 0.9, effect_text, transform=plt.gca().transAxes, 
                ha='center', color='darkred', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8))
    else:
        plt.text(0.5, 0.9, 'No Significant Difference', transform=plt.gca().transAxes, 
                ha='center', color='gray', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add statistics
    with_silence_mean = comparison['With_Silence'].mean()
    without_silence_mean = comparison['Without_Silence'].mean()
    difference = (without_silence_mean - with_silence_mean) * 100
    
    stats_text = f"With Silence: {with_silence_mean:.3f}\nWithout Silence: {without_silence_mean:.3f}\nDifference: {difference:.2f}%"
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, ha='left')
    
    plt.tight_layout()
    plt.savefig('Fig3_Silence_Cost_Effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: Fig3_Silence_Cost_Effect.png")

def plot_confidence_difference_analysis(df):
    """Create confidence difference vs switching rate plot"""
    bp_df = df[df['group'] == 'BP'].copy()
    
    if len(bp_df) == 0:
        print("⚠ No BP group data for confidence analysis")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create confidence difference variable
    bp_df['conf_diff'] = bp_df['AI_conf'] - bp_df['human_conf']
    bp_df['conf_diff_bin'] = pd.cut(bp_df['conf_diff'], bins=5, 
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Calculate switching rate by confidence difference
    conf_switch = bp_df.groupby('conf_diff_bin', observed=True)['switched'].mean()
    conf_switch_std = bp_df.groupby('conf_diff_bin', observed=True)['switched'].std()
    
    # Bar plot
    ax = conf_switch.plot(kind='bar', color='skyblue', alpha=0.7, 
                         yerr=conf_switch_std, capsize=4)
    
    plt.title('Confidence Difference vs Switching Rate (Group BP)', fontweight='bold', fontsize=14)
    plt.ylabel('Switching Rate')
    plt.xlabel('AI Confidence - Human Confidence')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(conf_switch.values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('Fig4_Confidence_Difference_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: Fig4_Confidence_Difference_Analysis.png")

def create_summary_statistics_table(df, p_values):
    """Create and save summary statistics table"""
    plt.figure(figsize=(12, 8))
    
    # Create summary data
    summary_data = []
    
    # Group statistics
    for group in ['A', 'B', 'BP', 'C']:
        group_data = df[df['group'] == group]
        if len(group_data) > 0:
            summary_data.append([
                f"Group {group}",
                len(group_data['participant_id'].unique()),
                len(group_data),
                f"{group_data['utility_gain'].mean():.4f}",
                f"{group_data['switched'].mean():.4f}",
                f"{group_data['initial_correct'].mean():.4f}",
                f"{group_data['final_correct'].mean():.4f}"
            ])
    
    # Statistical test results
    test_results = [
        ["Utility Gain ANOVA", f"p = {p_values.get('utility', 'N/A'):.4f}" if p_values.get('utility') else "N/A"],
        ["Switching Rate ANOVA", f"p = {p_values.get('switching', 'N/A'):.4f}" if p_values.get('switching') else "N/A"],
        ["Silence Cost (Paired t-test)", f"p = {p_values.get('silence', 'N/A'):.4f}" if p_values.get('silence') else "N/A"],
        ["Education Effect (t-test)", f"p = {p_values.get('education', 'N/A'):.4f}" if p_values.get('education') else "N/A"]
    ]
    
    # Create table
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Group statistics table
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=summary_data,
                      colLabels=['Group', 'N Participants', 'N Trials', 'Mean Utility Gain', 
                               'Mean Switch Rate', 'Initial Accuracy', 'Final Accuracy'],
                      cellLoc='center',
                      loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.5)
    ax1.set_title('Group Statistics Summary', fontweight='bold', fontsize=14, pad=20)
    
    # Statistical test results table
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=test_results,
                      colLabels=['Statistical Test', 'Result'],
                      cellLoc='center',
                      loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.5)
    ax2.set_title('Statistical Test Results', fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('Fig5_Summary_Statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: Fig5_Summary_Statistics.png")

# ==========================================
# 4. Main Execution Function
# ==========================================

def main():
    """Main analysis workflow"""
    print("Starting Quantitative Analysis...")
    
    try:
        # Load and preprocess data
        df = load_data()
        df_clean = preprocess_data(df)
        
        # Perform statistical analyses
        print("\n" + "="*80)
        print("                    STATISTICAL ANALYSIS REPORT")
        print("="*80)
        
        p_utility = analyze_utility_gain(df_clean)
        p_switch = analyze_switching_behavior(df_clean)
        silence_result = analyze_silence_cost(df_clean)
        p_education = analyze_education_effect(df_clean)
        
        # Store p-values for summary
        p_values = {
            'utility': p_utility,
            'switching': p_switch,
            'silence': silence_result[0] if silence_result else None,
            'education': p_education
        }
        
        # Generate individual plots
        print("\n" + "="*60)
        print("Generating Individual Plots...")
        print("="*60)
        
        plot_utility_gain_distribution(df_clean)
        plot_switching_rate_distribution(df_clean)
        plot_silence_cost_effect(silence_result)
        plot_confidence_difference_analysis(df_clean)
        create_summary_statistics_table(df_clean, p_values)
        
        # Summary report
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"1. Utility gain group differences: {'Significant' if p_utility < 0.05 else 'Not significant'} (p={p_utility:.4f})")
        print(f"2. Switching behavior group differences: {'Significant' if p_switch < 0.05 else 'Not significant'} (p={p_switch:.4f})")
        
        if silence_result:
            p_silence = silence_result[0]
            print(f"3. Silence cost effect: {'Significant' if p_silence < 0.05 else 'Not significant'} (p={p_silence:.4f})")
        else:
            print("3. Silence cost effect: Insufficient data for analysis")
        
        if p_education:
            print(f"4. Education level impact: {'Significant' if p_education < 0.05 else 'Not significant'} (p={p_education:.4f})")
        else:
            print("4. Education level impact: Insufficient data for analysis")
        
        print(f"\n✓ Analysis completed! Generated 5 individual plots.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()