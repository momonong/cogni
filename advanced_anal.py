import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 環境設置與資料讀取
# ==========================================
# 請確保 study_data 資料夾在你的路徑中，或修改此處路徑
DATA_DIR = "study_data" 

def load_file(filename):
    # 嘗試多種路徑讀取，增加穩健性
    if os.path.exists(filename):
        print(f"[Loading] {filename}...")
        return pd.read_csv(filename)
    elif os.path.exists(os.path.join(DATA_DIR, filename)):
        print(f"[Loading] {os.path.join(DATA_DIR, filename)}...")
        return pd.read_csv(os.path.join(DATA_DIR, filename))
    else:
        print(f"[Error] {filename} not found.")
        return None

# 讀取資料
demo = load_file("demo_survey.csv")
group_A = load_file("group_A.csv")
group_B = load_file("group_B.csv")
group_BP = load_file("group_BP.csv")
group_C = load_file("group_C.csv")

if any(df is None for df in [demo, group_A, group_B, group_BP, group_C]):
    print("Warning: Some files are missing. Please check your directory.")
    # 這裡為了讓你複製貼上能跑，加上了簡單的防呆，實際執行請確保檔案存在

# 標記組別
if group_A is not None:
    group_A['group'] = 'A (Biased Away)'
    group_B['group'] = 'B (Biased Towards)'
    group_BP['group'] = 'BP (Realigned)'
    group_C['group'] = 'C (No Bias)'

    # 合併資料
    all_games = pd.concat([group_A, group_B, group_BP, group_C], ignore_index=True)
    full_df = all_games.merge(demo, on='participant_id', how='left')

    # ==========================================
    # 2. 特徵工程 (Feature Engineering)
    # ==========================================

    # 確保有 AI_decision
    if 'AI_decision' not in full_df.columns:
        full_df['AI_decision'] = (full_df['AI_conf'] > 50).astype(int)

    # 1. 定義依從 (Switching)
    full_df['switched'] = full_df['initial_decision'] != full_df['final_decision']

    # 2. 定義意見不合 (Disagreement)
    # 只有當人類初始決策與 AI 決策不同時，才納入分析
    disagreement_df = full_df[full_df['initial_decision'] != full_df['AI_decision']].copy()

    # 3. 定義情境 (Scenarios)
    # Scenario 1: AI Correct (Helpful) -> AI 決策與結果一致
    # Scenario 2: AI Incorrect (Trap) -> AI 決策與結果不一致
    disagreement_df['scenario'] = np.where(
        disagreement_df['AI_decision'] == disagreement_df['outcome'], 
        'AI Correct (Helpful)', 
        'AI Incorrect (Trap)'
    )

    # ==========================================
    # 3. 繪圖與輸出 (Plotting)
    # ==========================================
    sns.set_theme(style="whitegrid", font_scale=1.2)
    group_palette = {
        'A (Biased Away)': '#d62728',   # 紅
        'B (Biased Towards)': '#ff7f0e', # 橘
        'BP (Realigned)': '#2ca02c',     # 綠
        'C (No Bias)':  '#1f77b4'      # 藍
    }

    print("\nGenerating optimized individual plots...")

    # --- Figure 1: Scenario Counts (The "Fundamental Flaw") ---
    # 這張圖證明 AI 幾乎都是來亂的 (100% Trap)
    plt.figure(figsize=(8, 6))
    sns.countplot(data=disagreement_df, x='scenario', hue='group', palette=group_palette)
    plt.title('Figure 1: The Distractor AI Problem\n(Count of Helpful vs. Trap Events)', fontweight='bold')
    plt.xlabel('Scenario')
    plt.ylabel('Count of Events')
    plt.tight_layout()
    plt.savefig('Fig1_Distractor_AI_Counts.png', dpi=300)
    plt.close()
    print("✓ Saved Fig1_Distractor_AI_Counts.png")

    # --- Figure 2: Compliance in Traps (The "Resistance" Check) ---
    # 只分析 Trap 情境，看誰最容易上當
    trap_data = disagreement_df[disagreement_df['scenario'] == 'AI Incorrect (Trap)']
    if not trap_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=trap_data,
            x='group', 
            y='switched', 
            hue='group',
            errorbar=('ci', 95),
            palette=group_palette,
            edgecolor='black',
            alpha=0.8
        )
        plt.title('Figure 2: Vulnerability to AI Traps\n(Compliance Rate when AI is Wrong)', fontweight='bold')
        plt.ylabel('Compliance Rate (Switching %)')
        plt.xlabel('Group')
        plt.ylim(0, 0.25) # 設定上限以便看清差異
        plt.tight_layout()
        plt.savefig('Fig2_Trap_Compliance.png', dpi=300)
        plt.close()
        print("✓ Saved Fig2_Trap_Compliance.png")

    # --- Figure 3: Temporal Dynamics (Learning to Resist) ---
    # 觀察隨時間推移，人們是否學會了不聽 AI 的話
    if not trap_data.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=trap_data, 
            x='trial_id', 
            y='switched', 
            hue='group', 
            marker='o',
            palette=group_palette,
            errorbar=None
        )
        plt.title('Figure 3: Learning to Resist Bad Advice Over Time', fontweight='bold')
        plt.xlabel('Round (Trial ID)')
        plt.ylabel('Compliance Rate')
        plt.ylim(0, 0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('Fig3_Temporal_Resistance.png', dpi=300)
        plt.close()
        print("✓ Saved Fig3_Temporal_Resistance.png")

    # --- Figure 4: Cost of Silence (Distribution) ---
    # 證明 BP 組的極化/龜縮策略
    plt.figure(figsize=(10, 6))
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
    plt.axvline(50, color='black', linestyle='--', alpha=0.5)
    plt.axvspan(40, 60, color='gray', alpha=0.1, label='Silence Zone (Avoided by BP)')
    plt.title('Figure 4: Evidence of "Cowardly AI" (Confidence Distribution)', fontweight='bold')
    plt.xlabel('AI Confidence')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig4_Silence_Cost.png', dpi=300)
    plt.close()
    print("✓ Saved Fig4_Silence_Cost.png")

    # --- Figure 5: Calibration / Danger Zone (NEW!) ---
    # 這是回答 Markdown 中 "Alignment Map" 的關鍵圖
    # 我們畫出「AI 信心」vs「實際準確率」
    plt.figure(figsize=(10, 8))
    
    for name, group_data in full_df.groupby('group'):
        # 將 AI 信心分箱 (0-10, 10-20...)
        bins = np.linspace(0, 100, 11)
        labels = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        group_data = group_data.copy()
        group_data['conf_bin'] = pd.cut(group_data['AI_conf'], bins=bins, labels=labels)
        
        # 計算每個信心區間的實際準確率 (Observed Accuracy)
        # 注意：這裡假設 AI 信心 > 50 代表預測 1 (Red)，否則預測 0 (Black)
        # 準確率 = (Outcome == AI_Decision) 的平均值
        calibration = group_data.groupby('conf_bin', observed=False).agg(
            acc=('outcome', lambda x: (x == group_data.loc[x.index, 'AI_decision']).mean())
        ).reset_index()
        
        plt.plot(calibration['conf_bin'], calibration['acc'], marker='o', label=name, color=group_palette[name], linewidth=2)

    # 畫出完美校準線 (對角線)
    plt.plot([0, 100], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
    
    plt.title('Figure 5: AI Calibration Analysis (Defensive Alignment)', fontweight='bold')
    plt.xlabel('AI Confidence')
    plt.ylabel('Observed Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Fig5_Calibration_Map.png', dpi=300)
    plt.close()
    print("✓ Saved Fig5_Calibration_Map.png")

    # --- Figure 6: Education (Fairness) ---
    plt.figure(figsize=(10, 6))
    if 'degree' in full_df.columns:
        df_edu = full_df.dropna(subset=['degree'])
        sns.barplot(data=df_edu, x='degree', y='switched', hue='group', palette=group_palette, errorbar=('ci', 95))
        plt.title('Figure 6: Fairness Check (Compliance by Education)', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Compliance Rate')
        plt.tight_layout()
        plt.savefig('Fig6_Education_Fairness.png', dpi=300)
        plt.close()
        print("✓ Saved Fig6_Education_Fairness.png")

    print("\nAll optimized plots generated successfully!")