import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 資料讀取 (Data Loading)
# ==========================================
DATA_DIR = "study_data" # 請確保你的資料夾名稱正確

def load_file(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    elif os.path.exists(os.path.join(DATA_DIR, filename)):
        return pd.read_csv(os.path.join(DATA_DIR, filename))
    else:
        print(f"[Error] {filename} not found.")
        return None

# 讀取檔案
group_A = load_file("group_A.csv")
group_B = load_file("group_B.csv")
group_BP = load_file("group_BP.csv")
group_C = load_file("group_C.csv")

# 合併資料 (如果有讀到檔案)
dfs = []
if group_A is not None: 
    group_A['group'] = 'A'; dfs.append(group_A)
if group_B is not None: 
    group_B['group'] = 'B'; dfs.append(group_B)
if group_BP is not None: 
    group_BP['group'] = 'BP'; dfs.append(group_BP)
if group_C is not None: 
    group_C['group'] = 'C'; dfs.append(group_C)

if dfs:
    full_df = pd.concat(dfs, ignore_index=True)

    # ==========================================
    # 2. 計算比例 (Calculation)
    # 邏輯：在 human_conf > 50 的情況下，initial_decision 不等於 "Red" 的比例
    # ==========================================

    # 篩選出高信心的樣本
    high_conf_df = full_df[full_df['human_conf'] > 50]

    # 計算分子：高信心且決策不是 Red (即選擇了 Black)
    not_red_high_conf = high_conf_df[high_conf_df['initial_decision'] != 'Red']

    # 計算整體比例
    if len(high_conf_df) > 0:
        overall_ratio = len(not_red_high_conf) / len(high_conf_df)
        print(f"=== 整體統計 ===")
        print(f"高信心樣本數 (>50): {len(high_conf_df)}")
        print(f"高信心但選 Black (非 Red) 的次數: {len(not_red_high_conf)}")
        print(f"比例: {overall_ratio:.2%} ({overall_ratio:.4f})")
    
    # 計算各組比例
    print(f"\n=== 各組統計 ===")
    group_stats = high_conf_df.groupby('group').apply(
        lambda x: (x['initial_decision'] != 'Red').mean()
    )
    # 將比例轉為百分比顯示
    for group, ratio in group_stats.items():
        print(f"Group {group}: {ratio:.2%} ({ratio:.4f})")

else:
    print("沒有讀取到任何資料檔案，請檢查路徑。")