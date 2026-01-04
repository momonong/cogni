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
    # 邏輯：在 human_conf < 50 的情況下，計算 initial_decision 為 Red 與 Black 的比例
    # ==========================================

    # 篩選出低信心的樣本 (< 50)
    low_conf_df = full_df[full_df['human_conf'] < 50]

    if len(low_conf_df) > 0:
        print(f"=== 整體統計 (Human Confidence < 50) ===")
        total_low = len(low_conf_df)
        
        # 計算個別數量
        counts = low_conf_df['initial_decision'].value_counts()
        red_count = counts.get('Red', 0)
        black_count = counts.get('Black', 0)
        
        # 計算比例
        red_ratio = red_count / total_low
        black_ratio = black_count / total_low
        
        print(f"低信心樣本總數: {total_low}")
        print(f"初始選擇 'Red': {red_count} ({red_ratio:.2%})")
        print(f"初始選擇 'Black': {black_count} ({black_ratio:.2%})")
        
        # 計算各組比例
        print(f"\n=== 各組統計 (比例) ===")
        # normalize=True 會自動算出比例
        group_stats = low_conf_df.groupby('group')['initial_decision'].value_counts(normalize=True).unstack().fillna(0)
        
        # 為了閱讀方便，轉成百分比格式顯示
        pd.options.display.float_format = '{:.2%}'.format
        print(group_stats)
    else:
        print("未發現任何信心小於 50 的數據。")

else:
    print("沒有讀取到任何資料檔案，請檢查路徑。")