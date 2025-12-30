## 一、 核心運算與硬體指標 (必考計算大題)

### 1. 卷積運算量計算 (Conv2D MACs)

- **物理意義**：衡量推理時的運算負荷、延遲與耗電。
- **標準卷積公式**：
$$\text{MACs} = (H_{out} \times W_{out} \times C_{out}) \times (K \times K \times C_{in})$$
- **深度可分離卷積 (DSC) 公式**：
$$\text{MACs} = \underbrace{(H_{out} \times W_{out} \times C_{in} \times K^2)}*{\text{Depthwise (DW)}} + \underbrace{(H*{out} \times W_{out} \times C_{in} \times C_{out})}_{\text{Pointwise (PW)}}$$
- **輸出尺寸計算 (考 Stride 時必用)**：
$$H_{out} = \lfloor \frac{H_{in} + 2P - K}{S} \rfloor + 1$$
    - $P$：Padding（填充）；$S$：Stride（步長）；$K$：Kernel Size（卷積核大小）。

### 2. 模型體重 (Parameters)

- **物理意義**：決定模型檔案大小 (.bin/.pth) 與儲存空間 (Flash)。
- **公式**：$K \times K \times C_{in} \times C_{out}$

### 3. Roofline Model (性能瓶頸分析)

- **算術強度 (Arithmetic Intensity)**：$\text{MACs} / \text{Memory Access (Bytes)}$。
- **運算密集型 (Compute-bound)**：落在水平線，優化 MACs 有效。
- **記憶體密集型 (Memory-bound)**：落在斜線，優化頻寬 (Bandwidth) 或量化有效。

---

## 二、 輕量化網路架構對比 (是非/複選題專用)

| 架構名稱 | 核心技術 | 物理意義 / 解決問題 |
| --- | --- | --- |
| **MobileNet V1** | **Depthwise Separable Conv (DSC)** | 將空間與通道去耦合，計算量降至約 $1/K^2$。 |
| **MobileNet V2** | **Inverted Residual & Linear Bottleneck** | 先擴張通道再運算，最後一層不加 ReLU 以防止低維資訊流失。 |
| **ShuffleNet V1** | **Group Conv & Channel Shuffle** | 用分組降低 $1 \times 1$ 卷積成本，透過洗牌解決資訊不通。 |
| **ShuffleNet V2** | **Channel Split** | 減少資料搬移與加法 (Add)，一半通道直接流過，對硬體最友善。 |
| **MobileNet V3** | **NAS + Hard-Swish + SE Block** | 用電腦搜出架構；用分段線性取代指數運算 (Exp) 省電。 |

---

## 三、 模型壓縮技術 (內部消息重點)

### 1. 線性量化公式 (Linear Quantization)

- **公式**：$r = S(q - Z)$ 或 $q = \text{round}(r/S + Z)$
- **Scale ($S$)**：$\frac{r_{max} - r_{min}}{q_{max} - q_{min}}$ (代表量化的解析度)。
- **Zero-point ($Z$)**：$\text{round}(q_{min} - \frac{r_{min}}{S})$。將真實世界 0.0 對齊到整數。
- **對稱量化 (Symmetric)**：$Z=0$，範圍 $[-max, max]$，計算快但精確度較低。
- **非對稱量化 (Asymmetric)**：$Z \neq 0$，範圍 $[min, max]$，能完美利用 INT8 的 256 個格子，適合 ReLU 後分佈。

### 2. NVIDIA 2:4 結構化稀疏 (Sparse Matrix)

- **核心機制**：每 4 個連續權重只存 2 個數值最大的，其餘強制為 0。
- **儲存方式**：**Data** (非零數值) + **Index** (位置索引)。
- **效益**：空間節省 50%，在硬體支援下（如 Ampere 架構）運算速度翻倍 (2x)。

### 3. 剪枝 (Pruning)

- **結構化剪枝 (Structured)**：砍掉整個 Channel，規則規整，硬體直接加速。
- **非結構化剪枝 (Unstructured)**：隨機變 0，矩陣稀疏不規整，僅省參數空間，不一定加速。
- **樂透彩票假說 (LTH)**：大網路裡藏著只要初始值對了就「中獎」的子網路。

---

## 四、 訓練優化與觀念複習 (是非/觀念題)

### 1. 知識蒸餾 (Knowledge Distillation)

- **目標**：大老師教小學生，補償瘦身後的準確度損失。
- **溫度 ($T$)**：$q_i = \frac{\exp(z_i / T)}{\sum \exp(z_j / T)}$。
- **物理意義**：$T$ 越大，分佈越平滑，能顯現類別間相似性的 **暗黑知識 (Dark Knowledge)**。

### 2. 訓練觀念 (Batch Size & Gradient Descent)

- **大 Batch Size**：梯度穩定、平行度高，但易落入鋒利局部最小值 (Sharp Minima)，泛化能力差。
- **小 Batch Size**：梯度帶隨機噪聲，助於跳出局部最小值，具正則化效果，但運算時間較長。
- **STE (Straight-Through Estimator)**：量化訓練時跳過不可導的 round 函數，讓梯度流回權重。

### 3. NAS 與 OFA

- **NAS (神經架構搜索)**：讓電腦自動搜索最佳模型結構（如 MobileNet V3）。
- **OFA (Once-for-All)**：**Train Once, Deploy Everywhere**。訓練一個包含子網路的超大網路。

---

### 考前提醒

1. **檢查 Stride ($S$)**：若題目 $S > 1$，輸出解析度會縮減，MACs 會大幅下降。
2. **是非題陷阱**：遇到「所有剪枝都能加速」或「量化絕對不掉準確率」通常是 **錯** 的。
3. **稀疏加速條件**：只有「結構化稀疏」且「硬體支援」才能達到實質推理加速。