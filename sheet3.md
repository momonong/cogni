# Edge AI (EAI) 開書考專用：精華重點筆記

## 一、 Efficient Model (高效模型) 核心觀念
* [cite_start]**定義**：在維持準確度前提下，極小化運算資源（MACs）、參數（Params）與能耗的模型 [cite: 143]。
* **必要性**：
    * [cite_start]**硬體限制**：邊緣設備記憶體（VRAM）有限，過大模型會導致記憶體溢位 (OOM) [cite: 87]。
    * [cite_start]**效能與能耗**：FP32 乘法能耗是 INT8 的 18.5 倍；面積佔用是 27 倍 [cite: 144, 145]。
* **CNN 效率特性**：
    * [cite_start]**稀疏連接 (Sparse Connectivity)**：卷積核尺寸遠小於輸入，僅與局部區域相連 [cite: 68, 69]。
    * [cite_start]**參數共享 (Parameter Sharing)**：同卷積核在整張圖滑動，共用一組權重 [cite: 70, 71]。

## 二、 訓練動態與最佳化 (是非/複選題區)
* **Batch Size 的權衡**：
    * [cite_start]**Small Batch**：梯度雜訊多，助於跳出局部最小值或「鞍點」，泛化能力（Generalization）較好 [cite: 85]。
    * [cite_start]**Large Batch**：梯度估計準，利於平行運算（Throughput），但易導致 OOM [cite: 87, 91]。
* [cite_start]**學習率連動**：當 Batch Size 增加（如 10 倍）時，學習率通常也應調大 [cite: 90, 91]。
* **激活函數優勢**：
    * [cite_start]**ReLU**：解決梯度消失問題（導數恆為 1），且計算效率高（無指數運算） [cite: 31, 32]。
    * [cite_start]**Softmax**：輸出轉換為機率分布（總和為 1），適合多類別分類 [cite: 34]。
* **正規化比較**：
    * [cite_start]**L1**：容易產生「稀疏權重 (Sparse Weights)」，具特徵選擇效果 [cite: 48]。
    * [cite_start]**L2 (Weight Decay)**：使權重均勻變小，但不為零 [cite: 47]。

## 三、 模型壓縮：量化 (Quantization)
* **Affine Quantization 計算**：
    * [cite_start]**公式**：$r = S(q - Z)$ 或 $q = r/S + Z$ [cite: 160]。
    * [cite_start]**Scale ($S$)**：$\frac{r_{max} - r_{min}}{2^b - 1}$（每一格代表的真實數值） [cite: 161]。
    * [cite_start]**Zero-point ($Z$)**：$round(q_{min} - \frac{r_{min}}{S})$（確保 0.0 精確對應整數） [cite: 163]。
* **量化流程對比**：
    * [cite_start]**PTQ (訓練後量化)**：不需重訓、不需原始訓練數據，資源有限時的首選 [cite: 147, 148]。
    * [cite_start]**QAT (量化感知訓練)**：在訓練中模擬量化雜訊，低位元（如 4-bit）下準確度最高 [cite: 148]。
* **進階量化技術**：
    * [cite_start]**KL Divergence (TensorRT)**：尋找最佳截斷閾值，使資訊損失（散度）最小 [cite: 172, 177]。
    * [cite_start]**Outlier Channel Splitting**：拆分極大值的通道，提高整體解析度 [cite: 180]。

## 四、 模型壓縮：剪枝 (Pruning)
* [cite_start]**剪枝流程**：偏好「迭代式剪枝 (Iterative)」，優於「一次性剪枝 (One-shot)」，能維持精度 [cite: 94, 95]。
* **剪枝粒度**：
    * [cite_start]**非結構化 (Unstructured/Fine-grained)**：壓縮率高，但記憶體存取不規律，硬體難加速 [cite: 102, 104]。
    * [cite_start]**結構化 (Structured/Coarse-grained)**：砍掉整個 Channel，規律性高，利於 GPU 加速 [cite: 100, 104]。
* **NVIDIA 2:4 結構化稀疏 (內部消息必看)**：
    * [cite_start]**定義**：每 4 個連續權重強留 2 個非零值 [cite: 113]。
    * [cite_start]**儲存**：存 2 個數值 + 2-bit Indices 紀錄原始位置 [cite: 114]。
    * [cite_start]**效益**：配合 Sparse Tensor Core 可達約 2 倍吞吐量 [cite: 115]。

## 五、 知識蒸餾 (Knowledge Distillation)
* [cite_start]**溫度 ($T$) 作用**：平滑化分布。$T > 1$ 能顯現類別間的「暗黑知識 (Dark Knowledge)」 [cite: 117, 118, 122]。
* [cite_start]**Dark Knowledge**：Soft Targets 中隱含的豐富資訊，如「這隻貓長得有點像狗」 [cite: 123]。
* [cite_start]**維度匹配**：使用「轉換層 (Regressor)」解決師生層數或通道數不同的問題（如 FitNets） [cite: 124, 125]。

## 六、 運算量 (MACs) 與硬體計算
* **標準卷積 MACs 公式**：
    * $H_{out} \times W_{out} \times C_{out} \times (K \times K \times C_{in})$
* **DSC (深度可分離卷積)**：分兩步運算：
    1. **Depthwise**：每通道獨立卷積（$K \times K \times C_{in}$）。
    2. **Pointwise**：$1 \times 1$ 卷積融合通道（$C_{in} \times C_{out}$）。
* [cite_start]**Bit-Serial 架構 (Stripes)**：逐位元處理，將精度從 16-bit 降至 8-bit 可線性提升 2 倍速度 [cite: 181, 183]。