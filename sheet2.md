# 嵌入式人工智慧與深度學習系統優化綜述報告：理論、架構設計與硬體協同

## 1. 機器學習基礎理論與優化 (Machine Learning Basics & Optimization)

本章節深入探討機器學習的核心數學基礎，涵蓋估計理論、偏差與變異數的權衡、以及深度學習優化演算法的數學原理。這部分是理解後續模型壓縮與高效能設計的基石。

### 1.1 學習演算法與估計理論

機器學習的本質是從經驗 $E$ 中學習，以在任務 $T$ 上提升性能度量 $P$ 1。在統計學習框架下，我們通常關注點估計 (Point Estimation)，即利用有限的獨立同分佈 (i.i.d.) 樣本 $x^{(1)}, \dots, x^{(m)}$ 來估計真實參數 $\theta$。

#### 1.1.1 估計量的性質

**偏差 (Bias)**：估計量的期望值與真實參數之間的差距。

$$Bias(\hat{\theta}_m) = \mathbb{E}(\hat{\theta}_m) - \theta$$

若 $Bias(\hat{\theta}_m) = 0$，則稱為不偏估計量 (Unbiased Estimator)。例如，伯努利分佈的樣本均值是不偏的；然而，常態分佈變異數的極大似然估計 (MLE) 是有偏的，其偏差為 $-\frac{1}{m}\sigma^2$，這解釋了為何樣本變異數需除以 $m-1$ (Bessel's correction) 1。

**一致性 (Consistency)**：隨著樣本數 $m \rightarrow \infty$，估計量是否依機率收斂於真實值。這比不偏性更為重要，因為即使是有偏估計量（如 MLE），若具備一致性，在大數據下仍是可靠的 1。

#### 1.1.2 偏差-變異數權衡 (Bias-Variance Tradeoff)

均方誤差 (MSE) 可分解為偏差與變異數的總和，這是模型選擇的核心依據：

$$MSE = \mathbb{E}[(\hat{\theta}_m - \theta)^2] = Bias(\hat{\theta}_m)^2 + Var(\hat{\theta}_m)$$

- **高偏差 (High Bias)**：模型容量 (Capacity) 不足，導致欠擬合 (Underfitting)。模型無法捕捉數據的複雜結構。
- **高變異數 (High Variance)**：模型容量過大，導致過擬合 (Overfitting)。模型記住了訓練數據中的隨機噪聲，導致泛化能力 (Generalization) 差 1。
- **最佳容量 (Optimal Capacity)**：應選擇在驗證集誤差最低點的模型複雜度。

### 1.2 極大似然估計 (MLE) 與 貝氏推論 (Bayesian Inference)

#### 1.2.1 極大似然估計 (MLE)

頻率學派認為參數 $\theta$ 是固定但未知的常數。MLE 試圖找到一組參數，使得觀測數據出現的機率最大化：

$$\theta_{ML} = \arg\max_{\theta} \sum_{i=1}^m \log p_{model}(x^{(i)}; \theta)$$

在深度學習中，最小化交叉熵 (Cross-Entropy) 等價於最大化似然函數。例如，對於高斯輸出分佈，MLE 等價於最小化均方誤差 (MSE) 1。

#### 1.2.2 貝氏估計 (Bayesian Estimation)

貝氏學派將 $\theta$ 視為隨機變數，並引入先驗分佈 (Prior Distribution) $p(\theta)$。

$$p(\theta | x) = \frac{p(x | \theta)p(\theta)}{p(x)}$$

**最大後驗估計 (MAP)**：

$$\theta_{MAP} = \arg\max_{\theta} \log p(x|\theta) + \log p(\theta)$$

這為正則化提供了理論解釋：

- **$L_2$ 正則化 (Weight Decay)**：等價於假設權重服從高斯先驗 ($w \sim N(0, \alpha^{-1}I)$)。這會迫使權重接近原點，抑制模型複雜度 1。
- **$L_1$ 正則化 (Lasso)**：等價於假設權重服從拉普拉斯先驗 (Laplace Prior)。這會導致權重稀疏化 (Sparsity)，即許多權重變為零，具有特徵選擇的效果 1。

### 1.3 優化演算法與梯度下降 (Optimization)

深度神經網路的損失函數通常是非凸的 (Non-convex)，充滿了局部極小值 (Local Minima) 和鞍點 (Saddle Points)。

#### 1.3.1 梯度下降法 (Gradient Descent)

利用泰勒展開 (Taylor Expansion) 近似損失函數，沿著負梯度方向更新參數：

$$w^{(t+1)} = w^{(t)} - \eta \nabla_w J(w)$$

其中 $\eta$ 為學習率 (Learning Rate)。

#### 1.3.2 批次大小 (Batch Size) 的影響 [重點考題]

批次大小是訓練中最關鍵的超參數之一，直接影響梯度的估計與模型的泛化能力。

**大批次 (Large Batch)**：
- **優點**：梯度估計準確，變異數小；能充分利用 GPU 的並行計算能力，提高 Throughput。
- **缺點**：容易收斂到尖銳極小值 (Sharp Minima)。在尖銳極小值處，損失函數曲率大，測試數據稍有偏移（由訓練與測試分佈的微小差異引起），Loss 就會劇烈上升，導致泛化能力差 2。

**小批次 (Small Batch)**：
- **優點**：梯度估計帶有較大的噪聲 (Noise)。這種噪聲具有正則化效果，能幫助模型跳出尖銳極小值，傾向於收斂到平坦極小值 (Flat Minima)。在平坦區域，參數的微小擾動不會導致 Loss 大幅變化，因此泛化能力較好 4。
- **缺點**：訓練時間可能較長（因為不能並行處理大量數據），且梯度震盪較大需要較小的學習率。

#### 1.3.3 進階優化器

**Momentum**：引入動量項 $v$，累積過去的梯度方向，有助於穿過平坦區域並抑制震盪。

$$v \leftarrow \alpha v - \epsilon \nabla J(\theta), \quad \theta \leftarrow \theta + v$$

**RMSProp**：解決 AdaGrad 學習率過早消失的問題。透過指數加權移動平均來計算梯度平方的累積，從而自適應地調整每個參數的學習率 1。

**Adam**：結合了 Momentum (一階矩) 和 RMSProp (二階矩) 的優點，並引入偏差修正 (Bias Correction) 以解決初期估計偏差問題。是目前最常用的優化器 1。

## 2. 深度神經網路架構 (Deep Neural Networks)

### 2.1 前饋網路原理

前饋網路 (Feedforward Networks/MLP) 旨在近似某個函數 $f^*$。與循環神經網路 (RNN) 不同，資訊流僅向單一方向傳遞，無反饋迴路 1。

#### 2.1.1 線性模型的局限性：XOR 問題 [模擬題考點]

線性模型 $y = w^Tx + b$ 只能解決線性可分問題。

**考點**：XOR（互斥或）問題在二維空間是線性不可分的。若強行使用線性模型優化 MSE，權重 $w$ 會收斂至 0，偏差 $b$ 收斂至 0.5，導致模型對所有輸入都預測 0.5，完全失效 1。

**解法**：引入隱藏層 (Hidden Layer) 與非線性激活函數 (Non-linear Activation)，將輸入映射到高維空間，使其線性可分。

### 2.2 激活函數 (Activation Functions)

- **Sigmoid / Tanh**：早期常用，但存在梯度消失 (Vanishing Gradient) 問題。當輸入值絕對值較大時，導數趨近於 0，導致深層網路無法訓練 1。

- **ReLU (Rectified Linear Unit)**：$g(z) = \max(0, z)$。
  - **優點**：解決了正區間的梯度消失問題；計算極快（僅需閾值判斷）；誘導稀疏激發 (Sparse Activation)。
  - **缺點**：Dead ReLU 問題（負區間梯度為 0，神經元可能永久死亡）。

- **Leaky ReLU / PReLU**：$g(z) = \max(0, z) + \alpha \min(0, z)$。給予負區間微小梯度 $\alpha$，解決 Dead ReLU 問題 1。

### 2.3 反向傳播 (Back-Propagation) 與 計算圖

反向傳播是基於微積分的連鎖律 (Chain Rule)。對於複合函數 $z = f(g(h(x)))$，其導數計算為：

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

在計算圖 (Computational Graph) 中，這對應於雅可比矩陣 (Jacobian Matrix) 的連乘。反向傳播利用動態規劃 (Dynamic Programming) 的思想，儲存中間節點的梯度，避免重複計算，是訓練深度網路的唯一高效途徑 1。

### 2.4 批次正規化 (Batch Normalization, BN)

**原理**：在每一層的激活函數之前，對輸入進行標準化（減均值，除標準差），使其分佈固定為 $N(0, 1)$，然後引入可學習的縮放參數 $\gamma$ 和平移參數 $\beta$ 以恢復表達能力。

$$\hat{x}^{(k)} = \frac{x^{(k)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y^{(k)} = \gamma \hat{x}^{(k)} + \beta$$

**訓練與推論的差異 [重點考題]**：
- **訓練時 (Training)**：使用當前 Batch 的均值 $\mu_B$ 和變異數 $\sigma_B^2$。
- **推論時 (Inference)**：使用訓練過程中累積的移動平均 (Running Average) $\mu_{running}$ 和 $\sigma_{running}^2$。这是因为推論時可能只有單一樣本，無法計算 Batch 統計量 1。

## 3. 智慧體育與姿態估計 (Smart Sport & Pose Estimation)

### 3.1 智慧體育與 Sabermetrics

智慧體育利用數據分析取代傳統的主觀判斷。

**Sabermetrics**：源於棒球數據分析（如電影《魔球》），強調使用經驗數據（如上壘率、長打率）而非傳統球探的直覺（如打擊姿勢、心理素質）來評估球員價值 1。

**技術應用**：包括動作捕捉 (Motion Capture)、虛擬實境 (VR) 訓練、以及基於 AI 的戰術分析。

### 3.2 姿態估計技術 (Pose Estimation) [重點考題]

姿態估計旨在從影像中定位人體關鍵點 (Keypoints)。

#### 3.2.1 Top-down vs. Bottom-up 方法比較

| 特性 | Top-down 方法 (由上而下) | Bottom-up 方法 (由下而上) |
|------|-------------------------|--------------------------|
| **流程** | 1. 物件偵測 (先找出所有人)<br>2. 單人姿態估計 (對每個人做關鍵點檢測) | 1. 關鍵點偵測 (找出圖中所有關鍵點)<br>2. 關鍵點分組 (Grouping/Association) |
| **代表模型** | RMPE (AlphaPose), HRNet, Mask R-CNN | OpenPose, PifPaf, DeepCut |
| **計算複雜度** | 與人數成正比 ($O(N)$)。人越多，推理時間越長。 | 與人數無關 ($O(1)$)。主要取決於圖像解析度。 |
| **準確度** | 通常較高。因為先進行了裁切 (Crop) 和歸一化，尺度較一致。 | 較受挑戰。尤其是關鍵點分組是 NP-hard 問題，容易連錯人。 |
| **缺點** | 依賴偵測器性能；在人群密集處若偵測失敗則無法補救；速度慢。 | 小物體或擁擠場景下，關鍵點容易丟失或誤連。 |
| **適用場景** | 少人、高精度需求（如單人健身教練、體育動作分析）。 | 多人擁擠、即時監控（如馬拉松、廣場人流分析）。 |

#### 3.2.2 回歸 (Regression) vs. 熱圖 (Heatmap) 方法

**回歸法 (Regression-based)**：網路直接輸出關鍵點的 $(x, y)$ 座標。
- **優點**：速度快，輸出即座標。
- **缺點**：高度非線性，難以學習空間泛化特徵，精度通常不如熱圖法 (DeepPose) 1。

**熱圖法 (Heatmap-based)**：網路輸出與輸入圖像同尺寸（或縮放後）的高斯熱圖 (Gaussian Heatmap)，每個像素值代表該點是關鍵點的機率。
- **優點**：保留了空間結構資訊，精度極高，是目前 SOTA 方法的主流 (Hourglass, CPM)。
- **缺點**：計算量大，且存在量化誤差 (Quantization Error)（受限於 Heatmap 解析度，取最大值時會有座標偏差）。

### 3.3 評估指標

- **PCK (Percentage of Correct Keypoints)**：在給定閾值內預測正確的關鍵點比例。
- **MPJPE (Mean Per Joint Position Error)**：預測點與真實點的平均歐式距離，常用於 3D 姿態估計 1。

## 4. 高效能模型指標與硬體 (Efficiency Metrics)

### 4.1 邊緣運算趨勢與硬體分級

AI 正從雲端遷移至邊緣 (Edge)，主要驅動力為：隱私 (Privacy)、低延遲 (Latency) 與 頻寬成本 (Cost) 1。

| 分級 | 運算能力 (Compute) | 記憶體 (Activation) | 儲存 (Weights) | 應用場景 |
|------|-------------------|-------------------|---------------|----------|
| **Cloud AI** | XX TOPS | 32 GB+ | TB/PB 級 | 數據中心訓練 |
| **Mobile AI** | X TOPS | 4 GB | 256 GB | 手機端推論 |
| **Tiny AI** | XX-XXX MOPS | 320 KB | 1 MB | MCU, IoT 感測器 |

**考點提示**：Tiny AI 的資源極度受限（如 SRAM 僅 320KB），這使得模型壓縮（量化、剪枝）成為部署的必要條件，而非選項 1。

### 4.2 運算量計算 (MACs Calculation) [必考模擬題]

MAC (Multiply-Accumulate) 是深度學習最基本的運算單元：$a \leftarrow a + (b \times c)$。通常 1 MAC $\approx$ 2 FLOPs。

#### 4.2.1 卷積層 MACs 公式

假設輸入特徵圖尺寸為 $H \times W \times C_{in}$，卷積核尺寸為 $K \times K$，輸出通道數為 $C_{out}$，輸出特徵圖尺寸為 $H_{out} \times W_{out}$。

**標準卷積 (Standard Conv2D)**：每個輸出像素需要 $K \times K \times C_{in}$ 次乘加運算。

$$MACs = H_{out} \times W_{out} \times C_{out} \times (K \times K \times C_{in})$$

**深度可分離卷積 (Depthwise Separable Conv)** (MobileNet 的核心)：分解為兩步：

1. **深度卷積 (Depthwise Conv)**：每個輸入通道獨立卷積。
   $$MACs_{dw} = H_{out} \times W_{out} \times C_{in} \times (K \times K)$$

2. **點卷積 (Pointwise Conv)**：$1 \times 1$ 卷積，融合通道資訊。
   $$MACs_{pw} = H_{out} \times W_{out} \times C_{out} \times C_{in}$$

**總 MACs** = $MACs_{dw} + MACs_{pw}$

#### 4.2.2 運算量比較 (Reduction Factor)

$$\frac{MACs_{separable}}{MACs_{standard}} = \frac{H_{out}W_{out}C_{in}K^2 + H_{out}W_{out}C_{in}C_{out}}{H_{out}W_{out}C_{in}C_{out}K^2} = \frac{1}{C_{out}} + \frac{1}{K^2}$$

當 $K=3$ 時，運算量約減少 8 到 9 倍。這是 MobileNet 高效的核心原因 1。

## 5. 模型剪枝 (Pruning)

### 5.1 剪枝原理與粒度 (Granularity)

剪枝靈感來自人腦突觸的發育：先過度生長 (2-4歲達頂峰 15,000 突觸/神經元)，再修剪至成人 (7,000 突觸/神經元) 1。

#### 5.1.1 剪枝粒度分類 [重點考題]

**非結構化剪枝 (Fine-grained / Unstructured)**：隨機修剪權重矩陣中的單個元素。
- **優點**：壓縮率最高，對模型精度傷害最小。
- **缺點**：產生稀疏矩陣 (Sparse Matrix)，需要專用硬體 (如 EIE) 才能加速。在一般 GPU 上因記憶體存取不連續，速度反而可能變慢。

**結構化剪枝 (Coarse-grained / Structured)**：修剪整個 Channel 或 Filter。
- **優點**：直接減少矩陣維度，無需特殊硬體即可在 CPU/GPU 上加速。
- **缺點**：精度損失較大，靈活性低。

**模式剪枝 (Pattern-based) - NVIDIA 2:4 Sparsity [必考]**：這是 NVIDIA Ampere 架構引入的半結構化稀疏。

- **定義**：在每 4 個連續的權重元素中，強制要求至少 2 個為零 (50% 稀疏度) 6。
- **儲存格式 (Compressed Format)**：
  - **Data**：僅儲存 2 個非零值 (FP16/INT8)。
  - **Indices (Metadata)**：需要額外的索引來記錄非零值的位置。通常每 4 個元素需要 2-bit indices 來標記非零位置 7。
- **優點**：結合了結構化的硬體效率 (Tensor Core 支援 2 倍吞吐量) 與非結構化的靈活性。

### 5.2 彩票假設 (Lottery Ticket Hypothesis)

**定義**：一個隨機初始化的密集網路包含一個子網路 (Winning Ticket)，若將該子網路單獨取出並使用原始初始化權重進行訓練，可以達到與原網路相當的精度 1。

**權重回溯 (Rewinding)**：研究發現，將權重重置為訓練早期 (如 epoch k) 的狀態，比重置為初始狀態 (epoch 0) 更能穩定地找到彩票。

## 6. 知識蒸餾 (Knowledge Distillation)

### 6.1 概念與溫度參數 (Temperature)

知識蒸餾旨在將大模型 (Teacher) 的知識轉移給小模型 (Student)。核心在於利用 Teacher 的 Soft Targets（軟標籤）1。

#### 6.1.1 溫度 $T$ 的作用 [模擬題考點]

Softmax 輸出機率為 $q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$。

- **$T=1$**：標準 Softmax，輸出通常接近 One-hot（某類別機率極高，其他接近 0）。
- **$T > 1$**：軟化 (Soften) 機率分佈。這會放大非正確類別的機率值，揭示了 Teacher 模型對類別間相似度的判斷（例如：圖片是「狗」，但像「貓」的機率遠高於像「車」）。這被稱為暗知識 (Dark Knowledge)。
- **極限情況**：當 $T \rightarrow \infty$ 時，Softmax 輸出趨近於均勻分佈 ($1/N$)。

### 6.2 損失函數

$$L_{KD} = \alpha L_{CE}(y_{true}, P_{student}) + \beta T^2 L_{KL}(P_{teacher}, P_{student})$$

**注意係數 $T^2$**：因為在求導時，Soft Targets 的梯度會縮小約 $1/T^2$ 倍，因此需要乘上 $T^2$ 來保持梯度量級一致，避免蒸餾項的影響力消失 1。

### 6.3 蒸餾策略

**自蒸餾 (Self-Distillation / Born-Again NN)**：Student 結構與 Teacher 相同。第一代 Student 訓練完後成為 Teacher 教導第二代，反覆迭代可提升精度。

**線上蒸餾 (Deep Mutual Learning)**：多個 Student 模型同時從頭訓練，並互相學習 (最小化彼此的 KL 散度)，無需預訓練的 Teacher。

## 7. 量化 (Quantization)

### 7.1 數值表示法：浮點數 vs. 定點數

**IEEE 754 浮點數 (FP32)**：由 Sign (1 bit), Exponent (8 bits), Mantissa (23 bits) 組成。數值 = $(-1)^S \times (1.M) \times 2^{E - Bias}$。Bias = 127 (對於 FP32)。

**定點數 (Fixed Point / INT8)**：$r = S \times (q - Z)$
- $r$: 真實浮點數
- $S$: Scale (FP32)
- $q$: 量化整數
- $Z$: Zero-point (整數)

### 7.2 量化方法分類 [比較題考點]

#### 7.2.1 對稱 vs. 非對稱量化

**對稱量化 (Symmetric)**：強制 $Z=0$。量化範圍對稱於 0 (如 $[-127, 127]$)。
- **優點**：計算簡單 ($r = S \times q$)，無需處理零點偏移，硬體效率高。
- **適用**：權重 (Weights) 分佈通常以 0 為中心，適合對稱量化。

**非對稱量化 (Asymmetric)**：$Z \neq 0$。量化範圍可以是任意區間。
- **優點**：能充分利用位元寬度來表示非對稱分佈的數據。
- **適用**：ReLU 後的激活值 (Activations) 全為非負，適合非對稱量化 (映射到 uint8)。

#### 7.2.2 QAT vs. PTQ

**PTQ (Post-Training Quantization)**：模型訓練完後直接進行量化。
- 需要校準 (Calibration)：使用少量數據來決定最佳的 Scale 和 Clipping 閾值 (如使用 KL 散度最小化)。
- **缺點**：低位元 (如 INT4) 精度損失大。

**QAT (Quantization-Aware Training)**：在訓練過程中模擬量化。
- **Fake Quantization**：前向傳播時模擬量化帶來的精度損失 ($x \rightarrow q \rightarrow \hat{x}$)。
- **STE (Straight-Through Estimator) [關鍵考點]**：由於量化函數 (Round) 不可導，反向傳播時假設導數為 1 (即直接將梯度穿過量化層)，以更新 FP32 的權重。這能讓模型適應量化雜訊，精度最高 1。

### 7.3 截斷技術 (Clipping)

直接使用最大/最小值 ($Min-Max$) 進行量化容易受離群值 (Outliers) 影響，導致 Scale 過大，降低整體解析度。

- **ACIQ**：假設權重服從高斯或拉普拉斯分佈，利用統計理論計算最佳截斷值以最小化 MSE。
- **KL Divergence (TensorRT)**：尋找一個閾值，使得截斷後的量化分佈與原始分佈之間的 KL 散度 (資訊損失) 最小 1。

## 8. 高效能模型架構 (Efficient Models)

### 8.1 SqueezeNet

**Fire Module**：包含 Squeeze 層 (全 $1\times1$ 卷積) 和 Expand 層 ($1\times1$ 與 $3\times3$ 混合)。

**策略**：利用 $1\times1$ 卷積壓縮輸入 Channel 數，從而減少後續 $3\times3$ 卷積的參數與運算量 1。

### 8.2 MobileNet

**核心**：深度可分離卷積 (Depthwise Separable Convolution) (如前所述，減少約 9 倍運算量)。

**MobileNetV2**：引入 Inverted Residuals (兩頭窄中間寬) 與 Linear Bottlenecks (最後輸出不加 ReLU，保留特徵資訊) 1。

### 8.3 ShuffleNet

**問題**：在極致輕量化網路中，$1\times1$ Pointwise Conv 佔據了大部分運算量。若改用 Group Convolution 雖可減少運算，但會導致通道間資訊不流通。

**解法**：Channel Shuffle (通道重排)。將 Group Convolution 的輸出通道均勻打亂，使不同 Group 的資訊能夠交換 1。

---

## 附錄：模擬試題與解析 (Simulated Exam)

### Q1 (觀念): 為什麼在知識蒸餾中，計算 Loss 時需要乘以 $T^2$？

**答**：Softmax 對 logits 進行縮放 ($z_i/T$)。在反向傳播求導時，梯度的量級會縮小約 $1/T^2$。為了讓蒸餾 Loss (Soft Target) 與分類 Loss (Hard Target) 的梯度保持在同一量級，避免蒸餾信號過小，因此需乘以 $T^2$ 進行補償。

### Q2 (計算): 假設輸入特徵圖 $112 \times 112 \times 64$，輸出通道 128，Kernel $3 \times 3$。請計算使用標準卷積與深度可分離卷積的 MACs 差異。

**答**：
- **標準卷積**：$112 \times 112 \times 128 \times 64 \times 3 \times 3 \approx 924 \text{ M MACs}$。
- **深度可分離**：
  - DW: $112 \times 112 \times 64 \times 3 \times 3 \approx 7.2 \text{ M}$
  - PW: $112 \times 112 \times 128 \times 64 \approx 102.7 \text{ M}$
  - 總和 $\approx 110 \text{ M MACs}$。
- **結論**：約減少 8.4 倍。

### Q3 (NVIDIA 量化): 請描述 NVIDIA Ampere 架構中 2:4 Sparsity 的儲存格式。

**答**：對於每 4 個連續權重，只儲存 2 個非零數值 (Data)。同時，需要額外的 Metadata (Indices) 來記錄這兩個數值在原始 4 個位置中的哪裡。通常使用 2-bit indices 來編碼每個非零值的位置。

### Q4 (優化): 大批次 (Large Batch) 訓練通常會導致什麼樣的泛化問題？為什麼？

**答**：大批次訓練容易收斂到尖銳極小值 (Sharp Minima)。在這些區域，損失函數的曲率很大，測試數據分佈的微小偏移就會導致 Loss 大幅上升，因此泛化能力較差。相反，小批次帶來的梯度噪聲有助於模型收斂到平坦極小值。

### Q5 (姿態估計): 比較 Top-down 與 Bottom-up 方法在計算複雜度上的差異。

**答**：Top-down 方法的計算量與人數成正比 ($O(N)$)，因為需要對每個檢測到的人執行一次姿態估計。Bottom-up 方法的計算量主要取決於圖像解析度 ($O(1)$ w.r.t 人數)，只執行一次全圖關鍵點檢測，因此在多人擁擠場景下 Bottom-up 效率較高。