# 論文現況整理 v6（完整版）

> **目標：ACL ARR 2026/12 cycle**
> v6 相對 v5 的主要更新：**接上古典 ensemble 文獻**、**RQ 回到原版**、**修正 v5 的錯誤宣稱**、**新增 recovery_blind 與難度分層**。
>
> **標記**
> ✅ 我實際重算驗證過　⚠️ 你的文件宣稱但我無法重現　❓ 尚未查證，投稿前必須確認

---

# 目錄

0. [一頁摘要](#0-一頁摘要)
1. [最初的困境](#1-最初的困境)
2. [現有實驗結果盤點](#2-現有實驗結果盤點)
3. [建議的方向](#3-建議的方向)
4. [完整邏輯流程](#4-完整邏輯流程)
5. [待辦清單與投稿機率](#5-待辦清單與投稿機率)
6. [完整文獻表](#6-完整文獻表)

---

# 0. 一頁摘要

## 核心恆等式

```
Gain = (1−d) × Δ_agree  +  d × (c − m) × recovery

d        = 候選答案不全相同的比例（聚合器被觸發的頻率）
c        = 分歧題中至少一個候選正確的比例（oracle）
m        = 分歧題中正確候選的平均比例（基線）
recovery = (acc_agg − m) / (c − m)，聚合器吃到幾成
Δ_agree  = 聚合器在候選一致題目上造成的淨變化（觸發式聚合器 = 0）
```

K=2、單一正解、觸發式時退化成 `Gain = d × (c/2) × recovery`。

## ⚠️ 最重要的一件事：這不是新的

**你的 `Gain`（相對候選平均）就是 Wood et al. (JMLR 2023) 的 ambiguity-effect。** 古典 ensemble 文獻三十年前就有這條分解。

**你真正的立足點是三個結構差異：**

| | 古典 ensemble | LLM 推理 pipeline |
|---|---|---|
| ① 聚合器什麼時候動作 | 每一題都動作 | **只在分歧時觸發** → `d` 從指標變成**成本變數** |
| ② 候選帶什麼 | 只有標籤或機率 | **完整推理過程** → 聚合器可以「讀論證」 |
| ③ 聚合器是誰 | 獨立規則或 meta-model | **和生成者同一個模型**（自我聚合） |

**① 是 `d × (c−m)` 拆分唯一真正說得過去的理由。** 古典文獻把兩者合併成 `Oracle − Average`，因為 combiner 永遠動作，沒有理由拆。

## 三個 RQ

```
RQ1  任何「產生多條候選 → 聚合」的推理方法，其增益是否都能拆成 d × (c−m) × recovery？
     → 數學上已解決，降級為 §3 的 Proposition + 四個假設的實測驗證

RQ2（headline）
     不同的多樣性來源（語言 / 採樣 / 推理強度 / 角色 / 改寫），
     在控制題目組成之後，是否給出相同的 c 與 recovery？

RQ3  取得多樣性的代價是什麼？語言是不是最貴的一種？
     判準：recovery > Δm / Δh
```

## 已有的關鍵數字

| | 數值 | 來源 |
|---|---|---|
| 跨語言辯論的 recovery | 0.34 | 你的資料 |
| 同語言 EN / ZH 辯論的 recovery | 0.322 / 0.346 | legacy |
| Gao et al. 反推的 recovery（K=4 多數決） | 0.329–0.415，平均 0.357 | 外部 |
| recovery 的跨格異質性 I² | **14.8%**（對比 excess 的 96%） | legacy |
| Gain 變異的來源 | 生成端 86%、聚合端 39% | legacy |

---

# 1. 最初的困境

## 1.1 原始論文

**Iterative Multilingual Self-Reflection (IMSR)**：同一個 LLM 扮演兩個語言的 agent，答案不一致時進行最多 3 輪跨語言辯論，仍不一致則由 Judgement Agent 裁決。用 XLM-RoBERTa 做 per-query 的語言對路由（10 個組合，來自 EN/ZH/JA/RU/ES）。4 模型 × 4 資料集 = 16 格。

## 1.2 審稿結果

| | Soundness | Excitement | Overall | Confidence |
|---|---|---|---|---|
| R1 | 2.5 | 2.5 | 2.5（Borderline Findings） | 4 |
| R2 | 2.0 | 2.5 | 2.0（Resubmit next cycle） | 4 |
| R3 | 2.0 | 1.5 | 1.5（Resubmit after next cycle） | 5 |

## 1.3 批評歸類

**A. 因果識別不足**
- R1-W1：沒有同語言雙 agent 控制組。增益可能來自「兩個 agent」而非「語言」
- R3：EN-SR / ZH-SR 的迭代輪數是否與 IMSR 對齊？若不對齊，增益可能只是測試時計算量增加

**B. Novelty 不足**
- R2：只是 cross-lingual prompting + self-reflection + debate + routing 的組合
- R2：沒有討論 AutoCAP、mGRPO
- R3：只有 17 篇引用；§2.4 完全沒有引用

**C. 統計不透明**
- R3：p < 0.05 怎麼算的？跑幾次？報平均還是最大值？和誰比？

**D. 設計質疑**
- R1-W2/R2：五種語言的選擇沒有理據
- R1-W3/R2：router 的泛化範圍未討論
- R3：辯論 prompt 是混語，已知不穩定
- R3：為什麼 best fixed pair 在不同模型上不同？

**E. 可重現性**：三位都給 Datasets / Software 最低分

## 1.4 教授的兩個批評

**① 「如果只影響幾篇現有 Paper，那會沒什麼 value。」**

成立。逐篇盤點後，原方案真正咬到的只有 6 篇，全在跨語言 CoT：AutoCAP、Gao et al.、mGRPO、CLP、XLT、Cross-ToT。原因是 RQ 的自變數是「語言」和「語言能力落差」，別人的 pipeline 沒有這兩個變數。

**② 「這個命題跟 ML 的 ensemble 研究很類似。」**

也成立，而且比表面更精確。見 §2.6 的對照字典。

---

# 2. 現有實驗結果盤點

## 2.1 資料集 A｜跨語言配對網格

**規模**：4 模型 × 4 資料集 × 5 語言 → 16 格 × 10 語言對 = 160 觀測
**n**：CSQA/MathQA/MMLU 各 2000 題；TruthfulQA 817 題 ✅

### A-1｜跨語言配對優於最強單語

| 量 | 數值 | 驗證 |
|---|---|---|
| 16 格中 excess > 0 | **15/16** | ✅ |
| 符號檢定 p | **5.19e-4** | ✅ |
| 平均 excess | **+1.518pp**（你的文件寫 +1.46）⚠️ | ✅ |
| 160 對中 excess > 0 | 86.9% | ✅ |

### A-2｜「侵蝕」是基準選擇的算術後果

恆等式 `excess = delta − gap/2`（160 列驗證，誤差 2.4e-14）✅

| 迴歸 | 斜率 | t |
|---|---|---|
| excess ~ gap（pooled OLS） | **−0.383** | −10.35 |
| excess ~ gap（cell FE） | −0.373 | −15.99 |
| excess ~ gap（16 格逐格平均） | −0.412 | 16/16 為負 ← 你文件標成「pooled OLS」的是這個 ⚠️ |
| **delta ~ gap（pooled OLS）** | **+0.117** | +3.17 |

**160/160 個語言對的 delta > 0，平均 +2.73pp** ✅

| gap 分箱中心（pp） | 0.4 | 1.4 | 2.4 | 3.5 | 4.7 | 6.8 | 10.4 |
|---|---|---|---|---|---|---|---|
| 平均 excess | 2.24 | 1.86 | 1.45 | 1.57 | 0.83 | −0.57 | −1.60 |
| 平均 delta | 2.46 | 2.57 | 2.67 | 3.30 | 3.18 | 2.83 | 3.59 |

### A-3｜兩平點的機制意義

`excess = 0 ⟺ Gain = gap/2 ⟺ gap* = 2 × Gain = d × c × recovery`

delta 平均 2.73pp → gap\* ≈ 5.46pp；用 `d≈0.21, c≈0.76, recovery=0.34` 代入 → **5.43pp** ✅

per-cell gap\* 範圍 0.85–19.54pp（相差 23 倍）→ **不是常數** ✅

### A-4｜異質性

| 量 | 數值 |
|---|---|
| a_excess 觀測 SD / 格內平均 SE | 1.147pp / 0.338pp ✅ |
| Cochran Q（df=15） | **372.9**，p = 3.4e-70 ✅ |
| **I²** | **96.0%** ✅ |
| 真實格間 SD τ | 1.288pp ✅ |
| 95% 預測區間 | 我算 [−0.43, +5.24]（你寫 [−0.77, +4.94]）⚠️ 都含 0 |

### A-5｜識別力診斷（λ）

`λ = 1 − Var(gap 雜訊)/Var(觀測 gap)`，**8/16 格 λ ≤ 0** ✅

例：DeepSeek/MathQA 五語言準確率 90.45–91.00（全距 0.55pp），但單一準確率的 SE = 0.64pp。**全距比單點誤差還小。**

可用的 8 格：raw g = −0.340 → EIV 校正後 **−0.243**

### A-6｜gap 與 max 在格內共線

| 量 | 數值 |
|---|---|
| 格內 corr(max_mono, gap) 平均 | **0.583**（15/16 為正，最高 0.994）✅ |
| 格內 corr(mean_mono, gap) 平均 | **0.185** ✅ |
| 英文錨點區塊內斜率 | −0.073（7/11 為負，不顯著）✅ |

GPT-4o mini/CSQA 英文錨點：gap 從 6.45 拉到 11.30，excess 幾乎不動（−1.30/−1.95/−1.40/−1.80）✅

### A-7｜Oldham 等變異前提檢查

殘餘偏誤 = `(σ²_min − σ²_max) / (2·Var(gap))`
**在 λ > 0 的可識別格子裡只有 +0.002 ~ +0.012** ✅ — 不影響結論。

### A-8｜LOO 交叉驗證（你的資料，我未重跑）

```
LOO MAE = 1.033pp（常數 baseline 1.311pp，改善 21.2%）
LOO R²_oos = 0.343；cell-cluster bootstrap 95% CI = [−0.109, 0.567]
```

⚠️ pair 層級 Fisher p = 4e-12，**cell 層級 Fisher p = 0.106（不顯著）** → 統一用 cell 層級。

### A-9｜虧損的結構

實質虧損（excess < −1.0pp）13 個 → **全部來自 3 個 cell，全部是 CommonsenseQA，12/13 含英文**。Gemini 零虧損。

---

## 2.2 資料集 B｜同語言辯論（legacy）

`legacy_em_samelang.csv`，EN-EN 與 ZH-ZH，3 模型 × 4 資料集 = 24 格
**⚠️ GPT 4.1 mini 不是 4o mini；無 Gemini；n=3000–6000；T=1**

### B-1｜恆等式驗證

`accuracy = (1−d)·a_agree + d·acc_debate`，**24/24 誤差 1.1e-16** ✅
`Gain = d(c/2)·recovery`，**24/24 誤差 < 1e-4** ✅

⚠️ 但這代表計分程式**預設** Δ_agree = 0，資料本身測不到違反。逐題驗證待 0A-2。

### B-2｜三個量的配對比較

| 量 | EN | ZH | 配對差異 | p | 判定 |
|---|---|---|---|---|---|
| **d** | 0.075 | 0.132 | −0.057 | **0.001** | ✅ 顯著 |
| **c** | 0.760 | 0.748 | +0.012 | 0.424 | ✗ |
| **recovery** | 0.322 | 0.346 | −0.024 | 0.486 | ✗ |

全部我重算驗證 ✅ **只有 d 會動。**

### B-3｜檢定力（12 格配對）

| 量 | MDE（80% power, α=.05） |
|---|---|
| d | ±0.039 |
| c | ±0.042 |
| **recovery** | **±0.097** |

✅ 硬約束：**只能宣稱「±0.10 內無法區分」，必須用 TOST。**

### B-4｜總 headroom

| 條件 | d | c | H = d(c/2) | recovery | Gain |
|---|---|---|---|---|---|
| 同語言 EN | 0.075 | 0.760 | **0.0282** | 0.327 | 0.93pp |
| 同語言 ZH | 0.132 | 0.748 | **0.0494** | 0.351 | 1.63pp |
| 跨語言（推估） | ~0.211 | — | **0.0803** | ~0.34 | 2.73pp |

**跨語言的 H 是同語言 EN 的 2.8 倍，但 recovery 三者相同，c 幾乎不變 → H 的差異幾乎全部來自 d。** ✅

### B-5｜recovery 的異質性 ← 新

| | recovery | excess（對照） |
|---|---|---|
| 觀測 SD | 0.1095 | 1.147pp |
| 格內平均 SE | 0.1011 | 0.338pp |
| **真實格間 SD τ** | **0.042** | **1.288pp** |
| **I²** | **14.8%** | **96.0%** |
| Cochran Q | 59.2（df=23），p=5e-5 | 372.9，p=3.4e-70 |

**recovery 的變異 85% 是量測雜訊，excess 的變異 96% 是真實格間差異。**

→ 生成端跨 setting 差異巨大，聚合端相對穩定。

⚠️ τ = 0.042 是兩個相近數字相減開根號得到，24 個點上估計誤差很大，需要 bootstrap。

### B-6｜變異數分解 ← 新

```
Var(log Gain)     = 0.265
Var(log H)        = 0.229   ← 生成端佔 86%
Var(log recovery) = 0.104   ← 聚合端佔 39%
2·Cov             = −0.068  ← 輕微負相關，互相抵銷
```

### B-7｜⚠️ 一個被推翻的假說

我曾假設「把 Gain 拆成三個因子會讓 diversity 與效能的關係變乾淨」。**在這份資料上不成立：**

| 預測 Gain 的變數 | r | r² |
|---|---|---|
| d（只看分歧率） | +0.737 | 54% |
| c | −0.184 | 3% |
| H = d(c−m) | +0.723 | 52% |

**H 沒有比 d 好。** 因為 c 在這裡幾乎是常數，所以 `H ≈ d × 0.38`。

**真正的抵銷機制是 `gap/2`，不是 `(c−m)`：**

```
同語言（gap≈0）   d 與 Gain 正相關 +0.737
跨語言（gap 變動）  gap 與 excess 負相關，斜率 −0.383
```

→ **「diversity 有沒有用」取決於基準選擇。** 這是可以寫進論文的版本。

---

## 2.3 資料集 C｜單語改寫與推理強度（部分完成）

**已跑**：english_cot / short_cot / direct。**⚠️ DeepSeek 有幾格因 bug 缺失。**

| 資料集 | 改寫效果 | 等價檢定（±3pp） | direct 掉幅 | 階梯可行 |
|---|---|---|---|---|
| MMLU | −1.85/−1.30/−0.35/0.00 | ✅ | 4–6.9pp | ✅ |
| MathQA | −2.55/+0.60/+0.65 | ✅ 大致 | 30–55pp | ✅ |
| CommonSenseQA | −2.85/−3.55/−3.70 | ❌ 難度掉 3pp | 0.8–1.5pp | ❌ |
| TruthfulQA | +2.32/+2.57/−1.35 | ⚠️ 方向不一致 | 方向相反 | ❌ |

**🔴 A-9 的虧損現象全在 CommonsenseQA，而 CSQA 正好是階梯做不了的資料集。**
**解法**：改用 Shi et al.（ICML 2023）的分級干擾脈絡注入。

### ⚠️ 混合策略的數學問題

逐題按 p 混合 full-CoT 與 direct 時，**所有聚合量對 p 精確線性**：

```
excess(p) = (1−p)·excess(cot) + p·excess(direct)
```

**斜率恆等於兩端點連線，中間的 p 不提供新資訊，也無法檢測非線性。** 而要驗證的「階梯 vs 線性」正是曲率問題。→ 改用 token 預算截斷或多級劣化。

### 另外發現

short_cot ≈ full_cot（差 < 1pp）。**CoT 長度是二元開關不是斜坡。**

---

## 2.4 資料集 D｜Gao et al. 表格反推（零成本外部驗證）

**來源**：arXiv 2504.11833 Table 1（Acc̄ = m、Acc@4 = c）+ Table 2（Vote@4 = acc_agg）

**為什麼 d 不需要知道**：三個量都是非條件的，`(Vote−Acc̄)/(Acc@k−Acc̄)` 中 d 會約掉 ✅

| 模型 | 組合 | m | c | Vote@4 | **recovery** |
|---|---|---|---|---|---|
| Qwen2.5-72B | Best | 43.7 | 74.3 | 54.2 | **0.343** |
| Qwen2.5-72B | Random | 41.5 | 70.0 | 51.7 | **0.358** |
| LLaMA3.1-70B | Best | 38.0 | 73.9 | 49.8 | **0.329** |
| LLaMA3.1-70B | Random | 36.9 | 70.2 | 48.8 | **0.357** |
| R1-Distill-70B | Best | 51.6 | 80.1 | 61.2 | **0.337** |
| R1-Distill-70B | Random | 49.0 | 75.5 | 60.0 | **0.415** |

**平均 0.357，SD 0.031** ✅。用單一常數 1/3 反推 Vote@4，六個預測五個誤差在 1 點以內 ✅

### 解開他們的矛盾

```
Multilingual: m=41.5, c=70.0, headroom=28.5, Gain = 10.2 → Vote 51.7
Repeat:       c=65.9, Vote=53.6 → 反推 m ≈ 47.5, headroom ≈ 18.4, Gain = 6.2
```

**Multilingual 的增益大得多（+10.2 vs +6.2），輸掉是因為基線低了 6 點。**
兩平條件：`recovery > Δm/Δh = 6.0/10.1 ≈ 0.60`，現有聚合器全在 0.33–0.42。

⚠️ Repeat 的 m 是反推不是實測，需用開源資料確認。

---

## 2.5 recovery_blind：一個免費的量

K=2、單一正解時有閉式解。令 `w_A` = 可救的分歧題中錨點是對的那一方的比例：

$$\texttt{recovery}_{\text{blind}} = 2w_A - 1 = \frac{\texttt{gap}}{d \times c}$$

**可以是負的**（若搭檔在分歧題上比錨點常對）。

**三個用途**：
1. **真正的部署基準**——聚合器要值得用，必須贏過「直接用最強的」
2. **組成差異的免費診斷**——高 recovery_blind 代表 D 裡混了很多「錨點明顯正確」的簡單題
3. **統一恆等式**：`excess = d(c/2)(recovery − recovery_blind)`，所以 `excess > 0 ⟺ recovery > recovery_blind`

**衍生量**：`skill = (recovery − recovery_blind)/(1 − recovery_blind)`

### ⚠️ 各軸的不對稱程度差很多

| 軸 | 預期 recovery_blind |
|---|---|
| 推理強度（full CoT vs direct） | 接近 +1 |
| 語言（英文 vs 日文） | 明顯 > 0 |
| 角色 | 略 > 0 |
| 採樣（T=0 vs T=0.7） | ≈ 0 |

**在極度不對稱的軸上，judge 只要「猜哪個是完整 CoT」就能拿高分，recovery 會虛高。**
→ **每個軸必須同時報 recovery 與 recovery_blind。**

### 從 design matrix 的推估

| 分組 | n | gap 平均 | recovery_blind 預估 | excess 平均 | excess<0 |
|---|---|---|---|---|---|
| 含英文 | 64 | 3.77pp | ≈ 0.24 | +1.20pp | 16/64 |
| 不含英文 | 96 | 1.53pp | ≈ 0.12 | +1.73pp | 4/96 |

---

## 2.6 ⭐ 與古典 ensemble 的對照字典

| 你的框架 | Ensemble 文獻的名字 |
|---|---|
| K 條推理路徑 | m 個 ensemble members |
| 每一題 | 一個 instance / test point |
| 聚合器 | combiner |
| **Gain（相對候選平均）** | **ambiguity-effect / diversity-effect** |
| **c**（至少一個正確） | **oracle accuracy**（Kuncheva 標準術語） |
| **m** | average member accuracy |
| **d** | disagreement measure（K&W 十個指標之一） |
| 救援率 − 破壞率 | **good / bad diversity** |
| excess（相對最強單語） | ensemble vs best individual |
| pilot / holdout | training / validation split、**stacking** |
| per-query router | **Dynamic Ensemble Selection (DES)** |

**重要化簡**：因為一致題目上 `c_i = m_i`，所以

```
d × (c − m) = E[c_i − m_i] = Oracle 準確率 − 平均準確率
```

於是 `recovery = Gain / (Oracle − Average)`。**這是一個很自然的比值，必須自己搜尋有沒有既有名稱。** ❓

---

## 2.7 佐證與反對的文獻對照

| 你的結果 | 佐證 | 該 paper 的論點 | 反對 / 需區分 | 該 paper 的論點 |
|---|---|---|---|---|
| Gain 分解 | **Wood et al. (JMLR 2023)** | ensemble 風險 = 平均個體風險 − ambiguity-effect，適用任何損失 | **Wood et al.（同篇）** | 🔴 **你的 Gain 就是他們的 ambiguity-effect，不能宣稱新** |
| 0/1 loss 的分解形式 | **Wood et al. Thm 10** | 對 0/1 loss、任何 combiner，該差值必然依賴標籤 | — | 你用 c、m（含正解）不是缺點，是唯一可能 |
| 救援 / 破壞分解 | **Brown & Kuncheva (2010)** | diversity 在 ensemble 答對時減少錯誤（good）、答錯時增加（bad） | — | 🔴 **已經有名字，直接沿用術語** |
| A-1 跨語言優於最強單語 | Gao et al. 2504.11833 | 多語言 Acc@k 上界比英語高近 10 點 | Gao et al.（同篇） | Vote@k 沒有優勢；歸因於 judge 語言偏誤 |
| A-2「侵蝕」是算術 | Oldham (1962)、Blomqvist (1977) | 差值對基線迴歸會因共用誤差產生假負斜率 | — | — |
| A-2 excess vs delta | **Krogh & Vedelsby (1994)** | ensemble 誤差 ≤ **個體平均**誤差（不保證贏最強個體） | — | 🟠 1994 年就存在的問題 |
| A-2 max 基準偏誤 | Dodge et al. (EMNLP 2019) | expected validation performance 修正 best-of-K 的高估 | — | — |
| B-7 diversity 關聯薄弱 | **Kuncheva & Whitaker (2003)** | 十個 diversity 指標與 ensemble 效能關聯都很弱 | — | 🔴 **RQ2 的前車之鑑** |
| 實驗設計需控制能力 | **Wood et al. Figure 14** | 只在單獨改變 m 時 diversity 乾淨預測效能（r²=0.99）；改變個體能力後崩到 0.59 | — | 🔴 **你的五個軸都會同時改變能力** |
| A-4 兩平點非常數 | IntHout et al. (2016)、Higgins et al. (2003) | 高異質性時應報預測區間 | — | — |
| A-5 λ≤0 | Card et al. (EMNLP 2020) | NLP 實驗普遍檢定力不足 | — | — |
| A-6/A-9 英文錨定虧損 | Wendler et al. (ACL 2024) | 模型在英語為中心的概念空間運算 | Chua et al. (2024) | 知識有跨語言屏障，某些任務語言應真能提高 c |
| B-2 只有 d 會動 | Gao et al. | 多語言優勢在 Acc@k 而非 Vote@k | Gao et al. | 但歸因於語言偏誤而非 recovery |
| B-4 recovery ≈ 0.34 | **Gao et al.（反推）** | 0.329–0.415 | Song et al. (ICLR 2025) | relative GV-Gap 隨 flops 上升（但分母不同） |
| recovery 可被推動 | Song et al. Fig.3 | 固定 generator、換大 verifier，gap 3.39→24.91 | — | 🟠 **反對「普世不變量」** |
| recovery 可被推動 | Weaver (NeurIPS 2025) | 加權集成弱 verifier，Llama 3.3 70B 達 o3-mini 級 | — | 但用**外部** verifier，支持「自我聚合被鎖死」 |
| per-query router 難做 | Cruz et al. (2018) DES 綜述 | 動態選擇的實際效能離 oracle 上界很遠 | — | 你的 router 在古典文獻叫 DES，落差研究二十年 |
| 聚合器達不到 oracle | Brown et al. (2024) | coverage 冪次成長至 95%+，但多數決與 RM 在幾百樣本後飽和 | — | — |
| MAD 沒有特殊性 | Smit et al. (ICML 2024) | 對齊 prompt 與預算後，MAD 贏不過單一 well-prompted agent | — | — |
| recovery 不高 | Huang et al. (ICLR 2024) | 內在自我修正不可靠；response 數對齊時 MAD 輸給 SC | — | — |
| C 改寫在 CSQA/TQA 失敗 | Sprague et al. (2025) | CoT 主要在數學與符號推理有效 | — | — |
| C 改寫移動準確率 | Mirzadeh et al. (ICLR 2025) | GSM-Symbolic：符號化改寫系統性移動準確率 | — | — |

---

# 3. 建議的方向

## 3.1 定位

**從「提出一個方法（IMSR）」轉向「把 ensemble diversity 理論搬到 LLM 推理，並處理三個結構差異」。**

語言降級成五個多樣性軸之一，IMSR 降級成被測量的代表性 pipeline。

## 3.2 三個 RQ

```
RQ1（工具，降級為 Proposition）
  任何「產生多條候選 → 聚合」的方法，其增益能否拆成 d × (c−m) × recovery？
  → 數學上已解決；價值在於「為什麼在 LLM 設定下需要拆得比古典更細」
  → §3 寫成 Proposition + 四個假設 + 逐項實測驗證

RQ2（實質，headline）
  不同的多樣性來源（語言 / 採樣 / 推理強度 / 角色 / 改寫），
  在控制題目組成之後，是否給出相同的 c 與 recovery？
  → diversity 是可替代的商品，還是有品質之分？

RQ3（成本）
  取得多樣性的代價是什麼？語言是不是最貴的一種？
  → 判準：一個多樣性來源值得採用 ⟺ recovery > Δm / Δh
  → 用 Gao et al. 的數字：多語言要贏英語重採樣需 recovery ≥ 0.60，現有全在 0.33–0.42
```

## 3.3 生成端：五個多樣性軸

> **Agent A（錨點，全實驗共用）**：原始英文題 + 完整 CoT + T=0。生成一次，所有軸重複使用。

| 軸 | Agent B | 變量 | 資料集 | 現況 |
|---|---|---|---|---|
| **L 語言** | 翻成 L + 完整 CoT + T=0 | ZH, JA, RU, ES | 全部 4 個 | ✅ 已有 |
| **S 採樣** | 同英文題 + T>0、換 seed | T = 0.7/1.0/1.3 | 全部 4 個 | ⬜ |
| **R 推理強度** | direct / short CoT / token 截斷 | 3 級 | MathQA, MMLU | 🟡 有單語 |
| **P 角色** | persona prompt | 2 級異議強度 | 全部 4 個 | ⬜ |
| **W 改寫** | 改寫題 / 干擾注入 | 2 級 | MathQA, MMLU | 🟡 有單語 |
| **K 候選數**（對照組） | 逐步加入候選 | K = 2,3,4,5 | MathQA, MMLU | ⬜ |

**🔴 K 軸是唯一「只改變 diversity 不改變個體能力」的軸**，正是 Wood et al. Fig.14 說的乾淨情況。應該當作陽性對照。

## 3.4 聚合端：五個聚合器

| 代號 | 聚合器 | K | 封閉性 | recovery 性質 | 用途 |
|---|---|---|---|---|---|
| **V2** | Vote@2（隨機破結） | 2 | ✅ | **恆等於 0** | 尺規零點 |
| **Blind** | 固定選錨點 | 2 | ✅ | `2w_A−1`，**可為負** | **部署基準** |
| **Judge** | 單次，看完整推理 | 2 | ✅ | 待測 | **主臂** |
| **Debate** | IMSR | 2 | ⚠️ 可能違反 | 實測 0.34 | 既有資料 |
| **Vote@3/5, Judge@5** | — | 3,5 | ✅ | 有理論值 | **校準測試** |

**為什麼主臂用 Judge**：Debate 會改寫候選，引入第二個變動源；Judge 限制輸出只能是 A 或 B，封閉性構造上成立；成本固定；與 Gao et al. 的 Judge@k 可比。

**⚠️ K=2 的 Vote 是退化的**（無多數，只能隨機挑，`recovery ≡ 0`），**所以校準測試必須 K ≥ 3。**

## 3.5 題目端：難度分層

### 為什麼需要（假想例子，100 題）

| 題型 | 題數 | 性質 | L 軸分歧 | S 軸分歧 |
|---|---|---|---|---|
| T1 | 60 | 所有條件都答對 | ❌ | ❌ |
| T2 | 25 | 英文穩定對、日文錯 | ✅ | ❌ |
| T3 | 15 | 模型本身不確定 | ✅ | ✅ |

```
D_L = T2∪T3 = 40 → d=0.40, c = (25×1.00+15×0.50)/40 = 0.81
D_S = T3    = 15 → d=0.15, c = 0.50
若 judge 在 T2 答對 90%、T3 答對 35% → recovery_L ≈ 0.71, recovery_S = 0.40
```

**judge 在相同題目（T3）上表現完全一樣，但你會看到 0.71 vs 0.40。100% 是組成造成的。**

### 調 d 解決不了
旋鈕同時減少 T2 和 T3 的分歧，比例不變。**d 對齊了，組成沒對齊。**

### ❌ 逐題配對也不行

`d_跨語言 ≈ 0.211`、`d_採樣 ≈ 0.075`：

| 相關性 | 交集題數（n=2000） | SE(recovery) |
|---|---|---|
| 完全獨立 | 32 | 0.234 |
| 中度正相關 | 60 | 0.169 |
| 高度正相關 | 98 | 0.133 |

**需要 ≤ 0.035。而且高 d 的軸被砍 86%、低 d 的只砍 60%，製造新的差別偏誤。**

### ✅ 正確做法：事後分層

```
Step 1  q_i = 第 i 題在所有 run 中被答對的比例（leave-one-axis-out 避免循環）
Step 2  按 q_i 五分位切成 5 層
Step 3  層內分別算 c(A,s)、recovery(A,s)
Step 4  用共同權重 w_s 加總
Step 5  敏感度：3/5/10 層各做一次
```

**⭐ 免費捷徑**：先看各軸的 `recovery_blind` 差多少。差很多 → 組成一定差很多 → 需要分層。

## 3.6 ⭐ 與古典 ensemble 的三個結構差異（必須寫進 §2）

| | 古典 ensemble | LLM 推理 | 為什麼重要 |
|---|---|---|---|
| **① 觸發時機** | 每題都合併 | **只在分歧時觸發** | 🔴 `d` 從指標變**成本變數**。古典沒有理由拆出 d |
| **② 候選內容** | 標籤或機率 | **完整推理過程** | 🔴 聚合器可以「讀論證」，recovery 可能遠高於多數決理論值 |
| **③ 聚合器身分** | 獨立規則或 meta-model | **生成者本身** | 🔴 自我聚合，古典無對應物 |
| ④ 多樣性來源 | 統計性（重抽樣、特徵子集） | **語義性**（語言、角色、推理風格） | 🟠 是否等價是實證問題 |
| ⑤ 成本結構 | 訓練時，成員成本相同 | **推論時，路徑成本差好幾倍** | 🔴 RQ3 在古典文獻不存在 |
| ⑥ 候選數 K | 幾十到幾百 | **通常 2–5** | 🟠 大數法則不適用 |

**①③⑤ 是真正的立足點。只講「他們做 ML、我們做 LLM」會被判定 incremental。**

## 3.7 標題方向

```
主推  Where Does the Gain Come From?
      Decomposing Test-Time Diversity in LLM Reasoning

備選  Not All Diversity Is Equal: Disentangling Trigger Rate,
      Information, and Aggregation Efficiency in LLM Ensembles
```

---

# 4. 完整邏輯流程

## §1 Introduction

| 段落 | 主張 | 支撐 | 該來源的論點 |
|---|---|---|---|
| 開場 | test-time scaling 方法百花齊放，但都只報最終準確率 | Brown et al. (2024) | coverage 冪次成長，但多數決與 RM 在幾百樣本後飽和 |
| 問題 | 兩個方法分數相同可能原因完全不同，缺共同語言 | — | — |
| 承接 | 古典 ensemble 已有 ambiguity-effect | **Wood et al. (2023)** | ensemble 增益 = Oracle − Average |
| 差異 | 但 LLM 有三個結構差異使這個量需要再拆 | §3.6 | — |
| 貢獻 | (1) 觸發式聚合下的三因子分解 (2) 五軸受控比較 (3) 成本判準 | — | — |

## §2 Related Work（四小節）

```
2.1 Ensemble diversity（古典）
    Krogh & Vedelsby → Kuncheva & Whitaker → Brown & Kuncheva → Wood et al.
    Breiman / Dietterich / Wolpert / Kuncheva 教科書 / Lam & Suen
    最後一段：三個結構差異

2.2 Test-time diversity in LLMs
    SC / MAD(Du) / MAD(Liang) / Self-Refine / MoA / ReConcile /
    DiVeRSe / Ask Me Anything / Universal SC / LLM-Blender

2.3 Cross-lingual reasoning as a diversity source
    Gao / AutoCAP / CLP / XLT / Cross-ToT / MGSM / mGRPO /
    Wendler / Chua / Global-MMLU

2.4 Coverage, verification, and selection
    Brown / Song / Weaver / Yue / Cruz(DES) / RouteLLM / RouterBench
    ← 解決 R3 的「§2.4 無引用」

2.5 Measurement practice
    Dodge / Tang / Card / Dror / Schaeffer / Oldham 系列
```

## §3 The Decomposition

### §3.1 一般式

```
A（K 個候選全相同，佔 1−d）、D（不全相同，佔 d）
m = E[候選正確比例 | D]，c = P(至少一個正確 | D)
recovery = (acc_agg − m)/(c − m)

Gain = (1−d)·Δ_agree + d·(c−m)·recovery
```

**明確標註**：`Gain` 即 Wood et al. 的 ambiguity-effect；`c` 即 Kuncheva 的 oracle accuracy。

**d 為什麼定義為「不全相同」**：不變性條件——一致題目上 headroom 逐題為零，搬動題目不改變 recovery。

**headroom 展開**：`c − m = c(1 − r/K)`，K=2 時 r=1 退化成 c/2。

### §3.2 四個假設與診斷

| 假設 | 違反時 | 診斷 | 現況 |
|---|---|---|---|
| 單一正解、可自動判分 | 🔴 硬邊界 | — | 滿足 |
| 聚合器在一致題目上是 no-op | Δ_agree ≠ 0 | 逐題比對 | ⬜ 0A-2 |
| 聚合器封閉 | recovery > 1 | 查 acc_agg ≤ c | ✅ legacy 24/24 |
| K = 2 | 不能化簡成 c/2 | 用一般式 | — |

### §3.3 Max-baseline 與 Blind 基準

```
excess = Gain − gap/2 = d(c/2)(recovery − recovery_blind)
recovery_blind = 2w_A − 1 = gap/(d·c)
skill = (recovery − recovery_blind)/(1 − recovery_blind)
gap* = d × c × recovery
```

**支撐**：Krogh & Vedelsby（ensemble 只保證贏過平均）、Oldham/Blomqvist（耦合）、Dodge et al.（best-of-K 高估）

### §3.4 為什麼 LLM 設定需要拆得比古典更細
引 §3.6 的三個結構差異。

## §4 Experiments

| 小節 | 內容 | 資料 |
|---|---|---|
| 4.1 | 五軸的 d / c / m / H / recovery_blind / recovery / 成本（難度分層後） | 臂 1 |
| 4.2 | 陽性對照：固定軸換聚合器（V2/Blind/Judge/Debate） | 臂 2 |
| 4.3 | K 軸（唯一不改變個體能力的軸） | 臂 3 |
| 4.4 | K>2 校準：Vote@3/5 實測 vs 理論；r 診斷 | 臂 3 |
| 4.5 | 外部驗證：Gao et al. 五個設定重算 | 臂 5 |
| 4.6 | 外部方法：開源 MAD 重跑 | 臂 4 |

## §5 Analysis

| 小節 | 主張 | 支撐 |
|---|---|---|
| 5.1 | 各軸的 H 差異大、recovery 等價（若成立） | §4.1 + TOST |
| 5.2 | H 拆成 d 與 (c−m)：語言靠「常觸發」還是「觸發得好」 | §4.1 |
| 5.3 | **基準選擇決定「diversity 有沒有用」的結論** | B-7 + A-2 |
| 5.4 | 兩平條件與各文獻的門檻 | §3.3 + Gao et al. |
| 5.5 | Robustness：耦合、EIV、split-half、異質性、λ | A-2/A-5/A-7 |

**§5.3 的具體寫法**：

> 「diversity 與效能的關聯之所以薄弱（Kuncheva & Whitaker, 2003），一個被忽略的原因是基準的選擇。在我們的資料上，相對候選平均時分歧率與增益強相關（r = 0.74）；相對最強候選時關係反轉為負（斜率 −0.38）。兩者的差恰好是 gap/2，一個純粹來自基準選擇的算術項。」

## §6 Implications

```
6.1 對 test-time diversity 方法的意涵（談「類」不談個別論文）
6.2 對跨語言推理方法的意涵（可點名 Gao / AutoCAP / CLP）
6.3 報告建議：請同時報 (d, c, recovery, recovery_blind)
    ← 讓你被後續論文引用的關鍵
```

## §7 Limitations

- recovery 的 MDE ±0.10，只能宣稱等價不能宣稱相同
- 硬邊界：需要單一正解，不適用自由生成
- 錨點只用一組，僅有限輪換
- R 軸與 W 軸只在 MathQA + MMLU
- Kuncheva & Whitaker：diversity 與效能的關聯在古典設定下就很弱
- Song et al. Fig.3 / Weaver：外部 verifier 會推動 recovery，我們的結論限於自我聚合

---

# 5. 待辦清單與投稿機率

## Stage 0A｜現有資料重算（零成本，5–7 天）

| # | 待辦 | 產出 | 用在 |
|---|---|---|---|
| **0A-1** | ⭐ **recovery_blind 分析**（160 對 + 24 legacy，含英文 vs 不含英文） | 配對不對稱性的分布 | §4.1, §5.5 |
| 0A-2 | 逐題檢查 Δ_agree = 0 與封閉性 | 兩個假設的實測驗證率 | §3.2 |
| 0A-3 | 各條件的解析失敗率稽核 | 資料品質 | Appendix |
| 0A-4 | Split-half 重新分析（200 次） | 耦合 + 選擇偏誤校正 | §5.5 |
| 0A-5 | 建立難度指標 q_i（LOAO）與分層工具 | 分析基礎設施 | §4 方法 |
| 0A-6 | recovery 異質性的 bootstrap CI（τ = 0.042 的不確定性） | §4 的關鍵數字 | §4.1 |
| 0A-7 | Table 1 三格標錯修正；Table 2 用新數字重算 | 正確性 | §4 |
| 0A-8 | 數字標籤修正（−0.412 是 per-cell 平均） | 精確性 | §5.5 |
| 0A-9 | 盈虧宣稱降級為 cell 層級 | 邏輯一致 | §5.5 |
| 0A-10 | G-4：legacy JSON 對回 2000 題重新計分 | 救回 DeepSeek/QWEN | §4.1 |

### 0A-1 規格（最高投報率）

```
對每個 cell × 每個語言對：
  1. 錨點 A = 單語準確率較高者（⚠️ 用 split-half 決定，避免 winner's curse）
  2. D = 兩者初答不同的題目；d = |D|/N
  3. n_A = D 中 A 正確的題數；n_B = D 中 B 正確的題數
     c = (n_A+n_B)/|D|；m = c/2；w_A = n_A/(n_A+n_B)
     recovery_blind = 2·w_A − 1
  4. 若有辯論結果：recovery、skill

輸出：
  表 1  pair 層級（160 + 24 列）
  表 2  含英文 vs 不含英文的分組比較（cell-cluster bootstrap 5000 次）
  表 3  三個虧損 cell 的診斷（看 skill 是否 ≤ 0）
  主分析：recovery_i ~ α + β·recovery_blind_i + (1|cell)
         β ≈ 0 → recovery 是純聚合器性質
         β ≈ 1 → 聚合器只是在模仿 blind
```

**⚠️ 需要逐題資料。先確認辯論 run 的 JSON 有沒有保留兩個 agent 的初答。**

### 決策點 D0
```
0A-1 顯示 β > 0.5     → recovery 不是純聚合器性質，RQ2 的設計要重想
0A-2 顯示封閉性違反 > 5% → Debate 的 recovery 解讀要加大量 caveat
```

## Stage 0B｜外部資料重算（零成本，7–10 天）

| # | 待辦 | 產出 |
|---|---|---|
| 0B-1 | 🔴 **Gao et al. 重算**（`github.com/CONE-MT/multilingual_reasoning`）：5 種多樣性來源的 d, c, m, recovery | 外部驗證 + 區隔 |
| 0B-2 | **搜尋 `Gain/(Oracle−Average)` 有無既有名稱** | novelty 確認 ❓ |
| 0B-3 | 讀 Wood et al. (2023) §1/§2/§4.2/§6.2 全文 | §2、§3 的承接寫法 |
| 0B-4 | Weaver 重算（`github.com/HazyResearch/scaling-verification`） | §7 Limitations |
| 0B-5 | 撰寫 "Reuse of public artifacts"（授權、commit hash、與原論文差異） | Reproducibility 1→4 |

### 決策點 D1
```
0B-2 若發現該比值已有名稱與文獻
    → RQ1 的殘餘 novelty 只剩「觸發式聚合下的 d 拆分」
    → 必須把 §3 的份量壓縮，把重心全部移到 RQ2/RQ3
0B-1 若顯示 Repeat/Paraphrase 的 recovery 也 ≈ 1/3
    → 語言偏誤不是主因，可寫進 §5
```

**Stage 0 完成後**

| 項目 | 狀態 |
|---|---|
| Contribution | 分解（承接 Wood et al.）+ max-baseline 分析 + 外部重算 |
| Novelty | 🟡 中偏低。**沒有自己的新實驗** |
| Soundness / Excitement | 3.0–3.5 / 2.5 |
| **P(main) / P(≥Findings)** | **12% / 50%** |

## Stage 1｜Pilot（~$5，3 天）

`1 模型 × 2 資料集 × 4 軸 × Judge`

| # | 檢查 | 門檻 |
|---|---|---|
| ① | 各軸 d 範圍重疊 | 兩兩至少 0.05 重疊 |
| ② | 難度分層後每軸每層樣本數 | ≥ 100 |
| ③ | 各軸解析失敗率差異 | < 1% |
| ④ | 實測重算的 recovery MDE | ≤ 0.10 |
| ⑤ | 各軸 recovery_blind 差多少 | 決定分層是否必要 |

## Stage 2｜主臂 + 對照（~$45，4 週）

| # | 待辦 | 規模 | 說明 |
|---|---|---|---|
| 2-1 | **S 採樣軸** × Judge | 16 格 × 2 T | 最便宜，先跑 |
| 2-2 | **P 角色軸** × Judge | 16 格 × 2 persona | 一個 prompt |
| 2-3 | 🔴 **陽性對照**：L 軸 × {V2, Blind, Judge, Debate} | 16 格（$3） | **必要，證明 recovery 有辨別力** |
| 2-4 | 🔴 **K 軸**：K = 2,3,4,5 × Vote/Judge | 8 格（$10） | **唯一不改變個體能力的軸** |
| 2-5 | **R 推理強度軸** × Judge | 8 格 | 已有單語 |
| 2-6 | **W 改寫軸**（CSQA 用 Shi et al. 干擾注入） | 8 格 | 可選 |
| 2-7 | 錨點輪換敏感度 | 2 格 | 回應「錨點只有一組」 |
| 2-8 | 難度分層 + TOST | — | headline |
| 2-9 | d–H 曲線（每軸 3 點），斜率 dH/dd | — | 多樣性品質的正面定義 |

**每格必記六個數字**：`acc_A`、`acc_B`、`d`、`c`、`m`、`recovery`（+ token 成本）。前五個不需跑聚合器。

### 決策點 D2
```
2-3 若顯示換聚合器時 recovery 也不動 → 🔴 指標可能無辨別力，停下來檢查測量
2-4 若 K 軸顯示 d 與增益強相關 → ✅ 可寫「乾淨情況下關係成立，改變來源時崩掉」
```

**Stage 2 完成後**

| 項目 | 狀態 |
|---|---|
| Novelty | 🟠 中。⚠️ Gao et al. 已比過語言/採樣/改寫三軸；增量是難度分層 + 分解 + 陽性對照 + K 軸 |
| Soundness / Excitement | 3.5–4.0 / 3.0 |
| **P(main) / P(≥Findings)** | **32% / 72%** |

## Stage 3｜K>2 校準（~$20，1 週）

| # | 待辦 | 產出 |
|---|---|---|
| 3-1 | 🔴 Vote@3/5 實測 vs **理論** recovery | **尺的校準測試** |
| 3-2 | Judge@5（與 Gao 的 Judge@k 可比） | 跨論文可比性 |
| 3-3 | `r` 診斷（可救題目上平均幾個候選正確） | K>2 才有的量 |
| 3-4 | recovery vs K 曲線 | 解釋 Gao 的「k 越大 Vote@k 越差」 |

| **P(main) / P(≥Findings)** | **42% / 78%** |
|---|---|

## Stage 4｜外部方法重跑（~$20，3 週）

| # | 待辦 | 為什麼 |
|---|---|---|
| 4-1 | 🔴 **開源 MAD 重跑**（Du et al. 或 Liang et al.） | 證明不只對自己的 pipeline 有效 |
| 4-2 | Self-Refine（可選） | 示範 Δ_agree ≠ 0 |
| 4-3 | **核心 figure**：recovery 分布圖 | 見下 |

```
Vote@2（隨機）                    0.00   ← 尺規零點
Blind（固定選錨點）                 ?     ← 部署基準（可為負）
自我聚合（SC / Judge / Debate / MAD）0.33–0.42
外部 verifier 集成（Weaver）         ?     ← 預期明顯更高
Oracle                            1.00
```

| **P(main) / P(≥Findings)** | **58% / 83%** |
|---|---|

## Stage 5｜寫作與開源（2 週）

| # | 待辦 |
|---|---|
| 5-1 | §2 重寫成五小節，引用 17 → 55+ |
| 5-2 | §6.3 報告建議「請同時報 (d, c, recovery, recovery_blind)」 |
| 5-3 | §7 Limitations 誠實列出 |
| 5-4 | 開源程式碼與資料 |
| 5-5 | 措辭檢查：對 Wood et al. / Kuncheva & Whitaker / Gao et al. 一律「承接」不「糾正」 |

| **P(main) / P(≥Findings)** | **64% / 86%** |
|---|---|

## 機率總表

| 階段 | 累積時間 | 累積成本 | P(main) | P(≥Findings) |
|---|---|---|---|---|
| 現況（v3 原稿） | — | — | 5% | 30% |
| Stage 0 | 3 週 | $0 | 12% | 50% |
| + Stage 1–2 | 8 週 | $50 | 32% | 72% |
| + Stage 3 | 9 週 | $70 | 42% | 78% |
| + Stage 4 | 12 週 | $90 | 58% | 83% |
| **+ Stage 5** | **14 週** | **$90** | **64%** | **86%** |

⚠️ 相對 v5 全面下修約 4pp，因為 ensemble 文獻壓縮了 RQ1 的 novelty。

### 下修分支

| 若發生 | P(main) | P(≥Findings) |
|---|---|---|
| 0B-2 發現該比值已有名稱 | 45% | 78% |
| D2 失敗（換聚合器 recovery 也不動） | 10% | 35% |
| 各軸 recovery 與 H 都等價（純負向） | 38% | 72% |

⚠️ 純負向結果仍可發表（Smit et al. → ICML、Huang et al. → ICLR），前提是尺架在別人的方法上。

## 12 月時程

```
9/10 – 9/30    Stage 0A + 0B（決策點 D0, D1）
               並行：立刻開始寫 §1, §2, §3
10/1 – 10/7    Stage 1 Pilot
10/8 – 11/4    Stage 2（順序：2-3 → 2-4 → 2-1 → 2-2 → 2-5）
11/5 – 11/11   Stage 3
11/12 – 12/2   Stage 4
12/3 – 12/15   Stage 5
```

**⚠️ 兩個提醒**
1. **論文從第一天開始寫。** §1/§2/§3 不依賴新結果。
2. **本週確認：你的論文算新投稿還是 resubmission？** 若算 resubmission，1.5 分的 review 會跟著你。

---

# 6. 完整文獻表

**圖例**：🔴 必引　🟢 建議引　⚪ 不必引

## 6.1 古典 Ensemble（新增，必須處理）

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **A Unified Theory of Diversity in Ensemble Learning**<br>Wood, Mu, Webb, Reeve, Luján, Brown — JMLR 2023, arXiv 2301.03962 | 🔴 | **最危險的前作**，你的 Gain 就是他們的 ambiguity-effect | (1) Prop 7：ensemble 風險 = 平均個體風險 − ambiguity-effect，適用任何損失；(2) Thm 10：0/1 loss 下任何 combiner 的該差值都必然依賴標籤；(3) Fig.14：只在單獨改 m 時 diversity 才乾淨預測效能（r² 0.99 → 0.59） |
| **Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy**<br>Kuncheva & Whitaker — Machine Learning 51(2), 2003 | 🔴 | **RQ2 的前車之鑑** | 研究十個 diversity 指標（Q statistic、correlation、disagreement、double fault、entropy、difficulty index、KW variance 等），發現與 ensemble 效能關聯薄弱，質疑其實用性 |
| **Good and Bad Diversity in Majority Voting Ensembles**<br>Brown & Kuncheva — MCS 2010 | 🔴 | 你的救援/破壞分解已有名字 | `ℓ(y,q̄) = 平均個體損失 − yq̄·(diversity)`。`yq̄=+1` 時 diversity 減少錯誤（good），`=−1` 時增加（bad） |
| **Neural Network Ensembles, Cross Validation and Active Learning**<br>Krogh & Vedelsby — NIPS 1994 | 🔴 | excess vs delta 的 1994 年版本 | ambiguity decomposition：ensemble 誤差 = 個體平均誤差 − ambiguity。**只保證贏過平均，不保證贏過最強** |
| **Combining Pattern Classifiers: Methods and Algorithms** (2nd ed.)<br>Kuncheva — Wiley 2014 | 🔴 | `c` 的定義來源 | "oracle accuracy"（至少一個成員正確）的標準定義；majority vote 的理論極限 |
| **Ensemble Methods in Machine Learning**<br>Dietterich — MCS 2000 | 🟢 | §2.1 的框架性引用 | ensemble 有效的三個理由：statistical、computational、representational。「兩個分類器 diverse = 在新資料上犯不同的錯」 |
| **Bagging Predictors / Random Forests**<br>Breiman — 1996 / 2001 | 🟢 | 「製造 diversity」的經典方法 | 用 bootstrap 重抽樣（Bagging）與特徵隨機化（RF）製造成員差異 |
| **Stacked Generalization**<br>Wolpert — Neural Networks 1992 | 🟢 | 教授的 100/900 切分 | 用 held-out 資料訓練 meta-learner 學習如何合併成員輸出 |
| **Application of Majority Voting to Pattern Recognition**<br>Lam & Suen — 1997 | 🟢 | recovery 的理論基準 | 個體獨立且 p > 0.5 時，多數決保證優於個體 |
| **Dynamic Classifier Selection: Recent Advances and Perspectives**<br>Cruz, Sabourin, Cavalcanti — Information Fusion 2018 | 🟢 | 你的 per-query router 的古典版 | DES 綜述；以 oracle 為上界，實際選擇方法與 oracle 落差很大 |
| **From Dynamic Classifier Selection to Dynamic Ensemble Selection**<br>Ko, Sabourin, Britto — Pattern Recognition 2008 ❓ | ⚪ | 與 Cruz 綜述重複 | DES 的原始提出 |
| **On Over-fitting in Model Selection and Subsequent Selection Bias**<br>Cawley & Talbot — JMLR 2010 ❓ | 🟢 | pilot winner's curse 的古典版 | 在同一批資料上做模型選擇與效能評估會產生選擇偏誤 |
| **Bias-Variance-Covariance decomposition**<br>Ueda & Nakano — 1996 | ⚪ | Wood et al. 已涵蓋 | 平方損失下的三項分解 |

## 6.2 核心競品（LLM 側）

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **Could Thinking Multilingually Empower LLM Reasoning?**<br>Gao, X. Huang, Zhu, S. Huang, Li, Yuan — arXiv 2504.11833 | 🔴 | **RQ2 最直接的競品**，也是 recovery 的外部驗證來源 | (1) 已比較 Multilingual / Repeat / Paraphrase / 兩種 Mix；(2) Multilingual 的 Acc@k 在 k=17 高約 8 點；(3) Vote@k 優勢消失甚至輸給 Repeat；(4) 歸因於 judge 的語言偏誤 |
| **Mind the Gap: Examining the Self-Improvement Capabilities of LLMs**<br>Song, H. Zhang, Eisenach, Kakade, Foster, Ghai — ICLR 2025 | 🟢 | §3 的方法論註記 + §7 | (1) relative GV-Gap 的分母是 `1−m`（距離滿分），我們用 `c−m`（可達上限）；(2) relative gap 隨 pretraining flops 上升；(3) cross-verification：固定 generator 換大 verifier，gap 3.39→24.91 |
| **Shrinking the Generation-Verification Gap with Weak Verifiers (Weaver)**<br>Saad-Falcon et al. — NeurIPS 2025, arXiv 2506.18203 | 🟢 | §7 Limitations | 加權集成多個弱 verifier + weak supervision，Llama 3.3 70B 達 o3-mini 級（87.7%）；蒸餾成 400M cross-encoder 保留 98.7%。**用外部 verifier** |
| **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling**<br>Brown, Juravsky, Ehrlich, Clark, Le, Ré, Mirhoseini — 2024 | 🔴 | §1 的問題背景 | coverage 隨採樣冪次成長至 95%+，但多數決與 reward model 在幾百樣本後飽和 |
| **Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?**<br>Yue et al. — NeurIPS 2025 | 🟢 | §6 的延伸 | RLVR 提升 pass@1 但大 k 時 pass@k 不超過 base model |
| **Cross-lingual Self-Consistency for Multilingual Reasoning**<br>arXiv 2606.01464 ❓ | 🔴（待查） | 2026 年、題目高度重疊 | ❓ **必須自己讀** |

## 6.3 多語言推理

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **AutoCAP**<br>Y. Zhang, Chen, Li, Che, Qin — Findings ACL 2024 | 🔴 | R2 點名；§6.2 對象 | 自動語言選擇 prompting + 自動權重分配 |
| **mGRPO: Unlocking LLM Reasoning through Multilingual Thinking**<br>OpenReview ❓ | 🔴 | R2 點名 | 不限制語言時表現最好；用受限/不受限 prompt 生成偏好 group 做 GRPO，四 benchmark 平均 +7.5% |
| **Cross-Lingual Prompting (CLP)**<br>Qin, Chen, Wei, S. Huang, Che — EMNLP 2023 | 🔴 | 跨語言 SC 的代表 | 跨語言對齊 + self-consistent prompting 聚合多語推理路徑 |
| **Not All Languages Are Created Equal in LLMs (XLT)**<br>H. Huang et al. — Findings EMNLP 2023 | 🔴 | 已引用 | cross-lingual-thought 模板把非英語導向英語推理 |
| **Language Models are Multilingual CoT Reasoners (MGSM)**<br>Shi, Suzgun, Freitag, X. Wang et al. — ICLR 2023 | 🔴 | R3 點名 | 多語 CoT benchmark；非英語落後英語；translate-to-English 強 |
| **Do Llamas Work in English?**<br>Wendler, Veselovsky, Monea, West — ACL 2024 | 🔴 | A-6/A-9 的機制解釋 | 模型在英語為中心的概念空間運算，非英語輸入先映射過去 |
| **Crosslingual Capabilities and Knowledge Barriers**<br>Chua et al. — 2024 | 🟢 | 反向預測 | 知識在語言間不共享，存在跨語言知識屏障 |
| **Cross-lingual Consistency of Factual Knowledge**<br>Qi, Fernández, Bisazza — EMNLP 2023 | 🟢 | 同受耦合影響 | RankC 指標量測跨語一致性 |
| **Global-MMLU**<br>Singh et al. — 2024 | 🟢 | 回應翻譯 artifact | 機器翻譯評測的偏差與文化敏感題目 |
| **Language Imbalance Can Boost Cross-Lingual Generalisation**<br>Schäfer et al. — 2024 | 🟢 | **方向相反** | 訓練期語言不均衡反而提升泛化 |
| **Do Multilingual Language Models Think Better in English?**<br>Etxaniz et al. — NAACL 2024 | 🟢 | self-translate | 模型自我翻譯成英語後表現更好 |
| **Cross-ToT**<br>Ranaldi et al. | 🟢 | 另一個跨語言聚合 | 跨語言 ToT 整合多語推理路徑 |
| **Multilingual LLMs Are Not (Yet) Code-Switchers**<br>R. Zhang et al. — EMNLP 2023 | 🟢 | 回應 R3 的混語質疑 | 多語 LLM 在 code-switching 表現不佳 |
| **Do Multilingual LLMs Think in English?**<br>Schut et al. — 2025 ❓ | 🟢 | 與 Wendler 同方向 | 表徵層的英語中心證據 |
| **MAPO / mCoT / Question Translation Training** | ⚪ | 訓練期方法 | — |

## 6.4 LLM 聚合器與多 agent

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **Self-Consistency Improves CoT Reasoning**<br>X. Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou — ICLR 2023 | 🔴 | 校準測試對象 | 多次採樣取多數決取代 greedy decoding |
| **Improving Factuality and Reasoning through Multiagent Debate**<br>Du, Li, Torralba, Tenenbaum, Mordatch — ICML 2024 | 🔴 | Stage 4 重跑對象 | 多個 LLM 實例並行提案、多輪互看修正 |
| **Encouraging Divergent Thinking through Multi-Agent Debate**<br>Liang et al. — EMNLP 2024 | 🔴 | 他們的解法就是拉高 d | Degeneration-of-Thought；用異議 agent + judge 維持分歧 |
| **Should We Be Going MAD?**<br>Smit, Grinsztajn, Duckworth, Barrett, Pretorius — ICML 2024 | 🔴 | 同方向盟友；敘事範本 | 對齊 prompt 品質與預算後，MAD 贏不過單一 well-prompted agent |
| **Large Language Models Cannot Self-Correct Reasoning Yet**<br>J. Huang et al. — ICLR 2024 | 🔴 | recovery 是其量化 | 內在自我修正不可靠；response 數對齊時 MAD 輸給 SC |
| **Self-Refine**<br>Madaan et al. — NeurIPS 2023 | 🔴 | R3 點名；Δ_agree ≠ 0 的案例 | 單模型自我回饋迭代改進 |
| **Debating with More Persuasive LLMs**<br>Khan et al. — ICML 2024 | 🔴 | recovery 可獨立操作的證據 | judge 與 debater 的相對強度是關鍵變數 |
| **ReConcile**<br>J. Chen, Saha, Bansal — 2024 | 🟢 | 多樣性來源 = 模型 | 不同 LLM 圓桌討論 + 信心加權投票 |
| **Mixture-of-Agents**<br>J. Wang et al. — 2024 | 🟢 | 相對「最佳單一模型」 | 分層聚合多個 LLM 回應 |
| **LLM-Blender**<br>D. Jiang, Ren, Lin — ACL 2023 | 🟢 | oracle best-of-N | pairwise ranking + generative fusion |
| **Universal Self-Consistency**<br>X. Chen et al. — 2023 | 🟢 | recovery 換成 LLM judge | 用 LLM 選最一致的回應 |
| **DiVeRSe**<br>Y. Li et al. — ACL 2023 | 🟢 | 多樣性來源 = prompt | 多樣 prompt 生成多路徑 + verifier 加權 |
| **Ask Me Anything**<br>Arora et al. — ICLR 2023 | 🟢 | 最便宜的對照軸 | 多個 prompt 格式的輸出用弱監督聚合 |
| **ChatEval**<br>Chan et al. — ICLR 2024 | 🟢 | `c` 的定義邊界 | 多 agent 辯論做評估（無 ground truth） |
| **Why Do Multi-Agent LLM Systems Fail? (MAST)**<br>Cemri et al. — 2025 ❓ | 🟢 | 救援/破壞的質性版本 | 多 agent 系統失效模式分類 |
| **Reflexion**<br>Shinn et al. — NeurIPS 2023 | 🟢 | 原論文的 SR baseline | 語言 agent 的口語強化學習 |

## 6.5 Verifier 與 Routing

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **Training Verifiers to Solve Math Word Problems (GSM8K)**<br>Cobbe et al. — 2021 | 🟢 | verifier 路線起點 | 訓練 verifier 排序候選，大幅優於 fine-tuning |
| **Let's Verify Step by Step**<br>Lightman et al. — ICLR 2024 | 🟢 | 提升 recovery 的既有路線 | 過程監督優於結果監督 |
| **Scaling LLM Test-Time Compute Optimally**<br>Snell, Lee, Xu, Kumar — 2024 | 🟢 | RQ3 背景 | 依難度分配 test-time compute 比放大模型有效 |
| **RouteLLM**<br>Ong et al. — 2024 | 🟢 | per-query 選擇 | 用偏好資料訓練 router 在強弱模型間分流 |
| **RouterBench**<br>Q. Hu et al. — 2024 | 🟢 | oracle router 上界受雜訊灌水 | routing 的系統性評測 |
| **LLM Routing with Benchmark Datasets**<br>Shnitzer et al. — 2023 | 🟢 | 事後選擇偏誤 | 用 benchmark 為每任務選模型 |
| **FrugalGPT**<br>L. Chen, Zaharia, Zou — 2023 | 🟢 | 成本層前例 | 級聯降低 LLM API 成本 |
| **Hybrid LLM**<br>Ding et al. — ICLR 2024 | ⚪ | 與 RouteLLM 重複 | 成本效率的 query routing |
| **Self-Improvement: The Sharpening Mechanism**<br>A. Huang et al. — ICLR 2025 | ⚪ | 理論取向，與你的實證框架距離遠 | 自我改進是 sharpening |

## 6.6 評測方法論

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **Show Your Work: Improved Reporting of Experimental Results**<br>Dodge, Gururangan, Card, Schwartz, Smith — EMNLP 2019 | 🔴 | max-baseline 偏誤的既有工具 | expected validation performance：把 best-of-K 畫成 K 的函數，去除搜尋預算造成的高估 |
| **Showing Your Work Doesn't Always Work**<br>Tang et al. — ACL 2020 | 🟢 | EVP 的無偏版本 | EVP 估計量本身有偏 |
| **With Little Power Comes Great Responsibility**<br>Card et al. — EMNLP 2020 | 🔴 | A-5 的框架 | NLP 實驗普遍檢定力不足 |
| **The Hitchhiker's Guide to Testing Statistical Significance in NLP**<br>Dror, Baumer, Shlomov, Reichart — ACL 2018 | 🔴 | 回應 R3 的統計批評 | 檢定選擇流程與常見誤用 |
| **Are Emergent Abilities of LLMs a Mirage?**<br>Schaeffer, Miranda, Koyejo — NeurIPS 2023 | 🔴 | **敘事範本** | 「湧現」很大程度是不連續 metric 造成的假象 |
| **Accounting for Variance in ML Benchmarks**<br>Bouthillier et al. — MLSys 2021 | 🟢 | 支撐 I²=96% | 分解 ML benchmark 的變異來源 |
| **On the State of the Art of Evaluation in Neural LMs**<br>Melis, Dyer, Blunsom — ICLR 2018 | 🟢 | winner's curse 先例 | 控制調參預算後 SOTA 差距大幅縮水 |
| **Equivalence Testing (TOST)**<br>Lakens — SPPS 2017 | 🔴 | 主結論是「證明相同」 | 兩個單邊檢定：定實務界線，檢定差異是否小到可忽略 |
| **Quantifying Heterogeneity in a Meta-Analysis (I²)**<br>Higgins & Thompson — Stat Med 2002 | 🔴 | I² 的定義 | — |
| **Measuring Inconsistency in Meta-Analyses**<br>Higgins, Thompson, Deeks, Altman — BMJ 2003 | 🟢 | 25/50/75% 解讀基準 | — |
| **Plea for Routinely Presenting Prediction Intervals**<br>IntHout et al. — BMJ Open 2016 | 🔴 | 「兩平點不是常數」 | 高異質性時應報預測區間 |
| **Fixed-Effect and Random-Effects Models**<br>Borenstein et al. — RSM 2010 | 🟢 | 隨機效果彙總 | — |
| **A Note on the Analysis of Change**<br>Oldham — J Chronic Dis 1962 | 🔴 | 耦合的數學基礎 | 差值對基線迴歸會因共用誤差產生假負斜率；改用前後平均 |
| **On the Relation Between Change and Initial Value**<br>Blomqvist — JASA 1977 | 🟢 | 閉式校正 | 偏誤的閉式解 |
| **Revisiting the Relation Between Change and Initial Value**<br>Tu & Gilthorpe — Stat Med 2007 | 🟢 | coupling 回顧 | 耦合與 regression to the mean |
| **Correcting for Regression Dilution Bias**<br>Frost & Thompson — JRSS-A 2000 | 🟢 | λ 校正 | 用信度比 λ 校正 regression dilution |
| **Mismeasured Variables in Econometric Analysis**<br>Hausman — JEP 2001 | ⚪ | 非技術導論 | — |

## 6.7 實驗設計依據

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **LLMs Can Be Easily Distracted by Irrelevant Context**<br>Shi, Chen, Misra, Scales, Dohan, Chi, Schärli, Zhou — ICML 2023 | 🔴 | **CSQA 的替代旋鈕** | 注入不相關脈絡系統性降低推理，格式不變 |
| **GSM-Symbolic**<br>Mirzadeh et al. — ICLR 2025 | 🔴 | 改寫的先例與 caveat | 符號化改寫系統性移動準確率 |
| **FormatSpread**<br>Sclar, Choi, Tsvetkov, Suhr — ICLR 2024 | 🟢 | format confound | 格式微調造成巨大分數變動 |
| **Rethinking the Role of Demonstrations**<br>Min et al. — EMNLP 2022 | 🟢 | 分級破壞的方法論範本 | 分級破壞 demonstration 以隔離貢獻 |
| **To CoT or Not to CoT?**<br>Sprague et al. — 2025 | 🔴 | 解釋 CSQA/TQA 的設計邊界 | CoT 主要在數學與符號推理有效 |
| **Chain-of-Thought Prompting**<br>Wei et al. — NeurIPS 2022 | 🔴 | 基礎 | — |
| **MMLU-Pro**<br>Y. Wang et al. — 2024 | 🟢 | 已引用 | 更穩健的多任務理解 benchmark |
| **MMLU / CommonsenseQA / TruthfulQA / MathQA** | 🔴 | 資料集 | — |
| **Semantic Uncertainty**<br>Kuhn, Gal, Farquhar — ICLR 2023 | ⚪ | 你不做不確定性偵測 | — |

---

# 附錄：三句話總結給教授

> 1. 原本的論文是一個多語言辯論方法，真正影響的只有 6 篇跨語言 CoT 論文，value 質疑成立。
>
> 2. 我們的 Gain 分解對應到古典 ensemble 的 ambiguity-effect（Wood et al., JMLR 2023），所以不能宣稱原創。但 LLM 設定有三個結構差異——**聚合器只在分歧時觸發**（所以 `d` 是成本變數）、**候選帶完整推理**（聚合器可以讀論證）、**聚合器就是生成者本身**——使得把增益拆成「觸發率 × 可用資訊 × 聚合效率」有操作意義。
>
> 3. 現在的核心問題是：五種多樣性來源（語言、採樣、推理強度、角色、改寫）在控制題目組成後，是否給出相同的 c 與 recovery？以及取得多樣性的代價是什麼。這兩題在古典 ensemble 文獻中沒有對應物，因為古典成員的成本相同且 combiner 永遠動作。
