# 論文現況整理 v7（完整版）

> **目標：ACL ARR 2026/12 cycle**
> v7 相對 v6 的更新：**0A-1 / 0A-4 / 0A-5 結果全部到位**、**新增 `has_english` 發現**、**多個舊宣稱被推翻或修正**。
>
> **標記**
> ✅ 我實際重算驗證過　⚠️ 宣稱與資料不符或需修正　❓ 尚未查證

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
Gain = (1−d) × Δ_agree + d × (c − m) × recovery         一般式
Gain = d × (c/2) × recovery                             K=2、單一正解、觸發式

excess = Gain − gap/2 = d × (c/2) × (recovery − recovery_blind)
recovery_blind = 2·w_A − 1 = gap / (d·c)
skill = (recovery − recovery_blind) / (1 − recovery_blind)
gap* = d × c × recovery
```

## ⚠️ 最重要的定位：分解式不是新的

**你的 `Gain`（相對候選平均）= Wood et al.（JMLR 2023）的 ambiguity-effect。`c` = Kuncheva 的 oracle accuracy。救援/破壞分解 = Brown & Kuncheva (2010) 的 good/bad diversity。**

**真正的立足點是三個結構差異：**

| | 古典 ensemble | LLM 推理 |
|---|---|---|
| ① 聚合器何時動作 | 每題都動作 | **只在分歧時觸發** → `d` 是成本變數 |
| ② 候選帶什麼 | 標籤或機率 | **完整推理過程** → 表面形式會影響聚合效率 |
| ③ 聚合器是誰 | 獨立規則 | **生成者本身** |

**②現在有實證了**（見 §2.6 的 `has_english` 發現）。

## 三個 RQ

```
RQ1  增益能否拆成 d × (c−m) × recovery？
     → 數學已解決，降級為 §3 Proposition + 四假設驗證

RQ2（headline）
     不同多樣性來源（語言/採樣/推理強度/角色/改寫），
     控制題目組成後是否給出相同的 c 與 recovery？

RQ3  取得多樣性的代價？語言是不是最貴的一種？
     判準：recovery > Δm / Δh
```

## 已確立的關鍵數字

| 項目 | 數值 |
|---|---|
| 跨語言辯論 recovery | 0.34 |
| 同語言 EN / ZH recovery | 0.322 / 0.346 |
| Gao et al. 反推 recovery（K=4 多數決） | 0.329–0.415，平均 0.357 |
| recovery 的跨格 I² | 14.8%（對比 excess 的 96%） |
| **`has_english` 對 recovery 的效果** | **+0.093（格內，p<0.0001）** |
| excess ~ gap（pooled，split-half 後） | −0.361 |

---

# 1. 最初的困境

## 1.1 原始論文

**IMSR**：同一 LLM 扮演兩個語言的 agent，答案不一致時最多 3 輪跨語言辯論，仍不一致由 Judgement Agent 裁決。XLM-RoBERTa 做 per-query 語言對路由。4 模型 × 4 資料集 = 16 格。

## 1.2 審稿結果

| | Soundness | Excitement | Overall | Confidence |
|---|---|---|---|---|
| R1 | 2.5 | 2.5 | 2.5（Borderline Findings） | 4 |
| R2 | 2.0 | 2.5 | 2.0（Resubmit next cycle） | 4 |
| R3 | 2.0 | 1.5 | 1.5（Resubmit after next cycle） | 5 |

## 1.3 批評歸類

**A. 因果識別不足**
- R1-W1：沒有同語言雙 agent 控制組
- R3：EN-SR / ZH-SR 的輪數是否與 IMSR 對齊

**B. Novelty 不足**
- R2：只是 cross-lingual prompting + self-reflection + debate + routing 的組合
- R2：沒討論 AutoCAP、mGRPO
- R3：只有 17 篇引用；§2.4 無引用

**C. 統計不透明**
- R3：p < 0.05 怎麼算的？跑幾次？和誰比？

**D. 設計質疑**
- 五種語言的選擇沒理據；router 泛化範圍未討論
- R3：辯論 prompt 是混語
- R3：為什麼 best fixed pair 因模型而異

**E. 可重現性**：三位都給最低分

## 1.4 教授的兩個批評

**① 「只影響幾篇論文就沒 value」** — 成立。原方案只咬到 6 篇跨語言 CoT。

**② 「這跟 ML 的 ensemble 研究很類似」** — 成立，見 §2.7 對照字典。

---

# 2. 現有實驗結果盤點

## 2.1 資料集 A｜跨語言配對網格

4 模型 × 4 資料集 × 5 語言 → 16 格 × 10 對 = 160 觀測。n：CSQA/MathQA/MMLU 各 2000，TruthfulQA 817 ✅

### A-1｜跨語言配對優於最強單語

| 量 | 數值 |
|---|---|
| 16 格中 excess > 0 | **15/16** ✅ |
| 符號檢定 p | **5.19e-4** ✅ |
| 平均 excess | **+1.518pp**（舊文件寫 +1.46）⚠️ |

### A-2｜「侵蝕」是基準選擇的算術後果

恆等式 `excess = delta − gap/2`（160 列，誤差 2.4e-14）✅

| 迴歸 | 斜率 |
|---|---|
| excess ~ gap（pooled OLS） | **−0.383** |
| excess ~ gap（cell FE） | −0.373 |
| **delta ~ gap（pooled OLS）** | **+0.117**（t=3.17） |

**160/160 個配對的 delta > 0，平均 +2.73pp** ✅

| gap 分箱（pp） | 0.4 | 1.4 | 2.4 | 3.5 | 4.7 | 6.8 | 10.4 |
|---|---|---|---|---|---|---|---|
| excess | 2.24 | 1.86 | 1.45 | 1.57 | 0.83 | −0.57 | −1.60 |
| delta | 2.46 | 2.57 | 2.67 | 3.30 | 3.18 | 2.83 | 3.59 |

### A-3｜兩平點

`gap* = d × c × recovery`。代入 0.21 × 0.76 × 0.34 = **5.43pp**，實測 5.46pp ✅
per-cell 範圍 0.85–19.54pp（**不是常數**）

### A-4｜異質性

Cochran Q = 372.9（df=15，p=3.4e-70）、**I² = 96.0%**、τ = 1.288pp ✅

### A-5｜λ 識別力

**8/16 格 λ ≤ 0** ✅。例：DeepSeek/MathQA 五語言全距 0.55pp < 單點 SE 0.64pp。

### A-6｜共線性

格內 corr(max, gap) = **0.583**；corr(mean, gap) = 0.185 ✅

### A-7｜Oldham 等變異檢查

殘餘偏誤在 λ>0 的格子只有 +0.002~+0.012 ✅

### A-8｜LOO

MAE 1.033pp（常數 baseline 1.311pp）、R²_oos = 0.343
⚠️ pair 層級 Fisher p=4e-12，**cell 層級 p=0.106（不顯著）**→ 統一用 cell 層級

### A-9｜虧損結構

實質虧損 13 個 → **全部 3 個 cell、全部 CommonsenseQA、12/13 含英文**

---

## 2.2 資料集 B｜同語言辯論（legacy）

3 模型 × 4 資料集 × 2 語言 = 24 格。⚠️ GPT **4.1** mini、無 Gemini、n=3000–6000、T=1

### B-1｜恆等式驗證

`accuracy = (1−d)·a_agree + d·acc_debate` 24/24 誤差 1.1e-16 ✅
`Gain = d(c/2)·recovery` 24/24 誤差 < 1e-4 ✅
⚠️ 但計分程式**預設** Δ_agree = 0，逐題驗證待做

### B-2｜三量配對比較

| 量 | EN | ZH | 差 | p |
|---|---|---|---|---|
| **d** | 0.075 | 0.132 | −0.057 | **0.001** ✅ |
| c | 0.760 | 0.748 | +0.012 | 0.424 ✗ |
| recovery | 0.322 | 0.346 | −0.024 | 0.486 ✗ |

**只有 d 會動。**

### B-3｜檢定力（12 格配對）

d ±0.039、c ±0.042、**recovery ±0.097** → 只能宣稱「±0.10 內無法區分」，必須用 TOST

### B-4｜總 headroom

| | d | c | H = d(c/2) | recovery | Gain |
|---|---|---|---|---|---|
| 同語言 EN | 0.075 | 0.760 | 0.0282 | 0.327 | 0.93pp |
| 同語言 ZH | 0.132 | 0.748 | 0.0494 | 0.351 | 1.63pp |
| 跨語言 | ~0.211 | — | 0.0803 | ~0.34 | 2.73pp |

**跨語言 H 是同語言 EN 的 2.8 倍，但 recovery 三者相同 → 差異幾乎全部來自 d** ✅

### B-5｜recovery 的異質性

| | recovery | excess |
|---|---|---|
| 觀測 SD | 0.1095 | 1.147pp |
| 格內平均 SE | 0.1011 | 0.338pp |
| 真實格間 SD τ | **0.042** | 1.288pp |
| **I²** | **14.8%** | **96.0%** |

**recovery 的變異 85% 是雜訊，excess 的 96% 是真實差異。**
⚠️ τ=0.042 由兩個相近數相減得到，24 點上估計誤差大，需 bootstrap

### B-6｜變異數分解

Var(log Gain) = 0.265；**Var(log H) = 0.229（86%）**；Var(log recovery) = 0.104（39%）；2Cov = −0.068

### B-7｜⚠️ 一個被推翻的假說

我曾假設「拆成三因子會讓 diversity 與效能的關係變乾淨」。**不成立：**

| 預測 Gain 的變數 | r² |
|---|---|
| d | 54% |
| c | 3% |
| H = d(c−m) | **52%** ← 沒比 d 好 |

因為 c 幾乎是常數，`H ≈ d × 0.38`。

**真正的抵銷機制是 `gap/2`：**
```
同語言（gap≈0）    d 與 Gain 正相關 +0.737
跨語言（gap 變動）  gap 與 excess 負相關 −0.383
```
→ **「diversity 有沒有用」取決於基準選擇。**

---

## 2.3 資料集 C｜單語改寫與推理強度（部分完成）

| 資料集 | 改寫效果 | 等價檢定 | direct 掉幅 | 階梯可行 |
|---|---|---|---|---|
| MMLU | −1.85/−1.30/−0.35 | ✅ | 4–6.9pp | ✅ |
| MathQA | −2.55/+0.60/+0.65 | ✅ 大致 | 30–55pp | ✅ |
| CommonSenseQA | −2.85/−3.55/−3.70 | ❌ 難度掉 3pp | 0.8–1.5pp | ❌ |
| TruthfulQA | +2.32/+2.57/−1.35 | ⚠️ | 方向相反 | ❌ |

**🔴 A-9 的虧損全在 CSQA，而 CSQA 正好做不了階梯。** 解法：Shi et al.（ICML 2023）的分級干擾注入。

### ⚠️ 混合策略的數學問題

逐題按 p 混合時所有聚合量對 p 精確線性 → 斜率恆等於兩端點連線，**無法檢測非線性**。改用 token 預算截斷。

### 另外發現
short_cot ≈ full_cot（差 <1pp）→ **CoT 長度是二元開關不是斜坡。**

---

## 2.4 資料集 D｜Gao et al. 表格反推（零成本外部驗證）

他們的 Acc̄ = m、Acc@4 = c、Vote@4 = acc_agg。d 在相除時約掉 ✅

| 模型 | 組合 | m | c | Vote@4 | **recovery** |
|---|---|---|---|---|---|
| Qwen2.5-72B | Best | 43.7 | 74.3 | 54.2 | **0.343** |
| Qwen2.5-72B | Random | 41.5 | 70.0 | 51.7 | **0.358** |
| LLaMA3.1-70B | Best | 38.0 | 73.9 | 49.8 | **0.329** |
| LLaMA3.1-70B | Random | 36.9 | 70.2 | 48.8 | **0.357** |
| R1-Distill-70B | Best | 51.6 | 80.1 | 61.2 | **0.337** |
| R1-Distill-70B | Random | 49.0 | 75.5 | 60.0 | **0.415** |

**平均 0.357，SD 0.031** ✅。用單一常數 1/3 反推，六個預測五個誤差 <1 點 ✅

### 解開他們的矛盾

```
Multilingual: m=41.5, c=70.0, headroom=28.5, Gain=10.2 → Vote 51.7
Repeat:       c=65.9, Vote=53.6 → 反推 m≈47.5, headroom≈18.4, Gain=6.2
```
**Multilingual 增益大得多，輸掉是因為基線低 6 點。**
兩平條件：`recovery > Δm/Δh = 0.60`，現有聚合器全在 0.33–0.42。

---

## 2.5 🆕 0A-1｜recovery_blind 分析（184 對）

### 結果 1：recovery **不受**配對不對稱性驅動 ✅

| 分析 | β | p |
|---|---|---|
| 全部混在一起 | +0.136 | 0.006 |
| 只用跨語言 160 對 | +0.136 | 0.004 |
| **只看含英文 64 對** | **+0.020** | **0.85** |
| **只看不含英文 96 對** | **−0.038** | **0.59** |
| **加入 has_english 共變量** | **−0.025（CI [−0.12,+0.07]）** | **0.61** |

**組內完全無關聯。** 表面的 +0.136 來自 `recovery_blind` 與 `has_english` 共線（r=0.457）。

→ **D0 通過：辯論不是靠「認出誰比較強」。CI 上界 +0.07 排除了 β=1（模仿 blind）的假說。**

⚠️ 論文要報 CI 不是 p 值，因為你在宣稱「沒有效果」。TOST 在 ±0.15 界內會通過。

### 結果 2：🔴 但 `has_english` 是強效果

| 量 | 含英文(64) | 不含(96) | 差 | bootstrap CI |
|---|---|---|---|---|
| d | 0.180 | 0.191 | −0.011 | [−0.019,−0.002] |
| c | 0.776 | 0.767 | +0.009 | [−0.008,+0.023] |
| m | 0.388 | 0.384 | +0.004 | [−0.004,+0.012] |
| **recovery_blind** | **0.220** | **0.065** | **+0.155** | [+0.069,+0.241] |
| **recovery** | **0.456** | **0.366** | **+0.089** | **[+0.051,+0.131]** |
| skill | 0.232 | 0.308 | −0.076 | [−0.204,+0.042] |

**加入 cell 固定效果後：`has_english` = +0.093（SE 0.014, p<0.0001）**

**生成端三量幾乎相同，只有聚合效率差。**

### 結果 3：被排除的子解釋

「英文當錨點 vs 當搭檔」看似差很多（recovery 0.399 vs 0.611），**但加入 cell FE 後完全消失**（係數 −0.0006, p=0.99）。那 17 對集中在 6 個高 recovery 的 cell，純組成偏誤。

### 結果 4：虧損機制確認 ✅

- `loss_pair=True` 的 13 個，**skill ≤ 0 的比例 = 100%**
- `excess` 與 `excess_H2` 相關 **0.998**，符號一致 100%
- `excess_H2` 與 `skill_H2` 符號一致 96.7%（唯一例外是 excess_H2=0.000 的邊界）

```
gpt4omini|CSQA|english_vs_chinese
    recovery_blind = 0.570   ← 無腦選英文
    recovery       = 0.385   ← 辯論
    skill          = −0.439
```

**虧損不是「辯論壞掉」，是「英文本來就該信，辯論反而講歪」。**

### 結果 5：recovery_blind 可以是負的

```
不含英文組  平均 +0.065，最低 −0.106
含英文組    平均 +0.220，最低 −0.074
```

負值 ⟺ H1 選出的較強者在 H2 上反而較弱 ⟺ 真實落差被雜訊蓋過。
**這是 split-half 修正生效的直接證據**（若沒切分，`recovery_blind` 在定義上不可能為負）。
與 A-5 的 λ≤0 診斷互相印證。

---

## 2.6 🆕 0A-5｜難度分層（`has_english` 的組成解釋被排除）

### 結果 1：方向與「組成假象」的預測**相反**

`Δq = 含英文 − 不含英文`（q 越低越難）：

| 基底 | scope | Δq | 含英文較難的格數 |
|---|---|---|---|
| **cross_model（最可信）** | per_pair | **−0.0118** | **13/16** |
| all5 | per_pair | −0.0171 | 13/16 |
| loo | per_pair | −0.0316 | 12/16 |
| loo_noen | per_pair | −0.0172 | 14/16 |
| cross_model | union | +0.0038 | 8/16 |

**四種基底全部為負 → 含英文的配對觸發在更難的題目上，但 recovery 更高。**
組成偏誤的方向在**對抗**觀察到的效果，不是製造它。

### 結果 2：量級不可能解釋 +0.093

需要 `d(recovery)/dq = 0.093 / (−0.0118) = −7.9`

對照生成端實測斜率：`dc/dq=+0.466`、`dm/dq=+0.233`、`d(rb)/dq=+0.142`、`dd/dq=−0.341`

**全在 0.1–0.5。需要的是 17 倍且方向相反 → 不可能。**

### 結果 3：分層調整幾乎不動

| 量（cross_model） | raw 差 | adj 差 | 變動 |
|---|---|---|---|
| c | +0.0088 | +0.0139 | +0.005 |
| m | +0.0044 | +0.0070 | +0.003 |
| recovery_blind | +0.1419 | +0.1388 | −0.003 |

權重保留率 **1.000**（所有層樣本都夠）

### 結果 4：LOO 基底汙染確認

```
loo −0.0316 vs cross_model −0.0118 → 汙染量 ≈ 0.020
```
英文把不含英文組的 q 墊高。**主分析必須用 cross_model。**

### 結果 5：逐資料集與逐模型拆解 ← 決定性

**逐資料集**（控制 cell 後的 `has_english` 係數）：

| 資料集 | 係數 | p |
|---|---|---|
| **CommonsenseQA** | **+0.248** | <0.0001 |
| MMLU | +0.124 | 0.002 |
| TruthfulQA | +0.072 | 0.007 |
| **MathQA** | **+0.040** | **0.109（不顯著）** |

**逐模型**：

| 模型 | 係數 | p |
|---|---|---|
| **Gemini 2.5 Flash Lite** | **+0.162** | <0.0001 |
| **DeepSeek v3.2** | **+0.119** | 0.0004 |
| GPT-4o mini | +0.053 | 0.084 |
| **QWEN3-8b** | **+0.023** | **0.300（無效果）** |

**🔴 這一組是決定性的**：四個模型用的是**同一批翻譯**（同一翻譯器、同樣 2000 題），所以翻譯品質對四個模型是常數。**常數不可能產生 7 倍變異 → 翻譯損失不是唯一解釋。**

而 QWEN3（中文訓練佔比高）效果最小，與「英語中心概念空間」一致。

---

## 2.7 🆕 0A-4｜Split-half

| 變體 | 全樣本 | split-half | 200 次 [2.5%,97.5%] | bootstrap CI | 變動 |
|---|---|---|---|---|---|
| pooled OLS | −0.383 | **−0.361** | [−0.440,−0.290] | [−0.470,−0.185] | 0.022 |
| cell FE | −0.373 | **−0.362** | [−0.446,−0.276] | [−0.434,−0.250] | 0.011 |
| **per-cell 平均** | **−0.412** | **−0.281** | [−0.446,−0.116] | [−0.364,−0.193] | **0.131** |

### 三個判讀

**① 彙總層級的耦合偏誤很小（0.011–0.022）。** 原本預期掉到 −0.20~−0.25，**沒有**。→ 「耦合造成假斜率」的說法必須弱化。

**② per-cell 平均偏誤最大**（0.131），而且切半後的 −0.281 的 CI [−0.364,−0.193] **包含 EIV 的 −0.243** ✅ 兩法收斂。

**③ ⚠️ −0.412 應該從所有文件撤除**，它是三個估計中偏誤最大的。

### ⚠️ 版本 D 尚未執行

split-half 修掉耦合與選擇偏誤，但**讓衰減偏誤變嚴重**（n 減半）。所以 −0.361（無耦合、有衰減）與 −0.243（有耦合、無衰減）**不可直接比**。需跑 split-half + EIV。

### ⚠️ 另一個提醒

**split-half 修不掉那個 −0.5。** `excess = Gain − gap/2` 對真值也成立，那是基準選擇的真實後果，不是偏誤。

---

## 2.8 與古典 ensemble 的對照字典

| 你的框架 | Ensemble 文獻的名字 |
|---|---|
| K 條推理路徑 | m 個 ensemble members |
| 聚合器 | combiner |
| **Gain（相對平均）** | **ambiguity-effect / diversity-effect** |
| **c** | **oracle accuracy** |
| **d** | disagreement measure |
| 救援 − 破壞 | **good / bad diversity** |
| excess（相對最強） | ensemble vs best individual |
| pilot / holdout | **stacking** |
| per-query router | **Dynamic Ensemble Selection (DES)** |

**重要化簡**：一致題目上 `c_i = m_i`，所以 `d × (c−m) = Oracle − Average`，於是
```
recovery = Gain / (Oracle − Average)
```
❓ **必須搜尋這個比值有無既有名稱。**

---

## 2.9 佐證與反對的文獻對照

| 你的結果 | 佐證 | 該 paper 的論點 | 反對/需區分 | 該 paper 的論點 |
|---|---|---|---|---|
| Gain 分解 | **Wood et al. (JMLR 2023)** | ensemble 風險 = 平均個體風險 − ambiguity-effect | **同篇** | 🔴 **你的 Gain 就是他們的 ambiguity-effect** |
| 0/1 loss 的分解形式 | **Wood et al. Thm 10** | 0/1 loss 下任何 combiner 的該差值必然依賴標籤 | — | 用 c、m 不是缺點，是唯一可能 |
| 救援/破壞 | **Brown & Kuncheva (2010)** | diversity 在答對時減少錯誤、答錯時增加 | — | 🔴 已有名字 |
| B-7 diversity 關聯薄弱 | **Kuncheva & Whitaker (2003)** | 十個 diversity 指標與效能關聯都弱 | — | 🔴 RQ2 的前車之鑑 |
| 五軸實驗需控制能力 | **Wood et al. Fig.14** | 只在單獨改 m 時 diversity 乾淨預測效能（r² 0.99→0.59） | — | 🔴 五軸都同時改變能力 |
| excess vs delta | **Krogh & Vedelsby (1994)** | ensemble 誤差 ≤ **個體平均**誤差 | — | 🟠 1994 就存在的問題 |
| max 基準偏誤 | Dodge et al. (2019) | EVP 修正 best-of-K 高估 | — | — |
| A-1 跨語言優於最強單語 | Gao et al. 2504.11833 | 多語言 Acc@k 上界高近 10 點 | 同篇 | Vote@k 無優勢；歸因語言偏誤 |
| A-2 耦合 | Oldham (1962)、Blomqvist (1977) | 差值對基線迴歸產生假負斜率 | — | — |
| A-4 兩平點非常數 | IntHout et al. (2016) | 高異質性應報預測區間 | — | — |
| A-5 λ≤0 | Card et al. (2020) | NLP 實驗普遍檢定力不足 | — | — |
| **has_english 效果** | **Wendler et al. (ACL 2024)** | 模型在英語為中心的概念空間運算 | Global-MMLU (Singh 2024) | 🟡 翻譯偏差可能是部分原因（但無法解釋模型間 7 倍差異） |
| **has_english 的資料集梯度** | **Chua et al. (2024)** | 知識在語言間不共享 | — | 知識題比數學題更需要英文 |
| B-2 只有 d 會動 | Gao et al. | 多語言優勢在 Acc@k 而非 Vote@k | Gao et al. | 但歸因於 judge 語言偏誤 |
| B-4 recovery ≈ 0.34 | **Gao et al.（反推）** | 0.329–0.415 | Song et al. (ICLR 2025) | relative GV-Gap 隨 flops 上升（分母不同） |
| recovery 可被推動 | Song et al. Fig.3；Weaver | verifier 容量 3.39→24.91；弱 verifier 集成達 o3-mini 級 | — | 🟠 反對「普世不變量」，但用外部 verifier |
| per-query router 難做 | Cruz et al. (2018) | DES 的實際效能離 oracle 很遠 | — | — |
| 聚合器達不到 oracle | Brown et al. (2024) | coverage 冪次成長，但多數決飽和 | — | — |
| MAD 沒有特殊性 | Smit et al. (ICML 2024) | 對齊 prompt 與預算後 MAD 輸單一 agent | — | — |
| recovery 不高 | Huang et al. (ICLR 2024) | response 數對齊時 MAD 輸給 SC | — | — |
| C 改寫在 CSQA/TQA 失敗 | Sprague et al. (2025) | CoT 主要在數學與符號推理有效 | — | — |
| 混語疑慮 | Zhang et al. (EMNLP 2023) | 多語 LLM 在 code-switching 表現不佳 | — | 方向對我們保守 |

---

# 3. 建議的方向

## 3.1 定位

**從「提出方法」轉向「把 ensemble diversity 理論搬到 LLM 推理，並處理三個結構差異」。**

## 3.2 三個 RQ（見 §0）

## 3.3 生成端：五個軸 + 一個對照軸

> **Agent A（錨點）**：原始英文題 + 完整 CoT + T=0，全實驗共用

| 軸 | Agent B | 變量 | 資料集 | 現況 |
|---|---|---|---|---|
| **L 語言** | 翻成 L + CoT + T=0 | ZH,JA,RU,ES | 全 4 個 | ✅ |
| **S 採樣** | 同題 + T>0、換 seed | T=0.7/1.0/1.3 | 全 4 個 | ⬜ |
| **R 推理強度** | direct/short CoT/token 截斷 | 3 級 | MathQA,MMLU | 🟡 |
| **P 角色** | persona prompt | 2 級 | 全 4 個 | ⬜ |
| **W 改寫** | 改寫題/干擾注入 | 2 級 | MathQA,MMLU | 🟡 |
| **K 候選數**（對照） | 逐步加入候選 | K=2,3,4,5 | MathQA,MMLU | ⬜ |

**🔴 K 軸是唯一「只改 diversity 不改個體能力」的軸**，正是 Wood et al. Fig.14 的乾淨情況。

## 3.4 聚合端：五個聚合器

| 代號 | 聚合器 | K | recovery 性質 | 用途 |
|---|---|---|---|---|
| V2 | Vote@2（隨機破結） | 2 | **恆等於 0** | 尺規零點 |
| Blind | 固定選錨點 | 2 | `2w_A−1`，可為負 | **部署基準** |
| **Judge** | 單次，看完整推理 | 2 | 待測 | **主臂** |
| Debate | IMSR | 2 | 0.34 | 既有資料 |
| Vote@3/5, Judge@5 | — | 3,5 | 有理論值 | **校準測試** |

**⚠️ K=2 的 Vote 退化（recovery ≡ 0）→ 校準必須 K ≥ 3。**

## 3.5 題目端：難度分層

**已建立基礎設施（0A-5）**，基底用 **cross_model**（所有配對同一基底，無汙染）。
Stage 2 的五軸比較會遇到同樣問題，直接沿用。

## 3.6 🆕 `has_english` 的處理

**這是一個支線發現，值得半頁，不值得兩週。**

已排除的解釋：

| 解釋 | 證據 |
|---|---|
| prompt 語言一致性 | 你的設計對稱（各 agent 用自己語言的 prompt） |
| 題目難度組成 | Δq 方向相反、量級差 17 倍 |
| 配對不對稱性 | 組內 β = +0.02/−0.04 |
| 生成端差異 | d、c、m 三者差異都在雜訊裡 |
| 翻譯損失（唯一原因） | 四模型共用同一批翻譯，但效果差 7 倍 |

**仍未排除**：翻譯損失作為**部分**原因；知識 vs 推理任務的梯度。

**決定性實驗（回譯法）**：

```
英文原題 → 翻成日文 → 翻回英文 = 「英文但被翻譯損傷」
比較：基準(英文原題+日文) / 回譯(英文回譯+日文) / 對照(中文+日文)

回譯 ≈ 基準  → 是「英文」這個語言 → 英語中心解釋成立
回譯 ≈ 對照  → 是原文品質 → 翻譯損失解釋成立
```

成本約 $5，2 格（CSQA + MathQA，效果最大與最小）。

## 3.7 對部署的結論

| | d | c/2 | recovery | recovery_blind | **excess** |
|---|---|---|---|---|---|
| 含英文 | 0.180 | 0.388 | 0.456 | **0.220** | **+1.20pp** |
| 不含英文 | 0.191 | 0.384 | 0.366 | **0.065** | **+1.73pp** |

> **「配英文比較好裁決，但不值得配」** —— 只有拆成三因子才說得出來。

## 3.8 標題方向

```
主推  Where Does the Gain Come From?
      Decomposing Test-Time Diversity in LLM Reasoning
```

---

# 4. 完整邏輯流程

## §1 Introduction

| 段落 | 主張 | 支撐 | 該來源的論點 |
|---|---|---|---|
| 開場 | test-time scaling 方法只報最終準確率 | Brown et al. (2024) | coverage 冪次成長但多數決飽和 |
| 承接 | 古典 ensemble 已有 ambiguity-effect | **Wood et al. (2023)** | ensemble 增益 = Oracle − Average |
| 差異 | LLM 有三個結構差異使該量需再拆 | §3.6 | — |
| 貢獻 | (1) 觸發式聚合下的三因子分解 (2) 五軸受控比較 (3) 成本判準 | — | — |

## §2 Related Work（五小節，引用 55+）

```
2.1 Ensemble diversity（古典）
    Krogh & Vedelsby → Kuncheva & Whitaker → Brown & Kuncheva → Wood et al.
    Breiman / Dietterich / Wolpert / Kuncheva 教科書 / Lam & Suen
    結尾：三個結構差異
2.2 Test-time diversity in LLMs
    SC / MAD(Du) / MAD(Liang) / Self-Refine / MoA / ReConcile / DiVeRSe /
    Ask Me Anything / Universal SC / LLM-Blender
2.3 Cross-lingual reasoning as a diversity source
    Gao / AutoCAP / CLP / XLT / Cross-ToT / MGSM / mGRPO / Wendler / Chua / Global-MMLU
2.4 Coverage, verification, selection
    Brown / Song / Weaver / Yue / Cruz(DES) / RouteLLM / RouterBench  ← 解決 R3
2.5 Measurement practice
    Dodge / Tang / Card / Dror / Schaeffer / Oldham 系列
```

## §3 The Decomposition

### §3.1 一般式
`Gain = (1−d)·Δ_agree + d·(c−m)·recovery`
**明確標註**：`Gain` = Wood et al. 的 ambiguity-effect；`c` = Kuncheva 的 oracle accuracy
d 定義為「不全相同」的不變性條件；`c − m = c(1 − r/K)`

### §3.2 四個假設與診斷
| 假設 | 診斷 | 現況 |
|---|---|---|
| 單一正解 | — | 滿足 |
| 一致時 no-op | 逐題比對 | ⬜ 0A-2 |
| 聚合器封閉 | acc_agg ≤ c | ✅ legacy 24/24 |
| K=2 | 用一般式 | — |

### §3.3 Max-baseline、Blind 與 skill
```
excess = Gain − gap/2 = d(c/2)(recovery − recovery_blind)
recovery_blind = 2w_A − 1 = gap/(d·c)
gap* = d·c·recovery
```
**支撐**：Krogh & Vedelsby（只保證贏平均）、Oldham/Blomqvist、Dodge et al.

### §3.4 為什麼 LLM 設定需要拆得更細
引 §3.6 的三個結構差異

## §4 Experiments

| 小節 | 內容 | 資料 |
|---|---|---|
| 4.1 | 五軸的六個量（難度分層後） | 臂 1 |
| 4.2 | 陽性對照：換聚合器 | 臂 2 |
| 4.3 | K 軸（不改變個體能力） | 臂 3 |
| 4.4 | K>2 校準 + r 診斷 | 臂 3 |
| 4.5 | 外部驗證：Gao et al. 重算 | 臂 5 |
| 4.6 | 外部方法：開源 MAD | 臂 4 |

## §5 Analysis

| 小節 | 主張 | 支撐 |
|---|---|---|
| 5.1 | 各軸 H 差異大、recovery 等價（若成立） | §4.1 + TOST |
| 5.2 | H 拆成 d 與 (c−m) | §4.1 |
| 5.3 | **基準選擇決定「diversity 有沒有用」的結論** | B-7 + A-2 |
| 5.4 | 兩平條件與各文獻門檻 | §3.3 + Gao |
| 5.5 | **聚合效率依賴候選表面形式**（has_english） | 0A-1 + 0A-5 + 回譯 |
| 5.6 | Robustness：耦合、EIV、split-half、異質性、λ | 0A-4 + A-5/A-7 |

**§5.3 寫法**：
> 「diversity 與效能的關聯薄弱（Kuncheva & Whitaker, 2003），一個被忽略的原因是基準選擇。相對候選平均時分歧率與增益強相關（r=0.74）；相對最強候選時反轉為負（−0.38）。兩者的差恰好是 gap/2。」

**§5.5 寫法**：
> 「在古典 ensemble 中 combiner 只見標籤，候選的表面形式不可能影響聚合效率。我們觀察到含英文的配對 recovery 高 0.093（格內，p<0.0001），而生成端三量（d, c, m）無差異；難度組成的方向相反且量級差 17 倍，可排除。逐模型拆解顯示效果從 +0.023（QWEN3）到 +0.162（Gemini），而四個模型共用同一批翻譯，因此翻譯品質不足以解釋。」

## §6 Implications
```
6.1 對 test-time diversity 方法的意涵
6.2 對跨語言推理方法的意涵（可點名 Gao / AutoCAP / CLP）
6.3 報告建議：請同時報 (d, c, recovery, recovery_blind)
```

## §7 Limitations
- recovery MDE ±0.10，只能宣稱等價
- 硬邊界：需要單一正解
- 錨點只用一組
- R/W 軸只在 MathQA + MMLU
- Kuncheva & Whitaker：古典設定下 diversity 與效能關聯就弱
- Song et al. / Weaver：外部 verifier 會推動 recovery，結論限自我聚合
- has_english 的「英文 = 原文」共線尚未完全分離（回譯實驗）

---

# 5. 待辦清單與投稿機率

## Stage 0A｜已完成 ✅

| # | 項目 | 狀態 |
|---|---|---|
| 0A-1 | recovery_blind 分析 | ✅ 完成，D0 通過，發現 has_english |
| 0A-4 | Split-half | ✅ 完成，耦合偏誤小 |
| 0A-5 | 難度分層 | ✅ 完成，組成解釋排除 |

## Stage 0A'｜本週剩下（零成本）

| # | 動作 | 成本 | 為什麼 |
|---|---|---|---|
| 1 | ✅ prompt 語言檢查（已完成，排除） | — | — |
| **2** | **逐語言拆開**（has_EN/ZH/RU/ES，日文參考組） | 半天 | 若只有英文有效果，英語中心解釋更強 |
| 3 | 兩組解析失敗率比較 | 半小時 | 排除「英文好解析」 |
| **4** | **撤掉 −0.412** | 半小時 | 它是偏誤最大的估計 |
| 5 | 0A-2 逐題檢查 Δ_agree 與封閉性 | 半天 | §3.2 的一行 |
| 6 | 版本 D（split-half + EIV），重算 λ' | 半天 | −0.361 與 −0.243 目前不可比 |
| 7 | recovery 異質性的 bootstrap CI | 半天 | τ=0.042 的不確定性 |
| 8 | Table 1 三格標錯修正；Table 2 重算 | 半天 | 正確性 |
| 9 | G-4：legacy JSON 對回 2000 題重新計分 | 半天 | 救回 DeepSeek/QWEN |

**⚠️ 逐語言拆開的設計**：每配對恰好 2 語言，五個 dummy 和恆為 2，與截距共線。**必須挑一個當參考組**（建議日文），放另外四個。

## Stage 0B｜外部資料（零成本，7–10 天）

| # | 待辦 | 產出 |
|---|---|---|
| 0B-1 | 🔴 **Gao et al. 重算** | 5 種多樣性來源的 d,c,m,recovery |
| 0B-2 | 🔴 **搜尋 `Gain/(Oracle−Average)` 有無既有名稱** | novelty 確認 ❓ |
| 0B-3 | **讀 Wood et al. §1/§2/§4.2/§6.2** | §2/§3 的承接寫法 |
| 0B-4 | Weaver 重算 | §7 Limitations |
| 0B-5 | "Reuse of public artifacts" 小節 | Reproducibility 1→4 |

### 決策點 D1
```
0B-2 若該比值已有名稱 → RQ1 殘餘 novelty 只剩「觸發式聚合下的 d 拆分」
                       → §3 份量壓縮，重心全移到 RQ2/RQ3
```

**Stage 0 完成後**：P(main) **14%** / P(≥Findings) **52%**
（比 v6 微升，因為 has_english 是一個真實發現）

## Stage 1｜Pilot（~$5，3 天）

`1 模型 × 2 資料集 × 4 軸 × Judge`

| # | 檢查 | 門檻 |
|---|---|---|
| ① | 各軸 d 範圍重疊 | 兩兩至少 0.05 |
| ② | 難度分層後每軸每層樣本 | ≥ 100 |
| ③ | 解析失敗率差異 | < 1% |
| ④ | recovery MDE | ≤ 0.10 |
| ⑤ | 各軸 recovery_blind 差多少 | 決定分層必要性 |

## Stage 2｜🔴 五軸主實驗（~$45，4 週）← 這是論文

| # | 待辦 | 成本 | 順序理由 |
|---|---|---|---|
| 2-3 | **陽性對照**：L 軸 × {V2,Blind,Judge,Debate} | $3 | **先跑**，證明 recovery 有辨別力 |
| 2-4 | **K 軸**：K=2,3,4,5 | $10 | 唯一不改變個體能力的軸 |
| 2-1 | S 採樣軸 × Judge | $10 | 最便宜 |
| 2-2 | P 角色軸 × Judge | $10 | 一個 prompt |
| 2-5 | R 推理強度軸 × Judge | $12 | 已有單語 |
| 2-6 | W 改寫軸（CSQA 用干擾注入） | $8 | 可選 |
| 2-7 | 錨點輪換敏感度 | $2 | 回應「錨點只有一組」 |
| 2-8 | 難度分層 + TOST | $0 | headline |
| 2-9 | d–H 曲線 | $0 | 多樣性品質 |

**每格必記七個數字**：`acc_A`、`acc_B`、`d`、`c`、`m`、`recovery`、`recovery_blind`（+ token 成本）。前六個不需跑聚合器。

### 決策點 D2
```
2-3 若換聚合器時 recovery 也不動 → 🔴 指標可能無辨別力，停下來
```

**Stage 2 完成後**：P(main) **33%** / P(≥Findings) **73%**

## Stage 3｜K>2 校準（~$20，1 週）

Vote@3/5 實測 vs 理論 recovery；`r` 診斷；recovery vs K 曲線

**完成後**：P(main) **43%** / P(≥Findings) **79%**

## Stage 4｜外部方法（~$20，3 週）

開源 MAD 重跑（Du et al. 或 Liang et al.）；核心 figure（recovery 分布圖）

**完成後**：P(main) **59%** / P(≥Findings) **84%**

## Stage 5｜支線（可砍，$5）

| # | 待辦 | 可砍嗎 |
|---|---|---|
| 5-1 | **回譯實驗**（分離「英文」vs「原文」） | ✅ 可砍，但做了 §5.5 會強很多 |
| 5-2 | Pilot/holdout 預測分析 | ✅ 可砍（模擬顯示訊噪比太差） |

⚠️ **pilot 預測的誠實評估**：n=100 時 excess 估計 SD 是 4–7pp，效果量只有 ±1.5pp；要可靠需 pilot ≈ 2700 題（比資料集還大）；且整個決策的頭空間只有 0.45pp。**只能寫成「pilot 規模與精度的關係 + 天真做法有 −3.5pp 偏誤」，不能當結論。**

## Stage 6｜寫作與開源（2 週）

**完成後**：P(main) **65%** / P(≥Findings) **87%**

## 機率總表

| 階段 | 累積時間 | 累積成本 | P(main) | P(≥Findings) |
|---|---|---|---|---|
| v3 原稿 | — | — | 5% | 30% |
| Stage 0 | 3 週 | $0 | 14% | 52% |
| + Stage 1–2 | 8 週 | $50 | 33% | 73% |
| + Stage 3 | 9 週 | $70 | 43% | 79% |
| + Stage 4 | 12 週 | $90 | 59% | 84% |
| **+ Stage 6** | **14 週** | **$95** | **65%** | **87%** |

### 下修分支

| 若發生 | P(main) | P(≥Findings) |
|---|---|---|
| 0B-2 發現該比值已有名稱 | 46% | 79% |
| D2 失敗（換聚合器 recovery 不動） | 10% | 35% |
| 各軸 recovery 與 H 都等價（純負向） | 39% | 73% |

## 12 月時程

```
本週         Stage 0A'（逐語言、解析率、撤 −0.412、版本 D）
9/15 – 9/30  Stage 0B（決策點 D1）；並行寫 §1, §2, §3
10/1 – 10/7  Stage 1 Pilot
10/8 – 11/4  🔴 Stage 2 五軸（2-3 → 2-4 → 2-1 → 2-2 → 2-5）
11/5 – 11/11 Stage 3
11/12 – 12/2 Stage 4；若有餘力插入 Stage 5-1 回譯
12/3 – 12/15 Stage 6
```

**⚠️ 兩個提醒**
1. **論文從第一天開始寫**（§1/§2/§3 不依賴新結果）
2. **本週確認：算新投稿還是 resubmission？** 若算 resubmission，1.5 分的 review 會跟著你

---

# 6. 完整文獻表

**圖例**：🔴 必引　🟢 建議引　⚪ 不必引

## 6.1 古典 Ensemble

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **A Unified Theory of Diversity in Ensemble Learning**<br>Wood, Mu, Webb, Reeve, Luján, Brown — JMLR 2023, arXiv 2301.03962 | 🔴 | **最危險的前作**，你的 Gain 就是他們的 ambiguity-effect | (1) Prop 7：ensemble 風險 = 平均個體風險 − ambiguity-effect，適用任何損失；(2) Thm 10：0/1 loss 下任何 combiner 的該差值必然依賴標籤；(3) Fig.14：只在單獨改 m 時 diversity 才乾淨預測效能（r² 0.99→0.59） |
| **Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy**<br>Kuncheva & Whitaker — Machine Learning 51(2), 2003 | 🔴 | **RQ2 的前車之鑑** | 十個 diversity 指標與 ensemble 效能關聯薄弱，質疑其實用性 |
| **Good and Bad Diversity in Majority Voting Ensembles**<br>Brown & Kuncheva — MCS 2010 | 🔴 | 救援/破壞分解已有名字 | `yq̄=+1` 時 diversity 減少錯誤（good），`=−1` 時增加（bad） |
| **Neural Network Ensembles, Cross Validation and Active Learning**<br>Krogh & Vedelsby — NIPS 1994 | 🔴 | excess vs delta 的 1994 版 | ambiguity decomposition；**只保證贏過平均，不保證贏過最強** |
| **Combining Pattern Classifiers** (2nd ed.)<br>Kuncheva — Wiley 2014 | 🔴 | `c` 的定義來源 | "oracle accuracy" 標準定義；majority vote 的理論極限 |
| **Ensemble Methods in Machine Learning**<br>Dietterich — MCS 2000 | 🟢 | §2.1 框架 | ensemble 有效的三個理由：statistical、computational、representational |
| **Bagging Predictors / Random Forests**<br>Breiman — 1996 / 2001 | 🟢 | 「製造 diversity」的經典 | bootstrap 重抽樣與特徵隨機化 |
| **Stacked Generalization**<br>Wolpert — Neural Networks 1992 | 🟢 | pilot/holdout 的古典版 | 用 held-out 資料訓練 meta-learner |
| **Application of Majority Voting to Pattern Recognition**<br>Lam & Suen — 1997 | 🟢 | recovery 的理論基準 | 個體獨立且 p>0.5 時多數決保證優於個體 |
| **Dynamic Classifier Selection: Recent Advances and Perspectives**<br>Cruz, Sabourin, Cavalcanti — Information Fusion 2018 | 🟢 | per-query router 的古典版 | DES 以 oracle 為上界，實際選擇離 oracle 很遠 |
| **On Over-fitting in Model Selection and Subsequent Selection Bias**<br>Cawley & Talbot — JMLR 2010 ❓ | 🟢 | winner's curse 的古典版 | 同一批資料做選擇與評估會產生選擇偏誤 |
| **From DCS to DES**<br>Ko, Sabourin, Britto — 2008 ❓ | ⚪ | 與 Cruz 綜述重複 | — |
| **Bias-Variance-Covariance**<br>Ueda & Nakano — 1996 | ⚪ | Wood et al. 已涵蓋 | — |

## 6.2 核心競品（LLM 側）

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **Could Thinking Multilingually Empower LLM Reasoning?**<br>Gao, X. Huang, Zhu, S. Huang, Li, Yuan — arXiv 2504.11833 | 🔴 | RQ2 最直接的競品 + recovery 的外部驗證 | (1) 已比較 Multilingual/Repeat/Paraphrase/兩種 Mix；(2) Acc@k 在 k=17 高約 8 點；(3) Vote@k 優勢消失甚至輸 Repeat；(4) 歸因於 judge 的語言偏誤 |
| **Mind the Gap**<br>Song, H. Zhang, Eisenach, Kakade, Foster, Ghai — ICLR 2025 | 🟢 | §3 方法論註記 + §7 | (1) relative GV-Gap 分母是 `1−m`，我們用 `c−m`；(2) 隨 pretraining flops 上升；(3) cross-verification：gap 3.39→24.91 |
| **Shrinking the Generation-Verification Gap with Weak Verifiers (Weaver)**<br>Saad-Falcon et al. — NeurIPS 2025, arXiv 2506.18203 | 🟢 | §7 Limitations | 加權集成弱 verifier + weak supervision，Llama 3.3 70B 達 o3-mini 級（87.7%）；蒸餾成 400M 保留 98.7%。**用外部 verifier** |
| **Large Language Monkeys**<br>Brown, Juravsky, Ehrlich, Clark, Le, Ré, Mirhoseini — 2024 | 🔴 | §1 問題背景 | coverage 冪次成長至 95%+，但多數決與 RM 在幾百樣本後飽和 |
| **Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?**<br>Yue et al. — NeurIPS 2025 | 🟢 | §6 延伸 | RLVR 提升 pass@1 但大 k 時 pass@k 不超過 base model |
| **Cross-lingual Self-Consistency for Multilingual Reasoning**<br>arXiv 2606.01464 ❓ | 🔴（待查） | 2026 年、題目高度重疊 | ❓ **必須自己讀** |

## 6.3 多語言推理

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **Do Llamas Work in English?**<br>Wendler, Veselovsky, Monea, West — ACL 2024 | 🔴 | **has_english 的機制解釋** | 模型在英語為中心的概念空間運算，非英語輸入先被映射過去 |
| **AutoCAP**<br>Y. Zhang, Chen, Li, Che, Qin — Findings ACL 2024 | 🔴 | R2 點名 | 自動語言選擇 + 自動權重分配 |
| **mGRPO**<br>OpenReview ❓ | 🔴 | R2 點名 | 不限制語言時最好；GRPO 四 benchmark +7.5% |
| **Cross-Lingual Prompting (CLP)**<br>Qin, Chen, Wei, S. Huang, Che — EMNLP 2023 | 🔴 | 跨語言 SC 代表 | 跨語言對齊 + self-consistent prompting |
| **Not All Languages Are Created Equal (XLT)**<br>H. Huang et al. — Findings EMNLP 2023 | 🔴 | 已引用 | cross-lingual-thought 模板導向英語推理 |
| **Language Models are Multilingual CoT Reasoners (MGSM)**<br>Shi, Suzgun, Freitag, X. Wang et al. — ICLR 2023 | 🔴 | R3 點名 | 多語 CoT benchmark；translate-to-English 強 |
| **Crosslingual Capabilities and Knowledge Barriers**<br>Chua et al. — 2024 | 🔴 | **has_english 的資料集梯度** | 知識在語言間不共享，存在跨語言知識屏障 |
| **Global-MMLU**<br>Singh et al. — 2024 | 🔴 | **翻譯偏差的替代解釋** | 機器翻譯評測的偏差與文化敏感題目 |
| **Cross-lingual Consistency of Factual Knowledge**<br>Qi, Fernández, Bisazza — EMNLP 2023 | 🟢 | 同受耦合影響 | RankC 指標 |
| **Language Imbalance Can Boost Cross-Lingual Generalisation**<br>Schäfer et al. — 2024 | 🟢 | **方向相反** | 訓練期語言不均衡反而提升泛化 |
| **Do Multilingual LMs Think Better in English?**<br>Etxaniz et al. — NAACL 2024 | 🟢 | self-translate | 自我翻譯成英語後表現更好 |
| **Cross-ToT**<br>Ranaldi et al. | 🟢 | 另一跨語言聚合 | 跨語言 ToT |
| **Multilingual LLMs Are Not (Yet) Code-Switchers**<br>R. Zhang et al. — EMNLP 2023 | 🟢 | 回應 R3 混語質疑 | 多語 LLM 在 code-switching 表現不佳 |
| **Do Multilingual LLMs Think in English?**<br>Schut et al. — 2025 ❓ | 🟢 | 與 Wendler 同方向 | 表徵層證據 |
| **MAPO / mCoT** | ⚪ | 訓練期方法 | — |

## 6.4 LLM 聚合器與多 agent

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **Self-Consistency Improves CoT**<br>X. Wang et al. — ICLR 2023 | 🔴 | 校準測試對象 | 多次採樣取多數決 |
| **Improving Factuality and Reasoning through Multiagent Debate**<br>Du, Li, Torralba, Tenenbaum, Mordatch — ICML 2024 | 🔴 | Stage 4 重跑對象 | 多 LLM 實例並行提案、多輪互看修正 |
| **Encouraging Divergent Thinking through MAD**<br>Liang et al. — EMNLP 2024 | 🔴 | 他們的解法就是拉高 d | Degeneration-of-Thought；異議 agent + judge |
| **Should We Be Going MAD?**<br>Smit et al. — ICML 2024 | 🔴 | 同方向盟友；敘事範本 | 對齊 prompt 與預算後 MAD 贏不過單一 well-prompted agent |
| **LLMs Cannot Self-Correct Reasoning Yet**<br>J. Huang et al. — ICLR 2024 | 🔴 | recovery 是其量化 | response 數對齊時 MAD 輸給 SC |
| **Self-Refine**<br>Madaan et al. — NeurIPS 2023 | 🔴 | R3 點名；Δ_agree≠0 案例 | 單模型自我回饋迭代 |
| **Debating with More Persuasive LLMs**<br>Khan et al. — ICML 2024 | 🔴 | recovery 可獨立操作 | judge 與 debater 相對強度是關鍵變數 |
| **ReConcile**<br>J. Chen, Saha, Bansal — 2024 | 🟢 | 多樣性來源 = 模型 | 圓桌討論 + 信心加權投票 |
| **Mixture-of-Agents**<br>J. Wang et al. — 2024 | 🟢 | 相對最佳單一模型 | 分層聚合 |
| **LLM-Blender**<br>D. Jiang, Ren, Lin — ACL 2023 | 🟢 | oracle best-of-N | pairwise ranking + fusion |
| **Universal Self-Consistency**<br>X. Chen et al. — 2023 | 🟢 | recovery 換成 LLM judge | 用 LLM 選最一致的回應 |
| **DiVeRSe**<br>Y. Li et al. — ACL 2023 | 🟢 | 多樣性來源 = prompt | 多樣 prompt + verifier 加權 |
| **Ask Me Anything**<br>Arora et al. — ICLR 2023 | 🟢 | 最便宜的對照軸 | 多 prompt 格式 + 弱監督聚合 |
| **ChatEval**<br>Chan et al. — ICLR 2024 | 🟢 | `c` 的定義邊界 | 多 agent 辯論做評估 |
| **Why Do Multi-Agent LLM Systems Fail? (MAST)**<br>Cemri et al. — 2025 ❓ | 🟢 | 救援/破壞的質性版 | 失效模式分類 |
| **Reflexion**<br>Shinn et al. — NeurIPS 2023 | 🟢 | 原論文 SR baseline | 口語強化學習 |

## 6.5 Verifier 與 Routing

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **Training Verifiers (GSM8K)**<br>Cobbe et al. — 2021 | 🟢 | verifier 路線起點 | 訓練 verifier 排序候選 |
| **Let's Verify Step by Step**<br>Lightman et al. — ICLR 2024 | 🟢 | 提升 recovery 的路線 | 過程監督優於結果監督 |
| **Scaling LLM Test-Time Compute Optimally**<br>Snell et al. — 2024 | 🟢 | RQ3 背景 | 依難度分配 compute |
| **RouteLLM**<br>Ong et al. — 2024 | 🟢 | per-query 選擇 | 偏好資料訓練 router |
| **RouterBench**<br>Q. Hu et al. — 2024 | 🟢 | oracle router 受雜訊灌水 | routing 評測 |
| **LLM Routing with Benchmark Datasets**<br>Shnitzer et al. — 2023 | 🟢 | 事後選擇偏誤 | 用 benchmark 為每任務選模型 |
| **FrugalGPT**<br>L. Chen, Zaharia, Zou — 2023 | 🟢 | 成本層前例 | 級聯降低成本 |
| **Hybrid LLM**<br>Ding et al. — ICLR 2024 | ⚪ | 與 RouteLLM 重複 | — |
| **Self-Improvement: The Sharpening Mechanism**<br>A. Huang et al. — ICLR 2025 | ⚪ | 理論取向，距離遠 | — |

## 6.6 評測方法論

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **Show Your Work**<br>Dodge, Gururangan, Card, Schwartz, Smith — EMNLP 2019 | 🔴 | max-baseline 偏誤的既有工具 | EVP：把 best-of-K 畫成 K 的函數，去除高估 |
| **Showing Your Work Doesn't Always Work**<br>Tang et al. — ACL 2020 | 🟢 | EVP 無偏版本 | EVP 估計量本身有偏 |
| **With Little Power Comes Great Responsibility**<br>Card et al. — EMNLP 2020 | 🔴 | A-5 的框架 | NLP 實驗普遍檢定力不足 |
| **Hitchhiker's Guide to Testing Statistical Significance in NLP**<br>Dror et al. — ACL 2018 | 🔴 | 回應 R3 | 檢定選擇流程與誤用 |
| **Are Emergent Abilities a Mirage?**<br>Schaeffer, Miranda, Koyejo — NeurIPS 2023 | 🔴 | **敘事範本** | 「湧現」是不連續 metric 造成的假象 |
| **Accounting for Variance in ML Benchmarks**<br>Bouthillier et al. — MLSys 2021 | 🟢 | 支撐 I²=96% | 變異來源分解 |
| **On the State of the Art of Evaluation in Neural LMs**<br>Melis, Dyer, Blunsom — ICLR 2018 | 🟢 | winner's curse 先例 | 控制調參預算後差距縮水 |
| **Equivalence Testing (TOST)**<br>Lakens — SPPS 2017 | 🔴 | 主結論是「證明相同」 | 兩個單邊檢定 |
| **Quantifying Heterogeneity (I²)**<br>Higgins & Thompson — Stat Med 2002 | 🔴 | I² 定義 | — |
| **Measuring Inconsistency in Meta-Analyses**<br>Higgins et al. — BMJ 2003 | 🟢 | 25/50/75% 基準 | — |
| **Plea for Prediction Intervals**<br>IntHout et al. — BMJ Open 2016 | 🔴 | 兩平點非常數 | 高異質性應報預測區間 |
| **Fixed-Effect and Random-Effects Models**<br>Borenstein et al. — RSM 2010 | 🟢 | 隨機效果彙總 | — |
| **A Note on the Analysis of Change**<br>Oldham — J Chronic Dis 1962 | 🔴 | 耦合的數學基礎 | 差值對基線迴歸產生假負斜率 |
| **On the Relation Between Change and Initial Value**<br>Blomqvist — JASA 1977 | 🟢 | 閉式校正 | — |
| **Revisiting Change and Initial Value**<br>Tu & Gilthorpe — Stat Med 2007 | 🟢 | coupling 回顧 | — |
| **Correcting for Regression Dilution Bias**<br>Frost & Thompson — JRSS-A 2000 | 🟢 | λ 校正 | 用信度比校正 |
| **Mismeasured Variables**<br>Hausman — JEP 2001 | ⚪ | 非技術導論 | — |

## 6.7 實驗設計依據

| Paper | 引用? | 理由 | 要引用的論點 |
|---|---|---|---|
| **LLMs Can Be Easily Distracted by Irrelevant Context**<br>Shi et al. — ICML 2023 | 🔴 | **CSQA 的替代旋鈕** | 注入不相關脈絡降低推理，格式不變 |
| **GSM-Symbolic**<br>Mirzadeh et al. — ICLR 2025 | 🔴 | 改寫的先例 | 符號化改寫移動準確率 |
| **FormatSpread**<br>Sclar et al. — ICLR 2024 | 🟢 | format confound | 格式微調造成巨大變動 |
| **Rethinking the Role of Demonstrations**<br>Min et al. — EMNLP 2022 | 🟢 | 分級破壞範本 | — |
| **To CoT or Not to CoT?**<br>Sprague et al. — 2025 | 🔴 | CSQA/TQA 的設計邊界 | CoT 主要在數學與符號推理有效 |
| **Chain-of-Thought Prompting**<br>Wei et al. — NeurIPS 2022 | 🔴 | 基礎 | — |
| **MMLU-Pro**<br>Y. Wang et al. — 2024 | 🟢 | 已引用 | — |
| **MMLU / CommonsenseQA / TruthfulQA / MathQA** | 🔴 | 資料集 | — |
| **Semantic Uncertainty**<br>Kuhn et al. — ICLR 2023 | ⚪ | 不做不確定性偵測 | — |

---

# 附錄 A：v7 推翻或修正的舊宣稱

| 舊宣稱 | 現況 |
|---|---|
| 「分解成三因子會讓 diversity 與效能關係變乾淨」 | ❌ **推翻**。H 的 r²=52% 沒比 d 的 54% 好（c 幾乎是常數） |
| 「耦合造成大量假斜率」 | ⚠️ **弱化**。split-half 只動 0.022 |
| 「split-half 會落在 −0.20~−0.25」 | ❌ **未達成**。pooled 是 −0.361 |
| 「−0.412 是 pooled OLS」 | ❌ **標籤錯誤**。那是 per-cell 平均，且偏誤最大，應撤除 |
| 「平均 excess +1.46pp」 | ⚠️ 實際 +1.518pp |
| 「95% 預測區間 [−0.77,+4.94]」 | ⚠️ 我算 [−0.43,+5.24]，結論同（含 0） |
| 「逐題配對是最好的組成控制」 | ❌ **推翻**。交集僅 3%，SE(recovery)=0.13–0.23，遠超需要的 0.035 |
| 「recovery 是純聚合器性質」 | ⚠️ **部分推翻**。不受配對不對稱性影響 ✅，但受 `has_english` 影響 +0.093 |

# 附錄 B：三句話總結給教授

> 1. 原論文是一個多語言辯論方法，只影響 6 篇跨語言 CoT，value 質疑成立。
>
> 2. 我們的 Gain 分解對應古典 ensemble 的 ambiguity-effect（Wood et al., JMLR 2023），不能宣稱原創。但 LLM 有三個結構差異——聚合器只在分歧時觸發（`d` 是成本變數）、候選帶完整推理（**表面形式會影響聚合效率，我們量到 +0.093**）、聚合器就是生成者本身。
>
> 3. 核心問題是：五種多樣性來源在控制題目組成後是否給出相同的 c 與 recovery，以及取得多樣性的代價。這兩題在古典 ensemble 沒有對應物，因為古典成員成本相同且 combiner 永遠動作。
