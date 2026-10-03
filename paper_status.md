# 論文現況總整理：從 IMSR 到多樣性聚合的決策框架

Oct 4, 2026 · @MJLee

## 摘要

原本的 IMSR 方法論文被評為 novelty 不足、因果不明；我們把論文改成「什麼時候值得做多樣性聚合」的決策框架。現有結果已足以撐起一篇分析論文，若 probe 預測實驗成功，可升級為方法加分析論文。

| 項目 | 現況 |
| --- | --- |
| 原論文 | IMSR：跨語言辯論 + 語言配對路由；上一輪 Overall 2.5 / 2 / 1.5 |
| 新論文核心式 | Excess = d·(c−m)·(recovery − recovery\_blind)；聚合值得 ⟺ recovery > recovery\_blind |
| 已有最強結果 | 基準一換，聚合勝率 76% → 34%；Debate ≈ Judge 但聚合端貴 5–6 倍；同語言重抽的淨效果勝過換語言；自我修正失敗在「很少改答案」 |
| 最大缺口 | RQ1 的 probe → target 預測尚未做；Gemini 答案截斷、Judge off-menu 尚未修 |
| 錄取機會（判斷，非計算） | 現在 Findings 25–35%；修好資料 40–50%；RQ1 成功 55–65%，Main 20–30% |
| 截止日 | 以 2026-12-15 估算，須自行確認該 cycle 的實際日期 |

本文件的數字分兩批資料：「舊分析」來自跨語言 160 對與 legacy 同語言 24 格；「新實驗」來自 aggregation\_cells.csv（GPT-4o mini、Qwen、DeepSeek，Gemini 暫時排除）。兩批不可直接混比。

## 一、最初的困境

原論文是一個增益不大、又沒有解釋為什麼有效的方法論文，三位審稿人分別從 novelty、因果、透明度三個方向否定它。

### 原論文做了什麼

- **IMSR**：同一個 LLM 用兩種語言各答一次；答案不同就進行最多三輪跨語言辯論，仍不一致則由 Judgment Agent 裁決
- **動態路由**：XLM-RoBERTa 多標籤分類器，從 5 種語言（EN、ZH、JA、RU、ES）的 10 個配對中挑出最可能修正成功的一對
- **實驗**：4 模型 × 4 資料集（MathQA、MMLU、TruthfulQA、CommonsenseQA），宣稱 16 格中 12 格最佳且 p < 0.05

### 三位審稿人的批評

| 審稿人 | Soundness / Excitement / Overall | 核心批評 |
| --- | --- | --- |
| R1（信心 4） | 2.5 / 2.5 / 2.5 | W1 缺同語言雙 agent 對照組，增益可能來自辯論結構而非換語言；W2 五種語言的選擇沒有理由；W3 router 的泛化範圍未討論，「plug-and-play」不成立 |
| R2（信心 4） | 2 / 2.5 / 2 | novelty 有限，只是 cross-lingual prompting、self-reflection、debate、routing 的組合；漏引 AutoCAP、mGRPO；router 需要每題跑遍所有配對，並不輕量；未驗證新語言 |
| R3（信心 5） | 2 / 1.5 / 1.5 | 只有 17 篇引用，§2.4 沒有引用；統計顯著性不透明（跑幾次、報平均或最大、p 值怎麼算）；混語辯論 prompt 不穩；best fixed pair 的原因沒解釋；SR 與 IMSR 輪數不同，增益可能只是來自更多計算 |

Reproducibility 分別是 3 / 1 / 2，Software 三位都給 1。

### 根本問題

1. 增益小（多數格子不到 2 個百分點），而且沒有說明為什麼有效
2. 「換語言」是否必要從未被隔離出來
3. 原論文自己的 Table 1 就顯示 EN-SR 在 4 格勝過 IMSR，主張前後不一致
4. 沒有任何一個可以被推翻的假設

### 轉向歷程

| 階段 | 論文是什麼 | 為什麼被換掉 |
| --- | --- | --- |
| 投稿版 | IMSR 方法論文 | 審稿人：novelty、因果、透明度 |
| v7 | Gain 分解 + 比較五種多樣性來源的 recovery 是否相等（TOST 等價檢定） | 教授問「實驗在驗證什麼 assumption」；recovery 的 MDE 是 ±0.09，等價宣稱做不到 |
| 教授第二次提問 | 1a / 1b / 1c 是否都只是 cross validation | 隨機切分的 train / test 同分布，實驗保證成功，不構成檢驗 |
| 現在 | 決策框架：何時值得聚合、少量樣本能否預測、瓶頸在 generator 還是 aggregator | — |

教授兩次提問其實是同一個問題：**什麼結果會讓我們承認自己錯了？** 新方向的每一個主張都必須回答這一點。

## 二、現有實驗結果與相關文獻

新實驗有八項結果，其中「多樣性多不等於賺多」「Debate ≈ Judge」「同語言控制組」三項最穩；舊分析約一半的宣稱已撤銷。先看資料品質，再看結果。

### A. 資料品質

| 項目 | 數據 | 影響 | 處理 |
| --- | --- | --- | --- |
| Gemini 答案截斷 | 解析失敗平均 7.2%、最高 20%（其他模型 < 0.1%）；Spearman(輸出長度, 失敗率) = 0.95；分歧題中 57% 其實是一邊沒有答案 | Gemini 暫時排除；Gemini 每格「最強 path」都是 SR，是截斷造成的假象 | 只對失敗的約 7% 題目調大 max\_tokens 重跑 |
| Judge off-menu | GPT-4o mini 1.9%、Qwen 2.8%，四個資料集都有 | Judge 不是純選擇器，recovery 混入「自己重新解題」 | Judge 改為輸出 A 或 B |
| 恆等式 | Blind 的 recovery\_H2 − recovery\_blind\_H2 誤差 1×10⁻¹⁶；Gain 恆等式誤差 0 | 計算正確 | — |
| split-half | EN 在全樣本比 S1 高 0.2–1.7pp，但 recovery\_blind\_H2 ≈ 0.004 | 贏家詛咒有被消掉 | — |

### B. 新實驗的八項結果

以下為 Judge、both\_answered 子集、GPT-4o mini + Qwen（新來源只有這兩個模型），語言軸另含 DeepSeek。

#### 結果 1：基準決定結論

跟配對內較強者比，聚合勝出 76%；跟該格 14 條 path 中最強者比，只勝出 34%。最強單一 path 在 8 格中有 6 格是 persona。34% 偏低，因為全域最強是在同一批題目上從 14 條挑出的。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Kuncheva, *Combining Pattern Classifiers*（2014） | 佐證 | single best 與 oracle 是評估組合方法的兩個參考點，single best 應在驗證資料上選出 |
| Cawley & Talbot（JMLR 2010） | 警告 | 在同一批資料上選擇又評估會產生樂觀偏誤，可大到與方法間差異相當 → 34% 需用 split 重算 |
| Zheng et al., *Is "A Helpful Assistant" the Best Role*（Findings EMNLP 2024） | 反對 | 162 種角色下 persona 不比不加好，效果近乎隨機 → 我們的 persona 6/8 最強可能是贏家詛咒 |

#### 結果 2：決策不退化

配對內基準下 81% 該聚合、19% 不該，29% 落在 ±0.5pp 以內；不對稱配對 77%、對稱配對 91%。先前只有語言軸時是 92 / 8。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Geifman & El-Yaniv, *Selective Classification*（NeurIPS 2017） | 佐證 | 決定「要不要動用較貴處理」的規則，必須沿整條覆蓋範圍評估，不能只看單一操作點 |

#### 結果 3：多樣性多，不等於賺多

語言（EN+ZH）的 headroom 8.4pp，Excess 只有 0.45pp；採樣（EN+S1）的 headroom 4.1pp，Excess 卻有 1.14pp。差別在 recovery\_blind：0.27 對 0.00。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Kuncheva & Whitaker（Machine Learning 2003） | 佐證 | 十種多樣性度量與集成準確率的關聯都很弱；我們補上原因：多樣性只決定 headroom |
| Wood et al., *A Unified Theory of Diversity*（JMLR 2023） | 佐證 | diversity 不是可獨立調高的參數，它依賴標籤與個體能力 |
| Gao et al., *Could Thinking Multilingually Empower LLM Reasoning?*（2025） | 相容 | 多語言抬高 Acc@k 上界，但 Vote@k 優勢消失；與我們一致，我們提供分解式的讀法 |
| Wang et al., *Self-Consistency*（ICLR 2023）；Li et al., *More Agents Is All You Need*（2024） | 表面相反 | 更多樣本持續提升；但他們比的是平均或單次，不是單一最強 |

#### 結果 4：recovery 跨多樣性來源差很多

Persona 0.16–0.22、採樣 0.24–0.26、SR 0.31–0.39、改寫 0.32–0.34、推理強度 0.33、語言 0.35–0.44。在一種來源上量到的 recovery，不能沿用到另一種。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Lai, Zhang, Nissim, *Multidimensional Consistency*（2025） | 佐證 | 推理一致性依變異維度（順序、改寫、語言）而不同 |
| Gao et al.（2025） | 佐證 | 多語言、重複取樣、改寫是表現不同的多樣性來源 |

#### 結果 5：三輪 Debate 不比單次 Judge 好

同配對同格 40 組：recovery 平均差 −0.006（SD 0.119），Debate 勝 21 / 40；聚合端輸出 tokens Judge 15–33、Debate 74–198。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Smit et al., *Should We Be Going MAD?*（ICML 2024） | 佐證 | 對齊 prompt 與預算後，多 agent 辯論沒有可靠勝過單一 agent |
| Chen, Zaharia, Zou, *Are More LLM Calls All You Need?*（NeurIPS 2024） | 佐證 | 更多 LLM 呼叫對表現可能非單調 |
| Du et al., *Multiagent Debate*（ICML 2024） | 表面相反 | 辯論提升事實性與推理；但對照組是單一回答，不是單次 Judge |
| Liang et al., *Encouraging Divergent Thinking*（EMNLP 2024） | 表面相反 | 角色辯論優於 self-reflection；我們比的是 Debate 與 Judge |

#### 結果 6：同語言控制組（回答 R1-W1）

都用 Debate、都以 EN 為一方：EN+ZH 的 recovery 0.375、recovery\_blind 0.265、Excess 0.53pp；EN+S1 的 recovery 0.270、recovery\_blind 0.004、Excess 1.15pp。換語言提高挑選能力，但更提高「直接用英文」這個對照，淨效果是同語言重抽勝。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Du et al.（ICML 2024） | 佐證 | 同語言、同模型的多實例辯論本身就有效 |
| Qin et al., *Cross-lingual Prompting*（2023）；Huang et al., *Not All Languages Are Created Equal*（2023） | 反對（原 IMSR 立場） | 跨語言 prompting 能提升推理；但比較基準是單語平均，不是單一最強 |
| Wendler et al., *Do Llamas Work in English?*（2024） | 機制 | 模型內部以英文為樞紐表徵，解釋英文候選為何系統性較強 |

#### 結果 7：自我修正失敗，是因為幾乎不改答案

EN+S1 的 d = 0.110；EN+SR-EN 的 d = 0.030，只有四分之一。c（0.74 對 0.68）與 recovery（0.26 對 0.31）相近。標準 SR 相對原答案只多 0.37pp（EN）與 0.01pp（ZH）。SR 的 n\_R 只有 33–60，CI 很寬。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Huang et al., *LLMs Cannot Self-Correct Reasoning Yet*（ICLR 2024） | 佐證 | 無外部回饋時，內在自我修正常讓推理變差 |
| Liang et al.（EMNLP 2024） | 佐證 | Degeneration-of-Thought：模型一旦有信心就不再產生新想法；我們的 d 是它的直接測量 |
| Kamoi et al., *When Can LLMs Actually Correct Their Own Mistakes?*（TACL 2024） | 佐證 | 只有自身回饋時，沒有前作展示出可靠的自我修正 |
| Madaan et al., *Self-Refine*（NeurIPS 2023） | 反對 | 7 個任務平均提升約 20%；多為生成任務，headroom 較大 |
| Stechly, Marquez, Kambhampati, *GPT-4 Doesn't Know It's Wrong*（2023） | 反對 | 瓶頸在驗證；我們的資料顯示 recovery 不差，瓶頸在 d |
| Tyen et al., *LLMs Cannot Find Reasoning Errors, but Can Correct Them*（Findings ACL 2024） | 相關 | 把自我修正拆成「找錯」與「改錯」，與我們的 d / recovery 拆法同構 |
| Gou et al., *CRITIC*（ICLR 2024） | 相關 | 外部工具注入新資訊才讓自我修正有效，即擴大 headroom |

#### 結果 8：配對中的英文數

0 個英文：recovery 0.417、recovery\_blind 0.072、Excess 1.99pp；1 個：0.371、0.308、0.43pp；2 個：0.273、0.090、0.83pp。英文數與多樣性來源共線，不能單獨歸因。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Wendler et al.（2024） | 機制 | 英文作為潛在樞紐語言 |
| Wu et al., *The Semantic Hub Hypothesis*（ICLR 2025） | 機制 | 跨語言語意表徵集中在由主導語言支配的共享空間 |

### C. 舊分析（跨語言 160 對 + legacy 同語言 24 格）

**仍然成立：**

- recovery ≈ 0.34（跨語言 160 對）
- has\_english 讓 recovery +0.093，但 recovery\_blind +0.155，淨 Excess 較低（含英文 1.20pp、不含 1.73pp）
- 160 對中 13 對實質虧損，全在 CommonsenseQA，12 / 13 含英文
- recovery 變異：資料集 51.5%、模型 0.5%、EN / CN 1.3%；資料集內殘差 I² ≈ 4%
- legacy Debate 的 off-menu 0.64%
- gap 的信度 λ ≈ 0.95，不需要測量誤差校正

**是線索、不是結論：**

| 觀察 | 為什麼還不能宣稱 | 相關論文與論點 |
| --- | --- | --- |
| 換模型時 recovery 幾乎不變（平均名次 2.00 / 2.12 / 1.88） | generator 與 aggregator 是同一個模型，不可識別；強模型的 n\_R 少 2–3 倍，可能互相抵銷；配對 CI ±0.13 | Panickssery, Bowman, Feng（NeurIPS 2024）：LLM 評估者偏好自己的輸出；Zheng et al., *Judging LLM-as-a-Judge*（NeurIPS 2023）：評審能力隨模型強度上升 |
| 跨資料集 Spearman(recovery, n\_R) = −1.00 | 只有 3 點；各資料集 n 不同，n\_R 不是 d 的乾淨代理 | Baek et al., *Agreement-on-the-Line*（NeurIPS 2022）：模型一致率可免標註預測 OOD 準確率；Garg et al., ATC（ICLR 2022）；Lakshminarayanan et al., *Deep Ensembles*（NIPS 2017）：成員分歧與誤差相關 |
| 資料集 recovery 梯度與 has\_english 係數 Spearman = −1 | 4 點，而且來自兩批資料 | Sprague et al., *To CoT or Not to CoT*（2025）：CoT 主要幫助數學與符號推理 |

**已撤銷，不要再用：**

- recovery 跨格 I² = 14.8% → 實為 57–77%（SE 定義錯）
- EIV 校正、−0.412、「兩法收斂」
- recovery 跨來源等價的 TOST 宣稱
- 對 Gao et al. 的「需要 recovery ≥ 0.60 才划算」→ 循環論證
- 「換模型不影響 recovery」作為假設 → 降為範圍界定

## 三、建議的新方向

把論文定位成「測試時多樣性什麼時候值得付錢」：一個明列可推翻假設的分解框架、一條決策規則、以及用少量樣本預測決策的方法。

暫定標題（二選一）：

- *When Does Test-Time Diversity Pay Off? Decomposing LLM Aggregation Gains Against the Strongest Single Path*
- *Diversity Is Not Gain: A Decision Rule for When to Aggregate LLM Reasoning Paths*

### 核心式

```latex
\text{Excess} = d \cdot (c - m) \cdot (\textit{recovery} - \textit{recovery}_{\textit{blind}})
```

d 是候選答案不同的比例；c、m 是分歧題上的 oracle 與平均正確率；recovery 是聚合器賺到 headroom 的比例；recovery\_blind 是「永遠用較強那條」的同一個量。聚合值得，當且僅當 recovery > recovery\_blind。

### 三個 RQ（改寫版）

| RQ | 問題 | 評估方式 |
| --- | --- | --- |
| RQ1（主結果） | 給定一個任務的少量標註樣本，能否預測其餘題目該聚合，還是直接用單一最強 path？ | 在 probe 上估 recovery − recovery\_blind，預測 target 的決策；以 regret 評估；看跨模型、跨資料集、跨多樣性來源的遷移 |
| RQ2 | 聚合失利時，瓶頸在 generator（d、c−m）、aggregator（recovery），還是單一最強 path 太強（recovery\_blind）？ | 逐來源分解 + 生成模型 × 裁決模型交叉實驗 |
| RQ3 | K 條 path 中，哪一條該被改進或替換？ | 每條 path 的獨有貢獻 π\_i 與被採納率 recovery\_i，K ≥ 3 |

### 貢獻

1. **框架**：把「單一最強」放進同一把尺（recovery\_blind），得到一條決策規則；明列哪些假設可被推翻
2. **實證**：在 6 種多樣性來源 × 3 種聚合器 × 4 模型 × 4 資料集上，顯示基準選擇會翻轉結論、多樣性不等於增益、Debate 與 Judge 同樣好但貴 5–6 倍、自我修正失敗在 d
3. **方法**：用少量 probe 預測決策（取決於 RQ1 結果）
4. **診斷**：瓶頸隨設定改變，並可細到單一 path

### 兩種定位（第二週結束時決定）

| RQ1 的結果 | 定位 | 目標 | 主結果 |
| --- | --- | --- | --- |
| probe 100–200 題，regret 明顯低於「永遠聚合」與「永遠用單一最強」 | 方法 + 分析 | Main，Findings 保底 | RQ1 曲線 + 結果 1、3 |
| regret 與笨基線差不多 | 分析論文 | Findings | 結果 1、3、5、6、7 |

### 這個方向如何回應每一條審稿意見

| 審稿意見 | 回應 |
| --- | --- |
| R2：novelty 有限，只是元件組合 | 不再提出新的聚合方法；貢獻是分解、決策規則、可推翻的假設與跨 6 種來源的實證 |
| R1-W1：缺同語言對照 | 結果 6 直接回答：同語言重抽的淨效果勝過換語言 |
| R1-W2：五種語言沒有理由 | 語言只是六種多樣性來源之一 |
| R1-W3、R2：router 泛化與成本 | router 移除；RQ1 的跨來源遷移正面處理泛化 |
| R3：引用只有 17 篇 | 目標 55 篇以上，見第六節 |
| R3：統計不透明 | 統計透明度附錄：每格跑幾次、報什麼、用哪個檢定、與誰比 |
| R3：增益可能來自更多計算 | 結果 5：Debate 多花 5–6 倍 tokens，沒有換來更高的 recovery |

### 刻意放棄的主張

- recovery 跨多樣性來源相等（結果 4 已推翻）
- IMSR 作為方法貢獻（降為框架中的一個觀測點）
- 動態語言路由器

## 四、新論文的完整邏輯流程

論文的主線是：文獻對測試時多樣性的結論互相矛盾 → 因為大家拿錯基準 → 我們給出一把把「單一最強」放進去的尺 → 用它發現基準會翻轉結論 → 再證明少量樣本就能做出正確決策 → 最後診斷失敗在哪。標 ⏳ 者尚未完成。

### §1 Introduction

- **要說的話**：自我一致性、多語言、persona、辯論、自我修正都在製造多樣性再聚合，但結論互相矛盾；多數研究拿「平均」或「單次」當基準，而不是單一最強的 path。我們問：什麼時候聚合真的值得？
- **實驗證據**：結果 1 當首圖（76% → 34%）；結果 3 當第二個鉤子
- **文獻**：
  - Madaan et al.（2023）：Self-Refine 平均提升約 20%；Huang et al.（2024）：自我修正會讓推理變差 → 同一方法、相反結論
  - Du et al.（2024）：辯論有效；Smit et al.（2024）：對齊預算後辯論無優勢 → 相反結論
  - Gao et al.（2025）：多語言抬高 Acc@k，但 Vote@k 優勢消失 → 上界高不代表會贏
  - Brown et al., *Large Language Monkeys*（2024）：coverage 隨樣本數穩定上升，但實際選得出來的遠低於 coverage

### §2 Related Work

| 小節 | 收錄論文 | 我們的定位 |
| --- | --- | --- |
| 2.1 集成多樣性理論 | Kuncheva & Whitaker（2003）、Wood et al.（2023）、Kuncheva（2014）、Cruz et al.（2018）、Caruana et al.（2004） | 他們說明多樣性與準確率關係弱；我們用 recovery\_blind 解釋原因，並給出決策規則 |
| 2.2 LLM 測試時聚合 | Wang et al.（2023）、Chen et al. USC（2023）、Jiang et al. LLM-Blender（2023）、Wang et al. MoA（2024）、Li et al.（2024）、Chen, Zaharia, Zou（2024）、Brown et al.（2024） | 他們提出聚合方法；我們提供評估何時該用的尺 |
| 2.3 多 agent 辯論與自我修正 | Du et al.、Liang et al.、Smit et al.、Huang et al.（2024）、Kamoi et al.、Madaan et al.、Shinn et al.、Tyen et al.、Stechly et al.、Gou et al. | 我們把相反結論放進同一個分解 |
| 2.4 多語言推理 | Shi et al.（2023）、Qin et al.、Huang et al. XLT、AutoCAP、mGRPO、Gao et al.、Lai et al.、CLC、Elhady et al.、Wendler et al.、Wu et al. | 語言是多樣性來源之一，不是主角 |
| 2.5 預測與路由 | Song et al. GV-gap、Baek et al.、Miller et al.、Garg et al.、FrugalGPT、AutoMix、Maia Polo et al.、Geifman & El-Yaniv | 最接近 RQ1；差別在我們預測的是「該不該聚合」，不是準確率或該不該升級模型 |

### §3 Framework

- **3.1 分解**：Gain = d·(c−m)·recovery；「永遠用較強那條」本身是一種聚合器，其增益為 gap / 2，得 recovery\_blind；相減得 Excess
- **3.2 決策規則**：聚合值得 ⟺ recovery > recovery\_blind；並區分配對內基準與全域基準，差值為 Δ\_best
- **3.3 單一 path 的延伸**：recovery = Σ π\_i · recovery\_i；recovery\_blind 是其特例（K=2 時 = 2π\_A − 1）
- **3.4 假設表**：A1 封閉性、A2 一致題無作用、A7 決策不退化，以及 A3 / A4 / A5 的適用範圍
- **3.5 測量**：較強者在 held-out 的一半選出；不這樣做的偏誤 ≈ 0.8 / √n\_R；有效樣本是 n\_R = n × d × c
- **實驗證據**：恆等式誤差 1×10⁻¹⁶；legacy off-menu 0.64%；split-half 消掉 EN 與 S1 的假差距；結果 2 支撐 A7
- **文獻**：
  - Kuncheva（2014）：oracle 與 single best 是組合方法的標準參考點
  - Song et al., *Mind the Gap*（ICLR 2025）：GV-gap 以 1 − m 正規化；我們以 c − m 正規化，且把「不聚合」放上同一把尺
  - Jiang et al., LLM-Blender（ACL 2023）：選擇式與生成式融合之分 → 封閉性為何只對選擇式成立
  - Cawley & Talbot（2010）：選擇與評估要分開資料
  - Jacobs & Wallach（FAccT 2021）：discriminant 與 predictive validity 的詞彙
  - Lipton & Steinhardt（2018）：mathiness 的批評 → 為什麼要放假設表

### §4 Experimental Setup

- **內容**：4 模型 × 4 資料集；14 條 path（5 語言、S1 / S2 於 T=1.0、P1 / P2、W1 / W2、簡短 CoT、SR-EN、SR-ZH）；聚合器 Blind、Vote、Judge、Debate；n = 2000（TruthfulQA 817）；每個 pass 跑一次，以逐題 bootstrap 估不確定性；由 n\_R 事先算 MDE
- **文獻**：
  - 資料集：Hendrycks et al.（MMLU）、Talmor et al.（CommonsenseQA）、Amini et al.（MathQA）、Lin et al.（TruthfulQA）
  - Card et al.（EMNLP 2020）：先算 MDE 再跑
  - Dodge et al.（EMNLP 2019）、Biderman et al.（2024）：要揭露的實驗細節
  - Sclar et al., FormatSpread（ICLR 2024）：格式差異會大幅改變表現 → 各 path 格式必須鎖死
  - Wang et al., *LLMs Are Not Fair Evaluators*（ACL 2024）：位置偏誤 → Judge 呈現順序隨機化
  - Renze & Guven（2024）：T 在 0–1 對正確率無顯著影響 → 解釋 S1 與 EN 能力相近
  - Deng et al., *Rephrase and Respond*（2023）：改寫會改變答案

### §5 Results

| 小節 | 主張 | 實驗 | 文獻與論點 |
| --- | --- | --- | --- |
| 5.1 決策會翻轉，而且看基準 | 配對內 76% vs 全域 34%（split 後重算） | 結果 1、2 | Kuncheva（2014）；Cawley & Talbot：選擇偏誤；Geifman & El-Yaniv：沿整個範圍評估；Zheng et al.（persona）：persona 效果近乎隨機 |
| 5.2 多樣性不等於增益 | 語言 headroom 最大但 Excess 小 | 結果 3、8 | Kuncheva & Whitaker：多樣性度量預測力弱；Wood et al.：diversity 依賴個體能力 |
| 5.3 少量樣本預測決策 ⏳ | probe 100–200 題的 regret 低於兩個笨基線 | RQ1（待做） | Maia Polo et al., tinyBenchmarks：約 100 題可估準確率；Baek et al.：一致率免標註預測 OOD；AutoMix、FrugalGPT：預算內路由；Søgaard et al.、Koh et al.：隨機切分高估，要看結構化偏移 |
| 5.4 瓶頸診斷 | 瓶頸隨來源改變；Debate 不比 Judge 好 | 結果 4、5；交叉實驗 ⏳ | Smit et al.：辯論無優勢；Chen, Zaharia, Zou：更多呼叫不一定更好；Zheng et al.（MT-Bench）：評審能力隨模型而變；Panickssery et al.：自我偏好 |
| 5.5 該改哪條 path ⏳ | π\_i 與 recovery\_i 的四象限 | Judge@3（待做） | Caruana et al.：從模型庫中選子集；Ghorbani & Zou：Shapley 歸因；Wang et al. MoA（反對）：弱 path 也有貢獻；Chen et al. USC：Judge@K 的方法出處 |

### §6 Analysis

- **6.1 換語言真的有幫助嗎**：結果 6。Du et al.：同語言辯論本身有效；Qin et al. 與 Huang et al.（XLT）：跨語言 prompting 有效但以平均為基準；Wendler et al.、Wu et al.：英文是潛在樞紐
- **6.2 自我修正為什麼失敗**：結果 7。Huang et al.（2024）、Kamoi et al.、Liang et al. 佐證；Madaan et al. 與 Stechly et al. 反對；Tyen et al. 的拆法與我們同構；Gou et al. 說明外部資訊能擴大 headroom
- **6.3 對 Gao et al. 的互補讀法**：多語言同時抬高 headroom 與 recovery\_blind。措辭一律「補充」，不寫「糾正」；不放兩平條件的數字

### §7 Limitations

- 模型為中階非推理模型，沒有 o1 / R1 類模型
- 只有選擇題，答案可直接比對
- 封閉性只近似成立
- 全域最強 path 的選擇仍有殘餘偏誤
- 主網格中 generator 與 aggregator 是同一個模型
- Gemini 的截斷修補只重跑失敗題

### 附錄

- **A 統計透明度**：跑幾次、報什麼、bootstrap 細節、MDE（回應 R3）
- **B 可重現性**：照 Pineau et al.（JMLR 2021）的清單；prompt 全文與 hash；程式碼
- **C 逐格完整表格**

## 五、待辦清單、各階段錄取機會與 Novelty 比較

影響錄取機會最大的單一工作是 RQ1 的 probe 預測；最便宜也最必要的是修好 Gemini 與 Judge。以下機率是判斷而非計算，ARR 審稿雜訊很大。

### 各階段的錄取機會

| 階段 | 完成內容 | Findings 以上 | Main | 論文性質 |
| --- | --- | --- | --- | --- |
| 0（現在） | 八項結果，但 Gemini 壞、Judge 有 off-menu、沒有 RQ1 | 25–35% | < 10% | 有缺陷的分析 |
| 1 | 修好 Gemini、Judge，匯出逐題資料 | 30–40% | < 10% | 乾淨的分析 |
| 2 | + 全域基準 split、相關工作 55 篇以上、統計透明度與可重現性附錄、匿名程式碼 | 40–50% | 10–15% | 完整的分析論文 |
| 3 | + RQ1 有乾淨結果 | 55–65% | 20–30% | 方法 + 分析 |
| 4 | + RQ2 交叉實驗、RQ3 的 Judge@3 | 60–65% | 25–30% | 完整版 |

參考點：ACL 系列 Main 錄取率約兩成多，加上 Findings 合計約四成；上一輪平均 Overall 是 2.0。

### 待辦清單

**階段 1：修資料（第 1 週）**

- [ ] 抽 10 題 Gemini 解析失敗的原始輸出，確認是句子中間截斷
- [ ] 只對 Gemini 失敗的約 7% 題目調大 max\_tokens 重跑（約 $2）
- [ ] Judge prompt 改為輸出 A 或 B；呈現順序每題隨機並記錄
- [ ] 重跑 Judge：12 配對 × 12 格，含修好的 Gemini（約 $5–8）
- [ ] 匯出逐題資料：每題每條 path 的答案與對錯；每題每個聚合器的最終答案
- [ ] 確認 ARR 截止日，以及實質改寫算新投稿還是 resubmission
- [ ] 寫 §3（不依賴新結果）

**階段 2：補齊分析論文的必要件（第 2 週起，與寫作並行）**

- [ ] 全域基準用 split：一半題目從 14 條選最強，另一半驗證，重算 76% → 34%
- [ ] 讀並各寫兩句區隔：Song et al.、Baek et al.、Lai et al.、Gao et al.、AutoMix、Kuncheva
- [ ] 補上原論文漏引的資料集：MathQA（Amini et al. 2019）、TruthfulQA（Lin et al. 2022）
- [ ] 統計透明度附錄草稿
- [ ] 可重現性附錄與匿名 repo

**階段 3：RQ1 主結果（第 2 週）**

- [ ] probe 大小 25 / 50 / 100 / 200 / 500 題
- [ ] 四種遷移：同格隨機切（上界）、換模型、換資料集、換多樣性來源
- [ ] 三個對照：永遠聚合、永遠用單一最強、oracle 決策
- [ ] 指標 regret；配對內與全域兩種基準都做
- [ ] （可選）優先選「容易分歧」的題目當 probe，增加有效樣本
- [ ] 第 2 週結束：決定定位 Main 或 Findings

**階段 4：RQ2、RQ3（第 3 週，只跑聚合端）**

- [ ] 生成 × 裁決交叉：EN+ZH，3 × 3 模型，4 個資料集（約 $3–5）
- [ ] Judge@3：EN+ZH+JA、EN+ZH+S1，記錄每條 path 的 recovery\_i（約 $3–5）
- [ ] Vote@3 / @5 / @14 離線計算

**寫作（第 4–10 週）**

- [ ] 圖：兩種基準的 Excess、多樣性 vs Excess、Debate vs Judge、SR 平行 vs 序列
- [ ] §1、§2、§4、§5、§6、§7
- [ ] 教授 review
- [ ] 內容凍結（約 11 月中），之後只寫不跑
- [ ] 提交前檢查：頁數、匿名化、每個數字回到原始輸出核對

### 與最接近的論文比較 Contribution 與 Novelty

| 論文 | 他們做什麼 | 有分解 | 以單一最強為基準 | 多種多樣性來源 | 少量樣本做決策 | 我們的差異 |
| --- | --- | --- | --- | --- | --- | --- |
| Gao et al.（2025） | 比較多語言、重複、改寫的 Acc@k 與 Vote@k | 否 | 否 | 是 | 否 | 給出分解式，說明上界高為何不贏 |
| Lai et al.（2025） | 沿三個維度製造變異再以一致性聚合 | 否 | 否 | 是 | 否 | 他們提升準確率；我們回答何時值得 |
| Song et al.（ICLR 2025） | 生成與驗證能力的差距 GV-gap | 比值 | 否 | 否 | 否 | 正規化分母 c − m 而非 1 − m；把「不聚合」放上同一把尺 |
| Wood et al.（JMLR 2023） | bias–variance–diversity 分解 | 是 | 否（平均） | — | 否 | 再拆成 headroom 與 recovery，並納入單一最強 |
| Kuncheva & Whitaker（2003）、Kuncheva（2014） | 多樣性度量；oracle 與 single best | 否 | 是 | — | 否 | 移到 LLM 測試時，給出可分解的決策規則 |
| Cruz et al.（2018） | 逐題決定選單一或合併 | 否 | 是 | — | 需驗證集與鄰域 | 分布層級、封閉形式；候選來自同一 LLM 的輸入變異 |
| Baek et al.（NeurIPS 2022） | 以一致率免標註預測 OOD 準確率 | 否 | — | — | 預測準確率 | 我們預測的是該不該聚合 |
| AutoMix（NeurIPS 2024）、FrugalGPT（2023） | 在預算內決定是否升級到大模型 | 否 | — | — | 逐題，需自我驗證或訓練 scorer | 分布層級，用少量標註 |
| Smit et al.（ICML 2024） | 對齊預算後比較辯論與單 agent | 否 | 部分 | 否 | 否 | 結論一致；我們量化到 recovery 與成本 |
| Brown et al.（2024） | coverage 隨取樣數上升 | 否 | 否 | 只有採樣 | 否 | 把選擇效率正規化，跨來源比較 |
| **本論文** | 分解 + 決策規則 + probe 預測 + 診斷 | **是** | **是** | **6 種** | **是（待驗證）** | — |

**Novelty 風險**：Song et al. 的比值結構、Baek et al. 的免標註預測、Lai et al. 的多維變異、Kuncheva 的 single best 與 oracle 都已存在。我們的新意是四者的組合，加上「基準翻轉結論」的實證。若 RQ1 失敗，novelty 只剩實證發現，論文定位就是 Findings。

## 六、所有相關論文：是否引用與引用論點

建議引用約 59 篇（必引 31、建議引 28），可達成 55 篇以上的目標；撤銷的統計分析所帶出的論文大多不需要。投稿前每一筆的作者、年份、出處都要以原文核對，標「待核對」者資訊較不確定。

### A. 原論文已引用的 17 篇

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Hendrycks et al.（2021）MMLU | 必引 | 使用的資料集 | 資料集出處 |
| Talmor et al.（2019）CommonsenseQA | 必引 | 使用的資料集 | 資料集出處 |
| Wang et al.（ICLR 2023）Self-Consistency | 必引 | 測試時聚合的原點 | 對取樣的多條推理路徑做多數決可提升準確率 |
| Huang et al.（ICLR 2024）LLMs Cannot Self-Correct Reasoning Yet | 必引 | §1 矛盾、§6.2 | 無外部回饋時，內在自我修正常降低推理表現 |
| Huang et al.（2023）Not All Languages Are Created Equal（XLT） | 建議引 | §6.1 反方 | 以英文為樞紐的跨語言提示可提升推理 |
| Qin et al.（2023）Cross-lingual Prompting | 建議引 | §2.4、§6.1 | 跨語言對齊提示改善 zero-shot CoT |
| Shinn et al.（2023）Reflexion | 建議引 | §2.3 自我修正代表作 | 以語言化回饋與記憶反覆修正 agent 行為 |
| Sprague et al.（2025）To CoT or Not to CoT | 建議引 | 解釋資料集間 recovery 梯度 | CoT 主要幫助數學與符號推理 |
| Bang et al.（2023） | 可選 | 背景一句 | LLM 在不同語言間表現不一致 |
| Lai et al.（2023）ChatGPT Beyond English | 可選 | 背景一句 | 同上 |
| Xiong et al.（2025）Self-Rewarding Correction | 可選 | §2.3 對照 | 需要訓練的自我修正 |
| OpenAI（2024）GPT-4 Technical Report | 可選 | 只在需要引模型出處時 | 模型來源 |
| Ahuja et al.（2025）sphinx | 不需要 | 訓練式多語言方法，新論文不涉訓練 | — |
| Indurthi et al.（2024） | 不需要 | 同上 | — |
| Ramesh et al.（2023） | 不需要 | 公平性動機，新論文不談 | — |
| Renze & Guven（2024）concise CoT | 不需要 | 主題不同；注意與溫度那篇是不同論文 | — |
| Wang et al.（2024）MMLU-Pro | 不需要 | 原論文用來支持「CoT 傷害知識題」並不恰當 | — |

### B. 對話中指出的缺漏與審稿人點名

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Amini et al.（2019）MathQA | 必引 | 使用卻漏引 | 資料集出處 |
| Lin et al.（2022）TruthfulQA | 必引 | 使用卻漏引 | 資料集出處 |
| Zhang et al.（Findings ACL 2024）AutoCAP | 必引 | R2 點名 | 自動選擇語言並分配各語言推理路徑的權重，用於 zero-shot CoT |
| mGRPO: Unlocking LLM Reasoning through Multilingual Thinking（待核對） | 必引 | R2 點名 | 把多語言思考納入 RL 訓練以提升推理；與我們的推論期框架區隔 |
| Du et al.（ICML 2024）Multiagent Debate | 必引 | R3 點名；§6.1 同語言控制組的設定來源 | 同一模型多實例互相辯論可提升事實性與推理 |
| Madaan et al.（NeurIPS 2023）Self-Refine | 必引 | R3 點名；§1 矛盾 | 同一模型自我回饋並改寫，7 個任務平均提升約 20% |
| Shi et al.（ICLR 2023）Language Models Are Multilingual CoT Reasoners | 必引 | R3 點名；§2.4 | 提出 MGSM，顯示 LLM 具多語言 CoT 能力 |
| Ki et al.（ACL 2025）Multiple LLM Agents Debate for Equitable Cultural Alignment（待核對） | 建議引 | R3 點名 | 多個 agent 以不同視角辯論，改善文化情境下的決策與群體間公平 |

### C. 集成理論與組合方法

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Wood et al.（JMLR 2023）A Unified Theory of Diversity | 必引 | §2.1、§3 承接 | 集成誤差 = 平均個體誤差 − diversity 效應；diversity 依賴標籤與個體能力 |
| Kuncheva & Whitaker（Machine Learning 2003） | 必引 | §5.2 核心 | 十種多樣性度量與集成準確率關聯都很弱 |
| Kuncheva（2014）Combining Pattern Classifiers | 必引 | §3 基準定義 | oracle 與 single best 是評估組合方法的兩個參考點 |
| Cruz, Sabourin, Cavalcanti（Information Fusion 2018） | 建議引 | 最接近「選單一 vs 合併」 | 逐題決定用單一最有能力的分類器或合併子集 |
| Caruana et al.（ICML 2004）Ensemble Selection | 建議引 | §5.5 RQ3 | 從模型庫逐步選子集，勝過使用全部 |
| Cruz et al.（Pattern Recognition 2015）META-DES | 可選 | 引 2018 的綜述即可 | 以 meta-learning 估計分類器能力 |
| Ghorbani & Zou（ICML 2019）Data Shapley | 可選 | 只在 RQ3 用歸因時 | 以 Shapley value 公平分配個別貢獻；LOO 是其一階近似 |

### D. LLM 測試時聚合

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Song et al.（ICLR 2025）Mind the Gap（作者待核對） | 必引 | 最接近 recovery 的既有量，novelty 風險 | GV-gap 以 1 − m 正規化生成與驗證的差距 |
| Chen et al.（2023）Universal Self-Consistency | 必引 | Judge@K 的方法出處 | 讓 LLM 自己從多個候選中選最一致者 |
| Jiang, Ren, Lin（ACL 2023）LLM-Blender | 必引 | §3 封閉性 | 區分選擇式（PairRanker）與生成式（GenFuser）融合 |
| Chen, Zaharia, Zou（NeurIPS 2024）Are More LLM Calls All You Need? | 必引 | §5.4 | 複合推論系統的表現對呼叫次數可能非單調 |
| Brown et al.（2024）Large Language Monkeys | 必引 | §1、§2.2 | coverage 隨樣本數穩定上升，但選得出來的遠低於 coverage |
| Wang et al.（2024）Mixture-of-Agents | 建議引 | §5.5 反方 | 模型看到其他（即使較弱）模型的回答後會產生更好輸出 |
| Li et al.（2024）More Agents Is All You Need | 建議引 | §2.2 | 單純增加取樣 agent 數加多數決即可提升 |

### E. 多 agent 辯論、自我修正、LLM 評審

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Liang et al.（EMNLP 2024）Encouraging Divergent Thinking | 必引 | §6.2；P1+P2 Debate 的設定來源 | Degeneration-of-Thought：模型有信心後不再產生新想法 |
| Smit et al.（ICML 2024）Should We Be Going MAD? | 必引 | §1、§5.4 佐證 | 對齊預算後，辯論沒有可靠勝過單一 agent |
| Kamoi et al.（TACL 2024）Critical Survey of Self-Correction | 必引 | §2.3 主引用 | 只有自身回饋時，沒有前作展示出可靠的自我修正 |
| Tyen et al.（Findings ACL 2024） | 必引 | 最接近我們拆法的工作 | 自我修正可拆成「找錯」與「改錯」；LLM 不會找錯但會改錯 |
| Wang et al.（ACL 2024）LLMs Are Not Fair Evaluators | 必引 | §4 Judge 順序隨機化的理由 | LLM 評審有強烈位置偏誤 |
| Stechly, Marquez, Kambhampati（2023） | 建議引 | §6.2 反方 | 自我批判讓表現變差，外部驗證器則有效 → 瓶頸在驗證 |
| Gou et al.（ICLR 2024）CRITIC | 建議引 | §6.2 | 外部工具注入新資訊才讓自我修正有效 |
| Panickssery, Bowman, Feng（NeurIPS 2024） | 建議引 | generator = aggregator 的混淆 | LLM 評估者偏好自己的輸出，且與自我辨識能力有因果關聯 |
| Zheng et al.（NeurIPS 2023）Judging LLM-as-a-Judge | 建議引 | §5.4 | 強模型評審與人類判斷一致率可達八成以上；評審能力隨模型而變 |

### F. 多語言推理

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Gao et al.（2025）Could Thinking Multilingually Empower LLM Reasoning? | 必引 | 最接近的應用論文 | 多語言抬高 Acc@k，但 Vote@k 優勢消失 |
| Lai, Zhang, Nissim（2025）Multidimensional Consistency | 必引 | 最直接撞 RQ 的論文 | 沿順序、改寫、語言三維度製造變異；一致性依維度而不同 |
| Elhady, Agirre, Artetxe（2026）Cross-lingual Self-Consistency（待核對） | 建議引 | 一句區隔 | 以跨語言一致性當無監督 RL 訊號；是訓練期方法 |
| Cross-Lingual Consistency, CLC（2025，作者待核對） | 建議引 | 語言軸 + 多數決的實例 | 整合多語言推理路徑的多數決可提升數學推理 |
| Wendler et al.（2024）Do Llamas Work in English? | 建議引 | §6.1 機制 | 模型內部以英文為潛在樞紐 |
| Wu et al.（ICLR 2025）The Semantic Hub Hypothesis | 可選 | §6.1 機制補充 | 跨語言語意表徵集中於由主導語言支配的共享空間 |
| Rajaee et al.（2025）Best-of-L（待核對） | 可選 | Limitations 一句 | 以跨語言 reward model 排序多語言候選 |

### G. 預測、路由與評估方法

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Baek et al.（NeurIPS 2022）Agreement-on-the-Line | 必引 | 最接近 RQ1，novelty 風險 | 模型一致率可免標註預測 OOD 準確率 |
| Chen, Zaharia, Zou（2023）FrugalGPT | 必引 | §2.5 | 級聯：先問便宜模型，必要時才升級 |
| Madaan et al.（NeurIPS 2024）AutoMix | 必引 | 最接近 RQ1 的路由 | 即使自我驗證訊號有噪音，路由決策仍有效 |
| Maia Polo et al.（ICML 2024）tinyBenchmarks | 必引（若 RQ1 做出來） | §5.3 | 約 100 題即可把準確率估到約 2% 誤差內 |
| Cawley & Talbot（JMLR 2010） | 必引 | §3.5 選最強要分資料 | 同一批資料選擇又評估會樂觀偏誤 |
| Miller et al.（ICML 2021）Accuracy on the Line | 建議引 | §2.5 | ID 與 OOD 準確率高度線性相關，但某些偏移下會崩壞 |
| Garg et al.（ICLR 2022）ATC | 建議引 | §2.5 | 在來源資料學信心門檻，免標註估計目標準確率 |
| Geifman & El-Yaniv（NeurIPS 2017） | 建議引 | §5.1 評估方式 | 棄答規則要沿整條覆蓋範圍評估 |
| Søgaard et al.（EACL 2021）We Need to Talk About Random Splits | 建議引 | §5.3 為何要結構化偏移 | 隨機切分系統性高估表現 |
| Koh et al.（ICML 2021）WILDS | 可選 | 同上 | i.i.d. 與真實偏移間有大幅落差 |
| Lakshminarayanan et al.（NIPS 2017）Deep Ensembles | 可選 | d 與難度的機制 | 成員分歧與預測誤差高度相關 |
| Jacobs & Wallach（FAccT 2021）Measurement and Fairness | 可選 | §3.4 效度詞彙 | discriminant、predictive validity 的定義 |

### H. 實驗設計與報告

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Card et al.（EMNLP 2020）With Little Power | 建議引 | §4.1 | 先算 MDE 再跑實驗 |
| Dodge et al.（EMNLP 2019）Show Your Work | 建議引 | 附錄 A | 單次或最佳結果無法重現，要揭露預算與次數 |
| Dror et al.（ACL 2018）Hitchhiker's Guide to Testing Statistical Significance | 建議引 | 附錄 A | NLP 中顯著性檢定的選擇與報告 |
| Biderman et al.（2024）Lessons from the Trenches | 建議引 | 附錄 B | prompt 格式、解析規則、非決定性是不可重現的主因 |
| Sclar et al.（ICLR 2024）FormatSpread | 建議引 | §4 格式鎖死 | 無意義的格式差異可大幅改變表現 |
| Renze & Guven（2024）Effect of Sampling Temperature | 建議引 | §4 採樣軸 | T 在 0–1 之間對解題正確率無顯著影響 |
| Deng, Zhang, Gu（2023）Rephrase and Respond | 建議引 | §4 改寫軸 | 問題的措辭會影響模型能否答對 |
| Zheng et al.（Findings EMNLP 2024）Is "A Helpful Assistant" the Best Role? | 建議引 | §4 persona 軸；§5.1 反方 | persona 不比不加好，效果近乎隨機 |
| Pineau et al.（JMLR 2021） | 可選 | ACL 已有 Responsible NLP checklist | 可重現性檢查表 |
| Holtzman et al.（ICLR 2020） | 可選 | 溫度作為多樣性旋鈕 | 解碼參數上的多樣性與品質取捨 |
| Cameron, Gelbach, Miller（2008） | 可選 | 只在用 cluster bootstrap 報 CI 時 | cluster 少時標準推論會過度拒絕 |

### I. 不需要引用的

| 論文 | 理由 |
| --- | --- |
| Higgins & Thompson（2002）、Higgins et al.（2003）、IntHout et al.（2016）、Rücker et al.（2008）、Thompson & Higgins（2002） | I² 異質性分析已撤銷 |
| Frost & Thompson（2000） | EIV 校正已撤銷；一句「衰減偏誤可忽略」不需引用 |
| Lakens（2017） | TOST 等價檢定已撤銷 |
| Lipton & Steinhardt（2018） | 寫作自我檢查用，放進論文讀起來突兀 |
| Ribeiro et al.（ACL 2020）CheckList、Schaeffer et al.（NeurIPS 2023） | 用來說服自己「診斷本身是貢獻」，論文內容不需要 |
| Cortes & Lawrence（2021） | 關於審稿雜訊，不是論文內容 |

模型本身（DeepSeek-V3、Qwen3、Gemini 2.5、GPT-4o mini）也要各引其技術報告或官方說明，對話中沒有討論，投稿前補上。
