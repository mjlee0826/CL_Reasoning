# 論文現況總整理 v2：多樣性聚合何時值得

Oct 4, 2026 · @MJLee

## 摘要（10/4 晚間更新：加入 RQ2 交叉實驗與 RQ1-K）

論文是分析論文：同一個模型內，測試時的多樣性聚合什麼時候值得。主網格、RQ1、RQ2、RQ1-K 四組實驗都已完成，三份判定標準都是事先寫好的。下一步是 K 條 path 的 Judge 版本，判定標準已把它列為必要的後續實驗。

| 項目 | 現況 |
| --- | --- |
| 原論文 | IMSR：跨語言辯論 + 語言配對路由；上一輪 Overall 2.5 / 2 / 1.5 |
| 資料 | 4 模型（GPT-4o mini、Qwen3-8b、DeepSeek V4.1 Flash、Gemini 3.1 Flash-Lite）× 4 資料集 = 16 區塊；14 條 path；聚合器有 Judge（只輸出選項）、Debate、多數決 |
| 核心式 | Excess = d·(c−m)·(recovery − recovery\_blind) |
| 主張一：增益取決於跟誰比 | 12 條 path 多數決比預設的英文 path 高 1.5pp，比最強的單一 path 只高 0.30pp（區間 −0.63 到 +1.23），輸出 tokens 是 11.5 倍 |
| 主張二：多樣性多不等於賺多 | 語言的 headroom 是採樣的兩倍，聚合後沒有比較高；增益不需要換語言。兩條與三條 path 都成立 |
| 主張三：要不要聚合由資料集決定 | 多數決在三個資料集小贏、在 TruthfulQA 輸 2pp，四個模型同向。用少量標註逐任務決定，贏不過固定做法（RQ1，負面結果） |
| 主張四：recovery 是裁判能力減分歧題難度 | 裁判效果 +0.32、候選效果 −0.22，都是 16 個區塊幾乎全部同向；模型裁決自己的答案最差（RQ2） |
| 改寫過的結論 | 「聚合只在 27% 的情況贏過單一最強」降為描述；Debate 不再「等於」Judge；「重抽勝過換語言」改成「換語言沒有比較好」；「強模型 recovery 較高」是兩個效果相抵的淨值 |
| 最大缺口 | K 條 path 的 Judge 版本未跑；RQ3 未做（可離線）；「強裁判是否只是自己知道答案」的離線檢查未做；論文尚未動筆 |
| 錄取機會（判斷，非計算） | Findings 以上 35–45%、Main 約 10% |
| 投稿時程 | ARR 沒有 12 月 cycle。下一輪是 2027 年 1 月（對應 ACL 2027，確切日期未公布），必須走 resubmission |

數字來源：aggregation\_cells.csv（主網格）、RQ1 報告、RQ2 報告（cross\_cells.csv）、RQ1-K 報告（rq1k\_blocks.csv），都是 2026-10-04 的版本。區間都是跨 16 個區塊（模型 × 資料集）的 95% 區間；區間含 0 代表分不出正負。v1 文件的數字來自舊 Judge 與已換掉的模型，不要再引用。

## 一、最初的困境

原論文是一個增益不大、又沒有解釋為什麼有效的方法論文，三位審稿人分別從 novelty、因果、透明度否定它。

### 原論文做了什麼

- **IMSR**：同一個 LLM 用兩種語言各答一次；答案不同就進行最多三輪跨語言辯論，仍不一致由 Judgment Agent 裁決。
- **動態路由**：XLM-RoBERTa 分類器，從 5 種語言的 10 個配對中挑出最可能修正成功的一對。
- **實驗**：4 模型 × 4 資料集，宣稱 16 格中 12 格最佳且 p < 0.05。

### 三位審稿人的批評

| 審稿人 | Soundness / Excitement / Overall | 核心批評 |
| --- | --- | --- |
| R1（信心 4） | 2.5 / 2.5 / 2.5 | W1 缺同語言雙 agent 對照組，增益可能來自辯論結構而非換語言；W2 五種語言的選擇沒有理由；W3 router 的泛化未討論，「plug-and-play」不成立 |
| R2（信心 4） | 2 / 2.5 / 2 | novelty 有限，只是既有元件的組合；漏引 AutoCAP、mGRPO；router 要每題跑遍所有配對，不輕量；未驗證新語言 |
| R3（信心 5） | 2 / 1.5 / 1.5 | 只有 17 篇引用；統計顯著性不透明；混語辯論 prompt 不穩；best fixed pair 的原因沒解釋；SR 與 IMSR 輪數不同，增益可能只是更多計算 |

Reproducibility 是 3 / 1 / 2，Software 三位都給 1。

### 根本問題

1. 增益小（多數格子不到 2pp），而且沒有說明為什麼有效。
2. 「換語言」是否必要從未被隔離出來。
3. 原論文自己的 Table 1 就顯示 EN-SR 在 4 格勝過 IMSR。
4. 沒有任何一個可以被推翻的假設。

### 轉向歷程

| 階段 | 論文是什麼 | 為什麼被換掉 |
| --- | --- | --- |
| 投稿版 | IMSR 方法論文 | 審稿人：novelty、因果、透明度 |
| v7 | Gain 分解 + 檢定五種多樣性來源的 recovery 是否相等 | 教授問「實驗在驗證什麼假設」；樣本量做不到等價宣稱 |
| 教授第二次提問 | 用隨機切分的 train / test 驗證預測 | 同分布切分保證成功，不構成檢驗 |
| v1 文件（10/4 上午） | 決策框架：用少量標註預測「該不該聚合」（RQ1）當主結果 | RQ1 照事先寫好的標準判定為沒有空間 |
| 現在 | 分析論文：增益取決於跟誰比；要不要聚合由資料集決定；recovery 是裁判能力減分歧題難度 | — |

教授的兩次提問是同一個問題：**什麼結果會讓我們承認自己錯了？** RQ1、RQ2、RQ1-K 三個實驗都照這個原則做：判定標準先寫好、記下雜湊值再跑。RQ1 的結果不如預期，照標準降為負面結果。

## 二、現有實驗結果與相關文獻

現在有十五項結果。結果 1 到 11 的編號沿用上一版，其中結果 1、3、6、9、10 已依 RQ2 與 RQ1-K 的資料更新；結果 12 到 15 是新增的。支柱是結果 1、3、10。

名詞：d 是兩條 path 答案不同的比例；headroom = d·(c−m)，是聚合最多能賺的正確率；recovery 是聚合器實際賺到 headroom 的比例；recovery\_blind 是「永遠用較強那條」賺到的比例；Excess 是聚合比較強那條多出的正確率。區間都是跨 16 個區塊（模型 × 資料集）的 95% 區間；區間含 0 代表分不出正負。

### A. 資料與資料品質

| 項目 | 數據 | 影響與處理 |
| --- | --- | --- |
| 模型 | GPT-4o mini、Qwen3-8b、DeepSeek V4.1 Flash、Gemini 3.1 Flash-Lite | 舊的 DeepSeek V3.2、Gemini 2.5 已換掉；CSV 裡還留著 V3.2 的列，分析時要排除 |
| 各模型單一 path 正確率 | GPT 68–84%、Qwen 65–83%、DeepSeek 73–93%、Gemini 75–95% | 形成弱、強兩群，可以看強度的影響 |
| 分歧率 d | GPT 0.13、Qwen 0.14、DeepSeek 0.08、Gemini 0.06 | 強模型每格只有約 100–125 題分歧題，單格數字不能單獨下結論 |
| Gemini 3.1 解析失敗 | 平均 0.07%，最高 0.4% | 迴圈問題沒有出現 |
| DeepSeek 在 MathQA 解析失敗 | 每條 path 1.4–3.3%（其他都 < 0.3%） | 待查：抽 10 題原始輸出看原因 |
| Judge off-menu | 平均 < 0.1%，最高 1.2% | 新 Judge 幾乎是純挑選器 |
| Debate off-menu | 平均 2–3%，最高 12% | Debate 會自己產生新答案，不是純挑選器，論文要寫明 |
| Judge 與 Debate 的配對 | 各 12 個，只有 5 個重疊（EN+JA、EN+S1、EN+ZH、P1+P2、ZH+JA） | 兩者只能在這 5 個配對上比較 |
| 恆等式核對 | 逐題重算與格層級 CSV 一致 | 計算正確 |

RQ2 與 RQ1-K 補充的三點：同一個 Judge 用相同設定重跑，有 1.5–20% 的選擇會變（結果 14）；12 條 path 都有答案的題目，DeepSeek 在 MathQA 只有 92.2%，其餘區塊都在 97.9% 以上；Judge 每次呼叫輸出約 105–245 tokens，是先推理再選，不是只輸出一個字母。

### B. 十一項結果

以下未註明者為 Judge、both\_answered 子集、192 格（16 區塊 × 12 配對）。

#### 結果 1：增益取決於跟誰比（支柱，已依 RQ1-K 改寫）

在同一份菜單裡公平比較：12 條 path 一起多數決的正確率是 84.67%。跟預設的英文 path 比多 1.5pp，跟最強的單一 path 比只多 0.3pp，而且分不出正負。

| 跟誰比 | 對手的正確率 | 多數決多出多少 |
| --- | --- | --- |
| 預設的英文 path | 83.17% | +1.50pp |
| 12 條中最強的單一 path（用另一半題目選出） | 84.37% | +0.30pp（區間 −0.63 到 +1.23），12 / 16 區塊為正 |

- 成本：多數決要 12 次呼叫，輸出 tokens 是單一 path 的 11.5 倍。
- 小菜單跟 12 條中最強的比：語言三條 −1.40pp（區間 −2.31 到 −0.49），確定較差；採樣三條 −0.75pp、五種語言（原論文的 5-Lang MV）−0.83pp，區間都含 0；persona 三條 +0.04pp。
- 不含 persona 的菜單都輸，含 persona 的才打平。三條 persona 的多數決（約 84.3%）和 12 條（84.67%）幾乎一樣，所以增益主要來自菜單裡有沒有強的 path。
- 最強單一 path 多半是 persona；TruthfulQA 上幾乎每次都選到 skeptic。
- 舊的說法「兩條 path 的聚合在配對內基準下 81% 勝出，對 14 條中最強只剩 27%」保留為描述，不當主張，因為兩條對 14 選 1 不公平。
- 範圍：以上只針對多數決。事先登記的判定結果是「無法判定」，所以 Judge 版本是必要的後續實驗。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Kuncheva, *Combining Pattern Classifiers*（2014） | 佐證 | single best 與 oracle 是評估組合方法的兩個參考點 |
| Cawley & Talbot（JMLR 2010） | 警告，已處理 | 同一批資料選擇又評估會樂觀偏誤；最強 path 都用另一半題目選 |
| Li et al., *Rethinking Mixture-of-Agents*（2025，待核對） | 類似 | 只重複取樣單一最強模型，勝過混合多個模型 |
| Wang et al., *Rethinking the Bounds of LLM Reasoning*（ACL 2024，作者待核對） | 類似 | 單一 agent 配上好的 prompt 可追平多 agent 討論 |
| Li et al., *More Agents Is All You Need*（2024） | 部分相反 | 增加成員加多數決就能提升；我們的 12 條比 3 條 persona 只多約 0.3pp |
| Zheng et al., *Is "A Helpful Assistant" the Best Role?*（Findings EMNLP 2024） | 相反 | persona 不比不加好；我們的最強單一 path 幾乎都是 persona |
| Kong et al., *Role-Play Prompting*（NAACL 2024，待核對） | 類似 | 角色扮演提示可提升零樣本推理 |

#### 結果 2：多數格子該聚合，但近半數差距很小

配對內基準下 81% 該聚合、19% 不該；45% 的格子落在 ±0.5pp 以內。不含英文的配對 84%，含英文的 79%。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Geifman & El-Yaniv（NeurIPS 2017） | 相關 | 決定要不要動用較貴處理的規則，要沿整個範圍評估 |

#### 結果 3：多樣性多，不等於賺多（支柱）

| 配對 | headroom | recovery | recovery\_blind | Excess |
| --- | --- | --- | --- | --- |
| EN+ZH（換語言） | 6.4pp | 0.35 | 0.26 | 0.21pp |
| EN+S1（同語言重抽） | 3.1pp | 0.25 | 0.00 | 0.84pp |

語言的 headroom 是採樣的兩倍，Excess 卻只有四分之一。同區塊內 EN+S1 的 Excess 比 EN+ZH 高 0.63pp（區間 0.21 到 1.06，16 個區塊中 11 個同向）。原因在 recovery\_blind：英文那條本來就強，直接用它已拿走大部分好處。

三條 path 加多數決也重現了（RQ1-K）：

| 菜單 | headroom | recovery | recovery\_blind | 比菜單內最強的一條高多少 |
| --- | --- | --- | --- | --- |
| 語言三條（英、中、日） | 9.5pp | 0.26 | 0.27 | −0.28pp（區間 −1.03 到 +0.46） |
| 採樣三條（英文、S1、S2） | 4.7pp | 0.12 | 0.00 | +0.64pp（區間 +0.22 到 +1.05），14 / 16 |

語言的 headroom 仍是採樣的兩倍，但它的多數決沒有贏過自己最強的一條；採樣的有。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Kuncheva & Whitaker（Machine Learning 2003） | 佐證 | 十種多樣性度量與集成準確率的關聯都很弱 |
| Wood et al., *A Unified Theory of Diversity*（JMLR 2023） | 佐證 | diversity 依賴標籤與個體能力，不是可獨立調高的量 |
| Gao et al., *Could Thinking Multilingually Empower LLM Reasoning?*（2025） | 相容 | 多語言抬高 Acc@k 上界，但 Vote@k 的優勢消失 |
| Wang et al., *Self-Consistency*（ICLR 2023）；Li et al., *More Agents Is All You Need*（2024） | 表面相反 | 更多樣本持續提升；但他們比的是單次取樣，不是單一最強 |

#### 結果 4：recovery 隨多樣性來源而不同

Persona 0.24、採樣 0.26、自我修正 0.27、改寫 0.32、簡短 CoT 0.32、語言 0.38。差距比舊資料小，但在一種來源量到的 recovery 仍不能直接沿用到另一種。

RQ2 之後要加一句限制：recovery 同時取決於裁判能力與分歧題的難度（結果 10），所以來源之間的差異可能來自分歧題難度不同，不一定是聚合器對不同來源的處理能力不同。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Lai, Zhang, Nissim, *Multidimensional Consistency*（2025） | 佐證 | 推理一致性依變異維度（順序、改寫、語言）而不同 |
| Gao et al.（2025） | 佐證 | 多語言、重複取樣、改寫是表現不同的多樣性來源 |

#### 結果 5：Debate 比 Judge 多一點，但貴很多（已改寫）

在 5 個共同配對 × 16 區塊（80 格）上：Debate 的 recovery 高 0.065（區間 0.02 到 0.11，13 個區塊同向）；換算成正確率 +0.28pp（區間 −0.01 到 +0.58）。聚合端輸出 tokens：Judge 15–22，Debate 73–164，是 4–8 倍。差距集中在 Qwen 與 DeepSeek，GPT 與 Gemini 接近 0。舊結論「兩者一樣好」不再成立。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Du et al., *Multiagent Debate*（ICML 2024） | 類似 | 同一模型多實例辯論可提升事實性與推理 |
| Liang et al., *Encouraging Divergent Thinking*（EMNLP 2024） | 類似 | 辯論優於自我反思 |
| Smit et al., *Should We Be Going MAD?*（ICML 2024） | 部分相反 | 對齊預算後辯論沒有可靠優勢；我們是有小幅優勢、成本高 |
| Chen, Zaharia, Zou, *Are More LLM Calls All You Need?*（NeurIPS 2024） | 相關 | 更多呼叫不一定更好，要看題目難度 |

#### 結果 6：換語言沒有比同語言重抽好（已改寫，回答 R1-W1）

| 比單用英文高多少 | Judge | Debate |
| --- | --- | --- |
| EN+S1（同語言重抽） | +0.64pp（0.22 到 1.06），13 / 16 | +0.87pp（0.43 到 1.30），16 / 16 |
| EN+ZH（換語言） | +0.31pp（−0.42 到 1.04），9 / 16 | +1.00pp（0.04 到 1.96），14 / 16 |
| 兩者相減（重抽 − 換語言） | +0.33pp（−0.15 到 0.81） | −0.13pp（−0.91 到 0.64） |

兩種聚合器下，兩者的最終正確率都分不出高下；換語言就算有優勢，最多也只有 0.9pp。可以主張的是「增益不需要換語言」，不能主張「重抽勝過換語言」。

三條 path 加多數決的版本（RQ1-K 判定三，同一批題目）：採樣三條比語言三條高 0.65pp（區間 −0.06 到 +1.37），10 / 16 區塊為正，事先登記的判定是「無法判定」。方向隨資料集而變，四個模型一致：CommonsenseQA 上重抽好 2.0–3.4pp；TruthfulQA 上換語言略好 0.1–2.3pp。五種語言的多數決（原論文的基準）只比單用英文高約 0.4pp，輸出 tokens 是 5.4 倍。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Du et al.（ICML 2024） | 佐證 | 同語言、同模型的多實例辯論本身就有效 |
| Qin et al., *Cross-lingual Prompting*（2023）；Huang et al., XLT（2023） | 相反 | 跨語言提示能提升推理；但比較基準是單語平均，不是單一最強 |
| Zhang et al., AutoCAP（Findings ACL 2024） | 相反 | 自動挑語言並加權整合多語言推理路徑，可提升 zero-shot CoT |
| Wendler et al., *Do Llamas Work in English?*（2024） | 機制 | 模型內部以英文為樞紐，解釋英文候選為何系統性較強 |

#### 結果 7：自我修正很少改答案（部分成立）

自我修正的 d 是 0.035，同語言重抽是 0.081，不到一半。兩者的 recovery 相近（0.25 對 0.25）。標準自我修正相對原答案只多 0.37pp（英文，區間 0.00 到 0.74）與 0.06pp（中文，區間 −0.28 到 0.39）。例外：DeepSeek 的自我修正改答案的頻率和重抽差不多（0.059 對 0.066）。每格只有約 55 題分歧題，recovery 的數字很晃。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Huang et al., *LLMs Cannot Self-Correct Reasoning Yet*（ICLR 2024） | 佐證 | 無外部回饋時，自我修正常讓推理變差 |
| Liang et al.（EMNLP 2024） | 佐證 | Degeneration-of-Thought：模型有信心後不再產生新想法；d 是它的直接測量 |
| Kamoi et al.（TACL 2024） | 佐證 | 只有自身回饋時，沒有前作展示出可靠的自我修正 |
| Tyen et al.（Findings ACL 2024） | 相關 | 把自我修正拆成找錯與改錯，與我們的 d / recovery 拆法同構 |
| Madaan et al., *Self-Refine*（NeurIPS 2023） | 相反 | 7 個任務平均提升約 20%；多為生成任務 |
| Stechly, Marquez, Kambhampati（2023） | 相反 | 瓶頸在驗證；我們的資料顯示瓶頸在 d |
| Gou et al., CRITIC（ICLR 2024） | 相關 | 外部工具注入新資訊才讓自我修正有效 |

#### 結果 8：配對裡的英文數

| 英文數 | recovery | recovery\_blind | Excess |
| --- | --- | --- | --- |
| 0 個（如 ZH+JA） | 0.34 | 0.06 | 1.15pp |
| 1 個（如 EN+ZH） | 0.37 | 0.29 | 0.27pp |
| 2 個（如 EN+S1） | 0.28 | 0.08 | 0.62pp |

ZH+JA 的 Excess 最高（1.99pp），但聚合後仍比單用英文低 1.6pp，16 個區塊只贏 3 個。Excess 高是因為兩條 path 都弱，這說明為什麼需要全域基準。英文數與多樣性來源共線，不能單獨歸因。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Wendler et al.（2024） | 機制 | 英文作為潛在樞紐語言 |
| Wu et al., *The Semantic Hub Hypothesis*（ICLR 2025） | 機制 | 跨語言語意表徵集中在由主導語言支配的共享空間 |

#### 結果 9：用少量標註決定要不要聚合，贏不過固定做法（新，負面結果）

做法：題目分兩半，只用前一半做決定，在後一半評分，重複 200 次。判定標準在跑之前寫好。

| 設定 | 比「永遠聚合」高多少 | 95% 區間 |
| --- | --- | --- |
| 隨機標 200 題 | −0.24pp | −0.32 到 −0.16 |
| 只標分歧題 200 題 | −0.04pp | −0.10 到 +0.02 |
| 借用另一個模型的決定 | −0.18pp | −0.32 到 −0.05 |
| 借用另一個資料集的決定 | −0.29pp | −0.43 到 −0.15 |
| 借用另一種多樣性來源的決定 | −0.15pp | −0.20 到 −0.11 |

- 空間（完美決策最多能贏多少）只有 0.17pp，低於事先訂的 0.2pp 門檻。
- 全域基準下，「永遠用最強單一 path」是較好的固定做法；決策最多再多 0.12pp（區間 0.03 到 0.20），遠低於門檻。
- 只標分歧題省約 8 倍標註：25 題的效果（−0.22）接近隨機 200 題（−0.24）。

之後又檢查了兩個可能有空間的情境，結論不變：

- **裁判和作答者不同（RQ2 的 192 格）**：空間 0.15pp。連最不利的組合（強模型的候選、弱裁判），聚合平均也只輸 0.13pp。配對內基準下聚合輸的時候輸很少，所以永遠聚合就對了。
- **12 條多數決對最強單一 path（RQ1-K 判定二）**：空間 0.52pp，剛過事先訂的 0.5pp 門檻，但全部來自 TruthfulQA 的四個區塊。就算每次都猜對，也只比永遠聚合多 0.52pp（區間 −0.10 到 +1.13），含 0。會翻轉的單位是資料集，而我們只有四個資料集、其中一個該不聚合，所以現有資料不可能驗證一個決策方法。這一點寫成觀察（結果 15），不寫成方法。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Hu et al., RouterBench（2024，待核對） | 類似 | 實際的路由器多半只和最佳單一模型打平 |
| Ong et al., RouteLLM（2024，待核對）；Madaan et al., AutoMix（NeurIPS 2024） | 相反 | 路由有明顯價值；但他們逐題決定，且兩個選項的成本與正確率差很多 |
| Maia Polo et al., tinyBenchmarks（ICML 2024） | 相反 | 約 100 題可估準確率；估「該不該聚合」時不成立 |
| Kossen et al., *Active Testing*（ICML 2021） | 佐證 | 主動挑要標註的題目可省標註 |

#### 結果 10：recovery 是裁判能力減分歧題難度（支柱，RQ2 交叉實驗）

做法：候選答案固定，只換裁判。4 個模型的候選各交給 4 個模型裁決，3 個配對（EN+ZH、EN+S1、P1+P2）、4 個資料集；prompt 與選項順序都和主網格相同。事先登記的判定是 (c)：兩個效果都存在。

| 效果 | 數字 | 意思 |
| --- | --- | --- |
| 裁判效果（強裁判 − 弱裁判，候選相同） | +0.32（區間 0.22 到 0.42），16 / 16 區塊為正 | 同一批候選，強模型當裁判挑得好很多 |
| 候選效果（強模型的候選 − 弱模型的候選，裁判相同） | −0.22（區間 −0.30 到 −0.15），0 / 16 為正 | 強模型的候選反而難挑；強模型自己兩次答案不同的題目，本來就是最難的題目 |

- 換成「剛好一條對時挑對的比例」（亂猜 50%）：弱模型當裁判約 59%，強模型約 72–78%；弱模型的候選配強裁判 79–87%；強模型的候選配弱裁判 53–58%。
- 主網格看到「強模型的 recovery 高約 0.11」，是 +0.32 與 −0.22 相抵的淨值。
- 分資料集：裁判效果在 TruthfulQA 最大（0.55），CommonsenseQA 最小（0.18）；候選效果在 MathQA（−0.34）與 MMLU（−0.30）最大。
- 限制：強模型沒有更強的裁判可以測，分不出它們的 recovery 是被裁判限制，還是那些題目本來就救不了。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Zheng et al., *Judging LLM-as-a-Judge*（NeurIPS 2023） | 類似 | 評審能力隨模型強度上升，對應裁判效果 |
| Wang et al., Mixture-of-Agents（2024） | 類似 | 提案與聚合是不同的能力，對應兩個效果可以分開（細節待核對） |
| Tyen et al.（Findings ACL 2024） | 方法範本 | 固定其他部分、只替換一段，來驗證哪一段是瓶頸 |

#### 結果 11：決策空間集中在 path 正確率低的地方（新）

| path 平均正確率 | 格數 | 聚合反而較差的比例 | 空間 | recovery | recovery\_blind |
| --- | --- | --- | --- | --- | --- |
| 72% 以下 | 15 | 67% | 0.44pp | 0.21 | 0.17 |
| 72–78% | 49 | 31% | 0.22pp | 0.22 | 0.12 |
| 78–84% | 56 | 5% | 0.02pp | 0.30 | 0.09 |
| 84–90% | 22 | 5% | 0.00pp | 0.51 | 0.18 |
| 90% 以上 | 50 | 16% | 0.01pp | 0.33 | 0.07 |

正確率越低空間越大（相關係數 −0.33）。最弱那組的 recovery 已貼近 recovery\_blind，代表裁判快挑不出來。這 15 格多半來自 GPT 與 Qwen 的 TruthfulQA，分不清是模型弱還是資料集特別。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Chen, Zaharia, Zou（NeurIPS 2024） | 類似 | 多次呼叫加投票在難題上反而有害 |
| Li et al., *Rethinking Mixture-of-Agents*（2025，待核對） | 類似 | 混入較弱的模型會拉低聚合品質 |

#### 結果 12：模型裁決自己的答案最差（新，事後觀察）

把 4×4 表用列平均與欄平均算出每格的預期值，自己裁決自己的格子比預期低 0.085（區間 −0.110 到 −0.060），16 個全部為負。對照組（同強度但不是自己）是 −0.011（區間 −0.034 到 +0.013），所以這是「自己」特有的，不是強弱相同造成的。

- 事先登記的檢驗只看正方向（自我偏好），結論是「不能說有自我偏好」。負方向是事後觀察，論文裡要照實標明。
- 主網格全部是自己裁決自己，正好是裁判最吃虧的組合。所以論文的範圍要寫明是單一模型內的聚合。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Panickssery, Bowman, Feng（NeurIPS 2024） | 表面相反 | LLM 評審偏好自己的輸出。我們的設定裡兩個候選都是自己的，測到的是「挑不出自己哪個對」，不是同一件事 |
| Jiang et al., *Self-\[In\]Correct*（2024，作者與出處待核對） | 類似 | LLM 分辨自己多個答案孰優孰劣的能力，不比直接作答可靠 |
| Huang et al.（ICLR 2024）；Stechly et al.（2023） | 類似 | 模型無法可靠地評判自己的推理 |
| Koo et al., *Benchmarking Cognitive Biases in LLMs as Evaluators*（Findings ACL 2024，待核對） | 相關 | 系統性測量 LLM 評審的多種偏誤，包含自我中心偏誤 |

#### 結果 13：弱模型的候選換強裁判，正確率多 2.47pp（新）

GPT-4o mini 與 Qwen 的候選改由 DeepSeek、Gemini 裁決，正確率比自己裁決高 2.47pp（區間 1.65 到 3.29），8 個區塊全為正，TruthfulQA 上多 3.9pp。公式說弱模型的瓶頸在裁判，換掉裁判後真的變好，診斷通過了介入檢驗。

- 還不能當實務建議：有強模型時，直接讓它作答可能更好。「強裁判是不是只是自己知道答案」的離線檢查還沒做（見第五節）。
- 只驗證了 aggregator 這一側。公式說 generator 有問題的地方（例如自我修正很少改答案），沒有做介入。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Chen, Zaharia, Zou, FrugalGPT（2023）；Madaan et al., AutoMix（NeurIPS 2024） | 相關 | 便宜模型先答、必要時才升級到貴的模型；「答案不同才請強模型」是同一類設計 |

#### 結果 14：同一個 Judge 重跑，選擇會變（新）

流程核對時，用相同的 prompt、相同的設定（T = 0）、相同的模型版本重跑「自己裁決自己」，和主網格逐題比對：

| 模型 | 題數 | 選擇相同的比例 |
| --- | --- | --- |
| Qwen3-8B | 194 | 98.5% |
| Gemini 3.1 Flash-Lite | 60 | 91.7% |
| GPT-4o mini | 178 | 89.3% |
| DeepSeek V4.1 Flash | 91 | 80.2% |

- 單一格的 recovery 不可信：DeepSeek 的 MathQA × EN+S1 重跑後從 0.16 變成 −0.10。
- 跨 16 個區塊的平均與區間仍可用，這種雜訊已包含在區塊之間的差異裡。
- 附錄要報這四個比例，並寫明每題只跑一次。這也回應 R3 問的「每個設定跑幾次」。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Atil et al., *Non-Determinism of "Deterministic" LLM Settings*（2024，待核對） | 類似 | 即使設定成確定性輸出，LLM 重跑的結果仍會不同，正確率可差到十幾個百分點 |

#### 結果 15：多數決贏不贏，由資料集決定（新，RQ1-K）

12 條 path 多數決減掉最強單一 path：

| 資料集 | 四個模型的平均 | 各模型的方向 |
| --- | --- | --- |
| MathQA | +1.36pp | 4 個都為正（弱模型 +2.2、+3.0；強模型約 +0.1） |
| CommonsenseQA | +1.01pp | 4 個都為正 |
| MMLU | +0.90pp | 4 個都為正 |
| TruthfulQA | −2.07pp | 4 個都為負（−0.3 到 −4.0） |

- TruthfulQA 上有一條特別適合這個任務的 path（skeptic persona），多數決把它和其他 11 條一起投票，就被票數壓過。用公式的話說，是 recovery\_blind 很高。
- 能寫的是觀察：只有當某一條 path 特別適合任務時，「不聚合」才有價值，在那種任務上值 2–4pp；而且這是資料集的性質，四個模型方向一致。
- 不能寫成經過檢驗的決策方法：只有四個資料集，其中一個該不聚合。

| 論文 | 立場 | 論點 |
| --- | --- | --- |
| Lin et al., TruthfulQA（2022） | 解釋 | 這個資料集專門收集人類常見的錯誤信念，模型傾向重複它們，所以多數的 path 會一起答錯 |
| Sprague et al., *To CoT or Not to CoT*（2025） | 類似 | 推理類技巧的好處集中在數學與符號推理，對應 MathQA 上多數決贏最多 |
| Wang et al., *Self-Consistency*（ICLR 2023） | 類似 | 多數決的增益在算術推理上最明顯 |
| Dehghani et al., *The Benchmark Lottery*（2021，出處待核對） | 相關 | 方法的相對優劣會隨選用的 benchmark 改變 |

### C. 已撤回、不要再用的主張

- 「Debate ≈ Judge」：改成結果 5 的寫法。
- 「同語言重抽勝過換語言」：改成結果 6 的寫法。
- 「決策可以跨模型沿用」：兩個模型時打平，四個模型時 −0.18pp，預測被推翻。
- 「recovery 的變異只有 0.5% 來自模型」：強弱模型差 0.13–0.16，不成立。
- 舊分析的 160 對與 legacy 24 格全部數字（recovery ≈ 0.34、13 對虧損等）：來自已換掉的模型與舊 Judge。
- v1 文件已列的撤銷項目仍然撤銷：I² 異質性分析、EIV 校正、跨來源等價的 TOST 宣稱、對 Gao et al. 的「recovery ≥ 0.60 才划算」。

RQ2 與 RQ1-K 之後新增的撤回：

- 「聚合只在 27% 的情況贏過單一最強 path」當作主張：那是兩條對 14 選 1，降為描述，主張改用結果 1 的公平比較。
- 「recovery 是 aggregator 的性質」以及「強模型的 recovery 較高，代表強模型比較會挑」：recovery 是裁判能力減分歧題難度，主網格的差距是淨值。
- 「模型會偏好自己的答案」：資料顯示相反，模型裁決自己的答案最差。
- 「RQ1 可以在 12 條多數決的情境救回來」：空間過了門檻，但全部來自一個資料集，現有資料無法驗證。
- 「加更弱的模型可以讓 RQ1 過關」：弱模型的決策空間較大（約 0.3pp 對 0.05pp），但決策在弱模型上也只是打平。

## 三、建議的新方向

定位成分析論文：同一個模型內，測試時的多樣性聚合什麼時候值得付錢。不提出新方法。貢獻是一把把「單一最強 path」放進去的尺、用它得到的四個主張，以及三個照事先標準做完的實驗（其中一個是負面結果）。目標是 Findings，Main 機會不高。

暫定標題（二選一）：

- *When Does Test-Time Diversity Pay Off? Decomposing LLM Aggregation Gains Against the Strongest Single Path*
- *Diversity Is Not Gain: Measuring LLM Aggregation Against the Strongest Single Path*

### 核心式

```latex
\text{Excess} = d \cdot (c - m) \cdot (\textit{recovery} - \textit{recovery}_{\textit{blind}})
```

這是恆等式，像記帳本，永遠成立，不能單獨當貢獻。貢獻在於用它讀出來的東西，以及驗證「照讀數去做有沒有用」。它對兩條 path 與 K 條 path 都成立（RQ1-K 的 25,600 次切分全部核對通過）。

### 論文要講的四個主張，加一個成本結論

1. **增益取決於跟誰比。** 12 條 path 多數決比預設的英文 path 高 1.5pp，比最強的單一 path 只高 0.3pp（分不出正負），成本是 11.5 倍。增益主要來自菜單裡有沒有強的 path（結果 1）。
2. **多樣性多不等於賺多。** 語言的 headroom 是採樣的兩倍，聚合後沒有比較高；增益不需要換語言。兩條與三條 path 都成立（結果 3、6、8）。
3. **要不要聚合由資料集決定，逐任務決策沒有額外好處。** 多數決在三個資料集小贏、在 TruthfulQA 輸 2pp，四個模型同向；用少量標註逐任務決定，贏不過固定做法。附一個任何人都能用的檢查：先算固定做法的 regret（結果 9、11、15）。
4. **recovery 是裁判能力減分歧題難度。** 裁判效果 +0.32，候選效果 −0.22；模型裁決自己的答案最差；弱模型換強裁判正確率多 2.47pp（結果 10、12、13）。
5. **成本與穩定性。** Debate 的 recovery 只高 0.065，聚合端 tokens 是 4–8 倍；同一個 Judge 重跑有 1.5–20% 的選擇會變（結果 5、14）。

### 三個 RQ 的現況

| RQ | 問題 | 答案 | 還缺什麼 |
| --- | --- | --- | --- |
| RQ1 | 聚合什麼時候贏過單一最強 path？能不能事先決定？ | 增益取決於基準；方向由資料集決定；少量標註的逐任務決策贏不過固定做法（負面結果） | K 條 path 的 Judge 版本（事先登記為必要） |
| RQ2 | 聚合沒賺到時，瓶頸在 generator、aggregator，還是單一 path 太強？ | 弱模型的瓶頸在 aggregator，經介入驗證；自我修正卡在很少改答案；語言卡在英文那條太強 | generator 這一側沒有做介入；「強裁判是否只是自己知道答案」的離線檢查 |
| RQ3 | K 條 path 中該換掉或改進哪一條？ | 還沒做 | 可以在 RQ1-K 的逐題資料上離線做：拿掉一條再多數決，用另一半題目驗證 |

### 範圍聲明（要寫進論文）

- 研究對象是單一模型內的聚合：候選答案與裁判是同一個模型。跨模型裁判只用來診斷，不是推薦的系統。
- 目前關於「聚合對單一最強」的結論只針對多數決；Judge 版本完成後才能下一般性的結論。
- 「由資料集決定」建立在四個資料集上，其中只有一個方向相反。

### 這個方向如何回應每一條審稿意見

| 審稿意見 | 回應 |
| --- | --- |
| R2：novelty 有限，只是元件組合 | 不再提出聚合方法；貢獻是分解、以單一最強為基準的實證、事先訂標準的實驗（含一個負面結果）、交叉實驗 |
| R1-W1：缺同語言對照組 | 結果 6：兩條、三條 path 下，同語言重抽與換語言都分不出高下，增益不需要換語言 |
| R1-W2：五種語言沒有理由 | 語言只是六種多樣性來源之一；結果 8 說明英文在配對中的作用；五種語言的多數決只比單用英文高約 0.4pp |
| R1-W3、R2：router 的泛化與成本 | router 移除；結果 9 顯示即使完美的逐任務決策，空間也只有 0.17pp |
| R3：只有 17 篇引用 | 約 70 篇，見第六節 |
| R3：統計不透明、每個設定跑幾次 | 附錄 A：每題跑一次；Judge 重跑的一致率（結果 14）；區間怎麼算；三份判定標準檔與雜湊值 |
| R3：增益可能來自更多計算 | 結果 5：Debate 多花 4–8 倍 tokens 只多約 0.28pp；結果 1：12 條 path 花 11.5 倍 tokens 只比最強單一 path 多 0.3pp |
| R3：混語辯論不穩 | 結果 6 的 Debate 欄：混語（EN+ZH）與同語（EN+S1）的辯論結果相當 |
| R3：best fixed pair 為何因模型而異 | 結果 3、8、10：差別來自配對中較強那條有多強，以及分歧題的難度 |
| Reproducibility 1–3、Software 1 | 匿名 repo：prompt 全文、逐題資料、分析程式、三份判定標準檔 |

### 刻意放棄的主張

- IMSR 作為方法貢獻，以及動態語言路由器。
- 「用少量標註決定要不要聚合」作為方法貢獻（降為負面結果）。
- 「聚合很少贏過單一最強」這個較強的說法（改成「跟最強的比，增益分不出正負」）。
- 「同語言重抽勝過換語言」「Debate 等於 Judge」「recovery 是 aggregator 的性質」。

## 四、新論文的完整邏輯流程

主線：文獻對測試時多樣性的結論不一致，部分原因是比較的基準不同 → 我們給一把把「單一最強」放進去的尺 → 用它發現增益取決於跟誰比、多樣性多不等於賺多 → 檢驗「能不能事先決定要不要聚合」，答案是不能，方向由資料集決定 → 用交叉實驗把 recovery 拆成裁判能力與分歧題難度。標「待做」者尚未完成。

### §1 Introduction

- **要說的話**：自我一致性、多語言、persona、辯論、自我修正都在製造多樣性再聚合，但結論不一致。多數研究拿「平均」或「單次作答」當基準，不是單一最強的 path。
- **實驗證據**：結果 1 當首圖（多數決比英文高 1.5pp，比最強單一 path 只高 0.3pp，成本 11.5 倍）；結果 3 當第二個鉤子（headroom 兩倍、聚合後沒有比較高）。
- **文獻與論點**（哪些不一致可以用基準解釋，要分開寫）：
  - 辯論：Du et al.（2024）說有效；Smit et al.（2024）說對齊預算後沒有可靠優勢；Wang et al.（ACL 2024）說單一 agent 配好 prompt 可追平。基準不同是合理解釋。
  - 跨語言：Qin et al.、Huang et al. XLT、AutoCAP 說有效；Gao et al.（2025）說上界變高但投票後優勢消失。基準不同是合理解釋。
  - 多取樣：Wang et al.（2023）、Li et al.（2024）說越多越好；Chen, Zaharia, Zou（2024）說難題上變差；Self-MoA（2025，待核對）說混合不如只用最強的。基準不同是合理解釋。
  - 自我修正：Madaan et al.（2023）說平均提升約 20%；Huang et al.（2024）說變差。這一組主要是任務不同，以及前人是否用標準答案決定何時停，不能歸給基準。
  - 論文只能寫「部分不一致可以由基準解釋」。我們直接證明的是自己資料裡的那個落差。

### §2 Related Work

| 小節 | 收錄論文 | 我們的定位 |
| --- | --- | --- |
| 2.1 集成多樣性理論 | Kuncheva & Whitaker（2003）、Wood et al.（2023）、Kuncheva（2014）、Cruz et al.（2018）、Caruana et al.（2004） | 他們指出多樣性與準確率關係弱；我們用 recovery\_blind 說明原因 |
| 2.2 LLM 測試時聚合 | Wang et al.（2023）、Chen et al. USC（2023）、LLM-Blender（2023）、Mixture-of-Agents（2024）、Self-MoA（2025）、Li et al.（2024）、Chen, Zaharia, Zou（2024）、Brown et al.（2024） | 他們提出或檢視聚合方法；我們提供以單一最強為基準的尺 |
| 2.3 辯論、自我修正、LLM 評審 | Du、Liang、Smit、Wang（Rethinking the Bounds）、Huang（2024）、Kamoi、Madaan、Shinn、Tyen、Stechly、Gou、Zheng（MT-Bench）、Panickssery、Self-\[In\]Correct、Koo | 把相反的結論放進同一個分解；用交叉實驗分開裁判與候選 |
| 2.4 多語言推理 | Shi et al.（2023）、Qin、Huang XLT、AutoCAP、mGRPO、Gao、Lai、CLC、Elhady、Wendler、Wu | 語言是多樣性來源之一，不是主角 |
| 2.5 路由與何時升級 | FrugalGPT、AutoMix、RouteLLM、RouterBench、Song et al.（GV-gap）、Maia Polo | 他們逐題決定要不要升級；我們檢驗逐任務的聚合決策，並給出決策空間的上限 |

### §3 Framework

| 小節 | 內容 | 證據 | 文獻與論點 |
| --- | --- | --- | --- |
| 3.1 分解 | Gain = d·(c−m)·recovery；「永遠用最強那條」也是一種聚合器，得到 recovery\_blind；相減得 Excess。K 條 path 的版本定義相同 | 恆等式在兩條與 K 條都核對通過 | Wood et al.：把集成誤差拆成個體誤差與多樣性；Song et al.（待核對）：GV-gap 以 1−m 正規化，我們以 c−m 正規化並放入「不聚合」 |
| 3.2 兩種基準 | 菜單內最強的一條；所有 path 中最強的一條 | 結果 1 | Kuncheva（2014）：single best 與 oracle 兩個參考點 |
| 3.3 決策的上限 | 兩個固定做法的 regret 取較小者，就是任何決策規則最多能贏的量 | 結果 9、11、15 | RouterBench：上限高不代表路由器抓得到 |
| 3.4 假設 | 封閉性（最終答案在候選之中）；一致題不受聚合影響 | Judge 與多數決成立；Debate 有 2–3% 不在候選中，只近似成立 | LLM-Blender：區分選擇式與生成式融合 |
| 3.5 測量 | 最強 path 在另一半題目上選出；統計單位是模型 × 資料集的區塊；判定標準事先登記 | 16 區塊；三份判定標準檔 | Cawley & Talbot：選擇與評估要分開資料；Cameron et al.：分組少時標準誤會偏小；van Miltenburg et al.（待核對）：事先登記；Altman & Bland：不顯著不等於沒有效果 |

### §4 Experimental Setup

- **主網格**：4 模型 × 4 資料集；14 條 path（5 語言、採樣 2、persona 2、改寫 2、簡短 CoT、自我修正 2）；兩條 path 的聚合器 Blind、Judge、Debate。
- **RQ1**：題目分兩半，一半做決定、一半評分，200 次；五種資訊來源。
- **RQ2**：候選固定，4 個 generator × 4 個 Judge，3 個配對。
- **RQ1-K**：8 份菜單（三條的同來源菜單四份、五種語言、12 條、8 條英文、14 條）加多數決；Judge 版本待做。
- **要寫明的細節**：各模型版本與日期、thinking 設定、temperature；Judge 輸出選項、順序隨機並沿用；每題跑一次與重跑的一致率；Judge 與 Debate 只有 5 個共同配對。
- **文獻與論點**：
  - 資料集：MMLU（Hendrycks et al.）、CommonsenseQA（Talmor et al.）、MathQA（Amini et al.）、TruthfulQA（Lin et al.）。
  - Wang et al., *LLMs Are Not Fair Evaluators*（ACL 2024）：評審有位置偏誤，所以順序要隨機。
  - Zheng et al., *Not Robust Multiple Choice Selectors*（ICLR 2024）：LLM 偏好特定選項代號。
  - Sclar et al., FormatSpread（ICLR 2024）：格式差異會改變表現，所以各 path 格式鎖死。
  - Card et al.（EMNLP 2020）、Dodge et al.（EMNLP 2019）、Biderman et al.（2024）：先估能分辨的最小差距，並揭露實驗細節。
  - Deng et al., *Rephrase and Respond*（2023）：改寫軸的依據。Renze & Guven（2024）：簡短 CoT 的依據。
  - Atil et al.（2024，待核對）：設定成確定性輸出，重跑仍會不同。

### §5 Results

| 小節 | 主張 | 實驗 | 文獻與論點 |
| --- | --- | --- | --- |
| 5.1 增益取決於跟誰比 | 多數決比英文高 1.5pp，比最強單一 path 只高 0.3pp；增益來自菜單裡有沒有強的 path | 結果 1、2；Judge 版本待做 | Kuncheva（2014）：兩個參考點；Self-MoA：只用最強的較好；Li et al.（More Agents）：部分相反；Zheng et al.（persona）：相反；Kong et al.：類似 |
| 5.2 多樣性多不等於賺多 | 語言 headroom 兩倍但聚合後沒有比較高；增益不需要換語言 | 結果 3、6、8 | Kuncheva & Whitaker：多樣性度量預測力弱；Wood et al.；Gao et al.：上界高但投票後消失；Qin、Huang XLT、AutoCAP：相反，基準不同；Wendler、Wu：英文是樞紐 |
| 5.3 能不能事先決定要不要聚合 | 逐任務決策贏不過固定做法；方向由資料集決定；空間集中在一條 path 特別強或 path 正確率低的地方 | 結果 9、11、15 | RouterBench：類似；RouteLLM、AutoMix：相反，逐題且選項差距大；Maia Polo：估準確率 100 題夠，估決策不夠；Kossen：只標有資訊的題目；Søgaard：隨機切分高估；Lin et al.：TruthfulQA 的特性 |
| 5.4 瓶頸在哪 | recovery 是裁判能力減分歧題難度；模型裁決自己最差；弱模型換強裁判 +2.47pp；自我修正卡在很少改答案 | 結果 4、7、10、12、13 | Tyen：固定其餘、只換一段的驗證法；Zheng et al.（MT-Bench）：評審能力隨模型上升；Panickssery：表面相反；Self-\[In\]Correct、Huang（2024）、Stechly：類似；Mixture-of-Agents；Kamoi、Liang、Madaan |
| 5.5 成本與穩定性 | Debate 多 0.065 的 recovery、4–8 倍 tokens；Judge 重跑不穩定 | 結果 5、14 | Du、Liang：辯論有效；Smit：對齊預算後無可靠優勢；Chen, Zaharia, Zou；Atil et al. |
| 5.6 該換哪條 path | 拿掉一條後的變化，用另一半題目驗證 | 待做（離線） | Caruana：從模型庫選子集；Ghorbani & Zou：貢獻歸因 |

### §6 Discussion

- **給使用者的建議**：先找單一最強的 path（多半是和任務相關的 persona）；要聚合就用同語言重抽；蓋決策或路由系統之前，先算固定做法的 regret；不要讓弱模型裁決自己。
- **對 Gao et al. 的互補讀法**：多語言同時抬高 headroom 與 recovery\_blind。措辭用「補充」，不寫「糾正」。
- **與 Self-MoA 的關係**：結論同方向；我們多了分解式、同一模型內的六種來源、決策層面的負面結果，以及裁判與候選的拆解。
- **強裁判的解讀**：待離線檢查完成後再寫。若強裁判的正確率和它自己直接作答差不多，就寫「裁判只是知道答案」。

### §7 Limitations

- 只有選擇題，答案可直接比對；自由作答未測。
- 只有四個資料集；「方向由資料集決定」只有一個反向的例子（TruthfulQA）。
- 沒有開啟 thinking 的推理模型；Gemini 3.1 的 thinking 無法完全關閉。
- 強模型沒有更強的裁判可測；generator 這一側沒有做介入。
- 「模型裁決自己較差」是事後觀察，不是事先登記的假設。
- Judge 每題只跑一次，重跑有 1.5–20% 的選擇不同；單格估計不穩。
- 12 條 path 的菜單組成是實驗設計的偶然，結果會隨組成而變（已並列 8 條英文的版本）。
- Debate 不是純挑選器；Judge 與 Debate 只有 5 個共同配對。
- RQ1 的決策規則本身略偏向不聚合。

### 附錄

- **A 統計透明度**：每題跑一次、重跑一致率；區間與四種狀態的判定方式；三份判定標準檔、雜湊值與確認時間（回應 R3）。
- **B 可重現性**：prompt 全文、模型版本、逐題資料、程式碼。
- **C 逐格完整表格**。
- **D RQ1 的完整曲線、遷移與敏感度分析**。
- **E RQ2 的分資料集、分配對結果；RQ1-K 的各菜單與敏感度**。

## 五、待辦清單、各階段錄取機會與 Novelty 比較

下一步是 K 條 path 的 Judge 版本，它在 RQ1-K 的判定標準裡已被列為必要。之後是兩個不花錢的離線分析，然後就是寫作。以下機率是判斷而非計算，ARR 審稿雜訊很大，而且都以論文寫完為前提。

### 各階段的錄取機會

| 階段 | 完成內容 | Findings 以上 | Main | 說明 |
| --- | --- | --- | --- | --- |
| 0（現在） | 主網格、RQ1、RQ2、RQ1-K | 35–45% | 約 10% | RQ2 的結果乾淨，是目前最強的一塊 |
| 1 | + K 條 path 的 Judge 版本、強裁判的離線檢查、資料收尾 | 35–45% | 約 10% | 這是補洞。不做的話「聚合對單一最強」只能講多數決，審稿人一定會問 Judge，機率下修到 30–40% |
| 2 | + RQ3（離線） | 35–45% | 約 10% | 三個 RQ 都有答案，機率變化不大 |
| 3 | + 寫作完成：引用約 70 篇、附錄、匿名 repo、修改說明 | 同上 | 同上 | 上面的數字都以這一階段完成為前提 |
| 4（可選） | + 自由作答資料集或更多資料集，且主張重現 | 40–50% | 10–15% | 回應「只有選擇題」「只有四個資料集」 |

參考點：ACL 系列 Main 約兩成多，加上 Findings 合計約四成，所以做完全部大約回到平均水準。壓低機率的因素：是 resubmission（上一輪 2.5 / 2 / 1.5 會跟著走）；沒有新方法；一個主結果是負面的；只有選擇題與四個資料集；Self-MoA、Gao et al.、Smit et al. 已有相近結論。加分的因素：每條審稿意見都有對應回答；三份判定標準都事先寫好；宣稱沒有效果時都附信賴區間；交叉實驗的兩個效果在 16 個區塊幾乎全部同向。

### 待辦清單

**投稿行政（現在就做）**

- [ ] 確認 2027 年 1 月 cycle 的確切截止日（ARR 官網的 dates 頁，目前只寫 January 2027）
- [ ] 以 resubmission 投稿：附上一輪的連結、逐點回應的修改說明 PDF，並要求全新的審稿人與 AC
- [ ] 問教授 service contributor 的名額（每人每輪最多支援 2 篇；沒有的投稿進抽籤）
- [ ] 所有作者的 OpenReview 補上 ORCID
- [ ] ARR 作者指引更新時，重讀 resubmission 的規定
- [ ] 跟教授確認：論文重心已從 1a 移到 1b；要不要做 1c（RQ3）

**下一步實驗（依序）**

- [ ] K 條 path 的 Judge 版本（必要，約一萬次呼叫）
  - 先寫判定標準檔，確認後再跑
  - 菜單：12 條為主；另跑語言三條、採樣三條、persona 三條，和兩條 path 的 Judge、三條的多數決對照
  - 裁判：自己裁決自己，和主網格一致
  - 只跑菜單內答案不一致的題目（12 條時約 36%）
  - prompt 沿用主網格 Judge 的形式，擴充到 K 個候選；候選順序隨機並記錄
  - 判定：Judge 版本減最強單一 path（四種狀態，門檻 0.5pp）；Judge 版本減多數決的差
  - 重點看 TruthfulQA：Judge 能不能認出 skeptic 那條是對的
  - 開跑前先重跑一個區塊，報重跑的一致率
- [ ] 離線檢查：在弱模型的分歧題上，比「強模型自己作答的正確率」「強裁判挑選的正確率」「兩者答案相同的比例」。決定 Discussion 裡實務建議怎麼寫
- [ ] RQ3（離線）：在 RQ1-K 的逐題資料上，拿掉一條 path 再多數決；用一半題目找出拿掉後最有幫助的那條，在另一半驗證
- [ ] 抽 10 題 DeepSeek 在 MathQA 解析失敗的原始輸出，確認原因

**資料收尾**

- [ ] 記錄四個模型的版本 ID、呼叫日期、thinking 設定、temperature，並查各自的下架日期
- [ ] Gemini 3 官方建議 temperature 留在 1.0；我們用 T = 0，要寫進 setup 並附解析失敗率
- [ ] 檢查 Judge 選第一個選項的比例是否接近 50%
- [ ] 把所有筆記裡的舊數字換成本文件的數字
- [ ] （可選）補跑幾個配對，讓 Judge 與 Debate 的共同配對多於 5 個

**寫作（現在就能開始）**

- [ ] §3 框架（不依賴新結果）
- [ ] 圖：增益隨基準縮小（首圖）、各來源的 headroom 對 Excess、RQ2 的 4×4 熱圖、12 條多數決的逐區塊圖、RQ1 的 k 曲線（附錄）
- [ ] §1、§2、§4、§5、§6、§7
- [ ] 對最接近的論文各寫兩句區隔：Self-MoA、Gao et al.、Lai et al.、Song et al.、Smit et al.、RouterBench、AutoMix、Zheng et al.（MT-Bench）、Self-\[In\]Correct
- [ ] 核對第六節所有標「待核對」的論文（作者、年份、出處）
- [ ] 補上四個模型的引用
- [ ] 附錄 A 到 E；匿名 repo（含三份判定標準檔）
- [ ] 修改說明（回應三位審稿人，可用第三節的表當底稿）
- [ ] 教授 review
- [ ] 內容凍結（約 12 月初），之後只寫不跑
- [ ] 提交前檢查：頁數、匿名化、每個數字回到原始輸出核對

**可選**

- [ ] 一個自由作答資料集，或幾個新的資料集：先寫下預測（例如增益隨基準縮小仍成立、方向仍由資料集決定）再跑
- [ ] 推理模型：不做就寫進 Limitations

**不建議做**

- 在 12 條多數決上再跑一次驗證性的 RQ1：空間全部來自一個資料集，連完美決策都過不了顯著性。
- 為了讓 RQ1 過關而加更弱的模型：決策空間最多約 0.4pp，而且看到結果後才加，只能算探索性分析。
- 再加同類的選擇題資料集或中階模型，除非是為了增加資料集的數量來檢驗「方向由資料集決定」。

### 與最接近的論文比較 Contribution 與 Novelty

| 論文 | 他們做什麼 | 與我們重疊的地方 | 我們的差異 |
| --- | --- | --- | --- |
| Li et al., *Rethinking Mixture-of-Agents*（2025，待核對） | 比較混合多個模型與只重複取樣最強模型 | 「品質比多樣性重要」；增益來自強的成員 | 同一個模型內的六種來源；分解式；公平的同菜單比較；決策層面的負面結果 |
| Gao et al.（2025） | 比較多語言、重複取樣、改寫的上界與投票結果 | 「上界高不代表會贏」 | 給出原因（recovery\_blind 同時被抬高）；以單一最強為基準；兩條與三條都驗證 |
| Lai, Zhang, Nissim（2025） | 沿順序、改寫、語言製造變異再用一致性聚合 | 多種變異來源 | 他們提升準確率；我們問何時值得 |
| Smit et al.（ICML 2024） | 對齊預算後比較辯論與單一 agent | 辯論優勢有限 | 量化到 recovery 與 token 成本，並與單次 Judge 比 |
| Zheng et al., *Judging LLM-as-a-Judge*（NeurIPS 2023） | 評估 LLM 當評審的能力 | 強模型是較好的評審 | 把裁判與候選分開；發現強模型的候選反而難挑，主網格的差距是 +0.32 與 −0.22 的淨值 |
| Panickssery et al.（NeurIPS 2024）；Jiang et al., *Self-\[In\]Correct*（2024，待核對） | 模型評判自己輸出時的行為 | 自己評自己 | 用交叉設計量出自己裁決自己較差，並排除「強弱相同」的解釋 |
| Song et al., *Mind the Gap*（ICLR 2025，待核對） | 生成與驗證能力的差距 | 比值形式的量 | 分母是 c−m；把「不聚合」放上同一把尺 |
| Wood et al.（JMLR 2023） | 誤差的 bias–variance–diversity 分解 | 分解的想法 | 再拆成 headroom 與 recovery，並納入單一最強 |
| Kuncheva（2014）、Kuncheva & Whitaker（2003） | single best 與 oracle；多樣性度量 | 兩個參考點 | 移到 LLM 測試時；用 recovery\_blind 解釋多樣性為何預測不了增益 |
| Hu et al., RouterBench（2024，待核對） | 多 LLM 路由的基準 | 「路由器只和最佳單一模型打平」 | 決策是逐任務的聚合與否；給出可事先計算的空間上限 |
| AutoMix（NeurIPS 2024）、FrugalGPT（2023） | 逐題決定要不要升級到大模型 | 何時動用較貴的處理 | 他們成功的條件（逐題、選項差距大）在聚合決策上不成立 |
| Chen, Zaharia, Zou（NeurIPS 2024） | 多次呼叫加投票的表現與題目難度 | 更多呼叫不一定更好 | 基準是單一最強 path；path 性質不同；方向隨資料集翻轉 |
| Huang et al.（ICLR 2024）、Kamoi et al.（TACL 2024） | 自我修正不可靠 | 自我修正失效 | 指出失效在「很少改答案」（d），不是挑不出來 |
| Brown et al.（2024） | 至少一個答對的比例隨取樣數上升 | 上界與實際選出的差距 | 把選擇效率正規化成 recovery，可跨來源比較 |
| **本論文** | 分解 + 兩種基準 + 六種來源 + 三個事先登記的實驗（決策檢驗、交叉實驗、公平的多數決比較） | — | — |

**Novelty 的誠實評估**：多數單一發現都有前例。新意最高的三項是：(1) 候選效果，以及主網格差距是兩個效果相抵的拆解，我沒有找到直接的前例；(2) 兩條與三條 path 下的同語言對照，直接回答跨語言是否必要；(3) 決策空間的上限與事先登記的負面結果。整體是中等偏低的 novelty，貢獻在於把相近的結論放上同一把尺。審稿人最可能的評語是「紮實，但不意外」，所以寫作要把「1.5pp 縮到 0.3pp」和交叉實驗放在最前面。

## 六、所有相關論文：是否引用與引用論點

建議引用約 68 篇（必引 34、建議引 34），另有 17 篇可選、22 篇不需要。RQ2 與 RQ1-K 之後的變動集中在 H2 小節：新增四篇，Panickssery et al. 由建議引升為必引。投稿前每一筆的作者、年份、出處都要以原文核對；標「待核對」者資訊較不確定，不要直接寫進 bib。表中「理由」欄的結果編號對應第二節。

### A. 原論文已引用的 17 篇

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Hendrycks et al.（2021）MMLU | 必引 | 使用的資料集 | 資料集出處 |
| Talmor et al.（2019）CommonsenseQA | 必引 | 使用的資料集 | 資料集出處 |
| Wang et al.（ICLR 2023）Self-Consistency | 必引 | 測試時聚合的原點；結果 3 的表面反方 | 對取樣的多條推理路徑做多數決可提升準確率 |
| Huang et al.（ICLR 2024）LLMs Cannot Self-Correct Reasoning Yet | 必引 | §1 的矛盾；結果 7 | 無外部回饋時，自我修正常降低推理表現 |
| Huang et al.（2023）Not All Languages Are Created Equal（XLT） | 建議引 | 結果 6 的反方 | 以英文為樞紐的跨語言提示可提升推理 |
| Qin et al.（2023）Cross-lingual Prompting | 建議引 | 結果 6 的反方 | 跨語言對齊提示改善 zero-shot CoT |
| Shinn et al.（2023）Reflexion | 建議引 | §2.3 自我修正代表作 | 以語言化回饋與記憶反覆修正 agent 行為 |
| Sprague et al.（2025）To CoT or Not to CoT | 建議引 | 結果 15：多數決在 MathQA 上贏最多 | CoT 的好處集中在數學與符號推理 |
| Renze & Guven（2024）Concise Chain of Thought | 建議引 | 簡短 CoT 這條 path 的出處（v1 文件列為不需要，現在改列） | 精簡的 CoT 大幅縮短輸出，對多數題型的正確率影響很小（細節待核對） |
| Bang et al.（2023） | 可選 | 背景一句 | LLM 在不同語言間表現不一致 |
| Lai et al.（2023）ChatGPT Beyond English | 可選 | 背景一句 | 同上 |
| Xiong et al.（2025）Self-Rewarding Correction | 可選 | §2.3 對照 | 需要訓練的自我修正 |
| OpenAI（2024）GPT-4 Technical Report | 不需要 | 改引 GPT-4o mini 的官方來源 | — |
| Ahuja et al.（2025）sphinx | 不需要 | 訓練式多語言方法，新論文不涉訓練 | — |
| Indurthi et al.（2024） | 不需要 | 同上 | — |
| Ramesh et al.（2023） | 不需要 | 公平性動機，新論文不談 | — |
| Wang et al.（2024）MMLU-Pro | 不需要 | 原論文用它支持「CoT 傷害知識題」並不恰當 | — |

### B. 審稿人點名與漏引的資料集

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Amini et al.（2019）MathQA | 必引 | 使用卻漏引 | 資料集出處 |
| Lin et al.（2022）TruthfulQA | 必引 | 使用卻漏引 | 資料集出處 |
| Zhang et al.（Findings ACL 2024）AutoCAP | 必引 | R2 點名；結果 6 的反方 | 自動選語言並分配各語言推理路徑的權重，提升 zero-shot CoT |
| mGRPO: Unlocking LLM Reasoning through Multilingual Thinking（待核對） | 必引 | R2 點名 | 把多語言思考納入 RL 訓練；與我們的推論期分析區隔 |
| Du et al.（ICML 2024）Multiagent Debate | 必引 | R3 點名；結果 5、6 | 同一模型多實例互相辯論可提升事實性與推理 |
| Madaan et al.（NeurIPS 2023）Self-Refine | 必引 | R3 點名；§1 的矛盾 | 同一模型自我回饋並改寫，7 個任務平均提升約 20% |
| Shi et al.（ICLR 2023）Language Models Are Multilingual CoT Reasoners | 必引 | R3 點名；§2.4 | 提出 MGSM，顯示 LLM 具多語言 CoT 能力 |
| Ki et al.（ACL 2025）Multiple LLM Agents Debate for Equitable Cultural Alignment（待核對） | 建議引 | R3 點名 | 多個 agent 以不同視角辯論，改善文化情境下的決策 |

### C. 集成理論與組合方法

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Wood et al.（JMLR 2023）A Unified Theory of Diversity | 必引 | §2.1、§3.1 | 集成誤差 = 平均個體誤差 − diversity 效應；diversity 依賴標籤與個體能力 |
| Kuncheva & Whitaker（Machine Learning 2003） | 必引 | 結果 3 的理論背景 | 十種多樣性度量與集成準確率關聯都很弱 |
| Kuncheva（2014）Combining Pattern Classifiers | 必引 | §3.2、§3.3 的基準定義 | single best 與 oracle 是評估組合方法的兩個參考點 |
| Cruz, Sabourin, Cavalcanti（Information Fusion 2018） | 建議引 | 最接近「選單一或合併」 | 逐題選分類器只有在各分類器擅長不同區域時才贏過固定組合 |
| Caruana et al.（ICML 2004）Ensemble Selection | 建議引（做 RQ3 才引） | §5.6 | 從模型庫逐步選子集，勝過使用全部 |
| Ghorbani & Zou（ICML 2019）Data Shapley | 可選 | 只在 RQ3 用歸因時 | 以 Shapley value 分配個別貢獻 |
| Cruz et al.（Pattern Recognition 2015）META-DES | 不需要 | 引 2018 的綜述即可 | — |

### D. LLM 測試時聚合

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Li et al.（2025）Rethinking Mixture-of-Agents（Self-MoA，作者與出處待核對） | 必引 | 最接近的結論，novelty 風險 | 只重複取樣單一最強模型再聚合，勝過混合不同模型；品質比多樣性重要 |
| Song et al.（ICLR 2025）Mind the Gap（作者待核對） | 必引 | 最接近 recovery 的既有量 | 以 1−m 正規化生成與驗證能力的差距 |
| Chen et al.（2023）Universal Self-Consistency | 必引 | Judge 的方法出處 | 讓 LLM 自己從多個候選中選最一致者 |
| Jiang, Ren, Lin（ACL 2023）LLM-Blender | 必引 | §3.4 封閉性 | 區分選擇式與生成式融合 |
| Chen, Zaharia, Zou（NeurIPS 2024）Are More LLM Calls All You Need? | 必引 | 結果 5、11 | 更多呼叫對簡單題有幫助、對難題有害，整體可能非單調 |
| Brown et al.（2024）Large Language Monkeys | 必引 | §1 | 至少一個答對的比例隨樣本數上升，但選得出來的遠低於它 |
| Wang et al.（2024）Mixture-of-Agents | 必引 | 結果 10 與 RQ2；Self-MoA 的對照 | 模型看到其他模型的回答後能產生更好輸出；提案與聚合是不同的能力 |
| Li et al.（2024）More Agents Is All You Need | 建議引 | 結果 3 的表面反方 | 增加取樣數加多數決即可提升，難題與弱模型上增益較大（細節待核對） |

### E. 多 agent 辯論、自我修正、LLM 評審

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Liang et al.（EMNLP 2024）Encouraging Divergent Thinking | 必引 | 結果 5、7 | Degeneration-of-Thought：模型有信心後不再產生新想法；辯論優於自我反思 |
| Smit et al.（ICML 2024）Should We Be Going MAD? | 必引 | 結果 5，引用時措辭要改 | 對齊預算後，辯論沒有可靠勝過單一 agent |
| Kamoi et al.（TACL 2024） | 必引 | §2.3 主引用；結果 7 | 只有自身回饋時，沒有前作展示出可靠的自我修正 |
| Tyen et al.（Findings ACL 2024） | 必引 | 最接近我們拆法；RQ2 的方法範本 | 自我修正可拆成找錯與改錯；直接給錯誤位置可驗證哪一段是瓶頸 |
| Wang et al.（ACL 2024）LLMs Are Not Fair Evaluators | 必引 | §4 Judge 順序隨機化 | LLM 評審有位置偏誤，調換順序就能改變結果 |
| Zheng et al.（NeurIPS 2023）Judging LLM-as-a-Judge | 必引 | 結果 10 | 評審能力隨模型強度上升 |
| Wang et al.（ACL 2024）Rethinking the Bounds of LLM Reasoning（作者待核對） | 建議引 | 結果 1、5 | 單一 agent 配上好的 prompt 可追平多 agent 討論 |
| Stechly, Marquez, Kambhampati（2023） | 建議引 | 結果 7 的反方 | 自我批判讓表現變差，外部驗證器才有效，瓶頸在驗證 |
| Gou et al.（ICLR 2024）CRITIC | 建議引 | 結果 7 | 外部工具注入新資訊才讓自我修正有效 |
| Panickssery, Bowman, Feng（NeurIPS 2024） | 必引（由建議引升級） | 結果 12：我們的資料顯示模型裁決自己較差，必須說明兩者測的不是同一件事 | LLM 評估者偏好自己的輸出 |
| Zheng et al.（ICLR 2024）LLMs Are Not Robust Multiple Choice Selectors | 建議引 | §4 Judge 只輸出 A 或 B | LLM 偏好特定選項代號，和內容無關 |

### F. 多語言推理

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Gao et al.（2025）Could Thinking Multilingually Empower LLM Reasoning? | 必引 | 最接近的多語言論文；結果 3 | 多語言抬高 Acc@k，但 Vote@k 的優勢消失 |
| Lai, Zhang, Nissim（2025）Multidimensional Consistency | 必引 | 結果 4 | 沿順序、改寫、語言三個維度製造變異；一致性依維度而不同 |
| Wendler et al.（2024）Do Llamas Work in English? | 建議引 | 結果 6、8 的機制 | 模型內部以英文為潛在樞紐 |
| Elhady, Agirre, Artetxe（2026）Cross-lingual Self-Consistency（待核對） | 建議引 | 一句區隔 | 以跨語言一致性當無監督訓練訊號，是訓練期方法 |
| Cross-Lingual Consistency, CLC（2025，作者待核對） | 建議引 | 語言軸加多數決的實例 | 整合多語言推理路徑的多數決可提升數學推理 |
| Wu et al.（ICLR 2025）The Semantic Hub Hypothesis | 可選 | 結果 8 的機制補充 | 跨語言語意表徵集中於主導語言支配的共享空間 |
| Rajaee et al.（2025）Best-of-L（待核對） | 可選 | Limitations 一句 | 以跨語言 reward model 排序多語言候選 |

### G. 路由、預測與何時升級

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Chen, Zaharia, Zou（2023）FrugalGPT | 必引 | §2.5；結果 9 的對照 | 級聯：先問便宜模型，必要時才升級 |
| Madaan et al.（NeurIPS 2024）AutoMix | 必引 | 結果 9 的相反結果 | 即使自我驗證訊號有雜訊，逐題路由仍有效 |
| Hu et al.（2024）RouterBench（細節待核對） | 建議引 | 結果 9 的類似結果 | 完美路由上限很高，但實際路由器多半只和最佳單一模型打平 |
| Ong et al.（2024）RouteLLM（出處待核對） | 建議引 | 結果 9 的相反結果 | 用偏好資料訓練路由器，維持品質下明顯降低成本 |
| Maia Polo et al.（ICML 2024）tinyBenchmarks | 建議引 | 結果 9 的對比（v1 文件是「RQ1 成功才必引」） | 約 100 題可把準確率估到約 2% 內；估聚合決策時不成立 |
| Kossen et al.（ICML 2021）Active Testing | 建議引 | 結果 9：只標分歧題省 8 倍標註 | 主動挑要標註的測試題，用較少標註達到同樣精度 |
| Søgaard et al.（EACL 2021）We Need to Talk About Random Splits | 建議引 | 結果 9 為何做遷移 | 隨機切分系統性高估表現 |
| Baek et al.（NeurIPS 2022）Agreement-on-the-Line | 可選 | RQ1 不再是預測方法，相關性下降 | 模型一致率可免標註預測 OOD 準確率 |
| Miller et al.（ICML 2021）Accuracy on the Line | 可選 | 背景 | ID 與 OOD 準確率高度線性相關，某些偏移下會崩壞 |
| Garg et al.（ICLR 2022）ATC | 可選 | 背景 | 在來源資料學信心門檻，免標註估計目標準確率 |
| Geifman & El-Yaniv（NeurIPS 2017） | 可選 | 結果 2 | 棄答規則要沿整個覆蓋範圍評估 |
| Koh et al.（ICML 2021）WILDS | 可選 | 同 Søgaard | i.i.d. 與真實偏移間有大幅落差 |
| Dehghani et al.（2021）The Benchmark Lottery（出處待核對） | 可選 | 「結論由資料集決定」 | 方法的相對優劣會隨選用的 benchmark 改變 |
| Lakshminarayanan et al.（NIPS 2017）Deep Ensembles | 可選 | d 與難度的關係 | 成員分歧與預測誤差高度相關 |
| Jacobs & Wallach（FAccT 2021） | 不需要 | 論文不再使用效度的詞彙 | — |

### H. 實驗設計、統計與報告

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Cawley & Talbot（JMLR 2010） | 必引 | §3.5；結果 1 的切分 | 同一批資料選擇又評估會樂觀偏誤 |
| Zheng et al.（Findings EMNLP 2024）Is "A Helpful Assistant" the Best Role? | 必引（由建議引升級） | 最強單一 path 有 11 / 16 是 persona，必須面對這個反方 | 162 種角色下 persona 不比不加好，效果近乎隨機 |
| Kong et al.（NAACL 2024）Role-Play Prompting（待核對） | 建議引 | persona 軸的佐證 | 角色扮演提示可提升零樣本推理 |
| Cameron, Gelbach, Miller（2008） | 建議引（由可選升級） | 統計單位只有 16 個區塊 | 分組數少時，一般的標準誤會太小、容易誤判顯著 |
| van Miltenburg et al.（NAACL 2021）Preregistering NLP Research（待核對） | 建議引 | RQ1 的判定標準事先寫好 | 主張 NLP 研究在實驗前先寫下假設與判定標準 |
| Card et al.（EMNLP 2020）With Little Power | 建議引 | §4；強模型分歧題少 | 樣本太少偵測不到真實差異，應先估能分辨的最小差距 |
| Dodge et al.（EMNLP 2019）Show Your Work | 建議引 | 附錄 A | 要揭露實驗預算與次數 |
| Dror et al.（ACL 2018）Hitchhiker's Guide | 建議引 | 附錄 A | NLP 中顯著性檢定的選擇與報告 |
| Biderman et al.（2024）Lessons from the Trenches | 建議引 | 附錄 B | prompt 格式、解析規則、非決定性是不可重現的主因 |
| Sclar et al.（ICLR 2024）FormatSpread | 建議引 | §4 格式鎖死 | 無意義的格式差異可大幅改變表現 |
| Deng, Zhang, Gu（2023）Rephrase and Respond | 建議引 | §4 改寫軸 | 問題的措辭會影響模型能否答對 |
| Renze & Guven（2024）Effect of Sampling Temperature | 建議引，引用要小心 | §4 採樣軸 | T 在 0–1 對解題正確率無顯著影響；這是舊模型的結論，Google 對 Gemini 3 的建議相反 |
| Elangovan et al.（EACL 2021）Memorization vs. Generalization（待核對） | 可選 | 附錄 D：來源與目標用不相交的題目 | 訓練與測試重疊會讓評估虛高 |
| Pineau et al.（JMLR 2021） | 可選 | ACL 已有 Responsible NLP checklist | 可重現性檢查表 |
| Holtzman et al.（ICLR 2020） | 可選 | 溫度作為多樣性旋鈕 | 低溫或貪婪解碼容易產生重複的退化文字 |

### H2. RQ2 與 RQ1-K 之後新增

| 論文 | 引用 | 理由 | 要引用的論點 |
| --- | --- | --- | --- |
| Jiang et al., *Self-\[In\]Correct: LLMs Struggle with Discriminating Self-Generated Responses*（2024，作者與出處待核對） | 建議引 | 結果 12 最接近的前例 | LLM 分辨自己先前產生的多個答案孰優孰劣的能力，不比直接作答可靠 |
| Atil et al., *Non-Determinism of "Deterministic" LLM Settings*（2024，待核對） | 建議引 | 結果 14；附錄 A | 即使設定成確定性輸出，LLM 重跑的結果仍會不同，正確率可差到十幾個百分點 |
| Altman & Bland, *Absence of Evidence Is Not Evidence of Absence*（BMJ 1995） | 建議引 | 判定標準把「看不出來」和「沒有」分開的依據 | 檢定不顯著不能當成沒有效果的證據，要看信賴區間涵蓋的範圍 |
| Koo et al., *Benchmarking Cognitive Biases in LLMs as Evaluators*（Findings ACL 2024，待核對） | 可選 | 結果 12 的背景 | 系統性測量 LLM 評審的多種偏誤，包含偏好自己輸出的自我中心偏誤 |

已在前面各表、但用途因新結果而改變的論文：

- Chen et al., *Universal Self-Consistency*（2023）：除了是 Judge 的出處，也是 K 條 path Judge 版本的依據。它的結果是讓模型挑選與多數決大致打平。
- Li et al., *More Agents Is All You Need*（2024）：現在也是結果 1 的部分反方（12 條只比 3 條 persona 多約 0.3pp）。
- Lin et al., TruthfulQA（2022）：除了資料集出處，也用來解釋結果 15。
- Zheng et al.（persona）：最強單一 path 幾乎都是 persona，這個反方在 §5.1 一定要處理。
- Baek et al.、Garg et al.、Miller et al.：RQ1 不再是預測方法，維持可選。

### I. 不需要引用的

| 論文 | 理由 |
| --- | --- |
| Xu et al.（NeurIPS 2022）Learning to Break the Loop | 用來解釋 Gemini 2.5 的迴圈；該模型已換掉 |
| Efron & Morris（1977）Stein's Paradox | 用來討論平手規則，最後沒有採用 |
| Simmons, Nelson, Simonsohn（2011）False-Positive Psychology | 用來決定不要事後加模型，是工作原則，不是論文內容 |
| Rogers & Augenstein（Findings EMNLP 2020） | 關於審稿人的捷思，用來準備寫作，不是論文內容 |
| Cortes & Lawrence（2021） | 關於審稿雜訊，不是論文內容 |
| Higgins & Thompson（2002）、Higgins et al.（2003）、IntHout et al.（2016）、Rücker et al.（2008）、Thompson & Higgins（2002） | I² 異質性分析已撤銷 |
| Frost & Thompson（2000） | EIV 校正已撤銷 |
| Lakens（2017） | TOST 等價檢定已撤銷 |
| Lipton & Steinhardt（2018） | 寫作自我檢查用 |
| Ribeiro et al.（ACL 2020）CheckList、Schaeffer et al.（NeurIPS 2023） | 用來說服自己「診斷本身是貢獻」，論文內容不需要 |

### J. 模型與官方文件

四個模型（GPT-4o mini、Qwen3-8b、DeepSeek V4.1 Flash、Gemini 3.1 Flash-Lite）各要引用技術報告、模型卡或官方說明，並在 setup 寫明版本 ID 與呼叫日期。Gemini 的 temperature 若不是預設的 1.0，要用註腳引 Google 的 Gemini 3 開發指南說明原因。

### 本文件引用的網頁（2026-10-04 查閱）

- [ARR Dates and Venues](https://aclrollingreview.org/dates)：10 月 cycle 是 10/12；ACL 2027 對應 2027 年 1 月，日期未公布。
- [ARR Authors Guidelines](https://aclrollingreview.org/authors)：什麼情況算新投稿、resubmission 要交什麼。
- [ACL sustainable reviewing policy](https://aclrollingreview.org/sustainable-reviewing-2026)：service contributor、ORCID、抽籤。
- [Gemini 3 developer guide](https://ai.google.dev/gemini-api/docs/gemini-3)：temperature 建議留在 1.0；minimal 不保證 thinking 關閉。
- [DeepSeek V4.1 Flash 發布說明](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)：V4 Flash 下架、名稱導向 V4.1 Flash。
