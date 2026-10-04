# RQ1-KJ 判定標準（事先登記）：K 條 path 的 Judge 版本 vs 單一最強 path

- 狀態：**已確認**（之後不得修改）
- 建立：2026-10-04 23:10 CST
- 確認：2026-10-04 23:11 CST，使用者確認（含門檻 0.5pp；§3.3 分組隨機順序；§6.4 試跑的無效比例為三種合計）
- 規則：
  - 確認後本檔不得再修改。每份 RQ1-KJ 輸出都記錄本檔的 sha256。
  - 本檔確認之前，呼叫 API 的程式與分析程式一律拒跑。
  - 不重跑任何生成端；只呼叫 Judge。
  - 開跑前的四項檢查（§6）依序做，每一步都回報；四項都通過後停下來，使用者確認後才跑全量。
  - API 由使用者執行；呼叫可續跑、有快取、失敗自動重試，已寫入的題目不重複計費。
  - 做完就停。結果不如預期時，不改設計、門檻、菜單、題目子集或 prompt，先回報。

## 1. 目的

RQ1-K 用多數決比較「整份菜單一起聚合」與「菜單內最強的單一 path」。M12 的判定是「無法判定」（+0.30pp，95% 區間 −0.63 到 +1.23），RQ1-K 的判定標準因此把 Judge 版本列為必要的後續實驗。

RQ1-KJ 把聚合方式換成 Judge：模型看完菜單內全部 K 個候選後挑一個。它回答三件事：

1. Judge 版本的聚合有沒有贏過最強的單一 path。
2. Judge 版本和多數決差多少。
3. 同樣三條 path、同樣用 Judge 時，同語言重抽和換語言誰比較好。

## 2. 資料、菜單與裁判

### 2.1 模型、資料集、候選

- 模型：gpt4omini、qwen、deepseek4.1flash、gemini3.1flashlite。
  - 不納入舊的 DeepSeek V3.2（`deepseek`）與 Gemini 2.5（`gemini`）。
- 資料集：mmlu、mathqa、truthfulqa、commonsenseqa。
- 區塊 = 模型 × 資料集，共 16 個。
- 候選來源：`result/arms` 中各 path 的逐題記錄：
  - 原始輸出 `raw_text`
  - 解析答案 `parsed_answer`
  - 是否有答案 `parse_ok`
  - 輸出 tokens `tokens_out`
- 正確 = 答案與 gold 相同，用該資料集的 `compareTwoAnswer`；這四個資料集都是字串相等。

### 2.2 菜單

沿用 RQ1-K 的定義，不增不減。括號內為 path 短代號（與 RQ1-K 相同）。表中列出的順序是 §3.3 候選順序的基準順序；它和 §4.2 的平手優先順序無關。

| 菜單 | K | path（基準順序） | 角色 |
|---|---|---|---|
| **M12** | 12 | L:en (EN)、L:zh (ZH)、L:ja (JA)、L:ru (RU)、L:es (ES)、S:T1.0:seed1 (S1)、S:T1.0:seed2 (S2)、P:expert (P1)、P:skeptic (P2)、W:rewrite1 (W1)、W:rewrite2 (W2)、R:short_cot (R) | **主要** |
| M3L | 3 | L:en、L:zh、L:ja | 同來源：語言 |
| M3S | 3 | L:en、S:T1.0:seed1、S:T1.0:seed2 | 同來源：採樣 |
| M3P | 3 | L:en、P:expert、P:skeptic | 同來源：persona |

### 2.3 裁判

- 自己裁決自己：Judge 模型 = 產生候選的模型，和主網格一致。
- 解碼與 thinking 設定和該模型在主網格當 Judge 時相同：每次呼叫 T = 0、max_tokens 8192、不傳 seed。

| 模型 | 呼叫的模型名 | thinking 設定 |
|---|---|---|
| gpt4omini | gpt-4o-mini-2024-07-18 | 無 thinking 參數 |
| qwen | qwen3-8b（DashScope 國際站） | enable_thinking = False |
| deepseek4.1flash | deepseek-flash（官方 API，= DeepSeek V4.1 Flash） | thinking disabled |
| gemini3.1flashlite | gemini-3.1-flash-lite | thinking_level minimal（reasoning_effort "minimal"；Google 註明不保證完全不思考） |

### 2.4 題目

- 子集_in(菜單)：菜單內所有 path 都有答案（`parse_ok`）的題目。和 RQ1-K 算 Excess_in 的子集相同。
- 不一致題：子集內，菜單各 path 的解析答案不完全相同的題目（用 `compareTwoAnswer` 比較）。
- 只對不一致題呼叫 Judge。答案完全一致的題目不呼叫，最終答案就是那個共同答案。
- 每個區塊、每個菜單都報兩個比例：
  - 保留比例 = 子集題數 ÷ 資料集題數
  - 不一致題比例 = 不一致題數 ÷ 子集題數
- 事先離線算好的 Judge 呼叫數（16 個區塊合計）：

| 菜單 | 呼叫數 | 占子集 |
|---|---|---|
| M12 | 9,691 | 35.9% |
| M3L | 6,537 | 24.1% |
| M3S | 3,173 | 11.7% |
| M3P | 3,231 | 11.9% |
| 合計 | 22,632 | — |

## 3. Judge 的 prompt、候選順序與計分

### 3.1 prompt（choice-k-v1）

- prompt_version 命名為 `choice-k-v1`。
- 文字由主網格 choice-v1 的同一個函式產生（`PromptJudgeChoiceFactory`），只把候選數從 2 擴成 K。
  - 該函式本來就以 K 為參數：K 出現在 "There are {K} answers"、"these {K} answers"、"an integer from 1 to {K}" 三處。
  - 其餘文字、每個候選呈現的內容、輸出格式都不變。
  - K = 2 時，渲染結果與 choice-v1 逐字相同。
- 全文見附錄 A。
- 呈現的內容：
  - 題目：資料集的英文原題。所有 path 都一樣，包括 L:zh 等翻譯 path 與 W 改寫 path，和主網格相同。
  - 候選：每條 path 的完整原始輸出（`raw_text`），包含推理與最後的答案 JSON，各自包在 ``` 中，標為 "Answer 1" 到 "Answer K"。
  - 候選保留原本的語言：不翻譯、不截斷、不合併答案相同的候選。
  - 不告訴 Judge 每個候選來自哪條 path，也不提供票數。
- 呼叫形式：整段 prompt 是單一則 user 訊息，沒有 system 訊息。
- 每題只呼叫一次（試跑的第二次除外，見 §6.4）。

### 3.2 解析與最終答案

- 取輸出中**最後一個** `{"choice":"N"}`（主網格的 `JudgeAggregator.CHOICE_PATTERN`）。
  - 數字沒加引號或寫成 "Answer N" 也能解析。
  - 兩位數（10–12）也能解析。
- 最終答案 = 被選中的候選的解析答案。

### 3.3 候選順序

不直接沿用 `Aggregate.balancedOrders`。它只把同一個基準順序循環旋轉 K 種，相鄰關係固定。依使用者確認（2026-10-04），改用「每 K 題一組，每組用不同的隨機基準順序」。

**作法**：每個區塊 × 菜單各自計算。

1. 把該格的不一致題 item_id 由小到大排好，共 n 題。
2. 建立 `rng = numpy.random.default_rng(0)`，用 `rng.permutation(n)` 打亂這 n 題（和 balancedOrders 相同）。
3. 打亂後的第 i 題（i 從 0 起算）屬於第 g = i // K 組。
4. 依 g = 0, 1, 2, … 的順序，每一組用同一個 rng 接著抽一次 `rng.permutation(K)`，照它重排 §2.2 的基準順序，得到該組的基準順序 B_g。
   - 最後一組不足 K 題時也一樣抽一次。
5. 第 i 題的候選順序 = B_g 循環旋轉 r = i mod K 位，即 B_g[r:] + B_g[:r]。

**效果**：

- 一個完整的組裡，K 種旋轉各用一次，所以每條 path 在每個位置恰好出現一次。
- 最後一組不足 K 題時，每條 path 在每個位置最多出現一次。
- 因此整體而言，每條 path 出現在每個位置的次數相差不超過 1。
- 相鄰關係隨組改變。

**已知限制**（事先寫明，不因結果更改）：

- 同一組的 K 題用的都是 B_g 的循環旋轉，所以組內各題的循環相鄰關係（哪條 path 接在哪條後面）相同。
- 不同組的相鄰關係不同。
- §7.6 報告位置與 path 的分布。

**其他**：

- 每題的順序記在 `presentation_order`。
- §6.3 的流程核對不受影響，它用主網格記錄的順序。
- §6.4 的試跑用本節在全量不一致題上算出的順序。
- §2.4 的呼叫數不變。

### 3.4 計分

以下三種情況都沒有最終答案，記為 agg_no_answer、算錯：

- Judge 輸出中找不到 `{"choice":"N"}`（解析失敗）
- 輸出的編號不在 1 到 K 之內（超出範圍）
- 供應商擋下呼叫，輸出為空（refused）

這與主網格的計分方式相同：主網格的 `JudgeAggregator` 在這三種情況都給空的最終答案；`Analysis/metrics.py` 的 both_answered 子集把它算錯，並以 agg_no_answer 報比例。

API 呼叫失敗（重試用完）的題目不寫入，續跑時補上，不計分。

## 4. 定義

### 4.1 切分

- 每個資料集用 `Analysis.splitHalf.makeSplits(n_d, reps=200, seed=0)`，和 RQ1、RQ1-K 相同。
  - n_d = 資料集題數：truthfulqa 817，其餘 2000。
  - 第 r 次切分得到 H1_r 與 H2_r，同一資料集的所有模型與菜單共用。
- 每次切分再和題目子集取交集：選擇只用 H1_r ∩ 子集，評估只用 H2_r ∩ 子集。

### 4.2 每次切分、每個區塊、每個菜單的量（正確率）

- A_J：Judge 版本的最終答案在 H2_r ∩ 子集_in 上的正確率。一致題用共同答案；agg_no_answer 算錯。
- A_V：多數決在 H2_r ∩ 子集_in 上的正確率。定義與平手規則和 RQ1-K 完全相同：
  - 得票最多的答案勝出。
  - 平手時，依優先順序找第一條答案落在平手答案中的 path，取它的答案。
  - 程式為 `Analysis.menuVote.menuFinals`，平手方式 `priority`。
- S_in：在 H1_r ∩ 子集_in 上，選出菜單內答對最多的 path（平手依同一個優先順序，即 `Analysis.probe.strongest`）。報它在 H2_r ∩ 子集_in 上的正確率。
- 平手優先順序（A_V 與 S_in 共用；RQ1-K 實際使用的順序，程式為 `Analysis.probe.pathKey`）：
  - 排序依據：L:en（EN）第一；其餘依 §2.2 括號內的 path **短代號**，做字串的字母序排序。不是依 arm_id（L:zh、P:expert……）排序。
  - M12 的完整順序：**EN、ES、JA、P1、P2、R、RU、S1、S2、W1、W2、ZH**。
  - 三條的菜單取這個順序的子序列：M3L 為 EN、JA、ZH；M3S 為 EN、S1、S2；M3P 為 EN、P1、P2。
  - 對照：若依 arm_id 排序，會是 EN、ES、JA、RU、ZH、P1、P2、R、S1、S2、W1、W2。本研究不用這個順序。
- EN：L:en 在 H2_r ∩ 子集_in 上的正確率，當參考。

### 4.3 每個區塊的量（先對 200 次切分取平均，單位是百分點）

- Excess_J = A_J − S_in
- Excess_V = A_V − S_in
- Diff_JV = A_J − A_V

不使用「在 H2 上從所有 path 挑最大值」當基準。

## 5. 統計與判定（門檻 0.5pp，確認時一併確認）

### 5.1 統計量

統計單位是區塊。跨 16 個區塊報：

- 平均
- SE = sd / √16
- 95% t 區間（自由度 15）
- 幾個區塊為正

### 5.2 四種狀態

先對區塊取平均，再和門檻比：

| 狀態 | 條件 |
|---|---|
| 正向成立 | 平均 ≥ +0.5pp，且 95% 區間不含 0 |
| 反向成立 | 平均 ≤ −0.5pp，且 95% 區間不含 0 |
| 兩者相當 | 整個 95% 區間落在 [−0.5, +0.5] 之內 |
| 無法判定 | 其他情況 |

參與判定的只有以下三項。

### 5.3 判定一（主要）：M12 的 Excess_J

- 正向 = Judge 版本的聚合較好；反向 = 最強單一 path 較好。
- 事先寫下的讀法：
  - **正向成立**：「聚合對最強單一 path 分不出正負」只適用於多數決。論文要寫 Judge 版本贏過最強單一 path，並報幅度與成本（§7.8）。
  - **反向成立**：Judge 版本輸給最強單一 path。
  - **兩者相當**：Judge 版本與最強單一 path 相當（差距在 ±0.5pp 內）。
  - **無法判定**：兩種聚合方式下都分不出正負。報兩個區間（M12 的 Excess_J 與 Excess_V），不下「沒有效果」的結論。

### 5.4 判定二：M12 的 Diff_JV

- 正向 = Judge 較好；反向 = 多數決較好。
- 事先寫下的讀法：
  - **正向成立**：Judge 版本比多數決好。報幅度與 Judge 多花的成本（§7.8）。
  - **反向成立**：多數決比 Judge 版本好。
  - **兩者相當**：Judge 版本與多數決相當（差距在 ±0.5pp 內），Judge 多花的呼叫沒有換到正確率。
  - **無法判定**：只報區間，不下誰比較好的結論。

### 5.5 判定三：M3S 的 A_J − M3L 的 A_J

- 題目：L:en、L:zh、L:ja、S:T1.0:seed1、S:T1.0:seed2 五條都有答案的題目。兩個菜單在同一批題目上比。
- 每個區塊：在 H2_r ∩ 五條子集上，M3S 的 A_J − M3L 的 A_J，對 200 次切分平均，單位是百分點。
- 正向 = 同語言重抽較好；反向 = 換語言較好。
- 事先寫下的讀法：
  - **正向成立**：三條 path 的 Judge 下，同語言重抽比換語言好。
  - **反向成立**：「增益不需要換語言」在三條 path 的 Judge 下不成立，論文的這個主張要加上這個限制。
  - **兩者相當**或**無法判定**：維持「換語言沒有比較好」的寫法，並報區間。

### 5.6 不參與判定的部分

- 其他菜單的數字、分資料集與分模型的數字都只報告，不參與判定。
- 結果不如預期時不加菜單、不改門檻。

## 6. 開跑前的檢查（依序做，每一步的結果都回報）

### 6.1 離線重現

- 用這次的程式，算四份菜單 × 16 個區塊的 A_V、S_in、Excess_V。
- 和 `result/analysis/rq1k/rq1k_blocks.csv` 的 `A_in`、`S_in`、`excess_in` 比對，誤差都要在 1e-9 以內。
- 不符就停。

### 6.2 模板核對

主網格的 Judge 紀錄沒有存 prompt 全文，也沒有存 prompt 的 hash；只存了每題的 `tokens_in`（用該模型的 tokenizer 重算 prompt 的 tokens）。所以逐字比對改成以下三項：

1. **重算 tokens 比對**：
   - 範圍：mmlu × EN+S1，四個模型各自兩條 path 都有答案的不一致題（gpt4omini 178、qwen 194、deepseek4.1flash 91、gemini3.1flashlite 60）。
   - 用 choice-k-v1 的程式在 K = 2 時渲染 prompt，候選順序用主網格檔案記錄的 `presentation_order`。
   - 用該模型的 `countTokens` 重算，每題都要和主網格存下的 `tokens_in` 相等。
   - Gemini 的 `countTokens` 走免費的 count_tokens API，不產生文字。
2. **git 紀錄**：`PromptJudgeChoiceFactory.py` 與 `JudgeAggregator.py` 最後一次變動是 b290e88（2026-10-04 02:58 CST），早於主網格 mmlu × EN+S1 Judge 檔的寫入時間。
   - GPT 與 Qwen 是 13:34。
   - DeepSeek 與 Gemini 是 16:18，在另一台機器執行，它們的程式版本只能靠第 1 項核對。
3. **措辭差異**：choice-k-v1 與 choice-v1 用同一個函式，沒有措辭差異。

若第 1 項有任何一題不相等，停下來，逐題列出差異。

### 6.3 流程核對

- 範圍：四個模型各用這次的程式，重跑「自己裁決自己」的 mmlu × EN+S1。
  - K = 2 的菜單是 [L:en, S:T1.0:seed1]，prompt 文字即 choice-v1。
  - 題目同 §6.2 第 1 項，與 RQ2 核對時相同。
  - 候選順序用主網格檔案 `result/aggregations/{模型}/mmlu/judge__L_en__S_T1.0_seed1.json` 逐題記錄的順序，不重新平衡；找不到記錄的順序就停。
  - 結果另存，不進正式分析。
- 和主網格逐題比對選中的 path，報一致率，並和 RQ2 核對時同一格的一致率並列：

| 模型 | RQ2 核對時 |
|---|---|
| qwen | 98.5%（191/194） |
| gemini3.1flashlite | 91.7%（55/60） |
| gpt4omini | 89.3%（159/178） |
| deepseek4.1flash | 80.2%（73/91） |

- 任何一個模型比 RQ2 核對時低 10 個百分點以上（含 10），停下來回報。
- 同時回報 API 回傳的模型版本 ID 是否和記錄相同。主網格的 Judge 檔沒有記錄回傳的版本，所以和以下兩者比對：
  - 該模型 arm 檔的 `model_version_string`
  - RQ2 Judge 呼叫的紀錄

  目前的紀錄是：`gpt-4o-mini-2024-07-18`、`qwen3-8b`、`deepseek-flash@DeepSeek-V4.1-Flash`、`gemini-3.1-flash-lite`。DeepSeek 的 API 只回傳別名 `deepseek-flash`，「V4.1-Flash」是我們加上的標籤，所以 DeepSeek 只能核對別名。

### 6.4 試跑

**抽樣與執行**

- 每個模型在 M12 上跑 100 題。
- 四個資料集各 25 題：從該區塊 M12 的不一致題（item_id 由小到大）中，用 `numpy.random.default_rng(0).choice(題數, 25, replace=False)` 抽出。
- 順序用 §3.3 在全量不一致題上算出的順序，和全量相同。
- 每題跑兩次：
  - 第一次寫進正式的 `judge_outputs/`，全量時沿用。正式分析一律用第一次的結果。
  - 第二次另存 `pilot_rep2/`，只用來算一致率。

**回報**

1. 每個模型的解析失敗率、編號超出範圍的比例、被擋下的比例（兩次分開列）。
2. Judge 的 input 與 output tokens 的平均與最大值，分兩種：用該模型 tokenizer 重算的值，以及 API 計費的值。
3. context 上限：
   - 試跑題目用實際值。
   - 全量 22,632 題用離線估計：「該題各候選 `tokens_out` 的總和 + L:en 生成 prompt 的 `tokens_in`」，乘上試跑的校正係數（該模型試跑的 API 計費 input ÷ 同一批題目的離線估計）。
   - 判斷方式：input tokens + max_tokens（8192）超過該模型官方文件的 context 上限，就算超過。各模型採用的上限值記在 `pilot.json`。
4. 兩次的一致率：
   - 選到同一個候選的比例（兩次都無效時算相同）。
   - 最終答案相同的比例（兩次都沒有答案時算相同）。
   - 另列兩次都有效的題數。
5. 全量的呼叫次數與成本，四份菜單分開列：
   - 呼叫次數用 §2.4 的確切值。
   - M12 每次呼叫的 input 與 output tokens，用試跑第一次的 API 計費平均。
   - M3 三份菜單的 input 用離線估計乘校正係數；output 每次呼叫用 M12 試跑的平均（偏保守）。
   - 價格（美元 / 百萬 tokens，輸入 / 輸出）與 RQ2 相同：

| 模型 | 輸入 | 輸出 | 備註 |
|---|---|---|---|
| gpt4omini | 0.15 | 0.60 | |
| qwen | 0.18 | 0.70 | |
| deepseek4.1flash | 0.30 | 1.20 | 以尖峰價估 |
| gemini3.1flashlite | 0.25 | 1.50 | |

**停止條件**（任一成立就停下來回報）

- 任一模型第一次的無效比例超過 1%。無效 = 解析失敗 + 編號超出範圍 + 被擋下，即 agg_no_answer；也就是 100 題中有 2 題以上。
- 試跑或全量的估計中，有任何題目超過 context 上限。
- 全量（四份菜單合計，含試跑已花的）推估成本超過 30 美元。

### 6.5 停下來

四項都通過後停下來，使用者確認後才跑全量。

## 7. 另外要報告的量（不參與判定）

1. **總表**：
   - 列 = 菜單。
   - 欄 = A_J、A_V、S_in、EN、Excess_J、Excess_V、Diff_JV。
   - 三個差值各附 95% 區間與為正的區塊數。
2. **逐區塊與彙總**：
   - 每個菜單的逐區塊數字，含 §2.4 的保留比例與不一致題比例。
   - 分資料集、分模型的平均。
3. **TruthfulQA**：事先指定的次要分析。只有 4 個區塊，不套四種狀態。
   - 列出四個模型在 M12 的 Diff_JV 與 Excess_J。
   - 讀法：
     - 4 個 Diff_JV 全為正，才寫「Judge 在 TruthfulQA 上比多數決好」。
     - 4 個 Excess_J 全不為負，才寫「Judge 補回了多數決在 TruthfulQA 的損失」。
     - 其餘情況只列數字。
   - 在 M12 子集中，P:skeptic 的答案和多數決答案不同的題目上，報：
     - Judge 最終答案的歸屬比例：等於 skeptic 答案、等於多數決答案、其他（agg_no_answer 歸入其他）。
     - 這些題目上 skeptic、多數決、Judge 各自的正確率。
     - 用全部子集題目，不切分。
4. **Judge 與多數決的逐題對照**：每個區塊、每個菜單，只算不一致題，用全部子集題目，不切分。
   - 兩者答案相同的比例。
   - 四格的題數：都對、都錯、多數決對而 Judge 錯、多數決錯而 Judge 對。
   - 依正確答案的位置分三類，各類報題數，以及 Judge 選到正確答案的比例：
     - 最多票：正確答案的票數等於最高票數，含與其他答案平手；平手的題數另外列出。
     - 少數：正確答案在候選中，但票數低於最高票。
     - 不在候選中：沒有任何候選答對。
5. **K 條 path 的分解**：用 S_in；每次切分在 H2_r ∩ 子集_in 上計算。
   - d：菜單內 path 答案不完全一致的題目比例。
   - c：這些題目中至少一條 path 答對的比例。
   - m：這些題目上各 path 的平均正確率（對 path 與題目平均）。
   - headroom = d·(c − m)
   - recovery_J =（Judge 在這些題目的正確率 − m）÷（c − m）。agg_no_answer 算錯。
   - recovery_blind =（S_in 在這些題目的正確率 − m）÷（c − m），與 RQ1-K 相同。
   - 核對：
     - 每次切分核對 Excess_J = headroom ×（recovery_J − recovery_blind），以正確率為單位，誤差 ≤ 1e-9。不符就停下來回報。
     - 沒有不一致題（d = 0）或 c = 0 時，recovery 無定義。這時核對改為 Excess_J = 0，並報告這種切分的次數。
   - 各量報對切分的平均，無定義的切分不計入。
6. **位置與 path**：每個區塊 × 菜單，以及每個菜單的合計，報：
   - 被選中的候選落在各位置（1 到 K）的比例。
   - 被選中的候選來自各 path 的比例。
   - 每條 path 出現在各位置的次數，用來確認順序有打散。
7. **和兩條 path 的 Judge 對照**：用主網格現有結果，在同一批題目上比正確率。
   - 對照組：
     - M3L 對 EN+ZH（`judge__L_en__L_zh.json`）
     - M3S 對 EN+S1（`judge__L_en__S_T1.0_seed1.json`）
     - M3P 對 P1+P2（`judge__P_expert__P_skeptic.json`）
   - 兩條 path 的 Judge 結果取自 `result/aggregations/{模型}/{資料集}/`，不用 RQ2 對角線的重跑。
   - 題目：該 M3 菜單的子集_in。
   - 兩條 path Judge 的最終答案：一致題是共同答案，不一致題是 Judge 的選擇；沒有答案算錯。
   - 每次切分在 H2_r 上算 A_J(M3) − A(兩條 Judge)，對 200 次切分平均，單位是百分點。
   - 跨 16 個區塊報平均、95% 區間、為正的區塊數。
8. **成本**：每個區塊 × 菜單，在 H2_r ∩ 子集_in 上對切分平均。
   - 每題的呼叫次數 = K + 不一致題比例（K 次生成加上不一致題的 1 次 Judge）。
   - Judge 的 input 與 output tokens 分開報。主要用 tokenizer 重算的值，與主網格相同；另列 API 計費的值。
   - 整個流程的輸出 tokens 對 S_in 那條 path 的輸出 tokens：
     - 分子 = 菜單內 K 條 path 的 `tokens_out` 總和，加上 Judge 的 output tokens（一致題為 0）。
     - 分母 = S_in 的 `tokens_out`。
9. **決策空間**：只報告，不是 go / no-go。
   - E_b = M12 區塊 b 的 Excess_J。
   - 永遠聚合的 regret = 16 個區塊的 max(0, −E_b) 的平均。
   - 永遠用最強單一 path 的 regret = 16 個區塊的 max(0, E_b) 的平均。
   - 空間 = 兩者取較小者。
10. **全量的失敗比例**：每個模型（再分菜單）報解析失敗率、編號超出範圍的比例、被擋下的比例、agg_no_answer 比例。

## 8. 輸出（`result/analysis/rq1kj/`）

- `rq1kj_criteria.md`：本檔。
- `judge_outputs/`：逐題的 Judge 原始輸出，每個模型 × 資料集 × 菜單一個檔。每題記錄：
  - 模型、資料集、菜單、題目 id
  - `presentation_order`、選中的編號與 path
  - 最終答案、是否正確
  - input / output tokens（重算值與 API 計費值）
  - `prompt_version`、prompt 的 sha256、Judge 原始輸出
  - API 回傳的模型版本、呼叫時間（UTC）

  另有一份攤平的 `items.csv.gz`。
- `precheck/`、`template_check.json`、`precheck.json`、`pilot_rep2/`、`pilot.json`：§6 的檢查結果。
- `rq1kj_blocks.csv`：每個區塊 × 菜單一列，含 §4 與 §7 的量。
- `report.md`，順序固定：
  1. 判定標準檔的雜湊值與確認時間
  2. §6 四項檢查的結果
  3. 解析失敗與重跑一致率
  4. 總表
  5. 三項判定對照判定標準的結論
  6. TruthfulQA 的次要分析
  7. 其餘只報告的量
- 兩張圖：
  - 各菜單的 Excess_J 與 Excess_V：平均與 95% 區間。
  - M12 逐區塊的 Excess_J：標出 0 與 ±0.5。
- `models.csv`：每個模型的版本 ID、呼叫日期、解碼與 thinking 設定。

## 附錄 A：choice-k-v1 的 prompt 全文

{英文原題}、{候選 i 的完整原始輸出}、{K} 為代入處，其餘逐字不變：

````text
For the following question
```
{英文原題}
```
There are {K} answers as follows
Answer 1
```
{候選 1 的完整原始輸出}
```
Answer 2
```
{候選 2 的完整原始輸出}
```
…（依序到 Answer K）
Based on the question, select the most correct one of these {K} answers. You must think step by step about which parts of the reasoning in each answer are incorrect, and output your reasoning process.
Please strictly follow the format below for output
Reasoning process
{your reasoning process - Note: **Do not restate the original question text or add content not required by the question**}

Final Choice
{"choice":"answer number"}
(You shouldn't output "answer number" directly. Replace it with the number of the answer you select, an integer from 1 to {K}. Do not output an option letter of the question or an answer of your own. The entire final choice block must only be that one line of JSON, with no extra text or explanation before or after.)
````
