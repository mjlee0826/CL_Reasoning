# RQ2 交叉實驗判定標準（事先登記）

- 狀態：**已確認**（之後不得修改）
- 建立：2026-10-04 17:22 CST（依使用者四點修改後更新；初稿 2026-10-04 17:17 CST）
- 確認：2026-10-04 17:25 CST，使用者確認（不加預測；API 由使用者自行執行）
- 規則：確認後本檔不得再修改。每份 RQ2 輸出都記錄本檔的 sha256。結果不如預期時，不修改設計、門檻或配對，先回報。

## 1. 目的

主網格裡，產生候選答案的模型和當 Judge 的模型是同一個。強模型（DeepSeek、Gemini）的 recovery 高於弱模型（GPT、Qwen），但分不出差距來自哪裡：是「強模型比較會挑」，還是「強模型的候選答案比較好挑」。本實驗固定候選答案，只換 Judge 模型，把兩者拆開。

## 2. 設計

- 模型：gpt4omini、qwen、deepseek4.1flash、gemini3.1flashlite。
  - 四個都當 generator，也都當 Judge，共 16 種組合。
  - 不納入舊的 DeepSeek V3.2（`deepseek`）與 Gemini 2.5（`gemini`）。
- 強弱分組（事先固定）：
  - 強 S = {deepseek4.1flash, gemini3.1flashlite}
  - 弱 W = {gpt4omini, qwen}
- 配對：EN+ZH（`L:en`,`L:zh`）、EN+S1（`L:en`,`S:T1.0:seed1`）、P1+P2（`P:expert`,`P:skeptic`）。
- 資料集：mmlu、mathqa、truthfulqa、commonsenseqa。
- 對角線（自己裁決自己）：
  - 4 種組合沿用主網格的 Judge 檔（`result/aggregations/{model}/{dataset}/judge__*.json`，prompt_version `choice-v1`），不重跑。
  - 例外：4.1 流程核對一致率 < 95% 的模型，其對角線用交叉流程重跑（見 4.1）。
  - 新跑的是其餘 12 種組合。不重跑任何生成端。
- 題目：每個 generator × 資料集 × 配對，只取兩條 path 都有解析出答案、且答案不同的題目。
  - 同一組候選答案，四個 Judge 看到的題目集合必須完全相同；不同就停下來回報。
- Judge 的輸入：
  - prompt 與主網格 Judge 完全相同（`JudgeAggregator`，`PromptJudgeChoiceFactory`，prompt_version `choice-v1`）。
  - Judge 看到原始英文題目，加上兩條 path 的完整原始輸出，標成 Answer 1 / Answer 2。
  - Judge 輸出 `{"choice":"N"}`，最終答案就是被選中那條 path 的答案。
- A、B 的順序（Answer 1 / Answer 2）：
  - 沿用主網格 Judge 檔逐題記錄的 `presentation_order`。
  - 2026-10-04 已核對：48 個主網格檔在每一題分歧題上都有紀錄，且與 `Aggregate.balancedOrders(seed 0)` 逐題相同。
  - 找不到紀錄，或紀錄和平衡順序不一致，就停下來回報，不重新隨機。
  - 這些順序是在全部分歧題上平衡的。在本實驗的題目子集裡，錨點在前的比例是 0.457–0.56；四個 Judge 相同，報告中列出。
- 解碼與 thinking 設定：與各模型在主網格當 Judge 時相同，即同一個模型類別。
  - 所有模型 T = 0，Judge 呼叫不傳 seed，max_tokens 8192。
  - Qwen：`enable_thinking` False。
  - DeepSeek：thinking disabled。
  - Gemini 3.1：thinking_level minimal（不保證完全不思考）。
  - GPT-4o mini 沒有 thinking 設定。
- 每次新呼叫都記錄供應商回傳的模型版本 ID 與呼叫時間（UTC）。

## 3. 每一格的量

一格 = generator g × Judge j × 資料集 d × 配對 p。題目子集是 both_answered（兩條 path 都有答案）。

- n_dis：分歧題數。
- n_R：剛好一條 path 答對的題數。
- recovery：用和 `aggregation_cells.csv` 的 recovery_H2 完全相同的算法。
  - 函式是 `Analysis.metrics.computeRow`：在該格的 both_answered 子集內做 split-half，seed 0、200 次。
  - 以下所有分析裡的「recovery」都指它。
  - 全樣本的 recovery 一併列出，只供參考。
- 挑對比例：在 n_R 題上，Judge 選中答對那條 path 的比例（亂猜是 50%）。無效選擇算挑錯。
- acc_final、off_menu、agg_no_answer：定義與 `computeRow` 相同，即與主網格相同。一致題上 Judge 不動作，答案就是兩條 path 共同的答案。
- Judge 輸出 tokens：列兩種，tokenizer 重算。
  - 每題平均：與 CSV 的 `tok_out_agg` 同義，一致題算 0。
  - 每次呼叫平均。
- 對角線核對（只適用於沿用主網格舊檔的模型；依 4.1 重跑對角線的模型不做此核對）：
  - 對角線格子用主網格 Judge 檔計算，所有欄位都必須和 `aggregation_cells.csv` 的 both_answered 列完全相同。
  - 把對角線也經過交叉實驗的計算路徑算一次，結果也必須相同。
  - 任一不同就停。
- 若有任何一格的 recovery 無定義（200 次切分都沒有可救的分歧題），停下來回報。

## 4. 執行前的兩個檢查（4.1 照下面的規則處理；4.2 不通過就停下來回報）

### 4.1 流程核對

- 四個模型各測一格：用交叉實驗的程式重跑該模型 mmlu × EN+S1「自己裁決自己」。
  - 題數：gpt4omini 178、qwen 194、deepseek4.1flash 91、gemini3.1flashlite 60。
  - 輸出寫在 `precheck/`，不覆蓋主網格。
- 各自和主網格存下來的 Judge 選擇逐題比對，分別報一致率。
  - 一致 = 選中的 path（`chosen_arm`）相同；兩次都是無效選擇也算一致。
- 某個模型一致率 < 95% 時，不停整個實驗，改成：
  - 把該模型的對角線（4 個資料集 × 3 個配對）用交叉流程重跑，分析用重跑的結果。
  - 4.1 已重跑的 mmlu × EN+S1 那一格直接沿用，不再呼叫。
  - 報告中註明，並同時列出舊檔算出的數字。

### 4.2 小規模試跑

- 每個 Judge 模型抽 100 題：從它要裁決的題目池均勻隨機抽，seed 0。
  - 題目池 = 另外 3 個 generator × 4 個資料集 × 3 個配對。
- 報告四個比例：
  1. 解析失敗：輸出中找不到 `{"choice"}`
  2. 編號超出範圍
  3. 被供應商擋下
  4. off-menu
- 另外報告每次呼叫的 token 數（API 計費的輸入 / 輸出，以及 tokenizer 重算的輸出）。
- 全量成本估算 = 各 Judge 每次呼叫的平均 API 成本 × 該 Judge 的全量呼叫數，四個 Judge 相加。
  - 全量呼叫數包含 4.1 決定要重跑的對角線。
  - 單價（美元 / 百萬 tokens，輸入 / 輸出）：

    | 模型 | 輸入 | 輸出 | 備註 |
    |---|---|---|---|
    | GPT-4o mini | 0.15 | 0.60 | |
    | Qwen3-8B | 0.18 | 0.70 | DashScope 國際站，非 thinking |
    | DeepSeek V4.1 Flash | 0.30 | 1.20 | 以尖峰價估，離峰為一半 |
    | Gemini 3.1 Flash-Lite | 0.25 | 1.50 | |

- 停止條件：
  - 任一 Judge 的上述任一比例 > 1%，也就是 100 題中超過 1 題；或
  - 估算總成本 > 20 美元。
- 試跑的紀錄留在正式快取中，全量執行時跳過，不重複計費。

## 5. 分析與統計

R(g, j, d) = 該格 recovery 對 3 個配對的平均。

### 5.1 4×4 表

- 列 = generator，欄 = Judge。
- 總表的格子 = recovery 對 4 個資料集 × 3 個配對的平均。
- 另外分資料集出 4 張（對配對平均），分配對出 3 張（對資料集平均）。

### 5.2 裁判效果（區塊 = generator × 資料集，16 個）

- 每個區塊：Δ_J(g, d) = mean_{j∈S, j≠g} R(g, j, d) − mean_{j∈W, j≠g} R(g, j, d)。
- 排除對角線後兩組的 Judge 數不一定相等。例如 g = deepseek4.1flash 時，強組只剩 gemini3.1flashlite。

### 5.3 候選效果（區塊 = Judge × 資料集，16 個）

- 每個區塊：Δ_C(j, d) = mean_{g∈S, g≠j} R(g, j, d) − mean_{g∈W, g≠j} R(g, j, d)。

### 5.4 統計量

- 跨 B 個區塊報四個量：
  - 平均
  - SE = sd / √B
  - 95% 區間 = 平均 ± t(0.975, B−1)·SE
  - 幾個區塊為正
- 主分析排除對角線格子。含對角線的版本（上面的 j≠g、g≠j 改為不排除）另外報告，不參與判定。

### 5.5 自我偏好

- 對每個資料集的 4×4 表（5.1 的分資料集表，含對角線）：
  - 預期值 E(g, j) = 列平均 + 欄平均 − 總平均。
  - 對角線殘差 = R(g, g, d) − E(g, g)。
- 共 16 個殘差（4 個資料集 × 4 個模型），報平均與 95% 區間（t，自由度 15）。
- 已知性質：列平均與欄平均包含對角線本身。若只有對角線高出 δ，殘差會是 0.75δ。照此算法，不另外校正。

## 6. 判定

### 6.1 效果的三種狀態（門檻為暫定值，確認時一併確認）

裁判效果、候選效果各自判斷。先對 16 個區塊的 Δ 取平均，再取絕對值：

- **存在**：跨 16 個區塊的平均 Δ 的絕對值 ≥ 0.05，而且 95% t 區間不含 0。
- **不存在**：整個 95% t 區間落在 [−0.05, +0.05] 之內。
- **無法判定**：其他情況。

方向（正 = 強組較高）另外報告。

### 6.2 結論與讀法

| 結果 | 條件 | 讀法 |
|---|---|---|
| (a) | 裁判效果存在、候選效果不存在 | recovery 是裁判的性質，「瓶頸在 aggregator」的診斷可信 |
| (b) | 候選效果存在、裁判效果不存在 | recovery 由候選答案決定，「該改進 aggregator」的診斷不成立 |
| (c) | 兩者都存在 | 報各自的大小 |
| (d) | 兩者都不存在 | 強弱模型的 recovery 差距不能歸因於裁判或候選（可能是交互作用或雜訊），兩種診斷都不支持 |
| (e) | 任一效果無法判定 | 只陳述已判定的那個效果；對無法判定的效果不下「有」或「沒有」的結論，並報它的區間 |

### 6.3 自我偏好

- 16 個對角線殘差的平均 ≥ +0.05，而且 95% 區間不含 0，才說有自我偏好。只看正方向。

### 6.4 判定只用

- 5.2、5.3 的主分析（排除對角線）。
- 5.5 的殘差。

其餘只報告。

## 7. 補充分析（只報告，不參與判定）

### 7.1 逐題配對（McNemar）

- 每一組強–弱 Judge 各做一次，共 4 組：
  - deepseek4.1flash–gpt4omini
  - deepseek4.1flash–qwen
  - gemini3.1flashlite–gpt4omini
  - gemini3.1flashlite–qwen
- 題目：兩個 Judge 都不是 generator 的那兩個 generator 的 n_R 題，跨資料集與配對合併。
  - 例：deepseek4.1flash–gpt4omini 用 qwen 與 gemini3.1flashlite 的候選。
- 計數：
  - b = 強 Judge 挑對、弱 Judge 挑錯
  - c = 弱 Judge 挑對、強 Judge 挑錯
  - 無效選擇算挑錯。
- 檢定：精確二項檢定（b 對 b + c，p = 0.5，雙尾）。

### 7.2 實際影響

- 對弱 generator g ∈ W，每個區塊（g × 資料集，8 個）：
  Δacc = mean_{j∈S} acc_final(g, j, d) − acc_final(g, g, d)
  - 兩項都先對 3 個配對平均。
  - 單位是百分點，在 both_answered 子集上算。
- 報平均、95% t 區間（自由度 7），以及各區塊的值。

### 7.3 分資料集、分配對

- 裁判效果與候選效果分資料集報告：每個資料集 4 個區塊，報平均與各區塊的值。
- 分配對時，把 R 換成單一配對的 recovery 重算 5.2、5.3。
- 預期 TruthfulQA 的差距最大、MathQA 最小。這個預期不影響任何算法。

### 7.4 同組但非自己的組合

- 四格（箭頭左邊是 Judge，右邊是 generator）：
  - deepseek4.1flash → gemini3.1flashlite
  - gemini3.1flashlite → deepseek4.1flash
  - gpt4omini → qwen
  - qwen → gpt4omini
- 殘差算法與 5.5 相同：殘差 = R(g, j, d) − E(g, j)，E 用同一張分資料集的 4×4 表（含對角線）。
- 每個資料集 4 個、共 16 個殘差，報平均與 95% 區間（t，自由度 15）。
- 用途：分辨對角線的正殘差是自我偏好，還是強弱相同的組合本來就比較合。

## 8. 輸出（`result/analysis/rq2/`）

- `judge_outputs/`：
  - `{judge}/{generator}/{dataset}/judge__{arm}__{arm}.json`：新呼叫的原始紀錄，同時是可續跑的快取。
  - `items.csv.gz`：逐題攤平表，含對角線。欄位：
    - generator、judge、dataset、pair、item_id
    - A/B 順序、choice、chosen_arm、最終答案、是否正確
    - Judge 原始輸出、tokens
- `precheck/`：4.1 的四個重跑檔與一致率。試跑結果另存 `pilot.json`。
- `cross_cells.csv`：第 3 節的量，每格一列，含對角線，共 192 列。
  - 對角線列標明來源（主網格舊檔 / 交叉流程重跑）。
  - 依 4.1 重跑的模型，另外列出舊檔算出的對角線列，標為不用於分析。
- `report.md`，順序固定：
  1. 本檔的 sha256 與確認時間
  2. 兩個執行前檢查的結果
  3. 各 Judge 的解析失敗與 off-menu
  4. 4×4 總表
  5. 裁判效果、候選效果、自我偏好，以及對照第 6 節的結論
  6. 分資料集、分配對
  7. 補充分析
- 熱圖：總表 1 張、分資料集 4 張。
- 模型紀錄：每個模型的版本 ID、呼叫日期、解碼與 thinking 設定。
  - 對角線沿用的主網格 Judge 檔沒有逐次記錄版本與日期，只能報設定與檔案日期。

## 9. 執行規則

- 呼叫可續跑、有快取（已寫入的題目跳過），失敗自動重試，不重複計費。
- 本檔確認之前，不做任何 API 呼叫。4.1 完成、4.2 通過之前，不跑全量。
- DeepSeek 盡量在離峰時段一次跑完，因為官方別名背後的模型可能更換；以回傳的版本 ID 核對。
- 結果不如預期時，不自行改設計、改門檻或換配對，先回報。
- 只用 both_answered 的題目。
- 環境：conda 環境 clreasoning，Python / pandas。
