# cl_reasoning

LLM test-time diversity 的分解實驗：用不同的多樣性來源（語言 / 採樣 / persona / 改寫 / 推理強度 / 自我修正）產生 K=2 的候選，
再用聚合器（Judge / Debate / Blind）合併，量測 `d`、`c`、`m`、`recovery`、`recovery_blind`。
研究脈絡與目前結論見 `paper_status.md`，實驗網格見 `todo.png`。

---

## 1. 環境

- conda env `clreasoning`（Python 3.13）；依賴列在 `pyproject.toml`（repo 內有 `poetry.lock` 與 `uv.lock`）。
- **所有指令都從 repo root 執行**：資料路徑（`result/...`、`Data/...`）都相對於 root。
- API key（環境變數）：

| 模型（`-m`） | 環境變數 | 備註 |
|---|---|---|
| `gpt4omini`、`gpt4.1mini` | `OPENAI_API_KEY` | `gpt4.1mini` 是改寫器 |
| `qwen` | `QWEN_API_KEY` | DashScope，Qwen3-8B，關閉 thinking |
| `deepseek4.1flash` | `DEEPSEEK_API_KEY` | DeepSeek 官方 API 的 `deepseek-flash`（= DeepSeek-V4.1-Flash），關閉 thinking |
| `gemini3.1flashlite` | `GEMINI_API_KEY` | `gemini-3.1-flash-lite`，thinking_level minimal，上限 8192，不接受 `seed` |
| `gemini` | `GEMINI_API_KEY` | Gemini 2.5 Flash-Lite，**已退出新框架**，結果在 `result/archive/`（見 §7） |
| `deepseek` | `GMI_API_KEY` | GMI Cloud 的 DeepSeek-V3.2，**新實驗不用**（見 §7） |

---

## 2. 目錄結構

```
run_generate.py      產生 arm（一個模型 × 資料集 × arm 的所有回答）
run_aggregate.py     聚合候選 arm（judge / debate / ...）
run_rewrite.py       產生英文改寫題（W arm 的題目來源）
run_analysis.py      K=2 分析表 -> result/analysis/aggregation_cells.csv
import_legacy.py     把舊格式結果匯入 result/arms、result/aggregations（一次性，已完成）
check_framework.py   離線檢查（不呼叫 API），改程式後必跑

Arm/                 ArmSpec（arm_id 文法）、PromptBuilder、GenerationRecord
Aggregator/          聚合器（Template Method：一致題 no-op，分歧題 resolve）、AggregationRecord
Analysis/            實驗計畫（14 條 path / 19 個配對）、對齊、指標、split-half
Runner/              runner 共用：路徑、task 排程、model / dataset builder
Strategy/            Generate / Aggregate / Rewrite + prompt factories
Model/  Dataset/  File/  Log/
Test/                舊格式結果檔的評分（TestEM、TestRecoveryBlind、TestTokenNums ...）
scripts/analysis_0A/ paper_status 0A 系列分析（split-half、EIV、難度分層 ...）
scripts/analysis_rq1/ RQ1：少量標註預測「聚合或用單一 path」
scripts/analysis_rq2/ RQ2：交叉實驗（候選固定、只換 Judge 模型）的呼叫與分析
scripts/analysis_rq1k/ RQ1-K：K 條 path 的多數決 vs 同一份菜單內最強的單一 path（離線）
scripts/analysis_rq1kj/ RQ1-KJ：K 條 path 的 Judge 版本 vs 同一份菜單內最強的單一 path（呼叫 API）
scripts/analysis_rq2da/ RQ2-DA：強模型從弱模型的候選中挑 vs 強模型自己直接作答（離線）
scripts/analysis_rq3/ RQ3：拿掉 path 之後的多數決（M12 去掉一條、去掉兩條 persona；離線）
scripts/analysis_decomposition/ 主網格 12 組配對的分解總表與圖（Judge、Debate；只描述，離線）
scripts/analysis_rq3g/ RQ3-G：隨機改進一條 path 對聚合的影響、跨模型替換 vs 隨機模型（離線）
scripts/analysis_rq3gk/ RQ3-GK：RQ3-G 推廣到 K = 3–12 的菜單與三種平手規則（離線）
scripts/analysis_rq3gs/ RQ3-GS：在 187 份沒看過的三條菜單上確認挑法（離線）
scripts/analysis_rq3gsk/ RQ3-GSK：落單程度推廣到 K = 5、7，落單的那條還是最不值得改嗎（離線）
scripts/analysis_rq3gj/ RQ3-GJ：宿主自己當裁判時，該改哪一條 path（K = 3 的四組菜單與 K = 2；呼叫 API）
scripts/analysis_rq3gjr/ RQ3-GJR：用 RQ3-GJ 的裁判紀錄做條件式 logit，預測裁判的選擇（離線）
scripts/analysis_rq3gsx/ RQ3-GSX：依正確率加權的投票（WV）下，落單的那條還是最不值得改嗎（K = 3、5、7；離線）
scripts/analysis_rq3gjt/ RQ3-GJT：只用原本版本的裁判紀錄與假設的改進，預測裁判的變化（三層、五種切法、資料量曲線；離線）
scripts/legacy_eval/ 舊格式結果的評分與彙整（test_em、test_tokens、test_em_legacy ...）
scripts/router/      XLM-R router（原論文的 per-query 語言對路由）

Data/data/           MathQA、CommonsenseQA 原始檔（MMLU、TruthfulQA 由 HF datasets 下載）
Data/v2_translated/  四個資料集的 ZH / JA / RU / ES 翻譯（產生它的程式已刪除，無法重產）
Data/rewritten/      英文改寫 v1（{d}_english.json）、v2（{d}_english_v2.json）

result/arms/{model}/{dataset}/{arm}.json                     generation 紀錄
result/aggregations/{model}/{dataset}/{agg}__{arm}__{arm}.json  aggregation 紀錄
result/analysis/aggregation_cells.csv                        run_analysis 輸出
result/analysis/items/    run_analysis 的逐題匯出（paths.csv.gz、aggregations.csv.gz）
result/analysis/0A/       0A 系列的 CSV
result/analysis/rq1/      RQ1 的判定標準與輸出
result/analysis/rq2/      RQ2 的判定標準、交叉 Judge 的原始輸出（judge_outputs/，付費取得、不可重產）與分析輸出
result/analysis/rq1k/     RQ1-K 的判定標準與輸出
result/analysis/rq1kj/    RQ1-KJ 的判定標準、開跑前檢查、K 條 path Judge 的原始輸出（judge_outputs/，付費取得、不可重產）與分析輸出
result/analysis/rq1k/     RQ1-K 的判定標準與輸出
result/analysis/legacy/   舊 summary 表
result/archive/           退出新框架的結果（gemini-2.5-flash-lite/{arms,aggregations}/gemini/...）
result/baseline/ challenge/ self_reflection/ english_*/ tempature*/ voting/ oldresult/   舊格式結果（唯讀）
```

---

## 3. 概念

- **arm**：一種產生候選的方式；錨點是 `L:en`（原始英文題 + 完整 CoT + T=0）。每個 arm 相對錨點只改一個因素。
- **arm_id**：

| arm_id | 改變的因素 | 例子 |
|---|---|---|
| `L:{lang}` | 語言（題目翻譯 + 該語言的 prompt） | `L:zh` `L:ja` `L:ru` `L:es` |
| `S:T{temp}:seed{n}` | 採樣 | `S:T1.0:seed1` |
| `P:{persona}` | persona（`expert`、`skeptic`） | `P:expert` |
| `W:{rewrite1\|rewrite2}` | 英文改寫題 | `W:rewrite1` |
| `R:{short_cot\|direct}` | 推理強度 | `R:short_cot` |
| `F:{lang}` | 自我修正：prompt 含 base arm `L:{lang}` 的輸出（衍生 arm） | `F:en` |

- **聚合器**：`judge`（單次 LLM 裁決，看完整推理後選一個候選、輸出編號，最終答案一定是某個候選的答案）、
  `debate`（IMSR 跨 agent 辯論）、`blind`（固定選某一方）、
  `revise`（固定選衍生 arm）、`v2`（K=2 投票，recovery ≡ 0）、`vote3` / `vote5` / `judge5`（K>2）。
  所有聚合器在兩個候選答案相同時都不動作（no-op），只在分歧題呼叫 LLM。
- **item_id** 對齊所有 arm；`--nums 2000` 讓 item_id 和舊資料一致（TruthfulQA 只有 817 題）。

---

## 4. 實驗流程（SOP）

每個指令都可以中斷後**用同一個指令重跑**：已寫入的題目會跳過，只補失敗的。
被供應商擋下的內容（安全過濾、內容審查）記成「沒有答案」（算答錯），不會一直重試。

**① 改寫題（只有 W arm 需要）**

```bash
python run_rewrite.py -m gpt4.1mini -d mmlu mathqa truthfulqa commonsenseqa --nums 2000 --version 1
python run_rewrite.py -m gpt4.1mini -d mmlu mathqa truthfulqa commonsenseqa --nums 2000 --version 2   # 看過 v1、措辭必須不同
```

**② 產生 arm**

```bash
python run_generate.py -m deepseek4.1flash gemini3.1flashlite -d mmlu mathqa truthfulqa commonsenseqa --nums 2000 \
    --arms L:en L:zh L:ja L:ru L:es S:T1.0:seed1 S:T1.0:seed2 P:expert P:skeptic W:rewrite1 W:rewrite2 R:short_cot
# F arm 需要完整的 base arm，所以要在 base 跑完之後另外下指令
python run_generate.py -m deepseek4.1flash gemini3.1flashlite -d mmlu mathqa truthfulqa commonsenseqa --nums 2000 --arms F:en F:zh
```

- W arm 的改寫檔必須涵蓋每一題，否則拒跑（Dataset 會靜默保留原題）。
- F arm：base 輸出是空的題目不呼叫模型，直接記成沒有答案（列在 metadata `no_answer_fill`）。

**③ 聚合**

```bash
python run_aggregate.py -m gpt4omini qwen deepseek4.1flash gemini3.1flashlite -d mmlu mathqa truthfulqa commonsenseqa --nums 2000 \
    -a judge --candidates L:en,L:zh --candidates L:en,S:T1.0:seed1 --candidates P:expert,P:skeptic
python run_aggregate.py -m gpt4omini qwen deepseek4.1flash gemini3.1flashlite -d mmlu mathqa truthfulqa commonsenseqa --nums 2000 \
    -a debate --candidates L:en,S:T1.0:seed1
```

- 候選的順序就是檔名順序（錨點放第一個）；Judge 的呈現順序會在分歧題間自動平衡（K=2 一半錨點在前）。
- Aggregate 會用 prompt_hash 驗證每個候選 arm 的 prompt；衍生 arm 只有在 base arm 也是候選時才能驗證。
- 聚合檔的 metadata 記 `prompt_version`（Judge 目前是 `choice-v1`）。用舊 prompt 寫的檔案不會被續跑：
  要重跑時先把舊檔移到 `result/archive/`。2026-10 以前的 judge 檔是舊 prompt（輸出選項本身，會 off-menu）。

**④ 分析**

```bash
python run_analysis.py
```

- 輸出 `result/analysis/aggregation_cells.csv`：model × dataset × 配對 × {blind, judge, debate} × {all, both_answered}。
  配對與 path 代號（EN、ZH、S1、P1、W1、R、SR-EN ...）定義在 `Analysis/experimentPlan.py`；欄位定義在 `Analysis/metrics.py`。
- `blind` 由分析程式計算（不跑聚合器）；`*_H2` 欄位 = split-half：在 H1 選較強的一方、在 H2 量測（seed 0、200 次）。
- `both_answered`：只留兩個候選都有解析出答案的題目（處理 Gemini 的截斷）；聚合器沒給答案仍算答錯，比例在 `agg_no_answer`。
- token 欄位只算輸出 token（每題平均；一致題的聚合成本為 0）。
- 內建 0A-1 比對：舊跨語言辯論配對的 split-half 值必須和 `result/challenge` 的 RecoveryBlind metadata 完全相同
  （GPT-4o mini / Qwen / DeepSeek 共 120 個；Gemini 2.5 的 40 個隨它退出 `result/arms`）。
- 逐題匯出 `result/analysis/items/`（`--items-dir ""` 可跳過）：`paths.csv.gz` 是每條 path 每題的答案與對錯，
  `aggregations.csv.gz` 是每個 judge / debate 每題的最終答案與對錯。Blind 不匯出（選哪一方取決於 split，可由 paths 算）。
- 分析已退出的 Gemini 2.5：`python run_analysis.py --armdir result/archive/gemini-2.5-flash-lite/arms --aggdir result/archive/gemini-2.5-flash-lite/aggregations -m gemini --items-dir "" --challenge-dir ""`

**⑤ 匯入舊資料（一次性，已完成）**

```bash
python import_legacy.py -m gpt4omini -d mathqa --sr-dir result/self_reflection
```

**⑥ RQ1：少量標註預測「聚合或用單一 path」**（先跑完 ④，它讀 `result/analysis/items/`）

```bash
python scripts/analysis_rq1/probe_regret.py --check-only          # 只做執行前核對（逐題資料 vs aggregation_cells.csv）
python scripts/analysis_rq1/probe_regret.py -m gpt4omini qwen     # preliminary -> result/analysis/rq1/preliminary/
python scripts/analysis_rq1/probe_regret.py                       # 最終（四個模型）-> result/analysis/rq1/
```

- 規格與判定標準在 `result/analysis/rq1/rq1_criteria.md`。它的「確認」欄填上時間之前，程式只允許 `--check-only`；
  確認之後不得修改，每份輸出都記錄它的 sha256。
- 每次重複（200 次，seed 0）把每個資料集切成 H1 / H2（`makeSplits`，同一資料集的模型與配對共用）；
  決策只用 H1（同格 probe 取 H1 的 k 題；遷移用來源格的整個 H1），評估一律在 H2。程式在 `Analysis/probe.py`，輸入層在 `Analysis/itemMatrix.py`。
- 輸出：`cells.csv.gz`、`blocks.csv`（每區塊原始數字）、`summary.csv`、`criteria.csv`、`report.md`、`k_curves_{judge,debate}.png`。

**⑦ RQ2：交叉實驗（候選答案固定，只換 Judge 模型）**（需要主網格的 arm 檔與 Judge 檔，以及 ④ 的 `aggregation_cells.csv`）

```bash
python scripts/analysis_rq2/run_cross_judge.py check     # §4.1 流程核對：四個模型各重跑 mmlu × EN+S1 自己裁決自己
python scripts/analysis_rq2/run_cross_judge.py pilot     # §4.2 試跑：每個 Judge 100 題，估全量成本
python scripts/analysis_rq2/run_cross_judge.py full      # 12 種非對角組合（+ §4.1 判定要重跑的對角線）；-j 可只跑某些 Judge
python scripts/analysis_rq2/cross_judge.py               # 分析 -> result/analysis/rq2/
```

- 規格與判定標準在 `result/analysis/rq2/rq2_criteria.md`（已確認，不得修改）；「確認」欄空著時兩支程式都拒跑，輸出記錄它的 sha256。
- 4 個模型 × 4 個 Judge × 4 個資料集 × 3 個配對（EN+ZH、EN+S1、P1+P2）。只裁決兩條 path 都有答案的分歧題；
  prompt、候選與呈現順序（主網格 Judge 檔逐題記錄的順序）都和主網格相同，只換 Judge 模型（`Strategy/CrossJudge.py`）。
- 原始輸出在 `judge_outputs/{judge}/{generator}/{dataset}/judge__*.json`，也是續跑的快取；每筆記錄供應商回傳的版本、呼叫時間與 API token。
- 每一步都要等上一步完成：`full` 需要 `precheck/precheck.json` 與通過的 `pilot.json`。
  流程核對一致率 < 95% 的模型，`full` 會用交叉流程重跑它的整條對角線。
- 分析程式先核對：沿用舊檔的對角線必須和 `aggregation_cells.csv` 完全相同；交叉檔必須完整、順序與主網格相同。不符就停。
- DeepSeek 在尖峰時段（週一至五 UTC 01–04、06–10）價格加倍，程式會提醒；可用 `-j deepseek4.1flash` 另外排在離峰。

**⑧ RQ1-K：多條 path 的多數決 vs 單一最強 path**（離線，只讀 `result/arms`）

```bash
python scripts/analysis_rq1k/menu_vote.py      # -> result/analysis/rq1k/
```

- 規格與判定標準在 `result/analysis/rq1k/rq1k_criteria.md`（已確認，不得修改）；「確認」欄空著時拒跑，輸出記錄它的 sha256。
- 菜單（M3L / M3S / M3P / M3W / M5L / M12 / M8EN / M14）定義在 `Analysis/menuVote.py`；多數決平手取優先順序最高的 path
  （L:en 第一，其餘依 path 短代號的字母序：EN、ES、JA、P1、P2、R、RU、S1、S2、SR-EN、SR-ZH、W1、W2、ZH；不是依 arm_id）。切分與 RQ1 相同（`makeSplits(n, 200, seed=0)`），S_in / S_all 在 H1 上選、在 H2 上評。
- 每次切分核對 Excess_in = headroom × (recovery − recovery_blind)（誤差 ≤ 1e-9），不符就停。
- 輸出：`rq1k_items.csv.gz`、`rq1k_blocks.csv`、`rq1k_compare.csv`（判定三的逐區塊值）、`report.md`、`excess_m12_blocks.png`、`excess_menus.png`。

**⑨ RQ1-KJ：K 條 path 的 Judge 版本 vs 單一最強 path**（需要 `result/arms`、主網格 Judge 檔、`result/analysis/rq1k/rq1k_blocks.csv`、
`result/analysis/rq2/judge_outputs/`；呼叫 API）

```bash
python scripts/analysis_rq1kj/run_menu_judge.py reproduce   # §6.1 離線重現 RQ1-K 的 A_V、S_in、Excess_V（1e-9）
python scripts/analysis_rq1kj/run_menu_judge.py template    # §6.2 K = 2 的 prompt 重算 tokens = 主網格 tokens_in（Gemini 走免費的 count_tokens）
python scripts/analysis_rq1kj/run_menu_judge.py check       # §6.3 四個模型重跑 mmlu × EN+S1 自己裁決自己（523 次呼叫）
python scripts/analysis_rq1kj/run_menu_judge.py pilot       # §6.4 每個模型 M12 100 題 × 2 次，估全量成本（800 次呼叫）
python scripts/analysis_rq1kj/run_menu_judge.py full        # 全量（共 22,632 次，試跑的 400 次沿用）；-m 可只跑某些模型
python scripts/analysis_rq1kj/menu_judge.py                 # 分析 -> result/analysis/rq1kj/
```

- 規格與判定標準在 `result/analysis/rq1kj/rq1kj_criteria.md`（已確認，不得修改）；「確認」欄空著時兩支程式都拒跑，輸出記錄它的 sha256。
- 每一步都要上一步的結果檔（`reproduce.json` → `template_check.json` → `precheck.json` → `pilot.json`）存在、
  寫於同一份判定標準之下且通過；沒通過就停下來回報，不跑下一步。`pilot` 通過後要使用者確認才跑 `full`。
- 自己裁決自己；prompt `choice-k-v1`（主網格 choice-v1 的同一個函式，候選數換成 K；`Aggregator/MenuJudgeAggregator.py`）；
  只裁決菜單內每條 path 都有答案且答案不完全一致的題目；候選順序是「每 K 題一組、每組不同的隨機基準順序」的循環旋轉
  （`Strategy/MenuJudge.groupedOrders`，seed 0）。菜單：M12（主要）、M3L、M3S、M3P。
- 原始輸出在 `judge_outputs/{model}/{dataset}/{menu}.json`，也是續跑的快取；每筆記錄供應商回傳的版本、呼叫時間、API token 與 prompt 的 sha256。
  `precheck/` 是 §6.3 的重跑，`pilot_rep2/` 是試跑的第二次（只用來算一致率）。
- 分析程式先核對：每個 Judge 檔都完整、prompt 為 choice-k-v1、順序等於 `groupedOrders`；每次切分核對
  Excess = headroom × (recovery − recovery_blind)（Judge 與多數決各一，誤差 ≤ 1e-9）。不符就停。
- 輸出：`rq1kj_blocks.csv`、`rq1kj_compare.csv`（判定三的逐區塊值）、`judge_outputs/items.csv.gz`、`models.csv`、`report.md`、
  `excess_menus.png`、`excess_j_m12_blocks.png`。
- DeepSeek 在尖峰時段（週一至五 UTC 01–04、06–10）價格加倍，程式會提醒；可用 `-m deepseek4.1flash` 另外排在離峰。

**⑩ RQ2-DA：強裁判挑選 vs 強模型直接作答**（離線；需要 `result/arms`、主網格 Judge 檔、`result/analysis/rq2/` 的交叉檔與 `cross_cells.csv`）

```bash
python scripts/analysis_rq2da/direct_answer.py               # -> result/analysis/rq2da/
```

- 規格與判定標準在 `result/analysis/rq2da/rq2da_criteria.md`（已確認，不得修改）；「確認」欄空著時程式拒跑，輸出記錄它的 sha256。
- 弱模型（gpt4omini、qwen）的兩條 path 都有答案的題目上，答案不同時：Sys_J 用強模型（deepseek4.1flash、gemini3.1flashlite）
  在 RQ2 挑出的答案，Sys_D 用強模型自己的 L:en。每個 generator × Judge 用 RQ2 判定時實際採用的版本（`cross_cells.csv` 的
  `used_in_analysis`：GPT、DeepSeek、Gemini 的自己裁決自己是 RQ2 重跑版，Qwen 是主網格）。逐格計算在 `Analysis/directAnswer.py`。
- 先做三項檢查，任何一項不符就停、不寫輸出：重現 RQ2 結果 13（+2.47 [+1.65, +3.29]，8/8）、48 格的不一致題數 = `cross_cells.csv`、
  每格 Sys_J − Sys_D = d × (a − b)（1e-9）。
- 輸出：`rq2da_cells.csv`、`rq2da_blocks.csv`、`report.md`、`systems_by_block.png`。

**⑪ RQ3：拿掉 path 之後的多數決**（離線；需要 `result/arms` 與 `result/analysis/rq1k/rq1k_blocks.csv`）

```bash
python scripts/analysis_rq3/menu_prune.py                    # -> result/analysis/rq3/
```

- 規格與判定標準在 `result/analysis/rq3/rq3_criteria.md`（已確認，不得修改）；「確認」欄空著時程式拒跑，輸出記錄它的 sha256。
- 菜單：M12、12 份 M12−p（去掉一條）、M10（去掉 P1、P2），全部在 M12 的子集（12 條都有答案）上算；多數決、平手規則、S_in、
  切分（makeSplits(n, 200, 0)）都沿用 RQ1-K。逐切分計算在 `Analysis/menuPrune.py`。
- 判定一 Prune1：在 H1 上選出拿掉後多數決最好的 p*（平手時先拿掉 ZH、W2、W1、S2、S1、RU、R、P2、P1、JA、ES，EN 最後），H2 上
  A_V(M12−p*) − A_V(M12)。判定二 ΔExcess = Excess_in(M10) − Excess_in(M12)。
- 先核對 M12 的 A_V、S_in、Excess_in = `rq1k_blocks.csv`（1e-9）與子集題數；分解的恆等式每次切分核對。不符就停、不寫輸出。
- 輸出：`rq3_blocks.csv`、`report.md`、`prune_each_path.png`、`excess_m12_m10.png`。

**⑫ 12 組配對的分解總表與圖**（離線、只描述，沒有判定；需要 `aggregation_cells.csv` 與 `result/analysis/items/`）

```bash
python scripts/analysis_decomposition/pair_tables.py         # -> result/analysis/decomposition/
```

- 四個新模型、both_answered：Judge 與 Debate 各 12 組配對（只有 5 組共同，分開列）。headroom = 100·d·(c − m)，recovery 用
  `recovery_H2` / `recovery_blind_H2`；單一最強拿回、聚合拿回、Excess 都是每個區塊先算再平均；「聚合 − L:en」用逐題匯出算。
- 先核對現況文件的數字（EN+ZH、EN+S1、自我修正的 d、依來源的 recovery），對不上就停、不寫輸出。
- 輸出：`pair_table_judge`、`pair_table_debate`、`pair_table_judge_by_strength`（.csv/.md）、`cells_long.csv`、
  `fig_pairs_judge`、`fig_pairs_judge_by_strength`、`fig_pairs_debate`（.png 300 dpi / .pdf）、`report.md`。

**⑬ RQ3-G：改進一條 path，聚合會多多少？該改哪一條？**（離線；需要 `result/arms` 與 `result/analysis/rq3/rq3_blocks.csv`，約 10 分鐘）

```bash
python scripts/analysis_rq3g/path_improve.py                 # -> result/analysis/rq3g/
```

- 規格與判定標準在 `result/analysis/rq3g/rq3g_criteria.md`（已確認，不得修改）；「確認」欄空著時程式拒跑，輸出記錄它的 sha256。
- 隨機改進模型：把 path p 在選擇半、評分半各自隨機改對 t 題，直接算期望值（WV、SB 的規則由選擇半改進後的題數決定）。
  逐切分計算在 `Analysis/pathImprove.py`；答案編成整數，載入時核對 compareTwoAnswer = 字串相等、整數投票 = `vote`。
- 判定一 D1：M12、子集一，在選擇半挑「+5pp 時 V 增加最多」的 p*，評分半上 gain(p*) − 其餘 11 條的平均。判定二 D2：宿主
  gpt4omini / qwen 的 path p 換成供體 deepseek4.1flash / gemini3.1flashlite 的同一條，A_real − A_sim（子集二，有效切分）。
- 先做四項檢查（重現 RQ3、蒙地卡羅 2,000 次、把 path 換成它自己、子集二的保留比例），前三項不過就停、不寫輸出。
- 輸出：`rq3g_cells.csv`、`rq3g_substitutions.csv`、`rq3g_curves.csv`、`report.md`（不畫圖）。

**⑭ RQ3-GK：不同的 K 下，改進一條 path 聚合會多多少？**（離線；需要 `result/arms` 與 `result/analysis/rq3g/rq3g_cells.csv`，約 15 分鐘）

```bash
python scripts/analysis_rq3gk/path_improve_k.py              # -> result/analysis/rq3gk/
```

- 規格與判定標準在 `result/analysis/rq3gk/rq3gk_criteria.md`（已確認，不得修改；抽到的 120 份菜單列在它的 §3）；
  其餘定義沿用 `rq3g_criteria.md`。輸出記錄判定標準檔的 sha256。
- 菜單：K = 3、5、7、9 各 30 份（一個 `default_rng(0)`，`rng.choice(12, K, replace=False)`，重複就重抽），K = 11 的 12 份、
  M12；另有 M3L、M3S、M3P 只用在檢查。平手規則 A（原本的順序）、B（反過來）、C（平分）；逐切分計算在 `Analysis/pathImproveK.py`。
- 判定一 E1：K = 3、規則 A，替換後 V − 替換後 SB（8 個弱模型區塊）。判定二：K = 3、規則 C 的 D1（16 個區塊）。
- 先做四項檢查（重現 RQ3-G 的 M12 與 M3 菜單、規則 C 的核對與蒙地卡羅、菜單清單），不過就停、不寫輸出。
- 輸出：`rq3gk_k_blocks.csv`、`rq3gk_menu_blocks.csv`、`rq3gk_substitutions.csv`、`fig_a_conversion`、`fig_b_vote_minus_sb`、
  `fig_c_d1`（.png / .pdf）、`report.md`。

**⑮ RQ3-GS：在沒看過的三條組合上，確認挑法有沒有挑對**（離線；需要 `result/arms` 與 `result/analysis/rq3gk/rq3gk_k_blocks.csv`，約 2 分鐘）

```bash
python scripts/analysis_rq3gs/path_improve_gs.py             # -> result/analysis/rq3gs/
```

- 規格與判定標準在 `result/analysis/rq3gs/rq3gs_criteria.md`（已確認，不得修改；187 份確認用的菜單與分組列在它的 §4）；
  其餘定義沿用 `rq3g_criteria.md`、`rq3gk_criteria.md`。逐切分計算在 `Analysis/pathImproveGS.py`。
- 效果 = 10 × Σ分子 ÷ Σ分母（替換後 − 替換前的 V 得分，規則 C；分母 = 供體 − 宿主那條 path 的正確率；評分半 ∩ 子集二），
  任一條 path 無效的（菜單、供體、切分）整個不計。判定一 E1 = 挑出的（選擇半規則 C 的 gain）− 其他的；判定二 E2 =
  相像那一對 − 落單那條（「明顯落單」組）。
- 先做四項檢查（重現 RQ3-GK、菜單與分組 = 判定標準檔的表、換成自己、純迴圈手算 gpt4omini × mmlu × GS-001），不過就停、不寫輸出。
- 輸出：`rq3gs_blocks.csv`、`rq3gs_menu_blocks.csv`、`rq3gs_substitutions.csv`、`fig_a_groups`、`fig_b_degree_vs_e2`（.png / .pdf）、`report.md`。

**⑯ RQ3-GSK：K = 5、7 時，和其他條最不像的那條還是最不值得改嗎？**（離線；需要 `result/arms`、`result/analysis/rq3gs/` 與 `result/analysis/rq3gk/rq3gk_k_blocks.csv`）

```bash
python scripts/analysis_rq3gsk/path_improve_gsk.py           # -> result/analysis/rq3gsk/
```

- 規格與判定標準在 `result/analysis/rq3gsk/rq3gsk_criteria.md`（已確認，不得修改）；792 種菜單的表 `rq3gsk_menus_K5.csv`、
  `rq3gsk_menus_K7.csv` 是確認前寫好的，sha256 記在判定標準檔裡，程式只核對、不覆寫。逐切分計算在 `Analysis/pathImproveGSK.py`。
- 落單程度推廣到 K 條：s_i = 其他 K − 1 條彼此的一致率平均 − i 和其他條的一致率平均；s_i 相同時平手順序在後的算比較落單
  （K = 3 時和 RQ3-GS 完全相同）。判定 E_K = 其他 K − 1 條的效果 − 落單那條的效果（「明顯落單」組；門檻 K = 5 為 0.30、K = 7 為 0.20）。
- 先做六項檢查（K = 3 重現 RQ3-GS 判定二與菜單表、重現 RQ3-GK K = 5、7 的 D1 與轉換率、菜單表 = CSV、換成自己、
  純迴圈手算 gpt4omini × mmlu × GS5-001），不過就停、不寫輸出。
- 輸出：`rq3gsk_blocks.csv`、`rq3gsk_menu_blocks.csv`、`rq3gsk_substitutions.csv`、`fig_a_groups`、`fig_b_rank_effect`（.png / .pdf）、`report.md`。

**⑰ RQ3-GJ：讓模型當裁判時，該改哪一條 path？**（呼叫 API；規格與判定標準在 `result/analysis/rq3gj/rq3gj_criteria.md`，已確認，不得修改）

```bash
python scripts/analysis_rq3gj/run_gj_judge.py check      # §10 流程核對（310 次）            -> precheck/、precheck.json
python scripts/analysis_rq3gj/run_gj_judge.py pilot      # §11 試跑（582 + 680 次）           -> judge_outputs/、judge_outputs_k2/（pilot: true）、pilot.json
python scripts/analysis_rq3gj/run_gj_judge.py predict    # §7、§8.12、§8.13 的預測（離線，只能寫一次） -> rq3gj_predictions*.csv.gz、manifest
python scripts/analysis_rq3gj/run_gj_judge.py prerun     # §12 正式跑之前的五項檢查（離線）     -> prerun.json
python scripts/analysis_rq3gj/run_gj_judge.py full -w 16 # 正式跑（約 100 萬次；可中斷後續跑）
python scripts/analysis_rq3gj/path_improve_gj.py --dump  # 分析之前：印出 §8.8 抽到的 20 題 Qwen 輸出，標記寫進 LABELS
python scripts/analysis_rq3gj/path_improve_gj.py         # 分析（離線，約 1–2 分鐘）-> rq3gj_*.csv、圖、report.md
```

- 每一步都要上一步的結果檔存在、寫於同一份判定標準之下且通過；不過就停，回報後才往下。呼叫可續跑，已寫入的題目不重複計費。
- 宿主自己當裁判（gpt4omini、qwen；T = 0、max_tokens 8192、不傳 seed；Qwen 關 thinking）。K = 3 用 choice-k-v1，K = 2 用 choice-v1；
  被換進來的那條候選讀供體（deepseek4.1flash、gemini3.1flashlite）的 arm 檔，供體本身不被呼叫。
- 計畫（四組 39 份菜單、每份 7 個版本、K = 2 的 3 個配對 × 5 個版本 × 2 種順序、要呼叫的題目、候選順序）在 `Analysis/judgeSubstitution.py`，
  每一步開始時重算並和第零階段的 `rq3gj_stage0_calls.csv` 核對；預測在 `Analysis/judgeSubstitutionPredict.py`；
  Judge 呼叫用 `Strategy/SubstitutionJudge.py`；分析的逐區塊計算在 `Analysis/judgeSubstitutionStats.py`（只讀存好的預測檔，sha256 不符就停）。

**⑱ RQ3-GJR：用裁判紀錄做 regression，預測裁判選哪一個**（離線；需要 ⑰ 的 Judge 紀錄與預測檔、RQ1-KJ 的 Judge 紀錄、`result/arms`）

```bash
python scripts/analysis_rq3gjr/judge_regression.py --checks-only   # 只跑第 10 節的六項檢查（約 2 分鐘），不寫輸出
python scripts/analysis_rq3gjr/judge_regression.py                 # 檢查全過才正式計算（約 6 分鐘）-> result/analysis/rq3gjr/
```

- 規格與判定標準在 `result/analysis/rq3gjr/rq3gjr_criteria.md`（已確認，不得修改）。每次呼叫是一組，候選依顯示的位置排；
  條件式 logit（不加截距）用自己寫的牛頓法配適（`Analysis/judgeRegression.py`），M0–M4 逐步加入 correct、support、位置、
  誰寫的、文字類型、loglen（o200k tokens）；M5a、M5b 加 pathacc，只做逐區塊配適。每個裁判分開配適。
- 判定一 = M3 在留一個資料集下的 D_P；判定二 = T_new（同樣訓練資料重估的 9 格表）與 M3 的平均絕對誤差之差。評分的題目、切分與
  D_P 的算法和 RQ3-GJ 判定三相同；T_old 是 RQ3-GJ 存好的留一個資料集預測（只讀，sha256 核對）。
- 先做六項檢查（重現 RQ3-GJ 判定三與留一個資料集、呼叫數、§8.3 採用率表、T_new 的算法重現 RQ3-GJ 的預測、資料結構、
  牛頓法 vs statsmodels ConditionalLogit 的牛頓法），任何一項不過就停、不寫輸出。
- 輸出：`rq3gjr_blocks.csv`、`rq3gjr_substitutions.csv`、`rq3gjr_coefficients.csv`（128 次配適）、
  `rq3gjr_predictions_{T_new,M0,M1,M2,M3,M4,M3chars}.csv.gz` 與 manifest、`fig_a_predictions`、`fig_b_scenarios`（.png / .pdf）、`report.md`。

**⑲ RQ3-GSX：依正確率加權的投票下，和其他條最不像的那條還是最不值得改嗎？**（離線；需要 `result/arms`、`result/analysis/rq3g/`、`rq3gk/`、`rq3gs/`、`rq3gsk/` 的既有輸出）

```bash
python scripts/analysis_rq3gsx/path_improve_gsx.py --workers 14   # -> result/analysis/rq3gsx/（14 個行程約 1 小時）
```

- 規格與判定標準在 `result/analysis/rq3gsx/rq3gsx_criteria.md`（已確認，不得修改）。菜單與分組完全沿用 RQ3-GS（K = 3 的 187 份）與
  RQ3-GSK（K = 5、7 各 762 份，`rq3gsk_menus_K{5,7}.csv`）；每次切分的落單 path 同 RQ3-GSK 第 3 節。
- WV 的權重 = max(0, ln(p̂ / (1 − p̂)))，p̂ 用選擇半 ∩ 子集二的答對題數；被替換的 path 用供體的答對題數重算。200 次切分的權重一起計分
  （`Analysis/pathImproveGSX.py` 的 `wvScores`）。效果 = 10 × Σ分子 ÷ Σ分母（規則 C；排除同 RQ3-GS），判定 E_W(K) = 其他條 − 落單那條
  （明顯落單組；門檻 0.50 / 0.30 / 0.20）；同一批資料上的 V 並排。另有隨機改進的模擬與權重固定的版本（分開答案與權重）。
- 先做六項檢查（重現 RQ3-GS / RQ3-GSK 的 V、RQ3-G 的 12 條 WV 與 RQ3-GK K = 3 的 WV − SB、菜單表、權重相同時 WV = V、換成自己、
  純迴圈手算 gpt4omini × mmlu × GS-001 / GS5-001），不過就停、不寫輸出。計算分成（K、區塊、一段菜單）的工作平行跑，依工作順序合併。
- 輸出：`rq3gsx_blocks.csv`、`rq3gsx_menu_blocks.csv`、`rq3gsx_substitutions.csv`、`fig_a_judgments`、`fig_b_rank_effect`（.png / .pdf）、`report.md`。

**⑳ RQ3-GJT：不用真的去改 path，只用原本版本的裁判紀錄，估得出裁判的變化嗎？**（離線；需要 ⑰ 的 K = 3 Judge 紀錄與預測檔、⑱ 的預測檔與輸出、`result/arms`）

```bash
python scripts/analysis_rq3gjt/judge_transfer.py --checks-only   # 只跑第 10 節的六項檢查，不寫輸出
python scripts/analysis_rq3gjt/judge_transfer.py                 # 檢查全過 → 曲線 → 預測檔與 manifest → 評分與報告 -> result/analysis/rq3gjt/
python scripts/analysis_rq3gjt/judge_transfer.py --score-only    # 預測檔已寫好：核對 sha256 後只重做評分與報告
```

- 規格與判定標準在 `result/analysis/rq3gjt/rq3gjt_criteria.md`（已確認，不得修改）。三層：A = 只用原本版本的紀錄配適 M3h（M3 拿掉 donor），
  輸入是原本的三份候選加上「path p 進步」的假設（RQ3-G 的隨機改進，直接算期望值）；B = 7 個版本的紀錄配適 M3，同樣的假設但被改進的那份
  設 donor = 1；C = RQ3-GJR（真的換進來的候選）。每層另有同樣訓練資料估的 9 格比例表。假設狀態的因素在 `Analysis/judgeTransfer.py`
  （算法同 `Analysis/judgeRegression.buildGroups`），配適沿用 `Analysis/judgeRegression.py` 的牛頓法。
- 切法：① 每個資料集各自、② 四個資料集合併（前 20 次切分）、③ 留一個資料集（主要；C 層讀 RQ3-GJR 的存檔）、④ 留一組菜單、⑥ 換裁判
  （200 次切分）；⑤ 照抄 RQ3-GJR 的 K = 2。判定一 = 明顯落單組的 Δ_E（A 層、M3h、③）；判定二 = 只抽 1% 訓練題目時變化誤差的增加（曲線 (i)）。
- 先做六項檢查（重現 RQ3-GJR 的判定、係數與預測，第 7 節 5、6；呼叫數；f = 0、換成自己、B 層的路徑餵真的候選 = C 層、純迴圈手算；
  洩漏；牛頓法 vs statsmodels），任何一項不過就停、不寫輸出。之後才跑曲線並寫預測檔，評分程式讀預測檔前核對 sha256。
- 輸出：`rq3gjt_blocks.csv`（長表）、`rq3gjt_substitutions.csv`、`rq3gjt_coefficients.csv`、`rq3gjt_curves.csv`、
  `rq3gjt_predictions_manifest.json`、`fig_a_gain`、`fig_b_cuts`、`fig_c_curves`（.png / .pdf）、`report.md`；
  預測檔 `rq3gjt_pred_*.csv.gz`、`rq3gjt_curve_*.npz`（約 200 MB，不進 git，manifest 進 git）。
- `--fake-choices SEED --out-dir <別的目錄>`：乾跑，把裁判的有效選擇換成隨機的，只用來測程式（不能寫到正式的目錄）。

---

## 5. 驗證

- **改任何程式之後**：`python check_framework.py`（離線、約 2 秒，必須 All checks passed）。
  它也保存了已刪除的舊策略（OnlyOneLanguage / SelfReflection / Challenge）的 golden prompt hash 與辯論 digest。
- **改到分析相關程式**：`python run_analysis.py` 的 0A-1 比對必須通過（120 個）。
- **改到 0A 相關程式**：依序重跑，輸出到 `result/analysis/0A/`，再和 git 版本比對：

```bash
python scripts/legacy_eval/test_em.py -t testrecoveryblind --testdir result/challenge      # 寫回 challenge 檔的 metadata
python scripts/legacy_eval/test_em_legacy.py result/tempature1/challenge_CN result/tempature1/challenge_EN \
    --csv result/analysis/0A/legacy_em_samelang.csv
python scripts/analysis_0A/regress_pair_acc.py
python scripts/analysis_0A/loo_gap_star.py
python scripts/analysis_0A/split_half_gap.py
python scripts/analysis_0A/split_half_eiv.py
python scripts/analysis_0A/analyze_recovery_blind.py
python scripts/analysis_0A/difficulty_strata.py
python scripts/analysis_0A/recovery_heterogeneity.py
```

  `difficulty_q_groups`、`difficulty_q_pairs`、`difficulty_strata_adjusted`、`loo_folds`、`loo_predictions`、
  `pair_acc_cell_estimates` 重跑後會和 git 版本差 ≤1e-13（套件版本造成的浮點誤差），其餘必須逐 byte 相同。

---

## 6. 舊格式結果與腳本

- `result/baseline`（單語）、`result/challenge`（跨語言辯論）、`result/self_reflection`、`result/english_*`、`result/tempature*`、
  `result/voting`、`result/oldresult` 是重構前的結果，**唯讀**。前三者已匯入 `result/arms` / `result/aggregations`。
- 讀它們的工具在 `scripts/legacy_eval/`（`test_em.py -t {testem,testp,testrecoveryblind,testmissing}`、
  `test_tokens.py`、`test_em_legacy.py`、`check_results.py`、`table_result*.py`）。
- 每題 output token 的舊定義保存在 `Test/TestTokenNums.py` 的 `TOKEN_COUNTERS`。

---

## 7. 已知限制

- **DeepSeek**：官方 API 已換掉舊模型，`deepseek` 改打 GMI Cloud 上的 V3.2，但約 30% 請求回 400（model deprecated），服務不穩定。
  舊的 DeepSeek 資料（L 軸、辯論、SR）照常分析，但**不要用 `-m deepseek` 跑新的生成、judge 或 debate**。
- **Gemini 2.5 Flash-Lite（`gemini`，已退出）**：T=0 時約 7% 的輸出陷入重複、跑到 4096 上限而沒有答案（上限放到 16384 也只救回 2/26），
  使 d / recovery 量到的是截斷而不是多樣性。2026-10-04 起移到 `result/archive/gemini-2.5-flash-lite/`，不再進主分析。
- **Gemini 3.1 Flash-Lite（`gemini3.1flashlite`）**：Google 建議 Gemini 3 用 T=1.0（較低可能重複），這裡照協定用 T=0；
  2.5 曾重複 / 截斷的 150 題在 3.1 上全部正常收尾（最長 1,513 token）。thinking_level minimal 不保證完全不思考。不接受 `seed`。
- **DeepSeek V4.1 Flash（`deepseek4.1flash`）**：官方 API 只回傳別名 `deepseek-flash`，版本對應（V4.1-Flash）來自 API 文件。
  DeepSeek 換過別名背後的模型，所以同一批實驗要一次跑完。尖峰時段（週一至五 UTC 01–04、06–10）價格加倍。
- **平行執行**：qwen 偶爾出現 tqdm `_lock` 的 thread race，單獨重跑那個 task 即可。
- **router**：`scripts/router/train.py` 需要 `scikit-learn`（不在 `pyproject.toml`）。
- `scripts/legacy_eval/table_result*.py` 依檔案系統順序讀檔，輸出的列順序不固定（內容相同）。
