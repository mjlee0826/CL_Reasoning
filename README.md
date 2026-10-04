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
