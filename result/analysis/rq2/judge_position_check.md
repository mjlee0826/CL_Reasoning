# 核對六：兩個候選的 Judge 有沒有偏向某個位置

產生時間 2026-10-07 13:46 CST；程式 `scripts/analysis_rq2/judge_position_check.py`。不呼叫 API，不改任何判定、計分與現有檔案。

## (1) 讀了哪些檔案

- 主網格兩個候選的 Judge（prompt choice-v1，主網格正式的那一次，各模型裁決自己）：`result/aggregations/{模型}/{資料集}/judge__*.json`，4 個模型 × 12 組配對 × 4 個資料集 = 192 個檔。欄位：`item_id`、`presentation_order`、`final_answer`、`off_menu`、`trace.choice`、`trace.chosen_arm`；metadata 的 `candidate_arms`、`prompt_version`。不讀舊模型的 `deepseek/`、`gemini/`，也不讀 RQ2 的重跑（`result/analysis/rq2/precheck/` 與 `judge_outputs/{m}/{m}/` 的對角線）。
- 候選答案與對錯：`result/arms/{模型}/{資料集}/` 的對應 path 檔，欄位 `gold`、`parsed_answer`、`parse_ok`；對錯用 `compareTwoAnswer`（字串相等）。subset = both_answered。
- 第零節 1：`result/analysis/aggregation_cells.csv`（qwen × judge × both_answered 的 12 列：`d`、`c`、`m`、`recovery`）與 `result/analysis/rq2/qwen_k2_check.md` 第零節的表。
- 第五節（只報告）：`result/analysis/rq2/judge_outputs/{裁判}/{候選模型}/{資料集}/judge__*.json`，裁判 ≠ 候選模型的 144 格（只收 both_answered 的分歧題）；候選答案讀 `result/arms/{候選模型}/`。

## (2) 第零節的檢查

1. 重現核對四第零節（qwen 12 格）：和 `aggregation_cells.csv` 的最大差 5.6e-17（門檻 1e-12），四捨五入到小數四位後和核對四的表逐格相同 → 通過。

| 配對    | 資料集           | d      | c      | m      | 分歧題正確率   | 與 CSV 的最大差   | 與核對四的表相同   |
|:------|:--------------|:-------|:-------|:-------|:---------|:-------------|:-----------|
| EN+ZH | mmlu          | 0.2245 | 0.7840 | 0.3920 | 0.4922   | 5.6e-17      | 是          |
| EN+ZH | mathqa        | 0.1885 | 0.7507 | 0.3753 | 0.5093   | 0.0e+00      | 是          |
| EN+ZH | truthfulqa    | 0.2228 | 0.7198 | 0.3599 | 0.3352   | 5.6e-17      | 是          |
| EN+ZH | commonsenseqa | 0.2860 | 0.8182 | 0.4091 | 0.5455   | 0.0e+00      | 是          |
| EN+S1 | mmlu          | 0.0970 | 0.7629 | 0.3814 | 0.4639   | 5.6e-17      | 是          |
| EN+S1 | mathqa        | 0.1277 | 0.7373 | 0.3686 | 0.4667   | 5.6e-17      | 是          |
| EN+S1 | truthfulqa    | 0.1176 | 0.6667 | 0.3333 | 0.3646   | 1.4e-17      | 是          |
| EN+S1 | commonsenseqa | 0.0865 | 0.7283 | 0.3642 | 0.3642   | 5.6e-17      | 是          |
| P1+P2 | mmlu          | 0.0891 | 0.7697 | 0.3848 | 0.4888   | 4.2e-17      | 是          |
| P1+P2 | mathqa        | 0.1203 | 0.7280 | 0.3640 | 0.4644   | 5.6e-17      | 是          |
| P1+P2 | truthfulqa    | 0.1420 | 0.7500 | 0.3750 | 0.4224   | 0.0e+00      | 是          |
| P1+P2 | commonsenseqa | 0.1315 | 0.8289 | 0.4144 | 0.4411   | 0.0e+00      | 是          |

2. 每格檔名第一條 path 排在位置 1 的比例（both_answered、有呼叫 Judge 的分歧題）：

| 模型                    | 格數   | 合併比例   | 最低   | 最高   | 不在 45%–55% 的格子   |
|:----------------------|:-----|:-------|:-----|:-----|:-----------------|
| GPT-4o mini           | 48   | 50.1   | 49.6 | 51.3 | 0                |
| Qwen3-8B              | 48   | 50.1   | 49.3 | 54.5 | 0                |
| DeepSeek V4.1 Flash   | 48   | 50.1   | 44.4 | 61.1 | 3                |
| Gemini 3.1 Flash-Lite | 48   | 50.3   | 48.5 | 53.1 | 0                |

不在 45%–55% 之間的 3 格（多半是題數少的格子）：

| 模型                  | 資料集    | 配對       | 第一條 path   | 題數   | 排在位置 1（%）   |
|:--------------------|:-------|:---------|:-----------|:-----|:------------|
| DeepSeek V4.1 Flash | mathqa | EN+SR-EN | L:en       | 18   | 61.1        |
| DeepSeek V4.1 Flash | mathqa | ZH+SR-ZH | L:zh       | 27   | 44.4        |
| DeepSeek V4.1 Flash | mathqa | P1+P2    | P:expert   | 50   | 56.0        |

## (3) 四個模型的合併結果與讀法

只用有呼叫 Judge、choice 為 1 或 2、final_answer 有解析出來、不是 off-menu 的分歧題（both_answered）。合併 = 題數加總後再算比例；區間 = 95% Clopper–Pearson。資料集欄是各資料集合併 12 組配對的值（%）。

| 模型                    | 有呼叫 Judge 的分歧題   | 排除   | 使用    | 編號 → 候選 → 答案對不上   |
|:----------------------|:-----------------|:-----|:------|:------------------|
| GPT-4o mini           | 10618            | 0    | 10618 | 0                 |
| Qwen3-8B              | 11592            | 1    | 11591 | 0                 |
| DeepSeek V4.1 Flash   | 6015             | 4    | 6011  | 0                 |
| Gemini 3.1 Flash-Lite | 4816             | 1    | 4815  | 0                 |

第 1 項（全部分歧題中選位置 2 的比例）與讀法：

| 模型                    | 第 1 項：選位置 2                    | mmlu   | mathqa   | truthfulqa   | commonsenseqa   | 讀法                                               |
|:----------------------|:-------------------------------|:-------|:---------|:-------------|:----------------|:-------------------------------------------------|
| GPT-4o mini           | 55.5% [54.6, 56.5]（5898/10618） | 49.4   | 56.1     | 55.9         | 60.3            | 其他情況 → 只回報，不下結論。                                 |
| Qwen3-8B              | 70.4% [69.6, 71.2]（8161/11591） | 67.3   | 61.7     | 75.0         | 79.0            | **在 60% 以上或 40% 以下，而且 4 個資料集方向一致 → 有位置偏向，特別標出。** |
| DeepSeek V4.1 Flash   | 63.2% [62.0, 64.5]（3801/6011）  | 61.1   | 56.2     | 62.8         | 65.8            | **在 60% 以上或 40% 以下，而且 4 個資料集方向一致 → 有位置偏向，特別標出。** |
| Gemini 3.1 Flash-Lite | 51.4% [50.0, 52.8]（2475/4815）  | 52.9   | 48.8     | 46.6         | 52.8            | 選位置 2 的比例在 45% 到 55% 之間 → 沒有看到位置偏向。              |

第 2 項（一對一錯的題目）：

| 模型                    | (a) 正確在位置 1 時選對               | (b) 正確在位置 2 時選對               | (c) (b) − (a)（pp）   | (c) mmlu   | (c) mathqa   | (c) truthfulqa   | (c) commonsenseqa   |
|:----------------------|:------------------------------|:------------------------------|:--------------------|:-----------|:-------------|:-----------------|:--------------------|
| GPT-4o mini           | 59.5% [57.9, 61.0]（2425/4078） | 70.1% [68.7, 71.5]（2861/4080） | +10.7               | -0.7       | +10.2        | +9.8             | +20.8               |
| Qwen3-8B              | 42.7% [41.3, 44.2]（1909/4469） | 82.3% [81.1, 83.4]（3693/4488） | +39.6               | +33.3      | +19.8        | +50.0            | +57.7               |
| DeepSeek V4.1 Flash   | 56.6% [54.6, 58.6]（1398/2470） | 80.9% [79.3, 82.5]（1998/2469） | +24.3               | +20.2      | +12.9        | +22.6            | +29.2               |
| Gemini 3.1 Flash-Lite | 68.0% [65.9, 70.1]（1357/1995） | 71.2% [69.1, 73.1]（1411/1983） | +3.1                | +3.9       | -3.3         | -6.0             | +6.9                |

第 3 項：

| 模型                    | 第 3 項：兩個都錯時選位置 2              | mmlu   | mathqa   | truthfulqa   | commonsenseqa   |
|:----------------------|:------------------------------|:-------|:---------|:-------------|:----------------|
| GPT-4o mini           | 56.3% [54.3, 58.2]（1384/2460） | 49.7   | 58.2     | 57.3         | 59.3            |
| Qwen3-8B              | 72.4% [70.7, 74.1]（1908/2634） | 68.8   | 66.6     | 74.6         | 81.1            |
| DeepSeek V4.1 Flash   | 68.2% [65.3, 71.0]（731/1072）  | 65.0   | 56.7     | 72.6         | 71.0            |
| Gemini 3.1 Flash-Lite | 50.9% [47.5, 54.3]（426/837）   | 58.2   | 51.2     | 46.5         | 49.1            |

特別標出：Qwen3-8B、DeepSeek V4.1 Flash 有位置偏向；第 2 項 (c) 的大小見上表。

## (4) 逐格的表

各資料集合併 12 組配對（%；(c) 為 pp）：

| 模型                    | 資料集           | 題數   | 選位置 2   | (a)   | (b)   | (c)   | 兩個都錯時選位置 2   | 兩個都錯的題數   |
|:----------------------|:--------------|:-----|:--------|:------|:------|:------|:-------------|:----------|
| GPT-4o mini           | mmlu          | 2741 | 49.4    | 61.9  | 61.2  | -0.7  | 49.7         | 580       |
| GPT-4o mini           | mathqa        | 3304 | 56.1    | 66.2  | 76.4  | +10.2 | 58.2         | 847       |
| GPT-4o mini           | truthfulqa    | 1452 | 55.9    | 51.3  | 61.1  | +9.8  | 57.3         | 485       |
| GPT-4o mini           | commonsenseqa | 3121 | 60.3    | 53.9  | 74.8  | +20.8 | 59.3         | 548       |
| Qwen3-8B              | mmlu          | 3093 | 67.3    | 44.9  | 78.1  | +33.3 | 68.8         | 688       |
| Qwen3-8B              | mathqa        | 3338 | 61.7    | 58.4  | 78.3  | +19.8 | 66.6         | 799       |
| Qwen3-8B              | truthfulqa    | 1454 | 75.0    | 32.7  | 82.7  | +50.0 | 74.6         | 418       |
| Qwen3-8B              | commonsenseqa | 3706 | 79.0    | 31.5  | 89.3  | +57.7 | 81.1         | 729       |
| DeepSeek V4.1 Flash   | mmlu          | 1289 | 61.1    | 54.6  | 74.8  | +20.2 | 65.0         | 197       |
| DeepSeek V4.1 Flash   | mathqa        | 657  | 56.2    | 61.2  | 74.1  | +12.9 | 56.7         | 150       |
| DeepSeek V4.1 Flash   | truthfulqa    | 987  | 62.8    | 68.5  | 91.1  | +22.6 | 72.6         | 212       |
| DeepSeek V4.1 Flash   | commonsenseqa | 3078 | 65.8    | 52.8  | 81.9  | +29.2 | 71.0         | 513       |
| Gemini 3.1 Flash-Lite | mmlu          | 958  | 52.9    | 64.8  | 68.7  | +3.9  | 58.2         | 146       |
| Gemini 3.1 Flash-Lite | mathqa        | 856  | 48.8    | 68.5  | 65.2  | -3.3  | 51.2         | 213       |
| Gemini 3.1 Flash-Lite | truthfulqa    | 558  | 46.6    | 79.3  | 73.3  | -6.0  | 46.5         | 101       |
| Gemini 3.1 Flash-Lite | commonsenseqa | 2443 | 52.8    | 66.6  | 73.5  | +6.9  | 49.1         | 377       |

逐（模型 × 資料集 × 配對）（%；(c) 為 pp；n = 使用的分歧題數）：

| 模型                    | 資料集           | 配對       | n   | 選位置 2   | (a)   | (b)   | (c)    | 兩個都錯時選位置 2   | 兩個都錯   |
|:----------------------|:--------------|:---------|:----|:--------|:------|:------|:-------|:-------------|:-------|
| GPT-4o mini           | mmlu          | EN+ZH    | 357 | 51.8    | 61.4  | 68.4  | +7.0   | 50.0         | 66     |
| GPT-4o mini           | mmlu          | EN+JA    | 391 | 52.2    | 62.5  | 64.2  | +1.7   | 61.3         | 75     |
| GPT-4o mini           | mmlu          | ZH+JA    | 407 | 48.2    | 67.3  | 61.8  | -5.5   | 50.0         | 100    |
| GPT-4o mini           | mmlu          | EN+S1    | 178 | 43.3    | 62.5  | 50.8  | -11.7  | 41.9         | 43     |
| GPT-4o mini           | mmlu          | S1+S2    | 217 | 47.0    | 58.3  | 51.9  | -6.4   | 50.0         | 44     |
| GPT-4o mini           | mmlu          | EN+P1    | 150 | 49.3    | 55.6  | 53.1  | -2.4   | 50.0         | 32     |
| GPT-4o mini           | mmlu          | P1+P2    | 164 | 45.7    | 61.1  | 55.0  | -6.1   | 43.8         | 32     |
| GPT-4o mini           | mmlu          | EN+W1    | 244 | 46.7    | 67.3  | 61.6  | -5.7   | 45.5         | 44     |
| GPT-4o mini           | mmlu          | W1+W2    | 219 | 49.8    | 66.3  | 65.9  | -0.4   | 51.0         | 51     |
| GPT-4o mini           | mmlu          | EN+R     | 168 | 50.6    | 62.0  | 65.2  | +3.2   | 48.4         | 31     |
| GPT-4o mini           | mmlu          | EN+SR-EN | 96  | 59.4    | 37.9  | 62.2  | +24.2  | 53.3         | 30     |
| GPT-4o mini           | mmlu          | ZH+SR-ZH | 150 | 50.0    | 55.9  | 62.7  | +6.8   | 37.5         | 32     |
| GPT-4o mini           | mathqa        | EN+ZH    | 358 | 53.6    | 69.2  | 80.7  | +11.6  | 49.4         | 77     |
| GPT-4o mini           | mathqa        | EN+JA    | 413 | 54.7    | 71.3  | 80.5  | +9.2   | 53.3         | 92     |
| GPT-4o mini           | mathqa        | ZH+JA    | 397 | 57.7    | 65.0  | 74.8  | +9.8   | 69.8         | 86     |
| GPT-4o mini           | mathqa        | EN+S1    | 294 | 58.8    | 63.1  | 78.8  | +15.7  | 59.0         | 78     |
| GPT-4o mini           | mathqa        | S1+S2    | 322 | 55.6    | 68.2  | 77.3  | +9.1   | 57.8         | 102    |
| GPT-4o mini           | mathqa        | EN+P1    | 223 | 56.1    | 71.2  | 81.2  | +10.0  | 58.7         | 63     |
| GPT-4o mini           | mathqa        | P1+P2    | 215 | 59.5    | 60.3  | 74.7  | +14.4  | 63.5         | 63     |
| GPT-4o mini           | mathqa        | EN+W1    | 265 | 52.8    | 67.4  | 70.9  | +3.5   | 52.9         | 70     |
| GPT-4o mini           | mathqa        | W1+W2    | 252 | 59.9    | 55.7  | 76.1  | +20.4  | 60.3         | 63     |
| GPT-4o mini           | mathqa        | EN+R     | 374 | 51.1    | 72.1  | 68.8  | -3.4   | 58.9         | 90     |
| GPT-4o mini           | mathqa        | EN+SR-EN | 78  | 65.4    | 46.4  | 80.8  | +34.3  | 62.5         | 24     |
| GPT-4o mini           | mathqa        | ZH+SR-ZH | 113 | 60.2    | 55.2  | 75.6  | +20.4  | 53.8         | 39     |
| GPT-4o mini           | truthfulqa    | EN+ZH    | 167 | 62.9    | 35.7  | 66.0  | +30.3  | 57.4         | 47     |
| GPT-4o mini           | truthfulqa    | EN+JA    | 169 | 53.3    | 53.1  | 60.9  | +7.8   | 49.0         | 51     |
| GPT-4o mini           | truthfulqa    | ZH+JA    | 185 | 61.6    | 44.6  | 60.5  | +15.9  | 69.8         | 53     |
| GPT-4o mini           | truthfulqa    | EN+S1    | 106 | 57.5    | 67.6  | 73.7  | +6.0   | 64.7         | 34     |
| GPT-4o mini           | truthfulqa    | S1+S2    | 124 | 57.3    | 51.6  | 61.1  | +9.5   | 59.6         | 57     |
| GPT-4o mini           | truthfulqa    | EN+P1    | 106 | 57.5    | 48.6  | 68.6  | +19.9  | 52.9         | 34     |
| GPT-4o mini           | truthfulqa    | P1+P2    | 104 | 48.1    | 47.1  | 43.9  | -3.2   | 48.3         | 29     |
| GPT-4o mini           | truthfulqa    | EN+W1    | 149 | 57.7    | 48.3  | 66.0  | +17.7  | 56.8         | 44     |
| GPT-4o mini           | truthfulqa    | W1+W2    | 117 | 47.0    | 71.4  | 45.0  | -26.4  | 64.3         | 42     |
| GPT-4o mini           | truthfulqa    | EN+R     | 115 | 48.7    | 72.4  | 62.8  | -9.6   | 48.8         | 43     |
| GPT-4o mini           | truthfulqa    | EN+SR-EN | 52  | 59.6    | 50.0  | 60.0  | +10.0  | 66.7         | 24     |
| GPT-4o mini           | truthfulqa    | ZH+SR-ZH | 58  | 53.4    | 46.7  | 68.8  | +22.1  | 44.4         | 27     |
| GPT-4o mini           | commonsenseqa | EN+ZH    | 463 | 62.4    | 57.2  | 84.8  | +27.6  | 55.4         | 65     |
| GPT-4o mini           | commonsenseqa | EN+JA    | 471 | 57.1    | 62.4  | 72.4  | +10.0  | 61.8         | 68     |
| GPT-4o mini           | commonsenseqa | ZH+JA    | 479 | 63.5    | 54.6  | 84.5  | +29.9  | 60.6         | 104    |
| GPT-4o mini           | commonsenseqa | EN+S1    | 174 | 67.8    | 38.5  | 74.7  | +36.2  | 66.7         | 21     |
| GPT-4o mini           | commonsenseqa | S1+S2    | 199 | 61.3    | 48.8  | 70.1  | +21.3  | 65.8         | 38     |
| GPT-4o mini           | commonsenseqa | EN+P1    | 159 | 56.6    | 48.6  | 59.3  | +10.7  | 65.4         | 26     |
| GPT-4o mini           | commonsenseqa | P1+P2    | 189 | 60.8    | 39.8  | 62.7  | +22.9  | 58.1         | 31     |
| GPT-4o mini           | commonsenseqa | EN+W1    | 304 | 59.5    | 57.6  | 72.9  | +15.2  | 63.0         | 46     |
| GPT-4o mini           | commonsenseqa | W1+W2    | 301 | 57.8    | 59.0  | 76.7  | +17.7  | 53.1         | 64     |
| GPT-4o mini           | commonsenseqa | EN+R     | 165 | 54.5    | 52.9  | 64.8  | +11.9  | 45.8         | 24     |
| GPT-4o mini           | commonsenseqa | EN+SR-EN | 69  | 58.0    | 57.1  | 62.5  | +5.4   | 68.8         | 16     |
| GPT-4o mini           | commonsenseqa | ZH+SR-ZH | 148 | 60.1    | 50.9  | 77.1  | +26.2  | 55.6         | 45     |
| Qwen3-8B              | mmlu          | EN+ZH    | 449 | 65.7    | 48.0  | 77.1  | +29.1  | 69.1         | 97     |
| Qwen3-8B              | mmlu          | EN+JA    | 480 | 65.0    | 52.1  | 82.3  | +30.2  | 64.6         | 96     |
| Qwen3-8B              | mmlu          | ZH+JA    | 538 | 64.3    | 50.5  | 77.2  | +26.7  | 64.7         | 133    |
| Qwen3-8B              | mmlu          | EN+S1    | 194 | 73.2    | 39.2  | 82.4  | +43.2  | 78.3         | 46     |
| Qwen3-8B              | mmlu          | S1+S2    | 199 | 72.9    | 35.2  | 75.3  | +40.1  | 81.4         | 43     |
| Qwen3-8B              | mmlu          | EN+P1    | 204 | 70.6    | 43.4  | 83.5  | +40.1  | 71.4         | 49     |
| Qwen3-8B              | mmlu          | P1+P2    | 178 | 71.9    | 43.3  | 79.2  | +35.9  | 80.5         | 41     |
| Qwen3-8B              | mmlu          | EN+W1    | 281 | 64.8    | 48.4  | 80.8  | +32.4  | 64.2         | 53     |
| Qwen3-8B              | mmlu          | W1+W2    | 260 | 65.0    | 39.2  | 68.9  | +29.7  | 65.4         | 52     |
| Qwen3-8B              | mmlu          | EN+R     | 223 | 69.5    | 31.8  | 71.2  | +39.4  | 69.1         | 55     |
| Qwen3-8B              | mmlu          | EN+SR-EN | 20  | 80.0    | 16.7  | 100.0 | +83.3  | 57.1         | 7      |
| Qwen3-8B              | mmlu          | ZH+SR-ZH | 67  | 70.1    | 34.8  | 82.1  | +47.4  | 56.2         | 16     |
| Qwen3-8B              | mathqa        | EN+ZH    | 377 | 64.2    | 53.7  | 81.0  | +27.3  | 63.8         | 94     |
| Qwen3-8B              | mathqa        | EN+JA    | 428 | 59.6    | 65.9  | 81.1  | +15.1  | 71.2         | 80     |
| Qwen3-8B              | mathqa        | ZH+JA    | 495 | 60.0    | 62.2  | 79.0  | +16.9  | 66.4         | 116    |
| Qwen3-8B              | mathqa        | EN+S1    | 255 | 61.2    | 56.0  | 71.6  | +15.6  | 73.1         | 67     |
| Qwen3-8B              | mathqa        | S1+S2    | 271 | 64.6    | 59.4  | 83.5  | +24.1  | 66.7         | 60     |
| Qwen3-8B              | mathqa        | EN+P1    | 232 | 63.8    | 45.6  | 72.8  | +27.3  | 63.9         | 72     |
| Qwen3-8B              | mathqa        | P1+P2    | 239 | 64.9    | 48.2  | 78.0  | +29.8  | 63.1         | 65     |
| Qwen3-8B              | mathqa        | EN+W1    | 302 | 57.9    | 59.5  | 71.8  | +12.3  | 65.6         | 64     |
| Qwen3-8B              | mathqa        | W1+W2    | 270 | 61.5    | 59.4  | 79.1  | +19.7  | 69.9         | 73     |
| Qwen3-8B              | mathqa        | EN+R     | 362 | 61.3    | 62.6  | 79.0  | +16.4  | 62.3         | 77     |
| Qwen3-8B              | mathqa        | EN+SR-EN | 39  | 56.4    | 66.7  | 68.8  | +2.1   | 75.0         | 8      |
| Qwen3-8B              | mathqa        | ZH+SR-ZH | 68  | 69.1    | 45.0  | 84.0  | +39.0  | 65.2         | 23     |
| Qwen3-8B              | truthfulqa    | EN+ZH    | 182 | 72.5    | 25.7  | 70.5  | +44.8  | 72.5         | 51     |
| Qwen3-8B              | truthfulqa    | EN+JA    | 214 | 74.3    | 38.6  | 85.5  | +47.0  | 77.1         | 48     |
| Qwen3-8B              | truthfulqa    | ZH+JA    | 204 | 76.0    | 29.9  | 81.8  | +52.0  | 73.5         | 49     |
| Qwen3-8B              | truthfulqa    | EN+S1    | 96  | 70.8    | 31.2  | 78.1  | +46.9  | 65.6         | 32     |
| Qwen3-8B              | truthfulqa    | S1+S2    | 109 | 79.8    | 29.4  | 81.6  | +52.2  | 86.5         | 37     |
| Qwen3-8B              | truthfulqa    | EN+P1    | 121 | 78.5    | 28.6  | 90.5  | +61.9  | 73.0         | 37     |
| Qwen3-8B              | truthfulqa    | P1+P2    | 116 | 83.6    | 20.9  | 90.9  | +70.0  | 79.3         | 29     |
| Qwen3-8B              | truthfulqa    | EN+W1    | 134 | 70.9    | 41.9  | 85.1  | +43.2  | 68.2         | 44     |
| Qwen3-8B              | truthfulqa    | W1+W2    | 111 | 76.6    | 35.3  | 88.9  | +53.6  | 75.6         | 41     |
| Qwen3-8B              | truthfulqa    | EN+R     | 117 | 70.9    | 42.9  | 83.8  | +40.9  | 73.7         | 38     |
| Qwen3-8B              | truthfulqa    | EN+SR-EN | 17  | 64.7    | 50.0  | 66.7  | +16.7  | 71.4         | 7      |
| Qwen3-8B              | truthfulqa    | ZH+SR-ZH | 33  | 72.7    | 40.0  | 72.2  | +32.2  | 100.0        | 5      |
| Qwen3-8B              | commonsenseqa | EN+ZH    | 572 | 74.3    | 45.5  | 90.1  | +44.6  | 87.5         | 104    |
| Qwen3-8B              | commonsenseqa | EN+JA    | 604 | 76.2    | 42.4  | 93.4  | +51.0  | 76.5         | 102    |
| Qwen3-8B              | commonsenseqa | ZH+JA    | 658 | 76.9    | 36.1  | 90.5  | +54.3  | 77.9         | 154    |
| Qwen3-8B              | commonsenseqa | EN+S1    | 173 | 85.5    | 15.4  | 86.9  | +71.5  | 85.1         | 47     |
| Qwen3-8B              | commonsenseqa | S1+S2    | 183 | 78.7    | 24.3  | 81.9  | +57.6  | 78.4         | 37     |
| Qwen3-8B              | commonsenseqa | EN+P1    | 198 | 87.9    | 12.5  | 91.7  | +79.2  | 81.6         | 38     |
| Qwen3-8B              | commonsenseqa | P1+P2    | 262 | 84.0    | 18.4  | 85.1  | +66.6  | 86.7         | 45     |
| Qwen3-8B              | commonsenseqa | EN+W1    | 389 | 81.5    | 24.6  | 90.0  | +65.4  | 79.7         | 74     |
| Qwen3-8B              | commonsenseqa | W1+W2    | 395 | 79.2    | 28.3  | 86.9  | +58.6  | 79.7         | 69     |
| Qwen3-8B              | commonsenseqa | EN+R     | 215 | 83.3    | 22.7  | 89.0  | +66.3  | 84.4         | 45     |
| Qwen3-8B              | commonsenseqa | EN+SR-EN | 11  | 90.9    | 0.0   | 100.0 | +100.0 | 66.7         | 3      |
| Qwen3-8B              | commonsenseqa | ZH+SR-ZH | 46  | 71.7    | 27.3  | 61.5  | +34.3  | 81.8         | 11     |
| DeepSeek V4.1 Flash   | mmlu          | EN+ZH    | 155 | 61.9    | 58.7  | 77.0  | +18.3  | 74.2         | 31     |
| DeepSeek V4.1 Flash   | mmlu          | EN+JA    | 187 | 61.0    | 63.1  | 84.6  | +21.5  | 68.0         | 25     |
| DeepSeek V4.1 Flash   | mmlu          | ZH+JA    | 182 | 56.0    | 79.4  | 79.3  | -0.1   | 70.4         | 27     |
| DeepSeek V4.1 Flash   | mmlu          | EN+S1    | 91  | 59.3    | 54.1  | 75.0  | +20.9  | 50.0         | 14     |
| DeepSeek V4.1 Flash   | mmlu          | S1+S2    | 91  | 60.4    | 53.8  | 75.6  | +21.8  | 54.5         | 11     |
| DeepSeek V4.1 Flash   | mmlu          | EN+P1    | 81  | 66.7    | 34.4  | 69.2  | +34.9  | 60.0         | 10     |
| DeepSeek V4.1 Flash   | mmlu          | P1+P2    | 95  | 58.9    | 43.1  | 68.8  | +25.6  | 41.7         | 12     |
| DeepSeek V4.1 Flash   | mmlu          | EN+W1    | 103 | 60.2    | 50.0  | 61.4  | +11.4  | 82.4         | 17     |
| DeepSeek V4.1 Flash   | mmlu          | W1+W2    | 84  | 59.5    | 54.3  | 77.8  | +23.5  | 46.2         | 13     |
| DeepSeek V4.1 Flash   | mmlu          | EN+R     | 89  | 62.9    | 56.0  | 70.8  | +14.8  | 68.8         | 16     |
| DeepSeek V4.1 Flash   | mmlu          | EN+SR-EN | 66  | 71.2    | 29.6  | 72.4  | +42.8  | 70.0         | 10     |
| DeepSeek V4.1 Flash   | mmlu          | ZH+SR-ZH | 65  | 63.1    | 38.2  | 65.0  | +26.8  | 63.6         | 11     |
| DeepSeek V4.1 Flash   | mathqa        | EN+ZH    | 74  | 55.4    | 72.2  | 90.5  | +18.3  | 70.6         | 17     |
| DeepSeek V4.1 Flash   | mathqa        | EN+JA    | 71  | 54.9    | 61.9  | 69.7  | +7.8   | 47.1         | 17     |
| DeepSeek V4.1 Flash   | mathqa        | ZH+JA    | 72  | 55.6    | 73.1  | 77.1  | +4.1   | 54.5         | 11     |
| DeepSeek V4.1 Flash   | mathqa        | EN+S1    | 46  | 54.3    | 55.6  | 64.3  | +8.7   | 57.1         | 14     |
| DeepSeek V4.1 Flash   | mathqa        | S1+S2    | 71  | 73.2    | 46.7  | 96.0  | +49.3  | 75.0         | 16     |
| DeepSeek V4.1 Flash   | mathqa        | EN+P1    | 48  | 56.2    | 55.0  | 60.0  | +5.0   | 69.2         | 13     |
| DeepSeek V4.1 Flash   | mathqa        | P1+P2    | 50  | 54.0    | 50.0  | 61.1  | +11.1  | 50.0         | 12     |
| DeepSeek V4.1 Flash   | mathqa        | EN+W1    | 55  | 47.3    | 63.6  | 55.0  | -8.6   | 53.8         | 13     |
| DeepSeek V4.1 Flash   | mathqa        | W1+W2    | 62  | 50.0    | 75.0  | 71.4  | -3.6   | 69.2         | 13     |
| DeepSeek V4.1 Flash   | mathqa        | EN+R     | 63  | 50.8    | 60.9  | 73.1  | +12.2  | 28.6         | 14     |
| DeepSeek V4.1 Flash   | mathqa        | EN+SR-EN | 18  | 77.8    | 0.0   | 83.3  | +83.3  | 50.0         | 6      |
| DeepSeek V4.1 Flash   | mathqa        | ZH+SR-ZH | 27  | 55.6    | 70.0  | 84.6  | +14.6  | 25.0         | 4      |
| DeepSeek V4.1 Flash   | truthfulqa    | EN+ZH    | 112 | 73.2    | 50.0  | 95.9  | +45.9  | 76.9         | 13     |
| DeepSeek V4.1 Flash   | truthfulqa    | EN+JA    | 120 | 56.7    | 75.9  | 88.4  | +12.4  | 73.9         | 23     |
| DeepSeek V4.1 Flash   | truthfulqa    | ZH+JA    | 99  | 68.7    | 64.9  | 95.1  | +30.3  | 76.2         | 21     |
| DeepSeek V4.1 Flash   | truthfulqa    | EN+S1    | 79  | 58.2    | 75.8  | 91.3  | +15.5  | 73.9         | 23     |
| DeepSeek V4.1 Flash   | truthfulqa    | S1+S2    | 83  | 61.4    | 65.8  | 92.6  | +26.8  | 72.2         | 18     |
| DeepSeek V4.1 Flash   | truthfulqa    | EN+P1    | 68  | 60.3    | 76.0  | 92.9  | +16.9  | 60.0         | 15     |
| DeepSeek V4.1 Flash   | truthfulqa    | P1+P2    | 73  | 63.0    | 64.3  | 82.4  | +18.1  | 72.7         | 11     |
| DeepSeek V4.1 Flash   | truthfulqa    | EN+W1    | 86  | 53.5    | 81.8  | 87.9  | +6.1   | 55.0         | 20     |
| DeepSeek V4.1 Flash   | truthfulqa    | W1+W2    | 83  | 56.6    | 69.7  | 83.3  | +13.6  | 60.0         | 20     |
| DeepSeek V4.1 Flash   | truthfulqa    | EN+R     | 70  | 72.9    | 80.0  | 100.0 | +20.0  | 87.5         | 24     |
| DeepSeek V4.1 Flash   | truthfulqa    | EN+SR-EN | 75  | 66.7    | 60.6  | 91.7  | +31.1  | 83.3         | 18     |
| DeepSeek V4.1 Flash   | truthfulqa    | ZH+SR-ZH | 39  | 61.5    | 68.4  | 92.9  | +24.4  | 83.3         | 6      |
| DeepSeek V4.1 Flash   | commonsenseqa | EN+ZH    | 415 | 64.6    | 57.1  | 82.5  | +25.5  | 71.0         | 62     |
| DeepSeek V4.1 Flash   | commonsenseqa | EN+JA    | 420 | 65.0    | 61.1  | 85.0  | +23.9  | 74.2         | 66     |
| DeepSeek V4.1 Flash   | commonsenseqa | ZH+JA    | 437 | 60.2    | 61.6  | 84.7  | +23.0  | 60.7         | 89     |
| DeepSeek V4.1 Flash   | commonsenseqa | EN+S1    | 193 | 72.0    | 42.3  | 81.2  | +38.9  | 83.3         | 30     |
| DeepSeek V4.1 Flash   | commonsenseqa | S1+S2    | 196 | 65.8    | 51.1  | 82.5  | +31.4  | 71.4         | 28     |
| DeepSeek V4.1 Flash   | commonsenseqa | EN+P1    | 172 | 63.4    | 52.9  | 75.3  | +22.5  | 72.4         | 29     |
| DeepSeek V4.1 Flash   | commonsenseqa | P1+P2    | 184 | 67.9    | 44.9  | 79.5  | +34.6  | 69.6         | 23     |
| DeepSeek V4.1 Flash   | commonsenseqa | EN+W1    | 303 | 65.0    | 52.5  | 78.5  | +26.0  | 70.8         | 48     |
| DeepSeek V4.1 Flash   | commonsenseqa | W1+W2    | 285 | 68.4    | 45.3  | 81.5  | +36.2  | 69.4         | 49     |
| DeepSeek V4.1 Flash   | commonsenseqa | EN+R     | 157 | 64.3    | 56.5  | 83.1  | +26.6  | 73.9         | 23     |
| DeepSeek V4.1 Flash   | commonsenseqa | EN+SR-EN | 201 | 74.6    | 34.1  | 85.0  | +50.9  | 72.2         | 36     |
| DeepSeek V4.1 Flash   | commonsenseqa | ZH+SR-ZH | 115 | 66.1    | 53.5  | 76.2  | +22.7  | 80.0         | 30     |
| Gemini 3.1 Flash-Lite | mmlu          | EN+ZH    | 144 | 54.9    | 70.5  | 75.0  | +4.5   | 68.4         | 19     |
| Gemini 3.1 Flash-Lite | mmlu          | EN+JA    | 139 | 58.3    | 60.3  | 75.9  | +15.5  | 60.9         | 23     |
| Gemini 3.1 Flash-Lite | mmlu          | ZH+JA    | 145 | 52.4    | 71.9  | 75.8  | +3.9   | 53.3         | 15     |
| Gemini 3.1 Flash-Lite | mmlu          | EN+S1    | 60  | 55.0    | 60.0  | 79.2  | +19.2  | 36.4         | 11     |
| Gemini 3.1 Flash-Lite | mmlu          | S1+S2    | 61  | 42.6    | 65.5  | 55.6  | -10.0  | 42.9         | 14     |
| Gemini 3.1 Flash-Lite | mmlu          | EN+P1    | 68  | 54.4    | 68.0  | 60.0  | -8.0   | 84.6         | 13     |
| Gemini 3.1 Flash-Lite | mmlu          | P1+P2    | 58  | 51.7    | 55.2  | 50.0  | -5.2   | 77.8         | 9      |
| Gemini 3.1 Flash-Lite | mmlu          | EN+W1    | 88  | 48.9    | 70.3  | 68.4  | -1.8   | 46.2         | 13     |
| Gemini 3.1 Flash-Lite | mmlu          | W1+W2    | 80  | 56.2    | 54.3  | 64.9  | +10.6  | 62.5         | 8      |
| Gemini 3.1 Flash-Lite | mmlu          | EN+R     | 63  | 49.2    | 60.0  | 56.5  | -3.5   | 53.3         | 15     |
| Gemini 3.1 Flash-Lite | mmlu          | EN+SR-EN | 20  | 50.0    | 80.0  | 53.8  | -26.2  | 100.0        | 2      |
| Gemini 3.1 Flash-Lite | mmlu          | ZH+SR-ZH | 32  | 50.0    | 61.5  | 66.7  | +5.1   | 25.0         | 4      |
| Gemini 3.1 Flash-Lite | mathqa        | EN+ZH    | 83  | 39.8    | 81.2  | 60.6  | -20.6  | 38.9         | 18     |
| Gemini 3.1 Flash-Lite | mathqa        | EN+JA    | 107 | 50.5    | 70.0  | 63.6  | -6.4   | 60.9         | 23     |
| Gemini 3.1 Flash-Lite | mathqa        | ZH+JA    | 113 | 48.7    | 72.2  | 65.2  | -7.0   | 48.4         | 31     |
| Gemini 3.1 Flash-Lite | mathqa        | EN+S1    | 68  | 51.5    | 40.6  | 47.6  | +7.0   | 40.0         | 15     |
| Gemini 3.1 Flash-Lite | mathqa        | S1+S2    | 73  | 45.2    | 64.9  | 57.1  | -7.7   | 53.3         | 15     |
| Gemini 3.1 Flash-Lite | mathqa        | EN+P1    | 69  | 44.9    | 79.3  | 66.7  | -12.6  | 56.2         | 16     |
| Gemini 3.1 Flash-Lite | mathqa        | P1+P2    | 73  | 42.5    | 80.0  | 81.8  | +1.8   | 30.8         | 26     |
| Gemini 3.1 Flash-Lite | mathqa        | EN+W1    | 67  | 53.7    | 61.5  | 63.0  | +1.4   | 64.3         | 14     |
| Gemini 3.1 Flash-Lite | mathqa        | W1+W2    | 67  | 58.2    | 65.2  | 73.1  | +7.9   | 66.7         | 18     |
| Gemini 3.1 Flash-Lite | mathqa        | EN+R     | 86  | 50.0    | 72.4  | 64.1  | -8.3   | 55.6         | 18     |
| Gemini 3.1 Flash-Lite | mathqa        | EN+SR-EN | 17  | 47.1    | 75.0  | 71.4  | -3.6   | 50.0         | 2      |
| Gemini 3.1 Flash-Lite | mathqa        | ZH+SR-ZH | 33  | 60.6    | 60.0  | 100.0 | +40.0  | 58.8         | 17     |
| Gemini 3.1 Flash-Lite | truthfulqa    | EN+ZH    | 68  | 38.2    | 77.4  | 62.5  | -14.9  | 30.8         | 13     |
| Gemini 3.1 Flash-Lite | truthfulqa    | EN+JA    | 73  | 54.8    | 70.4  | 79.3  | +8.9   | 52.9         | 17     |
| Gemini 3.1 Flash-Lite | truthfulqa    | ZH+JA    | 82  | 46.3    | 82.9  | 78.1  | -4.7   | 46.7         | 15     |
| Gemini 3.1 Flash-Lite | truthfulqa    | EN+S1    | 35  | 54.3    | 90.0  | 71.4  | -18.6  | 75.0         | 4      |
| Gemini 3.1 Flash-Lite | truthfulqa    | S1+S2    | 34  | 58.8    | 57.1  | 78.6  | +21.4  | 50.0         | 6      |
| Gemini 3.1 Flash-Lite | truthfulqa    | EN+P1    | 45  | 35.6    | 94.4  | 70.6  | -23.9  | 30.0         | 10     |
| Gemini 3.1 Flash-Lite | truthfulqa    | P1+P2    | 45  | 48.9    | 75.0  | 68.4  | -6.6   | 66.7         | 6      |
| Gemini 3.1 Flash-Lite | truthfulqa    | EN+W1    | 50  | 40.0    | 83.3  | 76.5  | -6.9   | 33.3         | 9      |
| Gemini 3.1 Flash-Lite | truthfulqa    | W1+W2    | 45  | 48.9    | 73.9  | 76.5  | +2.6   | 60.0         | 5      |
| Gemini 3.1 Flash-Lite | truthfulqa    | EN+R     | 38  | 39.5    | 85.7  | 71.4  | -14.3  | 30.0         | 10     |
| Gemini 3.1 Flash-Lite | truthfulqa    | EN+SR-EN | 23  | 52.2    | 90.0  | 88.9  | -1.1   | 75.0         | 4      |
| Gemini 3.1 Flash-Lite | truthfulqa    | ZH+SR-ZH | 20  | 50.0    | 83.3  | 58.3  | -25.0  | 100.0        | 2      |
| Gemini 3.1 Flash-Lite | commonsenseqa | EN+ZH    | 369 | 53.9    | 70.0  | 78.0  | +8.0   | 54.0         | 50     |
| Gemini 3.1 Flash-Lite | commonsenseqa | EN+JA    | 366 | 51.9    | 74.7  | 77.4  | +2.7   | 52.8         | 53     |
| Gemini 3.1 Flash-Lite | commonsenseqa | ZH+JA    | 387 | 49.9    | 74.5  | 73.1  | -1.4   | 46.5         | 71     |
| Gemini 3.1 Flash-Lite | commonsenseqa | EN+S1    | 92  | 53.3    | 54.8  | 63.2  | +8.4   | 50.0         | 12     |
| Gemini 3.1 Flash-Lite | commonsenseqa | S1+S2    | 99  | 49.5    | 58.1  | 63.4  | +5.3   | 33.3         | 15     |
| Gemini 3.1 Flash-Lite | commonsenseqa | EN+P1    | 153 | 54.2    | 56.6  | 77.8  | +21.2  | 34.8         | 23     |
| Gemini 3.1 Flash-Lite | commonsenseqa | P1+P2    | 172 | 54.7    | 53.2  | 70.1  | +16.9  | 39.3         | 28     |
| Gemini 3.1 Flash-Lite | commonsenseqa | EN+W1    | 271 | 52.0    | 68.6  | 73.8  | +5.2   | 55.8         | 43     |
| Gemini 3.1 Flash-Lite | commonsenseqa | W1+W2    | 241 | 53.1    | 73.9  | 77.1  | +3.2   | 50.0         | 40     |
| Gemini 3.1 Flash-Lite | commonsenseqa | EN+R     | 117 | 54.7    | 53.7  | 60.4  | +6.7   | 66.7         | 15     |
| Gemini 3.1 Flash-Lite | commonsenseqa | EN+SR-EN | 86  | 62.8    | 48.5  | 68.8  | +20.3  | 80.0         | 5      |
| Gemini 3.1 Flash-Lite | commonsenseqa | ZH+SR-ZH | 90  | 51.1    | 69.0  | 71.8  | +2.8   | 40.9         | 22     |

## (5) 交叉實驗的結果（只報告）

RQ2 交叉實驗中裁判 ≠ 候選模型的 144 格（3 組配對 EN+ZH、EN+S1、P1+P2 × 4 個資料集 × 12 種組合），題目的條件同第 (3) 節。位置 1 的順序沿用主網格的紀錄：檔名第一條 path 排在位置 1 的比例合併為 50.1%，3 格不在 45%–55% 之間。「三者合併」那一列的選位置 2 附 95% Clopper–Pearson 區間；其餘為 %，(c) 為 pp。

| 裁判                    | 候選模型                  | 題數   | 選位置 2                         | (a)   | (b)   | (c)（pp）   | 兩個都錯時選位置 2   | 排除   | 對應不上   |
|:----------------------|:----------------------|:-----|:------------------------------|:------|:------|:----------|:-------------|:-----|:-------|
| GPT-4o mini           | Qwen3-8B              | 3094 | 52.7                          | 64.8  | 68.6  | +3.8      | 55.0         | 0    | 0      |
| GPT-4o mini           | DeepSeek V4.1 Flash   | 1567 | 59.9                          | 50.0  | 68.4  | +18.4     | 64.1         | 0    | 0      |
| GPT-4o mini           | Gemini 3.1 Flash-Lite | 1267 | 56.7                          | 49.6  | 64.5  | +14.8     | 54.5         | 0    | 0      |
| GPT-4o mini           | 三者合併                  | 5928 | 55.4% [54.2, 56.7]（3287/5928） | 57.2  | 67.6  | +10.4     | 56.9         | 0    | 0      |
| Qwen3-8B              | GPT-4o mini           | 2767 | 72.0                          | 45.5  | 87.3  | +41.8     | 77.6         | 2    | 0      |
| Qwen3-8B              | DeepSeek V4.1 Flash   | 1550 | 80.0                          | 29.0  | 90.4  | +61.5     | 76.9         | 17   | 0      |
| Qwen3-8B              | Gemini 3.1 Flash-Lite | 1267 | 78.1                          | 29.2  | 85.0  | +55.7     | 80.6         | 0    | 0      |
| Qwen3-8B              | 三者合併                  | 5584 | 75.6% [74.4, 76.7]（4221/5584） | 37.0  | 87.7  | +50.6     | 78.0         | 19   | 0      |
| DeepSeek V4.1 Flash   | GPT-4o mini           | 2758 | 58.4                          | 71.1  | 87.9  | +16.8     | 61.4         | 11   | 0      |
| DeepSeek V4.1 Flash   | Qwen3-8B              | 3077 | 59.2                          | 71.3  | 87.8  | +16.4     | 62.4         | 17   | 0      |
| DeepSeek V4.1 Flash   | Gemini 3.1 Flash-Lite | 1265 | 63.2                          | 53.9  | 79.6  | +25.7     | 67.8         | 2    | 0      |
| DeepSeek V4.1 Flash   | 三者合併                  | 7100 | 59.6% [58.5, 60.7]（4232/7100） | 67.9  | 86.3  | +18.4     | 62.8         | 30   | 0      |
| Gemini 3.1 Flash-Lite | GPT-4o mini           | 2769 | 49.8                          | 85.7  | 87.3  | +1.7      | 49.8         | 0    | 0      |
| Gemini 3.1 Flash-Lite | Qwen3-8B              | 3093 | 48.6                          | 86.2  | 83.9  | -2.3      | 47.4         | 1    | 0      |
| Gemini 3.1 Flash-Lite | DeepSeek V4.1 Flash   | 1567 | 51.5                          | 72.4  | 76.2  | +3.8      | 51.1         | 0    | 0      |
| Gemini 3.1 Flash-Lite | 三者合併                  | 7429 | 49.6% [48.5, 50.8]（3687/7429） | 82.9  | 83.4  | +0.5      | 48.9         | 1    | 0      |
