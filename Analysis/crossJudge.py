import os
from collections import Counter

import numpy as np
import pandas as pd

from Arm.ArmSpec import ArmSpec
from Arm.GenerationRecord import GenerationRecord
from Aggregator.JudgeAggregator import JudgeAggregator
from Analysis import preregistration
from Analysis.alignment import CellData, PairArrays, alignPair, loadRecords
from Analysis.experimentPlan import PATHS, Pair
from Analysis.metrics import VALUE_COLUMNS, subsetMask, computeRow
from Runner.paths import ACTIVE_DATASETS, aggregationPath, crossJudgePath

# ------------------------------------------------------------------
# RQ2 交叉實驗（result/analysis/rq2/rq2_criteria.md）：候選答案固定，只換 Judge 模型
#   計畫常數、每格的題目集合、檔案位置、逐格的陣列與數字、兩個執行前檢查的評估
#   統計與判定在 Analysis/crossJudgeStats.py
# ------------------------------------------------------------------
MODELS = ["gpt4omini", "qwen", "deepseek4.1flash", "gemini3.1flashlite"]
STRONG = ["deepseek4.1flash", "gemini3.1flashlite"]
WEAK = ["gpt4omini", "qwen"]
DATASETS = list(ACTIVE_DATASETS)
PAIRS = [Pair(PATHS["EN"], PATHS["ZH"]), Pair(PATHS["EN"], PATHS["S1"]), Pair(PATHS["P1"], PATHS["P2"])]
PAIR_BY_LABEL = {pair.label: pair for pair in PAIRS}
NUMS = 2000           # 主網格的 --nums（arm 檔與 Judge 檔都用它）
SEED, REPS = 0, 200   # split-half，與 aggregation_cells.csv 相同

# §4 執行前檢查
CHECK_DATASET, CHECK_PAIR = "mmlu", PAIR_BY_LABEL["EN+S1"]
AGREEMENT_MIN = 0.95
PILOT_N, PILOT_SEED = 100, 0
PILOT_MAX_RATE, PILOT_MAX_COST = 0.01, 20.0
# 美元 / 百萬 tokens（輸入, 輸出）；DeepSeek 以尖峰價估
PRICES = {"gpt4omini": (0.15, 0.60), "qwen": (0.18, 0.70), "deepseek4.1flash": (0.30, 1.20), "gemini3.1flashlite": (0.25, 1.50)}
# 各模型當 Judge 時的解碼與 thinking 設定（Model/*.py；Judge 呼叫一律 T=0、不傳 seed）
JUDGE_SETTINGS = {
    "gpt4omini": "T=0，max_tokens 8192，不傳 seed；沒有 thinking 設定",
    "qwen": "T=0，max_tokens 8192，不傳 seed；enable_thinking False（DashScope 國際站）",
    "deepseek4.1flash": "T=0，max_tokens 8192，不傳 seed；thinking disabled（官方 API deepseek-flash）",
    "gemini3.1flashlite": "T=0，max_tokens 8192，不傳 seed；thinking_level minimal（不保證完全不思考）",
}

# 輸出位置（腳本可用參數覆寫）
OUT_DIR = "result/analysis/rq2"
CRITERIA_FILE = "rq2_criteria.md"
JUDGE_DIR, PRECHECK_DIR = "judge_outputs", "precheck"
PRECHECK_FILE, PILOT_FILE = "precheck.json", "pilot.json"

SOURCE_MAIN, SOURCE_CROSS, SOURCE_RERUN = "main_grid", "cross", "cross_rerun"
CELL_ID_COLUMNS = ["generator", "judge", "dataset", "pair", "arm_a", "arm_b", "source", "used_in_analysis"]
CELL_EXTRA_COLUMNS = ["n_R", "pick_right", "tok_out_per_call", "n_invalid_choice", "n_refused", "anchor_first"]
ITEM_COLUMNS = ["generator", "judge", "dataset", "pair", "arm_a", "arm_b", "item_id", "source", "gold", "answer_a", "answer_b",
                "correct_a", "correct_b", "presentation_order", "first_shown", "choice", "chosen_arm", "final_answer",
                "correct", "off_menu", "refused", "judge_output", "tokens_out", "model_version", "called_at"]


# ------------------------------------------------------------------
# 題目集合與檔案位置
# ------------------------------------------------------------------
def judgedItems(cell: CellData, pair: Pair) -> list[int]:
    """兩條 path 都有解析出答案、且答案不同的題目（四個 Judge 共用）。"""
    arrays = alignPair(cell, pair)
    mask = arrays.dis & arrays.answered_a & arrays.answered_b
    return [int(item_id) for item_id in arrays.item_ids[mask]]


def itemPools(armdir: str, aggdir: str, generators: list[str] = MODELS) -> dict[tuple[str, str, str], list[int]]:
    """{(generator, dataset, pair label): 要裁決的題目}"""
    pools = {}
    for generator in generators:
        for dataset in DATASETS:
            cell = CellData(armdir, aggdir, generator, dataset)
            for pair in PAIRS:
                pools[(generator, dataset, pair.label)] = judgedItems(cell, pair)
    return pools


def pilotSample(pools: dict, judge: str) -> dict[tuple[str, str, str], set[int]]:
    """§4.2：從該 Judge 要裁決的題目池（另外 3 個 generator × 資料集 × 配對）均勻抽 PILOT_N 題，seed PILOT_SEED。"""
    pool = [(generator, dataset, pair.label, item_id)
            for generator in MODELS if generator != judge
            for dataset in DATASETS for pair in PAIRS
            for item_id in pools[(generator, dataset, pair.label)]]
    picked = np.random.default_rng(PILOT_SEED).choice(len(pool), size=min(PILOT_N, len(pool)), replace=False)
    sample = {}
    for index in sorted(picked):
        generator, dataset, label, item_id = pool[index]
        sample.setdefault((generator, dataset, label), set()).add(item_id)
    return sample


def mainGridPath(aggdir: str, generator: str, dataset: str, pair: Pair) -> str:
    return aggregationPath(aggdir, generator, dataset, "judge", [ArmSpec.from_arm_id(arm_id) for arm_id in pair.arms])


def judgeOutputPath(outdir: str, judge: str, generator: str, dataset: str, pair: Pair) -> str:
    return crossJudgePath(outdir, judge, generator, dataset, [ArmSpec.from_arm_id(arm_id) for arm_id in pair.arms])


def diagonalRerunCalls(pools: dict, model: str) -> int:
    """§4.1 重跑對角線時新增的呼叫數（mmlu × EN+S1 沿用流程核對的結果）。"""
    return sum(len(pools[(model, d, p.label)]) for d in DATASETS for p in PAIRS) - len(pools[(model, CHECK_DATASET, CHECK_PAIR.label)])


def loadStepFile(path: str, step: str, criteria_sha256: str) -> dict:
    """precheck.json / pilot.json；檔案不存在或寫於另一份判定標準下就停。"""
    return preregistration.loadStepFile(path, step, criteria_sha256, "scripts/analysis_rq2/run_cross_judge.py")


# ------------------------------------------------------------------
# 逐題分類
# ------------------------------------------------------------------
def classifyRecord(record: dict) -> dict:
    """一筆 Judge 紀錄：被擋下 / 找不到 {"choice"} / 編號超出範圍 / off-menu。"""
    trace = record.get("trace") or {}
    refused = bool(trace.get("refused_calls"))
    matches = JudgeAggregator.CHOICE_PATTERN.findall(trace.get("judge_output") or "")
    return {
        "refused": refused,
        "no_choice": not refused and not matches,
        "out_of_range": not refused and bool(matches) and trace.get("choice") is None,
        "off_menu": bool(record["off_menu"]),
    }


def chosenArm(record: dict) -> str | None:
    """選中的 path；無效選擇或被擋下時為 None。"""
    return (record.get("trace") or {}).get("chosen_arm")


def callCost(judge: str, call: dict | None) -> float | None:
    if not call or call.get("usage_in") is None or call.get("usage_out") is None:
        return None
    price_in, price_out = PRICES[judge]
    return (call["usage_in"] * price_in + call["usage_out"] * price_out) / 1e6


# ------------------------------------------------------------------
# §4 執行前檢查的評估
# ------------------------------------------------------------------
def evaluatePrecheck(precheck_dir: str, armdir: str, aggdir: str, models: list[str] = MODELS) -> dict:
    """§4.1：每個模型 mmlu × EN+S1 自己裁決自己的重跑，逐題和主網格的 chosen_arm 比對。"""
    result = {}
    for model in models:
        judged = judgedItems(CellData(armdir, aggdir, model, CHECK_DATASET), CHECK_PAIR)
        path = judgeOutputPath(precheck_dir, model, model, CHECK_DATASET, CHECK_PAIR)
        records = loadRecords(path)[1] if os.path.exists(path) else {}
        _, main = loadRecords(mainGridPath(aggdir, model, CHECK_DATASET, CHECK_PAIR))
        agree = sum(chosenArm(records[i]) == chosenArm(main[i]) for i in judged if i in records)
        costs = [callCost(model, record.get("call")) for record in records.values()]
        result[model] = {
            "file": path, "n": len(judged), "n_done": len(records), "complete": set(records) == set(judged),
            "n_agree": agree, "agreement": agree / len(judged) if judged else float("nan"),
            "rerun_diagonal": bool(judged) and agree / len(judged) < AGREEMENT_MIN,
            "cost_usd": sum(c for c in costs if c is not None),
        }
    return result


def evaluatePilot(judge_dir: str, pools: dict, precheck: dict, models: list[str] = MODELS) -> dict:
    """§4.2：每個 Judge 的抽樣題目上的比例、每次呼叫的 token 與成本，以及全量成本估算。"""
    judges, total = {}, 0.0
    for judge in models:
        counts, calls, n_done = Counter(), [], 0
        sample = pilotSample(pools, judge)
        for (generator, dataset, label), items in sample.items():
            path = judgeOutputPath(judge_dir, judge, generator, dataset, PAIR_BY_LABEL[label])
            records = loadRecords(path)[1] if os.path.exists(path) else {}
            for item_id in items:
                if item_id not in records:
                    continue
                n_done += 1
                record = records[item_id]
                counts.update(key for key, flag in classifyRecord(record).items() if flag)
                calls.append({"cost": callCost(judge, record.get("call")), "tokens_out": record["tokens_out"],
                              **{key: (record.get("call") or {}).get(key) for key in ("usage_in", "usage_out")}})
        n = sum(len(items) for items in sample.values())
        with_usage = [c for c in calls if c["cost"] is not None]
        n_calls = sum(len(pools[(g, d, p.label)]) for g in MODELS if g != judge for d in DATASETS for p in PAIRS)
        if precheck.get(judge, {}).get("rerun_diagonal"):
            n_calls += diagonalRerunCalls(pools, judge)
        cost_per_call = float(np.mean([c["cost"] for c in with_usage])) if with_usage else float("nan")
        rates = {key: counts[key] / n_done if n_done else float("nan") for key in ("no_choice", "out_of_range", "refused", "off_menu")}
        judges[judge] = {
            "n": n, "n_done": n_done, "complete": n_done == n, "calls_without_usage": len(calls) - len(with_usage),
            **{f"n_{key}": counts[key] for key in rates}, **{f"rate_{key}": value for key, value in rates.items()},
            "api_in_per_call": float(np.mean([c["usage_in"] for c in with_usage])) if with_usage else float("nan"),
            "api_out_per_call": float(np.mean([c["usage_out"] for c in with_usage])) if with_usage else float("nan"),
            "tokens_out_per_call": float(np.mean([c["tokens_out"] for c in calls])) if calls else float("nan"),
            "cost_per_call_usd": cost_per_call, "full_calls": n_calls, "full_cost_usd": cost_per_call * n_calls,
            "rate_ok": all(value <= PILOT_MAX_RATE for value in rates.values()),
        }
        total += cost_per_call * n_calls
    complete = all(j["complete"] and j["calls_without_usage"] == 0 for j in judges.values())
    return {
        "judges": judges, "full_cost_usd": total, "complete": complete,
        "passed": complete and all(j["rate_ok"] for j in judges.values()) and total <= PILOT_MAX_COST,
    }


# ------------------------------------------------------------------
# 逐格的陣列與數字
# ------------------------------------------------------------------
def crossArrays(cell: CellData, pair: Pair, records: dict) -> PairArrays:
    """
    both_answered 子集的逐題陣列，最終答案取自 records（只含兩條 path 都有答案的分歧題）。
    一致題上 Judge 不動作：最終答案 = 兩條 path 共同的答案、off_menu False、tokens 0（與主網格的 no-op 紀錄相同）。
    records 必須剛好涵蓋該子集的全部分歧題，否則報錯。
    """
    arrays = alignPair(cell, pair)
    arrays = arrays.subset(subsetMask(arrays, "both_answered"))
    dis_ids = {int(item_id) for item_id in arrays.item_ids[arrays.dis]}
    if set(records) != dis_ids:
        raise ValueError(f"{cell.model_name} | {cell.dataset_name} | {pair.label}: the judge records cover {len(records)} items, "
                         f"expected the {len(dis_ids)} both-answered disagreements "
                         f"(missing {len(dis_ids - set(records))}, extra {len(set(records) - dis_ids)})")
    recordsA = cell.arm(pair.arm_a)
    finals, off_menu, tokens = [], [], []
    for item_id, dis in zip(arrays.item_ids, arrays.dis):
        record = records.get(int(item_id)) if dis else None
        finals.append(record["final_answer"] if dis else recordsA[int(item_id)]["parsed_answer"])
        off_menu.append(bool(record["off_menu"]) if dis else False)
        tokens.append(record["tokens_out"] if dis else 0)
    golds = [recordsA[int(item_id)]["gold"] for item_id in arrays.item_ids]
    arrays.final_correct = np.array([cell.compare(gold, final) for gold, final in zip(golds, finals)], dtype=bool)
    arrays.final_answered = np.array([GenerationRecord.isParseOk(final) for final in finals], dtype=bool)
    arrays.off_menu = np.array(off_menu, dtype=bool)
    arrays.tokens_out_agg = np.array(tokens, dtype=float)
    return arrays


def cellRow(arrays: PairArrays, pair: Pair, records: dict) -> dict:
    """computeRow（aggregation_cells.csv 的欄位與算法）加上 §3 的其他量。records = 該格分歧題的 Judge 紀錄。"""
    row = computeRow(arrays, pair, "judge", SEED, REPS)
    rescuable = arrays.dis & (arrays.correct_a ^ arrays.correct_b)
    flags = [classifyRecord(record) for record in records.values()]
    orders = [record.get("presentation_order") for record in records.values()]
    row.update({
        "n_R": int(rescuable.sum()),
        "pick_right": float(arrays.final_correct[rescuable].mean()) if rescuable.any() else float("nan"),
        "tok_out_per_call": float(arrays.tokens_out_agg[arrays.dis].mean()) if arrays.dis.any() else float("nan"),
        "n_invalid_choice": sum(f["no_choice"] or f["out_of_range"] for f in flags),
        "n_refused": sum(f["refused"] for f in flags),
        "anchor_first": float(np.mean([order[0] == pair.arm_a for order in orders])) if orders else float("nan"),
    })
    return row


def itemRows(cell: CellData, pair: Pair, judge: str, records: dict, source: str) -> list[dict]:
    """judge_outputs/items.csv.gz 的列（只有被裁決的題目）。"""
    recordsA, recordsB = cell.arm(pair.arm_a), cell.arm(pair.arm_b)
    rows = []
    for item_id in sorted(records):
        record, a, b = records[item_id], recordsA[item_id], recordsB[item_id]
        trace, call = record.get("trace") or {}, record.get("call") or {}
        order = record.get("presentation_order") or []
        rows.append({
            "generator": cell.model_name, "judge": judge, "dataset": cell.dataset_name, "pair": pair.label,
            "arm_a": pair.arm_a, "arm_b": pair.arm_b, "item_id": item_id, "source": source, "gold": a["gold"],
            "answer_a": a["parsed_answer"], "answer_b": b["parsed_answer"],
            "correct_a": cell.compare(a["gold"], a["parsed_answer"]), "correct_b": cell.compare(b["gold"], b["parsed_answer"]),
            "presentation_order": "|".join(order), "first_shown": order[0] if order else None,
            "choice": trace.get("choice"), "chosen_arm": trace.get("chosen_arm"), "final_answer": record["final_answer"],
            "correct": cell.compare(a["gold"], record["final_answer"]), "off_menu": record["off_menu"],
            "refused": bool(trace.get("refused_calls")), "judge_output": trace.get("judge_output"),
            "tokens_out": record["tokens_out"], "model_version": call.get("model_version"), "called_at": call.get("called_at"),
        })
    return rows


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def compareRows(found: dict, expected: dict, columns: list[str], where: str) -> list[str]:
    return [f"{where}: {column} = {found[column]!r}, expected {expected[column]!r}"
            for column in columns if not _same(found[column], expected[column])]


def buildCells(armdir: str, aggdir: str, judge_dir: str, rerun_models: set[str], cells_csv: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    192 格（generator × Judge × 資料集 × 配對）的 §3 數字，與逐題列。回傳 (cells, items, problems)；problems 非空就該停。
      - 對角線、沿用舊檔：主網格 Judge 檔。所有 VALUE_COLUMNS 必須和 aggregation_cells.csv 的 both_answered 列完全相同，
        而且同一批紀錄走交叉流程（crossArrays）也要得到同一列。
      - 對角線、依 §4.1 重跑：judge_dir 的重跑檔；舊檔的列另外保留（used_in_analysis False）。
      - 非對角線：judge_dir 的交叉檔。每題的呈現順序必須等於主網格紀錄的順序。
    """
    table = pd.read_csv(cells_csv, float_precision="round_trip")   # 逐位比對需要精確讀回
    cells, items, problems = [], [], []
    for generator in MODELS:
        for dataset in DATASETS:
            cell = CellData(armdir, aggdir, generator, dataset)
            for pair in PAIRS:
                judged = judgedItems(cell, pair)
                main_path = mainGridPath(aggdir, generator, dataset, pair)
                _, main_all = loadRecords(main_path)
                main = {item_id: main_all[item_id] for item_id in judged}
                ids = {"generator": generator, "dataset": dataset, "pair": pair.label, "arm_a": pair.arm_a, "arm_b": pair.arm_b}

                full = alignPair(cell, pair, main_path)
                main_row = cellRow(full.subset(subsetMask(full, "both_answered")), pair, main)
                if generator not in rerun_models:
                    where = f"{generator} | {dataset} | {pair.label} | diagonal"
                    csv_rows = table[(table["model"] == generator) & (table["dataset"] == dataset) & (table["pair"] == pair.label)
                                     & (table["aggregator"] == "judge") & (table["subset"] == "both_answered")]
                    if len(csv_rows) != 1:
                        problems.append(f"{where}: {len(csv_rows)} rows in {cells_csv}")
                    else:
                        problems += compareRows(main_row, csv_rows.iloc[0].to_dict(), VALUE_COLUMNS, f"{where} vs {cells_csv}")
                    via_cross = cellRow(crossArrays(cell, pair, main), pair, main)
                    problems += compareRows(via_cross, main_row, list(main_row), f"{where} via the cross path")

                for judge in MODELS:
                    if judge == generator and generator not in rerun_models:
                        source, records, row = SOURCE_MAIN, main, main_row
                    else:
                        source = SOURCE_RERUN if judge == generator else SOURCE_CROSS
                        path = judgeOutputPath(judge_dir, judge, generator, dataset, pair)
                        if not os.path.exists(path):
                            problems.append(f"missing {path}")
                            continue
                        metadata, records = loadRecords(path)
                        where = f"{path}"
                        if (metadata.get("generator"), metadata.get("judge")) != (generator, judge):
                            problems.append(f"{where}: generator / judge {(metadata.get('generator'), metadata.get('judge'))}")
                            continue
                        changed = [i for i, r in records.items() if r.get("presentation_order") != main_all[i].get("presentation_order")]
                        if changed:
                            problems.append(f"{where}: {len(changed)} items differ from the recorded presentation order")
                        try:
                            row = cellRow(crossArrays(cell, pair, records), pair, records)
                        except ValueError as e:
                            problems.append(str(e))
                            continue
                    cells.append({**ids, "judge": judge, "source": source, "used_in_analysis": True, **row})
                    items += itemRows(cell, pair, judge, records, source)
                if generator in rerun_models:
                    cells.append({**ids, "judge": generator, "source": SOURCE_MAIN, "used_in_analysis": False, **main_row})

    cells = pd.DataFrame(cells, columns=CELL_ID_COLUMNS + VALUE_COLUMNS + CELL_EXTRA_COLUMNS)
    used = cells[cells["used_in_analysis"]]
    undefined = used[used["recovery_H2"].isna()]
    problems += [f"{r.generator} → judge {r.judge} | {r.dataset} | {r.pair}: recovery_H2 undefined" for r in undefined.itertuples()]
    if len(used) != len(MODELS) ** 2 * len(DATASETS) * len(PAIRS) and not problems:
        problems.append(f"{len(used)} cells, expected {len(MODELS) ** 2 * len(DATASETS) * len(PAIRS)}")
    return cells, pd.DataFrame(items, columns=ITEM_COLUMNS), problems
