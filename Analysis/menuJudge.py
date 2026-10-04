import glob
import json
import os
from collections import Counter

import numpy as np

from Arm.ArmSpec import ArmSpec
from Arm.PromptBuilder import PromptBuilder
from Analysis.alignment import CellData, loadRecords
from Analysis.crossJudge import PRICES, JUDGE_SETTINGS, PAIR_BY_LABEL, classifyRecord, callCost, mainGridPath
from Analysis.experimentPlan import PATHS
from Analysis.menuVote import MODELS, DATASETS, MENUS as VOTE_MENUS, COMPARE_PATHS
from Runner.paths import armPath, menuJudgePath

# ------------------------------------------------------------------
# RQ1-KJ（result/analysis/rq1kj/rq1kj_criteria.md）：K 條 path 的 Judge 版本 vs 單一最強 path
#   計畫常數、每格要裁決的題目、檔案位置、§6 四項開跑前檢查的評估
#   逐切分的量、判定與 §7 的報告量在 Analysis/menuJudgeStats.py
# ------------------------------------------------------------------
MENU_NAMES = ["M12", "M3L", "M3S", "M3P"]
MENUS = {menu: VOTE_MENUS[menu] for menu in MENU_NAMES}   # 路徑短代號，順序 = §2.2 的基準順序（§3.3 用）
MAIN_MENU = "M12"
COMPARE_MENUS = ("M3S", "M3L")                              # 判定三：M3S 的 A_J − M3L 的 A_J
TWO_PATH = {"M3L": "EN+ZH", "M3S": "EN+S1", "M3P": "P1+P2"}  # §7.7 對照的兩條 path Judge（主網格）
THRESHOLD = 0.5                    # 百分點（§5）
SEED, REPS = 0, 200                # 切分（§4.1），與 RQ1、RQ1-K 相同
TOLERANCE = 1e-9                   # §6.1 重現與 §7.5 恆等式
NUMS = 2000                        # 主網格的 --nums
AGGREGATOR_SEED = 0                # §3.3 候選順序的 seed（也是主網格 Judge 檔的 seed）
ARM_TO_CODE = {arm_id: code for code, arm_id in PATHS.items()}

# §6 開跑前檢查
CHECK_DATASET, CHECK_MENU, CHECK_CODES = "mmlu", "EN+S1", ["EN", "S1"]
RQ2_AGREEMENT = {"qwen": (191, 194), "gemini3.1flashlite": (55, 60), "gpt4omini": (159, 178), "deepseek4.1flash": (73, 91)}
MAX_DROP = 0.10                    # §6.3：比 RQ2 核對時低 10 個百分點以上（含）就停
PILOT_PER_DATASET, PILOT_SEED = 25, 0
PILOT_MAX_INVALID = 0.01           # §6.4：第一次的無效比例（解析失敗 + 超出範圍 + 被擋下）> 1% 就停
MAX_COST = 30.0                    # 美元，全量四份菜單合計（含試跑第二次已花的）
MAX_TOKENS = 8192                  # 四個模型當 Judge 時的 max_tokens（§2.3）
# 官方文件的上限（2026-10-04 查閱）。超過 = input + MAX_TOKENS > context，或 input > max_input（有公布時）
CONTEXT_LIMITS = {
    "gpt4omini": {"context": 128_000, "max_input": None,
                  "source": "https://developers.openai.com/api/docs/models/gpt-4o-mini (128,000 context window; 16,384 max output)"},
    "qwen": {"context": 131_072, "max_input": 98_304,
             "source": "https://www.alibabacloud.com/help/en/model-studio/qwen3-8b (context 131,072; max input 98,304; max output 8,192)"},
    "deepseek4.1flash": {"context": 1_000_000, "max_input": None,
                         "source": "https://api-docs.deepseek.com/quick_start/pricing (deepseek-flash: context length 1M)"},
    "gemini3.1flashlite": {"context": 1_048_576, "max_input": 1_048_576,
                           "source": "https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite (input 1,048,576; output 65,536)"},
}

# 輸出位置（腳本可用參數覆寫）
OUT_DIR = "result/analysis/rq1kj"
CRITERIA_FILE = "rq1kj_criteria.md"
JUDGE_DIR, PRECHECK_DIR, REP2_DIR = "judge_outputs", "precheck", "pilot_rep2"
REPRODUCE_FILE, TEMPLATE_FILE, PRECHECK_FILE, PILOT_FILE = "reproduce.json", "template_check.json", "precheck.json", "pilot.json"
RUNNER = "scripts/analysis_rq1kj/run_menu_judge.py"


# ------------------------------------------------------------------
# 菜單與題目
# ------------------------------------------------------------------
def codesOf(menu: str) -> list[str]:
    """菜單的 path 短代號（§2.2 的基準順序）；CHECK_MENU 是 §6.3 的 K = 2 菜單。"""
    return CHECK_CODES if menu == CHECK_MENU else MENUS[menu]


def armIdsOf(menu: str) -> list[str]:
    return [PATHS[code] for code in codesOf(menu)]


def judgedItems(cell: CellData, codes: list[str]) -> list[int]:
    """§2.4：菜單內每條 path 都有答案（parse_ok）、且答案不完全一致的題目（item_id 由小到大）。"""
    records = [cell.arm(PATHS[code]) for code in codes]
    judged = []
    for item_id in sorted(records[0]):
        recs = [r[item_id] for r in records]
        if all(r["parse_ok"] for r in recs) and not all(cell.compare(recs[0]["parsed_answer"], r["parsed_answer"]) for r in recs[1:]):
            judged.append(int(item_id))
    return judged


def inputEstimates(cell: CellData, codes: list[str], item_ids: list[int]) -> dict[int, float]:
    """§6.4 的離線 input 估計：各候選 tokens_out 的總和 + L:en 生成 prompt 的 tokens_in。"""
    records = [cell.arm(PATHS[code]) for code in codes]
    anchor = cell.arm(PATHS["EN"])
    return {i: float(sum(r[i]["tokens_out"] for r in records) + anchor[i]["tokens_in"]) for i in item_ids}


def itemPools(armdir: str, aggdir: str, models: list[str] = MODELS) -> tuple[dict, dict]:
    """({(model, dataset, menu): 要裁決的題目}, {(model, dataset, menu): {item_id: 離線 input 估計}})"""
    pools, estimates = {}, {}
    for model in models:
        for dataset in DATASETS:
            cell = CellData(armdir, aggdir, model, dataset)
            for menu in MENU_NAMES:
                key = (model, dataset, menu)
                pools[key] = judgedItems(cell, codesOf(menu))
                estimates[key] = inputEstimates(cell, codesOf(menu), pools[key])
    return pools, estimates


def pilotSample(pools: dict, model: str) -> dict[str, list[int]]:
    """§6.4：每個資料集從 M12 的不一致題（由小到大）用 default_rng(0).choice(題數, 25, replace=False) 抽 25 題。"""
    sample = {}
    for dataset in DATASETS:
        ids = pools[(model, dataset, MAIN_MENU)]
        picked = np.random.default_rng(PILOT_SEED).choice(len(ids), size=min(PILOT_PER_DATASET, len(ids)), replace=False)
        sample[dataset] = sorted(ids[int(i)] for i in picked)
    return sample


def mainGridJudgePath(aggdir: str, model: str, dataset: str, pair_label: str) -> str:
    """主網格兩條 path 的 Judge 檔（§6.2、§6.3 用 EN+S1，§7.7 用 TWO_PATH）。"""
    return mainGridPath(aggdir, model, dataset, PAIR_BY_LABEL[pair_label])


def exceedsContext(model: str, tokens_in: float) -> bool:
    limit = CONTEXT_LIMITS[model]
    return bool(tokens_in + MAX_TOKENS > limit["context"] or (limit["max_input"] and tokens_in > limit["max_input"]))


def loadMenuRecords(directory: str, model: str, dataset: str, menu: str) -> tuple[dict, dict]:
    path = menuJudgePath(directory, model, dataset, menu)
    return loadRecords(path) if os.path.exists(path) else ({}, {})


# ------------------------------------------------------------------
# §6.1 離線重現
# ------------------------------------------------------------------
REPRODUCE_COLUMNS = {"A_V": "A_in", "S_in": "S_in", "excess_V": "excess_in"}   # 這次的欄位 -> rq1k_blocks.csv 的欄位


def evaluateReproduce(rows, rq1k_blocks) -> dict:
    """rows：這次的程式算出的區塊 × 菜單（A_V、S_in、excess_V）；和 rq1k_blocks.csv 比，誤差都要 ≤ TOLERANCE。"""
    keys = ["model", "dataset", "menu"]
    theirs = rq1k_blocks[keys + list(REPRODUCE_COLUMNS.values())].rename(
        columns={column: f"{column}_rq1k" for column in REPRODUCE_COLUMNS.values()})
    merged = rows[keys + list(REPRODUCE_COLUMNS)].merge(theirs, on=keys, how="left")
    max_diff, mismatches = {}, []
    for ours, column in REPRODUCE_COLUMNS.items():
        diff = (merged[ours] - merged[f"{column}_rq1k"]).abs()
        max_diff[ours] = float(diff.max()) if diff.notna().all() else float("nan")
        for row, d in zip(merged.itertuples(), diff):
            if not d <= TOLERANCE:   # NaN（rq1k 沒有這一列）也算不符
                mismatches.append({"model": row.model, "dataset": row.dataset, "menu": row.menu, "column": ours, "abs_diff": float(d)})
    return {"n_rows": len(merged), "max_abs_diff": max_diff, "tolerance": TOLERANCE, "mismatches": mismatches,
            "passed": len(merged) == len(MODELS) * len(DATASETS) * len(MENU_NAMES) and not mismatches}


# ------------------------------------------------------------------
# §6.2 模板核對
# ------------------------------------------------------------------
def templateRows(model: str, cell: CellData, main_records: dict, questions: dict, render, count) -> list[dict]:
    """
    mmlu × EN+S1 兩條都有答案的不一致題：用 choice-k-v1 的程式在 K = 2 時渲染 prompt（主網格記錄的候選順序），
    用該模型的 countTokens 重算，和主網格存下的 tokens_in 比。render(question, raw_texts) -> messages；count(text) -> tokens。
    """
    rows = []
    for item_id in judgedItems(cell, CHECK_CODES):
        record = main_records[item_id]
        order = record.get("presentation_order")
        if not order:
            rows.append({"item_id": item_id, "stored": record.get("tokens_in"), "recount": None, "equal": False,
                         "prompt_sha256": None, "note": "no recorded presentation order"})
            continue
        messages = render(questions[item_id], [cell.arm(arm_id)[item_id]["raw_text"] for arm_id in order])
        recount = sum(count(m.get("content", "")) for m in messages)
        rows.append({"item_id": item_id, "stored": record["tokens_in"], "recount": recount, "equal": recount == record["tokens_in"],
                     "prompt_sha256": PromptBuilder.promptSha256(messages)})
    return rows


def evaluateTemplate(per_model: dict[str, list[dict]], git_log: str) -> dict:
    models = {}
    for model, rows in per_model.items():
        bad = [r for r in rows if not r["equal"]]
        models[model] = {"n": len(rows), "n_equal": len(rows) - len(bad), "mismatches": bad[:50], "items": rows}
    return {"dataset": CHECK_DATASET, "menu": CHECK_MENU, "models": models, "git_log": git_log,
            "wording_difference": "none: choice-k-v1 and choice-v1 render through the same function (PromptJudgeChoiceFactory)",
            "passed": bool(models) and all(m["n"] > 0 and m["n_equal"] == m["n"] for m in models.values())}


# ------------------------------------------------------------------
# §6.3 流程核對
# ------------------------------------------------------------------
def recordedVersions(armdir: str, rq2_dir: str, model: str) -> dict:
    """主網格沒有記錄 Judge 回傳的版本：改列該模型 mmlu arm 檔的 model_version_string 與 RQ2 Judge 呼叫的紀錄。"""
    arms = Counter()
    for code in CHECK_CODES:
        _, records = loadRecords(armPath(armdir, model, CHECK_DATASET, ArmSpec.from_arm_id(PATHS[code])))
        arms.update(r.get("model_version_string") for r in records.values())
    rq2 = Counter()
    for path in glob.glob(os.path.join(rq2_dir, "judge_outputs", model, "*", "*", "*.json")):
        with open(path, encoding="utf-8") as f:
            rq2.update(json.load(f)[0].get("model_versions", {}))
    return {"arms": dict(arms), "rq2": dict(rq2)}


def evaluatePrecheck(precheck_dir: str, armdir: str, aggdir: str, rq2_dir: str, models: list[str] = MODELS) -> dict:
    """§6.3：每個模型重跑 mmlu × EN+S1 自己裁決自己，逐題和主網格的選擇比對，並和 RQ2 核對時的一致率並列。"""
    result = {}
    for model in models:
        cell = CellData(armdir, aggdir, model, CHECK_DATASET)
        judged = judgedItems(cell, CHECK_CODES)
        meta, records = loadMenuRecords(precheck_dir, model, CHECK_DATASET, CHECK_MENU)
        _, main = loadRecords(mainGridJudgePath(aggdir, model, CHECK_DATASET, CHECK_MENU))
        done = [i for i in judged if i in records]
        chosen = lambda r: (r.get("trace") or {}).get("chosen_arm")
        n_agree = sum(chosen(records[i]) == chosen(main[i]) for i in done)
        rq2_agree, rq2_n = RQ2_AGREEMENT[model]
        agreement = n_agree / len(judged) if judged else float("nan")
        drop = rq2_agree / rq2_n - agreement
        versions = recordedVersions(armdir, rq2_dir, model)
        returned = meta.get("model_versions", {})
        costs = [callCost(model, records[i].get("call")) for i in done]
        result[model] = {
            "file": menuJudgePath(precheck_dir, model, CHECK_DATASET, CHECK_MENU), "n": len(judged), "n_done": len(done),
            "complete": len(done) == len(judged) and len(judged) > 0,
            "n_agree": n_agree, "agreement": agreement, "rq2_n_agree": rq2_agree, "rq2_n": rq2_n,
            "rq2_agreement": rq2_agree / rq2_n, "drop": drop, "stop": bool(drop >= MAX_DROP - 1e-12),
            "n_same_final": sum(records[i]["final_answer"] == main[i]["final_answer"] for i in done),
            "n_same_tokens_in": sum(records[i]["tokens_in"] == main[i]["tokens_in"] for i in done),
            "n_same_output_text": sum((records[i].get("trace") or {}).get("judge_output") == (main[i].get("trace") or {}).get("judge_output")
                                      for i in done),
            "same_order_as_main_grid": all(records[i]["presentation_order"] == main[i]["presentation_order"] for i in done),
            "returned_versions": returned, "recorded_versions": versions,
            "version_matches_arms": bool(returned) and set(returned) <= set(versions["arms"]),
            "version_matches_rq2": bool(returned) and bool(versions["rq2"]) and set(returned) <= set(versions["rq2"]),
            "cost_usd": sum(c for c in costs if c is not None),
        }
    return {"dataset": CHECK_DATASET, "menu": CHECK_MENU, "max_drop": MAX_DROP, "models": result,
            "complete": all(r["complete"] for r in result.values()),
            "passed": all(r["complete"] and not r["stop"] for r in result.values())}


# ------------------------------------------------------------------
# §6.4 試跑
# ------------------------------------------------------------------
def _stats(values: list) -> dict:
    values = [v for v in values if v is not None]
    return {"mean": float(np.mean(values)) if values else float("nan"), "max": float(np.max(values)) if values else float("nan")}


def evaluatePilot(judge_dir: str, rep2_dir: str, pools: dict, estimates: dict, models: list[str] = MODELS) -> dict:
    """
    §6.4：每個模型 M12 的 100 題各跑兩次。第一次（judge_dir，正式）用來算無效比例、tokens、context 與成本；
    第二次（rep2_dir）只用來算一致率。全量成本：M12 用第一次的 API 計費平均；M3 菜單的 input 用離線估計 × 校正係數，
    output 每次呼叫用 M12 的平均。
    """
    out, total = {}, 0.0
    for model in models:
        sample = pilotSample(pools, model)
        runs = {1: {}, 2: {}}
        for dataset, ids in sample.items():
            for run, directory in ((1, judge_dir), (2, rep2_dir)):
                _, records = loadMenuRecords(directory, model, dataset, MAIN_MENU)
                runs[run].update({(dataset, i): records[i] for i in ids if i in records})
        keys = [(d, i) for d, ids in sample.items() for i in ids]
        n = len(keys)

        def counts(run: int) -> dict:
            c = Counter()
            for key in keys:
                if key in runs[run]:
                    flags = classifyRecord(runs[run][key])
                    c.update(k for k, flag in flags.items() if flag)
                    c["invalid"] += flags["no_choice"] or flags["out_of_range"] or flags["refused"]
            return {k: c[k] for k in ("no_choice", "out_of_range", "refused", "invalid")}

        first = [runs[1][k] for k in keys if k in runs[1]]
        calls = [r.get("call") or {} for r in first]
        usage_in = [c.get("usage_in") for c in calls]
        usage_out = [c.get("usage_out") for c in calls]
        with_usage = [c for c in calls if c.get("usage_in") is not None and c.get("usage_out") is not None]

        both = [k for k in keys if k in runs[1] and k in runs[2]]
        choice = lambda r: (r.get("trace") or {}).get("choice")
        same_choice = sum(choice(runs[1][k]) == choice(runs[2][k]) for k in both)
        same_final = sum(runs[1][k]["final_answer"] == runs[2][k]["final_answer"] for k in both)
        both_valid = sum(choice(runs[1][k]) is not None and choice(runs[2][k]) is not None for k in both)

        estimate_pilot = sum(estimates[(model, d, MAIN_MENU)][i] for d, i in keys if (d, i) in runs[1])
        factor = (sum(c["usage_in"] for c in with_usage) / estimate_pilot) if with_usage and estimate_pilot else float("nan")
        in_per_call = float(np.mean([c["usage_in"] for c in with_usage])) if with_usage else float("nan")
        out_per_call = float(np.mean([c["usage_out"] for c in with_usage])) if with_usage else float("nan")
        price_in, price_out = PRICES[model]

        full_max, full_max_item, menus = 0.0, None, {}
        for menu in MENU_NAMES:
            n_calls = sum(len(pools[(model, d, menu)]) for d in DATASETS)
            est = [(estimates[(model, d, menu)][i] * factor, (d, menu, i)) for d in DATASETS for i in pools[(model, d, menu)]]
            menu_in = in_per_call if menu == MAIN_MENU else (float(np.mean([e for e, _ in est])) if est else 0.0)
            cost = n_calls * (menu_in * price_in + out_per_call * price_out) / 1e6
            menus[menu] = {"calls": n_calls, "input_per_call": menu_in, "output_per_call": out_per_call, "cost_usd": cost}
            if est:
                value, item = max(est)
                if value > full_max:
                    full_max, full_max_item = value, item
        pilot_max_in = max([u for u in usage_in if u is not None] or [0])
        rep2_cost = sum(c for c in (callCost(model, r.get("call")) for r in runs[2].values()) if c is not None)
        model_cost = sum(m["cost_usd"] for m in menus.values()) + rep2_cost
        total += model_cost
        first_counts = counts(1)
        out[model] = {
            "n": n, "n_done_run1": len(runs[1]), "n_done_run2": len(runs[2]),
            "complete": len(runs[1]) == n and len(runs[2]) == n and len(with_usage) == n,
            "run1": first_counts, "run2": counts(2),
            "invalid_rate_run1": first_counts["invalid"] / n if n else float("nan"),
            "invalid_ok": bool(n) and first_counts["invalid"] / n <= PILOT_MAX_INVALID,
            "tokens_in_recount": _stats([r["tokens_in"] for r in first]), "tokens_out_recount": _stats([r["tokens_out"] for r in first]),
            "tokens_in_api": _stats(usage_in), "tokens_out_api": _stats(usage_out),
            "n_both_runs": len(both), "same_choice": same_choice, "same_final": same_final, "n_both_valid": both_valid,
            "same_choice_rate": same_choice / len(both) if both else float("nan"),
            "same_final_rate": same_final / len(both) if both else float("nan"),
            "calibration_factor": factor,
            "context_limit": CONTEXT_LIMITS[model], "max_tokens": MAX_TOKENS,
            "pilot_max_input": pilot_max_in, "pilot_exceeds": exceedsContext(model, pilot_max_in),
            "full_max_input_estimate": full_max, "full_max_input_item": full_max_item,
            "full_exceeds": exceedsContext(model, full_max),
            "menus": menus, "rep2_cost_usd": rep2_cost, "pilot_run1_cost_usd": sum(c for c in (callCost(model, r.get("call")) for r in first) if c is not None),
            "full_cost_usd": model_cost,
        }
    complete = all(m["complete"] for m in out.values())
    return {
        "models": out, "full_cost_usd": total, "max_cost_usd": MAX_COST, "complete": complete,
        "passed": complete and all(m["invalid_ok"] and not m["pilot_exceeds"] and not m["full_exceeds"] for m in out.values())
                  and total <= MAX_COST,
    }
