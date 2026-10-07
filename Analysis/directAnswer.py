import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Arm.GenerationRecord import GenerationRecord
from Analysis.alignment import CellData, loadRecords
from Analysis.blockStats import summarize
from Analysis.crossJudge import (MODELS, STRONG, WEAK, DATASETS, PAIRS, SOURCE_MAIN, judgedItems, mainGridPath,
                                 judgeOutputPath, crossArrays)
from Analysis.experimentPlan import PATHS, Pair

# ------------------------------------------------------------------
# RQ2-DA（result/analysis/rq2da/rq2da_criteria.md）：強模型從弱模型的兩份候選中挑一份（RQ2 交叉檔）
# vs 強模型自己直接作答（result/arms 的 L:en）。全程離線。
#   一格 = 弱模型 g × 強模型 s × 資料集 × 配對，都在 g 的 both_answered 題目上（RQ2 的 crossArrays）
# ------------------------------------------------------------------
OUT_DIR = "result/analysis/rq2da"
CRITERIA_FILE = "rq2da_criteria.md"
RQ2_DIR = "result/analysis/rq2"
RQ2_CELLS, RQ2_JUDGE_DIR = "cross_cells.csv", "judge_outputs"
THRESHOLD = 0.5                    # 百分點（§4）
TOLERANCE = 1e-9                   # §3.3 恆等式（正確率單位）
RESULT13 = {"mean": 2.47, "ci_low": 1.65, "ci_high": 3.29, "n_positive": 8}   # RQ2 報告 §7.2，比到小數第二位（§5.1）
DIRECT_ARM = PATHS["EN"]           # 強模型直接作答 = 它的 L:en
FORWARD, REVERSE, EQUIVALENT, UNDETERMINED = "正向成立", "反向成立", "兩者相當", "無法判定"
SYSTEMS = ["Weak_EN", "Sys_self", "Sys_J", "Sys_D", "Strong_alone"]
DIFFS = [("Sys_J", "Sys_self"), ("Sys_D", "Sys_self"), ("Sys_J", "Sys_D"), ("Strong_alone", "Sys_J"), ("Strong_alone", "Sys_D")]
CLASSES = ["own_chosen", "own_not_chosen", "own_neither"]   # §6.3
CANDIDATE_CLASSES = ["one_right", "both_wrong"]             # §6.4
TABLE = ["J1_D1", "J1_D0", "J0_D1", "J0_D0"]                # §6.5：裁判對錯 × 直接作答對錯


# ------------------------------------------------------------------
# 檔案：每個 generator × Judge 用 RQ2 判定時實際採用的版本（cross_cells.csv 的 used_in_analysis）
# ------------------------------------------------------------------
def usedCells(rq2_dir: str) -> pd.DataFrame:
    cells = pd.read_csv(os.path.join(rq2_dir, RQ2_CELLS))
    return cells[cells["used_in_analysis"]].set_index(["generator", "judge", "dataset", "pair"])


def judgeRecords(rq2_dir: str, aggdir: str, cell: CellData, judge: str, pair: Pair, source: str) -> tuple[str, dict]:
    """(檔案路徑, 該格 both_answered 不一致題的紀錄)。主網格檔含全部題目，只取這些題目（與 RQ2 的 buildCells 相同）。"""
    if source == SOURCE_MAIN:
        path = mainGridPath(aggdir, cell.model_name, cell.dataset_name, pair)
        _, records = loadRecords(path)
        return path, {item_id: records[item_id] for item_id in judgedItems(cell, pair)}
    path = judgeOutputPath(os.path.join(rq2_dir, RQ2_JUDGE_DIR), judge, cell.model_name, cell.dataset_name, pair)
    metadata, records = loadRecords(path)
    if (metadata.get("generator"), metadata.get("judge")) != (cell.model_name, judge):
        raise ValueError(f"{path}: generator / judge {(metadata.get('generator'), metadata.get('judge'))}")
    return path, records


# ------------------------------------------------------------------
# 逐格陣列（both_answered 題目，依 item_id 排序）
# ------------------------------------------------------------------
@dataclass
class DirectCell:
    dis: np.ndarray               # 兩條 path 答案不同
    judge_correct: np.ndarray     # 一致題 = 共同答案；不一致題 = 裁判挑出的答案（無效 = 空，算錯）
    judge_final: list
    chosen: list                  # 不一致題上被選中的 arm（無效為 None）；一致題為 None
    answer_a: list
    answer_b: list
    correct_a: np.ndarray
    correct_b: np.ndarray
    direct: list                  # 裁判模型自己的 L:en 答案
    direct_ok: np.ndarray
    direct_correct: np.ndarray    # 沒有答案算錯
    weak_en_correct: np.ndarray   # 產生候選的模型自己的 L:en（沒有答案算錯）
    judge_tokens_in: np.ndarray   # 不一致題上的裁判呼叫（重算）；一致題為 0
    judge_tokens_out: np.ndarray
    judge_api_in: np.ndarray      # API 計費；沒有 usage 為 nan
    judge_api_out: np.ndarray
    direct_tokens_in: np.ndarray  # L:en 的呼叫（重算）
    direct_tokens_out: np.ndarray
    compare: object


def directCell(cell: CellData, judge_cell: CellData, pair: Pair, records: dict) -> DirectCell:
    arrays = crossArrays(cell, pair, records)    # 核對紀錄剛好涵蓋 both_answered 的不一致題
    ids = [int(i) for i in arrays.item_ids]
    recA, recB, weak_en, direct = cell.arm(pair.arm_a), cell.arm(pair.arm_b), cell.arm(DIRECT_ARM), judge_cell.arm(DIRECT_ARM)
    if not set(ids) <= set(direct):
        raise ValueError(f"{judge_cell.model_name} | {judge_cell.dataset_name}: L:en misses items of {cell.model_name}'s cell")
    compare = cell.compare
    golds = [recA[i]["gold"] for i in ids]
    if any(direct[i]["gold"] != g for i, g in zip(ids, golds)):
        raise ValueError(f"{judge_cell.model_name} | {judge_cell.dataset_name}: L:en disagrees on gold answers")
    n = len(ids)
    finals, chosen = [], []
    tokens = {key: np.zeros(n) for key in ("in", "out")}
    api = {key: np.full(n, np.nan) for key in ("in", "out")}
    for k, (item_id, dis) in enumerate(zip(ids, arrays.dis)):
        if not dis:
            finals.append(recA[item_id]["parsed_answer"])
            chosen.append(None)
            continue
        record = records[item_id]
        finals.append(record["final_answer"])
        chosen.append((record.get("trace") or {}).get("chosen_arm"))
        tokens["in"][k], tokens["out"][k] = record["tokens_in"], record["tokens_out"]
        call = record.get("call") or {}
        if call.get("usage_in") is not None:
            api["in"][k] = call["usage_in"]
        if call.get("usage_out") is not None:
            api["out"][k] = call["usage_out"]
    direct_answers = [direct[i]["parsed_answer"] for i in ids]
    direct_ok = np.array([direct[i]["parse_ok"] for i in ids], dtype=bool)
    return DirectCell(
        dis=arrays.dis, judge_correct=arrays.final_correct, judge_final=finals, chosen=chosen,
        answer_a=[recA[i]["parsed_answer"] for i in ids], answer_b=[recB[i]["parsed_answer"] for i in ids],
        correct_a=arrays.correct_a, correct_b=arrays.correct_b,
        direct=direct_answers, direct_ok=direct_ok,
        direct_correct=direct_ok & np.array([compare(g, a) for g, a in zip(golds, direct_answers)], dtype=bool),
        weak_en_correct=np.array([weak_en[i]["parse_ok"] and compare(g, weak_en[i]["parsed_answer"]) for i, g in zip(ids, golds)],
                                 dtype=bool),
        judge_tokens_in=tokens["in"], judge_tokens_out=tokens["out"], judge_api_in=api["in"], judge_api_out=api["out"],
        direct_tokens_in=np.array([direct[i]["tokens_in"] for i in ids], dtype=float),
        direct_tokens_out=np.array([direct[i]["tokens_out"] for i in ids], dtype=float),
        compare=compare,
    )


def agreeMask(c: DirectCell) -> np.ndarray:
    """§3.2：裁判的答案與 L:en 答案都非空，且 compareTwoAnswer 為 True。"""
    return np.array([GenerationRecord.isParseOk(f) and ok and c.compare(f, d)
                     for f, d, ok in zip(c.judge_final, c.direct, c.direct_ok)], dtype=bool)


def judgeOnly(c: DirectCell) -> dict:
    """§6.8 用：不一致題上的 a、b、agree、d。"""
    dis = c.dis
    return {"n": len(dis), "n_dis": int(dis.sum()), "d": float(dis.mean()),
            "a": float(c.judge_correct[dis].mean()) if dis.any() else float("nan"),
            "b": float(c.direct_correct[dis].mean()) if dis.any() else float("nan"),
            "agree": float(agreeMask(c)[dis].mean()) if dis.any() else float("nan")}


def cellRow(c: DirectCell, self_correct: np.ndarray, pair: Pair) -> dict:
    """一格的 §3 與 §6 的量。self_correct = Sys_self 的逐題對錯（同一批 both_answered 題目）。"""
    dis = c.dis
    sys_J = c.judge_correct
    sys_D = np.where(dis, c.direct_correct, c.judge_correct)        # 一致題兩者都用共同答案
    row = judgeOnly(c)
    row.update({
        "Sys_J": float(sys_J.mean()), "Sys_D": float(sys_D.mean()), "Sys_self": float(self_correct.mean()),
        "Strong_alone": float(c.direct_correct.mean()), "Weak_EN": float(c.weak_en_correct.mean()),
    })
    row["Diff_pp"] = 100 * (row["Sys_J"] - row["Sys_D"])
    row["identity_error"] = abs((row["Sys_J"] - row["Sys_D"]) - row["d"] * (row["a"] - row["b"])) if dis.any() \
        else abs(row["Sys_J"] - row["Sys_D"])

    # §6.3：強模型自己的答案落在哪裡
    match_a = np.array([ok and c.compare(d, a) for d, a, ok in zip(c.direct, c.answer_a, c.direct_ok)], dtype=bool)
    match_b = np.array([ok and c.compare(d, b) for d, b, ok in zip(c.direct, c.answer_b, c.direct_ok)], dtype=bool)
    chose_a = np.array([x == pair.arm_a for x in c.chosen], dtype=bool)
    chose_b = np.array([x == pair.arm_b for x in c.chosen], dtype=bool)
    invalid = dis & np.array([x is None for x in c.chosen], dtype=bool)
    classes = {
        "own_chosen": dis & ((match_a & chose_a) | (match_b & chose_b)),
        "own_not_chosen": dis & (match_a | match_b) & ~((match_a & chose_a) | (match_b & chose_b)),
        "own_neither": dis & ~(match_a | match_b),
    }
    classes.update({"one_right": dis & (c.correct_a ^ c.correct_b), "both_wrong": dis & ~c.correct_a & ~c.correct_b})
    for name, mask in classes.items():
        row.update({f"n_{name}": int(mask.sum()), f"J_right_{name}": int(c.judge_correct[mask].sum()),
                    f"D_right_{name}": int(c.direct_correct[mask].sum())})
    row["n_own_not_chosen_invalid"] = int((classes["own_not_chosen"] & invalid).sum())
    row["n_invalid_choice"] = int(invalid.sum())
    # §6.5
    for name, (j, d) in zip(TABLE, [(True, True), (True, False), (False, True), (False, False)]):
        row[f"n_{name}"] = int((dis & (c.judge_correct == j) & (c.direct_correct == d)).sum())
    # §6.6：不一致題上 L:en 沒有答案；排除後的 Diff
    missing = dis & ~c.direct_ok
    keep = ~missing
    row["n_direct_missing"] = int(missing.sum())
    row["Diff_sens_pp"] = 100 * float(sys_J[keep].mean() - sys_D[keep].mean())
    # §6.7：tokens 的合計（之後依題目合併取平均）
    row.update({
        "judge_calls": int(dis.sum()), "judge_tokens_in_sum": float(c.judge_tokens_in[dis].sum()),
        "judge_tokens_out_sum": float(c.judge_tokens_out[dis].sum()),
        "judge_api_calls": int(np.isfinite(c.judge_api_in[dis]).sum()),
        "judge_api_in_sum": float(np.nansum(c.judge_api_in[dis])), "judge_api_out_sum": float(np.nansum(c.judge_api_out[dis])),
        "direct_dis_tokens_in_sum": float(c.direct_tokens_in[dis].sum()),
        "direct_dis_tokens_out_sum": float(c.direct_tokens_out[dis].sum()),
        "direct_all_calls": len(dis), "direct_all_tokens_in_sum": float(c.direct_tokens_in.sum()),
        "direct_all_tokens_out_sum": float(c.direct_tokens_out.sum()),
    })
    return row


# ------------------------------------------------------------------
# 全部格子
# ------------------------------------------------------------------
def buildAll(armdir: str, aggdir: str, rq2_dir: str) -> dict:
    """
    回傳 {"cells": 48 格（g × s × 資料集 × 配對）, "matrix": 192 格（4 × 4 × 資料集 × 配對，§6.8）, "files": 讀過的 Judge 檔}。
    每個 generator × Judge 用 RQ2 判定時實際採用的版本。
    """
    used = usedCells(rq2_dir)
    data = {(m, d): CellData(armdir, aggdir, m, d) for m in MODELS for d in DATASETS}
    cells, matrix, files = [], [], []
    for generator in MODELS:
        for dataset in DATASETS:
            cell = data[(generator, dataset)]
            for pair in PAIRS:
                loaded = {}
                for judge in MODELS:
                    row = used.loc[(generator, judge, dataset, pair.label)]
                    path, records = judgeRecords(rq2_dir, aggdir, cell, judge, pair, row["source"])
                    files.append({"generator": generator, "judge": judge, "dataset": dataset, "pair": pair.label,
                                  "source": row["source"], "path": path})
                    c = directCell(cell, data[(judge, dataset)], pair, records)
                    loaded[judge] = (c, row)
                    matrix.append({"generator": generator, "judge": judge, "dataset": dataset, "pair": pair.label,
                                   "source": row["source"], **judgeOnly(c)})
                if generator not in WEAK:
                    continue
                self_cell, self_row = loaded[generator]
                for strong in STRONG:
                    c, rq2_row = loaded[strong]
                    cells.append({"generator": generator, "judge": strong, "dataset": dataset, "pair": pair.label,
                                  "judge_source": rq2_row["source"], "self_source": self_row["source"],
                                  "rq2_n_dis": int(rq2_row["n_dis"]), **cellRow(c, self_cell.judge_correct, pair)})
    return {"cells": pd.DataFrame(cells), "matrix": pd.DataFrame(matrix), "files": pd.DataFrame(files)}


# ------------------------------------------------------------------
# 區塊、判定、核對
# ------------------------------------------------------------------
def blockTable(cells: pd.DataFrame) -> pd.DataFrame:
    """8 個區塊（g × 資料集），各為 2 個強模型 × 3 個配對的 6 格等權平均；差值以百分點。"""
    rows = []
    for generator in WEAK:
        for dataset in DATASETS:
            group = cells[(cells.generator == generator) & (cells.dataset == dataset)]
            if len(group) != len(STRONG) * len(PAIRS):
                raise ValueError(f"{generator} | {dataset}: {len(group)} cells, expected {len(STRONG) * len(PAIRS)}")
            row = {"generator": generator, "dataset": dataset, "n_cells": len(group)}
            row.update({name: float(group[name].mean()) for name in SYSTEMS})
            row.update({f"{x}-{y}_pp": 100 * float((group[x] - group[y]).mean()) for x, y in DIFFS})
            row["Diff_sens_pp"] = float(group["Diff_sens_pp"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def state(summary: dict, threshold: float = THRESHOLD) -> str:
    """§4 的四種狀態（先對區塊取平均，再和門檻比）。"""
    excludes_zero = summary["ci_low"] > 0 or summary["ci_high"] < 0
    if summary["mean"] >= threshold and excludes_zero:
        return FORWARD
    if summary["mean"] <= -threshold and excludes_zero:
        return REVERSE
    if summary["ci_low"] >= -threshold and summary["ci_high"] <= threshold:
        return EQUIVALENT
    return UNDETERMINED


def checks(cells: pd.DataFrame, blocks: pd.DataFrame) -> dict:
    """§5 的三項核對。"""
    r13 = summarize(blocks["Sys_J-Sys_self_pp"])
    found = {"mean": round(r13["mean"], 2), "ci_low": round(r13["ci_low"], 2), "ci_high": round(r13["ci_high"], 2),
             "n_positive": r13["n_positive"]}
    mismatched = cells[cells["n_dis"] != cells["rq2_n_dis"]]
    return {
        "result13": {"summary": r13, "rounded": found, "expected": RESULT13, "passed": found == RESULT13},
        "n_dis": {"n_cells": len(cells), "mismatches": mismatched[["generator", "judge", "dataset", "pair", "n_dis", "rq2_n_dis"]]
                  .to_dict("records"), "passed": mismatched.empty and len(cells) == len(WEAK) * len(STRONG) * len(DATASETS) * len(PAIRS)},
        "identity": {"max_error": float(cells["identity_error"].max()), "tolerance": TOLERANCE,
                     "passed": bool((cells["identity_error"] <= TOLERANCE).all())},
    }


def pooled(cells: pd.DataFrame, names: list[str]) -> list[dict]:
    """§6.3–§6.4：題目合併的題數、比例、a、b。"""
    total = int(cells["n_dis"].sum())
    out = []
    for name in names:
        n = int(cells[f"n_{name}"].sum())
        out.append({"class": name, "n": n, "share": n / total if total else float("nan"),
                    "a": cells[f"J_right_{name}"].sum() / n if n else float("nan"),
                    "b": cells[f"D_right_{name}"].sum() / n if n else float("nan")})
    return out
