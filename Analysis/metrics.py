import numpy as np

from Analysis.alignment import PairArrays
from Analysis.experimentPlan import Pair, TIE_BREAK_KEY
from Analysis.splitHalf import pickMax, recoveryStats, splitHalfStats

# ------------------------------------------------------------------
# aggregation_cells.csv 一列的數值欄
#   all            全部題目
#   both_answered  兩個候選都有解析出答案的題目（在子集內重新切半）；聚合器沒給答案仍算答錯，比例見 agg_no_answer
# ------------------------------------------------------------------
SUBSETS = ("all", "both_answered")

# splitHalfStats 的輸出欄（與 0A-1 RecoveryBlind metadata 相同）
SPLIT_COLUMNS = ["anchor", "anchor_rate", "n_A_H2", "n_B_H2", "w_A_H2",
                 "recovery_blind_H2", "recovery_H2", "skill_H2", "reps_undefined", "reps_skill_undefined"]
VALUE_COLUMNS = (["n", "n_dis", "acc_a", "acc_b", "d", "c", "m", "parse_fail_a", "parse_fail_b",
                  "acc_final", "recovery", "agg_no_answer", "off_menu"]
                 + SPLIT_COLUMNS + ["tok_out_a", "tok_out_b", "tok_out_agg"])


def subsetMask(arrays: PairArrays, subset: str) -> np.ndarray:
    if subset == "all":
        return np.ones(len(arrays.item_ids), dtype=bool)
    if subset == "both_answered":
        return arrays.answered_a & arrays.answered_b
    raise ValueError(f"Unknown subset '{subset}'")


def blindFinal(arrays: PairArrays, pair: Pair, seed: int, reps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    全樣本的 Blind：永遠選全樣本答對較多的一方（有選擇偏誤，以 split-half 的 recovery_H2 為準）。
    平手用 pickMax，rep = reps 與任何一次切分都不重複。回傳 (final_correct, final_answered)。
    """
    first = pickMax(int(arrays.correct_a.sum()), int(arrays.correct_b.sum()), seed, reps,
                    TIE_BREAK_KEY[pair.arm_a], TIE_BREAK_KEY[pair.arm_b])
    return (arrays.correct_a, arrays.answered_a) if first else (arrays.correct_b, arrays.answered_b)


def onDisagreement(values: np.ndarray, dis: np.ndarray) -> float:
    """分歧題上的平均（聚合器只在分歧題動作）。"""
    return float(values[dis].mean()) if dis.any() else float("nan")


def computeRow(arrays: PairArrays, pair: Pair, aggregator_id: str, seed: int = 0, reps: int = 200) -> dict:
    """arrays 已是該 subset 的題目；aggregator_id == "blind" 時最終答案由候選計算，否則取自聚合檔。"""
    n = len(arrays.item_ids)
    blind = aggregator_id == "blind"
    if blind:
        final_correct, final_answered = blindFinal(arrays, pair, seed, reps)
        off_menu, tokens_agg = np.zeros(n, dtype=bool), np.zeros(n)
    else:
        final_correct, final_answered = arrays.final_correct, arrays.final_answered
        off_menu, tokens_agg = arrays.off_menu, arrays.tokens_out_agg

    dis = arrays.dis
    full = recoveryStats(arrays.correct_a, arrays.correct_b, final_correct, dis, np.ones(n, dtype=bool))
    split = splitHalfStats(arrays.correct_a, arrays.correct_b, None if blind else final_correct, dis,
                           pair.arm_a, pair.arm_b, TIE_BREAK_KEY[pair.arm_a], TIE_BREAK_KEY[pair.arm_b], seed, reps)
    row = {
        "n": n,
        "n_dis": int(dis.sum()),
        "acc_a": float(arrays.correct_a.mean()),
        "acc_b": float(arrays.correct_b.mean()),
        "d": full["d"],
        "c": full["c"],
        "m": full["m"],
        "parse_fail_a": float(1 - arrays.answered_a.mean()),
        "parse_fail_b": float(1 - arrays.answered_b.mean()),
        "acc_final": float(final_correct.mean()),
        "recovery": full["recovery"],
        "agg_no_answer": onDisagreement(~final_answered, dis),
        "off_menu": onDisagreement(off_menu, dis),
        **{key: split[key] for key in SPLIT_COLUMNS},
        "tok_out_a": float(arrays.tokens_out_a.mean()),
        "tok_out_b": float(arrays.tokens_out_b.mean()),
        "tok_out_agg": float(tokens_agg.mean()),
    }
    checkIdentities(row, full, split, blind)
    return row


def checkIdentities(row: dict, full: dict, split: dict, blind: bool):
    """md §0 的恆等式；不成立代表對齊或聚合檔有問題。"""
    assert all(full[key] == split[key] or np.isnan(full[key]) and np.isnan(split[key]) for key in ("N", "D", "d", "c", "m"))
    if np.isfinite(row["recovery"]):
        gain = row["acc_final"] - (row["acc_a"] + row["acc_b"]) / 2
        assert abs(gain - row["d"] * row["c"] / 2 * row["recovery"]) < 1e-12, f"Gain = d(c/2)·recovery fails: {row}"
    if blind and np.isfinite(row["recovery_H2"]):
        assert abs(row["recovery_H2"] - row["recovery_blind_H2"]) < 1e-12, f"Blind recovery_H2 != recovery_blind_H2: {row}"
