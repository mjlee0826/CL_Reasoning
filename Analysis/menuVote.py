from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from Analysis.alignment import CellData
from Analysis.experimentPlan import PATHS
from Analysis.probe import pathKey, strongest
from Runner.paths import ACTIVE_DATASETS

# ------------------------------------------------------------------
# RQ1-K（result/analysis/rq1k/rq1k_criteria.md）：同一份菜單（一組 path）內，
# 「全部一起多數決」對「H1 上最強的單一 path」。全程離線，資料來自 result/arms。
# ------------------------------------------------------------------
MODELS = ["gpt4omini", "qwen", "deepseek4.1flash", "gemini3.1flashlite"]
DATASETS = list(ACTIVE_DATASETS)
_M12 = ["EN", "ZH", "JA", "RU", "ES", "S1", "S2", "P1", "P2", "W1", "W2", "R"]
MENUS = {
    "M3L": ["EN", "ZH", "JA"],
    "M3S": ["EN", "S1", "S2"],
    "M3P": ["EN", "P1", "P2"],
    "M3W": ["EN", "W1", "W2"],
    "M5L": ["EN", "ZH", "JA", "RU", "ES"],
    "M12": _M12,
    "M8EN": [code for code in _M12 if code not in ("ZH", "JA", "RU", "ES")],
    "M14": _M12 + ["SR-EN", "SR-ZH"],
}
MAIN_MENU = "M12"                 # 判定一、二；它的 12 條 path 也定義子集_all
SENSITIVITY_MENUS = ("M14",)
COMPARE_MENUS = ("M3S", "M3L")    # 判定三：M3S − M3L
COMPARE_PATHS = ["EN", "ZH", "JA", "S1", "S2"]
THRESHOLD = 0.5                   # 百分點（§4）
SEED, REPS = 0, 200               # 與 RQ1 相同的切分
TOLERANCE = 1e-9                  # 恆等式（正確率單位）

TIE_PRIORITY, TIE_RANDOM = "priority", "random"
SUBSET_ANSWERED, SUBSET_ALL_ITEMS = "answered", "all_items"


@dataclass
class PathBlock:
    """一個區塊（模型 × 資料集）的 14 條 path，逐題陣列依 item_id 排序（與 RQ1 的切分位置對齊）。"""
    model: str
    dataset: str
    item_ids: np.ndarray
    gold: list
    answers: dict[str, list[str]]
    answered: dict[str, np.ndarray]
    correct: dict[str, np.ndarray]
    tokens: dict[str, np.ndarray]
    compare: Callable[[str, str], bool]


def loadPathBlock(armdir: str, model: str, dataset: str, aggdir: str = "result/aggregations") -> PathBlock:
    cell = CellData(armdir, aggdir, model, dataset)
    records = {code: cell.arm(arm_id) for code, arm_id in PATHS.items()}
    ids = sorted(records["EN"])
    for code, recs in records.items():
        if sorted(recs) != ids:
            raise ValueError(f"{model} | {dataset}: {code} covers different items than EN")
    gold = [records["EN"][i]["gold"] for i in ids]
    for code, recs in records.items():
        if any(recs[i]["gold"] != g for i, g in zip(ids, gold)):
            raise ValueError(f"{model} | {dataset}: {code} disagrees on gold answers")
    answers = {code: [recs[i]["parsed_answer"] for i in ids] for code, recs in records.items()}
    return PathBlock(
        model=model, dataset=dataset, item_ids=np.array(ids), gold=gold, answers=answers,
        answered={code: np.array([recs[i]["parse_ok"] for i in ids], dtype=bool) for code, recs in records.items()},
        correct={code: np.array([cell.compare(g, a) for g, a in zip(gold, answers[code])], dtype=bool) for code in records},
        tokens={code: np.array([recs[i]["tokens_out"] for i in ids], dtype=float) for code, recs in records.items()},
        compare=cell.compare,
    )


# ------------------------------------------------------------------
# 多數決（§3.1）
# ------------------------------------------------------------------
def ordered(codes: list[str]) -> list[str]:
    """平手的優先順序：EN 第一，其餘依 path 代號字母序（= RQ1 的 pathKey）。"""
    return sorted(codes, key=pathKey)


def vote(answers: list[str], answered: list[bool], compare, rng: np.random.Generator | None = None) -> tuple[str, bool]:
    """
    answers / answered 依優先順序排列。沒答案的 path 不投票；全部沒答案時回傳 ""。
    平手時：rng 為 None 取優先順序最高的 path 所支持的答案，否則在平手答案（依優先順序排列）中均勻抽一個。
    回傳 (聚合答案, 是否平手)。
    """
    groups = []   # [答案, 票數]，依「支持它的最高優先 path」排列
    for answer, ok in zip(answers, answered):
        if not ok:
            continue
        for group in groups:
            if compare(group[0], answer):
                group[1] += 1
                break
        else:
            groups.append([answer, 1])
    if not groups:
        return "", False
    top = max(count for _, count in groups)
    tied = [answer for answer, count in groups if count == top]
    if len(tied) == 1:
        return tied[0], False
    return (tied[0] if rng is None else tied[int(rng.integers(len(tied)))]), True


def menuFinals(block: PathBlock, menu: str, ties: str = TIE_PRIORITY) -> tuple[list[str], np.ndarray]:
    """每題的菜單聚合答案與是否平手。"""
    codes = ordered(MENUS[menu])
    finals, tied = [], []
    for k, item_id in enumerate(block.item_ids):
        rng = np.random.default_rng([SEED, int(item_id)]) if ties == TIE_RANDOM else None
        final, tie = vote([block.answers[c][k] for c in codes], [block.answered[c][k] for c in codes], block.compare, rng)
        finals.append(final)
        tied.append(tie)
    return finals, np.array(tied, dtype=bool)


def allAnswered(block: PathBlock, codes: list[str]) -> np.ndarray:
    return np.all([block.answered[c] for c in codes], axis=0)


# ------------------------------------------------------------------
# 逐切分的量（§3.4–§3.5、§5.3–§5.5）
# ------------------------------------------------------------------
def evaluateMenu(block: PathBlock, menu: str, splits: np.ndarray, ties: str = TIE_PRIORITY,
                 subset: str = SUBSET_ANSWERED, decompose: bool = True) -> dict:
    """
    一個區塊 × 菜單：對 200 次切分平均的 A / S_in / S_all / EN / Excess（Excess 為百分點），
    以及（decompose 時）分解、恆等式的最大誤差、被選中的 path、成本。
    subset = all_items：敏感度，所有題目都納入（沒答案的 path 不投票；單一 path 沒答案算錯）。
    """
    codes = ordered(MENUS[menu])
    finals, tied = menuFinals(block, menu, ties)
    agg_correct = np.array([block.compare(g, f) for g, f in zip(block.gold, finals)], dtype=bool)
    n = len(block.item_ids)
    if subset == SUBSET_ALL_ITEMS:
        sub_in = sub_all = np.ones(n, dtype=bool)
    else:
        sub_in, sub_all = allAnswered(block, codes), allAnswered(block, MENUS[MAIN_MENU])

    path_correct = np.array([block.correct[c] for c in codes])                    # K × n
    any_correct, mean_correct = path_correct.any(axis=0), path_correct.mean(axis=0)
    agree = np.array([all(block.compare(block.answers[codes[0]][k], block.answers[c][k]) for c in codes[1:])
                      for k in range(n)], dtype=bool)
    menu_tokens = np.sum([block.tokens[c] for c in codes], axis=0)

    per_split = {key: [] for key in ("A_in", "S_in", "EN_in", "A_all", "S_all", "EN_all", "excess_in", "excess_all",
                                     "d", "c", "m", "headroom", "recovery", "recovery_blind", "tokens_menu", "tokens_S_in")}
    chosen_in, chosen_all = Counter(), Counter()
    undefined, max_error = 0, 0.0
    for h1 in splits:
        h2 = ~h1
        in1, in2, all1, all2 = h1 & sub_in, h2 & sub_in, h1 & sub_all, h2 & sub_all
        s_in = strongest({c: int(block.correct[c][in1].sum()) for c in codes})
        s_all = strongest({c: int(block.correct[c][all1].sum()) for c in MENUS[MAIN_MENU]})
        chosen_in[s_in] += 1
        chosen_all[s_all] += 1
        if menu == MAIN_MENU and subset == SUBSET_ANSWERED and s_in != s_all:
            raise ValueError(f"{block.model} | {block.dataset}: M12's S_in ({s_in}) != S_all ({s_all})")
        values = {
            "A_in": agg_correct[in2].mean(), "S_in": block.correct[s_in][in2].mean(), "EN_in": block.correct["EN"][in2].mean(),
            "A_all": agg_correct[all2].mean(), "S_all": block.correct[s_all][all2].mean(),
            "EN_all": block.correct["EN"][all2].mean(),
        }
        values["excess_in"] = values["A_in"] - values["S_in"]
        values["excess_all"] = values["A_all"] - values["S_all"]
        if decompose:
            dis = in2 & ~agree
            d = dis.sum() / in2.sum()
            c = any_correct[dis].mean() if dis.any() else float("nan")
            m = mean_correct[dis].mean() if dis.any() else float("nan")
            if dis.any() and c > m:
                recovery = (agg_correct[dis].mean() - m) / (c - m)
                recovery_blind = (block.correct[s_in][dis].mean() - m) / (c - m)
                error = abs(values["excess_in"] - d * (c - m) * (recovery - recovery_blind))
            else:   # 沒有分歧題或分歧題全部無人答對：recovery 無定義，Excess_in 必須是 0
                recovery = recovery_blind = float("nan")
                undefined += 1
                error = abs(values["excess_in"])
            max_error = max(max_error, error)
            values.update(d=d, c=c, m=m, headroom=d * (c - m) if dis.any() else 0.0, recovery=recovery,
                          recovery_blind=recovery_blind, tokens_menu=menu_tokens[in2].mean(),
                          tokens_S_in=block.tokens[s_in][in2].mean())
        for key, value in values.items():
            per_split[key].append(value)

    def mean(key: str, scale: float = 1.0) -> float:
        array = np.asarray(per_split[key], dtype=float)
        return float(np.nanmean(array) * scale) if np.isfinite(array).any() else float("nan")

    row = {
        "model": block.model, "dataset": block.dataset, "menu": menu, "K": len(codes), "n": n,
        "keep_in": float(sub_in.mean()), "keep_all": float(sub_all.mean()), "tie_rate": float(tied[sub_in].mean()),
        "A_in": mean("A_in"), "S_in": mean("S_in"), "EN_in": mean("EN_in"),
        "A_all": mean("A_all"), "S_all": mean("S_all"), "EN_all": mean("EN_all"),
        "excess_in": mean("excess_in", 100), "excess_all": mean("excess_all", 100),
    }
    if decompose:
        top_in, top_all = chosen_in.most_common(2), chosen_all.most_common(2)
        row.update({
            "d": mean("d"), "c": mean("c"), "m": mean("m"), "headroom_pp": mean("headroom", 100),
            "recovery": mean("recovery"), "recovery_blind": mean("recovery_blind"),
            "undefined_splits": undefined, "identity_max_error": max_error,
            "calls_menu": len(codes), "calls_single": 1, "tokens_menu": mean("tokens_menu"), "tokens_S_in": mean("tokens_S_in"),
            "S_in_top": " / ".join(f"{code} {count / len(splits):.0%}" for code, count in top_in),
            "S_all_top": " / ".join(f"{code} {count / len(splits):.0%}" for code, count in top_all),
        })
    return {"row": row, "finals": finals, "tied": tied, "agg_correct": agg_correct, "sub_in": sub_in, "sub_all": sub_all}


def compareMenus(block: PathBlock, splits: np.ndarray, finals: dict[str, np.ndarray]) -> float:
    """§4.5：EN、ZH、JA、S1、S2 都有答案的題目上，M3S 聚合 − M3L 聚合在 H2 的正確率，對切分平均（百分點）。"""
    subset = allAnswered(block, COMPARE_PATHS)
    first, second = (finals[menu] for menu in COMPARE_MENUS)
    diffs = [first[~h1 & subset].mean() - second[~h1 & subset].mean() for h1 in splits]
    return float(np.mean(diffs) * 100)


def regrets(excess: np.ndarray) -> dict:
    """§4.4：E_b（百分點）-> 永遠聚合 / 永遠單一最強的 regret 與空間。"""
    excess = np.asarray(excess, dtype=float)
    regret_A, regret_S = float(np.maximum(0, -excess).mean()), float(np.maximum(0, excess).mean())
    return {"regret_aggregate": regret_A, "regret_single": regret_S, "space": min(regret_A, regret_S)}
