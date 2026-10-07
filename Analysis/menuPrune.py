from collections import Counter

import numpy as np

from Analysis.menuVote import PathBlock, MENUS, allAnswered, ordered, vote
from Analysis.probe import pathKey, strongest

# ------------------------------------------------------------------
# RQ3（result/analysis/rq3/rq3_criteria.md）：拿掉 path 之後的多數決。全程離線，資料與多數決都沿用 RQ1-K。
#   所有菜單都在 M12 的子集（12 條都有答案）上算；挑選在 H1 ∩ 子集、評分在 H2 ∩ 子集
# ------------------------------------------------------------------
M12 = list(MENUS["M12"])                        # EN ZH JA RU ES S1 S2 P1 P2 W1 W2 R
PERSONAS = ["P1", "P2"]
DROP = {p: f"M12-{p}" for p in M12}             # M12−p 的菜單名稱
MENU_CODES = {"M12": M12, **{name: [c for c in M12 if c != p] for p, name in DROP.items()},
              "M10": [c for c in M12 if c not in PERSONAS]}
MENU_NAMES = list(MENU_CODES)                   # M12、12 份 M12−p、M10
REMOVAL_ORDER = sorted(M12, key=pathKey, reverse=True)   # §3.1 平手時先拿掉的：ZH W2 W1 S2 S1 RU R P2 P1 JA ES EN
CATEGORIES = {"語言": ["ZH", "JA", "RU", "ES"], "L:en": ["EN"], "採樣": ["S1", "S2"], "persona": ["P1", "P2"],
              "改寫": ["W1", "W2"], "簡短 CoT": ["R"]}
THRESHOLD = 0.5      # 百分點（§4）
TOLERANCE = 1e-9     # §6.1 重現與 §7.7 恆等式
DECOMPOSED = ["M12", "M10"]
DECOMPOSITION = ["d", "c", "m", "headroom", "recovery_V", "recovery_blind"]


def menuCorrect(block: PathBlock, codes: list[str]) -> np.ndarray:
    """菜單的多數決（RQ1-K 的 vote，平手依優先順序）每題是否答對。"""
    codes = ordered(codes)
    finals = [vote([block.answers[c][k] for c in codes], [block.answered[c][k] for c in codes], block.compare)[0]
              for k in range(len(block.item_ids))]
    return np.array([block.compare(g, f) for g, f in zip(block.gold, finals)], dtype=bool)


def agreeMask(block: PathBlock, codes: list[str]) -> np.ndarray:
    first = block.answers[codes[0]]
    return np.array([all(block.compare(first[k], block.answers[c][k]) for c in codes[1:]) for k in range(len(block.item_ids))],
                    dtype=bool)


def pickRemoval(counts: dict[str, int]) -> str:
    """§3.1：H1 上讓 M12−p 答對最多的 p；平手時拿掉 REMOVAL_ORDER 中較前（優先順序較後）的那條。"""
    return max(M12, key=lambda p: (counts[DROP[p]], -REMOVAL_ORDER.index(p)))


def decompose(in2: np.ndarray, dis: np.ndarray, path_correct: np.ndarray, agg_correct: np.ndarray, s_correct: np.ndarray,
              excess: float) -> dict:
    """§7.7（與 RQ1-K 相同）：H2 ∩ 子集上的 d、c、m、headroom、recovery；回傳恆等式誤差與是否無定義。"""
    dis = in2 & dis
    d = dis.sum() / in2.sum()
    if dis.any():
        c, m = path_correct.any(axis=0)[dis].mean(), path_correct.mean(axis=0)[dis].mean()
    else:
        c = m = float("nan")
    if dis.any() and c > m:
        rv = (agg_correct[dis].mean() - m) / (c - m)
        rb = (s_correct[dis].mean() - m) / (c - m)
        return {"d": d, "c": c, "m": m, "headroom": d * (c - m), "recovery_V": rv, "recovery_blind": rb,
                "error": abs(excess - d * (c - m) * (rv - rb)), "undefined": False}
    return {"d": d, "c": c, "m": m, "headroom": d * (c - m) if dis.any() else 0.0, "recovery_V": float("nan"),
            "recovery_blind": float("nan"), "error": abs(excess), "undefined": True}


def evaluateBlock(block: PathBlock, splits: np.ndarray) -> dict:
    """
    一個區塊、200 次切分。回傳：
      menus：每份菜單的 A_V、S_in（正確率）、Excess_in（百分點）與 S_in 前兩名，以及（M12−p）不切分的差值
      block：判定一、二與 §7 的區塊層級量（百分點）、p* 與 13 選一的選擇比例、分解、恆等式最大誤差
    """
    sub = allAnswered(block, M12)
    correct = {name: menuCorrect(block, codes) for name, codes in MENU_CODES.items()}
    agree = {name: agreeMask(block, MENU_CODES[name]) for name in DECOMPOSED}
    path_correct = {name: np.array([block.correct[c] for c in MENU_CODES[name]]) for name in DECOMPOSED}

    per_split = {key: [] for key in ("prune1", "prune13", "dA", "dS", "dExcess", "AV10_minus_S12")}
    menu_values = {name: {"A_V": [], "S_in": []} for name in MENU_NAMES}
    s_choice = {name: Counter() for name in MENU_NAMES}
    removed, chose_none = Counter(), 0
    parts = {name: {key: [] for key in DECOMPOSITION} for name in DECOMPOSED}
    undefined, max_error = Counter(), 0.0
    for h1 in splits:
        in1, in2 = h1 & sub, ~h1 & sub
        counts = {name: int(correct[name][in1].sum()) for name in MENU_NAMES}
        acc = {name: float(correct[name][in2].mean()) for name in MENU_NAMES}
        s_in = {}
        for name, codes in MENU_CODES.items():
            s_code = strongest({c: int(block.correct[c][in1].sum()) for c in codes})
            s_in[name] = s_code
            s_choice[name][s_code] += 1
            menu_values[name]["A_V"].append(acc[name])
            menu_values[name]["S_in"].append(float(block.correct[s_code][in2].mean()))
        p_star = pickRemoval(counts)
        removed[p_star] += 1
        none = counts["M12"] >= counts[DROP[p_star]]     # §7.4：平手時優先不拿
        chose_none += none
        S = {name: menu_values[name]["S_in"][-1] for name in ("M12", "M10")}
        per_split["prune1"].append(acc[DROP[p_star]] - acc["M12"])
        per_split["prune13"].append(0.0 if none else acc[DROP[p_star]] - acc["M12"])
        per_split["dA"].append(acc["M10"] - acc["M12"])
        per_split["dS"].append(S["M10"] - S["M12"])
        per_split["dExcess"].append((acc["M10"] - S["M10"]) - (acc["M12"] - S["M12"]))
        per_split["AV10_minus_S12"].append(acc["M10"] - S["M12"])
        for name in DECOMPOSED:
            excess = acc[name] - S[name]
            part = decompose(in2, ~agree[name], path_correct[name], correct[name], block.correct[s_in[name]], excess)
            max_error = max(max_error, part["error"])
            undefined[name] += part["undefined"]
            for key in DECOMPOSITION:
                parts[name][key].append(part[key])

    reps = len(splits)
    menus = {}
    for name in MENU_NAMES:
        a_v, s = np.mean(menu_values[name]["A_V"]), np.mean(menu_values[name]["S_in"])
        menus[name] = {"A_V": float(a_v), "S_in": float(s), "excess_in_pp": 100 * float(np.mean(
            np.array(menu_values[name]["A_V"]) - np.array(menu_values[name]["S_in"]))),
            "S_in_top": " / ".join(f"{code} {count / reps:.0%}" for code, count in s_choice[name].most_common(2))}
        if name != "M12":
            menus[name]["unsplit_vs_M12_pp"] = 100 * float(correct[name][sub].mean() - correct["M12"][sub].mean())

    def nanmean(values):
        values = np.asarray(values, dtype=float)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    block_row = {f"{key}_pp": 100 * float(np.mean(values)) for key, values in per_split.items()}
    block_row.update({"n_subset": int(sub.sum()), "none_share": chose_none / reps, "identity_max_error": max_error,
                      **{f"p_star_{p}": removed[p] / reps for p in M12},
                      **{f"p_star_{cat}": sum(removed[p] for p in codes) / reps for cat, codes in CATEGORIES.items()}})
    for name in DECOMPOSED:
        for key in DECOMPOSITION:
            scale = 100 if key == "headroom" else 1
            block_row[f"{name}_{key}"] = scale * nanmean(parts[name][key])
        block_row[f"{name}_undefined_splits"] = undefined[name]
    return {"menus": menus, "block": block_row}
