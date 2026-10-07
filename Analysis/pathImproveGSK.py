import itertools
from math import comb

import numpy as np
import pandas as pd

from Analysis.pathImprove import M12, STEP_PCT, CodedBlock, Splits, roundPct, fraction
from Analysis.pathImproveK import SCALE, Ans, HostDonorK, drawMenus, exactArgmax
from Analysis.pathImproveGS import GROUPS

# ------------------------------------------------------------------
# RQ3-GSK（result/analysis/rq3gsk/rq3gsk_criteria.md）：把 RQ3-GS 的落單程度推廣到 K 條。全程離線。
#   s_i = 其他 K − 1 條彼此的一致率平均 − i 和其他 K − 1 條的一致率平均（只看答案，不用 gold）。
#   排名：s_i 由大到小，相同時平手順序在後的排前面（K = 3 時等於 RQ3-GS「最像的一對相同時取字典序在前的那一對」）。
#   第 1 名 = 落單的 path，第 K 名 = 最不落單的 path。
# ------------------------------------------------------------------
KS = (5, 7)
THRESHOLDS = {5: 0.30, 7: 0.20}       # 判定一、二的門檻（第 6 節）
REFERENCE_THRESHOLD = 0.5             # 第 8 節 10 的對照
TRANSLATED = {"ES", "JA", "RU", "ZH"}


def rankOrder(values) -> list[int]:
    """菜單內的位置依 s 由大到小排序；相同時位置（= 平手順序）在後的排前面。"""
    return sorted(range(len(values)), key=lambda i: (-values[i], -i))


def sScores(codes: list[str], rates: dict) -> np.ndarray:
    """菜單表的 s_i（比例，不是 pp）。rates 的鍵是依平手順序的 (a, b)，同 Analysis.pathImproveGS.agreementRates。"""
    rate = lambda a, b: rates[(a, b)] if (a, b) in rates else rates[(b, a)]
    out = []
    for i, c in enumerate(codes):
        others = [d for j, d in enumerate(codes) if j != i]
        among = np.mean([rate(a, b) for a, b in itertools.combinations(others, 2)])
        out.append(among - np.mean([rate(c, d) for d in others]))
    return np.array(out)


def menuRow(codes: list[str], rates: dict) -> dict:
    s = sScores(codes, rates)
    order = rankOrder(s.tolist())
    return {"codes": codes, "s": s, "lone": codes[order[0]], "least": codes[order[-1]], "degree": float(s[order[0]])}


def menuSetsK(K: int) -> tuple[list, list]:
    """第 4 節：(確認用的菜單, RQ3-GK 的菜單)，各為 [(編號, 代號串列)]；確認用的依平手順序索引的字典序編為 GS{K}-001 起。"""
    menus, _ = drawMenus()
    old = [(name, list(codes)) for name, codes in menus[K]]
    used = {tuple(sorted(M12.index(c) for c in codes)) for _, codes in old}
    conf = [m for m in itertools.combinations(range(len(M12)), K) if m not in used]
    return [(f"GS{K}-{n:03d}", [M12[i] for i in m]) for n, m in enumerate(conf, 1)], old


def menuTableFrame(K: int, rates: dict) -> tuple[pd.DataFrame, list, list, float, float]:
    """
    第 4 節的 792 種菜單表（含扣掉的），欄位與排序同 rq3gsk_menus_K{K}.csv；另回傳確認用與扣掉的菜單（含分組）與兩個門檻。
    分組：確認用的依落單程度由高到低（相同時依編號）分三等份；扣掉的用門檻分組。
    """
    conf, old = menuSetsK(K)
    conf_rows = [{"mid": mid, **menuRow(codes, rates), "confirm": True} for mid, codes in conf]
    old_rows = [{"mid": mid, **menuRow(codes, rates), "confirm": False} for mid, codes in old]
    n = len(conf_rows)
    third = n // 3
    order = sorted(range(n), key=lambda i: (-conf_rows[i]["degree"], i))
    for rank, i in enumerate(order):
        conf_rows[i]["group"] = GROUPS[0] if rank < third else (GROUPS[2] if rank >= n - third else GROUPS[1])
        conf_rows[i]["rank"] = rank + 1
    top_min = min(r["degree"] for r in conf_rows if r["group"] == GROUPS[0])
    bot_max = max(r["degree"] for r in conf_rows if r["group"] == GROUPS[2])
    for r in old_rows:
        r["group"] = GROUPS[0] if r["degree"] >= top_min else (GROUPS[2] if r["degree"] <= bot_max else GROUPS[1])
        r["rank"] = None
    rows = []
    for r in sorted(conf_rows, key=lambda r: r["mid"]) + sorted(old_rows, key=lambda r: r["mid"]):
        row = {"menu": r["mid"], "K": K, "codes": " ".join(r["codes"]), "confirmation": r["confirm"]}
        for c in M12:
            row[f"s_{c}_pp"] = 100 * float(r["s"][r["codes"].index(c)]) if c in r["codes"] else np.nan
        row.update({"lone": r["lone"], "least_lone": r["least"], "degree_pp": 100 * r["degree"], "group": r["group"],
                    "degree_rank": r["rank"] if r["rank"] else np.nan})
        rows.append(row)
    return pd.DataFrame(rows), conf_rows, old_rows, top_min, bot_max


def menuTableCsv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, float_format="%.10f")


def blockLone(block: CodedBlock, codes: list[str]) -> str:
    """第 8 節 9：用一個區塊子集一的全部題目算 s_i（整數題數），回傳落單的 path。"""
    rows = [M12.index(c) for c in codes]
    sub = block.sub
    a = {(i, j): int((block.codes[rows[i]][sub] == block.codes[rows[j]][sub]).sum()) for i, j in itertools.combinations(range(len(rows)), 2)}
    return codes[rankOrder(integerS(a, len(rows)).tolist())[0]]


def integerS(a: dict, K: int) -> np.ndarray:
    """
    第 3 節的整數版本：S_i = (K − 1) × Σ（其他條兩兩的一致題數）− C(K−1, 2) × Σ_j（i 和 j 的一致題數）。
    a[(i, j)]（i < j）可以是純量或 (R,) 陣列；回傳 (K,) 或 (R, K)。S_i 和 s_i 只差所有 path 共用的正數倍數。
    """
    out = []
    for i in range(K):
        others = [j for j in range(K) if j != i]
        among = sum(a[(p, q)] for p, q in itertools.combinations(others, 2))
        withi = sum(a[(min(i, j), max(i, j))] for j in others)
        out.append((K - 1) * np.asarray(among, dtype=np.int64) - comb(K - 1, 2) * np.asarray(withi, dtype=np.int64))
    return np.stack(out, axis=-1)


def splitRanks(S: np.ndarray) -> np.ndarray:
    """S (R, K) -> 名次 (R, K)，1 = 最落單；s 相同時平手順序在後的排前面。"""
    R, K = S.shape
    ranks = np.empty((R, K), dtype=int)
    for r, row in enumerate(S.tolist()):
        for place, i in enumerate(rankOrder(row), 1):
            ranks[r, i] = place
    return ranks


# ------------------------------------------------------------------
# 宿主這一側（子集一）：每次切分在選擇半的名次、正確率最低的 path；評分半的 π；第 9 節 3 的 D1
# ------------------------------------------------------------------
def hostSide(block: CodedBlock, codes: list[str], splits: np.ndarray, pi_rules=("C",), d1_rules=()) -> dict:
    rows = [M12.index(c) for c in codes]
    K = len(rows)
    ans = Ans.of(block.codes[rows], block.gold)
    sp = Splits(ans.correct, block.sub, splits)
    h1, h2 = sp.h1.astype(np.int64), sp.h2.astype(np.int64)
    a = {(i, j): h1 @ (ans.codes[i] == ans.codes[j]).astype(np.int64) for i, j in itertools.combinations(range(K), 2)}
    ranks = splitRanks(integerS(a, K))
    out = {"ranks": ranks, "lone": ranks.argmin(axis=1), "lowest": sp.k1.argmin(axis=1), "pi2": {}, "D1": {}}
    rules = tuple(sorted(set(pi_rules) | set(d1_rules)))
    if not rules:
        return out
    ones = np.ones(K)
    base = ans.scores(ones, rules)
    fixed = [ans.withRow(j, block.gold).scores(ones, rules) for j in range(K)]
    w1, w2 = sp.n1[:, None] - sp.k1, sp.n2[:, None] - sp.k2
    R = sp.reps
    for rule in rules:
        delta = np.array([f[rule] - base[rule] for f in fixed])
        if (delta < 0).any():
            raise ValueError(f"{block.model} | {block.dataset} | {codes} | rule {rule}: fixing a path lowered an item score")
        F2 = h2 @ delta.T
        if rule in pi_rules:
            out["pi2"][rule] = np.divide(F2.astype(float), (w2 * SCALE).astype(float), out=np.full((R, K), np.nan), where=w2 > 0)
        if rule in d1_rules:
            F1 = h1 @ delta.T
            t1 = np.minimum(w1, roundPct(sp.n1, STEP_PCT)[:, None])
            t2 = np.minimum(w2, roundPct(sp.n2, STEP_PCT)[:, None])
            gain2 = 100 * np.divide((t2 * F2).astype(float), (w2 * sp.n2[:, None] * SCALE).astype(float),
                                    out=np.zeros((R, K)), where=w2 > 0)
            pstar, _ = exactArgmax(t1 * F1, np.where(w1 > 0, w1, 1))
            chosen = gain2[np.arange(R), pstar]
            out["D1"][rule] = float((chosen - (gain2.sum(axis=1) - chosen) / (K - 1)).mean())
    return out


# ------------------------------------------------------------------
# 第 9 節 3：RQ3-GK 的轉換率（逐替換：有效切分上的平均；只用排除比例 ≤ 20% 的替換）
# ------------------------------------------------------------------
def conversionParts(ctx: HostDonorK, codes: list[str], rules=("A", "C")) -> list[dict]:
    """每條 path：(是否參與, 有效切分上 A_real − V_before 的平均, A_sim − V_before 的平均, path 進步的平均)，依規則。"""
    host, donor, sp = ctx.host, ctx.donor, ctx.sp
    rows = [M12.index(c) for c in codes]
    ans = Ans.of(host.codes[rows], host.gold)
    ones = np.ones(len(rows))
    base = ans.scores(ones, rules)
    h2 = sp.h2.astype(np.int64)
    n2 = sp.n2
    out = []
    for j, p in enumerate(rows):
        valid = ctx.valid[:, p]
        after = ans.withRow(j, donor.codes[p]).scores(ones, rules)
        fixed = ans.withRow(j, host.gold).scores(ones, rules)
        t2, w2 = ctx.kd2[:, p] - sp.k2[:, p], n2 - sp.k2[:, p]
        f2 = fraction(t2, w2)
        mean = lambda x: float(np.mean(x[valid])) if valid.any() else float("nan")
        res = {"participates": bool(ctx.participates[p]), "path_inc": mean(t2 / n2)}
        for rule in rules:
            res[f"real_{rule}"] = mean((h2 @ (after[rule] - base[rule])) / (n2 * SCALE))
            res[f"sim_{rule}"] = mean(f2 * (h2 @ (fixed[rule] - base[rule])) / (n2 * SCALE))
        out.append(res)
    return out
