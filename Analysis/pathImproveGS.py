import itertools

import numpy as np

from Analysis.pathImprove import M12, STEP_PCT, CodedBlock, Splits, roundPct
from Analysis.pathImproveK import SCALE, FIXED_MENUS, Ans, HostDonorK, drawMenus, exactArgmax

# ------------------------------------------------------------------
# RQ3-GS（result/analysis/rq3gs/rq3gs_criteria.md）：在沒看過的三條組合上確認兩個預測。全程離線。
#   菜單的列依平手順序；三組兩兩的配對依字典序 (0,1)、(0,2)、(1,2)。
#   真實替換的效果 = 10 × Σ分子 ÷ Σ分母（評分半 ∩ 子集二；任一條 path 無效的（菜單、供體、切分）整個不計）。
# ------------------------------------------------------------------
RULES = ("C", "A")
PAIR_SLOTS = [(0, 1), (0, 2), (1, 2)]
GROUPS = ["明顯落單", "中間", "對稱"]


def menuSets() -> tuple[list, list]:
    """§3：(確認用的 187 份, 舊的 K3-01..K3-30)，各為 [(編號, 代號串列)]，代號依平手順序。"""
    menus, _ = drawMenus()
    old = [(name, list(codes)) for name, codes in menus[3]]
    used = {tuple(sorted(M12.index(c) for c in codes)) for _, codes in old}
    used |= {tuple(sorted(M12.index(c) for c in codes)) for codes in FIXED_MENUS.values()}
    conf = [m for m in itertools.combinations(range(len(M12)), 3) if m not in used]
    return [(f"GS-{n:03d}", [M12[i] for i in m]) for n, m in enumerate(conf, 1)], old


def agreementRates(blocks: list[CodedBlock]) -> dict:
    """§4：兩兩一致率 = 子集一中答案相同的比例；每個區塊算完再平均。只用答案代碼，不用 gold。"""
    rates = {}
    for a, b in itertools.combinations(range(len(M12)), 2):
        rates[(M12[a], M12[b])] = float(np.mean([(blk.codes[a][blk.sub] == blk.codes[b][blk.sub]).mean() for blk in blocks]))
    return rates


def menuGrouping(codes: list[str], rates: dict) -> dict:
    """§4：相像的一對（一致率最高；相同時字典序在前）、落單的 path、落單程度。"""
    vals = [rates[(codes[i], codes[j])] for i, j in PAIR_SLOTS]
    best = max(range(3), key=lambda t: (vals[t], -t))
    i, j = PAIR_SLOTS[best]
    lone = ({0, 1, 2} - {i, j}).pop()
    others = [vals[t] for t in range(3) if t != best]
    return {"rates": vals, "similar": (codes[i], codes[j]), "lone": codes[lone], "degree": vals[best] - float(np.mean(others))}


def assignGroups(groupings: list[dict]) -> tuple[list[str], float, float]:
    """依落單程度由高到低（相同時依編號）分三等份：前 ⌊n/3⌋ 份明顯落單、後 ⌊n/3⌋ 份對稱。回傳組別與兩個門檻。"""
    n = len(groupings)
    third = n // 3
    order = sorted(range(n), key=lambda i: (-groupings[i]["degree"], i))
    groups = [None] * n
    for rank, i in enumerate(order):
        groups[i] = GROUPS[0] if rank < third else (GROUPS[2] if rank >= n - third else GROUPS[1])
    top_min = min(g["degree"] for g, grp in zip(groupings, groups) if grp == GROUPS[0])
    bot_max = max(g["degree"] for g, grp in zip(groupings, groups) if grp == GROUPS[2])
    return groups, top_min, bot_max


def thresholdGroup(degree: float, top_min: float, bot_max: float) -> str:
    """§9.9：舊菜單用確認菜單的精確門檻分組。"""
    return GROUPS[0] if degree >= top_min else (GROUPS[2] if degree <= bot_max else GROUPS[1])


# ------------------------------------------------------------------
# 宿主這一側（子集一）：每次切分的 p*、各種挑法、落單的 path；評分半的 gain、π、D1
# ------------------------------------------------------------------
def hostSplits(block: CodedBlock, codes: list[str], splits: np.ndarray, rules=RULES) -> dict:
    rows = [M12.index(c) for c in codes]
    K = len(rows)
    ans = Ans.of(block.codes[rows], block.gold)
    sp = Splits(ans.correct, block.sub, splits)
    R, ones = sp.reps, np.ones(K)
    base = ans.scores(ones, rules)
    fixed = [ans.withRow(j, block.gold).scores(ones, rules) for j in range(K)]
    h1, h2 = sp.h1.astype(np.int64), sp.h2.astype(np.int64)
    w1, w2 = sp.n1[:, None] - sp.k1, sp.n2[:, None] - sp.k2
    t1 = np.minimum(w1, roundPct(sp.n1, STEP_PCT)[:, None])
    t2 = np.minimum(w2, roundPct(sp.n2, STEP_PCT)[:, None])
    out = {"pstar": {}, "gain2": {}, "pi2": {}, "D1": {}}
    for rule in rules:
        delta = np.array([f[rule] - base[rule] for f in fixed])
        if (delta < 0).any():
            raise ValueError(f"{block.model} | {block.dataset} | {codes} | rule {rule}: fixing a path lowered an item score")
        F1, F2 = h1 @ delta.T, h2 @ delta.T
        gain2 = 100 * np.divide((t2 * F2).astype(float), (w2 * sp.n2[:, None] * SCALE).astype(float),
                                out=np.zeros((R, K)), where=w2 > 0)
        pstar, _ = exactArgmax(t1 * F1, np.where(w1 > 0, w1, 1))
        chosen = gain2[np.arange(R), pstar]
        out["pstar"][rule], out["gain2"][rule] = pstar, gain2
        out["pi2"][rule] = np.divide(F2.astype(float), (w2 * SCALE).astype(float), out=np.full((R, K), np.nan), where=w2 > 0)
        out["D1"][rule] = chosen - (gain2.sum(axis=1) - chosen) / (K - 1)
    # 選擇半正確率最高 / 最低（相同時平手順序在前；列已依平手順序，argmax / argmin 取第一個）
    out["highest"], out["lowest"] = sp.k1.argmax(axis=1), sp.k1.argmin(axis=1)
    # 落單的 path：選擇半的整數一致題數，最高的一對相同時取字典序在前的
    agree = np.stack([h1 @ (ans.codes[i] == ans.codes[j]).astype(np.int64) for i, j in PAIR_SLOTS], axis=1)
    best = agree.argmax(axis=1)
    out["lone"] = np.array([({0, 1, 2} - set(PAIR_SLOTS[b])).pop() for b in best])
    out["simfirst"] = np.array([PAIR_SLOTS[b][0] for b in best])
    return out


def piFull(block: CodedBlock, codes: list[str]) -> np.ndarray:
    """§9.4：規則 C、子集一的全部題目（不切分）的 π，依菜單內的順序。"""
    rows = [M12.index(c) for c in codes]
    ans = Ans.of(block.codes[rows], block.gold)
    ones = np.ones(len(rows))
    base = ans.scores(ones, ("C",))["C"]
    out = []
    for j in range(len(rows)):
        wrong = block.sub & ~ans.correct[j]
        delta = ans.withRow(j, block.gold).scores(ones, ("C",))["C"] - base
        out.append(float(delta[wrong].sum() / (wrong.sum() * SCALE)) if wrong.any() else float("nan"))
    return np.array(out)


# ------------------------------------------------------------------
# 替換這一側（子集二）：每條 path、每次切分的分子與分母
# ------------------------------------------------------------------
def substitutionSplits(ctx: HostDonorK, codes: list[str], rules=RULES) -> dict:
    """
    回傳 num[rule] (R, 3)：替換後 V 的得分平均 − 替換前（評分半 ∩ 子集二）；num_int[rule]：同一個量 × n2 × SCALE（整數）；
    den (R, 3)：供體 − 宿主那條 path 的正確率；den_int：答對題數的差；valid (R, 3)：有效切分；included (R,)：三條都有效。
    """
    host, donor, sp = ctx.host, ctx.donor, ctx.sp
    rows = [M12.index(c) for c in codes]
    ans = Ans.of(host.codes[rows], host.gold)
    ones = np.ones(len(rows))
    base = ans.scores(ones, rules)
    h2 = sp.h2.astype(np.int64)
    num_int = {rule: np.empty((sp.reps, len(rows)), dtype=np.int64) for rule in rules}
    for j, p in enumerate(rows):
        after = ans.withRow(j, donor.codes[p]).scores(ones, rules)
        for rule in rules:
            num_int[rule][:, j] = h2 @ (after[rule] - base[rule])
    den_int = ctx.kd2[:, rows] - sp.k2[:, rows]
    valid = ctx.valid[:, rows]
    return {"num_int": num_int, "num": {r: v / (sp.n2[:, None] * SCALE) for r, v in num_int.items()},
            "den_int": den_int, "den": den_int / sp.n2[:, None], "valid": valid, "included": valid.all(axis=1)}


def onehot(idx: np.ndarray, K: int = 3) -> np.ndarray:
    out = np.zeros((len(idx), K), dtype=bool)
    out[np.arange(len(idx)), idx] = True
    return out


class GroupSums:
    """一群 path 的 Σ分子、Σ分母與替換數（依名稱累加）。效果 = 10 × Σ分子 ÷ Σ分母。"""
    def __init__(self):
        self.num, self.den, self.n = {}, {}, {}

    def add(self, name: str, num: np.ndarray, den: np.ndarray, mask: np.ndarray, rows: np.ndarray):
        """num、den、mask 都是 (R, 3)；rows (R,) = 這次要算的（菜單、供體、切分）。"""
        m = mask & rows[:, None]
        self.num[name] = self.num.get(name, 0.0) + float(num[m].sum())
        self.den[name] = self.den.get(name, 0.0) + float(den[m].sum())
        self.n[name] = self.n.get(name, 0) + int(m.sum())

    def effect(self, name: str) -> float:
        d = self.den.get(name, 0.0)
        return 10 * self.num[name] / d if d > 0 else float("nan")

    def diff(self, first: str, second: str) -> float:
        return self.effect(first) - self.effect(second)


def hitCredit(num_int: np.ndarray, den_int: np.ndarray, pick: np.ndarray, rows: np.ndarray, lowest: bool = False) -> tuple[float, int]:
    """
    §9.2：每個（菜單、供體、切分），每條 path 的真實效果 = num_int / (SCALE·den_int)（整數交叉相乘精確比較）；
    pick 剛好是最高（lowest 時為最低）那條時算 1，與 m 條相同時算 1/m。回傳 (總分, 次數)。
    """
    total, count = 0.0, 0
    for r in np.flatnonzero(rows):
        a, b = num_int[r].tolist(), den_int[r].tolist()
        best = [0]
        for j in (1, 2):
            cmp = a[j] * b[best[0]] - a[best[0]] * b[j]       # > 0：j 比較大
            if (cmp < 0) if lowest else (cmp > 0):
                best = [j]
            elif cmp == 0:
                best.append(j)
        total += (1 / len(best)) if pick[r] in best else 0.0
        count += 1
    return total, count
