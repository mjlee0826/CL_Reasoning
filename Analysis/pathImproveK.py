from dataclasses import dataclass

import numpy as np

from Analysis.pathImprove import (M12, MENU_ORDER, STEP_PCT, TIE_TOL, MAX_INVALID, CodedBlock, Splits, roundPct, logOdds,
                                  pickStrongest, fraction)

# ------------------------------------------------------------------
# RQ3-GK（result/analysis/rq3gk/rq3gk_criteria.md）：RQ3-G 推廣到任意 K 條的菜單與三種平手規則。全程離線。
#   規則 A = 原本的平手順序（RQ3-G 的 V）；B = 反過來；C = 平分（正確答案在 m 個平手答案中時得 1/m）。
#   每題的得分乘上 SCALE（1 到 12 的最小公倍數）後是整數，p* 與「相同 gain」都用整數比較。
# ------------------------------------------------------------------
KS = [3, 5, 7, 9, 11, 12]
RANDOM_KS = [3, 5, 7, 9]
N_MENUS, MENU_SEED = 30, 0
SCALE = 27720
FIXED_MENUS = {name: list(MENU_ORDER[name]) for name in ("M3L", "M3S", "M3P")}
MAIN_RULES = ("A", "C")


def drawMenus() -> tuple[dict, dict]:
    """§3：一個 default_rng(0)，依 K = 3、5、7、9 的順序抽 rng.choice(12, K, replace=False)，重複就重抽，各 30 組。"""
    rng = np.random.default_rng(MENU_SEED)
    menus, draws = {}, {}
    for K in RANDOM_KS:
        found, n = [], 0
        while len(found) < N_MENUS:
            n += 1
            m = tuple(sorted(int(i) for i in rng.choice(len(M12), size=K, replace=False)))
            if m not in found:
                found.append(m)
        menus[K] = [(f"K{K}-{j:02d}", [M12[i] for i in m]) for j, m in enumerate(found, 1)]
        draws[K] = n
    menus[11] = [(f"K11-{p}", [c for c in M12 if c != p]) for p in M12]
    menus[12] = [("M12", list(M12))]
    return menus, draws


@dataclass
class Ans:
    """一份菜單的逐題答案；leader[k, i] = 與 path k 答案相同的 path 中列號最小者（答案組的代表）。"""
    codes: np.ndarray
    gold: np.ndarray
    correct: np.ndarray
    eq: np.ndarray
    leader: np.ndarray

    @classmethod
    def of(cls, codes: np.ndarray, gold: np.ndarray) -> "Ans":
        same = codes[:, None, :] == codes[None, :, :]
        return cls(codes, gold, codes == gold[None, :], same.astype(float), same.argmax(axis=0))

    def withRow(self, k: int, row: np.ndarray) -> "Ans":
        codes = self.codes.copy()
        codes[k] = row
        return Ans.of(codes, self.gold)

    def scores(self, weights, rules) -> dict:
        """
        §4：每題的得分 × SCALE（int64）。加權分數在最高分 1e-9 以內的答案都是平手；
        A 取列號最小的候選（= Analysis.menuVote.vote），B 取列號最大的候選，C 平分。另回傳 m = 平手的答案數。
        """
        K, N = self.codes.shape
        score = np.tensordot(np.asarray(weights, dtype=float), self.eq, axes=(0, 0))
        cand = score >= score.max(axis=0) - TIE_TOL
        cols = np.arange(N)
        m = (cand & (self.leader == np.arange(K)[:, None])).sum(axis=0)
        out = {"m": m}
        for rule in rules:
            if rule == "A":
                out["A"] = self.correct[cand.argmax(axis=0), cols].astype(np.int64) * SCALE
            elif rule == "B":
                out["B"] = self.correct[K - 1 - cand[::-1].argmax(axis=0), cols].astype(np.int64) * SCALE
            elif rule == "C":
                out["C"] = np.where((cand & self.correct).any(axis=0), SCALE // m, 0).astype(np.int64)
            else:
                raise ValueError(rule)
        return out


def exactArgmax(num: np.ndarray, den: np.ndarray, reverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """每一列取 num/den 最大的位置（整數交叉相乘，精確）；平手取順序中最先的（reverse 時從最後一個往前）。另回傳是否平手。"""
    R, K = num.shape
    idx, tie = np.empty(R, dtype=int), np.zeros(R, dtype=bool)
    order = list(range(K - 1, -1, -1)) if reverse else list(range(K))
    for r, (nums, dens) in enumerate(zip(num.tolist(), den.tolist())):
        best, a0, b0, ties = None, 0, 1, 0
        for j in order:
            a, b = nums[j], dens[j]
            if best is None or a * b0 > a0 * b:
                best, a0, b0, ties = j, a, b, 1
            elif a * b0 == a0 * b:
                ties += 1
        idx[r], tie[r] = best, ties > 1
    return idx, tie


def atLeast(acc: np.ndarray, m: int) -> np.ndarray:
    """§9.2：Poisson-binomial，各 path 獨立、答對機率 acc[:, k]；回傳「至少 m 條答對」的機率（每列一個）。"""
    R, K = acc.shape
    dist = np.zeros((R, K + 1))
    dist[:, 0] = 1.0
    for k in range(K):
        a = acc[:, k:k + 1]
        nxt = dist * (1 - a)
        nxt[:, 1:] += dist[:, :-1] * a
        dist = nxt
    return dist[:, m:].sum(axis=1)


def _meanOrNan(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


# ------------------------------------------------------------------
# 沒有替換的量（16 個區塊，子集一）：判定二、§9.1 前半、§9.3–§9.5、§10.3 的第一部分
# ------------------------------------------------------------------
def menuBlock(block: CodedBlock, codes: list[str], splits: np.ndarray, rules=MAIN_RULES, alternatives: bool = False) -> dict:
    rows = [M12.index(c) for c in codes]
    K = len(rows)
    ans = Ans.of(block.codes[rows], block.gold)
    sp = Splits(ans.correct, block.sub, splits)
    R, ones = sp.reps, np.ones(K)
    needed = tuple(sorted(set(rules) | {"A", "C"}))
    base = ans.scores(ones, needed)
    fixed = [ans.withRow(j, block.gold).scores(ones, needed) for j in range(K)]
    h1, h2 = sp.h1.astype(np.int64), sp.h2.astype(np.int64)
    w1, w2 = sp.n1[:, None] - sp.k1, sp.n2[:, None] - sp.k2
    t1 = np.minimum(w1, roundPct(sp.n1, STEP_PCT)[:, None])
    t2 = np.minimum(w2, roundPct(sp.n2, STEP_PCT)[:, None])
    notie = block.sub & (base["m"] == 1)
    sb = np.array([pickStrongest(sp.k1[r], codes) for r in range(R)])
    SB = sp.k2[np.arange(R), sb] / sp.n2
    out = {"menu_codes": codes, "K": K, "n_subset": int(block.sub.sum()), "SB": float(SB.mean()),
           "tie_share": float(np.mean((h2 @ (base["m"] >= 2).astype(np.int64)) / sp.n2)),
           "C_vs_A_mismatch": int((base["C"][notie] != base["A"][notie]).sum()), "rules": {}}
    for rule in rules:
        delta = np.array([f[rule] - base[rule] for f in fixed])               # (K, N)
        if (delta < 0).any():
            raise ValueError(f"{block.model} | {block.dataset} | {codes} | rule {rule}: fixing a path lowered an item score")
        F1, F2 = h1 @ delta.T, h2 @ delta.T
        gain2 = 100 * np.divide((t2 * F2).astype(float), (w2 * sp.n2[:, None] * SCALE).astype(float),
                                out=np.zeros((R, K)), where=w2 > 0)
        pi2 = np.divide(F2.astype(float), (w2 * SCALE).astype(float), out=np.full((R, K), np.nan), where=w2 > 0)
        V = (h2 @ base[rule]) / (sp.n2 * SCALE)
        p_star, tie = exactArgmax(t1 * F1, np.where(w1 > 0, w1, 1), reverse=rule == "B")

        def d1(pick: np.ndarray) -> np.ndarray:
            chosen = gain2[np.arange(R), pick]
            return chosen - (gain2.sum(axis=1) - chosen) / (K - 1)

        pi_paths = [_meanOrNan(pi2[:, j]) for j in range(K)]
        res = {"D1": float(d1(p_star).mean()), "V": float(V.mean()), "VmSB_pp": 100 * float(np.mean(V - SB)),
               "pi_paths": dict(zip(codes, pi_paths)), "gain_paths": dict(zip(codes, gain2.mean(axis=0).tolist())),
               "pstar_share": dict(zip(codes, [float((p_star == j).mean()) for j in range(K)])),
               "pi_mean": _meanOrNan(pi_paths), "gain_mean": float(gain2.mean()), "gain_tie_share": float(tie.mean()),
               "pi_first": pi_paths[0], "pi_others": _meanOrNan(pi_paths[1:])}
        if alternatives:
            highest = sb                                                        # 選擇半答對最多（平手取順序較前）
            lowest = sp.k1.argmin(axis=1)                                       # 選擇半答對最少（平手取順序較前）
            first = np.zeros(R, dtype=int)
            res.update({"D1_highest": float(d1(highest).mean()), "D1_lowest": float(d1(lowest).mean()),
                        "D1_first": float(d1(first).mean()), "pstar_is_highest": float((p_star == highest).mean()),
                        "pstar_is_lowest": float((p_star == lowest).mean()), "pstar_is_first": float((p_star == first).mean())})
        out["rules"][rule] = res
    return out


def monteCarloC(block: CodedBlock, codes: list[str], path: str, splits: np.ndarray, draws: int = 2000, seed: int = 0) -> dict:
    """§10.3：規則 C、提高 5 個百分點；實際抽樣後用一般的計分函式（Ans.scores）算評分半的 V。"""
    rows = [M12.index(c) for c in codes]
    j = codes.index(path)
    ans = Ans.of(block.codes[rows], block.gold)
    sp = Splits(ans.correct, block.sub, splits)
    w1, w2 = sp.n1 - sp.k1[:, j], sp.n2 - sp.k2[:, j]
    t1, t2 = np.minimum(w1, roundPct(sp.n1, STEP_PCT)), np.minimum(w2, roundPct(sp.n2, STEP_PCT))
    ones = np.ones(len(rows))
    base = ans.scores(ones, ("C",))["C"]
    delta = ans.withRow(j, block.gold).scores(ones, ("C",))["C"] - base
    h2 = sp.h2.astype(np.int64)
    expected = float(np.mean((h2 @ base + fraction(t2, w2) * (h2 @ delta)) / (sp.n2 * SCALE)))

    rng = np.random.default_rng(seed)
    idx1 = [np.flatnonzero(sp.h1[r]) for r in range(sp.reps)]
    idx2 = [np.flatnonzero(sp.h2[r]) for r in range(sp.reps)]
    wrong1 = [i[~ans.correct[j, i]] for i in idx1]
    wrong2 = [i[~ans.correct[j, i]] for i in idx2]
    values = np.empty(draws)
    for d in range(draws):
        acc = np.empty(sp.reps)
        for r in range(sp.reps):
            menu_codes = ans.codes.copy()
            for wrong, t in ((wrong1[r], t1[r]), (wrong2[r], t2[r])):
                chosen = rng.choice(wrong, size=int(t), replace=False)
                menu_codes[j, chosen] = block.gold[chosen]
            part = Ans.of(menu_codes[:, idx2[r]], block.gold[idx2[r]])
            acc[r] = part.scores(ones, ("C",))["C"].mean() / SCALE
        values[d] = acc.mean()
    se = 0.0 if np.ptp(values) == 0 else float(values.std(ddof=1) / np.sqrt(draws))
    return {"expected": expected, "mc_mean": float(values.mean()), "mc_se": se}


# ------------------------------------------------------------------
# 替換（8 個弱模型區塊，子集二）：判定一、§9.1 後半、§9.2、§9.6、§9.8
# ------------------------------------------------------------------
class HostDonorK:
    """宿主 h × 供體 g × 資料集：子集二上的切分、兩者 12 條的答對題數、逐 path 的有效切分與是否參與判定。"""
    def __init__(self, host: CodedBlock, donor: CodedBlock, splits: np.ndarray):
        if host.dataset != donor.dataset or not np.array_equal(host.gold, donor.gold):
            raise ValueError(f"{host.model} / {donor.model} | {host.dataset}: gold differs")
        self.host, self.donor = host, donor
        self.sub = host.sub & donor.sub
        self.sp = Splits(host.correct, self.sub, splits)
        c = donor.correct.T.astype(np.int64)
        self.kd1, self.kd2 = self.sp.h1.astype(np.int64) @ c, self.sp.h2.astype(np.int64) @ c
        self.valid = (self.kd1 > self.sp.k1) & (self.kd2 > self.sp.k2)          # (R, 12)
        self.invalid = (~self.valid).sum(axis=0)
        self.participates = self.invalid <= MAX_INVALID


def substituteMenu(ctx: HostDonorK, codes: list[str], host_ans: Ans | None = None, rules=MAIN_RULES) -> list[dict]:
    """一份菜單在一個 (h, g, 資料集) 的所有替換。每個替換回傳逐切分的陣列（呼叫端依 valid 取平均）。"""
    host, donor, sp = ctx.host, ctx.donor, ctx.sp
    rows = [M12.index(c) for c in codes]
    K, R = len(rows), sp.reps
    ans = host_ans if host_ans is not None else Ans.of(host.codes[rows], host.gold)
    ones = np.ones(K)
    base = ans.scores(ones, rules)
    h1, h2 = sp.h1.astype(np.int64), sp.h2.astype(np.int64)
    n1, n2 = sp.n1, sp.n2
    k1m, k2m = sp.k1[:, rows], sp.k2[:, rows]
    V_before = {rule: (h2 @ base[rule]) / (n2 * SCALE) for rule in rules}
    threshold = K // 2 + 1
    acc_before = k2m / n2[:, None]
    P_before = atLeast(acc_before, threshold)
    out = []
    for j, p in enumerate(rows):
        fixed = ans.withRow(j, host.gold).scores(ones, rules)
        sub = ans.withRow(j, donor.codes[p])
        after = sub.scores(ones, rules)
        t1, t2 = ctx.kd1[:, p] - sp.k1[:, p], ctx.kd2[:, p] - sp.k2[:, p]
        w1, w2 = n1 - sp.k1[:, p], n2 - sp.k2[:, p]
        f1, f2 = fraction(t1, w1), fraction(t2, w2)
        hits1, hits2 = k1m.copy(), k2m.copy()
        hits1[:, j], hits2[:, j] = ctx.kd1[:, p], ctx.kd2[:, p]
        sb = np.array([pickStrongest(hits1[r], codes) for r in range(R)])
        SB_after = hits2[np.arange(R), sb] / n2
        WV_after = {rule: np.empty(R) for rule in rules}
        for r in range(R):
            sc = sub.scores(logOdds(hits1[r], n1[r]), rules)
            for rule in rules:
                WV_after[rule][r] = (h2[r] @ sc[rule]) / (n2[r] * SCALE)
        acc_after = acc_before.copy()
        acc_after[:, j] = ctx.kd2[:, p] / n2
        others1 = np.delete(k1m, j, axis=1).max(axis=1)
        res = {"path": codes[j], "valid": ctx.valid[:, p], "invalid_splits": int(ctx.invalid[p]),
               "participates": bool(ctx.participates[p]),
               "path_inc": (ctx.kd2[:, p] - sp.k2[:, p]) / n2, "SB_after": SB_after,
               "lead_H1_pp": 100 * (ctx.kd1[:, p] - others1) / n1,
               "P_before": P_before, "P_after": atLeast(acc_after, threshold), "rules": {}}
        for rule in rules:
            delta = fixed[rule] - base[rule]
            if (delta < 0).any():
                raise ValueError(f"{host.model} | {host.dataset} | {codes} | {codes[j]} | rule {rule}: fixing lowered a score")
            A_real = (h2 @ after[rule]) / (n2 * SCALE)
            A_sim = (h2 @ base[rule] + f2 * (h2 @ delta)) / (n2 * SCALE)
            res["rules"][rule] = {
                "V_before": V_before[rule], "A_real": A_real, "A_sim": A_sim,
                "actual_gain": (h2 @ after[rule] - h2 @ base[rule]) / (n2 * SCALE),
                "pred_H1": np.divide(t1 * (h1 @ delta).astype(float), (w1 * n1 * SCALE).astype(float),
                                     out=np.zeros(R), where=w1 > 0),
                "VmSB_after_pp": 100 * (A_real - SB_after), "WVmSB_after_pp": 100 * (WV_after[rule] - SB_after),
                "WV_after": WV_after[rule]}
        out.append(res)
    return out
