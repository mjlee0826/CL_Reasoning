from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.stats import rankdata

from Analysis.menuVote import PathBlock, MENUS, ordered, vote
from Analysis.probe import strongest

# ------------------------------------------------------------------
# RQ3-G（result/analysis/rq3g/rq3g_criteria.md）：改進一條 path，聚合會多多少？全程離線，資料、多數決、切分都沿用 RQ3。
#   答案編成整數：這四個資料集的 compareTwoAnswer 是字串相等（載入時逐對核對）。
#   每次切分：選擇半 = H1 ∩ 子集、評分半 = H2 ∩ 子集；正確率都在評分半，需要標註的選擇與權重都只用選擇半。
# ------------------------------------------------------------------
M12 = ordered(MENUS["M12"])          # 優先順序 EN ES JA P1 P2 R RU S1 S2 W1 W2 ZH，也是逐題陣列的列順序
MENU_ORDER = {"M12": M12, **{name: ordered(MENUS[name]) for name in ("M3L", "M3S", "M3P")}}
MAIN_MENU = "M12"
HOSTS = ["gpt4omini", "qwen"]
DONORS = ["deepseek4.1flash", "gemini3.1flashlite"]
STEP_PCT = 5                         # 判定一：提高 5 個百分點（§5）
CURVE_TARGETS = ["+0", "+2", "+5", "+10", "+20", "100%"]   # §8.3
TIE_TOL = 1e-9                       # 加權投票：分數差在 1e-9 以內視為平手（§3）
DS_ROUNDS = 20
MAX_INVALID = 40                     # 排除比例超過 20%（200 次中超過 40 次）的替換不參與判定（§6）
THRESHOLD = 0.5                      # 百分點（§7）


@dataclass
class CodedBlock:
    """一個區塊（模型 × 資料集）的 M12 逐題答案編碼；列依 M12 的優先順序，欄依 item_id（與切分位置對齊）。"""
    model: str
    dataset: str
    item_ids: np.ndarray
    codes: np.ndarray        # (12, N) 答案代碼
    gold: np.ndarray         # (N,) gold 代碼
    correct: np.ndarray      # (12, N)
    sub: np.ndarray          # (N,) 子集一：12 條都有答案


def encodeBlocks(blocks: list[PathBlock]) -> list[CodedBlock]:
    """同一資料集的各模型共用一份答案編碼。核對題目 id 與 gold 相同、compareTwoAnswer 等於字串相等、代碼的對錯 = block.correct。"""
    ref = blocks[0]
    vocab: dict = {}
    code = lambda answer: vocab.setdefault(answer, len(vocab))
    for b in blocks[1:]:
        if b.dataset != ref.dataset or not np.array_equal(b.item_ids, ref.item_ids) or b.gold != ref.gold:
            raise ValueError(f"{b.model} | {b.dataset}: item ids or gold differ from {ref.model}")
    gold = np.array([code(g) for g in ref.gold])
    out = []
    for b in blocks:
        codes = np.array([[code(a) for a in b.answers[c]] for c in M12])
        correct = codes == gold[None, :]
        for k, c in enumerate(M12):
            if not np.array_equal(correct[k], b.correct[c]):
                raise ValueError(f"{b.model} | {b.dataset} | {c}: encoded correctness != compareTwoAnswer")
        sub = np.all([b.answered[c] for c in M12], axis=0)
        out.append(CodedBlock(b.model, b.dataset, b.item_ids, codes, gold, correct, sub))
    words = list(vocab)
    for i, a in enumerate(words):
        for b in words[i:]:
            if bool(ref.compare(a, b)) != (a == b):
                raise ValueError(f"{ref.dataset}: compareTwoAnswer({a!r}, {b!r}) is not string equality")
    return out


def checkVote(block: PathBlock, coded: CodedBlock):
    """內部核對：每份菜單的整數投票逐題等於 Analysis.menuVote.vote。"""
    for menu, codes in MENU_ORDER.items():
        rows = [M12.index(c) for c in codes]
        mine = Answers.of(coded.codes[rows], coded.gold).voteCorrect()
        theirs = np.array([block.compare(g, vote([block.answers[c][k] for c in codes], [True] * len(codes), block.compare)[0])
                           for k, g in enumerate(block.gold)])
        if not np.array_equal(mine, theirs):
            raise ValueError(f"{block.model} | {block.dataset} | {menu}: integer vote != Analysis.menuVote.vote")


# ------------------------------------------------------------------
# 聚合（§3）
# ------------------------------------------------------------------
def logOdds(hits, n):
    """max(0, ln(p̂ / (1 − p̂)))，p̂ = (hits + 1) / (n + 2)。WV 的權重與 DS 的權重都用它。"""
    hits = np.asarray(hits, dtype=float)
    return np.maximum(0.0, np.log((hits + 1) / (n - hits + 1)))


def winner(eq: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    加權投票：eq[j, k, i] = path j 與 path k 在第 i 題答案相同。path k 的分數 = 它的答案的權重總和。
    分數在最高分 1e-9 以內的 path 都是候選；取列號最小的候選（列依優先順序，所以它就是平手答案中最先出現的那一個，
    與 Analysis.menuVote.vote 相同）。回傳每題勝出的 path 列號。等權投票 = weights 全為 1。
    """
    score = np.tensordot(weights, eq, axes=(0, 0))
    return (score >= score.max(axis=0) - TIE_TOL).argmax(axis=0)


@dataclass
class Answers:
    """一份菜單的逐題答案（K 條 path）與它們兩兩是否相同。"""
    codes: np.ndarray        # (K, N)
    gold: np.ndarray
    correct: np.ndarray      # (K, N)
    eq: np.ndarray           # (K, K, N) float

    @classmethod
    def of(cls, codes: np.ndarray, gold: np.ndarray) -> "Answers":
        return cls(codes, gold, codes == gold[None, :], (codes[:, None, :] == codes[None, :, :]).astype(float))

    def withRow(self, k: int, row: np.ndarray) -> "Answers":
        codes = self.codes.copy()
        codes[k] = row
        return Answers.of(codes, self.gold)

    def aggregateCorrect(self, weights) -> np.ndarray:
        return self.correct[winner(self.eq, np.asarray(weights, dtype=float)), np.arange(self.codes.shape[1])]

    def voteCorrect(self) -> np.ndarray:
        return self.aggregateCorrect(np.ones(len(self.codes)))


def dawidSkene(ans: Answers, items: np.ndarray) -> tuple[float, bool]:
    """§3 的 DS：只用 items（評分半）的題目，不用 gold。回傳 (正確率, 是否收斂)。"""
    idx = np.flatnonzero(items)
    codes, eq, cols, m = ans.codes[:, idx], ans.eq[:, :, idx], np.arange(len(idx)), len(idx)
    consensus = codes[winner(eq, np.ones(len(codes))), cols]          # 第 0 輪 = V
    for _ in range(DS_ROUNDS):
        agree = (codes == consensus[None, :]).sum(axis=1)
        new = codes[winner(eq, logOdds(agree, m)), cols]
        if np.array_equal(new, consensus):
            return float((new == ans.gold[idx]).mean()), True
        consensus = new
    return float((consensus == ans.gold[idx]).mean()), False


class Splits:
    """一份答案在某個子集上的 200 次切分：兩半的題數與各 path 的答對題數。"""
    def __init__(self, correct: np.ndarray, subset: np.ndarray, splits: np.ndarray):
        self.h1, self.h2 = splits & subset[None, :], ~splits & subset[None, :]
        self.n1, self.n2 = self.h1.sum(axis=1), self.h2.sum(axis=1)
        c = correct.T.astype(np.int64)
        self.k1, self.k2 = self.h1.astype(np.int64) @ c, self.h2.astype(np.int64) @ c     # (R, K)
        self.reps = len(splits)


def roundPct(n: np.ndarray, pct: int) -> np.ndarray:
    """⌊pct/100 × n + 0.5⌋，用整數算，沒有浮點誤差。"""
    return (2 * pct * np.asarray(n, dtype=np.int64) + 100) // 200


def pickStrongest(hits: np.ndarray, names: list[str]) -> int:
    """SB：答對題數最多的 path，平手依優先順序（Analysis.probe.strongest）。"""
    return names.index(strongest(dict(zip(names, (int(h) for h in hits)))))


# ------------------------------------------------------------------
# 隨機改進模型（§4）
# ------------------------------------------------------------------
class PathFix:
    """把 ans 的第 p 條改成正確答案的版本，與 V 在兩者下的逐題對錯。"""
    def __init__(self, ans: Answers, p: int):
        self.ans, self.p = ans, p
        self.fixed = ans.withRow(p, ans.gold)
        self.wrong = ~ans.correct[p]
        self.V0, self.V1 = ans.voteCorrect(), self.fixed.voteCorrect()
        if (self.V0 & ~self.V1).any():
            raise ValueError("fixing a path turned the majority vote from right to wrong (§4 says this cannot happen)")
        self.flips = (self.V1 & ~self.V0).astype(np.int64)          # 只會落在 p 答錯的題目


def fraction(t, w) -> np.ndarray:
    """f = t / w；w = 0 時 f = 0。"""
    t, w = np.asarray(t, dtype=float), np.asarray(w, dtype=float)
    return np.divide(t, w, out=np.zeros_like(t), where=w > 0)


def expectedV(fix: PathFix, half: np.ndarray, n: np.ndarray, t, w) -> np.ndarray:
    """某一半的 V 期望正確率 = (V 答對題數 + f × 改對後由錯變對的題數) / n。"""
    i64 = half.astype(np.int64)
    return (i64 @ fix.V0.astype(np.int64) + fraction(t, w) * (i64 @ fix.flips)) / n


def expectedAll(fix: PathFix, sp: Splits, t1, t2, names: list[str]) -> dict:
    """
    §4：p 在選擇半改對 t1 題、評分半改對 t2 題時，評分半上 V、WV、SB 的期望正確率與 p 的正確率（每次切分一個值）。
    WV、SB 的規則由選擇半改進後的答對題數 k1 + t1 決定；在 p 答對的題目上也用新的規則。
    """
    p, R = fix.p, sp.reps
    w2 = sp.n2 - sp.k2[:, p]
    f2 = fraction(t2, w2)
    out = {"V": expectedV(fix, sp.h2, sp.n2, t2, w2), "path": (sp.k2[:, p] + t2) / sp.n2,
           "WV": np.empty(R), "SB": np.empty(R)}
    wrong = fix.wrong.astype(np.int64)
    for r in range(R):
        hits = sp.k1[r].copy()
        hits[p] += t1[r]
        weights = logOdds(hits, sp.n1[r])
        a0, a1 = fix.ans.aggregateCorrect(weights), fix.fixed.aggregateCorrect(weights)
        h2 = sp.h2[r].astype(np.int64)
        out["WV"][r] = (h2 @ a0 + f2[r] * (h2 @ ((a1.astype(np.int64) - a0) * wrong))) / sp.n2[r]
        s = pickStrongest(hits, names)
        out["SB"][r] = (sp.k2[r, p] + t2[r] if s == p else sp.k2[r, s]) / sp.n2[r]
    return out


# ------------------------------------------------------------------
# 判定一（§5）與 §8.1–§8.3、§8.9 的判定一部分
# ------------------------------------------------------------------
def judgment1(block: CodedBlock, menu: str, splits: np.ndarray) -> dict:
    """一個區塊 × 菜單：200 次切分的 D1、p*、逐 path 的 π_p / 正確率 / gain_2；M12 另回傳 A_V 與 SB（§9.1）。"""
    names = MENU_ORDER[menu]
    rows = [M12.index(c) for c in names]
    ans = Answers.of(block.codes[rows], block.gold)
    sp = Splits(ans.correct, block.sub, splits)
    K, R = len(names), sp.reps
    flips = np.array([PathFix(ans, j).flips for j in range(K)])                 # (K, N)
    F1, F2 = sp.h1.astype(np.int64) @ flips.T, sp.h2.astype(np.int64) @ flips.T
    w1, w2 = sp.n1[:, None] - sp.k1, sp.n2[:, None] - sp.k2
    t1 = np.minimum(w1, roundPct(sp.n1, STEP_PCT)[:, None])
    t2 = np.minimum(w2, roundPct(sp.n2, STEP_PCT)[:, None])
    gain2 = 100 * np.divide((t2 * F2).astype(float), (w2 * sp.n2[:, None]).astype(float),
                            out=np.zeros((R, K)), where=w2 > 0)
    pi2 = np.divide(F2.astype(float), w2.astype(float), out=np.full((R, K), np.nan), where=w2 > 0)
    acc2 = sp.k2 / sp.n2[:, None]

    p_star, tie = np.empty(R, dtype=int), np.zeros(R, dtype=bool)
    for r in range(R):
        # gain_1 ∝ t1 × F1 / w1（n1 相同），用有理數比較
        values = [Fraction(int(t1[r, j] * F1[r, j]), int(w1[r, j])) if w1[r, j] > 0 else Fraction(0) for j in range(K)]
        best = max(values)
        tops = [j for j, v in enumerate(values) if v == best]
        p_star[r], tie[r] = tops[0], len(tops) > 1
    pick = gain2[np.arange(R), p_star]
    others = (gain2.sum(axis=1) - pick) / (K - 1)
    out = {
        "D1": float(np.mean(pick - others)), "gain_pstar": float(pick.mean()), "gain_others": float(others.mean()),
        "tie_share": float(tie.mean()), "n_subset": int(block.sub.sum()),
        "paths": {c: {"acc_H2": float(acc2[:, j].mean()), "pi_H2": float(np.nanmean(pi2[:, j])) if np.isfinite(pi2[:, j]).any()
                      else float("nan"), "pi_undefined_splits": int(np.isnan(pi2[:, j]).sum()),
                      "gain_H2_pp": float(gain2[:, j].mean()), "p_star_share": float((p_star == j).mean())}
                  for j, c in enumerate(names)},
    }
    if menu == MAIN_MENU:
        V0 = ans.voteCorrect().astype(np.int64)
        sb = np.array([sp.k2[r, pickStrongest(sp.k1[r], names)] for r in range(R)]) / sp.n2
        A_V = (sp.h2.astype(np.int64) @ V0) / sp.n2
        out.update(A_V=float(A_V.mean()), SB=float(sb.mean()), excess_pp=100 * float(np.mean(A_V - sb)))
    return out


def curves(block: CodedBlock, splits: np.ndarray) -> list[dict]:
    """§8.3：M12、子集一，每條 path × 目標的 V、WV、SB 期望正確率（評分半，200 次切分平均）。"""
    ans = Answers.of(block.codes, block.gold)
    sp = Splits(ans.correct, block.sub, splits)
    rows = []
    for p, code in enumerate(M12):
        fix = PathFix(ans, p)
        w1, w2 = sp.n1 - sp.k1[:, p], sp.n2 - sp.k2[:, p]
        for target in CURVE_TARGETS:
            if target == "100%":
                t1, t2 = w1, w2
            else:
                pct = int(target[1:])
                t1, t2 = roundPct(sp.n1, pct), roundPct(sp.n2, pct)
            short = int(((t1 > w1) | (t2 > w2)).sum())
            row = {"path": code, "target": target, "skipped": short > 0, "insufficient_splits": short}
            if short == 0:
                e = expectedAll(fix, sp, t1, t2, M12)
                row.update({"path_acc_H2": float(e["path"].mean()), "V": float(e["V"].mean()),
                            "WV": float(e["WV"].mean()), "SB": float(e["SB"].mean())})
            rows.append(row)
    return rows


def monteCarlo(block: CodedBlock, splits: np.ndarray, path: str = "ZH", draws: int = 2000, seed: int = 0) -> dict:
    """
    §9.2：實際抽樣。每一次依序對 200 次切分，先選擇半、後評分半，從 path 答錯的題目中不放回抽 t_h 題改成正確答案，
    再用一般的聚合函式（winner、strongest；不經期望值的算法）算評分半的 V、WV、SB。回傳期望值、抽樣平均與標準誤。
    """
    p = M12.index(path)
    ans = Answers.of(block.codes, block.gold)
    sp = Splits(ans.correct, block.sub, splits)
    w1, w2 = sp.n1 - sp.k1[:, p], sp.n2 - sp.k2[:, p]
    t1, t2 = np.minimum(w1, roundPct(sp.n1, STEP_PCT)), np.minimum(w2, roundPct(sp.n2, STEP_PCT))
    e = expectedAll(PathFix(ans, p), sp, t1, t2, M12)
    expected = {key: float(e[key].mean()) for key in ("V", "WV", "SB")}

    rng = np.random.default_rng(seed)
    h1 = [np.flatnonzero(sp.h1[r]) for r in range(sp.reps)]
    h2 = [np.flatnonzero(sp.h2[r]) for r in range(sp.reps)]
    wrong1 = [i[~block.correct[p, i]] for i in h1]
    wrong2 = [i[~block.correct[p, i]] for i in h2]
    ones = np.ones(len(M12))
    values = np.empty((draws, 3))
    for d in range(draws):
        acc = np.empty((sp.reps, 3))
        for r in range(sp.reps):
            codes = block.codes.copy()
            for wrong, t in ((wrong1[r], t1[r]), (wrong2[r], t2[r])):
                chosen = rng.choice(wrong, size=int(t), replace=False)
                codes[p, chosen] = block.gold[chosen]
            correct = codes == block.gold[None, :]
            hits = correct[:, h1[r]].sum(axis=1)
            c2, g2 = codes[:, h2[r]], block.gold[h2[r]]
            eq = (c2[:, None, :] == c2[None, :, :]).astype(float)
            cols = np.arange(len(h2[r]))
            acc[r] = [(c2[winner(eq, ones), cols] == g2).mean(), (c2[winner(eq, logOdds(hits, len(h1[r]))), cols] == g2).mean(),
                      (c2[pickStrongest(hits, M12)] == g2).mean()]
        values[d] = acc.mean(axis=0)
    mean, se = values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(draws)
    # 2,000 次的值完全相同時標準誤就是 0（np.std 對相同的浮點數會因捨入給出約 1e-16，不能拿來當 0 判斷）
    se = np.where(np.ptp(values, axis=0) == 0, 0.0, se)
    return {"expected": expected, "mc_mean": dict(zip(("V", "WV", "SB"), mean.tolist())),
            "mc_se": dict(zip(("V", "WV", "SB"), se.tolist()))}


# ------------------------------------------------------------------
# 替換（§6、§8.4–§8.8、§8.9 的判定二部分）
# ------------------------------------------------------------------
class Substitution:
    """宿主 h × 供體 g × 資料集：子集二、替換前的量（每次切分一個值），與逐 path 的替換。"""
    def __init__(self, host: CodedBlock, donor: CodedBlock, splits: np.ndarray):
        if host.dataset != donor.dataset or not np.array_equal(host.gold, donor.gold):
            raise ValueError(f"{host.model} / {donor.model} | {host.dataset}: gold differs")
        self.host, self.donor = host, donor
        self.sub = host.sub & donor.sub
        self.ans = Answers.of(host.codes, host.gold)
        self.sp = Splits(self.ans.correct, self.sub, splits)
        self.kd1 = self.sp.h1.astype(np.int64) @ donor.correct.T.astype(np.int64)
        self.kd2 = self.sp.h2.astype(np.int64) @ donor.correct.T.astype(np.int64)
        self.before = self._measure(self.ans, self.sp.k1, self.sp.k2)

    @staticmethod
    def rows(menu: str) -> list[int]:
        return [M12.index(c) for c in MENU_ORDER[menu]]

    def _measure(self, ans: Answers, k1: np.ndarray, k2: np.ndarray) -> dict:
        """一份 M12 答案在評分半的 V、WV、SB、DS、Oracle（每次切分一個值；SB 的答對題數另外回傳）。"""
        sp, R = self.sp, self.sp.reps
        h2 = sp.h2.astype(np.int64)
        V = ans.voteCorrect().astype(np.int64)
        out = {"V_count": h2 @ V, "V": (h2 @ V) / sp.n2, "Oracle": (h2 @ ans.correct.any(axis=0)) / sp.n2,
               "WV_count": np.empty(R, dtype=np.int64), "SB_count": np.empty(R, dtype=np.int64),
               "DS": np.empty(R), "DS_converged": np.empty(R, dtype=bool)}
        for r in range(R):
            out["WV_count"][r] = h2[r] @ ans.aggregateCorrect(logOdds(k1[r], sp.n1[r]))
            out["SB_count"][r] = k2[r, pickStrongest(k1[r], M12)]
            out["DS"][r], out["DS_converged"][r] = dawidSkene(ans, sp.h2[r])
        out["WV"], out["SB"] = out["WV_count"] / sp.n2, out["SB_count"] / sp.n2
        return out

    def substitute(self, p: int, full: bool = True) -> dict:
        """
        把宿主的第 p 條（M12 的列號）換成供體的同一條。回傳每次切分的 M12 量（full 時）與各菜單的 V 量，
        以及在整個子集二上數的修對 / 弄壞 / 錯換錯題數。無效切分不遮掉，由呼叫端依 valid 取用。
        """
        host, donor, sp = self.host, self.donor, self.sp
        code = M12[p]
        t1, t2 = self.kd1[:, p] - sp.k1[:, p], self.kd2[:, p] - sp.k2[:, p]
        w1, w2 = sp.n1 - sp.k1[:, p], sp.n2 - sp.k2[:, p]
        hc, dc = host.correct[p], donor.correct[p]
        out = {"valid": (t1 > 0) & (t2 > 0), "t1": t1, "t2": t2,
               "n_fixed": int((self.sub & ~hc & dc).sum()), "n_broken": int((self.sub & hc & ~dc).sum()),
               "n_wrong_to_other_wrong": int((self.sub & ~hc & ~dc & (host.codes[p] != donor.codes[p])).sum()),
               "host_acc": sp.k2[:, p] / sp.n2, "donor_acc": self.kd2[:, p] / sp.n2, "menus": {}}
        h1, h2 = sp.h1.astype(np.int64), sp.h2.astype(np.int64)
        for menu, names in MENU_ORDER.items():
            if code not in names:
                continue
            j = names.index(code)
            menu_ans = Answers.of(host.codes[self.rows(menu)], host.gold)
            fix = PathFix(menu_ans, j)
            after = menu_ans.withRow(j, donor.codes[p]).voteCorrect().astype(np.int64)
            out["menus"][menu] = {"V_before": (h2 @ fix.V0) / sp.n2, "A_real": (h2 @ after) / sp.n2,
                                  "A_sim": expectedV(fix, sp.h2, sp.n2, t2, w2)}
            if menu != MAIN_MENU or not full:
                continue
            m = out["menus"][menu]
            m["real_gain_count"] = h2 @ after - h2 @ fix.V0.astype(np.int64)        # 第 8 節 6、7：整數分子，避免假的不平手
            m["pred_H1"] = np.divide((t1 * (h1 @ fix.flips)).astype(float), (w1 * sp.n1).astype(float),
                                     out=np.zeros(sp.reps), where=w1 > 0)          # = 正確率增加量 × π_p（選擇半）
            # 只修不壞：只把「宿主錯、供體對」的題目換成供體的答案（= 正確答案）
            fixmask = ~hc & dc
            fix_only = np.where(fixmask, fix.V1, fix.V0).astype(np.int64)
            m["A_fix"] = (h2 @ fix_only) / sp.n2
            m["A_fix_sim"] = expectedV(fix, sp.h2, sp.n2, h2 @ fixmask.astype(np.int64), w2)
            sim = expectedAll(fix, sp, t1, t2, M12)
            m["WV_sim"], m["SB_sim"] = sim["WV"], sim["SB"]
            k1, k2 = sp.k1.copy(), sp.k2.copy()
            k1[:, p], k2[:, p] = self.kd1[:, p], self.kd2[:, p]
            m["after"] = self._measure(menu_ans.withRow(j, donor.codes[p]), k1, k2)
            others1 = np.delete(sp.k1, p, axis=1).max(axis=1)
            m["lead_H1_pp"] = 100 * (self.kd1[:, p] - others1) / sp.n1
            m["V_minus_SB_pp"] = 100 * (m["after"]["V_count"] - m["after"]["SB_count"]) / sp.n2
            m["WV_minus_SB_pp"] = 100 * (m["after"]["WV_count"] - m["after"]["SB_count"]) / sp.n2
        return out


def spearmanBySplit(x: np.ndarray, y: np.ndarray, use: np.ndarray) -> tuple[float, int]:
    """
    §8.7：x、y、use 都是 (替換數, 切分數)。每次切分對 use 為 True 的替換算 Spearman（平手用平均排名）；
    少於 3 個或任一邊全部相同時該次無定義。回傳 (有定義的切分的平均, 無定義的切分數)。
    """
    rhos, undefined = [], 0
    for r in range(x.shape[1]):
        m = use[:, r]
        a, b = x[m, r], y[m, r]
        if m.sum() < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
            undefined += 1
            continue
        rhos.append(float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]))
    return (float(np.mean(rhos)) if rhos else float("nan")), undefined
