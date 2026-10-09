from dataclasses import dataclass, field

import numpy as np

from Analysis.experimentPlan import PATHS
from Analysis.menuVote import DATASETS
from Analysis.pathImprove import M12, HOSTS, DONORS
from Analysis.judgeRegression import BASE, TRANS, SHORT, design, fitConditionalLogit, itemVector, halfMean
from Analysis.judgeSubstitution import MENU_ORDER
from Analysis.judgeSubstitutionPredict import voteCells, UNANIMOUS

# ------------------------------------------------------------------
# RQ3-GJT（result/analysis/rq3gjt/rq3gjt_criteria.md）：不用真的去改 path 的裁判預測。全程離線。
#   A 層：只用原本版本的紀錄配適 M3h，輸入 = 原本的三份候選 + 「path p 進步」的假設（RQ3-G 的隨機改進，直接算期望值）。
#   B 層：7 個版本的紀錄配適 M3，輸入同 A 層的假設，但被改進的那份每一題都設 donor = 1。
#   C 層：RQ3-GJR 的做法（真的換進來的候選）。
#   每一層另有同樣訓練資料估的 9 格比例表（T_A、T_B、T_C）。
# ------------------------------------------------------------------
M3H = ["correct", "support", "first", "last", "trans", "short", "loglen"]
M3 = ["correct", "support", "first", "last", "donor", "trans", "short", "loglen"]
SEED = 20261009
CURVE_Q = [0.003, 0.01, 0.03, 0.10, 0.30, 1.00]                 # 第 8 節 6 (i)(ii)
CURVE_N = [50, 100, 200, 500, None]                               # 第 8 節 6 (iii)；None = 全部
DIRECT_Q = [0.01, 0.03, 0.10, 0.30]                               # 第 8 節 7
CURVE_SPLITS = 20                                                 # 前 20 次切分
CURVE_REPS = 20
TIE_TOL = 1e-12                                                   # 第 8 節 12（補充 25）
DONOR_COL = BASE.index("donor")


# ------------------------------------------------------------------
# 一個狀態（三份候選的答案）的因素：依顯示的位置排，算法同 Analysis.judgeRegression.buildGroups
# ------------------------------------------------------------------
def stateX(codes: np.ndarray, gold: np.ndarray, perm: np.ndarray, tok: np.ndarray, paths: list) -> tuple:
    """
    codes (3, n)：每個菜單位置（slot）的答案代碼；gold (n,)；perm (n, 3)：顯示位置 k 是 slot perm[i, k]；
    tok (3, n)：每個 slot 那份文字的 tokens 數；paths：三個 slot 的 path 代號。
    回傳 X (n, 3, len(BASE))（donor、pathacc、logchars 都是 0）與 ans (n, 3)（依顯示位置的答案代碼）。
    """
    n = codes.shape[1]
    ans = np.take_along_axis(codes.T, perm, axis=1)
    t = np.take_along_axis(tok.T, perm, axis=1)
    if (t <= 0).any():
        raise ValueError("a candidate in a hypothetical state has an empty raw_text")
    X = np.zeros((n, 3, len(BASE)))
    X[:, :, 0] = ans == gold[:, None]
    X[:, :, 1] = ((ans[:, :, None] == ans[:, None, :]).sum(axis=2) - 1) / 2
    X[:, 0, 2] = 1.0
    X[:, 2, 3] = 1.0
    flags = np.array([[c in TRANS, c in SHORT] for c in paths], dtype=float)
    X[:, :, 5:7] = flags[perm]
    X[:, :, 7] = np.log(t)
    return X, ans


def regValue(X: np.ndarray, names: list, beta: np.ndarray, donor_pos: np.ndarray | None = None) -> np.ndarray:
    """Σ 候選被選的機率 × correct。donor_pos (n,)：該位置的候選設 donor = 1（B 層；X 的 donor 欄必須是 0）。"""
    u = design(X, names) @ beta
    if donor_pos is not None:
        if X[:, :, DONOR_COL].any():
            raise ValueError("donor_pos given for a state that already has a donor candidate")
        u[np.arange(len(u)), donor_pos] += beta[names.index("donor")]
    u -= u.max(axis=1, keepdims=True)
    p = np.exp(u)
    p /= p.sum(axis=1, keepdims=True)
    return (p * X[:, :, 0]).sum(axis=1)


def cellsOfAns(ans: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """K = 3 的 9 格（ans 依顯示位置）；三條相同不該出現在這裡。"""
    cell = voteCells(ans.T, gold, np.tile(np.arange(3)[:, None], (1, len(gold))))
    if (cell == UNANIMOUS).any():
        raise ValueError("a model-evaluated state has three identical answers")
    return cell


def tabValue(cell: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.where(cell >= 0, q[np.clip(cell, 0, 8)], 0.0)


# ------------------------------------------------------------------
# 配適：無法配適的判定（第 8 節 6，補充 16）
# ------------------------------------------------------------------
def noVariation(Z: np.ndarray) -> list:
    """在每一組的三個候選之間都相同的因素（條件式 logit 無法識別）。Z (G, K, P)。"""
    if len(Z) == 0:
        return ["(no groups)"]
    return [j for j in range(Z.shape[2]) if not (np.ptp(Z[:, :, j], axis=1) > 0).any()]


def fitSafe(Z: np.ndarray, y: np.ndarray, names: list) -> tuple:
    """回傳 (結果或 None, 原因)。原因：某個因素沒有變化、牛頓法沒有收斂、Hessian 不可逆、非有限值。"""
    flat = noVariation(Z)
    if flat:
        return None, "no variation: " + ",".join(names[j] if isinstance(j, int) else j for j in flat)
    try:
        with np.errstate(all="ignore"):
            res = fitConditionalLogit(Z, y)
    except RuntimeError as e:
        return None, f"not converged ({e})"
    except np.linalg.LinAlgError as e:
        return None, f"singular Hessian ({e})"
    if not (np.isfinite(res["beta"]).all() and np.isfinite(res["se"]).all() and np.isfinite(res["loglik"])):
        return None, "non-finite estimate"
    res["names"] = names
    return res, ""


# ------------------------------------------------------------------
# 一個區塊：原本的版本、每個（菜單、slot）的「改對之後」狀態、真的替換後狀態
# ------------------------------------------------------------------
@dataclass
class SlotState:
    """一個（菜單、slot j）的假設改進：宿主的 p 答錯、在兩個供體子集二的聯集內的題目，把 p 改成 gold。"""
    path: str
    host_correct: np.ndarray           # (N,) 宿主那條 p 答對
    gold_known: np.ndarray             # (N,) 改對之後三條相同（= 全對）的題目
    gold_cols: np.ndarray              # 改對之後三條不同、要用模型 / 表的題目
    gold_X: np.ndarray                 # (n, 3, BASE)，donor = 0
    gold_cell: np.ndarray              # (n,)
    gold_pos: np.ndarray               # (n,) slot j 的顯示位置
    orig_pos: np.ndarray               # 原本版本有呼叫的題目上 slot j 的顯示位置（依 orig cols 的順序）


@dataclass
class MenuState:
    menu: str
    paths: list
    base: np.ndarray                   # (3, N) 宿主的答案代碼（依 slot）
    perm: np.ndarray                   # (N, 3) 計畫的順序（顯示位置 → slot）；沒有順序的題目為 -1
    orig: object                       # VersionItems（原本的版本，RQ3-GJR）
    orig_idx: np.ndarray               # 原本版本的組在 G3 的索引（依 cols）
    slots: list = field(default_factory=list)


class TBlock:
    """一個（宿主、資料集）區塊的 A、B 層假設改進所需的靜態資料。"""

    def __init__(self, D: dict, h: str, d: str):
        coded, plan, G3, lengths = D["coded"], D["plan"], D["G3"], D["lengths"]
        self.h, self.d = h, d
        H = coded[(h, d)]
        self.H = H
        self.N = len(H.item_ids)
        self.gold = H.gold
        self.sub2 = {g: H.sub & coded[(g, d)].sub for g in DONORS}
        self.union = self.sub2[DONORS[0]] | self.sub2[DONORS[1]]
        self.H2 = ~D["splits"][d]                                        # (200, N) 評分半
        self.mask = {g: self.H2 & self.sub2[g][None, :] for g in DONORS}
        self.donor_correct = {g: coded[(g, d)].correct for g in DONORS}
        item_ids = [int(i) for i in H.item_ids]
        self.menus = {}
        for menu in MENU_ORDER:
            paths = list(plan.menus[menu]["codes"])
            arm_ids = [PATHS[c] for c in paths]
            order = plan.orders[(h, d, menu)]
            perm = np.full((self.N, 3), -1, dtype=int)
            for col, i in enumerate(item_ids):
                o = order.get(i)
                if o is not None:
                    perm[col] = [arm_ids.index(a) for a in o]
            if (perm[self.union] < 0).any():
                raise ValueError(f"{h} | {d} | {menu}: an item in the domain has no planned order")
            base = H.codes[[M12.index(c) for c in paths]]
            orig = D["items3"][(h, d, menu, "orig")]
            orig_idx = orig.idx[0]
            # 計畫的順序必須等於紀錄的 presentation_order（有呼叫的題目）
            if not np.array_equal(G3.ans[orig_idx], np.take_along_axis(base.T[orig.cols], perm[orig.cols], axis=1)):
                raise ValueError(f"{h} | {d} | {menu}: planned orders differ from the recorded presentation_order")
            ms = MenuState(menu, paths, base, perm, orig, orig_idx)
            tok_host = np.stack([lengths[(h, d, c)][0] for c in paths])
            for j, p in enumerate(paths):
                hc = H.correct[M12.index(p)]
                new = base.copy()
                new[j] = H.gold
                unan = (new == new[0]).all(axis=0)
                wrong = self.union & ~hc
                cols = np.flatnonzero(wrong & ~unan)
                X, ans = stateX(new[:, cols], H.gold[cols], perm[cols], tok_host[:, cols], paths)
                ms.slots.append(SlotState(
                    path=p, host_correct=hc, gold_known=wrong & unan, gold_cols=cols, gold_X=X,
                    gold_cell=cellsOfAns(ans, H.gold[cols]), gold_pos=np.argmax(perm[cols] == j, axis=1),
                    orig_pos=np.argmax(perm[orig.cols] == j, axis=1)))
            self.menus[menu] = ms

    # ------------------------------------------------------------------
    # 狀態的值（每題）
    # ------------------------------------------------------------------
    def values(self, G3, cell3: np.ndarray, kind: str, params, layer: str) -> dict:
        """
        kind = "reg"：params = (names, beta)；"tab"：params = q (9,)。
        回傳 {menu: (V_plain (N,), [V_o (N,) 每個 slot], [V_g (N,) 每個 slot])}：
          V_plain = 原本候選的值（預測的原本版本；donor 全為 0）；
          V_o = 假設版本裡「原本狀態」的值（B 層的 regression：slot j 設 donor = 1；其餘 = V_plain）；
          V_g = 宿主的 p 答錯的題目「改對之後」的值（其餘題目 0）。
        """
        out = {}
        for menu, ms in self.menus.items():
            o = ms.orig
            Xo = G3.X[ms.orig_idx]
            if kind == "reg":
                names, beta = params
                po = regValue(Xo, names, beta)
            else:
                po = tabValue(cell3[ms.orig_idx], params)
            v_plain = o.known.copy()
            v_plain[o.cols] = po
            vo, vg = [], []
            for s in ms.slots:
                if kind == "reg" and layer == "B":
                    v = o.known.copy()
                    v[o.cols] = regValue(Xo, names, beta, donor_pos=s.orig_pos)
                    vo.append(v)
                    gv = regValue(s.gold_X, names, beta, donor_pos=s.gold_pos)
                else:
                    vo.append(v_plain)
                    gv = regValue(s.gold_X, names, beta) if kind == "reg" else tabValue(s.gold_cell, params)
                g = np.zeros(self.N)
                g[s.gold_known] = 1.0
                g[s.gold_cols] = gv
                vg.append(g)
            out[menu] = (v_plain, vo, vg)
        return out

    def hypoPredict(self, vals: dict, rows, menus: list | None = None, f_zero: bool = False, self_sub: bool = False) -> dict:
        """
        第 4 節：{(菜單, 版本, 子集二的供體): (R,)} 的預測正確率（評分半 H2_r ∩ 子集二）。
        替換的版本 = (p ← g)，原本的版本 = (菜單, "orig", g)。
        f_zero：強制 f = 0（第 10 節 4 的第一項）；self_sub：供體 = 宿主（第二項，t = 0）。
        """
        out = {}
        for g in DONORS:
            M = self.mask[g][rows].astype(float)
            n = M.sum(axis=1)
            cols, todo = [], []                                          # todo：(鍵, 欄位起點, f 或 None)
            for menu in (menus or MENU_ORDER):
                ms = self.menus[menu]
                v_plain, vo, vg = vals[menu]
                todo.append(((menu, "orig", g), len(cols), None))
                cols.append(v_plain)
                for j, s in enumerate(ms.slots):
                    c = s.host_correct.astype(float)
                    cg = c if self_sub else self.donor_correct[g][M12.index(s.path)].astype(float)
                    t = M @ cg - M @ c                                   # 供體在這一半的答對題數 − 宿主的
                    w = M @ (1 - c)                                      # 宿主在這一半答錯的題數
                    f = np.where((t > 0) & (w > 0), t / np.where(w > 0, w, 1.0), 0.0)
                    if f_zero:
                        f = np.zeros_like(f)
                    todo.append(((menu, f"{s.path}-{g}", g), len(cols), f))
                    cols += [c * vo[j], (1 - c) * vo[j], (1 - c) * vg[j]]
            A = M @ np.stack(cols, axis=1)                               # (R, 欄數)
            for key, i, f in todo:
                out[key] = A[:, i] / n if f is None else (A[:, i] + (1 - f) * A[:, i + 1] + f * A[:, i + 2]) / n
        return out

    def fractions(self, rows, menu: str, j: int, g: str) -> np.ndarray:
        """f_r（第 4 節）：第 10 節 4 的手算用。"""
        s = self.menus[menu].slots[j]
        M = self.mask[g][rows].astype(float)
        c = s.host_correct.astype(float)
        t = M @ self.donor_correct[g][M12.index(s.path)].astype(float) - M @ c
        w = M @ (1 - c)
        return np.where((t > 0) & (w > 0), t / np.where(w > 0, w, 1.0), 0.0)

    def realValues(self, kind: str, params, menu: str, j: int, g: str, coded: dict, lengths: dict) -> np.ndarray:
        """第 10 節 4 的第三項：B 層的計算路徑餵真的換進來的候選（供體的答案、donor = 1、供體文字的 loglen；權重 1）。回傳 (N,)。"""
        ms = self.menus[menu]
        p = ms.paths[j]
        new = ms.base.copy()
        new[j] = coded[(g, self.d)].codes[M12.index(p)]
        tok = np.stack([lengths[(g if k == j else self.h, self.d, c)][0] for k, c in enumerate(ms.paths)])
        unan = (new == new[0]).all(axis=0)
        dom = self.sub2[g]
        v = np.zeros(self.N)
        v[dom & unan] = (new[0] == self.gold)[dom & unan]
        cols = np.flatnonzero(dom & ~unan)
        X, ans = stateX(new[:, cols], self.gold[cols], ms.perm[cols], tok[:, cols], ms.paths)
        pos = np.argmax(ms.perm[cols] == j, axis=1)
        if kind == "reg":
            names, beta = params
            v[cols] = regValue(X, names, beta, donor_pos=pos)
        else:
            v[cols] = tabValue(cellsOfAns(ans, self.gold[cols]), params)
        return v


# ------------------------------------------------------------------
# C 層：真的換進來的候選（RQ3-GJR 的 BlockEval.predFromGroups，切分可選）
# ------------------------------------------------------------------
def realPredict(ev, p: np.ndarray, rows, menus: list | None = None) -> dict:
    """ev：RQ3-GJR 的 BlockEval；p：G3 每組的預測正確機率。回傳 {(菜單, 版本, 子集二的供體): (R,)}。"""
    keep = set(menus or MENU_ORDER)
    vec = {k: itemVector(it, p) for k, it in ev.items.items() if k[0] in keep}
    return {k: halfMean(vec[k[:2]], ev.mask[k[2]][rows]) for k in ev.pairs if k[0] in keep}


# ------------------------------------------------------------------
# 第 8 節 12 的命中率
# ------------------------------------------------------------------
def hitRate(eff_pred: np.ndarray, eff_act: np.ndarray, top: bool) -> float:
    """第 8 節 12：命中 = Σ（預測最高的 k 條）(1/k) × [在實際最高的 m 條之中] × (1/m)；並列 = 相差 ≤ 1e-12。"""
    sp, sa = (eff_pred, eff_act) if top else (-eff_pred, -eff_act)
    P = np.flatnonzero(sp >= sp.max() - TIE_TOL)
    A = set(np.flatnonzero(sa >= sa.max() - TIE_TOL).tolist())
    return float(sum((1 / len(P)) * (1 / len(A)) for j in P if j in A))


# ------------------------------------------------------------------
# 抽樣（第 8 節 6、7；補充 13、14、17）
# ------------------------------------------------------------------
def nOf(q: float, N: int) -> int:
    return max(1, int(np.floor(q * N + 0.5)))


def curveSamples(q: float, k: int, universe: dict) -> dict:
    """(i)(ii)：{(裁判, 資料集): 抽到的題目在逐題陣列的位置}。每個（q、k）一個 rng，依序 gpt4omini、qwen × 四個資料集。"""
    rng = np.random.default_rng([SEED, k])
    out = {}
    for h in HOSTS:
        for d in DATASETS:
            U = universe[(h, d)]
            out[(h, d)] = U[np.sort(rng.choice(len(U), nOf(q, len(U)), replace=False))]
    return out


def cutOneSamples(n: int | None, r: int, universe: dict, splits: dict) -> tuple:
    """(iii)：H1_r ∩ 母體抽 n 題；n 大於可抽的題數時用全部（不抽、不消耗 rng）。回傳 ({(裁判, 資料集): 位置}, {(裁判, 資料集): 是否用全部})。"""
    rng = np.random.default_rng([SEED, r])
    out, capped = {}, {}
    for h in HOSTS:
        for d in DATASETS:
            U = universe[(h, d)]
            U = U[splits[d][r][U]]
            if n is None or n >= len(U):
                out[(h, d)], capped[(h, d)] = U, n is not None
            else:
                out[(h, d)], capped[(h, d)] = U[np.sort(rng.choice(len(U), n, replace=False))], False
    return out, capped


def directSamples(q: float, r: int, blocks: dict) -> dict:
    """第 8 節 7：{(裁判, 資料集, 供體): 抽到的題目}；母體 = H2_r ∩ 子集二 (h, g)。每個（q、r）一個 rng，依序區塊 → 供體。"""
    rng = np.random.default_rng([SEED, r])
    out = {}
    for h in HOSTS:
        for d in DATASETS:
            b = blocks[(h, d)]
            for g in DONORS:
                U = np.flatnonzero(b.mask[g][r])
                out[(h, d, g)] = U[np.sort(rng.choice(len(U), nOf(q, len(U)), replace=False))]
    return out
