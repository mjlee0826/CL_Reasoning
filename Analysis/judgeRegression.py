import os
from dataclasses import dataclass

import numpy as np

from Analysis.alignment import loadRecords
from Analysis.experimentPlan import PATHS
from Analysis.menuVote import DATASETS
from Analysis.pathImprove import M12, HOSTS
from Analysis.judgeSubstitutionPredict import NONE, UNANIMOUS, voteCells

# ------------------------------------------------------------------
# RQ3-GJR（result/analysis/rq3gjr/rq3gjr_criteria.md）：用 RQ3-GJ 的裁判紀錄配適條件式 logit，預測裁判選到的候選。全程離線。
#   每次呼叫是一組：K 個候選依顯示的位置排；被選的機率 = softmax(因素 · 係數)，不加截距、不做正則化。
#   一筆（版本、題目）預測的正確機率 = Σ 候選被選的機率 × correct；不用呼叫的題目（答案全部相同）用已知的對錯。
# ------------------------------------------------------------------
TRANS, SHORT = {"JA", "ZH", "RU", "ES"}, {"R"}
BASE = ["correct", "support", "first", "last", "donor", "trans", "short", "loglen", "pathacc", "logchars"]
_M0 = ["correct", "support", "first", "last"]
_M3 = _M0 + ["donor", "trans", "short", "loglen"]
_M4 = _M3 + ["correct:donor", "correct:trans", "correct:short"]
MODELS = {
    "M0": _M0,
    "M1": _M0 + ["donor"],
    "M2": _M0 + ["donor", "trans", "short"],
    "M3": _M3,
    "M4": _M4,
    "M5a": _M3 + ["pathacc"],
    "M5b": _M4 + ["pathacc"],
    "M3chars": _M0 + ["donor", "trans", "short", "logchars"],     # 第 7 節 9：loglen = ln(字元數)
}
NEWTON_TOL, NEWTON_MAX = 1e-10, 50


@dataclass
class Groups:
    """一批呼叫。候選依顯示的位置排：X (G, K, len(BASE))、ans (G, K) 答案代碼；chosen = 被選的位置（沒有有效選擇 = −1）。"""
    K: int
    X: np.ndarray
    ans: np.ndarray
    gold: np.ndarray
    chosen: np.ndarray
    final_ok: np.ndarray        # 紀錄的 final_answer 等於 gold（RQ3-GJ、RQ1-KJ 的計分；沒有有效選擇算錯）
    judge: np.ndarray           # 在 HOSTS 的索引
    dataset: np.ndarray         # 在 DATASETS 的索引
    col: np.ndarray             # 題目在該資料集逐題陣列（切分）中的位置
    vkey: np.ndarray            # 在 keys 的索引
    keys: list                  # (宿主, 資料集, 菜單, 版本, 順序)

    def __len__(self):
        return len(self.chosen)

    @property
    def valid(self) -> np.ndarray:
        return self.chosen >= 0

    @property
    def correct(self) -> np.ndarray:
        return self.X[:, :, 0] > 0.5

    @property
    def actual(self) -> np.ndarray:
        """Judge 的實際對錯：有效選擇而且選到的候選答對；沒有有效選擇算錯。"""
        return self.valid & self.correct[np.arange(len(self)), np.clip(self.chosen, 0, self.K - 1)]

    @property
    def menu(self) -> np.ndarray:
        names = np.array([k[2] for k in self.keys], dtype=object)
        return names[self.vkey]

    def subset(self, mask: np.ndarray) -> "Groups":
        return Groups(self.K, self.X[mask], self.ans[mask], self.gold[mask], self.chosen[mask], self.final_ok[mask], self.judge[mask],
                      self.dataset[mask], self.col[mask], self.vkey[mask], self.keys)


# ------------------------------------------------------------------
# 候選的長度與 pathacc
# ------------------------------------------------------------------
def textLengths(armdir: str, coded: dict, enc) -> dict:
    """{(model, dataset, path): (tokens (N,), 字元數 (N,))}，依 coded 的 item_ids 對齊；arm 檔沒有的題目為 0。"""
    from Runner.paths import armPath
    from Arm.ArmSpec import ArmSpec
    out = {}
    for (m, d), block in coded.items():
        index = {int(i): n for n, i in enumerate(block.item_ids)}
        for c in M12:
            _, recs = loadRecords(armPath(armdir, m, d, ArmSpec.from_arm_id(PATHS[c])))
            ids = [i for i in recs if int(i) in index]
            texts = [recs[i].get("raw_text") or "" for i in ids]
            toks = enc.encode_ordinary_batch(texts, num_threads=8)
            tok, chars = np.zeros(len(index)), np.zeros(len(index))
            cols = np.array([index[int(i)] for i in ids], dtype=int)
            tok[cols] = [len(t) for t in toks]
            chars[cols] = [len(x) for x in texts]
            out[(m, d, c)] = (tok, chars)
    return out


def pathAccuracy(coded: dict) -> dict:
    """pathacc：{(model, dataset, path): 該模型自己的子集一（12 條都有答案）上的正確率}（第 3 節 M5a、M5b）。"""
    return {(m, d, c): float(b.correct[M12.index(c)][b.sub].mean()) for (m, d), b in coded.items() for c in M12}


# ------------------------------------------------------------------
# 呼叫 -> 組
# ------------------------------------------------------------------
def versionCodes(coded: dict, host: str, dataset: str, codes: list, slot: int | None = None, donor: str | None = None) -> np.ndarray:
    """(K, N) 一個版本每條候選的答案代碼（依菜單列出的順序）；替換的版本把 slot 換成供體同一條 path 的答案。"""
    out = coded[(host, dataset)].codes[[M12.index(c) for c in codes]].copy()
    if donor:
        out[slot] = coded[(donor, dataset)].codes[M12.index(codes[slot])]
    return out


def buildGroups(sources: list, coded: dict, gold_text: dict, lengths: dict, pathacc: dict, K: int) -> Groups:
    """
    sources：[(鍵 (宿主, 資料集, 菜單, 版本, 順序), Judge 檔, 菜單的代號串列, slot, 供體)]。每個檔的每筆紀錄是一組；
    候選依該筆的 presentation_order 排。chosen_arm 必須在候選之中、而且和 choice 的位置一致（第 10 節 5）。
    """
    parts, keys = [], []
    for key, path, codes, slot, donor in sources:
        h, d = key[0], key[1]
        _, recs = loadRecords(path)
        if not recs:
            raise ValueError(f"{path}: no records")
        H = coded[(h, d)]
        index = {int(i): n for n, i in enumerate(H.item_ids)}
        arm_ids = [PATHS[c] for c in codes]
        writer = [donor if (donor and j == slot) else h for j in range(len(codes))]
        ver_codes = versionCodes(coded, h, d, codes, slot, donor)
        tok = np.stack([lengths[(writer[j], d, c)][0] for j, c in enumerate(codes)])
        chars = np.stack([lengths[(writer[j], d, c)][1] for j, c in enumerate(codes)])
        flags = np.array([[writer[j] != h, c in TRANS, c in SHORT] for j, c in enumerate(codes)], dtype=float)
        acc = np.array([pathacc[(writer[j], d, c)] for j, c in enumerate(codes)])
        ids = sorted(recs, key=int)
        n = np.array([index[int(i)] for i in ids])
        perm = np.array([[arm_ids.index(a) for a in recs[i]["presentation_order"]] for i in ids])
        if perm.shape[1] != K or (np.sort(perm, axis=1) != np.arange(K)).any():
            raise ValueError(f"{path}: presentation_order is not a permutation of the {K} candidates")
        chosen, final_ok = np.full(len(ids), -1), np.zeros(len(ids), dtype=bool)
        for g, i in enumerate(ids):
            r = recs[i]
            trace = r.get("trace") or {}
            c, arm = trace.get("choice"), trace.get("chosen_arm")
            if isinstance(c, int) and 1 <= c <= K:
                if arm not in r["presentation_order"] or r["presentation_order"][c - 1] != arm:
                    raise ValueError(f"{path} item {i}: chosen_arm {arm} does not match choice {c}")
                chosen[g] = c - 1
            elif arm is not None:
                raise ValueError(f"{path} item {i}: chosen_arm {arm} without a valid choice")
            final = r.get("final_answer")
            final_ok[g] = bool(final) and final == gold_text[(h, d)][n[g]]
        ans = ver_codes[perm, n[:, None]]                                    # (g, K) 依顯示位置
        X = np.zeros((len(ids), K, len(BASE)))
        X[:, :, 0] = ans == H.gold[n][:, None]
        X[:, :, 1] = ((ans[:, :, None] == ans[:, None, :]).sum(axis=2) - 1) / (K - 1)
        X[:, 0, 2] = 1.0
        X[:, K - 1, 3] = 1.0
        X[:, :, 4:7] = flags[perm]
        t, ch = tok[perm, n[:, None]], chars[perm, n[:, None]]
        if (t <= 0).any() or (ch <= 0).any():
            raise ValueError(f"{path}: a candidate has an empty raw_text")
        X[:, :, 7] = np.log(t)
        X[:, :, 8] = acc[perm]
        X[:, :, 9] = np.log(ch)
        parts.append((X, ans, H.gold[n], chosen, final_ok, np.full(len(ids), HOSTS.index(h)), np.full(len(ids), DATASETS.index(d)), n,
                      np.full(len(ids), len(keys))))
        keys.append(key)
    cat = [np.concatenate([p[k] for p in parts]) for k in range(9)]
    return Groups(K, *cat, keys)


# ------------------------------------------------------------------
# 條件式 logit
# ------------------------------------------------------------------
def design(X: np.ndarray, names: list) -> np.ndarray:
    cols = []
    for name in names:
        if ":" in name:
            a, b = name.split(":")
            cols.append(X[:, :, BASE.index(a)] * X[:, :, BASE.index(b)])
        else:
            cols.append(X[:, :, BASE.index(name)])
    return np.stack(cols, axis=2)


def _probs(Z: np.ndarray, b: np.ndarray) -> np.ndarray:
    u = Z @ b
    u -= u.max(axis=1, keepdims=True)
    p = np.exp(u)
    return p / p.sum(axis=1, keepdims=True)


def fitConditionalLogit(Z: np.ndarray, y: np.ndarray) -> dict:
    """最大概似：牛頓法、精確的梯度與 Hessian；每個係數的步長 < 1e-10 就停，最多 50 步，沒有收斂就停下來（第 3 節）。"""
    G, K, P = Z.shape
    rows = np.arange(G)
    b = np.zeros(P)
    for it in range(1, NEWTON_MAX + 1):
        p = _probs(Z, b)
        xbar = np.einsum("gk,gkp->gp", p, Z)
        grad = (Z[rows, y] - xbar).sum(axis=0)
        hess = np.einsum("gk,gkp,gkq->pq", p, Z, Z, optimize=True) - xbar.T @ xbar
        step = np.linalg.solve(hess, grad)
        b = b + step
        if np.abs(step).max() < NEWTON_TOL:
            break
    else:
        raise RuntimeError(f"conditional logit did not converge in {NEWTON_MAX} steps (last step {np.abs(step).max():.2e})")
    u = Z @ b
    m = u.max(axis=1)
    loglik = float((u[rows, y] - m - np.log(np.exp(u - m[:, None]).sum(axis=1))).sum())
    p = _probs(Z, b)
    xbar = np.einsum("gk,gkp->gp", p, Z)
    hess = np.einsum("gk,gkp,gkq->pq", p, Z, Z, optimize=True) - xbar.T @ xbar
    return {"beta": b, "se": np.sqrt(np.diag(np.linalg.inv(hess))), "loglik": loglik, "iterations": it, "n_groups": G}


def fit(groups: Groups, model: str) -> dict:
    """只用有效選擇的組配適（第 2 節）。"""
    g = groups.subset(groups.valid)
    out = fitConditionalLogit(design(g.X, MODELS[model]), g.chosen)
    out["names"] = MODELS[model]
    return out


def choiceProbs(X: np.ndarray, model: str, beta: np.ndarray) -> np.ndarray:
    return _probs(design(X, MODELS[model]), beta)


def predictCorrect(groups: Groups, model: str, beta: np.ndarray) -> np.ndarray:
    """每組預測的正確機率 = Σ 候選被選的機率 × correct。"""
    return (choiceProbs(groups.X, model, beta) * groups.X[:, :, 0]).sum(axis=1)


def scenarioProbability(model: str, beta: np.ndarray, correct: int, donor: int, kind: str, loglen: float) -> float:
    """
    第 7 節 3 的固定情境（K = 3）：落單那份 support 0；另外兩份 support 0.5、宿主寫的、英文完整、對錯和落單那份相反；
    三份的 loglen 都是 loglen（互相抵銷）。落單那份分別放在位置 1、2、3，回傳它被選的機率的平均。
    """
    X = np.zeros((3, 3, len(BASE)))                     # (落單那份的位置, 顯示位置, 因素)
    for lone in range(3):
        for j in range(3):
            is_lone = j == lone
            X[lone, j, 0] = correct if is_lone else 1 - correct
            X[lone, j, 1] = 0.0 if is_lone else 0.5
            X[lone, j, 4] = donor if is_lone else 0
            X[lone, j, 5] = float(is_lone and kind == "trans")
            X[lone, j, 6] = float(is_lone and kind == "short")
            X[lone, j, 7] = X[lone, j, 9] = loglen
        X[lone, 0, 2] = 1.0
        X[lone, 2, 3] = 1.0
    p = choiceProbs(X, model, beta)
    return float(np.mean([p[lone, lone] for lone in range(3)]))


# ------------------------------------------------------------------
# 9 格比例表（T_new、第 10 節 4）：RQ3-GJ §7 的算法
# ------------------------------------------------------------------
def cellsOf(groups: Groups) -> np.ndarray:
    """K = 3 的格子（票型 × 位置，0–8；沒有一條對 = NONE）。候選已依顯示的位置排，所以位置 = 欄的索引。"""
    if groups.K != 3:
        raise ValueError("the 9-cell table is defined for K = 3")
    pos = np.tile(np.arange(3)[:, None], (1, len(groups)))
    cell = voteCells(groups.ans.T, groups.gold, pos)
    if (cell == UNANIMOUS).any():
        raise ValueError("a called item has three identical answers")
    return cell


def ratioTable(cell: np.ndarray, correct: np.ndarray, weights: np.ndarray | None = None) -> tuple:
    """
    (選對 + 1) ÷ (題數 + 2)；每一筆紀錄都算進題數，correct 是 Judge 的實際對錯（沒有有效選擇 = 沒選對）。
    weights (R, G) 時逐切分估（第 10 節 4：選擇半的紀錄），回傳 (q, k, n) 各 (R, 9)；否則各 (9,)。
    """
    keep = cell >= 0
    onehot = np.eye(9)[np.clip(cell, 0, 8)] * keep[:, None]
    if weights is None:
        n, k = onehot.sum(axis=0), (onehot * correct[:, None]).sum(axis=0)
    else:
        n, k = weights @ onehot, weights @ (onehot * correct[:, None])
    return (k + 1) / (n + 2), k, n


def tableCorrect(cell: np.ndarray, q: np.ndarray) -> np.ndarray:
    """每組用 9 格表預測的正確機率；沒有一條對 = 0。"""
    return np.where(cell >= 0, q[np.clip(cell, 0, 8)], 0.0)


# ------------------------------------------------------------------
# 版本的逐題資料：已知的對錯、要呼叫的題目、實際的 Judge 對錯
# ------------------------------------------------------------------
@dataclass
class VersionItems:
    codes: np.ndarray           # (K, N)
    unanimous: np.ndarray       # (N,) 答案全部相同
    known: np.ndarray           # (N,) 全部相同且答對（float）
    cols: np.ndarray            # 要呼叫的題目在逐題陣列的位置
    idx: list                   # 每種順序一個：cols 對應到 Groups 的索引
    actual: np.ndarray          # (N,) 已知的對錯；要呼叫的題目 = Judge 的實際對錯（K = 2 是兩種順序的平均）


def versionItems(codes: np.ndarray, gold: np.ndarray, groups: Groups, vkeys: list, domain: np.ndarray) -> VersionItems:
    """vkeys：這個版本在 groups.keys 的索引（K = 2 兩種順序各一）。domain 內答案不全相同的題目必須恰好都有呼叫。"""
    unanimous = (codes == codes[0]).all(axis=0)
    known = (unanimous & (codes[0] == gold)).astype(float)
    idx, cols = [], None
    for k in vkeys:
        ix = np.flatnonzero(groups.vkey == k)
        ix = ix[np.argsort(groups.col[ix])]
        c = groups.col[ix]
        if cols is None:
            cols = c
        elif not np.array_equal(cols, c):
            raise ValueError(f"{groups.keys[k]}: the two orders were called on different items")
        idx.append(ix)
    if not np.array_equal(cols, np.flatnonzero(domain & ~unanimous)):
        raise ValueError(f"{groups.keys[vkeys[0]]}: called items are not the disagreement items of the domain")
    actual = known.copy()
    actual[cols] = np.mean([groups.actual[ix] for ix in idx], axis=0)
    return VersionItems(codes, unanimous, known, cols, idx, actual)


def itemVector(items: VersionItems, p: np.ndarray) -> np.ndarray:
    """p：每組的預測正確機率（Groups 的索引）。回傳 (N,)；要呼叫的題目 = 各順序預測的平均。"""
    vec = items.known.copy()
    vec[items.cols] = np.mean([p[ix] for ix in items.idx], axis=0)
    return vec


def halfMean(vec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mask (R, N) 評分半 ∩ 範圍；回傳每次切分的平均 (R,)。"""
    m = mask.astype(float)
    return (m @ vec) / m.sum(axis=1)
