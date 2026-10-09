import numpy as np

from Analysis.pathImprove import M12, TIE_TOL, logOdds, fraction
from Analysis.pathImproveK import SCALE, Ans, HostDonorK

# ------------------------------------------------------------------
# RQ3-GSX（result/analysis/rq3gsx/rq3gsx_criteria.md）：依正確率加權的投票（WV）下，落單的那條還是最不值得改嗎？全程離線。
#   WV 的權重 = max(0, ln(p̂ / (1 − p̂)))，p̂ = (選擇半答對題數 + 1) / (選擇半題數 + 2)；選擇半 = H1 ∩ 子集二（替換所在的子集）。
#   200 次切分的權重一起算：分數 (R, K, N) = W (R, K) × 「兩條 path 答案相同」(K, K × N)，平手與得分同 Analysis.pathImproveK.Ans.scores。
#   逐切分的分子都存成「評分半的得分總和 × SCALE」的差（整數；模擬是浮點），呼叫端再除以 n2 × SCALE。
# ------------------------------------------------------------------
RULES = ("C", "A")


def wvScores(codes: np.ndarray, correct: np.ndarray, W: np.ndarray, rules=RULES) -> dict:
    """
    W (R, K) 是每次切分的權重。回傳 m (R, N) = 平手的答案數，與每個規則的逐題得分 × SCALE（int64，(R, N)）：
    加權分數在最高分 1e-9 以內的答案都是平手；A 取列號最小的候選（菜單內的平手順序），C 平分。
    W 只有一列時就是 Ans.scores(weights, rules)。
    """
    K, N = codes.shape
    same = codes[:, None, :] == codes[None, :, :]
    leader = same.argmax(axis=0)
    score = (np.asarray(W, dtype=float) @ same.reshape(K, K * N).astype(float)).reshape(len(W), K, N)
    cand = score >= score.max(axis=1, keepdims=True) - TIE_TOL
    m = (cand & (leader == np.arange(K)[:, None])[None]).sum(axis=1)
    out = {"m": m}
    for rule in rules:
        if rule == "C":
            out["C"] = np.where((cand & correct[None]).any(axis=1), SCALE // m, 0).astype(np.int64)
        elif rule == "A":
            out["A"] = correct[cand.argmax(axis=1), np.arange(N)[None, :]].astype(np.int64) * SCALE
        else:
            raise ValueError(rule)
    return out


def rowSums(h2: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """每次切分評分半的得分總和：h2 (R, N) 0/1、scores (R, N) 或 (N,) → (R,)。"""
    return (h2 * scores).sum(axis=1) if scores.ndim == 2 else h2 @ scores


def menuSplits(ctx: HostDonorK, codes: list[str], wv: bool = True, sim_rules=("C",)) -> dict:
    """
    一份菜單在一個（宿主、供體、資料集）的逐切分量（評分半 ∩ 子集二；R 次切分 × 菜單內 K 條 path 的替換）：
      den_int (R, K)：供體 − 宿主那條 path 的答對題數；valid (R, K)；included (R,) = K 條都有效。
      base[key] (R,)：替換前的得分總和 × SCALE；key = V_C、V_A、W_C、W_A。
      num[key] (R, K)：替換後 − 替換前（整數）；另有 Wfix_C = p 換成供體的答案、權重沿用替換前（第 7 節 11）。
      sim[key] (R, K)：隨機改進模型的期望 − 替換前（浮點）；key = V_r、W_r（r ∈ sim_rules）。
      SB_after (R, K)：替換後 SB 在評分半的答對題數（RQ3-GK 判定一；選擇半答對最多，平手取菜單內順序在前的）。
      tie0 (R,)、tie1 (R, K)：WV 平手（m ≥ 2）的評分半題數，替換前 / 替換後；zero0 (R,)、zero1 (R, K)：權重為 0 的 path 數。
    wv = False 時只算 V 的部分（第 8 節 1）。
    """
    host, donor, sp = ctx.host, ctx.donor, ctx.sp
    rows = [M12.index(c) for c in codes]
    K, R = len(rows), sp.reps
    gold = host.gold
    hcodes = host.codes[rows]
    ans = Ans.of(hcodes, gold)
    ones = np.ones(K)
    h2 = sp.h2.astype(np.int64)
    n1, n2 = sp.n1, sp.n2
    k1, k2 = sp.k1[:, rows], sp.k2[:, rows]
    kd1, kd2 = ctx.kd1[:, rows], ctx.kd2[:, rows]
    valid = ctx.valid[:, rows]
    t2, w2 = kd2 - k2, n2[:, None] - k2
    f2 = fraction(t2, w2)
    out = {"den_int": kd2 - k2, "valid": valid, "included": valid.all(axis=1), "n2": n2, "base": {}, "num": {}, "sim": {}}

    vrules = tuple(sorted(set(RULES) | set(sim_rules)))
    baseV = ans.scores(ones, vrules)
    for r in RULES:
        out["base"][f"V_{r}"] = h2 @ baseV[r]
        out["num"][f"V_{r}"] = np.empty((R, K), dtype=np.int64)
    for r in sim_rules:
        out["sim"][f"V_{r}"] = np.empty((R, K))
    SB = np.empty((R, K), dtype=np.int64)
    for j, p in enumerate(rows):
        after = ans.withRow(j, donor.codes[p]).scores(ones, RULES)
        for r in RULES:
            out["num"][f"V_{r}"][:, j] = h2 @ after[r] - out["base"][f"V_{r}"]
        if sim_rules:
            fixed = ans.withRow(j, gold).scores(ones, sim_rules)
            for r in sim_rules:
                out["sim"][f"V_{r}"][:, j] = f2[:, j] * (h2 @ (fixed[r] - baseV[r]))
        hits1, hits2 = k1.copy(), k2.copy()
        hits1[:, j], hits2[:, j] = kd1[:, j], kd2[:, j]
        SB[:, j] = hits2[np.arange(R), hits1.argmax(axis=1)]     # argmax 取第一個最大的 = 菜單內順序在前（= probe.strongest）
    out["SB_after"] = SB
    if not wv:
        return out

    correct = hcodes == gold[None, :]
    W0 = logOdds(k1, n1[:, None])
    s0 = wvScores(hcodes, correct, W0, vrules)
    for r in RULES:
        out["base"][f"W_{r}"] = rowSums(h2, s0[r])
        out["num"][f"W_{r}"] = np.empty((R, K), dtype=np.int64)
    out["num"]["Wfix_C"] = np.empty((R, K), dtype=np.int64)
    for r in sim_rules:
        out["sim"][f"W_{r}"] = np.empty((R, K))
    out["tie0"], out["zero0"] = rowSums(h2, (s0["m"] >= 2).astype(np.int64)), (W0 == 0).sum(axis=1)
    out["tie1"], out["zero1"] = np.empty((R, K), dtype=np.int64), np.empty((R, K), dtype=np.int64)
    for j, p in enumerate(rows):
        Wj = W0.copy()
        Wj[:, j] = logOdds(kd1[:, j], n1)
        sub = hcodes.copy()
        sub[j] = donor.codes[p]
        sub_correct = sub == gold[None, :]
        sa = wvScores(sub, sub_correct, Wj, RULES)
        for r in RULES:
            out["num"][f"W_{r}"][:, j] = rowSums(h2, sa[r]) - out["base"][f"W_{r}"]
        out["num"]["Wfix_C"][:, j] = rowSums(h2, wvScores(sub, sub_correct, W0, ("C",))["C"]) - out["base"]["W_C"]
        if sim_rules:
            # 隨機改進：p 的權重 = 供體在選擇半的答對題數（k1 + t1）；原答案與改對答案都用改進後的權重（RQ3-G §4）
            fixed = hcodes.copy()
            fixed[j] = gold
            so = wvScores(hcodes, correct, Wj, sim_rules)
            sf = wvScores(fixed, fixed == gold[None, :], Wj, sim_rules)
            for r in sim_rules:
                out["sim"][f"W_{r}"][:, j] = (rowSums(h2, so[r]) - out["base"][f"W_{r}"]) + f2[:, j] * rowSums(h2, sf[r] - so[r])
        out["tie1"][:, j] = rowSums(h2, (sa["m"] >= 2).astype(np.int64))
        out["zero1"][:, j] = (Wj == 0).sum(axis=1)
    out["W0"] = W0
    return out


def equalWeightMismatch(ctx: HostDonorK, codes: list[str], constants=(1.0, 0.5)) -> tuple[int, int]:
    """
    第 8 節 4：權重全設成同一個常數 c 時，WV 的逐題得分（規則 C、A）= 等權投票（Ans.scores(ones)）。
    範圍：替換前的答案與每條 path 替換後的答案，子集二的每一題。回傳 (不相等的題數, 比較的題數)。
    """
    host, donor = ctx.host, ctx.donor
    rows = [M12.index(c) for c in codes]
    K = len(rows)
    sets = [host.codes[rows]]
    for j, p in enumerate(rows):
        s = host.codes[rows].copy()
        s[j] = donor.codes[p]
        sets.append(s)
    bad = total = 0
    sub = ctx.sub
    for codes_ in sets:
        correct = codes_ == host.gold[None, :]
        ref = Ans.of(codes_, host.gold).scores(np.ones(K), RULES)
        for c in constants:
            mine = wvScores(codes_, correct, np.full((1, K), c), RULES)
            for r in RULES:
                bad += int((mine[r][0][sub] != ref[r][sub]).sum())
                total += int(sub.sum())
    return bad, total
