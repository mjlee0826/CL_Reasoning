import numpy as np

# ------------------------------------------------------------------
# split-half 切分、錨點選擇、recovery 系列公式
# TestRecoveryBlind / test_em_legacy.py / split_half_gap.py / difficulty_strata.py / run_analysis.py / check_framework.py
# 共用，確保定義一致
# ------------------------------------------------------------------


def makeSplits(n: int, reps: int = 200, seed: int = 0) -> np.ndarray:
    """(reps, n) 的 bool 矩陣，True = H1（n // 2 題）、False = H2。同 n、同 seed 必得同一組切分。"""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n) < n // 2 for _ in range(reps)])


def pickMax(k1: int, k2: int, seed: int, rep: int, key1: int, key2: int) -> bool:
    """
    在 H1 上決定 L_max：k1 / k2 是兩者在 H1 的答對題數，回傳「第 1 個是否為 L_max」。
    平手時用 (seed, rep, 兩個 key) 決定的亂數，與參數順序無關，不同程式呼叫會得到同一個結果。
    """
    if k1 != k2:
        return k1 > k2
    lower_wins = np.random.default_rng([seed, rep, min(key1, key2), max(key1, key2)]).random() < 0.5
    return (key1 < key2) == lower_wins


def recoveryStats(cA, cB, cf, dis, mask) -> dict:
    """
    paper_status 0A-1 的公式，只用 mask 內的題目（A = 錨點）:
        D = 兩個初答不同的題目；d = |D| / N
        n_A / n_B = D 中 A / B 正確的題數；c = (n_A + n_B) / |D|；m = c / 2；w_A = n_A / (n_A + n_B)
        recovery_blind = 2·w_A − 1
        recovery = (acc_agg − m) / (c − m)，acc_agg = D 中最終答案正確的比例
        skill = (recovery − recovery_blind) / (1 − recovery_blind)
    cA / cB: 初答是否正確；cf: 最終答案是否正確；dis: 兩個初答是否不同（皆為同長度的 bool 陣列）
    """
    nan = float("nan")
    D = dis & mask
    N, nD = int(mask.sum()), int(D.sum())
    nA, nB = int((D & cA).sum()), int((D & cB).sum())
    nR = nA + nB
    d = nD / N if N else nan
    c = nR / nD if nD else nan
    m = c / 2
    if nR == 0:
        return dict(N=N, D=nD, d=d, c=c, m=m, n_A=nA, n_B=nB,
                    w_A=nan, recovery_blind=nan, recovery=nan, skill=nan)
    w_A = nA / nR
    recovery_blind = 2 * w_A - 1
    recovery = (int((D & cf).sum()) / nD - m) / (c - m)
    skill = (recovery - recovery_blind) / (1 - recovery_blind) if recovery_blind < 1 else nan
    return dict(N=N, D=nD, d=d, c=c, m=m, n_A=nA, n_B=nB, w_A=w_A,
                recovery_blind=recovery_blind, recovery=recovery, skill=skill)


# 在 H2 上計算、再對 reps 次切分取平均的量（輸出時加 _H2 字尾）
H2_KEYS = ("n_A", "n_B", "w_A", "recovery_blind", "recovery", "skill")


def splitHalfStats(c1, c2, cf, dis, name1: str, name2: str, key1: int, key2: int,
                   seed: int = 0, reps: int = 200) -> dict:
    """
    paper_status 0A-1 的 split-half 量（TestRecoveryBlind / test_em_legacy.py / run_analysis.py 共用）:
        d, c, m                                          全部題目（不涉及選擇）
        錨點 A = 兩者中答對較多者                          在 H1 決定（makeSplits / pickMax）
        n_A, n_B, w_A, recovery_blind, recovery, skill     在同一個 H2 上計算，再對 reps 次取平均
    recovery 必須和 recovery_blind 用同一批 H2 分歧題：recovery 的分母 c − m 與 recovery_blind 的分子共用這些題目。

    c1 / c2: 候選 1 / 2 初答是否正確；cf: 最終答案是否正確；dis: 兩個初答是否不同
    （皆為依題目排序的 bool 陣列）。name1 / name2 是錨點的顯示名稱，key1 / key2 用於平手判定。
    cf=None 表示 Blind：每次切分的最終答案就是 H1 選出的一方，因此 recovery_H2 = recovery_blind_H2。
    """
    blind = cf is None
    c1, c2, dis = (np.asarray(x, dtype=bool) for x in (c1, c2, dis))
    N = len(c1)
    cf = np.zeros(N, dtype=bool) if blind else np.asarray(cf, dtype=bool)
    full = recoveryStats(c1, c2, cf, dis, np.ones(N, dtype=bool))   # 只取 N, D, d, c, m（與 cf 無關）

    stats_H2, first_picked = [], []
    for rep, h1 in enumerate(makeSplits(N, reps, seed)):
        first = pickMax(int(c1[h1].sum()), int(c2[h1].sum()), seed, rep, key1, key2)  # H1 選錨點
        cA, cB = (c1, c2) if first else (c2, c1)
        stats_H2.append(recoveryStats(cA, cB, cA if blind else cf, dis, ~h1))         # H2 計算
        first_picked.append(first)

    first_rate = float(np.mean(first_picked))
    result = {
        "seed": seed,
        "reps": reps,
        "N": full["N"],
        "D": full["D"],
        "d": full["d"],
        "c": full["c"],
        "m": full["m"],
        "anchor": name1 if first_rate >= 0.5 else name2,
        "anchor_rate": max(first_rate, 1 - first_rate),
    }
    for key in H2_KEYS:
        values = np.array([s[key] for s in stats_H2], dtype=float)
        result[f"{key}_H2"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    # H2 中沒有可救分歧題（recovery 無定義）/ recovery_blind = 1（skill 無定義）的次數
    result["reps_undefined"] = int(sum(np.isnan(s["recovery"]) for s in stats_H2))
    result["reps_skill_undefined"] = int(sum(np.isnan(s["skill"]) for s in stats_H2))
    return result
