"""
difficulty_strata.py — 0A-5：難度指標 q_i 與分層工具（Step 1 + Step 2 + 生成端的分層診斷）

問題 (paper_status 0A-1 的表 2):
    含英文的 64 個配對 recovery 比不含英文的 96 個高約 0.09，但 d / c / m 三個生成端的量幾乎一樣。
    兩種解釋：(a) 兩組觸發的分歧題目難度組成不同 → 假象；(b) 聚合器真的偏好有英文的配對。
    Step 2 若顯示兩組的 q 分布幾乎重疊，(a) 被排除，不必做完整分層。

Step 1  q_i = 第 i 題在「基底 run」中被答對的比例，四種基底（見 difficulty_bases）:
    all5        五個語言全用           所有配對同基底、6 級   ← 主分析
    loo         排掉配對自己的兩個語言   3 個 run、4 級        ← 規格原文，robustness
    loo_noen    再排掉英文             3 或 2 個 run         ← 消掉「英文較強」造成的基底落差
    cross_model 其他三個模型 × 五語言    15 個 run、16 級      ← 同基底且完全不循環
    ⚠️ all5 是循環的（基底含配對自己的兩個語言），而且兩種影響差很多:
       對 Δmean q 影響很小 —— D 上「自己兩列」的貢獻恰為 c/5，兩組的 c 只差 0.009（0A-1 表 2）→ 約 0.002
       對分布形狀與逐層 c 影響很大 —— q=1 的題目結構上不可能落在 D 裡（五個語言全對 ⇒ 全都同答案）；
       q=0.8 的題目只要落在 D 裡 c 就必然 = 1；q=0 則 c 必然 = 0。
       → KS D 與 ④ 的逐層 c 一定要同時看 cross_model（基底是別的模型，完全不循環）。

Step 2  兩組分歧題目的 q 分布比較。全部在 cell 內做，再對 16 個 cell 等權平均。
    逐對（主）: 每個配對帶自己的 D；以題目數加權，含英文組 = 4 個配對的 D 之總和
    聯集（副）: cell 內 4 個 / 6 個配對的 D 取聯集，每題只算一次（只對 pair-invariant 的基底有定義）
    ⚠️ 同一題會出現在多個配對的 D 裡 → ks_2samp 的 p 值偏小不可用。
       改用 cell 內重排 10 個配對的 has_english 標籤的 permutation test（C(10,4)=210）。
    ⚠️ KS D 只在兩組的 q 支撐集相同時可比。loo_noen 的含英文配對基底 3 個 run、不含英文配對 2 個 run，
       支撐集不同（{0,⅓,⅔,1} vs {0,½,1}）→ 它的 KS 不納入判定（KS_COMPARABLE）。

④ 的 baseline-only 部分  每個配對 × 每個層的 d / c / m / recovery_blind（公式直接呼叫
    Test.recoveryStats，與 0A-1 同一份程式），以及用「兩組合併後的層分布」為共同權重的 q-adjusted 值。
    分層基底跑 STRATA_BASES = (all5, cross_model)；相異 q 值 ≤ NATURAL_LEVEL_MAX 用自然層，
    否則用 make_strata 的 tie-aware 分位分層（相同 q 值絕不拆開，實際層數可能少於 --strata）。
    ⚠️ recovery 需要辯論的最終答案，這裡不算，留給 0A-5 的 ④。
    ⚠️ 這裡的 recovery_blind 是「全樣本 + 0A-1 的 H1 多數決錨點」，與 0A-1 表 1 在 H2 上算的
       recovery_blind_H2 不是同一個數，不要直接比。

資料:
    result/baseline/*.json    逐題對錯（q 與 c/m/recovery_blind）與逐題初答（分歧集合 D）
    result/challenge/         只用到檔名（取得 10 個配對的命名），不讀內容
    recovery_blind_pairs.csv  只用到 anchor 欄（0A-1 在 H1 決定的錨點）；缺檔時退回全樣本 argmax
    D 的定義與 split_half_gap.build_cell 相同，該腳本已逐點 assert 過它與 challenge 檔的
    metadata["RecoveryBlind"] 一致，所以這裡不需要讀 2.6 GB 的 challenge 內容。

輸出:
    difficulty_q_pairs.csv       pair × basis (640 列)：|D| 與 q 的 mean / median / Q1 / Q3
    difficulty_q_groups.csv      cell × basis × {逐對, 聯集}：兩組摘要、Δmean q、KS D、permutation p
    difficulty_q_items.csv       逐 (cell, item)：pair-invariant 基底的 q、被幾個配對觸發
    difficulty_strata_pairs.csv  逐 (cell, basis, pair, 層)：N / |D| / d / c / m / n_A / n_B / recovery_blind
    difficulty_strata_adjusted.csv  逐 (cell, basis, pair)：raw 與 q-adjusted 的 c / m / recovery_blind

判定 (主分析基底):
    兩組「幾乎重疊」 ⟺ |cell 平均 Δmean q| ≤ 0.02 且 cell 平均 KS D ≤ 0.05 且 permutation p > 0.05

用法:
    conda run -n clreasoning python difficulty_strata.py
    conda run -n clreasoning python difficulty_strata.py --perm 10000 --seed 0 --outdir .
"""

import glob
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy import stats

from split_half_gap import extract_baseline
from Strategy.StrategyType import LANGUAGE_STR_LIST
from Test.Test import Test

CHALLENGE_DIR = "result/challenge"
RB_PAIRS_CSV = "recovery_blind_pairs.csv"
ENGLISH = "english"
LANG_INDEX = {lang: i for i, lang in enumerate(LANGUAGE_STR_LIST)}

BASES = ("all5", "loo", "loo_noen", "cross_model")
PRIMARY_BASIS = "all5"
PAIR_INVARIANT = ("all5", "cross_model")      # q 與配對無關 → 聯集版與分層診斷才有單一 q
STRATA_BASES = ("all5", "cross_model")        # 分層診斷跑這兩個：一個是主分析，一個不循環
NATURAL_LEVEL_MAX = 8                         # 相異 q 值 ≤ 8 個就直接用自然層，否則 tie-aware 分位分層

# KS D 只在「兩組的 q 支撐集相同」時可比。loo_noen 的含英文配對基底 3 個 run、
# 不含英文配對 2 個 run，支撐集本來就不同（{0,⅓,⅔,1} vs {0,½,1}），KS 會被撐大 → 判定時不看它的 KS。
KS_COMPARABLE = ("all5", "loo", "cross_model")

# 判定門檻（事前定好，不看數字再改）
OVERLAP_DQ, OVERLAP_KS, OVERLAP_P = 0.02, 0.05, 0.05
MIN_STRATUM = 30                      # |D(P,s)| 的下限，不足的層要記錄

# Test.recoveryStats 中不依賴最終答案的量（recovery / skill 需要 challenge 內容，這裡不算）
BASELINE_KEYS = ("N", "D", "d", "c", "m", "n_A", "n_B", "w_A", "recovery_blind")
ADJUSTABLE = ("c", "m", "recovery_blind")     # 條件在 D 上的量，才需要用 D 的層分布標準化


# ----------------------------------------------------------------------------
# 資料組裝
# ----------------------------------------------------------------------------
def list_pairs(model, dataset):
    """從 challenge 檔名取得這個 cell 的配對與命名（只 listdir，不讀檔案內容）。"""
    pattern = os.path.join(CHALLENGE_DIR, f"{model}_{dataset}_challenge_*.json")
    pairs = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path).split("_challenge_", 1)[1][: -len(".json")]
        l1, l2 = name.split("_vs_")
        pairs.append((l1, l2))
    return pairs


def build_mono(cell, base_cell):
    """回傳 (ids, mono)；mono 是 (5, N) 的 bool，列序 = LANGUAGE_STR_LIST（與 Test.pickMax 的 key 相同）。"""
    if set(base_cell) != set(LANGUAGE_STR_LIST):
        raise ValueError(f"{cell}: baseline 語言不齊 {sorted(base_cell)}")
    ids = sorted(base_cell[LANGUAGE_STR_LIST[0]][0])
    for lang in LANGUAGE_STR_LIST:
        if sorted(base_cell[lang][0]) != ids:
            raise ValueError(f"{cell}: {lang} 的題目 id 與其他語言不一致")
    mono = np.array([[base_cell[lang][0][q] for q in ids] for lang in LANGUAGE_STR_LIST], dtype=bool)
    return ids, mono


def build_disagreement(cell, base_cell, ids, pairs):
    """{(l1, l2): (N,) bool}：兩個 baseline 檔的 MyAnswer 不同。定義同 split_half_gap.build_cell。"""
    DatasetClass = base_cell[LANGUAGE_STR_LIST[0]][2]
    out = {}
    for l1, l2 in pairs:
        a1, a2 = base_cell[l1][1], base_cell[l2][1]
        out[(l1, l2)] = np.array([not DatasetClass.compareTwoAnswer(a1[q], a2[q]) for q in ids], dtype=bool)
    return out


def load_anchors(path):
    """{(cell, pair): anchor}，來自 0A-1 表 1 的 H1 多數決錨點。缺檔回傳空 dict（退回全樣本 argmax）。"""
    if not os.path.exists(path):
        print(f"⚠️  找不到 {path}，錨點退回「全樣本準確率較高者」（有 winner's curse，僅影響 recovery_blind）")
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["source"] == "current"]
    return {(r["cell"], r["pair"]): r["anchor"] for _, r in df.iterrows()}


# ----------------------------------------------------------------------------
# Step 1：難度指標
# ----------------------------------------------------------------------------
def _mean_rows(mono, rows):
    if not rows:
        return np.full(mono.shape[1], np.nan)
    return mono[rows].mean(axis=0)


def difficulty_bases(mono, pairs, cross_mono):
    """
    回傳 {basis: (q_by_pair, n_runs_by_pair)}。q_by_pair[pair] 是 (N,) float。
    q_i = 第 i 題在基底 run 中被答對的比例；基底為空時整列 nan。
    """
    n_lang = mono.shape[0]
    out = {}

    shared = _mean_rows(mono, list(range(n_lang)))
    out["all5"] = ({p: shared for p in pairs}, {p: n_lang for p in pairs})

    loo, loo_n, noen, noen_n = {}, {}, {}, {}
    for p in pairs:
        drop = {LANG_INDEX[p[0]], LANG_INDEX[p[1]]}
        keep = [i for i in range(n_lang) if i not in drop]
        keep_noen = [i for i in keep if i != LANG_INDEX[ENGLISH]]
        loo[p], loo_n[p] = _mean_rows(mono, keep), len(keep)
        noen[p], noen_n[p] = _mean_rows(mono, keep_noen), len(keep_noen)
    out["loo"] = (loo, loo_n)
    out["loo_noen"] = (noen, noen_n)

    if cross_mono is not None and len(cross_mono):
        q = cross_mono.mean(axis=0)
        out["cross_model"] = ({p: q for p in pairs}, {p: cross_mono.shape[0] for p in pairs})
    return out


# ----------------------------------------------------------------------------
# Step 2：把每個配對的 D 壓成「q 層計數向量」，之後全部是矩陣運算
# ----------------------------------------------------------------------------
def level_grid(q_by_pair, dis, pairs):
    """一個 cell × 一個基底裡所有出現過的相異 q 值（排序），當作直方圖與 KS 的共同格點。"""
    vals = [np.round(q_by_pair[p][dis[p]], 12) for p in pairs]
    vals = [v[np.isfinite(v)] for v in vals]
    return np.unique(np.concatenate(vals)) if any(len(v) for v in vals) else np.array([])


def counts_matrix(q_by_pair, dis, pairs, levels):
    """(n_pairs, n_levels)：第 k 列是第 k 個配對的 D 在各 q 層的題數。"""
    C = np.zeros((len(pairs), len(levels)))
    for k, p in enumerate(pairs):
        q = np.round(q_by_pair[p][dis[p]], 12)
        q = q[np.isfinite(q)]
        if len(q):
            np.add.at(C, (np.full(len(q), k), np.searchsorted(levels, q)), 1.0)
    return C


def _cdf(counts):
    total = counts.sum(axis=-1, keepdims=True)
    return np.divide(np.cumsum(counts, axis=-1), total, out=np.full(counts.shape, np.nan), where=total > 0)


def _mean_q(counts, levels):
    total = counts.sum(axis=-1)
    return np.divide(counts @ levels, total, out=np.full(total.shape, np.nan), where=total > 0)


def _quantile_q(counts, levels, p):
    """離散分布的第 p 分位：第一個累積比例 ≥ p 的 q 層。"""
    cdf = _cdf(counts)
    if not np.isfinite(cdf).all():
        return np.nan
    return float(levels[int(np.argmax(cdf >= p - 1e-12))])


def _ks(c1, c2):
    return np.nanmax(np.abs(_cdf(c1) - _cdf(c2)), axis=-1)


def describe(counts, levels):
    return {
        "n_items": float(counts.sum()),
        "q_mean": float(_mean_q(counts, levels)),
        "q_p25": _quantile_q(counts, levels, 0.25),
        "q_median": _quantile_q(counts, levels, 0.50),
        "q_p75": _quantile_q(counts, levels, 0.75),
    }


def permute_masks(rng, n_pairs, n_with, n_perm):
    """(n_perm, n_with)：每列是被標成「含英文」的配對索引。cell 之間各自獨立重排。"""
    return rng.random((n_perm, n_pairs)).argsort(axis=1)[:, :n_with]


def cell_step2(C, has_en, levels, rng, n_perm):
    """一個 cell 的觀測統計量與 n_perm 次重排的統計量。"""
    with_c, without_c = C[has_en].sum(axis=0), C[~has_en].sum(axis=0)
    obs = {
        "with": describe(with_c, levels),
        "without": describe(without_c, levels),
        "d_mean_q": float(_mean_q(with_c, levels) - _mean_q(without_c, levels)),
        "ks": float(_ks(with_c, without_c)),
    }
    idx = permute_masks(rng, len(C), int(has_en.sum()), n_perm)
    perm_with = C[idx].sum(axis=1)                       # (n_perm, L)
    perm_without = C.sum(axis=0)[None, :] - perm_with
    return obs, _mean_q(perm_with, levels) - _mean_q(perm_without, levels), _ks(perm_with, perm_without)


def perm_pvalue(obs, draws, two_sided):
    draws = np.asarray(draws, dtype=float)
    hit = np.abs(draws) >= abs(obs) - 1e-12 if two_sided else draws >= obs - 1e-12
    return float((1 + hit.sum()) / (len(draws) + 1))


# ----------------------------------------------------------------------------
# ④ 的 baseline-only 部分：逐層的 d / c / m / recovery_blind
# ----------------------------------------------------------------------------
def baseline_stats(cA, cB, dis, mask):
    """沿用 Test.recoveryStats 的公式（與 0A-1 同一份程式），只取不依賴最終答案的量。"""
    s = Test.recoveryStats(cA, cB, np.zeros(len(cA), dtype=bool), dis, mask)
    return {k: s[k] for k in BASELINE_KEYS}


def make_strata(q, n_strata):
    """
    tie-aware 分位分層：相同的 q 值絕不被拆到兩層。每個相異 q 值依自己在全體題目中的
    累積比例中點落到 floor(mid · n_strata) 層，再壓成連續編號 → 實際層數可能少於 n_strata。
    """
    qr = np.round(q, 12)
    levels, counts = np.unique(qr[np.isfinite(qr)], return_counts=True)
    share = counts / counts.sum()
    mid = np.cumsum(share) - share / 2
    raw = np.minimum((mid * n_strata).astype(int), n_strata - 1)
    _, label_of_level = np.unique(raw, return_inverse=True)
    lookup = dict(zip(levels, label_of_level))
    return np.array([lookup.get(v, -1) for v in qr]), levels, label_of_level


def strata_labels(q, n_strata):
    """
    回傳 (labels (N,) int，-1 = q 無定義；info DataFrame[stratum, q_level, q_lo, q_hi, N_s])。
    相異 q 值不多時直接用自然層（資訊最完整），否則用 tie-aware 分位分層。
    """
    qr = np.round(q, 12)
    levels = np.unique(qr[np.isfinite(qr)])
    if len(levels) <= NATURAL_LEVEL_MAX:
        label_of_level = np.arange(len(levels))
        labels = np.array([{v: k for k, v in enumerate(levels)}.get(v, -1) for v in qr])
    else:
        labels, levels, label_of_level = make_strata(q, n_strata)
    info = []
    for s in range(int(label_of_level.max()) + 1):
        member = levels[label_of_level == s]
        mask = labels == s
        info.append({"stratum": f"S{s}" if len(levels) > NATURAL_LEVEL_MAX else f"{member[0]:.4f}",
                     "stratum_id": s, "q_level": float(q[mask].mean()) if mask.any() else np.nan,
                     "q_lo": float(member.min()), "q_hi": float(member.max()), "N_s": int(mask.sum())})
    return labels, pd.DataFrame(info)


def strata_rows(cell, model, dataset, basis, mono, pairs, dis, labels, info, anchors):
    """逐 (pair, 層) 一列，外加該配對的全樣本 ALL 列。"""
    rows = []
    for l1, l2 in pairs:
        pair = f"{l1}_vs_{l2}"
        i, j = LANG_INDEX[l1], LANG_INDEX[l2]
        anchor = anchors.get((cell, pair))
        if anchor not in (l1, l2):
            anchor = l1 if mono[i].mean() >= mono[j].mean() else l2      # 退回全樣本 argmax
        hi, lo = (i, j) if anchor == l1 else (j, i)
        head = {"cell": cell, "model": model, "dataset": dataset, "basis": basis, "pair": pair,
                "has_english": ENGLISH in (l1, l2), "anchor": anchor}
        masks = [{"stratum": "ALL", "stratum_id": -1, "q_level": np.nan,
                  "mask": np.ones(len(labels), dtype=bool)}]
        masks += [{"stratum": r["stratum"], "stratum_id": r["stratum_id"], "q_level": r["q_level"],
                   "mask": labels == r["stratum_id"]} for _, r in info.iterrows()]
        for spec in masks:
            s = baseline_stats(mono[hi], mono[lo], dis[(l1, l2)], spec["mask"])
            rows.append({**head, "stratum": spec["stratum"], "stratum_id": spec["stratum_id"],
                         "q_level": spec["q_level"], **s, "enough": bool(s["D"] >= MIN_STRATUM)})
    return rows


def adjust_rows(strata):
    """
    共同權重 = cell 內十個配對的 D 合併後的層分布（⑤ 的「兩組合併後的層分布」）。
    只標準化條件在 D 上的量；d 不標準化 —— 主分析基底的 q 與配對無關，各配對的題目層分布本來就相同。
    """
    per_level = strata[strata["stratum"] != "ALL"]
    keys = ["cell", "basis", "stratum_id"]
    weight = per_level.groupby(keys, as_index=False)["D"].sum().rename(columns={"D": "w"})
    joined = per_level.merge(weight, on=keys, how="left")
    out = []
    for (cell, basis, pair), g in joined.groupby(["cell", "basis", "pair"], sort=True):
        row = {"cell": cell, "basis": basis, "pair": pair, "has_english": bool(g["has_english"].iloc[0])}
        total_w = float(g["w"].sum())
        for q in ADJUSTABLE:
            ok = g[np.isfinite(g[q].to_numpy(dtype=float))]
            w = ok["w"].to_numpy(dtype=float)
            row[f"{q}_adj"] = float(w @ ok[q].to_numpy(dtype=float) / w.sum()) if w.sum() > 0 else np.nan
            row[f"{q}_weight_kept"] = float(w.sum() / total_w) if total_w > 0 else np.nan
        out.append(row)
    adj = pd.DataFrame(out)
    raw = strata[strata["stratum"] == "ALL"][["cell", "basis", "pair"] + list(ADJUSTABLE)]
    return adj.merge(raw.rename(columns={q: f"{q}_raw" for q in ADJUSTABLE}), on=["cell", "basis", "pair"])


def composition_counterfactual(strata):
    """
    只換權重的反事實：把「不含英文組的逐層 recovery_blind profile」套上「含英文組的層分布」，
    得到題目組成單獨能造成的組間差，再與實際觀測的組間差比較。
    recovery_blind 是 baseline 唯一算得出來的聚合端量（recovery 要 challenge 內容），
    這裡拿它當「組成最多能把聚合端推動多少」的校準：
        comp = Σ_s (w_含英文[s] − w_不含英文[s]) · recovery_blind_不含英文[s]
    """
    lv = strata[strata["stratum"] != "ALL"].copy()
    lv["n_R"] = lv["n_A"] + lv["n_B"]
    rows = []
    for basis, b in lv.groupby("basis", sort=True):
        comps, obs = [], []
        for _, g in b.groupby("cell", sort=True):
            w = g[g["has_english"]].groupby("stratum")[["D", "n_R", "n_A"]].sum()
            o = g[~g["has_english"]].groupby("stratum")[["D", "n_R", "n_A"]].sum()
            idx = w.index.union(o.index)
            w, o = w.reindex(idx, fill_value=0), o.reindex(idx, fill_value=0)
            keep = (w["n_R"] > 0) & (o["n_R"] > 0)                 # 兩組都算得出 recovery_blind 的層
            if not keep.any():
                continue
            r_w = (2 * w["n_A"][keep] / w["n_R"][keep] - 1).to_numpy()
            r_o = (2 * o["n_A"][keep] / o["n_R"][keep] - 1).to_numpy()
            pw = w["D"][keep].to_numpy(dtype=float); pw /= pw.sum()
            po = o["D"][keep].to_numpy(dtype=float); po /= po.sum()
            comps.append(float((pw - po) @ r_o))
            obs.append(float(pw @ r_w - po @ r_o))
        comp, observed = float(np.mean(comps)), float(np.mean(obs))
        rows.append({"basis": basis, "comp_effect": comp, "observed": observed,
                     "share_explained": comp / observed if observed else np.nan, "cells": len(comps)})
    return pd.DataFrame(rows)


def cell_mean_diff(df, column, rng, n_perm):
    """cell 內 (含英文 − 不含英文) 的等權平均，以及 cell 內重排 has_english 標籤的 permutation p。"""
    obs, draws = [], []
    for _, g in df.groupby("cell", sort=True):
        v = g[column].to_numpy(dtype=float)
        en = g["has_english"].to_numpy(dtype=bool)
        ok = np.isfinite(v)
        vals = np.where(ok, v, 0.0)
        obs.append(np.nanmean(v[en]) - np.nanmean(v[~en]))

        # 重排後「不含英文」組就是補集，用總和相減即可，不必逐次重算
        idx = permute_masks(rng, len(v), int(en.sum()), n_perm)
        s_with, n_with = vals[idx].sum(axis=1), ok[idx].sum(axis=1)
        s_rest, n_rest = vals.sum() - s_with, ok.sum() - n_with
        mean_with = np.divide(s_with, n_with, out=np.full(s_with.shape, np.nan), where=n_with > 0)
        mean_rest = np.divide(s_rest, n_rest, out=np.full(s_rest.shape, np.nan), where=n_rest > 0)
        draws.append(mean_with - mean_rest)
    return float(np.mean(obs)), perm_pvalue(float(np.mean(obs)), np.mean(draws, axis=0), True)


# ----------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="0A-5: 難度指標 q_i 與兩組分歧題目的難度分布比較")
    parser.add_argument("--perm", type=int, default=10000, help="permutation 次數（cell 內重排 has_english）")
    parser.add_argument("--seed", type=int, default=0, help="permutation 的 seed")
    parser.add_argument("--strata", type=int, default=5, help="相異 q 值太多時 tie-aware 分位分幾層")
    parser.add_argument("--rb-pairs", default=RB_PAIRS_CSV, help="0A-1 表 1，用它的 anchor 欄")
    parser.add_argument("--outdir", default=".", help="CSV 輸出目錄")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    rng = np.random.default_rng(args.seed)

    # ---- 讀取 ----
    print("📦 讀取 baseline（逐檔載入）...")
    baseline = extract_baseline()
    cell_keys = sorted(baseline)
    anchors = load_anchors(args.rb_pairs)

    mono_by_key, ids_by_key, pairs_by_key, dis_by_key = {}, {}, {}, {}
    for model, dataset in cell_keys:
        cell = f"{model}|{dataset}"
        ids, mono = build_mono(cell, baseline[(model, dataset)])
        pairs = list_pairs(model, dataset)
        if len(pairs) != 10:
            raise ValueError(f"{cell}: challenge 檔名推出的配對應為 10 個，實際 {len(pairs)} 個")
        ids_by_key[(model, dataset)] = ids
        mono_by_key[(model, dataset)] = mono
        pairs_by_key[(model, dataset)] = pairs
        dis_by_key[(model, dataset)] = build_disagreement(cell, baseline[(model, dataset)], ids, pairs)
    print(f"   {len(cell_keys)} 個 cell，各 10 個配對（含英文 4 / 不含英文 6）")

    # cross_model 基底：同資料集的其他模型 × 五語言
    cross_by_key = {}
    for model, dataset in cell_keys:
        others = [(m, d) for (m, d) in cell_keys if d == dataset and m != model
                  and ids_by_key[(m, d)] == ids_by_key[(model, dataset)]]
        cross_by_key[(model, dataset)] = np.vstack([mono_by_key[k] for k in others]) if others else None

    # ---- Step 1 + Step 2 ----
    pair_rows, group_rows, item_rows = [], [], []
    perm_acc = {b: {"dq": [], "ks": []} for b in BASES}
    obs_acc = {b: {"dq": [], "ks": []} for b in BASES}

    for model, dataset in cell_keys:
        cell = f"{model}|{dataset}"
        mono, pairs = mono_by_key[(model, dataset)], pairs_by_key[(model, dataset)]
        dis = dis_by_key[(model, dataset)]
        has_en = np.array([ENGLISH in p for p in pairs])
        bases = difficulty_bases(mono, pairs, cross_by_key[(model, dataset)])

        for basis in BASES:
            if basis not in bases:
                continue
            q_by_pair, n_runs = bases[basis]
            levels = level_grid(q_by_pair, dis, pairs)
            C = counts_matrix(q_by_pair, dis, pairs, levels)

            for k, p in enumerate(pairs):
                pair_rows.append({"cell": cell, "model": model, "dataset": dataset,
                                  "pair": f"{p[0]}_vs_{p[1]}", "has_english": bool(has_en[k]),
                                  "basis": basis, "n_runs": n_runs[p], "n_levels": len(levels),
                                  **describe(C[k], levels)})

            obs, dq_draws, ks_draws = cell_step2(C, has_en, levels, rng, args.perm)
            group_rows.append({
                "cell": cell, "model": model, "dataset": dataset, "basis": basis, "scope": "per_pair",
                "n_levels": len(levels),
                **{f"with_{k}": v for k, v in obs["with"].items()},
                **{f"without_{k}": v for k, v in obs["without"].items()},
                "d_mean_q": obs["d_mean_q"], "ks_D": obs["ks"],
                "ks_p_scipy": np.nan, "perm_p_d_mean_q": perm_pvalue(obs["d_mean_q"], dq_draws, True),
                "perm_p_ks": perm_pvalue(obs["ks"], ks_draws, False),
            })
            obs_acc[basis]["dq"].append(obs["d_mean_q"])
            obs_acc[basis]["ks"].append(obs["ks"])
            perm_acc[basis]["dq"].append(dq_draws)
            perm_acc[basis]["ks"].append(ks_draws)

            # 聯集版（只有 pair-invariant 的基底才有單一 q）
            if basis in PAIR_INVARIANT:
                q = q_by_pair[pairs[0]]
                u_en = np.any([dis[p] for k, p in enumerate(pairs) if has_en[k]], axis=0)
                u_no = np.any([dis[p] for k, p in enumerate(pairs) if not has_en[k]], axis=0)
                cu = counts_matrix({0: q, 1: q}, {0: u_en, 1: u_no}, [0, 1], levels)
                group_rows.append({
                    "cell": cell, "model": model, "dataset": dataset, "basis": basis, "scope": "union",
                    "n_levels": len(levels),
                    **{f"with_{k}": v for k, v in describe(cu[0], levels).items()},
                    **{f"without_{k}": v for k, v in describe(cu[1], levels).items()},
                    "d_mean_q": float(_mean_q(cu[0], levels) - _mean_q(cu[1], levels)),
                    "ks_D": float(_ks(cu[0], cu[1])),
                    "ks_p_scipy": float(stats.ks_2samp(q[u_en], q[u_no]).pvalue),
                    "perm_p_d_mean_q": np.nan, "perm_p_ks": np.nan,
                })

        # 逐題表：pair-invariant 基底的 q + 被幾個配對觸發
        q_all5 = bases["all5"][0][pairs[0]]
        q_cross = bases["cross_model"][0][pairs[0]] if "cross_model" in bases else np.full(len(q_all5), np.nan)
        n_en = np.sum([dis[p] for k, p in enumerate(pairs) if has_en[k]], axis=0)
        n_no = np.sum([dis[p] for k, p in enumerate(pairs) if not has_en[k]], axis=0)
        for pos, q_id in enumerate(ids_by_key[(model, dataset)]):
            item_rows.append({"cell": cell, "model": model, "dataset": dataset, "id": q_id,
                              "q_all5": float(q_all5[pos]), "q_cross_model": float(q_cross[pos]),
                              "n_pairs_with_en_D": int(n_en[pos]), "n_pairs_without_en_D": int(n_no[pos])})

    pair_table = pd.DataFrame(pair_rows)
    group_table = pd.DataFrame(group_rows)
    item_table = pd.DataFrame(item_rows)

    # ---- 列印 Step 1 ----
    print("\n" + "=" * 110)
    print("📋 Step 1  難度指標 q_i：四種基底")
    print("=" * 110)
    step1 = pair_table.groupby("basis").agg(
        n_runs_min=("n_runs", "min"), n_runs_max=("n_runs", "max"),
        n_levels_max=("n_levels", "max"), D_mean=("n_items", "mean"),
        q_mean_with_en=("q_mean", lambda s: s[pair_table.loc[s.index, "has_english"]].mean()),
        q_mean_without_en=("q_mean", lambda s: s[~pair_table.loc[s.index, "has_english"]].mean()),
    ).reindex([b for b in BASES if b in set(pair_table["basis"])])
    print(step1.to_string(float_format=lambda x: f"{x:.4f}"))

    # ---- 列印 Step 2 ----
    print("\n" + "=" * 110)
    print(f"📊 Step 2  兩組分歧題目的 q 分布（逐對；cell 內比較後對 {len(cell_keys)} 個 cell 等權平均）")
    print("=" * 110)
    summary = []
    for basis in BASES:
        if not obs_acc[basis]["dq"]:
            continue
        dq = float(np.mean(obs_acc[basis]["dq"]))
        ks = float(np.mean(obs_acc[basis]["ks"]))
        p_dq = perm_pvalue(dq, np.mean(perm_acc[basis]["dq"], axis=0), True)
        p_ks = perm_pvalue(ks, np.mean(perm_acc[basis]["ks"], axis=0), False)
        ks_ok = ks <= OVERLAP_KS if basis in KS_COMPARABLE else True
        summary.append({"basis": basis, "d_mean_q": dq, "abs_d_mean_q": float(np.mean(np.abs(obs_acc[basis]["dq"]))),
                        "ks_D": ks, "ks_comparable": basis in KS_COMPARABLE,
                        "perm_p_d_mean_q": p_dq, "perm_p_ks": p_ks,
                        "cells_dq_lt_0": int(np.sum(np.array(obs_acc[basis]["dq"]) < 0)),
                        "pass_dq": abs(dq) <= OVERLAP_DQ, "pass_ks": ks_ok, "pass_p": p_dq > OVERLAP_P,
                        "overlap": bool(abs(dq) <= OVERLAP_DQ and ks_ok and p_dq > OVERLAP_P),
                        "primary": basis == PRIMARY_BASIS})
    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n   判定門檻：|Δmean q| ≤ {OVERLAP_DQ}  且  KS D ≤ {OVERLAP_KS}  且  permutation p > {OVERLAP_P}")
    print("   Δmean q = 含英文 − 不含英文（負 = 含英文組觸發的題目比較「難」）；cells_dq_lt_0 = 16 格中為負的格數")
    print("   abs_d_mean_q 是各 cell 取絕對值再平均，只是參考；判定用有號的 Δmean q。")
    print("   ⚠️ loo_noen 兩組的 q 支撐集不同（含英文 3 個 run、不含英文 2 個 run），KS D 會被撐大 → 不納入判定。")

    union = group_table[group_table["scope"] == "union"].groupby("basis")[
        ["with_q_mean", "without_q_mean", "d_mean_q", "ks_D", "with_n_items", "without_n_items"]].mean()
    print("\n   聯集版（cell 內 4 個 / 6 個配對的 D 取聯集，每題只算一次）：")
    print("   " + union.to_string(float_format=lambda x: f"{x:.4f}").replace("\n", "\n   "))

    # 文字直方圖（主分析基底，cell 內比例先算再等權平均）
    hist = []
    for model, dataset in cell_keys:
        mono, pairs = mono_by_key[(model, dataset)], pairs_by_key[(model, dataset)]
        dis = dis_by_key[(model, dataset)]
        has_en = np.array([ENGLISH in p for p in pairs])
        q_by_pair, _ = difficulty_bases(mono, pairs, cross_by_key[(model, dataset)])[PRIMARY_BASIS]
        levels = level_grid(q_by_pair, dis, pairs)
        C = counts_matrix(q_by_pair, dis, pairs, levels)
        w, wo = C[has_en].sum(axis=0), C[~has_en].sum(axis=0)
        hist.append(pd.DataFrame({"q": levels, "with": w / w.sum(), "without": wo / wo.sum()}))
    hist = pd.concat(hist).groupby("q").mean()
    print("\n" + "-" * 110)
    print(f"   分歧題目的 q 分布（基底 {PRIMARY_BASIS}，cell 等權平均）   █ 含英文 (n=4/cell)   ░ 不含英文 (n=6/cell)")
    print("-" * 110)
    for q, row in hist.iterrows():
        print(f"   q={q:<6.2f}  含英文 {row['with']:6.1%} {'█' * round(row['with'] * 60)}")
        print(f"   {'':<8}  不含英  {row['without']:6.1%} {'░' * round(row['without'] * 60)}")

    verdict = summary.set_index("basis").loc[PRIMARY_BASIS]
    mark = lambda ok: "✓" if ok else "✗"
    print("\n" + "=" * 110)
    print(f"   主分析基底 {PRIMARY_BASIS} 的三個條件："
          f"  |Δmean q| ≤ {OVERLAP_DQ} {mark(verdict['pass_dq'])}"
          f"   KS D ≤ {OVERLAP_KS} {mark(verdict['pass_ks'])}"
          f"   perm p > {OVERLAP_P} {mark(verdict['pass_p'])}")
    if verdict["overlap"]:
        print(f"✅ 兩組 q 分布幾乎重疊 → 題目組成沒有差異 → 解釋 (a) 排除，是 (b) → 不必做完整分層（③~⑦ 可省）")
    else:
        print(f"❌ 兩組 q 分布明顯不同 → 無法排除 (a) → 繼續做 ③~⑦")
    others = summary[summary["basis"] != PRIMARY_BASIS]
    agree = (others["overlap"] == verdict["overlap"]).all()
    print(f"   其他三個基底的判定{'一致' if agree else '不一致 ⚠️ 需要看是哪個基底翻盤'}："
          + "  ".join(f"{r['basis']}={'重疊' if r['overlap'] else '不同'}"
                      f"({mark(r['pass_dq'])}{mark(r['pass_ks'])}{mark(r['pass_p'])})"
                      for _, r in others.iterrows()))
    print("=" * 110)

    # ---- ④ 的 baseline-only 部分 ----
    strata_all = []
    for basis in STRATA_BASES:
        for model, dataset in cell_keys:
            cell = f"{model}|{dataset}"
            mono, pairs = mono_by_key[(model, dataset)], pairs_by_key[(model, dataset)]
            bases = difficulty_bases(mono, pairs, cross_by_key[(model, dataset)])
            if basis not in bases:
                continue
            labels, info = strata_labels(bases[basis][0][pairs[0]], args.strata)
            strata_all.extend(strata_rows(cell, model, dataset, basis, mono, pairs,
                                          dis_by_key[(model, dataset)], labels, info, anchors))
    strata = pd.DataFrame(strata_all)
    adjusted = adjust_rows(strata)

    print("\n" + "=" * 110)
    print(f"🔍 生成端的分層診斷（recovery 需要 challenge 內容，留給 ④）")
    print("=" * 110)
    for basis in STRATA_BASES:
        per_level = strata[(strata["basis"] == basis) & (strata["stratum"] != "ALL")]
        if per_level.empty:
            continue
        n_levels = per_level["stratum_id"].nunique()
        print(f"\n── 基底 {basis}：{n_levels} 層"
              f"（相異 q 值 ≤ {NATURAL_LEVEL_MAX} 用自然層，否則 tie-aware 分位分 {args.strata} 層）；"
              f"|D(P,s)| ≥ {MIN_STRATUM} 的層 {int(per_level['enough'].sum())}/{len(per_level)}")
        by_level = per_level.groupby(["stratum", "has_english"]).agg(
            pairs=("pair", "size"), D=("D", "mean"), c=("c", "mean"),
            recovery_blind=("recovery_blind", "mean"), enough=("enough", "sum")).unstack()
        print(by_level.to_string(float_format=lambda x: f"{x:.4f}"))

        # 循環診斷：層內 c 的 SD。若 ≈ 0，代表 c 是 q 的算術後果而不是實測量
        sd = per_level.groupby("stratum")["c"].std()
        degenerate = sd[(sd < 1e-9) | sd.isna()].index.tolist()
        if degenerate:
            print(f"   ⚠️ 層 {degenerate} 的 c 在 160 個配對上 SD = 0 → c 是 q 的算術後果，不是實測量。"
                  f"（{basis} 的基底含配對自己的兩個語言 → 循環）")

    print("\n   raw vs q-adjusted（共同權重 = cell 內十個配對合併後的 D 層分布；cell 內差異再等權平均）")
    rows = []
    for basis in STRATA_BASES:
        sub = adjusted[adjusted["basis"] == basis]
        if sub.empty:
            continue
        for q in ADJUSTABLE:
            raw_d, raw_p = cell_mean_diff(sub, f"{q}_raw", np.random.default_rng(args.seed), args.perm)
            adj_d, adj_p = cell_mean_diff(sub, f"{q}_adj", np.random.default_rng(args.seed), args.perm)
            rows.append({"basis": basis, "quantity": q, "diff_raw": raw_d, "perm_p_raw": raw_p,
                         "diff_adjusted": adj_d, "perm_p_adjusted": adj_p, "shrink": adj_d - raw_d,
                         "weight_kept": float(sub[f"{q}_weight_kept"].mean())})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("   diff = 含英文 − 不含英文。diff_adjusted ≈ diff_raw → 這個量的組間差不是題目組成造成的。")
    print("   ⚠️ recovery_blind 的 raw 是「全樣本 + H1 多數決錨點」，與 0A-1 表 1 的 recovery_blind_H2（H2 上算）不同。")

    print("\n" + "-" * 110)
    print("   組成單獨能造成多大的聚合端差距？（把不含英文組的逐層 recovery_blind profile 套上含英文組的層分布）")
    print("-" * 110)
    counterfactual = composition_counterfactual(strata)
    print(counterfactual.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("   comp_effect = 只換權重造成的差；observed = 實際組間差；share_explained = 組成解釋掉的比例。")
    print("   ⚠️ 這是用 recovery_blind 當代理。真正的 recovery 要等 ④ 讀 result/challenge 才能算。")

    # ---- 輸出 ----
    os.makedirs(args.outdir, exist_ok=True)
    outputs = [(pair_table, "difficulty_q_pairs.csv"), (group_table, "difficulty_q_groups.csv"),
               (item_table, "difficulty_q_items.csv"), (strata, "difficulty_strata_pairs.csv"),
               (adjusted, "difficulty_strata_adjusted.csv")]
    print()
    for table, name in outputs:
        path = os.path.join(args.outdir, name)
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"💾 {name} ({len(table)} 列) → {path}")


if __name__ == "__main__":
    main()
