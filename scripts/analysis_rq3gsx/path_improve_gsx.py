"""
path_improve_gsx.py — RQ3-GSX：依正確率加權的投票（WV）下，和其他條最不像的那條還是最不值得改嗎？
（result/analysis/rq3gsx/rq3gsx_criteria.md）

全程離線：不呼叫 API，不重跑任何東西，不修改現有檔案。判定標準確認前不執行。
    1. 載入 16 個區塊並編碼（沿用 Analysis/pathImprove.py 的核對）；兩兩一致率（8 個弱模型區塊，子集一）；三個 K 的菜單與分組
    2. 開跑前的檢查（第 8 節，依序做，任何一項不過就停，不寫輸出）：
       重現 RQ3-GS / RQ3-GSK 的 V；重現 RQ3-G 的 12 條 WV 與 RQ3-GK K = 3 的 WV − SB；菜單表；權重相同時 WV = V；換成自己；手算
    3. K = 3、5、7：每個（區塊、菜單、供體）的逐切分分子與分母（Analysis/pathImproveGSX.py），依第 4 節加總成各群的效果
       （多個行程平行；每個工作是（K、區塊、一段菜單），結果依工作順序合併，所以和行程數無關）
    4. 輸出（--out-dir）：rq3gsx_blocks.csv、rq3gsx_menu_blocks.csv、rq3gsx_substitutions.csv、兩張圖、report.md

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq3gsx/path_improve_gsx.py [--workers N]
"""
import os
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # 每個行程單執行緒，平行靠多個行程

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import hashlib
import itertools
import math
import multiprocessing as mp
import pickle
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Analysis.preregistration import sha256, confirmationLine
from Analysis.blockStats import summarize, fourState
from Analysis.menuJudgeStats import blocksWithSplits
from Analysis.menuVote import MODELS, DATASETS, loadPathBlock
from Analysis.splitHalf import makeSplits
from Analysis.pathImprove import M12, HOSTS, DONORS, encodeBlocks, checkVote
from Analysis.pathImproveK import SCALE, HostDonorK, drawMenus
from Analysis.pathImproveGS import GROUPS, menuSets, agreementRates, menuGrouping, assignGroups, onehot, GroupSums
from Analysis.pathImproveGSK import TRANSLATED, menuTableFrame, menuTableCsv, hostSide
from Analysis.pathImproveGSX import menuSplits, equalWeightMismatch

OUT_DIR = "result/analysis/rq3gsx"
CRITERIA_FILE = "rq3gsx_criteria.md"
KS = (3, 5, 7)
THRESHOLDS = {3: 0.50, 5: 0.30, 7: 0.20}
REFERENCE_THRESHOLD = 0.5
CLEAR = GROUPS[0]
RQ3G_SUBS = "result/analysis/rq3g/rq3g_substitutions.csv"
RQ3GK_BLOCKS = "result/analysis/rq3gk/rq3gk_k_blocks.csv"
RQ3GS_BLOCKS = "result/analysis/rq3gs/rq3gs_blocks.csv"
RQ3GS_MENU_BLOCKS = "result/analysis/rq3gs/rq3gs_menu_blocks.csv"
RQ3GSK_BLOCKS = "result/analysis/rq3gsk/rq3gsk_blocks.csv"
REPRO_V = {3: ("+1.75", "+1.03", "+2.46", 8), 5: ("+0.98", "+0.48", "+1.47", 8), 7: ("+0.63", "+0.33", "+0.94", 8)}
REPRO_GK3 = ("-2.42", "-3.95", "-0.88")
HAND = ("gpt4omini", "mmlu", ("GS-001", "GS5-001"))
TOL = 1e-9
CHUNK = 48
KEYS = ("V_C", "V_A", "W_C", "W_A", "Wfix_C", "V_Csim", "W_Csim")
MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
GROUP_EN = {"明顯落單": "clear odd-one-out", "中間": "middle", "對稱": "symmetric"}
READINGS = {
    "正向成立": "依正確率加權的投票下，和其他條最不像的那條仍然最不值得改。",
    "兩者相當": "加權之後落單與否沒有實質差別；落單規則只適用於等權投票。",
    "無法判定": "只列數字；不能把落單規則寫成適用於加權投票。",
}
LIMITS = ["(a) 明顯落單組裡落單的 path 全是翻譯的語言 path，分不開「不像」「是翻譯的」「比較弱」。",
          "(b) 菜單只由 12 條 path 排出來，彼此不獨立；三項判定用同一批題目與 path。",
          "(c) 這些菜單上等權投票的結果已經看過，而加權投票和等權投票高度相關，所以這不是獨立的確認。",
          "(d) 加權投票的權重要用標準答案（選擇半）；落單分數不用。",
          "(e) 替換同時改變那條 path 的答案與它的權重，兩者沒有分開。第 7 節 11 把兩者拆開，只報告。"]
WV_COLOR, V_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6a64", "#d9d8d2", "#fcfcfb"

G = {}   # 平行的工作行程（fork）共用：coded、splits


def parseArgs():
    parser = ArgumentParser(description="RQ3-GSX: is the odd-one-out path still the least worth improving under accuracy-weighted voting?")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--cache", default=None, help="（除錯用）算完後把檢查與區塊值存成 pickle；有這個檔就跳過計算，只重寫輸出")
    return parser.parse_args()


# ------------------------------------------------------------------
# 格式
# ------------------------------------------------------------------
def pp(x: float, d: int = 2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:+.{d}f}"


def S(values) -> dict:
    return summarize(np.asarray(values, dtype=float))


def ciText(s: dict, d: int = 2, npos: bool = True) -> str:
    out = f"{pp(s['mean'], d)}（{pp(s['ci_low'], d)} 到 {pp(s['ci_high'], d)}）"
    return out + (f"，{s['n_positive']}/{s['n_blocks']}" if npos else "")


def shown(s: dict) -> tuple:
    return (f"{s['mean']:+.2f}", f"{s['ci_low']:+.2f}", f"{s['ci_high']:+.2f}", s["n_positive"])


def md(df: pd.DataFrame) -> str:
    return df.astype(object).where(df.notna(), "—").astype(str).to_markdown(index=False, disable_numparse=True)


def label(h: str, d: str) -> str:
    return f"{MODEL_LABELS[h]} · {d}"


def mergeGS(target: GroupSums, res: dict):
    for name, v in res["num"].items():
        target.num[name] = target.num.get(name, 0.0) + v
        target.den[name] = target.den.get(name, 0.0) + res["den"][name]
        target.n[name] = target.n.get(name, 0) + res["n"][name]


# ------------------------------------------------------------------
# 平行的工作（fork 之後讀 G）
# ------------------------------------------------------------------
def runTask(task):
    mode, K, host, dataset, menus = task
    return {"v": taskV, "eq": taskEqual, "self": taskSelf, "main": taskMain}[mode](K, host, dataset, menus)


def taskV(K, host, dataset, menus):
    """第 8 節 1：只走等權投票的路徑，明顯落單組的其他條 / 落單那條（規則 C）。"""
    block, splits = G["coded"][(host, dataset)], G["splits"][dataset]
    ctxs = {g: HostDonorK(block, G["coded"][(g, dataset)], splits) for g in DONORS}
    gs = GroupSums()
    for m in menus:
        lone = onehot(hostSide(block, m["codes"], splits, pi_rules=())["lone"], len(m["codes"]))
        for donor in DONORS:
            s = menuSplits(ctxs[donor], m["codes"], wv=False, sim_rules=())
            n2 = s["n2"][:, None].astype(float)
            num, den = s["num"]["V_C"] / (n2 * SCALE), s["den_int"] / n2
            gs.add("others", num, den, ~lone, s["included"])
            gs.add("lone", num, den, lone, s["included"])
    return {"num": gs.num, "den": gs.den, "n": gs.n}


def taskEqual(K, host, dataset, menus):
    """第 8 節 4：權重全設成同一個常數（1、0.5）時 WV 的逐題得分 = V。"""
    block, splits = G["coded"][(host, dataset)], G["splits"][dataset]
    bad = total = 0
    for donor in DONORS:
        ctx = HostDonorK(block, G["coded"][(donor, dataset)], splits)
        for m in menus:
            b, t = equalWeightMismatch(ctx, m["codes"])
            bad, total = bad + b, total + t
    return {"bad": bad, "total": total}


def taskSelf(K, host, dataset, menus):
    """第 8 節 5：供體 = 宿主時，每次切分 WV / V 的分子（C、A）、模擬的分子、分子_固定、分母都是 0，且沒有有效切分。"""
    block, splits = G["coded"][(host, dataset)], G["splits"][dataset]
    ctx = HostDonorK(block, block, splits)
    bad = count = 0
    for m in menus:
        s = menuSplits(ctx, m["codes"])
        count += len(m["codes"])
        zero = all(not np.any(v) for v in s["num"].values()) and all(not np.any(v) for v in s["sim"].values())
        bad += int(not zero or np.any(s["den_int"]) or s["valid"].any() or s["included"].any())
    return {"bad": bad, "count": count}


def taskMain(K, host, dataset, menus):
    """正式計算：一個（K、區塊）的一段菜單。回傳各群的 Σ分子、Σ分母、個數，以及計數、逐替換與逐菜單的列。"""
    block, splits = G["coded"][(host, dataset)], G["splits"][dataset]
    ctxs = {g: HostDonorK(block, G["coded"][(g, dataset)], splits) for g in DONORS}
    gs, cnt = GroupSums(), Counter()
    sub_rows, menu_rows, item6 = [], [], []
    for m in menus:
        codes, group = m["codes"], m["group"]
        Kk = len(codes)
        info = hostSide(block, codes, splits, pi_rules=())
        lone, low, ranks = onehot(info["lone"], Kk), onehot(info["lowest"], Kk), info["ranks"]
        every = np.ones_like(lone)
        mrow, sub6 = defaultdict(float), {"V": [], "W": []}
        for donor in DONORS:
            s = menuSplits(ctxs[donor], codes)
            inc = s["included"]
            n2i = s["n2"]
            n2 = n2i[:, None].astype(float)
            den = s["den_int"] / n2
            vals = {k: v / (n2 * SCALE) for k, v in s["num"].items()}
            vals.update({f"{k}sim": v / (n2 * SCALE) for k, v in s["sim"].items()})

            def add(name, mask, rows=inc, keys=KEYS):
                for k in keys:
                    gs.add(f"{name}|{k}", vals[k], den, mask, rows)

            cnt["excluded"] += int((~inc).sum())
            cnt["total"] += len(inc)
            for name in (group, "all"):
                add(f"E_{name}_others", ~lone)
                add(f"E_{name}_lone", lone)
            add("conv_all", every, keys=("V_C", "W_C", "V_Csim", "W_Csim"))
            if K in (5, 7):
                for r in range(1, K + 1):
                    add(f"rank{r}", ranks == r, keys=("V_C", "W_C"))
            if m["lone"] not in TRANSLATED:
                add("4a_others", ~lone, keys=("W_C",))
                add("4a_lone", lone, keys=("W_C",))
            if group == CLEAR:
                rows_b = inc & (info["lone"] != info["lowest"])
                add("4b_others", ~lone, rows_b, ("W_C",))
                add("4b_lone", lone, rows_b, ("W_C",))
                add("4c_other", ~low, rows_b, ("W_C",))
                add("4c_low", low, rows_b, ("W_C",))
                cnt["use4"] += int(rows_b.sum())
                cnt["tot4"] += int(inc.sum())
            # 第 7 節 8：平手的題目、權重為 0 的 path（可用的切分）
            cnt["tie0_num"] += int(s["tie0"][inc].sum())
            cnt["tie0_den"] += int(n2i[inc].sum())
            cnt["tie1_num"] += int(s["tie1"][inc].sum())
            cnt["tie1_den"] += int(n2i[inc].sum()) * Kk
            cnt["zero0_num"] += int(s["zero0"][inc].sum())
            cnt["zero0_den"] += int(inc.sum()) * Kk
            cnt["zero1_num"] += int(s["zero1"][inc].sum())
            cnt["zero1_den"] += int(inc.sum()) * Kk * Kk
            # 第 7 節 6：每個替換在可用切分上的 100 × (替換後 − 替換後 SB) 的平均
            if inc.any():
                nv = n2i[inc].astype(float)[:, None]
                sb = s["SB_after"][inc] / nv
                for key, agg in (("V", "V_C"), ("W", "W_C")):
                    after = (s["base"][agg][inc][:, None] + s["num"][agg][inc]) / (nv * SCALE)
                    sub6[key].extend((100 * (after - sb)).mean(axis=0).tolist())
            for name, mask in (("others", ~lone), ("lone", lone)):
                mm = mask & inc[:, None]
                for k in ("W_C", "V_C"):
                    mrow[f"{name}_num_{k}"] += float(vals[k][mm].sum())
                mrow[f"{name}_den"] += float(den[mm].sum())
            mrow["excluded_splits"] += int((~inc).sum())
            for j, c in enumerate(codes):
                row = {"K": K, "model": host, "dataset": dataset, "menu": m["mid"], "codes": " ".join(codes), "group": group,
                       "donor": donor, "path": c, "n_valid_splits": int(s["valid"][:, j].sum()), "n_included_splits": int(inc.sum())}
                for k, col in (("W_C", "sum_num_WV_C"), ("W_A", "sum_num_WV_A"), ("V_C", "sum_num_V_C"), ("V_A", "sum_num_V_A"),
                               ("Wfix_C", "sum_num_WVfix_C"), ("W_Csim", "sum_num_WVsim_C"), ("V_Csim", "sum_num_Vsim_C")):
                    row[col] = float(vals[k][inc, j].sum())
                row["sum_den"] = float(den[inc, j].sum())
                row["n_lone"] = int((info["lone"][inc] == j).sum())
                for r in range(1, 8):
                    row[f"n_rank{r}"] = int((ranks[inc, j] == r).sum()) if r <= Kk else np.nan
                sub_rows.append(row)
        item6.append({"menu": m["mid"], "V": float(np.mean(sub6["V"])) if sub6["V"] else np.nan,
                      "W": float(np.mean(sub6["W"])) if sub6["W"] else np.nan})
        menu_rows.append({"K": K, "model": host, "dataset": dataset, "menu": m["mid"], "codes": " ".join(codes), "group": group,
                          "lone_table": m["lone"], "degree_pp": 100 * m["degree"],
                          "excluded_share": mrow["excluded_splits"] / (len(DONORS) * len(info["lone"])),
                          **{k: v for k, v in mrow.items() if k != "excluded_splits"}})
    return {"num": gs.num, "den": gs.den, "n": gs.n, "cnt": cnt, "subs": sub_rows, "menus": menu_rows, "item6": item6}


def runParallel(pool, tasks, what: str, elapsed):
    out = []
    for i, res in enumerate(pool.imap(runTask, tasks, chunksize=1), 1):
        out.append(res)
        if i % max(1, len(tasks) // 10) == 0 or i == len(tasks):
            print(f"  {what}：{i}/{len(tasks)}（{elapsed()}）", flush=True)
    return out


def chunks(menus: list, size: int = CHUNK) -> list:
    return [menus[i:i + size] for i in range(0, len(menus), size)]


# ------------------------------------------------------------------
# §8.6 手算（純迴圈、字串答案、Fraction 的規則 C 得分、math.log 的權重；不用 pathImproveK / GS / GSK / GSX）
# ------------------------------------------------------------------
def handCheck(args, codes: list[str]) -> dict:
    host_model, dataset, _ = HAND
    host = loadPathBlock(args.armdir, host_model, dataset, args.aggdir)
    N = len(host.item_ids)
    splits = makeSplits(N, 200, 0)
    gold = host.gold
    K = len(codes)
    sub1 = [all(host.answered[c][i] for c in M12) for i in range(N)]
    ans = {c: host.answers[c] for c in codes}

    def scoreC(answers: list, weights: list, i: int) -> Fraction:
        tot = {}
        for a, w in zip(answers, weights):
            tot[a] = tot.get(a, 0.0) + w
        top = max(tot.values())
        tied = [a for a, v in tot.items() if v >= top - 1e-9]
        return Fraction(1, len(tied)) if gold[i] in tied else Fraction(0)

    def weight(hits: int, n: int) -> float:
        return max(0.0, math.log((hits + 1) / (n - hits + 1)))

    lone = []
    for h1 in splits:
        idx = [i for i in range(N) if h1[i] and sub1[i]]
        agree = {(a, b): sum(ans[a][i] == ans[b][i] for i in idx) for a, b in itertools.combinations(codes, 2)}
        get = lambda a, b: agree[(a, b)] if (a, b) in agree else agree[(b, a)]
        s = []
        for c in codes:
            others = [d for d in codes if d != c]
            pairs = list(itertools.combinations(others, 2))
            s.append(Fraction(sum(get(a, b) for a, b in pairs), len(pairs)) - Fraction(sum(get(c, d) for d in others), len(others)))
        best = max(s)
        lone.append(max(j for j in range(K) if s[j] == best))          # 相同時取平手順序在後的
    out = {"per_path": {}, "excluded": {}, "sums": defaultdict(float)}
    cor = {c: [ans[c][i] == gold[i] for i in range(N)] for c in codes}
    ones = [1.0] * K
    for donor_model in DONORS:
        donor = loadPathBlock(args.armdir, donor_model, dataset, args.aggdir)
        sub2 = [sub1[i] and all(donor.answered[c][i] for c in M12) for i in range(N)]
        dcor = {c: [donor.answers[c][i] == gold[i] for i in range(N)] for c in codes}
        excluded, sums = 0, {c: {"W": 0.0, "V": 0.0, "den": 0.0} for c in codes}
        for r, h1 in enumerate(splits):
            H1 = [i for i in range(N) if h1[i] and sub2[i]]
            H2 = [i for i in range(N) if not h1[i] and sub2[i]]
            k1 = {c: sum(cor[c][i] for i in H1) for c in codes}
            d1 = {c: sum(dcor[c][i] for i in H1) for c in codes}
            k2 = {c: sum(cor[c][i] for i in H2) for c in codes}
            d2 = {c: sum(dcor[c][i] for i in H2) for c in codes}
            if not all(d1[c] > k1[c] and d2[c] > k2[c] for c in codes):
                excluded += 1
                continue
            w = [weight(k1[c], len(H1)) for c in codes]
            base = {i: [ans[c][i] for c in codes] for i in H2}
            baseW = sum((scoreC(base[i], w, i) for i in H2), Fraction(0))
            baseV = sum((scoreC(base[i], ones, i) for i in H2), Fraction(0))
            for j, c in enumerate(codes):
                wj = list(w)
                wj[j] = weight(d1[c], len(H1))
                after = {i: [donor.answers[c][i] if d == c else ans[d][i] for d in codes] for i in H2}
                numW = float((sum((scoreC(after[i], wj, i) for i in H2), Fraction(0)) - baseW) / len(H2))
                numV = float((sum((scoreC(after[i], ones, i) for i in H2), Fraction(0)) - baseV) / len(H2))
                den = (d2[c] - k2[c]) / len(H2)
                sums[c]["W"] += numW
                sums[c]["V"] += numV
                sums[c]["den"] += den
                key = "lone" if j == lone[r] else "others"
                out["sums"][f"{key}_W"] += numW
                out["sums"][f"{key}_V"] += numV
                out["sums"][f"{key}_den"] += den
        out["per_path"][donor_model] = sums
        out["excluded"][donor_model] = excluded
    sm = out["sums"]
    out["effects"] = {f"{g}_{k}": 10 * sm[f"{g}_{k}"] / sm[f"{g}_den"] for g in ("others", "lone") for k in ("W", "V")}
    return out


def programHand(codes: list[str]) -> dict:
    """§8.6 的程式那一邊：同一個區塊、菜單，用正式計算的函式。"""
    host_model, dataset, _ = HAND
    block, splits = G["coded"][(host_model, dataset)], G["splits"][dataset]
    lone_idx = hostSide(block, codes, splits, pi_rules=())["lone"]
    lone = onehot(lone_idx, len(codes))
    out = {"per_path": {}, "excluded": {}, "sums": defaultdict(float)}
    for donor in DONORS:
        s = menuSplits(HostDonorK(block, G["coded"][(donor, dataset)], splits), codes)
        inc = s["included"]
        n2 = s["n2"][:, None].astype(float)
        W, V, den = s["num"]["W_C"] / (n2 * SCALE), s["num"]["V_C"] / (n2 * SCALE), s["den_int"] / n2
        out["per_path"][donor] = {c: {"W": float(W[inc, j].sum()), "V": float(V[inc, j].sum()), "den": float(den[inc, j].sum())}
                                  for j, c in enumerate(codes)}
        out["excluded"][donor] = int((~inc).sum())
        for g, mask in (("others", ~lone), ("lone", lone)):
            mm = mask & inc[:, None]
            out["sums"][f"{g}_W"] += float(W[mm].sum())
            out["sums"][f"{g}_V"] += float(V[mm].sum())
            out["sums"][f"{g}_den"] += float(den[mm].sum())
    sm = out["sums"]
    out["effects"] = {f"{g}_{k}": 10 * sm[f"{g}_{k}"] / sm[f"{g}_den"] for g in ("others", "lone") for k in ("W", "V")}
    return out


# ------------------------------------------------------------------
# 圖
# ------------------------------------------------------------------
def styleAxes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)


def saveFig(fig, path: str, sha: str):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", facecolor=SURFACE,
                    metadata={"Subject" if ext == "pdf" else "Description": f"rq3gsx_criteria.md sha256 {sha}"})
    plt.close(fig)


def plotJudgments(stats: dict, path: str, sha: str):
    """(a) K = 3、5、7 明顯落單組的 E：WV 與 V 並排（8 個區塊的平均與 95% t 區間）。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(KS))
    for agg, color, marker, dx, name in (("W", WV_COLOR, "o", -0.09, "WV (accuracy-weighted vote)"),
                                         ("V", V_COLOR, "s", 0.09, "V (equal-weight vote)")):
        s = [stats[(K, agg)] for K in KS]
        ax.vlines(x + dx, [v["ci_low"] for v in s], [v["ci_high"] for v in s], color=color, linewidth=1.8, zorder=2)
        ax.scatter(x + dx, [v["mean"] for v in s], s=56, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.4,
                   zorder=3, label=name)
        for xi, v in zip(x + dx, s):
            ax.annotate(f"{v['mean']:+.2f}", (xi, v["mean"]), xytext=(7 if dx > 0 else -7, 0), textcoords="offset points",
                        ha="left" if dx > 0 else "right", va="center", fontsize=8, color=INK)
    for xi, K in zip(x, KS):
        ax.hlines(THRESHOLDS[K], xi - 0.28, xi + 0.28, color=MUTED, linewidth=0.9, linestyles=(0, (3, 2)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {K}\n({62 if K == 3 else 254} menus)" for K in KS], fontsize=8.5, color=INK)
    styleAxes(ax)
    ax.set_ylabel("E = others − odd one out\n(vote pp per 10pp path gain)", color=MUTED, fontsize=9)
    ax.set_title("RQ3-GSX (a): clear odd-one-out group, rule C\n8 weak blocks, 95% t interval; dashed = threshold",
                 color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
    saveFig(fig, path, sha)


def plotRanks(stats: dict, path: str, sha: str):
    """(b) 依落單名次的效果（全部確認用的菜單）：K = 5、7 各一格，WV 與 V 各一條線。"""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, K in zip(axes, (5, 7)):
        ranks = np.arange(1, K + 1)
        for agg, color, marker, dx, name in (("W", WV_COLOR, "o", -0.07, "WV"), ("V", V_COLOR, "s", 0.07, "V")):
            s = [stats[(K, agg, r)] for r in ranks]
            ax.vlines(ranks + dx, [v["ci_low"] for v in s], [v["ci_high"] for v in s], color=color, linewidth=1.5, zorder=2)
            ax.plot(ranks + dx, [v["mean"] for v in s], color=color, linewidth=2, marker=marker, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label=name)
        ax.set_xticks(ranks)
        ax.set_xticklabels(["1\nmost odd"] + [str(r) for r in range(2, K)] + [f"{K}\nleast odd"], fontsize=8.5, color=INK)
        styleAxes(ax)
        ax.set_title(f"K = {K} (762 menus)", color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("rank by s_i on the selection half", color=MUTED, fontsize=9)
    axes[0].set_ylabel("effect: vote pp per 10pp path gain", color=MUTED, fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("RQ3-GSX (b): effect by odd-one-out rank, rule C (8 weak blocks, 95% t interval)", color=INK, fontsize=10, x=0.01,
                 ha="left")
    saveFig(fig, path, sha)


# ------------------------------------------------------------------
# 區塊層級的量
# ------------------------------------------------------------------
def blockValues(K: int, h: str, d: str, gs: GroupSums, cnt: Counter, item6: list) -> dict:
    eff = lambda name, key: gs.effect(f"{name}|{key}") if gs.den.get(f"{name}|{key}", 0) > 0 else np.nan
    E = lambda grp, key: eff(f"E_{grp}_others", key) - eff(f"E_{grp}_lone", key)
    num = lambda name, key: gs.num.get(f"{name}|{key}", np.nan)
    den = lambda name, key: gs.den.get(f"{name}|{key}", np.nan)
    row = {"K": K, "model": h, "dataset": d, "excluded_share": cnt["excluded"] / cnt["total"]}
    for agg, key in (("W", "W_C"), ("V", "V_C")):
        row[f"E_{agg}"] = E(CLEAR, key)
        row[f"others_{agg}"], row[f"lone_{agg}"] = eff(f"E_{CLEAR}_others", key), eff(f"E_{CLEAR}_lone", key)
        for grp in GROUPS[1:] + ["all"]:
            row[f"E_{agg}_{grp}"] = E(grp, key)
        row[f"ratio_{agg}"] = row[f"lone_{agg}"] / row[f"others_{agg}"]
        if K in (5, 7):
            for r in range(1, K + 1):
                row[f"rank{r}_{agg}"] = eff(f"rank{r}", key)
            row[f"rankdiff_{agg}"] = row[f"rank{K}_{agg}"] - row[f"rank1_{agg}"]
        # 第 7 節 5：轉換率（全部 K 條、全部確認用菜單）
        real, sim = num("conv_all", key), num("conv_all", f"{key}sim")
        dd = den("conv_all", key)
        row[f"conv_real_{agg}"], row[f"conv_sim_{agg}"] = real / dd, sim / dd
        row[f"conv_diff_{agg}"] = row[f"conv_real_{agg}"] - row[f"conv_sim_{agg}"]
        row[f"underest_{agg}"] = 1 - row[f"conv_sim_{agg}"] / row[f"conv_real_{agg}"] if row[f"conv_real_{agg}"] > 0 else np.nan
        row[f"sum_num_real_{agg}"], row[f"sum_num_sim_{agg}"], row["sum_den_conv"] = real, sim, dd
        row[f"simratio_{agg}"] = eff(f"E_{CLEAR}_lone", f"{key}sim") / eff(f"E_{CLEAR}_others", f"{key}sim")
        row[f"sim_others_{agg}"], row[f"sim_lone_{agg}"] = eff(f"E_{CLEAR}_others", f"{key}sim"), eff(f"E_{CLEAR}_lone", f"{key}sim")
        # 第 7 節 9：規則 A
        keyA = key.replace("_C", "_A")
        row[f"E_{agg}_A"] = E(CLEAR, keyA)
    # 第 7 節 4
    row["E_W_4a"] = eff("4a_others", "W_C") - eff("4a_lone", "W_C") if "4a_others|W_C" in gs.num else np.nan
    row["E_W_4b"] = eff("4b_others", "W_C") - eff("4b_lone", "W_C")
    row["E_W_4c"] = eff("4c_other", "W_C") - eff("4c_low", "W_C")
    row["share_4bc"] = cnt["use4"] / cnt["tot4"] if cnt["tot4"] else np.nan
    row["n_4bc"] = cnt["use4"]
    # 第 7 節 6
    v6 = [x["V"] for x in item6 if np.isfinite(x["V"])]
    w6 = [x["W"] for x in item6 if np.isfinite(x["W"])]
    row["VmSB_after_pp"], row["WVmSB_after_pp"] = float(np.mean(v6)), float(np.mean(w6))
    row["n_menus6_used"], row["n_menus6_dropped"] = len(w6), len(item6) - len(w6)
    # 第 7 節 7（WV，明顯落單組）
    for g in ("others", "lone"):
        name = f"E_{CLEAR}_{g}|W_C"
        row[f"{g}_vote_pp"] = 100 * gs.num[name] / gs.n[name]
        row[f"{g}_path_pp"] = 100 * gs.den[name] / gs.n[name]
    # 第 7 節 8
    row["tie_before"] = cnt["tie0_num"] / cnt["tie0_den"]
    row["tie_after"] = cnt["tie1_num"] / cnt["tie1_den"]
    row["zero_before"] = cnt["zero0_num"] / cnt["zero0_den"]
    row["zero_after"] = cnt["zero1_num"] / cnt["zero1_den"]
    # 第 7 節 11：權重固定的版本；兩群的效果拆成答案的部分與權重的部分
    for grp, tag in ((CLEAR, "clear"), ("all", "all")):
        row[f"E_Wfix_{tag}"] = E(grp, "Wfix_C")
        for g in ("others", "lone"):
            name = f"E_{grp}_{g}"
            dsum = gs.den[f"{name}|W_C"]
            row[f"{g}_answer_{tag}"] = 10 * gs.num[f"{name}|Wfix_C"] / dsum
            row[f"{g}_weight_{tag}"] = 10 * (gs.num[f"{name}|W_C"] - gs.num[f"{name}|Wfix_C"]) / dsum
            row[f"{g}_answer_num_{tag}"] = gs.num[f"{name}|Wfix_C"]
            row[f"{g}_weight_num_{tag}"] = gs.num[f"{name}|W_C"] - gs.num[f"{name}|Wfix_C"]
    return row


# ------------------------------------------------------------------
# 主程式
# ------------------------------------------------------------------
def main():
    args = parseArgs()
    started = time.time()
    elapsed = lambda: f"{time.time() - started:.0f}s"
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty)")
    criteria_sha = sha256(criteria)
    criteria_text = Path(criteria).read_text(encoding="utf-8")
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    if args.cache and os.path.exists(args.cache):
        with open(args.cache, "rb") as fh:
            saved = pickle.load(fh)
        if saved["criteria_sha"] != criteria_sha:
            raise SystemExit("❌ cache was written under another criteria file")
        print(f"讀取 {args.cache}，只重寫輸出", flush=True)
        writeOutputs(args, criteria_sha, confirmed, saved["gen_time"], **saved["data"])
        return

    # 載入與編碼
    raw, splits = defaultdict(dict), {}
    for block, sp in blocksWithSplits(args.armdir, args.aggdir):
        raw[block.dataset][block.model] = block
        splits[block.dataset] = sp
    coded = {}
    for dataset in DATASETS:
        blocks = [raw[dataset][model] for model in MODELS]
        for pb, cb in zip(blocks, encodeBlocks(blocks)):
            checkVote(pb, cb)
            coded[(cb.model, cb.dataset)] = cb
    del raw
    G["coded"], G["splits"] = coded, splits
    weak_keys = [(h, d) for h in HOSTS for d in DATASETS]
    rates = agreementRates([coded[k] for k in weak_keys])
    print(f"載入與編碼：{len(coded)} 個區塊（{elapsed()}）", flush=True)

    # 菜單與分組（第 3 節）
    conf3, _ = menuSets()
    gro3 = [menuGrouping(codes, rates) for _, codes in conf3]
    groups3, top3, bot3 = assignGroups(gro3)
    menus = {3: [{"mid": mid, "codes": codes, "group": g, "lone": gr["lone"], "degree": gr["degree"]}
                 for (mid, codes), gr, g in zip(conf3, gro3, groups3)]}
    frames, thresholds = {}, {3: (top3, bot3)}
    for K in (5, 7):
        frame, conf_rows, _, top_min, bot_max = menuTableFrame(K, rates)
        frames[K] = frame
        thresholds[K] = (top_min, bot_max)
        menus[K] = [{"mid": r["mid"], "codes": r["codes"], "group": r["group"], "lone": r["lone"], "degree": r["degree"]}
                    for r in sorted(conf_rows, key=lambda r: r["mid"])]

    ctx_mp = mp.get_context("fork")
    pool = ctx_mp.Pool(args.workers)
    print(f"平行行程：{args.workers}", flush=True)
    checks = {}

    # ------------------------------------------------------------------
    # 開跑前的檢查（第 8 節，依序做）
    # ------------------------------------------------------------------
    def stop(n: int, msg: str):
        pool.terminate()
        raise SystemExit(f"❌ 第 8 節 {n} 沒有通過：{msg}（沒有寫任何輸出）")

    # §8.1 重現 RQ3-GS 判定二、RQ3-GSK 判定一二（等權投票的路徑）
    gsb = pd.read_csv(RQ3GS_BLOCKS).set_index(["model", "dataset"])
    gskb = pd.read_csv(RQ3GSK_BLOCKS).set_index(["K", "model", "dataset"])
    c1 = {}
    for K in KS:
        clear = [m for m in menus[K] if m["group"] == CLEAR]
        tasks = [("v", K, h, d, ch) for h, d in weak_keys for ch in chunks(clear)]
        res = runParallel(pool, tasks, f"§8.1 K = {K}", elapsed)
        e, diff = {}, 0.0
        for (h, d) in weak_keys:
            g = GroupSums()
            for t, r in zip(tasks, res):
                if (t[2], t[3]) == (h, d):
                    mergeGS(g, r)
            e[(h, d)] = g.diff("others", "lone")
            ref = gsb.loc[(h, d), "E2"] if K == 3 else gskb.loc[(K, h, d), "E"]
            diff = max(diff, abs(e[(h, d)] - ref))
        s = S(list(e.values()))
        c1[K] = {"shown": shown(s), "diff": diff, "ok": diff <= TOL and shown(s) == REPRO_V[K]}
        print(f"§8.1 K = {K}：{shown(s)}，逐區塊最大差 {diff:.1e}", flush=True)
    checks[1] = {"ok": all(v["ok"] for v in c1.values()), "detail": c1}
    if not checks[1]["ok"]:
        stop(1, str(c1))

    # §8.2 重現 RQ3-G 的 12 條 WV（逐替換）與 RQ3-GK K = 3 的 WV − SB
    rg = pd.read_csv(RQ3G_SUBS)
    rg = rg[rg.menu == "M12"].set_index(["host", "donor", "dataset", "path"])
    diff2, n2cmp = {"WV_before": 0.0, "WV_after": 0.0, "WV_sim": 0.0, "A_sim_V": 0.0}, 0
    for h, d in weak_keys:
        for donor in DONORS:
            ctx = HostDonorK(coded[(h, d)], coded[(donor, d)], splits[d])
            s = menuSplits(ctx, list(M12), sim_rules=("C", "A"))
            n2 = s["n2"].astype(float) * SCALE
            for j, c in enumerate(M12):
                v = ctx.valid[:, j]
                mine = {"WV_before": s["base"]["W_A"] / n2, "WV_after": (s["base"]["W_A"] + s["num"]["W_A"][:, j]) / n2,
                        "WV_sim": (s["base"]["W_A"] + s["sim"]["W_A"][:, j]) / n2,
                        "A_sim_V": (s["base"]["V_A"] + s["sim"]["V_A"][:, j]) / n2}
                ref = rg.loc[(h, donor, d, c)]
                for k, x in mine.items():
                    diff2[k] = max(diff2[k], abs(float(x[v].mean()) - ref[k]))
                n2cmp += 1
    gk = pd.read_csv(RQ3GK_BLOCKS, dtype={"K": str})
    gk = gk[gk.K == "3"].set_index(["model", "dataset"])
    menus_gk3 = drawMenus()[0][3]
    wvsb, diff2b = {"A": {}, "C": {}}, 0.0
    for h, d in weak_keys:
        per_menu = {"A": [], "C": []}
        ctxs = {g: HostDonorK(coded[(h, d)], coded[(g, d)], splits[d]) for g in DONORS}
        for _, codes in menus_gk3:
            subs = {"A": [], "C": []}
            rows = [M12.index(c) for c in codes]
            for donor in DONORS:
                ctx = ctxs[donor]
                s = menuSplits(ctx, codes, sim_rules=())
                n2 = s["n2"].astype(float)
                for j, p in enumerate(rows):
                    if not ctx.participates[p]:
                        continue
                    v = ctx.valid[:, p]
                    for r in ("A", "C"):
                        x = 100 * ((s["base"][f"W_{r}"] + s["num"][f"W_{r}"][:, j]) / (n2 * SCALE) - s["SB_after"][:, j] / n2)
                        subs[r].append(float(x[v].mean()))
            for r in ("A", "C"):
                if subs[r]:
                    per_menu[r].append(float(np.mean(subs[r])))
        for r in ("A", "C"):
            wvsb[r][(h, d)] = float(np.mean(per_menu[r]))
            diff2b = max(diff2b, abs(wvsb[r][(h, d)] - gk.loc[(h, d), f"WVmSB_after_{r}_pp"]))
    s2 = S(list(wvsb["A"].values()))
    shown2 = shown(s2)[:3]
    checks[2] = {"ok": max(diff2.values()) <= TOL and diff2b <= TOL and shown2 == REPRO_GK3, "diff": diff2, "n": n2cmp,
                 "diff_gk": diff2b, "shown_gk": shown2}
    print(f"§8.2 RQ3-G：{n2cmp} 個替換，最大差 {max(diff2.values()):.1e}；RQ3-GK K = 3 的 WV − SB {shown2}，逐區塊最大差 {diff2b:.1e}"
          f"（{elapsed()}）", flush=True)
    if not checks[2]["ok"]:
        stop(2, str(checks[2]))

    # §8.3 菜單表與分組
    problems3 = []
    rs = pd.read_csv(RQ3GS_MENU_BLOCKS)
    rs = rs[rs.menu_set == "conf"]
    if rs.groupby("menu")[["group", "lone_table", "degree_pp"]].nunique().max().max() != 1:
        problems3.append("rq3gs_menu_blocks.csv：同一份菜單在各區塊的列不同")
    rsm = rs.drop_duplicates("menu").set_index("menu")
    bad3 = [m["mid"] for m in menus[3] if rsm.loc[m["mid"], "group"] != m["group"] or rsm.loc[m["mid"], "lone_table"] != m["lone"]]
    deg3 = max(abs(100 * m["degree"] - rsm.loc[m["mid"], "degree_pp"]) for m in menus[3])
    if bad3 or deg3 > 1e-12 or len(menus[3]) != 187:
        problems3.append(f"K = 3：{len(bad3)} 份不同，落單程度最大差 {deg3:.1e}")
    sha3 = {}
    for K in (5, 7):
        text = menuTableCsv(frames[K]).encode()
        recorded = re.search(rf"`rq3gsk_menus_K{K}\.csv`，sha256 `([0-9a-f]{{64}})`", criteria_text).group(1)
        disk = Path(f"result/analysis/rq3gsk/rq3gsk_menus_K{K}.csv").read_bytes()
        sha3[K] = hashlib.sha256(text).hexdigest()
        if sha3[K] != recorded or disk != text or len(menus[K]) != 762:
            problems3.append(f"K = {K}：重算的菜單表和 CSV 不同（{sha3[K]} vs {recorded}）")
    checks[3] = {"ok": not problems3, "problems": problems3, "deg3": deg3, "sha": sha3}
    print(f"§8.3 菜單：{'通過' if not problems3 else problems3}", flush=True)
    if not checks[3]["ok"]:
        stop(3, str(problems3))

    # §8.4 權重相同時 WV = V
    tasks = [("eq", K, h, d, ch) for K in KS for h, d in weak_keys for ch in chunks(menus[K])]
    res = runParallel(pool, tasks, "§8.4", elapsed)
    bad4, total4 = sum(r["bad"] for r in res), sum(r["total"] for r in res)
    checks[4] = {"ok": bad4 == 0, "bad": bad4, "total": total4}
    print(f"§8.4 權重相同：比較 {total4} 題次，不相等 {bad4}", flush=True)
    if not checks[4]["ok"]:
        stop(4, f"{bad4} 題次不相等")

    # §8.5 換成自己
    tasks = [("self", K, h, d, ch) for K in KS for h, d in weak_keys for ch in chunks(menus[K])]
    res = runParallel(pool, tasks, "§8.5", elapsed)
    bad5, count5 = sum(r["bad"] for r in res), sum(r["count"] for r in res)
    checks[5] = {"ok": bad5 == 0, "bad": bad5, "count": count5}
    print(f"§8.5 換成自己：{count5} 個替換，不符的（菜單）{bad5}", flush=True)
    if not checks[5]["ok"]:
        stop(5, f"{bad5} 份菜單不符")

    # §8.6 手算
    hand_rows, diff6 = [], 0.0
    for mid in HAND[2]:
        K = 3 if mid.startswith("GS-") else 5
        codes = next(m["codes"] for m in menus[K] if m["mid"] == mid)
        hand, mine = handCheck(args, codes), programHand(codes)
        for donor in DONORS:
            if hand["excluded"][donor] != mine["excluded"][donor]:
                diff6 = max(diff6, float("inf"))
            for c in codes:
                for k in ("W", "V", "den"):
                    dlt = abs(hand["per_path"][donor][c][k] - mine["per_path"][donor][c][k])
                    diff6 = max(diff6, dlt)
                    hand_rows.append({"菜單": mid, "供體": donor, "path": c, "量": k, "手算": hand["per_path"][donor][c][k],
                                      "程式": mine["per_path"][donor][c][k], "差": dlt})
        for k in hand["effects"]:
            diff6 = max(diff6, abs(hand["effects"][k] - mine["effects"][k]))
        hand_rows.append({"菜單": mid, "供體": "合計", "path": "被排除的切分",
                          "量": "；".join(f"{g}: {hand['excluded'][g]}" for g in DONORS), "手算": np.nan, "程式": np.nan,
                          "差": float(any(hand["excluded"][g] != mine["excluded"][g] for g in DONORS))})
        for k in hand["effects"]:
            hand_rows.append({"菜單": mid, "供體": "合計", "path": "效果", "量": k, "手算": hand["effects"][k],
                              "程式": mine["effects"][k], "差": abs(hand["effects"][k] - mine["effects"][k])})
    checks[6] = {"ok": diff6 <= TOL, "diff": diff6}
    print(f"§8.6 手算：最大差 {diff6:.1e}（{elapsed()}）", flush=True)
    if not checks[6]["ok"]:
        stop(6, f"最大差 {diff6}")

    # ------------------------------------------------------------------
    # 正式計算
    # ------------------------------------------------------------------
    tasks = [("main", K, h, d, ch) for K in KS for h, d in weak_keys for ch in chunks(menus[K])]
    res = runParallel(pool, tasks, "正式計算", elapsed)
    pool.close()
    pool.join()
    agg = defaultdict(lambda: {"gs": GroupSums(), "cnt": Counter(), "item6": []})
    sub_rows, menu_rows = [], []
    for t, r in zip(tasks, res):
        a = agg[(t[1], t[2], t[3])]
        mergeGS(a["gs"], r)
        a["cnt"].update(r["cnt"])
        a["item6"].extend(r["item6"])
        sub_rows.extend(r["subs"])
        menu_rows.extend(r["menus"])
    block_rows = []
    for K in KS:
        for h, d in weak_keys:
            a = agg[(K, h, d)]
            for nm in (f"E_{CLEAR}_others|W_C", f"E_{CLEAR}_lone|W_C"):
                if a["gs"].den.get(nm, 0) <= 0:
                    raise SystemExit(f"❌ 第 4 節：K = {K} | {h} | {d} 在判定中沒有可用的（菜單、供體、切分），停下來回報")
            block_rows.append(blockValues(K, h, d, a["gs"], a["cnt"], a["item6"]))
    B = pd.DataFrame(block_rows)
    print(f"正式計算完成（{elapsed()}）", flush=True)
    data = {"B": B, "menu_rows": menu_rows, "sub_rows": sub_rows, "checks": checks, "hand_rows": hand_rows, "menus": menus,
            "thresholds": thresholds}
    if args.cache:
        with open(args.cache, "wb") as fh:
            pickle.dump({"criteria_sha": criteria_sha, "gen_time": gen_time, "data": data}, fh)
    writeOutputs(args, criteria_sha, confirmed, gen_time, **data)
    print(f"完成（{elapsed()}）", flush=True)


def writeOutputs(args, criteria_sha, confirmed, gen_time, B, menu_rows, sub_rows, checks, hand_rows, menus, thresholds):
    os.makedirs(args.out_dir, exist_ok=True)
    for df, name in ((B, "rq3gsx_blocks.csv"), (pd.DataFrame(menu_rows), "rq3gsx_menu_blocks.csv"),
                     (pd.DataFrame(sub_rows), "rq3gsx_substitutions.csv")):
        df = df.copy()
        df["criteria_sha256"] = criteria_sha
        df.to_csv(os.path.join(args.out_dir, name), index=False)

    st = {}
    blk = lambda K: B[B.K == K]
    for K in KS:
        for agg_ in ("W", "V"):
            st[(K, agg_)] = S(blk(K)[f"E_{agg_}"])
            if K in (5, 7):
                for r in range(1, K + 1):
                    st[(K, agg_, r)] = S(blk(K)[f"rank{r}_{agg_}"])
    plotJudgments(st, os.path.join(args.out_dir, "fig_a_judgments"), criteria_sha)
    plotRanks(st, os.path.join(args.out_dir, "fig_b_rank_effect"), criteria_sha)

    report = buildReport(B, st, checks, hand_rows, menus, thresholds, criteria_sha, confirmed, gen_time, args)
    Path(args.out_dir, "report.md").write_text(report, encoding="utf-8")


# ------------------------------------------------------------------
# 報告
# ------------------------------------------------------------------
def perBlock(B: pd.DataFrame, K: int, cols: dict, d: int = 2) -> str:
    sub = B[B.K == K]
    rows = []
    for r in sub.itertuples():
        row = {"區塊": label(r.model, r.dataset)}
        for col, name in cols.items():
            v = getattr(r, col)
            row[name] = "—" if pd.isna(v) else f"{v:+.{d}f}"
        rows.append(row)
    return md(pd.DataFrame(rows))


def ciTable(rows: list) -> str:
    return md(pd.DataFrame(rows))


def buildReport(B, st, checks, hand_rows, menus, thresholds, sha, confirmed, gen_time, args) -> str:
    blk = lambda K: B[B.K == K]
    L = ["# RQ3-GSX：依正確率加權的投票下，和其他條最不像的那條還是最不值得改嗎？", "",
         f"判定標準 `result/analysis/rq3gsx/rq3gsx_criteria.md`，sha256 `{sha}`；確認：{confirmed}。",
         f"產生時間 {gen_time}；程式 `scripts/analysis_rq3gsx/path_improve_gsx.py`（逐切分計算在 `Analysis/pathImproveGSX.py`）；"
         f"平行行程 {args.workers} 個（結果依工作順序合併，與行程數無關）。不呼叫 API，不重跑任何東西，不修改現有檔案。", "",
         "## (1) 讀了哪些檔案", "",
         "- `result/arms/{模型}/{資料集}/`：4 個模型 × 4 個資料集，經 `Analysis.menuVote.loadPathBlock` 載入（同 RQ3-G）。欄位 `item_id`、`gold`、"
         "`parsed_answer`、`parse_ok`。載入時的核對（題目 id 與 gold、`compareTwoAnswer` = 字串相等、整數投票 = `vote`）全部通過。",
         f"- `{RQ3GS_BLOCKS}`（`E2`）、`{RQ3GSK_BLOCKS}`（`E`）：§8.1 的核對。",
         f"- `{RQ3G_SUBS}`（M12 的 `WV_before`、`WV_after`、`WV_sim`、`A_sim_V`）、`{RQ3GK_BLOCKS}`（K = 3 的 `WVmSB_after_A_pp`、"
         "`WVmSB_after_C_pp`）：§8.2 的核對。",
         f"- `{RQ3GS_MENU_BLOCKS}`（`menu_set`、`group`、`lone_table`、`degree_pp`）、`result/analysis/rq3gsk/rq3gsk_menus_K5.csv`、"
         "`rq3gsk_menus_K7.csv`：§8.3 的核對。",
         "- 沿用的程式：`Analysis.pathImprove`（編碼、`logOdds`、`fraction`）、`Analysis.pathImproveK`（`Ans.scores`、`HostDonorK`、`drawMenus`）、"
         "`Analysis.pathImproveGS`（菜單、一致率、分組、`GroupSums`）、`Analysis.pathImproveGSK`（`menuTableFrame`、`hostSide` 的落單名次）、"
         "`Analysis.blockStats`。", "",
         "## (2) 第零階段", "",
         "確認前做的第零階段寫在判定標準檔的附錄 A（菜單表與兩個 CSV 的 sha256、排除比例、計時）。這次開跑時：",
         f"- 菜單表與分組重算後和既有的相同（§8.3）。組別門檻：" + "；".join(
             f"K = {K} 明顯落單 ≥ {100 * thresholds[K][0]:.10f}pp、對稱 ≤ {100 * thresholds[K][1]:.10f}pp" for K in KS) + "。",
         "- 各區塊被排除的（菜單、供體、切分）比例（全部確認用菜單，兩個供體合計；應和附錄 A.2 的「全部」欄相同）：", ""]
    ex = B.pivot_table(index=["model", "dataset"], columns="K", values="excluded_share", sort=False).reset_index()
    ex.insert(0, "區塊", [label(h, d) for h, d in zip(ex.model, ex.dataset)])
    ex = ex.drop(columns=["model", "dataset"])
    ex.columns = ["區塊"] + [f"K = {K}" for K in KS]
    for K in KS:
        ex[f"K = {K}"] = ex[f"K = {K}"].map(lambda x: f"{100 * x:.2f}%")
    L += [md(ex), ""]

    # (3) 開跑前的檢查
    c = checks
    L += ["## (3) 開跑前的檢查（第 8 節）", "",
          "1. 等權投票的路徑重現 RQ3-GS 判定二與 RQ3-GSK 判定一、二 → 通過。", ""]
    L.append(md(pd.DataFrame([{"K": K, "重算": f"{v['shown'][0]}（{v['shown'][1]} 到 {v['shown'][2]}），{v['shown'][3]}/8",
                               "既有": f"{REPRO_V[K][0]}（{REPRO_V[K][1]} 到 {REPRO_V[K][2]}），{REPRO_V[K][3]}/8",
                               "逐區塊最大差": f"{v['diff']:.1e}"} for K, v in c[1]["detail"].items()])))
    L += ["",
          f"2. RQ3-G 的 12 條 WV：{c[2]['n']} 個替換（M12，含 2 個不參與判定的），每個替換在自己的有效切分上平均（規則 A），和 "
          "`rq3g_substitutions.csv` 的最大差：" + "、".join(f"{k} {v:.1e}" for k, v in c[2]["diff"].items())
          + f"。RQ3-GK K = 3 的替換後 WV − SB（規則 A）：{c[2]['shown_gk'][0]}（{c[2]['shown_gk'][1]} 到 {c[2]['shown_gk'][2]}），"
          f"逐區塊和 `rq3gk_k_blocks.csv`（規則 A、C）最大差 {c[2]['diff_gk']:.1e} → 通過。",
          f"3. 菜單表：K = 3 的 187 份和 `rq3gs_menu_blocks.csv` 的組別、落單 path 相同，落單程度最大差 {c[3]['deg3']:.1e}pp；"
          f"K = 5、7 重算的表和兩個 CSV 的 bytes 相同，sha256 {c[3]['sha'][5][:8]}…、{c[3]['sha'][7][:8]}…（= 第 3 節）→ 通過。",
          f"4. 權重都設成 1 或 0.5 時，WV 的逐題得分（規則 C、A）和等權投票相同：比較 {c[4]['total']:,} 題次，不相等 {c[4]['bad']} → 通過。",
          f"5. 換成自己：{c[5]['count']:,} 個替換 × 200 次切分，WV 與 V 的分子（C、A）、模擬的分子、分子_固定、分母都是 0，"
          "且沒有有效切分 → 通過。",
          f"6. 手算（gpt4omini × mmlu，GS-001 與 GS5-001，兩個供體；純迴圈、`Fraction` 的規則 C 得分）：最大差 {c[6]['diff']:.1e} → 通過。", ""]
    hr = pd.DataFrame(hand_rows)
    hr_eff = hr[hr.path.isin(["效果", "被排除的切分"])].copy()
    hr_eff["手算"] = hr_eff["手算"].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    hr_eff["程式"] = hr_eff["程式"].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    hr_eff["差"] = hr_eff["差"].map(lambda x: f"{x:.1e}")
    L += ["手算的兩群效果與被排除的切分數（逐 path 的分子、分母總和也都比過，最大差同上）：", "", md(hr_eff), ""]

    # (4) 三項判定
    L += ["## (4) 三項判定（8 個弱模型區塊，各 K 的明顯落單組，規則 C）", "",
          "E_W = 其他 K − 1 條的效果 − 落單那條的效果（WV；單位：那條 path 每進步 10pp，加權投票多幾個 pp）。同一批資料上 V 的 E 並排（不參與判定）。", ""]
    rows, states = [], {}
    for n, K in enumerate(KS, 1):
        s = st[(K, "W")]
        states[K] = fourState(s, THRESHOLDS[K])
        b = blk(K)
        rows.append({"判定": f"判定{'一二三'[n - 1]}（K = {K}）", "菜單": 62 if K == 3 else 254,
                     "其他條（WV）": f"{b.others_W.mean():.2f}", "落單那條（WV）": f"{b.lone_W.mean():.2f}",
                     "E_W": pp(s["mean"]), "SE": f"{s['se']:.2f}", "95% 區間": f"[{pp(s['ci_low'])}, {pp(s['ci_high'])}]",
                     "為正的區塊": f"{s['n_positive']}/8", "門檻": f"{THRESHOLDS[K]:.2f}", "狀態": states[K],
                     "（對照）V 的 E": ciText(st[(K, "V")])})
    L += [ciTable(rows), "", "![判定](fig_a_judgments.png)", ""]
    for K in KS:
        L += [f"K = {K} 逐區塊：", "",
              perBlock(B, K, {"others_W": "其他條（WV）", "lone_W": "落單那條（WV）", "E_W": "E_W", "others_V": "其他條（V）",
                              "lone_V": "落單那條（V）", "E_V": "E（V）"}), ""]

    # (5) 只報告的量
    L += ["## (5) 只報告的量（不參與判定）", "", "### 7.1 各組的 E（WV 與 V 並排）", ""]
    rows = []
    for K in KS:
        for grp, col in ((CLEAR, ""), ("中間", "_中間"), ("對稱", "_對稱"), ("全部", "_all")):
            rows.append({"K": K, "組": grp, "E_W": ciText(S(blk(K)[f"E_W{col}"])), "E（V）": ciText(S(blk(K)[f"E_V{col}"]))})
    L += [ciTable(rows), "", "### 7.2 落單那條的效果 ÷ 其他條的效果（明顯落單組，逐區塊）", ""]
    rows = []
    for K in KS:
        for agg_ in ("W", "V"):
            v = blk(K)[f"ratio_{agg_}"]
            rows.append({"K": K, "聚合": "WV" if agg_ == "W" else "V", "8 個值": "、".join(f"{x:.2f}" for x in v),
                         "中位數": f"{v.median():.3f}", "範圍": f"{v.min():.2f} 到 {v.max():.2f}"})
    L += [ciTable(rows), "", "### 7.3 依落單名次的效果（全部確認用菜單；1 = 最落單）", ""]
    rows = []
    for K in (5, 7):
        for agg_ in ("W", "V"):
            r = {"K": K, "聚合": "WV" if agg_ == "W" else "V"}
            for k in range(1, 8):
                r[f"名次 {k}"] = f"{blk(K)[f'rank{k}_{agg_}'].mean():.2f}" if k <= K else "—"
            r["最不落單 − 最落單"] = ciText(S(blk(K)[f"rankdiff_{agg_}"]))
            rows.append(r)
    L += [ciTable(rows), "", "（各名次是 8 個區塊的平均；區間在圖 (b)。）", "", "![名次](fig_b_rank_effect.png)", "",
          "### 7.4 分開原因（WV）", ""]
    rows = []
    for K in KS:
        b = blk(K)
        nonT = [m for m in menus[K] if m["lone"] not in TRANSLATED]
        rng_ = (f"{100 * min(m['degree'] for m in nonT):.3f} 到 {100 * max(m['degree'] for m in nonT):.3f}pp") if nonT else "—"
        rows.append({"K": K, "(a) 落單的不是翻譯 path：E_W": ciText(S(b.E_W_4a)) + f"（{len(nonT)} 份，落單程度 {rng_}）",
                     "(b) 落單的不是正確率最低：E_W": ciText(S(b.E_W_4b)),
                     "(c) 其他條 − 正確率最低那條": ciText(S(b.E_W_4c)),
                     "(b)(c) 用到的比例": f"{b.share_4bc.mean():.3f}（{b.share_4bc.min():.3f} 到 {b.share_4bc.max():.3f}）"})
    L += [ciTable(rows), "", "### 7.5 模擬（隨機改進模型）", "",
          "轉換率 = 該區塊所有可用的（菜單、供體、切分）× K 條 path 的分子總和 ÷ 分母總和（全部確認用菜單）。", ""]
    rows, read5 = [], {}
    for K in KS:
        b = blk(K)
        for agg_ in ("W", "V"):
            ii = b[f"underest_{agg_}"]
            undefined = [label(h, d) for h, d, x in zip(b.model, b.dataset, ii) if pd.isna(x)]
            pooled = 1 - b[f"sum_num_sim_{agg_}"].sum() / b[f"sum_num_real_{agg_}"].sum()
            read5[(K, agg_)] = (float(np.nanmedian(ii)), pooled)
            rows.append({"K": K, "聚合": "WV" if agg_ == "W" else "V",
                         "轉換率，真實": f"{b[f'conv_real_{agg_}'].mean():.4f}", "轉換率，隨機": f"{b[f'conv_sim_{agg_}'].mean():.4f}",
                         "(i) 真實 − 隨機": ciText(S(b[f"conv_diff_{agg_}"]), 4),
                         "(ii) 1 − 隨機 ÷ 真實：8 個值": "、".join("無定義" if pd.isna(x) else f"{x:.3f}" for x in ii),
                         "(ii) 中位數（範圍）": f"{np.nanmedian(ii):.3f}（{np.nanmin(ii):.3f} 到 {np.nanmax(ii):.3f}）",
                         "(iii) 合在一起": f"{pooled:.3f}",
                         "無定義的區塊": "、".join(undefined) or "無"})
    L += [ciTable(rows), "", "（轉換率兩欄是 8 個區塊的平均，只描述。）", "",
          "明顯落單組：模擬算出的「落單那條的效果 ÷ 其他條的效果」和真實的並排（逐區塊，報中位數與範圍）：", ""]
    rows, ratio5 = [], {}
    for K in KS:
        b = blk(K)
        for agg_ in ("W", "V"):
            sim, real = b[f"simratio_{agg_}"], b[f"ratio_{agg_}"]
            ratio5[(K, agg_)] = (sim.median(), real.median())
            rows.append({"K": K, "聚合": "WV" if agg_ == "W" else "V", "模擬：8 個值": "、".join(f"{x:.2f}" for x in sim),
                         "模擬：中位數（範圍）": f"{sim.median():.3f}（{sim.min():.2f} 到 {sim.max():.2f}）",
                         "真實：中位數（範圍）": f"{real.median():.3f}（{real.min():.2f} 到 {real.max():.2f}）"})
    L += [ciTable(rows), "", "### 7.6 替換後 WV − SB 與 V − SB（全部確認用菜單，規則 C；pp）", ""]
    rows = []
    for K in KS:
        b = blk(K)
        rows.append({"K": K, "WV − SB": ciText(S(b.WVmSB_after_pp)), "V − SB": ciText(S(b.VmSB_after_pp)),
                     "沒有可用切分而不納入的菜單（8 個區塊合計）": int(b.n_menus6_dropped.sum())})
    L += [ciTable(rows), "", "區塊值：每個替換在可用切分上平均 → 菜單內的替換平均 → 菜單平均（RQ3-GK 第 5 節的三段平均，排除用第 4 節的規則）。", "",
          "### 7.7 不除以分母的版本（WV，明顯落單組；pp）", ""]
    rows = []
    for K in KS:
        b = blk(K)
        rows.append({"K": K, "其他條：投票多": ciText(S(b.others_vote_pp), 2, False), "其他條：path 進步": ciText(S(b.others_path_pp), 2, False),
                     "落單那條：投票多": ciText(S(b.lone_vote_pp), 2, False), "落單那條：path 進步": ciText(S(b.lone_path_pp), 2, False)})
    L += [ciTable(rows), "", "### 7.8 加權分數平手的題目比例、權重為 0 的 path 比例（逐區塊）", ""]
    rows = []
    for r in B.itertuples():
        rows.append({"K": r.K, "區塊": label(r.model, r.dataset), "平手：替換前": f"{100 * r.tie_before:.3f}%",
                     "平手：替換後": f"{100 * r.tie_after:.3f}%", "權重 0：替換前": f"{100 * r.zero_before:.3f}%",
                     "權重 0：替換後": f"{100 * r.zero_after:.3f}%"})
    L += [ciTable(rows), "", "### 7.9 規則 A 的值；K = 5、7 門檻改用 0.5 的狀態", ""]
    rows = []
    for K in KS:
        b = blk(K)
        r = {"K": K, "E_W（規則 A）": ciText(S(b.E_W_A)), "（對照）E（V，規則 A）": ciText(S(b.E_V_A))}
        r["門檻 0.5 時的狀態（規則 C）"] = fourState(st[(K, "W")], REFERENCE_THRESHOLD) if K in (5, 7) else "—"
        rows.append(r)
    L += [ciTable(rows), "", "### 7.10 圖", "", "圖 (a)、(b) 在上面（PNG 與 PDF 都在 `result/analysis/rq3gsx/`）。", "",
          "### 7.11 權重固定的版本：分開「答案」與「權重」", "",
          "分子_固定 = p 換成供體的答案、權重沿用替換前；答案的部分 = 10 × Σ分子_固定 ÷ Σ分母；權重的部分 = 10 × Σ(完整的分子 − 分子_固定) ÷ Σ分母"
          "（兩部分相加 = 完整的效果）。", ""]
    rows, pooled11 = [], {}
    for K in KS:
        b = blk(K)
        for tag, name in (("clear", "明顯落單"), ("all", "全部")):
            rows.append({"K": K, "範圍": name, "E_W_固定": ciText(S(b[f"E_Wfix_{tag}"])),
                         "其他條：答案": ciText(S(b[f"others_answer_{tag}"]), 2, False), "其他條：權重": ciText(S(b[f"others_weight_{tag}"]), 2, False),
                         "落單：答案": ciText(S(b[f"lone_answer_{tag}"]), 2, False), "落單：權重": ciText(S(b[f"lone_weight_{tag}"]), 2, False)})
        pooled11[K] = (b.lone_weight_num_clear.sum(), b.lone_answer_num_clear.sum())
    L += [ciTable(rows), "",
          "落單那條（明顯落單組）8 個區塊的分子合計（第 5 節反向成立的讀法用）：" + "；".join(
              f"K = {K} 權重的部分 {w:.4f}、答案的部分 {a:.4f}" for K, (w, a) in pooled11.items()) + "。", ""]

    # (6) 對照讀法的結論
    L += ["## (6) 對照讀法的結論", ""]
    for n, K in enumerate(KS, 1):
        s, state = st[(K, "W")], states[K]
        if state == "反向成立":
            w, a = pooled11[K]
            text = ("加權之後落單的那條反而比較值得改。" +
                    (f"第 7 節 11：落單那條的效果裡，8 個區塊合計時權重的部分（{w:.4f}）大於答案的部分（{a:.4f}），"
                     "所以可以寫成換進來的強 path 權重變高、壓過相像的那幾條。" if w > a else
                     f"第 7 節 11：權重的部分（{w:.4f}）沒有大於答案的部分（{a:.4f}），只列數字。"))
        else:
            text = READINGS[state]
        L.append(f"- 判定{'一二三'[n - 1]}（K = {K}，E_W {pp(s['mean'])}，[{pp(s['ci_low'])}, {pp(s['ci_high'])}]，"
                 f"{s['n_positive']}/8，門檻 {THRESHOLDS[K]:.2f}）：**{state}**。{text}")
    if all(states[K] == "正向成立" for K in KS):
        L.append("- 三項合起來：三項都正向成立 → 可以寫「K = 3、5、7 的加權投票下都成立」。")
    else:
        L.append("- 三項合起來：不是三項都正向成立 → 逐 K 照各自的狀態寫（上面三行）。")
    L += ["- 事先寫下的限制（論文要照寫）：", *[f"  - {x}" for x in LIMITS], "",
          "第 7 節 5 事先寫好的讀法（只描述）：", ""]
    for K in KS:
        (mw, pw), (mv, pv) = read5[(K, "W")], read5[(K, "V")]
        gap = max(abs(mw - pw), abs(mv - pv))
        if gap > 0.10:
            L.append(f"- K = {K}：(ii) 的中位數與 (iii) 相差 {gap:.3f} > 0.10（WV {mw:.3f} 對 {pw:.3f}；V {mv:.3f} 對 {pv:.3f}）→ "
                     "不寫這一句，只列 (i)、(ii)、(iii)（第 7 節 5 的表）。")
        else:
            L.append(f"- K = {K}：「加權投票下，隨機改進的模擬低估真實替換約 {10 * mw:.1f} 成（等權投票是 {10 * mv:.1f} 成）；"
                     f"8 個區塊合在一起是 {pw:.3f}（等權投票 {pv:.3f}）。」")
        (sw, rw), (sv, rv) = ratio5[(K, "W")], ratio5[(K, "V")]
        L.append(f"  - 「模擬算出的比例是 {sw:.2f}，真實的是 {rw:.2f}」（WV，明顯落單組，8 個區塊的中位數）；V 是 {sv:.2f} 對 {rv:.2f}。")
    L += ["", "- 不寫「模擬可用」或「模擬準」。其餘都只報告，不套四種狀態。", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
