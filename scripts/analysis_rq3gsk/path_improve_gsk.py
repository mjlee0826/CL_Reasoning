"""
path_improve_gsk.py — RQ3-GSK：K = 5、7 時，和其他條最不像的那條還是最不值得改嗎？（result/analysis/rq3gsk/rq3gsk_criteria.md）

全程離線：不呼叫 API，不重跑任何東西，不修改現有檔案。判定標準確認前不執行。
    1. 載入 16 個區塊並編碼（沿用 Analysis/pathImprove.py 的核對）；兩兩一致率（8 個弱模型區塊，子集一）
    2. 開跑前的檢查（第 9 節，任何一項不過就停，不寫輸出）：
       K = 3 重現 RQ3-GS 判定二；K = 3 的菜單表 = rq3gs_menu_blocks.csv；重現 RQ3-GK K = 5、7 的 D1 與轉換率；
       菜單表 = 確認前寫好的 CSV（sha256）；換成自己；手算 gpt4omini × mmlu × GS5-001
    3. K = 5、7：宿主這一側（每次切分在選擇半的名次）與替換這一側（分子、分母），依第 5 節加總成各群的效果
    4. 輸出（--out-dir）：rq3gsk_blocks.csv、rq3gsk_menu_blocks.csv、rq3gsk_substitutions.csv、兩張圖、report.md
       （rq3gsk_menus_K5.csv、rq3gsk_menus_K7.csv 是確認前寫好的，這裡只核對、不覆寫）

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq3gsk/path_improve_gsk.py
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import hashlib
import itertools
import os
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from Analysis.preregistration import sha256, confirmationLine
from Analysis.blockStats import summarize, fourState
from Analysis.menuJudgeStats import blocksWithSplits
from Analysis.menuVote import MODELS, DATASETS, loadPathBlock
from Analysis.splitHalf import makeSplits
from Analysis.pathImprove import M12, HOSTS, DONORS, encodeBlocks, checkVote
from Analysis.pathImproveK import HostDonorK
from Analysis.pathImproveGS import GROUPS, menuSets, agreementRates, menuGrouping, assignGroups, substitutionSplits, onehot, GroupSums
from Analysis.pathImproveGSK import (KS, THRESHOLDS, REFERENCE_THRESHOLD, TRANSLATED, menuRow, menuTableFrame, menuTableCsv,
                                     blockLone, hostSide, conversionParts)

OUT_DIR = "result/analysis/rq3gsk"
CRITERIA_FILE = "rq3gsk_criteria.md"
RQ3GS_BLOCKS = "result/analysis/rq3gs/rq3gs_blocks.csv"
RQ3GS_MENU_BLOCKS = "result/analysis/rq3gs/rq3gs_menu_blocks.csv"
RQ3GK_BLOCKS = "result/analysis/rq3gk/rq3gk_k_blocks.csv"
RQ3GS_SUMMARY = ("+1.75", "+1.03", "+2.46", 8)
GK_COLS_16 = ("D1_A", "D1_C")
GK_COLS_8 = ("conversion_real_A", "conversion_real_C", "conversion_sim_A", "conversion_sim_C")
HAND = ("gpt4omini", "mmlu", "GS5-001")
TOL = 1e-9
MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
GROUP_EN = {"明顯落單": "clear odd-one-out", "中間": "middle", "對稱": "symmetric"}
READINGS = {
    "正向成立": "K 條 path 投票時，和其他條最不像的那條仍然最不值得改。（依第 6 節的限制 (a)，論文只能寫「K 條 path 投票時，"
            "和其他條最不像的那條翻譯的 path 最不值得改」；要去掉「翻譯的」，還要看第 8 節 4。）",
    "兩者相當": "K 條時落單與否沒有實質差別；K = 3 的規則不能推廣。",
    "反向成立": "K 條時落單的那條反而比較值得改。",
    "無法判定": "只列數字；不能把 K = 3 的規則寫成適用於更大的 K。",
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6a64", "#d9d8d2", "#fcfcfb"


def parseArgs():
    parser = ArgumentParser(description="RQ3-GSK: is the odd-one-out path still the least worth improving at K = 5, 7?")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--out-dir", default=OUT_DIR)
    return parser.parse_args()


def pp(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:+.2f}"


def ciRow(name: str, s: dict, threshold: float | None = None) -> dict:
    row = {"量": name, "平均": pp(s["mean"]), "SE": f"{s['se']:.2f}", "95% 區間": f"[{pp(s['ci_low'])}, {pp(s['ci_high'])}]",
           "為正的區塊": f"{s['n_positive']}/{s['n_blocks']}"}
    if threshold is not None:
        row["狀態"] = fourState(s, threshold)
    return row


def md(df: pd.DataFrame, floatfmt: str = ".2f", text: bool = False) -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt, disable_numparse=text)


def withLabels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = out.model.map(MODEL_LABELS)
    return out.rename(columns={"model": "宿主", "dataset": "資料集"})


# ------------------------------------------------------------------
# §9.6 手算（純迴圈、字串答案、Fraction；不用 Analysis.pathImproveK / pathImproveGS / pathImproveGSK）
# ------------------------------------------------------------------
def handCheck(args, codes: list[str]) -> dict:
    host_model, dataset, _ = HAND
    host = loadPathBlock(args.armdir, host_model, dataset, args.aggdir)
    N = len(host.item_ids)
    splits = makeSplits(N, 200, 0)
    gold = host.gold
    K = len(codes)
    sub1 = [all(host.answered[c][i] for c in M12) for i in range(N)]

    def scoreC(answers: list, i: int) -> Fraction:
        votes = Counter(answers)
        top = max(votes.values())
        tied = [a for a, v in votes.items() if v == top]
        return Fraction(1, len(tied)) if gold[i] in tied else Fraction(0)

    ans = {c: host.answers[c] for c in codes}
    base = [scoreC([ans[c][i] for c in codes], i) for i in range(N)]
    cor = {c: [ans[c][i] == gold[i] for i in range(N)] for c in codes}
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
    out = {"per_path": {}, "excluded": {}, "others_num": 0.0, "others_den": 0.0, "lone_num": 0.0, "lone_den": 0.0}
    for donor_model in DONORS:
        donor = loadPathBlock(args.armdir, donor_model, dataset, args.aggdir)
        sub2 = [sub1[i] and all(donor.answered[c][i] for c in M12) for i in range(N)]
        dcor = {c: [donor.answers[c][i] == gold[i] for i in range(N)] for c in codes}
        delta = {c: [scoreC([donor.answers[c][i] if d == c else ans[d][i] for d in codes], i) - base[i] for i in range(N)]
                 for c in codes}
        excluded, sums = 0, {c: [0.0, 0.0] for c in codes}
        for r, h1 in enumerate(splits):
            H1 = [i for i in range(N) if h1[i] and sub2[i]]
            H2 = [i for i in range(N) if not h1[i] and sub2[i]]
            valid = all(sum(dcor[c][i] for i in H1) > sum(cor[c][i] for i in H1) and
                        sum(dcor[c][i] for i in H2) > sum(cor[c][i] for i in H2) for c in codes)
            if not valid:
                excluded += 1
                continue
            for j, c in enumerate(codes):
                num = float(sum((delta[c][i] for i in H2), Fraction(0)) / len(H2))
                den = (sum(dcor[c][i] for i in H2) - sum(cor[c][i] for i in H2)) / len(H2)
                sums[c][0] += num
                sums[c][1] += den
                key = "lone" if j == lone[r] else "others"
                out[f"{key}_num"] += num
                out[f"{key}_den"] += den
        out["per_path"][donor_model] = sums
        out["excluded"][donor_model] = excluded
    out["others_effect"] = 10 * out["others_num"] / out["others_den"]
    out["lone_effect"] = 10 * out["lone_num"] / out["lone_den"]
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
        fig.savefig(f"{path}.{ext}", facecolor=SURFACE, metadata={"Subject" if ext == "pdf" else "Description": f"rq3gsk_criteria.md sha256 {sha}"})
    plt.close(fig)


def plotGroups(stats: dict, counts: dict, path: str, sha: str):
    """(a) K = 3、5、7 三組菜單的 E_K（8 個區塊的平均與 95% t 區間）；K = 3 是 RQ3-GS 的值。"""
    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(GROUPS))
    for (K, color, marker, dx) in zip((3, 5, 7), SERIES, MARKERS, (-0.16, 0.0, 0.16)):
        s = [stats[(K, g)] for g in GROUPS]
        ax.vlines(x + dx, [v["ci_low"] for v in s], [v["ci_high"] for v in s], color=color, linewidth=1.8, zorder=2)
        ax.scatter(x + dx, [v["mean"] for v in s], s=52, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.4, zorder=3,
                   label=f"K = {K}" + (" (RQ3-GS)" if K == 3 else f" ({counts[K]} menus per group)"))
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_EN[g] for g in GROUPS], fontsize=8.5, color=INK)
    styleAxes(ax)
    ax.set_ylabel("E_K = others − odd one out\n(vote pp per 10pp path gain)", color=MUTED, fontsize=9)
    ax.set_title("RQ3-GSK (a): E_K by menu group, rule C (8 weak blocks, 95% t interval)", color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
    saveFig(fig, path, sha)


def plotRanks(stats: dict, path: str, sha: str):
    """(b) 依 s_i 名次的效果（全部確認用的菜單），K = 5、7 各一條線。"""
    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for K, color, marker, dx in zip(KS, SERIES[1:], MARKERS[1:], (-0.06, 0.06)):
        ranks = np.arange(1, K + 1)
        s = [stats[(K, r)] for r in ranks]
        ax.vlines(ranks + dx, [v["ci_low"] for v in s], [v["ci_high"] for v in s], color=color, linewidth=1.6, zorder=2)
        ax.plot(ranks + dx, [v["mean"] for v in s], color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label=f"K = {K}")
    ax.set_xticks(np.arange(1, 8))
    ax.set_xticklabels(["1\nmost odd"] + [str(r) for r in range(2, 7)] + ["7"], fontsize=8.5, color=INK)
    styleAxes(ax)
    ax.set_xlabel("rank by s_i on the selection half (1 = odd one out, K = least odd)", color=MUTED, fontsize=9)
    ax.set_ylabel("effect: vote pp per 10pp path gain", color=MUTED, fontsize=9)
    ax.set_title("RQ3-GSK (b): effect by odd-one-out rank, rule C\n762 confirmation menus per K (8 weak blocks, 95% t interval)",
                 color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    saveFig(fig, path, sha)


# ------------------------------------------------------------------
# 一個 K 的正式計算（第 5、6、8 節）
# ------------------------------------------------------------------
def runK(K: int, coded: dict, splits: dict, rates: dict, weak_keys: list, elapsed) -> dict:
    frame, conf_rows, old_rows, top_min, bot_max = menuTableFrame(K, rates)
    menus = [("conf", r) for r in conf_rows] + [("old", r) for r in old_rows]
    sim_rows, block_rows, menu_rows, sub_rows = [], [], [], []
    menu_sums = defaultdict(lambda: np.zeros(4))
    for key in [(m, d) for m in MODELS for d in DATASETS]:     # 宿主在外、資料集在內（報告與 CSV 的列順序）
        block = coded[key]
        weak = key in weak_keys
        h, d = key
        sim_vals = {"all": ([], []), GROUPS[0]: ([], [])}
        gs = GroupSums()
        use4 = Counter()
        excl, total = Counter(), Counter()
        contexts = {g: HostDonorK(block, coded[(g, d)], splits[d]) for g in DONORS} if weak else {}
        for kind, m in menus:
            codes, group, mid = m["codes"], m["group"], m["mid"]
            info = hostSide(block, codes, splits[d], pi_rules=("C",) if kind == "conf" else ())
            R = len(info["lone"])
            if kind == "conf":
                pi = info["pi2"]["C"]
                lone_pi = pi[np.arange(R), info["lone"]]
                others_pi = np.array([np.nanmean(np.delete(pi[r], info["lone"][r])) if np.isfinite(np.delete(pi[r], info["lone"][r])).any()
                                      else np.nan for r in range(R)])
                for scope in (("all",) + ((GROUPS[0],) if group == GROUPS[0] else ())):
                    sim_vals[scope][0].extend(lone_pi[np.isfinite(lone_pi)].tolist())
                    sim_vals[scope][1].extend(others_pi[np.isfinite(others_pi)].tolist())
            if not weak:
                continue
            lone, low = onehot(info["lone"], K), onehot(info["lowest"], K)
            ranks = info["ranks"]
            mrow = defaultdict(float)
            for donor in DONORS:
                s = substitutionSplits(contexts[donor], codes)
                inc = s["included"]
                excl[kind] += int((~inc).sum())
                total[kind] += R
                numC, numA, den = s["num"]["C"], s["num"]["A"], s["den"]
                if kind == "conf":
                    for name in (group, "all"):
                        gs.add(f"E_{name}_others", numC, den, ~lone, inc); gs.add(f"E_{name}_lone", numC, den, lone, inc)
                    for r in range(1, K + 1):
                        gs.add(f"rank{r}", numC, den, ranks == r, inc)
                        if group == GROUPS[0]:
                            gs.add(f"rankclear{r}", numC, den, ranks == r, inc)
                    if group == GROUPS[0]:
                        gs.add("EA_others", numA, den, ~lone, inc); gs.add("EA_lone", numA, den, lone, inc)
                        rows_b = inc & (info["lone"] != info["lowest"])
                        gs.add("4b_others", numC, den, ~lone, rows_b); gs.add("4b_lone", numC, den, lone, rows_b)
                        gs.add("4c_other", numC, den, ~low, rows_b); gs.add("4c_low", numC, den, low, rows_b)
                        use4["used"] += int(rows_b.sum())
                        use4["total"] += int(inc.sum())
                    if m["lone"] not in TRANSLATED:
                        gs.add("4a_others", numC, den, ~lone, inc); gs.add("4a_lone", numC, den, lone, inc)
                    lm, om = lone & inc[:, None], ~lone & inc[:, None]
                    menu_sums[mid] += [numC[om].sum(), den[om].sum(), numC[lm].sum(), den[lm].sum()]
                else:
                    for name in (group, "all"):
                        gs.add(f"old_{name}_others", numC, den, ~lone, inc); gs.add(f"old_{name}_lone", numC, den, lone, inc)
                for name, mask in (("others", ~lone), ("lone", lone)):
                    mm = mask & inc[:, None]
                    mrow[f"{name}_num_C"] += float(numC[mm].sum())
                    mrow[f"{name}_num_A"] += float(numA[mm].sum())
                    mrow[f"{name}_den"] += float(den[mm].sum())
                mrow["excluded_splits"] += int((~inc).sum())
                for j, c in enumerate(codes):
                    sub_rows.append({"K": K, "model": h, "dataset": d, "menu_set": kind, "menu": mid, "codes": " ".join(codes),
                                     "donor": donor, "path": c, "n_valid_splits": int(s["valid"][:, j].sum()),
                                     "n_included_splits": int(inc.sum()), "sum_num_C": float(numC[inc, j].sum()),
                                     "sum_num_A": float(numA[inc, j].sum()), "sum_den": float(den[inc, j].sum()),
                                     "n_lone": int((info["lone"][inc] == j).sum()),
                                     **{f"n_rank{r}": int((ranks[inc, j] == r).sum()) for r in range(1, K + 1)}})
            menu_rows.append({"K": K, "model": h, "dataset": d, "menu_set": kind, "menu": mid, "codes": " ".join(codes), "group": group,
                              "lone_table": m["lone"], "degree_pp": 100 * m["degree"],
                              "excluded_share": mrow["excluded_splits"] / (len(DONORS) * R),
                              **{k: v for k, v in mrow.items() if k != "excluded_splits"}})
        sim_rows.append({"K": K, "model": h, "dataset": d,
                         "pi_lone_all": float(np.mean(sim_vals["all"][0])), "pi_others_all": float(np.mean(sim_vals["all"][1])),
                         "pi_lone_clear": float(np.mean(sim_vals[GROUPS[0]][0])), "pi_others_clear": float(np.mean(sim_vals[GROUPS[0]][1]))})
        if not weak:
            continue
        clear = GROUPS[0]
        if any(gs.den.get(name, 0) <= 0 for name in (f"E_{clear}_others", f"E_{clear}_lone")):
            raise SystemExit(f"❌ §5：K = {K} | {h} | {d} 在判定中沒有可用的（菜單、供體、切分），停下來回報")
        row = {"K": K, "model": h, "dataset": d,
               "excluded_share_conf": excl["conf"] / total["conf"], "excluded_share_old": excl["old"] / total["old"],
               "E": gs.diff(f"E_{clear}_others", f"E_{clear}_lone"),
               "others_effect": gs.effect(f"E_{clear}_others"), "lone_effect": gs.effect(f"E_{clear}_lone"),
               **{f"E_{g}": gs.diff(f"E_{g}_others", f"E_{g}_lone") for g in GROUPS}, "E_all": gs.diff("E_all_others", "E_all_lone"),
               **{f"rank{r}_effect": gs.effect(f"rank{r}") for r in range(1, K + 1)},
               **{f"rankclear{r}_effect": gs.effect(f"rankclear{r}") for r in range(1, K + 1)},
               "rank_diff": gs.diff(f"rank{K}", "rank1"), "rankclear_diff": gs.diff(f"rankclear{K}", "rankclear1"),
               "E_A": gs.diff("EA_others", "EA_lone"),
               "E_4a": gs.diff("4a_others", "4a_lone") if "4a_others" in gs.num else float("nan"),
               "E_4b": gs.diff("4b_others", "4b_lone") if gs.den.get("4b_lone", 0) > 0 else float("nan"),
               "E_4c": gs.diff("4c_other", "4c_low") if gs.den.get("4c_low", 0) > 0 else float("nan"),
               "share_4bc": use4["used"] / use4["total"] if use4["total"] else float("nan"), "n_4bc": use4["used"],
               **{f"old_E_{g}": gs.diff(f"old_{g}_others", f"old_{g}_lone") if f"old_{g}_others" in gs.num else float("nan")
                  for g in GROUPS},
               "old_E_all": gs.diff("old_all_others", "old_all_lone"),
               "others_vote_pp": 100 * gs.num[f"E_{clear}_others"] / gs.n[f"E_{clear}_others"],
               "others_path_pp": 100 * gs.den[f"E_{clear}_others"] / gs.n[f"E_{clear}_others"],
               "lone_vote_pp": 100 * gs.num[f"E_{clear}_lone"] / gs.n[f"E_{clear}_lone"],
               "lone_path_pp": 100 * gs.den[f"E_{clear}_lone"] / gs.n[f"E_{clear}_lone"]}
        row["ratio_lone_others"] = row["lone_effect"] / row["others_effect"]
        block_rows.append(row)
        print(f"  K = {K} {h:10s} {d:14s} E {row['E']:+.3f}（{elapsed()}）")
    conf_menus = pd.DataFrame([{"menu": m["mid"], "codes": " ".join(m["codes"]), "group": m["group"], "lone_table": m["lone"],
                                "degree_pp": 100 * m["degree"],
                                "E_menu": 10 * (menu_sums[m["mid"]][0] / menu_sums[m["mid"]][1] - menu_sums[m["mid"]][2] / menu_sums[m["mid"]][3])}
                               for m in conf_rows])
    return {"frame": frame, "conf_rows": conf_rows, "old_rows": old_rows, "top_min": top_min, "bot_max": bot_max,
            "blocks": pd.DataFrame(block_rows), "sim": pd.DataFrame(sim_rows), "menu_blocks": pd.DataFrame(menu_rows),
            "subs": pd.DataFrame(sub_rows), "conf_menus": conf_menus,
            "rho": float(spearmanr(conf_menus.degree_pp, conf_menus.E_menu).statistic)}


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
    weak_keys = [(h, d) for h in HOSTS for d in DATASETS]
    rates = agreementRates([coded[k] for k in weak_keys])
    print(f"載入與編碼：{len(coded)} 個區塊（{elapsed()}）")

    # ------------------------------------------------------------------
    # 開跑前的檢查（第 9 節）
    # ------------------------------------------------------------------
    checks = {}
    # §9.2 K = 3 的菜單表
    conf3, _ = menuSets()
    rs_menu = pd.read_csv(RQ3GS_MENU_BLOCKS)
    rs_menu = rs_menu[rs_menu.menu_set == "conf"].drop_duplicates("menu").set_index("menu")
    rows3 = {mid: menuRow(codes, rates) for mid, codes in conf3}
    lone_bad = [mid for mid, r in rows3.items() if r["lone"] != rs_menu.loc[mid, "lone_table"]]
    deg_diff = max(abs(100 * r["degree"] - rs_menu.loc[mid, "degree_pp"]) for mid, r in rows3.items())
    checks[2] = {"ok": not lone_bad and deg_diff <= 1e-12 and len(rows3) == 187, "lone_bad": lone_bad, "deg_diff": deg_diff}
    print(f"§9.2 K = 3 的菜單表：落單的 path 不同 {len(lone_bad)} 份，落單程度最大差 {deg_diff:.1e}pp（{elapsed()}）")

    # §9.1 K = 3 重現 RQ3-GS 判定二（明顯落單組，用 RQ3-GS 的分組）
    gro3 = [menuGrouping(codes, rates) for _, codes in conf3]
    groups3, _, _ = assignGroups(gro3)
    clear3 = [codes for (mid, codes), g in zip(conf3, groups3) if g == GROUPS[0]]
    rs_blocks = pd.read_csv(RQ3GS_BLOCKS).set_index(["model", "dataset"])
    e2, diff1 = {}, 0.0
    for h, d in weak_keys:
        gs = GroupSums()
        for codes in clear3:
            lone = onehot(hostSide(coded[(h, d)], codes, splits[d])["lone"], 3)
            for donor in DONORS:
                s = substitutionSplits(HostDonorK(coded[(h, d)], coded[(donor, d)], splits[d]), codes, ("C",))
                gs.add("pair", s["num"]["C"], s["den"], ~lone, s["included"]); gs.add("lone", s["num"]["C"], s["den"], lone, s["included"])
        e2[(h, d)] = gs.diff("pair", "lone")
        diff1 = max(diff1, abs(e2[(h, d)] - rs_blocks.loc[(h, d), "E2"]))
    s1 = summarize(list(e2.values()))
    shown1 = (f"{s1['mean']:+.2f}", f"{s1['ci_low']:+.2f}", f"{s1['ci_high']:+.2f}", s1["n_positive"])
    checks[1] = {"ok": diff1 <= TOL and shown1 == RQ3GS_SUMMARY, "diff": diff1, "shown": shown1}
    print(f"§9.1 重現 RQ3-GS 判定二：{shown1}，逐區塊最大差 {diff1:.1e}（{elapsed()}）")

    # §9.4 菜單表 = 確認前寫好的 CSV
    frames, problems4 = {}, []
    for K in KS:
        frame, conf_rows, old_rows, _, _ = menuTableFrame(K, rates)
        frames[K] = frame
        text = menuTableCsv(frame)
        recorded = re.search(rf"`result/analysis/rq3gsk/rq3gsk_menus_K{K}\.csv` \| `([0-9a-f]{{64}})`", criteria_text).group(1)
        on_disk = Path(args.out_dir, f"rq3gsk_menus_K{K}.csv").read_bytes()
        if hashlib.sha256(text.encode()).hexdigest() != recorded or on_disk != text.encode():
            problems4.append(f"K = {K}: recomputed menu table differs from the CSV recorded in the criteria")
        old_set = {tuple(r["codes"]) for r in old_rows}
        if len(conf_rows) != 762 or len(old_rows) != 30 or any(tuple(r["codes"]) in old_set for r in conf_rows):
            problems4.append(f"K = {K}: {len(conf_rows)} confirmation menus / overlap with RQ3-GK")
    checks[4] = {"ok": not problems4, "problems": problems4}
    print(f"§9.4 菜單：{'通過' if not problems4 else problems4}（{elapsed()}）")

    # §9.3 重現 RQ3-GK 在 K = 5、7 的 D1 與轉換率
    gk = pd.read_csv(RQ3GK_BLOCKS)
    gk["K"] = gk.K.astype(str)
    gk = gk.set_index(["K", "model", "dataset"])
    diff3, rows_3 = 0.0, []
    for K in KS:
        old = [r for r in menuTableFrame(K, rates)[2]]
        for (m, d), block in coded.items():
            for rule in ("A", "C"):
                v = float(np.mean([hostSide(block, r["codes"], splits[d], pi_rules=(), d1_rules=(rule,))["D1"][rule] for r in old]))
                diff3 = max(diff3, abs(v - gk.loc[(str(K), m, d), f"D1_{rule}"]))
        for h, d in weak_keys:
            parts = []
            for donor in DONORS:
                ctx = HostDonorK(coded[(h, d)], coded[(donor, d)], splits[d])
                for r in old:
                    parts += [p for p in conversionParts(ctx, r["codes"]) if p["participates"]]
            inc = sum(p["path_inc"] for p in parts)
            for rule in ("A", "C"):
                for kind in ("real", "sim"):
                    v = sum(p[f"{kind}_{rule}"] for p in parts) / inc
                    ref = gk.loc[(str(K), h, d), f"conversion_{kind}_{rule}"]
                    diff3 = max(diff3, abs(v - ref))
                    rows_3.append({"K": K, "model": h, "dataset": d, "col": f"conversion_{kind}_{rule}", "mine": v, "rq3gk": ref})
    checks[3] = {"ok": diff3 <= TOL, "diff": diff3}
    print(f"§9.3 重現 RQ3-GK（K = 5、7 的 D1_A、D1_C、轉換率）：最大差 {diff3:.1e}（{elapsed()}）")

    # §9.5 換成自己
    self_bad, self_count = 0, 0
    for K in KS:
        conf_rows = menuTableFrame(K, rates)[1]
        for h, d in weak_keys:
            ctx = HostDonorK(coded[(h, d)], coded[(h, d)], splits[d])
            for r in conf_rows:
                s = substitutionSplits(ctx, r["codes"])
                self_count += K
                self_bad += int(np.any(s["num_int"]["C"] != 0) or np.any(s["num_int"]["A"] != 0) or np.any(s["den_int"] != 0)
                                or s["valid"].any() or s["included"].any())
    checks[5] = {"ok": self_bad == 0, "count": self_count}
    print(f"§9.5 換成自己：{self_count} 個替換，不符 {self_bad}（{elapsed()}）")

    # §9.6 手算
    hand_codes = frames[5].set_index("menu").loc[HAND[2], "codes"].split()
    hand = handCheck(args, hand_codes)
    hblock = coded[(HAND[0], HAND[1])]
    info = hostSide(hblock, hand_codes, splits[HAND[1]])
    lone = onehot(info["lone"], len(hand_codes))
    mine = {"others_num": 0.0, "others_den": 0.0, "lone_num": 0.0, "lone_den": 0.0}
    diff6, hand_rows = 0.0, []
    for donor in DONORS:
        s = substitutionSplits(HostDonorK(hblock, coded[(donor, HAND[1])], splits[HAND[1]]), hand_codes)
        inc = s["included"]
        for key, mask in (("others", ~lone), ("lone", lone)):
            mm = mask & inc[:, None]
            mine[f"{key}_num"] += float(s["num"]["C"][mm].sum())
            mine[f"{key}_den"] += float(s["den"][mm].sum())
        for j, c in enumerate(hand_codes):
            a_num, a_den = float(s["num"]["C"][inc, j].sum()), float(s["den"][inc, j].sum())
            h_num, h_den = hand["per_path"][donor][c]
            diff6 = max(diff6, abs(a_num - h_num), abs(a_den - h_den))
            hand_rows.append({"供體": MODEL_LABELS[donor], "path": c, "程式 Σ分子": f"{a_num:.10f}", "手算 Σ分子": f"{h_num:.10f}",
                              "程式 Σ分母": f"{a_den:.10f}", "手算 Σ分母": f"{h_den:.10f}"})
        if int((~inc).sum()) != hand["excluded"][donor]:
            diff6 = float("inf")
        hand_rows.append({"供體": MODEL_LABELS[donor], "path": "被排除的切分", "程式 Σ分子": str(int((~inc).sum())),
                          "手算 Σ分子": str(hand["excluded"][donor]), "程式 Σ分母": "", "手算 Σ分母": ""})
    mine_others, mine_lone = 10 * mine["others_num"] / mine["others_den"], 10 * mine["lone_num"] / mine["lone_den"]
    diff6 = max(diff6, abs(mine_others - hand["others_effect"]), abs(mine_lone - hand["lone_effect"]))
    checks[6] = {"ok": diff6 <= TOL, "diff": diff6}
    print(f"§9.6 手算：最大差 {diff6:.1e}（{elapsed()}）")
    if not all(c["ok"] for c in checks.values()):
        for n in sorted(checks):
            print(f"  §9.{n}: {'通過' if checks[n]['ok'] else '❌ 不符'} {checks[n]}")
        raise SystemExit("❌ 開跑前的檢查沒有全部通過：停下來回報，不寫輸出")

    # ------------------------------------------------------------------
    # 正式計算
    # ------------------------------------------------------------------
    res = {K: runK(K, coded, splits, rates, weak_keys, elapsed) for K in KS}

    # 第 8 節 9：各區塊自己算出的落單 path
    own = []
    menu_lists = {3: [(codes, rows3[mid]["lone"]) for mid, codes in conf3]}
    for K in KS:
        menu_lists[K] = [(r["codes"], r["lone"]) for r in res[K]["conf_rows"]]
    for h, d in weak_keys:
        row = {"model": h, "dataset": d}
        for K, lst in menu_lists.items():
            row[f"own_lone_same_K{K}"] = float(np.mean([blockLone(coded[(h, d)], codes) == lone for codes, lone in lst]))
        own.append(row)
    own = pd.DataFrame(own)
    k3_ratio = (rs_blocks.loc[weak_keys, "E2_lone_effect"] / rs_blocks.loc[weak_keys, "E2_pair_effect"]).to_numpy()

    # 輸出
    os.makedirs(args.out_dir, exist_ok=True)
    blocks = pd.concat([res[K]["sim"].merge(res[K]["blocks"], on=["K", "model", "dataset"], how="left") for K in KS])
    blocks = blocks.merge(own, on=["model", "dataset"], how="left")
    blocks.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq3gsk_blocks.csv"), index=False)
    menu_blocks = pd.concat([res[K]["menu_blocks"].merge(res[K]["conf_menus"][["menu", "E_menu"]], on="menu", how="left") for K in KS])
    menu_blocks.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq3gsk_menu_blocks.csv"), index=False)
    pd.concat([res[K]["subs"] for K in KS]).assign(criteria_sha256=criteria_sha).to_csv(
        os.path.join(args.out_dir, "rq3gsk_substitutions.csv"), index=False)

    weak = {K: res[K]["blocks"] for K in KS}
    S = lambda K, col: summarize(weak[K][col].to_numpy())
    group_stats = {(K, g): S(K, f"E_{g}") for K in KS for g in GROUPS}
    group_stats.update({(3, g): summarize(rs_blocks.loc[weak_keys, f"E2_{g}"].to_numpy()) for g in GROUPS})
    plotGroups(group_stats, {K: Counter(r["group"] for r in res[K]["conf_rows"])[GROUPS[0]] for K in KS},
               os.path.join(args.out_dir, "fig_a_groups"), criteria_sha)
    rank_stats = {(K, r): S(K, f"rank{r}_effect") for K in KS for r in range(1, K + 1)}
    plotRanks(rank_stats, os.path.join(args.out_dir, "fig_b_rank_effect"), criteria_sha)

    # ------------------------------------------------------------------
    # 報告
    # ------------------------------------------------------------------
    J = {K: S(K, "E") for K in KS}
    state = {K: fourState(J[K], THRESHOLDS[K]) for K in KS}
    out = ["# RQ3-GSK：K = 5、7 時，和其他條最不像的那條還是最不值得改嗎？", "",
           f"判定標準 `{criteria}`，sha256 `{criteria_sha}`；確認：{confirmed}。"
           f"產生時間 {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}；程式 `scripts/analysis_rq3gsk/path_improve_gsk.py`"
           "（逐切分計算在 `Analysis/pathImproveGSK.py`，沿用 `Analysis/pathImprove.py`、`pathImproveK.py`、`pathImproveGS.py`）。"
           "不呼叫 API，不重跑任何東西，不修改現有檔案。", "",
           "效果的單位：那條 path 每進步 10 個百分點，投票多幾個百分點（第 5 節）。E_K = 其他 K − 1 條的效果 − 落單那條的效果；"
           "門檻 K = 5 用 0.30、K = 7 用 0.20，單位相同。", ""]
    out += ["## (1) 讀了哪些檔案", "",
            f"- `{args.armdir}/{{模型}}/{{資料集}}/`：4 個模型 × 4 個資料集的 12 條 path，經 `Analysis.menuVote.loadPathBlock` 載入；"
            "欄位 `item_id`、`gold`、`parsed_answer`、`parse_ok`。`CellData` 對 `result/aggregations/` 只列出檔名。",
            f"- `{args.out_dir}/rq3gsk_menus_K5.csv`、`rq3gsk_menus_K7.csv`：確認前寫好的菜單表，只核對（§9.4），不覆寫。",
            f"- `{RQ3GS_BLOCKS}`：§9.1 的核對（`E2`）、第 8 節 11 的 K = 3 對照（`E2_lone_effect`、`E2_pair_effect`）、圖 (a) 的 K = 3"
            "（`E2_明顯落單`、`E2_中間`、`E2_對稱`）。",
            f"- `{RQ3GS_MENU_BLOCKS}`：§9.2 的核對（`lone_table`、`degree_pp`）。",
            f"- `{RQ3GK_BLOCKS}`：§9.3 的核對（K = 5、7 的 `D1_A`、`D1_C`、`conversion_real_A/C`、`conversion_sim_A/C`）。",
            "- RQ3-GK 的菜單由 `Analysis.pathImproveK.drawMenus` 重抽（與 `rq3gk_criteria.md` §3 的表相同）。",
            "- §9.6 的手算另外用 `Analysis.menuVote.loadPathBlock` 的字串答案、純迴圈與分數重算（不用 `pathImproveK`、`pathImproveGS`、`pathImproveGSK`）。", ""]

    tbl = []
    for K in KS:
        conf_rows = res[K]["conf_rows"]
        for g in GROUPS:
            rr = [r for r in conf_rows if r["group"] == g]
            deg = [100 * r["degree"] for r in rr]
            tbl.append({"K": K, "組別": g, "份數": len(rr), "落單程度範圍（pp）": f"{min(deg):.3f} – {max(deg):.3f}",
                        "落單的 path": "、".join(f"{k} {v}" for k, v in Counter(r["lone"] for r in rr).most_common())})
    ex = pd.DataFrame([{"K": K, "宿主": MODEL_LABELS[r.model], "資料集": r.dataset, "被排除的（菜單、供體、切分）%": 100 * r.excluded_share_conf}
                       for K in KS for r in weak[K].itertuples()])
    out += ["## (2) 第零階段的表", "",
            "菜單：K = 5、7 各 792 種，扣掉 RQ3-GK 的 30 種，各 762 份確認用的菜單，每組 254 份（判定標準檔第 4 節與附錄 A）。", "",
            md(pd.DataFrame(tbl), text=True), "",
            "各區塊被有效切分的規則排除的比例（確認用的菜單，兩個供體合計；第 5 節）：", "",
            md(ex.pivot_table(index=["宿主", "資料集"], columns="K", values="被排除的（菜單、供體、切分）%").reset_index(), ".2f"), ""]

    out += ["## (3) 開跑前的檢查", "",
            f"1. K = 3 重現 RQ3-GS 判定二（s_i 與第 3 節的平手規則）：{shown1[0]}（{shown1[1]} 到 {shown1[2]}），{shown1[3]} / 8；"
            f"逐區塊和 `rq3gs_blocks.csv` 的 `E2` 最大差 {checks[1]['diff']:.1e}（門檻 1e-9）→ 通過。",
            f"2. K = 3 的菜單表（187 份）：落單的 path 全部相同；落單程度最大差 {checks[2]['deg_diff']:.1e}pp（門檻 1e-12）→ 通過。",
            f"3. 重現 RQ3-GK 在 K = 5、7 的 `D1_A`、`D1_C`（16 個區塊）與 `conversion_real_A/C`、`conversion_sim_A/C`（8 個區塊）："
            f"最大差 {checks[3]['diff']:.1e}（門檻 1e-9）→ 通過。",
            "4. 確認用的菜單和 RQ3-GK 的 K5、K7 菜單沒有重複，各 762 份；重算的菜單表與兩個 CSV 逐位元組相同，sha256 等於判定標準檔第 4 節的值 → 通過。",
            f"5. 換成自己：{checks[5]['count']} 個替換（K = 5、7 × 8 個區塊 × 762 份 × 每條 path），分子、分母每次切分都是 0，"
            "而且都不是有效切分 → 通過。",
            f"6. 手算（{MODEL_LABELS[HAND[0]]} · {HAND[1]}、{HAND[2]}：{' '.join(hand_codes)}，兩個供體）：最大差 {checks[6]['diff']:.1e}"
            f"（門檻 1e-9）→ 通過。其他條的效果：程式 {mine_others:.6f}、手算 {hand['others_effect']:.6f}；落單那條的效果：程式 "
            f"{mine_lone:.6f}、手算 {hand['lone_effect']:.6f}。", "",
            md(pd.DataFrame(hand_rows), text=True), ""]

    out += ["## (4) 兩項判定（8 個弱模型區塊，各自的「明顯落單」組 254 份，規則 C）", "",
            "限制（判定標準檔第 6 節）：(a) 明顯落單組裡落單的 path 全是翻譯的語言 path，判定分不開「不像」「是翻譯的」「比較弱」；"
            "(b) 菜單只由 12 條 path 排出來，彼此不獨立；(c) 兩項判定用同一批題目與 path，不是兩次獨立的確認。", ""]
    for n, K in enumerate(KS, 1):
        per = withLabels(weak[K][["model", "dataset", "E", "others_effect", "lone_effect", "excluded_share_conf"]].assign(
            excluded_share_conf=lambda x: 100 * x.excluded_share_conf))
        out += [f"### 判定{'一二'[n - 1]}：K = {K}（門檻 {THRESHOLDS[K]:.2f}）", "",
                md(pd.DataFrame([ciRow(f"E_{K} = 其他 {K - 1} 條的效果 − 落單那條的效果", J[K], THRESHOLDS[K])]), text=True), "",
                f"**{state[K]}** → {READINGS[state[K]]}", "",
                md(per.rename(columns={"E": f"E_{K}", "others_effect": f"其他 {K - 1} 條的效果", "lone_effect": "落單那條的效果",
                                       "excluded_share_conf": "被排除的（菜單、供體、切分）%"})), ""]

    out += ["## (5) 只報告的量（不參與判定）", ""]
    out += ["### 8.1 E_K 在中間組、對稱組、全部 762 份", "",
            md(pd.DataFrame([ciRow(f"K = {K}，{name}", S(K, col)) for K in KS
                             for name, col in (("明顯落單（判定）", "E"), ("中間", f"E_{GROUPS[1]}"), ("對稱", f"E_{GROUPS[2]}"),
                                               ("全部 762 份", "E_all"))]), text=True), "",
            "![三組菜單的 E_K](fig_a_groups.png)", ""]
    rk = []
    for K in KS:
        for scope, pre, diffcol in (("全部 762 份", "rank", "rank_diff"), ("明顯落單組", "rankclear", "rankclear_diff")):
            rk += [ciRow(f"K = {K}，{scope}，第 {r} 名{'（最落單）' if r == 1 else '（最不落單）' if r == K else ''}的效果", S(K, f"{pre}{r}_effect"))
                   for r in range(1, K + 1)]
            rk.append(ciRow(f"K = {K}，{scope}：最不落單的效果 − 最落單的效果", S(K, diffcol)))
    out += ["### 8.2 依名次的效果", "", "名次每次切分在選擇半依第 3 節決定（1 = 最落單）。", "", md(pd.DataFrame(rk), text=True), "",
            "![依名次的效果](fig_b_rank_effect.png)", ""]
    out += ["### 8.3 落單程度對上菜單的 E_K", "",
            "菜單的 E_K = 8 個區塊、2 個供體、所有可用切分合併成一個值。菜單只由 12 條 path 排出來，彼此不獨立，相關係數看起來會比實際的證據強。", "",
            md(pd.DataFrame([{"K": K, "菜單數": len(res[K]["conf_menus"]), "Spearman": f"{res[K]['rho']:+.3f}"} for K in KS]), text=True),
            "", "逐菜單的值在 `rq3gsk_menu_blocks.csv` 的 `E_menu`。", ""]
    a4 = []
    for K in KS:
        cm = res[K]["conf_menus"]
        nt = cm[~cm.lone_table.isin(TRANSLATED)]
        a4 += [ciRow(f"K = {K} (a) 落單的 path 不是翻譯的語言 path 的菜單（{len(nt)} 份，落單程度 {nt.degree_pp.min():.3f}–"
                     f"{nt.degree_pp.max():.3f}pp）：E_K", S(K, "E_4a")),
               ciRow(f"K = {K} (b) 落單的不是正確率最低那條時的 E_K", summarize(weak[K].E_4b.dropna())),
               ciRow(f"K = {K} (c) 正確率最低的不是落單那條時：其他條 − 最低那條", summarize(weak[K].E_4c.dropna()))]
    t4 = pd.concat([withLabels(weak[K][["model", "dataset", "share_4bc", "n_4bc", "E_4b", "E_4c"]]).assign(K=K) for K in KS])
    t4["share_4bc"] = 100 * t4.share_4bc
    out += ["### 8.4 分開原因", "",
            "(b) 與 (c) 用的是同一個條件（落單的 path ≠ 選擇半正確率最低的那條），範圍是明顯落單組，所以用到的（菜單、供體、切分）相同。"
            "沒有這種情況的區塊值無定義，摘要只用有資料的區塊。", "",
            md(pd.DataFrame(a4), text=True), "",
            md(t4[["K", "宿主", "資料集", "share_4bc", "n_4bc", "E_4b", "E_4c"]].rename(columns={
                "share_4bc": "(b)(c) 用到的比例（%）", "n_4bc": "用到的（菜單、供體、切分）", "E_4b": "(b) E_K", "E_4c": "(c) 其他條 − 最低那條"})), ""]
    sim_txt = []
    for K in KS:
        sm = res[K]["sim"]
        sim_txt.append(f"- K = {K}：全部 762 份 {sm.pi_lone_all.mean() / sm.pi_others_all.mean():.3f}"
                       f"（{100 * sm.pi_lone_all.mean():.2f}% ÷ {100 * sm.pi_others_all.mean():.2f}%）；明顯落單組 "
                       f"{sm.pi_lone_clear.mean() / sm.pi_others_clear.mean():.3f}"
                       f"（{100 * sm.pi_lone_clear.mean():.2f}% ÷ {100 * sm.pi_others_clear.mean():.2f}%）。")
    out += ["### 8.5 模擬這一側（16 個區塊，子集一，規則 C）", "",
            "落單的 path 的 π ÷ 其他 K − 1 條 π 的平均（分子、分母各自先對切分、菜單、16 個區塊平均，再相除）：", "", *sim_txt, ""]
    out += ["### 8.6 規則 A 下的值", "",
            md(pd.DataFrame([ciRow(f"K = {K}（規則 A）：E_K", S(K, "E_A")) for K in KS]), text=True), ""]
    o7 = []
    for K in KS:
        oc = Counter(r["group"] for r in res[K]["old_rows"])
        o7 += [ciRow(f"K = {K}，{g}（{oc[g]} 份）", S(K, f"old_E_{g}")) for g in GROUPS if oc[g] > 0]
        o7.append(ciRow(f"K = {K}，全部 30 份", S(K, "old_E_all")))
    out += ["### 8.7 RQ3-GK 既有的菜單（K5-01 到 K5-30、K7-01 到 K7-30）", "",
            "用第 4 節的門檻分組。", "", md(pd.DataFrame(o7), text=True), ""]
    out += ["### 8.8 不除以分母的版本（明顯落單組，pp）", "",
            md(pd.DataFrame([ciRow(f"K = {K}，{name}", S(K, col)) for K in KS
                             for name, col in (("其他條：投票實際多幾個百分點", "others_vote_pp"),
                                               ("其他條：那條 path 實際進步幾個百分點", "others_path_pp"),
                                               ("落單那條：投票實際多幾個百分點", "lone_vote_pp"),
                                               ("落單那條：那條 path 實際進步幾個百分點", "lone_path_pp"))]), text=True), ""]
    own_t = withLabels(own).rename(columns={f"own_lone_same_K{K}": f"K = {K}" for K in (3, 5, 7)})
    for K in (3, 5, 7):
        own_t[f"K = {K}"] = 100 * own_t[f"K = {K}"]
    out += ["### 8.9 各區塊自己算出的落單 path 和菜單表相同的比例（%）", "",
            "每個區塊用自己子集一的全部題目算 s_i；K = 3 用 RQ3-GS 的 187 份，K = 5、7 用各自的 762 份。", "",
            md(own_t, ".1f"), "",
            "8 個區塊的平均：" + "、".join(f"K = {K} {100 * own[f'own_lone_same_K{K}'].mean():.1f}%" for K in (3, 5, 7)) + "。", ""]
    out += ["### 8.10 門檻改用 0.5 時的狀態（對照）", "",
            md(pd.DataFrame([ciRow(f"K = {K}：E_K（門檻 0.5）", J[K], REFERENCE_THRESHOLD) for K in KS]), text=True), ""]
    r11 = [{"K": 3, "8 個值": "、".join(f"{v:.3f}" for v in k3_ratio), "中位數": f"{np.median(k3_ratio):.3f}",
            "範圍": f"{k3_ratio.min():.3f} – {k3_ratio.max():.3f}"}]
    for K in KS:
        v = weak[K].ratio_lone_others.to_numpy()
        r11.append({"K": K, "8 個值": "、".join(f"{x:.3f}" for x in v), "中位數": f"{np.median(v):.3f}", "範圍": f"{v.min():.3f} – {v.max():.3f}"})
    out += ["### 8.11 落單那條的效果 ÷ 其他條的效果（逐區塊）", "",
            "K = 3 用 RQ3-GS 明顯落單組的 `E2_lone_effect` ÷ `E2_pair_effect`。8 個值的順序：GPT-4o mini 的 mmlu、mathqa、truthfulqa、"
            "commonsenseqa，再來 Qwen 的同樣四個。", "", md(pd.DataFrame(r11), text=True), ""]

    out += ["## (6) 對照讀法的結論", ""]
    for n, K in enumerate(KS, 1):
        out.append(f"- 判定{'一二'[n - 1]}（K = {K}，E_{K} {pp(J[K]['mean'])}，[{pp(J[K]['ci_low'])}, {pp(J[K]['ci_high'])}]，"
                   f"{J[K]['n_positive']}/8 為正，門檻 {THRESHOLDS[K]:.2f}）：**{state[K]}**。{READINGS[state[K]]}")
    out += ["- 其餘都只報告，不套四種狀態（第 8 節 10 只是門檻 0.5 的對照）。", ""]
    Path(os.path.join(args.out_dir, "report.md")).write_text("\n".join(out), encoding="utf-8")
    for K in KS:
        print(f"K = {K}: E {pp(J[K]['mean'])} [{pp(J[K]['ci_low'])}, {pp(J[K]['ci_high'])}] {J[K]['n_positive']}/8 -> {state[K]}")
    print(f"-> {args.out_dir}/report.md（{elapsed()}）")


if __name__ == "__main__":
    main()
