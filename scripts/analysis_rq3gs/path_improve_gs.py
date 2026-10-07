"""
path_improve_gs.py — RQ3-GS：在沒看過的三條組合上，確認挑法有沒有挑對（result/analysis/rq3gs/rq3gs_criteria.md）

全程離線：不呼叫 API，不重跑任何東西，不修改現有檔案。判定標準確認前不執行。
    1. 載入 16 個區塊並編碼（沿用 Analysis/pathImprove.py 的核對）；確認用的 187 份與舊的 30 份菜單；第 4 節的分組
    2. 開跑前的檢查（§10，任何一項不過就停，不寫輸出）：重現 RQ3-GK；菜單與分組 = 判定標準檔的表；換成自己；手算
    3. 宿主這一側（每次切分的 p*、落單的 path …）與替換這一側（分子、分母），依第 5 節加總成各群的效果
    4. 輸出（--out-dir）：rq3gs_blocks.csv、rq3gs_menu_blocks.csv、rq3gs_substitutions.csv、兩張圖、report.md

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq3gs/path_improve_gs.py
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
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
from Analysis.blockStats import summarize, fourState, FORWARD, REVERSE, EQUIVALENT, UNDETERMINED
from Analysis.menuJudgeStats import blocksWithSplits
from Analysis.menuVote import MODELS, DATASETS, loadPathBlock
from Analysis.splitHalf import makeSplits
from Analysis.pathImprove import M12, HOSTS, DONORS, THRESHOLD, encodeBlocks, checkVote
from Analysis.pathImproveK import HostDonorK, drawMenus, substituteMenu
from Analysis.pathImproveGS import (GROUPS, PAIR_SLOTS, menuSets, agreementRates, menuGrouping, assignGroups, thresholdGroup,
                                    hostSplits, piFull, substitutionSplits, onehot, GroupSums, hitCredit)

OUT_DIR = "result/analysis/rq3gs"
CRITERIA_FILE = "rq3gs_criteria.md"
RQ3GK_BLOCKS = "result/analysis/rq3gk/rq3gk_k_blocks.csv"
MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
READINGS_1 = {
    FORWARD: "在沒看過的組合上，挑法挑出的 path 真的改進時比其他條有效；K = 3 時挑法有效。",
    EQUIVALENT: "挑出的和其他條差不到 0.5；事後看到的差別沒有重現。",
    REVERSE: "挑出的反而較差。",
    UNDETERMINED: "只列數字；事後分析看到的差別不能寫成結論。",
}
READINGS_2 = {
    FORWARD: "最不像的那條最不值得改，在沒看過的組合上成立。（依第 7 節的限制，論文只能寫「和兩條相像的 path 放在一起時，"
             "那條翻譯的 path 最不值得改」；要寫成「最不像的那條最不值得改」，還要看第 9 節 5 與 11。）",
    EQUIVALENT: "落單與否沒有實質差別。",
    REVERSE: "落單的那條反而比較值得改。",
    UNDETERMINED: "只列數字。",
}
RQ3GK_SUMMARY = {"D1_C": ("+0.16", "+0.13", "+0.20", 16), "E1_pp": ("-4.92", "-7.77", "-2.07", 0)}
HAND = ("gpt4omini", "mmlu", "GS-001")
TRANSLATED = {"ES", "JA", "RU", "ZH"}
TOL = 1e-9
GROUP_EN = {"明顯落單": "clear odd-one-out", "中間": "middle", "對稱": "symmetric"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6a64", "#d9d8d2", "#fcfcfb"


def parseArgs():
    parser = ArgumentParser(description="RQ3-GS: confirm the pick and odd-one-out predictions on unseen K=3 menus")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--rq3gk-blocks", default=RQ3GK_BLOCKS)
    parser.add_argument("--out-dir", default=OUT_DIR)
    return parser.parse_args()


def pp(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:+.2f}"


def ciRow(name: str, s: dict, with_state: bool = False) -> dict:
    row = {"量": name, "平均": pp(s["mean"]), "SE": f"{s['se']:.2f}", "95% 區間": f"[{pp(s['ci_low'])}, {pp(s['ci_high'])}]",
           "為正的區塊": f"{s['n_positive']}/{s['n_blocks']}"}
    if with_state:
        row["狀態"] = fourState(s, THRESHOLD)
    return row


def md(df: pd.DataFrame, floatfmt: str = ".2f", text: bool = False) -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt, disable_numparse=text)


def label(model: str, dataset: str) -> str:
    return f"{MODEL_LABELS[model]} · {dataset}"


# ------------------------------------------------------------------
# §10.4 手算（純迴圈、字串答案；不用 Analysis.pathImproveK / pathImproveGS）
# ------------------------------------------------------------------
def handCheck(args, menu_codes: list[str]) -> dict:
    host_model, dataset, _ = HAND
    host = loadPathBlock(args.armdir, host_model, dataset, args.aggdir)
    N = len(host.item_ids)
    splits = makeSplits(N, 200, 0)
    gold = host.gold
    sub1 = [all(host.answered[c][i] for c in M12) for i in range(N)]

    def scoreC(answers: list, i: int) -> Fraction:
        votes = Counter(answers)
        top = max(votes.values())
        tied = [a for a, v in votes.items() if v == top]
        return Fraction(1, len(tied)) if gold[i] in tied else Fraction(0)

    ans = {c: host.answers[c] for c in menu_codes}
    base = [scoreC([ans[c][i] for c in menu_codes], i) for i in range(N)]
    cor = {c: [ans[c][i] == gold[i] for i in range(N)] for c in menu_codes}
    fixed_delta = {}
    for c in menu_codes:
        fixed_delta[c] = [scoreC([gold[i] if d == c else ans[d][i] for d in menu_codes], i) - base[i] for i in range(N)]
    pstar = []
    for h1 in splits:
        idx = [i for i in range(N) if h1[i] and sub1[i]]
        n1 = len(idx)
        best, best_val = None, None
        for c in menu_codes:
            wrong = [i for i in idx if not cor[c][i]]
            w = len(wrong)
            t = min(w, (10 * n1 + 100) // 200)
            val = Fraction(t) * sum((fixed_delta[c][i] for i in wrong), Fraction(0)) / w if w else Fraction(0)
            if best_val is None or val > best_val:
                best, best_val = c, val
        pstar.append(best)

    out = {"per_path": {}, "excluded": {}, "pick_num": 0.0, "pick_den": 0.0, "other_num": 0.0, "other_den": 0.0}
    for donor_model in DONORS:
        donor = loadPathBlock(args.armdir, donor_model, dataset, args.aggdir)
        sub2 = [sub1[i] and all(donor.answered[c][i] for c in M12) for i in range(N)]
        dcor = {c: [donor.answers[c][i] == gold[i] for i in range(N)] for c in menu_codes}
        sub_delta = {c: [scoreC([donor.answers[c][i] if d == c else ans[d][i] for d in menu_codes], i) - base[i]
                         for i in range(N)] for c in menu_codes}
        excluded = 0
        sums = {c: [0.0, 0.0] for c in menu_codes}
        for r, h1 in enumerate(splits):
            H1 = [i for i in range(N) if h1[i] and sub2[i]]
            H2 = [i for i in range(N) if not h1[i] and sub2[i]]
            valid = all(sum(dcor[c][i] for i in H1) > sum(cor[c][i] for i in H1) and
                        sum(dcor[c][i] for i in H2) > sum(cor[c][i] for i in H2) for c in menu_codes)
            if not valid:
                excluded += 1
                continue
            for c in menu_codes:
                num = float(sum((sub_delta[c][i] for i in H2), Fraction(0)) / len(H2))
                den = (sum(dcor[c][i] for i in H2) - sum(cor[c][i] for i in H2)) / len(H2)
                sums[c][0] += num
                sums[c][1] += den
                key = "pick" if c == pstar[r] else "other"
                out[f"{key}_num"] += num
                out[f"{key}_den"] += den
        out["per_path"][donor_model] = sums
        out["excluded"][donor_model] = excluded
    out["pick_effect"] = 10 * out["pick_num"] / out["pick_den"]
    out["other_effect"] = 10 * out["other_num"] / out["other_den"]
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


def plotGroups(stats: dict, counts: dict, path: str, sha: str):
    """(a) 三組菜單的 E1 與 E2（8 個區塊的平均與 95% t 區間）。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(GROUPS))
    for (name, color, marker, dx) in (("E1", SERIES[0], "o", -0.1), ("E2", SERIES[1], "s", 0.1)):
        s = [stats[(name, g)] for g in GROUPS]
        mean = np.array([v["mean"] for v in s])
        ax.vlines(x + dx, [v["ci_low"] for v in s], [v["ci_high"] for v in s], color=color, linewidth=1.8, zorder=2)
        ax.scatter(x + dx, mean, s=52, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.4, zorder=3,
                   label={"E1": "E1 = picked − others", "E2": "E2 = similar pair − odd one out"}[name])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{GROUP_EN[g]}\n({counts[g]} menus)" for g in GROUPS], fontsize=8.5, color=INK)
    styleAxes(ax)
    ax.set_ylabel("vote gain per 10pp path gain, pp (difference)", color=MUTED, fontsize=9)
    ax.set_title("RQ3-GS (a): E1 and E2 by menu group, rule C (8 weak blocks, 95% t interval)", color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", facecolor=SURFACE, metadata={"Subject" if ext == "pdf" else "Description": f"rq3gs_criteria.md sha256 {sha}"})
    plt.close(fig)


def plotScatter(menus: pd.DataFrame, rho: float, path: str, sha: str):
    """(b) 187 份菜單的落單程度對上菜單的 E2（8 個區塊合併）。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for g, color, marker in zip(GROUPS, SERIES, ("o", "s", "^")):
        m = menus[menus.group == g]
        ax.scatter(m.degree_pp, m.E2_menu, s=22, color=color, marker=marker, edgecolors=SURFACE, linewidths=0.6, alpha=0.9,
                   label=f"{GROUP_EN[g]} ({len(m)})", zorder=3)
    styleAxes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel("odd-one-out degree, pp (answer agreement only)", color=MUTED, fontsize=9)
    ax.set_ylabel("E2 of the menu (8 weak blocks pooled)", color=MUTED, fontsize=9)
    ax.set_title(f"RQ3-GS (b): odd-one-out degree vs E2 of each menu\n187 confirmation menus, Spearman {rho:+.2f}", color=INK,
                 fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", facecolor=SURFACE, metadata={"Subject" if ext == "pdf" else "Description": f"rq3gs_criteria.md sha256 {sha}"})
    plt.close(fig)


def main():
    args = parseArgs()
    started = time.time()
    elapsed = lambda: f"{time.time() - started:.0f}s"
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty)")
    criteria_sha = sha256(criteria)

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
    print(f"載入與編碼：{len(coded)} 個區塊（{elapsed()}）")

    # 菜單與分組（§10.2）
    conf, old = menuSets()
    rates = agreementRates([coded[k] for k in weak_keys])
    conf_g = [menuGrouping(codes, rates) for _, codes in conf]
    groups, top_min, bot_max = assignGroups(conf_g)
    for g, grp in zip(conf_g, groups):
        g["group"] = grp
    old_g = [menuGrouping(codes, rates) for _, codes in old]
    for g in old_g:
        g["group"] = thresholdGroup(g["degree"], top_min, bot_max)
    table = {r[1].strip(): r for r in (line.split("|") for line in Path(criteria).read_text(encoding="utf-8").splitlines()
                                        if line.startswith("| GS-"))}
    problems2 = []
    used = {tuple(c) for _, c in old} | {tuple(c) for c in (["EN", "JA", "ZH"], ["EN", "S1", "S2"], ["EN", "P1", "P2"])}
    if len(conf) != 187 or any(tuple(c) in used for _, c in conf):
        problems2.append(f"{len(conf)} confirmation menus or overlap with the old / fixed menus")
    for (mid, codes), g in zip(conf, conf_g):
        r = table.get(mid)
        if r is None or r[2].strip() != " ".join(codes) or r[4].strip() != "-".join(g["similar"]) or r[5].strip() != g["lone"] \
                or r[6].strip() != f"{100 * g['degree']:.3f}" or r[8].strip() != g["group"]:
            problems2.append(f"{mid}: grouping differs from the criteria table")
    ok2 = not problems2 and len(table) == 187
    print(f"§10.2 菜單與分組：187 份，與判定標準檔的表相同 {ok2}")

    # §10.1 重現 RQ3-GK
    menus_k, _ = drawMenus()
    ref = pd.read_csv(args.rq3gk_blocks)
    ref = ref[ref.K == 3].set_index(["model", "dataset"])
    d1, e1, diff1 = {}, {}, 0.0
    for key, block in coded.items():
        d1[key] = float(np.mean([hostSplits(block, codes, splits[key[1]], ("C",))["D1"]["C"].mean() for _, codes in menus_k[3]]))
        diff1 = max(diff1, abs(d1[key] - ref.loc[key, "D1_C"]))
    for h, d in weak_keys:
        menu_vals = []
        for _, codes in menus_k[3]:
            vals = []
            for donor in DONORS:
                ctx = HostDonorK(coded[(h, d)], coded[(donor, d)], splits[d])
                for res in substituteMenu(ctx, codes, rules=("A",)):
                    if res["participates"]:
                        vals.append(float(np.mean(res["rules"]["A"]["VmSB_after_pp"][res["valid"]])))
            if vals:
                menu_vals.append(np.mean(vals))
        e1[(h, d)] = float(np.mean(menu_vals))
        diff1 = max(diff1, abs(e1[(h, d)] - ref.loc[(h, d), "E1_pp"]))
    s_d1, s_e1 = summarize(list(d1.values())), summarize(list(e1.values()))
    shown = {"D1_C": (f"{s_d1['mean']:+.2f}", f"{s_d1['ci_low']:+.2f}", f"{s_d1['ci_high']:+.2f}", s_d1["n_positive"]),
             "E1_pp": (f"{s_e1['mean']:+.2f}", f"{s_e1['ci_low']:+.2f}", f"{s_e1['ci_high']:+.2f}", s_e1["n_positive"])}
    ok1 = diff1 <= TOL and shown == RQ3GK_SUMMARY
    print(f"§10.1 重現 RQ3-GK：{shown}，逐區塊最大差 {diff1:.1e} -> {'通過' if ok1 else '不符'}（{elapsed()}）")

    # §10.3 換成自己
    self_bad, self_count = 0, 0
    for h, d in weak_keys:
        ctx = HostDonorK(coded[(h, d)], coded[(h, d)], splits[d])
        for _, codes in conf:
            s = substitutionSplits(ctx, codes)
            self_count += 3
            self_bad += int(np.any(s["num_int"]["C"] != 0) or np.any(s["num_int"]["A"] != 0) or np.any(s["den_int"] != 0)
                            or s["valid"].any() or s["included"].any())
    ok3 = self_bad == 0
    print(f"§10.3 換成自己：{self_count} 個替換，不符 {self_bad} -> {'通過' if ok3 else '不符'}（{elapsed()}）")

    # §10.4 手算
    hand_codes = dict(conf)[HAND[2]]
    hand = handCheck(args, hand_codes)
    hblock = coded[(HAND[0], HAND[1])]
    info = hostSplits(hblock, hand_codes, splits[HAND[1]])
    mine = {"pick_num": 0.0, "pick_den": 0.0, "other_num": 0.0, "other_den": 0.0}
    diff4, hand_rows = 0.0, []
    for donor in DONORS:
        s = substitutionSplits(HostDonorK(hblock, coded[(donor, HAND[1])], splits[HAND[1]]), hand_codes)
        inc = s["included"]
        pick = onehot(info["pstar"]["C"])
        for key, mask in (("pick", pick), ("other", ~pick)):
            m = mask & inc[:, None]
            mine[f"{key}_num"] += float(s["num"]["C"][m].sum())
            mine[f"{key}_den"] += float(s["den"][m].sum())
        for j, c in enumerate(hand_codes):
            a_num, a_den = float(s["num"]["C"][inc, j].sum()), float(s["den"][inc, j].sum())
            h_num, h_den = hand["per_path"][donor][c]
            diff4 = max(diff4, abs(a_num - h_num), abs(a_den - h_den))
            hand_rows.append({"供體": MODEL_LABELS[donor], "path": c, "程式 Σ分子": f"{a_num:.10f}", "手算 Σ分子": f"{h_num:.10f}",
                              "程式 Σ分母": f"{a_den:.10f}", "手算 Σ分母": f"{h_den:.10f}"})
        if int((~inc).sum()) != hand["excluded"][donor]:
            diff4 = float("inf")
        hand_rows.append({"供體": MODEL_LABELS[donor], "path": "被排除的切分", "程式 Σ分子": str(int((~inc).sum())),
                          "手算 Σ分子": str(hand["excluded"][donor]), "程式 Σ分母": "", "手算 Σ分母": ""})
    mine_pick, mine_other = 10 * mine["pick_num"] / mine["pick_den"], 10 * mine["other_num"] / mine["other_den"]
    diff4 = max(diff4, abs(mine_pick - hand["pick_effect"]), abs(mine_other - hand["other_effect"]))
    ok4 = diff4 <= TOL
    print(f"§10.4 手算：最大差 {diff4:.1e} -> {'通過' if ok4 else '不符'}（{elapsed()}）")
    if not (ok1 and ok2 and ok3 and ok4):
        for p in problems2[:10]:
            print(f"  ❌ {p}")
        raise SystemExit("❌ 開跑前的檢查沒有全部通過：停下來回報，不寫輸出")

    # ------------------------------------------------------------------
    # 宿主這一側（16 個區塊 × 確認用與舊的菜單）
    # ------------------------------------------------------------------
    all_menus = [("conf", mid, codes, g) for (mid, codes), g in zip(conf, conf_g)] + \
                [("old", mid, codes, g) for (mid, codes), g in zip(old, old_g)]
    hs = {(key, mid): hostSplits(block, codes, splits[key[1]]) for key, block in coded.items() for _k, mid, codes, _g in all_menus}
    pi_full = {(key, mid): piFull(block, codes) for key, block in coded.items() for kind, mid, codes, _g in all_menus if kind == "conf"}
    print(f"宿主這一側：{len(hs)} 個（區塊 × 菜單）（{elapsed()}）")

    # 第 9 節 6（16 個區塊）
    sim_rows = []
    for key in coded:
        vals = {"all": ([], []), GROUPS[0]: ([], [])}
        d1s = []
        for kind, mid, codes, g in all_menus:
            if kind != "conf":
                continue
            info = hs[(key, mid)]
            d1s.append(float(info["D1"]["C"].mean()))
            pi = info["pi2"]["C"]
            R = len(pi)
            lone = pi[np.arange(R), info["lone"]]
            pair = np.array([np.nanmean(np.delete(pi[r], info["lone"][r])) if np.isfinite(np.delete(pi[r], info["lone"][r])).any()
                             else np.nan for r in range(R)])
            for scope in (("all",) + ((GROUPS[0],) if g["group"] == GROUPS[0] else ())):
                vals[scope][0].extend(lone[np.isfinite(lone)].tolist())
                vals[scope][1].extend(pair[np.isfinite(pair)].tolist())
        sim_rows.append({"model": key[0], "dataset": key[1], "D1_C_conf": float(np.mean(d1s)),
                         "pi_lone_all": float(np.mean(vals["all"][0])), "pi_pair_all": float(np.mean(vals["all"][1])),
                         "pi_lone_clear": float(np.mean(vals[GROUPS[0]][0])), "pi_pair_clear": float(np.mean(vals[GROUPS[0]][1]))})
    sim = pd.DataFrame(sim_rows)

    # 第 9 節 4：用其他 15 個區塊的 π 平均挑
    pick4 = {}
    for key in weak_keys:
        for kind, mid, codes, g in all_menus:
            if kind == "conf":
                others = np.array([pi_full[(k2, mid)] for k2 in coded if k2 != key])
                pick4[(key, mid)] = int(np.argmax(np.nanmean(others, axis=0)))     # 相同時取平手順序在前（argmax 取第一個）

    # ------------------------------------------------------------------
    # 替換這一側（8 個弱模型區塊 × 2 個供體 × 菜單）
    # ------------------------------------------------------------------
    set45 = {mid for (mid, codes), g in zip(conf, conf_g) if g["lone"] not in TRANSLATED}
    set25 = {mid for (mid, codes), g in zip(conf, conf_g) if g["lone"] in ("W1", "W2")}
    block_rows, menu_rows, sub_rows = [], [], []
    menu_sums = defaultdict(lambda: np.zeros(4))     # 第 9 節 7：mid -> [pair Σnum, pair Σden, lone Σnum, lone Σden]
    for key in weak_keys:
        h, d = key
        gs = GroupSums()
        hit2 = {"max": [0.0, 0], "min": [0.0, 0]}
        hit4 = {"max": [], "min": []}
        excl, total = Counter(), Counter()
        use5 = Counter()
        for kind, mid, codes, g in all_menus:
            info = hs[(key, mid)]
            R = len(info["lone"])
            pick, pickA = onehot(info["pstar"]["C"]), onehot(info["pstar"]["A"])
            lone, low, high, simf = onehot(info["lone"]), onehot(info["lowest"]), onehot(info["highest"]), onehot(info["simfirst"])
            group = g["group"]
            per_path = np.zeros((3, 2))
            mrow = defaultdict(float)
            for donor in DONORS:
                ctx = HostDonorK(coded[key], coded[(donor, d)], splits[d])
                s = substitutionSplits(ctx, codes)
                inc = s["included"]
                excl[kind] += int((~inc).sum())
                total[kind] += R
                numC, numA, den = s["num"]["C"], s["num"]["A"], s["den"]
                if kind == "conf":
                    gs.add("J1_pick", numC, den, pick, inc); gs.add("J1_other", numC, den, ~pick, inc)
                    gs.add("J1A_pick", numA, den, pickA, inc); gs.add("J1A_other", numA, den, ~pickA, inc)
                    gs.add(f"E1_{group}_pick", numC, den, pick, inc); gs.add(f"E1_{group}_other", numC, den, ~pick, inc)
                    gs.add(f"E2_{group}_pair", numC, den, ~lone, inc); gs.add(f"E2_{group}_lone", numC, den, lone, inc)
                    gs.add("E2_all_pair", numC, den, ~lone, inc); gs.add("E2_all_lone", numC, den, lone, inc)
                    if group == GROUPS[0]:
                        gs.add("J2A_pair", numA, den, ~lone, inc); gs.add("J2A_lone", numA, den, lone, inc)
                        rows_a = inc & (info["lone"] != info["lowest"])
                        gs.add("5a_pair", numC, den, ~lone, rows_a); gs.add("5a_lone", numC, den, lone, rows_a)
                        gs.add("5b_other", numC, den, ~low, rows_a); gs.add("5b_low", numC, den, low, rows_a)
                        use5["used"] += int(rows_a.sum())
                        use5["total"] += int(inc.sum())
                    for name, mask in (("high", high), ("low", low), ("lone", lone), ("simf", simf)):
                        gs.add(f"alt_{name}_pick", numC, den, mask, inc); gs.add(f"alt_{name}_other", numC, den, ~mask, inc)
                    fixed4 = onehot(np.full(R, pick4[(key, mid)]))
                    gs.add("x4_pick", numC, den, fixed4, inc); gs.add("x4_other", numC, den, ~fixed4, inc)
                    if mid in set45:
                        gs.add("E2_45_pair", numC, den, ~lone, inc); gs.add("E2_45_lone", numC, den, lone, inc)
                    if mid in set25:
                        gs.add("E2_25_pair", numC, den, ~lone, inc); gs.add("E2_25_lone", numC, den, lone, inc)
                    for which in ("max", "min"):
                        t, n = hitCredit(s["num_int"]["C"], s["den_int"], info["pstar"]["C"], inc, lowest=which == "min")
                        hit2[which][0] += t
                        hit2[which][1] += n
                    for j in range(3):
                        per_path[j] += [numC[inc, j].sum(), den[inc, j].sum()]
                    lm = lone & inc[:, None]
                    pm = ~lone & inc[:, None]
                    menu_sums[mid] += [numC[pm].sum(), den[pm].sum(), numC[lm].sum(), den[lm].sum()]
                else:
                    gs.add("old_E1_pick", numC, den, pick, inc); gs.add("old_E1_other", numC, den, ~pick, inc)
                    gs.add(f"old_E2_{group}_pair", numC, den, ~lone, inc); gs.add(f"old_E2_{group}_lone", numC, den, lone, inc)
                for name, mask in (("pick", pick), ("pair", ~lone), ("lone", lone)):
                    m = mask & inc[:, None]
                    mrow[f"{name}_num_C"] += float(numC[m].sum())
                    mrow[f"{name}_den"] += float(den[m].sum())
                mrow["other_num_C"] += float(numC[~pick & inc[:, None]].sum())
                mrow["other_den"] += float(den[~pick & inc[:, None]].sum())
                mrow["excluded_splits"] += int((~inc).sum())
                for j, c in enumerate(codes):
                    sub_rows.append({"model": h, "dataset": d, "menu_set": kind, "menu": mid, "codes": " ".join(codes), "donor": donor,
                                     "path": c, "n_valid_splits": int(s["valid"][:, j].sum()), "n_included_splits": int(inc.sum()),
                                     "sum_num_C": float(numC[inc, j].sum()), "sum_num_A": float(numA[inc, j].sum()),
                                     "sum_den": float(den[inc, j].sum()), "n_pstar_C": int((info["pstar"]["C"][inc] == j).sum()),
                                     "n_pstar_A": int((info["pstar"]["A"][inc] == j).sum()), "n_lone": int((info["lone"][inc] == j).sum())})
            if kind == "conf" and (per_path[:, 1] > 0).all():
                eff = 10 * per_path[:, 0] / per_path[:, 1]
                for which, target in (("max", eff.max()), ("min", eff.min())):
                    tied = np.flatnonzero(np.abs(eff - target) <= 1e-12 * max(1.0, abs(target)))
                    hit4[which].append((1 / len(tied)) if pick4[(key, mid)] in tied else 0.0)
            menu_rows.append({"model": h, "dataset": d, "menu_set": kind, "menu": mid, "codes": " ".join(codes), "group": group,
                              "lone_table": g["lone"], "degree_pp": 100 * g["degree"],
                              "excluded_share": mrow["excluded_splits"] / (len(DONORS) * R), **{k: v for k, v in mrow.items()
                                                                                                if k != "excluded_splits"}})
        row = {"model": h, "dataset": d,
               "excluded_share_conf": excl["conf"] / total["conf"], "excluded_share_old": excl["old"] / total["old"],
               "E1": gs.diff("J1_pick", "J1_other"), "E2": gs.diff(f"E2_{GROUPS[0]}_pair", f"E2_{GROUPS[0]}_lone"),
               "E1_pick_effect": gs.effect("J1_pick"), "E1_other_effect": gs.effect("J1_other"),
               "E2_pair_effect": gs.effect(f"E2_{GROUPS[0]}_pair"), "E2_lone_effect": gs.effect(f"E2_{GROUPS[0]}_lone"),
               "E1_A": gs.diff("J1A_pick", "J1A_other"), "E2_A": gs.diff("J2A_pair", "J2A_lone"),
               **{f"E1_{g}": gs.diff(f"E1_{g}_pick", f"E1_{g}_other") for g in GROUPS},
               **{f"E2_{g}": gs.diff(f"E2_{g}_pair", f"E2_{g}_lone") for g in GROUPS},
               "E2_all": gs.diff("E2_all_pair", "E2_all_lone"),
               "hit2_max": hit2["max"][0] / hit2["max"][1], "hit2_min": hit2["min"][0] / hit2["min"][1],
               **{f"E1_alt_{n}": gs.diff(f"alt_{n}_pick", f"alt_{n}_other") for n in ("high", "low", "lone", "simf")},
               "E1_x4": gs.diff("x4_pick", "x4_other"), "hit4_max": float(np.mean(hit4["max"])), "hit4_min": float(np.mean(hit4["min"])),
               "hit4_menus": len(hit4["max"]),
               "E2_5a": gs.diff("5a_pair", "5a_lone"), "E_5b": gs.diff("5b_other", "5b_low"),
               "share_5": use5["used"] / use5["total"] if use5["total"] else float("nan"), "n_5": use5["used"],
               "old_E1": gs.diff("old_E1_pick", "old_E1_other"),
               **{f"old_E2_{g}": (gs.diff(f"old_E2_{g}_pair", f"old_E2_{g}_lone") if f"old_E2_{g}_pair" in gs.num else float("nan"))
                  for g in GROUPS},
               "pick_vote_pp": 100 * gs.num["J1_pick"] / gs.n["J1_pick"], "pick_path_pp": 100 * gs.den["J1_pick"] / gs.n["J1_pick"],
               "other_vote_pp": 100 * gs.num["J1_other"] / gs.n["J1_other"], "other_path_pp": 100 * gs.den["J1_other"] / gs.n["J1_other"],
               "E2_45": gs.diff("E2_45_pair", "E2_45_lone"), "E2_25": gs.diff("E2_25_pair", "E2_25_lone")}
        if any(gs.den.get(name, 0) <= 0 for name in ("J1_pick", "J1_other", f"E2_{GROUPS[0]}_pair", f"E2_{GROUPS[0]}_lone")):
            raise SystemExit(f"❌ §5：{h} | {d} 在判定中沒有可用的（菜單、供體、切分），停下來回報")
        block_rows.append(row)
        print(f"  {h:10s} {d:14s} E1 {row['E1']:+.3f}  E2 {row['E2']:+.3f}（{elapsed()}）")
    blocks = pd.DataFrame(block_rows)
    menu_blocks = pd.DataFrame(menu_rows)
    subs = pd.DataFrame(sub_rows)

    # 第 9 節 7
    conf_menus = pd.DataFrame([{"menu": mid, "codes": " ".join(codes), "group": g["group"], "lone_table": g["lone"],
                                "degree_pp": 100 * g["degree"],
                                "E2_menu": 10 * (menu_sums[mid][0] / menu_sums[mid][1] - menu_sums[mid][2] / menu_sums[mid][3])}
                               for (mid, codes), g in zip(conf, conf_g)])
    rho = float(spearmanr(conf_menus.degree_pp, conf_menus.E2_menu).statistic)

    os.makedirs(args.out_dir, exist_ok=True)
    blocks_out = sim.merge(blocks, on=["model", "dataset"], how="left")
    blocks_out.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq3gs_blocks.csv"), index=False)
    menu_blocks.merge(conf_menus[["menu", "E2_menu"]], on="menu", how="left").assign(criteria_sha256=criteria_sha).to_csv(
        os.path.join(args.out_dir, "rq3gs_menu_blocks.csv"), index=False)
    subs.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq3gs_substitutions.csv"), index=False)

    S = lambda col: summarize(blocks[col].to_numpy())
    group_stats = {(n, g): S(f"{n}_{g}") for n in ("E1", "E2") for g in GROUPS}
    plotGroups(group_stats, Counter(groups), os.path.join(args.out_dir, "fig_a_groups"), criteria_sha)
    plotScatter(conf_menus, rho, os.path.join(args.out_dir, "fig_b_degree_vs_e2"), criteria_sha)

    # ------------------------------------------------------------------
    # 報告
    # ------------------------------------------------------------------
    j1, j2 = S("E1"), S("E2")
    state1, state2 = fourState(j1, THRESHOLD), fourState(j2, THRESHOLD)
    out = ["# RQ3-GS：在沒看過的三條組合上，確認挑法有沒有挑對", "",
           f"判定標準 `{criteria}`，sha256 `{criteria_sha}`；確認：{confirmed}。"
           f"產生時間 {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}；程式 `scripts/analysis_rq3gs/path_improve_gs.py`"
           "（逐切分計算在 `Analysis/pathImproveGS.py`，沿用 `Analysis/pathImprove.py` 與 `Analysis/pathImproveK.py`）。"
           "不呼叫 API，不重跑任何東西，不修改現有檔案。", "",
           "效果的單位：那條 path 每進步 10 個百分點，投票多幾個百分點（第 5 節）。E1、E2 是兩群效果的差，門檻 0.5 用同樣的單位。", ""]
    out += ["## (1) 讀了哪些檔案", "",
            f"- `{args.armdir}/{{模型}}/{{資料集}}/`：4 個模型 × 4 個資料集，經 `Analysis.menuVote.loadPathBlock` 載入（與 RQ3-G 相同）；"
            "欄位 `item_id`、`gold`、`parsed_answer`、`parse_ok`。`CellData` 對 `result/aggregations/` 只列出檔名。",
            f"- `{args.rq3gk_blocks}`：§10.1 的核對（K = 3 的 `D1_C`、`E1_pp`）。",
            f"- `{criteria}` 第 4 節的表：§10.2 的核對。`rq3g_criteria.md`、`rq3gk_criteria.md`：只引用定義。",
            "- §10.4 的手算另外用 `Analysis.menuVote.loadPathBlock` 的字串答案、純迴圈重算（不用 `pathImproveK` / `pathImproveGS`）。", ""]

    out += ["## (2) 開跑前的檢查", "",
            f"1. 重現 RQ3-GK：K = 3 的 D1（規則 C）{shown['D1_C'][0]}pp（{shown['D1_C'][1]} 到 {shown['D1_C'][2]}），"
            f"{shown['D1_C'][3]} / 16；K = 3 的判定一 {shown['E1_pp'][0]}pp（{shown['E1_pp'][1]} 到 {shown['E1_pp'][2]}），"
            f"{shown['E1_pp'][3]} / 8。逐區塊和 `rq3gk_k_blocks.csv` 的最大差 {diff1:.1e}（門檻 1e-9）→ 通過。",
            f"2. 確認用的菜單 187 份，和 K3-01 到 K3-30、M3L、M3S、M3P 沒有重複；重算的分組（相像的一對、落單的 path、落單程度、組別）"
            "與判定標準檔第 4 節的表逐份相同 → 通過。",
            f"3. 換成自己：{self_count} 個替換（8 個區塊 × 187 份 × 3 條），分子、分母每次切分都是 0，而且都不是有效切分 → 通過。",
            f"4. 手算（{label(HAND[0], HAND[1])}、{HAND[2]}：{' '.join(hand_codes)}，兩個供體）：最大差 {diff4:.1e}（門檻 1e-9）→ 通過。"
            f"挑出的效果：程式 {mine_pick:.6f}、手算 {hand['pick_effect']:.6f}；其他的效果：程式 {mine_other:.6f}、手算 {hand['other_effect']:.6f}。", "",
            md(pd.DataFrame(hand_rows), text=True), ""]

    excl_s = summarize(100 * blocks.excluded_share_conf)
    per1 = blocks[["model", "dataset", "E1", "E1_pick_effect", "E1_other_effect", "excluded_share_conf"]].copy()
    per1["model"] = per1.model.map(MODEL_LABELS)
    per1["excluded_share_conf"] = 100 * per1.excluded_share_conf
    out += ["## (3) 判定一：挑出來的那條，真的改進時比較有效嗎（8 個弱模型區塊，187 份，規則 C）", "",
            md(pd.DataFrame([ciRow("判定一：E1 = 挑出的效果 − 其他的效果", j1, True)]), text=True), "",
            f"門檻 {THRESHOLD}。**{state1}** → {READINGS_1[state1]}", "",
            md(per1.rename(columns={"model": "宿主", "dataset": "資料集", "E1_pick_effect": "挑出的效果", "E1_other_effect": "其他的效果",
                                    "excluded_share_conf": "被排除的（菜單、供體、切分）%"})), "",
            f"被排除的（菜單、供體、切分）比例：8 個區塊平均 {excl_s['mean']:.1f}%。", ""]

    per2 = blocks[["model", "dataset", "E2", "E2_pair_effect", "E2_lone_effect"]].copy()
    per2["model"] = per2.model.map(MODEL_LABELS)
    out += ["## (4) 判定二：最不像的那一條，真的改進時最沒效嗎（8 個弱模型區塊，「明顯落單」組 62 份，規則 C）", "",
            "限制（判定標準檔第 7 節）：「明顯落單」組的落單 path 全部是翻譯的語言 path（JA、RU、ZH、ES），這項判定分不開"
            "「和其他條不像」「是翻譯的」「正確率較低」三種原因。", "",
            md(pd.DataFrame([ciRow("判定二：E2 = 相像那一對的效果 − 落單那條的效果", j2, True)]), text=True), "",
            f"門檻 {THRESHOLD}。**{state2}** → {READINGS_2[state2]}", "",
            md(per2.rename(columns={"model": "宿主", "dataset": "資料集", "E2_pair_effect": "相像那一對的效果",
                                    "E2_lone_effect": "落單那條的效果"})), ""]

    out += ["## (5) 只報告的量（不參與判定）", "", "### 9.1 分組", "",
            md(pd.DataFrame([ciRow(f"E1，{g}", S(f"E1_{g}")) for g in GROUPS]
                            + [ciRow(f"E2，{g}", S(f"E2_{g}")) for g in GROUPS[1:]] + [ciRow("E2，全部 187 份", S("E2_all"))]), text=True), "",
            "![三組菜單的 E1 與 E2](fig_a_groups.png)", ""]
    h2 = blocks[["model", "dataset", "hit2_max", "hit2_min"]].copy()
    h2["model"] = h2.model.map(MODEL_LABELS)
    h2[["hit2_max", "hit2_min"]] *= 100
    out += ["### 9.2 命中率（逐切分）", "",
            "p\\* 剛好是評分半真實效果最高（最低）那條的比例，%；亂挑是 33.3%。註：逐切分的真實效果雜訊較大，這個命中率不能和"
            "事後分析的 53% 直接比；可比的是第 4 項。", "",
            md(h2.rename(columns={"model": "宿主", "dataset": "資料集", "hit2_max": "最高", "hit2_min": "最低"}), ".1f"), "",
            f"8 個區塊的平均：最高 {100 * blocks.hit2_max.mean():.1f}%、最低 {100 * blocks.hit2_min.mean():.1f}%。", ""]
    out += ["### 9.3 換一種挑法的 E1", "",
            md(pd.DataFrame([ciRow("判定一的 p*（規則 C 的 gain）", j1), ciRow("選擇半正確率最高", S("E1_alt_high")),
                             ciRow("選擇半正確率最低", S("E1_alt_low")), ciRow("落單的 path", S("E1_alt_lone")),
                             ciRow("相像那一對中平手順序在前的", S("E1_alt_simf"))]), text=True), ""]
    out += ["### 9.4 用其他 15 個區塊的 π 平均來挑", "",
            md(pd.DataFrame([ciRow("E1（其他區塊的 π 挑）", S("E1_x4"))]), text=True), "",
            f"命中率在（區塊、菜單）層級：真實效果最高 {100 * blocks.hit4_max.mean():.1f}%、最低 {100 * blocks.hit4_min.mean():.1f}%"
            f"（8 個區塊的平均；每個區塊用 {int(blocks.hit4_menus.min())} 份菜單；亂挑 33.3%）。逐區塊：", "",
            md(blocks.assign(model=blocks.model.map(MODEL_LABELS), hit4_max=100 * blocks.hit4_max, hit4_min=100 * blocks.hit4_min)[
                ["model", "dataset", "hit4_max", "hit4_min"]].rename(columns={"model": "宿主", "dataset": "資料集", "hit4_max": "最高",
                                                                             "hit4_min": "最低"}), ".1f"), ""]
    has5 = blocks[blocks.n_5 > 0]
    t5 = blocks[["model", "dataset", "share_5", "n_5", "E2_5a", "E_5b"]].copy()
    t5["model"] = t5.model.map(MODEL_LABELS)
    t5["share_5"] = 100 * t5.share_5
    out += ["### 9.5 分開「不像」和「比較弱」（「明顯落單」組）", "",
            "(a) 與 (b) 用的是同一個條件（落單的 path ≠ 選擇半正確率最低的那條），所以用到的（菜單、供體、切分）相同。"
            f"有 {len(blocks) - len(has5)} 個區塊完全沒有這種情況（落單的 path 每次都是正確率最低的），這些區塊的值無定義；"
            "下面的摘要只用有資料的區塊，而且有些區塊只有很少的情況，數字要看逐區塊的表。", "",
            md(pd.DataFrame([ciRow(f"(a) 落單的不是正確率最低那條時的 E2（{len(has5)} 個區塊）", summarize(has5.E2_5a)),
                             ciRow(f"(b) 正確率最低的不是落單那條時：其他兩條 − 最低那條（{len(has5)} 個區塊）", summarize(has5.E_5b))]),
               text=True), "",
            md(t5.rename(columns={"model": "宿主", "dataset": "資料集", "share_5": "用到的比例（%）", "n_5": "用到的（菜單、供體、切分）",
                                  "E2_5a": "(a) E2", "E_5b": "(b) 其他兩條 − 最低那條"}), ".2f"), ""]
    ratio_all = sim.pi_lone_all.mean() / sim.pi_pair_all.mean()
    ratio_clear = sim.pi_lone_clear.mean() / sim.pi_pair_clear.mean()
    out += ["### 9.6 模擬這一側（16 個區塊，子集一，規則 C）", "",
            md(pd.DataFrame([ciRow("確認用的菜單上的 D1（pp）", summarize(sim.D1_C_conf))]), text=True), "",
            f"- 落單的 path 的 π ÷ 相像那一對的 π 平均：全部 187 份 {ratio_all:.3f}"
            f"（{100 * sim.pi_lone_all.mean():.2f}% ÷ {100 * sim.pi_pair_all.mean():.2f}%）；「明顯落單」組 {ratio_clear:.3f}"
            f"（{100 * sim.pi_lone_clear.mean():.2f}% ÷ {100 * sim.pi_pair_clear.mean():.2f}%）。", ""]
    out += ["### 9.7 落單程度對上菜單的 E2", "",
            f"187 份菜單（8 個區塊、2 個供體合併成一個值）的 Spearman 相關：{rho:+.3f}。逐菜單的值在 `rq3gs_menu_blocks.csv` 的 `E2_menu`。", "",
            "![落單程度對上 E2](fig_b_degree_vs_e2.png)", ""]
    out += ["### 9.8 規則 A 下的值", "",
            md(pd.DataFrame([ciRow("判定一（規則 A）：E1", S("E1_A")), ciRow("判定二（規則 A）：E2", S("E2_A"))]), text=True), ""]
    old_counts = Counter(g["group"] for g in old_g)
    out += ["### 9.9 舊的 30 份菜單（K3-01 到 K3-30）", "",
            f"依同樣的門檻分組：明顯落單 {old_counts[GROUPS[0]]} 份、中間 {old_counts[GROUPS[1]]} 份、對稱 {old_counts[GROUPS[2]]} 份。", "",
            md(pd.DataFrame([ciRow("E1（30 份）", S("old_E1"))]
                            + [ciRow(f"E2，{g}（{old_counts[g]} 份）", S(f"old_E2_{g}")) for g in GROUPS if old_counts[g] > 0]), text=True), ""]
    out += ["### 9.10 不除以分母的版本（判定一的兩群，pp）", "",
            md(pd.DataFrame([ciRow("挑出的：投票實際多幾個百分點", S("pick_vote_pp")),
                             ciRow("挑出的：那條 path 實際進步幾個百分點", S("pick_path_pp")),
                             ciRow("其他的：投票實際多幾個百分點", S("other_vote_pp")),
                             ciRow("其他的：那條 path 實際進步幾個百分點", S("other_path_pp"))]), text=True), ""]
    d45 = conf_menus[conf_menus.menu.isin(set45)].degree_pp
    d25 = conf_menus[conf_menus.menu.isin(set25)].degree_pp
    out += ["### 9.11 落單的 path 不是翻譯的語言 path 的菜單", "",
            md(pd.DataFrame([ciRow(f"E2，落單 path 不是 ES、JA、RU、ZH（{len(set45)} 份）", S("E2_45")),
                             ciRow(f"E2，其中落單 path 是 W1 或 W2（{len(set25)} 份）", S("E2_25"))]), text=True), "",
            f"落單程度的範圍：45 份 {d45.min():.3f}–{d45.max():.3f}pp；25 份 {d25.min():.3f}–{d25.max():.3f}pp"
            f"（「明顯落單」組 ≥ {100 * top_min:.3f}pp）。", ""]

    out += ["## (6) 對照讀法的結論", "",
            f"- 判定一（E1 {pp(j1['mean'])}，[{pp(j1['ci_low'])}, {pp(j1['ci_high'])}]，{j1['n_positive']}/8 為正）：**{state1}**。"
            f"{READINGS_1[state1]}",
            f"- 判定二（E2 {pp(j2['mean'])}，[{pp(j2['ci_low'])}, {pp(j2['ci_high'])}]，{j2['n_positive']}/8 為正）：**{state2}**。"
            f"{READINGS_2[state2]}",
            "- 其餘都只報告，不套四種狀態。", ""]
    Path(os.path.join(args.out_dir, "report.md")).write_text("\n".join(out), encoding="utf-8")
    print(f"判定一 E1 {pp(j1['mean'])} [{pp(j1['ci_low'])}, {pp(j1['ci_high'])}] {j1['n_positive']}/8 -> {state1}")
    print(f"判定二 E2 {pp(j2['mean'])} [{pp(j2['ci_low'])}, {pp(j2['ci_high'])}] {j2['n_positive']}/8 -> {state2}")
    print(f"-> {args.out_dir}/report.md（{elapsed()}）")


if __name__ == "__main__":
    main()
