"""
path_improve_k.py — RQ3-GK：不同的 K 下，改進一條 path 聚合會多多少？（result/analysis/rq3gk/rq3gk_criteria.md）

全程離線：不呼叫 API，不重跑任何東西，不修改現有檔案。判定標準確認前不執行。
    1. 載入 16 個區塊並編碼（沿用 Analysis/pathImprove.py 的核對），抽菜單（§3）
    2. 沒有替換的量：每個區塊 × 每份菜單（Analysis/pathImproveK.menuBlock）
    3. 開跑前的檢查（§10，任何一項不過就停，不寫輸出）：重現 RQ3-G 的 M12 與 M3L、M3S、M3P；規則 C 的核對與
       蒙地卡羅 2,000 次；菜單的清單
    4. 替換：每個 K 的每份菜單 × 8 個弱模型區塊 × 2 個供體 × 菜單內的 path（Analysis/pathImproveK.substituteMenu）
    5. 輸出（--out-dir）：rq3gk_k_blocks.csv、rq3gk_menu_blocks.csv、rq3gk_substitutions.csv、三張圖（PNG、PDF）、report.md

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq3gk/path_improve_k.py
"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
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

from Analysis.preregistration import sha256, confirmationLine
from Analysis.blockStats import summarize, fourState, FORWARD, REVERSE, EQUIVALENT, UNDETERMINED
from Analysis.menuJudgeStats import blocksWithSplits
from Analysis.menuVote import MODELS, DATASETS
from Analysis.pathImprove import M12, HOSTS, DONORS, THRESHOLD, encodeBlocks, checkVote, spearmanBySplit
from Analysis.pathImproveK import (KS, FIXED_MENUS, MAIN_RULES, Ans, drawMenus, menuBlock, monteCarloC, HostDonorK,
                                   substituteMenu)

OUT_DIR = "result/analysis/rq3gk"
CRITERIA_FILE = "rq3gk_criteria.md"
RQ3G_CELLS = "result/analysis/rq3g/rq3g_cells.csv"
MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
READINGS_1 = {
    REVERSE: "（投票較低）K 小到 3，等權投票仍用不到明顯領先的 path；12 條時吃不到，不只是因為 K 太大。",
    EQUIVALENT: "K = 3 時投票和最強單一打平；K 夠小可以補回差距。",
    FORWARD: "（投票較高）K = 3 時投票勝過最強單一；12 條時的落差主要是 K 太大，主軸要加上 K 這個條件。",
    UNDETERMINED: "只列數字。",
}
READINGS_2 = {
    FORWARD: "K = 3 時，用一半題目挑得出比較值得改的 path（提高 5 個百分點時多 0.5pp 以上）。",
    EQUIVALENT: "連 K = 3 都挑不出實質的差別。",
    REVERSE: "挑出來的反而較差，排名不穩定。",
    UNDETERMINED: "只列數字。",
}
RQ3G_M12 = {"D1": ("+0.23", "+0.17", "+0.30", 16), "D2": ("+0.23", "+0.13", "+0.33", 8), "conv_real": "0.077", "conv_sim": "0.053"}
RQ3G_M3 = {"M3L": ("+0.82", "+0.55"), "M3S": ("+0.59", "+0.64"), "M3P": ("+0.51", "+0.54")}   # (D1, D2)
MC_BLOCK, MC_MENU, MC_PATH, MC_DRAWS = ("qwen", "truthfulqa"), "M3S", "S1", 2000
TOL = 1e-9
ALT_KS = (3, 12)                    # §9.5
TIE_CHECK_MENUS = ("M12", "M3S")    # §9.4：規則 A、B、C
# 類別色（dataviz 參考調色盤的前三個，依固定順序；已用 validate_palette.js 檢查）與墨色
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6a64", "#d9d8d2", "#fcfcfb"


def parseArgs():
    parser = ArgumentParser(description="RQ3-GK: RQ3-G across menu sizes K and tie rules (rq3gk_criteria.md)")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--rq3g-cells", default=RQ3G_CELLS)
    parser.add_argument("--out-dir", default=OUT_DIR)
    return parser.parse_args()


def pp(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:+.2f}"


def ciRow(name: str, s: dict, with_state: bool = False, fmt: str = "pp") -> dict:
    f = pp if fmt == "pp" else (lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    row = {"量": name, "平均": f(s["mean"]), "SE": f"{s['se']:.3f}" if fmt != "pp" else f"{s['se']:.2f}",
           "95% 區間": f"[{f(s['ci_low'])}, {f(s['ci_high'])}]", "為正的區塊": f"{s['n_positive']}/{s['n_blocks']}"}
    if with_state:
        row["狀態"] = fourState(s, THRESHOLD)
    return row


def md(df: pd.DataFrame, floatfmt: str = ".2f", text: bool = False) -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt, disable_numparse=text)


def validMean(x, valid: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x[valid])) if valid.any() else float("nan")


def menuKey(K) -> str:
    return "M3" if K == "M3" else str(K)


# ------------------------------------------------------------------
# 替換的逐替換列與逐切分陣列
# ------------------------------------------------------------------
def substitutionRows(K, name: str, ctx: HostDonorK, results: list[dict], n: int) -> tuple[list[dict], list[dict]]:
    rows, split_data = [], []
    for res in results:
        valid = res["valid"]
        vm = lambda x: validMean(x, valid)
        row = {"K": menuKey(K), "menu": name, "host": ctx.host.model, "donor": ctx.donor.model, "dataset": ctx.host.dataset,
               "path": res["path"], "n": n, "n_subset2": int(ctx.sub.sum()), "invalid_splits": res["invalid_splits"],
               "excluded_share": res["invalid_splits"] / len(valid), "participates": res["participates"],
               "path_inc": vm(res["path_inc"]), "SB_after": vm(res["SB_after"]), "lead_H1_pp": vm(res["lead_H1_pp"]),
               "P_before": vm(res["P_before"]), "P_after": vm(res["P_after"]),
               "indep_num": vm(res["P_after"] - res["P_before"])}
        for rule, r in res["rules"].items():
            row.update({f"V_before_{rule}": vm(r["V_before"]), f"A_real_{rule}": vm(r["A_real"]), f"A_sim_{rule}": vm(r["A_sim"]),
                        f"D2_{rule}_pp": 100 * vm(r["A_real"] - r["A_sim"]),
                        f"conv_num_real_{rule}": vm(r["A_real"] - r["V_before"]),
                        f"conv_num_sim_{rule}": vm(r["A_sim"] - r["V_before"]),
                        f"VmSB_after_{rule}_pp": vm(r["VmSB_after_pp"]), f"WVmSB_after_{rule}_pp": vm(r["WVmSB_after_pp"]),
                        f"WV_after_{rule}": vm(r["WV_after"])})
            row[f"conversion_real_{rule}"] = row[f"conv_num_real_{rule}"] / row["path_inc"] if valid.any() else np.nan
            row[f"conversion_sim_{rule}"] = row[f"conv_num_sim_{rule}"] / row["path_inc"] if valid.any() else np.nan
        row["E1_pp"] = row["VmSB_after_A_pp"]
        rows.append(row)
        a = res["rules"]["A"]
        split_data.append({"use": valid & res["participates"], "pred": a["pred_H1"], "actual": a["actual_gain"],
                           "lead": res["lead_H1_pp"], "VmSB": a["VmSB_after_pp"]})
    return rows, split_data


SUB_MEAN_COLS = (["path_inc", "SB_after", "lead_H1_pp", "P_before", "P_after", "E1_pp"]
                 + [f"{key}_{rule}" for rule in MAIN_RULES for key in ("V_before", "A_real", "A_sim", "WV_after")]
                 + [f"{key}_{rule}_pp" for rule in MAIN_RULES for key in ("D2", "VmSB_after", "WVmSB_after")])


def weakBlocks(subs: pd.DataFrame, split_data: dict, K) -> tuple[list[dict], list[dict]]:
    """§5：每份菜單先對參與判定的替換平均，再對菜單平均；轉換率是全部菜單、全部參與判定的替換的總和比。"""
    menu_rows, block_rows = [], []
    for host in HOSTS:
        for dataset in DATASETS:
            g = subs[(subs.K == menuKey(K)) & (subs.host == host) & (subs.dataset == dataset)]
            part = g[g.participates]
            per_menu = []
            for name, m in g.groupby("menu", sort=False):
                mp = m[m.participates]
                row = {"K": menuKey(K), "menu": name, "model": host, "dataset": dataset, "n_subs": len(m),
                       "n_subs_participating": len(mp)}
                row.update({col: float(mp[col].mean()) if len(mp) else np.nan for col in SUB_MEAN_COLS})
                for rule in MAIN_RULES:
                    row[f"V_inc_{rule}"] = row[f"A_real_{rule}"] - row[f"V_before_{rule}"]
                menu_rows.append(row)
                if len(mp):
                    per_menu.append(row)
            used = pd.DataFrame(per_menu)
            b = {"K": menuKey(K), "model": host, "dataset": dataset, "n_menus_used": len(used),
                 "n_menus_skipped": g.menu.nunique() - len(used)}
            if len(used):
                for col in SUB_MEAN_COLS + [f"V_inc_{rule}" for rule in MAIN_RULES]:
                    b[col] = float(used[col].mean())
            for rule in MAIN_RULES:
                b[f"conversion_real_{rule}"] = float(part[f"conv_num_real_{rule}"].sum() / part.path_inc.sum())
                b[f"conversion_sim_{rule}"] = float(part[f"conv_num_sim_{rule}"].sum() / part.path_inc.sum())
            b["conversion_indep"] = float(part.indep_num.sum() / part.path_inc.sum())
            data = split_data[(menuKey(K), host, dataset)]
            use = np.array([d["use"] for d in data])
            stack = lambda key: np.array([d[key] for d in data], dtype=float)
            b["spearman6"], b["spearman6_undefined"] = spearmanBySplit(stack("pred"), stack("actual"), use)
            b["spearman8"], b["spearman8_undefined"] = spearmanBySplit(stack("lead"), stack("VmSB"), use)
            b["n_lead_positive"], b["n_lead_total"] = int((part.lead_H1_pp > 0).sum()), len(part)
            block_rows.append(b)
    return menu_rows, block_rows


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


def plotLines(series: list[tuple[str, list[dict]]], ylabel: str, title: str, path: str, sha: str):
    """橫軸 K；每條線是 16 或 8 個區塊的平均與 95% t 區間（同一個 K 的點左右錯開，區間才不會疊在一起）。"""
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(KS))
    offsets = np.linspace(-0.12, 0.12, len(series)) if len(series) > 1 else [0.0]
    for (label, stats), color, marker, dx in zip(series, SERIES, MARKERS, offsets):
        mean = np.array([s["mean"] for s in stats])
        low, high = np.array([s["ci_low"] for s in stats]), np.array([s["ci_high"] for s in stats])
        ax.vlines(x + dx, low, high, color=color, linewidth=1.6, zorder=2)
        ax.plot(x + dx, mean, color=color, linewidth=2, marker=marker, markersize=6.5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(-0.4, len(KS) - 0.4 + 1.2)
    styleAxes(ax)
    # 直接標在最右端；標籤之間至少相隔 11pt，避免互相蓋住
    fig.canvas.draw()
    ends = [(ax.transData.transform((x[-1] + dx, s[-1]["mean"]))[1], label, x[-1] + dx, s[-1]["mean"])
            for (label, s), dx in zip(series, offsets)]
    gap, last = 11 * fig.dpi / 72, None
    for y_pix, label, xe, ye in sorted(ends):
        placed = y_pix if last is None else max(y_pix, last + gap)
        last = placed
        ax.annotate(label, (xe, ye), xytext=(10, (placed - y_pix) * 72 / fig.dpi), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("K (paths in the menu)", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(series), frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", facecolor=SURFACE, metadata={"Subject" if ext == "pdf" else "Description":
                                                                   f"rq3gk_criteria.md sha256 {sha}"})
    plt.close(fig)


def main():
    args = parseArgs()
    started = time.time()
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty)")
    criteria_sha = sha256(criteria)
    elapsed = lambda: f"{time.time() - started:.0f}s"

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
    n_items = {dataset: len(raw[dataset][MODELS[0]].item_ids) for dataset in DATASETS}
    del raw
    print(f"載入與編碼：{len(coded)} 個區塊（{elapsed()}）")

    # §10.4 菜單
    menus, draws = drawMenus()
    listed = dict(re.findall(r"(K[3579]-\d\d) \| ([A-Z0-9 ]+?) \|", Path(criteria).read_text(encoding="utf-8")))
    drawn = {name: " ".join(codes) for K in (3, 5, 7, 9) for name, codes in menus[K]}
    counts = {K: len(menus[K]) for K in KS}
    distinct = {K: len({tuple(c) for _, c in menus[K]}) == len(menus[K]) for K in KS}
    ok4 = listed == drawn and all(distinct.values()) and counts == {3: 30, 5: 30, 7: 30, 9: 30, 11: 12, 12: 1}
    print(f"§10.4 菜單：{counts}，抽的次數 {draws}，與判定標準檔的表相同 {listed == drawn}，不重複 {all(distinct.values())}")
    all_menus = [(K, name, codes) for K in KS for name, codes in menus[K]] + [("M3", n, c) for n, c in FIXED_MENUS.items()]

    # 沒有替換的量（16 個區塊 × 每份菜單）
    mb = {}
    for (model, dataset), block in coded.items():
        for K, name, codes in all_menus:
            rules = ("A", "B", "C") if name in TIE_CHECK_MENUS else MAIN_RULES
            mb[(menuKey(K), name, model, dataset)] = menuBlock(block, codes, splits[dataset], rules, alternatives=K in ALT_KS)
    mismatch = sum(r["C_vs_A_mismatch"] for r in mb.values())
    print(f"沒有替換的量：{len(mb)} 個（區塊 × 菜單）；§10.3 沒有平手的題目上規則 C ≠ 規則 A：{mismatch} 題（{elapsed()}）")

    # 替換：先算 K = 12 與 M3（檢查 1、2 要用）
    contexts = {(h, g, d): HostDonorK(coded[(h, d)], coded[(g, d)], splits[d]) for d in DATASETS for h in HOSTS for g in DONORS}
    sub_rows, split_data = [], defaultdict(list)

    def runSubstitutions(K, name, codes):
        for dataset in DATASETS:
            for host in HOSTS:
                rows = [M12.index(c) for c in codes]
                host_ans = Ans.of(coded[(host, dataset)].codes[rows], coded[(host, dataset)].gold)
                for donor in DONORS:
                    ctx = contexts[(host, donor, dataset)]
                    rows_out, data = substitutionRows(K, name, ctx, substituteMenu(ctx, codes, host_ans), n_items[dataset])
                    sub_rows.extend(rows_out)
                    split_data[(menuKey(K), host, dataset)].extend(data)

    for K, name, codes in all_menus:
        if K in (12, "M3"):
            runSubstitutions(K, name, codes)
    print(f"替換（K = 12 與 M3）：{len(sub_rows)} 個（{elapsed()}）")

    # §10.1、§10.2
    ref = pd.read_csv(args.rq3g_cells)
    ref = ref[ref.path == "ALL"].set_index(["menu", "model", "dataset"])
    subs_so_far = pd.DataFrame(sub_rows)
    problems, max_diff = [], 0.0
    check_rows = []
    for name in ("M12", "M3L", "M3S", "M3P"):
        K = 12 if name == "M12" else "M3"
        d1 = {(m, d): mb[(menuKey(K), name, m, d)]["rules"]["A"]["D1"] for (m, d) in coded}
        d2, cr, cs = {}, {}, {}
        for host in HOSTS:
            for dataset in DATASETS:
                g = subs_so_far[(subs_so_far.menu == name) & (subs_so_far.host == host) & (subs_so_far.dataset == dataset)]
                part = g[g.participates]
                d2[(host, dataset)] = float(part.D2_A_pp.mean())
                cr[(host, dataset)] = float(part.conv_num_real_A.sum() / part.path_inc.sum())
                cs[(host, dataset)] = float(part.conv_num_sim_A.sum() / part.path_inc.sum())
        for key, mine in d1.items():
            max_diff = max(max_diff, abs(mine - ref.loc[(name, *key), "D1_pp"]))
        for key, mine in d2.items():
            max_diff = max(max_diff, abs(mine - ref.loc[(name, *key), "D2_pp"]))
            if name == "M12":
                max_diff = max(max_diff, abs(cr[key] - ref.loc[(name, *key), "conversion_real"]),
                               abs(cs[key] - ref.loc[(name, *key), "conversion_sim"]))
        s1, s2 = summarize(list(d1.values())), summarize(list(d2.values()))
        shown1 = (f"{s1['mean']:+.2f}", f"{s1['ci_low']:+.2f}", f"{s1['ci_high']:+.2f}", s1["n_positive"])
        shown2 = (f"{s2['mean']:+.2f}", f"{s2['ci_low']:+.2f}", f"{s2['ci_high']:+.2f}", s2["n_positive"])
        if name == "M12":
            conv = (f"{np.mean(list(cr.values())):.3f}", f"{np.mean(list(cs.values())):.3f}")
            if shown1 != RQ3G_M12["D1"] or shown2 != RQ3G_M12["D2"] or conv != (RQ3G_M12["conv_real"], RQ3G_M12["conv_sim"]):
                problems.append(f"M12 summary D1 {shown1} D2 {shown2} conversion {conv}")
            check_rows.append({"菜單": name, "D1": f"{shown1[0]}（{shown1[1]} 到 {shown1[2]}），{shown1[3]}/16",
                               "D2": f"{shown2[0]}（{shown2[1]} 到 {shown2[2]}），{shown2[3]}/8",
                               "轉換率（真實 / 隨機模型）": f"{conv[0]} / {conv[1]}"})
        else:
            if (shown1[0], shown2[0]) != RQ3G_M3[name]:
                problems.append(f"{name} D1 {shown1[0]} D2 {shown2[0]} != {RQ3G_M3[name]}")
            check_rows.append({"菜單": name, "D1": shown1[0], "D2": shown2[0], "轉換率（真實 / 隨機模型）": "—"})
    if max_diff > TOL:
        problems.append(f"per-block max difference vs rq3g_cells.csv {max_diff:.1e} > {TOL}")
    ok12 = not problems
    print(f"§10.1–§10.2 重現 RQ3-G：逐區塊最大差 {max_diff:.1e} -> {'通過' if ok12 else '不符'}")

    # §10.3 蒙地卡羅（規則 C）
    mc = monteCarloC(coded[MC_BLOCK], FIXED_MENUS[MC_MENU], MC_PATH, splits[MC_BLOCK[1]], MC_DRAWS, 0)
    mc_diff = abs(mc["mc_mean"] - mc["expected"])
    ok_mc = mc_diff <= TOL if mc["mc_se"] == 0 else mc_diff <= 3 * mc["mc_se"]
    ok3 = mismatch == 0 and ok_mc
    print(f"§10.3 蒙地卡羅：差 {mc_diff:.2e}，標準誤 {mc['mc_se']:.2e} -> {'通過' if ok_mc else '不符'}（{elapsed()}）")
    if not (ok12 and ok3 and ok4):
        for problem in problems:
            print(f"  ❌ {problem}")
        raise SystemExit("❌ 開跑前的檢查沒有全部通過：停下來回報，不寫輸出")

    # 其餘的替換
    for K, name, codes in all_menus:
        if K not in (12, "M3"):
            runSubstitutions(K, name, codes)
        if name.endswith("-30") or name == "K11-ZH":
            print(f"  替換 K = {K} 完成（{elapsed()}）")
    subs = pd.DataFrame(sub_rows)

    # ------------------------------------------------------------------
    # 區塊值
    # ------------------------------------------------------------------
    menu_rows, weak_rows = [], []
    for K in KS + ["M3"]:
        m, b = weakBlocks(subs, split_data, K)
        menu_rows += m
        weak_rows += b
    weak_menu, weak = pd.DataFrame(menu_rows), pd.DataFrame(weak_rows)
    empty = weak[(weak.K == "3") & (weak.n_menus_used == 0)]
    if len(empty):
        raise SystemExit(f"❌ §5：判定一有 {len(empty)} 個區塊在 K = 3 沒有任何可用的菜單，停下來回報，不算判定一")

    strong_menu = []
    for (Kk, name, model, dataset), r in mb.items():
        row = {"K": Kk, "menu": name, "model": model, "dataset": dataset, "K_size": r["K"], "codes": " ".join(r["menu_codes"]),
               "n_subset1": r["n_subset"], "SB": r["SB"], "tie_share": r["tie_share"], "C_vs_A_mismatch": r["C_vs_A_mismatch"]}
        for rule, res in r["rules"].items():
            for key in ("D1", "V", "VmSB_pp", "pi_mean", "gain_mean", "gain_tie_share", "pi_first", "pi_others",
                        "D1_highest", "D1_lowest", "D1_first", "pstar_is_highest", "pstar_is_lowest", "pstar_is_first"):
                if key in res:
                    row[f"{key}_{rule}"] = res[key]
            for code in M12:
                row[f"pi_{rule}_{code}"] = res["pi_paths"].get(code, np.nan)
                row[f"pstar_{rule}_{code}"] = res["pstar_share"].get(code, np.nan)
        strong_menu.append(row)
    strong_menu = pd.DataFrame(strong_menu)
    menu_blocks = strong_menu.merge(weak_menu, on=["K", "menu", "model", "dataset"], how="left")

    k_blocks = []
    num_cols = [c for c in strong_menu.columns if c not in ("K", "menu", "model", "dataset", "codes", "K_size", "n_subset1")
                and not c.startswith(("pi_A_", "pi_B_", "pi_C_", "pstar_A_", "pstar_B_", "pstar_C_"))]
    for K in KS:
        for (model, dataset) in coded:
            g = strong_menu[(strong_menu.K == str(K)) & (strong_menu.model == model) & (strong_menu.dataset == dataset)]
            row = {"K": K, "model": model, "dataset": dataset, "n_menus": len(g)}
            row.update({c: float(g[c].mean()) for c in num_cols if g[c].notna().any()})
            w = weak[(weak.K == str(K)) & (weak.model == model) & (weak.dataset == dataset)]
            if len(w):
                row.update({c: w.iloc[0][c] for c in w.columns if c not in ("K", "model", "dataset")})
            k_blocks.append(row)
    k_blocks = pd.DataFrame(k_blocks)
    weak_k = k_blocks[k_blocks.model.isin(HOSTS)]

    os.makedirs(args.out_dir, exist_ok=True)
    k_blocks.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq3gk_k_blocks.csv"), index=False)
    menu_blocks.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq3gk_menu_blocks.csv"), index=False)
    subs.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq3gk_substitutions.csv"), index=False)

    perK = lambda df, col: [summarize(df[df.K == K][col].to_numpy()) for K in KS]
    # 圖
    plotLines([("real", perK(weak_k, "conversion_real_A")), ("random model", perK(weak_k, "conversion_sim_A")),
               ("independence", perK(weak_k, "conversion_indep"))],
              "conversion rate (vote gain / path gain)", "RQ3-GK (a): conversion rate after substitution, rule A (8 weak blocks)",
              os.path.join(args.out_dir, "fig_a_conversion"), criteria_sha)
    plotLines([("V − SB", perK(weak_k, "VmSB_after_A_pp")), ("WV − SB", perK(weak_k, "WVmSB_after_A_pp"))],
              "after substitution, pp", "RQ3-GK (b): vote minus strongest single path after substitution, rule A (8 weak blocks)",
              os.path.join(args.out_dir, "fig_b_vote_minus_sb"), criteria_sha)
    plotLines([("D1, rule A", perK(k_blocks, "D1_A")), ("D1, rule C", perK(k_blocks, "D1_C"))],
              "D1, pp", "RQ3-GK (c): D1 = gain(p*) − mean gain of the others (16 blocks)",
              os.path.join(args.out_dir, "fig_c_d1"), criteria_sha)

    # ------------------------------------------------------------------
    # 判定與報告
    # ------------------------------------------------------------------
    j1 = summarize(weak_k[weak_k.K == 3].E1_pp)
    state1 = fourState(j1, THRESHOLD)
    j2 = summarize(k_blocks[k_blocks.K == 3].D1_C)
    state2 = fourState(j2, THRESHOLD)
    label = lambda m, d: f"{MODEL_LABELS[m]} · {d}"

    out = ["# RQ3-GK：不同的 K 下，改進一條 path 聚合會多多少？", "",
           f"判定標準 `{criteria}`，sha256 `{criteria_sha}`；確認：{confirmed}。"
           f"產生時間 {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}；程式 `scripts/analysis_rq3gk/path_improve_k.py`"
           "（逐切分計算在 `Analysis/pathImproveK.py`，沿用 `Analysis/pathImprove.py`）。不呼叫 API，不重跑任何東西，不修改現有檔案。", ""]
    out += ["## (1) 讀了哪些檔案", "",
            f"- `{args.armdir}/{{模型}}/{{資料集}}/`：4 個模型 × 4 個資料集，經 `Analysis.menuVote.loadPathBlock` 載入（與 RQ3-G 相同）。"
            "欄位 `item_id`、`gold`、`parsed_answer`、`parse_ok`。`CellData` 對 `result/aggregations/` 只列出檔名，不讀內容。",
            f"- `{args.rq3g_cells}`：§10.1–§10.2 的核對（M12、M3L、M3S、M3P 的 `ALL` 列：`D1_pp`、`D2_pp`、`conversion_real`、`conversion_sim`）。",
            "- `result/analysis/rq3g/rq3g_criteria.md`：沿用的定義（只引用）與四種狀態的原文。",
            "- 載入時的核對（同 RQ3-G，不符就停）：題目 id 與 gold、`compareTwoAnswer` = 字串相等、整數編碼的對錯、整數投票 = `vote`。全部通過。", ""]

    out += ["## (2) 開跑前的檢查", "",
            f"1–2. 重現 RQ3-G（規則 A）：逐區塊和 `rq3g_cells.csv` 的最大差 {max_diff:.1e}（門檻 1e-9）→ 通過。", "",
            md(pd.DataFrame(check_rows), text=True), "",
            f"3. 規則 C：所有區塊 × 所有菜單（{len(all_menus)} 份），子集一中沒有平手的題目上規則 C 的得分 ≠ 規則 A 的題數：{mismatch} → 通過。"
            f"蒙地卡羅（{label(*MC_BLOCK)}、{MC_MENU}、path {MC_PATH}、提高 5 個百分點、`default_rng(0)` 抽 {MC_DRAWS:,} 次）："
            f"期望值 {100 * mc['expected']:.4f}%、抽樣平均 {100 * mc['mc_mean']:.4f}%、標準誤 {100 * mc['mc_se']:.4f}pp、"
            f"差 {100 * mc_diff:.4f}pp" + (f"（{mc_diff / mc['mc_se']:.2f} 個標準誤）" if mc["mc_se"] > 0 else "（標準誤 0）") + " → 通過。",
            f"4. 菜單：K = 3、5、7、9 各 30 組（抽的次數 {draws[3]}、{draws[5]}、{draws[7]}、{draws[9]}），K = 11 有 12 組，K = 12 有 1 組；"
            "同一個 K 內沒有重複，與判定標準檔 §3 的表逐組相同 → 通過。完整清單見判定標準檔 §3。", ""]

    per1 = weak_k[weak_k.K == 3][["model", "dataset", "E1_pp", "n_menus_used", "n_menus_skipped"]].copy()
    per1["model"] = per1.model.map(MODEL_LABELS)
    per1[["n_menus_used", "n_menus_skipped"]] = per1[["n_menus_used", "n_menus_skipped"]].astype(int)
    out += ["## (3) 判定一：K = 3 時，替換後投票還輸給最強單一嗎（8 個弱模型區塊，規則 A，子集二）", "",
            md(pd.DataFrame([ciRow("判定一：E1 = 替換後 V − 替換後 SB（pp）", j1, True)]), text=True), "",
            f"門檻 {THRESHOLD}pp。**{state1}** → {READINGS_1[state1]}", "",
            md(per1.rename(columns={"model": "宿主", "dataset": "資料集", "E1_pp": "E1", "n_menus_used": "納入的菜單",
                                    "n_menus_skipped": "沒有可用替換的菜單"})), "",
            f"K = 3 的替換：{int((subs.K == '3').sum())} 個，參與判定 {int(subs[subs.K == '3'].participates.sum())} 個。", ""]

    per2 = k_blocks[k_blocks.K == 3][["model", "dataset", "D1_C", "D1_A"]].copy()
    per2["model"] = per2.model.map(MODEL_LABELS)
    out += ["## (4) 判定二：K = 3 時挑得出比較值得改的 path 嗎（16 個區塊，規則 C，子集一）", "",
            md(pd.DataFrame([ciRow("判定二：D1（規則 C，pp）", j2, True)]), text=True), "",
            f"門檻 {THRESHOLD}pp。**{state2}** → {READINGS_2[state2]}", "",
            md(per2.rename(columns={"model": "模型", "dataset": "資料集", "D1_C": "D1（規則 C）", "D1_A": "（對照）D1（規則 A）"})), ""]

    out += ["## (5) 只報告的量（不參與判定）", ""]

    def kTable(df: pd.DataFrame, items: list[tuple[str, str]], fmt: str = "pp") -> str:
        rows = []
        for title, col in items:
            for K in KS:
                s = summarize(df[df.K == K][col].to_numpy())
                f = {"pp": lambda v: f"{v:+.2f}", "acc": lambda v: f"{100 * v:.2f}", "plain": lambda v: f"{v:.2f}",
                     "ratio": lambda v: f"{v:.3f}"}[fmt]
                rows.append({"量": title, "K": K, "平均": f(s["mean"]), "95% 區間": f"[{f(s['ci_low'])}, {f(s['ci_high'])}]",
                             "為正的區塊": f"{s['n_positive']}/{s['n_blocks']}" if fmt in ("pp", "ratio") else "—"})
        return md(pd.DataFrame(rows), text=True)

    out += ["### 9.1 每個 K 的曲線", "", "16 個區塊（子集一，沒有替換；π 為 %，其餘 pp；V、SB 為 %）：", "",
            kTable(k_blocks.assign(**{f"pi100_{r}": 100 * k_blocks[f"pi_mean_{r}"] for r in MAIN_RULES}),
                   [(f"π 的平均（%，規則 {r}）", f"pi100_{r}") for r in MAIN_RULES], fmt="plain"), "",
            kTable(k_blocks, [(f"gain 的平均（規則 {r}）", f"gain_mean_{r}") for r in MAIN_RULES]
                   + [(f"D1（規則 {r}）", f"D1_{r}") for r in MAIN_RULES]
                   + [(f"V − SB（規則 {r}）", f"VmSB_pp_{r}") for r in MAIN_RULES]), "",
            kTable(k_blocks, [(f"V（規則 {r}）", f"V_{r}") for r in MAIN_RULES] + [("SB", "SB")], fmt="acc"), "",
            "8 個弱模型區塊（子集二，替換；pp 或比值）：", "",
            kTable(weak_k.assign(path_inc_pp=100 * weak_k.path_inc, **{f"V_inc_pp_{r}": 100 * weak_k[f"V_inc_{r}"] for r in MAIN_RULES}),
                   [("那條 path 增加的正確率", "path_inc_pp")]
                   + [(f"V 增加的正確率（規則 {r}）", f"V_inc_pp_{r}") for r in MAIN_RULES]
                   + [(f"D2（規則 {r}）", f"D2_{r}_pp") for r in MAIN_RULES]
                   + [(f"替換後 V − SB（規則 {r}）", f"VmSB_after_{r}_pp") for r in MAIN_RULES]
                   + [(f"替換後 WV − SB（規則 {r}）", f"WVmSB_after_{r}_pp") for r in MAIN_RULES]), "",
            kTable(weak_k, [(f"轉換率，真實（規則 {r}）", f"conversion_real_{r}") for r in MAIN_RULES]
                   + [(f"轉換率，隨機模型（規則 {r}）", f"conversion_sim_{r}") for r in MAIN_RULES], fmt="ratio"), "",
            "![轉換率](fig_a_conversion.png)", "", "![替換後的 V − SB 與 WV − SB](fig_b_vote_minus_sb.png)", "",
            "![D1](fig_c_d1.png)", ""]

    out += ["### 9.2 獨立假設下的對照（8 個弱模型區塊，子集二，評分半；%）", "",
            kTable(weak_k, [("獨立假設，替換前", "P_before"), ("獨立假設，替換後", "P_after")]
                   + [(f"實際 V，替換前（規則 {r}）", f"V_before_{r}") for r in MAIN_RULES]
                   + [(f"實際 V，替換後（規則 {r}）", f"A_real_{r}") for r in MAIN_RULES], fmt="acc"), "",
            kTable(weak_k, [("轉換率，獨立假設", "conversion_indep")], fmt="ratio"), ""]

    out += ["### 9.3 最高票平手的題目比例（16 個區塊，子集一，評分半，%）", "",
            kTable(k_blocks, [("平手比例", "tie_share")], fmt="acc"), ""]

    tie_rows, ratio = [], {}
    for name in TIE_CHECK_MENUS:
        Kk = "12" if name == "M12" else "M3"
        g = strong_menu[(strong_menu.K == Kk) & (strong_menu.menu == name)]
        codes = FIXED_MENUS.get(name, M12)
        for rule in ("A", "B", "C"):
            pis = {c: float(g[f"pi_{rule}_{c}"].mean()) for c in codes}
            shares = {c: float(g[f"pstar_{rule}_{c}"].mean()) for c in codes}
            ratio[(name, rule)] = pis["EN"] / np.mean([pis["S1"], pis["S2"]])
            tie_rows.append({"菜單": name, "規則": rule, "量": "π（%）", **{c: f"{100 * pis[c]:.1f}" for c in codes}})
            tie_rows.append({"菜單": name, "規則": rule, "量": "p* 比例（%）", **{c: f"{100 * shares[c]:.1f}" for c in codes}})
    rc = [ratio[(name, "C")] for name in TIE_CHECK_MENUS]
    if all(0.8 <= r <= 1.25 for r in rc):
        tie_reading = "兩份菜單的比值都在 0.8 到 1.25 之間 → EN 的高影響力來自平手規則。"
    elif all(r >= 1.5 for r in rc):
        tie_reading = "兩份菜單的比值都在 1.5 以上 → 平手規則解釋不了，只回報。"
    else:
        tie_reading = "其他情況 → 只回報。"
    first_rows = []
    for K in KS:
        g = k_blocks[k_blocks.K == K]
        first_rows.append({"K": K, **{f"規則 {r}": f"{g[f'pi_first_{r}'].mean() / g[f'pi_others_{r}'].mean():.2f}" for r in MAIN_RULES}})
    out += ["### 9.4 平手規則的核對（事後分析，16 個區塊的平均）", "",
            md(pd.DataFrame(tie_rows).fillna(""), text=True), "",
            "EN 的 π ÷ (S1 的 π + S2 的 π) / 2（16 個區塊平均後相除）：", "",
            md(pd.DataFrame([{"菜單": name, **{f"規則 {r}": f"{ratio[(name, r)]:.2f}" for r in ("A", "B", "C")}}
                             for name in TIE_CHECK_MENUS]), text=True), "",
            f"事先寫下的讀法（看規則 C）：{tie_reading}", "",
            "每個 K：菜單內平手順序第一的 path 的 π ÷ 其餘 path 的 π 的平均（分子、分母各自對 16 個區塊與菜單平均後相除）：", "",
            md(pd.DataFrame(first_rows), text=True), ""]

    alt_rows = []
    for K in ALT_KS:
        g = k_blocks[k_blocks.K == K]
        for title, col in (("判定二的 p*", "D1_C"), ("選擇半正確率最高", "D1_highest_C"), ("選擇半正確率最低", "D1_lowest_C"),
                           ("平手順序第一", "D1_first_C")):
            alt_rows.append({"K": K, **ciRow(f"D1：{title}", summarize(g[col]))})
    share_rows = [{"K": K, "p* = 正確率最高（%）": f"{100 * k_blocks[k_blocks.K == K].pstar_is_highest_C.mean():.1f}",
                   "p* = 正確率最低（%）": f"{100 * k_blocks[k_blocks.K == K].pstar_is_lowest_C.mean():.1f}",
                   "p* = 平手順序第一（%）": f"{100 * k_blocks[k_blocks.K == K].pstar_is_first_C.mean():.1f}",
                   "選擇半出現相同 gain（%）": f"{100 * k_blocks[k_blocks.K == K].gain_tie_share_C.mean():.1f}"} for K in ALT_KS]
    out += ["### 9.5 和簡單做法比（規則 C，子集一，pp）", "", md(pd.DataFrame(alt_rows), text=True), "",
            md(pd.DataFrame(share_rows), text=True), ""]

    sp_rows = []
    for K in KS:
        g = weak_k[weak_k.K == K]
        sp_rows.append({"K": K, **{label(r.model, r.dataset): f"{r.spearman6:.3f}" for r in g.itertuples()},
                        "平均": f"{g.spearman6.mean():.3f}", "無定義的切分": int(g.spearman6_undefined.sum())})
    out += ["### 9.6 預測增加量（選擇半）與實際增加量（評分半）的 Spearman 相關（規則 A）", "",
            md(pd.DataFrame(sp_rows), text=True), ""]

    out += ["### 9.7 其他 K 的判定量", "",
            kTable(weak_k, [("判定一的量：E1（規則 A）", "E1_pp")]), "",
            kTable(k_blocks, [("判定二的量：D1（規則 C）", "D1_C"), ("判定二的量在規則 A 下：D1（規則 A）", "D1_A")]), ""]

    lead_rows = []
    for K in KS:
        g = weak_k[weak_k.K == K]
        s = summarize(g.lead_H1_pp)
        lead_rows.append({"K": K, "領先幅度的區塊平均（pp）": f"{s['mean']:+.2f}", "95% 區間": f"[{s['ci_low']:+.2f}, {s['ci_high']:+.2f}]",
                          "領先幅度 > 0 的替換": f"{int(g.n_lead_positive.sum())}/{int(g.n_lead_total.sum())}"
                                             f"（{100 * g.n_lead_positive.sum() / g.n_lead_total.sum():.1f}%）",
                          "Spearman（領先 vs 替換後 V − SB）": f"{g.spearman8.mean():.3f}",
                          "無定義的切分": int(g.spearman8_undefined.sum())})
    out += ["### 9.8 領先幅度（8 個弱模型區塊，子集二，規則 A）", "", md(pd.DataFrame(lead_rows), text=True), "",
            "逐替換的值在 `rq3gk_substitutions.csv`（`lead_H1_pp`、`VmSB_after_A_pp`）；Spearman 每次切分對該 K 所有菜單中參與判定、"
            "且該次有效的替換算一次，對有定義的切分平均，再對 8 個區塊平均。", ""]

    out += ["## (6) 對照讀法的結論", "",
            f"- 判定一（E1 {pp(j1['mean'])}pp，[{pp(j1['ci_low'])}, {pp(j1['ci_high'])}]，{j1['n_positive']}/8 為正）：**{state1}**。"
            f"{READINGS_1[state1]}",
            f"- 判定二（D1，規則 C，{pp(j2['mean'])}pp，[{pp(j2['ci_low'])}, {pp(j2['ci_high'])}]，{j2['n_positive']}/16 為正）："
            f"**{state2}**。{READINGS_2[state2]}",
            f"- 第 9 節 4 的事先讀法（規則 C）：{tie_reading}",
            "- 其餘都只報告，不套四種狀態。", ""]

    Path(os.path.join(args.out_dir, "report.md")).write_text("\n".join(out), encoding="utf-8")
    print(f"判定一 E1 {pp(j1['mean'])} [{pp(j1['ci_low'])}, {pp(j1['ci_high'])}] {j1['n_positive']}/8 -> {state1}")
    print(f"判定二 D1_C {pp(j2['mean'])} [{pp(j2['ci_low'])}, {pp(j2['ci_high'])}] {j2['n_positive']}/16 -> {state2}")
    print(f"§9.4：{tie_reading}")
    print(f"-> {args.out_dir}/report.md（{elapsed()}）")


if __name__ == "__main__":
    main()
