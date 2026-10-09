"""
path_improve_gj.py — RQ3-GJ 的分析（result/analysis/rq3gj/rq3gj_criteria.md）

正式跑完之後執行。離線：不呼叫 API、不改任何 Judge 輸出、不重算預測（只讀預測檔並核對 sha256）。
    1. 核對四個步驟檔（precheck、pilot、預測 manifest、prerun）都寫於同一份判定標準之下且通過；每個計畫的題目都有紀錄
    2. 逐區塊讀 Judge 輸出：每個版本在子集二上的正確率、多數決（規則 C）、逐切分的實際與預測、票型轉換、K = 2 的量
    3. 判定一到四（第 6 節）與第 8 節 1–15 的量
    4. 輸出（--out-dir）：rq3gj_blocks.csv、rq3gj_substitutions.csv、rq3gj_transitions.csv、rq3gj_k2_blocks.csv、
       rq3gj_k2_substitutions.csv、judge_outputs/items.csv.gz、judge_outputs_k2/items_k2.csv.gz、models.csv、兩張圖、report.md

§8.8 的人工標記：先跑 `--dump` 印出抽到的 20 題全文，把標記寫進 LABELS，再跑一次。

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq3gj/path_improve_gj.py --dump
    conda run -n clreasoning python scripts/analysis_rq3gj/path_improve_gj.py
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import csv
import gzip
import io
import json
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Analysis.preregistration import sha256, confirmationLine, loadStepFile
from Analysis.blockStats import FORWARD, REVERSE, EQUIVALENT, UNDETERMINED
from Analysis.menuVote import DATASETS
from Analysis.pathImprove import HOSTS, DONORS
from Analysis.alignment import loadRecords
from Analysis.judgeSubstitution import (OUT_DIR, CRITERIA_FILE, JUDGE_DIR, JUDGE_DIR_K2, PRECHECK_FILE, PILOT_FILE, PRERUN_FILE, RUNNER,
                                        GROUP_ORDER, GROUP_MENUS, MENU_ORDER, MIDDLE_TRANSLATED, PAIRS, PRICES, callCost)
from Analysis.judgeSubstitutionPredict import MANIFEST, trainingRecords, tableBySplit
from Analysis.judgeSubstitutionStats import (THRESHOLD, SMALL_DEN, STATES, CATEGORIES, loadPredictions, processBlock, substitutionRows,
                                             originalRows, transitionRows, groupEffect, ejOf, summ, state, spearmanSplits, splitEffects,
                                             trainingErrors, k2Counts, qwenPopulation)
import run_gj_judge as runner

MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B"}
CLEAR, MIDDLE, SYMMETRIC, ENGLISH = GROUP_ORDER
N_QWEN = 20
# §8.8 的人工標記：(資料集, 菜單, 版本, item_id) -> (類別, 引用的原文, 說明)。類別：矛盾 / 一致 / 其他。讀完 --dump 的全文後填入。
LABELS: dict = {
    ("mmlu", "GS-142", "P1-gemini3.1flashlite", 13193): ("一致", "Answer 2 correctly identifies Option A as the most accurate choice", "選 2"),
    ("mmlu", "GS-003", "P2-gemini3.1flashlite", 8539): ("一致", "Answer 1 correctly identifies that Scenario 2 involves morally wrong behavior", "選 1"),
    ("mmlu", "GS-181", "S2-deepseek4.1flash", 13409): ("一致", "Therefore, the correct answer is provided in Answer 1.", "選 1"),
    ("mmlu", "GS-068", "ES-gemini3.1flashlite", 4904): ("一致", "The reasoning in Answer 2 is more directly aligned with the purpose of universal screening", "選 2"),
    ("mmlu", "GS-021", "EN-gemini3.1flashlite", 3314): ("一致", "Answer 1 correctly identifies that Calvinists had a theocratic structure", "選 1"),
    ("mathqa", "GS-142", "orig", 1279): ("一致", "Answer 3 correctly identifies that none of the given choices are accurate", "選 3"),
    ("truthfulqa", "GS-139", "P1-deepseek4.1flash", 720): ("一致", "Therefore, the correct answer is A, making Answer 2 the most accurate.", "選 2"),
    ("truthfulqa", "GS-181", "W2-gemini3.1flashlite", 237): ("一致", "The correct interpretation should consider ... the trademarked slogan for Disneyland.",
                                                             "先說 Answer 1 也有缺點，結論仍支持 Answer 1；選 1"),
    ("truthfulqa", "GS-150", "P2-gemini3.1flashlite", 583): ("一致", "Answer 2 correctly identifies that while there are no blanket bans", "選 2"),
    ("truthfulqa", "GS-020", "EN-deepseek4.1flash", 27): ("一致", "The most specific and accurate answer is Answer 1.", "選 1"),
    ("truthfulqa", "GS-125", "P2-deepseek4.1flash", 95): ("一致", "Therefore, Answer 3 is the most accurate.", "選 3"),
    ("truthfulqa", "GS-041", "EN-deepseek4.1flash", 524): ("一致", "Answer 3 correctly identifies that ... is a broader and more accurate answer", "選 3"),
    ("commonsenseqa", "GS-105", "orig", 1350): ("一致", "Therefore, Answer 3 provides a more accurate and comprehensive reasoning.", "選 3"),
    ("commonsenseqa", "GS-095", "JA-gemini3.1flashlite", 1345): ("一致", "Therefore, Answer 3 provides a more accurate and direct interpretation", "選 3"),
    ("commonsenseqa", "GS-181", "orig", 63): ("一致", "Answer 3 correctly identifies \"under rocks\" (A) as a suitable environment", "選 3"),
    ("commonsenseqa", "GS-065", "ES-deepseek4.1flash", 1929): ("一致", "Therefore, Answer 3 provides the most accurate reasoning.", "選 3"),
    ("commonsenseqa", "GS-068", "P2-deepseek4.1flash", 111): ("一致", "Answer 1 correctly eliminates the dog park (B)", "選 1"),
    ("commonsenseqa", "GS-109", "S2-gemini3.1flashlite", 1683): ("一致", "Therefore, Answer 3 provides the most accurate reasoning.", "選 3"),
    ("commonsenseqa", "GS-022", "P1-gemini3.1flashlite", 124): ("一致", "Therefore, Answer 1 provides the most comprehensive and accurate reasoning.", "選 1"),
    ("commonsenseqa", "GS-021", "P1-gemini3.1flashlite", 1567): ("矛盾", "However, Answer 2 incorrectly assumes that a department store is the most likely place",
                                                              "文字說 Answer 1、3（gym）對、Answer 2 錯，卻輸出 {\"choice\":2}"),
}
READ_J1 = {
    FORWARD: "和投票一樣，裁判之下落單的那條也最不值得改。",
    REVERSE: "裁判之下落單的那條反而比較值得改。",
    EQUIVALENT: "裁判之下改哪一條差不多；落單的劣勢是數票造成的。",
    UNDETERMINED: "只列數字。",
}
READ_J2 = {
    FORWARD: "改進一條 path，裁判多拿到的正確率比投票多。",
    EQUIVALENT: "裁判和投票拿到的差不多。",
    REVERSE: "裁判拿到的比投票少。",
    UNDETERMINED: "只列數字。",
}
READ_J4 = {
    FORWARD: "三條都是英文時，落單的那條在裁判之下也最不值得改；落單本身有作用，不只是語言。",
    EQUIVALENT: "三條都是英文時，裁判之下改哪一條差不多。",
    REVERSE: "三條都是英文時，落單的那條反而比較值得改。",
    UNDETERMINED: "只列數字；判定一仍要帶著「落單的是翻譯的 path」這個條件寫。",
}
SERIES = ["#2a78d6", "#eb6834"]
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6a64", "#d9d8d2", "#fcfcfb"


def parseArgs():
    parser = ArgumentParser(description="RQ3-GJ analysis (rq3gj_criteria.md)")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--dump", action="store_true", help="Print the §8.8 sample (full texts) for labelling, then stop")
    parser.add_argument("--allow-unlabeled", action="store_true", help="Dry runs only: write the report without the §8.8 labels")
    return parser.parse_args()


def pp(x: float) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.2f}"


def ciRow(name: str, s: dict, threshold: float | None = None) -> dict:
    row = {"量": name, "平均": pp(s["mean"]), "SE": "—" if not np.isfinite(s["se"]) else f"{s['se']:.2f}",
           "95% 區間": f"[{pp(s['ci_low'])}, {pp(s['ci_high'])}]", "為正的區塊": f"{s['n_positive']}/{s['n_blocks']}"}
    if s.get("n_dropped"):
        row["為正的區塊"] += f"（{s['n_dropped']} 個區塊分母過小）"
    if threshold is not None:
        row["狀態"] = state(s, threshold)
    return row


def md(df: pd.DataFrame, floatfmt: str = ".2f", text: bool = False) -> str:
    if text:
        df = df.fillna("")
    return df.to_markdown(index=False, floatfmt=floatfmt, disable_numparse=text)


class GzCsv:
    """逐列寫 gzip CSV（mtime 0，可重現）。"""
    def __init__(self, path: str, columns: list):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.raw = open(path, "wb")
        self.gz = gzip.GzipFile(fileobj=self.raw, mode="wb", mtime=0)
        self.text = io.TextIOWrapper(self.gz, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.text, fieldnames=columns, extrasaction="ignore")
        self.writer.writeheader()

    def __call__(self, row: dict):
        self.writer.writerow(row)

    def close(self):
        self.text.close()
        self.raw.close()


ITEM_COLUMNS = ["host", "dataset", "K", "menu", "version", "order", "donor", "slot", "item_id", "presentation_order", "donor_position",
                "choice", "chosen_arm", "final_answer", "correct", "no_choice", "out_of_range", "refused", "tokens_in", "tokens_out",
                "usage_in", "usage_out", "model_version", "called_at", "pilot", "prompt_sha256"]


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


def saveFig(fig, path: str, sha: str):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", facecolor=SURFACE, metadata={"Subject" if ext == "pdf" else "Description": f"rq3gj_criteria.md sha256 {sha}"})
    plt.close(fig)


def plotGroups(stats: dict, path: str, sha: str):
    """(a) 明顯落單與對稱兩組：相像那一對與落單那條的效果，Judge 與多數決並排（8 個區塊的平均與 95% t 區間）。"""
    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    cats = [(CLEAR, "pair"), (CLEAR, "lone"), (SYMMETRIC, "pair"), (SYMMETRIC, "lone")]
    labels = ["clear: similar pair", "clear: odd one out", "symmetric: similar pair", "symmetric: odd one out"]
    x = np.arange(len(cats))
    for (agg, color, marker, dx) in (("J", SERIES[0], "o", -0.1), ("V", SERIES[1], "s", 0.1)):
        s = [stats[(agg, g, r)] for g, r in cats]
        ax.vlines(x + dx, [v["ci_low"] for v in s], [v["ci_high"] for v in s], color=color, linewidth=1.8, zorder=2)
        ax.scatter(x + dx, [v["mean"] for v in s], s=52, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.4, zorder=3,
                   label={"J": "Judge (host judges itself)", "V": "majority vote (rule C)"}[agg])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, color=INK)
    styleAxes(ax)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.set_ylabel("aggregator pp per 10pp path gain", color=MUTED, fontsize=9)
    ax.set_title("RQ3-GJ (a): effect of improving one path, K = 3 (8 weak blocks, 95% t interval)", color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
    saveFig(fig, path, sha)


def plotPrediction(subs: pd.DataFrame, path: str, sha: str):
    """(b) 每個替換：預測多的正確率對實際多的正確率（評分半，對 200 次切分平均）。"""
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for host, color, marker in zip(HOSTS, SERIES, ("o", "s")):
        s = subs[subs.host == host]
        ax.scatter(s.gain_pred_H2, s.gain_actual_H2, s=12, color=color, marker=marker, alpha=0.55, edgecolors="none",
                   label=f"{MODEL_LABELS[host]} ({len(s)} substitutions)", zorder=3)
    lo = float(min(subs.gain_pred_H2.min(), subs.gain_actual_H2.min()))
    hi = float(max(subs.gain_pred_H2.max(), subs.gain_actual_H2.max()))
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", zorder=2, label="predicted = actual")
    styleAxes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel("predicted Judge gain, pp (ratio table)", color=MUTED, fontsize=9)
    ax.set_ylabel("actual Judge gain, pp", color=MUTED, fontsize=9)
    ax.set_title("RQ3-GJ (b): predicted vs actual gain per substitution\n(evaluation half, mean of 200 splits)", color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    saveFig(fig, path, sha)


# ------------------------------------------------------------------
def main():
    args = parseArgs()
    started = time.time()
    elapsed = lambda: f"{time.time() - started:.0f}s"
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet")
    sha = sha256(criteria)
    steps = {name: loadStepFile(os.path.join(args.out_dir, f), name, sha, RUNNER)
             for name, f in (("check", PRECHECK_FILE), ("pilot", PILOT_FILE), ("predict", MANIFEST), ("prerun", PRERUN_FILE))}
    for name, data in steps.items():
        if not data.get("passed"):
            raise SystemExit(f"❌ the `{name}` step did not pass")
    rargs = type("A", (), {"armdir": args.armdir, "aggdir": args.aggdir, "out_dir": args.out_dir})()
    ctx = runner.loadAll(rargs)
    coded, plan, splits = ctx["coded"], ctx["plan"], ctx["splits"]
    preds = loadPredictions(args.out_dir)
    print(f"載入與計畫、預測（sha256 已核對）（{elapsed()}）")

    # ------------------------------------------------------------------
    # 逐區塊
    # ------------------------------------------------------------------
    w3 = GzCsv(os.path.join(args.out_dir, JUDGE_DIR, "items.csv.gz"), ITEM_COLUMNS)
    w2 = GzCsv(os.path.join(args.out_dir, JUDGE_DIR_K2, "items_k2.csv.gz"), ITEM_COLUMNS)
    writer = lambda row: (w3 if row["K"] == 3 else w2)(row)
    subs, origs, trans, records, missing, qwen_pop, returned = [], [], [], [], [], [], set()
    split_blocks, k2 = {}, {}
    for h in HOSTS:
        for d in DATASETS:
            block = processBlock(plan, coded, ctx["gold_text"], h, d, args.out_dir, None if args.dump else writer)
            if block["missing"]:
                missing += [(h, d, *m) for m in block["missing"]]
                continue
            rows, per_split = substitutionRows(plan, coded, block, splits[d], preds)
            subs += rows
            origs += originalRows(plan, coded, block, splits[d], preds)
            trans += transitionRows(plan, coded, block)
            sp, undefined = spearmanSplits(per_split)
            split_blocks[(h, d)] = {"act": {k: v.mean() for k, v in splitEffects(plan, per_split, "act").items()},
                                    "spearman": sp, "spearman_undefined": undefined}
            k2[(h, d)] = k2Counts(plan, coded, block)
            qwen_pop += qwenPopulation(plan, coded, block)
            returned |= {(h, mv) for mv in block["model_versions"]}
            keep = ["host", "dataset", "K", "menu", "version", "order", "donor", "slot", "item_id", "choice", "correct", "no_choice",
                    "out_of_range", "refused", "tokens_in", "tokens_out", "usage_in", "usage_out", "pilot", "cell", "alldiff",
                    "slot_positions", "lone_is_donor"]
            records.append(block["records"].reindex(columns=keep))
            print(f"  {h:10s} {d:14s} {len(block['records']):7d} 筆（{elapsed()}）")
            del block
    w3.close()
    w2.close()
    if missing:
        print(f"\n⚠️ {len(missing)} 個版本還有沒寫入的題目（正式跑沒有完成）：")
        for m in missing[:20]:
            print("   ", m)
        raise SystemExit("❌ 停：先把正式跑補完（同一個 run_gj_judge.py full 指令續跑）")

    # §8.8 Qwen 的抽樣
    pop = sorted(qwen_pop, key=lambda r: r["sort"])
    picked = sorted(np.random.default_rng(0).choice(len(pop), N_QWEN, replace=False).tolist()) if len(pop) >= N_QWEN else list(range(len(pop)))
    sample = [pop[i] for i in picked]
    if args.dump:
        dumpSample(plan, sample, args)
        return
    unlabeled = [s for s in sample if (s["dataset"], s["menu"], s["version"], s["item_id"]) not in LABELS]
    if unlabeled and not args.allow_unlabeled:
        raise SystemExit(f"❌ §8.8：{len(unlabeled)} 題還沒有標記；先跑 --dump，把標記寫進 LABELS")

    subs = pd.DataFrame(subs)
    origs = pd.DataFrame(origs)
    trans = pd.DataFrame(trans)
    rec = pd.concat(records, ignore_index=True)
    subs3, subs2 = subs[subs.K == 3].copy(), subs[subs.K == 2].copy()
    weak = [(h, d) for h in HOSTS for d in DATASETS]

    # ------------------------------------------------------------------
    # 逐區塊的值
    # ------------------------------------------------------------------
    rows = []
    for h, d in weak:
        s = subs3[(subs3.host == h) & (subs3.dataset == d)]
        o = origs[(origs.host == h) & (origs.dataset == d)]
        row = {"model": h, "dataset": d}
        for group in GROUP_ORDER:
            for agg, num in (("J", "numJ"), ("V", "numV")):
                e = ejOf(s, GROUP_MENUS[group], num)
                row.update({f"E_{agg}_{group}": e["E"], f"pair_{agg}_{group}": e["pair"], f"lone_{agg}_{group}": e["lone"],
                            f"den_pair_{group}": e["den_pair"], f"den_lone_{group}": e["den_lone"]})
        for agg, num in (("J", "numJ"), ("V", "numV")):
            row[f"E_{agg}_middle6"] = ejOf(s, MIDDLE_TRANSLATED, num)["E"]
        row["D_JV"] = float((100 * (s.numJ - s.numV)).mean())
        row["D_P"] = float(s.D_P.mean())
        for donor in DONORS:
            sd = s[s.donor == donor]
            tag = donor.split(".")[0][:8]
            row[f"E_J_{CLEAR}_{tag}"] = ejOf(sd, GROUP_MENUS[CLEAR])["E"]
            row[f"E_J_{ENGLISH}_{tag}"] = ejOf(sd, GROUP_MENUS[ENGLISH])["E"]
            row[f"D_JV_{tag}"] = float((100 * (sd.numJ - sd.numV)).mean())
        row.update({"spearman": split_blocks[(h, d)]["spearman"], "spearman_undefined": split_blocks[(h, d)]["spearman_undefined"],
                    "D_P_orig": float(o.D_P.mean()), "MAE_mean": float(s.MAE.mean()), "MAE_max": float(s.MAE.max()),
                    "D_P_lodo": float(s.D_P_lodo.mean()), "MAE_lodo": float(s.MAE_lodo.mean()),
                    "D_P_cross": float(s.D_P_cross.mean()), "MAE_cross": float(s.MAE_cross.mean())})
        clear = s[s.menu.isin(GROUP_MENUS[CLEAR])]
        for role in ("pair", "lone"):
            c = clear[clear.role == role]
            row.update({f"undiv_J_{role}": 100 * c.numJ.sum() / len(c), f"undiv_V_{role}": 100 * c.numV.sum() / len(c),
                        f"undiv_path_{role}": 100 * c.den.sum() / len(c)})
        row.update({"A_J_orig": o.A_J.mean(), "A_V_orig": o.A_V.mean(), "SB_orig": o.SB.mean(), "A_J_sub": s.A_J_sub.mean(),
                    "A_V_sub": s.A_V_sub.mean(), "SB_sub": s.SB_sub.mean(), "J_minus_SB_sub": float((100 * (s.A_J_sub - s.SB_sub)).mean())})
        act = split_blocks[(h, d)]["act"]
        pe = preds["effects"]
        pe = pe[(pe.host == h) & (pe.dataset == d)]
        for group in GROUP_ORDER:
            row[f"E_J_H2_act_{group}"] = act[f"E_J_{group}"]
            row[f"E_J_H2_pred_{group}"] = float(pe[f"E_J_pred_{group}"].mean())
        row["D_JV_H2_act"], row["D_JV_H2_pred"] = act["D_JV"], float(pe.D_JV_pred.mean())
        s2 = subs2[(subs2.host == h) & (subs2.dataset == d)]
        row.update({"k2_undiv_J": float((100 * s2.numJ).mean()), "k2_effect_J": 10 * s2.numJ.sum() / s2.den.sum(),
                    "k2_effect_V": 10 * s2.numV.sum() / s2.den.sum(),
                    "k2_effect_higher": groupEffect(s2[s2.higher.astype(bool)])[0],
                    "k2_effect_lower": groupEffect(s2[~s2.higher.astype(bool)])[0]})
        for pair in PAIRS:
            p2 = s2[s2.menu == pair]
            row[f"k2_effect_J_{pair}"] = groupEffect(p2)[0]
        cnt = k2[(h, d)]["counts"]
        share = lambda key: cnt[key][1] / cnt[key][0] if key in cnt and cnt[key][0] else float("nan")
        for kind in ("donor", "host", "base"):
            for pos in (1, 2):
                row[f"k2_b_{kind}_{pos}"] = share((kind, pos))
            row[f"k2_b_{kind}_avg"] = np.nanmean([row[f"k2_b_{kind}_1"], row[f"k2_b_{kind}_2"]])
        same = k2[(h, d)]["same"]
        row["k2_c_orig"] = same["orig"][1] / same["orig"][0] if same["orig"][0] else float("nan")
        row["k2_c_sub"] = same["sub"][1] / same["sub"][0] if same["sub"][0] else float("nan")
        rows.append(row)
    blocks = pd.DataFrame(rows)

    # §8.7(c)
    train = trainingRecords(coded, ctx["gold_text"])
    tables = {j: tableBySplit(train, j, splits)[0] for j in HOSTS}
    terr = trainingErrors(train, tables, coded, ctx["answered"], splits)
    blocks = blocks.merge(terr.rename(columns={"judge": "model"}), on=["model", "dataset"], how="left")

    # 第 8 節 1：逐菜單的 E_J（8 個區塊、2 個供體合併）
    menu_rows = []
    for menu in MENU_ORDER:
        m = subs3[subs3.menu == menu]
        rowm = {"menu": menu, "groups": "、".join(g for g in GROUP_ORDER if menu in GROUP_MENUS[g]), "lone": plan.menus[menu]["lone"],
                "degree_pp": 100 * plan.menus[menu]["degree"]}
        for agg, num in (("J", "numJ"), ("V", "numV")):
            pr, lo = m[m.role == "pair"], m[m.role == "lone"]
            rowm[f"E_{agg}"] = 10 * (pr[num].sum() / pr.den.sum() - lo[num].sum() / lo.den.sum())
        menu_rows.append(rowm)
    menus_df = pd.DataFrame(menu_rows)

    # 輸出 CSV
    os.makedirs(args.out_dir, exist_ok=True)
    blocks.assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gj_blocks.csv"), index=False)
    subs3.assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gj_substitutions.csv"), index=False)
    trans.assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gj_transitions.csv"), index=False)
    tb = []
    tcat = trans[trans.kind == "category"]
    for scope, menus, role in [(f"{g}|{r}", GROUP_MENUS[g], r) for g in GROUP_ORDER for r in ("pair", "lone")] + [("all|all", None, None)]:
        t_ = tcat if menus is None else tcat[tcat.menu.isin(menus) & (tcat.role == role)]
        for (h, d), g_ in t_.groupby(["host", "dataset"], sort=False):
            for c in CATEGORIES:
                gc = g_[g_.category == c]
                tb.append({"model": h, "dataset": d, "scope": scope, "category": c, "n_subs": len(gc),
                           "judge_pp_sum": gc.judge_pp.sum(), "vote_pp_sum": gc.vote_pp.sum()})
    pd.DataFrame(tb).assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gj_transitions_blocks.csv"), index=False)
    k2_cols = ["model", "dataset"] + [c for c in blocks.columns if c.startswith("k2_")]
    blocks[k2_cols].assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gj_k2_blocks.csv"), index=False)
    subs2.assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gj_k2_substitutions.csv"), index=False)
    menus_df.assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gj_menus.csv"), index=False)
    versions = sorted(returned)
    pd.DataFrame([{"judge": h, "returned_versions": "、".join(v for hh, v in versions if hh == h),
                   "settings": "T = 0、max_tokens 8192、不傳 seed" + ("、enable_thinking False（DashScope 國際站）" if h == "qwen" else ""),
                   "price_usd_per_M_in_out": f"{PRICES[h][0]} / {PRICES[h][1]}（2026-10-07 查閱）"} for h in HOSTS]).to_csv(
        os.path.join(args.out_dir, "models.csv"), index=False)

    S = lambda col: summ(blocks[col])
    fig_stats = {(agg, g, r): S(f"{r}_{agg}_{g}") for agg in ("J", "V") for g in (CLEAR, SYMMETRIC) for r in ("pair", "lone")}
    plotGroups(fig_stats, os.path.join(args.out_dir, "fig_a_groups"), sha)
    plotPrediction(subs3, os.path.join(args.out_dir, "fig_b_prediction"), sha)

    report = buildReport(args, sha, confirmed, steps, blocks, subs3, subs2, origs, trans, rec, menus_df, sample, plan, S)
    Path(os.path.join(args.out_dir, "report.md")).write_text("\n".join(report), encoding="utf-8")
    print(f"-> {args.out_dir}/report.md（{elapsed()}）")


def dumpSample(plan, sample: list, args):
    """§8.8：印出抽到的題目：每個位置的候選（path、來自哪個模型、解析答案）、gold、Qwen 的選擇與 Judge 原始輸出全文。"""
    from File.File import File
    from Arm.ArmSpec import ArmSpec
    from Runner.paths import armPath
    for n, s in enumerate(sample, 1):
        v = next(x for x in plan.versions if (x.K, x.host, x.dataset, x.menu, x.version) == (3, "qwen", s["dataset"], s["menu"], s["version"]))
        _, recs = loadRecords(v.path(args.out_dir))
        r = recs[s["item_id"]]
        files = {a: File(armPath(args.armdir, v.donor if j == v.slot else v.host, v.dataset, ArmSpec.from_arm_id(a)))
                 for j, a in enumerate(v.arm_ids)}
        shown = []
        for k, a in enumerate(r["presentation_order"], 1):
            arm = files[a].getRecordById(s["item_id"])
            shown.append(f"Answer {k} = {a}{'（供體）' if v.slot is not None and a == v.arm_ids[v.slot] else ''}：{arm['parsed_answer']}")
        gold = files[v.arm_ids[0]].getRecordById(s["item_id"])["gold"]
        print(f"===== {n}. {s['dataset']} | {s['menu']} | {s['version']} | item {s['item_id']} | gold {gold} | "
              f"choice {r['trace'].get('choice')} -> {r['trace'].get('chosen_arm')} | final {r['final_answer']}")
        print("   " + "；".join(shown))
        print(r["trace"]["judge_output"])


# ------------------------------------------------------------------
# 報告
# ------------------------------------------------------------------
def mainSource(trans: pd.DataFrame, menus: list | None, role: str | None) -> dict:
    """§6.2「主要來自」：8 個區塊合計，四類各佔 Judge 分子總和的比例；回傳 {類別: (pp, 比例)} 與最大的類別。"""
    t = trans[trans.kind == "category"]
    if menus is not None:
        t = t[t.menu.isin(menus)]
    if role is not None:
        t = t[t.role == role]
    if menus is None:
        t = t.drop_duplicates(["host", "dataset", "menu", "version", "category"])
    sums = t.groupby("category").judge_pp.sum()
    n_subs = len(t.drop_duplicates(["host", "dataset", "menu", "version"]))
    total = sums.sum()
    shares = {c: (float(sums.get(c, 0.0)) / n_subs if n_subs else float("nan"), float(sums.get(c, 0.0) / total) if total else float("nan"))
              for c in CATEGORIES}
    top = max(CATEGORIES, key=lambda c: shares[c][1]) if total else None
    return {"shares": shares, "top": top, "total_pp": float(total) / n_subs if n_subs else float("nan"), "n_subs": n_subs}


def sourceText(src: dict, a_text: str, b_text: str) -> str:
    top = src["top"]
    parts = "；".join(f"({c}) 平均每個替換 {src['shares'][c][0]:+.3f}pp（{100 * src['shares'][c][1]:.0f}%）" for c in CATEGORIES)
    if top == "b":
        return f"主要來自 (b)：{b_text}（{parts}）"
    if top == "a":
        return f"主要來自 (a)：{a_text}（{parts}）"
    return f"佔比最大的是 ({top})：只列數字，不下 (a)、(b) 的讀法（{parts}）"


def buildReport(args, sha, confirmed, steps, blocks, subs3, subs2, origs, trans, rec, menus_df, sample, plan, S) -> list:
    out = ["# RQ3-GJ：讓模型當裁判時，該改哪一條 path？", "",
           f"判定標準 `{os.path.join(args.out_dir, CRITERIA_FILE)}`，sha256 `{sha}`；確認：{confirmed}。"
           f"產生時間 {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}；程式 `scripts/analysis_rq3gj/path_improve_gj.py`"
           "（逐區塊的計算在 `Analysis/judgeSubstitutionStats.py`；計畫在 `Analysis/judgeSubstitution.py`；預測在 "
           "`Analysis/judgeSubstitutionPredict.py`，只讀存好的預測檔）。", "",
           "判定一、判定四的單位是「那條 path 每進步 10 個百分點，聚合多幾個百分點」；判定二、判定三的單位是 pp。門檻都是 0.5。", ""]

    # (1) 讀了哪些檔案
    out += ["## (1) 讀了哪些檔案", "",
            f"- `{args.armdir}/{{模型}}/{{資料集}}/`：4 個模型的 12 條 path（`item_id`、`gold`、`parsed_answer`、`parse_ok`），經 "
            "`Analysis.menuVote.loadPathBlock` 載入；用來決定每個版本的候選答案、子集二與多數決。",
            f"- `{args.out_dir}/{JUDGE_DIR}/`、`{JUDGE_DIR_K2}/`：這次的 Judge 輸出（`final_answer`、`presentation_order`、"
            "`trace.choice`、`trace.chosen_arm`、`trace.judge_output`、`tokens_in`、`tokens_out`、`call`、`pilot`、`donor_position`）。",
            f"- `{args.out_dir}/` 的預測檔（只讀，sha256 等於 `{MANIFEST}`）、`{PRECHECK_FILE}`、`{PILOT_FILE}`、`{PRERUN_FILE}`。",
            "- `result/analysis/rq1kj/judge_outputs/{gpt4omini, qwen}/{資料集}/M3L|M3S|M3P.json`：第 8 節 7(c) 的訓練用紀錄。",
            "- `result/analysis/rq3gj/rq3gj_stage0_calls.csv`：計畫和第零階段逐列核對。", ""]

    # (2) 第零階段
    calls = pd.read_csv(os.path.join(args.out_dir, "rq3gj_stage0_calls.csv"))
    out += ["## (2) 第零階段的表", "",
            f"- K = 3 共 39 份菜單（四組各 10 份，GS-181 在兩組），每份 7 個版本；K = 2 的 3 個配對 × 5 個版本 × 2 種順序。",
            f"- 計畫的呼叫數 {int(calls.calls.sum()):,} 次（K = 3 {int(calls[calls.K == 3].calls.sum()):,}、K = 2 "
            f"{int(calls[calls.K == 2].calls.sum()):,}），第零階段估計 {calls.cost_usd.sum():.2f} 美元；被排除的替換 0 個。",
            "- 四組菜單、落單的 path、分母表見判定標準檔第 3 節與附錄 A。", ""]

    # (3) 流程核對與試跑
    pc, pi = steps["check"], steps["pilot"]
    out += ["## (3) 流程核對與試跑", "", "### 第 10 節 流程核對", "",
            md(pd.DataFrame([{"裁判": MODEL_LABELS[h], "題數": c["n"], "最終答案相同": f"{c['same_final']}（{100 * c['agreement']:.1f}%）",
                              "門檻": f"{100 * c['min_agreement']:.0f}%", "選到同一候選": f"{c['same_choice']}（{100 * c['choice_agreement']:.1f}%）",
                              "版本": "、".join(c["returned_versions"])} for h, c in pc["cells"].items()]), text=True), "",
            "兩個比例當作 Judge 重跑雜訊的參考（判定一的限制 (c)）。", "", "### 第 11 節 試跑", "",
            md(pd.DataFrame([{"段": k, "呼叫數": s["calls"], "無效": f"{s['no_choice'] + s['out_of_range'] + s['refused']}（{100 * s['invalid_rate']:.2f}%）",
                              "答案不在候選之中": f"{100 * s['out_of_range_rate']:.2f}%",
                              "input 平均 / 最大": f"{s['tokens_in_api']['mean']:.0f} / {s['tokens_in_api']['max']}",
                              "output 平均 / 最大": f"{s['tokens_out_api']['mean']:.0f} / {s['tokens_out_api']['max']}",
                              "位置": "、".join(f"{p}: {100 * v:.1f}%" for p, v in s["position_share"].items())}
                             for k, s in pi["segments"].items()]), text=True), "",
            f"依實際用量推估的總費用 {pi['projected_total_usd']:.2f} 美元（門檻 300）。試跑的紀錄併入正式結果，標記 `pilot: true`。", ""]

    # (4) 正式跑之前的檢查
    pr = steps["prerun"]["checks"]
    out += ["## (4) 正式跑之前的檢查（第 12 節）", "",
            f"1. 重現 RQ3-GS 判定二 {pr['1_rq3gs']['summary'][0]}（{pr['1_rq3gs']['summary'][1]} 到 {pr['1_rq3gs']['summary'][2]}），"
            f"{pr['1_rq3gs']['summary'][3]} / 8；逐區塊最大差 {pr['1_rq3gs']['max_block_diff']:.1e}；39 份菜單的 E2_menu 最大差 "
            f"{pr['1_rq3gs']['max_menu_diff']:.1e} → 通過。",
            f"2. 同一份菜單、同一題，所有版本的順序相同；已寫入的 {pr['2_orders']['records_checked']} 筆紀錄都用計畫的順序 → 通過。",
            "3. 預測檔在正式跑之前存檔，sha256 記在 manifest；這次分析讀檔時再核對一次 → 通過。",
            f"4. 換成自己：{pr['4_self']['substitutions']} 個，候選、要呼叫的題目、順序都和原本的版本相同 → 通過。",
            f"5. K = 2 兩種順序互為對調；{pr['5_k2_swap']['records_checked']} 筆紀錄的 prompt 用的是對的檔案與順序 → 通過。", ""]

    # (5) 判定
    j1, j2, j3, j4 = S(f"E_J_{CLEAR}"), S("D_JV"), S("D_P"), S(f"E_J_{ENGLISH}")
    st1, st2, st3, st4 = state(j1), state(j2), state(j3), state(j4)
    s7a = S("D_P_orig")
    st7a = state(s7a)
    src1 = mainSource(trans, GROUP_MENUS[CLEAR], "lone")
    src4 = mainSource(trans, GROUP_MENUS[ENGLISH], "lone")
    src2 = mainSource(trans, None, None)
    m6 = S("E_J_middle6")
    reading1 = READ_J1[st1]
    if st1 == REVERSE:
        reading1 += " 原因看票型拆解：" + sourceText(src1, "只能寫成落單那條答錯時裁判原本會被它帶錯。", "裁判會採用落單但答對的答案。")
    reading2 = READ_J2[st2]
    if st2 == FORWARD:
        reading2 += " 原因看票型拆解：" + sourceText(src2, "只能寫成多數已對時裁判仍會選錯，改進 path 補回了這部分。",
                                               "讀內容的聚合更能用到單一 path 的進步。")
    reading3 = {EQUIVALENT: "平均誤差在 0.5pp 內。預測可以代替一部分 Judge 呼叫（仍需要標準答案）。"}.get(st3, "")
    if st3 in (FORWARD, REVERSE):
        word = "低估" if st3 == FORWARD else "高估"
        if st7a == EQUIVALENT:
            reading3 = (f"預測{word}；原本的版本估得準（第 8 節 7(a) 為兩者相當）→ "
                        + ("換進強模型的推理後，裁判更會選對。" if st3 == FORWARD else "換進強模型的推理後，裁判選對的比例低於比例表的預期。"))
        elif st7a == st3:
            reading3 = f"預測{word}；原本的版本也{word}（第 8 節 7(a) 為{st7a}）→ 只能說比例表在新的菜單上不準。"
        else:
            reading3 = f"預測{word}；第 8 節 7(a) 為{st7a}，不屬於事先寫好的組合 → 只列數字。"
    if st3 == UNDETERMINED:
        reading3 = "只列數字。"
    reading4 = READ_J4[st4]
    if st4 == EQUIVALENT:
        st6 = state(m6)
        if st6 == FORWARD:
            reading4 += " 第 8 節 15：翻譯落單的 6 份明顯為正 → 判定一的差別可能來自翻譯的 path。"
        elif st6 in (EQUIVALENT, REVERSE):
            reading4 += f" 第 8 節 15：翻譯落單的 6 份為{st6}（兩者相當或更小）→ 只能寫成落單不夠明顯時沒有差別。"
        else:
            reading4 += f" 第 8 節 15：翻譯落單的 6 份為{st6} → 其他情況，只列數字。"
    if st4 == REVERSE:
        reading4 += " 原因看票型拆解：" + sourceText(src4, "只能寫成落單那條答錯時裁判原本會被它帶錯。", "裁判會採用落單但答對的答案。")

    def perBlock(cols: dict) -> str:
        t = blocks[["model", "dataset", *cols]].copy()
        t["model"] = t.model.map(MODEL_LABELS)
        return md(t.rename(columns={"model": "宿主", "dataset": "資料集", **cols}))

    out += ["## (5) 判定一、二、三、四", "",
            f"### 判定一：裁判之下，落單的那條還是最不值得改嗎（明顯落單組的 10 份）", "",
            md(pd.DataFrame([ciRow("E_J = 相像那一對的 Judge 效果 − 落單那條的 Judge 效果", j1, THRESHOLD),
                             ciRow("（對照）同樣 10 份上的多數決", S(f"E_V_{CLEAR}"))]), text=True), "",
            f"**{st1}** → {reading1}", "",
            "限制：(a) 這 10 份落單的全是翻譯的語言 path（JA 6、ZH 3、ES 1），判定分不開「落單」和「文字是另一種語言」；"
            "(b) 區間只反映 8 個區塊之間的變動，不反映抽到哪 10 份，逐菜單的值見第 8 節 1；(c) 分子含 Judge 重跑的雜訊（第 10 節）。", "",
            perBlock({f"E_J_{CLEAR}": "E_J", f"pair_J_{CLEAR}": "相像那一對", f"lone_J_{CLEAR}": "落單那條", f"E_V_{CLEAR}": "多數決的 E"}), "",
            f"### 判定二：改進一條 path，裁判拿到的比投票多嗎（四組全部的菜單）", "",
            md(pd.DataFrame([ciRow("D_JV = 替換後 Judge 多的 − 多數決多的（pp）", j2, THRESHOLD)]), text=True), "",
            f"**{st2}** → {reading2}", "", perBlock({"D_JV": "D_JV"}), "",
            f"### 判定三：簡單的預測估得準嗎（四組全部的菜單）", "",
            md(pd.DataFrame([ciRow("D_P = 評分半上替換後 Judge 的實際 − 預測（pp）", j3, THRESHOLD),
                             ciRow("（對照，第 8 節 7(a)）原本的版本的 D_P", s7a, THRESHOLD)]), text=True), "",
            f"**{st3}** → {reading3}", "", "限制：D_P 是有正負號的平均，正負誤差會互相抵銷；每個替換的平均絕對誤差見第 8 節 7(b)。", "",
            perBlock({"D_P": "D_P", "D_P_orig": "原本的版本 D_P", "MAE_mean": "平均絕對誤差"}), "",
            f"### 判定四：三條都是英文時，落單的那條在裁判之下還是最不值得改嗎（英文落單組的 10 份）", "",
            md(pd.DataFrame([ciRow("E_J（英文落單組）", j4, THRESHOLD), ciRow("（對照）同樣菜單上的多數決", S(f"E_V_{ENGLISH}"))]), text=True), "",
            f"**{st4}** → {reading4}", "",
            "限制：這一組的落單程度比明顯落單組小（平均 4.694pp 對 9.502pp）；落單的都是改寫題目的 path（W1、W2）。", "",
            perBlock({f"E_J_{ENGLISH}": "E_J", f"pair_J_{ENGLISH}": "相像那一對", f"lone_J_{ENGLISH}": "落單那條", f"E_V_{ENGLISH}": "多數決的 E"}), ""]

    # (6) 只報告的量
    out += ["## (6) 只報告的量（不參與判定）", "", "### 8.1 四組的 E_J，Judge 與多數決並排", "",
            md(pd.DataFrame([ciRow(f"{g}：{agg}", S(f"E_{a}_{g}")) for g in GROUP_ORDER for a, agg in (("J", "Judge"), ("V", "多數決"))]), text=True), "",
            "逐菜單的 E_J（8 個區塊、2 個供體合併）與每一組內菜單之間的標準差：", "",
            md(menus_df.rename(columns={"menu": "菜單", "groups": "組", "lone": "落單的 path", "degree_pp": "落單程度（pp）",
                                        "E_J": "E_J（Judge）", "E_V": "E（多數決）"})), "",
            md(pd.DataFrame([{"組": g, "菜單數": len(menus_df[menus_df.groups.str.contains(g)]),
                              "Judge 的標準差": f"{menus_df[menus_df.groups.str.contains(g)].E_J.std(ddof=1):.2f}",
                              "多數決的標準差": f"{menus_df[menus_df.groups.str.contains(g)].E_V.std(ddof=1):.2f}"} for g in GROUP_ORDER]), text=True), "",
            "![兩組菜單的效果](fig_a_groups.png)", ""]
    acc_cols = {"A_J_orig": "原本：Judge", "A_V_orig": "原本：多數決", "SB_orig": "原本：最強單一", "A_J_sub": "替換後：Judge",
                "A_V_sub": "替換後：多數決", "SB_sub": "替換後：最強單一"}
    t82 = blocks[["model", "dataset", *acc_cols]].copy()
    t82[list(acc_cols)] *= 100
    t82["model"] = t82.model.map(MODEL_LABELS)
    out += ["### 8.2 三種聚合的正確率（%，子集二全部題目，區塊內對版本平均）", "",
            md(t82.rename(columns={"model": "宿主", "dataset": "資料集", **acc_cols})), "",
            md(pd.DataFrame([ciRow("替換後 Judge − 替換後選最強單一（pp）", S("J_minus_SB_sub"))]), text=True), ""]
    r3 = rec[rec.K == 3]
    adopt = []
    for h in HOSTS:
        for t in (1, 2, 3):
            for p in (1, 2, 3):
                g = r3[(r3.host == h) & (r3.cell == (t - 1) * 3 + (p - 1))]
                row = {"裁判": MODEL_LABELS[h], "票型": t, "位置": p, "題數": len(g), "選對": f"{100 * g.correct.mean():.1f}%" if len(g) else "—"}
                if t == 2:
                    for flag, name in ((False, "落單答對的是宿主自己的"), (True, "是換進來的強模型的")):
                        gg = g[g.lone_is_donor == flag]
                        row[name] = f"{100 * gg.correct.mean():.1f}%（{len(gg)}）" if len(gg) else "—"
                adopt.append(row)
    out += ["### 8.3 實際的採用率（K = 3 全部版本，每個（版本、題目）計一次）", "",
            "票型一的位置是落單那條（錯的）；票型二、三的位置是正確答案。", "", md(pd.DataFrame(adopt).fillna(""), text=True), ""]
    pos_rows = []
    for h in HOSTS:
        g = r3[(r3.host == h) & r3.alldiff]
        valid = g[g.choice.notna()]
        pos_rows.append({"裁判": MODEL_LABELS[h], "三條都不同的題數": len(g), "沒有有效選擇": len(g) - len(valid),
                         **{f"選位置 {p}": f"{100 * (valid.choice == p).mean():.1f}%" for p in (1, 2, 3)}})
    bal = []
    for h in HOSTS:
        g = r3[r3.host == h]
        for slot in range(3):
            c = Counter(int(s[slot]) + 1 for s in g.slot_positions)
            bal.append({"裁判": MODEL_LABELS[h], "菜單內第幾條（平手順序）": slot + 1, **{f"顯示在位置 {p}": c[p] for p in (1, 2, 3)}})
    out += ["### 8.4 位置", "", md(pd.DataFrame(pos_rows), text=True), "",
            "票型二依位置的採用率見第 8 節 3。實際呼叫的題目上，菜單內每條 path 出現在各位置的次數：", "", md(pd.DataFrame(bal), text=True), ""]
    tags = {d: d.split(".")[0][:8] for d in DONORS}
    out += ["### 8.5 分開報", "", "兩個供體分開（8 個區塊）：", "",
            md(pd.DataFrame([ciRow(f"{name}，供體 {d}", S(f"{col}_{tags[d]}")) for d in DONORS
                             for name, col in (("判定一 E_J", f"E_J_{CLEAR}"), ("判定二 D_JV", "D_JV"), ("判定四 E_J", f"E_J_{ENGLISH}"))]), text=True), "",
            "兩個宿主分開（各 4 個區塊，只描述）：", ""]
    host_rows = []
    for h in HOSTS:
        b = blocks[blocks.model == h]
        for name, col in (("判定一", f"E_J_{CLEAR}"), ("判定二", "D_JV"), ("判定三", "D_P"), ("判定四", f"E_J_{ENGLISH}")):
            v = b[col].to_numpy()
            host_rows.append({"宿主": MODEL_LABELS[h], "判定": name, "平均": pp(np.nanmean(v)), "4 個值": "、".join(pp(x) for x in v),
                              "為正": f"{int((v > 0).sum())}/4"})
    out += [md(pd.DataFrame(host_rows), text=True), ""]
    out += ["### 8.6 預測的排名（逐切分的 Spearman，再對切分平均）", "",
            perBlock({"spearman": "Spearman", "spearman_undefined": "無定義的切分"}), "",
            f"8 個區塊的平均：{blocks.spearman.mean():+.3f}。", ""]
    out += ["### 8.7 預測的誤差", "",
            md(pd.DataFrame([ciRow("(a) 原本的版本的 D_P（pp）", s7a, THRESHOLD)]), text=True), "",
            "(b) 替換的版本：每個替換的平均絕對誤差（pp），逐區塊的平均與最大值：", "",
            perBlock({"MAE_mean": "平均", "MAE_max": "最大"}), "",
            "(c) 在 RQ1-KJ 的紀錄上：訓練用的題目（選擇半）與沒看過的題目（評分半）的誤差（實際 − 預測，pp）：", "",
            perBlock({"train": "訓練用：有正負號", "train_abs": "訓練用：絕對值", "unseen": "沒看過：有正負號", "unseen_abs": "沒看過：絕對值"}), ""]
    sample_rows = []
    for n, s in enumerate(sample, 1):
        lab = LABELS.get((s["dataset"], s["menu"], s["version"], s["item_id"]), ("（未標記）", "", ""))
        sample_rows.append({"#": n, "資料集": s["dataset"], "菜單": s["menu"], "版本": s["version"], "item_id": s["item_id"],
                            "標記": lab[0], "引用": lab[1], "說明": lab[2]})
    c8 = Counter(r["標記"] for r in sample_rows)
    out += ["### 8.8 Qwen 的輸出檢查", "",
            "母體：qwen 的 K = 3 紀錄中，兩票對一票、而且 Qwen 選了落單那條的（版本、題目）；依資料集、菜單、版本、item_id 排序後 "
            "`numpy.random.default_rng(0).choice(N, 20, replace=False)`。類別定義同核對四。", "",
            md(pd.DataFrame(sample_rows), text=True), "",
            f"矛盾 {c8.get('矛盾', 0)}、一致 {c8.get('一致', 0)}、其他 {c8.get('其他', 0)}。" +
            ("**矛盾 ≥ 5 題，特別標出。**" if c8.get("矛盾", 0) >= 5 else ""), ""]
    fail = []
    for h in HOSTS:
        for K in (3, 2):
            for kind in ("orig", "sub"):
                g = rec[(rec.host == h) & (rec.K == K) & ((rec.version == "orig") == (kind == "orig"))]
                if not len(g):
                    continue
                inval = g.no_choice | g.out_of_range | g.refused
                fail.append({"裁判": MODEL_LABELS[h], "K": K, "版本": {"orig": "原本", "sub": "替換"}[kind], "呼叫數": len(g),
                             "找不到選擇": int(g.no_choice.sum()), "答案不在候選之中": int(g.out_of_range.sum()), "被擋下": int(g.refused.sum()),
                             "agg_no_answer": f"{100 * inval.mean():.2f}%",
                             "input（API / 重算）": f"{g.usage_in.mean():.0f} / {g.tokens_in.mean():.0f}",
                             "output（API / 重算）": f"{g.usage_out.mean():.0f} / {g.tokens_out.mean():.0f}",
                             "費用（美元）": f"{sum(callCost(h, i, o) for i, o in zip(g.usage_in.fillna(0), g.usage_out.fillna(0))):.2f}"})
    # 被擋下的呼叫沒有用量（API 不回傳 usage），費用以 0 計
    total_cost = sum(callCost(h, i, o) for h, i, o in zip(rec.host, rec.usage_in.fillna(0), rec.usage_out.fillna(0)))
    out += ["### 8.9 失敗與成本", "", md(pd.DataFrame(fail), text=True), "",
            f"實際費用合計 {total_cost:.2f} 美元（2026-10-07 查閱的定價；含試跑，不含流程核對）。流程核對的重跑一致率：" +
            "、".join(f"{MODEL_LABELS[h]} {100 * c['agreement']:.1f}%" for h, c in steps["check"]["cells"].items()) + "。", ""]
    out += ["### 8.10 不除以分母的版本（明顯落單組的 10 份，pp）", "",
            md(pd.DataFrame([ciRow(f"{role_name}：{name}", S(f"undiv_{col}_{role}")) for role, role_name in (("pair", "相像那一對"), ("lone", "落單那條"))
                             for col, name in (("J", "Judge 實際多"), ("V", "多數決實際多"), ("path", "那條 path 實際進步"))]), text=True), ""]
    tcat = trans[trans.kind == "category"]
    trows = []
    for scope, menus, role in [(f"{g}：{rn}", GROUP_MENUS[g], r) for g in GROUP_ORDER for r, rn in (("pair", "相像那一對"), ("lone", "落單那條"))] \
            + [("全部菜單", None, None)]:
        src = mainSource(trans, menus, role)
        trows.append({"範圍": scope, "替換數": src["n_subs"], **{f"({c}) pp": f"{src['shares'][c][0]:+.3f}" for c in CATEGORIES},
                      **{f"({c}) 比例": f"{100 * src['shares'][c][1]:.0f}%" for c in CATEGORIES},
                      "Judge 分子 pp": f"{src['total_pp']:+.3f}"})
    cells = trans[trans.kind == "cell"].drop_duplicates(["host", "dataset", "menu", "version", "before", "after"]).groupby(
        ["before", "after"], as_index=False)[["n", "judge_change", "vote_change"]].sum().sort_values("n", ascending=False)
    out += ["### 8.11 分子依票型的轉換拆開", "",
            "四類：(a) 替換前票型一、替換後三條相同且答對；(b) 替換後票型二、落單答對的是換進來的 path；(c) 替換後票型一或三條相同且答對、"
            "不屬於 (a)；(d) 其餘。每個替換的 pp = 100 × 該類的 Judge 變化總和 ÷ 子集二題數；下表是 8 個區塊合計後除以替換數"
            "（= 平均每個替換），比例 = 該類佔 Judge 分子總和的比例（「主要來自」用它）。逐區塊的合計在 `rq3gj_transitions_blocks.csv`。", "",
            md(pd.DataFrame(trows), text=True), "",
            "全部菜單的（替換前, 替換後）格子（8 個區塊合計；Judge 與多數決的變化是答對題數的差）。逐區塊與逐範圍的值在 `rq3gj_transitions.csv`：", "",
            md(cells.rename(columns={"before": "替換前", "after": "替換後", "n": "題數", "judge_change": "Judge 的變化", "vote_change": "多數決的變化"}),
               ".1f"), ""]
    diff_rows = [ciRow(f"E_J，{g}：實際 − 預測", summ(blocks[f"E_J_H2_act_{g}"] - blocks[f"E_J_H2_pred_{g}"])) for g in GROUP_ORDER] + \
                [ciRow("D_JV：實際 − 預測", summ(blocks.D_JV_H2_act - blocks.D_JV_H2_pred))]
    out += ["### 8.12 預測的 E_J 與 D_JV（評分半，對 200 次切分平均）", "",
            perBlock({**{f"E_J_H2_act_{g}": f"{g} 實際" for g in GROUP_ORDER}, **{f"E_J_H2_pred_{g}": f"{g} 預測" for g in GROUP_ORDER},
                      "D_JV_H2_act": "D_JV 實際", "D_JV_H2_pred": "D_JV 預測"}), "",
            md(pd.DataFrame(diff_rows), text=True), ""]
    lodo = S("D_P_lodo")
    st_lodo = state(lodo)
    out += ["### 8.13 預測的泛化", "",
            md(pd.DataFrame([ciRow("1. 留一個資料集：D_P（pp）", lodo, THRESHOLD), ciRow("2. 換裁判（3 格表）：D_P（pp）", S("D_P_cross"))]), text=True), "",
            perBlock({"MAE_lodo": "留一個資料集：平均絕對誤差", "MAE_cross": "換裁判：平均絕對誤差"}), "",
            "讀法：" + ("兩者相當 → 可以寫「換到沒看過的資料集不用重做表」。" if st_lodo == EQUIVALENT
                       else f"{st_lodo} → 只能寫到判定三的範圍（同一批資料集內），並列出數字。"), ""]
    k2_rows = [ciRow("(a) 替換後 Judge 多拿到的正確率（pp，替換的平均）", S("k2_undiv_J")),
               ciRow("(a) Judge：path 每進步 10pp 多幾個 pp", S("k2_effect_J")),
               ciRow("(a) 多數決（規則 C）：path 每進步 10pp 多幾個 pp", S("k2_effect_V")),
               *[ciRow(f"(b) {name}，對的在位置 {p}：裁判選它的比例（%）", summ(100 * blocks[f"k2_b_{kind}_{p}"])) for kind, name in
                 (("donor", "對的是換進來的強模型 path"), ("host", "對的是宿主自己的 path"), ("base", "基準：原本的版本")) for p in (1, 2)],
               *[ciRow(f"(b) {name}：兩種順序平均（%）", summ(100 * blocks[f"k2_b_{kind}_avg"])) for kind, name in
                 (("donor", "對的是換進來的強模型 path"), ("host", "對的是宿主自己的 path"), ("base", "基準：原本的版本"))],
               ciRow("(c) 兩種順序選到同一條 path 的比例（%）：原本的版本", summ(100 * blocks.k2_c_orig)),
               ciRow("(c) 同上：替換的版本（%）", summ(100 * blocks.k2_c_sub)),
               ciRow("(d) 被換的是正確率較高的那條：效果", S("k2_effect_higher")),
               ciRow("(d) 被換的是正確率較低的那條：效果", S("k2_effect_lower"))]
    k2_text = (f"(a) Judge 的效果 {pp(S('k2_effect_J')['mean'])}；多數決固定是 5（實際算出 {S('k2_effect_V')['mean']:.4f}）。"
               "(a) 高於 5 只表示兩條答案不同時裁判選得比擲銅板好，而且替換後這種題目變多，不能單獨說明裁判認得出強模型的候選。"
               f"認不認得出看 (b)：對的是強模型 path 時裁判選它的比例（兩種順序平均）{100 * S('k2_b_donor_avg')['mean']:.1f}%，"
               f"原本版本的選對率 {100 * S('k2_b_base_avg')['mean']:.1f}%。只描述，不套四種狀態。")
    out += ["### 8.14 K = 2 的附帶量（只報告）", "", md(pd.DataFrame(k2_rows), text=True), "", k2_text, "",
            "逐配對（只描述；分母總和 < 10pp 的格子依第 5 節不算，標為「分母過小」）：", "",
            md(blocks.assign(model=blocks.model.map(MODEL_LABELS))[["model", "dataset", *[f"k2_effect_J_{p}" for p in PAIRS]]]
               .rename(columns={"model": "宿主", "dataset": "資料集", **{f"k2_effect_J_{p}": f"{p}：Judge 的效果" for p in PAIRS}})
               .map(lambda x: "分母過小" if isinstance(x, float) and not np.isfinite(x) else (f"{x:.2f}" if isinstance(x, float) else x)),
               text=True), ""]
    deg6 = np.mean([100 * plan.menus[m]["degree"] for m in MIDDLE_TRANSLATED])
    deg10 = np.mean([100 * plan.menus[m]["degree"] for m in GROUP_MENUS[ENGLISH]])
    m6J = summ(blocks.E_J_middle6)
    m6V = summ(blocks.E_V_middle6)
    out += ["### 8.15 落單程度相近的對照（中間組翻譯落單的 6 份 vs 英文落單組的 10 份）", "",
            md(pd.DataFrame([ciRow("翻譯落單 6 份：Judge 的 E_J", m6J, THRESHOLD), ciRow("翻譯落單 6 份：多數決的 E", m6V),
                             ciRow("英文落單 10 份：Judge 的 E_J", j4), ciRow("英文落單 10 份：多數決的 E", S(f"E_V_{ENGLISH}")),
                             ciRow("兩邊之差（翻譯落單 − 英文落單，Judge，逐區塊相減）", summ(blocks.E_J_middle6 - blocks[f"E_J_{ENGLISH}"]))]), text=True), "",
            f"落單程度平均：翻譯落單 6 份 {deg6:.3f}pp，英文落單 10 份 {deg10:.3f}pp。限制：只有 6 份，其中 4 份落單的是 ES、6 份都含 P2；只當線索。"
            "「明顯為正」= 這 6 份的 Judge E_J 平均 ≥ 0.5 且 95% 區間不含 0（上表的狀態欄為正向成立）。"
            "判定四讀法裡的「兩者相當或更小」= 這 6 份的 Judge E_J 為兩者相當或反向成立；這個操作定義由使用者在 2026-10-08 23:36 CST、"
            "跑分析之前確認。", ""]
    # (7) 結論
    out += ["## (7) 對照讀法的結論", "",
            f"- 判定一（E_J {pp(j1['mean'])}，[{pp(j1['ci_low'])}, {pp(j1['ci_high'])}]，{j1['n_positive']}/{j1['n_blocks']} 為正）：**{st1}**。{reading1}"
            "任何狀態都帶著「落單的是翻譯的語言 path」這個條件。",
            f"- 判定二（D_JV {pp(j2['mean'])}，[{pp(j2['ci_low'])}, {pp(j2['ci_high'])}]，{j2['n_positive']}/{j2['n_blocks']} 為正）：**{st2}**。{reading2}",
            f"- 判定三（D_P {pp(j3['mean'])}，[{pp(j3['ci_low'])}, {pp(j3['ci_high'])}]，{j3['n_positive']}/{j3['n_blocks']} 為正）：**{st3}**。{reading3}",
            f"- 判定四（E_J {pp(j4['mean'])}，[{pp(j4['ci_low'])}, {pp(j4['ci_high'])}]，{j4['n_positive']}/{j4['n_blocks']} 為正）：**{st4}**。{reading4}",
            "- 其餘都只報告，不套四種狀態（第 8 節 7(a)、13、15 的狀態只用來套事先寫好的讀法）。", ""]
    return out


if __name__ == "__main__":
    main()
