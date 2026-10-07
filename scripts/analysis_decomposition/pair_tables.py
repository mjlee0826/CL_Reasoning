"""
pair_tables.py — 主網格兩條 path 配對的分解總表與圖（整理現有結果；不呼叫 API，不重跑，不修改現有檔案，沒有判定）

    來源：result/analysis/aggregation_cells.csv（四個新模型、both_answered、Judge 與 Debate 各 12 組配對）。
    每個區塊（模型 × 資料集）、每組配對：
        d、c、m；headroom = 100·d·(c − m)（pp）；recovery = recovery_H2、recovery_blind = recovery_blind_H2；
        單一最強拿回 = headroom × recovery_blind、聚合拿回 = headroom × recovery、Excess = 兩者相減（都先逐區塊算，再平均）；
        聚合後正確率 − L:en 正確率（pp；不含 L:en 的配對沒有這個欄位，用 result/analysis/items 的逐題匯出算）。
    再對 16 個區塊（或強、弱各 8 個）取平均、SE、95% t 區間（Analysis.blockStats.summarize）。
    先核對現況文件的數字（結果 3、4、7），對不上就停、不寫輸出。

輸出（--out-dir，預設 result/analysis/decomposition/）：
    pair_table_judge.csv/.md、pair_table_debate.csv/.md、pair_table_judge_by_strength.csv/.md、cells_long.csv、
    fig_pairs_judge、fig_pairs_judge_by_strength、fig_pairs_debate（.png 300 dpi 與 .pdf）、report.md

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_decomposition/pair_tables.py
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from Analysis.blockStats import summarize

MODELS = ["gpt4omini", "qwen", "deepseek4.1flash", "gemini3.1flashlite"]
WEAK, STRONG = ["gpt4omini", "qwen"], ["deepseek4.1flash", "gemini3.1flashlite"]
DATASETS = ["mmlu", "mathqa", "truthfulqa", "commonsenseqa"]
SUBSET = "both_answered"
# 依多樣性來源分組（由上到下）
JUDGE_GROUPS = [("語言", "Language", ["EN+ZH", "EN+JA", "ZH+JA"]), ("採樣", "Resampling", ["EN+S1", "S1+S2"]),
                ("persona", "Persona", ["EN+P1", "P1+P2"]), ("改寫", "Rewrite", ["EN+W1", "W1+W2"]),
                ("簡短 CoT", "Short CoT", ["EN+R"]), ("自我修正", "Self-correction", ["EN+SR-EN", "ZH+SR-ZH"])]
DEBATE_GROUPS = [("語言", "Language", ["EN+ZH", "EN+JA", "EN+RU", "EN+ES", "ZH+JA", "ZH+RU", "ZH+ES", "JA+RU", "JA+ES", "RU+ES"]),
                 ("採樣", "Resampling", ["EN+S1"]), ("persona", "Persona", ["P1+P2"])]
NAMES = {"EN": "English", "ZH": "Chinese", "JA": "Japanese", "RU": "Russian", "ES": "Spanish", "S1": "resample", "S2": "resample",
         "P1": "expert", "P2": "skeptic", "W1": "rewrite", "W2": "rewrite", "R": "short CoT", "SR-EN": "self-correction",
         "SR-ZH": "self-correction"}
QUANTITIES = [("d", "d"), ("c", "c"), ("m", "m"), ("headroom_pp", "headroom（pp）"), ("recovery", "recovery"),
              ("recovery_blind", "recovery_blind"), ("strongest_pp", "單一最強拿回（pp）"), ("aggregation_pp", "聚合拿回（pp）"),
              ("excess_pp", "Excess（pp）"), ("vs_en_pp", "聚合 − L:en（pp）")]
# §三 的核對值（現況文件；比到文件的位數）
CHECKS = [("judge", "EN+ZH", "headroom_pp", 1, 6.4), ("judge", "EN+ZH", "recovery", 2, 0.35),
          ("judge", "EN+ZH", "recovery_blind", 2, 0.26), ("judge", "EN+ZH", "excess_pp", 2, 0.21),
          ("judge", "EN+S1", "headroom_pp", 1, 3.1), ("judge", "EN+S1", "recovery", 2, 0.25),
          ("judge", "EN+S1", "recovery_blind", 2, 0.00), ("judge", "EN+S1", "excess_pp", 2, 0.84),
          ("judge", "EN+SR-EN", "d", 3, 0.035), ("judge", "EN+S1", "d", 3, 0.081)]
SOURCE_CHECKS = {"persona": 0.24, "採樣": 0.26, "自我修正": 0.27, "改寫": 0.32, "簡短 CoT": 0.32, "語言": 0.38}
# 圖（使用者指定的顏色與白底）
BLUE, ORANGE, GRAY, INK, MUTED = "#2F7BD9", "#F0692E", "#D9D9D9", "#1f1f1d", "#5f5f5f"
AGG_LABEL = {"judge": "Recovered by aggregation (judge)", "debate": "Recovered by aggregation (debate)"}


def parseArgs():
    parser = ArgumentParser(description="Decomposition tables and figures for the 12 two-path pairs (descriptive only)")
    parser.add_argument("--cells", default="result/analysis/aggregation_cells.csv")
    parser.add_argument("--items-dir", default="result/analysis/items")
    parser.add_argument("--out-dir", default="result/analysis/decomposition")
    return parser.parse_args()


def label(pair: str) -> str:
    """圖與表上的可讀名稱，例如 EN+ZH -> English + Chinese、S1+S2 -> Resample + resample。"""
    a, b = pair.split("+", 1)
    text = f"{NAMES[a]} + {NAMES[b]}"
    return text[0].upper() + text[1:]


def groupOf(groups: list, pair: str) -> tuple[str, str]:
    return next((zh, en) for zh, en, pairs in groups if pair in pairs)


# ------------------------------------------------------------------
# 逐區塊
# ------------------------------------------------------------------
def loadCells(path: str) -> pd.DataFrame:
    cells = pd.read_csv(path)
    rows = []
    for aggregator, groups in (("judge", JUDGE_GROUPS), ("debate", DEBATE_GROUPS)):
        pairs = [p for _, _, ps in groups for p in ps]
        sel = cells[cells.model.isin(MODELS) & (cells.subset == SUBSET) & (cells.aggregator == aggregator) & cells.pair.isin(pairs)].copy()
        if len(sel) != len(pairs) * len(MODELS) * len(DATASETS):
            raise SystemExit(f"❌ {aggregator}: {len(sel)} rows in {path}, expected {len(pairs) * len(MODELS) * len(DATASETS)}")
        sel["source"] = [groupOf(groups, p)[0] for p in sel.pair]
        rows.append(sel)
    cells = pd.concat(rows, ignore_index=True)
    cells["headroom_pp"] = 100 * cells.d * (cells.c - cells.m)
    cells["recovery"] = cells.recovery_H2
    cells["recovery_blind"] = cells.recovery_blind_H2
    cells["strongest_pp"] = cells.headroom_pp * cells.recovery_blind
    cells["aggregation_pp"] = cells.headroom_pp * cells.recovery
    cells["excess_pp"] = cells.aggregation_pp - cells.strongest_pp
    return cells


def addVersusEnglish(cells: pd.DataFrame, items_dir: str) -> tuple[pd.DataFrame, float]:
    """聚合後正確率 − L:en 正確率（pp），在該配對的 both_answered 題目上。含 L:en 的配對核對 = 100·(acc_final − acc_a)。"""
    paths = pd.read_csv(os.path.join(items_dir, "paths.csv.gz"))
    aggs = pd.read_csv(os.path.join(items_dir, "aggregations.csv.gz"))
    paths, aggs = paths[paths.model.isin(MODELS)], aggs[aggs.model.isin(MODELS)]
    answered = paths.pivot_table(index=["model", "dataset", "item_id"], columns="arm_id", values="answered", aggfunc="first").astype(bool)
    correct = paths.pivot_table(index=["model", "dataset", "item_id"], columns="arm_id", values="correct", aggfunc="first").astype(bool)
    grouped = {key: g.set_index("item_id") for key, g in aggs.groupby(["model", "dataset", "pair", "aggregator"])}
    values, max_error = [], 0.0
    for r in cells.itertuples():
        agg = grouped[(r.model, r.dataset, r.pair, r.aggregator)]
        ok = answered.loc[(r.model, r.dataset)]
        both = ok[r.arm_a] & ok[r.arm_b]
        ids = both[both].index
        if len(ids) != r.n:
            raise SystemExit(f"❌ {r.model} | {r.dataset} | {r.pair} | {r.aggregator}: {len(ids)} both-answered items in the export, CSV n = {r.n}")
        value = 100 * (agg.loc[ids, "correct"].astype(bool).mean() - correct.loc[(r.model, r.dataset)].loc[ids, "L:en"].mean())
        if r.arm_a == "L:en":
            max_error = max(max_error, abs(value - 100 * (r.acc_final - r.acc_a)))
        values.append(value)
    cells = cells.assign(vs_en_pp=values)
    return cells, max_error


# ------------------------------------------------------------------
# 跨區塊
# ------------------------------------------------------------------
def pairTable(cells: pd.DataFrame, groups: list, by_strength: bool = False) -> pd.DataFrame:
    rows = []
    for zh, _, pairs in groups:
        for pair in pairs:
            for strength, models in ([("弱", WEAK), ("強", STRONG)] if by_strength else [("全部", MODELS)]):
                g = cells[(cells.pair == pair) & cells.model.isin(models)]
                row = {"來源": zh, "配對": pair, "名稱": label(pair)}
                if by_strength:
                    row["模型"] = f"{strength}（{'、'.join(models)}）"
                row.update({"英文 path 數": int(g.n_english.iloc[0]), "對稱": bool(g.symmetric.iloc[0]), "區塊數": len(g)})
                for key, _ in QUANTITIES:
                    s = summarize(g[key])
                    row.update({f"{key}_mean": s["mean"], f"{key}_se": s["se"], f"{key}_ci_low": s["ci_low"], f"{key}_ci_high": s["ci_high"]})
                row["excess_pp_n_positive"] = int((g.excess_pp > 0).sum())
                rows.append(row)
    return pd.DataFrame(rows)


def fmt(row, key: str, digits: int) -> str:
    return (f"{row[f'{key}_mean']:.{digits}f}（{row[f'{key}_se']:.{digits}f}）"
            f"[{row[f'{key}_ci_low']:.{digits}f}, {row[f'{key}_ci_high']:.{digits}f}]")


def markdownTable(table: pd.DataFrame, by_strength: bool = False) -> str:
    """兩段：d、c、m、headroom；recovery 到 聚合 − L:en。每格 = 平均（SE）[95% t 區間]。"""
    digits = {"d": 3, "c": 3, "m": 3, "headroom_pp": 2, "recovery": 3, "recovery_blind": 3, "strongest_pp": 2,
              "aggregation_pp": 2, "excess_pp": 2, "vs_en_pp": 2}
    head = ["來源", "名稱"] + (["模型"] if by_strength else [])
    first, second = [], []
    for _, r in table.iterrows():
        base = {k: r[k] for k in head}
        first.append({**base, "英文 path 數": r["英文 path 數"], "對稱": "是" if r["對稱"] else "否",
                      **{title: fmt(r, key, digits[key]) for key, title in QUANTITIES[:4]}})
        second.append({**base, **{title: fmt(r, key, digits[key]) for key, title in QUANTITIES[4:]},
                       "Excess 為正的區塊": f"{r['excess_pp_n_positive']}/{r['區塊數']}"})
    return (pd.DataFrame(first).to_markdown(index=False, disable_numparse=True) + "\n\n"
            + pd.DataFrame(second).to_markdown(index=False, disable_numparse=True))


def checks(cells: pd.DataFrame) -> list[dict]:
    judge = cells[cells.aggregator == "judge"]
    out = []
    for aggregator, pair, key, digits, expected in CHECKS:
        value = float(judge[judge.pair == pair][key].mean())
        out.append({"項目": f"{pair} {key}", "算出": f"{value:.4f}", "四捨五入": f"{round(value, digits):.{digits}f}",
                    "現況文件": f"{expected:.{digits}f}", "一致": round(value, digits) == expected})
    pair_means = judge.groupby("pair").recovery.mean()
    for zh, _, pairs in JUDGE_GROUPS:
        value = float(pair_means[pairs].mean())
        out.append({"項目": f"{zh} 的平均 recovery（{', '.join(pairs)}）", "算出": f"{value:.4f}", "四捨五入": f"{round(value, 2):.2f}",
                    "現況文件": f"{SOURCE_CHECKS[zh]:.2f}", "一致": round(value, 2) == SOURCE_CHECKS[zh]})
    return out


# ------------------------------------------------------------------
# 圖
# ------------------------------------------------------------------
def layout(groups: list) -> tuple[list, list]:
    """由上到下的列：(y, 'group', 英文來源名) 與 (y, 'pair', 配對)；組與組之間空一列。"""
    rows, y = [], 0.0
    for i, (_, en, pairs) in enumerate(groups):
        if i:
            y -= 0.5
        rows.append((y, "group", en))
        y -= 1
        for pair in pairs:
            rows.append((y, "pair", pair))
            y -= 1
    return rows


def drawPanel(ax, table: pd.DataFrame, groups: list, show_labels: bool):
    ax.set_facecolor("white")
    rows = layout(groups)
    by_pair = table.set_index("配對")
    for y, kind, value in rows:
        if kind != "pair":
            continue
        r = by_pair.loc[value]
        ax.barh(y, r["headroom_pp_mean"], height=0.62, color=GRAY, zorder=1)
        s, a = r["strongest_pp_mean"], r["aggregation_pp_mean"]
        ax.plot([s, a], [y, y], color=INK, linewidth=0.8, zorder=2)
        ax.hlines(y, r["aggregation_pp_ci_low"], r["aggregation_pp_ci_high"], color=BLUE, linewidth=1.4, zorder=3)
        ax.scatter([s], [y], marker="^", s=46, color=ORANGE, edgecolors="white", linewidths=0.8, zorder=4)
        ax.scatter([a], [y], marker="o", s=40, color=BLUE, edgecolors="white", linewidths=0.8, zorder=5)
    ax.set_yticks([y for y, _, _ in rows])
    if show_labels:
        ax.set_yticklabels([label(v) if k == "pair" else v for _, k, v in rows])
        for tick, (_, kind, _) in zip(ax.get_yticklabels(), rows):
            if kind == "group":
                tick.set_fontweight("bold")
                tick.set_color(INK)
            else:
                tick.set_color(MUTED)
    else:
        ax.tick_params(axis="y", labelleft=False)   # 共用 y 軸：不能清空標籤，否則左邊面板也會不見
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.axvline(0, color=MUTED, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b0b0b0")
    ax.set_xlabel("Accuracy gain on both-answered items (pp)", fontsize=9, color=INK)


def legendHandles(aggregator: str) -> list:
    return [Patch(facecolor=GRAY, label="Headroom"),
            Line2D([], [], marker="^", color="none", markerfacecolor=ORANGE, markeredgecolor="white", markersize=8,
                   label="Recovered by the strongest single path"),
            Line2D([0, 1], [0, 0], marker="o", color=BLUE, markerfacecolor=BLUE, markeredgecolor="white", markersize=7, linewidth=1.4,
                   label=AGG_LABEL[aggregator] + " ± 95% CI")]


def xRange(tables: list[pd.DataFrame]) -> tuple[float, float]:
    low = min(min(t.aggregation_pp_ci_low.min(), t.strongest_pp_mean.min(), 0.0) for t in tables)
    high = max(max(t.headroom_pp_mean.max(), t.aggregation_pp_ci_high.max()) for t in tables)
    pad = 0.04 * (high - low)
    return low - pad, high + pad


def save(fig, out_dir: str, name: str):
    for ext, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), facecolor="white", bbox_inches="tight", **kwargs)
    plt.close(fig)


def plotSingle(table: pd.DataFrame, groups: list, aggregator: str, title: str, out_dir: str, name: str):
    n_rows = len(layout(groups))
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * n_rows + 1.6))
    fig.patch.set_facecolor("white")
    drawPanel(ax, table, groups, True)
    ax.set_xlim(*xRange([table]))
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.legend(handles=legendHandles(aggregator), loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=1, frameon=False, fontsize=8.5)
    save(fig, out_dir, name)


def plotStrength(table: pd.DataFrame, out_dir: str, name: str):
    weak = table[table["模型"].str.startswith("弱")]
    strong = table[table["模型"].str.startswith("強")]
    n_rows = len(layout(JUDGE_GROUPS))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.32 * n_rows + 1.6), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, t, show, title in ((axes[0], weak, True, "Weak models (GPT-4o mini, Qwen3-8B), 8 blocks"),
                               (axes[1], strong, False, "Strong models (DeepSeek V4.1 Flash, Gemini 3.1 Flash-Lite), 8 blocks")):
        drawPanel(ax, t, JUDGE_GROUPS, show)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    axes[0].set_xlim(*xRange([weak, strong]))
    fig.legend(handles=legendHandles("judge"), loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, out_dir, name)


# ------------------------------------------------------------------
# 報告
# ------------------------------------------------------------------
def describe(judge: pd.DataFrame, debate: pd.DataFrame, strength: pd.DataFrame) -> list[str]:
    """第 (5) 節：只寫數字顯示的事（句中的清單與範圍都由數字算出）。"""
    j = judge.set_index("配對")
    lang, samp, en_lang = ["EN+ZH", "EN+JA", "ZH+JA"], ["EN+S1", "S1+S2"], ["EN+ZH", "EN+JA"]
    rng = lambda series, d=2: f"{series.min():.{d}f}–{series.max():.{d}f}"
    top3 = j.recovery_mean.nlargest(3).index.tolist()
    below = j[j.vs_en_pp_mean < 0]
    above = j[j.vs_en_pp_mean >= 0]
    weak = strength[strength["模型"].str.startswith("弱")].set_index("配對")
    strong = strength[strength["模型"].str.startswith("強")].set_index("配對")
    higher = int((strong.recovery_mean > weak.recovery_mean.reindex(strong.index)).sum())
    weak_neg = weak[weak.excess_pp_mean < 0]
    strong_neg = strong[strong.excess_pp_mean < 0]
    return [
        f"- 三組語言配對的 headroom 是 {rng(j.loc[lang, 'headroom_pp_mean'], 1)}pp，其餘九組是 {rng(j.drop(lang).headroom_pp_mean, 1)}pp；"
        f"recovery 最高的三組是 {'、'.join(top3)}（{rng(j.loc[top3, 'recovery_mean'])}）。",
        f"- 含英文的兩組語言配對（EN+ZH、EN+JA）的 recovery_blind 是 {rng(j.loc[en_lang, 'recovery_blind_mean'])}，"
        f"Excess 是 {rng(j.loc[en_lang, 'excess_pp_mean'])}pp；採樣兩組的 recovery_blind 是 {rng(j.loc[samp, 'recovery_blind_mean'])}，"
        f"Excess 是 {rng(j.loc[samp, 'excess_pp_mean'])}pp。",
        f"- ZH+JA 的 Excess 是 12 組中最高（{j.loc['ZH+JA', 'excess_pp_mean']:.2f}pp）；聚合後正確率的平均低於 L:en 的有 "
        + "、".join(f"{p}（{r.vs_en_pp_mean:+.2f}pp）" for p, r in below.iterrows())
        + f"，其餘 {len(above)} 組的平均都高於 L:en（{rng(above.vs_en_pp_mean)}pp）。",
        f"- 依模型強弱分開時，{higher}/12 組配對強模型的 recovery 高於弱模型（例如 EN+ZH {strong.loc['EN+ZH', 'recovery_mean']:.2f} 對 "
        f"{weak.loc['EN+ZH', 'recovery_mean']:.2f}）；Excess 平均為負的，弱模型有 {len(weak_neg)} 組"
        + (f"（{'、'.join(f'{p} {r.excess_pp_mean:+.2f}' for p, r in weak_neg.iterrows())}pp）" if len(weak_neg) else "")
        + f"，強模型有 {len(strong_neg)} 組。",
        f"- Debate 的 12 組 recovery 是 {rng(debate.recovery_mean)}，Excess 是 {rng(debate.excess_pp_mean)}pp，"
        f"{int((debate.excess_pp_mean > 0).sum())}/12 組的平均為正。",
    ]


def main():
    args = parseArgs()
    cells = loadCells(args.cells)
    cells, en_error = addVersusEnglish(cells, args.items_dir)
    check_rows = checks(cells)
    print(pd.DataFrame(check_rows).to_string(index=False))
    print(f"含 L:en 的配對：逐題算的 聚合 − L:en 與 CSV 的 100·(acc_final − acc_a) 最大差 {en_error:.1e}")
    if not all(r["一致"] for r in check_rows) or en_error > 1e-9:
        raise SystemExit("❌ 核對沒有全部通過：停下來回報，不寫輸出")

    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams["font.family"] = "sans-serif"
    keep = ["model", "dataset", "aggregator", "source", "pair", "arm_a", "arm_b", "n_english", "symmetric", "n", "n_dis"] + \
           [key for key, _ in QUANTITIES] + ["acc_final", "acc_a", "acc_b"]
    cells[keep].to_csv(os.path.join(args.out_dir, "cells_long.csv"), index=False)
    tables = {}
    for name, aggregator, groups, by_strength in (("pair_table_judge", "judge", JUDGE_GROUPS, False),
                                                  ("pair_table_debate", "debate", DEBATE_GROUPS, False),
                                                  ("pair_table_judge_by_strength", "judge", JUDGE_GROUPS, True)):
        table = pairTable(cells[cells.aggregator == aggregator], groups, by_strength)
        table.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        Path(os.path.join(args.out_dir, f"{name}.md")).write_text(markdownTable(table, by_strength) + "\n", encoding="utf-8")
        tables[name] = table
    plotSingle(tables["pair_table_judge"], JUDGE_GROUPS, "judge",
               "Two-path aggregation (judge): headroom and what each choice recovers (mean of 16 blocks)", args.out_dir, "fig_pairs_judge")
    plotStrength(tables["pair_table_judge_by_strength"], args.out_dir, "fig_pairs_judge_by_strength")
    plotSingle(tables["pair_table_debate"], DEBATE_GROUPS, "debate",
               "Two-path aggregation (debate): headroom and what each choice recovers (mean of 16 blocks)", args.out_dir, "fig_pairs_debate")

    judge = cells[(cells.aggregator == "judge") & (cells.pair == "EN+ZH")]
    naive_agg = judge.headroom_pp.mean() * judge.recovery.mean()
    naive_ex = judge.headroom_pp.mean() * (judge.recovery.mean() - judge.recovery_blind.mean())
    note = (f"第 6 到 8 項都是每個區塊先算 headroom × recovery（或 recovery_blind），再對區塊平均；不是平均後的 headroom 乘平均後的 recovery。"
            f"以 Judge 的 EN+ZH 為例：逐區塊先乘再平均，聚合拿回 {judge.aggregation_pp.mean():.2f}pp、Excess {judge.excess_pp.mean():.2f}pp；"
            f"先平均再乘（{judge.headroom_pp.mean():.2f} × {judge.recovery.mean():.3f}）會得到 {naive_agg:.2f}pp，"
            f"Excess 變成 {judge.headroom_pp.mean():.2f} × ({judge.recovery.mean():.3f} − {judge.recovery_blind.mean():.3f}) = {naive_ex:.2f}pp。"
            "兩種算法的差，等於 headroom 與 recovery（或 recovery − recovery_blind）在 16 個區塊間的共變異數"
            f"（分母 16）：聚合拿回 {judge.aggregation_pp.mean() - naive_agg:+.2f}pp、Excess {judge.excess_pp.mean() - naive_ex:+.2f}pp。")
    en_s1 = tables["pair_table_judge"].set_index("配對").loc["EN+S1"]
    en_s1_debate = tables["pair_table_debate"].set_index("配對").loc["EN+S1"]

    out = ["# 12 組配對的分解總表與圖", "",
           f"產生時間：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}；程式：`scripts/analysis_decomposition/pair_tables.py`。"
           "整理現有結果：不呼叫 API，不重跑任何東西，不修改現有檔案；沒有判定，也不依門檻分類。", ""]
    out += ["## (1) 讀了哪些檔案與欄位", "",
            f"- `{args.cells}`：四個新模型（{'、'.join(MODELS)}；不含舊的 deepseek、gemini）、`subset = both_answered`、"
            "`aggregator` 為 judge 或 debate 的列。欄位：`model`、`dataset`、`pair`、`arm_a`、`arm_b`、`symmetric`、`n_english`、`n`、"
            "`d`、`c`、`m`、`recovery_H2`、`recovery_blind_H2`、`acc_final`、`acc_a`。",
            f"- `{args.items_dir}/paths.csv.gz`、`aggregations.csv.gz`（與 CSV 同一次 run_analysis 產生的逐題匯出）：只用於第 9 項。"
            "CSV 只有配對內兩條 path 的正確率，不含 L:en 的配對沒有 L:en 的數字，所以第 9 項一律在該配對的 both_answered 題目上，"
            "用逐題的聚合正確率減 L:en 正確率。",
            "- `result/aggregations/{4 模型}/*/judge__*.json` 的檔頭：192 個主網格 Judge 檔的 `prompt_version` 都是 `choice-v1`（開始前已核對）。",
            "- headroom = 100·d·(c − m)（d、c、m 是 both_answered 全部題目的值）；recovery、recovery_blind 用另一半題目上評估的 "
            "`recovery_H2`、`recovery_blind_H2`。區間都是 `Analysis.blockStats.summarize` 的 95% t 區間"
            "（16 個區塊自由度 15；強弱分開時各 8 個區塊，自由度 7）。", ""]
    out += ["## (2) 第三節的核對", "", pd.DataFrame(check_rows).to_markdown(index=False, disable_numparse=True), "",
            f"全部一致。另外，含 L:en 的配對上，逐題算的「聚合 − L:en」與 CSV 的 100·(acc_final − acc_a) 最大差 {en_error:.1e}。", "",
            f"不在核對清單內、只回報：現況文件結果 6 的 EN+S1「比單用英文高」是 Judge +0.64pp（0.22 到 1.06，13/16）、Debate +0.87pp；"
            f"本表第 9 項是 Judge {en_s1['vs_en_pp_mean']:+.2f}pp（{en_s1['vs_en_pp_ci_low']:.2f} 到 {en_s1['vs_en_pp_ci_high']:.2f}）、"
            f"Debate {en_s1_debate['vs_en_pp_mean']:+.2f}pp。EN+ZH（Judge +0.31、Debate +1.00）與 ZH+JA（−1.58，結果 8）都與現況文件一致；"
            "EN+S1 用配對自己的 both_answered、EN/ZH/S1 共同子集、全部題目、H2 平均四種算法都得不到 +0.64，原因未查。", ""]
    out += ["## (3) 三張表", "",
            "每格 = 16 個區塊（強弱分開時各 8 個）的平均（SE）[95% t 區間]。完整數字在同名的 .csv；逐區塊的值在 `cells_long.csv`"
            "（區塊 × 配對 × 聚合器一列，共 384 列）。", "", f"> {note}", "",
            "### 表 1：Judge 的 12 組配對（`pair_table_judge`）", "", markdownTable(tables["pair_table_judge"]), "",
            "### 表 2：Debate 的 12 組配對（`pair_table_debate`）", "",
            "Judge 與 Debate 只有 5 組共同配對（EN+ZH、EN+JA、ZH+JA、EN+S1、P1+P2），兩張表分開列，不合併平均。", "",
            markdownTable(tables["pair_table_debate"]), "",
            "### 表 3：Judge，依模型強弱分開（`pair_table_judge_by_strength`，只當描述）", "",
            markdownTable(tables["pair_table_judge_by_strength"], True), ""]
    out += ["## (4) 三張圖", "",
            "灰色橫條 = headroom；橘色三角形 = 單一最強拿回的量；藍色圓點 = 聚合拿回的量（附 95% t 區間）；兩者之間的距離 = Excess，"
            "藍點在橘色左邊時聚合輸。各有 .png（300 dpi）與 .pdf。", "",
            "![Judge](fig_pairs_judge.png)", "", "![Judge, weak vs strong](fig_pairs_judge_by_strength.png)", "",
            "![Debate](fig_pairs_debate.png)", ""]
    out += ["## (5) 看到的型態（只描述數字）", ""] + describe(tables["pair_table_judge"], tables["pair_table_debate"], tables["pair_table_judge_by_strength"]) + [""]
    Path(os.path.join(args.out_dir, "report.md")).write_text("\n".join(out), encoding="utf-8")
    print(f"-> {args.out_dir}/report.md")


if __name__ == "__main__":
    main()
