"""
probe_regret.py — RQ1：少量標註能否預測「該聚合，還是直接用單一最強 path」

規格與判定標準：result/analysis/rq1/rq1_criteria.md（確認後才能執行；每份輸出記錄它的 sha256）。
    1. 檢查判定標準已確認
    2. 執行前核對（§6）：逐題匯出重算的 d / c / m / recovery / recovery_blind_H2 必須與 aggregation_cells.csv 完全相同，否則停止
    3. 每次重複切 H1 / H2，算同格 probe（隨機、只抽分歧題）與三種遷移（Analysis/probe.py）
    4. 輸出（--out-dir；模型不是最終四個時寫到 preliminary/）：
         cells.csv.gz   每個（設定, 基準, 聚合器, k, 目標格, 來源）對重複的平均
         blocks.csv     每個區塊（模型 × 資料集）的原始數字（百分點）
         summary.csv    跨區塊平均、SE、95% t 區間
         criteria.csv   §5 判定表（Judge、配對內基準）
         report.md      報告（第一張表 = 空間）
         k_curves_{aggregator}.png

資料：result/analysis/items/（run_analysis.py 的逐題匯出）、result/analysis/aggregation_cells.csv

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq1/probe_regret.py --check-only              # 只做 §6 核對
    conda run -n clreasoning python scripts/analysis_rq1/probe_regret.py -m gpt4omini qwen         # preliminary
    conda run -n clreasoning python scripts/analysis_rq1/probe_regret.py                           # 最終（四個模型）
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import hashlib
import os
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root：Analysis 等套件

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from Analysis.itemMatrix import loadBlocks, crossCheckCells
from Analysis.probe import (KS, PROBE_RANDOM, PROBE_DIS, TRANSFER_MODEL, TRANSFER_DATASET, TRANSFER_SOURCE,
                            TRANSFER_SOURCE_SAME_AXIS, TRANSFERS, NO_K, CRITERIA,
                            runProbes, blockTable, summaryTable, criteriaTable, verdict)

FINAL_MODELS = ["gpt4omini", "qwen", "deepseek4.1flash", "gemini3.1flashlite"]
AGGREGATORS = ["judge", "debate"]
SETTING_NAMES = {
    PROBE_RANDOM: "同格 probe（隨機）", PROBE_DIS: "同格 probe（只抽分歧題）",
    TRANSFER_MODEL: "換模型", TRANSFER_DATASET: "換資料集", TRANSFER_SOURCE: "換多樣性來源（不同軸）",
    TRANSFER_SOURCE_SAME_AXIS: "換多樣性來源（同軸，參考）",
}
# matplotlib 的預設字型沒有中文字，圖上用英文
PLOT_NAMES = {
    PROBE_RANDOM: "same-cell probe (random)", PROBE_DIS: "same-cell probe (disagreements only)",
    TRANSFER_MODEL: "transfer: other model", TRANSFER_DATASET: "transfer: other dataset",
    TRANSFER_SOURCE: "transfer: other diversity source",
}
RATE_METRICS = {"tie_rate", "aggregate_rate", "short_rate"}


def parseArgs():
    parser = ArgumentParser(description="RQ1 probe → aggregate-or-not decision (rq1_criteria.md)")
    parser.add_argument("-m", "--models", nargs="+", default=FINAL_MODELS)
    parser.add_argument("--items-dir", default="result/analysis/items")
    parser.add_argument("--cells", default="result/analysis/aggregation_cells.csv")
    parser.add_argument("--criteria", default="result/analysis/rq1/rq1_criteria.md")
    parser.add_argument("--out-dir", default="result/analysis/rq1")
    parser.add_argument("--ks", type=int, nargs="+", default=list(KS))
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true", help="Only run the §6 cross-check")
    return parser.parse_args()


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def confirmationLine(criteria_path: str) -> str | None:
    """rq1_criteria.md 的「確認：」行；尚未填入確認時間時回傳 None。"""
    with open(criteria_path, encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\s*-\s*確認：(.*)", line)
            if match:
                text = match.group(1).strip()
                return None if "待填" in text or not text else text
    return None


def pp(value: float, signed: bool = True) -> str:
    return "—" if pd.isna(value) else (f"{value:+.2f}" if signed else f"{value:.2f}")


def ciText(summary: pd.DataFrame, key: dict, metric: str) -> str:
    row = summary
    for column, value in key.items():
        row = row[row[column] == value]
    row = row[row["metric"] == metric]
    if row.empty:
        return "—"
    row = row.iloc[0]
    signed = metric not in RATE_METRICS
    if pd.isna(row["ci_low"]):
        return pp(row["mean"], signed)
    return f"{pp(row['mean'], signed)} [{pp(row['ci_low'], signed)}, {pp(row['ci_high'], signed)}]"


def spaceTable(blocks: pd.DataFrame, aggregator: str, k: int) -> pd.DataFrame:
    """第一張表：同格隨機 probe（k）下，兩種基準各區塊的 regret_A、regret_S（空間 = 兩者較小值）。"""
    sub = blocks[(blocks["setting"] == PROBE_RANDOM) & (blocks["k"] == k) & (blocks["aggregator"] == aggregator)]
    table = sub.pivot_table(index=["model", "dataset"], columns="baseline", values=["regret_A", "regret_S"])
    table.columns = [f"{baseline}: {metric}" for metric, baseline in table.columns]
    table = table[[c for b in ("pair", "global") for c in (f"{b}: regret_A", f"{b}: regret_S") if c in table.columns]]
    table.loc[("平均", ""), :] = table.mean()
    return table.round(2)


def curveTable(summary: pd.DataFrame, aggregator: str, baseline: str, ks: list[int], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for setting in (PROBE_RANDOM, PROBE_DIS) + TRANSFERS + (TRANSFER_SOURCE_SAME_AXIS,):
        for k in (ks if setting in (PROBE_RANDOM, PROBE_DIS) else [NO_K]):
            key = {"setting": setting, "baseline": baseline, "aggregator": aggregator, "k": k}
            n = summary
            for column, value in key.items():
                n = n[n[column] == value]
            if n.empty:
                continue
            row = {"設定": SETTING_NAMES[setting], "k": k if k != NO_K else "—", "區塊數": int(n["n_blocks"].iloc[0])}
            row.update({metric: ciText(summary, key, metric) for metric in metrics})
            rows.append(row)
    return pd.DataFrame(rows)


def plotCurves(summary: pd.DataFrame, aggregator: str, path: str, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, metric, label in zip(axes, ("diff_D_A", "diff_D_S"), ("D − always aggregate (pp)", "D − always single path (pp)")):
        for setting, style in ((PROBE_RANDOM, "o-"), (PROBE_DIS, "s--")):
            sub = summary[(summary["setting"] == setting) & (summary["aggregator"] == aggregator)
                          & (summary["baseline"] == "pair") & (summary["metric"] == metric)].sort_values("k")
            if sub.empty:
                continue
            line, = ax.plot(sub["k"], sub["mean"], style, label=PLOT_NAMES[setting])
            ax.fill_between(sub["k"], sub["ci_low"], sub["ci_high"], color=line.get_color(), alpha=0.15)
        for setting, color in zip(TRANSFERS, ("tab:green", "tab:red", "tab:purple")):
            sub = summary[(summary["setting"] == setting) & (summary["aggregator"] == aggregator)
                          & (summary["baseline"] == "pair") & (summary["metric"] == metric)]
            if not sub.empty:
                ax.axhline(sub["mean"].iloc[0], color=color, linestyle=":", label=PLOT_NAMES[setting])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("probe size k (labelled items in H1)")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parseArgs()
    preliminary = sorted(args.models) != sorted(FINAL_MODELS)
    out_dir = os.path.join(args.out_dir, "preliminary") if preliminary else args.out_dir

    confirmed = confirmationLine(args.criteria)
    if not args.check_only and confirmed is None:
        raise SystemExit(f"❌ {args.criteria} has not been confirmed yet (the 確認 line is empty); only --check-only may run")

    print(f"Loading {args.items_dir} for {args.models} ...")
    blocks = loadBlocks(args.items_dir, args.models, AGGREGATORS)
    print(f"  {len(blocks)} blocks, {sum(len(b.cells) for b in blocks.values())} cells")

    mismatches = crossCheckCells(blocks, args.cells, args.seed, args.reps)
    if mismatches:
        raise SystemExit("❌ §6 cross-check failed, stopping:\n  " + "\n  ".join(mismatches[:50]))
    print(f"✅ §6 cross-check: every cell's d / c / m / recovery / recovery_blind_H2 matches {args.cells}")
    if args.check_only:
        return

    print(f"Running {args.reps} repetitions ...")
    cells = runProbes(blocks, args.ks, args.reps, args.seed)
    block_table = blockTable(cells)
    summary = summaryTable(block_table, CRITERIA["level"])
    criteria = criteriaTable(block_table, "judge", "pair", CRITERIA)
    decision = verdict(criteria, CRITERIA)

    os.makedirs(out_dir, exist_ok=True)
    cells.to_csv(os.path.join(out_dir, "cells.csv.gz"), index=False, compression={"method": "gzip", "mtime": 0})
    block_table.to_csv(os.path.join(out_dir, "blocks.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    criteria.to_csv(os.path.join(out_dir, "criteria.csv"), index=False)
    for aggregator in AGGREGATORS:
        if (summary["aggregator"] == aggregator).any():
            plotCurves(summary, aggregator, os.path.join(out_dir, f"k_curves_{aggregator}.png"),
                       f"RQ1 {'(preliminary) ' if preliminary else ''}{aggregator}, pair baseline: "
                       f"mean over {block_table['model'].nunique()}×{block_table['dataset'].nunique()} blocks, 95% t-CI")

    main_metrics = ["diff_D_A", "diff_D_S", "tie_rate", "aggregate_rate"]
    lines = [
        f"# RQ1 結果{'（preliminary，不作判定）' if preliminary else ''}",
        "",
        f"- 產生時間：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        f"- 模型：{', '.join(args.models)}；區塊數 {len(blocks)}；重複 {args.reps} 次（seed {args.seed}）；k = {args.ks}",
        f"- 判定標準：`{args.criteria}`，sha256 `{sha256(args.criteria)}`，確認：{confirmed}",
        f"- 輸入：`{args.items_dir}`（paths sha256 `{sha256(os.path.join(args.items_dir, 'paths.csv.gz'))[:16]}`，"
        f"aggregations sha256 `{sha256(os.path.join(args.items_dir, 'aggregations.csv.gz'))[:16]}`）",
        "- §6 執行前核對：通過",
        "- 單位：百分點（H2 上的正確率差值）；區間 = 跨區塊的 95% t 區間；子集 = both_answered",
        "",
        f"## 1. 空間：同格隨機 probe k={CRITERIA['k']} 時各區塊的 regret（Judge）",
        "",
        spaceTable(block_table, "judge", CRITERIA["k"]).to_markdown(),
        "",
        "## 2. 判定（Judge、配對內基準）",
        "",
        criteria.round(3).to_markdown(index=False),
        "",
        f"**{'preliminary：只有部分模型，不據以判定。' if preliminary else ''}{decision}**",
        "",
    ]
    for aggregator in AGGREGATORS:
        for baseline in ("pair", "global"):
            table = curveTable(summary, aggregator, baseline, args.ks, main_metrics)
            if table.empty:
                continue
            lines += [f"## 3. k 曲線與遷移：{aggregator}、{'配對內' if baseline == 'pair' else '全域'}基準",
                      "", "D − A：我們的決策 − 永遠聚合；D − S：我們的決策 − 永遠用單一 path；平手率 = Excess 剛好為 0 的比例（%）", "",
                      table.rename(columns={"diff_D_A": "D − A", "diff_D_S": "D − S", "tie_rate": "平手率",
                                            "aggregate_rate": "判為聚合（%）"}).to_markdown(index=False), ""]
    for name, metrics, note in [
        ("敏感度 (a)：平手判為聚合", ["diff_D_A_tieagg", "diff_D_S_tieagg"], "D 在 Excess = 0 時改判聚合"),
        ("敏感度 (b)：不聚合時改用 H2 上真正較強的 path", ["diff_D_A_truepath", "diff_D_S_truepath", "regret_S_truepath"],
         "D 與 S 都改用 H2 上的真正較強者；與主結果的差 = path 選錯造成的部分"),
    ]:
        for aggregator in AGGREGATORS:
            table = curveTable(summary, aggregator, "pair", args.ks, metrics)
            if not table.empty:
                lines += [f"## 4. {name}（{aggregator}、配對內基準）", "", note, "", table.to_markdown(index=False), ""]
    regret_table = curveTable(summary, "judge", "pair", args.ks, ["regret_D", "regret_A", "regret_S"])
    lines += ["## 5. regret 絕對值（參考；Oracle 在 H2 上挑選，會偏高）", "", regret_table.to_markdown(index=False), ""]
    f_pairs = sorted(cells.loc[cells["involves_F"] & cells["setting"].isin([TRANSFER_SOURCE, TRANSFER_SOURCE_SAME_AXIS]),
                               "pair"].unique())
    skipped = cells[cells["skipped_reps"] > 0].groupby(["setting", "baseline", "aggregator", "k", "dataset"])["skipped_reps"].sum()
    lines += ["## 6. 備註", "",
              f"- 換多樣性來源中，目標或來源含 F 軸的配對（F:en 的原答案就是 L:en，兩者不獨立）在 `cells.csv.gz` 的 `involves_F` 欄標記；"
              f"目標配對含 F 的有：{', '.join(f_pairs) if f_pairs else '無'}，來源含 F 的列同樣標記。",
              "- k 大於該格 H1 題數而跳過的重複次數（設定, 基準, 聚合器, k, 資料集）：",
              "", skipped.to_frame().to_markdown() if not skipped.empty else "（無）", ""]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{decision}" + ("（preliminary，不據以判定）" if preliminary else ""))
    print(f"💾 {out_dir}/: cells.csv.gz, blocks.csv, summary.csv, criteria.csv, report.md, k_curves_*.png")


if __name__ == "__main__":
    main()
