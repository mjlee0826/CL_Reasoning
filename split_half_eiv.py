"""
split_half_eiv.py — 0A-6 版本 D：split-half + EIV，重算 λ'

問題 (paper_status v7 §2.7):
    split-half (0A-4) 去掉了耦合與選擇偏誤，但 gap_H1 只用一半題目，衰減偏誤變嚴重 → −0.361（無耦合、有衰減）；
    A-5 的 EIV −0.243 是全樣本（有耦合、無衰減）。兩者不可直接比，要對 split-half 的斜率也做 EIV 校正。

Step 1  split-half 斜率        split_half_points.csv（split_half_gap.py），每次切分各自迴歸再對 reps 次取平均
Step 2  信度比（A-5 的定義；N 不扣兩個語言答同一批題目的共變異）
            每對  N = (p1·q1 + p2·q2) / n                    p = 單語準確率，n = 題數
            S  = Var(gap) − N̄                               Var 用 ddof = 1
            λ  = S / (S + N̄)                                全樣本 n 題
            λ' = S / (S + f·N̄)，f = n / ⌊n/2⌋               gap_H1 只用 H1：2000 → 1000 題 f = 2；817 → 408 題 f = 2.0025
Step 3  校正（皆為 g_excess 的尺度，與 −0.243 相同；g_delta = 校正值 + 0.5）
            全樣本 EIV (A-5)     (g + 0.5) / λ − 0.5         全樣本的 −0.5 是耦合的精確項，只有 delta 部分被衰減
            版本 D-公式          (g_split + 0.5) / λ' − 0.5  同一個公式，只把 λ 換成 λ'
            版本 D-重複測量      g_split / λ'_rep            Frost & Thompson (2000)：λ'_rep = slope(gap_H2 ~ gap_H1)
        split-half 下 excess_H2 = delta_H2 − gap_H2/2，自變數卻是 gap_H1:
            slope(excess_H2) = slope(delta_H2) − 0.5 · slope(gap_H2)
        −0.5 那一項也被衰減，所以重複測量版兩項一起除以 λ'_rep；公式版假設 −0.5 不衰減。
Step 4  與全樣本 EIV 並列比較

三種規格 (與 split_half_gap.py 相同):
    pooled_ols      16 格一條線；Var(gap) 用 160 對的總變異，N̄ 為 160 對的平均
    cell_fe         16 格共同斜率；Var(gap) 用 cell 內合併變異 Σ SS_c / Σ(k_c − 1)，N̄ 以 (k_c − 1) 加權
    per_cell_mean   逐格校正後再平均，只用 λ > 0 的格子（A-5：8 / 16 格）；
                    摘要表中它的 λ / λ' / λ'_rep 是這些格子的平均，僅供參考

資料:
    pair_acc_design_matrix.csv   全樣本的 gap / excess / 單語準確率（regress_pair_acc.py）
    split_half_points.csv        gap_H1 / excess_H2 / gap_H2（split_half_gap.py）
    result/baseline/*.json       單語檔的題數 n，並核對設計矩陣的單語準確率

輸出:
    split_half_eiv_cells.csv     逐格 Var(gap)、N̄、S、λ、λ'、λ'_rep、全樣本與 split-half 的斜率與校正值
    split_half_eiv_summary.csv   三種規格 × {全樣本 raw、全樣本 EIV、split-half raw、版本 D-公式、版本 D-重複測量}

用法:
    conda run -n clreasoning python split_half_eiv.py
    conda run -n clreasoning python split_half_eiv.py --design pair_acc_design_matrix.csv --split-points split_half_points.csv --outdir .
"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from split_half_gap import VARIANTS, estimate, extract_baseline, sufficient_stats, within_cell

A5 = {"n_nonpositive": 8, "raw": -0.340, "eiv": -0.243}          # paper_status v6 §A-5
SPLIT_COLUMNS = ("cell", "rep", "pair", "gap_H1", "excess_H2", "gap_H2")
SUMMARY_COLS = ["spec", "n_cells", "lam", "lam_split", "lam_rep",
                "g_full", "g_full_eiv", "g_split", "d_formula", "d_replicate"]
CELL_COLS = ["cell", "n", "var_gap", "noise", "S", "lam", "lam_split", "lam_rep", "kept",
             "g_full", "g_full_eiv", "g_split", "d_formula", "d_replicate"]


def eiv_correct(g, lam):
    """A-5 的 EIV 校正 (g + 0.5) / λ − 0.5；全樣本 EIV 與版本 D-公式共用。"""
    return (g + 0.5) / lam - 0.5


def reliability(var_gap, noise, split_noise):
    """回傳 (S, λ, λ')：S = Var(gap) − N；λ = S / (S + N)；λ' = S / (S + f·N)。"""
    signal = var_gap - noise
    return signal, signal / (signal + noise), signal / (signal + split_noise)


# ----------------------------------------------------------------------------
# 資料：題數與每對的雜訊
# ----------------------------------------------------------------------------
def attach_noise(design):
    """每對加上 n、noise = (p1·q1 + p2·q2)/n、split_noise = f·noise。n 取自 baseline，並核對單語準確率。"""
    baseline = extract_baseline()
    sizes = []
    for row in design.itertuples(index=False):
        langs = baseline[(row.model, row.dataset)]
        n_pair = []
        for lang, acc in ((row.l1, row.mono_l1), (row.l2, row.mono_l2)):
            correct = list(langs[lang][0].values())
            if not np.isclose(round(float(np.mean(correct)), 4), acc, rtol=0, atol=1e-12):     # TestEM 存到小數第 4 位
                raise ValueError(f"{row.cell} {lang}: baseline 準確率 {np.mean(correct)} ≠ 設計矩陣 {acc}")
            n_pair.append(len(correct))
        if n_pair[0] != n_pair[1]:
            raise ValueError(f"{row.cell} {row.pair}: 兩個語言的題數不同 {n_pair}")
        sizes.append(n_pair[0])

    out = design.assign(n=sizes)
    out["noise"] = (out["mono_l1"] * (1 - out["mono_l1"]) + out["mono_l2"] * (1 - out["mono_l2"])) / out["n"]
    out["split_noise"] = out["noise"] * out["n"] / (out["n"] // 2)
    return out


# ----------------------------------------------------------------------------
# Step 1：斜率（沿用 split_half_gap.py 的充分統計量）
# ----------------------------------------------------------------------------
def slopes(points, cells, reps, x_col, y_col):
    """回傳 ({variant: 斜率對 reps 的平均}, 每個 cell 的斜率對 reps 的平均 (C,))。"""
    stats = sufficient_stats(points, cells, reps, x_col, y_col)
    within_xx, within_xy, _, _ = within_cell(stats)
    if (within_xx <= 0).any():
        raise ValueError(f"{x_col}: 有 cell 的自變數沒有變異，per-cell 斜率無定義")
    est = estimate(stats, np.ones((1, len(cells))))
    return {v: float(est[v]["slope"][:, 0].mean()) for v in VARIANTS}, (within_xy / within_xx).mean(axis=0)


# ----------------------------------------------------------------------------
# Step 2：信度比
# ----------------------------------------------------------------------------
def cell_reliability(design, cells):
    """逐格的 S、λ、λ'。λ ≤ 0（全距小於雜訊）的格子不做校正，與 A-5 相同。"""
    g = design.groupby("cell")
    table = pd.DataFrame({
        "n": g["n"].first(),
        "var_gap": g["gap"].var(ddof=1),
        "noise": g["noise"].mean(),
        "split_noise": g["split_noise"].mean(),
    }).loc[cells]
    table["S"], table["lam"], table["lam_split"] = reliability(table["var_gap"], table["noise"], table["split_noise"])
    table["kept"] = table["lam"] > 0
    return table


def aggregate_reliability(design):
    """{variant: (S, λ, λ')}。pooled_ols 用 160 對的總變異；cell_fe 用 cell 內合併變異，N̄ 以 (k_c − 1) 加權。"""
    out = {"pooled_ols": reliability(design["gap"].var(ddof=1), design["noise"].mean(), design["split_noise"].mean())}
    g = design.groupby("cell")
    dof = g.size() - 1
    out["cell_fe"] = reliability((g["gap"].var(ddof=1) * dof).sum() / dof.sum(),
                                 (g["noise"].mean() * dof).sum() / dof.sum(),
                                 (g["split_noise"].mean() * dof).sum() / dof.sum())
    return out


# ----------------------------------------------------------------------------
# Step 3 / 4：校正與對照
# ----------------------------------------------------------------------------
def build_tables(design, points):
    cells = sorted(design["cell"].unique())
    if sorted(points["cell"].unique()) != cells:
        raise ValueError("split_half_points.csv 的 cell 與設計矩陣不一致")
    reps = int(points["rep"].nunique())

    full, full_cell = slopes(design.assign(rep=0), cells, 1, "gap", "excess")          # 全樣本
    split, split_cell = slopes(points, cells, reps, "gap_H1", "excess_H2")             # Step 1
    lam_rep, lam_rep_cell = slopes(points, cells, reps, "gap_H1", "gap_H2")            # λ'_rep

    table = cell_reliability(design, cells)
    table["lam_rep"] = lam_rep_cell
    table["g_full"] = full_cell
    table["g_full_eiv"] = eiv_correct(table["g_full"], table["lam"])
    table["g_split"] = split_cell
    table["d_formula"] = eiv_correct(table["g_split"], table["lam_split"])
    table["d_replicate"] = table["g_split"] / table["lam_rep"]
    table.loc[~table["kept"], ["g_full_eiv", "d_formula", "d_replicate"]] = np.nan      # λ ≤ 0：不校正

    reliab = aggregate_reliability(design)
    kept = table[table["kept"]]
    rows = []
    for spec in VARIANTS:
        if spec == "per_cell_mean":
            row = {"n_cells": len(kept), **kept[SUMMARY_COLS[2:]].mean().to_dict()}
        else:
            _, lam, lam_split = reliab[spec]
            row = {"n_cells": len(cells), "lam": lam, "lam_split": lam_split, "lam_rep": lam_rep[spec],
                   "g_full": full[spec], "g_full_eiv": eiv_correct(full[spec], lam),
                   "g_split": split[spec], "d_formula": eiv_correct(split[spec], lam_split),
                   "d_replicate": split[spec] / lam_rep[spec]}
        rows.append({"spec": spec, **row})
    return table.rename_axis("cell").reset_index()[CELL_COLS], pd.DataFrame(rows)[SUMMARY_COLS]


def print_report(table, summary):
    fmt = lambda x: f"{x:.4f}"
    print("\n" + "=" * 120)
    print("📐 Step 2  逐格信度比（var_gap、noise 單位 pp²；N = (p1·q1 + p2·q2)/n；λ' 用 H1 題數）")
    print("=" * 120)
    show = table.assign(var_gap=table["var_gap"] * 1e4, noise=table["noise"] * 1e4, S=table["S"] * 1e4)
    print(show.to_string(index=False, float_format=fmt))

    per_cell = summary.set_index("spec").loc["per_cell_mean"]
    print(f"\n🔁 A-5 重現：λ ≤ 0 的格子 {int((~table['kept']).sum())}/{len(table)}（A-5 {A5['n_nonpositive']}/16）；"
          f"其餘 {int(per_cell['n_cells'])} 格 raw g {per_cell['g_full']:.3f}（A-5 {A5['raw']:.3f}）"
          f" → EIV {per_cell['g_full_eiv']:.3f}（A-5 {A5['eiv']:.3f}）")

    print("\n" + "=" * 120)
    print("📊 Step 3 / 4  版本 D vs 全樣本 EIV（g_excess 的尺度；g_delta = 數值 + 0.5）")
    print("=" * 120)
    print(summary.to_string(index=False, float_format=fmt))
    print("\n   g_full → g_full_eiv   全樣本斜率 → A-5 的 EIV：(g + 0.5)/λ − 0.5")
    print("   g_split               split-half 斜率（Step 1）")
    print("   d_formula             版本 D-公式：(g_split + 0.5)/λ' − 0.5，λ' = S/(S + f·N)")
    print("   d_replicate           版本 D-重複測量：g_split / λ'_rep，λ'_rep = slope(gap_H2 ~ gap_H1)（Frost & Thompson 2000）")
    print("   per_cell_mean 只平均 λ > 0 的格子（逐格校正後再平均）；它的 λ / λ' / λ'_rep 是這些格子的平均，僅供參考。")
    print("   ⚠️ split-half 下 −0.5 那一項也被衰減：slope(excess_H2) = slope(delta_H2) − 0.5·λ'_rep。"
          "d_formula 假設它不衰減，λ' 與 λ'_rep 差距大時兩者會分歧。")


# ----------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="0A-6 版本 D：split-half + EIV，重算 λ'")
    parser.add_argument("--design", default="pair_acc_design_matrix.csv", help="regress_pair_acc.py 的設計矩陣")
    parser.add_argument("--split-points", default="split_half_points.csv", help="split_half_gap.py 的逐點資料（需含 gap_H2）")
    parser.add_argument("--outdir", default=".", help="CSV 輸出目錄")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    points = pd.read_csv(args.split_points, encoding="utf-8-sig")
    missing = [col for col in SPLIT_COLUMNS if col not in points.columns]
    if missing:
        print(f"❌ {args.split_points} 缺少欄位 {missing}（請先重跑 split_half_gap.py）")
        return

    print("📦 讀取設計矩陣與 baseline 題數（逐檔載入）...")
    design = attach_noise(pd.read_csv(args.design, encoding="utf-8-sig"))
    table, summary = build_tables(design, points)
    print_report(table, summary)

    os.makedirs(args.outdir, exist_ok=True)
    print()
    for frame, name in ((table, "split_half_eiv_cells.csv"), (summary, "split_half_eiv_summary.csv")):
        path = os.path.join(args.outdir, name)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"💾 {name} ({len(frame)} 列) → {path}")


if __name__ == "__main__":
    main()
