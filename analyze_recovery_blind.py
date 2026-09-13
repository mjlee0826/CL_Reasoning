"""
analyze_recovery_blind.py — 0A-1：recovery_blind 的表 1–3 與主分析

前置 (依序執行):
    conda run -n clreasoning python test_em.py -t testrecoveryblind --testdir result/challenge
    conda run -n clreasoning python test_em_legacy.py result/tempature1/challenge_EN result/tempature1/challenge_CN --csv legacy_em_samelang.csv
    conda run -n clreasoning python split_half_gap.py        # 表 3 的 excess_H2 來自 split_half_points.csv

資料:
    current 160 對  <- result/challenge/*.json 的 metadata["RecoveryBlind"]   (Test/TestRecoveryBlind.py)
    legacy   24 對  <- legacy_em_samelang.csv                                  (test_em_legacy.py)
    題目範圍: d, c, m 用全部題目；*_H2 = H1 決定錨點、H2 計算、200 次切分平均

輸出:
    表 1   recovery_blind_pairs.csv        pair 層級 184 列 (legacy 的 cell 加 "legacy:" 前綴)
    表 2   recovery_blind_groups.csv       含英文 vs 不含英文 (current 160 列)，cell-cluster bootstrap
    表 3   recovery_blind_loss_cells.csv   虧損 cell (有 pair 的 excess < −1pp) 的全部配對：skill_H2 是否 ≤ 0，
                                           並列全樣本 excess 與 excess_H2 (與 skill_H2 同一批 H2 題目)
    主分析 recovery_blind_mixedlm.csv     recovery_H2 ~ recovery_blind_H2 + (1|cell)，184 列
           決策點 D0：β > 0.5 → recovery 不是純聚合器性質

用法:
    conda run -n clreasoning python analyze_recovery_blind.py
    conda run -n clreasoning python analyze_recovery_blind.py --boot 5000 --seed 0 --outdir .
"""

import glob
import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from regress_pair_acc import build_design

CHALLENGE_DIR = "result/challenge"
LOSS_THRESHOLD = -0.01

RB_COLS = ["anchor", "anchor_rate", "n_A_H2", "n_B_H2", "w_A_H2",
           "recovery_blind_H2", "recovery_H2", "skill_H2", "reps_undefined", "reps_skill_undefined"]
TABLE1_COLS = ["source", "cell", "model", "dataset", "pair", "has_english", "N", "D", "d", "c", "m"] + RB_COLS
GROUP_QUANTITIES = ["d", "c", "m", "recovery_blind_H2", "recovery_H2", "skill_H2"]


# ----------------------------------------------------------------------------
# 表 1：pair 層級
# ----------------------------------------------------------------------------
def load_current():
    rows, missing = [], []
    for path in sorted(glob.glob(os.path.join(CHALLENGE_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)[0]
        rb = meta.get("RecoveryBlind")
        if rb is None:
            missing.append(os.path.basename(path))
            continue
        l1, l2 = meta["Strategy"]["languages"]
        model, dataset = meta["Model"]["modelType"], meta["Dataset"]["datasetType"]
        rows.append({"source": "current", "cell": f"{model}|{dataset}", "model": model, "dataset": dataset,
                     "pair": f"{l1}_vs_{l2}", "has_english": "english" in (l1, l2), **rb})
    if missing:
        print(f"⚠️  {len(missing)} 個 challenge 檔還沒有 RecoveryBlind（請先跑 test_em.py -t testrecoveryblind）：{missing[:3]}")
    return pd.DataFrame(rows)


def load_legacy(path):
    if not os.path.exists(path):
        print(f"⚠️  找不到 {path}，表 1 不含 legacy")
        return pd.DataFrame(columns=TABLE1_COLS)
    df = pd.read_csv(path, encoding="utf-8-sig")
    lacking = [col for col in ["m"] + RB_COLS if col not in df.columns]
    if lacking:
        print(f"⚠️  {path} 缺少欄位 {lacking}（請先重跑 test_em_legacy.py --csv），表 1 不含 legacy")
        return pd.DataFrame(columns=TABLE1_COLS)

    df = df[df["condition"].isin(["challenge_EN", "challenge_CN"])].reset_index(drop=True)
    lang = df["condition"].map({"challenge_EN": "english", "challenge_CN": "chinese"})
    out = pd.DataFrame({
        "source": "legacy",
        "cell": "legacy:" + df["model"] + "|" + df["dataset"],
        "model": df["model"],
        "dataset": df["dataset"],
        "pair": lang + "_vs_" + lang,
        "has_english": lang == "english",
        "N": df["n"],
        "D": df["n_disagree"],
        "d": df["debate_rate"],
        "c": df["c_recoverable"],
        "m": df["m"],
    })
    for col in RB_COLS:
        out[col] = df[col]
    return out


# ----------------------------------------------------------------------------
# 表 2：含英文 vs 不含英文 (cell-cluster bootstrap)
# ----------------------------------------------------------------------------
def group_comparison(current, boot, seed):
    cells = sorted(current["cell"].unique())
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(cells), np.full(len(cells), 1 / len(cells)), size=boot)   # (boot, C)

    rows = []
    for q in GROUP_QUANTITIES:
        valid = current.dropna(subset=[q])
        sums, counts = {}, {}
        for flag in (True, False):
            grouped = valid[valid["has_english"] == flag].groupby("cell")[q]
            sums[flag] = grouped.sum().reindex(cells, fill_value=0).to_numpy(dtype=float)
            counts[flag] = grouped.count().reindex(cells, fill_value=0).to_numpy(dtype=float)

        mean_en = sums[True].sum() / counts[True].sum()
        mean_no = sums[False].sum() / counts[False].sum()
        boot_diff = (weights @ sums[True]) / (weights @ counts[True]) \
                    - (weights @ sums[False]) / (weights @ counts[False])
        rows.append({
            "quantity": q,
            "mean_with_english": mean_en,
            "mean_without_english": mean_no,
            "diff": mean_en - mean_no,
            "ci_lo": float(np.percentile(boot_diff, 2.5)),
            "ci_hi": float(np.percentile(boot_diff, 97.5)),
            "n_with_english": int(counts[True].sum()),
            "n_without_english": int(counts[False].sum()),
            "boot": boot,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 表 3：虧損 cell 的診斷
# ----------------------------------------------------------------------------
def loss_cell_table(current, split_points):
    design = build_design()                                           # 全樣本 excess (A-9 的定義)
    loss_cells = sorted(design.loc[design["excess"] < LOSS_THRESHOLD, "cell"].unique())
    table = design[design["cell"].isin(loss_cells)][["cell", "pair", "gap", "excess"]].copy()

    if os.path.exists(split_points):
        points = pd.read_csv(split_points, encoding="utf-8-sig")
        excess_H2 = points.groupby(["cell", "pair"], as_index=False)["excess_H2"].mean()
        table = table.merge(excess_H2, on=["cell", "pair"], how="left")
    else:
        print(f"⚠️  找不到 {split_points}（請先跑 split_half_gap.py），表 3 的 excess_H2 留空")
        table["excess_H2"] = np.nan

    cols = ["cell", "pair", "anchor", "recovery_H2", "recovery_blind_H2", "skill_H2"]
    table = table.merge(current[cols], on=["cell", "pair"], how="left")
    table["loss_pair"] = table["excess"] < LOSS_THRESHOLD
    table["skill_H2_le_0"] = table["skill_H2"] <= 0
    return table.sort_values(["cell", "excess"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 主分析：recovery_H2 ~ recovery_blind_H2 + (1|cell)
# ----------------------------------------------------------------------------
def mixed_model(table1):
    df = table1.dropna(subset=["recovery_H2", "recovery_blind_H2"]).copy()
    df["recovery_H2"] = df["recovery_H2"].astype(float)
    df["recovery_blind_H2"] = df["recovery_blind_H2"].astype(float)
    fit = smf.mixedlm("recovery_H2 ~ recovery_blind_H2", df, groups=df["cell"]).fit()
    ci = fit.conf_int()

    rows = []
    for term in ("Intercept", "recovery_blind_H2"):
        rows.append({
            "term": term,
            "estimate": float(fit.params[term]),
            "se": float(fit.bse[term]),
            "ci_lo": float(ci.loc[term, 0]),
            "ci_hi": float(ci.loc[term, 1]),
            "p": float(fit.pvalues[term]),
        })
    out = pd.DataFrame(rows)
    out["n_obs"] = len(df)
    out["n_groups"] = df["cell"].nunique()
    out["group_var"] = float(fit.cov_re.iloc[0, 0])
    out["converged"] = bool(fit.converged)
    out["D0_beta_gt_0_5"] = bool(fit.params["recovery_blind_H2"] > 0.5)
    return out


# ----------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="0A-1: recovery_blind 的表 1–3 與主分析")
    parser.add_argument("--legacy-csv", default="legacy_em_samelang.csv", help="test_em_legacy.py 的輸出")
    parser.add_argument("--split-points", default="split_half_points.csv", help="split_half_gap.py 的輸出")
    parser.add_argument("--boot", type=int, default=5000, help="表 2 的 cell-cluster bootstrap 次數")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap 的 seed")
    parser.add_argument("--outdir", default=".", help="CSV 輸出目錄")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    # ---- 表 1 ----
    current = load_current()
    if current.empty:
        print("❌ 沒有任何 challenge 檔帶 RecoveryBlind metadata")
        return
    legacy = load_legacy(args.legacy_csv)
    parts = [current[TABLE1_COLS]] + ([legacy[TABLE1_COLS]] if not legacy.empty else [])
    table1 = pd.concat(parts, ignore_index=True)

    print("=" * 100)
    print(f"📋 表 1  pair 層級：{len(table1)} 列（current {len(current)}、legacy {len(legacy)}）")
    print("=" * 100)
    summary_cols = ["d", "c", "m", "recovery_blind_H2", "recovery_H2", "skill_H2"]
    print(table1.groupby("source")[summary_cols].mean().to_string(float_format=lambda x: f"{x:.4f}"))
    undefined = int(table1["reps_undefined"].fillna(0).sum())
    skill_undefined = int(table1["reps_skill_undefined"].fillna(0).sum())
    print(f"\n   H2 中 recovery 無定義的次數合計 {undefined}；skill 無定義（recovery_blind = 1）的次數合計 {skill_undefined}")

    # ---- 表 2 ----
    table2 = group_comparison(current, args.boot, args.seed)
    print("\n" + "=" * 100)
    print(f"📊 表 2  含英文 vs 不含英文（current {len(current)} 列；對 {current['cell'].nunique()} 個 cell 重抽 {args.boot} 次）")
    print("=" * 100)
    print(table2.drop(columns=["boot"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- 表 3 ----
    table3 = loss_cell_table(current, args.split_points)
    print("\n" + "=" * 100)
    print(f"🔍 表 3  虧損 cell（有 pair 的全樣本 excess < {LOSS_THRESHOLD:+.2f}）的全部配對")
    print("=" * 100)
    show = ["cell", "pair", "excess", "excess_H2", "anchor", "recovery_H2", "recovery_blind_H2",
            "skill_H2", "loss_pair", "skill_H2_le_0"]
    print(table3[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    per_cell = table3.groupby("cell").agg(pairs=("pair", "size"), loss_pairs=("loss_pair", "sum"),
                                          skill_le_0=("skill_H2_le_0", "sum"))
    print("\n" + per_cell.to_string())

    # ---- 主分析 ----
    table4 = mixed_model(table1)
    beta = table4.set_index("term").loc["recovery_blind_H2"]
    print("\n" + "=" * 100)
    print(f"🧮 主分析  recovery_H2 ~ recovery_blind_H2 + (1|cell)   "
          f"n={int(beta['n_obs'])}，cell={int(beta['n_groups'])}，converged={beta['converged']}")
    print("=" * 100)
    print(table4[["term", "estimate", "se", "ci_lo", "ci_hi", "p"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    verdict = "β > 0.5 → recovery 不是純聚合器性質，RQ2 的設計要重想" if beta["D0_beta_gt_0_5"] \
        else "β ≤ 0.5 → 未觸發 D0"
    print(f"\n   決策點 D0：β = {beta['estimate']:.4f}  [{beta['ci_lo']:.4f}, {beta['ci_hi']:.4f}]  → {verdict}")

    # ---- 輸出 ----
    outputs = [(table1, "recovery_blind_pairs.csv"), (table2, "recovery_blind_groups.csv"),
               (table3, "recovery_blind_loss_cells.csv"), (table4, "recovery_blind_mixedlm.csv")]
    print()
    for table, name in outputs:
        path = os.path.join(args.outdir, name)
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"💾 {name} ({len(table)} 列) → {path}")


if __name__ == "__main__":
    main()
