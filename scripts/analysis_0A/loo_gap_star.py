"""
loo_gap_star.py — 對「辯論盈虧門檻 gap*」做 Leave-One-Cell-Out 交叉驗證

模型:
    excess = pair_acc - max_mono          (辯論相對「直接用較強語言」的盈虧)
    excess ~ gap + C(cell)                (帶 cell 固定效應)
    盈虧翻轉點  gap* = -a / g

流程 (16 折, 每折留一個 model x dataset 格):
    for i in 1..16:
        train = 其他 15 格 (150 列)
        a, g <- OLS(excess ~ gap + C(cell), train)
                g = 固定效應的組內斜率
                a = 15 個 cell 截距的平均
                   ^ 留出格沒有自己的固定效應 (它不在訓練資料裡),
                     所以用訓練格截距的平均當作「典型格」的截距。
                     這是新群體預測的標準處理, 也是唯一不外加假設的選擇。
        for 第 i 格的每個 pair:
            excess_pred = a + g * gap
        比對 excess_true

評分 (兩個層級都做):
    Pair 層級   160 次預測, 比對 sign(excess_pred) vs sign(excess_true)
    Cell 層級    16 次預測, 比對「此格是否存在虧損 pair」

⚠️ 類別嚴重不平衡, 所有數字都必須跟對照基準並列:
    Pair 層級  139/160 (86.9%) 是盈  -> 「永遠猜盈」就有 86.9%
    Cell 層級    9/16 沒有虧損 pair  -> 「永遠猜無虧損」就有 56.3%
    因此真正有資訊量的是: 虧損那 20 個 pair 抓到幾個 (recall), 以及 balanced accuracy。

用法:
    conda run -n clreasoning python scripts/analysis_0A/loo_gap_star.py
    conda run -n clreasoning python scripts/analysis_0A/loo_gap_star.py --design result/analysis/0A/pair_acc_design_matrix.csv
"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_fe(train):
    """在訓練格上跑 excess ~ gap + C(cell) - 1。

    去掉截距後，C(cell) 展開成完整虛擬變數，係數就直接是各格截距，
    不必從 treatment coding 反推。gap 的係數是固定效應的組內斜率。
    """
    m = smf.ols("excess ~ gap + C(cell) - 1", data=train).fit()
    g = m.params["gap"]
    cell_intercepts = m.params[[k for k in m.params.index if k.startswith("C(cell)")]]
    a = float(cell_intercepts.mean())
    return a, g, m


def run_loo(df):
    cells = sorted(df["cell"].unique())
    fold_rows, pred_rows = [], []

    for cell in cells:
        train = df[df["cell"] != cell]
        test = df[df["cell"] == cell].copy()

        a, g, _ = fit_fe(train)
        gap_star = -a / g if g != 0 else np.nan

        test["excess_pred"] = a + g * test["gap"]
        test["a_used"] = a
        test["g_used"] = g
        test["gap_star"] = gap_star
        pred_rows.append(test)

        n_pred_loss = int((test["excess_pred"] < 0).sum())
        n_true_loss = int((test["excess_true_neg"]).sum())
        fold_rows.append({
            "cell": cell,
            "n_train": len(train),
            "a": a, "g": g, "gap_star": gap_star,
            "max_gap": test["gap"].max(),
            "pred_has_loss": n_pred_loss > 0,
            "true_has_loss": n_true_loss > 0,
            "n_pred_loss": n_pred_loss,
            "n_true_loss": n_true_loss,
            "cell_correct": (n_pred_loss > 0) == (n_true_loss > 0),
            "rmse": float(np.sqrt(((test["excess_pred"] - test["excess"]) ** 2).mean())),
        })

    return pd.DataFrame(fold_rows), pd.concat(pred_rows, ignore_index=True)


def confusion(y_true_loss, y_pred_loss):
    """回傳 (TP, FP, FN, TN)，以「虧損」為正類。"""
    t = np.asarray(y_true_loss, dtype=bool)
    p = np.asarray(y_pred_loss, dtype=bool)
    return int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum()), int((~t & ~p).sum())


def print_confusion(tp, fp, fn, tn, unit):
    n = tp + fp + fn + tn
    print(f"\n   混淆矩陣（正類 = 虧損）              預測虧   預測盈")
    print(f"      實際虧損                        {tp:>6} {fn:>8}")
    print(f"      實際盈利                        {fp:>6} {tn:>8}")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    bal = (recall + spec) / 2
    print(f"\n   準確率        {(tp + tn) / n:>7.1%}   ({tp + tn}/{n})")
    print(f"   虧損 recall   {recall:>7.1%}   ({tp}/{tp + fn})  ← 真正有資訊量的數字")
    print(f"   虧損 precision{prec:>7.1%}   ({tp}/{tp + fp})" if (tp + fp) else
          "   虧損 precision      —   (從未預測虧損)")
    print(f"   盈利 recall   {spec:>7.1%}   ({tn}/{tn + fp})")
    print(f"   balanced acc  {bal:>7.1%}   ← 對不平衡免疫，50% = 無資訊")
    return {"acc": (tp + tn) / n, "recall": recall, "spec": spec, "bal": bal}


def main():
    parser = ArgumentParser(description="gap* 的 Leave-One-Cell-Out 交叉驗證")
    parser.add_argument("--design", default="result/analysis/0A/pair_acc_design_matrix.csv",
                        help="regress_pair_acc.py 產生的設計矩陣")
    parser.add_argument("--outdir", default="result/analysis/0A", help="CSV 輸出目錄")
    args = parser.parse_args()

    df = pd.read_csv(args.design)
    # excess == 0 視為「不虧」（辯論與較強語言打平，不是損失）；資料中僅 1 筆
    df["excess_true_neg"] = df["excess"] < 0
    n_zero = int((df["excess"] == 0).sum())

    print("=" * 94)
    print(f"📦 {len(df)} 個 pair × {df['cell'].nunique()} 個 cell")
    print(f"   實際虧損 (excess<0)  {int(df['excess_true_neg'].sum()):>3} / {len(df)}"
          f"   ({df['excess_true_neg'].mean():.1%})"
          + (f"   [另有 {n_zero} 筆 excess=0，計為不虧]" if n_zero else ""))
    print("=" * 94)

    folds, preds = run_loo(df)

    # ---------------- 每折細節 ----------------
    print("\n📐 每折（留出一格）")
    print(f"{'留出的 cell':<25}{'a':>9}{'g':>9}{'gap*':>9}{'max gap':>10}"
          f"{'預測虧':>8}{'實際虧':>8}{'cell對?':>9}{'RMSE':>9}")
    print("-" * 94)
    for _, r in folds.iterrows():
        print(f"{r['cell']:<25}{r['a']:>9.4f}{r['g']:>9.3f}{r['gap_star']:>9.4f}"
              f"{r['max_gap']:>10.4f}{r['n_pred_loss']:>8}{r['n_true_loss']:>8}"
              f"{'O' if r['cell_correct'] else 'X':>9}{r['rmse']:>9.4f}")
    print("-" * 94)
    print(f"   gap* 跨 16 折：{folds['gap_star'].min():.4f} ~ {folds['gap_star'].max():.4f}"
          f"（平均 {folds['gap_star'].mean():.4f}，標準差 {folds['gap_star'].std():.4f}）")
    print(f"   → 標準差小代表 gap* 對留出哪一格不敏感，估計穩定")

    # ---------------- A. Cell 層級 16 次 ----------------
    print("\n" + "=" * 94)
    print("📊 A. Cell 層級：16 次預測「此格是否存在虧損 pair」")
    print("=" * 94)
    n_correct = int(folds["cell_correct"].sum())
    print(f"\n   ✅ 16 次中預測正確 {n_correct} 次  ({n_correct / 16:.1%})")
    base_cell = max(int((~folds["true_has_loss"]).sum()), int(folds["true_has_loss"].sum()))
    print(f"   對照基準（永遠猜同一類）：{base_cell}/16  ({base_cell / 16:.1%})")
    tp, fp, fn, tn = confusion(folds["true_has_loss"], folds["pred_has_loss"])
    print_confusion(tp, fp, fn, tn, "cell")

    # ---------------- B. Pair 層級 160 次 ----------------
    print("\n" + "=" * 94)
    print("📊 B. Pair 層級：160 次預測 sign(excess)")
    print("=" * 94)
    preds["pred_loss"] = preds["excess_pred"] < 0
    n_pair_correct = int((preds["pred_loss"] == preds["excess_true_neg"]).sum())
    print(f"\n   ✅ 160 次中預測正確 {n_pair_correct} 次  ({n_pair_correct / len(preds):.1%})")
    base_pair = max(int((~preds["excess_true_neg"]).sum()), int(preds["excess_true_neg"].sum()))
    print(f"   對照基準（永遠猜盈）：{base_pair}/{len(preds)}  ({base_pair / len(preds):.1%})")
    tp2, fp2, fn2, tn2 = confusion(preds["excess_true_neg"], preds["pred_loss"])
    print_confusion(tp2, fp2, fn2, tn2, "pair")

    # ---------------- C. 連續預測品質 ----------------
    print("\n" + "=" * 94)
    print("📊 C. 連續預測品質（excess_pred vs excess_true，不只看正負號）")
    print("=" * 94)
    err = preds["excess_pred"] - preds["excess"]
    ss_res = float((err ** 2).sum())
    ss_tot = float(((preds["excess"] - preds["excess"].mean()) ** 2).sum())
    r = float(np.corrcoef(preds["excess_pred"], preds["excess"])[0, 1])
    print(f"\n   RMSE            {np.sqrt((err ** 2).mean()):.5f}   "
          f"({np.sqrt((err ** 2).mean()) * 100:.3f} 個百分點)")
    print(f"   MAE             {err.abs().mean():.5f}   ({err.abs().mean() * 100:.3f} 個百分點)")
    print(f"   偏誤 (平均誤差) {err.mean():+.5f}")
    print(f"   樣本外 R²       {1 - ss_res / ss_tot:>8.4f}   ← 負值代表比「一律預測全體平均」還差")
    print(f"   Pearson r       {r:>8.4f}")
    print(f"   實測 excess 標準差 {preds['excess'].std():.5f}")

    os.makedirs(args.outdir, exist_ok=True)
    p1 = os.path.join(args.outdir, "loo_folds.csv")
    p2 = os.path.join(args.outdir, "loo_predictions.csv")
    folds.to_csv(p1, index=False, encoding="utf-8-sig")
    preds.to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"\n💾 每折摘要 → {p1}")
    print(f"💾 160 筆逐 pair 預測 → {p2}")


if __name__ == "__main__":
    main()
