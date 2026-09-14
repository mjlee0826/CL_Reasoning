"""
recovery_heterogeneity.py — 0A-7：recovery 跨格異質性（τ、I²）的 bootstrap CI

問題 (paper_status v7 §B-5):
    τ = 0.042 是兩個相近數字相減再開根號，24 格上估計誤差大，需要 bootstrap。
    ⚠️ B-5 的 τ = 0.042 / I² = 14.8% 無法用單一 SE 公式重現（B-5 自己的 Q = 59.2、df = 23 換算 I² ≈ 61%，
       與 14.8% 矛盾），這裡用下面寫明的 SE 重算，數字不會等於 B-5。

資料 (24 格 = 3 model × 4 dataset × {EN-EN, ZH-ZH}；legacy_em_samelang.csv，test_em_legacy.py 產生):
    recovery = 2·acc_debate / c − 1                        每格一個值（全樣本）
    n_R      = n_disagree × c_recoverable                  分歧題中有一方初答正確的題數
    SE       = 2·√(w(1 − w) / n_R)，w = (recovery + 1)/2   closure 下 recovery = 2w − 1 的二項 SE（= delta method）
               legacy 檔的分歧題中最後換成第三個答案而答對的只有 65 / 10108 題，closure 近乎成立

兩種估計（都報）:
    DerSimonian–Laird   v = 1/SE²；Q = Σ v(y − ȳ_v)²；I² = (Q − df)/Q；τ² = (Q − df)/(Σv − Σv²/Σv)   與 A-4 的 excess 同法
    動差法              τ² = SD² − (平均 SE)²；I² = τ²/SD²；SD 用 ddof = 1                              B-5 描述的算法
    τ²、I² 為負時截為 0。

Bootstrap:
    以 cell 為單位有放回重抽 24 格 --boot 次（每格被抽中的次數當權重），每次重算 τ 與 I²；
    95% CI = 2.5 / 97.5 百分位。boot_share_zero = 被截為 0 的比例（高時 CI 下界會貼在 0）。

輸出:
    recovery_heterogeneity_cells.csv   24 格的 recovery、n_R、SE
    recovery_heterogeneity.csv         兩種方法 × {tau, I2} 的點估計與 bootstrap CI；另附點估計的 Q / SD / mean_SE

用法:
    conda run -n clreasoning python recovery_heterogeneity.py
    conda run -n clreasoning python recovery_heterogeneity.py --legacy-csv legacy_em_samelang.csv --boot 5000 --seed 0 --outdir .
"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

CONDITIONS = ("challenge_EN", "challenge_CN")
N_CELLS = 24
METHODS = ("dersimonian_laird", "moment")
WITH_CI = ("tau", "I2")
B5 = {"tau": 0.042, "I2": 0.148}                      # paper_status v7 §B-5（無法重現，僅列出對照）


# ----------------------------------------------------------------------------
# 資料
# ----------------------------------------------------------------------------
def load_cells(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["condition"].isin(CONDITIONS)].reset_index(drop=True)
    if len(df) != N_CELLS:
        raise ValueError(f"{path}: {CONDITIONS} 應有 {N_CELLS} 格，實際 {len(df)} 格")

    n_R = df["n_disagree"] * df["c_recoverable"]
    if not np.allclose(n_R, n_R.round(), rtol=0, atol=1e-6):
        raise ValueError(f"{path}: n_disagree × c_recoverable 不是整數")
    cells = pd.DataFrame({
        "cell": df["condition"] + "|" + df["model"] + "|" + df["dataset"],
        "recovery": df["recovery"].astype(float),
        "n_R": n_R.round().astype(int),
    })
    w = (cells["recovery"] + 1) / 2
    cells["se"] = 2 * np.sqrt(w * (1 - w) / cells["n_R"])
    return cells


# ----------------------------------------------------------------------------
# τ 與 I²：給一組 cell 權重算一次，bootstrap 只是換權重
# ----------------------------------------------------------------------------
def heterogeneity(y, se, weights):
    """
    y, se  : 每格的 recovery 與 SE, (K,)
    weights: 每列是一組 cell 被抽中的次數（全 1 = 原樣本）, (B, K)
    回傳 {method: {quantity: (B,)}}
    """
    k = weights.sum(axis=1)
    df = k - 1
    out = {}

    # DerSimonian–Laird
    v = 1 / se ** 2
    sv, svy, svyy, svv = (weights @ a for a in (v, v * y, v * y * y, v * v))
    q = svyy - svy ** 2 / sv
    out["dersimonian_laird"] = {
        "tau": np.sqrt(np.maximum((q - df) / (sv - svv / sv), 0)),
        "I2": np.maximum((q - df) / q, 0),
        "Q": q,
    }

    # 動差法
    mean = (weights @ y) / k
    var = (weights @ (y * y) - k * mean ** 2) / df
    mean_se = (weights @ se) / k
    tau2 = np.maximum(var - mean_se ** 2, 0)
    out["moment"] = {"tau": np.sqrt(tau2), "I2": tau2 / var, "SD": np.sqrt(var), "mean_SE": mean_se}
    return out


def summarize(cells, boot, seed):
    y, se = cells["recovery"].to_numpy(dtype=float), cells["se"].to_numpy(dtype=float)
    K = len(y)
    point = heterogeneity(y, se, np.ones((1, K)))

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(K, np.full(K, 1 / K), size=boot)                 # (boot, K)
    booted = heterogeneity(y, se, weights)

    rows = []
    for method in METHODS:
        for quantity, values in point[method].items():
            draws = booted[method][quantity]
            with_ci = quantity in WITH_CI
            rows.append({
                "method": method,
                "quantity": quantity,
                "estimate": float(values[0]),
                "ci_lo": float(np.percentile(draws, 2.5)) if with_ci else np.nan,
                "ci_hi": float(np.percentile(draws, 97.5)) if with_ci else np.nan,
                "boot_share_zero": float(np.mean(draws == 0)) if with_ci else np.nan,
                "boot": boot,
                "seed": seed,
            })
    return pd.DataFrame(rows)


def print_report(cells, summary):
    print("=" * 96)
    print(f"📦 {len(cells)} 格 recovery：平均 {cells['recovery'].mean():.4f}、SD {cells['recovery'].std(ddof=1):.4f}、"
          f"平均 SE {cells['se'].mean():.4f}（n_R {cells['n_R'].min()}–{cells['n_R'].max()}）")
    print("=" * 96)
    print(f"{'method':<20} {'quantity':<9} {'estimate':>9} {'bootstrap 95% CI':>24} {'截為 0':>9}")
    print("-" * 96)
    for _, r in summary.iterrows():
        ci = f"[{r['ci_lo']:>8.4f}, {r['ci_hi']:>8.4f}]" if np.isfinite(r["ci_lo"]) else ""
        zero = f"{r['boot_share_zero']:.1%}" if np.isfinite(r["boot_share_zero"]) else ""
        print(f"{r['method']:<20} {r['quantity']:<9} {r['estimate']:>9.4f} {ci:>24} {zero:>9}")
    print(f"\n   B-5 原值 τ = {B5['tau']}、I² = {B5['I2']:.1%}（無法重現，見 docstring）")
    print(f"   bootstrap：以 cell 為單位重抽 {int(summary['boot'].iloc[0])} 次（seed={int(summary['seed'].iloc[0])}），CI = 2.5 / 97.5 百分位")


# ----------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="0A-7: recovery 跨格異質性 (τ、I²) 的 bootstrap CI")
    parser.add_argument("--legacy-csv", default="legacy_em_samelang.csv", help="test_em_legacy.py 的輸出")
    parser.add_argument("--boot", type=int, default=5000, help="cell bootstrap 次數")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap 的 seed")
    parser.add_argument("--outdir", default=".", help="CSV 輸出目錄")
    args = parser.parse_args()

    cells = load_cells(args.legacy_csv)
    summary = summarize(cells, args.boot, args.seed)
    print_report(cells, summary)

    os.makedirs(args.outdir, exist_ok=True)
    print()
    for frame, name in ((cells, "recovery_heterogeneity_cells.csv"), (summary, "recovery_heterogeneity.csv")):
        path = os.path.join(args.outdir, name)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"💾 {name} ({len(frame)} 列) → {path}")


if __name__ == "__main__":
    main()
