"""
regress_pair_acc.py — 配對辯論成績 vs 兩個成員語言的單語成績

問題:
    控制住「這個 model 在這個 dataset 上大概多強」之後,
    一個 language pair 的辯論成績, 還能不能被它兩個成員語言的單語成績解釋?

資料 (160 格 = 4 model x 4 dataset x 10 pair):
    pair_acc  <- result/challenge/{model}_{dataset}_challenge_{l1}_vs_{l2}.json  的 ExactMatch_Accuracy
    mono      <- result/baseline/{model}_{dataset}_onelanguage_{lang}.json       的 ExactMatch_Accuracy
    max_mono = max(mono[l1], mono[l2])
    min_mono = min(mono[l1], mono[l2])
    gap      = max_mono - min_mono

參數化:
    主  pair_acc ~ max_mono + gap
    輔  pair_acc ~ max_mono + min_mono
    兩者是線性重參數化 (gap = max - min), 配適度完全相同, 係數關係:
        B_gap = -b_min          B_max = b_max + b_min
    因此四個假設在 (max, gap) 下變得乾淨:
        b_max = b_min = 0    <=>  B_max = 0 且 B_gap = 0
        b_max = 1, b_min = 0 <=>  B_max = 1 且 B_gap = 0
        b_max + b_min = 1    <=>  B_max = 1
        b_min = 0            <=>  B_gap = 0

估計策略 (兩階段, 取代單一 pooled 迴歸 + 16 群 cluster SE):
    第一階段  在 16 個 cell 各跑一次迴歸 (每格 10 obs, 2 regressors, df=7)
    第二階段  對這 16 組估計值做單樣本 t 檢定 (df=15) 與符號檢定
    好處: 不必假設斜率跨 cell 同質, 也避開 16 群 cluster SE 向下偏誤的問題。

另外跑:
    超額模型  excess ~ gap,  excess = pair_acc - max_mono
              截距 = gap=0 時, 辯論相對「直接用較強語言」的加成。H0: 截距 = 0
    TOST      對 B_gap (等價於 -b_min) 做等價檢定, 判斷是否可宣稱「效應可忽略」

用法 (需要 statsmodels, 在 clreasoning 環境):
    conda run -n clreasoning python scripts/analysis_0A/regress_pair_acc.py
    conda run -n clreasoning python scripts/analysis_0A/regress_pair_acc.py --delta 0.15
    conda run -n clreasoning python scripts/analysis_0A/regress_pair_acc.py --outdir out/
"""

import glob
import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

BASELINE_DIR = "result/baseline"
CHALLENGE_DIR = "result/challenge"


# ----------------------------------------------------------------------------
# 資料組裝
# ----------------------------------------------------------------------------
def _meta(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)[0]


def load_mono():
    """{(model, dataset, language): accuracy}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(BASELINE_DIR, "*.json"))):
        m = _meta(path)
        if "ExactMatch_Accuracy" not in m:
            print(f"⚠️  {os.path.basename(path)} 缺 ExactMatch_Accuracy，已略過")
            continue
        langs = m.get("Strategy", {}).get("languages", [])
        if len(langs) != 1:
            continue
        key = (m["Model"]["modelType"], m["Dataset"]["datasetType"], langs[0])
        out[key] = m["ExactMatch_Accuracy"]
    return out


def load_pairs():
    """[(model, dataset, l1, l2, accuracy), ...]"""
    rows = []
    for path in sorted(glob.glob(os.path.join(CHALLENGE_DIR, "*.json"))):
        m = _meta(path)
        if "ExactMatch_Accuracy" not in m:
            print(f"⚠️  {os.path.basename(path)} 缺 ExactMatch_Accuracy，已略過")
            continue
        langs = m.get("Strategy", {}).get("languages", [])
        if len(langs) != 2:
            continue
        rows.append((m["Model"]["modelType"], m["Dataset"]["datasetType"],
                     langs[0], langs[1], m["ExactMatch_Accuracy"]))
    return rows


def build_design():
    mono = load_mono()
    pairs = load_pairs()

    records, missing = [], []
    for model, dataset, l1, l2, acc in pairs:
        a1 = mono.get((model, dataset, l1))
        a2 = mono.get((model, dataset, l2))
        if a1 is None or a2 is None:
            missing.append((model, dataset, l1, l2))
            continue
        records.append({
            "model": model,
            "dataset": dataset,
            "cell": f"{model}|{dataset}",
            "l1": l1,
            "l2": l2,
            "pair": f"{l1}_vs_{l2}",
            "pair_acc": acc,
            "mono_l1": a1,
            "mono_l2": a2,
            "max_mono": max(a1, a2),
            "min_mono": min(a1, a2),
            "gap": abs(a1 - a2),
            "mean_mono": (a1 + a2) / 2,
            "excess": acc - max(a1, a2),
        })

    if missing:
        print(f"⚠️  有 {len(missing)} 個 pair 找不到對應的單語 baseline，已略過：{missing[:5]}")

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# 識別力前置檢查
# ----------------------------------------------------------------------------
def identification_check(df):
    print("\n" + "=" * 92)
    print("🔎 識別力前置檢查  — 固定效應吸收 cell 後，識別完全來自 cell 內 10 個 pair 的變異")
    print("=" * 92)
    print(f"{'cell':<26} {'n':>3} {'max 相異':>9} {'gap 相異':>9} "
          f"{'sd(max)':>9} {'sd(gap)':>9} {'r(max,gap)':>11}")
    print("-" * 92)

    weak = []
    for cell, g in df.groupby("cell", sort=True):
        r = np.corrcoef(g["max_mono"], g["gap"])[0, 1] if len(g) > 2 else np.nan
        print(f"{cell:<26} {len(g):>3} {g['max_mono'].nunique():>9} {g['gap'].nunique():>9} "
              f"{g['max_mono'].std():>9.4f} {g['gap'].std():>9.4f} {r:>11.3f}")
        if g["max_mono"].nunique() < 3 or g["gap"].std() < 1e-6:
            weak.append(cell)

    print("\n   說明：5 個單語成績 a1>a2>a3>a4>a5 之下，max_mono 天生只有 4 個相異值")
    print("   （含最強語言的 4 個 pair 共用 a1，其餘依序 3/2/1 個）。這是結構性的上限，不是資料問題。")
    if weak:
        print(f"   ⚠️  識別力偏弱的 cell：{weak}（係數與 SE 不可靠）")
    else:
        print("   ✅ 所有 cell 都有足夠的 cell 內變異")


# ----------------------------------------------------------------------------
# 第一階段：16 個 cell 各跑一次迴歸
# ----------------------------------------------------------------------------
def stage1(df):
    rows = []
    for cell, g in df.groupby("cell", sort=True):
        g = g.copy()

        # 主參數化：pair_acc ~ max_mono + gap        (df = 10 - 3 = 7)
        m = smf.ols("pair_acc ~ max_mono + gap", data=g).fit()
        # 輔參數化：pair_acc ~ max_mono + min_mono   (同一個配適)
        a = smf.ols("pair_acc ~ max_mono + min_mono", data=g).fit()
        # 超額模型：excess ~ gap                     (df = 10 - 2 = 8)
        e = smf.ols("excess ~ gap", data=g).fit()

        rows.append({
            "cell": cell,
            "model": g["model"].iloc[0],
            "dataset": g["dataset"].iloc[0],
            "n": len(g),
            # --- 主 (max, gap) ---
            "B_max": m.params["max_mono"],
            "B_max_se": m.bse["max_mono"],
            "B_gap": m.params["gap"],
            "B_gap_se": m.bse["gap"],
            "r2": m.rsquared,
            "resid_sd": np.sqrt(m.mse_resid),
            # 每個 cell 內的假設檢定 p 值
            "p_joint_zero": float(m.f_test("max_mono = 0, gap = 0").pvalue),
            "p_joint_pure_max": float(m.f_test("max_mono = 1, gap = 0").pvalue),
            "p_Bmax_eq1": float(m.t_test("max_mono = 1").pvalue),
            "p_Bgap_eq0": m.pvalues["gap"],
            # --- 輔 (max, min) ---
            "b_max": a.params["max_mono"],
            "b_min": a.params["min_mono"],
            "b_sum": a.params["max_mono"] + a.params["min_mono"],
            # --- 超額模型 ---
            "a_excess": e.params["Intercept"],
            "a_excess_se": e.bse["Intercept"],
            "g_excess": e.params["gap"],
            # 描述統計
            "mean_pair_acc": g["pair_acc"].mean(),
            "mean_excess": g["excess"].mean(),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 第二階段：對 16 個估計值做檢定
# ----------------------------------------------------------------------------
def stage2_test(values, null, label):
    """單樣本 t 檢定 (df = k-1) + 符號檢定。"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    k = len(v)
    mean, sd = v.mean(), v.std(ddof=1)
    se = sd / np.sqrt(k)
    t = (mean - null) / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), k - 1) if se > 0 else np.nan
    crit = stats.t.ppf(0.975, k - 1)
    ci = (mean - crit * se, mean + crit * se)

    above = int((v > null).sum())
    ties = int((v == null).sum())
    n_eff = k - ties
    p_sign = stats.binomtest(above, n_eff, 0.5).pvalue if n_eff > 0 else np.nan

    return {
        "label": label, "null": null, "k": k,
        "mean": mean, "sd": sd, "se": se,
        "t": t, "p": p, "ci_lo": ci[0], "ci_hi": ci[1],
        "above": above, "n_eff": n_eff, "p_sign": p_sign,
    }


def print_stage2(results):
    print(f"\n{'量':<34} {'H0':>5} {'平均':>9} {'SE':>8} {'95% CI':>19} "
          f"{'t':>8} {'p':>9} {'符號':>8} {'p(符號)':>9}")
    print("-" * 122)
    for r in results:
        star = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        print(f"{r['label']:<34} {r['null']:>5.0f} {r['mean']:>9.4f} {r['se']:>8.4f} "
              f"[{r['ci_lo']:>7.4f},{r['ci_hi']:>8.4f}] {r['t']:>8.3f} "
              f"{r['p']:>9.4f}{star:<3} {r['above']:>3}/{r['n_eff']:<4} {r['p_sign']:>9.4f}")


# ----------------------------------------------------------------------------
# TOST 等價檢定
# ----------------------------------------------------------------------------
def tost(values, delta, alpha=0.05, label=""):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    k = len(v)
    mean, se = v.mean(), v.std(ddof=1) / np.sqrt(k)
    df = k - 1

    # H0a: mu <= -delta  (上尾)   H0b: mu >= +delta  (下尾)
    p_lower = stats.t.sf((mean + delta) / se, df)
    p_upper = stats.t.cdf((mean - delta) / se, df)
    p_tost = max(p_lower, p_upper)

    # 能宣稱等價的最小 delta：|mean| + t_{1-alpha,df} * se
    delta_min = abs(mean) + stats.t.ppf(1 - alpha, df) * se

    print(f"\n🔬 TOST 等價檢定 — {label}")
    print(f"   等價界限 δ = ±{delta:.3f}   平均 = {mean:+.4f}   SE = {se:.4f}   df = {df}")
    print(f"   p(下界) = {p_lower:.4f}   p(上界) = {p_upper:.4f}   → TOST p = {p_tost:.4f}")
    if p_tost < alpha:
        print(f"   ✅ 在 α={alpha} 下可宣稱等價：效應落在 (−{delta}, +{delta}) 內，可視為可忽略")
    else:
        print(f"   ❌ 無法宣稱等價（也不代表有效應，可能只是檢定力不足）")
    print(f"   能宣稱等價的最小界限 δ_min = {delta_min:.4f}")
    print(f"   → 你只能說「效應絕對值小於 {delta_min:.4f}」，無法排除比這更小的真實效應")
    return {"delta": delta, "p_tost": p_tost, "delta_min": delta_min, "mean": mean, "se": se}


# ----------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="配對辯論成績的兩階段迴歸分析")
    parser.add_argument("--delta", type=float, default=0.10,
                        help="TOST 等價界限（B_gap 的係數尺度，預設 0.10）")
    parser.add_argument("--outdir", default="result/analysis/0A", help="CSV 輸出目錄")
    args = parser.parse_args()

    df = build_design()
    if df.empty:
        print("❌ 沒有組出任何資料")
        return

    print("=" * 92)
    print(f"📦 設計矩陣：{len(df)} 列 = {df['cell'].nunique()} 個 cell × "
          f"{len(df) // max(df['cell'].nunique(), 1)} 個 pair")
    print(f"   pair_acc  範圍 {df['pair_acc'].min():.4f} ~ {df['pair_acc'].max():.4f}")
    print(f"   max_mono  範圍 {df['max_mono'].min():.4f} ~ {df['max_mono'].max():.4f}")
    print(f"   gap       範圍 {df['gap'].min():.4f} ~ {df['gap'].max():.4f}")
    print(f"   excess    平均 {df['excess'].mean():+.4f}  (= pair_acc − max_mono)")
    print("=" * 92)

    identification_check(df)

    est = stage1(df)

    print("\n" + "=" * 92)
    print("📐 第一階段：16 個 cell 各自的迴歸  pair_acc ~ max_mono + gap   (每格 10 obs, df=7)")
    print("=" * 92)
    print(f"{'cell':<26} {'B_max':>8} {'(se)':>8} {'B_gap':>9} {'(se)':>8} "
          f"{'R²':>7} {'a_excess':>10} {'(se)':>8}")
    print("-" * 92)
    for _, r in est.iterrows():
        print(f"{r['cell']:<26} {r['B_max']:>8.3f} {r['B_max_se']:>8.3f} "
              f"{r['B_gap']:>9.3f} {r['B_gap_se']:>8.3f} {r['r2']:>7.3f} "
              f"{r['a_excess']:>10.4f} {r['a_excess_se']:>8.4f}")

    print("\n" + "=" * 92)
    print("🧪 第二階段：對 16 個估計值做單樣本 t 檢定 (df=15) 與符號檢定")
    print("=" * 92)

    tests = [
        stage2_test(est["B_max"], 0, "B_max  (主) 較強語言傳導係數"),
        stage2_test(est["B_max"], 1, "B_max  (主) 對 1 —— 純 max 傳導"),
        stage2_test(est["B_gap"], 0, "B_gap  (主) ⇔ −b_min，語言差距效應"),
        stage2_test(est["a_excess"], 0, "a_excess 截距 —— 辯論加成"),
        stage2_test(est["b_max"], 0, "b_max  (輔)"),
        stage2_test(est["b_min"], 0, "b_min  (輔) 較弱語言傳導係數"),
        stage2_test(est["b_sum"], 1, "b_max+b_min (輔) 對 1 —— 凸組合"),
    ]
    print_stage2(tests)

    # ---- 聯合假設：彙總 16 個 cell 內的 F 檢定 ----
    print("\n" + "=" * 92)
    print("🧮 聯合假設：彙總 16 個 cell 內的 F 檢定")
    print("=" * 92)
    for col, name in [("p_joint_zero", "H0: B_max = 0 且 B_gap = 0   (配哪兩個語言完全不重要)"),
                      ("p_joint_pure_max", "H0: B_max = 1 且 B_gap = 0   (辯論 = 直接拿較強語言)")]:
        p = est[col].to_numpy(dtype=float)
        n_rej = int((p < 0.05).sum())
        fisher = -2 * np.log(np.clip(p, 1e-300, None)).sum()
        p_fisher = stats.chi2.sf(fisher, 2 * len(p))
        print(f"\n{name}")
        print(f"   16 個 cell 中在 α=0.05 下拒絕的個數：{n_rej} / 16   "
              f"（H0 全真時期望約 0.8 個）")
        print(f"   Fisher 合併 p = {p_fisher:.4g}   （⚠️ 16 個 cell 共用模型/資料集，並非完全獨立）")

    # ---- TOST ----
    print("\n" + "=" * 92)
    tost(est["B_gap"], args.delta, label="B_gap = 0（等價於 b_min = 0）")

    sd_gap = df.groupby("cell")["gap"].std().mean()
    print(f"\n   尺度換算：cell 內 gap 的平均標準差 = {sd_gap:.4f}")
    print(f"   → 係數 δ={args.delta} 相當於 pair_acc 變動約 "
          f"{args.delta * sd_gap * 100:.3f} 個百分點")

    # ---- 輸出 ----
    os.makedirs(args.outdir, exist_ok=True)
    p1 = os.path.join(args.outdir, "pair_acc_design_matrix.csv")
    p2 = os.path.join(args.outdir, "pair_acc_cell_estimates.csv")
    df.to_csv(p1, index=False, encoding="utf-8-sig")
    est.to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"\n💾 設計矩陣 ({len(df)} 列) → {p1}")
    print(f"💾 第一階段估計 ({len(est)} 列) → {p2}")


if __name__ == "__main__":
    main()
