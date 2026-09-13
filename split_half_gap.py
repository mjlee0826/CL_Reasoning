"""
split_half_gap.py — 0A-4：用 split-half 重新估計 excess ~ gap（去除耦合與選擇偏誤）

問題 (paper_status A-2):
    全樣本上 excess = pair_acc − max_mono 與 gap = max_mono − min_mono 共用同一個 max_mono 的量測誤差,
    而且「哪個語言是 max」也是在同一批題目上選的 → 斜率同時帶有耦合 (Oldham 1962) 與 winner's curse。

做法 (每個 cell = model × dataset, 重複 --reps 次):
    1. 題目隨機切成 H1 / H2 (2000 → 1000 / 1000；TruthfulQA 817 → 408 / 409)
    2. H1: 五個語言各自的準確率；每個配對在自己的兩個語言中選出 L_max / L_min      ← 選擇
           gap_H1 = acc(L_max, H1) − acc(L_min, H1)                                ← 自變數
    3. H2: excess_H2 = pair_acc(H2) − acc(L_max, H2)，語言身分沿用 H1，不重選         ← 應變數
    4. 每次記錄 160 個 (gap_H1, excess_H2)
    5. 每次各跑三種規格的 excess_H2 ~ gap_H1，再對 reps 次取平均:
           pooled OLS      160 點一條線
           cell FE         cell 內去平均後的共同斜率 (截距 = 各 cell 截距的平均, 同 loo_gap_star.py)
           per-cell 平均   16 格各自 10 點的斜率 / 截距再取平均
       切分變動: reps 次估計值的 2.5–97.5 百分位 (不是信賴區間)
       95% CI : 對 16 個 cell 有放回重抽 --boot 次 (沿用同一組切分), 取 2.5 / 97.5 百分位
    全樣本 (不切分) 的三種規格一併列出, 應重現 A-2 的 −0.383 / −0.373 / −0.412。

與 0A-1 (Test/TestRecoveryBlind.py) 的關係:
    共用 Test.makeSplits / Test.pickMax, 切分與錨點 (= L_max) 完全相同, 所以每一對、每一次都必須滿足
        excess_H2 = d_H2 · (c_H2/2) · (recovery_H2 − recovery_blind_H2)
    本腳本逐點 assert 這條恆等式; 若 challenge 檔已跑過 TestRecoveryBlind, 也核對 metadata 中的
    recovery_blind_H2 / recovery_H2 與這裡重算的 reps 次平均是否一致。

資料:
    result/baseline/*.json    五個語言的單語結果 (逐題 MyAnswer = 辯論的初答)
    result/challenge/*.json   十個語言配對的辯論結果 (逐題最終 MyAnswer)

輸出:
    split_half_points.csv   cell × rep × pair 的 (L_max, L_min, gap_H1, excess_H2)
    split_half_slopes.csv   三種規格 × {slope, intercept} 的全樣本值、split-half 平均、切分變動、bootstrap CI

用法:
    conda run -n clreasoning python split_half_gap.py
    conda run -n clreasoning python split_half_gap.py --reps 200 --boot 5000 --seed 0 --boot-seed 0 --outdir .
"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from File.FileFactory import FileFactory
from Strategy.StrategyType import LANGUAGE_STR_LIST
from Test.Test import Test

BASELINE_DIR = "result/baseline"
CHALLENGE_DIR = "result/challenge"
VARIANTS = ("pooled_ols", "cell_fe", "per_cell_mean")
A2_FULL_SAMPLE_SLOPE = {"pooled_ols": -0.383, "cell_fe": -0.373, "per_cell_mean": -0.412}


# ----------------------------------------------------------------------------
# 資料組裝：逐檔載入，只保留逐題的對錯與初答
# ----------------------------------------------------------------------------
def extract_baseline():
    """{(model, dataset): {language: (correct_map, answer_map, DatasetClass)}}"""
    out = {}
    for f in FileFactory().iterFileInDir(BASELINE_DIR):
        langs = f.getLanguage()
        if f.getStrategyConfig().strategyType != "onelanguage" or len(langs) != 1:
            continue
        key = (f.getModelConfig().modelType, f.getDatasetConfig().datasetType)
        answers = {q_id: str(r.get("MyAnswer", "")) for q_id, r in f.records_map.items()}
        out.setdefault(key, {})[langs[0]] = (Test.getCorrectMap(f), answers, Test.getDatasetClass(f))
    return out


def extract_challenge(model, dataset):
    """{(l1, l2): (correct_map, metadata["RecoveryBlind"] 或 None)}，只讀這個 cell 的檔案。"""
    out = {}
    for f in FileFactory().iterFileInDir(CHALLENGE_DIR, f"{model}_{dataset}_challenge_*.json"):
        if (f.getModelConfig().modelType, f.getDatasetConfig().datasetType) != (model, dataset):
            continue
        l1, l2 = f.getLanguage()
        out[(l1, l2)] = (Test.getCorrectMap(f), f.metadata.get("RecoveryBlind"))
    return out


def build_cell(cell, base, chal):
    """一個 cell 轉成依題目 id 排序的陣列：mono (5, N)、pairs {(l1, l2): (N,)}、dis {(l1, l2): (N,)}。"""
    if set(base) != set(LANGUAGE_STR_LIST):
        raise ValueError(f"{cell}: baseline 語言不齊 {sorted(base)}")
    if len(chal) != 10:
        raise ValueError(f"{cell}: challenge 配對應為 10 個，實際 {len(chal)} 個")

    ids = sorted(base[LANGUAGE_STR_LIST[0]][0])
    sources = [(lang, v[0]) for lang, v in base.items()] + [(pair, v[0]) for pair, v in chal.items()]
    for name, correct in sources:
        if sorted(correct) != ids:
            raise ValueError(f"{cell}: {name} 的題目 id 與其他檔案不一致")

    DatasetClass = base[LANGUAGE_STR_LIST[0]][2]
    # mono 第 i 列 = LANGUAGE_STR_LIST[i]，與 Test.pickMax 使用的 key 相同
    mono = np.array([[base[lang][0][q] for q in ids] for lang in LANGUAGE_STR_LIST], dtype=bool)
    pairs, dis = {}, {}
    for (l1, l2), (correct, _) in sorted(chal.items()):
        pairs[(l1, l2)] = np.array([correct[q] for q in ids], dtype=bool)
        a1, a2 = base[l1][1], base[l2][1]
        dis[(l1, l2)] = np.array([not DatasetClass.compareTwoAnswer(a1[q], a2[q]) for q in ids], dtype=bool)
    return ids, mono, pairs, dis


# ----------------------------------------------------------------------------
# split-half 與全樣本的點
# ----------------------------------------------------------------------------
def run_splits(cell, mono, pairs, dis, reps, seed):
    """回傳 (points, 核對 0A-1 用的逐次 recovery 值, 恆等式成立的點數)。"""
    rows = []
    checks = {pair: {"recovery_blind": [], "recovery": []} for pair in pairs}
    n_identity = 0

    for rep, h1 in enumerate(Test.makeSplits(mono.shape[1], reps, seed)):
        h2 = ~h1
        for (l1, l2), final in pairs.items():
            i, j = LANGUAGE_STR_LIST.index(l1), LANGUAGE_STR_LIST.index(l2)
            first = Test.pickMax(int(mono[i, h1].sum()), int(mono[j, h1].sum()), seed, rep, i, j)  # H1 選擇
            hi, lo = (i, j) if first else (j, i)
            gap_H1 = mono[hi, h1].mean() - mono[lo, h1].mean()
            excess_H2 = final[h2].mean() - mono[hi, h2].mean()                                    # H2 測量

            s = Test.recoveryStats(mono[hi], mono[lo], final, dis[(l1, l2)], h2)
            if s["n_A"] + s["n_B"] > 0:
                identity = s["d"] * s["c"] / 2 * (s["recovery"] - s["recovery_blind"])
                assert abs(excess_H2 - identity) < 1e-12, (cell, rep, l1, l2, excess_H2, identity)
                n_identity += 1
            checks[(l1, l2)]["recovery_blind"].append(s["recovery_blind"])
            checks[(l1, l2)]["recovery"].append(s["recovery"])

            rows.append({
                "cell": cell, "rep": rep, "pair": f"{l1}_vs_{l2}", "l1": l1, "l2": l2,
                "L_max": LANGUAGE_STR_LIST[hi], "L_min": LANGUAGE_STR_LIST[lo],
                "gap_H1": gap_H1, "excess_H2": excess_H2,
            })
    return rows, checks, n_identity


def full_sample_points(cell, mono, pairs):
    """不切分：gap = |acc_l1 − acc_l2|、excess = pair_acc − max，與 pair_acc_design_matrix.csv 同定義。"""
    acc = mono.mean(axis=1)
    rows = []
    for (l1, l2), final in pairs.items():
        a1, a2 = acc[LANGUAGE_STR_LIST.index(l1)], acc[LANGUAGE_STR_LIST.index(l2)]
        rows.append({"cell": cell, "rep": 0, "pair": f"{l1}_vs_{l2}",
                     "gap": abs(a1 - a2), "excess": final.mean() - max(a1, a2)})
    return rows


def compare_with_metadata(checks, chal_meta, reps, seed):
    """核對 TestRecoveryBlind 寫進 metadata 的 H2 平均。回傳 (比對的值數, 不一致數)。"""
    compared = mismatched = 0
    for pair, values in checks.items():
        meta = chal_meta.get(pair)
        if not meta or meta.get("reps") != reps or meta.get("seed") != seed:
            continue
        for key in ("recovery_blind", "recovery"):
            arr = np.array(values[key], dtype=float)
            mine = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
            theirs = meta.get(f"{key}_H2")
            compared += 1
            if theirs is None or not np.isclose(mine, theirs, rtol=0, atol=1e-9, equal_nan=True):
                mismatched += 1
    return compared, mismatched


# ----------------------------------------------------------------------------
# 迴歸：用每個 (rep, cell) 的充分統計量算三種規格，bootstrap 只是換 cell 權重
# ----------------------------------------------------------------------------
def sufficient_stats(points, cells, n_reps, x_col, y_col):
    """每個 (rep, cell) 的 n, Σx, Σy, Σxx, Σxy，shape = (n_reps, n_cells)。"""
    cell_index = {c: k for k, c in enumerate(cells)}
    r = points["rep"].to_numpy()
    c = points["cell"].map(cell_index).to_numpy()
    x = points[x_col].to_numpy(dtype=float)
    y = points[y_col].to_numpy(dtype=float)
    stats = {}
    for key, v in (("n", np.ones_like(x)), ("sx", x), ("sy", y), ("sxx", x * x), ("sxy", x * y)):
        stats[key] = np.zeros((n_reps, len(cells)))
        np.add.at(stats[key], (r, c), v)
    return stats


def estimate(stats, weights):
    """
    stats  : 每個 (rep, cell) 的充分統計量, (reps, C)
    weights: 每列是一組 cell 權重 (bootstrap 重抽次數；全 1 = 原樣本), (B, C)
    回傳 {variant: {"slope": (reps, B), "intercept": (reps, B)}}
    """
    W = weights.T                                                   # (C, B)
    n_cells = weights.sum(axis=1)                                   # (B,)
    out = {}

    # pooled OLS：所有點一條線
    n, sx, sy, sxx, sxy = (stats[k] @ W for k in ("n", "sx", "sy", "sxx", "sxy"))
    slope = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    out["pooled_ols"] = {"slope": slope, "intercept": (sy - slope * sx) / n}

    # cell 內去平均
    within_xx = stats["sxx"] - stats["sx"] ** 2 / stats["n"]
    within_xy = stats["sxy"] - stats["sx"] * stats["sy"] / stats["n"]
    x_bar, y_bar = stats["sx"] / stats["n"], stats["sy"] / stats["n"]

    # cell FE：共同斜率，截距 = 各 cell 截距 (ȳ_c − slope·x̄_c) 的平均
    slope = (within_xy @ W) / (within_xx @ W)
    out["cell_fe"] = {"slope": slope, "intercept": ((y_bar @ W) - slope * (x_bar @ W)) / n_cells}

    # per-cell 平均：每個 cell 自己的斜率 / 截距再平均
    cell_slope = within_xy / within_xx
    cell_intercept = y_bar - cell_slope * x_bar
    out["per_cell_mean"] = {"slope": (cell_slope @ W) / n_cells, "intercept": (cell_intercept @ W) / n_cells}
    return out


def summarize(points, full, cells, reps, boot, boot_seed):
    stats_split = sufficient_stats(points, cells, reps, "gap_H1", "excess_H2")
    stats_full = sufficient_stats(full, cells, 1, "gap", "excess")
    for name, stats in (("split-half", stats_split), ("全樣本", stats_full)):
        if (stats["sxx"] - stats["sx"] ** 2 / stats["n"] <= 0).any():
            raise ValueError(f"{name}: 有 cell 的 gap 完全沒有變異，per-cell 斜率無定義")

    ones = np.ones((1, len(cells)))
    est_split = estimate(stats_split, ones)                         # (reps, 1)
    est_full = estimate(stats_full, ones)                           # (1, 1)

    rng = np.random.default_rng(boot_seed)
    weights = rng.multinomial(len(cells), np.full(len(cells), 1 / len(cells)), size=boot)   # (boot, C)
    est_boot = estimate(stats_split, weights)                       # (reps, boot)

    rows = []
    for variant in VARIANTS:
        for coef in ("slope", "intercept"):
            per_rep = est_split[variant][coef][:, 0]
            boot_means = est_boot[variant][coef].mean(axis=0)       # 每次重抽：對 reps 取平均
            rows.append({
                "variant": variant,
                "coef": coef,
                "full_sample": float(est_full[variant][coef][0, 0]),
                "split_half_mean": float(per_rep.mean()),
                "split_p2_5": float(np.percentile(per_rep, 2.5)),
                "split_p97_5": float(np.percentile(per_rep, 97.5)),
                "boot_ci_lo": float(np.percentile(boot_means, 2.5)),
                "boot_ci_hi": float(np.percentile(boot_means, 97.5)),
                "reps": reps,
                "boot": boot,
            })
    return pd.DataFrame(rows)


def print_summary(summary):
    print("\n" + "=" * 104)
    print("📐 excess ~ gap：全樣本 vs split-half (gap 用 H1、excess 用 H2)")
    print("=" * 104)
    print(f"{'variant':<15} {'coef':<10} {'full sample':>12} {'split-half':>11} "
          f"{'split 2.5% ~ 97.5%':>22} {'bootstrap 95% CI':>22}")
    print("-" * 104)
    for _, r in summary.iterrows():
        print(f"{r['variant']:<15} {r['coef']:<10} {r['full_sample']:>12.4f} {r['split_half_mean']:>11.4f} "
              f"   [{r['split_p2_5']:>8.4f}, {r['split_p97_5']:>8.4f}]   [{r['boot_ci_lo']:>8.4f}, {r['boot_ci_hi']:>8.4f}]")
    print("\n   split 2.5% ~ 97.5% 是切分造成的變動，不是信賴區間；信賴區間看 bootstrap（對 cell 重抽）。")
    print("   ⚠️ gap_H1 只用一半題目，雜訊較大，斜率本來就會往 0 縮；與全樣本的差距不能全部歸因於去除耦合。")

    full = summary[summary["coef"] == "slope"].set_index("variant")["full_sample"]
    print("\n   全樣本斜率 vs paper_status A-2：" + "  ".join(
        f"{v} {full[v]:.3f} (A-2 {A2_FULL_SAMPLE_SLOPE[v]:.3f})" for v in VARIANTS))


# ----------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="0A-4: split-half 重新估計 excess ~ gap")
    parser.add_argument("--reps", type=int, default=200, help="切分次數")
    parser.add_argument("--seed", type=int, default=0, help="切分 seed（與 TestRecoveryBlind 相同才能核對 metadata）")
    parser.add_argument("--boot", type=int, default=5000, help="cell-cluster bootstrap 次數")
    parser.add_argument("--boot-seed", type=int, default=0, help="bootstrap 的 seed")
    parser.add_argument("--outdir", default=".", help="CSV 輸出目錄")
    args = parser.parse_args()

    print("📦 讀取 baseline（逐檔載入）...")
    baseline = extract_baseline()
    cell_keys = sorted(baseline)
    cells = [f"{model}|{dataset}" for model, dataset in cell_keys]
    print(f"   {len(cells)} 個 cell")

    print(f"\n✂️  split-half：每個 cell 切 {args.reps} 次（seed={args.seed}）")
    split_rows, full_rows = [], []
    n_identity = n_compared = n_mismatched = 0
    for (model, dataset), cell in zip(cell_keys, cells):
        chal = extract_challenge(model, dataset)
        ids, mono, pairs, dis = build_cell(cell, baseline[(model, dataset)], chal)
        rows, checks, identity_points = run_splits(cell, mono, pairs, dis, args.reps, args.seed)
        split_rows.extend(rows)
        full_rows.extend(full_sample_points(cell, mono, pairs))
        n_identity += identity_points
        compared, mismatched = compare_with_metadata(checks, {p: v[1] for p, v in chal.items()},
                                                     args.reps, args.seed)
        n_compared += compared
        n_mismatched += mismatched
        print(f"   {cell:<26} N={len(ids):>5}  H1={len(ids) // 2:>4}  H2={len(ids) - len(ids) // 2:>4}  配對={len(pairs)}")

    print(f"\n🔗 恆等式 excess_H2 = d_H2·(c_H2/2)·(recovery_H2 − recovery_blind_H2)：{n_identity} 點全部成立")
    if n_compared:
        status = "✅ 全部一致" if n_mismatched == 0 else f"❌ {n_mismatched} 個不一致"
        print(f"   與 metadata['RecoveryBlind'] 的 H2 平均核對：{n_compared} 個值，{status}")
    else:
        print("   challenge 檔尚無對應 seed / reps 的 RecoveryBlind metadata，略過核對")

    points = pd.DataFrame(split_rows)
    full = pd.DataFrame(full_rows)
    summary = summarize(points, full, cells, args.reps, args.boot, args.boot_seed)
    print_summary(summary)

    os.makedirs(args.outdir, exist_ok=True)
    p1 = os.path.join(args.outdir, "split_half_points.csv")
    p2 = os.path.join(args.outdir, "split_half_slopes.csv")
    points.to_csv(p1, index=False, encoding="utf-8-sig")
    summary.to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"\n💾 逐點資料 ({len(points)} 列) → {p1}")
    print(f"💾 迴歸摘要 ({len(summary)} 列) → {p2}")


if __name__ == "__main__":
    main()
