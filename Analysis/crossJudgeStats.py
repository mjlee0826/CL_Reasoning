import numpy as np
import pandas as pd
from scipy import stats

from Analysis.crossJudge import MODELS, STRONG, WEAK, DATASETS

# ------------------------------------------------------------------
# RQ2 的統計與判定（rq2_criteria.md §5–§7）。輸入是 buildCells 的 cells（只取 used_in_analysis）與逐題列。
#   R(g, j, d) = 該格 recovery（recovery_H2）對配對的平均
# ------------------------------------------------------------------
THRESHOLD = 0.05
LEVEL = 0.95
# §7.4 同組但非自己的組合：(Judge, generator)
SAME_GROUP = [("deepseek4.1flash", "gemini3.1flashlite"), ("gemini3.1flashlite", "deepseek4.1flash"),
              ("gpt4omini", "qwen"), ("qwen", "gpt4omini")]
EXISTS, ABSENT, UNDETERMINED = "存在", "不存在", "無法判定"
OUTCOMES = {
    "a": "裁判效果存在、候選效果不存在：recovery 是裁判的性質，「瓶頸在 aggregator」的診斷可信",
    "b": "候選效果存在、裁判效果不存在：recovery 由候選答案決定，「該改進 aggregator」的診斷不成立",
    "c": "兩者都存在：報各自的大小",
    "d": "兩者都不存在：強弱模型的 recovery 差距不能歸因於裁判或候選（可能是交互作用或雜訊），兩種診斷都不支持",
    "e": "任一效果無法判定：只陳述已判定的那個效果；對無法判定的效果不下「有」或「沒有」的結論，並報它的區間",
}


def blockMeans(cells: pd.DataFrame, value: str = "recovery_H2", pair: str | None = None) -> pd.Series:
    """R(g, j, d)：index (generator, judge, dataset)，對配對平均（pair 指定時只用該配對）。"""
    used = cells[cells["used_in_analysis"]]
    if pair is not None:
        used = used[used["pair"] == pair]
    return used.groupby(["generator", "judge", "dataset"])[value].mean()


def matrix(cells: pd.DataFrame, value: str = "recovery_H2", dataset: str | None = None, pair: str | None = None) -> pd.DataFrame:
    """4×4 表（列 = generator、欄 = Judge），格子 = value 對其餘維度的平均。"""
    used = cells[cells["used_in_analysis"]]
    if dataset is not None:
        used = used[used["dataset"] == dataset]
    if pair is not None:
        used = used[used["pair"] == pair]
    table = used.pivot_table(index="generator", columns="judge", values=value, aggfunc="mean")
    return table.reindex(index=MODELS, columns=MODELS)


def summarize(values, level: float = LEVEL) -> dict:
    """跨區塊：平均、SE = sd/√B、95% t 區間（自由度 B − 1）、幾個區塊為正。"""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * se) if n > 1 else float("nan")
    return {"n_blocks": n, "mean": mean, "se": se, "ci_low": mean - half, "ci_high": mean + half,
            "n_positive": int((values > 0).sum())}


def blockEffects(R: pd.Series, kind: str, exclude_diagonal: bool = True) -> pd.DataFrame:
    """
    §5.2 裁判效果（kind = "judge"）：區塊 = generator × 資料集，Δ = 強 Judge 的平均 R − 弱 Judge 的平均 R。
    §5.3 候選效果（kind = "candidate"）：區塊 = Judge × 資料集，Δ = 強 generator 的平均 R − 弱 generator 的平均 R。
    exclude_diagonal：主分析排除自己裁決自己的格子。
    """
    def groupMean(fixed: str, group: list[str], dataset: str) -> float:
        keys = [(fixed, other, dataset) if kind == "judge" else (other, fixed, dataset)
                for other in group if not (exclude_diagonal and other == fixed)]
        return float(np.mean([R[key] for key in keys]))

    block = "generator" if kind == "judge" else "judge"
    effects = pd.DataFrame([{block: fixed, "dataset": dataset,
                             "strong": groupMean(fixed, STRONG, dataset), "weak": groupMean(fixed, WEAK, dataset)}
                            for fixed in MODELS for dataset in DATASETS])
    effects["delta"] = effects["strong"] - effects["weak"]
    return effects


def effectState(summary: dict, threshold: float = THRESHOLD) -> str:
    """§6.1：先平均再取絕對值。存在 = |平均| ≥ 門檻且區間不含 0；不存在 = 整個區間在 [−門檻, +門檻] 內；其他 = 無法判定。"""
    if abs(summary["mean"]) >= threshold and (summary["ci_low"] > 0 or summary["ci_high"] < 0):
        return EXISTS
    if summary["ci_low"] >= -threshold and summary["ci_high"] <= threshold:
        return ABSENT
    return UNDETERMINED


def outcome(judge_state: str, candidate_state: str) -> str:
    """§6.2 的 (a)–(e)。"""
    if UNDETERMINED in (judge_state, candidate_state):
        return "e"
    return {(EXISTS, ABSENT): "a", (ABSENT, EXISTS): "b", (EXISTS, EXISTS): "c", (ABSENT, ABSENT): "d"}[(judge_state, candidate_state)]


def residuals(R: pd.Series) -> pd.DataFrame:
    """§5.5：每個資料集的 4×4 表（含對角線），預期值 = 列平均 + 欄平均 − 總平均，殘差 = 實際 − 預期。"""
    rows = []
    for dataset in DATASETS:
        table = pd.DataFrame([[R[(g, j, dataset)] for j in MODELS] for g in MODELS], index=MODELS, columns=MODELS)
        expected = table.mean(axis=1).values[:, None] + table.mean(axis=0).values[None, :] - table.values.mean()
        for i, generator in enumerate(MODELS):
            for k, judge in enumerate(MODELS):
                rows.append({"generator": generator, "judge": judge, "dataset": dataset, "R": table.iloc[i, k],
                             "expected": expected[i, k], "residual": table.iloc[i, k] - expected[i, k]})
    return pd.DataFrame(rows)


def diagonalResiduals(res: pd.DataFrame) -> pd.DataFrame:
    return res[res["generator"] == res["judge"]]


def sameGroupResiduals(res: pd.DataFrame) -> pd.DataFrame:
    """§7.4：(Judge, generator) ∈ SAME_GROUP 的格子。"""
    keys = set(SAME_GROUP)
    return res[[(j, g) in keys for j, g in zip(res["judge"], res["generator"])]]


def selfPreference(summary: dict, threshold: float = THRESHOLD) -> bool:
    """§6.3：16 個對角線殘差的平均 ≥ +門檻，且 95% 區間不含 0（只看正方向）。"""
    return summary["mean"] >= threshold and summary["ci_low"] > 0


def mcnemar(items: pd.DataFrame) -> pd.DataFrame:
    """
    §7.1：每組強–弱 Judge，用兩者都不是 generator 的那兩個 generator 的 n_R 題（跨資料集與配對合併），
    b = 強挑對、弱挑錯，c = 弱挑對、強挑錯（無效選擇算挑錯），精確二項檢定（雙尾，p = 0.5）。
    """
    used = items[items["correct_a"] != items["correct_b"]]
    key = ["generator", "dataset", "pair", "item_id"]
    rows = []
    for strong in STRONG:
        for weak in WEAK:
            generators = [g for g in MODELS if g not in (strong, weak)]
            sub = used[used["generator"].isin(generators)]
            s = sub[sub["judge"] == strong].set_index(key)["correct"]
            w = sub[sub["judge"] == weak].set_index(key)["correct"]
            if not s.index.sort_values().equals(w.index.sort_values()):
                raise ValueError(f"McNemar {strong} vs {weak}: the two judges cover different items")
            w = w.reindex(s.index)
            b, c = int((s & ~w).sum()), int((~s & w).sum())
            p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
            rows.append({"strong_judge": strong, "weak_judge": weak, "generators": ", ".join(generators), "n_R": len(s),
                         "strong_right": int(s.sum()), "weak_right": int(w.sum()), "b": b, "c": c, "p_value": p})
    return pd.DataFrame(rows)


def practicalImpact(cells: pd.DataFrame) -> pd.DataFrame:
    """§7.2：弱 generator 的候選改由強 Judge 裁決，acc_final 的變化（百分點）。區塊 = 弱 generator × 資料集。"""
    A = blockMeans(cells, "acc_final")
    rows = []
    for generator in WEAK:
        for dataset in DATASETS:
            strong = float(np.mean([A[(generator, judge, dataset)] for judge in STRONG]))
            own = float(A[(generator, generator, dataset)])
            rows.append({"generator": generator, "dataset": dataset, "acc_strong_judges": strong, "acc_self": own,
                         "delta_pp": 100 * (strong - own)})
    return pd.DataFrame(rows)
