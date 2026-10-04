import numpy as np
from scipy import stats

# ------------------------------------------------------------------
# 跨區塊（模型 × 資料集）的統計：RQ2（crossJudgeStats）與 RQ1-K（menuVote）共用
# ------------------------------------------------------------------
LEVEL = 0.95
# RQ1-K 的四種狀態（rq1k_criteria.md §4.2）
FORWARD, REVERSE, EQUIVALENT, UNDETERMINED = "正向成立", "反向成立", "兩者相當", "無法判定"


def summarize(values, level: float = LEVEL) -> dict:
    """跨區塊：平均、SE = sd/√B、95% t 區間（自由度 B − 1）、幾個區塊為正。"""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * se) if n > 1 else float("nan")
    return {"n_blocks": n, "mean": mean, "se": se, "ci_low": mean - half, "ci_high": mean + half,
            "n_positive": int((values > 0).sum())}


def fourState(summary: dict, threshold: float) -> str:
    """
    正向成立 = 平均 ≥ +門檻且區間不含 0；反向成立 = 平均 ≤ −門檻且區間不含 0；
    兩者相當 = 整個區間在 [−門檻, +門檻] 內；其他 = 無法判定。
    """
    excludes_zero = summary["ci_low"] > 0 or summary["ci_high"] < 0
    if summary["mean"] >= threshold and excludes_zero:
        return FORWARD
    if summary["mean"] <= -threshold and excludes_zero:
        return REVERSE
    if summary["ci_low"] >= -threshold and summary["ci_high"] <= threshold:
        return EQUIVALENT
    return UNDETERMINED
