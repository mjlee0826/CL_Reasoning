"""
judge_position_check.py — 核對六：兩個候選的 Judge 有沒有偏向某個位置（不呼叫 API，不改任何現有檔案）

    零、開始前的檢查
        1. 重現核對四第零節（qwen 12 格的 d、c、m、分歧題正確率）：和 aggregation_cells.csv 比（1e-12），
           也和 qwen_k2_check.md 第零節的表（小數四位）比。對不上就停。
        2. 每個模型：presentation_order 中，每格（配對 × 資料集）檔名第一條 path 排在位置 1 的比例；
           偏離 45%–55% 的格子列出。題目 = both_answered、有呼叫 Judge 的分歧題。
    一、每個裁判模型（主網格，各自裁決自己，12 組配對 × 4 個資料集；both_answered）：
        只用有 Judge 紀錄、choice 為 1 或 2、final_answer 有解析出來、不是 off-menu 的分歧題
        1. 分歧題中選位置 2 的比例
        2. 一對一錯：(a) 正確答案在位置 1 時選對的比例；(b) 在位置 2 時選對的比例；(c) (b) − (a)
        3. 兩個候選都錯：選位置 2 的比例
        合併 = 題數加總後再算比例；區間 = Clopper–Pearson（proportion_confint(method="beta")）
    二、事先寫下的讀法（第 1 項的合併比例）
    五、RQ2 交叉實驗（裁判 ≠ 候選模型的 144 格，只報告）

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq2/judge_position_check.py
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import glob
import os
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from Arm.ArmSpec import ArmSpec
from Arm.GenerationRecord import GenerationRecord
from Analysis.alignment import loadRecords
from Analysis.experimentPlan import PAIRS, STEM_TO_ARM, ARMS_TO_PAIR
from Dataset.DatasetType import DatasetType, get_dataset_map
from Runner.paths import armPath

MODELS = ["gpt4omini", "qwen", "deepseek4.1flash", "gemini3.1flashlite"]
DATASETS = ["mmlu", "mathqa", "truthfulqa", "commonsenseqa"]
MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
CHECK4_PAIRS = ["EN+ZH", "EN+S1", "P1+P2"]
CHECK4_REPORT = "result/analysis/rq2/qwen_k2_check.md"
CROSS_DIR = "result/analysis/rq2/judge_outputs"
OUT_FILE = "result/analysis/rq2/judge_position_check.md"
BAND = (0.45, 0.55)           # 第零節 2 與讀法的「沒有看到位置偏向」
BIAS = (0.40, 0.60)           # 讀法：≥ 60% 或 ≤ 40%
TOL = 1e-12


def parseArgs():
    parser = ArgumentParser(description="Check 6: position bias of the two-candidate Judge")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--cells", default="result/analysis/aggregation_cells.csv")
    parser.add_argument("--check4", default=CHECK4_REPORT)
    parser.add_argument("--cross-dir", default=CROSS_DIR)
    parser.add_argument("--out", default=OUT_FILE)
    return parser.parse_args()


def pairOf(path: str):
    stems = os.path.basename(path)[len("judge__"):-len(".json")].split("__")
    arms = [STEM_TO_ARM[s] for s in stems]
    pair = ARMS_TO_PAIR[frozenset(arms)]
    if pair.arms != arms:
        raise SystemExit(f"❌ {path}: arm order in the file name differs from the experiment plan")
    return pair


def cellItems(args, generator: str, dataset: str, path: str) -> pd.DataFrame:
    """一格的逐題資料（全部題目）：兩個候選的答案與對錯、Judge 的紀錄。"""
    pair = pairOf(path)
    compare = get_dataset_map()[DatasetType(dataset)].compareTwoAnswer
    A = loadRecords(armPath(args.armdir, generator, dataset, ArmSpec.from_arm_id(pair.arm_a)))[1]
    B = loadRecords(armPath(args.armdir, generator, dataset, ArmSpec.from_arm_id(pair.arm_b)))[1]
    meta, J = loadRecords(path)
    if meta.get("candidate_arms") != pair.arms or meta.get("prompt_version") != "choice-v1":
        raise SystemExit(f"❌ {path}: candidate_arms {meta.get('candidate_arms')} / prompt {meta.get('prompt_version')}")
    rows = []
    for i in sorted(A):
        a, b, r = A[i], B[i], J.get(i)
        if a["gold"] != b["gold"]:
            raise SystemExit(f"❌ {path} | {i}: gold differs between the arm files")
        trace = (r or {}).get("trace") or {}
        rows.append({"item_id": i, "gold": a["gold"], "ans_a": a["parsed_answer"], "ans_b": b["parsed_answer"],
                     "both": bool(a["parse_ok"] and b["parse_ok"]),
                     "correct_a": compare(a["gold"], a["parsed_answer"]), "correct_b": compare(b["gold"], b["parsed_answer"]),
                     "dis": not compare(a["parsed_answer"], b["parsed_answer"]), "has_record": r is not None,
                     "order": (r or {}).get("presentation_order"), "final": (r or {}).get("final_answer"),
                     "off_menu": bool((r or {}).get("off_menu")), "choice": trace.get("choice"), "chosen_arm": trace.get("chosen_arm")})
    df = pd.DataFrame(rows)
    df["final_ok"] = [GenerationRecord.isParseOk(f) for f in df.final]
    df["final_correct"] = [compare(g, f) if f is not None else False for g, f in zip(df.gold, df.final)]
    df["called"] = df.both & df.dis & df.has_record & df.order.notna()
    df["usable"] = df.called & df.choice.isin([1, 2]) & df.final_ok & ~df.off_menu
    df["a_first"] = [o is not None and o[0] == pair.arm_a for o in df.order]
    # 位置 1 的候選是否答對
    first_correct = np.where(df.a_first, df.correct_a, df.correct_b)
    second_correct = np.where(df.a_first, df.correct_b, df.correct_a)
    df["pos1_correct"], df["pos2_correct"] = first_correct, second_correct
    df["pick2"] = df.choice == 2
    # 編號 → 候選 → 答案的對應（只核對、不改任何東西）
    df["mapping_ok"] = [not u or (o[int(c) - 1] == arm and (a if arm == pair.arm_a else b) == f)
                        for u, o, c, arm, a, b, f in zip(df.usable, df.order, df.choice, df.chosen_arm, df.ans_a, df.ans_b, df.final)]
    return df.assign(pair=pair.label, arm_a=pair.arm_a, arm_b=pair.arm_b, dataset=dataset, generator=generator)


def counts(df: pd.DataFrame) -> dict:
    """第一節的四個量的分子與分母（只用 usable 的題目）。"""
    u = df[df.usable]
    one = u[u.pos1_correct ^ u.pos2_correct]
    c1, c2 = one[one.pos1_correct], one[one.pos2_correct]
    bw = u[~u.pos1_correct & ~u.pos2_correct]
    return {"n": len(u), "pick2": int(u.pick2.sum()),
            "n_c1": len(c1), "right_c1": int((~c1.pick2).sum()), "n_c2": len(c2), "right_c2": int(c2.pick2.sum()),
            "n_bw": len(bw), "bw_pick2": int(bw.pick2.sum()),
            "called": int(df.called.sum()), "excluded": int((df.called & ~df.usable).sum()),
            "mapping_bad": int((df.usable & ~df.mapping_ok).sum())}


def rates(c: dict) -> dict:
    div = lambda a, b: a / b if b else float("nan")
    out = {"pos2": div(c["pick2"], c["n"]), "a": div(c["right_c1"], c["n_c1"]), "b": div(c["right_c2"], c["n_c2"]),
           "bw_pos2": div(c["bw_pick2"], c["n_bw"])}
    out["c_pp"] = 100 * (out["b"] - out["a"])
    return out


def addCounts(rows: list[dict]) -> dict:
    keys = ("n", "pick2", "n_c1", "right_c1", "n_c2", "right_c2", "n_bw", "bw_pick2", "called", "excluded", "mapping_bad")
    return {k: sum(r[k] for r in rows) for k in keys}


def ci(k: int, n: int) -> str:
    if n == 0:
        return "—"
    low, high = proportion_confint(k, n, alpha=0.05, method="beta")
    return f"{100 * k / n:.1f}% [{100 * low:.1f}, {100 * high:.1f}]（{k}/{n}）"


def pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{100 * x:.1f}"


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, disable_numparse=True)


def reading(pooled: float, by_dataset: list[float]) -> str:
    if BAND[0] <= pooled <= BAND[1]:
        return "選位置 2 的比例在 45% 到 55% 之間 → 沒有看到位置偏向。"
    same_side = all(v > 0.5 for v in by_dataset) or all(v < 0.5 for v in by_dataset)
    if (pooled >= BIAS[1] or pooled <= BIAS[0]) and same_side:
        return "**在 60% 以上或 40% 以下，而且 4 個資料集方向一致 → 有位置偏向，特別標出。**"
    return "其他情況 → 只回報，不下結論。"


def main():
    args = parseArgs()

    # 主網格：每個模型 12 組配對 × 4 個資料集
    cells = {}
    for model in MODELS:
        for dataset in DATASETS:
            for path in sorted(glob.glob(os.path.join(args.aggdir, model, dataset, "judge__*.json"))):
                df = cellItems(args, model, dataset, path)
                cells[(model, dataset, df.pair.iloc[0])] = df
    order = [p.label for p in PAIRS]
    n_files = {m: sum(1 for k in cells if k[0] == m) for m in MODELS}
    if any(v != 48 for v in n_files.values()):
        raise SystemExit(f"❌ judge files per model {n_files} (expected 48)")

    # 第零節 1
    ref = pd.read_csv(args.cells)
    ref = ref[(ref.model == "qwen") & (ref.aggregator == "judge") & (ref.subset == "both_answered")]
    text = Path(args.check4).read_text(encoding="utf-8")
    printed = {(m[0], m[1]): [float(v) for v in m[2:]] for m in
               re.findall(r"^\| (EN\+ZH|EN\+S1|P1\+P2) \| (\w+)\s*\| \d+\s*\| \d+\s*\| ([\d.]+)\s*\| ([\d.]+)\s*\| ([\d.]+)\s*\| [\d.]+\s*\| ([\d.]+)",
                          text, flags=re.M)}
    check1, max_diff, printed_ok = [], 0.0, True
    for label in CHECK4_PAIRS:
        for dataset in DATASETS:
            df = cells[("qwen", dataset, label)]
            b = df[df.both]
            nD = int(b.dis.sum())
            mine = {"d": nD / len(b), "c": int((b.dis & (b.correct_a | b.correct_b)).sum()) / nD,
                    "m": float(((b.correct_a.astype(float) + b.correct_b.astype(float)) / 2)[b.dis].mean()),
                    "acc_dis": float(b.final_correct[b.dis].mean())}
            r = ref[(ref.pair == label) & (ref.dataset == dataset)].iloc[0]
            theirs = {"d": r.d, "c": r.c, "m": r.m, "acc_dis": r.m + r.recovery * (r.c - r.m)}
            diff = max(abs(mine[k] - theirs[k]) for k in mine)
            max_diff = max(max_diff, diff)
            shown = [round(mine[k], 4) for k in ("d", "c", "m", "acc_dis")]
            same = printed.get((label, dataset)) == shown
            printed_ok &= same
            check1.append({"配對": label, "資料集": dataset, "d": f"{mine['d']:.4f}", "c": f"{mine['c']:.4f}", "m": f"{mine['m']:.4f}",
                           "分歧題正確率": f"{mine['acc_dis']:.4f}", "與 CSV 的最大差": f"{diff:.1e}",
                           "與核對四的表相同": "是" if same else "**否**"})
    ok1 = max_diff <= TOL and printed_ok and len(printed) == 12
    print(f"第零節 1：最大差 {max_diff:.1e}，與核對四的表相同 {printed_ok}（{len(printed)} 格）-> {'通過' if ok1 else '不符'}")
    if not ok1:
        raise SystemExit("❌ 第零節 1 對不上：停下來回報")

    # 第零節 2
    balance = []
    for (model, dataset, label), df in cells.items():
        called = df[df.called]
        balance.append({"model": model, "dataset": dataset, "pair": label, "arm_a": df.arm_a.iloc[0], "n": len(called),
                        "a_first": float(called.a_first.mean()) if len(called) else float("nan")})
    balance = pd.DataFrame(balance)
    off = balance[(balance.a_first < BAND[0]) | (balance.a_first > BAND[1])]

    # 第一節
    per_cell = []
    for (model, dataset, label), df in cells.items():
        per_cell.append({"model": model, "dataset": dataset, "pair": label, **counts(df)})
    per_cell = pd.DataFrame(per_cell)
    per_cell["_p"] = per_cell.pair.map(order.index)
    per_cell["_d"] = per_cell.dataset.map(DATASETS.index)
    per_cell["_m"] = per_cell.model.map(MODELS.index)
    per_cell = per_cell.sort_values(["_m", "_d", "_p"]).drop(columns=["_p", "_d", "_m"]).reset_index(drop=True)

    pooled = {}
    for model in MODELS:
        g = per_cell[per_cell.model == model]
        pooled[model] = {"all": addCounts(g.to_dict("records")),
                         **{d: addCounts(g[g.dataset == d].to_dict("records")) for d in DATASETS}}

    # 第五節：交叉實驗（裁判 ≠ 候選模型）
    cross = []
    for judge in MODELS:
        for generator in MODELS:
            if generator == judge:
                continue
            for dataset in DATASETS:
                for path in sorted(glob.glob(os.path.join(args.cross_dir, judge, generator, dataset, "judge__*.json"))):
                    meta = loadRecords(path)[0]
                    if meta.get("judge") != judge or meta.get("generator") != generator:
                        raise SystemExit(f"❌ {path}: judge / generator metadata mismatch")
                    df = cellItems(args, generator, dataset, path)
                    cross.append({"judge": judge, "generator": generator, "dataset": dataset, "pair": df.pair.iloc[0],
                                  "a_first": float(df[df.called].a_first.mean()), **counts(df)})
    cross = pd.DataFrame(cross)
    if len(cross) != 144:
        raise SystemExit(f"❌ {len(cross)} cross cells (expected 144)")

    writeReport(args, check1, max_diff, balance, off, per_cell, pooled, cross)


def writeReport(args, check1, max_diff, balance, off, per_cell, pooled, cross):
    out = ["# 核對六：兩個候選的 Judge 有沒有偏向某個位置", "",
           f"產生時間 {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}；程式 `scripts/analysis_rq2/judge_position_check.py`。"
           "不呼叫 API，不改任何判定、計分與現有檔案。", ""]
    out += ["## (1) 讀了哪些檔案", "",
            f"- 主網格兩個候選的 Judge（prompt choice-v1，主網格正式的那一次，各模型裁決自己）：`{args.aggdir}/{{模型}}/{{資料集}}/judge__*.json`，"
            "4 個模型 × 12 組配對 × 4 個資料集 = 192 個檔。欄位：`item_id`、`presentation_order`、`final_answer`、`off_menu`、"
            "`trace.choice`、`trace.chosen_arm`；metadata 的 `candidate_arms`、`prompt_version`。不讀舊模型的 `deepseek/`、`gemini/`，"
            "也不讀 RQ2 的重跑（`result/analysis/rq2/precheck/` 與 `judge_outputs/{m}/{m}/` 的對角線）。",
            f"- 候選答案與對錯：`{args.armdir}/{{模型}}/{{資料集}}/` 的對應 path 檔，欄位 `gold`、`parsed_answer`、`parse_ok`；"
            "對錯用 `compareTwoAnswer`（字串相等）。subset = both_answered。",
            f"- 第零節 1：`{args.cells}`（qwen × judge × both_answered 的 12 列：`d`、`c`、`m`、`recovery`）與 `{args.check4}` 第零節的表。",
            f"- 第五節（只報告）：`{args.cross_dir}/{{裁判}}/{{候選模型}}/{{資料集}}/judge__*.json`，裁判 ≠ 候選模型的 144 格"
            "（只收 both_answered 的分歧題）；候選答案讀 `result/arms/{候選模型}/`。", ""]

    out += ["## (2) 第零節的檢查", "",
            f"1. 重現核對四第零節（qwen 12 格）：和 `aggregation_cells.csv` 的最大差 {max_diff:.1e}（門檻 1e-12），"
            "四捨五入到小數四位後和核對四的表逐格相同 → 通過。", "", md(pd.DataFrame(check1)), "",
            "2. 每格檔名第一條 path 排在位置 1 的比例（both_answered、有呼叫 Judge 的分歧題）："]
    summ = []
    for model in MODELS:
        g = balance[balance.model == model]
        w = (g.a_first * g.n).sum() / g.n.sum()
        summ.append({"模型": MODEL_LABELS[model], "格數": len(g), "合併比例": pct(w), "最低": pct(g.a_first.min()),
                     "最高": pct(g.a_first.max()), "不在 45%–55% 的格子": int(((g.a_first < BAND[0]) | (g.a_first > BAND[1])).sum())})
    out += ["", md(pd.DataFrame(summ)), ""]
    if len(off):
        show = off.assign(model=off.model.map(MODEL_LABELS), a_first=off.a_first.map(pct))
        out += [f"不在 45%–55% 之間的 {len(off)} 格（多半是題數少的格子）：", "",
                md(show[["model", "dataset", "pair", "arm_a", "n", "a_first"]].rename(columns={
                    "model": "模型", "dataset": "資料集", "pair": "配對", "arm_a": "第一條 path", "n": "題數", "a_first": "排在位置 1（%）"})), ""]
    else:
        out += ["所有格子都在 45%–55% 之間。", ""]

    # (3) 合併結果與讀法
    head, detail = [], []
    for model in MODELS:
        p = pooled[model]
        a = p["all"]
        r = rates(a)
        by = [rates(p[d])["pos2"] for d in DATASETS]
        head.append({"模型": MODEL_LABELS[model], "第 1 項：選位置 2": ci(a["pick2"], a["n"]),
                     **{d: pct(v) for d, v in zip(DATASETS, by)}, "讀法": reading(r["pos2"], by)})
        detail.append({"模型": MODEL_LABELS[model], "(a) 正確在位置 1 時選對": ci(a["right_c1"], a["n_c1"]),
                       "(b) 正確在位置 2 時選對": ci(a["right_c2"], a["n_c2"]), "(c) (b) − (a)（pp）": f"{r['c_pp']:+.1f}",
                       **{f"(c) {d}": f"{rates(p[d])['c_pp']:+.1f}" for d in DATASETS}})
    third = [{"模型": MODEL_LABELS[m], "第 3 項：兩個都錯時選位置 2": ci(pooled[m]["all"]["bw_pick2"], pooled[m]["all"]["n_bw"]),
              **{d: pct(rates(pooled[m][d])["bw_pos2"]) for d in DATASETS}} for m in MODELS]
    excl = [{"模型": MODEL_LABELS[m], "有呼叫 Judge 的分歧題": pooled[m]["all"]["called"], "排除": pooled[m]["all"]["excluded"],
             "使用": pooled[m]["all"]["n"], "編號 → 候選 → 答案對不上": pooled[m]["all"]["mapping_bad"]} for m in MODELS]
    out += ["## (3) 四個模型的合併結果與讀法", "",
            "只用有呼叫 Judge、choice 為 1 或 2、final_answer 有解析出來、不是 off-menu 的分歧題（both_answered）。"
            "合併 = 題數加總後再算比例；區間 = 95% Clopper–Pearson。資料集欄是各資料集合併 12 組配對的值（%）。", "",
            md(pd.DataFrame(excl).astype(str)), "", "第 1 項（全部分歧題中選位置 2 的比例）與讀法：", "", md(pd.DataFrame(head)), "",
            "第 2 項（一對一錯的題目）：", "", md(pd.DataFrame(detail)), "", "第 3 項：", "", md(pd.DataFrame(third)), ""]
    flagged = [MODEL_LABELS[m] for m, h in zip(MODELS, head) if "有位置偏向" in h["讀法"]]
    if flagged:
        out += [f"特別標出：{'、'.join(flagged)} 有位置偏向；第 2 項 (c) 的大小見上表。", ""]

    # (4) 逐格的表
    out += ["## (4) 逐格的表", "", "各資料集合併 12 組配對（%；(c) 為 pp）：", ""]
    rows = []
    for model in MODELS:
        for d in DATASETS:
            c = pooled[model][d]
            r = rates(c)
            rows.append({"模型": MODEL_LABELS[model], "資料集": d, "題數": c["n"], "選位置 2": pct(r["pos2"]),
                         "(a)": pct(r["a"]), "(b)": pct(r["b"]), "(c)": f"{r['c_pp']:+.1f}", "兩個都錯時選位置 2": pct(r["bw_pos2"]),
                         "兩個都錯的題數": c["n_bw"]})
    out += [md(pd.DataFrame(rows).astype(str)), "", "逐（模型 × 資料集 × 配對）（%；(c) 為 pp；n = 使用的分歧題數）：", ""]
    rows = []
    for rec in per_cell.to_dict("records"):
        r = rates(rec)
        rows.append({"模型": MODEL_LABELS[rec["model"]], "資料集": rec["dataset"], "配對": rec["pair"], "n": rec["n"],
                     "選位置 2": pct(r["pos2"]), "(a)": pct(r["a"]), "(b)": pct(r["b"]),
                     "(c)": "—" if pd.isna(r["c_pp"]) else f"{r['c_pp']:+.1f}", "兩個都錯時選位置 2": pct(r["bw_pos2"]), "兩個都錯": rec["n_bw"]})
    out += [md(pd.DataFrame(rows).astype(str)), ""]

    # (5) 交叉實驗
    rows = []
    for judge in MODELS:
        g = cross[cross.judge == judge]
        for generator in [m for m in MODELS if m != judge] + ["合併"]:
            h = g if generator == "合併" else g[g.generator == generator]
            c = addCounts(h.to_dict("records"))
            r = rates(c)
            rows.append({"裁判": MODEL_LABELS[judge], "候選模型": "三者合併" if generator == "合併" else MODEL_LABELS[generator],
                         "題數": c["n"], "選位置 2": ci(c["pick2"], c["n"]) if generator == "合併" else pct(r["pos2"]),
                         "(a)": pct(r["a"]), "(b)": pct(r["b"]), "(c)（pp）": f"{r['c_pp']:+.1f}", "兩個都錯時選位置 2": pct(r["bw_pos2"]),
                         "排除": c["excluded"], "對應不上": c["mapping_bad"]})
    w = (cross.a_first * cross.called).sum() / cross.called.sum()
    cross_off = cross[(cross.a_first < BAND[0]) | (cross.a_first > BAND[1])]
    out += ["## (5) 交叉實驗的結果（只報告）", "",
            "RQ2 交叉實驗中裁判 ≠ 候選模型的 144 格（3 組配對 EN+ZH、EN+S1、P1+P2 × 4 個資料集 × 12 種組合），題目的條件同第 (3) 節。"
            f"位置 1 的順序沿用主網格的紀錄：檔名第一條 path 排在位置 1 的比例合併為 {pct(w)}%，"
            f"{len(cross_off)} 格不在 45%–55% 之間。「三者合併」那一列的選位置 2 附 95% Clopper–Pearson 區間；其餘為 %，(c) 為 pp。", "",
            md(pd.DataFrame(rows).astype(str)), ""]
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    for h in head:
        print(f"{h['模型']}: 選位置 2 {h['第 1 項：選位置 2']} -> {h['讀法']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
