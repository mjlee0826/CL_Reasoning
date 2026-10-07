"""
path_improve.py — RQ3-G：改進一條 path，聚合會多多少？該改哪一條？（result/analysis/rq3g/rq3g_criteria.md）

全程離線：不呼叫 API，不重跑任何東西，不修改現有檔案。判定標準確認前不執行。
    1. 載入 16 個區塊（loadPathBlock），答案編成整數（Analysis/pathImprove.py）；核對題目 id、gold、compareTwoAnswer、
       整數投票 = Analysis.menuVote.vote
    2. §9 的四項檢查（任何一項不過就停，不寫輸出）：重現 RQ3；蒙地卡羅 2,000 次；把 path 換成它自己；子集二的保留比例
    3. 判定一（D1，16 個區塊）、判定二（D2，8 個弱模型區塊）與 §8 只報告的量
    4. 輸出（--out-dir）：rq3g_cells.csv、rq3g_substitutions.csv、rq3g_curves.csv、report.md

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq3g/path_improve.py
"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import numpy as np
import pandas as pd

from Analysis.preregistration import sha256, confirmationLine
from Analysis.blockStats import summarize, fourState, FORWARD, REVERSE, EQUIVALENT, UNDETERMINED
from Analysis.experimentPlan import PATHS
from Analysis.menuJudgeStats import blocksWithSplits
from Analysis.menuVote import MODELS, DATASETS
from Analysis.pathImprove import (M12, MENU_ORDER, MAIN_MENU, HOSTS, DONORS, CURVE_TARGETS, MAX_INVALID, THRESHOLD,
                                  encodeBlocks, checkVote, judgment1, curves, monteCarlo, Substitution, spearmanBySplit)

OUT_DIR = "result/analysis/rq3g"
CRITERIA_FILE = "rq3g_criteria.md"
RQ3_BLOCKS = "result/analysis/rq3/rq3_blocks.csv"
MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
READINGS_1 = {
    FORWARD: "path 的影響力不同，用一半題目就能挑出比較值得改的那一條。",
    EQUIVALENT: "把任何一條 path 提高 5 個百分點，投票的增加量彼此相差不到 0.5pp。這是絕對量的說法；相對的差別看第 8 節 2 的兩個組成。",
    REVERSE: "挑出來的反而較差，影響力的排名不穩定，只能當雜訊。",
    UNDETERMINED: "只列數字，不說有沒有差別。",
}
READINGS_2 = {
    REVERSE: "（真實低於預測）隨機模型高估；改對哪些題目，比正確率提高多少更重要。",
    EQUIVALENT: "隨機模型可以當作估算真實改進效果的工具。",
    FORWARD: "（真實高於預測）真實改進比隨機更常落在差一票的題目上，隨機模型低估。",
    UNDETERMINED: "只列數字。",
}
RQ3_SUMMARY = {"A_V": "84.67", "SB": "84.37", "mean": "+0.30", "ci_low": "-0.63", "ci_high": "+1.23", "n_positive": 12}   # §9.1
MC_BLOCK, MC_PATH, MC_DRAWS = ("qwen", "truthfulqa"), "ZH", 2000                                                         # §9.2
RETENTION_FLAG = 0.85                                                                                                    # §9.4
TOL_RQ3, TOL_SELF, TOL_MC_ZERO = 1e-9, 1e-12, 1e-9
AGGREGATORS = ["V", "WV", "SB", "DS", "Oracle"]


def parseArgs():
    parser = ArgumentParser(description="RQ3-G: which path is worth improving, random-fix model vs real substitution")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--rq3-blocks", default=RQ3_BLOCKS)
    parser.add_argument("--out-dir", default=OUT_DIR)
    return parser.parse_args()


def pp(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:+.2f}"


def ciRow(name: str, s: dict, with_state: bool = False) -> dict:
    row = {"量": name, "平均": pp(s["mean"]), "SE": f"{s['se']:.2f}", "95% 區間": f"[{pp(s['ci_low'])}, {pp(s['ci_high'])}]",
           "為正的區塊": f"{s['n_positive']}/{s['n_blocks']}"}
    if with_state:
        row["狀態"] = fourState(s, THRESHOLD)
    return row


def md(df: pd.DataFrame, floatfmt: str = ".2f", text: bool = False) -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt, disable_numparse=text)


def label(model: str, dataset: str) -> str:
    return f"{MODEL_LABELS[model]} · {dataset}"


def validMean(x: np.ndarray, valid: np.ndarray) -> float:
    return float(np.mean(x[valid])) if valid.any() else float("nan")


# ------------------------------------------------------------------
# §9 的檢查
# ------------------------------------------------------------------
def checkRQ3(j1_m12: dict, rq3_path: str) -> tuple[bool, list[str], dict]:
    """§9.1：逐區塊 = rq3_blocks.csv 的 M12 列（1e-9、題數相同），摘要四捨五入後 = RQ3 的數字。"""
    ref = pd.read_csv(rq3_path)
    ref = ref[ref.menu == "M12"].set_index(["model", "dataset"])
    diffs = {"A_V": 0.0, "S_in": 0.0, "excess_in_pp": 0.0}
    problems = []
    for (model, dataset), j in j1_m12.items():
        r = ref.loc[(model, dataset)]
        for key, mine in (("A_V", j["A_V"]), ("S_in", j["SB"]), ("excess_in_pp", j["excess_pp"])):
            diffs[key] = max(diffs[key], abs(mine - r[key]))
        if int(r["n_subset"]) != j["n_subset"]:
            problems.append(f"{model} | {dataset}: {j['n_subset']} items, RQ3 n_subset {int(r['n_subset'])}")
    if len(j1_m12) != len(ref):
        problems.append(f"{len(j1_m12)} blocks vs {len(ref)} RQ3 rows")
    if any(v > TOL_RQ3 for v in diffs.values()):
        problems.append(f"max differences {diffs} > {TOL_RQ3}")
    s = summarize([j["excess_pp"] for j in j1_m12.values()])
    shown = {"A_V": f"{100 * np.mean([j['A_V'] for j in j1_m12.values()]):.2f}",
             "SB": f"{100 * np.mean([j['SB'] for j in j1_m12.values()]):.2f}",
             "mean": f"{s['mean']:+.2f}", "ci_low": f"{s['ci_low']:+.2f}", "ci_high": f"{s['ci_high']:+.2f}", "n_positive": s["n_positive"]}
    if shown != RQ3_SUMMARY:
        problems.append(f"summary {shown} != {RQ3_SUMMARY}")
    return not problems, problems, {"diffs": diffs, "shown": shown}


def checkMonteCarlo(mc: dict) -> tuple[bool, list[dict]]:
    """§9.2：|抽樣平均 − 期望值| ≤ 3 × 蒙地卡羅標準誤；標準誤為 0 時差 ≤ 1e-9。"""
    rows, ok = [], True
    for key in ("V", "WV", "SB"):
        exp, mean, se = mc["expected"][key], mc["mc_mean"][key], mc["mc_se"][key]
        diff = abs(mean - exp)
        passed = diff <= TOL_MC_ZERO if se == 0 else diff <= 3 * se
        ok &= passed
        rows.append({"聚合": key, "期望值（%）": f"{100 * exp:.4f}", "2,000 次平均（%）": f"{100 * mean:.4f}",
                     "蒙地卡羅標準誤（pp）": f"{100 * se:.4f}", "差（pp）": f"{100 * diff:.4f}",
                     "差 ÷ 標準誤": "—（標準誤 0）" if se == 0 else f"{diff / se:.2f}", "結果": "通過" if passed else "不符"})
    return ok, rows


def checkSelf(coded: dict, splits: dict) -> tuple[bool, list[str], dict]:
    """§9.3：供體 = 宿主時，V、WV、SB、DS、Oracle 替換前後相同，計數為 0，子集二 = 子集一，t = 0 的隨機模型 = 原本。"""
    problems, max_error, count = [], 0.0, 0
    for (model, dataset), block in coded.items():
        s = Substitution(block, block, splits[dataset])
        if not np.array_equal(s.sub, block.sub):
            problems.append(f"{model} | {dataset}: subset 2 != subset 1")
        b = s.before
        for p, code in enumerate(M12):
            o = s.substitute(p)
            count += 1
            if o["n_fixed"] or o["n_broken"] or o["n_wrong_to_other_wrong"] or o["valid"].any():
                problems.append(f"{model} | {dataset} | {code}: non-zero counts or valid splits")
            m = o["menus"][MAIN_MENU]
            pairs = [(m["A_real"], b["V"]), (m["A_sim"], b["V"]), (m["V_before"], b["V"]),
                     (m["after"]["WV"], b["WV"]), (m["WV_sim"], b["WV"]), (m["after"]["SB"], b["SB"]), (m["SB_sim"], b["SB"]),
                     (m["after"]["DS"], b["DS"]), (m["after"]["Oracle"], b["Oracle"]), (m["A_fix"], b["V"]), (m["A_fix_sim"], b["V"])]
            pairs += [(mm["A_real"], mm["V_before"]) for mm in o["menus"].values()]
            pairs += [(mm["A_sim"], mm["V_before"]) for mm in o["menus"].values()]
            error = max(float(np.max(np.abs(x - y))) for x, y in pairs)
            max_error = max(max_error, error)
            if error > TOL_SELF:
                problems.append(f"{model} | {dataset} | {code}: max difference {error:.1e}")
    return not problems, problems, {"max_error": max_error, "substitutions": count}


# ------------------------------------------------------------------
# 替換的逐替換列（§6、§8.4–§8.8）
# ------------------------------------------------------------------
def substitutionRows(s: Substitution, p: int, n: int) -> tuple[list[dict], dict]:
    o = s.substitute(p)
    valid = o["valid"]
    invalid = int((~valid).sum())
    base = {"host": s.host.model, "donor": s.donor.model, "dataset": s.host.dataset, "path": M12[p], "path_id": PATHS[M12[p]],
            "n": n, "n_subset2": int(s.sub.sum()), "retention": float(s.sub.mean()), "invalid_splits": invalid,
            "excluded_share": invalid / len(valid), "participates": invalid <= MAX_INVALID}
    vm = lambda x: validMean(np.asarray(x, dtype=float), valid)
    rows, split_data = [], {}
    for menu, m in o["menus"].items():
        row = {"menu": menu, **base, "V_before": vm(m["V_before"]), "A_real_V": vm(m["A_real"]), "A_sim_V": vm(m["A_sim"]),
               "D2_pp": 100 * vm(m["A_real"] - m["A_sim"])}
        if menu == MAIN_MENU:
            a, b = m["after"], s.before
            row.update({
                "host_acc_H2": vm(o["host_acc"]), "donor_acc_H2": vm(o["donor_acc"]),
                "n_fixed": o["n_fixed"], "n_broken": o["n_broken"], "n_wrong_to_other_wrong": o["n_wrong_to_other_wrong"],
                "WV_before": vm(b["WV"]), "WV_after": vm(a["WV"]), "WV_sim": vm(m["WV_sim"]),
                "SB_before": vm(b["SB"]), "SB_after": vm(a["SB"]), "SB_sim": vm(m["SB_sim"]),
                "DS_before": vm(b["DS"]), "DS_after": vm(a["DS"]),
                "DS_nonconverged_before": int((~b["DS_converged"] & valid).sum()),
                "DS_nonconverged_after": int((~a["DS_converged"] & valid).sum()),
                "Oracle_before": vm(b["Oracle"]), "Oracle_after": vm(a["Oracle"]),
                "A_fix_V": vm(m["A_fix"]), "A_fix_sim_V": vm(m["A_fix_sim"]), "D5_pp": 100 * vm(m["A_fix"] - m["A_fix_sim"]),
                "conv_num_real": vm(m["A_real"] - m["V_before"]), "conv_num_sim": vm(m["A_sim"] - m["V_before"]),
                "conv_den": vm(o["donor_acc"] - o["host_acc"]),
                "lead_H1_pp": vm(m["lead_H1_pp"]), "V_minus_SB_after_pp": vm(m["V_minus_SB_pp"]),
                "WV_minus_SB_after_pp": vm(m["WV_minus_SB_pp"]),
            })
            row["conversion_real"] = row["conv_num_real"] / row["conv_den"] if valid.any() else float("nan")
            row["conversion_sim"] = row["conv_num_sim"] / row["conv_den"] if valid.any() else float("nan")
            split_data = {"valid": valid, "participates": base["participates"], "pred": m["pred_H1"],
                          "actual": m["real_gain_count"] / s.sp.n2, "lead": m["lead_H1_pp"],
                          "VmSB": m["V_minus_SB_pp"], "WVmSB": m["WV_minus_SB_pp"]}
        rows.append(row)
    return rows, split_data


def main():
    args = parseArgs()
    started = time.time()
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty)")
    criteria_sha = sha256(criteria)

    # 載入與編碼（同一資料集的 4 個模型共用編碼與切分）
    raw, splits = defaultdict(dict), {}
    for block, sp in blocksWithSplits(args.armdir, args.aggdir):
        raw[block.dataset][block.model] = block
        splits[block.dataset] = sp
    coded = {}
    for dataset in DATASETS:
        blocks = [raw[dataset][model] for model in MODELS]
        for pb, cb in zip(blocks, encodeBlocks(blocks)):
            checkVote(pb, cb)
            coded[(cb.model, cb.dataset)] = cb
    n_items = {dataset: len(raw[dataset][MODELS[0]].item_ids) for dataset in DATASETS}
    del raw
    print(f"載入與編碼：{len(coded)} 個區塊；題目 id、gold、compareTwoAnswer、整數投票 = vote 都核對過（{time.time() - started:.0f}s）")

    # 判定一（各菜單）；M12 的 A_V、SB 也給 §9.1
    j1 = {menu: {key: judgment1(block, menu, splits[key[1]]) for key, block in coded.items()} for menu in MENU_ORDER}

    # §9 的檢查
    ok1, prob1, info1 = checkRQ3(j1[MAIN_MENU], args.rq3_blocks)
    print(f"§9.1 重現 RQ3：{info1['shown']}，最大誤差 {info1['diffs']} -> {'通過' if ok1 else '不符'}")
    mc = monteCarlo(coded[MC_BLOCK], splits[MC_BLOCK[1]], MC_PATH, MC_DRAWS, 0)
    ok2, mc_rows = checkMonteCarlo(mc)
    print(f"§9.2 蒙地卡羅：{[(r['聚合'], r['差 ÷ 標準誤'], r['結果']) for r in mc_rows]}（{time.time() - started:.0f}s）")
    ok3, prob3, info3 = checkSelf(coded, splits)
    print(f"§9.3 換成自己：{info3['substitutions']} 個替換，最大差 {info3['max_error']:.1e} -> {'通過' if ok3 else '不符'}"
          f"（{time.time() - started:.0f}s）")
    retention = []
    for dataset in DATASETS:
        for host in HOSTS:
            for donor in DONORS:
                kept = coded[(host, dataset)].sub & coded[(donor, dataset)].sub
                retention.append({"host": host, "donor": donor, "dataset": dataset, "n": n_items[dataset],
                                  "n_subset2": int(kept.sum()), "retention": float(kept.mean())})
    retention = pd.DataFrame(retention)
    low = retention[retention.retention < RETENTION_FLAG]
    print(f"§9.4 保留比例：最低 {100 * retention.retention.min():.1f}%，低於 85% 的有 {len(low)} 個")
    if not (ok1 and ok2 and ok3):
        for problem in prob1 + prob3:
            print(f"  ❌ {problem}")
        raise SystemExit("❌ 開跑前的檢查沒有全部通過：停下來回報，不寫輸出")

    # 曲線（§8.3）
    curve_rows = []
    for (model, dataset), block in coded.items():
        for row in curves(block, splits[dataset]):
            curve_rows.append({"model": model, "dataset": dataset, **row})
    print(f"曲線：{len(curve_rows)} 點（{time.time() - started:.0f}s）")

    # 替換（§6、§8.4–§8.9）
    sub_rows, split_data = [], defaultdict(list)
    for dataset in DATASETS:
        for host in HOSTS:
            for donor in DONORS:
                s = Substitution(coded[(host, dataset)], coded[(donor, dataset)], splits[dataset])
                for p in range(len(M12)):
                    rows, data = substitutionRows(s, p, n_items[dataset])
                    sub_rows += rows
                    split_data[(host, dataset)].append(data)
        print(f"  替換 {dataset} 完成（{time.time() - started:.0f}s）")
    subs = pd.DataFrame(sub_rows)

    # 區塊層級的判定二與 §8.5–§8.8
    weak = []
    for host in HOSTS:
        for dataset in DATASETS:
            row = {"model": host, "dataset": dataset}
            for menu in MENU_ORDER:
                g = subs[(subs.menu == menu) & (subs.host == host) & (subs.dataset == dataset)]
                part = g[g.participates]
                row[f"{menu}_n_subs"], row[f"{menu}_n_participating"] = len(g), len(part)
                row[f"{menu}_D2_pp"] = float(part.D2_pp.mean()) if len(part) else float("nan")
                if menu == MAIN_MENU:
                    row["D5_pp"] = float(part.D5_pp.mean()) if len(part) else float("nan")
                    row["conversion_real"] = float(part.conv_num_real.sum() / part.conv_den.sum()) if len(part) else float("nan")
                    row["conversion_sim"] = float(part.conv_num_sim.sum() / part.conv_den.sum()) if len(part) else float("nan")
                    for key in ("V_before", "A_real_V", "A_sim_V", "WV_before", "WV_after", "WV_sim", "SB_before", "SB_after",
                                "SB_sim", "DS_before", "DS_after", "Oracle_before", "Oracle_after", "n_fixed", "n_broken",
                                "n_wrong_to_other_wrong", "host_acc_H2", "donor_acc_H2"):
                        row[f"mean_{key}"] = float(part[key].mean()) if len(part) else float("nan")
                    row["DS_nonconverged_before"] = int(part.DS_nonconverged_before.sum())
                    row["DS_nonconverged_after"] = int(part.DS_nonconverged_after.sum())
            data = split_data[(host, dataset)]
            use = np.array([d["valid"] & d["participates"] for d in data])
            stack = lambda key: np.array([d[key] for d in data], dtype=float)
            row["spearman7"], row["spearman7_undefined"] = spearmanBySplit(stack("pred"), stack("actual"), use)
            row["spearman8_V"], row["spearman8_V_undefined"] = spearmanBySplit(stack("lead"), stack("VmSB"), use)
            row["spearman8_WV"], row["spearman8_WV_undefined"] = spearmanBySplit(stack("lead"), stack("WVmSB"), use)
            weak.append(row)
    weak = pd.DataFrame(weak)
    empty = weak[weak[f"{MAIN_MENU}_n_participating"] == 0]
    if len(empty):
        raise SystemExit(f"❌ §6：{len(empty)} 個弱模型區塊沒有任何替換參與判定，停下來回報，不算判定二："
                         f"{list(zip(empty.model, empty.dataset))}")

    # cells CSV
    cell_rows = []
    weak_idx = weak.set_index(["model", "dataset"])
    for menu in MENU_ORDER:
        for (model, dataset), j in j1[menu].items():
            base = {"menu": menu, "model": model, "dataset": dataset, "n": n_items[dataset], "n_subset1": j["n_subset"]}
            for code, values in j["paths"].items():
                cell_rows.append({**base, "path": code, "path_id": PATHS[code], **values})
            block_row = {**base, "path": "ALL", "D1_pp": j["D1"], "gain_pstar_pp": j["gain_pstar"],
                         "gain_others_pp": j["gain_others"], "tie_share": j["tie_share"]}
            if menu == MAIN_MENU:
                block_row.update({"A_V": j["A_V"], "SB": j["SB"], "excess_pp": j["excess_pp"]})
            if (model, dataset) in weak_idx.index:
                w = weak_idx.loc[(model, dataset)]
                block_row.update({"n_subs": int(w[f"{menu}_n_subs"]), "n_subs_participating": int(w[f"{menu}_n_participating"]),
                                  "D2_pp": float(w[f"{menu}_D2_pp"])})
                if menu == MAIN_MENU:
                    block_row.update({key: w[key] for key in ("D5_pp", "conversion_real", "conversion_sim", "spearman7",
                                                              "spearman7_undefined", "spearman8_V", "spearman8_V_undefined",
                                                              "spearman8_WV", "spearman8_WV_undefined")})
            cell_rows.append(block_row)
    cells = pd.DataFrame(cell_rows)
    curves_df = pd.DataFrame(curve_rows)

    os.makedirs(args.out_dir, exist_ok=True)
    cells.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq3g_cells.csv"), index=False)
    subs.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq3g_substitutions.csv"), index=False)
    curves_df.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq3g_curves.csv"), index=False)

    # ------------------------------------------------------------------
    # 判定與報告
    # ------------------------------------------------------------------
    m12_all = cells[(cells.menu == MAIN_MENU) & (cells.path == "ALL")].reset_index(drop=True)
    j1s = summarize(m12_all.D1_pp)
    state1 = fourState(j1s, THRESHOLD)
    j2s = summarize(weak[f"{MAIN_MENU}_D2_pp"])
    state2 = fourState(j2s, THRESHOLD)

    out = ["# RQ3-G：改進一條 path，聚合會多多少？該改哪一條？", "",
           f"判定標準 `{criteria}`，sha256 `{criteria_sha}`；確認：{confirmed}。"
           f"產生時間 {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}；程式 `scripts/analysis_rq3g/path_improve.py`"
           "（逐切分計算在 `Analysis/pathImprove.py`）。不呼叫 API，不重跑任何東西，不修改現有檔案。", ""]

    out += ["## (1) 讀了哪些檔案", "",
            f"- `{args.armdir}/{{模型}}/{{資料集}}/`：4 個模型 × 4 個資料集，經 `Analysis.menuVote.loadPathBlock` 載入"
            "（與 RQ1-K、RQ3 相同；它讀 14 個 path 檔，這裡只用 M12 的 12 條）。欄位 `item_id`、`gold`、`parsed_answer`、`parse_ok`。",
            f"- `{args.aggdir}/`：`CellData` 只列出檔名，不讀內容。",
            f"- `{args.rq3_blocks}`：§9.1 的核對（M12 列的 `A_V`、`S_in`、`excess_in_pp`、`n_subset`）。",
            "- `result/analysis/rq3/rq3_criteria.md` §4：四種狀態與門檻的原文（已抄入本次判定標準檔）。",
            "- 沿用的程式：多數決 `Analysis.menuVote.vote`（整數版逐題核對過）、最強 path `Analysis.probe.strongest`、"
            "切分 `Analysis.menuJudgeStats.blocksWithSplits`（`makeSplits(n, 200, 0)`，同一資料集的模型共用）、"
            "統計 `Analysis.blockStats.summarize` / `fourState`。",
            "- 載入時的核對（不符就停）：同一資料集 4 個模型的題目 id 與 gold 相同；`compareTwoAnswer` 在出現過的答案上等於字串相等；"
            "整數編碼的對錯 = `loadPathBlock` 的 `correct`；四份菜單的整數投票逐題 = `vote`。全部通過。", ""]

    out += ["## (2) 開跑前的四項檢查", "",
            f"1. 重現 RQ3：逐區塊和 `rq3_blocks.csv` 的最大誤差 A_V {info1['diffs']['A_V']:.1e}、S_in {info1['diffs']['S_in']:.1e}、"
            f"Excess {info1['diffs']['excess_in_pp']:.1e}（門檻 1e-9），題數相同（共 {int(m12_all.n_subset1.sum())} 題）。"
            f"摘要：多數決 {info1['shown']['A_V']}%、最強單一 path {info1['shown']['SB']}%、差距 {info1['shown']['mean']}pp"
            f"（{info1['shown']['ci_low']} 到 {info1['shown']['ci_high']}）、{info1['shown']['n_positive']} / 16 為正 → 通過。",
            f"2. 期望值的算法：{label(*MC_BLOCK)}、子集一、path {MC_PATH}（{PATHS[MC_PATH]}）、提高 5 個百分點，"
            f"`numpy.random.default_rng(0)` 實際抽 {MC_DRAWS:,} 次（每次涵蓋 200 次切分）→ 通過。", "",
            md(pd.DataFrame(mc_rows), text=True), "",
            "   - 第一次執行（2026-10-06）在這一項停下、沒有寫輸出：SB 的 2,000 個值完全相同（規則只由題數決定），"
            "但 `np.std` 對相同的浮點數因捨入給出約 1e-16，程式沒有走「標準誤為 0 時差 ≤ 1e-9」的規則，把約 1e-16 的捨入差判成不符。"
            "經使用者同意，修正為「2,000 個值完全相同時標準誤 = 0」後重跑；判定標準檔沒有改。", "",
            f"3. 把一條 path 換成它自己：{info3['substitutions']} 個替換（16 個區塊 × 12 條）× 200 次切分，V、WV、SB、DS、Oracle、"
            f"A_sim（t = 0）、只修不壞版本與 M3 菜單的 V，替換前後最大差 {info3['max_error']:.1e}（門檻 1e-12）；修對、弄壞、"
            "錯換錯的題數都是 0；子集二 = 子集一 → 通過。",
            "4. 子集二的保留比例（低於 85% 的標 ⚠）：", ""]
    ret = retention.copy()
    ret["host"], ret["donor"] = ret.host.map(MODEL_LABELS), ret.donor.map(MODEL_LABELS)
    ret["retention"] = [f"{100 * v:.1f}%" + (" ⚠" if v < RETENTION_FLAG else "") for v in retention.retention]
    out += [md(ret.rename(columns={"host": "宿主", "donor": "供體", "dataset": "資料集", "n": "總題數", "n_subset2": "子集二",
                                   "retention": "保留比例"}), text=True), "",
            f"低於 85% 的：{len(low)} 個。" + ("" if len(low) == 0 else "（只標出，不停。）"), ""]

    per1 = m12_all[["model", "dataset", "D1_pp", "gain_pstar_pp", "gain_others_pp"]].copy()
    per1["model"] = per1.model.map(MODEL_LABELS)
    out += ["## (3) 判定一：有沒有哪一條 path 特別值得改（16 個區塊，M12，V）", "",
            md(pd.DataFrame([ciRow("判定一：D1 = gain_2(p*) − 其餘 11 條 gain_2 的平均（pp）", j1s, True)]), text=True), "",
            f"門檻 {THRESHOLD}pp。**{state1}** → {READINGS_1[state1]}", "",
            "逐區塊（pp，200 次切分平均；gain_2(p*) 與其餘的平均是第 8 節 2 的兩個組成）：", "",
            md(per1.rename(columns={"model": "模型", "dataset": "資料集", "D1_pp": "D1", "gain_pstar_pp": "gain_2(p*)",
                                    "gain_others_pp": "其餘 11 條的平均"})), ""]

    m12_subs = subs[subs.menu == MAIN_MENU]
    per2 = weak[["model", "dataset", f"{MAIN_MENU}_D2_pp", f"{MAIN_MENU}_n_participating", f"{MAIN_MENU}_n_subs"]].copy()
    per2["model"] = per2.model.map(MODEL_LABELS)
    excluded = m12_subs[m12_subs.invalid_splits > 0].sort_values("excluded_share", ascending=False)
    out += ["## (4) 判定二：隨機模型預測得準真實的替換嗎（8 個弱模型區塊，M12，V）", "",
            md(pd.DataFrame([ciRow("判定二：D2 = A_real − A_sim（pp）", j2s, True)]), text=True), "",
            f"門檻 {THRESHOLD}pp。**{state2}** → {READINGS_2[state2]}", "",
            "逐區塊（pp；先對每個替換的有效切分平均，再對參與判定的替換等權平均）：", "",
            md(per2.rename(columns={"model": "宿主", "dataset": "資料集", f"{MAIN_MENU}_D2_pp": "D2",
                                    f"{MAIN_MENU}_n_participating": "參與判定的替換", f"{MAIN_MENU}_n_subs": "替換數"})), "",
            f"有任何無效切分的替換：{len(excluded)} / {len(m12_subs)} 個；排除比例超過 20%、不參與判定的："
            f"{int((~m12_subs.participates).sum())} 個。", ""]
    if len(excluded):
        ex = excluded[["host", "donor", "dataset", "path", "excluded_share", "participates", "host_acc_H2", "donor_acc_H2"]].copy()
        ex["host"], ex["donor"] = ex.host.map(MODEL_LABELS), ex.donor.map(MODEL_LABELS)
        ex["excluded_share"] = [f"{100 * v:.1f}%" for v in ex.excluded_share]
        ex["participates"] = ["是" if v else "**否**" for v in ex.participates]
        ex["host_acc_H2"] = [f"{100 * v:.2f}" if pd.notna(v) else "—" for v in ex.host_acc_H2]
        ex["donor_acc_H2"] = [f"{100 * v:.2f}" if pd.notna(v) else "—" for v in ex.donor_acc_H2]
        out += [md(ex.rename(columns={"host": "宿主", "donor": "供體", "dataset": "資料集", "path": "path", "excluded_share": "排除比例",
                                      "participates": "參與判定", "host_acc_H2": "宿主正確率（%，有效切分）",
                                      "donor_acc_H2": "供體正確率（%，有效切分）"}), text=True), ""]

    # (5) 只報告的量
    out += ["## (5) 只報告的量（不參與判定）", ""]
    path_rows = cells[(cells.menu == MAIN_MENU) & (cells.path != "ALL")]
    def wide(column: str, scale: float) -> pd.DataFrame:
        t = path_rows.pivot_table(index=["model", "dataset"], columns="path", values=column, sort=False)[M12] * scale
        t = t.reset_index()
        t["model"] = t.model.map(MODEL_LABELS)
        t.loc[len(t)] = ["平均", ""] + list(t[M12].mean())
        return t.rename(columns={"model": "模型", "dataset": "資料集"})
    out += ["### 8.1 逐區塊、逐 path（M12，子集一，評分半，200 次切分平均）", "",
            "gain_2（pp；提高 5 個百分點時 V 的期望增加量）：", "", md(wide("gain_H2_pp", 1)), "",
            "π_p（%；p 答錯的題目中，改對後 V 由錯變對的比例）：", "", md(wide("pi_H2", 100), ".1f"), "",
            f"π_p 無定義（評分半沒有答錯的題目）的切分：共 {int(path_rows.pi_undefined_splits.sum())} 次。", "",
            "正確率（%）：", "", md(wide("acc_H2", 100), ".1f"), ""]

    share = wide("p_star_share", 100)
    tie = {(MODEL_LABELS[r.model], r.dataset): 100 * r.tie_share for r in m12_all.itertuples()}   # 依 (模型, 資料集) 對齊
    share["選擇半平手（%）"] = [tie.get((m, d), 100 * float(m12_all.tie_share.mean())) for m, d in zip(share["模型"], share["資料集"])]
    comp = []
    for menu in MENU_ORDER:
        g = cells[(cells.menu == menu) & (cells.path == "ALL")]
        comp.append({"菜單": menu, "gain_2(p*)": g.gain_pstar_pp.mean(), "其餘的平均": g.gain_others_pp.mean(), "D1": g.D1_pp.mean(),
                     "選擇半平手（%）": 100 * g.tie_share.mean()})
    m3_blocks = cells[(cells.menu != MAIN_MENU) & (cells.path == "ALL")].copy()
    m3_blocks["model"] = m3_blocks.model.map(MODEL_LABELS)
    out += ["### 8.2 判定一的選擇", "",
            "p* 落在各條 path 的比例（%），以及選擇半出現平手的切分比例：", "", md(share, ".1f"), "",
            "gain_2(p*) 與其餘各條 gain_2 的平均（pp，16 個區塊的平均；M12 的逐區塊值在第 (3) 節）：", "",
            md(pd.DataFrame(comp)), "", "M3 菜單的逐區塊值（pp）：", "",
            md(m3_blocks[["menu", "model", "dataset", "gain_pstar_pp", "gain_others_pp", "D1_pp", "tie_share"]].rename(
                columns={"menu": "菜單", "model": "模型", "dataset": "資料集", "gain_pstar_pp": "gain_2(p*)",
                         "gain_others_pp": "其餘的平均", "D1_pp": "D1", "tie_share": "平手比例"})), ""]

    base0 = curves_df[curves_df.target == "+0"].set_index(["model", "dataset", "path"])
    crow = []
    for target in CURVE_TARGETS:
        g = curves_df[curves_df.target == target]
        done = g[~g.skipped]
        b = base0.loc[list(zip(done.model, done.dataset, done.path))]
        crow.append({"目標": target, "算出的點": len(done), "略過的點": int(g.skipped.sum()),
                     "path 增加（pp）": 100 * float(np.mean(done.path_acc_H2.to_numpy() - b.path_acc_H2.to_numpy())) if len(done) else np.nan,
                     "V 增加（pp）": 100 * float(np.mean(done.V.to_numpy() - b.V.to_numpy())) if len(done) else np.nan,
                     "WV 增加（pp）": 100 * float(np.mean(done.WV.to_numpy() - b.WV.to_numpy())) if len(done) else np.nan,
                     "SB 增加（pp）": 100 * float(np.mean(done.SB.to_numpy() - b.SB.to_numpy())) if len(done) else np.nan})
    out += ["### 8.3 曲線資料", "",
            "完整資料在 `rq3g_curves.csv`（16 個區塊 × 12 條 × 6 個目標）。下表只是摘要：每個目標在算出的點上，"
            "相對於原本（+0）的平均增加量。各目標算出的點不同（越高的目標略過越多），所以各列不能直接互比。", "",
            md(pd.DataFrame(crow)), ""]

    agg_rows = []
    for key, before, after, sim in (("V", "mean_V_before", "mean_A_real_V", "mean_A_sim_V"),
                                    ("WV", "mean_WV_before", "mean_WV_after", "mean_WV_sim"),
                                    ("SB", "mean_SB_before", "mean_SB_after", "mean_SB_sim"),
                                    ("DS", "mean_DS_before", "mean_DS_after", None),
                                    ("Oracle", "mean_Oracle_before", "mean_Oracle_after", None)):
        agg_rows.append({"聚合": key, "替換前（%）": 100 * weak[before].mean(), "真實替換後（%）": 100 * weak[after].mean(),
                         "A_sim（%）": 100 * weak[sim].mean() if sim else np.nan,
                         "真實 − 替換前（pp）": 100 * (weak[after] - weak[before]).mean(),
                         "真實 − A_sim（pp）": 100 * (weak[after] - weak[sim]).mean() if sim else np.nan})
    counts = weak[["model", "dataset", "mean_n_fixed", "mean_n_broken", "mean_n_wrong_to_other_wrong", "mean_host_acc_H2",
                   "mean_donor_acc_H2"]].copy()
    counts["model"] = counts.model.map(MODEL_LABELS)
    counts[["mean_host_acc_H2", "mean_donor_acc_H2"]] *= 100
    out += ["### 8.4 每個替換", "",
            f"逐替換的完整數字在 `rq3g_substitutions.csv`（M12 {len(m12_subs)} 列）。下表：每個弱模型區塊先對參與判定的替換平均，"
            "再對 8 個區塊平均（評分半；有效切分）。", "", md(pd.DataFrame(agg_rows)), "",
            f"DS 未收斂（20 輪內）的切分：替換前 {int(weak.DS_nonconverged_before.sum())} 次、替換後 "
            f"{int(weak.DS_nonconverged_after.sum())} 次（參與判定的替換 × 有效切分）。", "",
            "修對、弄壞、錯換成另一個錯答案的題數（整個子集二，不切分；區塊內替換的平均）與那條 path 的正確率（%）：", "",
            md(counts.rename(columns={"model": "宿主", "dataset": "資料集", "mean_n_fixed": "修對", "mean_n_broken": "弄壞",
                                      "mean_n_wrong_to_other_wrong": "錯換錯", "mean_host_acc_H2": "宿主 path",
                                      "mean_donor_acc_H2": "供體 path"}), ".1f"), ""]

    d5 = summarize(weak.D5_pp)
    per5 = weak[["model", "dataset", "D5_pp", f"{MAIN_MENU}_D2_pp"]].copy()
    per5["model"] = per5.model.map(MODEL_LABELS)
    out += ["### 8.5 只修不壞版本", "", md(pd.DataFrame([ciRow("D5 = A_fix − A_fix,sim（pp）", d5)]), text=True), "",
            md(per5.rename(columns={"model": "宿主", "dataset": "資料集", "D5_pp": "D5", f"{MAIN_MENU}_D2_pp": "（對照）D2"})), ""]

    cr, cs = summarize(weak.conversion_real), summarize(weak.conversion_sim)
    per6 = weak[["model", "dataset", "conversion_real", "conversion_sim"]].copy()
    per6["model"] = per6.model.map(MODEL_LABELS)
    out += ["### 8.6 轉換率（V 增加的正確率 ÷ path 增加的正確率）", "",
            "區塊值 = 區塊內參與判定的替換，分子的總和 ÷ 分母的總和。逐替換的比值在 `rq3g_substitutions.csv`（`conversion_real`、"
            "`conversion_sim`）。", "",
            md(pd.DataFrame([{"量": "真實", "平均": f"{cr['mean']:.3f}", "95% 區間": f"[{cr['ci_low']:.3f}, {cr['ci_high']:.3f}]"},
                             {"量": "隨機模型", "平均": f"{cs['mean']:.3f}", "95% 區間": f"[{cs['ci_low']:.3f}, {cs['ci_high']:.3f}]"}]),
               text=True), "",
            md(per6.rename(columns={"model": "宿主", "dataset": "資料集", "conversion_real": "真實", "conversion_sim": "隨機模型"}), ".3f"), ""]

    per7 = weak[["model", "dataset", "spearman7", "spearman7_undefined"]].copy()
    per7["model"] = per7.model.map(MODEL_LABELS)
    out += ["### 8.7 預測（選擇半）與實際（評分半）的 Spearman 相關", "",
            "每次切分對區塊內參與判定、該次有效的替換算一次，再對有定義的切分平均。", "",
            md(per7.rename(columns={"model": "宿主", "dataset": "資料集", "spearman7": "Spearman",
                                    "spearman7_undefined": "無定義的切分"}), ".3f"), "",
            f"8 個區塊的平均：{weak.spearman7.mean():.3f}。", ""]

    per8 = weak[["model", "dataset", "spearman8_V", "spearman8_WV", "spearman8_V_undefined", "spearman8_WV_undefined"]].copy()
    per8["model"] = per8.model.map(MODEL_LABELS)
    lead = m12_subs[m12_subs.participates]
    out += ["### 8.8 領先幅度", "",
            "領先幅度 = 替換後那條 path 在選擇半的正確率 − 其餘 11 條在選擇半的最高正確率；逐替換的值（有效切分平均）在 "
            "`rq3g_substitutions.csv`（`lead_H1_pp`、`V_minus_SB_after_pp`、`WV_minus_SB_after_pp`）。"
            "Spearman 照 8.7 的做法逐切分算再平均。", "",
            md(per8.rename(columns={"model": "宿主", "dataset": "資料集", "spearman8_V": "領先 vs V − SB",
                                    "spearman8_WV": "領先 vs WV − SB", "spearman8_V_undefined": "無定義（V）",
                                    "spearman8_WV_undefined": "無定義（WV）"}), ".3f"), "",
            f"8 個區塊的平均：領先 vs V − SB {weak.spearman8_V.mean():.3f}；領先 vs WV − SB {weak.spearman8_WV.mean():.3f}。"
            f"參與判定的替換中，領先幅度 > 0 的有 {int((lead.lead_H1_pp > 0).sum())} / {len(lead)} 個。", ""]

    m3 = []
    for menu in MENU_ORDER:
        if menu == MAIN_MENU:
            continue
        g = cells[(cells.menu == menu) & (cells.path == "ALL")]
        m3.append(ciRow(f"{menu} 判定一的量：D1（16 個區塊）", summarize(g.D1_pp)))
        m3.append(ciRow(f"{menu} 判定二的量：D2（8 個區塊）", summarize(weak[f"{menu}_D2_pp"])))
    m3_part = {menu: f"{int(weak[f'{menu}_n_participating'].sum())} / {int(weak[f'{menu}_n_subs'].sum())}"
               for menu in MENU_ORDER if menu != MAIN_MENU}
    out += ["### 8.9 M3L、M3S、M3P（只用 V，不套四種狀態）", "", md(pd.DataFrame(m3), text=True), "",
            "參與判定的替換：" + "；".join(f"{menu} {v}" for menu, v in m3_part.items()) + "。逐 path 的量在 `rq3g_cells.csv`。", ""]

    out += ["## (6) 對照讀法的結論", "",
            f"- 判定一（D1 {pp(j1s['mean'])}pp，[{pp(j1s['ci_low'])}, {pp(j1s['ci_high'])}]，{j1s['n_positive']}/16 為正）：**{state1}**。"
            f"{READINGS_1[state1]}",
            f"- 判定二（D2 {pp(j2s['mean'])}pp，[{pp(j2s['ci_low'])}, {pp(j2s['ci_high'])}]，{j2s['n_positive']}/8 為正）：**{state2}**。"
            f"{READINGS_2[state2]}",
            "- 其餘都只報告，不套四種狀態。", ""]

    Path(os.path.join(args.out_dir, "report.md")).write_text("\n".join(out), encoding="utf-8")
    print(f"判定一 D1 {pp(j1s['mean'])} [{pp(j1s['ci_low'])}, {pp(j1s['ci_high'])}] {j1s['n_positive']}/16 -> {state1}")
    print(f"判定二 D2 {pp(j2s['mean'])} [{pp(j2s['ci_low'])}, {pp(j2s['ci_high'])}] {j2s['n_positive']}/8 -> {state2}")
    print(f"-> {args.out_dir}/report.md（{time.time() - started:.0f}s）")


if __name__ == "__main__":
    main()
