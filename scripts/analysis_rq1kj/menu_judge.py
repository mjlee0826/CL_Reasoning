"""
menu_judge.py — RQ1-KJ 的分析：K 條 path 的 Judge 版本 vs 單一最強 path（result/analysis/rq1kj/rq1kj_criteria.md）

先跑完 scripts/analysis_rq1kj/run_menu_judge.py 的 reproduce → template → check → pilot → full。判定標準確認前不執行。
    1. 讀四項開跑前檢查的結果檔（§6），都要寫於同一份判定標準之下且通過
    2. 核對 Judge 檔（任何一項不符就停）：每個區塊 × 菜單都存在、prompt_version 為 choice-k-v1、
       剛好涵蓋子集內的不一致題、候選順序等於 §3.3 的分組旋轉
    3. 每個區塊 × 菜單、每次切分（makeSplits(n, 200, seed=0)）：A_J、A_V、S_in、EN 與分解；
       逐切分核對恆等式（§7.5，Judge 與多數決各一，誤差 > 1e-9 就停）
    4. 三項判定（§5）與 §7 的報告量
    5. 輸出（--out-dir）：rq1kj_blocks.csv、rq1kj_compare.csv、judge_outputs/items.csv.gz、models.csv、
       excess_menus.png、excess_j_m12_blocks.png、report.md（順序依 §8）

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq1kj/menu_judge.py
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import glob
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Aggregator.MenuJudgeAggregator import MenuJudgeAggregator
from Strategy.MenuJudge import MenuJudge, ORDER_GROUPED
from Analysis.alignment import loadRecords
from Analysis.preregistration import sha256, confirmationLine, loadStepFile
from Analysis.blockStats import summarize, fourState, FORWARD, REVERSE, EQUIVALENT
from Analysis.menuVote import regrets
from Analysis.menuJudge import (MODELS, DATASETS, MENU_NAMES, MENUS, MAIN_MENU, COMPARE_MENUS, TWO_PATH, THRESHOLD, SEED, REPS,
                                TOLERANCE, AGGREGATOR_SEED, PRICES, JUDGE_SETTINGS, OUT_DIR, CRITERIA_FILE, JUDGE_DIR,
                                PRECHECK_DIR, REP2_DIR, REPRODUCE_FILE, TEMPLATE_FILE, PRECHECK_FILE, PILOT_FILE, RUNNER,
                                codesOf, armIdsOf, loadMenuRecords, mainGridJudgePath)
from Analysis.menuJudgeStats import (FAILURE_KINDS, blocksWithSplits, judgeArrays, evaluateBlockMenu, compareJudge,
                                     twoPathDiff, itemComparison, skepticComparison, positionStats)

MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
# 類別色（依固定順序指派給模型 / 序列）與墨色；圖上用英文（matplotlib 預設字型沒有中文字）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#d9d8d2"
DIFF_COLUMNS = {"excess_J": "Excess_J", "excess_V": "Excess_V", "diff_JV": "Diff_JV"}


def parseArgs():
    parser = ArgumentParser(description="RQ1-KJ analysis: K-path Judge vs the strongest single path (rq1kj_criteria.md)")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations", help="Main-grid two-path judge files (§7.7)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    return parser.parse_args()


def pp(value, sign: bool = True, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{value:{'+' if sign else ''}.{digits}f}"


def ciText(summary: dict) -> str:
    return f"{pp(summary['mean'])} [{pp(summary['ci_low'])}, {pp(summary['ci_high'])}]"


def summaryRow(name: str, values, state: bool = True) -> dict:
    s = summarize(values)
    row = {"": name, "區塊數": s["n_blocks"], "平均（pp）": pp(s["mean"]), "SE": pp(s["se"], False),
           "95% 區間": f"[{pp(s['ci_low'])}, {pp(s['ci_high'])}]", "為正的區塊": f"{s['n_positive']}/{s['n_blocks']}"}
    if state:
        row["狀態"] = fourState(s, THRESHOLD)
    return row


def blockLabel(model: str, dataset: str) -> str:
    return f"{MODEL_LABELS[model]} | {dataset}"


# ------------------------------------------------------------------
# 載入與核對
# ------------------------------------------------------------------
def loadJudge(judge_dir: str, block, menu: str, problems: list):
    """一個區塊 × 菜單的 Judge 紀錄 -> JudgeArrays；不符 §2–§3 的地方記進 problems（之後停）。"""
    label = f"{block.model} | {block.dataset} | {menu}"
    meta, records = loadMenuRecords(judge_dir, block.model, block.dataset, menu)
    if not meta:
        problems.append(f"{label}: no judge file")
        return None
    found = (meta.get("prompt_version"), meta.get("menu"), meta.get("order_scheme"), meta.get("candidate_arms"),
             meta.get("Aggregator", {}).get("seed"), meta.get("Aggregator", {}).get("k"))
    expected = (MenuJudgeAggregator.PROMPT_VERSION, menu, ORDER_GROUPED, armIdsOf(menu), AGGREGATOR_SEED, len(armIdsOf(menu)))
    if found != expected:
        problems.append(f"{label}: metadata {found} != {expected}")
        return None
    try:
        judge = judgeArrays(block, codesOf(menu), records, label)
    except ValueError as e:
        problems.append(str(e))
        return None
    orders = MenuJudge.groupedOrders(armIdsOf(menu), sorted(records), AGGREGATOR_SEED)
    wrong = [i for i, r in records.items() if r["presentation_order"] != orders[i]]
    if wrong:
        problems.append(f"{label}: {len(wrong)} presentation orders differ from §3.3 (e.g. {wrong[:5]})")
        return None
    return judge, records


def twoPathCorrect(aggdir: str, block, menu: str) -> np.ndarray:
    """§7.7：主網格兩條 path 的 Judge 的最終答案是否正確（一致題 = 共同答案；沒有答案算錯）。"""
    path = mainGridJudgePath(aggdir, block.model, block.dataset, TWO_PATH[menu])
    _, records = loadRecords(path)
    missing = [int(i) for i in block.item_ids if int(i) not in records]
    if missing:
        raise SystemExit(f"❌ {path} misses {len(missing)} items (e.g. {missing[:5]})")
    return np.array([block.compare(g, records[int(i)]["final_answer"]) for g, i in zip(block.gold, block.item_ids)], dtype=bool)


def itemRows(block, menu: str, result: dict, records: dict) -> list[dict]:
    """judge_outputs/items.csv.gz：逐題的 Judge 原始輸出（只有呼叫過 Judge 的題目）。"""
    judge, rows = result["judge"], []
    index = {int(i): k for k, i in enumerate(block.item_ids)}
    for item_id, record in sorted(records.items()):
        k = index[item_id]
        trace, call = record.get("trace") or {}, record.get("call") or {}
        rows.append({
            "model": block.model, "dataset": block.dataset, "menu": menu, "item_id": item_id, "gold": block.gold[k],
            "presentation_order": "|".join(record["presentation_order"]), "choice": trace.get("choice"),
            "chosen_path": judge.chosen[k], "chosen_arm": trace.get("chosen_arm"), "final_answer": record["final_answer"],
            "correct": bool(judge.correct[k]), "vote_answer": result["vote_finals"][k], "vote_correct": bool(result["vote_correct"][k]),
            **{kind: bool(judge.failure[kind][k]) for kind in FAILURE_KINDS},
            "tokens_in": record["tokens_in"], "tokens_out": record["tokens_out"],
            "usage_in": call.get("usage_in"), "usage_out": call.get("usage_out"),
            "prompt_version": MenuJudgeAggregator.PROMPT_VERSION, "prompt_sha256": trace.get("prompt_sha256"),
            "model_version": call.get("model_version"), "called_at": call.get("called_at"), "judge_output": trace.get("judge_output"),
        })
    return rows


def modelRecords(out_dir: str) -> pd.DataFrame:
    """每個模型的版本 ID（逐次呼叫回傳的）、呼叫時間、解碼與 thinking 設定、價格。"""
    rows = []
    for model in MODELS:
        versions, first, last, names = {}, [], [], set()
        for directory in (JUDGE_DIR, PRECHECK_DIR, REP2_DIR):
            for path in sorted(glob.glob(os.path.join(out_dir, directory, model, "*", "*.json"))):
                with open(path, encoding="utf-8") as f:
                    metadata = json.load(f)[0]
                for version, count in metadata.get("model_versions", {}).items():
                    versions[version] = versions.get(version, 0) + count
                if metadata.get("calls_utc"):
                    first.append(metadata["calls_utc"]["first"])
                    last.append(metadata["calls_utc"]["last"])
                names.add(metadata["Model"]["modelName"])
        rows.append({"model": model, "modelName": ", ".join(sorted(names)), "settings": JUDGE_SETTINGS[model],
                     "versions": "; ".join(f"{v} ×{c}" for v, c in sorted(versions.items())),
                     "calls_first_utc": min(first) if first else None, "calls_last_utc": max(last) if last else None,
                     "price_usd_per_M_in_out": f"{PRICES[model][0]} / {PRICES[model][1]}"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 圖
# ------------------------------------------------------------------
def styleAxes(ax, axis: str):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="best")


def plotMenus(summaries: dict, path: str):
    """各菜單的 Excess_J 與 Excess_V：跨 16 個區塊的平均與 95% t 區間。"""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(MENU_NAMES))
    for i, (column, label) in enumerate((("excess_J", "Excess_J (Judge − strongest in menu)"),
                                         ("excess_V", "Excess_V (majority vote − strongest in menu)"))):
        s = [summaries[(m, column)] for m in MENU_NAMES]
        ax.errorbar([x["mean"] for x in s], y + (0.15 if i else -0.15),
                    xerr=[[x["mean"] - x["ci_low"] for x in s], [x["ci_high"] - x["mean"] for x in s]],
                    fmt=MARKERS[i], color=SERIES[i], markersize=7, capsize=0, linewidth=2, markeredgecolor="white",
                    label=label, zorder=3)
    ax.axvline(0, color=INK, linewidth=1)
    for x in (THRESHOLD, -THRESHOLD):
        ax.axvline(x, color=MUTED, linewidth=1, linestyle="--")
    ax.set_yticks(y, [f"{m} (K={len(MENUS[m])})" for m in MENU_NAMES])
    ax.invert_yaxis()
    ax.set_xlabel("pp, mean over 16 blocks with 95% t-CI (dashed = ±0.5 pp)")
    ax.set_title("RQ1-KJ: aggregation − strongest single path, by menu", fontsize=10)
    styleAxes(ax, "x")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plotBlocks(blocks: pd.DataFrame, path: str):
    """M12 下 16 個區塊的 Excess_J：依資料集分組，模型 = 顏色 + 形狀；標出 0 與 ±門檻。"""
    main = blocks[blocks["menu"] == MAIN_MENU]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    offsets = np.linspace(-0.24, 0.24, len(MODELS))
    for i, model in enumerate(MODELS):
        sub = main[main["model"] == model].set_index("dataset").reindex(DATASETS)
        ax.scatter(np.arange(len(DATASETS)) + offsets[i], sub["excess_J"], s=64, marker=MARKERS[i], color=SERIES[i],
                   edgecolor="white", linewidth=1.5, zorder=3, label=MODEL_LABELS[model])
    ax.axhline(0, color=INK, linewidth=1)
    for y in (THRESHOLD, -THRESHOLD):
        ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(DATASETS)), DATASETS)
    ax.set_ylabel("Excess_J (pp): Judge − strongest single path")
    ax.set_title(f"RQ1-KJ, {MAIN_MENU}: Excess_J per block (dashed = ±{THRESHOLD} pp)", fontsize=10)
    styleAxes(ax, "y")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# 讀法（§5.3–§5.5，照判定標準檔的文字）
# ------------------------------------------------------------------
READING_1 = {
    FORWARD: "「聚合對最強單一 path 分不出正負」只適用於多數決；論文要寫 Judge 版本贏過最強單一 path，並報幅度與成本（§7.8）。",
    REVERSE: "Judge 版本輸給最強單一 path。",
    EQUIVALENT: "Judge 版本與最強單一 path 相當（差距在 ±0.5pp 內）。",
}
READING_1_UNDETERMINED = "兩種聚合方式下都分不出正負：報兩個區間（M12 的 Excess_J {j} 與 Excess_V {v}），不下「沒有效果」的結論。"
READING_2 = {
    FORWARD: "Judge 版本比多數決好。報幅度與 Judge 多花的成本（§7.8）。",
    REVERSE: "多數決比 Judge 版本好。",
    EQUIVALENT: "Judge 版本與多數決相當（差距在 ±0.5pp 內），Judge 多花的呼叫沒有換到正確率。",
}
READING_2_UNDETERMINED = "只報區間，不下誰比較好的結論。"
READING_3 = {
    FORWARD: "三條 path 的 Judge 下，同語言重抽比換語言好。",
    REVERSE: "「增益不需要換語言」在三條 path 的 Judge 下不成立，論文的這個主張要加上這個限制。",
}
READING_3_OTHER = "維持「換語言沒有比較好」的寫法，並報區間。"


def main():
    args = parseArgs()
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty)")
    criteria_sha = sha256(criteria)
    steps = {}
    for step, name in (("reproduce", REPRODUCE_FILE), ("template", TEMPLATE_FILE), ("check", PRECHECK_FILE), ("pilot", PILOT_FILE)):
        steps[step] = loadStepFile(os.path.join(args.out_dir, name), step, criteria_sha, RUNNER)
        if not steps[step].get("passed"):
            raise SystemExit(f"❌ {name}: the `{step}` step did not pass")

    judge_dir = os.path.join(args.out_dir, JUDGE_DIR)
    rows, compare_rows, items, skeptic, positions, problems = [], [], [], {}, {}, []
    for block, splits in blocksWithSplits(args.armdir, args.aggdir):
        results = {}
        for menu in MENU_NAMES:
            loaded = loadJudge(judge_dir, block, menu, problems)
            if loaded is None:
                continue
            judge, records = loaded
            result = evaluateBlockMenu(block, menu, splits, judge)
            row = result["row"]
            row.update(itemComparison(block, result))
            pos = positionStats(result)
            positions[(block.model, block.dataset, menu)] = pos
            row.update({"n_valid_choice": pos["n_valid"], "position_spread": pos["position_spread"],
                        **{f"pos_{p}": share for p, share in pos["position_share"].items()},
                        **{f"chosen_{c}": share for c, share in pos["path_share"].items()}})
            if menu in TWO_PATH:
                row["vs_two_path_judge"] = twoPathDiff(splits, result["sub"], judge.correct, twoPathCorrect(args.aggdir, block, menu))
                row["two_path"] = TWO_PATH[menu]
            if menu == MAIN_MENU and block.dataset == "truthfulqa":
                skeptic[block.model] = skepticComparison(block, result)
            rows.append(row)
            items += itemRows(block, menu, result, records)
            results[menu] = result
        if all(m in results for m in COMPARE_MENUS):
            compare_rows.append({"model": block.model, "dataset": block.dataset,
                                 "m3s_minus_m3l_J": compareJudge(block, splits, results[COMPARE_MENUS[0]]["judge"],
                                                                 results[COMPARE_MENUS[1]]["judge"])})
        print(f"📊 {block.model} | {block.dataset}: {len(results)} menus")
    if problems:
        raise SystemExit("❌ Stopping (rq1kj_criteria.md §2–§3):\n  " + "\n  ".join(problems[:50]))
    print("✅ Every judge file is complete, written with choice-k-v1, in the §3.3 orders")

    blocks = pd.DataFrame(rows)
    bad = blocks[(blocks["identity_max_error_J"] > TOLERANCE) | (blocks["identity_max_error_V"] > TOLERANCE)]
    if not bad.empty:
        raise SystemExit("❌ §7.5 identity check failed, stopping:\n"
                         + bad[["model", "dataset", "menu", "identity_max_error_J", "identity_max_error_V"]].to_string())
    max_error = float(blocks[["identity_max_error_J", "identity_max_error_V"]].max().max())
    print(f"✅ §7.5 identity: {len(blocks) * REPS} splits, max error {max_error:.2e}")
    compare = pd.DataFrame(compare_rows)

    # ---------------- 判定（§5） ----------------
    main_rows = blocks[blocks["menu"] == MAIN_MENU]
    j1, j1v = summarize(main_rows["excess_J"]), summarize(main_rows["excess_V"])
    j2, j3 = summarize(main_rows["diff_JV"]), summarize(compare["m3s_minus_m3l_J"])
    j1_state, j2_state, j3_state = fourState(j1, THRESHOLD), fourState(j2, THRESHOLD), fourState(j3, THRESHOLD)
    summaries = {(menu, column): summarize(blocks[blocks["menu"] == menu][column]) for menu in MENU_NAMES for column in DIFF_COLUMNS}

    # ---------------- 輸出 ----------------
    os.makedirs(args.out_dir, exist_ok=True)
    blocks.to_csv(os.path.join(args.out_dir, "rq1kj_blocks.csv"), index=False)
    compare.to_csv(os.path.join(args.out_dir, "rq1kj_compare.csv"), index=False)
    pd.DataFrame(items).to_csv(os.path.join(judge_dir, "items.csv.gz"), index=False, compression={"method": "gzip", "mtime": 0})
    models = modelRecords(args.out_dir)
    models.to_csv(os.path.join(args.out_dir, "models.csv"), index=False, encoding="utf-8-sig")
    plotMenus(summaries, os.path.join(args.out_dir, "excess_menus.png"))
    plotBlocks(blocks, os.path.join(args.out_dir, "excess_j_m12_blocks.png"))

    def blockTable(column: str, scale: float = 1.0, digits: int = 2, menus=MENU_NAMES) -> str:
        table = blocks.pivot_table(index=["model", "dataset"], columns="menu", values=column).reindex(columns=list(menus))
        table = (table * scale).round(digits)
        table.index = [blockLabel(m, d) for m, d in table.index]
        return table.to_markdown()

    def by(key: str, column: str) -> pd.DataFrame:
        table = blocks.pivot_table(index="menu", columns=key, values=column).reindex(MENU_NAMES).round(2)
        return table.rename(columns=MODEL_LABELS) if key == "model" else table

    # ---------------- report.md ----------------
    reproduce, template, precheck, pilot = steps["reproduce"], steps["template"], steps["check"], steps["pilot"]
    L = [
        "# RQ1-KJ 結果：K 條 path 的 Judge 版本 vs 同一份菜單內最強的單一 path",
        "",
        f"- 產生時間：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        f"- 輸入：`{args.armdir}`（候選）、`{judge_dir}`（Judge 輸出）、`{args.aggdir}`（§7.7 的兩條 path Judge）；"
        f"切分 = `makeSplits(n, {REPS}, seed={SEED})`（與 RQ1、RQ1-K 相同）",
        "- 單位：正確率為 %，Excess / Diff 為百分點（pp）；區間 = 跨 16 個區塊的 95% t 區間",
        "",
        "## (1) 判定標準",
        "",
        f"- `{criteria}`，sha256 `{criteria_sha}`",
        f"- 確認：{confirmed}",
        "",
        "## (2) 開跑前的四項檢查（§6）",
        "",
        f"- **§6.1 離線重現**（{reproduce['generated_at']}）：{reproduce['n_rows']} 列，和 `{reproduce['reference']}` 的最大誤差 "
        + "、".join(f"{c} {v:.1e}" for c, v in reproduce["max_abs_diff"].items()) + f"（門檻 {TOLERANCE:.0e}），通過。",
        f"- **§6.2 模板核對**（{template['generated_at']}）：choice-k-v1 在 K = 2 時渲染 {template['dataset']} × {template['menu']}，"
        "重算 tokens 等於主網格的 tokens_in："
        + "、".join(f"{MODEL_LABELS[m]} {r['n_equal']}/{r['n']}" for m, r in template["models"].items())
        + f"。模板檔最後一次變動（git）：{template['git_log']}。措辭差異：無（同一個函式）。",
        f"- **§6.3 流程核對**（{precheck['generated_at']}）：重跑「自己裁決自己」的 {precheck['dataset']} × {precheck['menu']}，"
        "用主網格記錄的順序，逐題和主網格比對：",
        "",
        pd.DataFrame([{"模型": MODEL_LABELS[m], "題數": r["n"], "選擇相同": f"{r['n_agree']}（{100 * r['agreement']:.1f}%）",
                       "RQ2 核對時": f"{r['rq2_n_agree']}/{r['rq2_n']}（{100 * r['rq2_agreement']:.1f}%）",
                       "差（pp）": pp(100 * -r["drop"], digits=1), "最終答案相同": r["n_same_final"],
                       "tokens_in 相同": r["n_same_tokens_in"], "輸出文字相同": r["n_same_output_text"],
                       "回傳的版本": ", ".join(r["returned_versions"]),
                       "與 arm 紀錄相同": "是" if r["version_matches_arms"] else "否",
                       "與 RQ2 紀錄相同": "是" if r["version_matches_rq2"] else "否"}
                      for m, r in precheck["models"].items()]).to_markdown(index=False),
        "",
        "  沒有任何模型比 RQ2 核對時低 10 個百分點以上。DeepSeek 的 API 只回傳別名 `deepseek-flash`，只能核對別名。",
        "",
        f"- **§6.4 試跑**（{pilot['generated_at']}）：每個模型 M12 100 題（四個資料集各 25 題），各跑兩次；"
        f"全量成本估算 ${pilot['full_cost_usd']:.2f}（門檻 ${pilot['max_cost_usd']:.0f}）。",
        "",
        pd.DataFrame([{"模型": MODEL_LABELS[m], "無效（第一次）": f"{r['run1']['invalid']}/{r['n']}",
                       "找不到 choice": r["run1"]["no_choice"], "超出範圍": r["run1"]["out_of_range"], "被擋下": r["run1"]["refused"],
                       "API input 平均 / 最大": f"{r['tokens_in_api']['mean']:.0f} / {r['tokens_in_api']['max']:.0f}",
                       "API output 平均 / 最大": f"{r['tokens_out_api']['mean']:.0f} / {r['tokens_out_api']['max']:.0f}",
                       "全量最大 input（估計）": f"{r['full_max_input_estimate']:.0f}", "context 上限": r["context_limit"]["context"],
                       "全量成本（$）": f"{r['full_cost_usd']:.2f}"} for m, r in pilot["models"].items()]).to_markdown(index=False),
        "",
        "## (3) 解析失敗與重跑一致率",
        "",
        "試跑第二次與第一次（正式）的比較（每個模型 100 題）：",
        "",
        pd.DataFrame([{"模型": MODEL_LABELS[m], "兩次都完成": r["n_both_runs"],
                       "選到同一個候選": f"{r['same_choice']}（{100 * r['same_choice_rate']:.1f}%）",
                       "最終答案相同": f"{r['same_final']}（{100 * r['same_final_rate']:.1f}%）",
                       "兩次都有效": r["n_both_valid"], "第二次無效": r["run2"]["invalid"]}
                      for m, r in pilot["models"].items()]).to_markdown(index=False),
        "",
        "全量的失敗比例（§7.10；分母 = Judge 呼叫數；agg_no_answer = 三種合計，算錯）：",
        "",
    ]
    failures = blocks.groupby(["model", "menu"])[["n_calls", "n_no_choice", "n_out_of_range", "n_refused", "n_agg_no_answer"]].sum()
    for kind in ("no_choice", "out_of_range", "refused", "agg_no_answer"):
        failures[f"{kind} (%)"] = (100 * failures[f"n_{kind}"] / failures["n_calls"]).round(2)
    failures = failures.reset_index()
    failures["model"] = failures["model"].map(MODEL_LABELS)
    L += [failures.to_markdown(index=False), ""]

    total = []
    for menu in MENU_NAMES:
        sub = blocks[blocks["menu"] == menu]
        row = {"菜單": menu, "K": len(MENUS[menu]), "A_J": f"{100 * sub['A_J'].mean():.2f}", "A_V": f"{100 * sub['A_V'].mean():.2f}",
               "S_in": f"{100 * sub['S_in'].mean():.2f}", "EN": f"{100 * sub['EN'].mean():.2f}"}
        for column, name in DIFF_COLUMNS.items():
            s = summaries[(menu, column)]
            row[f"{name} [95%]"] = ciText(s)
            row[f"{name} 為正"] = f"{s['n_positive']}/{s['n_blocks']}"
        total.append(row)
    L += ["## (4) 總表（跨 16 個區塊的平均）", "", pd.DataFrame(total).to_markdown(index=False), "",
          "圖：`excess_menus.png`（各菜單的 Excess_J 與 Excess_V）、`excess_j_m12_blocks.png`（M12 的 16 個區塊）。", ""]

    reading1 = READING_1.get(j1_state) or READING_1_UNDETERMINED.format(j=ciText(j1), v=ciText(j1v))
    reading2 = READING_2.get(j2_state, READING_2_UNDETERMINED)
    reading3 = READING_3.get(j3_state, READING_3_OTHER)
    L += ["## (5) 三項判定（門檻 0.5pp）", "",
          pd.DataFrame([summaryRow(f"判定一：{MAIN_MENU} 的 Excess_J（正向 = Judge 版本的聚合較好）", main_rows["excess_J"]),
                        summaryRow(f"判定二：{MAIN_MENU} 的 Diff_JV（正向 = Judge 較好）", main_rows["diff_JV"]),
                        summaryRow(f"判定三：{COMPARE_MENUS[0]} 的 A_J − {COMPARE_MENUS[1]} 的 A_J（正向 = 同語言重抽較好）",
                                   compare["m3s_minus_m3l_J"])]).to_markdown(index=False),
          "",
          f"- **判定一：{j1_state}**。{reading1}（參考：同一批區塊的 Excess_V {ciText(j1v)}。）",
          f"- **判定二：{j2_state}**。{reading2}",
          f"- **判定三：{j3_state}**。{reading3}",
          "- 其他菜單的數字、分資料集與分模型的數字都只報告，不參與判定。",
          ""]

    tq = main_rows[main_rows["dataset"] == "truthfulqa"].set_index("model").reindex(MODELS)
    all_diff_positive = bool((tq["diff_JV"] > 0).all())
    all_excess_nonneg = bool((tq["excess_J"] >= 0).all())
    L += ["## (6) TruthfulQA（事先指定的次要分析，不套四種狀態）", "",
          pd.DataFrame({"模型": [MODEL_LABELS[m] for m in MODELS], "Diff_JV（pp）": [pp(v) for v in tq["diff_JV"]],
                        "Excess_J（pp）": [pp(v) for v in tq["excess_J"]], "Excess_V（pp）": [pp(v) for v in tq["excess_V"]]}
                       ).to_markdown(index=False),
          "",
          "- " + ("4 個 Diff_JV 全為正：Judge 在 TruthfulQA 上比多數決好。" if all_diff_positive else "Diff_JV 不是 4 個全為正：只列數字。"),
          "- " + ("4 個 Excess_J 全不為負：Judge 補回了多數決在 TruthfulQA 的損失。" if all_excess_nonneg
                  else "Excess_J 不是 4 個全不為負：只列數字。"),
          "",
          "P:skeptic 的答案和多數決不同的題目（M12 子集，全部題目、不切分）：",
          "",
          pd.DataFrame([{"模型": MODEL_LABELS[m], "題數": s["n"],
                         **({"Judge = skeptic": f"{100 * s['judge_is_skeptic']:.1f}%", "Judge = 多數決": f"{100 * s['judge_is_vote']:.1f}%",
                             "其他": f"{100 * s['judge_other']:.1f}%", "skeptic 正確率": f"{100 * s['acc_skeptic']:.1f}%",
                             "多數決正確率": f"{100 * s['acc_vote']:.1f}%", "Judge 正確率": f"{100 * s['acc_judge']:.1f}%"} if s["n"] else {})}
                        for m, s in skeptic.items()]).to_markdown(index=False),
          ""]

    L += ["## (7) 其餘只報告的量", "",
          "### 每個菜單的逐區塊 Excess_J（pp）", "", blockTable("excess_J"), "",
          "### 每個菜單的逐區塊 Excess_V（pp）", "", blockTable("excess_V"), "",
          "### 每個菜單的逐區塊 Diff_JV（pp）", "", blockTable("diff_JV"), "",
          "### 保留比例與不一致題比例（%）", "", blockTable("keep", 100, 1), "", blockTable("disagree_rate", 100, 1), "",
          "### 分資料集的平均（pp）", "",
          *[x for column, name in DIFF_COLUMNS.items() for x in (f"{name}：", "", by("dataset", column).to_markdown(), "")],
          "### 分模型的平均（pp）", "",
          *[x for column, name in DIFF_COLUMNS.items() for x in (f"{name}：", "", by("model", column).to_markdown(), "")],
          "### H1 上被選為 S_in 的 path（前兩名與 200 次中被選中的比例）", ""]
    chosen = blocks.pivot_table(index=["model", "dataset"], columns="menu", values="S_in_top", aggfunc="first").reindex(columns=MENU_NAMES)
    chosen.index = [blockLabel(m, d) for m, d in chosen.index]
    L += [chosen.to_markdown(), ""]

    comparison = blocks.groupby("menu")[["n_judged", "same_answer_rate", "both_right", "both_wrong", "vote_right_judge_wrong",
                                         "vote_wrong_judge_right", "n_plurality", "plurality_ties", "n_minority", "n_absent"]].sum()
    for name in ("plurality", "minority", "absent"):
        rates = blocks.assign(right=blocks[f"judge_right_{name}"] * blocks[f"n_{name}"]).groupby("menu")["right"].sum()
        comparison[f"Judge 選對（{name}）"] = (100 * rates / comparison[f"n_{name}"]).round(1)
    comparison["same_answer_rate"] = (100 * blocks.assign(x=blocks["same_answer_rate"] * blocks["n_judged"]).groupby("menu")["x"].sum()
                                      / comparison["n_judged"]).round(1)
    L += ["### Judge 與多數決的逐題對照（§7.4；不一致題，16 個區塊合計，不切分）", "",
          "same_answer_rate = 兩者答案相同的比例（%）；正確答案的位置：plurality = 最多票（含平手，平手題數另列）、"
          "minority = 在候選中但票數較少、absent = 不在候選中。逐區塊的數字在 `rq1kj_blocks.csv`。", "",
          comparison.reindex(MENU_NAMES).to_markdown(), ""]

    decomposition = blocks.groupby("menu")[["d", "c", "m", "headroom_pp", "recovery_J", "recovery_V", "recovery_blind",
                                            "excess_J", "undefined_splits"]].mean().reindex(MENU_NAMES)
    L += ["### K 條 path 的分解（§7.5；跨區塊平均；H2 ∩ 子集，用 S_in，對切分平均）", "",
          "各量先對切分平均，所以平均後的乘積不必等於平均 Excess；恆等式 Excess_J = headroom × (recovery_J − recovery_blind) "
          f"在每次切分各自核對：{len(blocks) * REPS} 次切分，最大誤差 {max_error:.1e}（門檻 {TOLERANCE:.0e}），全部通過。"
          f"recovery 無定義的切分：{int(blocks['undefined_splits'].sum())} 次。", "",
          decomposition.round(3).to_markdown(), ""]

    pooled_pos, pooled_path = [], []
    for menu in MENU_NAMES:
        keys = [k for k in positions if k[2] == menu]
        n_valid = sum(positions[k]["n_valid"] for k in keys)
        k = len(MENUS[menu])
        pooled_pos.append({"菜單": menu, **{f"位置 {p}": f"{100 * sum(positions[x]['position_share'][p] * positions[x]['n_valid'] for x in keys) / n_valid:.1f}"
                                            for p in range(1, k + 1)}})
        pooled_path.append({"菜單": menu, **{c: f"{100 * sum(positions[x]['path_share'][c] * positions[x]['n_valid'] for x in keys) / n_valid:.1f}"
                                            for c in MENUS[menu]}})
    L += ["### 位置與 path（§7.6；有效選擇，16 個區塊合計，%）", "",
          "被選中的候選落在各位置的比例：", "", pd.DataFrame(pooled_pos).fillna("").to_markdown(index=False), "",
          "被選中的候選來自各 path 的比例：", "", pd.DataFrame(pooled_path).fillna("").to_markdown(index=False), "",
          f"順序打散的核對：每個區塊 × 菜單中，每條 path 在各位置出現次數的最大差距最多為 {int(blocks['position_spread'].max())}"
          "（§3.3 的設計是 ≤ 1）。各區塊的位置比例與 path 比例在 `rq1kj_blocks.csv`（pos_*、chosen_*）。", ""]

    two = blocks[blocks["menu"].isin(TWO_PATH)]
    L += ["### 和兩條 path 的 Judge 對照（§7.7；主網格現有結果，同一批題目，H2 對切分平均）", "",
          pd.DataFrame([summaryRow(f"{menu} 的 A_J − {TWO_PATH[menu]} Judge", two[two["menu"] == menu]["vs_two_path_judge"], state=False)
                        for menu in TWO_PATH]).to_markdown(index=False), ""]

    cost = blocks.groupby("menu")[["calls_per_item", "judge_tokens_in_per_call", "judge_tokens_out_per_call", "judge_api_in_per_call",
                                   "judge_api_out_per_call", "tokens_flow", "tokens_S_in"]].mean().reindex(MENU_NAMES)
    cost["輸出 tokens 倍數"] = cost["tokens_flow"] / cost["tokens_S_in"]
    L += ["### 成本（§7.8；跨區塊平均）", "",
          "calls_per_item = K 次生成 + 不一致題的 1 次 Judge；judge_tokens_* = 每次 Judge 呼叫（重算，與主網格相同的定義），"
          "judge_api_* = API 計費；tokens_flow = 每題整個流程的輸出 tokens（K 條 path + Judge），對 S_in 那一條。", "",
          cost.round(1).to_markdown(), ""]

    r = regrets(main_rows["excess_J"].to_numpy())
    L += ["### 決策空間（§7.9；只報告，不是 go / no-go）", "",
          f"用 {MAIN_MENU} 每個區塊的 Excess_J：永遠聚合的 regret {r['regret_aggregate']:.2f}pp，永遠用最強單一 path 的 regret "
          f"{r['regret_single']:.2f}pp，空間 = {r['space']:.2f}pp。", "",
          "### 模型版本與設定", "", models.to_markdown(index=False), ""]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"\n判定一（{MAIN_MENU} Excess_J）：{j1_state} {ciText(j1)}（Excess_V {ciText(j1v)}）")
    print(f"判定二（{MAIN_MENU} Diff_JV）：{j2_state} {ciText(j2)}")
    print(f"判定三（{COMPARE_MENUS[0]} − {COMPARE_MENUS[1]}，Judge）：{j3_state} {ciText(j3)}")
    print(f"💾 {args.out_dir}/: rq1kj_blocks.csv, rq1kj_compare.csv, models.csv, {JUDGE_DIR}/items.csv.gz, report.md, excess_*.png")


if __name__ == "__main__":
    main()
