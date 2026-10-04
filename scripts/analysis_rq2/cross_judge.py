"""
cross_judge.py — RQ2 交叉實驗的分析（result/analysis/rq2/rq2_criteria.md）

先跑完 scripts/analysis_rq2/run_cross_judge.py 的 check → pilot → full。判定標準確認前不執行。
    1. 讀 precheck.json（§4.1：哪些模型的對角線改用重跑結果）與 pilot.json（§4.2）
    2. 192 格的 §3 數字（Analysis/crossJudge.buildCells）。任何一項不符就停：
         沿用舊檔的對角線必須和 aggregation_cells.csv 的 both_answered 列完全相同；
         交叉檔必須剛好涵蓋兩條 path 都有答案的分歧題、呈現順序等於主網格紀錄；recovery 不可無定義
    3. §5–§7 的統計與 §6 的判定（Analysis/crossJudgeStats.py）
    4. 輸出（--out-dir）：cross_cells.csv、blocks.csv、models.csv、judge_outputs/items.csv.gz、
       heatmap_overall.png、heatmap_{dataset}.png、report.md（順序依 §8）

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq2/cross_judge.py
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import glob
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from Analysis.preregistration import sha256, confirmationLine
from Analysis.crossJudge import (MODELS, DATASETS, PAIRS, CHECK_DATASET, CHECK_PAIR, AGREEMENT_MIN,
                                 PILOT_MAX_RATE, PILOT_MAX_COST, PRICES, JUDGE_SETTINGS, OUT_DIR, CRITERIA_FILE, JUDGE_DIR,
                                 PRECHECK_DIR, PRECHECK_FILE, PILOT_FILE, SOURCE_MAIN, loadStepFile, mainGridPath, buildCells)
from Analysis.crossJudgeStats import (THRESHOLD, SAME_GROUP, OUTCOMES, UNDETERMINED, blockMeans, matrix, summarize, blockEffects,
                                      effectState, outcome, residuals, diagonalResiduals, sameGroupResiduals,
                                      selfPreference, mcnemar, practicalImpact)

# 圖上用英文（matplotlib 預設字型沒有中文字）
MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
# 單一色相的循序色階（淺 = 低、深 = 高）
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, SURFACE = "#1f1f1d", "#ffffff"


def parseArgs():
    parser = ArgumentParser(description="RQ2 cross-judge analysis (rq2_criteria.md)")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--cells", default="result/analysis/aggregation_cells.csv")
    parser.add_argument("--out-dir", default=OUT_DIR)
    return parser.parse_args()


def f3(value, digits: int = 3, sign: bool = True) -> str:
    return "—" if pd.isna(value) else f"{value:{'+' if sign else ''}.{digits}f}"


def ciRow(name: str, summary: dict, state: str | None = None) -> dict:
    row = {"效果": name, "區塊數": summary["n_blocks"], "平均": f3(summary["mean"]), "SE": f3(summary["se"], sign=False),
           "95% 區間": f"[{f3(summary['ci_low'])}, {f3(summary['ci_high'])}]",
           "為正的區塊": f"{summary['n_positive']}/{summary['n_blocks']}"}
    if state is not None:
        row["狀態"] = state
    return row


def markdownMatrix(table: pd.DataFrame, digits: int = 3) -> str:
    """4×4 表加上列平均與欄平均。"""
    shown = table.copy()
    shown["列平均"] = table.mean(axis=1)
    shown.loc["欄平均"] = list(table.mean(axis=0)) + [table.values.mean()]
    shown.index = [MODEL_LABELS.get(i, i) for i in shown.index]
    shown.columns = [MODEL_LABELS.get(c, c) for c in shown.columns]
    shown.index.name = "generator ＼ Judge"
    return shown.round(digits).to_markdown()


def heatmap(table: pd.DataFrame, path: str, title: str, vmin: float, vmax: float):
    """4×4 熱圖：列 = generator、欄 = Judge；對角線（自己裁決自己）加框。所有熱圖共用同一個色階範圍。"""
    cmap = LinearSegmentedColormap.from_list("rq2_blue", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    values = table.values.astype(float)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(values.shape[0]):
        for k in range(values.shape[1]):
            share = (values[i, k] - vmin) / (vmax - vmin) if vmax > vmin else 0
            ax.text(k, i, f"{values[i, k]:.3f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if share > 0.55 else INK)
        ax.add_patch(plt.Rectangle((i - 0.44, i - 0.44), 0.88, 0.88, fill=False, edgecolor=INK, linewidth=1.4, zorder=3))
    labels = [MODEL_LABELS[m] for m in MODELS]
    ax.set_xticks(range(len(MODELS)), labels, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(MODELS)), labels, fontsize=8)
    ax.set_xlabel("Judge")
    ax.set_ylabel("generator (candidates)")
    ax.set_xticks(np.arange(-0.5, len(MODELS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("recovery (split-half H2)", fontsize=8)
    colorbar.outline.set_visible(False)
    ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")   # tight_layout leaves the long y tick labels' axis title outside
    plt.close(fig)


def modelRecords(out_dir: str, aggdir: str) -> pd.DataFrame:
    """每個模型當 Judge 時的版本 ID、呼叫時間（新呼叫逐次記錄）與設定；主網格對角線只有檔案日期。"""
    rows = []
    for judge in MODELS:
        versions, first, last, names = {}, [], [], set()
        for path in sorted(glob.glob(os.path.join(out_dir, JUDGE_DIR, judge, "*", "*", "judge__*.json"))
                           + glob.glob(os.path.join(out_dir, PRECHECK_DIR, judge, "*", "*", "judge__*.json"))):
            with open(path, encoding="utf-8") as f:
                metadata = json.load(f)[0]
            for version, count in metadata.get("model_versions", {}).items():
                versions[version] = versions.get(version, 0) + count
            if metadata.get("calls_utc"):
                first.append(metadata["calls_utc"]["first"])
                last.append(metadata["calls_utc"]["last"])
            names.add(metadata["Model"]["modelName"])
        main_files = [mainGridPath(aggdir, judge, d, p) for d in DATASETS for p in PAIRS]
        mtimes = [datetime.fromtimestamp(os.path.getmtime(p), timezone.utc) for p in main_files if os.path.exists(p)]
        rows.append({
            "model": judge, "modelName": ", ".join(sorted(names)), "settings": JUDGE_SETTINGS[judge],
            "new_call_versions": "; ".join(f"{v} ×{c}" for v, c in sorted(versions.items())),
            "new_calls_first_utc": min(first) if first else None, "new_calls_last_utc": max(last) if last else None,
            "main_grid_judge_files_mtime_utc": f"{min(mtimes):%Y-%m-%d} – {max(mtimes):%Y-%m-%d}" if mtimes else None,
            "price_usd_per_M_in_out": f"{PRICES[judge][0]} / {PRICES[judge][1]}",
        })
    return pd.DataFrame(rows)


def main():
    args = parseArgs()
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty)")
    criteria_sha = sha256(criteria)
    precheck = loadStepFile(os.path.join(args.out_dir, PRECHECK_DIR, PRECHECK_FILE), "check", criteria_sha)
    pilot = loadStepFile(os.path.join(args.out_dir, PILOT_FILE), "pilot", criteria_sha)["pilot"]
    rerun = {model for model, r in precheck["models"].items() if r["rerun_diagonal"]}

    print("Building the 192 cells ...")
    cells, items, problems = buildCells(args.armdir, args.aggdir, os.path.join(args.out_dir, JUDGE_DIR), rerun, args.cells)
    if problems:
        raise SystemExit("❌ Stopping (rq2_criteria.md §2–§3):\n  " + "\n  ".join(problems[:50]))
    kept = sorted(set(MODELS) - rerun)
    if kept:
        print(f"✅ The main-grid diagonal of {kept} matches {args.cells} exactly (directly and via the cross path)")
    if rerun:
        print(f"ℹ️ Diagonal rerun through the cross path (§4.1): {sorted(rerun)}")
    print("✅ Every judge file is complete, with the recorded presentation orders")

    # §5–§6
    R = blockMeans(cells)
    judge_fx, candidate_fx = blockEffects(R, "judge"), blockEffects(R, "candidate")
    judge_sum, candidate_sum = summarize(judge_fx["delta"]), summarize(candidate_fx["delta"])
    judge_state, candidate_state = effectState(judge_sum), effectState(candidate_sum)
    result = outcome(judge_state, candidate_state)
    judge_incl, candidate_incl = blockEffects(R, "judge", False), blockEffects(R, "candidate", False)
    res = residuals(R)
    diag, same = diagonalResiduals(res), sameGroupResiduals(res)
    diag_sum, same_sum = summarize(diag["residual"]), summarize(same["residual"])
    self_pref = selfPreference(diag_sum)
    # §7
    tests = mcnemar(items)
    impact = practicalImpact(cells)
    impact_sum = summarize(impact["delta_pp"])

    # 輸出
    os.makedirs(args.out_dir, exist_ok=True)
    cells.to_csv(os.path.join(args.out_dir, "cross_cells.csv"), index=False, encoding="utf-8-sig")
    items_path = os.path.join(args.out_dir, JUDGE_DIR, "items.csv.gz")
    items.to_csv(items_path, index=False, compression={"method": "gzip", "mtime": 0})
    blocks = pd.concat([
        judge_fx.assign(kind="judge_effect", diagonal="excluded", pair="all"),
        candidate_fx.assign(kind="candidate_effect", diagonal="excluded", pair="all"),
        judge_incl.assign(kind="judge_effect", diagonal="included", pair="all"),
        candidate_incl.assign(kind="candidate_effect", diagonal="included", pair="all"),
        *[blockEffects(blockMeans(cells, pair=p.label), kind).assign(kind=f"{kind}_effect", diagonal="excluded", pair=p.label)
          for p in PAIRS for kind in ("judge", "candidate")],
    ], ignore_index=True)
    blocks.to_csv(os.path.join(args.out_dir, "blocks.csv"), index=False)
    res.to_csv(os.path.join(args.out_dir, "residuals.csv"), index=False)
    models = modelRecords(args.out_dir, args.aggdir)
    models.to_csv(os.path.join(args.out_dir, "models.csv"), index=False, encoding="utf-8-sig")

    overall = matrix(cells)
    tables = {d: matrix(cells, dataset=d) for d in DATASETS}
    vmin = min(0.0, min(t.values.min() for t in [overall, *tables.values()]))
    vmax = max(t.values.max() for t in [overall, *tables.values()])
    heatmap(overall, os.path.join(args.out_dir, "heatmap_overall.png"),
            "RQ2: recovery by generator × Judge (mean of 4 datasets × 3 pairs)", vmin, vmax)
    for d, table in tables.items():
        heatmap(table, os.path.join(args.out_dir, f"heatmap_{d}.png"), f"RQ2: recovery, {d} (mean of 3 pairs)", vmin, vmax)

    # ---------------- report.md ----------------
    used = cells[cells["used_in_analysis"]]
    L = [
        "# RQ2 交叉實驗結果",
        "",
        f"- 產生時間：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        f"- 輸入：`{args.armdir}`、`{args.aggdir}`（主網格 Judge）、`{os.path.join(args.out_dir, JUDGE_DIR)}`；"
        f"`{args.cells}` sha256 `{sha256(args.cells)[:16]}`",
        "- recovery = 與 aggregation_cells.csv 的 recovery_H2 相同的算法（both_answered 子集內 split-half，seed 0、200 次）；"
        "區間 = 跨區塊的 95% t 區間",
        "",
        "## (1) 判定標準",
        "",
        f"- `{criteria}`，sha256 `{criteria_sha}`",
        f"- 確認：{confirmed}",
        "",
        "## (2) 執行前檢查",
        "",
        f"### 4.1 流程核對（{CHECK_DATASET} × {CHECK_PAIR.label}，自己裁決自己，逐題比對主網格；門檻 {AGREEMENT_MIN:.0%}）",
        "",
        pd.DataFrame([{"模型": MODEL_LABELS[m], "題數": r["n"], "一致": r["n_agree"], "一致率": f"{r['agreement']:.3f}",
                       "對角線": "交叉流程重跑" if r["rerun_diagonal"] else "沿用主網格舊檔"}
                      for m, r in precheck["models"].items()]).to_markdown(index=False),
        "",
    ]
    if rerun:
        old = cells[(cells["source"] == SOURCE_MAIN) & ~cells["used_in_analysis"]]
        new = cells[(cells["generator"] == cells["judge"]) & cells["used_in_analysis"] & cells["generator"].isin(rerun)]
        compare = old[["generator", "dataset", "pair", "recovery_H2"]].merge(
            new[["generator", "dataset", "pair", "recovery_H2"]], on=["generator", "dataset", "pair"], suffixes=("_舊檔", "_重跑"))
        L += [f"以下模型一致率低於門檻，對角線改用交叉流程重跑的結果（mmlu × EN+S1 沿用流程核對的重跑）：{', '.join(sorted(rerun))}。"
              "舊檔算出的數字同列於此，也在 `cross_cells.csv`（`used_in_analysis` = False）。", "",
              compare.round(4).to_markdown(index=False), ""]
    pj = pd.DataFrame([{"Judge": MODEL_LABELS[j], "題數": r["n"], "找不到 choice": r["n_no_choice"], "超出範圍": r["n_out_of_range"],
                        "被擋下": r["n_refused"], "off-menu": r["n_off_menu"],
                        "API 輸入/次": round(r["api_in_per_call"]), "API 輸出/次": round(r["api_out_per_call"]),
                        "輸出 tokens/次（tokenizer）": round(r["tokens_out_per_call"], 1),
                        "美元/次": f"{r['cost_per_call_usd']:.5f}", "全量呼叫數": r["full_calls"],
                        "估算（美元）": f"{r['full_cost_usd']:.2f}"} for j, r in pilot["judges"].items()])
    L += [f"### 4.2 試跑（比例門檻 {PILOT_MAX_RATE:.0%}、成本門檻 {PILOT_MAX_COST:.0f} 美元）", "", pj.to_markdown(index=False), "",
          f"全量成本估算 {pilot['full_cost_usd']:.2f} 美元；結果：{'通過' if pilot['passed'] else '未通過'}。", ""]

    per_judge = used.groupby("judge").agg(呼叫數=("n_dis", "sum"), 無效選擇=("n_invalid_choice", "sum"), 被擋下=("n_refused", "sum"))
    weights = used.assign(w_off=used["off_menu"] * used["n_dis"], w_none=used["agg_no_answer"] * used["n_dis"]).groupby("judge")
    per_judge["off-menu 比例"] = (weights["w_off"].sum() / per_judge["呼叫數"]).round(4)
    per_judge["沒有答案比例"] = (weights["w_none"].sum() / per_judge["呼叫數"]).round(4)
    per_judge["輸出 tokens/次"] = (used.assign(t=used["tok_out_per_call"] * used["n_dis"]).groupby("judge")["t"].sum()
                                  / per_judge["呼叫數"]).round(1)
    per_judge.index = [MODEL_LABELS[j] for j in per_judge.index]
    L += ["## (3) 各 Judge 的解析失敗與 off-menu（全部 48 格 / Judge，含對角線）", "",
          "無效選擇 = 找不到 `{\"choice\"}` 或編號超出範圍；Judge 從候選中選，所以 off-menu 只來自無效選擇與被擋下。", "",
          per_judge.loc[[MODEL_LABELS[m] for m in MODELS]].to_markdown(), ""]

    L += ["## (4) 4×4 總表（recovery，對 4 個資料集 × 3 個配對平均）", "", markdownMatrix(overall), "",
          "熱圖：`heatmap_overall.png`；對角線 = 自己裁決自己。", ""]

    effects = pd.DataFrame([ciRow("裁判效果（強 Judge − 弱 Judge，區塊 = generator × 資料集）", judge_sum, judge_state),
                            ciRow("候選效果（強 generator − 弱 generator，區塊 = Judge × 資料集）", candidate_sum, candidate_state)])
    incl = pd.DataFrame([ciRow("裁判效果（含對角線）", summarize(judge_incl["delta"])),
                         ciRow("候選效果（含對角線）", summarize(candidate_incl["delta"]))])
    L += ["## (5) 裁判效果、候選效果、自我偏好與判定", "",
          f"主分析排除對角線。門檻（§6.1）：存在 = |平均 Δ| ≥ {THRESHOLD} 且 95% 區間不含 0；不存在 = 整個區間在 "
          f"[−{THRESHOLD}, +{THRESHOLD}] 內；其他 = 無法判定。正 = 強組較高。", "",
          effects.to_markdown(index=False), "",
          f"**判定 ({result})：{OUTCOMES[result]}**", ""]
    if result == "e":
        decided = [f"{name}{state}" for name, state in (("裁判效果", judge_state), ("候選效果", candidate_state)) if state != UNDETERMINED]
        L += [f"已判定：{'、'.join(decided) or '無'}；無法判定的效果不下「有」或「沒有」的結論，只報上表的區間。", ""]
    L += [f"自我偏好（§5.5、§6.3）：16 個對角線殘差 平均 {diag_sum['mean']:+.3f}，95% 區間 "
          f"[{diag_sum['ci_low']:+.3f}, {diag_sum['ci_high']:+.3f}]，{diag_sum['n_positive']}/16 為正 → "
          f"**{'有自我偏好' if self_pref else '不能說有自我偏好'}**（門檻：平均 ≥ +{THRESHOLD} 且區間不含 0）。"
          "殘差的預期值用含對角線的列平均與欄平均，對角線真實高出 δ 時殘差為 0.75δ。", "",
          "含對角線的版本（只報告，不參與判定）：", "", incl.to_markdown(index=False), ""]

    L += ["## (6) 分資料集、分配對", ""]
    for d in DATASETS:
        L += [f"### {d}（對 3 個配對平均）", "", markdownMatrix(tables[d]), ""]
    by_dataset = []
    for d in DATASETS:
        for name, fx, block in (("裁判效果", judge_fx, "generator"), ("候選效果", candidate_fx, "judge")):
            sub = fx[fx["dataset"] == d].set_index(block)["delta"]
            by_dataset.append({"資料集": d, "效果": name, "平均": f3(sub.mean()),
                               **{MODEL_LABELS[m]: f3(sub[m]) for m in MODELS}})
    L += ["### 裁判效果與候選效果，分資料集（每個資料集 4 個區塊；欄 = 區塊的 generator 或 Judge）", "",
          pd.DataFrame(by_dataset).to_markdown(index=False), "",
          "事先的預期是 TruthfulQA 的差距最大、MathQA 最小；這個預期不影響任何算法。", ""]
    for p in PAIRS:
        L += [f"### {p.label}（對 4 個資料集平均）", "", markdownMatrix(matrix(cells, pair=p.label)), ""]
    pair_rows = []
    for p in PAIRS:
        Rp = blockMeans(cells, pair=p.label)
        pair_rows += [{"配對": p.label, **ciRow("裁判效果", summarize(blockEffects(Rp, "judge")["delta"]))},
                      {"配對": p.label, **ciRow("候選效果", summarize(blockEffects(Rp, "candidate")["delta"]))}]
    L += ["### 裁判效果與候選效果，分配對（16 個區塊，排除對角線；只報告）", "", pd.DataFrame(pair_rows).to_markdown(index=False), ""]

    shown_tests = tests.assign(p_value=tests["p_value"].map(lambda p: f"{p:.4f}"))
    shown_tests["strong_judge"] = shown_tests["strong_judge"].map(MODEL_LABELS)
    shown_tests["weak_judge"] = shown_tests["weak_judge"].map(MODEL_LABELS)
    shown_impact = impact.assign(generator=impact["generator"].map(MODEL_LABELS)).round({"acc_strong_judges": 4, "acc_self": 4, "delta_pp": 2})
    L += ["## (7) 補充分析（只報告，不參與判定）", "",
          "### 7.1 逐題配對（McNemar，精確二項檢定）", "",
          "題目 = 兩個 Judge 都不是 generator 的那兩個 generator 的 n_R 題（剛好一條 path 答對），跨資料集與配對合併；"
          "b = 強 Judge 挑對、弱 Judge 挑錯；c = 相反；無效選擇算挑錯。", "",
          shown_tests.to_markdown(index=False), "",
          "### 7.2 實際影響：弱 generator 的候選改由強 Judge 裁決（acc_final，both_answered，百分點）", "",
          shown_impact.to_markdown(index=False), "",
          f"8 個區塊：平均 {impact_sum['mean']:+.2f}pp，95% 區間 [{impact_sum['ci_low']:+.2f}, {impact_sum['ci_high']:+.2f}]，"
          f"{impact_sum['n_positive']}/8 為正。", "",
          "### 7.4 同組但非自己的組合的殘差", "",
          f"四格（Judge → generator）：{'、'.join(f'{MODEL_LABELS[j]} → {MODEL_LABELS[g]}' for j, g in SAME_GROUP)}；"
          f"16 個殘差 平均 {same_sum['mean']:+.3f}，95% 區間 [{same_sum['ci_low']:+.3f}, {same_sum['ci_high']:+.3f}]，"
          f"{same_sum['n_positive']}/16 為正。對照：對角線殘差平均 {diag_sum['mean']:+.3f}。", "",
          "### 挑對比例（n_R 題上選中答對那條 path 的比例；亂猜 0.5）", "", markdownMatrix(matrix(cells, "pick_right")), "",
          "### 呈現順序", "",
          f"錨點（配對的第一條 path）在前的比例，在被裁決的題目上：{used['anchor_first'].min():.3f}–{used['anchor_first'].max():.3f}"
          "（同一組候選的四個 Judge 相同）。", "",
          "## 附錄：模型紀錄", "",
          "新呼叫的版本 ID 與時間取自每次回應；主網格對角線的 Judge 檔沒有逐次紀錄，只列檔案修改日期（同步時間可能晚於呼叫時間）。", "",
          models.to_markdown(index=False), ""]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"\n判定 ({result})：{OUTCOMES[result]}")
    print(f"  裁判效果 {judge_sum['mean']:+.3f} [{judge_sum['ci_low']:+.3f}, {judge_sum['ci_high']:+.3f}] {judge_state}；"
          f"候選效果 {candidate_sum['mean']:+.3f} [{candidate_sum['ci_low']:+.3f}, {candidate_sum['ci_high']:+.3f}] {candidate_state}")
    print(f"  自我偏好 {diag_sum['mean']:+.3f} [{diag_sum['ci_low']:+.3f}, {diag_sum['ci_high']:+.3f}] -> {'有' if self_pref else '不能說有'}")
    print(f"💾 {args.out_dir}/: cross_cells.csv, blocks.csv, residuals.csv, models.csv, {JUDGE_DIR}/items.csv.gz, "
          f"heatmap_*.png, report.md")


if __name__ == "__main__":
    main()
