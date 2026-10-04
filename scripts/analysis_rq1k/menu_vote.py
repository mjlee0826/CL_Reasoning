"""
menu_vote.py — RQ1-K：多條 path 的多數決 vs 同一份菜單內最強的單一 path（result/analysis/rq1k/rq1k_criteria.md）

全程離線（只讀 result/arms），判定標準確認前不執行。
    1. 每個區塊（模型 × 資料集）、每個菜單、每次切分（RQ1 的 makeSplits，seed 0、200 次）：
       A、S_in、S_all、EN、Excess_in、Excess_all；分解並逐切分核對恆等式（不符就停）
    2. 敏感度：全部題目、隨機平手、M14
    3. 三項判定（§4）與其他報告的量（§5）
    4. 輸出（--out-dir）：rq1k_items.csv.gz、rq1k_blocks.csv、rq1k_compare.csv、report.md、
       excess_m12_blocks.png、excess_menus.png

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq1k/menu_vote.py
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Analysis.preregistration import sha256, confirmationLine
from Analysis.blockStats import summarize, fourState, FORWARD
from Analysis.splitHalf import makeSplits
from Analysis.menuVote import (MODELS, DATASETS, MENUS, MAIN_MENU, SENSITIVITY_MENUS, COMPARE_MENUS, THRESHOLD, SEED, REPS,
                               TOLERANCE, TIE_RANDOM, SUBSET_ALL_ITEMS, loadPathBlock, evaluateMenu, compareMenus, regrets)

MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
# 類別色（依固定順序指派給模型 / 序列）與墨色；圖上用英文（matplotlib 預設字型沒有中文字）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#d9d8d2"
SENSITIVITY_COLUMNS = ["excess_in", "excess_all"]


def parseArgs():
    parser = ArgumentParser(description="RQ1-K: majority vote over K-path menus vs the strongest single path (rq1k_criteria.md)")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--out-dir", default="result/analysis/rq1k")
    parser.add_argument("--rq1-blocks", default="result/analysis/rq1/blocks.csv", help="RQ1 blocks for the §5.7 comparison")
    return parser.parse_args()


def pp(value, sign: bool = True) -> str:
    return "—" if pd.isna(value) else f"{value:{'+' if sign else ''}.2f}"


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


def plotBlocks(blocks: pd.DataFrame, path: str):
    """M12 下 16 個區塊的 Excess_in：依資料集分組，模型 = 顏色 + 形狀；標出 0 與 ±門檻。"""
    main = blocks[blocks["menu"] == MAIN_MENU]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    offsets = np.linspace(-0.24, 0.24, len(MODELS))
    for i, model in enumerate(MODELS):
        sub = main[main["model"] == model].set_index("dataset").reindex(DATASETS)
        ax.scatter(np.arange(len(DATASETS)) + offsets[i], sub["excess_in"], s=64, marker=MARKERS[i], color=SERIES[i],
                   edgecolor="white", linewidth=1.5, zorder=3, label=MODEL_LABELS[model])
    ax.axhline(0, color=INK, linewidth=1)
    for y in (THRESHOLD, -THRESHOLD):
        ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(DATASETS)), DATASETS)
    ax.set_ylabel("Excess_in (pp): majority vote − strongest single path")
    ax.set_title(f"RQ1-K, {MAIN_MENU}: Excess_in per block (dashed = ±{THRESHOLD} pp)", fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plotMenus(summaries: dict, path: str):
    """各菜單的 Excess_in 與 Excess_all：跨區塊平均與 95% t 區間。"""
    menus = list(MENUS)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(menus))
    for i, (column, label) in enumerate((("excess_in", "Excess_in (vs strongest in menu)"),
                                         ("excess_all", "Excess_all (vs strongest of M12)"))):
        means = [summaries[(m, column)]["mean"] for m in menus]
        low = [summaries[(m, column)]["mean"] - summaries[(m, column)]["ci_low"] for m in menus]
        high = [summaries[(m, column)]["ci_high"] - summaries[(m, column)]["mean"] for m in menus]
        ax.errorbar(means, y + (0.15 if i else -0.15), xerr=[low, high], fmt=MARKERS[i], color=SERIES[i], markersize=7,
                    capsize=0, linewidth=2, markeredgecolor="white", label=label, zorder=3)
    ax.axvline(0, color=INK, linewidth=1)
    for x in (THRESHOLD, -THRESHOLD):
        ax.axvline(x, color=MUTED, linewidth=1, linestyle="--")
    ax.set_yticks(y, [f"{m} (K={len(MENUS[m])})" + (" *" if m in SENSITIVITY_MENUS else "") for m in menus])
    ax.invert_yaxis()
    ax.set_xlabel("pp, mean over 16 blocks with 95% t-CI")
    ax.set_title("RQ1-K: majority vote − strongest single path, by menu (* = sensitivity only)", fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parseArgs()
    criteria = os.path.join(args.out_dir, "rq1k_criteria.md")
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty)")

    rows, items, compare_rows, splits_by_dataset = [], [], [], {}
    for model in MODELS:
        for dataset in DATASETS:
            block = loadPathBlock(args.armdir, model, dataset)
            n = len(block.item_ids)
            if dataset not in splits_by_dataset:
                splits_by_dataset[dataset] = (block.item_ids, makeSplits(n, REPS, SEED))
            ids, splits = splits_by_dataset[dataset]
            if not np.array_equal(ids, block.item_ids):
                raise SystemExit(f"❌ {model} | {dataset}: item ids differ from the other models (the split would not be shared)")
            finals = {}
            for menu in MENUS:
                main_result = evaluateMenu(block, menu, splits)
                random_ties = evaluateMenu(block, menu, splits, ties=TIE_RANDOM, decompose=False)["row"]
                all_items = evaluateMenu(block, menu, splits, subset=SUBSET_ALL_ITEMS, decompose=False)["row"]
                row = main_result["row"]
                row.update({f"{c}_random_ties": random_ties[c] for c in SENSITIVITY_COLUMNS})
                row.update({f"{c}_all_items": all_items[c] for c in SENSITIVITY_COLUMNS})
                row["tie_rate_all_items"] = all_items["tie_rate"]
                rows.append(row)
                finals[menu] = main_result["agg_correct"]
                for k, item_id in enumerate(block.item_ids):
                    items.append({"model": model, "dataset": dataset, "menu": menu, "item_id": int(item_id),
                                  "gold": block.gold[k], "final_answer": main_result["finals"][k],
                                  "tie": bool(main_result["tied"][k]), "correct": bool(main_result["agg_correct"][k]),
                                  "in_subset_in": bool(main_result["sub_in"][k]), "in_subset_all": bool(main_result["sub_all"][k])})
            compare_rows.append({"model": model, "dataset": dataset, "m3s_minus_m3l": compareMenus(block, splits, finals)})
            print(f"📊 {model} | {dataset}: {len(MENUS)} menus")

    blocks = pd.DataFrame(rows)
    bad = blocks[blocks["identity_max_error"] > TOLERANCE]
    if not bad.empty:
        raise SystemExit("❌ §5.4 identity check failed, stopping:\n" + bad[["model", "dataset", "menu", "identity_max_error"]].to_string())
    print(f"✅ §5.4 identity: {len(blocks) * REPS} splits, max error {blocks['identity_max_error'].max():.2e}")
    compare = pd.DataFrame(compare_rows)

    # ---------------- 判定 ----------------
    main = blocks[blocks["menu"] == MAIN_MENU]
    j1 = summarize(main["excess_in"])
    j1_state = fourState(j1, THRESHOLD)
    j2 = regrets(main["excess_in"].to_numpy())
    j3 = summarize(compare["m3s_minus_m3l"])
    j3_state = fourState(j3, THRESHOLD)
    summaries = {(menu, column): summarize(blocks[blocks["menu"] == menu][column])
                 for menu in MENUS for column in SENSITIVITY_COLUMNS}

    # ---------------- 輸出 ----------------
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(items).to_csv(os.path.join(args.out_dir, "rq1k_items.csv.gz"), index=False,
                               compression={"method": "gzip", "mtime": 0})
    blocks.to_csv(os.path.join(args.out_dir, "rq1k_blocks.csv"), index=False)
    compare.to_csv(os.path.join(args.out_dir, "rq1k_compare.csv"), index=False)
    plotBlocks(blocks, os.path.join(args.out_dir, "excess_m12_blocks.png"))
    plotMenus(summaries, os.path.join(args.out_dir, "excess_menus.png"))

    # ---------------- report.md ----------------
    def blockTable(column: str, scale: float = 1.0, digits: int = 2) -> str:
        """區塊 × 菜單。"""
        table = blocks.pivot_table(index=["model", "dataset"], columns="menu", values=column).reindex(columns=list(MENUS))
        table = (table * scale).round(digits)
        table.index = [blockLabel(m, d) for m, d in table.index]
        return table.to_markdown()

    L = [
        "# RQ1-K 結果：多數決 vs 同一份菜單內最強的單一 path",
        "",
        f"- 產生時間：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        f"- 輸入：`{args.armdir}`（離線，不呼叫 API）；切分 = `makeSplits(n, {REPS}, seed={SEED})`（與 RQ1 相同）",
        "- 單位：正確率為 %，Excess 為百分點（pp）；區間 = 跨 16 個區塊的 95% t 區間",
        "",
        "## (1) 判定標準",
        "",
        f"- `{criteria}`，sha256 `{sha256(criteria)}`",
        f"- 確認：{confirmed}",
        "",
        "## (2) 題目保留比例、平手比例、恆等式核對",
        "",
        "子集_in 的保留比例（%；菜單內所有 path 都有答案）：",
        "",
        blockTable("keep_in", 100, 1),
        "",
        "子集_all（M12 的 12 條都有答案）的保留比例：" + "、".join(
            f"{blockLabel(r.model, r.dataset)} {100 * r.keep_all:.1f}%" for r in main.itertuples() if r.keep_all < 0.99)
        + "；其餘區塊 ≥ 99%。",
        "",
        "平手比例（%，子集_in 內；跨區塊的平均 / 最小 / 最大）：",
        "",
        blocks.groupby("menu")["tie_rate"].agg(["mean", "min", "max"]).reindex(list(MENUS)).mul(100).round(2).to_markdown(),
        "",
        f"恆等式（§5.4）：{len(blocks) * REPS} 次切分，Excess_in 與 headroom ×（recovery − recovery_blind）的最大誤差 "
        f"{blocks['identity_max_error'].max():.1e}（門檻 {TOLERANCE:.0e}），全部通過。recovery 無定義的切分："
        f"{int(blocks['undefined_splits'].sum())} 次（"
        + ("、".join(f"{r.menu} {blockLabel(r.model, r.dataset)} {r.undefined_splits}" for r in blocks.itertuples() if r.undefined_splits)
           or "無") + "）。",
        "",
        "## (3) 三項判定",
        "",
        pd.DataFrame([summaryRow(f"判定一：{MAIN_MENU} 的 Excess_in（正向 = 聚合較好）", main["excess_in"]),
                      summaryRow(f"判定三：{COMPARE_MENUS[0]} − {COMPARE_MENUS[1]}（正向 = 同語言重抽較好）",
                                 compare["m3s_minus_m3l"])]).to_markdown(index=False),
        "",
        f"- **判定一：{j1_state}**。{MAIN_MENU} 的多數決比同一份菜單內 H1 上最強的單一 path 平均 {pp(j1['mean'])}pp。",
        f"- **判定二（決策空間）**：永遠聚合的 regret {j2['regret_aggregate']:.2f}pp，永遠用單一最強的 regret "
        f"{j2['regret_single']:.2f}pp，空間 = {j2['space']:.2f}pp → **"
        + ("空間 ≥ 0.5pp：值得再做一次 RQ1（另外寫判定標準，這次不做）" if j2["space"] >= THRESHOLD
           else "空間 < 0.5pp：RQ1 維持負面結果") + "**。",
        f"- **判定三：{j3_state}**。同樣三條 path，{COMPARE_MENUS[0]}（同語言重抽）比 {COMPARE_MENUS[1]}（換語言）平均 {pp(j3['mean'])}pp。",
        "- 範圍：以上判定只針對多數決。「聚合是否贏過單一最強 path」的一般性結論，要等同一份菜單上的 Judge 版本完成後才能下；"
        + ("判定一為正向成立，Judge 版本列為**可選**。" if j1_state == FORWARD else f"判定一為{j1_state}，Judge 版本列為**必要**的後續實驗。"),
        "",
        "## (4) 總表（跨 16 個區塊的平均；* = 只當敏感度）",
        "",
    ]
    total = []
    for menu in MENUS:
        sub = blocks[blocks["menu"] == menu]
        s_in, s_all = summaries[(menu, "excess_in")], summaries[(menu, "excess_all")]
        total.append({"菜單": menu + (" *" if menu in SENSITIVITY_MENUS else ""), "K": len(MENUS[menu]),
                      "聚合（子集_in）": f"{100 * sub['A_in'].mean():.2f}", "S_in": f"{100 * sub['S_in'].mean():.2f}",
                      "聚合（子集_all）": f"{100 * sub['A_all'].mean():.2f}", "S_all": f"{100 * sub['S_all'].mean():.2f}",
                      "L:en（子集_in）": f"{100 * sub['EN_in'].mean():.2f}",
                      "Excess_in [95%]": ciText(s_in), "Excess_in 為正": f"{s_in['n_positive']}/16",
                      "Excess_all [95%]": ciText(s_all), "Excess_all 為正": f"{s_all['n_positive']}/16"})
    L += [pd.DataFrame(total).to_markdown(index=False), "", "圖：`excess_menus.png`（各菜單）、`excess_m12_blocks.png`（M12 的 16 個區塊）。", ""]

    def by(key: str, column: str) -> pd.DataFrame:
        """菜單 × 資料集（或模型）的平均。"""
        return blocks.pivot_table(index="menu", columns=key, values=column).reindex(list(MENUS)).round(2)

    L += ["## (5) 分資料集、分模型、被選中的 path", "",
          "### 每個菜單的逐區塊 Excess_in（pp）", "", blockTable("excess_in"), "",
          "### 每個菜單的逐區塊 Excess_all（pp）", "", blockTable("excess_all"), "",
          "### 分資料集的平均 Excess_in（pp）", "", by("dataset", "excess_in").to_markdown(), "",
          "### 分資料集的平均 Excess_all（pp）", "", by("dataset", "excess_all").to_markdown(), "",
          "### 分模型的平均 Excess_in（pp）", "", by("model", "excess_in").rename(columns=MODEL_LABELS).to_markdown(), "",
          "### 分模型的平均 Excess_all（pp）", "", by("model", "excess_all").rename(columns=MODEL_LABELS).to_markdown(), "",
          "### H1 上被選為 S_in 的 path（前兩名與 200 次中被選中的比例；最後一欄 = S_all，M12 的 12 條中）", ""]
    chosen = blocks.pivot_table(index=["model", "dataset"], columns="menu", values="S_in_top", aggfunc="first").reindex(columns=list(MENUS))
    chosen["S_all"] = main.set_index(["model", "dataset"])["S_all_top"]
    chosen.index = [blockLabel(m, d) for m, d in chosen.index]
    L += [chosen.to_markdown(), ""]

    decomposition = blocks.groupby("menu")[["d", "c", "m", "headroom_pp", "recovery", "recovery_blind", "excess_in",
                                            "undefined_splits"]].mean().reindex(list(MENUS))
    cost = blocks.groupby("menu")[["calls_menu", "calls_single", "tokens_menu", "tokens_S_in"]].mean().reindex(list(MENUS))
    cost["tokens 倍數"] = cost["tokens_menu"] / cost["tokens_S_in"]
    L += ["## (6) 分解與成本（跨區塊平均；分解在 H2 ∩ 子集_in 上、用 S_in，對切分平均）", "",
          "d = 菜單內答案不完全一致的題目比例；c = 其中至少一條答對的比例；m = 其中各 path 的平均正確率；"
          "headroom = d·(c − m)；recovery / recovery_blind = 聚合 / S_in 在這些題目上的 (正確率 − m) ÷ (c − m)。"
          "各量先對切分平均，所以平均後的乘積不必等於平均 Excess；恆等式在每次切分各自成立（見 (2)）。", "",
          decomposition.round(3).to_markdown(), "",
          "成本：多數決沒有額外的聚合呼叫；tokens = 每題輸出 tokens（菜單內各 path 加總，對 S_in 那一條）。", "",
          cost.round(1).to_markdown(), ""]

    sens = []
    for menu in MENUS:
        sub = blocks[blocks["menu"] == menu]
        sens.append({"菜單": menu + (" *" if menu in SENSITIVITY_MENUS else ""),
                     "主分析 Excess_in": ciText(summaries[(menu, "excess_in")]),
                     "全部題目 Excess_in": ciText(summarize(sub["excess_in_all_items"])),
                     "全部題目 Excess_all": ciText(summarize(sub["excess_all_all_items"])),
                     "隨機平手 Excess_in": ciText(summarize(sub["excess_in_random_ties"])),
                     "隨機平手 Excess_all": ciText(summarize(sub["excess_all_random_ties"]))})
    spaces = []
    for menu in MENUS:
        r = regrets(blocks[blocks["menu"] == menu]["excess_in"].to_numpy())
        spaces.append({"菜單": menu + (" *" if menu in SENSITIVITY_MENUS else ""),
                       "永遠聚合 regret": f"{r['regret_aggregate']:.2f}", "永遠單一最強 regret": f"{r['regret_single']:.2f}",
                       "空間": f"{r['space']:.2f}",
                       "註": ("判定二" if menu == MAIN_MENU else "探索性（≥ 0.5pp，不算通過 go / no-go）" if r["space"] >= THRESHOLD else "")})
    L += ["## (7) 敏感度與其他菜單的空間", "",
          "敏感度：全部題目（沒答案的 path 不投票、單一 path 沒答案算錯、在全部 H1 題目上選 path）；"
          "隨機平手（每題 `default_rng([0, item_id])`）；M14 見總表與下表的 * 列。", "",
          pd.DataFrame(sens).to_markdown(index=False), "",
          "各菜單的決策空間（算法同判定二，用各自的 Excess_in）：", "",
          pd.DataFrame(spaces).to_markdown(index=False), ""]

    rq1 = pd.read_csv(args.rq1_blocks)
    rq1 = rq1[(rq1["baseline"] == "global") & (rq1["setting"] == "probe_random") & (rq1["k"] == 500) & (rq1["aggregator"] == "judge")]
    m14 = blocks[blocks["menu"] == "M14"].set_index(["model", "dataset"])["S_in"]
    match = rq1.set_index(["model", "dataset"])["acc_S"].reindex(m14.index)
    comparison = pd.DataFrame({"M14 的 S_in（%）": (100 * m14).round(2), "RQ1 全域基準 acc_S（%）": (100 * match).round(2),
                               "差（pp）": (100 * (m14 - match)).round(2)})
    comparison.index = [blockLabel(m, d) for m, d in comparison.index]
    gaps = (100 * (m14 - match)).dropna()
    missing = [blockLabel(m, d) for m, d in match.index[match.isna()]]
    L += ["### 與 RQ1 全域基準比對（§5.7）", "",
          f"RQ1：`{args.rq1_blocks}` 的全域基準 acc_S（Judge、隨機 probe、k = 500）。RQ1 只用 H1 的 500 題選 path、"
          "在各配對的 both_answered 子集上評估；這裡用整個 H1 ∩ 子集_in。", "",
          comparison.to_markdown(), "",
          (f"RQ1 沒有這些區塊的 k = 500 數字（H1 題數少於 500，RQ1 依其規則跳過）：{'、'.join(missing)}。" if missing else "")
          + f"可比的 {len(gaps)} 個區塊：平均差 {gaps.mean():+.2f}pp，平均絕對差 {gaps.abs().mean():.2f}pp"
          + ("（超過 0.5pp，需逐區塊說明）。" if abs(gaps.mean()) > THRESHOLD else "（在 0.5pp 內）。"), ""]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"\n判定一（{MAIN_MENU} Excess_in）：{j1_state} {ciText(j1)}")
    print(f"判定二：regret 聚合 {j2['regret_aggregate']:.2f} / 單一 {j2['regret_single']:.2f} → 空間 {j2['space']:.2f}pp")
    print(f"判定三（{COMPARE_MENUS[0]} − {COMPARE_MENUS[1]}）：{j3_state} {ciText(j3)}")
    print(f"💾 {args.out_dir}/: rq1k_items.csv.gz, rq1k_blocks.csv, rq1k_compare.csv, report.md, excess_*.png")


if __name__ == "__main__":
    main()
