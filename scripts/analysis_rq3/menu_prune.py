"""
menu_prune.py — RQ3：拿掉 path 之後的多數決（result/analysis/rq3/rq3_criteria.md）

全程離線：不呼叫 API，不重跑任何東西，不修改現有檔案。判定標準確認前不執行。
    1. 16 個區塊 × 14 份菜單（M12、12 份 M12−p、M10），都在 M12 的子集上、makeSplits(n, 200, 0) 的每次切分
       （Analysis/menuPrune.py；多數決、S_in、切分都沿用 RQ1-K）
    2. §6 的兩項檢查（任何一項不符就停，不寫輸出）：M12 的 A_V、S_in、Excess_in = rq1k_blocks.csv（1e-9）；
       子集題數 = RQ1-K 的 M12 子集。§7.7 的恆等式每次切分核對（1e-9），不符也停
    3. 判定一（Prune1）、判定二（ΔExcess）、TruthfulQA 的次要分析與 §7 只報告的量
    4. 輸出（--out-dir）：rq3_blocks.csv、report.md、prune_each_path.png、excess_m12_m10.png

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq3/menu_prune.py
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
from Analysis.blockStats import summarize, fourState, FORWARD, REVERSE, EQUIVALENT, UNDETERMINED
from Analysis.experimentPlan import PATHS
from Analysis.menuJudgeStats import blocksWithSplits
from Analysis.menuVote import MODELS, DATASETS
from Analysis.menuPrune import (M12, DROP, MENU_CODES, MENU_NAMES, REMOVAL_ORDER, CATEGORIES, THRESHOLD, TOLERANCE,
                                DECOMPOSED, DECOMPOSITION, evaluateBlock)

OUT_DIR = "result/analysis/rq3"
CRITERIA_FILE = "rq3_criteria.md"
RQ1K_BLOCKS = "result/analysis/rq1k/rq1k_blocks.csv"
MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
READINGS_1 = {
    FORWARD: "修剪菜單有好處：主張一要補一句「挑對成員可以再多 {x:.2f}pp」。",
    REVERSE: "用標註選一條拿掉，反而讓多數決變差。",
    EQUIVALENT: "拿掉一條對多數決幾乎沒有影響（±0.5pp 內）。",
    UNDETERMINED: "只報區間。",
}
READINGS_2 = {
    FORWARD: "支持「增益大小取決於菜單裡有沒有強的 path」。",
    REVERSE: "這個說法不成立，論文裡對應的句子要改寫。",
    EQUIVALENT: "這個說法不成立，論文裡對應的句子要改寫。",
    UNDETERMINED: "只報區間，這個說法只能寫成觀察。",
}
# 類別色（dataviz 參考調色盤，依固定順序指派）與墨色；圖上用英文（matplotlib 預設字型沒有中文字）
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6a64", "#d9d8d2", "#fcfcfb"
FIG_A, FIG_B = "prune_each_path.png", "excess_m12_m10.png"


def parseArgs():
    parser = ArgumentParser(description="RQ3: majority vote after removing paths (rq3_criteria.md)")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--rq1k-blocks", default=RQ1K_BLOCKS)
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


def styleAxes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)


def plotEachPath(summary: pd.DataFrame, path: str, sha: str):
    """(a) §7.3：拿掉每一條（不切分）的 A_V(M12−p) − A_V(M12)，16 個區塊的平均與 95% t 區間。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(summary))[::-1]
    ax.hlines(y, summary.ci_low, summary.ci_high, color=BLUE, linewidth=2, zorder=2)
    ax.scatter(summary["mean"], y, s=52, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"drop {c} ({PATHS[c]})" for c in summary.path])
    styleAxes(ax)
    ax.set_xlabel("A_V(M12 − p) − A_V(M12), pp (all subset items; mean of 16 blocks, 95% t interval)", color=MUTED, fontsize=8.5)
    ax.set_title("RQ3: effect of removing each path from the 12-path majority vote\n(12 comparisons, descriptive only)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, metadata={"Description": f"rq3_criteria.md sha256 {sha}"})
    plt.close(fig)


def plotExcess(m12: pd.DataFrame, m10: pd.DataFrame, path: str, sha: str):
    """(b) 16 個區塊的 Excess_in：M12 對 M10。"""
    fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    labels = [f"{MODEL_LABELS[r.model]} · {r.dataset}" for r in m12.itertuples()]
    y = np.arange(len(m12))[::-1]
    a, b = m12.excess_in_pp.to_numpy(), m10.excess_in_pp.to_numpy()
    ax.hlines(y, np.minimum(a, b), np.maximum(a, b), color=GRID, linewidth=2, zorder=1)
    # 兩個點上下錯開一點，數值相近時才不會互相蓋住
    ax.scatter(a, y + 0.12, s=52, color=BLUE, marker="o", edgecolors=SURFACE, linewidths=1.5, zorder=3, label="M12 (12 paths)")
    ax.scatter(b, y - 0.12, s=52, color=ORANGE, marker="s", edgecolors=SURFACE, linewidths=1.5, zorder=3, label="M10 (no personas)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    styleAxes(ax)
    ax.set_xlabel("Excess_in = A_V − S_in (pp, mean of 200 splits)", color=MUTED, fontsize=9)
    ax.set_title("RQ3: majority vote vs the strongest single path in the same menu", color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, metadata={"Description": f"rq3_criteria.md sha256 {sha}"})
    plt.close(fig)


def main():
    args = parseArgs()
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty)")
    criteria_sha = sha256(criteria)

    rows, blocks = [], []
    for block, splits in blocksWithSplits(args.armdir, args.aggdir):
        result = evaluateBlock(block, splits)
        base = {"model": block.model, "dataset": block.dataset, "n": len(block.item_ids)}
        blocks.append({**base, **result["block"]})
        for name in MENU_NAMES:
            row = {**base, "menu": name, "K": len(MENU_CODES[name]), "n_subset": result["block"]["n_subset"], **result["menus"][name]}
            if name == "M12":
                row.update(result["block"])
            rows.append(row)
        print(f"  {block.model:20s} {block.dataset:14s} Prune1 {result['block']['prune1_pp']:+.2f}  "
              f"ΔExcess {result['block']['dExcess_pp']:+.2f}")
    table, blocks = pd.DataFrame(rows), pd.DataFrame(blocks)
    m12 = table[table.menu == "M12"].reset_index(drop=True)
    m10 = table[table.menu == "M10"].reset_index(drop=True)

    # §6 的檢查與 §7.7 的恆等式
    rq1k = pd.read_csv(args.rq1k_blocks)
    ref = rq1k[rq1k.menu == "M12"].set_index(["model", "dataset"])
    diffs = {"A_V": [], "S_in": [], "Excess_in": []}
    count_problems = []
    for r in m12.itertuples():
        theirs = ref.loc[(r.model, r.dataset)]
        diffs["A_V"].append(abs(r.A_V - theirs["A_in"]))
        diffs["S_in"].append(abs(r.S_in - theirs["S_in"]))
        diffs["Excess_in"].append(abs(r.excess_in_pp - theirs["excess_in"]))
        expected = theirs["n"] * theirs["keep_in"]
        if abs(expected - round(expected)) > 1e-6 or int(round(expected)) != r.n_subset:
            count_problems.append(f"{r.model} | {r.dataset}: {r.n_subset} items, RQ1-K n × keep_in = {expected}")
    max_diff = {key: max(values) for key, values in diffs.items()}
    check1 = all(value <= TOLERANCE for value in max_diff.values()) and len(m12) == len(MODELS) * len(DATASETS)
    check2 = not count_problems
    identity = float(blocks.identity_max_error.max())
    print(f"§6.1 M12 重現：最大誤差 {max_diff} -> {'通過' if check1 else '不符'}")
    print(f"§6.2 子集題數：{'通過' if check2 else '不符'}")
    print(f"§7.7 恆等式：最大誤差 {identity:.1e} -> {'通過' if identity <= TOLERANCE else '不符'}")
    if not (check1 and check2 and identity <= TOLERANCE):
        for problem in count_problems:
            print(f"  ❌ {problem}")
        raise SystemExit("❌ 檢查沒有全部通過：停下來回報，不寫輸出")

    os.makedirs(args.out_dir, exist_ok=True)
    table.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq3_blocks.csv"), index=False)

    j1, j2 = summarize(blocks.prune1_pp), summarize(blocks.dExcess_pp)
    s1, s2 = fourState(j1, THRESHOLD), fourState(j2, THRESHOLD)
    each = []
    for p in M12:
        s = summarize(table[table.menu == DROP[p]].unsplit_vs_M12_pp.to_numpy())
        each.append({"path": p, **s})
    each = pd.DataFrame(each)
    plotEachPath(each, os.path.join(args.out_dir, FIG_A), criteria_sha)
    plotExcess(m12, m10, os.path.join(args.out_dir, FIG_B), criteria_sha)

    label = lambda r: f"{MODEL_LABELS[r.model]} · {r.dataset}"
    out = ["# RQ3：拿掉 path 之後的多數決", ""]
    out += ["## (1) 判定標準", "",
            f"- `{criteria}`，sha256 `{criteria_sha}`",
            f"- 確認：{confirmed}",
            f"- 產生時間：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}；程式：`scripts/analysis_rq3/menu_prune.py`"
            "（逐切分計算在 `Analysis/menuPrune.py`）。不呼叫 API，不重跑任何東西，不修改現有檔案。", ""]
    out += ["## (2) 讀了哪些檔案", "",
            f"- `{args.armdir}/{{模型}}/{{資料集}}/`：經 `Analysis.menuVote.loadPathBlock` 載入（與 RQ1-K 相同；它讀 14 個 path 檔，"
            "這裡只用 M12 的 12 條）。欄位 `parsed_answer`、`parse_ok`、`gold`。",
            f"- `{args.rq1k_blocks}`：§6 的核對（M12 列的 `A_in`、`S_in`、`excess_in`、`n`、`keep_in`）。",
            "- `result/analysis/rq1k/rq1k_criteria.md` §3.1：平手的優先順序（只引用）。",
            "- 沿用的程式：多數決 `Analysis.menuVote.vote`、最強 path `Analysis.probe.strongest`、"
            "切分 `Analysis.menuJudgeStats.blocksWithSplits`（`makeSplits(n, 200, 0)`，同一資料集的模型共用）、"
            "統計 `Analysis.blockStats.summarize`。",
            "- 「M12 只去掉 P:skeptic」= M12−P2，「M12 只去掉 P:expert」= M12−P1，都在 12 份 M12−p 之中。",
            f"- 平手時拿掉的先後（§3.1）：{'、'.join(REMOVAL_ORDER)}。", ""]

    out += ["## (3) 第 6 節的檢查", "",
            f"1. M12 的 A_V、S_in、Excess_in 與 `rq1k_blocks.csv`：最大誤差 A_V {max_diff['A_V']:.1e}、S_in {max_diff['S_in']:.1e}、"
            f"Excess_in {max_diff['Excess_in']:.1e}（門檻 1e-09）→ 通過。",
            f"2. 16 個區塊的子集題數 = RQ1-K 的 M12 子集（`n × keep_in`）→ 通過（共 {int(m12.n_subset.sum())} 題）。",
            f"- §7.7 的恆等式（M12 與 M10，16 個區塊 × 200 次切分）：最大誤差 {identity:.1e} → 通過。", ""]

    out += ["## (4) 判定一與判定二", "",
            md(pd.DataFrame([ciRow("判定一：Prune1 = A_V(M12−p*) − A_V(M12)", j1, True),
                             ciRow("判定二：ΔExcess = Excess_in(M10) − Excess_in(M12)", j2, True)]), text=True), "",
            f"單位：百分點；門檻 {THRESHOLD}pp。",
            f"- 判定一 **{s1}** → {READINGS_1[s1].format(x=j1['mean'])}",
            f"- 判定二 **{s2}** → {READINGS_2[s2]}", ""]

    tq = m12[m12.dataset == "truthfulqa"].model
    tq_rows = []
    for model in tq:
        get = lambda menu: table[(table.model == model) & (table.dataset == "truthfulqa") & (table.menu == menu)].iloc[0]
        e12, e10, e2 = get("M12").excess_in_pp, get("M10").excess_in_pp, get(DROP["P2"]).excess_in_pp
        tq_rows.append({"模型": MODEL_LABELS[model], "Excess_in(M12)": e12, "Excess_in(M10)": e10, "ΔExcess（M10）": e10 - e12,
                        "Excess_in(M12−P2)": e2, "M12−P2 − M12": e2 - e12})
    tq_df = pd.DataFrame(tq_rows)
    tq_df.loc[len(tq_df)] = {"模型": "平均", **tq_df.drop(columns="模型").mean().to_dict()}
    all_pos = bool((tq_df["ΔExcess（M10）"].iloc[:-1] > 0).all())
    mean10 = float(tq_df["Excess_in(M10)"].iloc[-1])
    if all_pos and mean10 >= -0.5:
        tq_reading = "4 個區塊的 ΔExcess 全為正，且 Excess_in(M10) 的平均 ≥ −0.5pp → 寫「拿掉 persona 後，TruthfulQA 上的差距消失」，現有的解釋得到支持。"
    elif mean10 <= -1.5:
        tq_reading = "Excess_in(M10) 的平均仍 ≤ −1.5pp → 寫「拿掉 persona 後差距仍在」，現有的解釋不成立，原因標為未知。"
    else:
        tq_reading = "其他情況 → 只列數字，不下結論。"
    out += ["## (5) TruthfulQA 的次要分析（4 個區塊，不套四種狀態）", "", md(tq_df), "",
            f"- ΔExcess（M10）全為正：{'是' if all_pos else '否'}；Excess_in(M10) 的平均 {mean10:+.2f}pp。",
            f"- 讀法：{tq_reading}", ""]

    out += ["## (6) 其餘只報告的量（不參與判定）", "", "### 7.1 判定一、二的逐區塊數字（pp）", ""]
    per = blocks[["model", "dataset", "prune1_pp", "dExcess_pp"]].copy()
    per["model"] = per.model.map(MODEL_LABELS)
    out += [md(per.rename(columns={"prune1_pp": "Prune1", "dExcess_pp": "ΔExcess"})), ""]
    for key, title in (("dataset", "分資料集"), ("model", "分模型")):
        g = blocks.groupby(key, sort=False)[["prune1_pp", "dExcess_pp"]].mean().reset_index()
        if key == "model":
            g["model"] = g.model.map(MODEL_LABELS)
        out += [f"{title}的平均：", "", md(g.rename(columns={"prune1_pp": "Prune1", "dExcess_pp": "ΔExcess"})), ""]

    shares = blocks[["model", "dataset"] + [f"p_star_{p}" for p in M12]].copy()
    shares["model"] = shares.model.map(MODEL_LABELS)
    shares.columns = ["模型", "資料集"] + M12
    shares[M12] = 100 * shares[M12]
    shares.loc[len(shares)] = ["平均", ""] + list(shares[M12].mean())
    cats = blocks[["model", "dataset"] + [f"p_star_{c}" for c in CATEGORIES]].copy()
    cats["model"] = cats.model.map(MODEL_LABELS)
    cats.columns = ["模型", "資料集"] + list(CATEGORIES)
    cats[list(CATEGORIES)] = 100 * cats[list(CATEGORIES)]
    cats.loc[len(cats)] = ["平均", ""] + list(cats[list(CATEGORIES)].mean())
    out += ["### 7.2 p* 是哪一條（200 次切分中被選中的比例，%）", "", md(shares, ".0f"), "",
            "依類別合計（語言 = ZH、JA、RU、ES；採樣 = S1、S2；persona = P1、P2；改寫 = W1、W2；簡短 CoT = R）：", "",
            md(cats, ".0f"), ""]

    each_show = pd.DataFrame([{"拿掉": f"{r.path}（{PATHS[r.path]}）", "平均": pp(r.mean), "SE": f"{r.se:.2f}",
                               "95% 區間": f"[{pp(r.ci_low)}, {pp(r.ci_high)}]", "為正的區塊": f"{r.n_positive}/{r.n_blocks}"}
                              for r in each.itertuples()])
    out += ["### 7.3 不經挑選的逐條結果", "",
            "A_V(M12−p) − A_V(M12)，用子集內全部題目（不切分），百分點。**這是 12 個比較，只當描述。**", "",
            md(each_show, text=True), "", f"![逐條拿掉的效果]({FIG_A})", ""]

    p13 = summarize(blocks.prune13_pp)
    none = blocks[["model", "dataset", "none_share"]].copy()
    none["model"] = none.model.map(MODEL_LABELS)
    none["none_share"] = 100 * none.none_share
    out += ["### 7.4 可以選擇不拿的版本（13 選一，平手優先不拿）", "",
            md(pd.DataFrame([ciRow("A_V(選出的菜單) − A_V(M12)", p13)]), text=True), "",
            f"選到「不拿」的切分比例：16 個區塊平均 {100 * blocks.none_share.mean():.1f}%。逐區塊（%）：", "",
            md(none.rename(columns={"model": "模型", "dataset": "資料集", "none_share": "選到不拿"}), ".1f"), ""]

    comp = [ciRow("A_V(M12)（%）", summarize(100 * m12.A_V)), ciRow("A_V(M10)（%）", summarize(100 * m10.A_V)),
            ciRow("S_in(M12)（%）", summarize(100 * m12.S_in)), ciRow("S_in(M10)（%）", summarize(100 * m10.S_in)),
            ciRow("ΔA（pp）", summarize(blocks.dA_pp)), ciRow("ΔS（pp）", summarize(blocks.dS_pp))]
    for row, values in zip(comp[:4], (m12.A_V, m10.A_V, m12.S_in, m10.S_in)):   # 正確率：不帶正負號，「為正」沒有意義
        s = summarize(100 * values)
        row.update({"平均": f"{s['mean']:.2f}", "95% 區間": f"[{s['ci_low']:.2f}, {s['ci_high']:.2f}]", "為正的區塊": "—"})
    tops = pd.DataFrame([{"區塊": label(r), "S_in(M10) 前兩名": r.S_in_top,
                          "（對照）S_in(M12) 前兩名": m12.iloc[i].S_in_top} for i, r in enumerate(m10.itertuples())])
    out += ["### 7.5 第二部分的組成", "",
            md(pd.DataFrame(comp), text=True), "", "S_in(M10) 最常選到的 path（200 次切分中的比例）：", "", md(tops, text=True), ""]
    out += ["### 7.6 沒有 persona 的多數決對上 12 條中最強的單一 path", "",
            md(pd.DataFrame([ciRow("A_V(M10) − S_in(M12)（pp）", summarize(blocks.AV10_minus_S12_pp))]), text=True), ""]

    dec = []
    for name in DECOMPOSED:
        row = {"菜單": name, **{key: blocks[f"{name}_{key}"].mean() for key in DECOMPOSITION},
               "無定義的切分": int(blocks[f"{name}_undefined_splits"].sum())}
        row["Excess_in（pp）"] = table[table.menu == name].excess_in_pp.mean()
        dec.append(row)
    out += ["### 7.7 分解（M12 的子集上，16 個區塊的平均）", "",
            "headroom 以百分點表示；各量先對切分平均，所以平均後的乘積不必等於平均 Excess_in。逐區塊的值在 `rq3_blocks.csv`（M12 那一列的 "
            "`M12_*`、`M10_*` 欄）。", "", md(pd.DataFrame(dec).rename(columns={"headroom": "headroom（pp）"}), ".3f"),
            "", f"恆等式 Excess_in = d × (c − m) × (recovery_V − recovery_blind)：每次切分核對，最大誤差 {identity:.1e}。", ""]

    side = []
    for name, title in (("M12", "M12"), ("M10", "M10（去掉 P1、P2）"), (DROP["P2"], "M12−P2（只去掉 P:skeptic）"),
                        (DROP["P1"], "M12−P1（只去掉 P:expert）")):
        g = table[table.menu == name]
        s = summarize(g.excess_in_pp)
        side.append({"菜單": title, "A_V（%）": f"{100 * g.A_V.mean():.2f}", "S_in（%）": f"{100 * g.S_in.mean():.2f}",
                     "Excess_in（pp）": pp(s["mean"]), "Excess_in 95% 區間": f"[{pp(s['ci_low'])}, {pp(s['ci_high'])}]",
                     "為正的區塊": f"{s['n_positive']}/16"})
    out += ["### 7.8 只去掉一條 persona 的版本（16 個區塊的平均）", "", md(pd.DataFrame(side), text=True), "",
            f"![16 個區塊的 Excess_in：M12 對 M10]({FIG_B})", ""]

    Path(os.path.join(args.out_dir, "report.md")).write_text("\n".join(out), encoding="utf-8")
    print(f"判定一 Prune1 {pp(j1['mean'])} [{pp(j1['ci_low'])}, {pp(j1['ci_high'])}] {j1['n_positive']}/16 -> {s1}")
    print(f"判定二 ΔExcess {pp(j2['mean'])} [{pp(j2['ci_low'])}, {pp(j2['ci_high'])}] {j2['n_positive']}/16 -> {s2}")
    print(f"TruthfulQA：{tq_reading}")
    print(f"-> {args.out_dir}/report.md")


if __name__ == "__main__":
    main()
