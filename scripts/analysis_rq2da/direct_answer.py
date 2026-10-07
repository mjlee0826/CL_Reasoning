"""
direct_answer.py — RQ2-DA：強裁判挑選 vs 強模型直接作答（result/analysis/rq2da/rq2da_criteria.md）

全程離線：不呼叫 API，不重跑任何東西，不修改現有檔案。判定標準確認前不執行。
    1. 48 格（弱模型 g × 強模型 s × 資料集 × 配對）在 g 的 both_answered 題目上：Sys_J、Sys_D、Sys_self、Strong_alone、
       Weak_EN 與不一致題上的 a、b、agree、d（Analysis/directAnswer.py）。每個 generator × Judge 用 RQ2 判定時實際採用的版本
    2. §5 的三項檢查（任何一項不符就停，不寫輸出）：重現結果 13（+2.47 [+1.65, +3.29]，8/8）、
       每格不一致題數 = RQ2 的 cross_cells.csv、每格恆等式 Sys_J − Sys_D = d × (a − b)（1e-9）
    3. 判定一（Diff = Sys_J − Sys_D，8 個區塊，門檻 0.5pp）與 §6 只報告的量
    4. 輸出（--out-dir）：rq2da_cells.csv、rq2da_blocks.csv、report.md、systems_by_block.png

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq2da/direct_answer.py
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
from Analysis.blockStats import summarize
from Analysis.crossJudge import MODELS, STRONG, WEAK, DATASETS, PAIRS
from Analysis.directAnswer import (OUT_DIR, CRITERIA_FILE, RQ2_DIR, RQ2_CELLS, THRESHOLD, SYSTEMS, DIFFS, CLASSES,
                                   CANDIDATE_CLASSES, TABLE, FORWARD, REVERSE, EQUIVALENT, buildAll, blockTable, state,
                                   checks, pooled)

MODEL_LABELS = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
                "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
READINGS = {
    FORWARD: "弱模型的候選對強模型有幫助。可以把「弱模型作答、答案不同時交給強模型裁決」寫成建議，並報幅度。",
    REVERSE: "裁決這一步沒有價值。結果 13 的 2.47pp 只當診斷（瓶頸在裁判），不當建議；答案不同時直接讓強模型作答比較好。",
    EQUIVALENT: "裁決和直接作答的正確率相當（差距在 ±0.5pp 內）。論文寫「裁判這一步在正確率上沒有額外好處」，選哪一個由成本決定。",
    "無法判定": "只報區間，不提出建議。",
}
CLASS_LABELS = {"own_chosen": "自己的答案 = 某個候選，裁判選了它", "own_not_chosen": "自己的答案 = 某個候選，裁判沒有選它",
                "own_neither": "自己的答案和兩個候選都不同（含沒有答案）",
                "one_right": "兩個候選一對一錯", "both_wrong": "兩個候選都錯"}
TABLE_LABELS = {"J1_D1": "裁判對、直接答對", "J1_D0": "裁判對、直接答錯", "J0_D1": "裁判錯、直接答對", "J0_D0": "兩者都錯"}
# 類別色（validated：dataviz 參考調色盤的前四格，依固定順序指派）與墨色；圖上用英文（matplotlib 預設字型沒有中文字）
SERIES = {"Sys_self": ("#2a78d6", "o"), "Sys_J": ("#eb6834", "s"), "Sys_D": ("#1baf7a", "^"), "Strong_alone": ("#eda100", "D")}
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6a64", "#d9d8d2", "#fcfcfb"
FIGURE = "systems_by_block.png"


def parseArgs():
    parser = ArgumentParser(description="RQ2-DA: strong judge picking vs the strong model answering directly (rq2da_criteria.md)")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations", help="Main-grid judge files (Qwen's self-judging)")
    parser.add_argument("--rq2-dir", default=RQ2_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    return parser.parse_args()


def pct(x: float, digits: int = 2) -> str:
    return "—" if pd.isna(x) else f"{100 * x:.{digits}f}"


def pp(x: float, digits: int = 2) -> str:
    return "—" if pd.isna(x) else f"{x:+.{digits}f}"


def ciRow(name: str, s: dict) -> dict:
    return {"量": name, "區塊數": s["n_blocks"], "平均（pp）": pp(s["mean"]), "SE": f"{s['se']:.2f}",
            "95% 區間": f"[{pp(s['ci_low'])}, {pp(s['ci_high'])}]", "為正的區塊": f"{s['n_positive']}/{s['n_blocks']}"}


def md(df: pd.DataFrame, floatfmt: str = ".3f", text: bool = False) -> str:
    """text = True：欄位已格式化成字串，不讓 tabulate 再解析成數字（保留正負號與位數）。"""
    return df.to_markdown(index=False, floatfmt=floatfmt, disable_numparse=text)


def groupMeans(cells: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """§6.2：各組的 a、b、agree、d（格的等權平均）與不一致題總數。"""
    groups = [("整體", cells)] if by is None else [(key, g) for key, g in cells.groupby(by, sort=False)]
    return pd.DataFrame([{"組": MODEL_LABELS.get(key, key), "格數": len(g), "不一致題": int(g.n_dis.sum()),
                          "a": g.a.mean(), "b": g.b.mean(), "agree": g.agree.mean(), "d": g.d.mean()} for key, g in groups])


def pooledTable(cells: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for label, group in [("整體", cells)] + [(MODEL_LABELS[s], cells[cells.judge == s]) for s in STRONG]:
        for r in pooled(group, names):
            rows.append({"範圍": label, "類別": CLASS_LABELS[r["class"]], "題數": r["n"], "比例": r["share"], "a": r["a"], "b": r["b"]})
    return pd.DataFrame(rows)


def costTable(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strong in STRONG:
        g = cells[cells.judge == strong]
        rows.append({
            "強模型": MODEL_LABELS[strong], "d（格平均）": round(g.d.mean(), 3),
            "裁判呼叫數": int(g.judge_calls.sum()),
            "裁判 input（重算）": g.judge_tokens_in_sum.sum() / g.judge_calls.sum(),
            "裁判 output（重算）": g.judge_tokens_out_sum.sum() / g.judge_calls.sum(),
            "裁判 input（API）": g.judge_api_in_sum.sum() / g.judge_api_calls.sum(),
            "裁判 output（API）": g.judge_api_out_sum.sum() / g.judge_api_calls.sum(),
            "直接作答 input（不一致題）": g.direct_dis_tokens_in_sum.sum() / g.judge_calls.sum(),
            "直接作答 output（不一致題）": g.direct_dis_tokens_out_sum.sum() / g.judge_calls.sum(),
            "Strong_alone 呼叫數": int(g.direct_all_calls.sum()),
            "直接作答 input（全部題目）": g.direct_all_tokens_in_sum.sum() / g.direct_all_calls.sum(),
            "直接作答 output（全部題目）": g.direct_all_tokens_out_sum.sum() / g.direct_all_calls.sum(),
        })
    return pd.DataFrame(rows)


def matrixTable(matrix: pd.DataFrame, value: str) -> str:
    table = matrix.pivot_table(index="generator", columns="judge", values=value, aggfunc="mean").reindex(index=MODELS, columns=MODELS)
    table.index = [MODEL_LABELS[m] for m in table.index]
    table.columns = [MODEL_LABELS[m] for m in table.columns]
    table.index.name = "產生候選 ＼ 裁判"
    return table.round(3).to_markdown()


def plotBlocks(blocks: pd.DataFrame, path: str, criteria_sha: str):
    """8 個區塊（列）× 四種系統的正確率（點），同一列以細線連起最小到最大。"""
    labels = [f"{MODEL_LABELS[r.generator]} · {r.dataset}" for r in blocks.itertuples()]
    y = np.arange(len(blocks))[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    values = 100 * blocks[list(SERIES)].to_numpy()
    ax.hlines(y, values.min(axis=1), values.max(axis=1), color=GRID, linewidth=2, zorder=1)
    for name, (color, marker) in SERIES.items():
        ax.scatter(100 * blocks[name], y, s=58, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=name)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=9)
    ax.set_xlabel("Accuracy on both-answered items (%)", color=MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("RQ2-DA: weak model's candidates, strong model as judge vs answering directly\n"
                 "(each block = mean of 2 strong models × 3 pairs)", color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, metadata={"Description": f"rq2da_criteria.md sha256 {criteria_sha}"})
    plt.close(fig)


def main():
    args = parseArgs()
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty)")
    criteria_sha = sha256(criteria)

    built = buildAll(args.armdir, args.aggdir, args.rq2_dir)
    cells, matrix, files = built["cells"], built["matrix"], built["files"]
    blocks = blockTable(cells)
    ck = checks(cells, blocks)
    r13 = ck["result13"]
    print(f"§5.1 結果 13：{r13['rounded']}（應為 {r13['expected']}）-> {'通過' if r13['passed'] else '不符'}")
    print(f"§5.2 不一致題數：{ck['n_dis']['n_cells']} 格，不符 {len(ck['n_dis']['mismatches'])} 格 -> {'通過' if ck['n_dis']['passed'] else '不符'}")
    print(f"§5.3 恆等式：最大誤差 {ck['identity']['max_error']:.1e} -> {'通過' if ck['identity']['passed'] else '不符'}")
    if not all(part["passed"] for part in ck.values()):
        for m in ck["n_dis"]["mismatches"][:20]:
            print(f"  ❌ {m}")
        raise SystemExit("❌ §5 的檢查沒有全部通過：停下來回報，不寫輸出")

    os.makedirs(args.out_dir, exist_ok=True)
    cells.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq2da_cells.csv"), index=False)
    blocks.assign(criteria_sha256=criteria_sha).to_csv(os.path.join(args.out_dir, "rq2da_blocks.csv"), index=False)
    plotBlocks(blocks, os.path.join(args.out_dir, FIGURE), criteria_sha)

    diff = summarize(blocks["Sys_J-Sys_D_pp"])
    verdict = state(diff)
    sens = summarize(blocks["Diff_sens_pp"])

    out = ["# RQ2-DA：強裁判挑選 vs 強模型直接作答", ""]
    out += ["## (1) 判定標準", "",
            f"- `{criteria}`，sha256 `{criteria_sha}`",
            f"- 確認：{confirmed}",
            f"- 產生時間：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}；程式：`scripts/analysis_rq2da/direct_answer.py`"
            "（逐格計算在 `Analysis/directAnswer.py`）。不呼叫 API，不重跑任何東西，不修改現有檔案。", ""]

    sources = files.groupby(["source"]).size().to_dict()
    self_files = files[(files.generator == files.judge) & files.generator.isin(WEAK)]
    out += ["## (2) 讀了哪些檔案", "",
            f"- 強裁判的選擇：`{args.rq2_dir}/judge_outputs/{{deepseek4.1flash,gemini3.1flashlite}}/{{gpt4omini,qwen}}/{{資料集}}/judge__{{配對}}.json`"
            "（RQ2 交叉檔，48 個）。欄位：`final_answer`、`trace.chosen_arm`、`tokens_in`、`tokens_out`、`call.usage_in`、`call.usage_out`。",
            "- g 自己裁決自己（Sys_self），用 RQ2 判定時實際採用的版本（`cross_cells.csv` 的 `used_in_analysis`）：",
            *[f"  - {MODEL_LABELS[g]}：`{grp.source.iloc[0]}`，例如 `{grp.path.iloc[0]}`（{len(grp)} 個檔）"
              for g, grp in self_files.groupby("generator", sort=False)],
            "- §6.8 的 4×4 表另外讀其餘的 generator × 裁判組合，同樣用 RQ2 實際採用的版本。全部 192 個組合的來源："
            + "、".join(f"{k} {v}" for k, v in sources.items()) + "（主網格檔只取 both_answered 的不一致題）。",
            f"- `{args.armdir}/{{模型}}/{{資料集}}/`：配對用到的 path（L_en、L_zh、S_T1.0_seed1、P_expert、P_skeptic）與四個模型的 L_en；"
            "欄位 `parsed_answer`、`parse_ok`、`gold`、`tokens_in`、`tokens_out`。",
            f"- `{args.rq2_dir}/{RQ2_CELLS}`：每格用的版本（`source`、`used_in_analysis`）與 `n_dis`（§5.2）。"
            f"結果 13 的數字取自 `{args.rq2_dir}/report.md` §7.2（+2.47pp，[+1.65, +3.29]，8/8 為正）。",
            "- 載入沿用 RQ2 的 `Analysis.crossJudge.crossArrays`：核對裁判紀錄剛好涵蓋 both_answered 的不一致題，計分與 RQ2 相同。", ""]

    out += ["## (3) 第 5 節三項檢查", "",
            f"1. 重現結果 13（Sys_J − Sys_self，8 個區塊）：平均 {r13['summary']['mean']:+.4f}pp，95% 區間 "
            f"[{r13['summary']['ci_low']:+.4f}, {r13['summary']['ci_high']:+.4f}]，{r13['summary']['n_positive']}/8 為正；"
            f"四捨五入到小數第二位 = {r13['rounded']['mean']:+.2f} [{r13['rounded']['ci_low']:+.2f}, {r13['rounded']['ci_high']:+.2f}]，"
            f"與 RQ2 報告的 +2.47 [+1.65, +3.29]、8/8 {'一致 → 通過' if r13['passed'] else '不符'}。",
            f"2. 48 格的不一致題數與 `cross_cells.csv` 的 `n_dis`：{len(ck['n_dis']['mismatches'])} 格不符 → "
            f"{'通過' if ck['n_dis']['passed'] else '不符'}（共 {int(cells.n_dis.sum())} 題）。",
            f"3. 恆等式 Sys_J − Sys_D = d × (a − b)：48 格最大誤差 {ck['identity']['max_error']:.1e}（門檻 1e-09）→ "
            f"{'通過' if ck['identity']['passed'] else '不符'}。", ""]

    means = pd.DataFrame([{"系統": name, "正確率（%，8 個區塊的平均）": pct(blocks[name].mean())} for name in SYSTEMS])
    diffs = pd.DataFrame([ciRow(f"{x} − {y}", summarize(blocks[f"{x}-{y}_pp"])) for x, y in DIFFS])
    out += ["## (4) 總表", "",
            "都在弱模型的 both_answered 題目上；每個區塊（弱模型 × 資料集）是 2 個強模型 × 3 個配對的 6 格等權平均。", "",
            md(means, text=True), "", md(diffs, text=True), "", f"![8 個區塊的四種系統]({FIGURE})", ""]

    out += ["## (5) 判定一：Diff = Sys_J − Sys_D", "",
            f"- 8 個區塊：平均 {pp(diff['mean'])}pp，SE {diff['se']:.2f}，95% 區間 [{pp(diff['ci_low'])}, {pp(diff['ci_high'])}]，"
            f"{diff['n_positive']}/8 為正。",
            f"- 門檻 {THRESHOLD}pp → **{verdict}**。",
            f"- 事先寫下的讀法：{READINGS[verdict]}", ""]
    per_block = blocks[["generator", "dataset", "Sys_J", "Sys_D", "Sys_J-Sys_D_pp"]].copy()
    per_block["generator"] = per_block["generator"].map(MODEL_LABELS)
    per_block["Sys_J"], per_block["Sys_D"] = 100 * per_block["Sys_J"], 100 * per_block["Sys_D"]
    out += ["逐區塊（%、pp）：", "", md(per_block.rename(columns={"Sys_J-Sys_D_pp": "Diff（pp）"}), ".2f"), ""]

    out += ["## (6) 其餘只報告的量（不參與判定）", "",
            "### 6.2 不一致題上的 a、b、agree、d", "",
            "a = 強模型當裁判的正確率；b = 強模型自己 L:en 的正確率；agree = 裁判選出的答案與 L:en 相同（兩邊都要有答案）；"
            "d = 不一致題占 both_answered 的比例。每組是該組各格的等權平均。", ""]
    for title, by in [("整體", None), ("分強模型", "judge"), ("分弱模型", "generator"), ("分配對", "pair"), ("分資料集", "dataset")]:
        out += [f"{title}：", "", md(groupMeans(cells, by)), ""]

    out += ["### 6.3 依強模型自己的答案落在哪裡（題目合併）", "",
            f"「裁判沒有選它」包括裁判沒有給出有效選擇：共 {int(cells.n_own_not_chosen_invalid.sum())} 題"
            f"（全部不一致題中無效選擇 {int(cells.n_invalid_choice.sum())} 題）。", "",
            md(pooledTable(cells, CLASSES)), ""]
    out += ["### 6.4 依候選的對錯（題目合併）", "", md(pooledTable(cells, CANDIDATE_CLASSES)), ""]
    rows = []
    for label, g in [("整體", cells)] + [(MODEL_LABELS[s], cells[cells.judge == s]) for s in STRONG]:
        rows.append({"範圍": label, **{TABLE_LABELS[t]: int(g[f"n_{t}"].sum()) for t in TABLE}})
    out += ["### 6.5 逐題對照（不一致題，題數）", "", md(pd.DataFrame(rows), ".0f"), ""]

    miss = [{"範圍": "整體", "不一致題": int(cells.n_dis.sum()), "L:en 沒有答案": int(cells.n_direct_missing.sum()),
             "比例": cells.n_direct_missing.sum() / cells.n_dis.sum()}]
    for strong in STRONG:
        for dataset in DATASETS:
            g = cells[(cells.judge == strong) & (cells.dataset == dataset)]
            miss.append({"範圍": f"{MODEL_LABELS[strong]} · {dataset}", "不一致題": int(g.n_dis.sum()),
                         "L:en 沒有答案": int(g.n_direct_missing.sum()), "比例": g.n_direct_missing.sum() / g.n_dis.sum()})
    out += ["### 6.6 強模型 L:en 在不一致題上沒有答案", "", md(pd.DataFrame(miss), ".4f"), "",
            f"敏感度（排除這些題目後的 Diff，不參與判定）：8 個區塊平均 {pp(sens['mean'])}pp，"
            f"95% 區間 [{pp(sens['ci_low'])}, {pp(sens['ci_high'])}]，{sens['n_positive']}/8 為正。", ""]

    out += ["### 6.7 呼叫強模型的成本", "",
            "- 呼叫次數：Sys_J 與 Sys_D 都只在不一致題上呼叫強模型（每題 d 次，d 見下表）；Strong_alone 每題一次。",
            "- 每次呼叫的平均 tokens，題目合併。重算 = 每題紀錄的 `tokens_in` / `tokens_out`（arm 紀錄只有重算值，兩種呼叫因此可比）；"
            "API = 裁判紀錄的 `call.usage_in` / `call.usage_out`。", "",
            md(costTable(cells).round(1).assign(**{"d（格平均）": costTable(cells)["d（格平均）"]}), "g"), ""]

    out += ["### 6.8 參考用的 4×4 表（附錄，不做判定）", "",
            "列 = 產生候選的模型，欄 = 裁判的模型；每格是 4 個資料集 × 3 個配對共 12 格的等權平均。b 與 agree 用的是裁判模型自己的 L:en。"
            "對角線（自己裁決自己）：EN+ZH 與 EN+S1 中，L:en 本身就是候選之一。", ""]
    for value in ("a", "b", "agree"):
        out += [f"{value}：", "", matrixTable(matrix, value), ""]

    Path(os.path.join(args.out_dir, "report.md")).write_text("\n".join(out), encoding="utf-8")
    print(f"判定一：{pp(diff['mean'])} [{pp(diff['ci_low'])}, {pp(diff['ci_high'])}]，{diff['n_positive']}/8 -> {verdict}")
    print(f"-> {args.out_dir}/report.md")


if __name__ == "__main__":
    main()
