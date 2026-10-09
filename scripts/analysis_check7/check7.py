"""
check7.py — 核對七：寫作前的數字清理（離線、不呼叫 API、不是新實驗；只確認與整理既有數字，不產生新的判定）

規格：使用者 2026-10-09 貼上的「核對七」。既有檔案一律唯讀；新檔案只寫到 result/analysis/check7/。
    A. 結果 25 的轉換率統一算法（rq3gk / rq3g 的逐替換輸出；規則 C 主表、規則 A 附錄）
    B. 結果 6（0.64 對 0.69）：items/{paths,aggregations}.csv.gz、aggregation_cells.csv
    C. Judge 與 Debate 的輸出 tokens：result/aggregations 的逐題紀錄與每檔的 api_usage
    D. RQ1 中 Debate 的決策結果：rq1/summary.csv、blocks.csv 已有 Debate 的列 → 直接整理，不重算
    E. 12 條菜單下最強單一 path 是 persona 的區塊數：result/arms，子集一，切分同 RQ1-K / RQ3
    F. 三份判定標準檔（RQ3-GSK、RQ3-GJ、RQ3-GJR）：確認版對確認前的草稿
    G. G1 RQ3-GSK 的 K = 7 門檻；G2 RQ3-GJ 分角色的數字

使用者確認的做法（2026-10-09）：
    - C：用逐題紀錄的 tokens_in / tokens_out（呼叫當下用模型自己的 tokenizer 重算），只用有 api_usage 的檔；
      GPT-4o mini 與 Qwen 的舊匯入 Debate 檔（沒有 api_usage）只報數量；輸入另列 API 的每次平均；
      「105–245」對不上時 C 照常算 (1)–(5)。
    - A：(1) 的加總範圍只用參與判定的替換；「轉換率 × K 在 0.80 到 1.00 之間」四捨五入到小數第二位再比。
    - F：確認前的最終草稿沒有存成檔案；把 session 紀錄裡的確認指令反推回去，sha256 等於確認前記錄的雜湊才算找到（標明是重建）。
我在計畫裡寫明、使用者沒有異議的預設：
    - A：相對差距 = |(1) − (2)| ÷ (2)。
    - B：(2) 兩種題目範圍都算；共同題目 = EN、S1、ZH 三條都有答案；(3) Debate 用同樣方式。
    - C：(4) 每個區塊 Σ ÷ Σ 再對區塊平均；(5)「任一輸出 ≥ 8,000」= 兩條 path 的輸出或聚合端在該題的輸出（Judge 就是那一次呼叫）≥ 8,000。
    - 任何一項的「先重現」對不上就停在那一項，其他項照做。

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_check7/check7.py
"""
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import difflib
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import numpy as np
import pandas as pd

from Analysis.blockStats import summarize, fourState
from Analysis.experimentPlan import STEM_TO_ARM, ARMS_TO_PAIR, PATHS
from Analysis.menuJudgeStats import blocksWithSplits
from Analysis.menuVote import allAnswered
from Analysis.menuPrune import M12, PERSONAS
from Analysis.probe import strongest

MODELS = ["gpt4omini", "qwen", "deepseek4.1flash", "gemini3.1flashlite"]
WEAK = ["gpt4omini", "qwen"]
DATASETS = ["mmlu", "mathqa", "truthfulqa", "commonsenseqa"]
LABEL = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
         "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
ANALYSIS = "result/analysis"
OUT = os.path.join(ANALYSIS, "check7")
ARMDIR, AGGDIR = "result/arms", "result/aggregations"
CST = timezone(timedelta(hours=8))
COMMON5 = ["EN+ZH", "EN+JA", "ZH+JA", "EN+S1", "P1+P2"]
TRANSCRIPT = os.path.expanduser(
    "~/.claude/projects/-home-mjlee-Desktop-cl-reasoning/dd2d53a9-8914-49b4-8899-941dc0267bd5.jsonl")
OLD_SCRATCH = "/tmp/claude-1000/-home-mjlee-Desktop-cl-reasoning/dd2d53a9-8914-49b4-8899-941dc0267bd5/scratchpad"

# ------------------------------------------------------------------
# 共用：sha256、來源登記、舊模型排除、跨區塊統計、格式
# ------------------------------------------------------------------
_SHA = {}


def sha256(path: str) -> str:
    if path not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA[path] = h.hexdigest()
    return _SHA[path]


def hashTree() -> list[tuple]:
    """result/analysis/ 底下 check7/ 以外的所有檔案（不用快取，每次重算）。"""
    rows = []
    for dirpath, _, names in os.walk(ANALYSIS):
        if os.path.relpath(dirpath, ANALYSIS).split(os.sep)[0] == "check7":
            continue
        for name in names:
            path = os.path.join(dirpath, name)
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            rows.append((path, h.hexdigest(), os.path.getsize(path)))
    return sorted(rows)


SOURCES, MANIFEST, EXCLUDED = [], [], []
CURRENT = ["—"]   # 目前在算哪一項（排除表的「用在」欄）


def source(item: str, quantity: str, files: list[str], filt: str, algo: str):
    """sources.csv 的一列。檔案超過 6 個時寫進 source_manifest.csv（同一個群組名稱），這裡只放群組。"""
    if len(files) > 6:
        group = f"{item}:{quantity}"
        for f in files:
            MANIFEST.append({"group": group, "path": f, "sha256": sha256(f)})
        files_col, sha_col = f"見 source_manifest.csv 群組「{group}」（{len(files)} 個檔案）", ""
    else:
        files_col, sha_col = ";".join(files), ";".join(sha256(f) for f in files)
    SOURCES.append({"item": item, "quantity": quantity, "files": files_col, "sha256": sha_col, "filter": filt,
                    "algorithm": algo})


def dropOld(df: pd.DataFrame, file: str, col: str = "model") -> pd.DataFrame:
    old = ~df[col].isin(MODELS)
    for e in EXCLUDED:
        if e["file"] == file and e["column"] == col:
            e["used_in"] = "、".join(dict.fromkeys(e["used_in"].split("、") + [CURRENT[0]]))
            break
    else:
        EXCLUDED.append({"file": file, "column": col, "rows": len(df), "excluded_rows": int(old.sum()),
                         "excluded_models": ", ".join(sorted(set(df.loc[old, col].astype(str)))) or "—", "used_in": CURRENT[0]})
    return df[~old]


def S(values) -> dict:
    return summarize(np.asarray(values, dtype=float))


def ci(s: dict, d: int = 2, npos: bool = True) -> str:
    f = f"{{:+.{d}f}}"
    out = f"{f.format(s['mean'])}（{f.format(s['ci_low'])} 到 {f.format(s['ci_high'])}）"
    return out + (f"，{s['n_positive']}/{s['n_blocks']}" if npos else "")


def plain(s: dict, d: int = 3) -> str:
    return f"{s['mean']:.{d}f}（{s['ci_low']:.{d}f} 到 {s['ci_high']:.{d}f}）"


def matches(s: dict, target: tuple) -> bool:
    """target = (mean, ci_low, ci_high, n_positive 或 None)，都到小數第二位。"""
    ok = all(f"{s[k]:.2f}" == f"{v:.2f}" for k, v in zip(("mean", "ci_low", "ci_high"), target[:3]))
    return ok and (target[3] is None or s["n_positive"] == target[3])


def md(df: pd.DataFrame, **kw) -> str:
    """表格：字串照原樣（不讓 tabulate 把 "0.350" 改成 0.35）。"""
    return df.astype(object).where(df.notna(), "").astype(str).to_markdown(index=False, disable_numparse=True, **kw)


def toCST(iso: str) -> str:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(CST).strftime("%Y-%m-%d %H:%M:%S CST")


def cst(s: str) -> datetime:
    """'2026-10-07 23:55 CST' → datetime"""
    return datetime.strptime(s.replace(" CST", ""), "%Y-%m-%d %H:%M").replace(tzinfo=CST)


def iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def mtime(path: str) -> str:
    return datetime.fromtimestamp(os.path.getmtime(path), CST).strftime("%Y-%m-%d %H:%M:%S CST")


SUMMARY = []   # 總表：項目｜舊數字｜新數字｜狀態


def summary(item: str, old: str, new: str, status: str):
    SUMMARY.append({"項目": item, "舊數字": old, "新數字": new, "狀態": status})


# ==================================================================
# A. 結果 25 的轉換率統一算法
# ==================================================================
OLD_A = {3: ("10.3", "3.21", "0.326", "0.265", "0.350"), 5: ("10.4", "1.84", "0.188", "0.140", "0.187"),
         7: ("10.4", "1.29", "0.131", "0.096", "0.113"), 9: ("10.4", "0.98", "0.102", "0.071", "0.073"),
         11: ("10.4", "0.80", "0.084", "0.057", "0.049"), 12: ("10.4", "0.78", "0.077", "0.053", "0.067")}
KS = [3, 5, 7, 9, 11, 12]
N_MENUS = {3: 30, 5: 30, 7: 30, 9: 30, 11: 12, 12: 1}


def sectionA() -> list[str]:
    f_sub = f"{ANALYSIS}/rq3gk/rq3gk_substitutions.csv"
    f_kb = f"{ANALYSIS}/rq3gk/rq3gk_k_blocks.csv"
    f_g = f"{ANALYSIS}/rq3g/rq3g_substitutions.csv"
    sub = dropOld(pd.read_csv(f_sub, dtype={"K": str}), f_sub, "host")
    sub = dropOld(sub, f_sub + "（donor 欄）", "donor")
    n_m3 = int((sub.K == "M3").sum())
    sub = sub[sub.K != "M3"].copy()
    sub["K"] = sub.K.astype(int)
    kb = dropOld(pd.read_csv(f_kb, dtype={"K": str}), f_kb)
    kb = kb[kb.K != "M3"].copy()
    kb["K"] = kb.K.astype(int)
    weak = kb[kb.model.isin(WEAK)]
    menus = sub.groupby("K").menu.nunique().to_dict()
    if menus != N_MENUS:
        raise SystemExit(f"A：菜單數 {menus} ≠ {N_MENUS}")
    part = sub[sub.participates]
    nonpart = sub[~sub.participates].groupby(["K", "host", "donor", "dataset"]).size()

    rows = []
    for (K, host, ds), g in part.groupby(["K", "host", "dataset"]):
        den, ind = g.path_inc.sum(), g.indep_num.sum()
        for rule in "AC":
            real, sim = g[f"conv_num_real_{rule}"].sum(), g[f"conv_num_sim_{rule}"].sum()
            rows.append({"K": K, "rule": rule, "model": host, "dataset": ds, "n_menus": g.menu.nunique(), "n_subs": len(g),
                         "sum_path_inc": den, "sum_vote_inc_real": real, "sum_vote_inc_sim": sim, "sum_vote_inc_indep": ind,
                         "conv_real": real / den, "conv_sim": sim / den, "conv_indep": ind / den,
                         "conv_real_x_K": K * real / den, "real_over_indep": real / ind, "real_over_sim": real / sim})
    blocks = pd.DataFrame(rows)
    blocks.to_csv(f"{OUT}/A_blocks.csv", index=False)

    # 重現：既有表（rq3gk_k_blocks.csv 的 8 個弱模型區塊）＋ 逐區塊和 k_blocks 的 conversion_* 比
    rep, maxdiff = [], 0.0
    for K in KS:
        w = weak[weak.K == K].set_index(["model", "dataset"])
        for rule in "AC":
            b = blocks[(blocks.K == K) & (blocks.rule == rule)].set_index(["model", "dataset"])
            for mine, theirs in (("conv_real", f"conversion_real_{rule}"), ("conv_sim", f"conversion_sim_{rule}"),
                                 ("conv_indep", "conversion_indep")):
                maxdiff = max(maxdiff, float((b[mine] - w.loc[b.index, theirs]).abs().max()))
        vals = {r: (f"{100 * w.path_inc.mean():.1f}", f"{100 * w[f'V_inc_{r}'].mean():.2f}",
                    f"{w[f'conversion_real_{r}'].mean():.3f}", f"{w[f'conversion_sim_{r}'].mean():.3f}",
                    f"{w.conversion_indep.mean():.3f}") for r in "AC"}
        rep.append({"K": K, "舊表": " / ".join(OLD_A[K]), "規則 A 重算": " / ".join(vals["A"]),
                    "規則 C 重算": " / ".join(vals["C"]), "規則 A 相同": vals["A"] == OLD_A[K], "規則 C 相同": vals["C"] == OLD_A[K]})
    rep = pd.DataFrame(rep)
    reproduced = bool(rep["規則 A 相同"].all())
    xk_old = [K * weak[weak.K == K].conversion_real_A.mean() for K in KS]

    # rq3g（K = 12 的 M12）和 rq3gk 的 K = 12 是同一批替換
    g12 = pd.read_csv(f_g)
    g12 = g12[g12.menu == "M12"].set_index(["host", "donor", "dataset", "path"])
    k12 = sub[sub.K == 12].set_index(["host", "donor", "dataset", "path"]).loc[g12.index]
    d12 = max(float((g12.conv_num_real - k12.conv_num_real_A).abs().max()), float((g12.conv_den - k12.path_inc).abs().max()),
              float((g12.conv_num_sim - k12.conv_num_sim_A).abs().max()))
    same_part = bool((g12.participates.values == k12.participates.values).all())

    source("A", "既有表的重現（那條 path 進步、多數決進步、三種轉換率）", [f_kb],
           "K ≠ M3；model ∈ {gpt4omini, qwen}（8 個弱模型區塊）",
           "每個 K：8 個區塊的 path_inc、V_inc_A、conversion_real_A、conversion_sim_A、conversion_indep 取平均（path_inc、V_inc × 100）")
    source("A", "(1)–(4) 逐區塊的 Σ", [f_sub], "K ≠ M3；participates = True（只用參與判定的替換）",
           "區塊（host × dataset）× K：Σconv_num_real_{A,C}、Σconv_num_sim_{A,C}、Σindep_num、Σpath_inc；轉換率 = Σ分子 ÷ Σpath_inc")
    source("A", "K = 12 與 RQ3-G 的一致性", [f_g, f_sub], "rq3g：menu = M12；rq3gk：K = 12",
           "逐替換比 conv_num_real / conv_num_sim / conv_den 與 conv_num_real_A / conv_num_sim_A / path_inc")

    # (1)–(4)
    byk = []
    for rule in "CA":
        for K in KS:
            b = blocks[(blocks.K == K) & (blocks.rule == rule)]
            pooled = {"conv_real": b.sum_vote_inc_real.sum() / b.sum_path_inc.sum(),
                      "conv_sim": b.sum_vote_inc_sim.sum() / b.sum_path_inc.sum(),
                      "conv_indep": b.sum_vote_inc_indep.sum() / b.sum_path_inc.sum()}
            for q in ("conv_real", "conv_sim", "conv_indep", "conv_real_x_K", "real_over_indep", "real_over_sim"):
                s = S(b[q])
                row = {"rule": rule, "K": K, "quantity": q, **s}
                if q in pooled:
                    row["pooled_all_blocks"] = pooled[q]
                    row["rel_gap"] = abs(s["mean"] - pooled[q]) / pooled[q]
                byk.append(row)
    byk = pd.DataFrame(byk)
    byk.to_csv(f"{OUT}/A_by_K.csv", index=False)

    def get(rule, K, q, key="mean"):
        return float(byk[(byk.rule == rule) & (byk.K == K) & (byk.quantity == q)][key].iloc[0])

    tables = {}
    for rule in "CA":
        t = []
        for K in KS:
            w = weak[weak.K == K]
            t.append({"K": K, "那條 path 進步（pp）": f"{100 * w.path_inc.mean():.1f}",
                      "多數決進步（pp）": f"{100 * w[f'V_inc_{rule}'].mean():.2f}",
                      "轉換率（真實）": plain(S(blocks[(blocks.K == K) & (blocks.rule == rule)].conv_real)),
                      "轉換率（隨機模型）": f"{get(rule, K, 'conv_sim'):.3f}",
                      "轉換率（假設獨立）": f"{get(rule, K, 'conv_indep'):.3f}",
                      "真實 × K": f"{get(rule, K, 'conv_real_x_K'):.2f}",
                      "真實 ÷ 假設獨立": f"{get(rule, K, 'real_over_indep'):.2f}",
                      "真實 ÷ 隨機模型": f"{get(rule, K, 'real_over_sim'):.2f}"})
        tables[rule] = pd.DataFrame(t)
    gaps = byk[byk.quantity.isin(["conv_real", "conv_sim", "conv_indep"])].copy()
    gapC = gaps[gaps.rule == "C"]
    gap_tab = gapC.pivot(index="K", columns="quantity", values=["mean", "pooled_all_blocks", "rel_gap"])
    gap_rows = []
    for K in KS:
        r = {"K": K}
        for q, name in (("conv_real", "真實"), ("conv_sim", "隨機模型"), ("conv_indep", "假設獨立")):
            r[f"{name}：(1)"] = f"{gap_tab.loc[K, ('mean', q)]:.4f}"
            r[f"{name}：(2)"] = f"{gap_tab.loc[K, ('pooled_all_blocks', q)]:.4f}"
            r[f"{name}：相對差距"] = f"{100 * gap_tab.loc[K, ('rel_gap', q)]:.1f}%"
        gap_rows.append(r)
    gapA = gaps[gaps.rule == "A"]
    max_gap_C, max_gap_A = float(gapC.rel_gap.max()), float(gapA.rel_gap.max())
    worst = gapC.loc[gapC.rel_gap.idxmax()]
    xkC = [f"{get('C', K, 'conv_real_x_K'):.2f}" for K in KS]
    xkA = [f"{get('A', K, 'conv_real_x_K'):.2f}" for K in KS]
    xkC_raw = [f"{get('C', K, 'conv_real_x_K'):.4f}" for K in KS]
    in_range = all(0.80 <= float(x) <= 1.00 for x in xkC)

    L = ["## A. 結果 25 的轉換率統一算法", "",
         f"來源：`{f_sub}`（逐替換）、`{f_kb}`（既有表的左兩欄）、`{f_g}`（K = 12 的一致性）。8 個弱模型區塊（宿主 gpt4omini、qwen）；"
         f"K = 3、5、7、9 各 30 種組合，K = 11 是 12 種，K = 12 是 1 種，共 {sum(menus.values())} 種。另有 {n_m3} 列 M3L / M3S / M3P（RQ3-GK 只用在檢查）不納入。",
         "",
         f"參與判定的替換：{len(part)} / {len(sub)}。不參與的 {len(sub) - len(part)} 個全在 "
         + "、".join(f"{h} × {d} × {ds}（{n}）" for (h, d, ds), n in nonpart.groupby(level=[1, 2, 3]).sum().items())
         + "（排除的切分 > 20%），依使用者的決定不納入 (1)。", "",
         "### A.1 重現既有的表", "",
         "格式：那條 path 進步 / 多數決進步 / 轉換率（真實）/（隨機模型）/（假設獨立）。", "",
         md(rep), "",
         f"- 既有的表**用的是平手規則 A**（規則 A 的重算逐字相同；規則 C 不同）。重現：{'通過' if reproduced else '對不上'}。",
         "- 既有表的**轉換率**是：每個區塊、每個 K，把該 K 全部菜單裡參與判定的替換的分子加總、分母加總再相除，然後對 8 個區塊取平均"
         "（`rq3gk_criteria.md` §5「轉換率」）。從逐替換的 CSV 重算，逐區塊和 `rq3gk_k_blocks.csv` 的 `conversion_*` 最大差 "
         f"{maxdiff:.1e}。",
         "- 既有表的**左兩欄**（那條 path 進步、多數決進步）是三段平均：每個替換 → 每份菜單平均 → 該 K 的菜單平均 → 8 個區塊平均"
         "（`rq3gk_criteria.md` §5「替換的量」）。所以左兩欄相除不等於轉換率（例：3.21 ÷ 10.3 = 0.312，表上是 0.326）。",
         f"- 也就是說，規格 (1) 的統一算法就是既有轉換率欄的算法；既有表要換的是平手規則，而不是轉換率的算法。",
         f"- 「轉換率 × K 在 0.92 到 0.98 之間」的重現（規則 A）：{min(xk_old):.2f} 到 {max(xk_old):.2f} → "
         f"{'通過' if (f'{min(xk_old):.2f}', f'{max(xk_old):.2f}') == ('0.92', '0.98') else '對不上'}。",
         f"- K = 12 和 RQ3-G（`rq3g_substitutions.csv`，M12）逐替換最大差 {d12:.1e}，參與判定的標記{'相同' if same_part else '不同'}。", ""]
    if not reproduced:
        summary("A 結果 25 的表", "0.326 …", "見 A.1", "對不上")
        return L + ["既有的表對不上，A 停在這裡。", ""]

    L += ["### A.2 新數字", "",
          "(1) 統一算法：區塊內 Σ（投票的進步）÷ Σ（那條 path 的進步），加總範圍是該 K 的全部參與判定的替換（組合 × path × 2 個供體），再對 8 個區塊取平均。"
          "三欄都用這個算法；假設獨立的分子是 Poisson-binomial 的「替換後 − 替換前」（不受平手規則影響）。"
          "左兩欄照既有算法（三段平均），只為了和既有的表對照。", "",
          "**主表（規則 C）**，轉換率（真實）附 95% t 區間（8 個區塊，全部為正）：", "", md(tables["C"]), "",
          "**附錄表（規則 A）**：", "", md(tables["A"]), "",
          "(2) 對照：8 個區塊全部合在一起的 Σ ÷ Σ（規則 C），以及和 (1) 的相對差距 = |(1) − (2)| ÷ (2)：", "",
          md(pd.DataFrame(gap_rows)), "",
          f"- 規則 C 的最大相對差距 {100 * max_gap_C:.1f}%（K = {int(worst.K)}，{ {'conv_real': '真實', 'conv_sim': '隨機模型', 'conv_indep': '假設獨立'}[worst.quantity] }）；"
          f"規則 A 的最大相對差距 {100 * max_gap_A:.1f}%。逐 K 的值在 `A_by_K.csv`（`pooled_all_blocks`、`rel_gap`）。",
          f"- (3) 真實轉換率 × K：規則 C {', '.join(xkC)}（未四捨五入 {', '.join(xkC_raw)}）；"
          f"規則 A {', '.join(xkA)}（K = 3、5、7、9、11、12）。"
          "真實 ÷ 假設獨立與真實 ÷ 隨機模型都是每個區塊內先算、再對區塊平均（上表）。", "",
          "### A.3 讀法（事先寫好的）", ""]
    if max_gap_C <= 0.02:
        L.append(f"- 每個 K 的 (1) 與 (2) 相對差距都在 2% 以內（最大 {100 * max_gap_C:.1f}%）→ 表不用加註。")
        note = "不用加註"
    else:
        L.append(f"- 有相對差距超過 2%（最大 {100 * max_gap_C:.1f}%）→ 表註寫明「轉換率是區塊內先加總再相除，不等於前兩欄相除」。")
        note = "要加表註"
    if in_range:
        L.append(f"- 六個 K 的（轉換率 × K）四捨五入到小數第二位都在 0.80 到 1.00 之間（{min(xkC)} 到 {max(xkC)}）"
                 f"→ 可以寫「約 K 分之一、略低」，並附範圍 {min(xkC)} 到 {max(xkC)}。")
    else:
        L.append(f"- 有 K 的（轉換率 × K）不在 0.80 到 1.00 之間（{', '.join(xkC)}）→ 不用「K 分之一」，直接列各 K 的數字。")
    L += ["- 「真實 ÷ 假設獨立」只列各 K 的數字（上表），不下判定："
          + "、".join(f"K = {K} {get('C', K, 'real_over_indep'):.2f}" for K in KS) + "（規則 C）。", ""]
    summary("A 結果 25 的表（既有，規則 A）", "0.326 / 0.188 / 0.131 / 0.102 / 0.084 / 0.077", "相同（規則 A，區塊內 Σ ÷ Σ 再平均）", "重現")
    summary("A 結果 25 主表改規則 C（真實轉換率）", "規則 A 的值",
            " / ".join(f"{get('C', K, 'conv_real'):.3f}" for K in KS), "更正")
    summary("A 轉換率 × K", "0.92 到 0.98（規則 A）", f"{min(xkC)} 到 {max(xkC)}（規則 C）", "更正")
    summary("A (1) 對 (2) 的相對差距", "—", f"最大 {100 * max_gap_C:.1f}%（規則 C）→ {note}", "—")
    return L


# ==================================================================
# B. 結果 6（0.64 對 0.69）
# ==================================================================
M12_CODES = ["EN", "ZH", "JA", "RU", "ES", "S1", "S2", "P1", "P2", "W1", "W2", "R"]


def loadItems():
    f_p, f_a = f"{ANALYSIS}/items/paths.csv.gz", f"{ANALYSIS}/items/aggregations.csv.gz"
    p_all, a_all = pd.read_csv(f_p), pd.read_csv(f_a)
    return f_p, f_a, p_all, a_all


def aggMinusEn(P: pd.DataFrame, A: pd.DataFrame, agg: str, pair: str, models: list[str]) -> pd.DataFrame:
    """每個區塊：聚合 − 單用英文（pp），幾種題目範圍。"""
    other = pair.split("+")[1]
    rows = []
    for model in models:
        for ds in DATASETS:
            a = A[(A.model == model) & (A.dataset == ds) & (A.aggregator == agg) & (A.pair == pair)]
            if a.empty:
                continue
            p = P[(P.model == model) & (P.dataset == ds)]
            ans = p.pivot(index="item_id", columns="path", values="answered").astype(bool)
            cor = p.pivot(index="item_id", columns="path", values="correct").astype(float)
            cj = a.set_index("item_id").correct.astype(float).reindex(ans.index)
            own = ans["EN"] & ans[other]
            masks = {"own": own}
            if {"S1", "ZH"} <= set(ans.columns):
                masks["common"] = ans["EN"] & ans["S1"] & ans["ZH"]
            if set(M12_CODES) <= set(ans.columns):
                masks["sub1"] = ans[M12_CODES].all(axis=1)
            for name, m in masks.items():
                rows.append({"model": model, "dataset": ds, "aggregator": agg, "pair": pair, "variant": name, "n_items": int(m.sum()),
                             "value_pp": 100 * (cj[m].mean() - cor.loc[m, "EN"].mean())})
            rows.append({"model": model, "dataset": ds, "aggregator": agg, "pair": pair, "variant": "enall", "n_items": int(own.sum()),
                         "value_pp": 100 * (cj[own].mean() - cor["EN"].mean())})
    return pd.DataFrame(rows)


def sectionB() -> list[str]:
    f_p, f_a, p_all, a_all = loadItems()
    f_c = f"{ANALYSIS}/aggregation_cells.csv"
    f_dj, f_dd = f"{ANALYSIS}/decomposition/pair_table_judge.csv", f"{ANALYSIS}/decomposition/pair_table_debate.csv"
    P, A = dropOld(p_all, f_p), dropOld(a_all, f_a)
    C = dropOld(pd.read_csv(f_c), f_c)
    old_models = sorted(set(a_all.model) - set(MODELS))

    vals = []
    for agg in ("judge", "debate"):
        for pair in ("EN+S1", "EN+ZH"):
            vals.append(aggMinusEn(P, A, agg, pair, MODELS))
            # (d) 含舊模型的列：舊模型有這個配對與聚合器時才有區塊可加
            if old_models:
                old = aggMinusEn(p_all, a_all, agg, pair, old_models)
                if not old.empty:
                    vals.append(old.assign(variant=lambda x: x.variant.map(lambda v: f"old_{v}")))
    V = pd.concat(vals, ignore_index=True)

    # (a)：aggregation_cells.csv 的 Excess = 100·d·(c−m)·(recovery_H2 − recovery_blind_H2)（both_answered）
    ex = C[C.subset == "both_answered"].copy()
    ex["excess_pp"] = 100 * ex.d * (ex.c - ex.m) * (ex.recovery_H2 - ex.recovery_blind_H2)
    exk = ex.set_index(["aggregator", "model", "dataset", "pair"]).excess_pp

    def block(agg, pair, variant, models=MODELS):
        x = V[(V.aggregator == agg) & (V.pair == pair) & (V.variant == variant) & V.model.isin(models)]
        return x.set_index(["model", "dataset"]).value_pp

    def excessDiff(agg):
        return (exk.loc[agg].xs("EN+S1", level="pair") - exk.loc[agg].xs("EN+ZH", level="pair")).dropna()

    def withOld(agg, pair):
        """(d)：新模型的 own 區塊加上舊模型的 own 區塊（若有）。"""
        new = block(agg, pair, "own")
        old = V[(V.aggregator == agg) & (V.pair == pair) & (V.variant == "old_own")].set_index(["model", "dataset"]).value_pp
        return pd.concat([new, old]), len(old)

    rows_out = V.copy()
    rows_out.to_csv(f"{OUT}/B_blocks.csv", index=False)

    # 先重現（Judge，各自的 both_answered）
    rj = {pair: S(block("judge", pair, "own")) for pair in ("EN+S1", "EN+ZH")}
    rep_ok = matches(rj["EN+S1"], (0.69, 0.29, 1.09, None)) and matches(rj["EN+ZH"], (0.31, -0.42, 1.04, 9))
    dj, dd = pd.read_csv(f_dj).set_index("配對"), pd.read_csv(f_dd).set_index("配對")
    dec_diff = max(abs(rj["EN+S1"]["mean"] - dj.loc["EN+S1", "vs_en_pp_mean"]), abs(rj["EN+ZH"]["mean"] - dj.loc["EN+ZH", "vs_en_pp_mean"]))
    source("B", "先重現：聚合 − 單用英文（Judge、Debate；各自的 both_answered）", [f_p, f_a, f_dj, f_dd],
           "model ∈ 四個新模型；pair ∈ {EN+S1, EN+ZH}；題目 = 該配對兩條 path 都有答案（answered）",
           "每個區塊：聚合正確率 − L:en 正確率（同一批題目，×100）；16 個區塊的平均、95% t 區間、為正的區塊數；和 decomposition 的 vs_en_pp_mean 對照")
    L = ["## B. 結果 6（0.64 對 0.69）", "",
         f"來源：`{f_p}`、`{f_a}`（逐題）、`{f_c}`（候選 (a)）、`{f_dj}`、`{f_dd}`（12 組配對的分解報告）。16 個區塊；主網格兩條 path、choice-v1。", "",
         "### B.1 重現", "",
         f"- EN+S1 聚合減單用英文（Judge，各自的 both_answered）：{ci(rj['EN+S1'])}；目標 +0.69（0.29 到 1.09）。",
         f"- EN+ZH：{ci(rj['EN+ZH'])}；目標 +0.31（−0.42 到 1.04），9 / 16。",
         f"- 和分解報告 `pair_table_judge.csv` 的 `vs_en_pp_mean` 最大差 {dec_diff:.1e}。重現：{'通過' if rep_ok else '對不上'}。", ""]
    if not rep_ok:
        summary("B 結果 6", "+0.69 / +0.31", "見 B.1", "對不上")
        return L + ["重現對不上，B 停在這裡。", ""]

    # (1) 0.64 是哪個量：依序試 (a)–(d)
    def candidates(agg, target_kind):
        """target_kind = 'S1'（EN+S1 − EN 類）或 'D'（EN+S1 與 EN+ZH 的差）。回傳 [(候選, 區塊值 Series, 說明)]。"""
        out = [("(a) EN+S1 的 Excess − EN+ZH 的 Excess", excessDiff(agg),
                "aggregation_cells.csv，both_answered，Excess = 100·d·(c−m)·(recovery_H2 − recovery_blind_H2)，每個區塊相減")]
        for v, name in (("sub1", "子集一（12 條 path 都有答案）"), ("own", "各自的 both_answered"), ("common", "EN／S1／ZH 三條都有答案")):
            if target_kind == "S1":
                out.append((f"(b) 題目範圍：{name}", block(agg, "EN+S1", v), f"聚合 − L:en，同一批題目（{name}）"))
            else:
                out.append((f"(b) 題目範圍：{name}", block(agg, "EN+S1", v) - block(agg, "EN+ZH", v), f"兩個配對各自的（聚合 − L:en）相減（{name}）"))
        if target_kind == "S1":
            out.append(("(c) 單用英文改用全部題目", block(agg, "EN+S1", "enall"), "聚合（各自的 both_answered）− L:en（該資料集全部題目）"))
        else:
            out.append(("(c) 單用英文改用全部題目", block(agg, "EN+S1", "enall") - block(agg, "EN+ZH", "enall"),
                        "兩個配對的聚合都減同一個 L:en（全部題目）再相減"))
        if target_kind == "S1":
            vals_d, n_old = withOld(agg, "EN+S1")
            out.append(("(d) 含舊模型的列", vals_d, f"各自的 both_answered，加上舊模型的區塊 {n_old} 個"))
        else:
            n1 = len(V[(V.aggregator == agg) & (V.pair == "EN+S1") & (V.variant == "old_own")])
            out.append(("(d) 含舊模型的列", block(agg, "EN+S1", "own") - block(agg, "EN+ZH", "own"),
                        f"舊模型有 EN+S1 的區塊 {n1} 個，差值無法多加區塊，等於各自的 both_answered"))
        return out

    def search(agg, kind, target, label):
        rows, hit = [], None
        for name, values, how in candidates(agg, kind):
            s = S(values.values)
            ok = matches(s, target)
            rows.append({"目標": label, "候選": name, "做法": how, "區塊數": s["n_blocks"], "平均": f"{s['mean']:+.3f}",
                         "95% 區間": f"{s['ci_low']:+.3f} 到 {s['ci_high']:+.3f}", "為正": s["n_positive"], "對得上": "是" if ok else "否"})
            if ok and hit is None:
                hit = name
        return rows, hit

    cand_rows = []
    r_j, hit_j = search("judge", "S1", (0.64, 0.22, 1.06, 13), "Judge EN+S1 +0.64（0.22 到 1.06），13 / 16")
    cand_rows += r_j

    # (2) D 兩種題目範圍
    D = {}
    for agg in ("judge", "debate"):
        for v in ("own", "common"):
            D[(agg, v)] = block(agg, "EN+S1", v) - block(agg, "EN+ZH", v)
    dsum = {k: S(v.values) for k, v in D.items()}
    pd.DataFrame([{"aggregator": a, "scope": v, "model": m, "dataset": d, "D_pp": x}
                  for (a, v), ser in D.items() for (m, d), x in ser.items()]).to_csv(f"{OUT}/B_D_blocks.csv", index=False)

    # (3) Debate：同樣方式
    rd = {pair: S(block("debate", pair, "own")) for pair in ("EN+S1", "EN+ZH")}
    dec_diff_d = max(abs(rd["EN+S1"]["mean"] - dd.loc["EN+S1", "vs_en_pp_mean"]), abs(rd["EN+ZH"]["mean"] - dd.loc["EN+ZH", "vs_en_pp_mean"]))
    deb_targets = [("EN+S1", (0.87, 0.43, 1.30, 16), "Debate EN+S1 +0.87（0.43 到 1.30），16 / 16", "S1"),
                   ("EN+ZH", (1.00, 0.04, 1.96, 14), "Debate EN+ZH +1.00（0.04 到 1.96），14 / 16", None),
                   ("D", (-0.13, -0.91, 0.64, None), "Debate 兩者相減 −0.13（−0.91 到 0.64）", "D")]
    deb_status = {}
    for key, target, label, kind in deb_targets:
        own = rd[key] if key != "D" else dsum[("debate", "own")]
        if matches(own, target):
            deb_status[key] = ("重現", "各自的 both_answered（和 Judge 的重現同一種算法）", own)
            continue
        rows, hit = search("debate", kind, target, label)
        cand_rows += rows
        deb_status[key] = ("對得上：" + hit if hit else "都對不上", "", own)
    pd.DataFrame(cand_rows).to_csv(f"{OUT}/B_candidates.csv", index=False)
    source("B", "(1) 候選 (a)", [f_c], "aggregator ∈ {judge, debate}；subset = both_answered；四個新模型",
           "每個區塊 Excess(EN+S1) − Excess(EN+ZH)，Excess = 100·d·(c−m)·(recovery_H2 − recovery_blind_H2)")
    source("B", "(1) 候選 (b)(c)(d) 與 (2)(3) 的 D", [f_p, f_a],
           "四個新模型（(d) 另加舊模型的列）；題目範圍見 B_candidates.csv 的「做法」",
           "每個區塊：聚合 − L:en；D = (EN+S1 − 英文) − (EN+ZH − 英文)；16 個區塊的平均、SE、95% t 區間、為正的區塊數")

    L += ["### B.2 (1) 舊數字 +0.64（0.22 到 1.06），13 / 16 是哪個量", "",
          "依序試 (a) → (d)；「對得上」= 平均、區間兩端到小數第二位相同，且為正的區塊數相同。", "",
          md(pd.DataFrame(r_j)), "",
          f"- 結果：{'對得上的是 ' + hit_j if hit_j else '**都對不上**'}。",
          f"- 舊模型（{', '.join(old_models)}）在 items 匯出裡只有 Debate 的 10 組語言配對（沒有 S1 這條 path、沒有 Judge），所以 (d) 對 Judge EN+S1 不會多出任何區塊。", "",
          "### B.3 (2) D =（EN+S1 的 Judge − 單用英文）−（EN+ZH 的 Judge − 單用英文）", "",
          md(pd.DataFrame([{"題目範圍": n, "D（pp）": ci(dsum[("judge", v)]), "SE": f"{dsum[('judge', v)]['se']:.3f}"}
                           for v, n in (("own", "各自的 both_answered（主要）"), ("common", "EN／S1／ZH 三條都有答案"))])), "",
          f"逐區塊的值在 `B_D_blocks.csv`。各自的 both_answered 下，EN+S1 與 EN+ZH 的 EN 題目不同，所以 D 不等於兩個 Judge 正確率直接相減。", "",
          "### B.4 (3) Debate 那一欄", "",
          f"- 用同樣的算法（各自的 both_answered）：EN+S1 {ci(rd['EN+S1'])}；EN+ZH {ci(rd['EN+ZH'])}；"
          f"D {ci(dsum[('debate', 'own')])}；共同題目的 D {ci(dsum[('debate', 'common')])}。"
          f"和 `pair_table_debate.csv` 的 `vs_en_pp_mean` 最大差 {dec_diff_d:.1e}。", ""]
    for key, target, label, kind in deb_targets:
        st, how, own = deb_status[key]
        L.append(f"- {label}：{st}" + (f"（{how}）" if how else f"；各自的 both_answered 是 {ci(own)}"))
    other_rows = [r for r in cand_rows if r["目標"].startswith("Debate")]
    if other_rows:
        L += ["", "對不上的 Debate 舊數字，依序試 (a) → (d)：", "", md(pd.DataFrame(other_rows))]
    main = dsum[("judge", "own")]
    contains0 = main["ci_low"] <= 0 <= main["ci_high"]
    L += ["", "### B.5 讀法（事先寫好的）", "",
          f"- 主要數字是各自的 both_answered：D = {ci(main)}。",
          ("- D 的區間含 0 → 結果 6 的寫法不變（分不出高下），只換數字。" if contains0
           else "- D 的區間不含 0 → 只回報，不改寫法，由使用者決定。"),
          f"- 新的結果 6 表（各自的 both_answered）：Judge EN+S1 {ci(rj['EN+S1'])}、EN+ZH {ci(rj['EN+ZH'])}、兩者相減 {ci(main)}；"
          f"Debate EN+S1 {ci(rd['EN+S1'])}、EN+ZH {ci(rd['EN+ZH'])}、兩者相減 {ci(dsum[('debate', 'own')])}。", ""]
    summary("B 結果 6 Judge EN+S1 − 英文", "+0.64（0.22 到 1.06），13/16", ci(rj["EN+S1"]), "更正（舊數字都對不上）" if not hit_j else "更正")
    summary("B 結果 6 Judge EN+ZH − 英文", "+0.31（−0.42 到 1.04），9/16", ci(rj["EN+ZH"]), "重現")
    summary("B 結果 6 Judge 兩者相減（D）", "+0.33（−0.15 到 0.81）", ci(main), "更正")
    for key, label in (("EN+S1", "Debate EN+S1 − 英文"), ("EN+ZH", "Debate EN+ZH − 英文"), ("D", "Debate 兩者相減（D）")):
        st, _, own = deb_status[key]
        old = {"EN+S1": "+0.87（0.43 到 1.30），16/16", "EN+ZH": "+1.00（0.04 到 1.96），14/16", "D": "−0.13（−0.91 到 0.64）"}[key]
        summary(f"B 結果 6 {label}", old, ci(own), "重現" if st == "重現" else f"更正（{st}）")
    return L


# ==================================================================
# C. Judge 與 Debate 的輸出 tokens
# ==================================================================
def loadAggFile(path: str) -> tuple[dict, list]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data[0], data[1:]


def pairOfFile(path: str) -> tuple[str, str, str, str]:
    stem = os.path.basename(path)[:-5]
    agg, sa, sb = stem.split("__")
    a, b = STEM_TO_ARM[sa], STEM_TO_ARM[sb]
    return agg, ARMS_TO_PAIR[frozenset((a, b))].label, a, b


def qstats(x: np.ndarray, prefix: str) -> dict:
    if len(x) == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "median", "q1", "q3")}
    return {f"{prefix}_mean": float(x.mean()), f"{prefix}_median": float(np.median(x)),
            f"{prefix}_q1": float(np.percentile(x, 25)), f"{prefix}_q3": float(np.percentile(x, 75))}


def sectionC() -> list[str]:
    f_p = f"{ANALYSIS}/items/paths.csv.gz"
    P = dropOld(pd.read_csv(f_p), f_p)
    ans = {(m, d, a): (dict(zip(g.item_id, g.answered.astype(bool))), dict(zip(g.item_id, g.tokens_out)))
           for (m, d, a), g in P.groupby(["model", "dataset", "arm_id"])}

    # 舊模型的聚合檔：只數，不用
    old_files = sorted(f for f in glob.glob(f"{AGGDIR}/*/*/*.json") if f.split(os.sep)[2] not in MODELS)
    old_records = sum(len(loadAggFile(f)[1]) for f in old_files)
    EXCLUDED.append({"file": f"{AGGDIR}/（舊模型資料夾）", "column": "資料夾", "rows": old_records, "excluded_rows": old_records,
                     "excluded_models": ", ".join(sorted({f.split(os.sep)[2] for f in old_files})) + f"（{len(old_files)} 個檔）", "used_in": "C"})

    files, items = [], []
    for model in MODELS:
        for ds in DATASETS:
            for path in sorted(glob.glob(f"{AGGDIR}/{model}/{ds}/*.json")):
                agg, pair, a, b = pairOfFile(path)
                if agg not in ("judge", "debate"):
                    continue
                meta, recs = loadAggFile(path)
                usage = meta.get("api_usage")
                A, B = ans[(model, ds, a)], ans[(model, ds, b)]
                n_calls_file = 0
                for r in recs:
                    t = r["trace"]
                    called = t is not None
                    if agg == "judge":
                        calls = 1 if called else 0
                    elif usage is not None:
                        final = isinstance(t, dict) and bool(t.get("Result3"))
                        calls = 2 * int(r["n_rounds"] or 0) + int(final) if called else 0
                    else:
                        calls = None
                    n_calls_file += calls or 0
                    i = r["item_id"]
                    both = A[0][i] and B[0][i]
                    big = A[1][i] >= 8000 or B[1][i] >= 8000 or r["tokens_out"] >= 8000
                    items.append({"model": model, "dataset": ds, "aggregator": agg, "pair": pair, "item_id": i,
                                  "has_usage": usage is not None, "called": called or (agg == "debate" and (r["n_rounds"] or 0) > 0),
                                  "n_calls": calls, "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
                                  "both_answered": both, "ge8000": bool(big)})
                files.append({"model": model, "dataset": ds, "aggregator": agg, "pair": pair, "file": path, "sha256": sha256(path),
                              "source": meta.get("source"), "prompt_version": meta.get("prompt_version"), "has_usage": usage is not None,
                              "records": len(recs), "called_items": sum(1 for r in recs if r["trace"] is not None),
                              "calls_counted": n_calls_file if usage is not None or agg == "judge" else None,
                              "api_calls": (usage or {}).get("calls"), "api_completion_tokens": (usage or {}).get("completion_tokens"),
                              "api_prompt_tokens": (usage or {}).get("prompt_tokens"), "api_calls_without_usage": (usage or {}).get("calls_without_usage"),
                              "sum_tokens_out": sum(r["tokens_out"] for r in recs), "sum_tokens_in": sum(r["tokens_in"] for r in recs)})
    F = pd.DataFrame(files)
    I = pd.DataFrame(items)
    F["out_diff_vs_api"] = F.sum_tokens_out - F.api_completion_tokens
    F["calls_diff_vs_api"] = F.calls_counted - F.api_calls
    F["in_overhead_per_call"] = (F.api_prompt_tokens - F.sum_tokens_in) / F.api_calls
    F.to_csv(f"{OUT}/C_files.csv", index=False)
    source("C", "Judge 與 Debate 的逐題紀錄與 api_usage", list(F.file), "四個新模型 × 四個資料集；檔名 judge__* 與 debate__*（各 12 組配對）",
           "見 C_files.csv：每檔的 sha256、api_usage、重算總和與差")
    source("C", "both_answered 與 path 輸出 tokens", [f_p], "四個新模型；每個配對的兩條 path（arm_id）",
           "answered 判斷 both_answered；tokens_out ≥ 8000 判斷 (5) 的排除")
    usageF = F[F.has_usage]
    out_eq = int((usageF.out_diff_vs_api == 0).sum())
    calls_eq = int((usageF.calls_diff_vs_api == 0).sum())

    # ---------- 重現四個舊數字 ----------
    f_c = f"{ANALYSIS}/aggregation_cells.csv"
    cells = dropOld(pd.read_csv(f_c), f_c)
    rep = {}
    for agg in ("judge", "debate"):
        x = cells[(cells.aggregator == agg) & (cells.subset == "both_answered") & cells.pair.isin(COMMON5)]
        rep[agg] = x.groupby("model").tok_out_agg.mean()
    # 單位核對：tok_out_agg = 該格 both_answered 題目上 tokens_out 的平均（沒呼叫的題目算 0）
    mine = I[I.both_answered].groupby(["model", "dataset", "aggregator", "pair"]).tokens_out.mean()
    cc = cells[(cells.subset == "both_answered") & cells.aggregator.isin(["judge", "debate"])].set_index(["model", "dataset", "aggregator", "pair"]).tok_out_agg
    unit_diff = float((mine.loc[cc.index] - cc).abs().max())
    j_rng = (f"{rep['judge'].min():.0f}", f"{rep['judge'].max():.0f}")
    d_rng = (f"{rep['debate'].min():.0f}", f"{rep['debate'].max():.0f}")
    # 105–245 的候選
    f_rq2 = f"{ANALYSIS}/rq2/cross_cells.csv"
    rq2 = pd.read_csv(f_rq2)
    u = rq2[rq2.used_in_analysis]
    rq2_per = (u.tok_out_per_call * u.n_dis).groupby(u.judge).sum() / u.n_dis.groupby(u.judge).sum()
    pre_rows = []
    for label, pattern in (("RQ2 §4.1 流程核對（mmlu × EN+S1 重跑）", f"{ANALYSIS}/rq2/precheck/*/*/mmlu/judge__L_en__S_T1.0_seed1.json"),
                           ("RQ1-KJ §6.3（mmlu × EN+S1 重跑）", f"{ANALYSIS}/rq1kj/precheck/*/mmlu/EN+S1.json")):
        for f in sorted(glob.glob(pattern)):
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            recs = data[1:] if isinstance(data, list) else data.get("records", [])
            t = [r["tokens_out"] for r in recs if isinstance(r, dict) and r.get("tokens_out")]
            pre_rows.append({"來源": label, "Judge": f.split(os.sep)[-3] if "rq1kj" in f else f.split(os.sep)[-4], "檔案": f,
                             "呼叫數": len(t), "每次平均": round(float(np.mean(t)), 1)})
    pre = pd.DataFrame(pre_rows)
    cand105 = [{"來源": "RQ2 §(3)（`cross_cells.csv`，used_in_analysis，每個 Judge 48 格，含交叉格）", "範圍": f"{rq2_per.min():.1f}–{rq2_per.max():.1f}",
                "每個 Judge": "、".join(f"{LABEL[j]} {v:.1f}" for j, v in rq2_per.items())}]
    for label, g in pre.groupby("來源"):
        cand105.append({"來源": label, "範圍": f"{g['每次平均'].min():.1f}–{g['每次平均'].max():.1f}",
                        "每個 Judge": "、".join(f"{LABEL[j]} {v:.1f}" for j, v in zip(g.Judge, g["每次平均"]))})
    hit105 = any(r["範圍"] in ("105.0–245.0",) or tuple(round(float(v)) for v in r["範圍"].split("–")) == (105, 245) for r in cand105)
    # Qwen 中位數 100（75 到 146）：核對四的 12 格，所有呼叫 Judge 的分歧題
    q = I[(I.model == "qwen") & (I.aggregator == "judge") & I.pair.isin(["EN+ZH", "EN+S1", "P1+P2"]) & I.called]
    qt = q.tokens_out.to_numpy(float)
    q_med, q10, q90 = np.median(qt), np.percentile(qt, 10), np.percentile(qt, 90)
    q_ok = (f"{q_med:.0f}", f"{q10:.0f}", f"{q90:.0f}") == ("100", "75", "146")
    source("C", "舊數字「15–22」「73–164」", [f_c], "aggregator ∈ {judge, debate}；subset = both_answered；pair ∈ 5 個共同配對；四個新模型",
           "每個模型：20 格（4 資料集 × 5 配對）的 tok_out_agg 取平均；取四個模型的最小與最大，四捨五入到整數")
    source("C", "舊數字「105–245」的候選", [f_rq2] + list(pre["檔案"]), "cross_cells：used_in_analysis；precheck：mmlu × EN+S1",
           "每個 Judge：Σ(tok_out_per_call × n_dis) ÷ Σn_dis；precheck：每次呼叫 tokens_out 的平均")
    source("C", "舊數字「Qwen 中位數 100（75 到 146）」", sorted(set(F[(F.model == "qwen") & (F.aggregator == "judge") & F.pair.isin(["EN+ZH", "EN+S1", "P1+P2"])].file)),
           "qwen；judge；EN+ZH、EN+S1、P1+P2 × 4 資料集；有呼叫 Judge 的分歧題（不限 both_answered，同核對四）",
           "tokens_out 的中位數、第 10 與第 90 百分位")

    # ---------- (1)–(3) ----------
    scopes = {"both_answered": I.both_answered, "both_answered_lt8000": I.both_answered & ~I.ge8000}
    brows = []
    for scope, mask in scopes.items():
        X = I[mask]
        for (model, ds), g in X.groupby(["model", "dataset"]):
            j = g[g.aggregator == "judge"]
            jc = j[j.called]
            brows.append({"scope": scope, "model": model, "dataset": ds, "aggregator": "judge", "pairs": j.pair.nunique(),
                          "n_items": len(j), "n_calls": int(jc.n_calls.sum()), "sum_out": int(j.tokens_out.sum()), "sum_in": int(j.tokens_in.sum()),
                          **qstats(jc.tokens_out.to_numpy(float), "out_per_call"), **qstats(jc.tokens_in.to_numpy(float), "in_per_call"),
                          "out_per_item": j.tokens_out.sum() / len(j), "calls_per_item": jc.n_calls.sum() / len(j)})
            d = g[(g.aggregator == "debate") & g.has_usage]
            calls = d.n_calls.sum()
            brows.append({"scope": scope, "model": model, "dataset": ds, "aggregator": "debate", "pairs": d.pair.nunique(),
                          "pair_list": " ".join(sorted(d.pair.unique())), "n_items": len(d), "n_called_items": int(d.called.sum()),
                          "n_calls": int(calls), "sum_out": int(d.tokens_out.sum()), "sum_in": int(d.tokens_in.sum()),
                          "out_per_call_mean": d.tokens_out.sum() / calls, "in_per_call_mean": d.tokens_in.sum() / calls,
                          "out_per_item": d.tokens_out.sum() / len(d), "calls_per_item": calls / len(d),
                          "calls_per_called_item": calls / d.called.sum()})
            leg = g[(g.aggregator == "debate") & ~g.has_usage]
            if len(leg):
                brows.append({"scope": scope, "model": model, "dataset": ds, "aggregator": "debate_no_usage（只報數量）",
                              "pairs": leg.pair.nunique(), "pair_list": " ".join(sorted(leg.pair.unique())), "n_items": len(leg),
                              "n_called_items": int(leg.called.sum())})
        # 每檔 api_usage（整檔，不限題目）的每次平均：只有 both_answered 那一份算（與範圍無關）
    Bk = pd.DataFrame(brows)
    api = usageF.groupby(["model", "dataset", "aggregator"]).agg(api_calls=("api_calls", "sum"), api_out=("api_completion_tokens", "sum"),
                                                               api_in=("api_prompt_tokens", "sum")).reset_index()
    api["api_out_per_call_file"] = api.api_out / api.api_calls
    api["api_in_per_call_file"] = api.api_in / api.api_calls
    Bk = Bk.merge(api[["model", "dataset", "aggregator", "api_out_per_call_file", "api_in_per_call_file"]], how="left", on=["model", "dataset", "aggregator"])
    Bk.to_csv(f"{OUT}/C_blocks.csv", index=False)

    mrows = []
    for scope in scopes:
        for agg, cols in (("judge", ["out_per_call_mean", "out_per_call_median", "out_per_call_q1", "out_per_call_q3", "in_per_call_mean",
                                     "in_per_call_median", "in_per_call_q1", "in_per_call_q3", "out_per_item", "api_out_per_call_file", "api_in_per_call_file"]),
                          ("debate", ["out_per_call_mean", "in_per_call_mean", "out_per_item", "calls_per_item", "calls_per_called_item",
                                      "api_out_per_call_file", "api_in_per_call_file"])):
            for model in MODELS:
                b = Bk[(Bk.scope == scope) & (Bk.aggregator == agg) & (Bk.model == model)]
                for c in cols:
                    s = S(b[c])
                    mrows.append({"scope": scope, "aggregator": agg, "model": model, "quantity": c, **s,
                                  "min_dataset": float(b[c].min()), "max_dataset": float(b[c].max())})
    Mk = pd.DataFrame(mrows)
    Mk.to_csv(f"{OUT}/C_models.csv", index=False)

    def mv(scope, agg, model, q):
        return float(Mk[(Mk.scope == scope) & (Mk.aggregator == agg) & (Mk.model == model) & (Mk.quantity == q)]["mean"].iloc[0])

    # (4) 共同配對的倍數
    rrows = []
    for scope, mask in scopes.items():
        X = I[mask & I.pair.isin(COMMON5)]
        for (model, ds), g in X.groupby(["model", "dataset"]):
            dpairs = sorted(g[(g.aggregator == "debate") & g.has_usage].pair.unique())
            jj = g[(g.aggregator == "judge") & g.pair.isin(dpairs)]
            dd = g[(g.aggregator == "debate") & g.pair.isin(dpairs)]
            rrows.append({"scope": scope, "model": model, "dataset": ds, "pairs": " ".join(dpairs), "n_pairs": len(dpairs),
                          "judge_items": len(jj), "debate_items": len(dd), "judge_out": int(jj.tokens_out.sum()), "debate_out": int(dd.tokens_out.sum()),
                          "judge_in": int(jj.tokens_in.sum()), "debate_in": int(dd.tokens_in.sum()),
                          "ratio_out": dd.tokens_out.sum() / jj.tokens_out.sum(), "ratio_in": dd.tokens_in.sum() / jj.tokens_in.sum()})
    R = pd.DataFrame(rrows)
    R.to_csv(f"{OUT}/C_ratio_blocks.csv", index=False)
    rs = {(sc, q): S(R[R.scope == sc][q]) for sc in scopes for q in ("ratio_out", "ratio_in")}
    source("C", "(1)–(3) 每次呼叫與每題平均", ["result/analysis/check7/C_files.csv"], "見 C_files.csv；both_answered（另一份再排除任一輸出 ≥ 8000）；Debate 只用有 api_usage 的檔",
           "區塊 = 模型 × 資料集；Judge 每次呼叫 = 有呼叫的題的 tokens_out（平均、中位數、四分位）；每題平均 = Σtokens_out ÷ 該範圍全部題數；"
           "Debate 每次呼叫 = Σtokens_out ÷ Σ呼叫數（呼叫數 = 2 × 輪數 + 最後裁決）；每個模型 = 4 個區塊的平均")
    source("C", "(4) Debate ÷ Judge 的倍數", ["result/analysis/check7/C_files.csv"],
           "5 個共同配對中 Debate 有 api_usage 的配對（GPT、Qwen 只有 EN+S1、P1+P2；DeepSeek、Gemini 五個都有），Judge 用同樣的配對",
           "每個區塊 Σ(Debate tokens) ÷ Σ(Judge tokens)，再對 16 個區塊平均（輸出、輸入各一）")

    # 表
    t1 = []
    for model in MODELS:
        t1.append({"模型": LABEL[model],
                   "輸出：平均": f"{mv('both_answered', 'judge', model, 'out_per_call_mean'):.1f}",
                   "中位數": f"{mv('both_answered', 'judge', model, 'out_per_call_median'):.1f}",
                   "Q1–Q3": f"{mv('both_answered', 'judge', model, 'out_per_call_q1'):.1f}–{mv('both_answered', 'judge', model, 'out_per_call_q3'):.1f}",
                   "API 輸出/次（整檔）": f"{mv('both_answered', 'judge', model, 'api_out_per_call_file'):.1f}",
                   "輸入：平均": f"{mv('both_answered', 'judge', model, 'in_per_call_mean'):.1f}",
                   "輸入中位數": f"{mv('both_answered', 'judge', model, 'in_per_call_median'):.1f}",
                   "輸入 Q1–Q3": f"{mv('both_answered', 'judge', model, 'in_per_call_q1'):.1f}–{mv('both_answered', 'judge', model, 'in_per_call_q3'):.1f}",
                   "API 輸入/次（整檔）": f"{mv('both_answered', 'judge', model, 'api_in_per_call_file'):.1f}",
                   "(2) 每題平均輸出": f"{mv('both_answered', 'judge', model, 'out_per_item'):.1f}"})
    t3 = []
    for model in MODELS:
        b = Bk[(Bk.scope == "both_answered") & (Bk.aggregator == "debate") & (Bk.model == model)]
        t3.append({"模型": LABEL[model], "配對": b.pair_list.iloc[0] if b.pairs.iloc[0] < 12 else "12 組全部",
                   "輸出/次（平均）": f"{mv('both_answered', 'debate', model, 'out_per_call_mean'):.1f}",
                   "API 輸出/次（整檔）": f"{mv('both_answered', 'debate', model, 'api_out_per_call_file'):.1f}",
                   "輸入/次（平均）": f"{mv('both_answered', 'debate', model, 'in_per_call_mean'):.1f}",
                   "API 輸入/次（整檔）": f"{mv('both_answered', 'debate', model, 'api_in_per_call_file'):.1f}",
                   "每題平均輸出": f"{mv('both_answered', 'debate', model, 'out_per_item'):.1f}",
                   "每題平均呼叫次數": f"{mv('both_answered', 'debate', model, 'calls_per_item'):.3f}",
                   "有呼叫的題平均呼叫次數": f"{mv('both_answered', 'debate', model, 'calls_per_called_item'):.2f}"})
    tmd = []
    for model in MODELS:
        for ds in DATASETS:
            j = Bk[(Bk.scope == "both_answered") & (Bk.aggregator == "judge") & (Bk.model == model) & (Bk.dataset == ds)].iloc[0]
            d = Bk[(Bk.scope == "both_answered") & (Bk.aggregator == "debate") & (Bk.model == model) & (Bk.dataset == ds)].iloc[0]
            tmd.append({"模型": LABEL[model], "資料集": ds, "Judge 呼叫數": int(j.n_calls),
                        "Judge 輸出/次：平均": f"{j.out_per_call_mean:.1f}", "中位數": f"{j.out_per_call_median:.1f}",
                        "Q1–Q3": f"{j.out_per_call_q1:.1f}–{j.out_per_call_q3:.1f}", "Judge 輸入/次：平均": f"{j.in_per_call_mean:.1f}",
                        "Judge 每題平均輸出": f"{j.out_per_item:.1f}", "Debate 配對數": int(d.pairs), "Debate 呼叫數": int(d.n_calls),
                        "Debate 輸出/次：平均": f"{d.out_per_call_mean:.1f}", "Debate 每題平均輸出": f"{d.out_per_item:.1f}"})
    tex = []
    for model in MODELS:
        tex.append({"模型": LABEL[model],
                    "Judge 輸出/次：平均": f"{mv('both_answered_lt8000', 'judge', model, 'out_per_call_mean'):.1f}",
                    "中位數": f"{mv('both_answered_lt8000', 'judge', model, 'out_per_call_median'):.1f}",
                    "Judge 輸入/次：平均": f"{mv('both_answered_lt8000', 'judge', model, 'in_per_call_mean'):.1f}",
                    "Judge 每題平均輸出": f"{mv('both_answered_lt8000', 'judge', model, 'out_per_item'):.1f}",
                    "Debate 輸出/次：平均": f"{mv('both_answered_lt8000', 'debate', model, 'out_per_call_mean'):.1f}",
                    "Debate 輸入/次：平均": f"{mv('both_answered_lt8000', 'debate', model, 'in_per_call_mean'):.1f}",
                    "Debate 每題平均輸出": f"{mv('both_answered_lt8000', 'debate', model, 'out_per_item'):.1f}"})
    legacy = Bk[(Bk.scope == "both_answered") & (Bk.aggregator.str.startswith("debate_no_usage"))]
    leg_files = F[(F.aggregator == "debate") & ~F.has_usage]
    t4 = []
    for sc, name in (("both_answered", "both_answered（主要）"), ("both_answered_lt8000", "再排除任一輸出 ≥ 8,000")):
        t4.append({"題目範圍": name, "輸出倍數（16 區塊）": ci(rs[(sc, "ratio_out")]), "輸入倍數（16 區塊）": ci(rs[(sc, "ratio_in")]),
                   **{f"輸出倍數：{LABEL[m]}": f"{R[(R.scope == sc) & (R.model == m)].ratio_out.mean():.2f}" for m in MODELS}})
    ex_n = I[I.both_answered & I.ge8000].groupby(["aggregator"]).size().to_dict()
    overhead = usageF.groupby(["aggregator", "model"]).in_overhead_per_call.agg(["mean", "min", "max"]).round(2).reset_index()
    jmin, jmax = min(mv("both_answered", "judge", m, "out_per_call_mean") for m in MODELS), max(mv("both_answered", "judge", m, "out_per_call_mean") for m in MODELS)
    jmedmin, jmedmax = min(mv("both_answered", "judge", m, "out_per_call_median") for m in MODELS), max(mv("both_answered", "judge", m, "out_per_call_median") for m in MODELS)
    main_ratio = rs[("both_answered", "ratio_out")]

    L = ["## C. Judge 與 Debate 的輸出 tokens", "",
         f"來源：`{AGGDIR}/{{模型}}/{{資料集}}/judge__*.json`、`debate__*.json`（{len(F)} 個檔，逐檔 sha256 在 `C_files.csv`）、`{f_p}`（both_answered 與 path 的輸出 tokens）。"
         "範圍：主網格、兩個候選、自己裁決自己。舊模型的聚合檔不讀內容，只數（見開頭的排除表）。", "",
         "### C.0 紀錄裡的 tokens 是什麼（依使用者的決定）", "",
         "- 主網格的逐題紀錄沒有 usage 欄位。每題的 `tokens_in` / `tokens_out` 是呼叫當下用該模型自己的 tokenizer 重算、該題所有聚合端呼叫的總和"
         "（`Aggregator/Aggregator.call`）；供應商回報的 usage 只有每檔 metadata 的 `api_usage` 總數。",
         f"- 有 api_usage 的 {len(usageF)} 個檔：重算的輸出總和等於 API 的 completion_tokens 有 {out_eq} 個；其餘 {len(usageF) - out_eq} 個"
         f"（全是 Gemini 的 Debate）差 {int(usageF.out_diff_vs_api.abs().max())} tokens 以內。我數的呼叫次數等於 API 的 calls 有 {calls_eq} / {len(usageF)} 個。",
         f"- 重算的輸入每次比 API 的 prompt_tokens 少一個固定的訊息格式開銷（下表，每次呼叫）；所以輸入另列 API 的每次平均（整檔，不限題目）。", "",
         md(overhead.rename(columns={"aggregator": "聚合器", "model": "模型", "mean": "平均", "min": "最小", "max": "最大"})), "",
         f"- 沒有 api_usage 的檔：{len(leg_files)} 個，全是 GPT-4o mini 與 Qwen 的 10 組語言配對 Debate（`source = legacy_import`），依決定只報數量："
         + "；".join(f"{LABEL[m]} {len(leg_files[leg_files.model == m])} 個檔、{int(leg_files[leg_files.model == m].records.sum())} 題紀錄"
                    f"（both_answered 中 {int(legacy[legacy.model == m].n_items.sum())} 題，其中有辯論的 {int(legacy[legacy.model == m].n_called_items.sum())} 題）"
                    for m in WEAK) + "。",
         "- Debate 的紀錄只有每題的總和，所以每次呼叫只能報平均（Σtokens ÷ Σ呼叫數），沒有中位數與四分位。", "",
         "### C.1 重現四個舊數字並標明單位", "",
         md(pd.DataFrame([
             {"舊數字": "Judge「15–22」（結果 5）", "重算": f"{rep['judge'].min():.1f}–{rep['judge'].max():.1f}（{j_rng[0]}–{j_rng[1]}）",
              "狀態": "重現" if j_rng == ("15", "22") else "對不上",
              "單位": "每題平均（分母 = both_answered 全部題目，沒呼叫的題算 0）；平均；範圍是跨模型（每個模型 = 5 個共同配對 × 4 資料集 = 20 格的格平均）；5 個共同配對；tokenizer 重算"},
             {"舊數字": "Judge「105–245」（A 節）", "重算": "見下表；沒有單一來源是 105–245", "狀態": "重現" if hit105 else "對不上",
              "單位": "候選都是每次呼叫、平均、跨 Judge 模型；tokenizer 重算"},
             {"舊數字": "Debate「73–164」（結果 5）", "重算": f"{rep['debate'].min():.1f}–{rep['debate'].max():.1f}（{d_rng[0]}–{d_rng[1]}）",
              "狀態": "重現" if d_rng == ("73", "164") else "對不上",
              "單位": "同「15–22」（每題平均、跨模型、5 個共同配對）；GPT、Qwen 的三組語言配對是舊匯入、沒有 api_usage 的檔"},
             {"舊數字": "Qwen「中位數 100（75 到 146）」（結果 30）", "重算": f"{q_med:.0f}（{q10:.0f} 到 {q90:.0f}），{len(qt)} 次呼叫",
              "狀態": "重現" if q_ok else "對不上",
              "單位": "每次呼叫；中位數；括號是第 10 到第 90 百分位（不是四分位）；只有 Qwen；EN+ZH、EN+S1、P1+P2 × 4 資料集 = 12 格；有呼叫的分歧題（不限 both_answered）"}])), "",
         f"- 「15–22」「73–164」用的 `tok_out_agg` 就是該格 both_answered 題目上 tokens_out 的平均（和逐題紀錄重算最大差 {unit_diff:.1e}）。"
         f"倍數 4–8 = 每個模型的 Debate ÷ Judge：" + "、".join(f"{LABEL[m]} {rep['debate'][m] / rep['judge'][m]:.1f}" for m in MODELS) + "。",
         "- 「105–245」的候選（都是每次呼叫的平均）：", "", md(pd.DataFrame(cand105)), "",
         "  下限 105 對得上 RQ2 §4.1 重跑的 Qwen（105.1），上限 245 對得上 RQ2 §(3) 的 DeepSeek（244.9），沒有任何一個來源同時給出兩端。"
         "依使用者的決定，C 照常算 (1)–(5)。", "",
         "### C.2 (1) Judge 每次呼叫（both_answered；每個模型 = 4 個資料集區塊的平均）", "",
         md(pd.DataFrame(t1)), "",
         "- 「API …/次（整檔）」是每檔 api_usage 的總數 ÷ 呼叫數，範圍是整檔、不限 both_answered：包含一條 path 沒有答案的分歧題"
         "（例如 DeepSeek 在 MathQA 寫到 8,000 tokens 上限的迴圈輸出會整段進到 Judge 的輸入），所以輸入的整檔平均可以遠高於 both_answered 的重算值。",
         f"- 每次呼叫的輸出：平均 {jmin:.0f}–{jmax:.0f}、中位數 {jmedmin:.0f}–{jmedmax:.0f} tokens（跨模型）。逐區塊（模型 × 資料集）在 `C_blocks.csv`，"
         "每個模型的 SE、95% 區間在 `C_models.csv`。", "",
         "### C.3 (3) Debate 聚合端（both_answered；只用有 api_usage 的檔）", "",
         "「聚合端」= 每一輪兩個 agent 各一次呼叫（最多 3 輪）＋ 三輪後仍不一致時的最後裁決一次；不含兩條 path 原本的作答。", "",
         md(pd.DataFrame(t3)), "",
         "### C.3b 逐模型 × 資料集（both_answered）", "", md(pd.DataFrame(tmd)), "",
         "### C.3c (5) 再排除任一輸出 ≥ 8,000 的版本（每個模型 = 4 個區塊的平均）", "", md(pd.DataFrame(tex)), "",
         "### C.4 (4)(5) Debate ÷ Judge 的倍數（5 個共同配對 × 16 區塊，每個區塊 Σ ÷ Σ 再平均）", "",
         "GPT-4o mini 與 Qwen 的 Debate 只有 EN+S1、P1+P2 有 api_usage，所以這 8 個區塊只用這兩個配對（Judge 也只用這兩個）；DeepSeek 與 Gemini 用 5 個。", "",
         md(pd.DataFrame(t4)), "",
         f"- (5) 排除任一輸出 ≥ 8,000 的題目數（both_answered 內）：Judge {ex_n.get('judge', 0)}、Debate {ex_n.get('debate', 0)}（逐區塊在 `C_blocks.csv` 兩種範圍的 n_items 差）。", "",
         "### C.5 讀法（事先寫好的）", "",
         "- 論文統一用「每次呼叫」描述 Judge 的輸出長度，用 (4) 的倍數描述 Debate 對 Judge 的成本。"]
    if 4 <= main_ratio["mean"] <= 8:
        L.append(f"- (4) 的輸出倍數，區塊平均 {main_ratio['mean']:.2f}，落在 4 到 8 之間 → 「4 到 8 倍」可以保留，註明單位（聚合端輸出 tokens，Debate ÷ Judge，每個區塊 Σ ÷ Σ 再平均）。")
        ratio_status = "保留「4 到 8 倍」並註明單位"
    else:
        L.append(f"- (4) 的輸出倍數，區塊平均 {main_ratio['mean']:.2f}，不在 4 到 8 之間 → 用 (4) 的數字取代「4 到 8 倍」：{ci(main_ratio)}。")
        ratio_status = f"改用 {main_ratio['mean']:.2f} 倍"
    L += ["- 結果 5 的其他數字不動。", ""]
    summary("C Judge「15–22」", "15–22", f"{rep['judge'].min():.1f}–{rep['judge'].max():.1f}，每題平均（含沒呼叫的題）",
            "重現" if j_rng == ("15", "22") else "對不上")
    summary("C Judge「105–245」", "105–245", "；".join(f"{r['範圍']}" for r in cand105) + "（每次平均）", "重現" if hit105 else "對不上")
    summary("C Debate「73–164」", "73–164", f"{rep['debate'].min():.1f}–{rep['debate'].max():.1f}，每題平均", "重現" if d_rng == ("73", "164") else "對不上")
    summary("C Qwen 中位數", "100（75 到 146）", f"{q_med:.0f}（{q10:.0f} 到 {q90:.0f}），第 10–90 百分位", "重現" if q_ok else "對不上")
    summary("C Judge 每次呼叫的輸出", "—", f"平均 {jmin:.0f}–{jmax:.0f}、中位數 {jmedmin:.0f}–{jmedmax:.0f}（跨模型）", "更正（統一單位）")
    summary("C Debate ÷ Judge（聚合端輸出）", "4–8 倍", f"{ci(main_ratio)} → {ratio_status}", "更正")
    return L


# ==================================================================
# D. RQ1 中 Debate 的決策結果
# ==================================================================
SETTINGS5 = [("probe_random", 200, "隨機標 200 題"), ("probe_dis", 200, "只標分歧題 200 題"), ("transfer_model", 0, "借用另一個模型的決定"),
             ("transfer_dataset", 0, "借用另一個資料集的決定"), ("transfer_source", 0, "借用另一種多樣性來源的決定")]
GLOBAL4 = [("probe_random", 200, "隨機標 200 題"), ("transfer_model", 0, "換模型"), ("transfer_dataset", 0, "換資料集"),
           ("transfer_source", 0, "換多樣性來源（不同軸）")]


def sectionD() -> list[str]:
    f_s, f_b, f_cr = f"{ANALYSIS}/rq1/summary.csv", f"{ANALYSIS}/rq1/blocks.csv", f"{ANALYSIS}/rq1/rq1_criteria.md"
    s = pd.read_csv(f_s)
    b = dropOld(pd.read_csv(f_b), f_b)

    def row(setting, baseline, agg, k, metric):
        x = s[(s.setting == setting) & (s.baseline == baseline) & (s.aggregator == agg) & (s.k == k) & (s.metric == metric)]
        if len(x) != 1:
            raise SystemExit(f"D：{setting} {baseline} {agg} {k} {metric} 有 {len(x)} 列")
        x = x.iloc[0]
        blk = b[(b.setting == setting) & (b.baseline == baseline) & (b.aggregator == agg) & (b.k == k)][metric]
        if abs(blk.mean() - x["mean"]) > 1e-9 or len(blk) != x.n_blocks:
            raise SystemExit(f"D：blocks.csv 和 summary.csv 不一致（{setting} {metric}）")
        return {"mean": x["mean"], "se": x.se, "ci_low": x.ci_low, "ci_high": x.ci_high, "n_blocks": int(x.n_blocks),
                "n_positive": int((blk > 0).sum())}

    jr = row("probe_random", "pair", "judge", 200, "diff_D_A")
    jspace = min(row("probe_random", "pair", "judge", 200, "regret_A")["mean"], row("probe_random", "pair", "judge", 200, "regret_S")["mean"])
    rep_ok = matches(jr, (-0.24, -0.32, -0.16, None)) and f"{jspace:.2f}" == "0.17"

    def globalBest(agg):
        rows = []
        for setting, k, name in GLOBAL4:
            ra, rs_ = row(setting, "global", agg, k, "regret_A")["mean"], row(setting, "global", agg, k, "regret_S")["mean"]
            better = "S" if rs_ < ra else "A"
            d = row(setting, "global", agg, k, f"diff_D_{better}")
            rows.append({"設定": name, "regret_A": ra, "regret_S": rs_, "較好的固定做法": "永遠用最強單一 path" if better == "S" else "永遠聚合",
                         "決策 − 較好的固定做法": ci(d), "_mean": d["mean"], "_d": d})
        return rows

    jg = globalBest("judge")

    def betterText(rows):
        kinds = {r["較好的固定做法"] for r in rows}
        return (f"四個設定 regret 較小的都是「{kinds.pop()}」" if len(kinds) == 1
                else "較好的固定做法隨設定而變（" + "、".join(f"{r['設定']}：{r['較好的固定做法']}" for r in rows) + "）")
    jbest = max(jg, key=lambda r: r["_mean"])
    rep_ok = rep_ok and matches(jbest["_d"], (0.12, 0.03, 0.20, None))
    source("D", "RQ1 的 Judge 與 Debate（不重算）", [f_s, f_b, f_cr],
           "aggregator ∈ {judge, debate}；baseline ∈ {pair, global}；k = 200（隨機與只抽分歧題）或 0（遷移）",
           "summary.csv 的 mean / ci；為正的區塊數由 blocks.csv 數；空間 = min(regret_A, regret_S)（rq1_criteria.md §5）；"
           "全域基準：每個設定取 regret 較小的固定做法，報決策減它，最多 = 四個設定中最大的")
    L = ["## D. RQ1 中 Debate 的決策結果", "",
         f"來源：`{f_s}`、`{f_b}`（RQ1 的輸出，判定標準 `{f_cr}` sha256 `{sha256(f_cr)[:16]}…`）。", "",
         "### D.1 重現（Judge）", "",
         f"- 隨機標 200 題相對於永遠聚合：{ci(jr)}；目標 −0.24（−0.32 到 −0.16）。",
         f"- 空間 = min(regret_A, regret_S)：{jspace:.3f}；目標 0.17。",
         f"- 全域基準：{betterText(jg)}；決策減它最大的是「{jbest['設定']}」{jbest['決策 − 較好的固定做法']}；目標 0.12（0.03 到 0.20）。",
         f"- 重現：{'通過' if rep_ok else '對不上'}。", ""]
    if not rep_ok:
        summary("D RQ1", "−0.24、0.17", "見 D.1", "對不上")
        return L + ["重現對不上，D 停在這裡。", ""]
    has_debate = bool(((s.aggregator == "debate")).any())
    L += [f"### D.2 Debate（RQ1 既有輸出{'已有' if has_debate else '沒有'} Debate 的列 → {'直接整理，不重算' if has_debate else '要補算'}）", ""]
    t = []
    for setting, k, name in SETTINGS5:
        d = row(setting, "pair", "debate", k, "diff_D_A")
        j = row(setting, "pair", "judge", k, "diff_D_A")
        t.append({"設定": name, "Debate：比「永遠聚合」高多少": ci(d), "（對照）Judge": ci(j)})
    dspace_A = row("probe_random", "pair", "debate", 200, "regret_A")["mean"]
    dspace_S = row("probe_random", "pair", "debate", 200, "regret_S")["mean"]
    dspace = min(dspace_A, dspace_S)
    dg = globalBest("debate")
    dbest = max(dg, key=lambda r: r["_mean"])
    out = b[((b.aggregator == "debate") | ((b.aggregator == "judge") & (b.setting == "probe_random"))) &
            b.setting.isin([x[0] for x in SETTINGS5]) & b.k.isin([0, 200])]
    out[["setting", "baseline", "aggregator", "k", "model", "dataset", "diff_D_A", "diff_D_S", "regret_A", "regret_S"]].to_csv(f"{OUT}/D_blocks.csv", index=False)
    L += ["配對內基準，16 個區塊（和結果 9 的表同一格式）：", "", md(pd.DataFrame(t)), "",
          f"- 空間（隨機標 200 題）：Debate min(regret_A {dspace_A:.3f}, regret_S {dspace_S:.3f}) = **{dspace:.3f}pp**（Judge {jspace:.3f}）。",
          "- 全域基準：", "",
          md(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in dg]).round(3)), "",
          f"  {betterText(dg)}；決策最多再多的是「{dbest['設定']}」{dbest['決策 − 較好的固定做法']}。", "",
          "### D.3 讀法（事先寫好的）", "",
          "- RQ1 的判定只針對 Judge，不因此改變。"]
    if dspace < 0.2:
        L.append(f"- Debate 的空間 {dspace:.2f}pp < 0.2pp → 論文寫「Debate 下同樣沒有空間」，放附錄。")
    else:
        L.append(f"- Debate 的空間 {dspace:.2f}pp ≥ 0.2pp → 只列數字並回報，不寫成 Debate 下有決策價值。")
    L.append("")
    summary("D RQ1 Judge 重現", "−0.24（−0.32 到 −0.16）；空間 0.17；全域最多 +0.12", f"{ci(jr, npos=False)}；{jspace:.2f}；{jbest['決策 − 較好的固定做法']}", "重現")
    summary("D RQ1 Debate（既有輸出，整理）", "—", f"隨機 200 題 {ci(row('probe_random', 'pair', 'debate', 200, 'diff_D_A'), npos=False)}；"
            f"空間 {dspace:.2f}；全域最多 {dbest['決策 − 較好的固定做法']}", "—")
    return L


# ==================================================================
# E. 12 條菜單下最強單一 path 是 persona 的區塊數
# ==================================================================
def sectionE() -> list[str]:
    f_r3 = f"{ANALYSIS}/rq3/rq3_blocks.csv"
    r3 = dropOld(pd.read_csv(f_r3), f_r3)
    r3 = r3[r3.menu.isin(["M12", "M10"])].set_index(["model", "dataset", "menu"])
    m10 = [c for c in M12 if c not in PERSONAS]
    rows, accrows, arm_files = [], [], []
    maxdiff = 0.0
    for block, splits in blocksWithSplits(ARMDIR, AGGDIR, MODELS):
        arm_files += [f"{ARMDIR}/{block.model}/{block.dataset}/{a.replace(':', '_')}.json" for a in PATHS.values()]
        sub = allAnswered(block, M12)
        choice, s_in = Counter(), {"M12": [], "M10": []}
        for h1 in splits:
            in1, in2 = h1 & sub, ~h1 & sub
            for name, codes in (("M12", M12), ("M10", m10)):
                code = strongest({c: int(block.correct[c][in1].sum()) for c in codes})
                s_in[name].append(float(block.correct[code][in2].mean()))
                if name == "M12":
                    choice[code] += 1
        for name in ("M12", "M10"):
            maxdiff = max(maxdiff, abs(np.mean(s_in[name]) - r3.loc[(block.model, block.dataset, name), "S_in"]))
        acc = {c: float(block.correct[c][sub].mean()) for c in M12}
        top = max(acc.values())
        tied = [c for c in M12 if acc[c] == top]
        bp = max(PERSONAS, key=lambda c: (acc[c], -M12.index(c)))
        bn = max(m10, key=lambda c: (acc[c], -M12.index(c)))
        reps = len(splits)
        other_top = max((c for c in choice if c not in PERSONAS), key=lambda c: choice[c], default="—")
        rows.append({"model": block.model, "dataset": block.dataset, "n_subset1": int(sub.sum()), "top_paths": " ".join(tied),
                     "top_acc": 100 * top, "top_is_persona": all(c in PERSONAS for c in tied), "tie_mixed": len({c in PERSONAS for c in tied}) > 1,
                     "best_persona": bp, "best_persona_acc": 100 * acc[bp], "best_nonpersona": bn, "best_nonpersona_acc": 100 * acc[bn],
                     "persona_minus_nonpersona_pp": 100 * (acc[bp] - acc[bn]),
                     "share_P1": choice["P1"] / reps, "share_P2": choice["P2"] / reps,
                     "share_other": sum(v for c, v in choice.items() if c not in PERSONAS) / reps,
                     "most_chosen_other": other_top, "majority_persona": (choice["P1"] + choice["P2"]) / reps > 0.5,
                     "S_in_M12": 100 * np.mean(s_in["M12"]), "S_in_M10": 100 * np.mean(s_in["M10"])})
        for c in M12:
            accrows.append({"model": block.model, "dataset": block.dataset, "path": c, "arm_id": PATHS[c], "acc_subset1": 100 * acc[c],
                            "share_selected_H1": choice[c] / reps})
    E = pd.DataFrame(rows)
    E.to_csv(f"{OUT}/E_blocks.csv", index=False)
    pd.DataFrame(accrows).to_csv(f"{OUT}/E_path_acc.csv", index=False)
    s12, s10 = E.S_in_M12.mean(), E.S_in_M10.mean()
    rep_ok = f"{s12:.2f}" == "84.37" and f"{s10:.2f}" == "83.31" and maxdiff <= 1e-12
    source("E", "子集一的 12 條 path（正確率與切分）", arm_files, "四個新模型；子集一 = M12 的 12 條 path 都有答案",
           "Analysis.menuJudgeStats.blocksWithSplits（makeSplits(n, 200, 0)）；每次切分在 H1 ∩ 子集一用 Analysis.probe.strongest 選最強（平手依 pathKey），在 H2 ∩ 子集一評分")
    source("E", "重現 84.37 / 83.31", [f_r3], "menu ∈ {M12, M10}", "16 個區塊 S_in 的平均（× 100）；逐區塊和重算比")
    L = ["## E. 12 條菜單下最強單一 path 是 persona 的區塊數", "",
         f"來源：`{ARMDIR}`（經 `Analysis.menuVote.loadPathBlock`，與 RQ1-K、RQ3 相同）、`{f_r3}`；子集一（12 條都有答案）；切分 `makeSplits(n, 200, 0)`。", "",
         "### E.1 重現", "",
         f"- 12 條中最強單一 path（用另一半題目選出）的正確率：{s12:.2f}%（目標 84.37%）；拿掉兩條 persona 後 {s10:.2f}%（目標 83.31%）；"
         f"逐區塊和 `rq3_blocks.csv` 最大差 {maxdiff:.1e}。重現：{'通過' if rep_ok else '對不上'}。",
         f"- P1 = `{PATHS['P1']}`（expert），P2 = `{PATHS['P2']}`（skeptic）。", ""]
    if not rep_ok:
        summary("E persona 區塊數", "84.37 / 83.31", f"{s12:.2f} / {s10:.2f}", "對不上")
        return L + ["重現對不上，E 停在這裡。", ""]
    t1 = E.assign(模型=E.model.map(LABEL))[["模型", "dataset", "top_paths", "top_acc", "best_persona", "best_persona_acc", "best_nonpersona",
                                           "best_nonpersona_acc", "persona_minus_nonpersona_pp", "share_P1", "share_P2", "share_other", "most_chosen_other"]]
    t1 = t1.rename(columns={"dataset": "資料集", "top_paths": "全部題目上最高", "top_acc": "最高正確率", "best_persona": "最強 persona",
                            "best_persona_acc": "其正確率", "best_nonpersona": "最強非 persona", "best_nonpersona_acc": "其正確率 ",
                            "persona_minus_nonpersona_pp": "差（pp）", "share_P1": "選到 P1", "share_P2": "選到 P2", "share_other": "選到其他",
                            "most_chosen_other": "其他中最常被選"})
    X = int(E.top_is_persona.sum())
    Xmaj = int(E.majority_persona.sum())
    cnt = []
    for key, col in (("資料集", "dataset"), ("模型", "model")):
        for v, g in E.groupby(col, sort=False):
            cnt.append({"分組": key, "值": LABEL.get(v, v), "區塊數": len(g), "全部題目上最強是 persona": int(g.top_is_persona.sum()),
                        "過半切分選到 persona": int(g.majority_persona.sum())})
    d_all = S(E.persona_minus_nonpersona_pp)
    d_tqa = S(E[E.dataset == "truthfulqa"].persona_minus_nonpersona_pp)
    d_oth = S(E[E.dataset != "truthfulqa"].persona_minus_nonpersona_pp)
    L += ["### E.2 逐區塊（子集一全部題目的正確率，%；選到的比例 = 200 次切分中在 H1 被選為最強的比例）", "",
          md(t1.round(2)), "",
          f"- 全部題目上的並列：{int((E.top_paths.str.count(' ') > 0).sum())} 個區塊有並列"
          + (f"（其中 {int(E.tie_mixed.sum())} 個並列同時含 persona 與非 persona："
             + "、".join(f"{LABEL[r.model]} × {r.dataset} 的 {r.top_paths}（{r.top_acc:.2f}%）" for r in E[E.tie_mixed].itertuples()) + "）"
             if E.tie_mixed.any() else "") + "。"
          "「全部題目上最強是 persona」只在並列的 path 全是 persona 時才算；"
          f"並列含 persona 就算的話是 {int((E.top_is_persona | E.tie_mixed).sum())} / 16。",
          "", "### E.3 區塊數", "", md(pd.DataFrame(cnt)), "",
          f"- 全部題目上最強的是 persona：{X} / 16；過半切分選到 persona：{Xmaj} / 16。", "",
          "### E.4 最強 persona − 最強非 persona（pp，子集一全部題目）", "",
          md(pd.DataFrame([{"範圍": "16 個區塊", "差": ci(d_all)}, {"範圍": "TruthfulQA 4 個區塊（事後依資料集拆分）", "差": ci(d_tqa)},
                           {"範圍": "其他 12 個區塊（事後依資料集拆分）", "差": ci(d_oth)}])), "",
          "### E.5 讀法（事先寫好的）", "",
          f"- 「{X} / 16 個區塊最強的是 persona；領先最強的非 persona path：TruthfulQA {d_tqa['mean']:+.2f} pp，其他三個資料集 {d_oth['mean']:+.2f} pp」。",
          "- 不論 X 多大，都不寫「先找 persona」或「persona 會提高表現」。", ""]
    summary("E 最強單一 path 的正確率", "84.37% / 83.31%", f"{s12:.2f}% / {s10:.2f}%", "重現")
    summary("E 最強是 persona 的區塊數", "—", f"{X}/16（全部題目）；{Xmaj}/16（過半切分）；領先 TruthfulQA {d_tqa['mean']:+.2f}、其他 {d_oth['mean']:+.2f} pp", "—")
    return L


# ==================================================================
# F. 三份判定標準檔：確認版對草稿
# ==================================================================
EXPS = {"RQ3-GSK": ("rq3gsk", "6b36382f", [], r"path_improve_gsk\.py", [f"{OLD_SCRATCH}/gsk_run.log", f"{OLD_SCRATCH}/gsk_run2.log"]),
        "RQ3-GJ": ("rq3gj", "4e96b914", ["rq3gj_criteria_template.md", "rq3gj_criteria_v1.md", "rq3gj_criteria_v2.md"], r"path_improve_gj\.py(?! --dump)", []),
        "RQ3-GJR": ("rq3gjr", "46c3cd12", ["rq3gjr_criteria_template.md", "rq3gjr_criteria_v1.md", "rq3gjr_criteria_v2.md"], r"judge_regression\.py",
                    [f"{OLD_SCRATCH}/gjr_checks.log", f"{OLD_SCRATCH}/gjr_full.log"])}
SHA_RE = re.compile(r"([0-9a-f]{64})\s+(\S+criteria\.md)")


def transcriptEvents():
    """
    session 紀錄裡依序的 (時間, 種類, 內容, 對應的指令)：
      cmd = Bash 指令；write = Write / Edit 的 file_path；result = tool result 的文字（對應的指令另存）；text = 助手訊息的文字。
    """
    events, cmd_of = [], {}
    with open(TRANSCRIPT, encoding="utf-8") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = o.get("message")
            if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
                continue
            for c in msg["content"]:
                kind = c.get("type")
                if kind == "tool_use" and c.get("name") == "Bash":
                    cmd = c["input"].get("command", "")
                    cmd_of[c.get("id")] = cmd
                    events.append((o.get("timestamp"), "cmd", cmd, ""))
                elif kind == "tool_use" and c.get("name") in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
                    cmd_of[c.get("id")] = ""
                    events.append((o.get("timestamp"), "write", c["input"].get("file_path", ""), ""))
                elif kind == "tool_result":
                    cc = c.get("content")
                    text = cc if isinstance(cc, str) else " ".join(x.get("text", "") for x in cc if isinstance(x, dict))
                    events.append((o.get("timestamp"), "result", text, cmd_of.get(c.get("tool_use_id"), "")))
                elif kind == "text" and msg.get("role") == "assistant":
                    events.append((o.get("timestamp"), "text", c.get("text", ""), ""))
    return events


WRITE_RE = re.compile(r"sed -i|>>?\s*\S*criteria\.md|open\([^)]*criteria\.md[^)]*['\"]w|write_text|\.write\(")


def unescapeSed(s: str) -> str:
    return s.replace("\\/", "/").replace("\\*", "*")


def sectionF() -> list[str]:
    events = transcriptEvents()
    os.makedirs(f"{OUT}/F_drafts", exist_ok=True)
    L = ["## F. 三份判定標準檔：確認版對草稿", "",
         f"草稿的來源：確認前的最終草稿沒有存成檔案。依使用者的決定，用寫這三份檔的 session 紀錄 `{TRANSCRIPT}`"
         f"（讀取時 sha256 `{sha256(TRANSCRIPT)}`）裡的確認指令反推：確認指令只用 `sed -i` 改了兩行（狀態、確認），把這兩行改回去，"
         "得到的內容的 sha256 等於該 session 在確認前印出的雜湊，才算找到（標明是重建）。磁碟上較早的草稿只列 sha256。git 歷史裡只有確認版。", ""]
    frows = []
    for name, (folder, recorded, older, run_re, logs) in EXPS.items():
        path = f"{ANALYSIS}/{folder}/{folder}_criteria.md"
        conf_sha = sha256(path)
        text = open(path, encoding="utf-8").read()
        # 確認指令
        fname = f"{folder}/{folder}_criteria.md"
        cmds = [(ts, c) for ts, kind, c, _ in events if kind == "cmd" and "sed -i" in c and fname in c and "已確認" in c]
        if len(cmds) != 1:
            L += [f"### {name}", "", f"確認指令找到 {len(cmds)} 個，無法核對。", ""]
            summary(f"F {name}", recorded, "—", "無法核對")
            continue
        conf_ts, cmd = cmds[0]
        subs = re.findall(r"s/\^(.*?)\$/(.*?)/(?:;|')", cmd)
        # session 印出的這個檔的雜湊，依序：(a) tool result 裡「雜湊 + 路徑」的列；(b) 對應指令提到這個檔且做 sha256sum 或 hashlib.sha256
        # （改草稿的 python 印出寫入內容的雜湊）的 tool result 裡、沒有跟著路徑的雜湊。確認版的雜湊第一次出現之前的最後一個不同的雜湊 = 確認前記錄的雜湊。
        seq = []
        for ts, kind, c, src in events:
            if kind != "result":
                continue
            with_path = SHA_RE.findall(c)
            for h, p in with_path:
                if p.endswith(fname) and not p.startswith("/tmp"):
                    seq.append((ts, h, "tool result 的 sha256sum 輸出（雜湊 + 路徑）"))
            if fname in src and ("sha256sum" in src or "hashlib.sha256" in src):
                for h in re.findall(r"(?m)^(?:\S+[ \t]+)?([0-9a-f]{64})[ \t]*$", c):   # 單獨一行的雜湊（前面可有時間）
                    if h not in {x for x, _ in with_path}:
                        seq.append((ts, h, "tool result（指令寫入或讀取這個檔並印出它的 sha256，輸出沒有路徑）"))
        first_conf = next(i for i, (_, h, _) in enumerate(seq) if h == conf_sha)
        draft_seen = [x for x in seq[:first_conf + 1] if x[1] != conf_sha]
        pre_ts, pre_hash, pre_how = draft_seen[-1]
        told = [ts for ts, kind, c, _ in events if kind == "text" and pre_hash in c and iso(ts) < iso(conf_ts)]
        # 確認前的雜湊印出之後、確認之前，有沒有寫入這個檔的動作
        mods = [(ts, kind, c[:160]) for ts, kind, c, _ in events if iso(pre_ts) < iso(ts) < iso(conf_ts)
                and ((kind == "write" and c.endswith(fname)) or (kind == "cmd" and fname in c and WRITE_RE.search(c)))]
        # 反推
        lines = text.split("\n")
        inv = 0
        for pat, rep in subs:
            pat, rep = unescapeSed(pat), unescapeSed(rep)
            idx = [i for i, ln in enumerate(lines) if ln == rep]
            if len(idx) == 1:
                lines[idx[0]] = pat
                inv += 1
        draft = "\n".join(lines)
        dpath = f"{OUT}/F_drafts/{folder}_criteria_draft_reconstructed.md"
        with open(dpath, "w", encoding="utf-8") as fh:
            fh.write(draft)
        draft_sha = sha256(dpath)
        found = draft_sha == pre_hash and inv == len(subs) == 2 and not mods
        diff = list(difflib.unified_diff(draft.split("\n"), text.split("\n"), fromfile=f"{folder}_criteria_draft_reconstructed.md（草稿，重建）",
                                         tofile=f"{folder}_criteria.md（確認版）", n=0, lineterm=""))
        changed = [ln for ln in diff if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---"))]
        only_conf = all(re.match(r"^[+-]- (狀態|確認)：", ln) for ln in changed)
        # git 歷史
        git = subprocess.run(["git", "log", "--format=%h %ad", "--date=iso", "--", path], capture_output=True, text=True).stdout.strip()
        git_rows = []
        for ln in git.splitlines():
            h = ln.split()[0]
            blob = subprocess.run(["git", "show", f"{h}:{path}"], capture_output=True).stdout
            git_rows.append(f"{ln}（sha256 {hashlib.sha256(blob).hexdigest()[:16]}…，{'= 確認版' if hashlib.sha256(blob).hexdigest() == conf_sha else '≠ 確認版'}）")
        older_rows = []
        for o in older:
            op = f"{OLD_SCRATCH}/{o}"
            if os.path.exists(op):
                older_rows.append({"檔案": op, "sha256": sha256(op), "修改時間": mtime(op)})
        # 確認時間與正式開跑的時間：times = (事件, 時間字串, datetime, 是不是開跑的證據)
        conf_line = re.search(r"^- 確認：(\S+ \S+) CST", text, re.M).group(1)
        times = [("判定標準檔確認（檔內「確認」行）", conf_line + " CST", cst(conf_line + " CST"), False),
                 ("確認指令（session 紀錄）", toCST(conf_ts), iso(conf_ts), False)]
        run_cmds = [(ts, c) for ts, kind, c, _ in events if kind == "cmd" and re.search(run_re, c) and "python" in c and "open(p" not in c
                    and "sed -n" not in c and "grep" not in c.split("python")[0] and "ast.parse" not in c
                    and "--out-dir" not in c and "--allow-unlabeled" not in c and "--dump" not in c and iso(ts) > iso(conf_ts)]
        if run_cmds:
            times.append(("確認後第一次執行分析程式的指令（session 紀錄；不含 dry run 與 --dump）", toCST(run_cmds[0][0]), iso(run_cmds[0][0]), True))
        for lg in logs:
            if os.path.exists(lg):
                times.append((f"log `{lg}` 的修改時間", mtime(lg), datetime.fromtimestamp(os.path.getmtime(lg), CST), True))
        if name == "RQ3-GJ":
            for label, f in (("§10 流程核對 precheck.json 的 generated_at", "precheck.json"), ("§11 試跑 pilot.json 的 generated_at", "pilot.json"),
                             ("§12 開跑前檢查 prerun.json 的 generated_at", "prerun.json")):
                g = json.load(open(f"{ANALYSIS}/rq3gj/{f}")).get("generated_at")
                if g:
                    times.append((label, g, cst(g), True))
            calls = pd.concat([pd.read_csv(f"{ANALYSIS}/rq3gj/judge_outputs/items.csv.gz", usecols=["called_at", "pilot"]),
                               pd.read_csv(f"{ANALYSIS}/rq3gj/judge_outputs_k2/items_k2.csv.gz", usecols=["called_at", "pilot"])])
            for label, sel in (("試跑的 API 呼叫（called_at 最早 – 最晚）", calls.pilot.astype(bool)),
                               ("正式跑的 API 呼叫（called_at 最早 – 最晚）", ~calls.pilot.astype(bool))):
                c = calls[sel].called_at
                times.append((label, f"{toCST(c.min())} – {toCST(c.max())}", iso(c.min()), True))
        rep_path = f"{ANALYSIS}/{folder}/report.md"
        gen = re.search(r"產生(?:時間)?[：:]?\s*(\d{4}-\d\d-\d\d \d\d:\d\d) CST", open(rep_path, encoding="utf-8").read())
        if gen:
            times.append(("report.md 內記的產生時間", gen.group(1) + " CST", cst(gen.group(1) + " CST"), True))
        times.append(("report.md 的修改時間", mtime(rep_path), datetime.fromtimestamp(os.path.getmtime(rep_path), CST), True))
        conf_dt = min(t[2] for t in times if not t[3])
        first_run_dt = min(t[2] for t in times if t[3])
        first_run = first_run_dt.isoformat()
        order = (f"確認在先（確認 {conf_dt.astimezone(CST).strftime('%m-%d %H:%M')}，最早的開跑證據 {first_run_dt.astimezone(CST).strftime('%m-%d %H:%M')} CST）"
                 if first_run_dt > conf_dt else "確認在後")
        times = [(a, b) for a, b, _, _ in times]
        L += [f"### {name}", "",
              f"- 確認版：`{path}`，sha256 `{conf_sha}`；記錄的開頭 `{recorded}` → {'對到確認版' if conf_sha.startswith(recorded) else '對不到'}。",
              f"- 確認前的草稿：{'**找到（重建）**' if found else '**無法核對**'}。重建檔 `{dpath}`，sha256 `{draft_sha}`；"
              f"確認版的雜湊第一次出現之前，session 最後印出的這個檔的雜湊是 `{pre_hash}`（{toCST(pre_ts)}，{pre_how}）→ "
              f"{'相同' if draft_sha == pre_hash else '不同'}。"
              + (f"session 也在 {'、'.join(toCST(t) for t in told)} 的訊息裡把這個雜湊告訴使用者。" if told else "")
              + f"這之後到確認之前寫入這個檔的動作：{'無' if not mods else '；'.join(f'{toCST(t)} {k} {c}' for t, k, c in mods)}。"
              f"確認指令（{toCST(conf_ts)}）：", "", "```bash", cmd.strip(), "```", "",
              "- 逐行 diff（草稿 → 確認版，完整）：", "", "```diff", *diff, "```", "",
              f"- 較早的草稿（磁碟上，只列）：" + ("；".join(f"`{r['檔案']}` sha256 `{r['sha256'][:16]}…`（{r['修改時間']}）" for r in older_rows) or "無") + "。",
              f"- git 歷史：" + ("；".join(git_rows) if git_rows else "沒有提交過（未追蹤）") + "。",
              "- 確認時間與正式開跑的時間：", "", md(pd.DataFrame(times, columns=["事件", "時間"])), "",
              f"  → {order}。", "",
              f"- 讀法：{'只差確認的那兩行（狀態、確認）→ 通過' if found and only_conf else '有其他差異（上面逐行列出），不判斷嚴不嚴重'}。", ""]
        frows.append({"experiment": name, "confirmed_file": path, "confirmed_sha256": conf_sha, "recorded_prefix": recorded,
                      "draft_reconstructed": dpath, "draft_sha256": draft_sha, "pre_confirmation_hash_in_session": pre_hash,
                      "pre_hash_time": pre_ts, "confirm_cmd_time": conf_ts, "found": found, "only_confirmation_lines": only_conf,
                      "first_run_cmd_time": first_run})
        for r in older_rows:
            MANIFEST.append({"group": f"F:{name} 較早的草稿", "path": r["檔案"], "sha256": r["sha256"]})
        source("F", f"{name} 確認版與重建的草稿", [path, dpath, TRANSCRIPT], "session 紀錄：含確認指令與確認前 sha256 的 tool result",
               "把確認指令的兩個 sed 替換反過來；比 sha256；difflib.unified_diff(n=0)")
        summary(f"F {name}", f"{recorded}（確認版）", f"草稿 {pre_hash[:8]}（重建）；diff {'只有確認兩行' if only_conf else '有其他差異'}；{order}",
                "重現（通過）" if found and only_conf else ("無法核對" if not found else "對不上"))
    pd.DataFrame(frows).to_csv(f"{OUT}/F_criteria.csv", index=False)
    return L


# ==================================================================
# G. 兩處文件內對不上的數字
# ==================================================================
def sectionG() -> list[str]:
    from Analysis.pathImproveGSK import THRESHOLDS
    f_cr = f"{ANALYSIS}/rq3gsk/rq3gsk_criteria.md"
    f_rep = f"{ANALYSIS}/rq3gsk/report.md"
    f_b = f"{ANALYSIS}/rq3gsk/rq3gsk_blocks.csv"
    cr = open(f_cr, encoding="utf-8").read().split("\n")
    cr_lines = [f"第 {i + 1} 行：{ln.strip()}" for i, ln in enumerate(cr) if re.search(r"門檻|0\.5 × 3", ln) and re.search(r"0\.20|0\.30|0\.5 × 3", ln)]
    rp = open(f_rep, encoding="utf-8").read().split("\n")
    rp_lines = [f"第 {i + 1} 行：{ln.strip()}" for i, ln in enumerate(rp) if ln.startswith("### 判定") or ln.startswith("- 判定二")]
    gb = dropOld(pd.read_csv(f_b), f_b)
    e7v = gb[(gb.K == 7) & gb.model.isin(WEAK)].E   # 判定只在 8 個弱模型區塊（強模型的列是 NaN）
    if len(e7v) != 8 or e7v.isna().any():
        raise SystemExit("G1：K = 7 的 E 不是 8 個弱模型區塊")
    e7 = S(e7v)
    st020, st0214 = fourState(e7, 0.20), fourState(e7, 0.5 * 3 / 7)
    source("G", "G1 K = 7 的門檻", [f_cr, f_rep, f_b, "Analysis/pathImproveGSK.py"], "rq3gsk_blocks：K = 7、model ∈ {gpt4omini, qwen}（8 個弱模型區塊）",
           "判定標準檔與 report 的原文；THRESHOLDS；E 的跨區塊統計套 fourState(0.20) 與 fourState(0.5 × 3 ÷ 7)")
    L = ["## G. 兩處文件內對不上的數字", "", "### G1. RQ3-GSK 的 K = 7 門檻", "",
         f"- 判定標準檔原文（`{f_cr}`）：", "", *[f"  - {x}" for x in cr_lines], "",
         f"- 程式：`Analysis/pathImproveGSK.py` 的 `THRESHOLDS = {THRESHOLDS}`。report（`{f_rep}`）：", "", *[f"  - {x}" for x in rp_lines], "",
         f"- 判定標準檔寫的是 **0.20**（0.5 × 3 ÷ 7 = 0.214，「取到 0.05」）；report 實際用的也是 0.20。",
         f"- E_7 = {ci(e7)}：門檻 0.20 → {st020}；門檻 0.214 → {st0214}（區間下限 {e7['ci_low']:+.2f}）。",
         "- 讀法：兩個值下判定相同，只統一論文的寫法，以判定標準檔原文為準 → 寫「0.20」。", ""]
    summary("G1 RQ3-GSK K = 7 門檻", "0.20 或 0.214", f"判定標準檔與 report 都是 0.20；兩者判定都是{st020}", "重現")

    # G2
    f_s = f"{ANALYSIS}/rq3gj/rq3gj_substitutions.csv"
    f_m = f"{ANALYSIS}/rq3gj/rq3gj_menus.csv"
    f_jb = f"{ANALYSIS}/rq3gj/rq3gj_blocks.csv"
    f_rb = f"{ANALYSIS}/rq3gjr/rq3gjr_blocks.csv"
    f_gjc, f_gjrc = f"{ANALYSIS}/rq3gj/rq3gj_criteria.md", f"{ANALYSIS}/rq3gjr/rq3gjr_criteria.md"
    s = dropOld(pd.read_csv(f_s), f_s, "host")
    s = s[s.K == 3]
    nblk = s.groupby(["host", "dataset"]).size()
    allm = {"path": 100 * s.den.mean(), "J": 100 * s.numJ.mean(), "V": 100 * s.numV.mean()}
    blk_means = s.groupby(["host", "dataset"])[["den", "numJ", "numV"]].mean().mean() * 100
    rep_all = (f"{allm['path']:.2f}", f"{allm['J']:.2f}", f"{allm['V']:.2f}") == ("10.03", "4.68", "3.04") and s.menu.nunique() == 39 and len(s) == 1872

    def roleTable(df):
        rows, brows = [], []
        for role, g in [("相像那一對（pair）", df[df.role == "pair"]), ("落單那條（lone）", df[df.role == "lone"]), ("全部", df)]:
            per = []
            for (h, d), b in g.groupby(["host", "dataset"]):
                per.append({"role": role, "model": h, "dataset": d, "n_subs": len(b), "path_inc_pp": 100 * b.den.mean(),
                            "judge_more_pp": 100 * b.numJ.mean(), "vote_more_pp": 100 * b.numV.mean(),
                            "judge_effect_per10": 10 * b.numJ.sum() / b.den.sum(), "vote_effect_per10": 10 * b.numV.sum() / b.den.sum()})
            p = pd.DataFrame(per)
            brows.append(p)
            rows.append({"角色": role, "替換數": len(g), **{k: p[k].mean() for k in ("path_inc_pp", "judge_more_pp", "vote_more_pp",
                                                                                   "judge_effect_per10", "vote_effect_per10")}})
        return pd.DataFrame(rows), pd.concat(brows)

    rt, rb = roleTable(s)
    rb.to_csv(f"{OUT}/G2_role_blocks.csv", index=False)
    rt.to_csv(f"{OUT}/G2_roles.csv", index=False)
    pair_r, lone_r, all_r = rt.iloc[0], rt.iloc[1], rt.iloc[2]
    rep_roles = (f"{pair_r.judge_effect_per10:.2f}", f"{pair_r.vote_effect_per10:.2f}", f"{lone_r.judge_effect_per10:.2f}",
                 f"{lone_r.vote_effect_per10:.2f}") == ("5.29", "3.64", "3.55", "2.49")
    rep_roles_pp = (f"{pair_r.judge_more_pp:.2f}", f"{pair_r.vote_more_pp:.2f}", f"{lone_r.judge_more_pp:.2f}",
                    f"{lone_r.vote_more_pp:.2f}") == ("5.29", "3.64", "3.55", "2.49")
    wcheck = []
    for c in ("path_inc_pp", "judge_more_pp", "vote_more_pp", "judge_effect_per10", "vote_effect_per10"):
        w = (pair_r["替換數"] * pair_r[c] + lone_r["替換數"] * lone_r[c]) / all_r["替換數"]
        wcheck.append({"欄": c, "依替換數加權": round(w, 6), "全部": round(all_r[c], 6), "差": f"{w - all_r[c]:.1e}",
                       "相等（1e-9）": abs(w - all_r[c]) <= 1e-9})
    # 效果：每個區塊內，分角色依 Σ（那條 path 進步）加權 = 全部
    den_w = 0.0
    for _, b in s.groupby(["host", "dataset"]):
        for num in ("numJ", "numV"):
            parts = [(g.den.sum(), 10 * g[num].sum() / g.den.sum()) for _, g in b.groupby("role")]
            den_w = max(den_w, abs(sum(w * e for w, e in parts) / sum(w for w, _ in parts) - 10 * b[num].sum() / b.den.sum()))
    # 2.83 / 2.84、0.74 / 0.76
    jb, rbk = dropOld(pd.read_csv(f_jb), f_jb), dropOld(pd.read_csv(f_rb), f_rb)
    menus = pd.read_csv(f_m)
    grp = {g: set(menus[menus.groups.str.contains(g)].menu) for g in ("明顯落單", "英文落單")}
    recomputed = {}
    for g, ms in grp.items():
        vals = []
        for (h, d), b in s[s.menu.isin(ms)].groupby(["host", "dataset"]):
            pr, lo = b[b.role == "pair"], b[b.role == "lone"]
            vals.append(((h, d), 10 * pr.numJ.sum() / pr.den.sum() - 10 * lo.numJ.sum() / lo.den.sum()))
        recomputed[g] = pd.Series(dict(vals))
    jbi = jb.set_index(["model", "dataset"])
    rbi = rbk.set_index(["model", "dataset"])
    ej = {g: S(jb[f"E_J_{g}"]) for g in grp}
    ejh2 = {g: S(jb[f"E_J_H2_act_{g}"]) for g in grp}
    ejr = {g: S(rbk[f"E_J_act_{g}"]) for g in grp}
    d_rec = max(float((recomputed[g] - jbi.loc[recomputed[g].index, f"E_J_{g}"]).abs().max()) for g in grp)
    d_h2 = max(float((jbi[f"E_J_H2_act_{g}"] - rbi.loc[jbi.index, f"E_J_act_{g}"]).abs().max()) for g in grp)
    cr_gj = open(f_gjc, encoding="utf-8").read().split("\n")
    gj_lines = [f"第 {i + 1} 行：{ln.strip()}" for i, ln in enumerate(cr_gj) if "同樣在評分半上算實際的 E_J" in ln or "每個版本 v 在子集二 (h, g, 資料集) 全部題目上的正確率（不切分）" in ln]
    source("G", "G2 全部替換與分角色", [f_s], "K = 3；39 份菜單、1,872 個替換；role ∈ {pair, lone}",
           "多的正確率 = 每個替換 den / numJ / numV（子集二全部題目）× 100 的平均（每區塊 234 個，等於區塊平均再平均）；"
           "效果 = 每個區塊 10 × ΣnumJ ÷ Σden（或 numV），再對 8 個區塊平均")
    source("G", "G2 2.83 / 2.84、0.74 / 0.76", [f_jb, f_rb, f_m, f_gjc, f_gjrc], "8 個弱模型區塊",
           "rq3gj_blocks 的 E_J_{組}（判定一、四，子集二全部題目、不切分）對 E_J_H2_act_{組} = rq3gjr_blocks 的 E_J_act_{組}"
           "（評分半 H2 ∩ 子集二，200 次切分平均）；另由 rq3gj_substitutions 重算 E_J_{組}")
    L += ["### G2. RQ3-GJ 分角色的數字", "",
          f"來源：`{f_s}`（K = 3，{s.menu.nunique()} 份菜單、{len(s)} 個替換；每個區塊 {int(nblk.iloc[0])} 個）。", "",
          f"- 重現全部替換的平均：那條 path 進步 {allm['path']:.2f}pp、裁判多 {allm['J']:.2f}pp、投票多 {allm['V']:.2f}pp（目標 10.03、4.68、3.04）→ "
          f"{'重現' if rep_all else '對不上'}。算法：1,872 個替換的簡單平均（子集二全部題目，不切分）；因為每個區塊都是 234 個，等於「區塊內平均、再對 8 個區塊平均」"
          f"（{blk_means['den']:.2f}、{blk_means['numJ']:.2f}、{blk_means['numV']:.2f}）。",
          f"- 分角色的四個數字（相像那一對 5.29 對 3.64，落單那條 3.55 對 2.49）是 **效果（每 10pp）**，不是「多的正確率」："
          f"每個區塊 10 × Σ（裁判或投票多的正確率）÷ Σ（那條 path 進步），再對 8 個區塊平均 → "
          f"{pair_r.judge_effect_per10:.2f}、{pair_r.vote_effect_per10:.2f}、{lone_r.judge_effect_per10:.2f}、{lone_r.vote_effect_per10:.2f}"
          f"（{'重現' if rep_roles else '對不上'}；若當成多的正確率則是 {pair_r.judge_more_pp:.2f}、{pair_r.vote_more_pp:.2f}、{lone_r.judge_more_pp:.2f}、"
          f"{lone_r.vote_more_pp:.2f}，{'也相同' if rep_roles_pp else '對不上'}）。", ""]
    if not (rep_all and rep_roles):
        summary("G2 RQ3-GJ 分角色", "10.03 / 4.68 / 3.04；5.29 / 3.64；3.55 / 2.49", "見 G2", "對不上")
        return L + ["重現對不上，G2 停在這裡。", ""]
    show = rt.copy()
    show.columns = ["角色", "替換數", "那條 path 進步（pp）", "裁判多（pp）", "投票多（pp）", "裁判效果（每 10pp）", "投票效果（每 10pp）"]
    L += ["同一種算法的表（多的正確率 = 區塊內平均再對 8 個區塊平均；效果 = 區塊內 Σ ÷ Σ × 10 再對 8 個區塊平均）：", "",
          md(show.round(2)), "",
          "依替換數加權後是否等於「全部」：", "", md(pd.DataFrame(wcheck)), "",
          "- 多的正確率三欄依替換數加權後等於「全部」。效果兩欄不等：效果是比值，在每個區塊內要依 Σ（那條 path 進步）加權才等於該區塊的「全部」"
          f"（最大差 {den_w:.1e}）；再對 8 個區塊平均之後，任何一種固定權重都不保證相等。", "",
          "兩組數字各是哪一種算法、哪一批資料（同一批 RQ3-GJ 的裁判紀錄，差在題目範圍）：", "",
          md(pd.DataFrame([
              {"量": "明顯落單組 E_J", "判定一（RQ3-GJ）": ci(ej["明顯落單"]), "regression 報告裡的實際值": ci(ejr["明顯落單"])},
              {"量": "英文落單組 E_J", "判定一（RQ3-GJ）": f"{ci(ej['英文落單'])}（判定四）", "regression 報告裡的實際值": ci(ejr["英文落單"])}])), "",
          f"- 判定一、四的 2.83 / 0.74 = `rq3gj_blocks.csv` 的 `E_J_明顯落單` / `E_J_英文落單`：子集二全部題目、不切分，每個區塊相像那一對的效果 − 落單那條的效果"
          f"（由逐替換的 CSV 重算，最大差 {d_rec:.1e}）。",
          f"- regression 報告的 2.84 / 0.76 = `rq3gjr_blocks.csv` 的 `E_J_act_*` = `rq3gj_blocks.csv` 的 `E_J_H2_act_*`（最大差 {d_h2:.1e}）：在每次切分的評分半 H2 ∩ 子集二上算，"
          f"再對 200 次切分平均（RQ3-GJ §8.12 的「實際值」，和預測放在同一批題目上比）。精確值 {ejh2['明顯落單']['mean']:.4f} 與 {ejh2['英文落單']['mean']:.4f}。",
          "- 判定標準檔的相關原文（RQ3-GJ）：", "", *[f"  - {x}" for x in gj_lines], "",
          "- 讀法：只統一標示，不改任何判定。", ""]
    summary("G2 全部替換的平均", "10.03 / 4.68 / 3.04", f"{allm['path']:.2f} / {allm['J']:.2f} / {allm['V']:.2f}（替換的簡單平均）", "重現")
    summary("G2 分角色的四個數字", "5.29 / 3.64；3.55 / 2.49", "相同；是效果（每 10pp），區塊內 Σ ÷ Σ 再平均", "重現（補標示）")
    summary("G2 明顯落單 2.83 對 2.84；英文落單 0.74 對 0.76", "2.83 / 2.84；0.74 / 0.76",
            f"{ej['明顯落單']['mean']:.4f}（子集二不切分）/ {ejr['明顯落單']['mean']:.4f}（H2 切分平均）；{ej['英文落單']['mean']:.4f} / {ejr['英文落單']['mean']:.4f}", "重現（補標示）")
    return L


# ==================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    started = datetime.now(CST)
    before = hashTree()
    pd.DataFrame(before, columns=["path", "sha256", "bytes"]).to_csv(f"{OUT}/hash_before.csv", index=False)
    print(f"hash_before：{len(before)} 個檔")

    sections = {}
    for key, fn in (("A", sectionA), ("B", sectionB), ("C", sectionC), ("D", sectionD), ("E", sectionE), ("F", sectionF), ("G", sectionG)):
        t0 = datetime.now()
        CURRENT[0] = key
        sections[key] = fn()
        print(f"{key} 完成（{(datetime.now() - t0).total_seconds():.0f}s）")

    pd.DataFrame(SOURCES).to_csv(f"{OUT}/sources.csv", index=False)
    pd.DataFrame(MANIFEST).to_csv(f"{OUT}/source_manifest.csv", index=False)
    pd.DataFrame(EXCLUDED).to_csv(f"{OUT}/excluded_rows.csv", index=False)
    after = hashTree()
    pd.DataFrame(after, columns=["path", "sha256", "bytes"]).to_csv(f"{OUT}/hash_after.csv", index=False)
    same = before == after

    head = ["# 核對七：寫作前的數字清理", "",
            f"- 產生時間：{started.strftime('%Y-%m-%d %H:%M CST')}；程式 `scripts/analysis_check7/check7.py`；離線、不呼叫任何 API；conda 環境 clreasoning。",
            "- 性質：核對，只確認與整理既有數字；沒有判定標準檔，不產生新的判定。既有檔案唯讀，新檔案只寫到 `result/analysis/check7/`。",
            "- 資料夾：先前的核對（核對四、核對六）把報告放在 `result/analysis/rq2/`，沒有專用資料夾的命名，所以這次照規格用 `check7/`。",
            f"- 既有檔案的雜湊：開始前與結束後各算一次（`hash_before.csv`、`hash_after.csv`，result/analysis/ 底下 check7/ 以外的 {len(before)} 個檔）→ "
            f"**{'完全相同' if same else '不同'}**。",
            "- 模型只用 gpt4omini、qwen、deepseek4.1flash、gemini3.1flashlite；各檔排除的舊模型列數在下表（也在 `excluded_rows.csv`）。",
            "- 使用者確認的做法（2026-10-09）：C 用逐題重算的 tokens、只用有 api_usage 的檔，「105–245」對不上時照常算；A 只用參與判定的替換、"
            "「× K 在 0.80–1.00」四捨五入到小數第二位比；F 的草稿用 session 紀錄裡的確認指令反推、雜湊相同才算找到。", "",
            "## 總表", "", md(pd.DataFrame(SUMMARY)), "",
            "## 排除的舊模型列", "", md(pd.DataFrame(EXCLUDED)), ""]
    body = []
    for key in "ABCDEFG":
        body += sections[key]
    files = sorted(p for p in glob.glob(f"{OUT}/**/*", recursive=True) if os.path.isfile(p) and not p.endswith("report.md"))
    tail = ["## check7/ 底下的檔案", "", "（`report.md` 本身無法列入自己的雜湊。）", "",
            md(pd.DataFrame([{"檔案": f, "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()} for f in files])), ""]
    with open(f"{OUT}/report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + body + tail))
    print(f"完成；既有檔案雜湊{'相同' if same else '不同'}")


if __name__ == "__main__":
    main()
