"""
check9.py — 核對九：加權投票的既有數字整理（離線、不呼叫 API、不是新實驗；只讀 RQ3-G、RQ3-GK 既有的輸出檔）

規格：使用者 2026-10-09 貼上的「加權投票：核對九 ＋ RQ3-GSX」的階段一。既有檔案一律唯讀；新檔案只寫到 result/analysis/check9/。
    1A. RQ3-G（12 條，8 個弱模型區塊，子集二；RQ3-G 只有平手規則 A）：先重現判定二、轉換率、替換後多的正確率；
        再把 WV 和 V 並排整理（替換前後、多的正確率、A_real − A_sim、轉換率與相對低估、§8.3 曲線的 +5 / +10）
    1B. RQ3-GK（K = 3、5、7、9、11、12，8 個弱模型區塊，子集二）：先重現替換後 V − SB 與 WV − SB；
        再整理每個 K 的 WV − SB（規則 A、C）、追回的比例、WV 的轉換率（既有輸出有分子才算）
    1C. 事先寫好的兩句讀法：數字符合才寫，否則只列數字
原則：既有輸出沒有的量寫「既有輸出沒有」，不從 result/arms 重算。比例類的量在區塊內先把分子、分母各自加總再相除；
      區塊平均後的數字不再互相乘除。統計單位是區塊（Analysis.blockStats.summarize）。

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_check9/check9.py
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import glob
import hashlib
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import numpy as np
import pandas as pd

from Analysis.blockStats import summarize
from Analysis.pathImprove import M12

MODELS = ["gpt4omini", "qwen", "deepseek4.1flash", "gemini3.1flashlite"]
WEAK = ["gpt4omini", "qwen"]
DATASETS = ["mmlu", "mathqa", "truthfulqa", "commonsenseqa"]
LABEL = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
         "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
ANALYSIS = "result/analysis"
OUT = os.path.join(ANALYSIS, "check9")
CST = timezone(timedelta(hours=8))
KS = [3, 5, 7, 9, 11, 12]
F_G_SUB = f"{ANALYSIS}/rq3g/rq3g_substitutions.csv"
F_G_CELLS = f"{ANALYSIS}/rq3g/rq3g_cells.csv"
F_G_CURVES = f"{ANALYSIS}/rq3g/rq3g_curves.csv"
F_G_REPORT = f"{ANALYSIS}/rq3g/report.md"
F_K_BLOCKS = f"{ANALYSIS}/rq3gk/rq3gk_k_blocks.csv"
F_K_SUB = f"{ANALYSIS}/rq3gk/rq3gk_substitutions.csv"
F_K_MENU = f"{ANALYSIS}/rq3gk/rq3gk_menu_blocks.csv"
F_K_REPORT = f"{ANALYSIS}/rq3gk/report.md"

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
    """result/analysis/ 底下 check9/ 以外的所有檔案（不用快取，每次重算）。"""
    rows = []
    for dirpath, _, names in os.walk(ANALYSIS):
        if os.path.relpath(dirpath, ANALYSIS).split(os.sep)[0] == "check9":
            continue
        for name in names:
            path = os.path.join(dirpath, name)
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            rows.append((path, h.hexdigest(), os.path.getsize(path)))
    return sorted(rows)


SOURCES, EXCLUDED, SUMMARY = [], [], []


def source(item: str, quantity: str, file: str, columns: str, filt: str, rule: str, avg: str):
    """sources.csv 的一列：來源檔、sha256、欄位、篩選、平手規則、哪一種平均。"""
    SOURCES.append({"item": item, "quantity": quantity, "file": file, "sha256": sha256(file), "columns": columns,
                    "filter": filt, "tie_rule": rule, "averaging": avg})


def summary(item: str, old: str, new: str, status: str):
    SUMMARY.append({"項目": item, "既有數字": old, "整理出的數字": new, "狀態": status})


def dropOld(df: pd.DataFrame, file: str, cols: tuple) -> pd.DataFrame:
    """只留 4 個新模型；每個檔每個欄位記一次排除的列數（核對時舊模型一律排除）。"""
    for col in cols:
        old = ~df[col].isin(MODELS)
        EXCLUDED.append({"file": file, "column": col, "rows": len(df), "excluded_rows": int(old.sum()),
                         "excluded_models": ", ".join(sorted(set(df.loc[old, col].astype(str)))) or "—"})
        df = df[~old]
    return df


def S(values) -> dict:
    return summarize(np.asarray(values, dtype=float))


def ci(s: dict, d: int = 2, sign: bool = True, npos: bool = True) -> str:
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    out = f"{f.format(s['mean'])}（{f.format(s['ci_low'])} 到 {f.format(s['ci_high'])}）"
    return out + (f"，{s['n_positive']}/{s['n_blocks']}" if npos else "")


def same(s: dict, target: tuple, d: int = 2) -> bool:
    """target = (mean, ci_low, ci_high, n_positive 或 None)；四捨五入到小數 d 位比。"""
    ok = all(f"{s[k]:.{d}f}" == f"{v:.{d}f}" for k, v in zip(("mean", "ci_low", "ci_high"), target[:3]) if v is not None)
    return ok and (target[3] is None or s["n_positive"] == target[3])


def md(df: pd.DataFrame) -> str:
    return df.astype(object).where(df.notna(), "").astype(str).to_markdown(index=False, disable_numparse=True)


def blockLabel(df: pd.DataFrame, host: str = "host") -> pd.Series:
    return df[host].map(LABEL) + " · " + df["dataset"]


# ==================================================================
# 1A. RQ3-G（12 條）
# ==================================================================
def section1A() -> tuple[list[str], dict]:
    sub = pd.read_csv(F_G_SUB)
    sub = dropOld(sub, F_G_SUB, ("host", "donor"))
    m12 = sub[sub.menu == "M12"]
    part = m12[m12.participates].copy()
    n_all, n_part = len(m12), len(part)
    excl = m12[~m12.participates][["host", "donor", "dataset", "path", "excluded_share"]]

    # 每個替換：有效切分上的平均（rq3g 的逐替換欄位本身就是有效切分的平均；差與和都是線性的）
    part["V_gain"] = part.A_real_V - part.V_before
    part["WV_gain"] = part.WV_after - part.WV_before
    part["SB_gain"] = part.SB_after - part.SB_before
    part["DS_gain"] = part.DS_after - part.DS_before
    part["D2_V"] = part.A_real_V - part.A_sim_V
    part["D2_WV"] = part.WV_after - part.WV_sim
    part["WV_num_real"] = part.WV_after - part.WV_before
    part["WV_num_sim"] = part.WV_sim - part.WV_before

    rows = []
    for (h, d), g in part.groupby(["host", "dataset"], sort=False):
        den = g.conv_den.sum()
        r = {"host": h, "dataset": d, "n_subs_participating": len(g)}
        for agg, (b, a) in {"V": ("V_before", "A_real_V"), "WV": ("WV_before", "WV_after"), "SB": ("SB_before", "SB_after"),
                            "DS": ("DS_before", "DS_after")}.items():
            r[f"{agg}_before_pct"] = 100 * g[b].mean()
            r[f"{agg}_after_pct"] = 100 * g[a].mean()
            r[f"{agg}_gain_pp"] = 100 * g[f"{agg}_gain"].mean()
        r["D2_V_pp"] = 100 * g.D2_V.mean()
        r["D2_V_pp_csv"] = g.D2_pp.mean()
        r["D2_WV_pp"] = 100 * g.D2_WV.mean()
        r["sum_path_inc"] = den
        r["sum_V_num_real"], r["sum_V_num_sim"] = g.conv_num_real.sum(), g.conv_num_sim.sum()
        r["sum_WV_num_real"], r["sum_WV_num_sim"] = g.WV_num_real.sum(), g.WV_num_sim.sum()
        r["conv_V_real"], r["conv_V_sim"] = r["sum_V_num_real"] / den, r["sum_V_num_sim"] / den
        r["conv_WV_real"], r["conv_WV_sim"] = r["sum_WV_num_real"] / den, r["sum_WV_num_sim"] / den
        r["underest_V"] = 1 - r["conv_V_sim"] / r["conv_V_real"]
        r["underest_WV"] = 1 - r["conv_WV_sim"] / r["conv_WV_real"]
        rows.append(r)
    B = pd.DataFrame(rows)
    order = [(h, d) for h in WEAK for d in DATASETS]
    B = B.set_index(["host", "dataset"]).loc[order].reset_index()

    # rq3g_cells.csv 的 ALL 列：逐區塊的 D2 與轉換率（核對我從逐替換算的與既有的區塊值相同）
    cells = pd.read_csv(F_G_CELLS)
    cells = dropOld(cells, F_G_CELLS, ("model",))
    cells = cells[(cells.menu == "M12") & (cells.path == "ALL") & cells.model.isin(WEAK)].set_index(["model", "dataset"])
    cmp = cells.loc[order]
    diff_cells = max(float(np.abs(B.D2_V_pp_csv.values - cmp.D2_pp.values).max()),
                     float(np.abs(B.conv_V_real.values - cmp.conversion_real.values).max()),
                     float(np.abs(B.conv_V_sim.values - cmp.conversion_sim.values).max()))
    B.drop(columns=["D2_V_pp_csv"]).to_csv(f"{OUT}/1A_blocks.csv", index=False)

    st = {k: S(B[k]) for k in ["D2_V_pp", "D2_WV_pp", "conv_V_real", "conv_V_sim", "conv_WV_real", "conv_WV_sim",
                               "underest_V", "underest_WV"] + [f"{a}_{q}" for a in ("V", "WV", "SB", "DS")
                                                                for q in ("before_pct", "after_pct", "gain_pp")]}
    rep = {
        "判定二 D2（V）": (st["D2_V_pp"], (0.23, 0.13, 0.33, 8), 2, "+0.23pp（+0.13 到 +0.33），8 / 8"),
        "轉換率，真實（V）": (st["conv_V_real"], (0.077, 0.065, 0.090, None), 3, "0.077（0.065 到 0.090）"),
        "轉換率，隨機模型（V）": (st["conv_V_sim"], (0.053, 0.044, 0.062, None), 3, "0.053（0.044 到 0.062）"),
        "替換後多的正確率 V": (st["V_gain_pp"], (0.78, None, None, None), 2, "+0.78pp"),
        "替換後多的正確率 WV": (st["WV_gain_pp"], (1.29, None, None, None), 2, "+1.29pp"),
        "替換後多的正確率 DS": (st["DS_gain_pp"], (0.56, None, None, None), 2, "+0.56pp"),
        "替換後多的正確率 SB": (st["SB_gain_pp"], (7.44, None, None, None), 2, "+7.44pp"),
    }
    rep_rows, all_ok = [], True
    for name, (s, target, d, old) in rep.items():
        ok = same(s, target, d)
        all_ok &= ok
        show = ci(s, d, sign=d == 2, npos=target[3] is not None)
        rep_rows.append({"項目": name, "既有數字": old, "重算": show, "結果": "重現" if ok else "對不上"})
        summary(f"1A {name}", old, show, "重現" if ok else "對不上")

    for q, cols in (("替換前後與多的正確率", "V_before, A_real_V, WV_before, WV_after, SB_before, SB_after, DS_before, DS_after"),
                    ("A_real − A_sim", "A_real_V − A_sim_V；WV_after − WV_sim（D2_pp 只用來核對 V）"),
                    ("轉換率（真實、隨機模型）", "分子：conv_num_real、conv_num_sim（V）；WV_after − WV_before、WV_sim − WV_before（WV）；"
                                         "分母：conv_den")):
        source("1A", q, F_G_SUB, cols, "menu = M12；participates = True（190 / 192）；host ∈ {gpt4omini, qwen}",
               "A（RQ3-G 只有原本的平手順序）",
               "替換前後、多的正確率、A_real − A_sim：每個替換的值（已是有效切分的平均）→ 區塊內參與判定的替換等權平均 → 8 個區塊平均"
               if "轉換率" not in q else "區塊內 Σ分子 ÷ Σ分母（參與判定的替換）→ 8 個區塊平均；相對低估 = 1 − 隨機 ÷ 真實，區塊內先算再平均")
    source("1A", "逐區塊核對（D2、V 的轉換率）", F_G_CELLS, "D2_pp、conversion_real、conversion_sim", "menu = M12、path = ALL、弱模型",
           "A", "逐區塊比（不平均）")

    L = ["## 1A. RQ3-G（12 條，8 個弱模型區塊，子集二）", "",
         f"來源：`{F_G_SUB}`（逐替換，M12 的 192 列；參與判定 {n_all - len(excl)} / {n_all}）。RQ3-G 只有一種平手規則"
         "（原本的平手順序 = RQ3-GK 的規則 A），WV 的平手也照這個順序（`rq3g_criteria.md` §3）。逐替換的欄位已經是有效切分上的平均，"
         "所以「替換後 − 替換前」「A_real − A_sim」直接用兩欄相減（線性，和逐切分先減再平均相同）。", "",
         "不參與判定的替換（排除的切分超過 20%，依 RQ3-G §6 不併入）：" +
         "、".join(f"{r.host} × {r.donor} × {r.dataset} × {r.path}（{100 * r.excluded_share:.1f}%）" for r in excl.itertuples()) + "。", "",
         "### 1A.1 先重現", "", md(pd.DataFrame(rep_rows)), "",
         f"- 逐區塊和 `rq3g_cells.csv`（M12、ALL 列的 `D2_pp`、`conversion_real`、`conversion_sim`）的最大差 {diff_cells:.1e}。",
         f"- 判定二、轉換率是 RQ3-G 報告 §(4)、§8.6 的數字；四個「多的正確率」是 §8.4 表的「真實 − 替換前」欄"
         "（每個區塊先對參與判定的替換平均，再對 8 個區塊平均）。", ""]
    if not all_ok:
        L.append("有「先重現」對不上，1A 停在這裡。")
        return L + [""], {}

    def tbl(keys, d=2, pct=False):
        return [ci(st[k], d, sign=not pct, npos=not pct) for k in keys]

    t1 = pd.DataFrame({"聚合": ["V（等權投票）", "WV（依正確率加權）"],
                       "替換前（%）": [ci(st["V_before_pct"], 2, False, False), ci(st["WV_before_pct"], 2, False, False)],
                       "替換後（%）": [ci(st["V_after_pct"], 2, False, False), ci(st["WV_after_pct"], 2, False, False)],
                       "多的正確率（pp）": tbl(["V_gain_pp", "WV_gain_pp"]),
                       "A_real − A_sim（pp）": tbl(["D2_V_pp", "D2_WV_pp"])})
    t2 = pd.DataFrame({"聚合": ["V", "WV"],
                       "轉換率，真實": [ci(st["conv_V_real"], 3, False, False), ci(st["conv_WV_real"], 3, False, False)],
                       "轉換率，隨機模型": [ci(st["conv_V_sim"], 3, False, False), ci(st["conv_WV_sim"], 3, False, False)],
                       "1 − 隨機 ÷ 真實": [ci(st["underest_V"], 3, True, True), ci(st["underest_WV"], 3, True, True)]})
    per = B[["host", "dataset", "V_gain_pp", "WV_gain_pp", "D2_V_pp", "D2_WV_pp", "conv_V_real", "conv_V_sim", "conv_WV_real",
             "conv_WV_sim", "underest_V", "underest_WV"]].copy()
    per.insert(0, "區塊", blockLabel(per))
    per = per.drop(columns=["host", "dataset"])
    for c in per.columns[1:]:
        per[c] = per[c].map(lambda x: f"{x:+.2f}" if c.endswith("_pp") else f"{x:.3f}" if c.startswith("conv") else f"{x:+.3f}")
    per.columns = ["區塊", "V 多的（pp）", "WV 多的（pp）", "V：A_real − A_sim", "WV：A_real − A_sim", "V 轉換率，真實", "V 轉換率，隨機",
                   "WV 轉換率，真實", "WV 轉換率，隨機", "V：1 − 隨機 ÷ 真實", "WV：1 − 隨機 ÷ 真實"]

    # 曲線（§8.3，16 個區塊，子集一）
    cu = pd.read_csv(F_G_CURVES)
    cu = dropOld(cu, F_G_CURVES, ("model",))
    base = cu[cu.target == "+0"].set_index(["model", "dataset", "path"])[["V", "WV", "path_acc_H2"]]
    crow = []
    for t in ("+5", "+10"):
        c = cu[cu.target == t].set_index(["model", "dataset", "path"]).join(base, rsuffix="_0")
        for (mdl, d, p), r in c.iterrows():
            crow.append({"model": mdl, "dataset": d, "path": p, "target": t, "skipped": bool(r.skipped),
                         "insufficient_splits": int(r.insufficient_splits),
                         "path_inc_pp": 100 * (r.path_acc_H2 - r.path_acc_H2_0) if not r.skipped else np.nan,
                         "V_inc_pp": 100 * (r.V - r.V_0) if not r.skipped else np.nan,
                         "WV_inc_pp": 100 * (r.WV - r.WV_0) if not r.skipped else np.nan})
    C = pd.DataFrame(crow)
    C.to_csv(f"{OUT}/1A_curves.csv", index=False)
    cblk = C[~C.skipped].groupby(["target", "model", "dataset"], sort=False)[["V_inc_pp", "WV_inc_pp", "path_inc_pp"]].mean().reset_index()
    cblk["n_paths"] = C[~C.skipped].groupby(["target", "model", "dataset"], sort=False).size().values
    cblk.to_csv(f"{OUT}/1A_curves_blocks.csv", index=False)
    curve_rows = []
    for t in ("+5", "+10"):
        b = cblk[cblk.target == t]
        n_pts = int((~C[C.target == t].skipped).sum())
        curve_rows.append({"目標": t, "算出的點（區塊 × path）": f"{n_pts} / 192", "有點的區塊": f"{len(b)} / 16",
                           "V 的期望增加量（pp）": ci(S(b.V_inc_pp), 2, True, True),
                           "WV 的期望增加量（pp）": ci(S(b.WV_inc_pp), 2, True, True)})
    bypath = (C[~C.skipped].groupby(["target", "path"])[["V_inc_pp", "WV_inc_pp"]].agg(["mean", "count"]))
    path_rows = []
    for p in M12:
        r = {"path": p}
        for t in ("+5", "+10"):
            if (t, p) in bypath.index:
                v = bypath.loc[(t, p)]
                r[f"{t} V"] = f"{v[('V_inc_pp', 'mean')]:.2f}"
                r[f"{t} WV"] = f"{v[('WV_inc_pp', 'mean')]:.2f}"
                r[f"{t} 區塊數"] = int(v[("V_inc_pp", "count")])
            else:
                r[f"{t} V"] = r[f"{t} WV"] = "—"
                r[f"{t} 區塊數"] = 0
        path_rows.append(r)
    missing10 = sorted(set(map(tuple, cu[["model", "dataset"]].drop_duplicates().values)) -
                       set(map(tuple, cblk[cblk.target == "+10"][["model", "dataset"]].values)))
    source("1A", "曲線：+5、+10 時 V 與 WV 的期望增加量", F_G_CURVES, "V、WV、path_acc_H2（target = +0、+5、+10）、skipped",
           "16 個區塊 × 12 條；skipped = False 的點", "A",
           "每個點：(+Δ 的值 − +0 的值)（已是 200 次切分平均）→ 區塊內對算出的 path 平均 → 有點的區塊平均（另列每條 path 對有點的區塊平均）")

    L += ["### 1A.2 整理：WV 和 V 並排（同一種算法）", "",
          "替換前後與多的正確率：每個區塊先對參與判定的替換平均，再對 8 個區塊平均（95% t 區間；多的正確率與 A_real − A_sim 另列為正的區塊）。"
          "A_real − A_sim 是判定二的兩段平均（每個替換的有效切分平均 → 參與判定的替換等權平均）。平手規則 A。", "",
          md(t1), "",
          "轉換率：區塊內參與判定的替換 Σ分子 ÷ Σ分母（RQ3-G §8.6），再對 8 個區塊平均；分母都是「那條 path 增加的正確率」"
          "（`conv_den`）。WV 的分子 = `WV_after − WV_before`（真實）與 `WV_sim − WV_before`（隨機模型；RQ3-G §4 的 WV 規則）。"
          "1 − 隨機 ÷ 真實：每個區塊內先算，再對區塊平均。平手規則 A。", "",
          md(t2), "",
          "逐區塊（`1A_blocks.csv` 有替換前後與 Σ分子、Σ分母）：", "", md(per), "",
          f"- 1 − 隨機 ÷ 真實為負的區塊：WV {int((B.underest_WV < 0).sum())} 個（{', '.join(blockLabel(B[B.underest_WV < 0]).tolist()) or '無'}）、"
          f"V {int((B.underest_V < 0).sum())} 個；為負代表該區塊隨機模型算出的轉換率高於真實。", "",
          "### 1A.3 整理：曲線（§8.3，16 個區塊，子集一，平手規則 A）", "",
          "每個點 = 某區塊的某條 path 提高 Δ 個百分點時，V 或 WV 在評分半的期望正確率減原本（+0），200 次切分的平均。"
          "只要有一次切分的任一半超過 100%，該點就略過（RQ3-G §8.3）。區塊值 = 區塊內算出的 path 的平均。", "",
          md(pd.DataFrame(curve_rows)), "",
          f"- +10 沒有任何點的區塊（每條 path 都會超過 100%）：{'、'.join(f'{LABEL[m]} · {d}' for m, d in missing10)}。"
          "+5 與 +10 算出的點不同，兩列不能直接互比。", "",
          "每條 path（對有點的區塊平均，pp；`1A_curves.csv` 有逐區塊 × path 的值）：", "", md(pd.DataFrame(path_rows)), ""]
    summary("1A WV 替換前 → 替換後", "—", f"{st['WV_before_pct']['mean']:.2f}% → {st['WV_after_pct']['mean']:.2f}%"
            f"（V {st['V_before_pct']['mean']:.2f}% → {st['V_after_pct']['mean']:.2f}%）", "整理")
    summary("1A WV 的 A_real − A_sim", "—", ci(st["D2_WV_pp"]) + f"（V {ci(st['D2_V_pp'])}）", "整理")
    summary("1A WV 的轉換率（真實 / 隨機模型）", "—",
            f"{st['conv_WV_real']['mean']:.3f} / {st['conv_WV_sim']['mean']:.3f}（V {st['conv_V_real']['mean']:.3f} / "
            f"{st['conv_V_sim']['mean']:.3f}）", "整理")
    summary("1A 1 − 隨機 ÷ 真實", "—", f"WV {st['underest_WV']['mean']:.3f}、V {st['underest_V']['mean']:.3f}（區塊內先算再平均）", "整理")
    c5, c10 = cblk[cblk.target == "+5"], cblk[cblk.target == "+10"]
    summary("1A 曲線 +5 / +10 的期望增加量", "—",
            f"+5：V {c5.V_inc_pp.mean():.2f}、WV {c5.WV_inc_pp.mean():.2f}（16 個區塊）；+10：V {c10.V_inc_pp.mean():.2f}、"
            f"WV {c10.WV_inc_pp.mean():.2f}（{len(c10)} 個區塊）", "整理")
    return L, {"underest_V": st["underest_V"]["mean"], "underest_WV": st["underest_WV"]["mean"], "blocks": B}


# ==================================================================
# 1B. RQ3-GK（K = 3 … 12）
# ==================================================================
OLD_VMSB = {3: -4.92, 5: -5.51, 7: -5.67, 9: -5.85, 11: -5.93, 12: -6.00}


def section1B(blocks1A: pd.DataFrame | None) -> tuple[list[str], dict]:
    kb = pd.read_csv(F_K_BLOCKS, dtype={"K": str})
    kb = dropOld(kb, F_K_BLOCKS, ("model",))
    kb = kb[kb.K != "M3"].copy()
    kb["K"] = kb.K.astype(int)
    w = kb[kb.model.isin(WEAK)].copy()
    sub = pd.read_csv(F_K_SUB, dtype={"K": str})
    sub = dropOld(sub, F_K_SUB, ("host", "donor"))
    sub = sub[sub.K != "M3"].copy()
    sub["K"] = sub.K.astype(int)
    part = sub[sub.participates].copy()

    # 先重現
    rep_rows, ok_all = [], True
    st = {}
    for K in KS:
        g = w[w.K == K]
        for col in ("VmSB_after_A_pp", "WVmSB_after_A_pp", "VmSB_after_C_pp", "WVmSB_after_C_pp", "E1_pp"):
            st[(K, col)] = S(g[col])
        if float(np.abs(g.E1_pp.values - g.VmSB_after_A_pp.values).max()) > 1e-12:
            raise SystemExit("E1_pp ≠ VmSB_after_A_pp")
        s = st[(K, "VmSB_after_A_pp")]
        ok = f"{s['mean']:.2f}" == f"{OLD_VMSB[K]:.2f}"
        ok_all &= ok
        rep_rows.append({"項目": f"替換後 V − SB（規則 A），K = {K}", "既有數字": f"{OLD_VMSB[K]:+.2f}", "重算": ci(s),
                         "結果": "重現" if ok else "對不上"})
    for K, target, old in ((3, (-2.42, -3.95, -0.88, None), "−2.42（−3.95 到 −0.88）"), (12, (-5.31, None, None, None), "−5.31")):
        s = st[(K, "WVmSB_after_A_pp")]
        ok = same(s, target)
        ok_all &= ok
        rep_rows.append({"項目": f"替換後 WV − SB（規則 A），K = {K}", "既有數字": old, "重算": ci(s), "結果": "重現" if ok else "對不上"})
    for r in rep_rows:
        summary(f"1B {r['項目']}", r["既有數字"], r["重算"], r["結果"])
    source("1B", "替換後 V − SB、WV − SB（規則 A、C）", F_K_BLOCKS,
           "VmSB_after_A_pp、WVmSB_after_A_pp、VmSB_after_C_pp、WVmSB_after_C_pp（E1_pp 核對 = VmSB_after_A_pp）",
           "K ≠ M3；model ∈ {gpt4omini, qwen}", "A 與 C 分欄（SB 不受平手規則影響）",
           "既有的區塊值（每個替換的有效切分平均 → 菜單內參與判定的替換平均 → 該 K 的菜單平均，RQ3-GK §5）→ 8 個區塊平均")

    L = ["## 1B. RQ3-GK（K = 3、5、7、9、11、12，8 個弱模型區塊，子集二）", "",
         f"來源：`{F_K_BLOCKS}`（逐 K × 區塊的既有區塊值）、`{F_K_SUB}`（逐替換，算追回的比例）、`{F_G_SUB}`（K = 12 的 WV 轉換率）。"
         "替換後 V − SB、WV − SB 的區塊值是 RQ3-GK §5 的三段平均（每個替換 → 每份菜單 → 該 K 的菜單平均）。", "",
         "### 1B.1 先重現", "", md(pd.DataFrame(rep_rows)), "",
         "- 既有數字是 RQ3-GK 報告 §9.1 / §9.7 的規則 A 欄（K = 3 的 −4.92 是判定一）。", ""]
    if not ok_all:
        L.append("有「先重現」對不上，1B 停在這裡。")
        return L + [""], {}

    # 追回的比例：區塊內 Σ(WV − V) ÷ Σ(SB − V)，範圍 = 該 K 全部菜單裡參與判定的替換（同 RQ3-GK §5 轉換率的範圍）
    rec_rows = []
    for (K, h, d), g in part.groupby(["K", "host", "dataset"]):
        for rule in "AC":
            num = (g[f"WVmSB_after_{rule}_pp"] - g[f"VmSB_after_{rule}_pp"]).sum()     # Σ (WV_after − V_after)，pp
            den = (-g[f"VmSB_after_{rule}_pp"]).sum()                                 # Σ (SB_after − V_after)，pp
            rec_rows.append({"K": K, "rule": rule, "host": h, "dataset": d, "n_subs": len(g), "sum_WV_minus_V_pp": num,
                             "sum_SB_minus_V_pp": den, "recovered_share": num / den})
    R = pd.DataFrame(rec_rows)
    kbw = w.set_index(["K", "model", "dataset"])
    out_blocks = []
    for (K, h, d), r in kbw.iterrows():
        row = {"K": K, "host": h, "dataset": d, "n_menus": int(r.n_menus_used) if "n_menus_used" in kbw.columns else np.nan}
        for rule in "AC":
            row[f"VmSB_after_{rule}_pp"] = r[f"VmSB_after_{rule}_pp"]
            row[f"WVmSB_after_{rule}_pp"] = r[f"WVmSB_after_{rule}_pp"]
            row[f"WV_minus_V_after_{rule}_pp"] = r[f"WVmSB_after_{rule}_pp"] - r[f"VmSB_after_{rule}_pp"]
            x = R[(R.K == K) & (R.rule == rule) & (R.host == h) & (R.dataset == d)].iloc[0]
            row[f"sum_WV_minus_V_{rule}_pp"], row[f"sum_SB_minus_V_{rule}_pp"] = x.sum_WV_minus_V_pp, x.sum_SB_minus_V_pp
            row[f"recovered_share_{rule}"] = x.recovered_share
        out_blocks.append(row)
    OB = pd.DataFrame(out_blocks)
    order = {(h, d): i for i, (h, d) in enumerate((h, d) for h in WEAK for d in DATASETS)}
    OB["_o"] = [order[(h, d)] for h, d in zip(OB.host, OB.dataset)]
    OB = OB.sort_values(["K", "_o"]).drop(columns="_o").reset_index(drop=True)
    OB.to_csv(f"{OUT}/1B_blocks.csv", index=False)
    source("1B", "追回的比例", F_K_SUB, "WVmSB_after_{A,C}_pp、VmSB_after_{A,C}_pp",
           "K ≠ M3；participates = True；host ∈ {gpt4omini, qwen}", "A 與 C 分開",
           "區塊 × K 內 Σ(WV − V 替換後) ÷ Σ(SB − V 替換後)（每個替換的值是有效切分平均；範圍 = 該 K 全部菜單的參與判定替換）"
           "→ 8 個區塊平均；另報 8 個值的中位數")
    source("1B", "替換後 WV − V", F_K_BLOCKS, "WVmSB_after_{A,C}_pp − VmSB_after_{A,C}_pp", "K ≠ M3；弱模型", "A 與 C 分開",
           "區塊內相減（三段平均的區塊值）→ 8 個區塊平均（差，不是比例）")

    # WV 的轉換率：既有輸出有沒有 WV 的分子（替換後 WV − 替換前 WV）？
    has_wv_before = {f: [c for c in pd.read_csv(f, nrows=0).columns if c.startswith("WV_before")] for f in (F_K_SUB, F_K_BLOCKS, F_K_MENU)}
    wv_conv = {}
    if blocks1A is not None and len(blocks1A):
        g12 = pd.read_csv(F_G_SUB)
        g12 = g12[g12.menu == "M12"].set_index(["host", "donor", "dataset", "path"])
        k12 = sub[sub.K == 12].set_index(["host", "donor", "dataset", "path"]).loc[g12.index]
        d12 = float(np.abs(g12.WV_after.values - k12.WV_after_A.values).max())
        wv_conv[12] = (S(blocks1A.conv_WV_real), S(blocks1A.conv_V_real), d12)
        source("1B", "K = 12 的 WV 轉換率", F_G_SUB, "WV_after − WV_before（分子）、conv_den（分母）",
               "menu = M12；participates = True", "A", "同 1A（區塊內 Σ ÷ Σ → 8 個區塊平均）")

    tab = []
    for K in KS:
        g = w[w.K == K]
        r = {"K": K}
        r["V − SB（A）"] = ci(st[(K, "VmSB_after_A_pp")])
        r["WV − SB（A）"] = ci(st[(K, "WVmSB_after_A_pp")])
        r["WV − SB（C）"] = ci(st[(K, "WVmSB_after_C_pp")])
        r["WV − V（A）"] = ci(S(g.WVmSB_after_A_pp - g.VmSB_after_A_pp))
        rr = OB[OB.K == K]
        r["追回的比例（A）：平均（區間）"] = ci(S(rr.recovered_share_A), 2, False, False)
        r["中位數"] = f"{rr.recovered_share_A.median():.2f}"
        tab.append(r)
    per_rec = OB.pivot_table(index=["host", "dataset"], columns="K", values="recovered_share_A", sort=False).reset_index()
    per_rec.insert(0, "區塊", blockLabel(per_rec))
    per_rec = per_rec.drop(columns=["host", "dataset"])
    per_rec.columns = ["區塊"] + [f"K = {K}" for K in per_rec.columns[1:]]
    per_rec = per_rec.set_index("區塊").loc[[f"{LABEL[h]} · {d}" for h in WEAK for d in DATASETS]].reset_index()
    for c in per_rec.columns[1:]:
        per_rec[c] = per_rec[c].map(lambda x: f"{x:.2f}")
    per_vsb = OB.pivot_table(index=["host", "dataset"], columns="K", values="VmSB_after_A_pp", sort=False).reset_index()
    per_vsb.insert(0, "區塊", blockLabel(per_vsb))
    per_vsb = per_vsb.drop(columns=["host", "dataset"])
    per_vsb.columns = ["區塊"] + [f"K = {K}" for K in per_vsb.columns[1:]]
    per_vsb = per_vsb.set_index("區塊").loc[[f"{LABEL[h]} · {d}" for h in WEAK for d in DATASETS]].reset_index()
    for c in per_vsb.columns[1:]:
        per_vsb[c] = per_vsb[c].map(lambda x: f"{x:+.2f}")

    conv_rows = []
    for K in KS:
        s_real = S(w[w.K == K].conversion_real_A)
        if K in wv_conv:
            conv_rows.append({"K": K, "V 的轉換率，真實（A）": ci(s_real, 3, False, False),
                              "WV 的轉換率，真實（A）": ci(wv_conv[K][0], 3, False, False) + "（來源 `rq3g_substitutions.csv`）"})
        else:
            conv_rows.append({"K": K, "V 的轉換率，真實（A）": ci(s_real, 3, False, False), "WV 的轉換率，真實（A）": "既有輸出沒有"})
    source("1B", "V 的轉換率（對照欄）", F_K_BLOCKS, "conversion_real_A", "K ≠ M3；弱模型", "A",
           "既有的區塊值（該 K 全部參與判定替換的 Σ ÷ Σ）→ 8 個區塊平均")

    L += ["### 1B.2 整理：每個 K 替換後的 WV − SB", "",
          "平均（95% t 區間），幾個區塊為正。V − SB、WV − SB、WV − V 是 RQ3-GK §5 三段平均的區塊值再對 8 個區塊平均"
          "（WV − V 是同一區塊的兩欄相減，是差不是比例）。WV 在規則 C 下的平分照 RQ3-GK §4。", "",
          "追回的比例 = 換進來的強 path 讓 WV 比 V 多追回「V 與 SB 的差距」的幾成："
          "每個區塊內，該 K 全部參與判定的替換 Σ(替換後 WV − 替換後 V) ÷ Σ(替換後 SB − 替換後 V)，再對 8 個區塊平均（規則 A）。"
          "依共同規則 5，不用「區塊平均後的 WV − SB ÷ V − SB」。", "",
          md(pd.DataFrame(tab)), "",
          f"規則 C 與規則 A 的 WV − SB 逐區塊（6 個 K × 8 個區塊）最大差 {float(np.abs(w.WVmSB_after_A_pp - w.WVmSB_after_C_pp).max()):.1e}pp；"
          "V − SB 規則 C 的值與規則 C 的追回比例在 `1B_blocks.csv`。", "",
          "逐區塊的追回比例（規則 A）：", "", md(per_rec), "",
          "逐區塊的替換後 V − SB（規則 A，pp；追回比例的分母）：", "", md(per_vsb), "",
          "- 兩個 CommonsenseQA 區塊的 V − SB 接近 0（供體只領先約 1pp），分母小，比例在 K ≥ 5 時超過 1 或變成負的（K = 11、12 時"
          "Qwen3-8B · commonsenseqa 的 V − SB 為正）。8 個區塊的平均因此不隨 K 單調；中位數另列。", "",
          "### 1B.3 整理：WV 的轉換率", "",
          "既有輸出有沒有「替換後 WV 多的正確率」（替換前的 WV）：`rq3gk_substitutions.csv`、`rq3gk_k_blocks.csv`、`rq3gk_menu_blocks.csv` "
          f"都只有替換後的 WV（`WV_after_A`、`WV_after_C`），沒有替換前的 WV（找到的 WV_before 欄：{sum(len(v) for v in has_wv_before.values())} 個）。"
          "K = 12 的 M12 和 RQ3-G 是同一批替換，RQ3-G 的逐替換有 `WV_before`，所以 K = 12 用 1A 的值"
          + (f"（兩邊替換後 WV 逐替換最大差 {wv_conv[12][2]:.1e}）" if 12 in wv_conv else "") + "。", "",
          md(pd.DataFrame(conv_rows)), ""]
    for K in KS:
        summary(f"1B 替換後 WV − SB，K = {K}", "—" if K not in (3, 12) else f"{-2.42 if K == 3 else -5.31:+.2f}",
                f"A {ci(st[(K, 'WVmSB_after_A_pp')])}；C {ci(st[(K, 'WVmSB_after_C_pp')])}", "整理")
    summary("1B 追回的比例（規則 A，區塊內 Σ ÷ Σ 再平均）", "「K = 3 追回一半」",
            "、".join(f"K={K} {OB[OB.K == K].recovered_share_A.mean():.2f}" for K in KS) + "（中位數 "
            + "、".join(f"{OB[OB.K == K].recovered_share_A.median():.2f}" for K in KS) + "）", "整理")
    summary("1B WV 的轉換率", "—", "K = 12：" + (f"{wv_conv[12][0]['mean']:.3f}（RQ3-G）" if 12 in wv_conv else "—")
            + "；K = 3、5、7、9、11：既有輸出沒有", "既有輸出沒有（K = 3–11）")
    return L, {"OB": OB, "st": st}


# ==================================================================
# 1C. 讀法
# ==================================================================
def section1C(r1A: dict, r1B: dict) -> list[str]:
    L = ["## 1C. 讀法（事先寫好；只描述，不下判定）", ""]
    if r1A:
        x, y = 10 * r1A["underest_WV"], 10 * r1A["underest_V"]
        L += [f"- 「12 條時，加權投票的隨機改進模型低估真實替換約 {x:.1f} 成（V 是 {y:.1f} 成）。」",
              f"  - X、Y = 1 − 隨機模型的轉換率 ÷ 真實的轉換率，每個區塊內先算再對 8 個區塊平均（{r1A['underest_WV']:.3f}、"
              f"{r1A['underest_V']:.3f}），平手規則 A。WV 有一個區塊為負，見 1A.2。", ""]
    else:
        L += ["- 第一句：1A 的先重現對不上，不寫。", ""]
    if not r1B:
        return L + ["- 第二句：1B 的先重現對不上，不寫。", ""]
    OB, st = r1B["OB"], r1B["st"]
    rec = [OB[OB.K == K].recovered_share_A.mean() for K in KS]
    half = round(rec[0], 1) == 0.5
    mono = all(a > b for a, b in zip(rec, rec[1:]))
    wv = "WVmSB_after_A_pp"
    highs = [st[(K, wv)]["ci_high"] for K in KS]
    below = all(h < 0 for h in highs)
    L += ["- 第二句「換進一條強的 path 後，加權投票在 K = 3 追回約一半的差距，K 越大追回越少；每個 K 都仍低於直接用最強的那條。」",
          "  我怎麼對照（三個條件都符合才寫這一句）：",
          f"  - 「約一半」：K = 3 的追回比例（1B.2，規則 A）四捨五入到小數一位是 0.5 → {rec[0]:.3f}，{'符合' if half else '不符合'}。",
          f"  - 「K 越大追回越少」：追回比例的 8 個區塊平均隨 K = 3、5、7、9、11、12 嚴格遞減 → "
          f"{'、'.join(f'{v:.2f}' for v in rec)}，{'符合' if mono else '不符合'}。",
          f"  - 「每個 K 都仍低於直接用最強的那條」：每個 K 的替換後 WV − SB（規則 A）95% 區間上緣 < 0 → "
          f"{'、'.join(f'{h:+.2f}' for h in highs)}，{'符合' if below else '不符合'}。"]
    if half and mono and below:
        L += ["", "  → 三個條件都符合，寫這一句。", ""]
    else:
        L += ["", "  → 有條件不符合，**不寫這一句，只列數字**：",
              "    - 替換後 WV − SB（規則 A）：" + "、".join(f"K = {K} {st[(K, 'WVmSB_after_A_pp')]['mean']:+.2f}" for K in KS) + "；",
              "    - 替換後 V − SB（規則 A）：" + "、".join(f"K = {K} {st[(K, 'VmSB_after_A_pp')]['mean']:+.2f}" for K in KS) + "；",
              "    - 追回的比例（規則 A，區塊內 Σ ÷ Σ 再平均；中位數）：" + "、".join(
                  f"K = {K} {OB[OB.K == K].recovered_share_A.mean():.2f}（{OB[OB.K == K].recovered_share_A.median():.2f}）" for K in KS) + "。",
              "  - 不符合的原因在 1B.2：兩個 CommonsenseQA 區塊的分母（V − SB）接近 0，比例不穩定。", ""]
    return L


def main():
    os.makedirs(OUT, exist_ok=True)
    started = datetime.now(CST)
    before = hashTree()
    pd.DataFrame(before, columns=["path", "sha256", "bytes"]).to_csv(f"{OUT}/hash_before.csv", index=False)
    print(f"hash_before：{len(before)} 個檔")

    L1A, r1A = section1A()
    L1B, r1B = section1B(r1A.get("blocks") if r1A else None)
    L1C = section1C(r1A, r1B)
    for f, what in ((F_G_REPORT, "既有數字的出處（RQ3-G 報告 §(4)、§8.4、§8.6）"), (F_K_REPORT, "既有數字的出處（RQ3-GK 報告 §9.1、§9.7）")):
        source("出處", what, f, "—", "—", "—", "只讀，不取數字（數字都從 CSV 算）")

    pd.DataFrame(SOURCES).to_csv(f"{OUT}/sources.csv", index=False)
    pd.DataFrame(EXCLUDED).to_csv(f"{OUT}/excluded_rows.csv", index=False)
    after = hashTree()
    pd.DataFrame(after, columns=["path", "sha256", "bytes"]).to_csv(f"{OUT}/hash_after.csv", index=False)
    same_hash = before == after

    head = ["# 核對九：加權投票的既有數字整理", "",
            f"- 產生時間：{started.strftime('%Y-%m-%d %H:%M CST')}；程式 `scripts/analysis_check9/check9.py`；離線、不呼叫任何 API；conda 環境 clreasoning。",
            "- 性質：核對，只讀 RQ3-G、RQ3-GK 既有的輸出檔（CSV）；沒有判定標準檔，不產生新的判定。沒有從 `result/arms` 重算任何量。"
            "既有檔案唯讀，新檔案只寫到 `result/analysis/check9/`。",
            f"- 既有檔案的雜湊：開始前與結束後各算一次（`hash_before.csv`、`hash_after.csv`，result/analysis/ 底下 check9/ 以外的 "
            f"{len(before)} 個檔）→ **{'完全相同' if same_hash else '不同'}**。",
            "- 模型只用 gpt4omini、qwen、deepseek4.1flash、gemini3.1flashlite；讀到的 CSV 都沒有舊模型的列（`excluded_rows.csv`）。",
            "- 每個數字的來源檔、欄位、平手規則與平均方式在各節的說明與 `sources.csv`。統計單位是區塊：平均（95% t 區間），幾個區塊為正。", "",
            "## 總表", "", md(pd.DataFrame(SUMMARY)), ""]
    files = sorted(p for p in glob.glob(f"{OUT}/**/*", recursive=True) if os.path.isfile(p) and not p.endswith("report.md"))
    tail = ["## 來源檔", "", md(pd.DataFrame(SOURCES)[["item", "quantity", "file", "sha256"]].drop_duplicates()), "",
            "## check9/ 底下的檔案", "", "（`report.md` 本身無法列入自己的雜湊。）", "",
            md(pd.DataFrame([{"檔案": f, "sha256": sha256(f)} for f in files])), ""]
    with open(f"{OUT}/report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + L1A + L1B + L1C + tail))
    print(f"完成；既有檔案雜湊{'相同' if same_hash else '不同'}")


if __name__ == "__main__":
    main()
