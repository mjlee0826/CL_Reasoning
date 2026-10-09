"""
check8.py — 核對八：Debate 舊匯入資料的一致性，以及判定標準檔的審查版本（離線、不呼叫 API、不是新實驗）

規格：使用者 2026-10-09 貼上的「核對八」。既有檔案一律唯讀；新檔案只寫到 result/analysis/check8/；不做 git commit。
    第一部分（D1–D5）：gpt4omini、qwen 的 80 個舊匯入 Debate 檔（source = legacy_import）和新跑的 16 + 96 個 Debate 檔是不是同一套做法
    第二部分（F1–F4）：RQ3-GSK、RQ3-GJ、RQ3-GJR 判定標準檔的各版時間線、改動對指示、審查雜湊（F3 留白 → 跳過）、草稿複本

使用者確認的做法（2026-10-09）：
    - 舊匯入檔本身沒有對話全文（trace 全是 null）；D2–D4 用 metadata 的 legacy_path 指向的原始紀錄 result/challenge/*.json，
      報告每一項標明「匯入來源」並附 sha256，同時寫明匯入檔本身沒有這些欄位。
    - F1 的重播：把 session 紀錄裡改草稿的程式在沙盒裡重新執行（寫入只准落在沙盒；datetime.now() 固定成紀錄裡印出的時間），
      每一步的雜湊和當時印出的比對。
    - F3 的審查雜湊留白 → 照規格跳過，三個實驗都寫「無法比對」，F1 的表放在報告最前面。
    - 「全文有沒有顯示給使用者」兩欄分開：(a) 助手訊息、(b) 工具輸出（Bash / Read）；全文 = 這一版所有非空行都出現在同一則；
      另報最高涵蓋率；雜湊用前 8 碼比對。
我在計畫裡寫明、使用者沒有異議的預設：
    - D2 抽樣：每個模型 × {舊匯入, 新跑} 各用新的 default_rng(0) 從有辯論的題（依檔名、item_id 排序）抽 20 題；
      非英文的固定文字和 DeepSeek、Gemini 新跑的同語言 Debate 比；最後裁決的 prompt 兩邊都沒有存 → 沒有紀錄。
    - D1：沒有欄位就寫「沒有紀錄」；對話紀錄裡看得到的（每輪幾次呼叫、最後裁決何時觸發）另列一欄，不拿來代替。
    - D4 的 off-menu、沒有答案、進到最後裁決的比例，分母是有辯論的題。
    - F2：「第一版草稿」= 模板填完後第一次寫出的完整檔；模板 → 填完只列在 F1。

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_check8/check8.py              # 全部
    conda run -n clreasoning python scripts/analysis_check8/check8.py --dump-hunks # 只重播並印出 F2 的改動區塊（標記用）
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import difflib
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import numpy as np
import pandas as pd

from Aggregator.DebateAggregator import DebateAggregator
from Analysis.blockStats import summarize
from Analysis.experimentPlan import STEM_TO_ARM, ARMS_TO_PAIR
from Arm.ArmSpec import ArmSpec
from Arm.GenerationRecord import GenerationRecord
from Dataset.DatasetType import DatasetType, get_dataset_map

MODELS = ["gpt4omini", "qwen", "deepseek4.1flash", "gemini3.1flashlite"]
WEAK, STRONG = ["gpt4omini", "qwen"], ["deepseek4.1flash", "gemini3.1flashlite"]
DATASETS = ["mmlu", "mathqa", "truthfulqa", "commonsenseqa"]
LABEL = {"gpt4omini": "GPT-4o mini", "qwen": "Qwen3-8B", "deepseek4.1flash": "DeepSeek V4.1 Flash",
         "gemini3.1flashlite": "Gemini 3.1 Flash-Lite"}
KIND_LABEL = {"legacy": "舊匯入", "new_weak": "新跑（GPT、Qwen）", "new_strong": "新跑（DeepSeek、Gemini）"}
ANALYSIS, AGGDIR = "result/analysis", "result/aggregations"
OUT = os.path.join(ANALYSIS, "check8")
COMMON5 = ["EN+ZH", "EN+JA", "ZH+JA", "EN+S1", "P1+P2"]
LANG_CODES = {"EN", "ZH", "JA", "RU", "ES"}
CST = timezone(timedelta(hours=8))
SESSION = "dd2d53a9-8914-49b4-8899-941dc0267bd5"
TRANSCRIPT = os.path.expanduser(f"~/.claude/projects/-home-mjlee-Desktop-cl-reasoning/{SESSION}.jsonl")
OLD_S = f"/tmp/claude-1000/-home-mjlee-Desktop-cl-reasoning/{SESSION}/scratchpad"
SANDBOX = "/tmp/claude-1000/-home-mjlee-Desktop-cl-reasoning/a97efb19-f59b-4f96-9885-e0eb77c10fbd/scratchpad/c8_sandbox"
CHECK7_TRANSCRIPT_SHA = "6121a5e566a06b918f45b46b731997c9a505655b6d2fce102b779463aea64c3c"

# ------------------------------------------------------------------
# 共用
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


def shaText(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hashTree() -> list[tuple]:
    rows = []
    for dirpath, _, names in os.walk(ANALYSIS):
        if os.path.relpath(dirpath, ANALYSIS).split(os.sep)[0] == "check8":
            continue
        for name in names:
            path = os.path.join(dirpath, name)
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            rows.append((path, h.hexdigest(), os.path.getsize(path)))
    return sorted(rows)


SOURCES, MANIFEST, SUMMARY = [], [], []


def source(item: str, quantity: str, files: list[str], filt: str, algo: str):
    if len(files) > 6:
        group = f"{item}:{quantity}"
        for f in files:
            MANIFEST.append({"group": group, "path": f, "sha256": sha256(f)})
        fcol, scol = f"見 source_manifest.csv 群組「{group}」（{len(files)} 個檔案）", ""
    else:
        fcol, scol = ";".join(files), ";".join(sha256(f) for f in files)
    SOURCES.append({"item": item, "quantity": quantity, "files": fcol, "sha256": scol, "filter": filt, "algorithm": algo})


def S(values) -> dict:
    return summarize(np.asarray(values, dtype=float))


def ci(s: dict, d: int = 2, npos: bool = True) -> str:
    f = f"{{:+.{d}f}}"
    out = f"{f.format(s['mean'])}（{f.format(s['ci_low'])} 到 {f.format(s['ci_high'])}）"
    return out + (f"，{s['n_positive']}/{s['n_blocks']}" if npos else "")


def md(df: pd.DataFrame) -> str:
    return df.astype(object).where(df.notna(), "").astype(str).to_markdown(index=False, disable_numparse=True)


def toCST(iso: str) -> str:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(CST).strftime("%Y-%m-%d %H:%M:%S")


def readJson(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ==================================================================
# 第一部分：資料
# ==================================================================
def pairOf(path: str):
    agg, sa, sb = os.path.basename(path)[:-5].split("__")
    a, b = STEM_TO_ARM[sa], STEM_TO_ARM[sb]
    return ARMS_TO_PAIR[frozenset((a, b))].label, a, b


def loadDebateFiles() -> list[dict]:
    files = []
    for model in MODELS:
        for ds in DATASETS:
            for path in sorted(glob.glob(f"{AGGDIR}/{model}/{ds}/debate__*.json")):
                data = readJson(path)
                meta, recs = data[0], data[1:]
                pair, a, b = pairOf(path)
                legacy = meta.get("source") == "legacy_import"
                kind = "legacy" if legacy else ("new_weak" if model in WEAK else "new_strong")
                files.append({"file": path, "model": model, "dataset": ds, "pair": pair, "arm_a": a, "arm_b": b, "kind": kind,
                              "meta": meta, "recs": recs, "by_id": {r["item_id"]: r for r in recs}})
    return files


_ARMS, _SRC = {}, {}


def arm(model: str, ds: str, arm_id: str) -> dict:
    key = (model, ds, arm_id)
    if key not in _ARMS:
        path = f"result/arms/{model}/{ds}/{ArmSpec.from_arm_id(arm_id).file_stem}.json"
        data = readJson(path)
        _ARMS[key] = (path, {r["item_id"]: r for r in data[1:]})
    return _ARMS[key][1]


def legacySource(path: str) -> tuple[dict, dict]:
    if path not in _SRC:
        data = readJson(path)
        _SRC[path] = (data[0], {r["id"]: r for r in data[1:]})
    return _SRC[path]


def compareFn(ds: str):
    return get_dataset_map()[DatasetType(ds)].compareTwoAnswer


def debated(rec: dict) -> bool:
    return (rec.get("n_rounds") or 0) > 0


def flatKeys(d: dict, prefix: str = ""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatKeys(v, prefix + k + ".")
        else:
            yield prefix + k


def lang(arm_id: str) -> str:
    return ArmSpec.from_arm_id(arm_id).language


def dialogue(f: dict, item_id: int):
    """
    回傳該題的辯論對話（不含兩條 path 原本的作答）：每個 agent 的 [(user prompt, assistant 輸出)] 每輪一組，
    起始答案、起始推理文字、最後裁決的輸出、來源。舊匯入用 result/challenge 的原始紀錄；新跑用 trace。
    """
    if f["kind"] == "legacy":
        _, src = legacySource(f["meta"]["legacy_path"])
        r = src[item_id]
        turns = []
        for rec in (r["Record1"], r["Record2"]):
            turns.append([(rec[i]["content"], rec[i + 1]["content"]) for i in range(2, len(rec) - 1, 2)])
        return {"turns": turns, "start_answers": [r["AnswerRecord1"][0], r["AnswerRecord2"][0]],
                "start_texts": [r["Record1"][1]["content"], r["Record2"][1]["content"]],
                "answers": [r["AnswerRecord1"], r["AnswerRecord2"]], "result3": r.get("Result3") or "", "rounds": r.get("Times")}
    rec = f["by_id"][item_id]
    t = rec["trace"]
    turns = [[(R[i]["content"], R[i + 1]["content"]) for i in range(0, len(R) - 1, 2)] for R in (t["Record1"], t["Record2"])]
    return {"turns": turns, "start_answers": [t["AnswerRecord1"][0], t["AnswerRecord2"][0]], "start_texts": None,
            "answers": [t["AnswerRecord1"], t["AnswerRecord2"]], "result3": t.get("Result3") or "", "rounds": rec["n_rounds"]}


def opponents(f: dict, item_id: int, d: dict) -> list[list[str]]:
    """每個 agent 每一輪 prompt 裡應該出現的對手前一次輸出。"""
    if d["start_texts"] is not None:
        starts = d["start_texts"]
    else:
        starts = [arm(f["model"], f["dataset"], f["arm_a"])[item_id]["raw_text"], arm(f["model"], f["dataset"], f["arm_b"])[item_id]["raw_text"]]
    outs = [[starts[k]] + [o for _, o in d["turns"][k]] for k in (0, 1)]
    return [[outs[1][r] for r in range(len(d["turns"][0]))], [outs[0][r] for r in range(len(d["turns"][1]))]]


# ==================================================================
# 第一部分：重現
# ==================================================================
def aggMinusEn(P: pd.DataFrame, A: pd.DataFrame, agg: str, pair: str) -> pd.Series:
    """核對七 B 的算法：每個區塊在該配對自己的 both_answered 題目上，聚合正確率 − L:en 正確率（pp）。"""
    other = pair.split("+")[1]
    out = {}
    for model in MODELS:
        for ds in DATASETS:
            a = A[(A.model == model) & (A.dataset == ds) & (A.aggregator == agg) & (A.pair == pair)]
            p = P[(P.model == model) & (P.dataset == ds)]
            ans = p.pivot(index="item_id", columns="path", values="answered").astype(bool)
            cor = p.pivot(index="item_id", columns="path", values="correct").astype(float)
            cj = a.set_index("item_id").correct.astype(float).reindex(ans.index)
            own = ans["EN"] & ans[other]
            out[(model, ds)] = 100 * (cj[own].mean() - cor.loc[own, "EN"].mean())
    return pd.Series(out)


def result5(cells: pd.DataFrame, include) -> tuple[dict, dict, pd.DataFrame]:
    """結果 5 的算法：每個區塊對納入的配對平均 (Debate − Judge) 的 recovery_H2 與正確率（pp），再對區塊平均。include(model, pair) → bool。"""
    x = cells[(cells.subset == "both_answered") & cells.pair.isin(COMMON5)]
    x = x[[include(m, p) for m, p in zip(x.model, x.pair)]]
    pv = x.pivot_table(index=["model", "dataset", "pair"], columns="aggregator", values=["recovery_H2", "acc_final"])
    rec = (pv[("recovery_H2", "debate")] - pv[("recovery_H2", "judge")]).groupby(level=[0, 1]).mean()
    acc = (100 * (pv[("acc_final", "debate")] - pv[("acc_final", "judge")])).groupby(level=[0, 1]).mean()
    npairs = pv.groupby(level=[0, 1]).size()
    blocks = pd.DataFrame({"recovery_diff": rec, "acc_diff_pp": acc, "n_pairs": npairs}).reset_index()
    return S(rec.values), S(acc.values), blocks


# ==================================================================
# 第一部分
# ==================================================================
def part1() -> tuple[list[str], str]:
    files = loadDebateFiles()
    leg = [f for f in files if f["kind"] == "legacy"]
    neww = [f for f in files if f["kind"] == "new_weak"]
    news = [f for f in files if f["kind"] == "new_strong"]
    L = ["# 第一部分：Debate 舊匯入資料和新跑的是不是同一套做法", "",
         f"對象：舊匯入 {len(leg)} 個檔（gpt4omini、qwen 的 10 組語言配對，`source = legacy_import`）；新跑 {len(neww)} 個（gpt4omini、qwen 的 EN+S1、P1+P2）"
         f"與 {len(news)} 個（deepseek4.1flash、gemini3.1flashlite 的 12 組）。舊模型資料夾（deepseek V3.2）不讀內容；Gemini 2.5 不在 `result/aggregations/`。", ""]
    MANIFEST.extend({"group": "D:Debate 檔", "path": f["file"], "sha256": sha256(f["file"])} for f in files)
    src_paths = sorted({f["meta"]["legacy_path"] for f in leg})
    MANIFEST.extend({"group": "D:匯入來源 result/challenge", "path": p, "sha256": sha256(p)} for p in src_paths)

    # ---------- 先重現 ----------
    n_files = {m: sum(1 for f in leg if f["model"] == m) for m in WEAK}
    n_recs = {m: sum(len(f["recs"]) for f in leg if f["model"] == m) for m in WEAK}
    rep0 = len(leg) == 80 and all(n_files[m] == 40 and n_recs[m] == 68170 for m in WEAK)
    f_c = f"{ANALYSIS}/aggregation_cells.csv"
    cells = pd.read_csv(f_c)
    n_old_cells = int((~cells.model.isin(MODELS)).sum())
    cells = cells[cells.model.isin(MODELS)]
    r5_rec, r5_acc, r5_blocks = result5(cells, lambda m, p: True)
    rep5 = (f"{r5_rec['mean']:.3f}", f"{r5_rec['ci_low']:.2f}", f"{r5_rec['ci_high']:.2f}", r5_rec["n_positive"]) == ("0.065", "0.02", "0.11", 13) \
        and (f"{r5_acc['mean']:.2f}", f"{r5_acc['ci_low']:.2f}", f"{r5_acc['ci_high']:.2f}") == ("0.28", "-0.01", "0.58")
    f_p, f_a = f"{ANALYSIS}/items/paths.csv.gz", f"{ANALYSIS}/items/aggregations.csv.gz"
    P = pd.read_csv(f_p)
    A = pd.read_csv(f_a)
    n_old_p, n_old_a = int((~P.model.isin(MODELS)).sum()), int((~A.model.isin(MODELS)).sum())
    P, A = P[P.model.isin(MODELS)], A[A.model.isin(MODELS)]
    s1, zh = aggMinusEn(P, A, "debate", "EN+S1"), aggMinusEn(P, A, "debate", "EN+ZH")
    r6 = {"EN+S1": S(s1.values), "EN+ZH": S(zh.values), "D": S((s1 - zh).values)}

    def m2(s, t):
        return (f"{s['mean']:.2f}", f"{s['ci_low']:.2f}", f"{s['ci_high']:.2f}", s["n_positive"]) == t

    rep6 = m2(r6["EN+S1"], ("0.92", "0.50", "1.34", 16)) and m2(r6["EN+ZH"], ("1.00", "0.04", "1.96", 14)) and m2(r6["D"], ("-0.08", "-0.86", "0.70", 8))
    source("D", "先重現：舊匯入檔數與題數", [f["file"] for f in leg], "gpt4omini、qwen；debate__*；metadata source = legacy_import", "數檔數與紀錄數")
    source("D", "先重現：結果 5", [f_c], f"subset = both_answered；pair ∈ 5 個共同配對；四個新模型（排除舊模型 {n_old_cells} 列）",
           "每個區塊對 5 個配對平均 (Debate − Judge) 的 recovery_H2 與 acc_final × 100，再對 16 個區塊平均")
    source("D", "先重現：結果 6 的 Debate", [f_p, f_a], f"四個新模型（排除舊模型 paths {n_old_p} 列、aggregations {n_old_a} 列）；各自的 both_answered",
           "核對七 B 的算法：每個區塊 Debate − L:en（pp）；D = EN+S1 − EN+ZH")
    L += ["## 先重現", "",
          md(pd.DataFrame([
              {"項目": "舊匯入檔數；每個模型檔數與題數", "目標": "80；各 40 個檔、各 68,170 題",
               "重算": f"{len(leg)}；" + "、".join(f"{LABEL[m]} {n_files[m]} 個檔、{n_recs[m]:,} 題" for m in WEAK), "相同": rep0},
              {"項目": "結果 5：Debate − Judge 的 recovery", "目標": "0.065（0.02 到 0.11），13 個區塊同向", "重算": ci(r5_rec, 3), "相同": rep5},
              {"項目": "結果 5：換算成正確率（pp）", "目標": "+0.28（−0.01 到 +0.58）", "重算": ci(r5_acc), "相同": rep5},
              {"項目": "結果 6 Debate：EN+S1 − 英文", "目標": "+0.92（+0.50 到 +1.34），16/16", "重算": ci(r6["EN+S1"]), "相同": rep6},
              {"項目": "結果 6 Debate：EN+ZH − 英文", "目標": "+1.00（+0.04 到 +1.96），14/16", "重算": ci(r6["EN+ZH"]), "相同": rep6},
              {"項目": "結果 6 Debate：兩者相減", "目標": "−0.08（−0.86 到 +0.70），8/16", "重算": ci(r6["D"]), "相同": rep6}])), "",
          "- 結果 5 的算法（找出來的）：`aggregation_cells.csv`，both_answered，5 個共同配對；每個區塊對 5 個配對平均（Debate − Judge）的 `recovery_H2`，"
          "以及 `acc_final` 之差 × 100，再對 16 個區塊平均。用不切分的 `recovery` 是 +0.064，不是 0.065。", ""]
    if not (rep0 and rep5 and rep6):
        return L + ["先重現對不上，第一部分停在這裡。", ""], "停（重現對不上）"

    verdict_flags = {}

    # ---------- D1 ----------
    rows_fields = []
    groups = {"舊匯入檔（80）": leg, "新跑（112）": neww + news}
    src_meta_keys, src_rec_keys = Counter(), Counter()
    for p in src_paths:
        m, recs = legacySource(p)
        src_meta_keys.update(set(flatKeys(m)))
        for r in recs.values():
            src_rec_keys.update(r.keys())
    inv = {}
    for name, fs in groups.items():
        mk, rk, tk = Counter(), Counter(), Counter()
        for f in fs:
            mk.update(set(flatKeys(f["meta"])))
            for r in f["recs"]:
                rk.update(r.keys())
                if isinstance(r.get("trace"), dict):
                    tk.update(r["trace"].keys())
        inv[name] = {"檔層級": (mk, len(fs)), "逐題層級": (rk, sum(len(f["recs"]) for f in fs)), "trace 內": (tk, sum(len(f["recs"]) for f in fs))}
    inv["匯入來源 result/challenge（80）"] = {"檔層級": (src_meta_keys, len(src_paths)), "逐題層級": (src_rec_keys, sum(len(legacySource(p)[1]) for p in src_paths)),
                                          "trace 內": (Counter(), 0)}
    for level in ("檔層級", "逐題層級", "trace 內"):
        keys = sorted(set().union(*[set(inv[n][level][0]) for n in inv]))
        for k in keys:
            row = {"層級": level, "欄位": k}
            for n in inv:
                c, tot = inv[n][level]
                row[n] = f"{c[k]}/{tot}" if c[k] else "—"
            row["舊匯入檔與新跑不同"] = (row["舊匯入檔（80）"] == "—") != (row["新跑（112）"] == "—")
            rows_fields.append(row)
    F1 = pd.DataFrame(rows_fields)
    F1.to_csv(f"{OUT}/D1_fields.csv", index=False)
    diff_fields = F1[F1["舊匯入檔與新跑不同"]]

    meta_rows = []
    for f in files:
        m = f["meta"]
        deb = [r for r in f["recs"] if debated(r)]
        rounds_obs = Counter()
        per_round_ok, n_dial, final_rule = 0, 0, Counter()
        cmp = compareFn(f["dataset"])
        for r in deb:
            d = dialogue(f, r["item_id"])
            n_dial += 1
            rounds_obs[d["rounds"]] += 1
            per_round_ok += int(len(d["turns"][0]) == d["rounds"] == len(d["turns"][1]))
            still = not cmp(d["answers"][0][-1], d["answers"][1][-1])
            final_rule[(bool(d["result3"]), still, d["rounds"] == 3)] += 1
        row = {"file": f["file"], "sha256": sha256(f["file"]), "model": f["model"], "dataset": f["dataset"], "pair": f["pair"],
               "kind": KIND_LABEL[f["kind"]], "source": m.get("source"),
               "legacy_path": m.get("legacy_path", ""), "legacy_path_sha256": sha256(m["legacy_path"]) if m.get("legacy_path") else "",
               "prompt_version": (repr(m["prompt_version"]) if "prompt_version" in m else "沒有紀錄"),
               "model_api_name": m["Model"].get("modelName", "沒有紀錄"), "model_version_string": "沒有紀錄",
               "temperature": m["Model"].get("temperature", "沒有紀錄"), "max_tokens": "沒有紀錄", "thinking": "沒有紀錄",
               "rounds_limit（Aggregator.threshold）": m.get("Aggregator", {}).get("threshold", "沒有紀錄"),
               "calls_per_round": "沒有紀錄", "final_trigger": "沒有紀錄", "call_dates": "沒有紀錄", "import_date": "沒有紀錄",
               "code_version": "沒有紀錄", "parse_rule": "沒有紀錄",
               "legacy_checks": json.dumps(m.get("legacy_checks"), ensure_ascii=False) if m.get("legacy_checks") else "",
               "api_usage_calls": m.get("api_usage", {}).get("calls", ""),
               "對話紀錄：有辯論的題": n_dial, "對話紀錄：輪數分布": " ".join(f"{k}輪×{v}" for k, v in sorted(rounds_obs.items())),
               "對話紀錄：兩個 agent 每輪各一次呼叫的題": per_round_ok,
               "對話紀錄：有最後裁決 = 三輪後仍不一致": sum(v for (r3, still, three), v in final_rule.items() if r3 == (still and three)),
               "對話紀錄：最大輪數": max(rounds_obs) if rounds_obs else ""}
        if f["kind"] == "legacy":
            sm, _ = legacySource(m["legacy_path"])
            row["匯入來源：Model.modelName"] = sm.get("Model", {}).get("modelName", "沒有紀錄")
            row["匯入來源：Model.temperature"] = sm.get("Model", {}).get("temperature", "沒有紀錄")
        meta_rows.append(row)
    M1 = pd.DataFrame(meta_rows)
    M1.to_csv(f"{OUT}/D1_files.csv", index=False)

    comp = []
    items = [("prompt 的版本或名稱", "prompt_version"), ("模型的 API 代號", "model_api_name"), ("模型的版本字串", "model_version_string"),
             ("temperature", "temperature"), ("max_tokens", "max_tokens"), ("thinking 設定", "thinking"),
             ("辯論的輪數上限", "rounds_limit（Aggregator.threshold）"), ("每輪幾次呼叫", "calls_per_round"), ("最後裁決的觸發條件", "final_trigger"),
             ("呼叫日期", "call_dates"), ("匯入日期", "import_date"), ("程式版本或 commit", "code_version"), ("答案的解析規則或版本", "parse_rule")]
    d1_diff = []
    for model in WEAK:
        a = M1[(M1.model == model) & (M1.kind == KIND_LABEL["legacy"])]
        b = M1[(M1.model == model) & (M1.kind == KIND_LABEL["new_weak"])]
        for name, col in items:
            va, vb = sorted(set(map(str, a[col]))), sorted(set(map(str, b[col])))
            if "沒有紀錄" in va or "沒有紀錄" in vb:
                st = "沒有紀錄"
            else:
                st = "相同" if va == vb and len(va) == 1 else "不同"
            if st == "不同":
                d1_diff.append(f"{LABEL[model]} {name}")
            comp.append({"模型": LABEL[model], "項目": name, "舊匯入": " / ".join(va), "新跑": " / ".join(vb), "比較": st})
    C1 = pd.DataFrame(comp)
    C1.to_csv(f"{OUT}/D1_compare.csv", index=False)
    obs = M1.groupby("kind").agg(檔數=("file", "size"), 有辯論的題=("對話紀錄：有辯論的題", "sum"),
                                 每輪兩次呼叫=("對話紀錄：兩個 agent 每輪各一次呼叫的題", "sum"),
                                 最後裁決規則相符=("對話紀錄：有最後裁決 = 三輪後仍不一致", "sum"), 最大輪數=("對話紀錄：最大輪數", "max")).reset_index()
    source("D1", "欄位與 metadata", [f"{OUT}/D1_files.csv"], "全部 192 個 Debate 檔；舊匯入另讀 legacy_path 的匯入來源",
           "欄位清單；每檔的 metadata 值；沒有欄位寫「沒有紀錄」；對話紀錄看得到的另列")
    verdict_flags["D1_different"] = d1_diff
    L += ["## D1. 檔案結構與 metadata", "",
          "兩邊欄位不同的地方（「舊匯入檔」= `result/aggregations` 裡的 80 個匯入檔；「匯入來源」= 它們 `legacy_path` 指向的 `result/challenge` 原始紀錄；完整清單在 `D1_fields.csv`）：", "",
          md(diff_fields), "",
          "- 舊匯入檔的逐題 `trace` 全是 null：匯入檔本身沒有對話全文、起始答案、最後裁決的輸出；這些只在匯入來源裡。",
          "- 兩邊都沒有的欄位：模型的版本字串（arm 檔才有）、max_tokens、thinking、呼叫日期、匯入日期、程式版本或 commit、解析規則。",
          "", "逐項比較（每個模型分開；有一邊沒有紀錄就寫「沒有紀錄」；逐檔的值在 `D1_files.csv`）：", "", md(C1), "",
          "對話紀錄裡看得到的（不是欄位，只描述，不拿來代替上表）：", "", md(obs), "",
          "- 「每輪兩次呼叫」= 兩個 agent 的對話裡每輪各有一組 user / assistant；「最後裁決規則相符」= 有最後裁決的輸出，剛好等於「三輪後兩個 agent 的答案仍不同」。", ""]

    # ---------- D2 ----------
    def sample(fs, model, n=20):
        pool = sorted(((f["file"], r["item_id"], f) for f in fs if f["model"] == model for r in f["recs"] if debated(r)), key=lambda x: (x[0], x[1]))
        rng = np.random.default_rng(0)
        idx = sorted(rng.choice(len(pool), min(n, len(pool)), replace=False))
        return [pool[i] for i in idx]

    fixed = {}   # (kind_group, language) -> Counter of fixed texts
    rebuild_rows, d2_rows = [], []
    samples = {}
    for model in MODELS:
        kinds = [("legacy", leg), ("new", neww if model in WEAK else news)] if model in WEAK else [("new", news)]
        for kname, fs in kinds:
            sm = sample(fs, model)
            samples[(model, kname)] = sm
            n_same = 0
            for fpath, iid, f in sm:
                d = dialogue(f, iid)
                opp = opponents(f, iid, d)
                langs = [lang(f["arm_a"]), lang(f["arm_b"])]
                ok_all, n_turns, n_found = True, 0, 0
                for k in (0, 1):
                    for r, (prompt, _) in enumerate(d["turns"][k]):
                        n_turns += 1
                        expected = DebateAggregator.getDebatePrompt(langs[k], opp[k][r])
                        ok = prompt == expected
                        ok_all &= ok
                        if opp[k][r] in prompt:
                            n_found += 1
                            pre, suf = prompt.split(opp[k][r], 1)
                            fixed.setdefault(("舊匯入" if kname == "legacy" else "新跑", langs[k]), Counter())[(pre, suf)] += 1
                        d2_rows.append({"model": model, "kind": kname, "file": fpath, "item_id": iid, "agent": k + 1, "round": r + 1,
                                        "language": langs[k], "opponent_text_in_prompt": opp[k][r] in prompt, "equals_current_template": ok})
                n_same += ok_all
                rebuild_rows.append({"model": model, "kind": kname, "file": fpath, "item_id": iid, "turns": n_turns, "all_turns_equal_template": ok_all})
    D2 = pd.DataFrame(d2_rows)
    D2.to_csv(f"{OUT}/D2_turns.csv", index=False)
    RB = pd.DataFrame(rebuild_rows)
    RB.to_csv(f"{OUT}/D2_items.csv", index=False)
    fx_rows, fx_diffs = [], []
    for language in sorted({k[1] for k in fixed}):
        lg, nw = fixed.get(("舊匯入", language), Counter()), fixed.get(("新跑", language), Counter())
        same = set(lg) == set(nw) if lg and nw else None
        fx_rows.append({"語言": language, "舊匯入：不同的固定文字數": len(lg), "舊匯入：prompt 數": sum(lg.values()),
                        "新跑：不同的固定文字數": len(nw), "新跑：prompt 數": sum(nw.values()),
                        "逐字相同": ("沒有可比的" if same is None else ("相同" if same else "不同"))})
        if same is False:
            a_txt = "\n=====\n".join(p + "{對手的輸出}" + s for p, s in sorted(lg))
            b_txt = "\n=====\n".join(p + "{對手的輸出}" + s for p, s in sorted(nw))
            fx_diffs.append((language, list(difflib.unified_diff(a_txt.split("\n"), b_txt.split("\n"), "舊匯入", "新跑", n=1, lineterm=""))))
    FX = pd.DataFrame(fx_rows)
    FX.to_csv(f"{OUT}/D2_fixed_text.csv", index=False)
    rb_tab = RB.groupby(["model", "kind"]).agg(題數=("item_id", "size"), 和目前樣板完全相同的題=("all_turns_equal_template", "sum"),
                                               prompt_數=("turns", "sum")).reset_index()
    rb_tab["model"] = rb_tab.model.map(LABEL)
    rb_tab["kind"] = rb_tab.kind.map({"legacy": "舊匯入（匯入來源）", "new": "新跑（對照）"})
    d2_same_legacy = bool(RB[RB.kind == "legacy"].all_turns_equal_template.all())
    d2_fixed_same = all(r["逐字相同"] in ("相同", "沒有可比的") for r in fx_rows)
    verdict_flags["D2"] = d2_same_legacy and d2_fixed_same
    source("D2", "prompt 全文", [f"{OUT}/D2_turns.csv", f"{OUT}/D2_items.csv"],
           "每個模型 × {舊匯入, 新跑} 各 default_rng(0) 抽 20 題有辯論的題；舊匯入讀匯入來源 result/challenge 的 Record1 / Record2",
           "每一輪的 user prompt：切掉對手前一次的輸出得到固定文字，依 agent 的語言比；另用 DebateAggregator.getDebatePrompt 重建逐字比")
    L += ["## D2. prompt 全文", "",
          "- 舊匯入檔本身：沒有紀錄（`trace` 是 null）。以下舊匯入一律用匯入來源 `result/challenge` 的對話（每一輪兩個 agent 的 user prompt 與輸出）。",
          "- 最後裁決的 prompt：兩邊都沒有存（只有輸出 `Result3`）→ 沒有紀錄。",
          "- 抽樣：每個模型 × {舊匯入, 新跑} 各用新的 `numpy.random.default_rng(0)`，從有辯論的題（依檔名、item_id 排序）抽 20 題；題號在 `D2_items.csv`。", "",
          "固定文字（每一輪的 prompt 切掉對手前一次的輸出；依 agent 的語言；新跑的非英文來自 DeepSeek、Gemini 的語言配對）：", "", md(FX), ""]
    for language, dl in fx_diffs:
        L += [f"語言 {language} 的差異（完整）：", "", "```diff", *dl, "```", ""]
    L += ["用目前程式的 Debate prompt 樣板（`DebateAggregator.getDebatePrompt`，套上同一題對手前一次的輸出）重建每一輪的 prompt，和紀錄逐字比：", "",
          md(rb_tab), "",
          f"- 舊匯入 20 題 × 2 個模型：{'全部逐字相同' if d2_same_legacy else '有不同（見 D2_turns.csv）'}；固定文字{'逐字相同' if d2_fixed_same else '有不同'}。"
          "有存 prompt 全文，所以不做間接的 tokens 比對。", ""]

    # ---------- D3 ----------
    d3 = []
    for f in files:
        A_, B_ = arm(f["model"], f["dataset"], f["arm_a"]), arm(f["model"], f["dataset"], f["arm_b"])
        cmp = compareFn(f["dataset"])
        deb = sorted(r["item_id"] for r in f["recs"] if debated(r))
        dif = sorted(i for i in A_ if not cmp(A_[i]["parsed_answer"], B_[i]["parsed_answer"]))
        only_deb, only_dif = sorted(set(deb) - set(dif)), sorted(set(dif) - set(deb))
        rng = np.random.default_rng(0)
        pick = lambda xs: " ".join(map(str, sorted(rng.choice(xs, min(5, len(xs)), replace=False)))) if xs else ""
        sa = st = n = 0
        for iid in deb:
            d = dialogue(f, iid)
            n += 1
            sa += int(d["start_answers"][0] == A_[iid]["parsed_answer"] and d["start_answers"][1] == B_[iid]["parsed_answer"])
            if d["start_texts"] is not None:
                st += int(d["start_texts"][0] == A_[iid]["raw_text"] and d["start_texts"][1] == B_[iid]["raw_text"])
        d3.append({"file": f["file"], "model": f["model"], "dataset": f["dataset"], "pair": f["pair"], "kind": KIND_LABEL[f["kind"]],
                   "debated": len(deb), "arms_differ": len(dif), "sets_equal": deb == dif, "only_debated": len(only_deb), "only_arms_differ": len(only_dif),
                   "only_debated_sample5": pick(only_deb), "only_arms_differ_sample5": pick(only_dif),
                   "start_answers_equal": sa, "start_answers_share": sa / n if n else float("nan"),
                   "start_texts_equal": st if f["kind"] == "legacy" else "沒有紀錄",
                   "start_texts_share": (st / n if n else float("nan")) if f["kind"] == "legacy" else "沒有紀錄",
                   "start_from": "匯入來源 result/challenge（AnswerRecord[0]、Record[1]）" if f["kind"] == "legacy" else "trace（AnswerRecord[0]；起始推理文字沒有存）"})
    D3 = pd.DataFrame(d3)
    D3.to_csv(f"{OUT}/D3_files.csv", index=False)
    d3_tab = D3.groupby("kind").agg(檔數=("file", "size"), 題目集合完全相同的檔=("sets_equal", "sum"), 有辯論的題=("debated", "sum"),
                                    只在有辯論的=("only_debated", "sum"), 只在答案不同的=("only_arms_differ", "sum"),
                                    起始答案相同的題=("start_answers_equal", "sum")).reset_index()
    leg3 = D3[D3.kind == KIND_LABEL["legacy"]]
    d3_ok = bool(leg3.sets_equal.all() and (leg3.start_answers_equal == leg3.debated).all())
    d3_text_ok = bool((leg3.start_texts_equal == leg3.debated).all())
    verdict_flags["D3"] = d3_ok and d3_text_ok
    MANIFEST.extend({"group": "D3:result/arms", "path": p, "sha256": sha256(p)} for p, _ in sorted(_ARMS.values(), key=lambda x: x[0]))
    source("D3", "辯論的題目集合與起始答案", [f"{OUT}/D3_files.csv"], "全部 192 個 Debate 檔；arm 檔見 source_manifest.csv 群組 D3:result/arms",
           "有辯論 = n_rounds > 0；答案不同 = 兩條 path 的 parsed_answer 用該資料集的 compareTwoAnswer 不相等；起始答案、起始推理文字逐字比")
    L += ["## D3. 辯論的是不是現在的那兩份答案", "",
          "- 有辯論的題 = 匯入檔（或新跑的檔）裡 `n_rounds > 0` 的題；答案不同 = `result/arms` 裡兩條 path 的 `parsed_answer` 用該資料集的 `compareTwoAnswer` 不相等。",
          "- 起始答案與起始推理文字：舊匯入用匯入來源（`AnswerRecord1[0]`、`AnswerRecord2[0]`、`Record1[1]`、`Record2[1]`）；新跑的 trace 只存起始答案，"
          "起始推理文字沒有存（`Aggregator/DebateAggregator.py` 只留辯論的輪次）→ 沒有紀錄。", "",
          md(d3_tab), "",
          f"- 舊匯入 80 個檔：題目集合{'全部完全相同' if leg3.sets_equal.all() else '有不同（見 D3_files.csv）'}；起始答案與 result/arms "
          f"{'全部相同' if (leg3.start_answers_equal == leg3.debated).all() else '有不同'}（{int(leg3.start_answers_equal.sum())} / {int(leg3.debated.sum())} 題）；"
          f"起始推理文字{'全部逐字相同' if d3_text_ok else '有不同'}（{int(leg3.start_texts_equal.sum())} / {int(leg3.debated.sum())} 題）。逐檔在 `D3_files.csv`。", ""]
    bad = D3[~D3.sets_equal]
    if len(bad):
        L += ["題目集合不同的檔：", "", md(bad[["file", "kind", "debated", "arms_differ", "only_debated", "only_arms_differ", "only_debated_sample5",
                                             "only_arms_differ_sample5"]]), ""]

    # ---------- D4 ----------
    d4 = []
    for f in files:
        rows = [r for r in f["recs"] if debated(r)]
        if not rows:
            continue
        rounds = np.array([r["n_rounds"] for r in rows], float)
        finals = np.array([bool(dialogue(f, r["item_id"])["result3"]) for r in rows])
        calls = 2 * rounds + finals
        d4.append({"kind": f["kind"], "model": f["model"], "dataset": f["dataset"], "pair": f["pair"], "n": len(rows), "rounds": rounds.sum(),
                   "calls": calls.sum(), "tokens_out": sum(r["tokens_out"] for r in rows), "off_menu": sum(r["off_menu"] for r in rows),
                   "no_answer": sum(not GenerationRecord.isParseOk(r["final_answer"]) for r in rows), "final": finals.sum()})
    D4 = pd.DataFrame(d4)
    g = D4.groupby(["kind", "model", "dataset"]).sum(numeric_only=True).reset_index()
    g["有辯論的題"] = g.n
    g["平均輪數"] = g.rounds / g.n
    g["平均呼叫次數"] = g.calls / g.n
    g["輸出 tokens／次"] = g.tokens_out / g.calls
    g["off-menu 比例"] = g.off_menu / g.n
    g["沒有答案的比例"] = g.no_answer / g.n
    g["進到最後裁決的比例"] = g.final / g.n
    g.to_csv(f"{OUT}/D4_blocks.csv", index=False)
    show = g.assign(kind=g.kind.map(KIND_LABEL), model=g.model.map(LABEL))[["kind", "model", "dataset", "有辯論的題", "平均輪數", "平均呼叫次數",
                                                                        "輸出 tokens／次", "off-menu 比例", "沒有答案的比例", "進到最後裁決的比例"]]
    for c in ("平均輪數", "平均呼叫次數"):
        show[c] = show[c].map(lambda v: f"{v:.2f}")
    show["輸出 tokens／次"] = show["輸出 tokens／次"].map(lambda v: f"{v:.1f}")
    for c in ("off-menu 比例", "沒有答案的比例", "進到最後裁決的比例"):
        show[c] = show[c].map(lambda v: f"{100 * v:.1f}%")
    source("D4", "行為描述", [f"{OUT}/D4_blocks.csv"], "有辯論的題（n_rounds > 0）；區塊內各配對先加總",
           "輪數、呼叫次數 = 2 × 輪數 + 最後裁決；tokens = 逐題 tokens_out（tokenizer 重算）÷ 呼叫次數；最後裁決：舊匯入看匯入來源的 Result3，新跑看 trace 的 Result3")
    L += ["## D4. 行為上的描述（只報告，不能當成相同或不同的證據）", "",
          "分母是有辯論的題；區塊內各配對先加總再相除；呼叫次數 = 2 × 輪數 + 有沒有最後裁決（舊匯入的最後裁決看匯入來源）；輸出 tokens 是逐題紀錄的 tokenizer 重算值。", "",
          md(show), ""]

    # ---------- D5 ----------
    subsets5 = [("(i) 既有的 80 格", lambda m, p: True),
                ("(ii) 只用新跑的格（強模型 × 5 配對；弱模型 × EN+S1、P1+P2）", lambda m, p: m in STRONG or p in ("EN+S1", "P1+P2")),
                ("(iii) 只用強模型 8 個區塊 × 5 配對", lambda m, p: m in STRONG),
                ("(iv) 只用舊匯入的格（弱模型 × EN+ZH、EN+JA、ZH+JA）", lambda m, p: m in WEAK and p in ("EN+ZH", "EN+JA", "ZH+JA"))]
    t5, b5 = [], []
    for name, inc in subsets5:
        sr, sa_, blk = result5(cells, inc)
        t5.append({"格子": name, "區塊數": sr["n_blocks"], "Debate − Judge 的 recovery": ci(sr, 3), "正確率（pp）": ci(sa_)})
        b5.append(blk.assign(subset=name))
    pd.concat(b5).to_csv(f"{OUT}/D5_result5_blocks.csv", index=False)
    Dser = s1 - zh
    t6 = []
    b6 = []
    for name, models in (("(i) 16 個區塊", MODELS), ("(ii) 只用強模型 8 個區塊", STRONG), ("(iii) 只用弱模型 8 個區塊（新跑的 EN+S1 對舊匯入的 EN+ZH）", WEAK)):
        sel = [i for i in Dser.index if i[0] in models]
        t6.append({"區塊": name, "EN+S1 − 英文": ci(S(s1[sel].values)), "EN+ZH − 英文": ci(S(zh[sel].values)), "兩者相減": ci(S(Dser[sel].values))})
        b6 += [{"subset": name, "model": i[0], "dataset": i[1], "EN+S1": s1[i], "EN+ZH": zh[i], "D": Dser[i]} for i in sel]
    pd.DataFrame(b6).to_csv(f"{OUT}/D5_result6_blocks.csv", index=False)
    # RQ1：用既有的 cells.csv.gz 重新加總（不重跑 RQ1）
    f_rc, f_rb = f"{ANALYSIS}/rq1/cells.csv.gz", f"{ANALYSIS}/rq1/blocks.csv"
    rc = pd.read_csv(f_rc)
    rb = pd.read_csv(f_rb)
    rc = rc[(rc.setting == "probe_random") & (rc.baseline == "pair") & (rc.aggregator == "debate") & (rc.k == 200) & rc.model.isin(MODELS)]
    rc = rc.assign(dDA=100 * (rc.acc_D - rc.acc_A), rA=100 * (rc.acc_O - rc.acc_A), rS=100 * (rc.acc_O - rc.acc_S),
                   langpair=[set(p.split("+")) <= LANG_CODES for p in rc.pair])
    full = rc.groupby(["model", "dataset"])[["dDA", "rA", "rS"]].mean()
    rbx = rb[(rb.setting == "probe_random") & (rb.baseline == "pair") & (rb.aggregator == "debate") & (rb.k == 200)].set_index(["model", "dataset"])
    rq1_diff = float(max((full.dDA - rbx.loc[full.index, "diff_D_A"]).abs().max(), (full.rA - rbx.loc[full.index, "regret_A"]).abs().max()))
    t1 = []
    rq1_blocks = []
    for name, sel in (("全部（16 個區塊、12 組配對）", rc.model.isin(MODELS)), ("弱模型 8 個區塊（全部配對）", rc.model.isin(WEAK)),
                      ("強模型 8 個區塊（全部配對）", rc.model.isin(STRONG)), ("語言配對（16 個區塊、10 組）", rc.langpair),
                      ("非語言配對（16 個區塊、EN+S1、P1+P2）", ~rc.langpair),
                      ("弱模型 × 語言配對（= 舊匯入的格）", rc.model.isin(WEAK) & rc.langpair), ("弱模型 × 非語言配對", rc.model.isin(WEAK) & ~rc.langpair),
                      ("強模型 × 語言配對", rc.model.isin(STRONG) & rc.langpair), ("強模型 × 非語言配對", rc.model.isin(STRONG) & ~rc.langpair)):
        bl = rc[sel].groupby(["model", "dataset"])[["dDA", "rA", "rS"]].mean()
        sp = min(bl.rA.mean(), bl.rS.mean())
        t1.append({"分開的部分": name, "區塊數": len(bl), "隨機標 200 題 − 永遠聚合": ci(S(bl.dDA.values)), "regret_A": f"{bl.rA.mean():.3f}",
                   "regret_S": f"{bl.rS.mean():.3f}", "空間": f"{sp:.3f}"})
        rq1_blocks.append(bl.reset_index().assign(subset=name))
    pd.concat(rq1_blocks).to_csv(f"{OUT}/D5_rq1_blocks.csv", index=False)
    source("D5", "結果 5 的敏感度", [f_c], "subset = both_answered；5 個共同配對；依 (i)–(iv) 只換納入的格", "同先重現的結果 5 算法")
    source("D5", "結果 6 的敏感度", [f_p, f_a], "同先重現的結果 6；只換區塊", "同核對七 B")
    source("D5", "RQ1 的 Debate", [f_rc, f_rb], "setting = probe_random、baseline = pair、aggregator = debate、k = 200",
           "每個區塊對納入的格平均 100 × (acc_D − acc_A)、100 × (acc_O − acc_A)、100 × (acc_O − acc_S)；空間 = min(regret_A, regret_S) 的區塊平均；"
           "全部格的區塊值和 blocks.csv 比對")

    # 用到這 80 個檔的報告與 CSV
    weak_lang = lambda df, mcol="model", pcol="pair": int((df[mcol].isin(WEAK) & df[pcol].map(lambda p: set(str(p).split("+")) <= LANG_CODES)).sum())
    uses = []
    ac = pd.read_csv(f_c)
    uses.append((f_c, "run_analysis.py：Debate 的分解欄位（每格一列）", weak_lang(ac[ac.aggregator == "debate"])))
    uses.append((f_a, "run_analysis.py：逐題匯出（每題一列）", weak_lang(pd.read_csv(f_a).query("aggregator == 'debate'"))))
    rcall = pd.read_csv(f_rc)
    uses.append((f_rc, "RQ1：逐格（Debate 的列）", weak_lang(rcall[rcall.aggregator == "debate"])))
    for p, what in ((f_rb, "RQ1：逐區塊（Debate 的列含這些格的平均）"), (f"{ANALYSIS}/rq1/summary.csv", "RQ1：跨區塊統計（Debate）"),
                    (f"{ANALYSIS}/rq1/report.md", "RQ1：report 的 Debate 各節"), (f"{ANALYSIS}/rq1/k_curves_debate.png", "RQ1：Debate 的 k 曲線圖")):
        uses.append((p, what, "（由上列加總，不逐列）"))
    for p in sorted(glob.glob(f"{ANALYSIS}/rq1/preliminary/*")):
        uses.append((p, "RQ1 preliminary（gpt4omini、qwen）", "（由逐題匯出算）"))
    dl = pd.read_csv(f"{ANALYSIS}/decomposition/cells_long.csv")
    uses.append((f"{ANALYSIS}/decomposition/cells_long.csv", "12 組配對的分解：逐格", weak_lang(dl[dl.aggregator == "debate"]) if "aggregator" in dl else "（見檔）"))
    for p in sorted(glob.glob(f"{ANALYSIS}/decomposition/*debate*")) + [f"{ANALYSIS}/decomposition/report.md"]:
        uses.append((p, "12 組配對的分解：Debate 的表、圖或 report", "（由逐格算）"))
    b7 = pd.read_csv(f"{ANALYSIS}/check7/B_blocks.csv")
    uses.append((f"{ANALYSIS}/check7/B_blocks.csv", "核對七 B：結果 6 的 Debate（EN+ZH）", weak_lang(b7[b7.aggregator == "debate"])))
    for p in (f"{ANALYSIS}/check7/B_candidates.csv", f"{ANALYSIS}/check7/B_D_blocks.csv", f"{ANALYSIS}/check7/report.md"):
        uses.append((p, "核對七 B：結果 6 的 Debate", "（由上列算）"))
    c7 = pd.read_csv(f"{ANALYSIS}/check7/C_files.csv")
    uses.append((f"{ANALYSIS}/check7/C_files.csv", "核對七 C：只數、不用（沒有 api_usage）", int(((c7.aggregator == "debate") & ~c7.has_usage).sum())))
    for p in ("difficulty_q_groups.csv", "difficulty_q_items.csv", "difficulty_q_pairs.csv", "difficulty_strata_adjusted.csv", "difficulty_strata_pairs.csv",
              "recovery_blind_groups.csv", "recovery_blind_loss_cells.csv", "recovery_blind_mixedlm.csv", "recovery_blind_pairs.csv",
              "pair_acc_cell_estimates.csv", "pair_acc_design_matrix.csv", "split_half_points.csv", "split_half_slopes.csv",
              "split_half_eiv_cells.csv", "split_half_eiv_summary.csv", "loo_folds.csv", "loo_predictions.csv",
              "recovery_heterogeneity_cells.csv", "recovery_heterogeneity.csv"):
        uses.append((f"{ANALYSIS}/0A/{p}", "0A 系列：直接讀匯入來源 result/challenge（含舊模型），不是匯入檔", "（舊框架）"))
    U = pd.DataFrame(uses, columns=["檔案", "用途", "這 80 個檔的列數"])
    U.to_csv(f"{OUT}/D5_uses.csv", index=False)
    L += ["## D5. 敏感度（只報告，不參與任何判定，不取代既有數字）", "",
          "結果 5（Debate − Judge；算法同先重現，只換納入的格）：", "", md(pd.DataFrame(t5)), "",
          "結果 6 的 Debate（各自的 both_answered；算法同先重現，只換區塊）：", "", md(pd.DataFrame(t6)), "",
          f"RQ1 的 Debate（隨機標 200 題、配對內基準）：用既有的 `rq1/cells.csv.gz` 逐格的值重新加總，沒有重跑 RQ1 的程式。"
          f"全部格加總後和 `rq1/blocks.csv` 的最大差 {rq1_diff:.1e}。", "", md(pd.DataFrame(t1)), "",
          "用到這 80 個舊匯入檔的報告與 CSV（列數是這些檔貢獻的列；0A 讀的是匯入來源，不是匯入檔）：", "", md(U), ""]

    # ---------- 讀法 ----------
    missing = sorted(set(C1[C1["比較"] == "沒有紀錄"]["項目"]))
    # 讀法丙的關鍵項目：prompt 全文（D2）、模型版本（D1 的版本字串）、起始答案（D3）
    key = {"prompt 全文": "有（舊匯入在匯入來源 result/challenge；新跑在 trace）",
           "模型版本": "沒有紀錄（兩邊都只有 API 代號，而且相同；呼叫回傳的版本字串兩邊都沒有存）"
           if "模型的版本字串" in missing else "有",
           "起始答案": "有（舊匯入在匯入來源；新跑在 trace 的 AnswerRecord[0]）"}
    key_missing = [k for k, v in key.items() if v.startswith("沒有紀錄")]
    different = (verdict_flags["D1_different"] or not verdict_flags["D2"] or not verdict_flags["D3"])
    if different:
        verdict, sentence = "乙（有差異）", "「有差異」"
    elif key_missing:
        verdict, sentence = "丙（無法確認）", "「無法確認」"
    else:
        verdict = "甲（相同）"
        sentence = "「舊匯入的 80 格和新跑的是同一套做法。setup 寫明這 80 格沿用較早執行的結果與執行日期；D5 只放附錄。」"
    L += ["## 第一部分的讀法（事先寫好的）", "",
          f"- D1 有紀錄的項目：{'全部相同' if not verdict_flags['D1_different'] else '有不同：' + '、'.join(verdict_flags['D1_different'])}"
          f"（模型的 API 代號、temperature、辯論的輪數上限）；兩邊都沒有紀錄的項目：{'、'.join(missing) if missing else '無'}。",
          f"- D2：prompt 逐字{'相同' if verdict_flags['D2'] else '有不同'}（舊匯入用匯入來源）。",
          f"- D3：題目集合與起始答案、起始推理文字{'和 result/arms 全部相同' if verdict_flags['D3'] else '有不同'}。",
          "- 關鍵項目：" + "；".join(f"{k}：{v}" for k, v in key.items()) + "。",
          f"- 選：**{verdict}** → {sentence}"]
    if verdict.startswith("丙"):
        L += [f"- 缺的紀錄：{'、'.join(key_missing)}（呼叫回傳的模型版本字串）；另外兩邊都沒有、但不屬於關鍵項目的："
              + "、".join(x for x in missing if x != "模型的版本字串") + "。",
              "- 處理同乙：不重跑、不改任何既有數字；結果 5、結果 6 的 Debate 欄、RQ1 的 Debate 要不要改寫由使用者決定，D5 的表給使用者判斷用。"]
    L += [          "- D4 不影響選哪一種讀法。", ""]
    return L, verdict


# ==================================================================
# 第二部分：session 紀錄與重播
# ==================================================================
def transcript():
    ev, uses, results = [], {}, {}
    with open(TRANSCRIPT, encoding="utf-8") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = o.get("message")
            if not isinstance(msg, dict):
                continue
            ts = o.get("timestamp")
            content = msg.get("content")
            if isinstance(content, str):
                ev.append({"ts": ts, "kind": "user_text" if msg.get("role") == "user" else "assistant_text", "text": content})
                continue
            for c in content or []:
                t = c.get("type")
                if t == "text":
                    ev.append({"ts": ts, "kind": "assistant_text" if msg.get("role") == "assistant" else "user_text", "text": c.get("text", "")})
                elif t == "tool_use":
                    uses[c["id"]] = {"ts": ts, "name": c["name"], "input": c.get("input", {})}
                    ev.append({"ts": ts, "kind": "tool_use", "id": c["id"], "name": c["name"], "input": c.get("input", {})})
                elif t == "tool_result":
                    cc = c.get("content")
                    text = cc if isinstance(cc, str) else " ".join(x.get("text", "") for x in cc if isinstance(x, dict))
                    results[c.get("tool_use_id")] = text
                    ev.append({"ts": ts, "kind": "tool_result", "id": c.get("tool_use_id"), "text": text})
    return ev, uses, results


def useAt(uses: dict, ts_prefix: str, name: str | None = None) -> dict:
    found = [(k, u) for k, u in uses.items() if u["ts"].startswith(ts_prefix) and (name is None or u["name"] == name)]
    if len(found) != 1:
        raise SystemExit(f"F：{ts_prefix} 找到 {len(found)} 個 tool use")
    return {"id": found[0][0], **found[0][1]}


def heredoc(cmd: str, name: str) -> str:
    m = re.search(r"cat > \$S/" + re.escape(name) + r" <<'EOF'\n(.*?)\nEOF\n", cmd, re.S)
    return m.group(1) + "\n"


RUNNER = r'''
import builtins, io, os, shutil, sys, datetime
SB = os.path.realpath(sys.argv[2])
def inside(p):
    return os.path.realpath(os.path.abspath(p)).startswith(SB + os.sep)
_open = builtins.open
def gopen(file, mode="r", *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)) and any(c in mode for c in "wax+") and not inside(file):
        raise PermissionError(f"blocked write outside sandbox: {file}")
    return _open(file, mode, *a, **k)
builtins.open = gopen; io.open = gopen
for name in ("copyfile", "copy", "copy2", "move"):
    f = getattr(shutil, name)
    def guard(src, dst, *a, _f=f, **k):
        if not inside(dst):
            raise PermissionError(f"blocked copy outside sandbox: {dst}")
        return _f(src, dst, *a, **k)
    setattr(shutil, name, guard)
_mk = os.makedirs
def gmk(p, *a, **k):
    if os.path.isdir(p):
        return
    if not inside(p):
        raise PermissionError(f"blocked makedirs outside sandbox: {p}")
    return _mk(p, *a, **k)
os.makedirs = gmk
for name in ("remove", "unlink", "rename", "replace"):
    setattr(os, name, lambda *a, **k: (_ for _ in ()).throw(PermissionError("blocked")))
NOW = datetime.datetime.fromisoformat(sys.argv[3]) if sys.argv[3] else None
src = _open(sys.argv[1], encoding="utf-8").read()
exec(compile(src, sys.argv[1], "exec"), {"__name__": "__main__", "__NOW__": NOW})
'''


def runSandboxed(script: str, name: str, subs: list[tuple[str, str]], now: str | None) -> str:
    for old, new in subs:
        if script.count(old) < 1:
            raise SystemExit(f"F：{name} 找不到要換的路徑 {old}")
        script = script.replace(old, new)
    script = script.replace("datetime.datetime.now()", "__NOW__")
    path = f"{SANDBOX}/run_{name}"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(script)
    with open(f"{SANDBOX}/runner.py", "w", encoding="utf-8") as fh:
        fh.write(RUNNER)
    res = subprocess.run([sys.executable, f"{SANDBOX}/runner.py", path, SANDBOX, now or ""], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[2]))
    if res.returncode != 0:
        raise SystemExit(f"F：沙盒執行 {name} 失敗：\n{res.stderr[-3000:]}")
    return res.stdout


def sedSandbox(cmd: str, target: str):
    exprs = re.findall(r"sed -i '((?:[^']|'\\'')*)'", cmd)
    if len(exprs) != 1:
        raise SystemExit(f"F：sed 指令解析失敗：{cmd[:120]}")
    subprocess.run(["sed", "-i", exprs[0], target], check=True)


def readText(p: str) -> str:
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def replay(uses: dict, results: dict) -> dict:
    """依 session 紀錄重播三個判定標準檔的每一次寫入。回傳 {實驗: [動作]}；每個動作有 sandbox 裡的狀態檔與 sha256。"""
    if os.path.exists(SANDBOX):
        shutil.rmtree(SANDBOX)
    for x in ("gsk", "gj", "gjr"):
        os.makedirs(f"{SANDBOX}/{x}")
    acts = {"RQ3-GSK": [], "RQ3-GJ": [], "RQ3-GJR": []}

    def snap(exp, ts, kind, what, target, expect=None, use=None, file_label="判定標準檔"):
        h = sha256_nocache(target)
        printed = None
        if use is not None:
            txt = results.get(use["id"], "")
            printed = expect if (expect and expect in txt) else ("（這一步沒有印出雜湊）" if not expect else f"（紀錄裡找不到 {expect[:8]}）")
        n = len(acts[exp]) + 1
        keep = f"{OUT}/F_versions/{exp}_step{n:02d}_{h[:8]}.md"
        shutil.copyfile(target, keep)
        acts[exp].append({"exp": exp, "step": n, "ts": ts, "kind": kind, "what": what, "file": file_label, "sha256": h, "expect": expect,
                          "printed": printed, "match": (expect is None) or (h == expect), "state": keep})

    os.makedirs(f"{OUT}/F_versions", exist_ok=True)
    # ---------- RQ3-GSK ----------
    w = useAt(uses, "2026-10-07T15:33:08", "Write")
    gsk = f"{SANDBOX}/gsk/rq3gsk_criteria.md"
    open(gsk, "w", encoding="utf-8").write(w["input"]["content"])
    snap("RQ3-GSK", w["ts"], "新建（Write，含待填的佔位字）", "Write 寫入 result/analysis/rq3gsk/rq3gsk_criteria.md", gsk, use=w)
    u = useAt(uses, "2026-10-07T15:33:27", "Bash")
    out = runSandboxed(heredoc(u["input"]["command"], "gsk_fill.py"), "gsk_fill.py",
                       [('p="result/analysis/rq3gsk/rq3gsk_criteria.md"', f'p="{gsk}"')], "2026-10-07T23:33:00")
    snap("RQ3-GSK", u["ts"], "整份重寫（填入第零階段的表）", "gsk_fill.py：填佔位字（時間固定 23:33）", gsk,
         "2c7af8b11d8924200cdc625c9b034912e526d0e34678c85243f15db608355188", u)
    c = useAt(uses, "2026-10-07T15:55:08", "Bash")
    sedSandboxMulti(c["input"]["command"], gsk)
    snap("RQ3-GSK", c["ts"], "局部修改（確認）", "確認指令的 sed（狀態、確認兩行）", gsk,
         "6b36382febd44d6d641845cecf80ff7311307a7a3977ec943428c5d0022f05c9", c)

    # ---------- RQ3-GJ ----------
    gj, gjt = f"{SANDBOX}/gj/rq3gj_criteria.md", f"{SANDBOX}/gj/rq3gj_criteria_template.md"
    w = useAt(uses, "2026-10-07T10:18:06", "Write")
    open(gjt, "w", encoding="utf-8").write(w["input"]["content"])
    snap("RQ3-GJ", w["ts"], "新建模板（scratchpad）", "Write 寫入 scratchpad/rq3gj_criteria_template.md", gjt, use=w, file_label="scratchpad 模板")
    u = useAt(uses, "2026-10-07T10:18:28", "Bash")
    runSandboxed(heredoc(u["input"]["command"], "fill.py"), "fill.py",
                 [('open(f"{S}/rq3gj_criteria_template.md")', f'open("{gjt}")'),
                  ('open("result/analysis/rq3gj/rq3gj_criteria.md","w",encoding="utf-8")', f'open("{gj}","w",encoding="utf-8")')], None)
    snap("RQ3-GJ", u["ts"], "新建（fill.py 填模板）", "fill.py：模板 + 第零階段的表 → result/analysis/rq3gj/rq3gj_criteria.md", gj,
         "662f95d11a1ce11f5f64308690b5386c650405ca47137208f475ea29a817d5ac", u)
    for ts, exp_h in (("2026-10-07T10:18:37", "72091a8cc32c85b91d58c368eb7d39e177cdd44c0a92f930879e41cc525f5958"),
                      ("2026-10-07T10:18:40", "7203409ac2ffed193438a11419815dad990eab596c8cfab11a308b0f5e019656")):
        s = useAt(uses, ts, "Bash")
        sedSandbox(s["input"]["command"], gj)
        snap("RQ3-GJ", s["ts"], "局部修改（sed）", re.search(r"sed -i '([^']*)'", s["input"]["command"]).group(1), gj, exp_h, s)
    s = useAt(uses, "2026-10-07T15:36:44", "Bash")
    shutil.copyfile(gj, f"{SANDBOX}/gj/rq3gj_criteria_v1.md")
    snap("RQ3-GJ", s["ts"], "複製備份（scratchpad）", "cp → scratchpad/rq3gj_criteria_v1.md", f"{SANDBOX}/gj/rq3gj_criteria_v1.md",
         "7203409ac2ffed193438a11419815dad990eab596c8cfab11a308b0f5e019656", s, file_label="scratchpad v1")
    wscript = useAt(uses, "2026-10-07T15:40:24", "Write")["input"]["content"]
    u = useAt(uses, "2026-10-07T15:40:34", "Bash")
    runSandboxed(wscript, "gj2_edit.py",
                 [('P = "result/analysis/rq3gj/rq3gj_criteria.md"', f'P = "{gj}"'),
                  ('open(f"{S}/rq3gj_criteria_v1.md", encoding="utf-8")', f'open("{SANDBOX}/gj/rq3gj_criteria_v1.md", encoding="utf-8")'),
                  ('CALLS_CSV = "result/analysis/rq3gj/rq3gj_stage0_calls.csv"', f'CALLS_CSV = "{SANDBOX}/gj/rq3gj_stage0_calls.csv"')],
                 "2026-10-07T23:40:00")
    snap("RQ3-GJ", u["ts"], "整份改寫（第二版）", "gj2_edit.py：逐段替換（時間固定 23:40；呼叫數 CSV 寫到沙盒）", gj,
         "4c41028cf132d0d15685c62f89a78a5c70dd1d818587bb16a0203f5f6d3a4607", u)
    u = useAt(uses, "2026-10-07T15:55:45", "Bash")
    shutil.copyfile(gj, f"{SANDBOX}/gj/rq3gj_criteria_v2.md")
    snap("RQ3-GJ", u["ts"], "複製備份（scratchpad）", "cp → scratchpad/rq3gj_criteria_v2.md", f"{SANDBOX}/gj/rq3gj_criteria_v2.md",
         file_label="scratchpad v2")
    runSandboxed(heredoc(u["input"]["command"], "gj3_edit.py"), "gj3_edit.py",
                 [('P="result/analysis/rq3gj/rq3gj_criteria.md"', f'P="{gj}"'),
                  ('open(f"{S}/rq3gj_criteria_v2.md",encoding="utf-8")', f'open("{SANDBOX}/gj/rq3gj_criteria_v2.md",encoding="utf-8")')],
                 "2026-10-07T23:55:00")
    snap("RQ3-GJ", u["ts"], "局部修改（第三版）", "gj3_edit.py：逐段替換（時間固定 23:55）", gj,
         "5e8c8b45877651f6dab3c959b19a6601d782c6a9fc80fdff821613291e1cdb34", u)
    c = useAt(uses, "2026-10-07T16:07:11", "Bash")
    sedSandboxMulti(c["input"]["command"], gj)
    snap("RQ3-GJ", c["ts"], "局部修改（確認）", "確認指令的 sed（狀態、確認兩行）", gj,
         "4e96b91475c30a490285a91faf6a878583d2691518c6ccff0b520dd6c154e700", c)

    # ---------- RQ3-GJR ----------
    gjr, gjrt = f"{SANDBOX}/gjr/rq3gjr_criteria.md", f"{SANDBOX}/gjr/rq3gjr_criteria_template.md"
    w = useAt(uses, "2026-10-08T16:17:17", "Write")
    open(gjrt, "w", encoding="utf-8").write(w["input"]["content"])
    snap("RQ3-GJR", w["ts"], "新建模板（scratchpad）", "Write 寫入 scratchpad/rq3gjr_criteria_template.md", gjrt, use=w, file_label="scratchpad 模板")
    u = useAt(uses, "2026-10-08T16:17:36", "Bash")
    runSandboxed(heredoc(u["input"]["command"], "gjr_fill.py"), "gjr_fill.py",
                 [('open(f"{S}/rq3gjr_criteria_template.md",encoding="utf-8")', f'open("{gjrt}",encoding="utf-8")'),
                  ('open("result/analysis/rq3gjr/rq3gjr_criteria.md","w",encoding="utf-8")', f'open("{gjr}","w",encoding="utf-8")')],
                 "2026-10-09T00:17:00")
    snap("RQ3-GJR", u["ts"], "新建（gjr_fill.py 填模板）", "gjr_fill.py：模板 + 第零階段的表（時間固定 00:17）", gjr,
         "9a18aa29a583f601e4491a10036c3a73ee6ffda0b27cdddaa9ecdd4857b23600", u)
    for ts, name, now, exp_h, label in (
            ("2026-10-08T16:22:21", "gjr_edit2.py", "2026-10-09T00:22:00", "e53805ce3308d53c49060f7dae73c2af1242ebb8cb504f336af9560216280ddd", "第二版"),
            ("2026-10-08T16:25:21", "gjr_edit3.py", "2026-10-09T00:25:00", "273ebf17984709979d9d27a7528e5ee3b54459db3c3f918ab7e77538037c5827", "第三版")):
        u = useAt(uses, ts, "Bash")
        vname = "v1" if name == "gjr_edit2.py" else "v2"
        shutil.copyfile(gjr, f"{SANDBOX}/gjr/rq3gjr_criteria_{vname}.md")
        snap("RQ3-GJR", u["ts"], "複製備份（scratchpad）", f"cp → scratchpad/rq3gjr_criteria_{vname}.md", f"{SANDBOX}/gjr/rq3gjr_criteria_{vname}.md",
             file_label=f"scratchpad {vname}")
        runSandboxed(heredoc(u["input"]["command"], name), name, [('P="result/analysis/rq3gjr/rq3gjr_criteria.md"', f'P="{gjr}"')], now)
        snap("RQ3-GJR", u["ts"], f"局部修改（{label}）", f"{name}：逐段替換（時間固定 {now[11:16]}）", gjr, exp_h, u)
    c = useAt(uses, "2026-10-08T16:28:17", "Bash")
    sedSandboxMulti(c["input"]["command"], gjr)
    snap("RQ3-GJR", c["ts"], "局部修改（確認）", "確認指令的 sed（狀態、確認兩行）", gjr,
         "46c3cd12ac5eec205317bb2ca9a69d49f86ffb00f93aac75f8c9b96831d945de", c)
    return acts


def sha256_nocache(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sedSandboxMulti(cmd: str, target: str):
    """確認指令：sed -i 's/…/…/; s/…/…/' $f"""
    m = re.search(r"sed -i '((?:[^'])*)' \$f", cmd)
    if not m:
        raise SystemExit(f"F：確認指令解析失敗：{cmd[:120]}")
    subprocess.run(["sed", "-i", m.group(1), target], check=True)


# ==================================================================
# 第二部分：F2 的指示與標記
# ==================================================================
# 使用者訊息裡要求修改判定標準檔的指示（時間 = session 紀錄，CST）。每一點有代號，F2 的改動區塊標上代號。
MSG = {"GJ2": ("RQ3-GJ", "2026-10-07T15:26:37"), "GJ3": ("RQ3-GJ", "2026-10-07T15:54:28"),
       "GJR2": ("RQ3-GJR", "2026-10-08T16:21:50"), "GJR3": ("RQ3-GJR", "2026-10-08T16:24:58")}
POINTS = [
    ("GJ2-0a", "成本不再是限制：刪掉「超過 60 美元就改成只用 gemini3.1flashlite」，兩個供體都用。"),
    ("GJ2-0b", "第零階段全部重算；總估計超過 300 美元先回報，等我回覆。"),
    ("GJ2-0c", "改完更新建立時間與 sha256，把修改處列給我，停，等我確認。"),
    ("GJ2-A1", "A 1. 明顯落單組：保留已抽的 GS-105、GS-095、GS-142，從其餘 59 份再抽 7 份，共 10 份。"),
    ("GJ2-A2", "A 2. 對稱組：保留已抽的 GS-019、GS-026、GS-020，從其餘 59 份再抽 7 份，共 10 份。"),
    ("GJ2-A3", "A 3. 中間組：從 63 份抽 10 份。抽法：新建一個 numpy.random.default_rng(1)，依序抽明顯落單 7 份、對稱 7 份、中間 10 份；各組依編號排序後用 rng.choice(…, replace=False)。"),
    ("GJ2-A4", "A 4. 英文落單組：187 份裡三條都不是翻譯的語言 path 的菜單，依落單程度由高到低取前 10 份（相同時依編號）；不足 10 份就全取。和前三組重複的菜單只跑一次，在兩組都計入。"),
    ("GJ2-A5", "A 末段：判定標準檔列出四組的菜單、三組一致率、落單的 path、落單程度、在 RQ3-GS 的 E2_menu，以及各組落單程度的平均與範圍。「第一份菜單」仍是 GS-105。"),
    ("GJ2-B1", "B 1. 判定一：範圍改成明顯落單組的 10 份。限制 (a)(c) 不變。限制 (b) 改為：「區間只反映 8 個區塊之間的變動，不反映抽到哪 10 份。對照用同樣 10 份上的多數決，不用 RQ3-GS 62 份的 +1.75。逐菜單的值另外報。」"),
    ("GJ2-B2", "B 2. 判定一「反向成立」的讀法改為：「裁判之下落單的那條反而比較值得改。原因看票型拆解（C 的第 11 項）：主要來自 (b)…；主要來自 (a)…」"),
    ("GJ2-B3", "B 3. 判定二：範圍改成四組全部的菜單。「正向成立」的讀法改為：…「主要來自」的操作定義：8 個區塊合計時，該類佔 Judge 分子總和的比例最大。"),
    ("GJ2-B4", "B 4. 判定三：範圍改成四組全部的菜單，其餘不變。"),
    ("GJ2-B5", "B 5. 新增判定四：三條都是英文時，落單的那條在裁判之下還是最不值得改嗎（英文落單組）；E_J 的算法、單位、門檻同判定一；四種讀法；限制。"),
    ("GJ2-B6", "B 末段：參與判定的是判定一、二、三、四，其餘只報告。"),
    ("GJ2-C1", "C：第 1 項改為：E_J 在四組各算一次，Judge 與多數決並排；另外報每份菜單的 E_J 與菜單之間的標準差。"),
    ("GJ2-C11", "C：新增第 11 項：分子依票型的轉換拆開（六種狀態、每格報題數與變化、四類 (a)–(d)、範圍）。"),
    ("GJ2-C12", "C：新增第 12 項：預測的 E_J 與 D_JV（只用預測檔；評分半；實際與預測；存檔時一併寫入）。"),
    ("GJ2-C13", "C：新增第 13 項：預測的泛化（1 留一個資料集與讀法；2 換裁判的 3 格表；3 正式跑之前存檔、記錄 sha256）。"),
    ("GJ2-C14", "C：新增第 14 項：K = 2 的附帶量（見 D）。"),
    ("GJ2-D", "D：K = 2 的附帶量（配對、prompt、設定、版本、題目、兩種順序、得分、報 (a)–(d)、讀法）。"),
    ("GJ2-E1", "E：第 9 節（第零階段）全部重算：四組菜單與 K = 2 的呼叫數、tokens、費用、附錄 A.5 的分母表（四組都列）。"),
    ("GJ2-E2", "E：第 11 節（試跑）加一段：gpt4omini × mmlu × EN+ZH，前 100 題，5 個版本、兩種順序；另外報選第二個的比例。"),
    ("GJ2-E3", "E：第 12 節加一項：K = 2 每一題兩次呼叫的候選文字逐字相同，只有位置對調。"),
    ("GJ3-0", "改完更新時間與 sha256，把修改前後的文字並排列給我，停。"),
    ("GJ3-1a", "1. 判定四「兩者相當」的讀法改為：「三條都是英文時，裁判之下改哪一條差不多。原因看第 8 節 15：…其他情況只列數字。」"),
    ("GJ3-1b", "1. 第 8 節新增第 15 項（只報告）：範圍 6 份對 10 份；報 E_J、差、落單程度平均；限制「只有 6 份，其中 4 份落單的是 ES、5 份含 P2；只當線索」；「明顯為正」的操作定義。"),
    ("GJ3-2a", "2. 第 8 節 14 (b) 加一項基準：原本的版本…裁判的選對率（依位置分開，並報兩種順序的平均）。"),
    ("GJ3-2b", "2. 第 8 節 14 事先寫好的讀法改為：「K = 2 沒有多數可以跟。(a) 高於 5 只表示…認不認得出看 (b)…」"),
    ("GJ3-3", "3. 第 6.2 節「主要來自」的操作定義後面加一句：「佔比最大的是 (c) 或 (d) 時，只列數字，不下 (a)、(b) 的讀法。」判定一、判定四的「反向成立」同樣適用。"),
    ("GJ3-4a", "4. 第 2.1 節：「判定一的單位是……」改成「判定一、判定四的單位是……」。"),
    ("GJ3-4b", "4. 第 8 節 5：「兩個供體分開的判定一、判定二」加上判定四；「兩個宿主分開的三項判定」改成「四項判定」。"),
    ("GJR2-0", "改完更新時間與 sha256，把修改處列給我，停。"),
    ("GJR2-1", "1. 第 7 節 2：實際 − 預測改成 T_old、T_new、M0、M1、M2、M3 並排；用途寫明（M1 加入誰寫的；M2 加入文字類型；M3 加入長度）。只描述。"),
    ("GJR2-2", "2. 第 3 節加 M5 = M4 + pathacc（pathacc 的定義、裁判看不到）；第 7 節 3 多報 M5 的係數、trans 與 short 在 M4 與 M5 之間的變化；事先寫下的讀法；限制。"),
    ("GJR3-0", "改完更新時間與 sha256，把修改處列給我，停。"),
    ("GJR3-1", "第 3 節與第 7 節 3 的 M5 改成兩個：M5a = M3 + pathacc（沒有交互作用項）；M5b = M4 + pathacc。"),
    ("GJR3-2", "報 M5a、M5b 的各係數（8 個區塊的平均、95% t 區間、幾個區塊為正）。"),
    ("GJR3-3", "trans、short 的變化改成 M5a − M3（逐區塊相減）。"),
    ("GJR3-4", "M5b 另外報翻譯候選的兩個量：答錯時 = trans；答對時 = trans + correct×trans。short 同樣處理。"),
    ("GJR3-5", "事先寫下的讀法改用 M5a 的 trans 係數；操作定義不變（95% 區間上限 < 0，且 8 個區塊至少 5 個為負）。"),
    ("GJR3-6", "理由寫進檔案：M4、M5b 有 correct×trans，trans 的係數只代表答錯的候選，不能當成整體的效果。"),
    ("GJR3-7", "附錄 A.4 的配適次數跟著更新。"),
]
PROCEDURAL = {   # 指示裡不是改檔的部分（只列）
    "GJ2": "確認前不呼叫 API，不算任何版本的正確率、效果或預測值；「先作RQ3-GSK再做RQ3-GJ(有內容有修改要注意)」。",
    "GJ3": "不用重抽菜單，不用重算第零階段；不要呼叫 API，不要算任何版本的正確率、效果或預測值。",
    "GJR2": "不配適任何模型。", "GJR3": "不配適任何模型。"}
A_ = ["GJ2-A1", "GJ2-A2", "GJ2-A3", "GJ2-A4"]
# (實驗, 步驟, 區塊) → (對應的指示代號, 指示以外的部分；None = 沒有)。步驟 = F1 的步驟序號。人工讀 --dump-hunks 的 diff 後填。
LABELS = {
    ("RQ3-GJ", 3, 1): ([], "第一版顯示給使用者之前的修正：sed 把附錄 A.4 的「都明顯低於」改成「都低於」；沒有對應的使用者指示"),
    ("RQ3-GJ", 4, 1): ([], "第一版顯示給使用者之前的修正：sed 把建立時間 18:30 改成 18:18；沒有對應的使用者指示"),
    ("RQ3-GJ", 6, 1): (["GJ2-0c"], "另加「修改：」一行，寫第一版的雜湊與這次改了什麼（指示只說更新建立時間與 sha256）"),
    ("RQ3-GJ", 6, 2): (["GJ2-E1"], None),
    ("RQ3-GJ", 6, 3): (["GJ2-B5"], None),
    ("RQ3-GJ", 6, 4): (["GJ2-B5", "GJ2-C14"], None),
    ("RQ3-GJ", 6, 5): (["GJ2-A1", "GJ2-A2", "GJ2-A3"], None),
    ("RQ3-GJ", 6, 6): (A_ + ["GJ2-A5"], None),
    ("RQ3-GJ", 6, 7): (["GJ2-A5"], "菜單表拿掉了第一版的「RQ3-GS 名次」欄（指示列出的欄位沒有它，也沒有說要刪）"),
    ("RQ3-GJ", 6, 8): (["GJ2-A5", "GJ2-B5"], "跟著新菜單改寫第 3 節的描述：英文落單組落單的是 W1、W2、相像的一對都在 EN、S1、S2、P1、P2 之中、對稱組 10 份中 5 份含翻譯的 path 等句子，指示沒有逐字要求"),
    ("RQ3-GJ", 6, 9): (["GJ2-E1"], None),
    ("RQ3-GJ", 6, 10): (["GJ2-E1"], "第 5 節分母規則的說明改寫，並新增 K = 2 的分母與「逐配對另列時最小 7.72pp，那一格會標為『分母過小』（只描述）」，指示沒有提"),
    ("RQ3-GJ", 6, 11): (["GJ2-B5"], None),
    ("RQ3-GJ", 6, 12): (["GJ2-B1"], None),
    ("RQ3-GJ", 6, 13): (["GJ2-B1"], None),
    ("RQ3-GJ", 6, 14): (["GJ2-B1"], None),
    ("RQ3-GJ", 6, 15): (["GJ2-B2"], "另加「【補充，待確認】這裡拆解的範圍是明顯落單組「落單那條」的替換。「主要來自」的定義見第 6.2 節。」"),
    ("RQ3-GJ", 6, 16): (["GJ2-B1"], "指示說限制 (a)(c) 不變；限制 (a) 括號裡的份數從「這 3 份是 JA、JA、ZH」改成「這 10 份是 JA 6、ZH 3、ES 1」；限制 (b) 句尾另加「（第 8 節 1）」"),
    ("RQ3-GJ", 6, 17): (["GJ2-B3"], None),
    ("RQ3-GJ", 6, 18): (["GJ2-B3", "GJ2-A4"], None),
    ("RQ3-GJ", 6, 19): (["GJ2-B3"], None),
    ("RQ3-GJ", 6, 20): (["GJ2-B3"], "另加「【補充，待確認】判定二拆解的範圍是全部菜單、全部參與的替換；判定一、判定四的「反向成立」用該組「落單那條」的替換。」"),
    ("RQ3-GJ", 6, 21): (["GJ2-B4"], None),
    ("RQ3-GJ", 6, 22): (["GJ2-B5", "GJ2-B6"], "另加：反向成立的拆解範圍「（第 8 節 11，英文落單組「落單那條」的替換）」、限制裡兩組的範圍數字、句尾「（第 8 節 7(a) 與 13 的四種狀態只用來套事先寫好的讀法）」"),
    ("RQ3-GJ", 6, 23): (["GJ2-C12", "GJ2-C13"], None),
    ("RQ3-GJ", 6, 24): (["GJ2-C1"], None),
    ("RQ3-GJ", 6, 25): (A_, None),
    ("RQ3-GJ", 6, 26): (A_ + ["GJ2-B3"], None),
    ("RQ3-GJ", 6, 27): (A_ + ["GJ2-B4"], None),
    ("RQ3-GJ", 6, 28): (["GJ2-A1", "GJ2-B1"], None),
    ("RQ3-GJ", 6, 29): (["GJ2-C11", "GJ2-C12", "GJ2-C13", "GJ2-C14", "GJ2-D"],
                        "另加三個【補充，待確認】（第 13 項 2 換裁判的表用另一個裁判 RQ1-KJ 的全部題目；第 14 項的順序甲、乙；第 14 項 (d) 的較高、較低怎麼定），"
                        "以及幾處細節：「四類互斥，合起來等於分子」「票型的定義同第 7 節」「分格與平滑同第 7 節」、K = 2 的 prompt 與 choice-k-v1 的關係、"
                        "「排除規則同第 4 節；第零階段 96 個替換全部參與」、原本的版本在兩個供體子集二的聯集上呼叫、agg_no_answer 算錯、多數決「程式也實際算一次核對」、"
                        "(b)「以每次呼叫為單位」、(c)「原本的版本與替換的版本分開」、「逐配對的值另外列，只描述」"),
    ("RQ3-GJ", 6, 30): (["GJ2-E1"], "另把逐列的呼叫數表存成 result/analysis/rq3gj/rq3gj_stage0_calls.csv（gj2_edit.py 在確認前就寫進 result/analysis/）"),
    ("RQ3-GJ", 6, 31): (["GJ2-E1"], None),
    ("RQ3-GJ", 6, 32): (["GJ2-E1"], None),
    ("RQ3-GJ", 6, 33): (["GJ2-0a", "GJ2-0b", "GJ2-E1"], None),
    ("RQ3-GJ", 6, 34): (["GJ2-E2"], "第 11 節試跑的停止門檻從「超過 60 美元就停」改成「超過 300 美元就停」（標「【待確認】」；指示只說刪掉換成只用 gemini 的 60 美元退路、總估計超過 300 美元先回報）"),
    ("RQ3-GJ", 6, 35): (["GJ2-E2"], None),
    ("RQ3-GJ", 6, 36): (A_, None),
    ("RQ3-GJ", 6, 37): (["GJ2-D"], "第 12 節「換成它自己」的檢查擴到 K = 2 的 3 個配對，指示沒有提"),
    ("RQ3-GJ", 6, 38): (["GJ2-E3"], None),
    ("RQ3-GJ", 6, 39): (["GJ2-C12", "GJ2-C13"], "輸出清單另列 rq3gj_stage0_calls.csv"),
    ("RQ3-GJ", 6, 40): (["GJ2-C11", "GJ2-C14", "GJ2-D"], None),
    ("RQ3-GJ", 6, 41): (["GJ2-B5"], None),
    ("RQ3-GJ", 6, 42): (["GJ2-E1"], None),
    ("RQ3-GJ", 6, 43): (["GJ2-A5"], None),
    ("RQ3-GJ", 6, 44): (["GJ2-E1"], None),
    ("RQ3-GJ", 6, 45): (["GJ2-E1"], "附錄 A.2 改成指向 rq3gj_stage0_calls.csv（附 sha256 與欄位說明）"),
    ("RQ3-GJ", 6, 46): (["GJ2-E1"], "附錄 A.2 第一版逐菜單 × 版本的呼叫數表拿掉，改放 CSV；表改成依宿主 × 資料集"),
    ("RQ3-GJ", 6, 47): (["GJ2-E1"], None),
    ("RQ3-GJ", 6, 48): (["GJ2-E1"], None),
    ("RQ3-GJ", 6, 49): (["GJ2-E1"], None),
    ("RQ3-GJ", 6, 50): (["GJ2-E1", "GJ2-0a", "GJ2-0b"], None),
    ("RQ3-GJ", 6, 51): (["GJ2-E1"], None),
    ("RQ3-GJ", 6, 52): (["GJ2-E1"], None),
    ("RQ3-GJ", 6, 53): (["GJ2-E1", "GJ2-D"], None),
    ("RQ3-GJ", 8, 1): (["GJ3-0"], "「修改：」一行加上第二版的雜湊與這次改的四處"),
    ("RQ3-GJ", 8, 2): (["GJ3-4a"], None),
    ("RQ3-GJ", 8, 3): (["GJ3-3"], None),
    ("RQ3-GJ", 8, 4): (["GJ3-1a"], None),
    ("RQ3-GJ", 8, 5): (["GJ3-4b"], None),
    ("RQ3-GJ", 8, 6): (["GJ3-2a"], None),
    ("RQ3-GJ", 8, 7): (["GJ3-2b", "GJ3-1b"], "第 15 項的限制寫成「其中 4 份落單的是 ES、6 份都含 P2」；指示原文是「其中 4 份落單的是 ES、5 份含 P2」"),
    ("RQ3-GJR", 4, 1): (["GJR2-0"], "另加「修改：」一行，寫第一版的雜湊與這次改了什麼"),
    ("RQ3-GJR", 4, 2): (["GJR2-2"], "另加「【補充，待確認】子集一用寫這份候選的模型自己的子集一（…）。供體寫的候選用供體的子集一。」"),
    ("RQ3-GJR", 4, 3): (["GJR2-1"], None),
    ("RQ3-GJR", 4, 4): (["GJR2-2"], "另加「【補充，待確認】操作定義：「仍為負」= …；「多數同號」= 8 個區塊中至少 5 個為負。…」；變化寫成「M5 − M4，逐區塊相減後報平均、95% t 區間、幾個區塊為正」（指示只說「在 M4 與 M5 之間的變化」）"),
    ("RQ3-GJR", 6, 1): (["GJR3-0"], "「修改：」一行加上第二版的雜湊與這次的修改"),
    ("RQ3-GJR", 6, 2): (["GJR3-1"], None),
    ("RQ3-GJR", 6, 3): (["GJR3-2", "GJR3-3", "GJR3-4", "GJR3-5", "GJR3-6"], None),
    ("RQ3-GJR", 6, 4): (["GJR3-5"], "操作定義前的「【補充，待確認】」拿掉了（指示說操作定義不變，沒有說要拿掉這個標記）"),
    ("RQ3-GJR", 6, 5): (["GJR3-7"], None),
    ("RQ3-GJR", 6, 6): (["GJR3-7"], None),
}
# 指示有、但檔案沒有照原文執行的（人工讀 diff 時記下；另外程式會補上「沒有任何改動區塊對應」的指示）
NOT_DONE = [
    ("RQ3-GJ", "GJ3-1b", "限制的原文「其中 4 份落單的是 ES、5 份含 P2」沒有照寫；檔案寫的是「6 份都含 P2」（第 8 步區塊 7）"),
    ("RQ3-GJ", "GJ2-B1", "「限制 (a)(c) 不變」：限制 (a) 括號裡的份數被改了（第 6 步區塊 16）"),
]


def hunks(a: str, b: str) -> list[dict]:
    al, bl = a.split("\n"), b.split("\n")
    sm = difflib.SequenceMatcher(a=al, b=bl, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.append({"tag": tag, "a_start": i1 + 1, "a_end": i2, "b_start": j1 + 1, "b_end": j2, "removed": al[i1:i2], "added": bl[j1:j2]})
    return out


def userMessage(ev: list, ts_prefix: str) -> str:
    txt = [e["text"] for e in ev if e["kind"] == "user_text" and e["ts"] and e["ts"].startswith(ts_prefix)]
    if len(txt) != 1:
        raise SystemExit(f"F2：{ts_prefix} 的使用者訊息找到 {len(txt)} 則")
    return txt[0]


def part2(ev: list, uses: dict, results: dict, acts: dict) -> tuple[list[str], list[str], dict]:
    exps = {"RQ3-GSK": "rq3gsk", "RQ3-GJ": "rq3gj", "RQ3-GJR": "rq3gjr"}
    expected_last = {"RQ3-GSK": "2c7af8b1", "RQ3-GJ": "5e8c8b45", "RQ3-GJR": "273ebf17"}
    confirmed = {"RQ3-GSK": "6b36382f", "RQ3-GJ": "4e96b914", "RQ3-GJR": "46c3cd12"}
    tsha = sha256(TRANSCRIPT)
    L = ["# 第二部分：判定標準檔——審查的是哪一版，之後還有沒有改", "",
         f"- session 紀錄：`{TRANSCRIPT}`，{os.path.getsize(TRANSCRIPT):,} bytes，sha256 `{tsha}`"
         f"（核對七讀到的是 `{CHECK7_TRANSCRIPT_SHA[:8]}…`：{'相同' if tsha == CHECK7_TRANSCRIPT_SHA else '不同'}）。",
         f"- 重播：每一步用 session 紀錄裡的原始內容（Write 的全文、`sed` 指令、fill / edit 程式的原文）在沙盒 `{SANDBOX}` 重新執行；"
         "程式裡的路徑換成沙盒、`datetime.datetime.now()` 固定成紀錄裡印出的時間、所有寫入都被限制在沙盒（`gj2_edit.py` 原本會寫的 "
         "`rq3gj_stage0_calls.csv` 也寫到沙盒）。每一步的結果存在 `F_versions/`。", ""]
    # ---------- 先重現 ----------
    rows, ok_all = [], True
    for exp in exps:
        st = [a for a in acts[exp] if a["file"] == "判定標準檔"]
        conf = st[-1]
        last = st[-2]
        ok = conf["sha256"].startswith(confirmed[exp]) and last["sha256"].startswith(expected_last[exp]) and \
            conf["sha256"] == sha256(f"{ANALYSIS}/{exps[exp]}/{exps[exp]}_criteria.md") and \
            last["sha256"] == sha256(f"{ANALYSIS}/check7/F_drafts/{exps[exp]}_criteria_draft_reconstructed.md")
        ok_all &= ok
        rows.append({"實驗": exp, "確認版（目標）": confirmed[exp], "重播得到的確認版": conf["sha256"][:8], "確認前最後一版（目標）": expected_last[exp],
                     "重播得到的確認前最後一版": last["sha256"][:8], "和現在的檔案、核對七的重建檔逐 byte 相同": ok})
    L += ["## 先重現（核對七）", "", md(pd.DataFrame(rows)), ""]
    if not ok_all:
        return L + ["重播的最後一版對不上，第二部分停在這裡。", ""], [], {}

    # ---------- F1 ----------
    conf_ts = {exp: acts[exp][-1]["ts"] for exp in exps}
    a_texts = [(e["ts"], e["text"]) for e in ev if e["kind"] == "assistant_text"]
    t_texts = [(e["ts"], e["text"]) for e in ev if e["kind"] == "tool_result"]
    line_sets = {"a": [(ts, set(x.strip() for x in t.split("\n") if x.strip())) for ts, t in a_texts],
                 "b": [(ts, set(x.strip() for x in t.split("\n") if x.strip())) for ts, t in t_texts]}
    f1_rows, front, timelines = [], [], {}
    for exp in exps:
        vno = 0
        tl = []
        for a in acts[exp]:
            text = readText(a["state"])
            vlines = set(x.strip() for x in text.split("\n") if x.strip())
            before = lambda ts: ts <= conf_ts[exp]
            hash_times = [toCST(ts)[5:16] for ts, t in a_texts if before(ts) and a["sha256"][:8] in t]
            cov = {}
            for k in ("a", "b"):
                best = (0.0, None)
                for ts, ls in line_sets[k]:
                    if not before(ts):
                        continue
                    c = len(vlines & ls) / len(vlines)
                    if c > best[0]:
                        best = (c, ts)
                full = [toCST(ts)[5:16] for ts, ls in line_sets[k] if before(ts) and vlines <= ls]
                cov[k] = (best, full)
            if a["file"] == "判定標準檔":
                vno += 1
                label = f"第 {vno} 版" + ("（確認版）" if a is acts[exp][-1] else "")
            else:
                label = f"—（{a['file']}）"
            row = {"實驗": exp, "版本序號": label, "步驟": a["step"], "時間（CST）": toCST(a["ts"]), "動作": a["kind"], "內容": a["what"],
                   "該動作之後檔案的 sha256": a["sha256"],
                   "當時印出的雜湊": ("相同" if a["match"] and a["expect"] else ("沒有印出" if not a["expect"] else "不同")),
                   "雜湊出現在給使用者的訊息（時間）": "、".join(hash_times) if hash_times else "沒有",
                   "全文：助手訊息": (f"有（{'、'.join(cov['a'][1])}）" if cov["a"][1] else f"沒有（最高涵蓋 {100 * cov['a'][0][0]:.0f}%"
                                                                              + (f"，{toCST(cov['a'][0][1])[5:16]}）" if cov["a"][0][1] else "）")),
                   "全文：工具輸出": (f"有（{'、'.join(cov['b'][1])}）" if cov["b"][1] else f"沒有（最高涵蓋 {100 * cov['b'][0][0]:.0f}%"
                                                                            + (f"，{toCST(cov['b'][0][1])[5:16]}）" if cov["b"][0][1] else "）")),
                   "_seen": bool(hash_times) or bool(cov["a"][1]) or bool(cov["b"][1]), "_state": a["state"], "_file": a["file"]}
            tl.append(row)
        timelines[exp] = tl
        f1_rows += tl
    F1 = pd.DataFrame(f1_rows)
    F1.drop(columns=[c for c in F1.columns if c.startswith("_")]).to_csv(f"{OUT}/F1_timeline.csv", index=False)
    # 寫入的完整性：session 裡所有提到這些檔的 tool use，分成重播過的寫入、唯讀
    replayed_ids = set()
    for exp in exps:
        for a in acts[exp]:
            replayed_ids.add(a["ts"][:19])
    scan = []
    for e in ev:
        if e["kind"] != "tool_use":
            continue
        inp = e["input"]
        blob = json.dumps(inp, ensure_ascii=False)
        for exp, x in exps.items():
            names = [f"{x}/{x}_criteria.md", f"{x}_criteria_template.md", f"{x}_criteria_v1.md", f"{x}_criteria_v2.md"]
            if not any(n in blob for n in names):
                continue
            if e["ts"] > conf_ts[exp]:
                continue
            cmd = inp.get("command", "")
            writes = (e["name"] in ("Write", "Edit") and any(inp.get("file_path", "").endswith(n) for n in names)) or \
                bool(re.search(r"sed -i|cp [^;]*criteria|open\([^)]*['\"]w|> *\S*criteria", cmd)) or \
                (e["name"] == "Bash" and re.search(r"python\S*\s+\S*(fill|edit)\S*\.py|conda run[^;]*(fill|edit)\S*\.py", cmd) is not None)
            scan.append({"實驗": exp, "時間（CST）": toCST(e["ts"]), "工具": e["name"], "寫入": writes, "已重播": e["ts"][:19] in replayed_ids,
                         "摘要": (inp.get("file_path") or cmd)[:160].replace("\n", " ")})
    SC = pd.DataFrame(scan)
    SC.to_csv(f"{OUT}/F1_scan.csv", index=False)
    unreplayed = SC[SC["寫入"] & ~SC["已重播"]]
    source("F1", "時間線", [TRANSCRIPT] + sorted(glob.glob(f"{OUT}/F_versions/*.md")), "session dd2d53a9 的 tool use / tool result / 助手訊息；確認之前",
           "沙盒重播每一步；sha256；雜湊前 8 碼出現在助手訊息；全文 = 這一版所有非空行（去掉前後空白）都出現在同一則助手訊息或工具輸出")
    seen_rows = []
    for exp, tl in timelines.items():
        crit = [r for r in tl if r["_file"] == "判定標準檔" and "確認版" not in r["版本序號"]]
        seen = [r for r in crit if r["_seen"]]
        last_seen = seen[-1] if seen else None
        after = [r for r in tl if last_seen and r["步驟"] > last_seen["步驟"] and "確認版" not in r["版本序號"]]
        seen_rows.append({"實驗": exp, "使用者最後看到全文或雜湊的那一版": last_seen["版本序號"] if last_seen else "沒有",
                          "那一版的 sha256": last_seen["該動作之後檔案的 sha256"][:8] if last_seen else "",
                          "之後到確認之間的寫入": "；".join(f"步驟 {r['步驟']} {r['動作']}（{r['內容']}）" for r in after) if after else "沒有",
                          "_after_writes": [r for r in after if r["_file"] == "判定標準檔"]})
    for r in timelines.values():
        pass
    front_tables = []
    for exp, tl in timelines.items():
        t = pd.DataFrame(tl).drop(columns=[c for c in tl[0] if c.startswith("_") or c == "實驗"])
        t["該動作之後檔案的 sha256"] = t["該動作之後檔案的 sha256"].str[:16] + "…"
        front_tables += [f"### {exp}", "", md(t), ""]

    # ---------- F2 ----------
    pts = dict(POINTS)
    f2_hunks, extra, notdone = [], [], []
    texts = {}
    for key, (exp, ts) in MSG.items():
        texts[key] = userMessage(ev, ts)
    for exp in exps:
        crit = [a for a in acts[exp] if a["file"] == "判定標準檔"][:-1]   # 不含確認版
        first = 1 if exp != "RQ3-GSK" else 2                              # GSK 的第 1 版是含佔位字的 Write
        steps = crit[first - 1:]
        for i in range(1, len(steps)):
            prev, cur = readText(steps[i - 1]["state"]), readText(steps[i]["state"])
            for k, h in enumerate(hunks(prev, cur), 1):
                ids, ext = LABELS.get((exp, steps[i]["step"], k), (None, "（未標記）"))
                f2_hunks.append({"實驗": exp, "步驟": steps[i]["step"], "區塊": k, "類型": h["tag"],
                                 "舊行": f"{h['a_start']}–{h['a_end']}", "新行": f"{h['b_start']}–{h['b_end']}",
                                 "對應的指示": "、".join(ids) if ids else "（沒有）", "指示以外的部分": ext or "",
                                 "刪去": "\n".join(h["removed"]), "加入": "\n".join(h["added"])})
                if not ids or ext:
                    extra.append({"實驗": exp, "步驟": steps[i]["step"], "區塊": k, "對應的指示": "、".join(ids) if ids else "（沒有）",
                                  "指示以外的部分": ext})
    F2 = pd.DataFrame(f2_hunks)
    F2.to_csv(f"{OUT}/F2_hunks.csv", index=False)
    unlabeled = int((F2["指示以外的部分"] == "（未標記）").sum()) if len(F2) else 0
    used = set()
    for ids in F2["對應的指示"]:
        used.update(x for x in ids.split("、") if x and x != "（沒有）")
    for pid, txt in POINTS:
        exp = "RQ3-GJ" if pid.startswith("GJ2") or pid.startswith("GJ3") else "RQ3-GJR"
        if pid not in used:
            notdone.append({"實驗": exp, "指示": pid, "原文（摘要）": txt, "說明": "沒有任何改動區塊對應"})
    for exp, pid, note in NOT_DONE:
        notdone.append({"實驗": exp, "指示": pid, "原文（摘要）": pts[pid], "說明": note})
    EX, ND = pd.DataFrame(extra), pd.DataFrame(notdone)
    EX.to_csv(f"{OUT}/F2_extra_changes.csv", index=False)
    ND.to_csv(f"{OUT}/F2_not_done.csv", index=False)
    pd.DataFrame([{"代號": p, "原文（摘要）": t} for p, t in POINTS]).to_csv(f"{OUT}/F2_instructions.csv", index=False)
    source("F2", "指示與改動區塊", [TRANSCRIPT, f"{OUT}/F2_hunks.csv"], "使用者在第一版草稿到確認之間要求修改的訊息（GJ 兩則、GJR 兩則；GSK 沒有）",
           "difflib.SequenceMatcher 逐行比對相鄰兩版；每個改動區塊人工標上對應的指示代號（LABELS）與指示以外的部分")

    # ---------- F3 ----------
    # ---------- F4 ----------
    os.makedirs(f"{OUT}/F_drafts_disk", exist_ok=True)
    f4 = []
    for name in ("rq3gj_criteria_template.md", "rq3gj_criteria_v1.md", "rq3gj_criteria_v2.md",
                 "rq3gjr_criteria_template.md", "rq3gjr_criteria_v1.md", "rq3gjr_criteria_v2.md"):
        src = f"{OLD_S}/{name}"
        dst = f"{OUT}/F_drafts_disk/{name}"
        shutil.copyfile(src, dst)
        same_ver = [r["版本序號"] + f"（步驟 {r['步驟']}）" for tl in timelines.values() for r in tl if r["該動作之後檔案的 sha256"] == sha256(src)]
        f4.append({"scratchpad 原檔": src, "複本": dst, "sha256": sha256(src), "複本 sha256": sha256_nocache(dst),
                   "修改時間（CST）": datetime.fromtimestamp(os.path.getmtime(src), CST).strftime("%Y-%m-%d %H:%M:%S"),
                   "等於 F1 的哪一步": "、".join(sorted(set(same_ver))) or "沒有"})
    others = []
    for p in sorted(glob.glob(f"{OLD_S}/**/*criteria*.md", recursive=True)):
        if os.path.basename(p) in [os.path.basename(r["scratchpad 原檔"]) for r in f4] and os.path.dirname(p) == OLD_S:
            continue
        others.append({"檔案": p, "sha256": sha256(p)})
    F4 = pd.DataFrame(f4)
    F4.to_csv(f"{OUT}/F4_drafts_disk.csv", index=False)
    source("F4", "scratchpad 的草稿", list(F4["scratchpad 原檔"]), "session dd2d53a9 的 scratchpad", "原樣複製到 check8/F_drafts_disk/")

    # ---------- 讀法 ----------
    verdict = {}
    for exp in exps:
        n_extra = int((EX["實驗"] == exp).sum()) if len(EX) else 0
        n_nd = int((ND["實驗"] == exp).sum()) if len(ND) else 0
        verdict[exp] = (f"無法比對（F3 留白）；F2 另有指示以外的改動 {n_extra} 處、沒有照原文執行或沒有對應改動的指示 {n_nd} 項，完整列在 F2"
                        if exp != "RQ3-GSK" else "無法比對（F3 留白）；第一版草稿到確認之間沒有改動，也沒有修改指示")
    L += ["## F1. 每一版的時間線", "",
          "（同一張表也放在報告最前面。）「全文」= 這一版的所有非空行（去掉前後空白）都出現在同一則助手訊息或同一則工具輸出裡；雜湊用前 8 碼比對；都只看確認之前。",
          "逐步的檔案在 `F_versions/`；CSV 在 `F1_timeline.csv`。", "",
          f"- 重播得到的最後一版雜湊（確認前）：GSK `{expected_last['RQ3-GSK']}`、GJ `{expected_last['RQ3-GJ']}`、GJR `{expected_last['RQ3-GJR']}`，等於核對七的「確認前最後一版」。",
          f"- session 紀錄裡提到這些檔、而且在確認之前的 tool use 共 {len(SC)} 個（`F1_scan.csv`）；看起來是寫入的 {int(SC['寫入'].sum())} 個，"
          f"其中沒有重播到的：{len(unreplayed)} 個" + ("（" + "；".join(f"{r['時間（CST）']} {r['摘要'][:60]}" for _, r in unreplayed.iterrows()) + "）" if len(unreplayed) else "") + "。",
          "", md(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in seen_rows])), "",
          "- 「使用者最後看到」依上面兩種顯示（雜湊出現在助手訊息、或全文出現在助手訊息或工具輸出）判斷。", ""]
    L += ["## F2. 每次改動對不對得上修改指示", "",
          "範圍：第一版草稿（模板填完後第一次寫出的完整檔）到確認之前；確認那一步（狀態、確認兩行）不列。RQ3-GSK 在這段期間沒有任何改動，使用者只說「GSK沒問題」。", "",
          "### 使用者的修改指示（時間、原文）", ""]
    for key, (exp, ts) in MSG.items():
        t = texts[key]
        if key == "GJ2":
            start = t.index("# RQ3-GJ 判定標準檔（草稿）的修改")
            t = t[start:]
        L += [f"**{exp}，{toCST(ts + '.000Z')}**（代號 {key}-…）：", "", "````text", t.strip(), "````", "",
              f"不是改檔的部分（只列）：{PROCEDURAL[key]}", ""]
    L += ["指示的代號：", "", md(pd.DataFrame([{"代號": p, "原文（摘要）": t} for p, t in POINTS])), "",
          "### 逐步的 diff 與標記", "",
          f"共 {len(F2)} 個改動區塊（未標記 {unlabeled} 個）；完整的刪去與加入的文字在 `F2_hunks.csv`。下面每一步完整列出。", ""]
    for (exp, step), g in F2.groupby(["實驗", "步驟"], sort=False):
        L += [f"#### {exp} 步驟 {step}", ""]
        for r in g.itertuples():
            L += [f"區塊 {r.區塊}（{r.類型}；舊 {r.舊行} → 新 {r.新行}）：對應 **{r.對應的指示}**" + (f"；指示以外：{r.指示以外的部分}" if r.指示以外的部分 else ""), "",
                  "```diff", *[("- " + x) for x in r.刪去.split("\n") if r.刪去], *[("+ " + x) for x in r.加入.split("\n") if r.加入], "```", ""]
    L += ["### 指示以外的改動（只列，不判斷）", "", md(EX) if len(EX) else "沒有。", "",
          "### 沒有執行的指示（只列，不判斷）", "", md(ND) if len(ND) else "沒有。", ""]
    L += ["## F3. 和聊天裡的審查雜湊比對", "",
          "三個實驗的審查雜湊都留白 → 照規格跳過這一步，只輸出 F1 的表（放在報告最前面）。", ""]
    L += ["## F4. scratchpad 裡現存的各版草稿", "",
          "原樣複製到 `check8/F_drafts_disk/`（原檔不動）：", "", md(F4), "",
          "scratchpad 裡其他含 criteria 的檔（不是草稿，只列）：", "", md(pd.DataFrame(others)) if others else "沒有。", "",
          f"session 紀錄不複製：`{TRANSCRIPT}`，{os.path.getsize(TRANSCRIPT):,} bytes，sha256 `{tsha}`。", ""]
    L += ["## 第二部分的讀法（事先寫好的；每個實驗各選一種）", ""]
    for exp in exps:
        L.append(f"- {exp}：**{verdict[exp]}**。")
    L.append("")
    return L, front_tables, verdict


def main():
    ap = ArgumentParser()
    ap.add_argument("--dump-hunks", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    started = datetime.now(CST)
    if not args.dump_hunks:
        before = hashTree()
        pd.DataFrame(before, columns=["path", "sha256", "bytes"]).to_csv(f"{OUT}/hash_before.csv", index=False)
    ev, uses, results = transcript()
    acts = replay(uses, results)
    if args.dump_hunks:
        for exp, al in acts.items():
            states = [a for a in al if a["file"] == "判定標準檔"]
            for i in range(1, len(states)):
                prev, cur = readText(states[i - 1]["state"]), readText(states[i]["state"])
                for k, h in enumerate(hunks(prev, cur), 1):
                    print(f"##### {exp} step{states[i]['step']} hunk {k} [{h['tag']}] a{h['a_start']}-{h['a_end']} b{h['b_start']}-{h['b_end']}")
                    for ln in h["removed"]:
                        print("- " + ln)
                    for ln in h["added"]:
                        print("+ " + ln)
        return
    L1, v1 = part1()
    print("第一部分完成：", v1)
    L2, front, v2 = part2(ev, uses, results, acts)
    print("第二部分完成：", v2)
    pd.DataFrame(SOURCES).to_csv(f"{OUT}/sources.csv", index=False)
    pd.DataFrame(MANIFEST).to_csv(f"{OUT}/source_manifest.csv", index=False)
    after = hashTree()
    pd.DataFrame(after, columns=["path", "sha256", "bytes"]).to_csv(f"{OUT}/hash_after.csv", index=False)
    same = before == after
    head = ["# 核對八：Debate 舊匯入資料的一致性，以及判定標準檔的審查版本", "",
            f"- 產生時間：{started.strftime('%Y-%m-%d %H:%M CST')}；程式 `scripts/analysis_check8/check8.py`；離線、不呼叫任何 API；沒有做 git commit。",
            "- 性質：核對，只確認與整理既有資料；沒有判定標準檔，不產生新的判定。既有檔案唯讀，新檔案只寫到 `result/analysis/check8/`。",
            f"- 既有檔案的雜湊：開始前與結束後各算一次（`hash_before.csv`、`hash_after.csv`，result/analysis/ 底下 check8/ 以外的 {len(before)} 個檔）→ "
            f"**{'完全相同' if same else '不同'}**。",
            "- 使用者確認的做法（2026-10-09）：舊匯入的對話用匯入來源 `result/challenge`（標明）；F1 在沙盒重新執行紀錄裡的程式；F3 留白、跳過；"
            "「全文顯示」分助手訊息與工具輸出兩欄。", "",
            "## 總表", "",
            md(pd.DataFrame([{"部分": "第一部分（Debate 舊匯入對新跑）", "讀法": v1}] +
                            [{"部分": f"第二部分 {exp}", "讀法": v} for exp, v in v2.items()])), "",
            "## F1 的表（F3 留白，照讀法放在最前面讓使用者自己對）", ""] + front
    files = sorted(p for p in glob.glob(f"{OUT}/**/*", recursive=True) if os.path.isfile(p) and not p.endswith("report.md"))
    tail = ["# check8/ 底下的檔案", "", "（`report.md` 本身無法列入自己的雜湊。）", "",
            md(pd.DataFrame([{"檔案": f, "sha256": sha256_nocache(f)} for f in files])), ""]
    with open(f"{OUT}/report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + L1 + L2 + tail))
    print(f"完成；既有檔案雜湊{'相同' if same else '不同'}")


if __name__ == "__main__":
    main()
