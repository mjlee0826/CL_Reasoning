"""
judge_transfer.py — RQ3-GJT：不用真的去改 path 的裁判預測——準確度、成本與泛化（result/analysis/rq3gjt/rq3gjt_criteria.md）

判定標準確認前拒跑。全程離線：不呼叫 API；既有檔案一律唯讀（RQ3-GJ、RQ3-GJR 的預測檔讀之前核對 sha256）；新檔案只寫到 --out-dir。
    1. 讀 RQ3-GJ 的 K = 3 裁判紀錄與 result/arms（沿用 RQ3-GJR 的 loadData），建出每次呼叫的候選因素與每題的假設狀態
    2. 第 5 節各切法（①②③④⑥）、A／B／C 三層、regression 與比例表的配適與預測（先放在記憶體）
    3. 第 10 節的六項檢查；任何一項不過就停，不寫任何輸出
    4. 第 8 節 6 的曲線；把所有預測值寫成檔案，sha256 記在 rq3gjt_predictions_manifest.json
    5. 評分：讀預測檔並核對 sha256，不符就停 → rq3gjt_blocks.csv、rq3gjt_substitutions.csv、rq3gjt_curves.csv、三張圖、report.md

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq3gjt/judge_transfer.py --checks-only   # 只跑第 10 節的檢查，不寫輸出
    conda run -n clreasoning python scripts/analysis_rq3gjt/judge_transfer.py                 # 檢查 → 預測檔 → 評分與報告
    conda run -n clreasoning python scripts/analysis_rq3gjt/judge_transfer.py --score-only    # 預測檔已寫好：核對 sha256 後只重做評分與報告
"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_rq3gjr"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_rq3gj"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from Analysis.preregistration import sha256, confirmationLine
from Analysis.blockStats import summarize, fourState, FORWARD, REVERSE, EQUIVALENT, UNDETERMINED
from Analysis.experimentPlan import PATHS
from Analysis.menuVote import DATASETS
from Analysis.pathImprove import M12, HOSTS, DONORS
from Analysis.alignment import loadRecords
from Analysis.judgeSubstitution import GROUP_ORDER, GROUP_MENUS, MENU_ORDER, OUT_DIR as GJ_DIR
from Analysis.judgeSubstitutionStats import splitEffects
from Analysis.judgeRegression import TRANS, SHORT, design, fitConditionalLogit, ratioTable, cellsOf
from Analysis.judgeTransfer import (M3H, M3, CURVE_Q, CURVE_N, DIRECT_Q, CURVE_SPLITS, CURVE_REPS, regValue, tabValue, fitSafe,
                                    TBlock, realPredict, hitRate, curveSamples, cutOneSamples, directSamples)
import judge_regression as jr
from path_improve_gj import md, pp, MODEL_LABELS, styleAxes, SERIES, INK, MUTED, GRID, SURFACE

OUT_DIR = "result/analysis/rq3gjt"
CRITERIA_FILE = "rq3gjt_criteria.md"
GJR_DIR = "result/analysis/rq3gjr"
SOURCE_CRITERIA = {"result/analysis/rq3gj/rq3gj_criteria.md": "4e96b91475c30a490285a91faf6a878583d2691518c6ccff0b520dd6c154e700",
                   "result/analysis/rq3gjr/rq3gjr_criteria.md": "46c3cd12ac5eec205317bb2ca9a69d49f86ffb00f93aac75f8c9b96831d945de",
                   "result/analysis/rq3g/rq3g_criteria.md": "478fe4a43b69b467e634171e5a8674a9f4754f78dfb6225a357a82e0f3b735d1"}
GJR_PRED = {"reg": "rq3gjr_predictions_M3.csv.gz", "tab": "rq3gjr_predictions_T_new.csv.gz"}
MANIFEST = "rq3gjt_predictions_manifest.json"
THRESHOLD = 0.5
EXPECTED_CALLS, EXPECTED_ORIG = 936_892, 126_692
CHECK1_TARGET = {"J1": ("-0.11", "-1.16", "+0.95"), "J2": ("+0.33", "+0.06", "+0.61")}
CHECK_TOL, CHECK4_TOL, CHECK6_TOL, CHECK6_GROUPS = 1e-9, 1e-12, 1e-6, 5000
HAND = ("gpt4omini", "mmlu", "GS-105")
CUT_LABEL = {"c1": "①", "c2": "②", "c3": "③", "c4": "④", "c5": "⑤", "c6": "⑥"}
MAIN_RUNS = [(c, l) for c in ("c1", "c2", "c3", "c4", "c6") for l in (("A", "B") if c == "c3" else ("A", "C"))]
METHOD_NAME = {("A", "reg"): "M3h", ("A", "tab"): "T_A", ("B", "reg"): "M3", ("B", "tab"): "T_B", ("C", "reg"): "M3", ("C", "tab"): "T_C"}
METRICS = [("bias", "1. 變化的偏差"), ("err_change", "2. 變化每個替換的誤差"), ("spearman", "3. 排名相關"),
           ("err_abs", "4. 絕對正確率每個替換的誤差"), ("D_P", "5. D_P")]
BLOCKS = [(h, d) for h in HOSTS for d in DATASETS]
CLEAR = GROUP_ORDER[0]
LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # 參考調色盤前四格（已用 validate_palette.js 檢查）
READ_J1 = {
    EQUIVALENT: "只用原本版本的裁判紀錄與假設的改進，就估得出落單那條和相像那一對的差距（誤差 0.5 內）。",
    FORWARD: "預測低估了差距。",
    REVERSE: "預測高估了差距。",
    UNDETERMINED: "只列數字；不能寫「不用真的去改 path 就估得出該改哪一條」。",
}
READ_J2 = {
    EQUIVALENT: "約 1% 的原本版本紀錄，估變化的誤差就和用全部差不多。",
    FORWARD: "1% 不夠，誤差明顯變大。",
    REVERSE: "只列數字。",
    UNDETERMINED: "只列數字。",
}
LIMITS = [
    "(a) 這批紀錄已經看過，這是事後分析，不是在新資料上的檢驗。",
    "(b) A 層估的是「同一個模型自己變強」；對答案用的是「換成強模型寫的」。兩者差一個寫的人，沒有完全對得上的答案。",
    "(c) 預測需要每份候選的對錯，也就是需要標準答案。",
    "(d) 菜單、path、模型都看過；A 層的訓練資料用了 39 份菜單的原本版本。",
    "(e) 只有兩個弱裁判，而且兩者的採用率相近；只有 K = 3。",
    "(f) 假設的改進是隨機挑答錯的題目改對；真實的改進不是隨機的，也會把原本對的改錯。",
]


def parseArgs():
    parser = ArgumentParser(description="RQ3-GJT (rq3gjt_criteria.md)")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--gj-dir", default=GJ_DIR)
    parser.add_argument("--gjr-dir", default=GJR_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--checks-only", action="store_true", help="Run the §10 checks, print them, write nothing")
    parser.add_argument("--score-only", action="store_true", help="Prediction files already written: verify sha256, redo scoring and report")
    parser.add_argument("--fake-choices", type=int, default=None,
                        help="Dry run only: replace every valid judge choice with a random one (this seed). Refuses the default --out-dir")
    return parser.parse_args()


def fileSha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def summ(values) -> dict:
    return summarize(np.asarray(values, dtype=float))


def cellText(s: dict, digits: int = 2) -> str:
    f = (lambda x: pp(x)) if digits == 2 else (lambda x: "—" if not np.isfinite(x) else f"{x:+.{digits}f}")
    return f"{f(s['mean'])} [{f(s['ci_low'])}, {f(s['ci_high'])}] {s['n_positive']}/{s['n_blocks']}"


def pctText(s: dict) -> str:
    return f"{100 * s['mean']:.1f}% [{100 * s['ci_low']:.1f}, {100 * s['ci_high']:.1f}]"


def elapsedFn(started: float):
    return lambda: f"{time.time() - started:.0f}s"


# ------------------------------------------------------------------
# 資料
# ------------------------------------------------------------------
class Ctx:
    def __init__(self, args):
        t0 = time.time()
        a = type("A", (), {"armdir": args.armdir, "aggdir": args.aggdir, "gj_dir": args.gj_dir})()
        D = jr.loadData(a)
        self.D, self.args = D, args
        self.G3, self.plan, self.coded, self.splits, self.lengths = D["G3"], D["plan"], D["coded"], D["splits"], D["lengths"]
        G3 = self.G3
        self.version = np.array([k[3] for k in G3.keys], dtype=object)[G3.vkey]
        self.menu = np.array([k[2] for k in G3.keys], dtype=object)[G3.vkey]
        self.orig = self.version == "orig"
        self.fake = args.fake_choices
        if self.fake is not None:
            self.fakeChoices(self.fake)
        self.cell3 = cellsOf(G3)
        self.usage_in, self.usage_out, self.usage_null = self.loadUsage()
        self.bidx = {(h, d): np.flatnonzero((G3.judge == hi) & (G3.dataset == di)) for hi, h in enumerate(HOSTS) for di, d in enumerate(DATASETS)}
        self.blocks = {b: TBlock(D, *b) for b in BLOCKS}
        self.evs = {b: jr.BlockEval(D, *b) for b in BLOCKS}
        self.universe = {b: np.flatnonzero(self.blocks[b].union) for b in BLOCKS}
        self.pairs = self.evs[BLOCKS[0]].pairs
        for b in BLOCKS:
            if self.evs[b].pairs != self.pairs:
                raise SystemExit(f"❌ {b}: the (menu, version, donor) pairs differ between blocks")
        self.subs = [k for k in self.pairs if k[1] != "orig"]
        self.actual = {b: self.actualArrays(b) for b in BLOCKS}
        self._h1 = {}
        print(f"資料與每題的假設狀態（{time.time() - t0:.0f}s）")

    def fakeChoices(self, seed: int):
        """乾跑用：有效選擇換成隨機的候選；版本的實際對錯跟著重算。結果沒有意義，只用來測程式。"""
        G3 = self.G3
        rng = np.random.default_rng(seed)
        chosen = G3.chosen.copy()
        valid = chosen >= 0
        chosen[valid] = rng.integers(0, 3, size=int(valid.sum()))
        G3.chosen = chosen
        for it in self.D["items3"].values():
            it.actual = it.known.copy()
            it.actual[it.cols] = np.mean([G3.actual[ix] for ix in it.idx], axis=0)
        print(f"⚠️  乾跑：{int(valid.sum()):,} 個有效選擇換成隨機（seed {seed}）")

    def loadUsage(self) -> tuple:
        """每組（呼叫）的 API 計費 tokens（call.usage_in、call.usage_out），和 G3 逐組對齊；null 記 0。"""
        G3 = self.G3
        u_in, u_out = np.zeros(len(G3)), np.zeros(len(G3))
        nulls = []
        order = np.argsort(G3.vkey, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(np.bincount(G3.vkey, minlength=len(G3.keys)))])
        for vk, key in enumerate(G3.keys):
            h, d, menu, version, _ = key
            v = self.plan.find(3, h, d, menu, version)
            _, recs = loadRecords(v.path(self.args.gj_dir))
            ids = sorted(recs, key=int)
            ix = order[bounds[vk]:bounds[vk + 1]]
            ix = ix[np.argsort(G3.col[ix], kind="stable")]
            index = {int(i): n for n, i in enumerate(self.coded[(h, d)].item_ids)}
            if not np.array_equal(G3.col[ix], [index[int(i)] for i in ids]):
                raise SystemExit(f"❌ {key}: records do not line up with the groups")
            for n, i in zip(ix, ids):
                c = recs[i].get("call") or {}
                ui, uo = c.get("usage_in"), c.get("usage_out")
                if not isinstance(ui, int) or not isinstance(uo, int):
                    nulls.append((h, d, menu, version, int(i)))
                    continue
                u_in[n], u_out[n] = ui, uo
        return u_in, u_out, nulls

    def actualArrays(self, b) -> dict:
        ev = self.evs[b]
        subs = self.subs
        orig = [(m, "orig", g) for m, _, g in subs]
        return {"act_s": np.stack([ev.act[k] for k in subs]), "act_o": np.stack([ev.act[k] for k in orig]),
                "vote_s": np.stack([ev.vote[k] for k in subs]), "vote_o": np.stack([ev.vote[k] for k in orig]),
                "den": np.stack([ev.den[k[:2]] for k in subs])}

    def h1(self, r: int) -> np.ndarray:
        """每組的題目在選擇半 H1_r。"""
        if r not in self._h1:
            G3 = self.G3
            m = np.zeros(len(G3), dtype=bool)
            for di, d in enumerate(DATASETS):
                s = G3.dataset == di
                m[s] = self.splits[d][r][G3.col[s]]
            self._h1[r] = m
        return self._h1[r]

    def itemSelect(self, samples: dict) -> np.ndarray:
        """{(裁判, 資料集): 題目位置} → 每組的題目是否被抽中。"""
        sel = np.zeros(len(self.G3), dtype=bool)
        for b, cols in samples.items():
            im = np.zeros(len(self.coded[b].item_ids), dtype=bool)
            im[cols] = True
            idx = self.bidx[b]
            sel[idx] = im[self.G3.col[idx]]
        return sel


# ------------------------------------------------------------------
# 配適
# ------------------------------------------------------------------
class Fits:
    """rq3gjt_coefficients.csv 的每一列；同時做第 10 節 5 的洩漏檢查。"""

    def __init__(self, ctx: Ctx):
        self.ctx, self.rows, self.leaks = ctx, [], []

    def fit(self, layer: str, rng: np.ndarray, tags: dict, safe: bool = False, leak: dict | None = None):
        """rng：訓練範圍（含沒有有效選擇的紀錄，用來數呼叫與 tokens）；配適只用有效選擇。回傳 ((names, beta) 或 None, 9 格表 q)。"""
        ctx, G3 = self.ctx, self.ctx.G3
        if layer == "A" and not ctx.orig[rng].all():
            self.leaks.append(f"{tags}: A-layer training data contains a substituted version")
        if leak:
            self.leakCheck(rng, tags, **leak)
        names = M3H if layer == "A" else M3
        mask = rng & G3.valid
        Z = design(G3.X[mask], names)
        y = G3.chosen[mask]
        if safe:
            res, reason = fitSafe(Z, y, names)
        else:
            try:
                res = fitConditionalLogit(Z, y)
            except (RuntimeError, np.linalg.LinAlgError) as e:
                raise SystemExit(f"❌ {tags}: the fit failed ({e}); stop and report (RQ3-GJR §3)")
            res["names"], reason = names, ""
        q = ratioTable(ctx.cell3[mask], G3.actual[mask])[0]
        row = {**tags, "layer": layer, "model": "M3h" if layer == "A" else "M3", "n_calls": int(rng.sum()), "n_groups": int(mask.sum()),
               "tokens_in": float(ctx.usage_in[rng].sum()), "tokens_out": float(ctx.usage_out[rng].sum()),
               "converged": res is not None, "reason": reason}
        if res is not None:
            row.update({"loglik": res["loglik"], "iterations": res["iterations"]})
            for n, b, se in zip(names, res["beta"], res["se"]):
                row[f"b_{n}"], row[f"se_{n}"] = float(b), float(se)
        self.rows.append(row)
        return ((names, res["beta"]) if res is not None else None), q

    def leakCheck(self, rng: np.ndarray, tags: dict, dataset: int | None = None, not_dataset: int | None = None, h1: int | None = None,
                  not_menus: list | None = None, judge: int | None = None):
        ctx, G3 = self.ctx, self.ctx.G3
        bad = []
        if judge is not None and (G3.judge[rng] != judge).any():
            bad.append("other judge")
        if dataset is not None and (G3.dataset[rng] != dataset).any():
            bad.append("other dataset")
        if not_dataset is not None and (G3.dataset[rng] == not_dataset).any():
            bad.append("test dataset")
        if h1 is not None:
            for di, d in enumerate(DATASETS):
                s = rng & (G3.dataset == di)
                if (~ctx.splits[d][h1][G3.col[s]]).any():
                    bad.append(f"H2_{h1} items of {d}")
        if not_menus is not None and np.isin(ctx.menu[rng], not_menus).any():
            bad.append("test-group menus")
        if bad:
            self.leaks.append(f"{tags}: {', '.join(bad)}")


def predictLayer(ctx: Ctx, layer: str, b: tuple, kind: str, params, rows, menus: list | None = None) -> dict:
    """{(菜單, 版本, 子集二的供體): (R,)}。A、B 層 = 第 4 節的假設改進；C 層 = 真的換進來的候選。"""
    G3 = ctx.G3
    if layer in ("A", "B"):
        blk = ctx.blocks[b]
        return blk.hypoPredict(blk.values(G3, ctx.cell3, kind, params, layer), rows, menus)
    idx = ctx.bidx[b]
    p = np.full(len(G3), np.nan)
    p[idx] = regValue(G3.X[idx], *params) if kind == "reg" else tabValue(ctx.cell3[idx], params)
    return realPredict(ctx.evs[b], p, rows, menus)


def store(P: dict, run: tuple, b: tuple, scope: str, preds: dict, R: int, r: int | None = None):
    """P[(切法, 層, 方法)][區塊][(範圍, 菜單, 版本, 供體)] = (R,)；r 不是 None 時只填第 r 格（① ②）。"""
    target = P.setdefault(run, {}).setdefault(b, {})
    for k, v in preds.items():
        key = (scope, *k)
        if r is None:
            target[key] = v
        else:
            target.setdefault(key, np.full(R, np.nan))[r] = v[0]


# ------------------------------------------------------------------
# 第 5 節的各切法（記憶體內）
# ------------------------------------------------------------------
def mainCuts(ctx: Ctx, F: Fits) -> dict:
    G3 = ctx.G3
    P, params = {}, {}
    t0 = time.time()
    # ① 一個資料集各自一套係數
    for hi, h in enumerate(HOSTS):
        for di, d in enumerate(DATASETS):
            for r in range(CURVE_SPLITS):
                base = (G3.judge == hi) & (G3.dataset == di) & ctx.h1(r)
                for layer in ("A", "C"):
                    rng = base & ctx.orig if layer == "A" else base
                    reg, q = F.fit(layer, rng, {"cut": "c1", "judge": h, "target": d, "split": r},
                                   leak={"judge": hi, "dataset": di, "h1": r})
                    for kind, prm in (("reg", reg), ("tab", q)):
                        store(P, ("c1", layer, kind), (h, d), "", predictLayer(ctx, layer, (h, d), kind, prm, [r]), CURVE_SPLITS, r)
    print(f"  ① 完成（{time.time() - t0:.0f}s）")
    # ② 四個資料集合併
    for hi, h in enumerate(HOSTS):
        for r in range(CURVE_SPLITS):
            base = (G3.judge == hi) & ctx.h1(r)
            for layer in ("A", "C"):
                rng = base & ctx.orig if layer == "A" else base
                reg, q = F.fit(layer, rng, {"cut": "c2", "judge": h, "target": "all", "split": r}, leak={"judge": hi, "h1": r})
                for d in DATASETS:
                    for kind, prm in (("reg", reg), ("tab", q)):
                        store(P, ("c2", layer, kind), (h, d), "", predictLayer(ctx, layer, (h, d), kind, prm, [r]), CURVE_SPLITS, r)
    print(f"  ② 完成（{time.time() - t0:.0f}s）")
    # ③ 留一個資料集（A、B；C 讀 RQ3-GJR 的存檔，這裡另外算一份給第 10 節 1、4 核對）
    c3check = {}
    for hi, h in enumerate(HOSTS):
        for di, d in enumerate(DATASETS):
            base = (G3.judge == hi) & (G3.dataset != di)
            for layer in ("A", "B"):
                rng = base & ctx.orig if layer == "A" else base
                reg, q = F.fit(layer, rng, {"cut": "c3", "judge": h, "target": d}, leak={"judge": hi, "not_dataset": di})
                params[("c3", layer, h, d)] = (reg, q)
                for kind, prm in (("reg", reg), ("tab", q)):
                    store(P, ("c3", layer, kind), (h, d), "", predictLayer(ctx, layer, (h, d), kind, prm, slice(None)), 200)
            reg, q = params[("c3", "B", h, d)]
            c3check[(h, d)] = {kind: predictLayer(ctx, "C", (h, d), kind, prm, slice(None)) for kind, prm in (("reg", reg), ("tab", q))}
    print(f"  ③ 完成（{time.time() - t0:.0f}s）")
    # ④ 留一組菜單
    for group in GROUP_ORDER:
        menus = list(GROUP_MENUS[group])
        in_group = np.isin(ctx.menu, menus)
        for hi, h in enumerate(HOSTS):
            base = (G3.judge == hi) & ~in_group
            for layer in ("A", "C"):
                rng = base & ctx.orig if layer == "A" else base
                reg, q = F.fit(layer, rng, {"cut": "c4", "judge": h, "target": group}, leak={"judge": hi, "not_menus": menus})
                for d in DATASETS:
                    for kind, prm in (("reg", reg), ("tab", q)):
                        store(P, ("c4", layer, kind), (h, d), group, predictLayer(ctx, layer, (h, d), kind, prm, slice(None), menus), 200)
    print(f"  ④ 完成（{time.time() - t0:.0f}s）")
    # ⑥ 換裁判：用另一個裁判的全部紀錄
    for hi, h in enumerate(HOSTS):
        oi = 1 - hi
        base = G3.judge == oi
        for layer in ("A", "C"):
            rng = base & ctx.orig if layer == "A" else base
            reg, q = F.fit(layer, rng, {"cut": "c6", "judge": HOSTS[oi], "target": h}, leak={"judge": oi})
            params[("c6", layer, HOSTS[oi])] = reg
            for d in DATASETS:
                for kind, prm in (("reg", reg), ("tab", q)):
                    store(P, ("c6", layer, kind), (h, d), "", predictLayer(ctx, layer, (h, d), kind, prm, slice(None)), 200)
    print(f"  ⑥ 完成（{time.time() - t0:.0f}s）")
    # 第 8 節 8：逐區塊的 A 層 M3h
    for hi, h in enumerate(HOSTS):
        for di, d in enumerate(DATASETS):
            F.fit("A", (G3.judge == hi) & (G3.dataset == di) & ctx.orig, {"cut": "block", "judge": h, "target": d},
                  leak={"judge": hi, "dataset": di})
    print(f"  逐區塊的 A 層係數（{time.time() - t0:.0f}s）")
    return {"P": P, "params": params, "c3check": c3check}


# ------------------------------------------------------------------
# 既有的預測檔（只讀，核對 sha256）
# ------------------------------------------------------------------
def loadGjrPredictions(gjr_dir: str) -> tuple:
    """RQ3-GJR 存檔的 M3 與 T_new（留一個資料集）：{方法: {區塊: {("", 菜單, 版本, 供體): (200,)}}}；回傳 (預測, {檔名: sha256})。"""
    with open(os.path.join(gjr_dir, "rq3gjr_predictions_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("criteria_sha256") != SOURCE_CRITERIA["result/analysis/rq3gjr/rq3gjr_criteria.md"]:
        raise SystemExit("❌ the RQ3-GJR prediction manifest was written under another criteria file")
    out, shas = {}, {}
    for kind, name in GJR_PRED.items():
        path = os.path.join(gjr_dir, name)
        sha = fileSha(path)
        if sha != manifest["files"][name]:
            raise SystemExit(f"❌ {path} does not match its sha256 in the RQ3-GJR manifest")
        shas[os.path.join(gjr_dir, name)] = sha
        frame = pd.read_csv(path).sort_values(["host", "dataset", "menu", "version", "subset_donor", "split"])
        m = {}
        for (h, d, menu, version, g), grp in frame.groupby(["host", "dataset", "menu", "version", "subset_donor"], sort=False):
            m.setdefault((h, d), {})[("", menu, version, g)] = grp.pred_acc.to_numpy()
        out[kind] = m
    return out, shas


def toldPredictions(ctx: Ctx) -> dict:
    """T_old：RQ3-GJ 存檔的留一個資料集預測（jr.loadData 讀時已核對 sha256）。"""
    out = {}
    for (h, d, menu, version, g), v in ctx.D["saved"]["lodo"].items():
        out.setdefault((h, d), {})[("", menu, version, g)] = v
    return out


# ------------------------------------------------------------------
# 指標（第 6 節）
# ------------------------------------------------------------------
def splitSpearman(gp: np.ndarray, ga: np.ndarray) -> tuple:
    """每次切分，替換之間預測多的對實際多的 Spearman（平手用平均排名）；少於 3 個替換或任一邊全部相同時無定義。"""
    if gp.shape[0] < 3:
        return float("nan"), gp.shape[1]
    vals, undefined = [], 0
    for r in range(gp.shape[1]):
        x, y = gp[:, r], ga[:, r]
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            undefined += 1
            continue
        vals.append(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return (float(np.mean(vals)) if vals else float("nan")), undefined


def metricsArr(ps, po, as_, ao) -> tuple:
    """(S, R) 陣列 → (區塊的五個指標, 每個替換的值)。單位 pp。"""
    gp, ga = 100 * (ps - po), 100 * (as_ - ao)
    dp = 100 * (as_ - ps)
    sub = {"bias": (ga - gp).mean(axis=1), "err_change": np.abs(ga - gp).mean(axis=1), "D_P": dp.mean(axis=1),
           "err_abs": np.abs(dp).mean(axis=1), "gain_pred": gp.mean(axis=1), "gain_act": ga.mean(axis=1)}
    sp, und = splitSpearman(gp, ga)
    blk = {"bias": float(sub["bias"].mean()), "err_change": float(sub["err_change"].mean()), "spearman": sp, "spearman_undefined": und,
           "err_abs": float(sub["err_abs"].mean()), "D_P": float(sub["D_P"].mean()), "n_subs": len(ps)}
    return blk, sub


def predArrays(ctx: Ctx, pm: dict, scope: str, sub_idx: np.ndarray | None = None) -> tuple:
    """一個區塊的預測（scope 的範圍）→ (pred_s, pred_o) (S, R)，替換依 ctx.subs 的順序（sub_idx 只取一部分）。"""
    subs = ctx.subs if sub_idx is None else [ctx.subs[i] for i in sub_idx]
    ps = np.stack([pm[(scope, *k)] for k in subs])
    po = np.stack([pm[(scope, k[0], "orig", k[2])] for k in subs])
    return ps, po


def runMetrics(ctx: Ctx, pmap: dict, cut: str, R: int) -> dict:
    """一個（切法、層、方法）：{區塊: (五個指標, 每個替換, 替換的 (範圍, 索引))}；④ 每個區塊 240 個（四組各自的測試替換放在一起）。"""
    out = {}
    for b in BLOCKS:
        pm, A = pmap[b], ctx.actual[b]
        if cut == "c4":
            parts = []
            for group in GROUP_ORDER:
                idx = np.array([i for i, k in enumerate(ctx.subs) if k[0] in GROUP_MENUS[group]])
                ps, po = predArrays(ctx, pm, group, idx)
                parts.append((group, idx, ps, po))
            ps = np.concatenate([p[2] for p in parts])
            po = np.concatenate([p[3] for p in parts])
            idx = np.concatenate([p[1] for p in parts])
            scopes = sum([[p[0]] * len(p[1]) for p in parts], [])
        else:
            ps, po = predArrays(ctx, pm, "")
            idx = np.arange(len(ctx.subs))
            scopes = [""] * len(idx)
        blk, sub = metricsArr(ps, po, A["act_s"][idx, :R], A["act_o"][idx, :R])
        out[b] = (blk, sub, list(zip(scopes, idx.tolist())), (ps, po))
        if cut == "c4":
            out[b][0]["groups"] = {}
            for group, gidx, gps, gpo in parts:
                out[b][0]["groups"][group] = metricsArr(gps, gpo, A["act_s"][gidx, :R], A["act_o"][gidx, :R])[0]
    return out


def perSplitDict(ctx: Ctx, b: tuple, ps: np.ndarray, po: np.ndarray, donor: str | None = None) -> dict:
    """RQ3-GJ 的 splitEffects 用的格式：{(菜單, 版本): {act_s, act_o, pred_s, pred_o, vote_s, vote_o, den}}（200 次切分）。"""
    A = ctx.actual[b]
    out = {}
    for i, (menu, version, g) in enumerate(ctx.subs):
        if donor and g != donor:
            continue
        out[(menu, version)] = {"act_s": A["act_s"][i], "act_o": A["act_o"][i], "pred_s": ps[i], "pred_o": po[i],
                                "vote_s": A["vote_s"][i], "vote_o": A["vote_o"][i], "den": A["den"][i]}
    return out


# ------------------------------------------------------------------
# 第 10 節的檢查
# ------------------------------------------------------------------
def runChecks(ctx: Ctx, F: Fits, M: dict, gjr: dict, args) -> list:
    checks = []
    G3, P, params = ctx.G3, M["P"], M["params"]
    gjr_blocks = pd.read_csv(os.path.join(args.gjr_dir, "rq3gjr_blocks.csv")).set_index(["model", "dataset"])
    gjr_coef = pd.read_csv(os.path.join(args.gjr_dir, "rq3gjr_coefficients.csv"))

    # 1. 重現 RQ3-GJR（切法 ③、C 層）
    c3 = {kind: runMetrics(ctx, gjr[kind], "c3", 200) for kind in ("reg", "tab")}
    dpv = [c3["reg"][b][0]["D_P"] for b in BLOCKS]
    gv = [c3["tab"][b][0]["err_abs"] - c3["reg"][b][0]["err_abs"] for b in BLOCKS]
    s1, s2 = summ(dpv), summ(gv)
    t1, t2 = (pp(s1["mean"]), pp(s1["ci_low"]), pp(s1["ci_high"])), (pp(s2["mean"]), pp(s2["ci_low"]), pp(s2["ci_high"]))
    worst_blk = max(max(abs(dpv[i] - gjr_blocks.loc[b, "D_P_M3"]), abs(gv[i] - gjr_blocks.loc[b, "G"])) for i, b in enumerate(BLOCKS))
    worst_b = 0.0
    for h in HOSTS:
        for d in DATASETS:
            reg, _ = params[("c3", "B", h, d)]
            row = gjr_coef[(gjr_coef.fit == "lodo") & (gjr_coef.model == "M3") & (gjr_coef.judge == h) & (gjr_coef.test_dataset == d)].iloc[0]
            worst_b = max(worst_b, max(abs(b - row[f"b_{n}"]) for n, b in zip(reg[0], reg[1])))
    worst_p, n_cmp = 0.0, 0
    for b in BLOCKS:
        for kind in ("reg", "tab"):
            mine, saved = M["c3check"][b][kind], gjr[kind][b]
            for k, v in mine.items():
                worst_p = max(worst_p, float(np.abs(v - saved[("", *k)]).max()))
                n_cmp += len(v)
    ok = t1 == CHECK1_TARGET["J1"] and t2 == CHECK1_TARGET["J2"] and worst_blk <= CHECK_TOL and worst_b <= CHECK_TOL and worst_p <= CHECK_TOL
    checks.append({"項": 1, "內容": "重現 RQ3-GJR（切法 ③、C 層）：判定一、二；M3 係數；M3 與 T_new 的預測逐筆",
                   "結果": f"判定一 D_P {t1[0]}（{t1[1]} 到 {t1[2]}），判定二 {t2[0]}（{t2[1]} 到 {t2[2]}）；逐區塊和 rq3gjr_blocks.csv 最大差 "
                           f"{worst_blk:.1e}；8 次 M3 係數和 rq3gjr_coefficients.csv 最大差 {worst_b:.1e}；預測 {n_cmp:,} 筆（M3 與 T_new）最大差 {worst_p:.1e}",
                   "通過": ok})

    # 2. 重現 RQ3-GJR 第 7 節 5（留一組菜單）與第 7 節 6（同一個資料集內）
    c4 = runMetrics(ctx, P[("c4", "C", "reg")], "c4", 200)
    c2 = runMetrics(ctx, P[("c2", "C", "reg")], "c2", CURVE_SPLITS)
    worst = 0.0
    for b in BLOCKS:
        for group in GROUP_ORDER:
            g = c4[b][0]["groups"][group]
            worst = max(worst, abs(g["D_P"] - gjr_blocks.loc[b, f"group_D_P_{group}"]), abs(g["err_abs"] - gjr_blocks.loc[b, f"group_MAE_{group}"]))
        worst = max(worst, abs(c2[b][0]["D_P"] - gjr_blocks.loc[b, "within_D_P"]), abs(c2[b][0]["err_abs"] - gjr_blocks.loc[b, "within_MAE"]))
    checks.append({"項": 2, "內容": "重現 RQ3-GJR 第 7 節 5（留一組菜單，逐組）與第 7 節 6（同一個資料集內）的 D_P 與平均絕對誤差",
                   "結果": f"8 個區塊 × （4 組 + 同一個資料集內）× 2 個量，和 rq3gjr_blocks.csv 最大差 {worst:.1e}", "通過": worst <= CHECK_TOL})

    # 3. 呼叫數
    n_orig = int(ctx.orig.sum())
    checks.append({"項": 3, "內容": "呼叫數：K = 3 共 936,892 次；原本的版本和第零階段相同（126,692 次）",
                   "結果": f"K = 3 {len(G3):,} 次；原本的版本 {n_orig:,} 次", "通過": len(G3) == EXPECTED_CALLS and n_orig == EXPECTED_ORIG})

    # 4. 假設的改進
    worst_f0, worst_self, n_sub = 0.0, 0.0, 0
    for b in BLOCKS:
        h, d = b
        blk = ctx.blocks[b]
        for layer, kind in (("A", "reg"), ("A", "tab"), ("B", "tab")):
            reg, q = params[("c3", layer, h, d)]
            vals = blk.values(G3, ctx.cell3, kind, reg if kind == "reg" else q, layer)
            for flag, store_ in (("f_zero", "f0"), ("self_sub", "self")):
                pr = blk.hypoPredict(vals, slice(None), **{flag: True})
                for (menu, version, g), v in pr.items():
                    if version == "orig":
                        continue
                    diff = float(np.abs(v - pr[(menu, "orig", g)]).max())
                    if store_ == "f0":
                        worst_f0 = max(worst_f0, diff)
                        n_sub += 1
                    else:
                        worst_self = max(worst_self, diff)
    worst_real, n_real = 0.0, 0
    for b in BLOCKS:
        h, d = b
        blk = ctx.blocks[b]
        reg, q = params[("c3", "B", h, d)]
        for menu in MENU_ORDER:
            for j, p in enumerate(blk.menus[menu].paths):
                for g in DONORS:
                    for kind, prm in (("reg", reg), ("tab", q)):
                        v = blk.realValues(kind, prm, menu, j, g, ctx.coded, ctx.lengths)
                        m = blk.mask[g].astype(float)
                        pred = (m @ v) / m.sum(axis=1)
                        worst_real = max(worst_real, float(np.abs(pred - gjr[kind][b][("", menu, f"{p}-{g}", g)]).max()))
                        n_real += len(pred)
    hand = handCheck(ctx, params)
    ok4 = worst_f0 <= CHECK4_TOL and worst_self <= CHECK4_TOL and worst_real <= CHECK_TOL and hand["worst"] <= CHECK_TOL
    checks.append({"項": 4, "內容": "假設的改進：f = 0；換成自己；B 層的路徑餵真的候選 = C 層；手算",
                   "結果": f"f = 0（A 層 M3h、T_A 與 T_B，{n_sub:,} 個替換 × 200 次切分）最大差 {worst_f0:.1e}；換成自己 最大差 {worst_self:.1e}；"
                           f"B 層的路徑餵真的候選 vs RQ3-GJR 存檔的 M3 與 T_new {n_real:,} 筆最大差 {worst_real:.1e}；"
                           f"手算（{' × '.join(HAND)}、兩個供體、第 0 次切分、M3h 與 T_A）{hand['n_items']:,} 個題目值與 {hand['n_gain']} 個預測多的正確率，"
                           f"最大差 {hand['worst']:.1e}",
                   "通過": ok4})

    # 5. 沒有洩漏
    checks.append({"項": 5, "內容": "沒有洩漏（① ② 沒有 H2_r 的題目；③ 沒有測試資料集；④ 沒有測試那一組菜單（含 GS-181）；⑥ 沒有目標裁判；A 層只有原本的版本）",
                   "結果": f"{len(F.rows)} 次配適的訓練範圍逐一檢查：" + ("沒有洩漏" if not F.leaks else "；".join(F.leaks[:5])),
                   "通過": not F.leaks})

    # 6. 自己寫的最大概似 vs statsmodels（M3h，原本的版本）
    from statsmodels.discrete.conditional_models import ConditionalLogit
    worst6, detail = 0.0, []
    for hi, h in enumerate(HOSTS):
        idx = np.flatnonzero(G3.valid & ctx.orig & (G3.judge == hi))
        pick = np.sort(idx[np.random.default_rng(0).choice(len(idx), CHECK6_GROUPS, replace=False)])
        Z = design(G3.X[pick], M3H)
        own = fitConditionalLogit(Z, G3.chosen[pick])
        y = np.zeros((len(pick), 3))
        y[np.arange(len(pick)), G3.chosen[pick]] = 1
        model = ConditionalLogit(y.ravel(), Z.reshape(-1, Z.shape[2]), groups=np.repeat(np.arange(len(pick)), 3))
        res = model.fit(method="newton", tol=1e-12, maxiter=100, disp=0)
        diff = float(np.abs(own["beta"] - res.params).max())
        worst6 = max(worst6, diff)
        detail.append(f"{MODEL_LABELS[h]} 最大差 {diff:.1e}（自己的牛頓法 {own['iterations']} 步）")
    checks.append({"項": 6, "內容": "自己寫的最大概似 vs statsmodels ConditionalLogit（M3h，每個裁判 5,000 組原本版本的紀錄，default_rng(0)；statsmodels 用 newton、tol 1e-12）",
                   "結果": "；".join(detail), "通過": worst6 <= CHECK6_TOL})
    return checks


def handCheck(ctx: Ctx, params: dict) -> dict:
    """第 10 節 4 的手算：純 Python 迴圈重算 A 層（切法 ③、M3h 與 T_A）每題的預測值與預測多的正確率。"""
    h, d, menu = HAND
    H, plan = ctx.coded[(h, d)], ctx.plan
    blk = ctx.blocks[(h, d)]
    ms = blk.menus[menu]
    reg, q = params[("c3", "A", h, d)]
    coef = dict(zip(reg[0], [float(x) for x in reg[1]]))
    qt = [float(x) for x in q]
    arm_ids = [PATHS[c] for c in ms.paths]
    order = plan.orders[(h, d, menu)]
    item_ids = [int(i) for i in H.item_ids]
    tokens = {c: ctx.lengths[(h, d, c)][0] for c in ms.paths}
    gold = [int(x) for x in H.gold]

    def value(answers: list, col: int, kind: str) -> float:
        perm = [arm_ids.index(a) for a in order[item_ids[col]]]
        disp = [answers[s] for s in perm]
        if disp[0] == disp[1] == disp[2]:
            return 1.0 if disp[0] == gold[col] else 0.0
        corr = [1.0 if a == gold[col] else 0.0 for a in disp]
        if kind == "tab":
            if corr == [0.0, 0.0, 0.0]:
                return 0.0
            if disp[0] != disp[1] and disp[0] != disp[2] and disp[1] != disp[2]:
                cell = 6 + corr.index(1.0)
            else:
                lone = [k for k in range(3) if disp[k] != disp[(k + 1) % 3] and disp[k] != disp[(k + 2) % 3]][0]
                cell = (3 if corr[lone] == 1.0 else 0) + lone
            return qt[cell]
        u = []
        for k in range(3):
            path = ms.paths[perm[k]]
            feat = {"correct": corr[k], "support": sum(1 for m in range(3) if m != k and disp[m] == disp[k]) / 2,
                    "first": 1.0 if k == 0 else 0.0, "last": 1.0 if k == 2 else 0.0, "trans": 1.0 if path in TRANS else 0.0,
                    "short": 1.0 if path in SHORT else 0.0, "loglen": math.log(tokens[path][col])}
            u.append(sum(coef[n] * feat[n] for n in M3H))
        top = max(u)
        e = [math.exp(x - top) for x in u]
        return sum(e[k] / sum(e) * corr[k] for k in range(3))

    worst, n_items, n_gain = 0.0, 0, 0
    for kind, prm in (("reg", reg), ("tab", q)):
        vals = blk.values(ctx.G3, ctx.cell3, kind, prm, "A")
        prog = blk.hypoPredict(vals, [0], [menu])
        v_plain, vo, vg = vals[menu]
        for g in DONORS:
            dom = [col for col in range(len(item_ids)) if blk.mask[g][0][col]]
            for j, p in enumerate(ms.paths):
                k12 = M12.index(p)
                hc = [bool(H.correct[k12][col]) for col in dom]
                dc = [bool(ctx.coded[(g, d)].correct[k12][col]) for col in dom]
                t = sum(dc) - sum(hc)
                w = sum(1 for x in hc if not x)
                f = t / w if (t > 0 and w > 0) else 0.0
                f_prog = float(blk.fractions([0], menu, j, g)[0])
                worst = max(worst, abs(f - f_prog))
                sub_vals, orig_vals = [], []
                for col, c in zip(dom, hc):
                    answers = [int(H.codes[M12.index(x)][col]) for x in ms.paths]
                    vo_ = value(answers, col, kind)
                    if c:
                        v = vo_
                    else:
                        changed = list(answers)
                        changed[j] = gold[col]
                        v = (1 - f) * vo_ + f * value(changed, col, kind)
                    cf = 1.0 if c else 0.0
                    mine = cf * vo[j][col] + (1 - cf) * ((1 - f_prog) * vo[j][col] + f_prog * vg[j][col])
                    worst = max(worst, abs(v - mine), abs(vo_ - v_plain[col]))
                    sub_vals.append(v)
                    orig_vals.append(vo_)
                    n_items += 1
                gain = sum(sub_vals) / len(sub_vals) - sum(orig_vals) / len(orig_vals)
                gain_prog = float(prog[(menu, f"{p}-{g}", g)][0] - prog[(menu, "orig", g)][0])
                worst = max(worst, abs(gain - gain_prog))
                n_gain += 1
    return {"worst": worst, "n_items": n_items, "n_gain": n_gain}


# ------------------------------------------------------------------
# 第 8 節 6 的曲線
# ------------------------------------------------------------------
def curves(ctx: Ctx, F: Fits, started: float) -> dict:
    G3 = ctx.G3
    S, NP = CURVE_SPLITS, len(ctx.pairs)
    rows = slice(0, S)
    out = {"i": {"reg": {}, "tab": {}, "failed": {}}, "ii": {"reg": {}, "tab": {}, "failed": {}}}
    elapsed = elapsedFn(started)
    for qi, q in enumerate(CURVE_Q):
        reps = 1 if q == 1.0 else CURVE_REPS
        for c in ("i", "ii"):
            for kind in ("reg", "tab"):
                out[c][kind][qi] = np.full((reps, len(BLOCKS), NP, S), np.nan)
            out[c]["failed"][qi] = np.zeros((reps, len(BLOCKS)), dtype=bool)
        for k in range(reps):
            sel = ctx.itemSelect(curveSamples(q, k, ctx.universe))
            for bi, (h, d) in enumerate(BLOCKS):
                hi, di = HOSTS.index(h), DATASETS.index(d)
                base = (G3.judge == hi) & (G3.dataset != di) & sel
                for c, layer in (("i", "A"), ("ii", "C")):
                    rng = base & ctx.orig if layer == "A" else base
                    reg, tq = F.fit(layer, rng, {"cut": f"curve_{c}", "judge": h, "target": d, "q": q, "rep": k}, safe=True,
                                    leak={"judge": hi, "not_dataset": di})
                    if reg is None:
                        out[c]["failed"][qi][k, bi] = True
                    else:
                        pr = predictLayer(ctx, layer, (h, d), "reg", reg, rows)
                        out[c]["reg"][qi][k, bi] = np.stack([pr[p] for p in ctx.pairs])
                    pr = predictLayer(ctx, layer, (h, d), "tab", tq, rows)
                    out[c]["tab"][qi][k, bi] = np.stack([pr[p] for p in ctx.pairs])
        print(f"  曲線 (i)(ii) q = {q:g}（{elapsed()}）")
    # (iii) A 層、切法 ①
    iii = {"reg": np.full((len(CURVE_N), S, len(BLOCKS), NP), np.nan), "tab": np.full((len(CURVE_N), S, len(BLOCKS), NP), np.nan),
           "failed": np.zeros((len(CURVE_N), S, len(BLOCKS)), dtype=bool), "capped": np.zeros((len(CURVE_N), S, len(BLOCKS)), dtype=bool)}
    for ni, n in enumerate(CURVE_N):
        for r in range(S):
            samp, capped = cutOneSamples(n, r, ctx.universe, ctx.splits)
            sel = ctx.itemSelect(samp)
            for bi, (h, d) in enumerate(BLOCKS):
                hi, di = HOSTS.index(h), DATASETS.index(d)
                rng = (G3.judge == hi) & (G3.dataset == di) & sel & ctx.orig
                iii["capped"][ni, r, bi] = capped[(h, d)]
                reg, tq = F.fit("A", rng, {"cut": "curve_iii", "judge": h, "target": d, "n": "all" if n is None else n, "split": r,
                                           "capped": capped[(h, d)]}, safe=True, leak={"judge": hi, "dataset": di, "h1": r})
                if reg is None:
                    iii["failed"][ni, r, bi] = True
                else:
                    pr = predictLayer(ctx, "A", (h, d), "reg", reg, [r])
                    iii["reg"][ni, r, bi] = [pr[p][0] for p in ctx.pairs]
                pr = predictLayer(ctx, "A", (h, d), "tab", tq, [r])
                iii["tab"][ni, r, bi] = [pr[p][0] for p in ctx.pairs]
        print(f"  曲線 (iii) n = {'全部' if n is None else n}（{elapsed()}）")
    out["iii"] = iii
    return out


# ------------------------------------------------------------------
# 預測檔
# ------------------------------------------------------------------
def writeGz(frame: pd.DataFrame, path: str):
    jr.writeGz(frame, path)


def mainFrame(pmap: dict) -> pd.DataFrame:
    parts = []
    for (h, d), m in pmap.items():
        keys = list(m)
        R = len(next(iter(m.values())))
        vals = np.stack([m[k] for k in keys])
        if np.isnan(vals).any():
            raise SystemExit(f"❌ {h} | {d}: some prediction is missing")
        parts.append(pd.DataFrame({"host": h, "dataset": d, "test_group": np.repeat([k[0] for k in keys], R),
                                   "menu": np.repeat([k[1] for k in keys], R), "version": np.repeat([k[2] for k in keys], R),
                                   "subset_donor": np.repeat([k[3] for k in keys], R), "split": np.tile(np.arange(R), len(keys)),
                                   "pred_acc": vals.ravel()}))
    return pd.concat(parts, ignore_index=True)


def writePredictions(ctx: Ctx, M: dict, C: dict, F: Fits, checks: list, gjr_shas: dict, args, sha: str) -> dict:
    os.makedirs(args.out_dir, exist_ok=True)
    files, shapes = {}, {}
    for (cut, layer, kind), pmap in sorted(M["P"].items()):
        name = f"rq3gjt_pred_{cut}_{layer}_{kind}.csv.gz"
        frame = mainFrame(pmap)
        writeGz(frame, os.path.join(args.out_dir, name))
        files[name], shapes[name] = fileSha(os.path.join(args.out_dir, name)), [len(frame)]
    pairs = np.array(ctx.pairs, dtype=str)
    blocks = np.array(BLOCKS, dtype=str)
    for c in ("i", "ii"):
        for kind in ("reg", "tab"):
            arrays = {"pairs": pairs, "blocks": blocks, "q": np.array(CURVE_Q)}
            for qi in range(len(CURVE_Q)):
                arrays[f"pred_{qi}"] = C[c][kind][qi]
                arrays[f"failed_{qi}"] = C[c]["failed"][qi]
            name = f"rq3gjt_curve_{c}_{kind}.npz"
            np.savez_compressed(os.path.join(args.out_dir, name), **arrays)
            files[name] = fileSha(os.path.join(args.out_dir, name))
            shapes[name] = [list(C[c][kind][qi].shape) for qi in range(len(CURVE_Q))]
    for kind in ("reg", "tab"):
        name = f"rq3gjt_curve_iii_{kind}.npz"
        np.savez_compressed(os.path.join(args.out_dir, name), pairs=pairs, blocks=blocks, n=np.array([-1 if n is None else n for n in CURVE_N]),
                            pred=C["iii"][kind], failed=C["iii"]["failed"], capped=C["iii"]["capped"])
        files[name], shapes[name] = fileSha(os.path.join(args.out_dir, name)), list(C["iii"][kind].shape)
    coefs = pd.DataFrame(F.rows).assign(criteria_sha256=sha)
    coefs.to_csv(os.path.join(args.out_dir, "rq3gjt_coefficients.csv"), index=False)
    files["rq3gjt_coefficients.csv"] = fileSha(os.path.join(args.out_dir, "rq3gjt_coefficients.csv"))
    manifest = {"criteria_sha256": sha, "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
                "fake_choices": ctx.fake, "files": files, "shapes": shapes, "sources": gjr_shas,
                "checks": [{**c, "通過": bool(c["通過"])} for c in checks],
                "null_usage_records": [list(x) for x in ctx.usage_null],
                "columns": ["host", "dataset", "test_group", "menu", "version", "subset_donor", "split", "pred_acc"],
                "note": "切法 ① ② 的 split = 前 20 次切分的第幾次（每次一個配適）；③ ④ ⑥ = 200 次切分。曲線的 npz：pred_{q 的索引} "
                        "(重複, 區塊, (菜單, 版本, 供體), 前 20 次切分)；(iii) pred (n 的索引, 切分, 區塊, (菜單, 版本, 供體))，只有第 r 次切分有值。"}
    with open(os.path.join(args.out_dir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def readPredictions(ctx: Ctx, args) -> tuple:
    """讀預測檔；每個檔的 sha256 要等於 manifest，不符就停。"""
    with open(os.path.join(args.out_dir, MANIFEST), encoding="utf-8") as f:
        manifest = json.load(f)
    for name, sha in manifest["files"].items():
        if fileSha(os.path.join(args.out_dir, name)) != sha:
            raise SystemExit(f"❌ {name} does not match its sha256 in {MANIFEST}; predictions must not change after they are written")
    for path, sha in manifest["sources"].items():
        if fileSha(path) != sha:
            raise SystemExit(f"❌ {path} changed since the predictions were written")
    if manifest.get("fake_choices") != ctx.fake:
        raise SystemExit("❌ the prediction files were written with another --fake-choices setting")
    P = {}
    for name in manifest["files"]:
        if not name.startswith("rq3gjt_pred_"):
            continue
        _, _, cut, layer, kind = name.replace(".csv.gz", "").split("_")
        frame = pd.read_csv(os.path.join(args.out_dir, name), keep_default_na=False).sort_values(
            ["host", "dataset", "test_group", "menu", "version", "subset_donor", "split"])
        m = {}
        for (h, d, sc, menu, version, g), grp in frame.groupby(["host", "dataset", "test_group", "menu", "version", "subset_donor"], sort=False):
            m.setdefault((h, d), {})[(sc, menu, version, g)] = grp.pred_acc.to_numpy(dtype=float)
        P[(cut, layer, kind)] = m
    C = {}
    for c in ("i", "ii"):
        C[c] = {"failed": {}}
        for kind in ("reg", "tab"):
            z = np.load(os.path.join(args.out_dir, f"rq3gjt_curve_{c}_{kind}.npz"))
            if [tuple(x) for x in z["pairs"].tolist()] != [tuple(map(str, p)) for p in ctx.pairs]:
                raise SystemExit(f"❌ rq3gjt_curve_{c}_{kind}.npz: pair order differs")
            C[c][kind] = {qi: z[f"pred_{qi}"] for qi in range(len(CURVE_Q))}
            C[c]["failed"] = {qi: z[f"failed_{qi}"] for qi in range(len(CURVE_Q))}
    C["iii"] = {}
    for kind in ("reg", "tab"):
        z = np.load(os.path.join(args.out_dir, f"rq3gjt_curve_iii_{kind}.npz"))
        C["iii"][kind], C["iii"]["failed"], C["iii"]["capped"] = z["pred"], z["failed"], z["capped"]
    coefs = pd.read_csv(os.path.join(args.out_dir, "rq3gjt_coefficients.csv"), keep_default_na=False, na_values=[""])
    return manifest, P, C, coefs


# ------------------------------------------------------------------
# 評分
# ------------------------------------------------------------------
class Long:
    """rq3gjt_blocks.csv：長表（量、切法、層、方法、範圍、區塊、值）。"""

    def __init__(self):
        self.rows = []

    def add(self, quantity: str, cut: str, layer: str, method: str, scope: str, b: tuple, value: float):
        self.rows.append({"quantity": quantity, "cut": cut, "layer": layer, "method": method, "scope": scope, "host": b[0], "dataset": b[1],
                          "value": float(value)})


def curveMetrics(ctx: Ctx, arr: np.ndarray, failed: np.ndarray, rows: np.ndarray | None = None) -> dict:
    """
    曲線的一個點：arr (重複, 區塊, 配對, 切分) → {區塊: 五個指標（對成功的重複平均）}；(iii) 用 rows 指定每個重複對應的切分。
    回傳 {"blocks": {b: {指標: 值}}, "per_rep": {b: [每個成功重複的指標]}}。
    """
    pidx = {p: i for i, p in enumerate(ctx.pairs)}
    s_idx = np.array([pidx[k] for k in ctx.subs])
    o_idx = np.array([pidx[(k[0], "orig", k[2])] for k in ctx.subs])
    out = {"blocks": {}, "per_rep": {}}
    for bi, b in enumerate(BLOCKS):
        A = ctx.actual[b]
        reps = []
        for k in range(arr.shape[0]):
            if failed[k, bi]:
                continue
            if rows is None:
                a = arr[k, bi]
                blk, _ = metricsArr(a[s_idx], a[o_idx], A["act_s"][:, :CURVE_SPLITS], A["act_o"][:, :CURVE_SPLITS])
            else:
                a = arr[k, bi][:, None]
                r = rows[k]
                blk, _ = metricsArr(a[s_idx], a[o_idx], A["act_s"][:, [r]], A["act_o"][:, [r]])
            reps.append(blk)
        out["per_rep"][b] = reps
        out["blocks"][b] = {m: (float(np.mean([x[m] for x in reps])) if reps else float("nan")) for m, _ in METRICS}
    return out


def directAsk(ctx: Ctx) -> dict:
    """第 8 節 7：從評分半 ∩ 子集二抽 q 的題目直接用既有紀錄算；誤差 = 和整個評分半 ∩ 子集二的絕對差。"""
    G3 = ctx.G3
    out = {}
    for q in DIRECT_Q:
        acc = {b: {"e2": [], "e4": [], "calls": [], "tin": [], "tout": []} for b in BLOCKS}
        for r in range(CURVE_SPLITS):
            samp = directSamples(q, r, ctx.blocks)
            for b in BLOCKS:
                ev = ctx.evs[b]
                e2, e4 = [], []
                calls = tin = tout = 0.0
                for menu in MENU_ORDER:
                    it_o = ev.items[(menu, "orig")]
                    union = np.union1d(samp[(*b, DONORS[0])], samp[(*b, DONORS[1])])
                    sel = it_o.idx[0][np.isin(it_o.cols, union)]
                    calls += len(sel)
                    tin += ctx.usage_in[sel].sum()
                    tout += ctx.usage_out[sel].sum()
                for menu, version, g in ctx.subs:
                    S = samp[(*b, g)]
                    it_s, it_o = ev.items[(menu, version)], ev.items[(menu, "orig")]
                    a_s, a_o = it_s.actual[S].mean(), it_o.actual[S].mean()
                    f_s, f_o = ev.act[(menu, version, g)][r], ev.act[(menu, "orig", g)][r]
                    e2.append(100 * abs((a_s - a_o) - (f_s - f_o)))
                    e4.append(100 * abs(a_s - f_s))
                    sel = it_s.idx[0][np.isin(it_s.cols, S)]
                    calls += len(sel)
                    tin += ctx.usage_in[sel].sum()
                    tout += ctx.usage_out[sel].sum()
                acc[b]["e2"].append(e2)
                acc[b]["e4"].append(e4)
                acc[b]["calls"].append(calls)
                acc[b]["tin"].append(tin)
                acc[b]["tout"].append(tout)
        out[q] = {b: {"err_change": float(np.mean(np.mean(acc[b]["e2"], axis=0))), "err_abs": float(np.mean(np.mean(acc[b]["e4"], axis=0))),
                      "calls": float(np.mean(acc[b]["calls"])), "tokens_in": float(np.mean(acc[b]["tin"])),
                      "tokens_out": float(np.mean(acc[b]["tout"]))} for b in BLOCKS}
    return out


def hitRates(ctx: Ctx, ps: np.ndarray, po: np.ndarray, b: tuple) -> dict:
    """第 8 節 12：每個（菜單、供體）三條 path 的效果（200 次切分的分子總和 ÷ 分母總和），預測最高 / 最低的命中。"""
    A = ctx.actual[b]
    num_p = (ps - po).sum(axis=1)
    num_a = (A["act_s"] - A["act_o"]).sum(axis=1)
    den = A["den"].sum(axis=1)
    if (den <= 0).any():
        raise SystemExit(f"❌ {b}: a substitution has a non-positive denominator sum")
    eff_p, eff_a = num_p / den, num_a / den
    groups = defaultdict(list)
    for i, (menu, version, g) in enumerate(ctx.subs):
        groups[(menu, g)].append(i)
    out = {}
    for (menu, g), idx in groups.items():
        if len(idx) != 3:
            raise SystemExit(f"❌ {b} {menu} {g}: expected 3 substitutions, got {len(idx)}")
        out[(menu, g)] = (hitRate(eff_p[idx], eff_a[idx], True), hitRate(eff_p[idx], eff_a[idx], False))
    return out


def score(ctx: Ctx, args, sha: str, confirmed: str, started: float):
    elapsed = elapsedFn(started)
    manifest, P, C, coefs = readPredictions(ctx, args)
    gjr, gjr_shas = loadGjrPredictions(args.gjr_dir)
    if any(manifest["sources"].get(k) != v for k, v in gjr_shas.items()):
        raise SystemExit("❌ the RQ3-GJR prediction files differ from the ones recorded in the manifest")
    P[("c3", "C", "reg")], P[("c3", "C", "tab")] = gjr["reg"], gjr["tab"]
    told = toldPredictions(ctx)
    gjr_blocks = pd.read_csv(os.path.join(args.gjr_dir, "rq3gjr_blocks.csv")).set_index(["model", "dataset"])
    gjr_coef = pd.read_csv(os.path.join(args.gjr_dir, "rq3gjr_coefficients.csv"))
    print(f"\n評分（預測檔的 sha256 已核對；{elapsed()}）")
    L = Long()
    plan = ctx.plan

    # ---------------- 第 6 節的指標：總表 ----------------
    runs = [(c, l, k) for c, l in MAIN_RUNS for k in ("reg", "tab")] + [("c3", "C", "reg"), ("c3", "C", "tab")]
    R_of = {"c1": CURVE_SPLITS, "c2": CURVE_SPLITS}
    MET = {}
    sub_rows = []
    for cut, layer, kind in runs:
        res = runMetrics(ctx, P[(cut, layer, kind)], cut, R_of.get(cut, 200))
        MET[(cut, layer, kind)] = res
        for b in BLOCKS:
            blk, sub, keys, _ = res[b]
            for m, _ in METRICS:
                L.add(m, cut, layer, METHOD_NAME[(layer, kind)], "all", b, blk[m])
            L.add("n_subs", cut, layer, METHOD_NAME[(layer, kind)], "all", b, blk["n_subs"])
            L.add("spearman_undefined_splits", cut, layer, METHOD_NAME[(layer, kind)], "all", b, blk["spearman_undefined"])
            if cut == "c4":
                for group, g in blk["groups"].items():
                    for m, _ in METRICS:
                        L.add(m, cut, layer, METHOD_NAME[(layer, kind)], group, b, g[m])
            for n, (scope, i) in enumerate(keys):
                menu, version, g = ctx.subs[i]
                path = version.split("-", 1)[0]
                sub_rows.append({"cut": cut, "layer": layer, "method": METHOD_NAME[(layer, kind)], "test_group": scope, "host": b[0],
                                 "dataset": b[1], "menu": menu, "version": version, "path": path, "donor": g,
                                 "role": "lone" if path == plan.menus[menu]["lone"] else "pair",
                                 **{k: float(v[n]) for k, v in sub.items()}})
    T_OLD = runMetrics(ctx, told, "c3", 200)
    for b in BLOCKS:
        for m, _ in METRICS:
            L.add(m, "c3", "T_old", "T_old", "all", b, T_OLD[b][0][m])
    S = lambda key, m: summ([MET[key][b][0][m] for b in BLOCKS])
    print(f"  總表（{elapsed()}）")

    # ---------------- 判定一 ----------------
    A3 = MET[("c3", "A", "reg")]
    j1 = {}
    for b in BLOCKS:
        ps, po = A3[b][3]
        ps_d = perSplitDict(ctx, b, ps, po)
        act = splitEffects(plan, ps_d, "act")
        pred = splitEffects(plan, ps_d, "pred")
        j1[b] = {"delta": float((act[f"E_J_{CLEAR}"] - pred[f"E_J_{CLEAR}"]).mean()), "act": float(act[f"E_J_{CLEAR}"].mean()),
                 "pred": float(pred[f"E_J_{CLEAR}"].mean())}
        L.add("J1_Delta_E", "c3", "A", "M3h", CLEAR, b, j1[b]["delta"])
        L.add("J1_E_J_actual", "c3", "A", "M3h", CLEAR, b, j1[b]["act"])
        L.add("J1_E_J_pred", "c3", "A", "M3h", CLEAR, b, j1[b]["pred"])
    sJ1 = summ([j1[b]["delta"] for b in BLOCKS])
    stJ1 = fourState(sJ1, THRESHOLD)

    # ---------------- 曲線（第 8 節 6）與判定二 ----------------
    CM = {}
    for c in ("i", "ii"):
        for kind in ("reg", "tab"):
            for qi, q in enumerate(CURVE_Q):
                CM[(c, kind, q)] = curveMetrics(ctx, C[c][kind][qi], C[c]["failed"][qi] if kind == "reg" else
                                                np.zeros(C[c]["failed"][qi].shape, dtype=bool))
    for kind in ("reg", "tab"):
        for ni, n in enumerate(CURVE_N):
            failed = C["iii"]["failed"][ni] if kind == "reg" else np.zeros(C["iii"]["failed"][ni].shape, dtype=bool)
            CM[("iii", kind, n)] = curveMetrics(ctx, C["iii"][kind][ni], failed, rows=np.arange(CURVE_SPLITS))
    print(f"  曲線（{elapsed()}）")
    n_failed = {}
    for c in ("i", "ii"):
        for qi, q in enumerate(CURVE_Q):
            n_failed[(c, q)] = int(C[c]["failed"][qi].sum())
    for ni, n in enumerate(CURVE_N):
        n_failed[("iii", n)] = int(C["iii"]["failed"][ni].sum())
    full = {b: CM[("i", "reg", 1.0)]["blocks"][b]["err_change"] for b in BLOCKS}
    T = 0.2 * float(np.mean(list(full.values())))
    Gq = {}
    for q in CURVE_Q[:-1]:
        if n_failed[("i", q)] > 5:
            Gq[q] = None
            continue
        per = CM[("i", "reg", q)]["per_rep"]
        vals = [float(np.mean([x["err_change"] for x in per[b]])) - full[b] for b in BLOCKS]
        Gq[q] = (summ(vals), vals)
    if Gq[0.01] is None:
        sJ2, stJ2 = None, UNDETERMINED
    else:
        sJ2 = Gq[0.01][0]
        stJ2 = fourState(sJ2, T)
        for i, b in enumerate(BLOCKS):
            L.add("J2_G_1pct", "curve_i", "A", "M3h", "all", b, Gq[0.01][1][i])
    within = [q for q in CURVE_Q[:-1] if Gq[q] is not None and Gq[q][0]["ci_low"] >= -T and Gq[q][0]["ci_high"] <= T]
    start_q = next((q for q in CURVE_Q[:-1] if Gq[q] is not None and all(x in within for x in CURVE_Q[CURVE_Q.index(q):-1])), None)

    # ---------------- 第 8 節 7 ----------------
    DIR = directAsk(ctx)
    print(f"  直接問一部分題目（{elapsed()}）")

    # ---------------- 曲線的呼叫數與 tokens ----------------
    def curveCost(cut: str, layer: str, key: str, x) -> dict:
        c = coefs[(coefs.cut == cut) & (coefs.layer == layer)]
        c = c[c[key].astype(str) == str(x)] if key else c
        per_b = c.groupby(["judge", "target"])[["n_calls", "n_groups", "tokens_in", "tokens_out"]].mean()
        return {k: float(per_b[k].mean()) for k in ("n_calls", "n_groups", "tokens_in", "tokens_out")}

    curve_rows = []
    for c, layer in (("i", "A"), ("ii", "C")):
        for q in CURVE_Q:
            cost = curveCost(f"curve_{c}", layer, "q", q)
            for kind in ("reg", "tab"):
                for b in BLOCKS:
                    curve_rows.append({"curve": c, "layer": layer, "method": METHOD_NAME[(layer, kind)], "x": q, "host": b[0], "dataset": b[1],
                                       **CM[(c, kind, q)]["blocks"][b], "failed_fits_at_point": n_failed[(c, q)] if kind == "reg" else 0,
                                       "calls_per_block_mean": cost["n_calls"], "valid_per_block_mean": cost["n_groups"],
                                       "tokens_in_per_block_mean": cost["tokens_in"], "tokens_out_per_block_mean": cost["tokens_out"]})
    for n in CURVE_N:
        cost = curveCost("curve_iii", "A", "n", "all" if n is None else n)
        for kind in ("reg", "tab"):
            for b in BLOCKS:
                curve_rows.append({"curve": "iii", "layer": "A", "method": METHOD_NAME[("A", kind)], "x": "all" if n is None else n, "host": b[0],
                                   "dataset": b[1], **CM[("iii", kind, n)]["blocks"][b], "failed_fits_at_point": n_failed[("iii", n)] if kind == "reg" else 0,
                                   "calls_per_block_mean": cost["n_calls"], "valid_per_block_mean": cost["n_groups"],
                                   "tokens_in_per_block_mean": cost["tokens_in"], "tokens_out_per_block_mean": cost["tokens_out"]})
    for q in DIRECT_Q:
        m = DIR[q]
        for b in BLOCKS:
            curve_rows.append({"curve": "direct", "layer": "C", "method": "direct", "x": q, "host": b[0], "dataset": b[1],
                               "err_change": m[b]["err_change"], "err_abs": m[b]["err_abs"], "calls_per_block": m[b]["calls"],
                               "tokens_in_per_block": m[b]["tokens_in"], "tokens_out_per_block": m[b]["tokens_out"]})
    curves_df = pd.DataFrame(curve_rows)

    # ---------------- 第 8 節 12 命中率 ----------------
    HIT = {}
    for layer, kind in (("A", "reg"), ("A", "tab"), ("B", "reg"), ("B", "tab"), ("C", "reg"), ("C", "tab")):
        res = MET[("c3", layer, kind)]
        for b in BLOCKS:
            ps, po = res[b][3]
            HIT[(layer, kind, b)] = hitRates(ctx, ps, po, b)
            for scope in ["all"] + GROUP_ORDER:
                keys = [k for k in HIT[(layer, kind, b)] if scope == "all" or k[0] in GROUP_MENUS[scope]]
                L.add("hit_top", "c3", layer, METHOD_NAME[(layer, kind)], scope, b, np.mean([HIT[(layer, kind, b)][k][0] for k in keys]))
                L.add("hit_bottom", "c3", layer, METHOD_NAME[(layer, kind)], scope, b, np.mean([HIT[(layer, kind, b)][k][1] for k in keys]))
    print(f"  命中率（{elapsed()}）")

    # ---------------- 寫檔 ----------------
    blocks_df = pd.DataFrame(L.rows).assign(criteria_sha256=sha)
    blocks_df.to_csv(os.path.join(args.out_dir, "rq3gjt_blocks.csv"), index=False)
    pd.DataFrame(sub_rows).assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gjt_substitutions.csv"), index=False)
    curves_df.assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gjt_curves.csv"), index=False)
    figA(ctx, MET, os.path.join(args.out_dir, "fig_a_gain"), sha)
    figB(MET, os.path.join(args.out_dir, "fig_b_cuts"), sha)
    figC(CM, DIR, coefs, n_failed, curveCost, os.path.join(args.out_dir, "fig_c_curves"), sha)

    report(ctx, args, sha, confirmed, manifest, MET, T_OLD, S, j1, sJ1, stJ1, CM, Gq, T, full, sJ2, stJ2, start_q, n_failed, DIR, HIT,
           coefs, gjr_blocks, gjr_coef, curveCost)
    print(f"\n判定一 Δ_E {cellText(sJ1)} → {stJ1}；判定二 G_1% {cellText(sJ2) if sJ2 else '—'}（T = {T:.3f}）→ {stJ2}（{elapsed()}）")
    print(f"💾 {args.out_dir}/report.md 與其他輸出")


# ------------------------------------------------------------------
# 圖
# ------------------------------------------------------------------
def saveFig(fig, path: str, sha: str):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", facecolor=SURFACE, metadata={"Subject" if ext == "pdf" else "Description": f"rq3gjt_criteria.md sha256 {sha}"})
    plt.close(fig)


def figA(ctx: Ctx, MET: dict, path: str, sha: str):
    """(a) 切法 ③：每個替換的預測多的對實際多的（對 200 次切分平均），A、B、C 三格（regression）。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    data = {l: {b: MET[("c3", l, "reg")][b][1] for b in BLOCKS} for l in ("A", "B", "C")}
    allv = np.concatenate([np.concatenate([d[b]["gain_pred"], d[b]["gain_act"]]) for d in data.values() for b in BLOCKS])
    lo, hi = float(allv.min()), float(allv.max())
    titles = {"A": "A: original records only, M3h + hypothetical fix", "B": "B: all 7 versions, M3 + hypothetical fix (donor = 1)",
              "C": "C: all 7 versions, M3 + real substituted candidate"}
    for ax, l in zip(axes, ("A", "B", "C")):
        for host, color, marker in zip(HOSTS, SERIES, ("o", "s")):
            x = np.concatenate([data[l][b]["gain_pred"] for b in BLOCKS if b[0] == host])
            y = np.concatenate([data[l][b]["gain_act"] for b in BLOCKS if b[0] == host])
            ax.scatter(x, y, s=11, color=color, marker=marker, alpha=0.5, edgecolors="none", label=f"{MODEL_LABELS[host]} ({len(x)})", zorder=3)
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", zorder=2, label="predicted = actual")
        styleAxes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_title(titles[l], color=INK, fontsize=9, loc="left")
        ax.set_xlabel("predicted Judge gain, pp", color=MUTED, fontsize=9)
    axes[0].set_ylabel("actual Judge gain, pp", color=MUTED, fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("RQ3-GJT (a): predicted vs actual gain per substitution, leave-one-dataset-out (evaluation half, mean of 200 splits)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    saveFig(fig, path, sha)


def figB(MET: dict, path: str, sha: str):
    """(b) 總表的指標 2 與指標 4：橫軸 = 切法 ①②③④⑥，A 層與 C 層（regression），95% t 區間。"""
    cuts = ["c1", "c2", "c3", "c4", "c6"]
    labels = ["(1) per dataset", "(2) pooled", "(3) leave-one-dataset", "(4) leave-one-group", "(6) other judge"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(cuts))
    for ax, (m, title) in zip(axes, (("err_change", "metric 2: error of the gain per substitution, pp"),
                                     ("err_abs", "metric 4: error of the accuracy per substitution, pp"))):
        for li, (layer, name) in enumerate((("A", "A (original records only)"), ("C", "C (real candidate)"))):
            s = [summ([MET[(c, layer, "reg")][b][0][m] for b in BLOCKS]) for c in cuts]
            mean = np.array([v["mean"] for v in s])
            err = np.array([[v["mean"] - v["ci_low"] for v in s], [v["ci_high"] - v["mean"] for v in s]])
            xs = x + (li - 0.5) * 0.12
            ax.errorbar(xs, mean, yerr=err, color=SERIES[li], linewidth=1.0, marker="o", markersize=4, capsize=2.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, label=name, zorder=3)
        styleAxes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, color=INK, rotation=15)
        ax.set_ylim(bottom=0)
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("RQ3-GJT (b): regression error by split (8 blocks, mean and 95% t interval)", color=INK, fontsize=10, x=0.01, ha="left")
    saveFig(fig, path, sha)


def figC(CM: dict, DIR: dict, coefs: pd.DataFrame, n_failed: dict, curveCost, path: str, sha: str):
    """(c) 第 8 節 6、7 的曲線：橫軸 = 每個區塊的呼叫數（對數）；regression 實線、比例表虛線；直接問一部分題目一條。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    series = [("i", "A", "q", CURVE_Q, "(i) A, leave-one-dataset"), ("ii", "C", "q", CURVE_Q, "(ii) C, leave-one-dataset"),
              ("iii", "A", "n", CURVE_N, "(iii) A, same dataset")]
    for ax, (m, title) in zip(axes, (("err_change", "metric 2: error of the gain per substitution, pp"),
                                     ("err_abs", "metric 4: error of the accuracy per substitution, pp"))):
        for si, (c, layer, key, xs, name) in enumerate(series):
            calls = [curveCost(f"curve_{c}", layer, key, ("all" if x is None else x))["n_calls"] for x in xs]
            for kind, style in (("reg", "-"), ("tab", "--")):
                ys = []
                for x in xs:
                    if kind == "reg" and n_failed[(c, x)] > 5:
                        ys.append(np.nan)
                    else:
                        ys.append(np.mean([CM[(c, kind, x)]["blocks"][b][m] for b in BLOCKS]))
                ax.plot(calls, ys, linestyle=style, color=LINE_COLORS[si], linewidth=1.0, marker="o" if kind == "reg" else None,
                        markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0,
                        label=f"{name}, {'regression' if kind == 'reg' else 'ratio table'}", zorder=3)
            ax.annotate(name.split(",")[0], (calls[-1], ys[-1]), xytext=(5, 3), textcoords="offset points", color=INK, fontsize=8)
        dx = [float(np.mean([DIR[q][b]["calls"] for b in BLOCKS])) for q in DIRECT_Q]
        dy = [float(np.mean([DIR[q][b][m] for b in BLOCKS])) for q in DIRECT_Q]
        ax.plot(dx, dy, color=LINE_COLORS[3], linewidth=1.0, marker="D", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0,
                label="direct: ask the judge on q of the items (C setting)", zorder=3)
        ax.annotate("direct", (dx[-1], dy[-1]), xytext=(5, 3), textcoords="offset points", color=INK, fontsize=8)
        styleAxes(ax)
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("judge calls per block (training records, or calls asked directly)", color=MUTED, fontsize=9)
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    axes[1].legend(loc="upper right", frameon=False, fontsize=7.5, labelcolor=INK)
    fig.suptitle("RQ3-GJT (c): error vs number of judge calls (first 20 splits; mean of 8 blocks)", color=INK, fontsize=10, x=0.01, ha="left")
    saveFig(fig, path, sha)


# ------------------------------------------------------------------
# report.md
# ------------------------------------------------------------------
def report(ctx, args, sha, confirmed, manifest, MET, T_OLD, S, j1, sJ1, stJ1, CM, Gq, T, full, sJ2, stJ2, start_q, n_failed, DIR, HIT,
           coefs, gjr_blocks, gjr_coef, curveCost):
    G3 = ctx.G3
    out = [f"判定標準 `{os.path.join(OUT_DIR, CRITERIA_FILE)}` 的 sha256 `{sha}`；確認：{confirmed}", "",
           "# RQ3-GJT：不用真的去改 path 的裁判預測——準確度、成本與泛化", "",
           f"產生：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}（`scripts/analysis_rq3gjt/judge_transfer.py`，`Analysis/judgeTransfer.py`，"
           "沿用 `Analysis/judgeRegression.py`）。全程離線，沒有呼叫 API，沒有修改任何既有的輸出。預測檔寫於 "
           f"{manifest['generated_at']}，評分前已核對 sha256。", ""]
    if ctx.fake is not None:
        out += [f"**⚠️ 乾跑：裁判的有效選擇換成隨機的候選（seed {ctx.fake}）。以下數字沒有意義，只用來測試程式。**", ""]
    out += ["單位：偏差、誤差、D_P、G 是 pp；E_J 是「那條 path 每進步 10pp，Judge 多幾個 pp」。區塊 = 宿主 × 資料集，共 8 個；"
            "區間 = 95% t 區間（自由度 = 區塊數 − 1）。每格寫成「平均 [95% 區間] 為正的區塊數」。", ""]

    # 1. 讀了哪些檔案
    out += ["## 1. 讀了哪些檔案", "",
            f"- RQ3-GJ 的 K = 3 Judge 紀錄 `{args.gj_dir}/judge_outputs/{{宿主}}/{{資料集}}/{{菜單}}__{{版本}}.json`（{len(G3.keys):,} 份）："
            "`presentation_order`、`trace.choice`、`trace.chosen_arm`、`final_answer`、`call.usage_in`、`call.usage_out`。",
            f"- `{args.armdir}` 的 4 個模型 × 4 個資料集 × 12 條 path：`raw_text`（tokens 用 tiktoken `o200k_base`）、`parsed_answer`、`gold`；"
            f"`{args.aggdir}`：切分（`makeSplits` 200 次）。讀法同 RQ3-GJR（`scripts/analysis_rq3gjr/judge_regression.py` 的 `loadData`）。",
            "- RQ3-GJ 的計畫（`Analysis/judgeSubstitution.py`：菜單、版本、每題的順序）；`rq3gj_predictions_lodo.csv.gz`（T_old；sha256 已核對 "
            "RQ3-GJ 的 manifest）。",
            f"- RQ3-GJR：`{args.gjr_dir}/rq3gjr_predictions_M3.csv.gz`、`rq3gjr_predictions_T_new.csv.gz`（切法 ③ 的 C 層；sha256 已核對 RQ3-GJR 的 "
            "manifest）、`rq3gjr_blocks.csv`、`rq3gjr_coefficients.csv`（第 10 節 1、2 與第 8 節 8 的並排）。",
            "- 三份判定標準檔（RQ3-GJ、RQ3-GJR、RQ3-G）的 sha256 和本檔第 2 節相同。只讀。", ""]

    # 2. 第零階段
    a1 = []
    for hi, h in enumerate(HOSTS):
        for di, d in enumerate(DATASETS):
            m = (G3.judge == hi) & (G3.dataset == di)
            a1.append({"裁判": h, "資料集": d, "原本的版本 呼叫": f"{int((m & ctx.orig).sum()):,}", "原本的版本 有效": f"{int((m & ctx.orig & G3.valid).sum()):,}",
                       "7 個版本 呼叫": f"{int(m.sum()):,}", "7 個版本 有效": f"{int((m & G3.valid).sum()):,}"})
    a1.append({"裁判": "合計", "資料集": "", "原本的版本 呼叫": f"{int(ctx.orig.sum()):,}", "原本的版本 有效": f"{int((ctx.orig & G3.valid).sum()):,}",
               "7 個版本 呼叫": f"{len(G3):,}", "7 個版本 有效": f"{int(G3.valid.sum()):,}"})
    out += ["## 2. 第零階段", "",
            "呼叫數與有效選擇數（由這次讀到的紀錄重數，應和判定標準附錄 A.1 相同）。訓練組數、因素、tokens、抽樣、計算時間見判定標準附錄 A.2–A.6。", "",
            md(pd.DataFrame(a1), text=True), "",
            f"沒有 API 計費值的紀錄：{len(ctx.usage_null)} 筆（{'；'.join(' '.join(map(str, x)) for x in ctx.usage_null) or '無'}），tokens 記 0。", ""]

    # 3. 檢查
    out += ["## 3. 開跑前的檢查（第 10 節）", "",
            md(pd.DataFrame([{"項": c["項"], "內容": c["內容"], "結果": c["結果"], "通過": "✅" if c["通過"] else "❌"} for c in manifest["checks"]]),
               text=True), ""]

    # 4. 判定
    b_tab = pd.DataFrame([{"宿主": MODEL_LABELS[b[0]], "資料集": b[1], "實際的 E_J": f"{j1[b]['act']:+.2f}", "A 層預測的 E_J": f"{j1[b]['pred']:+.2f}",
                           "Δ_E": f"{j1[b]['delta']:+.2f}"} for b in BLOCKS])
    out += ["## 4. 兩項判定（A 層、M3h、切法 ③）", "",
            "### 判定一：只用原本的資料，估得出「落單那條和相像那一對差多少」嗎（明顯落單組 10 份，門檻 0.5）", "",
            f"- Δ_E = 實際的 E_J − A 層預測的 E_J（每次切分在評分半上算，200 次平均）：**{cellText(sJ1)}** → **{stJ1}**。{READ_J1[stJ1]}", "",
            md(b_tab, text=True), "",
            "### 判定二：只用一小部分紀錄時，估變化還一樣準嗎（曲線 (i)，前 20 次切分）", ""]
    if sJ2 is None:
        out += [f"- 1% 那一點有 {n_failed[('i', 0.01)]} 次無法配適（超過 5 次）→ **無法判定**，只列次數。", ""]
    else:
        out += [f"- 用全部時的誤差（q = 100%，指標 2）8 個區塊平均 {np.mean(list(full.values())):.3f}pp → 門檻 T = 0.2 × 它 = **{T:.3f}pp**。",
                f"- G_1% = 抽 1% 時的誤差 − 用全部時的誤差（20 次重複的平均）：**{cellText(sJ2, 3)}** → **{stJ2}**。{READ_J2[stJ2]}"]
        if stJ2 == FORWARD:
            out.append("- 曲線上開始落在 ±T 內的比例（只描述）：" + (f"q = {100 * start_q:g}%。" if start_q is not None else "到 30% 都沒有落在 ±T 內。"))
        out.append("")
    gq_rows = []
    for q in CURVE_Q[:-1]:
        if Gq[q] is None:
            gq_rows.append({"q": f"{100 * q:g}%", "G_q": f"無法配適 {n_failed[('i', q)]} 次（超過 5 次，不報）", "整個區間在 ±T 內": ""})
        else:
            s = Gq[q][0]
            gq_rows.append({"q": f"{100 * q:g}%", "G_q": cellText(s, 3), "整個區間在 ±T 內": "是" if s["ci_low"] >= -T and s["ci_high"] <= T else "否"})
    out += ["各比例的 G_q（同樣的算法；只描述）：", "", md(pd.DataFrame(gq_rows), text=True), ""]

    # 5. 只報告的量
    out += ["## 5. 只報告的量（第 8 節）", ""]
    # 8.1 總表
    rows = []
    for cut in ("c1", "c2", "c3", "c4", "c6"):
        for layer in ("A", "C"):
            for kind in ("reg", "tab"):
                key = (cut, layer, kind)
                rows.append({"切法": CUT_LABEL[cut], "層": layer, "方法": METHOD_NAME[(layer, kind)],
                             **{label: (f"{S(key, m)['mean']:+.3f}" if m == "spearman" else cellText(S(key, m))) for m, label in METRICS}})
        if cut == "c3":
            for kind in ("reg", "tab"):
                key = ("c3", "B", kind)
                rows.append({"切法": "③", "層": "B", "方法": METHOD_NAME[("B", kind)],
                             **{label: (f"{S(key, m)['mean']:+.3f}" if m == "spearman" else cellText(S(key, m))) for m, label in METRICS}})
    rows.append({"切法": "⑤ 跨 K（照抄 RQ3-GJR，K = 2）", "層": "C", "方法": "M3", METRICS[0][1]: "—", METRICS[1][1]: "—", METRICS[2][1]: "—",
                 METRICS[3][1]: cellText(summ(gjr_blocks.k2_MAE)), METRICS[4][1]: cellText(summ(gjr_blocks.k2_D_P))})
    rows.append({"切法": "③（歷史結果）", "層": "T_old", "方法": "RQ3-GJ 的比例表",
                 **{label: (f"{summ([T_OLD[b][0][m] for b in BLOCKS])['mean']:+.3f}" if m == "spearman" else cellText(summ([T_OLD[b][0][m] for b in BLOCKS])))
                    for m, label in METRICS}})
    out += ["### 8.1 總表", "",
            "切法 ①、② 用前 20 次切分；③、④、⑥ 用 200 次。④ 每個區塊 240 個替換（GS-181 在中間組與英文落單組各算一次，各用那一次的模型）。"
            "③ 的 C 層讀 RQ3-GJR 的存檔。⑤ 只有 K = 2 的指標 4、5（RQ3-GJR 已有的數字）。排名相關報 8 個區塊的平均，8 個值在下表。", "",
            md(pd.DataFrame(rows), text=True), ""]
    sp_rows = []
    for cut in ("c1", "c2", "c3", "c4", "c6"):
        for layer in (("A", "B", "C") if cut == "c3" else ("A", "C")):
            for kind in ("reg", "tab"):
                v = [MET[(cut, layer, kind)][b][0]["spearman"] for b in BLOCKS]
                sp_rows.append({"切法": CUT_LABEL[cut], "層": layer, "方法": METHOD_NAME[(layer, kind)], "平均": f"{np.mean(v):+.3f}",
                                "8 個值（GPT 四個資料集、Qwen 四個資料集）": "、".join(f"{x:+.2f}" for x in v)})
    out += ["排名相關的 8 個值（資料集順序 mmlu、mathqa、truthfulqa、commonsenseqa）：", "", md(pd.DataFrame(sp_rows), text=True), ""]
    g_rows = []
    for layer in ("A", "C"):
        for kind in ("reg", "tab"):
            for group in GROUP_ORDER:
                g_rows.append({"層": layer, "方法": METHOD_NAME[(layer, kind)], "測試組": group,
                               **{label: cellText(summ([MET[("c4", layer, kind)][b][0]["groups"][group][m] for b in BLOCKS]))
                                  for m, label in METRICS if m != "spearman"}})
    out += ["④ 逐組（每組每個區塊 60 個替換）：", "", md(pd.DataFrame(g_rows), text=True), ""]

    # 8.2 三層的落差
    bias = {l: np.array([MET[("c3", l, "reg")][b][0]["bias"] for b in BLOCKS]) for l in ("A", "B", "C")}
    sA, sB, sC = summ(bias["A"]), summ(bias["B"]), summ(bias["C"])
    out += ["### 8.2 三層的落差（切法 ③，regression 的變化的偏差）", "",
            md(pd.DataFrame([{"量": "A 層的偏差", "值": cellText(sA)}, {"量": "B 層的偏差", "值": cellText(sB)}, {"量": "C 層的偏差", "值": cellText(sC)},
                             {"量": "A 到 B 的差（A − B）", "值": cellText(summ(bias["A"] - bias["B"]))},
                             {"量": "B 到 C 的差（B − C）", "值": cellText(summ(bias["B"] - bias["C"]))}]), text=True), "",
            f"讀法（只描述）：A 層低估變化 {sA['mean']:+.2f}pp；加入強模型寫的之後剩 {sB['mean']:+.2f}pp；用真的候選之後剩 {sC['mean']:+.2f}pp"
            "（正 = 實際多的比預測多的大，即低估）。", ""]

    # 8.3 E_J 與 D_JV
    ej_rows = []
    for layer in ("A", "B", "C"):
        for kind in ("reg", "tab"):
            res = MET[("c3", layer, kind)]
            diffs = defaultdict(list)
            for b in BLOCKS:
                ps, po = res[b][3]
                d_ = perSplitDict(ctx, b, ps, po)
                act, pred = splitEffects(ctx.plan, d_, "act"), splitEffects(ctx.plan, d_, "pred")
                for name in [f"E_J_{g}" for g in GROUP_ORDER] + ["D_JV"]:
                    diffs[name].append(float((act[name] - pred[name]).mean()))
            ej_rows.append({"層": layer, "方法": METHOD_NAME[(layer, kind)],
                            **{(f"E_J {n[4:]}" if n != "D_JV" else "D_JV"): cellText(summ(v)) for n, v in diffs.items()}})
    out += ["### 8.3 預測的 E_J 與 D_JV：實際 − 預測（切法 ③）", "",
            "每次切分在評分半上算（RQ3-GJ 第 8 節 12；多數決用實際值），200 次平均，再逐區塊相減。", "", md(pd.DataFrame(ej_rows), text=True), ""]

    # 8.4 ① 對 ②
    rows4, reads4 = [], []
    for layer in ("A", "C"):
        for m, label in (("err_abs", "指標 4"), ("err_change", "指標 2")):
            d12 = np.array([MET[("c2", layer, "reg")][b][0][m] - MET[("c1", layer, "reg")][b][0][m] for b in BLOCKS])
            Tp = 0.2 * float(np.mean([MET[("c2", layer, "reg")][b][0][m] for b in BLOCKS]))
            s = summ(d12)
            hit = s["mean"] >= Tp and (s["ci_low"] > 0 or s["ci_high"] < 0)
            rows4.append({"層": layer, "量": f"② − ① 的{label}", "值": cellText(s, 3), "T'": f"{Tp:.3f}", "平均 ≥ T' 且區間不含 0": "是" if hit else "否"})
            if m == "err_abs":
                reads4.append(f"- {layer} 層（指標 4）：" + ("各資料集各自估係數時明顯較準，支持絕對正確率估不準是因為各資料集的係數不同。" if hit else "只列數字。"))
    out += ["### 8.4 各資料集各自一套係數有沒有比較準（① 對 ②，regression，前 20 次切分）", "", md(pd.DataFrame(rows4), text=True), "",
            *reads4, "- 註：① 的平均偏差接近 0 是同分布造成的，不當成證據。", ""]

    # 8.5 換裁判
    s6 = S(("c6", "A", "reg"), "bias")
    ok6 = s6["ci_low"] >= -THRESHOLD and s6["ci_high"] <= THRESHOLD
    c6 = coefs[coefs.cut == "c6"]
    crow = []
    for layer, model, names in (("A", "M3h", M3H), ("C", "M3", M3)):
        for n in names:
            r = {"層": layer, "模型": model, "係數": n}
            for h in HOSTS:
                x = c6[(c6.layer == layer) & (c6.judge == h)].iloc[0]
                r[f"{MODEL_LABELS[h]} 的紀錄"] = f"{x[f'b_{n}']:+.3f}（SE {x[f'se_{n}']:.3f}）"
            crow.append(r)
    out += ["### 8.5 換裁判（⑥）", "",
            f"- A 層 regression 的指標 1（變化的偏差）：{cellText(s6)} → " +
            ("95% 區間整個在 ±0.5pp 內：搬得到另一個相近的弱裁判。" if ok6 else "95% 區間不在 ±0.5pp 內，只列數字。") + "（任何情況都不寫「任何裁判」。）", "",
            "兩個裁判各自配適時的係數（⑥ 用到的四次配適，各用該裁判四個資料集的全部紀錄）：", "", md(pd.DataFrame(crow), text=True), ""]

    # 8.6 曲線
    def curveTable(c: str, layer: str, key: str, xs: list) -> pd.DataFrame:
        rows_ = []
        for x in xs:
            xv = "all" if x is None else x
            cost = curveCost(f"curve_{c}", layer, key, xv)
            for kind in ("reg", "tab"):
                nf = n_failed[(c, x)] if kind == "reg" else 0
                label = f"{100 * x:g}%" if key == "q" else ("全部" if x is None else str(x))
                r = {key: label, "方法": METHOD_NAME[(layer, kind)], "無法配適": nf}
                for m, lab in METRICS:
                    if nf > 5:
                        r[lab] = "—"
                    else:
                        s = summ([CM[(c, kind, x)]["blocks"][b][m] for b in BLOCKS])
                        r[lab] = f"{s['mean']:+.3f}" if m == "spearman" else cellText(s)
                r.update({"每個區塊的呼叫": f"{cost['n_calls']:,.0f}", "有效": f"{cost['n_groups']:,.0f}",
                          "輸入 tokens": f"{cost['tokens_in']:,.0f}", "輸出 tokens": f"{cost['tokens_out']:,.0f}"})
                rows_.append(r)
        return pd.DataFrame(rows_)

    capped = cappedText(ctx)
    out += ["### 8.6 訓練資料量的曲線（評分只用前 20 次切分）", "",
            "每個點：各區塊先對成功的重複平均，再報 8 個區塊的平均 [95% 區間] 為正的區塊數。呼叫數與 tokens = 預測一個區塊的那次配適的訓練紀錄"
            "（含沒有有效選擇的），對重複平均、再對 8 個區塊平均。無法配適 = 該點 160 次配適中的次數；超過 5 次的點不報指標。", "",
            "(i) A 層、切法 ③（判定二用它）：", "", md(curveTable("i", "A", "q", CURVE_Q), text=True), "",
            "(ii) C 層、切法 ③：", "", md(curveTable("ii", "C", "q", CURVE_Q), text=True), "",
            "(iii) A 層、切法 ①（目標資料集 H1_r 抽 n 題）：", "", md(curveTable("iii", "A", "n", CURVE_N), text=True), "",
            f"n 大於可抽的題數、改用全部的：{capped}。", ""]

    # 8.7 直接問
    drows = []
    for q in DIRECT_Q:
        drows.append({"q": f"{100 * q:g}%", "指標 2 的對應值": cellText(summ([DIR[q][b]["err_change"] for b in BLOCKS])),
                      "指標 4 的對應值": cellText(summ([DIR[q][b]["err_abs"] for b in BLOCKS])),
                      "評完一個區塊的呼叫": f"{np.mean([DIR[q][b]['calls'] for b in BLOCKS]):,.0f}",
                      "輸入 tokens": f"{np.mean([DIR[q][b]['tokens_in'] for b in BLOCKS]):,.0f}",
                      "輸出 tokens": f"{np.mean([DIR[q][b]['tokens_out'] for b in BLOCKS]):,.0f}"})
    c_full = {m: float(np.mean([CM[("ii", "reg", 1.0)]["blocks"][b][m] for b in BLOCKS])) for m in ("err_change", "err_abs")}
    lower = {m: [f"{100 * q:g}%（{np.mean([DIR[q][b]['calls'] for b in BLOCKS]):,.0f} 次）" for q in DIRECT_Q
                 if c_full[m] < np.mean([DIR[q][b][m] for b in BLOCKS])] for m in c_full}
    out += ["### 8.7 直接問一部分題目（只適用 C 層的情境）", "",
            "每個（區塊、供體、切分、q）抽一次，原本的版本與該供體的所有替換共用；誤差 = 樣本上的值和整個評分半 ∩ 子集二上的值的絕對差，"
            "每個替換對前 20 次切分平均、再對區塊內的替換平均。呼叫數 = 原本的版本在兩個供體樣本的聯集中有呼叫的題數 + 每個替換在它的樣本中有呼叫的題數。", "",
            md(pd.DataFrame(drows), text=True), "",
            f"讀法（只描述）：用全部紀錄的 C 層 regression（曲線 (ii) q = 100%）指標 2 是 {c_full['err_change']:.3f}pp、指標 4 是 {c_full['err_abs']:.3f}pp。"
            f"預測的誤差低於直接問的：指標 2 在 {('、'.join(lower['err_change']) or '沒有任何一個 q')}；指標 4 在 {('、'.join(lower['err_abs']) or '沒有任何一個 q')}。"
            "不寫「預測可以取代裁判呼叫」。", "",
            "A 層不畫這條線：還沒改任何 path 時，沒有改好的候選可以直接問。", ""]

    # 8.8 A 層的係數
    c3a = coefs[(coefs.cut == "c3") & (coefs.layer == "A")]
    rows8 = []
    for _, r in c3a.iterrows():
        rows8.append({"裁判": MODEL_LABELS[r.judge], "測試資料集": r.target, "訓練組數": f"{int(r.n_groups):,}", **{n: f"{r[f'b_{n}']:+.3f}" for n in M3H}})
    blk_a = coefs[coefs.cut == "block"]
    blk_r = gjr_coef[(gjr_coef.fit == "block") & (gjr_coef.model == "M3")]
    rows8b = []
    for n in M3:
        r = {"係數": n}
        if n in M3H:
            s = summ(blk_a[f"b_{n}"])
            r["A 層 M3h（原本的版本）"] = f"{s['mean']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] {s['n_positive']}/8"
        else:
            r["A 層 M3h（原本的版本）"] = "—"
        s = summ(blk_r[f"b_{n}"])
        r["RQ3-GJR M3（7 個版本）"] = f"{s['mean']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] {s['n_positive']}/8"
        rows8b.append(r)
    out += ["### 8.8 A 層的係數（只描述）", "", "切法 ③ 每次配適的 M3h 係數：", "", md(pd.DataFrame(rows8), text=True), "",
            "逐區塊配適（8 個區塊的平均 [95% 區間] 為正的區塊數），和 RQ3-GJR 的 M3 逐區塊係數並排：", "", md(pd.DataFrame(rows8b), text=True), ""]

    # 8.9 依被改的 path
    subA = MET[("c3", "A", "reg")]
    res_rows = []
    for path in M12:
        vals = []
        for b in BLOCKS:
            sub = subA[b][1]
            idx = [i for i, k in enumerate(ctx.subs) if k[1].split("-", 1)[0] == path]
            if idx:
                vals.append(float(np.mean(sub["gain_act"][idx] - sub["gain_pred"][idx])))
        if len(vals) == len(BLOCKS):
            res_rows.append({"被改的 path": path, "實際多的 − 預測多的": cellText(summ(vals)), "替換數 / 區塊": len(idx)})
        elif vals:
            res_rows.append({"被改的 path": path, "實際多的 − 預測多的": f"{np.mean(vals):+.2f}（{len(vals)} 個區塊）", "替換數 / 區塊": len(idx)})
    out += ["### 8.9 依被改的 path（A 層、M3h、切法 ③）", "", "每個區塊、每條 path 先對它的替換（菜單 × 供體）平均，再報 8 個區塊。", "",
            md(pd.DataFrame(res_rows), text=True), ""]

    # 8.10 分開
    rows10 = []
    for g in DONORS:
        dv = []
        for b in BLOCKS:
            ps, po = subA[b][3]
            d_ = perSplitDict(ctx, b, ps, po, donor=g)
            # 只用明顯落單組。單一供體時，其他組的分母總和可能剛好是 0（gpt4omini × commonsenseqa × deepseek 的英文落單組落單那條，
            # 200 次切分中 1 次），splitEffects 會對那一組除以 0；那些組不報告，所以只在這裡關掉警告。
            with np.errstate(divide="ignore", invalid="ignore"):
                act, pred = splitEffects(ctx.plan, d_, "act"), splitEffects(ctx.plan, d_, "pred")
            if not (np.isfinite(act[f"E_J_{CLEAR}"]).all() and np.isfinite(pred[f"E_J_{CLEAR}"]).all()):
                raise SystemExit(f"❌ {b} {g}: the clear-group denominator sum is 0 in some split")
            dv.append(float((act[f"E_J_{CLEAR}"] - pred[f"E_J_{CLEAR}"]).mean()))
        idx = [i for i, k in enumerate(ctx.subs) if k[2] == g]
        m1 = [float(subA[b][1]["bias"][idx].mean()) for b in BLOCKS]
        m2 = [float(subA[b][1]["err_change"][idx].mean()) for b in BLOCKS]
        rows10 += [{"分開": f"供體 {g}", "量": "判定一 Δ_E", "值": cellText(summ(dv))},
                   {"分開": f"供體 {g}", "量": "指標 1", "值": cellText(summ(m1))}, {"分開": f"供體 {g}", "量": "指標 2", "值": cellText(summ(m2))}]
    rows10b = []
    for h in HOSTS:
        bs = [b for b in BLOCKS if b[0] == h]
        for name, vals in (("判定一 Δ_E", [j1[b]["delta"] for b in bs]), ("指標 1", [subA[b][0]["bias"] for b in bs]),
                           ("指標 2", [subA[b][0]["err_change"] for b in bs])):
            v = np.array(vals)
            rows10b.append({"裁判": MODEL_LABELS[h], "量": name, "平均": pp(v.mean()), "4 個值": "、".join(pp(x) for x in v), "為正": f"{int((v > 0).sum())}/4"})
    out += ["### 8.10 兩個供體分開、兩個裁判分開（A 層、M3h、切法 ③；只描述）", "", md(pd.DataFrame(rows10), text=True), "",
            md(pd.DataFrame(rows10b), text=True), ""]

    # 8.11 圖
    out += ["### 8.11 圖", "",
            "- `fig_a_gain.{pdf,png}`：切法 ③，每個替換的預測多的對實際多的（對 200 次切分平均），A、B、C 三格（regression）。",
            "- `fig_b_cuts.{pdf,png}`：總表的指標 2 與指標 4，橫軸是切法 ①②③④⑥，A 層與 C 層（regression，95% t 區間）。",
            "- `fig_c_curves.{pdf,png}`：第 8 節 6、7 的曲線；橫軸 = 每個區塊的呼叫數（對數），regression 實線、比例表虛線；直接問一部分題目一條。", ""]

    # 8.12 命中率
    hit_rows = []
    for layer, kind in (("A", "reg"), ("A", "tab"), ("B", "reg"), ("B", "tab"), ("C", "reg"), ("C", "tab")):
        for scope in ["all"] + GROUP_ORDER:
            top, bot = [], []
            for b in BLOCKS:
                H_ = HIT[(layer, kind, b)]
                keys = [k for k in H_ if scope == "all" or k[0] in GROUP_MENUS[scope]]
                top.append(np.mean([H_[k][0] for k in keys]))
                bot.append(np.mean([H_[k][1] for k in keys]))
            hit_rows.append({"層": layer, "方法": METHOD_NAME[(layer, kind)], "範圍": "全部（39 份 × 2 個供體）" if scope == "all" else f"{scope}（10 份 × 2）",
                             "最值得改的命中率": pctText(summ(top)), "最不值得改的命中率": pctText(summ(bot))})
    top_A = summ([np.mean([v[0] for v in HIT[("A", "reg", b)].values()]) for b in BLOCKS])
    top_B = summ([np.mean([v[0] for v in HIT[("B", "reg", b)].values()]) for b in BLOCKS])
    top_C = summ([np.mean([v[0] for v in HIT[("C", "reg", b)].values()]) for b in BLOCKS])
    out += ["### 8.12 菜單內的命中率（切法 ③；只報告）", "",
            "每個（區塊、菜單、供體）三條 path 的效果 = 200 次切分的分子總和 ÷ 分母總和；命中 = Σ（預測最高的 k 條）(1/k) × [在實際最高的 m 條之中] × (1/m)，"
            "並列 = 相差 ≤ 1e-12。每個區塊 = 區塊內（菜單、供體）的平均；8 個區塊的平均 [95% 區間]。亂挑的期望是 33.3%。", "",
            md(pd.DataFrame(hit_rows), text=True), "",
            f"讀法（只描述）：只用原本的資料，預測最值得改的那條有 {100 * top_A['mean']:.1f}% 是真的最值得改的（亂挑是 33%）；"
            f"B 層 {100 * top_B['mean']:.1f}%、C 層 {100 * top_C['mean']:.1f}%（regression）。", ""]

    # 6. 結論
    out += ["## 6. 對照讀法的結論", "",
            f"- 判定一（Δ_E {cellText(sJ1)}，門檻 0.5）：**{stJ1}**。{READ_J1[stJ1]}",
            (f"- 判定二（G_1% {cellText(sJ2, 3)}，T = {T:.3f}pp）：**{stJ2}**。{READ_J2[stJ2]}" if sJ2 is not None else
             f"- 判定二：1% 那一點無法配適 {n_failed[('i', 0.01)]} 次（超過 5 次）→ **無法判定**，只列次數。"),
            "- 其餘各項只報告，不套四種狀態（第 8 節 4、5 的讀法照事先寫的區間條件）。", "",
            "事先寫下的限制（論文要照寫）：", "", *[f"- {x}" for x in LIMITS], "",
            "## 輸出檔", "",
            f"`rq3gjt_blocks.csv`（長表：量、切法、層、方法、範圍、區塊、值）、`rq3gjt_substitutions.csv`、`rq3gjt_coefficients.csv`（{len(coefs)} 次配適）、"
            "`rq3gjt_curves.csv`、預測檔（`rq3gjt_pred_*.csv.gz`、`rq3gjt_curve_*.npz`；不進 git）與 `rq3gjt_predictions_manifest.json`、三張圖。"
            "每份都記錄判定標準的 sha256。", ""]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def cappedText(ctx: Ctx) -> str:
    rows = []
    for ni, n in enumerate(CURVE_N):
        if n is None:
            continue
        for r in range(CURVE_SPLITS):
            _, capped = cutOneSamples(n, r, ctx.universe, ctx.splits)
            for b, c in capped.items():
                if c:
                    rows.append((n, b))
    if not rows:
        return "無"
    counts = defaultdict(int)
    for n, b in rows:
        counts[(n, b)] += 1
    return "；".join(f"n = {n}：{b[0]} {b[1]} {c} 次切分" for (n, b), c in counts.items())


# ------------------------------------------------------------------
def main():
    args = parseArgs()
    started = time.time()
    if args.fake_choices is not None and os.path.abspath(args.out_dir) == os.path.abspath(OUT_DIR):
        raise SystemExit("❌ --fake-choices is a dry run: give a different --out-dir")
    criteria = os.path.join(OUT_DIR, CRITERIA_FILE)
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet")
    sha = sha256(criteria)
    for path, expect in SOURCE_CRITERIA.items():
        if sha256(path) != expect:
            raise SystemExit(f"❌ {path} does not match the sha256 recorded in rq3gjt_criteria.md §2")
    ctx = Ctx(args)
    if args.score_only:
        score(ctx, args, sha, confirmed, started)
        return
    F = Fits(ctx)
    gjr, gjr_shas = loadGjrPredictions(args.gjr_dir)
    print("\n第 5 節的各切法（配適與預測，先放在記憶體）")
    M = mainCuts(ctx, F)
    checks = runChecks(ctx, F, M, gjr, args)
    print("\n第 10 節的檢查：")
    for c in checks:
        print(f"  {'✅' if c['通過'] else '❌'} {c['項']}. {c['內容']}\n       {c['結果']}")
    print(f"（{time.time() - started:.0f}s）")
    if not all(c["通過"] for c in checks):
        if ctx.fake is None:
            raise SystemExit("❌ 開跑前的檢查沒有全部通過：停，不寫任何輸出")
        print("⚠️  乾跑：隨機的選擇本來就重現不了 RQ3-GJR，檢查不過照樣繼續")
    if args.checks_only:
        return
    print("\n第 8 節 6 的曲線")
    C = curves(ctx, F, started)
    manifest = writePredictions(ctx, M, C, F, checks, gjr_shas, args, sha)
    print(f"💾 預測檔 {len(manifest['files'])} 份，sha256 在 {MANIFEST}（{time.time() - started:.0f}s）")
    score(ctx, args, sha, confirmed, started)


if __name__ == "__main__":
    main()
