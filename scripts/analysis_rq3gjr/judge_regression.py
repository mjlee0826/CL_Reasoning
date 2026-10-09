"""
judge_regression.py — RQ3-GJR：用 RQ3-GJ 的裁判紀錄做條件式 logit，預測裁判的選擇（result/analysis/rq3gjr/rq3gjr_criteria.md）

判定標準確認前拒跑。全程離線：不呼叫 API，不修改 RQ3-GJ / RQ1-KJ 的任何輸出（只讀；RQ3-GJ 的預測檔讀之前核對 sha256）。
    1. 讀 RQ3-GJ 的 K = 3、K = 2 紀錄與 RQ1-KJ 的 M12、M3L/M3S/M3P 紀錄，建出每次呼叫的候選因素（第 3 節）
    2. 第 10 節的六項檢查；任何一項不過就停，不寫任何輸出
    3. 正式計算：留一個資料集的 T_new、M0–M4（判定一、二）與第 7 節 1–10
    4. 輸出（--out-dir）：rq3gjr_blocks.csv、rq3gjr_substitutions.csv、rq3gjr_coefficients.csv、
       rq3gjr_predictions_{模型}.csv.gz 與 rq3gjr_predictions_manifest.json、兩張圖、report.md

用法（從 repo root）：
    conda run -n clreasoning python scripts/analysis_rq3gjr/judge_regression.py --checks-only    # 只跑第 10 節的檢查
    conda run -n clreasoning python scripts/analysis_rq3gjr/judge_regression.py
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import gzip
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_rq3gj"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken

from Analysis.preregistration import sha256, confirmationLine
from Analysis.blockStats import summarize, fourState, FORWARD, REVERSE, EQUIVALENT, UNDETERMINED
from Analysis.menuVote import DATASETS
from Analysis.pathImprove import M12, HOSTS, DONORS
from Analysis.judgeSubstitution import GROUP_ORDER, GROUP_MENUS, MENU_ORDER, PAIRS, RQ1KJ_JUDGE_DIR, OUT_DIR as GJ_DIR
from Analysis.judgeSubstitutionPredict import M3_MENUS, NONE
from Analysis.judgeSubstitutionStats import loadPredictions, voteScore, spearmanSplits, splitEffects
from Analysis.judgeRegression import (BASE, MODELS, Groups, textLengths, pathAccuracy, versionCodes, buildGroups, design, fit,
                                      fitConditionalLogit, predictCorrect, scenarioProbability, cellsOf, ratioTable, tableCorrect,
                                      versionItems, itemVector, halfMean)
from path_improve_gj import md, pp, MODEL_LABELS, styleAxes, SERIES, INK, MUTED, GRID, SURFACE
import run_gj_judge as runner

OUT_DIR = "result/analysis/rq3gjr"
CRITERIA_FILE = "rq3gjr_criteria.md"
GJ_CRITERIA_SHA = "4e96b91475c30a490285a91faf6a878583d2691518c6ccff0b520dd6c154e700"
THRESHOLD = 0.5
LODO_MODELS = ["M0", "M1", "M2", "M3", "M4", "M3chars"]
BLOCK_MODELS = ["M3", "M4", "M5a", "M5b"]
PRED_MODELS = ["T_new", "M0", "M1", "M2", "M3", "M4", "M3chars"]          # 寫預測檔的模型（T_old 是 RQ3-GJ 已存的）
SIDE = ["T_old", "T_new", "M0", "M1", "M2", "M3", "M4"]                    # 第 7 節 1
EJ_MODELS = ["T_old", "T_new", "M0", "M1", "M2", "M3"]                      # 第 7 節 2
WITHIN_SPLITS = 20                                                           # 第 7 節 6
CHECK6_GROUPS, CHECK6_TOL, CHECK1_TOL, CHECK4_TOL = 5000, 1e-6, 1e-9, 1e-9
CHECK1_TARGET = {"D_P": ("+0.95", "+0.09", "+1.80"), "D_P_lodo": ("+0.91", "-0.09", "+1.92")}
EXPECTED_CALLS = {3: 936_892, 2: 70_034}
KINDS = [("en", "英文完整"), ("trans", "翻譯"), ("short", "短推理")]
PRED_FILE = "rq3gjr_predictions_{}.csv.gz"
MANIFEST = "rq3gjr_predictions_manifest.json"
READ_J1 = {
    EQUIVALENT: "誤差在 0.5pp 內；放進這些因素後，預測可以用在沒看過的資料集（仍需要標準答案）。",
    FORWARD: "仍然低估。",
    REVERSE: "高估。",
    UNDETERMINED: "只列數字。",
}
READ_J2 = {
    FORWARD: "誰寫的、文字類型、長度讓預測實質變準。",
    EQUIVALENT: "多放這些因素沒有實質幫助；比例表不準的原因不在這裡。",
    REVERSE: "多放因素反而變差。",
    UNDETERMINED: "只列數字。",
}


def parseArgs():
    parser = ArgumentParser(description="RQ3-GJR (rq3gjr_criteria.md)")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--gj-dir", default=GJ_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--checks-only", action="store_true", help="Run the §10 checks, print them, write nothing")
    return parser.parse_args()


def fileSha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def summ(values) -> dict:
    return summarize(np.asarray(values, dtype=float))


def ciRow(name: str, s: dict, judged: bool = False) -> dict:
    row = {"量": name, "平均": pp(s["mean"]), "SE": f"{s['se']:.2f}", "95% 區間": f"[{pp(s['ci_low'])}, {pp(s['ci_high'])}]",
           "為正的區塊": f"{s['n_positive']}/{s['n_blocks']}"}
    if judged:
        row["狀態"] = fourState(s, THRESHOLD)
    return row


def fmt(x: float, digits: int = 3) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{digits}f}"


# ------------------------------------------------------------------
# 資料
# ------------------------------------------------------------------
def loadData(args) -> dict:
    t0 = time.time()
    ctx = runner.loadAll(type("A", (), {"armdir": args.armdir, "aggdir": args.aggdir, "out_dir": args.gj_dir})())
    coded, plan, gold_text = ctx["coded"], ctx["plan"], ctx["gold_text"]
    saved = loadPredictions(args.gj_dir)
    lengths = textLengths(args.armdir, coded, tiktoken.get_encoding("o200k_base"))
    pathacc = pathAccuracy(coded)
    print(f"載入 arm、計畫、RQ3-GJ 的預測（sha256 已核對）、tokens（{time.time() - t0:.0f}s）")
    src3 = [((v.host, v.dataset, v.menu, v.version, None), v.path(args.gj_dir), v.codes, v.slot, v.donor)
            for v in plan.versions if v.K == 3 and v.participates]
    src2 = [((v.host, v.dataset, v.menu, v.version, v.order), v.path(args.gj_dir), v.codes, v.slot, v.donor)
            for v in plan.versions if v.K == 2 and v.participates]
    src12 = [((h, d, "M12", "orig", None), os.path.join(RQ1KJ_JUDGE_DIR, h, d, "M12.json"), M12, None, None) for h in HOSTS for d in DATASETS]
    srcm3 = [((h, d, menu, "orig", None), os.path.join(RQ1KJ_JUDGE_DIR, h, d, f"{menu}.json"), codes, None, None)
             for h in HOSTS for d in DATASETS for menu, codes in M3_MENUS.items()]
    G3 = buildGroups(src3, coded, gold_text, lengths, pathacc, 3)
    G2 = buildGroups(src2, coded, gold_text, lengths, pathacc, 2)
    G12 = buildGroups(src12, coded, gold_text, lengths, pathacc, 12)
    Gm = buildGroups(srcm3, coded, gold_text, lengths, pathacc, 3)
    print(f"呼叫：K = 3 {len(G3):,}、K = 2 {len(G2):,}、K = 12 {len(G12):,}、RQ1-KJ M3 菜單 {len(Gm):,}（{time.time() - t0:.0f}s）")

    # 逐區塊、逐版本的題目
    k3 = {key: i for i, key in enumerate(G3.keys)}
    k2 = {key: i for i, key in enumerate(G2.keys)}
    k12 = {key: i for i, key in enumerate(G12.keys)}
    items3, items2, items12, info = {}, {}, {}, {}
    for h in HOSTS:
        for d in DATASETS:
            H = coded[(h, d)]
            sub2 = {g: H.sub & coded[(g, d)].sub for g in DONORS}
            union = sub2[DONORS[0]] | sub2[DONORS[1]]
            info[(h, d)] = {"sub2": sub2, "union": union}
            for v in plan.versions:
                if v.host != h or v.dataset != d or not v.participates or (v.K == 2 and v.order != "A"):
                    continue
                codes = versionCodes(coded, h, d, v.codes, v.slot, v.donor)
                domain = union if v.version == "orig" else sub2[v.donor]
                if v.K == 3:
                    items3[(h, d, v.menu, v.version)] = versionItems(codes, H.gold, G3, [k3[(h, d, v.menu, v.version, None)]], domain)
                else:
                    items2[(h, d, v.menu, v.version)] = versionItems(codes, H.gold, G2, [k2[(h, d, v.menu, v.version, o)] for o in "AB"], domain)
            items12[(h, d)] = versionItems(versionCodes(coded, h, d, M12), H.gold, G12, [k12[(h, d, "M12", "orig", None)]], H.sub)
    versions3 = {}
    for v in plan.versions:
        if v.K == 3 and v.participates:
            versions3.setdefault((v.host, v.dataset), []).append(v)
    print(f"逐版本的題目：K = 3 {len(items3)}、K = 2 {len(items2)}、K = 12 {len(items12)} 個版本（{time.time() - t0:.0f}s）")
    return {**ctx, "saved": saved, "lengths": lengths, "pathacc": pathacc, "G3": G3, "G2": G2, "G12": G12, "Gm": Gm,
            "items3": items3, "items2": items2, "items12": items12, "info": info, "versions3": versions3}


def splitWeights(groups: Groups, splits: dict, rows: slice | np.ndarray) -> np.ndarray:
    """(R, G)：每組的題目在選擇半 H1_r 為 1。"""
    first = next(iter(splits.values()))
    W = np.zeros((first[rows].shape[0], len(groups)))
    for di, d in enumerate(DATASETS):
        m = groups.dataset == di
        W[:, m] = splits[d][rows][:, groups.col[m]]
    return W


# ------------------------------------------------------------------
# 一個區塊的評分：每個（版本、子集二的供體）在評分半 ∩ 子集二上的逐切分正確率
# ------------------------------------------------------------------
class BlockEval:
    def __init__(self, D: dict, h: str, d: str):
        self.h, self.d = h, d
        H = D["coded"][(h, d)]
        self.H = H
        self.sub2 = D["info"][(h, d)]["sub2"]
        self.mask = {g: ~D["splits"][d] & self.sub2[g][None, :] for g in DONORS}
        self.n2 = {g: self.mask[g].sum(axis=1) for g in DONORS}
        self.versions = D["versions3"][(h, d)]
        self.items = {(v.menu, v.version): D["items3"][(h, d, v.menu, v.version)] for v in self.versions}
        self.subs = [v for v in self.versions if v.version != "orig"]
        self.pairs = [(v.menu, v.version, v.donor) for v in self.subs] + [(m, "orig", g) for m in MENU_ORDER for g in DONORS]
        self.act = {k: halfMean(self.items[k[:2]].actual, self.mask[k[2]]) for k in self.pairs}
        self.vote = {k: halfMean(voteScore(self.items[k[:2]].codes, H.gold).astype(float), self.mask[k[2]]) for k in self.pairs}
        self.den = {}
        for v in self.subs:
            j = M12.index(v.codes[v.slot])
            m = self.mask[v.donor].astype(float)
            self.den[(v.menu, v.version)] = (m @ D["coded"][(v.donor, d)].correct[j].astype(float) - m @ H.correct[j].astype(float)) / m.sum(axis=1)

    def predFromGroups(self, p: np.ndarray) -> dict:
        """p：K = 3 每組的預測正確機率。回傳 {(菜單, 版本, 供體): (R,)}。"""
        vec = {k: itemVector(it, p) for k, it in self.items.items()}
        return {k: halfMean(vec[k[:2]], self.mask[k[2]]) for k in self.pairs}

    def perSplit(self, pred: dict) -> dict:
        """RQ3-GJ 的 splitEffects / spearmanSplits 用的格式：{(菜單, 版本): {act_s, act_o, pred_s, pred_o, vote_s, vote_o, den}}。"""
        out = {}
        for v in self.subs:
            g = v.donor
            out[(v.menu, v.version)] = {"act_s": self.act[(v.menu, v.version, g)], "act_o": self.act[(v.menu, "orig", g)],
                                        "pred_s": pred[(v.menu, v.version, g)], "pred_o": pred[(v.menu, "orig", g)],
                                        "vote_s": self.vote[(v.menu, v.version, g)], "vote_o": self.vote[(v.menu, "orig", g)],
                                        "den": self.den[(v.menu, v.version)]}
        return out

    def subStats(self, pred: dict) -> dict:
        """{(菜單, 版本): (D_P, MAE, 預測多的, 實際多的)}（pp，對切分平均）。"""
        out = {}
        for v in self.subs:
            k, g = (v.menu, v.version), v.donor
            dp = 100 * (self.act[(*k, g)] - pred[(*k, g)])
            out[k] = (float(dp.mean()), float(np.abs(dp).mean()), float(100 * (pred[(*k, g)] - pred[(v.menu, "orig", g)]).mean()),
                      float(100 * (self.act[(*k, g)] - self.act[(v.menu, "orig", g)]).mean()))
        return out


# ------------------------------------------------------------------
# 第 10 節的檢查
# ------------------------------------------------------------------
def runChecks(D: dict, args) -> list:
    checks = []
    G3, G2, G12, Gm, saved = D["G3"], D["G2"], D["G12"], D["Gm"], D["saved"]
    gj_blocks = pd.read_csv(os.path.join(args.gj_dir, "rq3gj_blocks.csv")).set_index(["model", "dataset"])

    # 1. 用存檔的預測重現 RQ3-GJ 的判定三與留一個資料集
    vals = {"D_P": [], "D_P_lodo": []}
    worst = 0.0
    for h in HOSTS:
        for d in DATASETS:
            ev = BlockEval(D, h, d)
            for name, src in (("D_P", "pred"), ("D_P_lodo", "lodo")):
                pred = {k: saved[src][(h, d, k[0], k[1], k[2])] for k in ev.pairs}
                v = float(np.mean([s[0] for s in ev.subStats(pred).values()]))
                vals[name].append(v)
                worst = max(worst, abs(v - gj_blocks.loc[(h, d), name]))
    got = {name: summ(v) for name, v in vals.items()}
    text = {name: (pp(s["mean"]), pp(s["ci_low"]), pp(s["ci_high"])) for name, s in got.items()}
    ok = worst <= CHECK1_TOL and all(text[n] == CHECK1_TARGET[n] for n in text)
    checks.append({"項": 1, "內容": "用存檔的預測重現 RQ3-GJ 判定三與留一個資料集",
                   "結果": f"判定三 {text['D_P'][0]}（{text['D_P'][1]} 到 {text['D_P'][2]}）；留一個資料集 {text['D_P_lodo'][0]}"
                           f"（{text['D_P_lodo'][1]} 到 {text['D_P_lodo'][2]}）；逐區塊和 rq3gj_blocks.csv 的最大差 {worst:.1e}",
                   "通過": ok})

    # 2. 呼叫數
    with open(os.path.join(args.gj_dir, "report.md"), encoding="utf-8") as f:
        report = f.read()
    m = re.search(r"K = 3 ([\d,]+)、K = 2 ([\d,]+)）", report)
    gj_calls = {3: int(m.group(1).replace(",", "")), 2: int(m.group(2).replace(",", ""))} if m else {}
    ok = len(G3) == EXPECTED_CALLS[3] == gj_calls.get(3) and len(G2) == EXPECTED_CALLS[2] == gj_calls.get(2)
    checks.append({"項": 2, "內容": "K = 3、K = 2 的呼叫數和 RQ3-GJ 的 report 相同",
                   "結果": f"K = 3 {len(G3):,}、K = 2 {len(G2):,}；RQ3-GJ report：K = 3 {gj_calls.get(3, 0):,}、K = 2 {gj_calls.get(2, 0):,}",
                   "通過": ok})

    # 3. RQ3-GJ 第 8 節 3 的採用率表
    cell = cellsOf(G3)
    act = G3.actual
    lone_donor = G3.X[np.arange(len(G3)), np.clip(cell % 3, 0, 2), BASE.index("donor")] > 0.5
    mine = []
    for hi, h in enumerate(HOSTS):
        for t in (1, 2, 3):
            for p in (1, 2, 3):
                m = (G3.judge == hi) & (cell == (t - 1) * 3 + (p - 1))
                row = [MODEL_LABELS[h], str(t), str(p), str(int(m.sum())), f"{100 * act[m].mean():.1f}%"]
                if t == 2:
                    for flag in (False, True):
                        mm = m & (lone_donor == flag)
                        row.append(f"{100 * act[mm].mean():.1f}%（{int(mm.sum())}）")
                else:
                    row += ["", ""]
                mine.append(row)
    sec = report.split("### 8.3")[1].split("###")[0]
    theirs = [[c.strip() for c in line.strip().strip("|").split("|")] for line in sec.splitlines()
              if line.startswith("|") and not line.startswith("|:") and "裁判" not in line]
    diff = [(a, b) for a, b in zip(mine, theirs) if a != b]
    ok = len(mine) == len(theirs) == 18 and not diff
    checks.append({"項": 3, "內容": "用紀錄重算 RQ3-GJ 第 8 節 3 的採用率表",
                   "結果": f"18 列逐格比對：{'全部相同' if ok else f'{len(diff)} 列不同：{diff[:3]}'}", "通過": ok})

    # 4. T_new 的算法：用 RQ1-KJ 選擇半的紀錄逐切分估表，預測值要等於 rq3gj_predictions.csv.gz
    cell_m = cellsOf(Gm)
    q = {}
    for hi, h in enumerate(HOSTS):
        sel = Gm.judge == hi
        g = Gm.subset(sel)
        q[h] = ratioTable(cell_m[sel], g.actual, splitWeights(g, D["splits"], slice(None)))[0]          # (R, 9)
    worst, n_cmp = 0.0, 0
    for h in HOSTS:
        for d in DATASETS:
            ev = BlockEval(D, h, d)
            for (menu, version), it in ev.items.items():
                cvec = np.full(len(ev.H.item_ids), NONE)
                cvec[it.cols] = cell[it.idx[0]]
                onehot = np.eye(9)[np.clip(cvec, 0, 8)] * (cvec >= 0)[:, None]
                for g in ([next(v.donor for v in ev.versions if (v.menu, v.version) == (menu, version))] if version != "orig" else DONORS):
                    m = ev.mask[g].astype(float)
                    pred = (m @ it.known + ((m @ onehot) * q[h]).sum(axis=1)) / m.sum(axis=1)
                    worst = max(worst, float(np.abs(pred - saved["pred"][(h, d, menu, version, g)]).max()))
                    n_cmp += len(pred)
    checks.append({"項": 4, "內容": "T_new 的算法用 RQ1-KJ 選擇半的紀錄估表，預測值和 rq3gj_predictions.csv.gz 逐筆相同（差 ≤ 1e-9）",
                   "結果": f"{n_cmp:,} 筆，最大差 {worst:.1e}", "通過": worst <= CHECK4_TOL and n_cmp == sum(len(v) for v in saved["pred"].values())})

    # 5. 資料的結構
    parts = []
    struct_ok = True
    for name, G in (("K = 3", G3), ("K = 2", G2), ("K = 12", G12), ("RQ1-KJ M3", Gm)):
        same = bool((G.actual == G.final_ok).all())
        struct_ok &= same
        parts.append(f"{name}：{len(G):,} 組，有效選擇 {int(G.valid.sum()):,} 組各恰好選一個（choice 與 chosen_arm 一致），"
                     f"選到的候選答對 = 紀錄的 final_answer 等於 gold：{'全部相同' if same else '有不同'}")
    leak = []
    for hi, h in enumerate(HOSTS):
        for di, d in enumerate(DATASETS):
            train = G3.valid & (G3.judge == hi) & (G3.dataset != di)
            test = (G3.judge == hi) & (G3.dataset == di)
            if (G3.dataset[train] == di).any() or (train & test).any():
                leak.append((h, d))
    struct_ok &= not leak
    parts.append(f"留一個資料集的 8 個訓練集裡沒有測試資料集的紀錄：{'是' if not leak else f'否 {leak}'}")
    checks.append({"項": 5, "內容": "資料的結構", "結果": "；".join(parts), "通過": struct_ok})

    # 6. 自己寫的最大概似 vs statsmodels ConditionalLogit（M4，每個裁判隨機 5,000 組）
    from statsmodels.discrete.conditional_models import ConditionalLogit
    worst, detail = 0.0, []
    for hi, h in enumerate(HOSTS):
        idx = np.flatnonzero(G3.valid & (G3.judge == hi))
        pick = np.sort(idx[np.random.default_rng(0).choice(len(idx), CHECK6_GROUPS, replace=False)])
        g = G3.subset(pick)
        own = fit(g, "M4")
        Z = design(g.X, MODELS["M4"])
        y = np.zeros((len(g), 3))
        y[np.arange(len(g)), g.chosen] = 1
        # statsmodels 自己的牛頓法（它的 Hessian 是數值微分）。2026-10-09 第一次用 method="bfgs"：BFGS 在 |score| 約 0.12 時就停了，
        # 差 6.7e-4 / 1.6e-3 沒過；回報後經使用者確認改用牛頓法，門檻與抽樣不變。
        model = ConditionalLogit(y.ravel(), Z.reshape(-1, Z.shape[2]), groups=np.repeat(np.arange(len(g)), 3))
        res = model.fit(method="newton", tol=1e-12, maxiter=100, disp=0)
        diff = float(np.abs(own["beta"] - res.params).max())
        worst = max(worst, diff)
        detail.append(f"{MODEL_LABELS[h]} 最大差 {diff:.1e}（自己的牛頓法 {own['iterations']} 步；statsmodels 的 |score| 在自己的係數上 "
                      f"{np.abs(model.score(own['beta'])).max():.1e}）")
    checks.append({"項": 6, "內容": "自己寫的最大概似 vs statsmodels ConditionalLogit（M4，每個裁判 5,000 組，default_rng(0)；statsmodels 用 method=\"newton\"）",
                   "結果": "；".join(detail), "通過": worst <= CHECK6_TOL})
    return checks


# ------------------------------------------------------------------
# 正式計算
# ------------------------------------------------------------------
class Coefficients:
    """rq3gjr_coefficients.csv：每個模型、每次配適一列。"""
    def __init__(self):
        self.rows = []

    def add(self, kind: str, model: str, res: dict, **tags) -> dict:
        row = {"fit": kind, "model": model, **tags, "n_groups": res["n_groups"], "loglik": res["loglik"], "iterations": res["iterations"]}
        for name, b, se in zip(res["names"], res["beta"], res["se"]):
            row[f"b_{name}"], row[f"se_{name}"] = float(b), float(se)
        self.rows.append(row)
        return res


def lodoFits(D: dict, coef: Coefficients) -> dict:
    """留一個資料集：每個裁判 × 被留下的資料集，用另外三個資料集的 K = 3 有效紀錄配適；T_new 用同樣的訓練資料估 9 格表。"""
    G3, G2, G12 = D["G3"], D["G2"], D["G12"]
    cell3 = D["cell3"]
    p3 = {m: np.full(len(G3), np.nan) for m in PRED_MODELS}
    p2, p12 = np.full(len(G2), np.nan), np.full(len(G12), np.nan)
    tables, betas = {}, {}
    for hi, h in enumerate(HOSTS):
        for di, d in enumerate(DATASETS):
            t0 = time.time()
            train_mask = G3.valid & (G3.judge == hi) & (G3.dataset != di)
            if (G3.dataset[train_mask] == di).any():
                raise SystemExit("❌ the leave-one-dataset-out training set contains the test dataset")
            train = G3.subset(train_mask)
            test = (G3.judge == hi) & (G3.dataset == di)
            gtest = G3.subset(test)
            for m in LODO_MODELS:
                res = coef.add("lodo", m, fit(train, m), judge=h, test_dataset=d)
                betas[(h, d, m)] = res["beta"]
                p3[m][test] = predictCorrect(gtest, m, res["beta"])
            q, k, n = ratioTable(cell3[train_mask], train.actual)
            tables[(h, d)] = {"q": q, "k": k, "n": n}
            p3["T_new"][test] = tableCorrect(cell3[test], q)
            t2, t12 = (G2.judge == hi) & (G2.dataset == di), (G12.judge == hi) & (G12.dataset == di)
            p2[t2] = predictCorrect(G2.subset(t2), "M3", betas[(h, d, "M3")])
            p12[t12] = predictCorrect(G12.subset(t12), "M3", betas[(h, d, "M3")])
            print(f"  留一個資料集 {h:10s} {d:14s} 訓練 {len(train):,} 組（{time.time() - t0:.0f}s）")
    for name, arr in [*p3.items(), ("K2", p2), ("K12", p12)]:
        if not np.isfinite(arr).all():
            raise SystemExit(f"❌ {name}: some group has no prediction")
    return {"p3": p3, "p2": p2, "p12": p12, "tables": tables, "betas": betas}


def evaluateK3(D: dict, evs: dict, L: dict) -> dict:
    """判定一、二與第 7 節 1、2、7、8：逐區塊、逐替換。"""
    plan = D["plan"]
    block_rows, sub_rows, resid_rows, pred_parts = [], [], [], {m: [] for m in PRED_MODELS}
    for h in HOSTS:
        for d in DATASETS:
            ev = evs[(h, d)]
            preds = {"T_old": {k: D["saved"]["lodo"][(h, d, *k)] for k in ev.pairs}}
            for m in PRED_MODELS:
                preds[m] = ev.predFromGroups(L["p3"][m])
            stats = {m: ev.subStats(p) for m, p in preds.items()}
            per_split = {m: ev.perSplit(p) for m, p in preds.items()}
            act_eff = {k: v.mean() for k, v in splitEffects(plan, per_split["M3"], "act").items()}
            row = {"model": h, "dataset": d, "n_subs": len(ev.subs)}
            for m in preds:
                row[f"D_P_{m}"] = float(np.mean([s[0] for s in stats[m].values()]))
                row[f"MAE_{m}"] = float(np.mean([s[1] for s in stats[m].values()]))
                row[f"spearman_{m}"] = spearmanSplits(per_split[m])[0]
            for group in GROUP_ORDER:
                row[f"E_J_act_{group}"] = act_eff[f"E_J_{group}"]
            row["D_JV_act"] = act_eff["D_JV"]
            for m in EJ_MODELS:
                pe = {k: v.mean() for k, v in splitEffects(plan, per_split[m], "pred").items()}
                for group in GROUP_ORDER:
                    row[f"E_J_pred_{group}_{m}"] = pe[f"E_J_{group}"]
                row[f"D_JV_pred_{m}"] = pe["D_JV"]
            row["G"] = row["MAE_T_new"] - row["MAE_M3"]
            row["G_chars"] = row["MAE_T_new"] - row["MAE_M3chars"]
            for g in DONORS:
                keys = [(v.menu, v.version) for v in ev.subs if v.donor == g]
                tag = g.split(".")[0][:8]
                row[f"D_P_M3_{tag}"] = float(np.mean([stats["M3"][k][0] for k in keys]))
                row[f"G_{tag}"] = float(np.mean([stats["T_new"][k][1] - stats["M3"][k][1] for k in keys]))
            block_rows.append(row)
            for v in ev.subs:
                k = (v.menu, v.version)
                path = v.codes[v.slot]
                srow = {"K": 3, "host": h, "dataset": d, "menu": v.menu, "version": v.version, "path": path, "donor": v.donor,
                        "role": "lone" if path == plan.menus[v.menu]["lone"] else "pair",
                        "groups": "|".join(g for g in GROUP_ORDER if v.menu in GROUP_MENUS[g]), "n2": int(ev.sub2[v.donor].sum()),
                        "gain_actual_H2": stats["M3"][k][3]}
                for m in preds:
                    srow.update({f"D_P_{m}": stats[m][k][0], f"MAE_{m}": stats[m][k][1], f"gain_pred_H2_{m}": stats[m][k][2]})
                sub_rows.append(srow)
                resid_rows.append({"host": h, "dataset": d, "path": path, "residual": stats["M3"][k][3] - stats["M3"][k][2]})
            for m in PRED_MODELS:
                for (menu, version, g), vals in preds[m].items():
                    pred_parts[m].append(pd.DataFrame({"host": h, "dataset": d, "menu": menu, "version": version, "subset_donor": g,
                                                       "split": np.arange(len(vals)), "n2": ev.n2[g], "pred_acc": vals}))
            print(f"  評分 {h:10s} {d:14s} {len(ev.subs)} 個替換")
    return {"blocks": pd.DataFrame(block_rows), "subs": pd.DataFrame(sub_rows), "resid": pd.DataFrame(resid_rows),
            "pred_frames": {m: pd.concat(parts, ignore_index=True) for m, parts in pred_parts.items()}}


def blockFits(D: dict, coef: Coefficients) -> dict:
    """第 7 節 3：M3、M4、M5a、M5b 逐區塊配適；12 個情境機率（M4，M3 並列）。"""
    G3 = D["G3"]
    betas, scen = {}, []
    for hi, h in enumerate(HOSTS):
        for di, d in enumerate(DATASETS):
            inblock = (G3.judge == hi) & (G3.dataset == di)
            g = G3.subset(inblock & G3.valid)
            for m in BLOCK_MODELS:
                betas[(h, d, m)] = coef.add("block", m, fit(g, m), judge=h, dataset=d)["beta"]
            med = float(np.median(G3.X[inblock][:, :, BASE.index("loglen")]))
            for correct in (1, 0):
                for donor in (0, 1):
                    for kind, _ in KINDS:
                        scen.append({"judge": h, "dataset": d, "correct": correct, "donor": donor, "kind": kind, "loglen_median": med,
                                     **{f"p_{m}": scenarioProbability(m, betas[(h, d, m)], correct, donor, kind, med) for m in ("M4", "M3")}})
    return {"betas": betas, "scenarios": pd.DataFrame(scen)}


def evaluateK2(D: dict, L: dict, gj_blocks: pd.DataFrame) -> tuple:
    """第 7 節 4(a)：K = 3 訓練的 M3（留一個資料集）預測 K = 2 各版本；兩種順序平均。"""
    coded, plan, splits = D["coded"], D["plan"], D["splits"]
    rows, sub_rows, worst = [], [], 0.0
    for h in HOSTS:
        for d in DATASETS:
            H = coded[(h, d)]
            sub2 = D["info"][(h, d)]["sub2"]
            vs = [v for v in plan.versions if v.K == 2 and v.participates and v.order == "A" and v.host == h and v.dataset == d]
            vec = {v.menu + "|" + v.version: itemVector(D["items2"][(h, d, v.menu, v.version)], L["p2"]) for v in vs}
            act = {v.menu + "|" + v.version: D["items2"][(h, d, v.menu, v.version)].actual for v in vs}
            dps, maes, num_a, num_p, dens = [], [], [], [], []
            for v in [v for v in vs if v.version != "orig"]:
                k, o = v.menu + "|" + v.version, v.menu + "|orig"
                s2 = sub2[v.donor]
                mask = ~splits[d] & s2[None, :]
                dp = 100 * (halfMean(act[k], mask) - halfMean(vec[k], mask))
                j = M12.index(v.codes[v.slot])
                den = coded[(v.donor, d)].correct[j][s2].mean() - H.correct[j][s2].mean()
                na, npred = act[k][s2].mean() - act[o][s2].mean(), vec[k][s2].mean() - vec[o][s2].mean()
                dps.append(dp.mean())
                maes.append(np.abs(dp).mean())
                num_a.append(na)
                num_p.append(npred)
                dens.append(den)
                sub_rows.append({"K": 2, "host": h, "dataset": d, "menu": v.menu, "version": v.version, "path": v.codes[v.slot],
                                 "donor": v.donor, "n2": int(s2.sum()), "D_P_M3": float(dp.mean()), "MAE_M3": float(np.abs(dp).mean()),
                                 "gain_actual_sub2": 100 * na, "gain_pred_sub2_M3": 100 * npred, "den": den})
            eff_a, eff_p = 10 * np.sum(num_a) / np.sum(dens), 10 * np.sum(num_p) / np.sum(dens)
            worst = max(worst, abs(eff_a - gj_blocks.loc[(h, d), "k2_effect_J"]))
            rows.append({"model": h, "dataset": d, "k2_D_P": float(np.mean(dps)), "k2_MAE": float(np.mean(maes)),
                         "k2_effect_actual": eff_a, "k2_effect_pred": eff_p, "k2_effect_diff": eff_a - eff_p})
    return pd.DataFrame(rows), pd.DataFrame(sub_rows), worst


def evaluateK12(D: dict, L: dict) -> pd.DataFrame:
    """第 7 節 4(b)：同樣的 M3 預測 RQ1-KJ 的 M12 原本的版本；M12 子集、評分半、200 次切分平均。"""
    rows = []
    for h in HOSTS:
        for d in DATASETS:
            it = D["items12"][(h, d)]
            mask = ~D["splits"][d] & D["coded"][(h, d)].sub[None, :]
            a, p = halfMean(it.actual, mask), halfMean(itemVector(it, L["p12"]), mask)
            rows.append({"model": h, "dataset": d, "k12_actual": 100 * a.mean(), "k12_pred": 100 * p.mean(),
                         "k12_D_P": float((100 * (a - p)).mean()), "k12_MAE": float(np.abs(100 * (a - p)).mean()),
                         "k12_calls": len(it.cols)})
    return pd.DataFrame(rows)


def leaveOneGroup(D: dict, evs: dict, coef: Coefficients) -> pd.DataFrame:
    """第 7 節 5：四組菜單輪流當測試；M3 用其餘菜單（四個資料集）的有效紀錄配適。GS-181 在測試組時不進訓練。"""
    G3 = D["G3"]
    menu = G3.menu
    rows = []
    for group in GROUP_ORDER:
        test_menus = set(GROUP_MENUS[group])
        in_group = np.isin(menu, list(test_menus))
        p = np.full(len(G3), np.nan)
        for hi, h in enumerate(HOSTS):
            res = coef.add("leave_group", "M3", fit(G3.subset(G3.valid & (G3.judge == hi) & ~in_group), "M3"), judge=h, test_group=group)
            t = (G3.judge == hi) & in_group
            p[t] = predictCorrect(G3.subset(t), "M3", res["beta"])
        for h in HOSTS:
            for d in DATASETS:
                ev = evs[(h, d)]
                dps, maes = [], []
                for v in [v for v in ev.subs if v.menu in test_menus]:
                    vec = itemVector(ev.items[(v.menu, v.version)], p)
                    dp = 100 * (ev.act[(v.menu, v.version, v.donor)] - halfMean(vec, ev.mask[v.donor]))
                    dps.append(dp.mean())
                    maes.append(np.abs(dp).mean())
                rows.append({"model": h, "dataset": d, "group": group, "n_subs": len(dps), "D_P": float(np.mean(dps)),
                             "MAE": float(np.mean(maes))})
        print(f"  留一組菜單：{group}")
    return pd.DataFrame(rows)


def withinDataset(D: dict, evs: dict, coef: Coefficients) -> pd.DataFrame:
    """第 7 節 6：前 20 次切分；每次用四個資料集中題目在 H1_r 的有效紀錄配適 M3，在 H2_r 上評分。"""
    G3 = D["G3"]
    acc = {(h, d): {} for h in HOSTS for d in DATASETS}
    for r in range(WITHIN_SPLITS):
        h1 = splitWeights(G3, D["splits"], [r])[0] > 0.5
        for hi, h in enumerate(HOSTS):
            res = coef.add("within_dataset", "M3", fit(G3.subset(G3.valid & (G3.judge == hi) & h1), "M3"), judge=h, split=r)
            mine = G3.judge == hi
            p = np.full(len(G3), np.nan)
            p[mine] = predictCorrect(G3.subset(mine), "M3", res["beta"])
            for d in DATASETS:
                ev = evs[(h, d)]
                for v in ev.subs:
                    vec = itemVector(ev.items[(v.menu, v.version)], p)
                    m = ev.mask[v.donor][r]
                    dp = 100 * (ev.act[(v.menu, v.version, v.donor)][r] - vec[m].mean())
                    acc[(h, d)].setdefault((v.menu, v.version), []).append(dp)
        print(f"  同一個資料集內：切分 {r + 1}/{WITHIN_SPLITS}")
    rows = []
    for (h, d), subs in acc.items():
        rows.append({"model": h, "dataset": d, "within_D_P": float(np.mean([np.mean(x) for x in subs.values()])),
                     "within_MAE": float(np.mean([np.mean(np.abs(x)) for x in subs.values()]))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 圖
# ------------------------------------------------------------------
def saveFig(fig, path: str, sha: str):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", facecolor=SURFACE,
                    metadata={"Subject" if ext == "pdf" else "Description": f"rq3gjr_criteria.md sha256 {sha}"})
    plt.close(fig)


def plotPredictions(subs: pd.DataFrame, path: str, sha: str):
    """(a) 每個替換的預測多的 vs 實際多的正確率（評分半，對 200 次切分平均）：T_old、T_new、M3（留一個資料集）。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    s3 = subs[subs.K == 3]
    cols = ["gain_pred_H2_T_old", "gain_pred_H2_T_new", "gain_pred_H2_M3"]
    lo = float(min(s3[cols].min().min(), s3.gain_actual_H2.min()))
    hi = float(max(s3[cols].max().max(), s3.gain_actual_H2.max()))
    titles = ["T_old (saved RQ3-GJ table)", "T_new (table, same training data)", "M3 (conditional logit)"]
    for ax, col, title in zip(axes, cols, titles):
        for host, color, marker in zip(HOSTS, SERIES, ("o", "s")):
            s = s3[s3.host == host]
            ax.scatter(s[col], s.gain_actual_H2, s=11, color=color, marker=marker, alpha=0.5, edgecolors="none",
                       label=f"{MODEL_LABELS[host]} ({len(s)})", zorder=3)
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", zorder=2, label="predicted = actual")
        styleAxes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("predicted Judge gain, pp", color=MUTED, fontsize=9)
    axes[0].set_ylabel("actual Judge gain, pp", color=MUTED, fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("RQ3-GJR (a): predicted vs actual gain per substitution, leave-one-dataset-out (evaluation half, mean of 200 splits)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    saveFig(fig, path, sha)


def plotScenarios(scen: pd.DataFrame, path: str, sha: str):
    """(b) 第 7 節 3 的 12 個機率（M4，兩個裁判分開；各為 4 個區塊的平均）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    labels = [f"{w} / {k}" for w in ("host", "donor") for k in ("English full", "translated", "short (R)")]
    y = np.arange(len(labels))[::-1]
    for ax, h in zip(axes, HOSTS):
        s = scen[scen.judge == h].groupby(["correct", "donor", "kind"], as_index=False).p_M4.mean()
        for correct, color, marker, name in ((1, SERIES[0], "o", "lone candidate is correct"), (0, SERIES[1], "s", "lone candidate is wrong")):
            vals = [float(s[(s.correct == correct) & (s.donor == dn) & (s.kind == k)].p_M4.iloc[0]) for dn in (0, 1) for k, _ in KINDS]
            ax.scatter(vals, y, s=46, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.2, zorder=3, label=name)
        styleAxes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_xlim(0, 1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5, color=INK)
        ax.set_xlabel("probability the lone candidate is picked (M4)", color=MUTED, fontsize=9)
        ax.set_title(MODEL_LABELS[h], color=INK, fontsize=10, loc="left")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.17), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("RQ3-GJR (b): K = 3, lone candidate vs two host-written English candidates (mean of 3 positions and 4 blocks)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    saveFig(fig, path, sha)


def writeGz(frame: pd.DataFrame, path: str):
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(frame.to_csv(index=False, float_format="%.12g").encode())


def cellText(s: dict) -> str:
    return f"{pp(s['mean'])} [{pp(s['ci_low'])}, {pp(s['ci_high'])}] {s['n_positive']}/{s['n_blocks']}"


def formalRun(D: dict, args, checks: list, sha: str, confirmed: str, started: float):
    elapsed = lambda: f"{time.time() - started:.0f}s"
    os.makedirs(args.out_dir, exist_ok=True)
    gj_blocks = pd.read_csv(os.path.join(args.gj_dir, "rq3gj_blocks.csv")).set_index(["model", "dataset"])
    D["cell3"] = cellsOf(D["G3"])
    coef = Coefficients()
    print("\n正式計算")
    L = lodoFits(D, coef)
    print(f"留一個資料集的配適（{elapsed()}）")
    evs = {(h, d): BlockEval(D, h, d) for h in HOSTS for d in DATASETS}
    K3 = evaluateK3(D, evs, L)
    blocks, subs = K3["blocks"], K3["subs"]
    print(f"K = 3 的評分（{elapsed()}）")
    BF = blockFits(D, coef)
    print(f"逐區塊的配適（{elapsed()}）")
    k2_blocks, k2_subs, k2_worst = evaluateK2(D, L, gj_blocks)
    k12_blocks = evaluateK12(D, L)
    print(f"跨 K（{elapsed()}）")
    log = leaveOneGroup(D, evs, coef)
    print(f"留一組菜單（{elapsed()}）")
    within = withinDataset(D, evs, coef)
    print(f"同一個資料集內（{elapsed()}）")

    # ------------------------------------------------------------------
    # 輸出檔
    # ------------------------------------------------------------------
    log_wide = log.pivot(index=["model", "dataset"], columns="group", values=["D_P", "MAE"])
    log_wide.columns = [f"group_{a}_{b}" for a, b in log_wide.columns]
    blocks = (blocks.merge(k2_blocks, on=["model", "dataset"]).merge(k12_blocks, on=["model", "dataset"])
              .merge(log_wide.reset_index(), on=["model", "dataset"]).merge(within, on=["model", "dataset"]))
    blocks.assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gjr_blocks.csv"), index=False)
    all_subs = pd.concat([subs, k2_subs], ignore_index=True)
    all_subs.assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gjr_substitutions.csv"), index=False)
    coefs = pd.DataFrame(coef.rows)
    coefs.assign(criteria_sha256=sha).to_csv(os.path.join(args.out_dir, "rq3gjr_coefficients.csv"), index=False)
    files = {}
    for m, frame in K3["pred_frames"].items():
        name = PRED_FILE.format(m)
        writeGz(frame, os.path.join(args.out_dir, name))
        files[name] = fileSha(os.path.join(args.out_dir, name))
    with open(os.path.join(args.out_dir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump({"criteria_sha256": sha, "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"), "files": files,
                   "rows": {PRED_FILE.format(m): len(fr) for m, fr in K3["pred_frames"].items()},
                   "columns": ["host", "dataset", "menu", "version", "subset_donor", "split", "n2", "pred_acc"],
                   "note": "留一個資料集的預測（K = 3 參與的版本；原本的版本對兩個供體的子集二各一組）；評分半 ∩ 子集二的預測正確率"},
                  f, ensure_ascii=False, indent=2)
    plotPredictions(all_subs, os.path.join(args.out_dir, "fig_a_predictions"), sha)
    plotScenarios(BF["scenarios"], os.path.join(args.out_dir, "fig_b_scenarios"), sha)

    # ------------------------------------------------------------------
    # report.md
    # ------------------------------------------------------------------
    S = lambda col, frame=blocks: summ(frame[col])
    G3, G2, G12 = D["G3"], D["G2"], D["G12"]
    out = [f"判定標準 `{os.path.join(args.out_dir, CRITERIA_FILE)}` 的 sha256 `{sha}`；確認：{confirmed}", "",
           "# RQ3-GJR：用 RQ3-GJ 的裁判紀錄做 regression，預測裁判的選擇", "",
           f"產生：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}（`scripts/analysis_rq3gjr/judge_regression.py`，"
           f"`Analysis/judgeRegression.py`）。全程離線，沒有呼叫 API，沒有修改任何既有的輸出。", "",
           "單位：D_P、平均絕對誤差、G、殘差都是 pp；E_J 是「path 每進步 10pp，Judge 多幾個 pp」。區塊 = 宿主 × 資料集，共 8 個；"
           "區間 = 95% t 區間（自由度 = 區塊數 − 1）。門檻 0.5pp。", ""]

    # 1. 讀了哪些檔案
    out += ["## 1. 讀了哪些檔案", "",
            f"- RQ3-GJ 的 K = 3 Judge 紀錄 `{args.gj_dir}/judge_outputs/{{宿主}}/{{資料集}}/{{菜單}}__{{版本}}.json`（{len(G3.keys):,} 份）、"
            f"K = 2 `{args.gj_dir}/judge_outputs_k2/…__{{A|B}}.json`（{len(G2.keys):,} 份）："
            "`presentation_order`、`trace.choice`、`trace.chosen_arm`、`final_answer`。",
            f"- RQ1-KJ 的 Judge 紀錄 `{RQ1KJ_JUDGE_DIR}/{{宿主}}/{{資料集}}/M12.json`（第 7 節 4(b)）與 `M3L/M3S/M3P.json`（第 10 節 4）：同上。",
            f"- `{args.armdir}` 的 4 個模型 × 4 個資料集 × 12 條 path：`raw_text`（tokens 用 tiktoken `o200k_base`、字元數）、`parsed_answer`、`gold`"
            f"（經 `Analysis.menuVote` / `Analysis.pathImprove` 的編碼，與 RQ3-GJ 相同）；`{args.aggdir}`：切分（`makeSplits` 200 次）。",
            f"- RQ3-GJ 的計畫（`Analysis/judgeSubstitution.py`；判定標準 sha256 `{GJ_CRITERIA_SHA}` 已核對）、"
            f"`rq3gj_predictions.csv.gz` 與 `rq3gj_predictions_lodo.csv.gz`（sha256 已核對 `rq3gj_predictions_manifest.json`）、"
            "`rq3gj_blocks.csv`、`report.md`（第 10 節 1–3）。只讀。", ""]

    # 2. 第零階段的表（由這次讀到的紀錄重數）
    a1 = []
    for hi, h in enumerate(HOSTS):
        for di, d in enumerate(DATASETS):
            row = {"裁判": h, "資料集": d}
            for K, G in ((3, G3), (2, G2), (12, G12)):
                m = (G.judge == hi) & (G.dataset == di)
                row[f"K = {K} 呼叫"], row[f"K = {K} 有效"] = f"{int(m.sum()):,}", f"{int((m & G.valid).sum()):,}"
            a1.append(row)
    a1.append({"裁判": "合計", "資料集": "", **{f"K = {K} {w}": f"{int(n):,}" for K, G in ((3, G3), (2, G2), (12, G12))
                                              for w, n in (("呼叫", len(G)), ("有效", G.valid.sum()))}})
    out += ["## 2. 第零階段的表", "",
            "呼叫數與有效選擇數（由這次讀到的紀錄重數，和判定標準附錄 A.1 相同）。因素的分布、缺 raw_text 的候選、計算時間見判定標準附錄 A.2–A.4。", "",
            md(pd.DataFrame(a1), text=True), ""]

    # 3. 檢查
    out += ["## 3. 開跑前的檢查（第 10 節）", "",
            md(pd.DataFrame([{"項": c["項"], "內容": c["內容"], "結果": c["結果"], "通過": "✅" if c["通過"] else "❌"} for c in checks]), text=True), "",
            "第 6 項的經過：2026-10-09 第一次跑時，statsmodels 用 `fit(method=\"bfgs\", gtol=1e-11)`，和自己的係數最大差 GPT-4o mini 6.7e-4、"
            "Qwen3-8B 1.6e-3，**沒過**，停下來回報。診斷：用 statsmodels 自己的 `score`，自己的係數 |score| 約 1e-13、對數概似較高；"
            "BFGS 的係數 |score| 約 0.12（gtol 沒有作用，沒有走到最大值）。經使用者確認，改用 statsmodels 的牛頓法（`method=\"newton\", tol=1e-12`），"
            "門檻 1e-6、抽樣（每個裁判 5,000 組、`default_rng(0)`）不變，六項全部重跑，結果如上表。", ""]

    # 4. 判定
    s1, sG = S("D_P_M3"), S("G")
    st1, st2 = fourState(s1, THRESHOLD), fourState(sG, THRESHOLD)
    def perBlock(cols: dict, frame: pd.DataFrame = blocks) -> str:
        t = frame.assign(model=frame.model.map(MODEL_LABELS))[["model", "dataset", *cols]]
        return md(t.rename(columns={"model": "宿主", "dataset": "資料集", **cols}))

    out += ["## 4. 判定（M3、留一個資料集；門檻 0.5pp）", "",
            md(pd.DataFrame([ciRow("判定一：D_P = 實際 − M3 預測（pp）", s1, True),
                             ciRow("判定二：G = T_new 的平均絕對誤差 − M3 的（pp）", sG, True)]), text=True), "",
            perBlock({"D_P_M3": "判定一 D_P", "MAE_T_new": "T_new 平均絕對誤差", "MAE_M3": "M3 平均絕對誤差", "G": "判定二 G"}), "",
            f"- 判定一：**{st1}**。{READ_J1[st1]}",
            f"- 判定二：**{st2}**。{READ_J2[st2]}",
            "- 事先寫下的限制：(a) 測試的資料集沒看過，但菜單、path、模型都看過；(b) 這批紀錄已經被用來形成「強模型的候選較常被選」等觀察，"
            "判定檢驗的是預測準不準，不是那些觀察本身；(c) 預測需要知道每份候選對不對。", ""]

    # 5. 只報告的量
    out += ["## 5. 只報告的量", ""]
    side = []
    for m in SIDE:
        side.append({"預測": m, "D_P": cellText(S(f"D_P_{m}")), "每個替換的平均絕對誤差": cellText(S(f"MAE_{m}")),
                     "Spearman（8 個區塊的平均）": f"{blocks[f'spearman_{m}'].mean():+.3f}"})
    out += ["### 7.1 T_old、T_new、M0–M4 並排（留一個資料集）", "",
            "每格：8 個區塊的平均 [95% 區間] 為正的區塊數。Spearman：每次切分、區塊內參與的替換之間算「預測多的」對「實際多的」，再對切分平均。", "",
            md(pd.DataFrame(side), text=True), "",
            "逐區塊的 D_P：", "", perBlock({f"D_P_{m}": m for m in SIDE}), "",
            "逐區塊的 Spearman：", "", perBlock({f"spearman_{m}": m for m in SIDE}), ""]
    ej_rows = []
    for group in GROUP_ORDER:
        ej_rows.append({"量": f"E_J {group}：實際 − 預測",
                        **{m: cellText(summ(blocks[f"E_J_act_{group}"] - blocks[f"E_J_pred_{group}_{m}"])) for m in EJ_MODELS}})
    ej_rows.append({"量": "D_JV：實際 − 預測", **{m: cellText(summ(blocks.D_JV_act - blocks[f"D_JV_pred_{m}"])) for m in EJ_MODELS}})
    out += ["### 7.2 預測的 E_J 與 D_JV（評分半，對 200 次切分平均）", "",
            "實際 − 預測；每格：8 個區塊的平均 [95% 區間] 為正的區塊數。M1 加入誰寫的；M2 加入文字類型；M3 加入長度。只描述。", "",
            md(pd.DataFrame(ej_rows), text=True), "",
            "實際的 E_J 與 D_JV（8 個區塊的平均）：" + "、".join(f"{g} {blocks[f'E_J_act_{g}'].mean():+.2f}" for g in GROUP_ORDER) +
            f"、D_JV {blocks.D_JV_act.mean():+.2f}。", ""]

    # 7.3 係數
    bc = coefs[coefs.fit == "block"]
    crow = []
    for m in BLOCK_MODELS:
        c = bc[bc.model == m]
        for name in MODELS[m]:
            s = summ(c[f"b_{name}"])
            crow.append({"模型": m, "係數": name, "平均": f"{s['mean']:+.3f}", "95% 區間": f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]",
                         "為正的區塊": f"{s['n_positive']}/8", "逐區塊 SE 的中位數": f"{c[f'se_{name}'].median():.3f}"})
    key = lambda m: bc[bc.model == m].set_index(["judge", "dataset"])
    m3, m5a, m5b = key("M3"), key("M5a"), key("M5b")
    chg = []
    for name in ("trans", "short"):
        s = summ(m5a[f"b_{name}"] - m3[f"b_{name}"])
        chg.append({"量": f"{name}：M5a − M3", "平均": f"{s['mean']:+.3f}", "95% 區間": f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]",
                    "為正的區塊": f"{s['n_positive']}/8"})
    for name in ("trans", "short"):
        for label, vals in (("答錯時", m5b[f"b_{name}"]), ("答對時", m5b[f"b_{name}"] + m5b[f"b_correct:{name}"])):
            s = summ(vals)
            chg.append({"量": f"M5b {name}，{label}", "平均": f"{s['mean']:+.3f}", "95% 區間": f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]",
                        "為正的區塊": f"{s['n_positive']}/8"})
    tr = summ(m5a["b_trans"])
    n_neg = int((m5a["b_trans"] < 0).sum())
    if tr["ci_high"] < 0 and n_neg >= 5:
        read3 = "翻譯的候選較少被選，不只是因為那條 path 較弱"
    elif tr["ci_low"] <= 0 <= tr["ci_high"]:
        read3 = "較弱的 path 的候選較少被選"
    else:
        read3 = None
    read3_text = (f"M5a 的 trans 係數 {tr['mean']:+.3f}，95% 區間 [{tr['ci_low']:+.3f}, {tr['ci_high']:+.3f}]，8 個區塊中 {n_neg} 個為負 → " +
                  (f"「{read3}」。" if read3 else "不符合兩句讀法的條件，只列數字。"))
    scen = BF["scenarios"].groupby(["judge", "correct", "donor", "kind"], as_index=False)[["p_M4", "p_M3"]].mean()
    srows = []
    for correct in (1, 0):
        for donor in (0, 1):
            for kind, kname in KINDS:
                r = {"落單那份": "對" if correct else "錯", "寫的": "供體（強模型）" if donor else "宿主", "文字類型": kname}
                for h in HOSTS:
                    x = scen[(scen.judge == h) & (scen.correct == correct) & (scen.donor == donor) & (scen.kind == kind)].iloc[0]
                    r[f"{MODEL_LABELS[h]}：M4"], r[f"{MODEL_LABELS[h]}：M3"] = f"{x.p_M4:.3f}", f"{x.p_M3:.3f}"
                srows.append(r)
    out += ["### 7.3 係數（逐區塊配適，只描述）", "",
            "**這批資料就是形成假設的資料，只當描述；不把係數寫成因果，也不當成「裁判看文字類型」的確認。**", "",
            "M3、M4、M5a、M5b 每個區塊各配適一次（該區塊全部 K = 3 有效紀錄）。係數是 log-odds 尺度；8 個區塊的平均與 95% t 區間：", "",
            md(pd.DataFrame(crow), text=True), "",
            "trans、short 的變化與 M5b 的翻譯 / 短推理候選（逐區塊算，再取平均與區間）：", "",
            md(pd.DataFrame(chg), text=True), "",
            f"- 讀法（用 M5a 的 trans 係數，事先寫下）：{read3_text}",
            "- 限制：pathacc 用到標準答案，而且和 trans 高度相關，係數會不穩。", "",
            "固定情境下落單那份被選的機率（K = 3；另外兩份同答案、宿主寫的、英文完整、對錯相反；三個位置平均；長度取區塊中位數、互相抵銷；"
            "每個區塊各算一次，再對裁判的 4 個區塊平均）。M4 是主要的表，M3 並列：", "",
            md(pd.DataFrame(srows), text=True), ""]

    # 7.4 跨 K
    sk2 = summ(blocks.k2_D_P)
    k2_ok = sk2["ci_low"] >= -THRESHOLD and sk2["ci_high"] <= THRESHOLD
    out += ["### 7.4 跨 K", "",
            "(a) K = 3 訓練的 M3（留一個資料集）→ 被留下資料集的 K = 2 各版本。題目的預測機率與實際對錯都取兩種順序的平均；"
            "D_P 在評分半 ∩ 子集二上算、200 次切分平均；K = 2 效果 = 10 × 分子總和 ÷ 分母總和（子集二全部題目）。"
            "**K = 2 的裁判 prompt 版本不同（choice-v1）。**【補充】這一項用 M3（主要模型）。", "",
            md(pd.DataFrame([ciRow("D_P（pp）", sk2), ciRow("每個替換的平均絕對誤差（pp）", S("k2_MAE")),
                             ciRow("實際的 K = 2 效果", S("k2_effect_actual")), ciRow("M3 預測的 K = 2 效果", S("k2_effect_pred")),
                             ciRow("效果：實際 − 預測", S("k2_effect_diff"))]), text=True), "",
            perBlock({"k2_D_P": "D_P", "k2_MAE": "平均絕對誤差", "k2_effect_actual": "實際效果", "k2_effect_pred": "預測效果"}), "",
            f"實際的 K = 2 效果和 RQ3-GJ 第 8 節 14(a) 的逐區塊值最大差 {k2_worst:.1e}。", "",
            "讀法：" + ("D_P 的區間在 ±0.5pp 內 → 可以跨到 K = 2。" if k2_ok else
                       f"D_P 的區間 [{pp(sk2['ci_low'])}, {pp(sk2['ci_high'])}] 不在 ±0.5pp 內 → 只列數字。"), ""]
    k12_rows = []
    for h in HOSTS:
        b = blocks[blocks.model == h]
        v = b.k12_D_P.to_numpy()
        k12_rows.append({"裁判": MODEL_LABELS[h], "D_P 平均": pp(v.mean()), "4 個值": "、".join(pp(x) for x in v),
                         "為正": f"{int((v > 0).sum())}/4", "實際正確率（平均）": f"{b.k12_actual.mean():.2f}%",
                         "預測正確率（平均）": f"{b.k12_pred.mean():.2f}%"})
    out += ["(b) 同樣的 M3 → RQ1-KJ 的 K = 12 原本的版本（M12 子集 = 12 條都有答案，評分半，200 次切分平均）。兩個裁判分開列：", "",
            md(pd.DataFrame(k12_rows), text=True), "",
            perBlock({"k12_actual": "實際正確率 %", "k12_pred": "預測正確率 %", "k12_D_P": "D_P", "k12_MAE": "平均絕對誤差", "k12_calls": "呼叫數"}), "",
            "註：Qwen 的 K = 12 輸出有問題（RQ1-KJ 結果 17：11 比 1 時常輸出它自己判斷為錯的候選），它的數字不能當成模型能否跨 K 的證據。", ""]

    # 7.5 留一組菜單
    lrows = []
    for group in GROUP_ORDER:
        g = log[log.group == group]
        lrows.append({"測試組": group, "替換數": int(g.n_subs.sum()), "D_P": cellText(summ(g.D_P)), "平均絕對誤差": cellText(summ(g.MAE))})
    out += ["### 7.5 留一組菜單（M3）", "",
            "四組輪流當測試；其餘菜單（四個資料集）的有效紀錄配適。GS-181 同時在中間組與英文落單組，測試這兩組的任一組時它不進訓練。"
            "每格：8 個區塊的平均 [95% 區間] 為正的區塊數。", "", md(pd.DataFrame(lrows), text=True), ""]

    # 7.6 同一個資料集內
    out += ["### 7.6 同一個資料集內（M3，前 20 次切分）", "",
            "每次切分 r：用該裁判 4 個資料集中題目在 H1_r 的 K = 3 有效紀錄配適，在 H2_r 上評分。留一個資料集的值（200 次切分）並列。", "",
            md(pd.DataFrame([ciRow("同一個資料集內：D_P", S("within_D_P")), ciRow("同一個資料集內：平均絕對誤差", S("within_MAE")),
                             ciRow("留一個資料集：D_P", S("D_P_M3")), ciRow("留一個資料集：平均絕對誤差", S("MAE_M3"))]), text=True), "",
            perBlock({"within_D_P": "同資料集 D_P", "within_MAE": "同資料集 MAE", "D_P_M3": "留一個資料集 D_P", "MAE_M3": "留一個資料集 MAE"}), ""]

    # 7.7 分開
    tags = {g: g.split(".")[0][:8] for g in DONORS}
    jrows = []
    for h in HOSTS:
        b = blocks[blocks.model == h]
        for name, col in (("判定一 D_P", "D_P_M3"), ("判定二 G", "G")):
            v = b[col].to_numpy()
            jrows.append({"裁判": MODEL_LABELS[h], "量": name, "平均": pp(v.mean()), "4 個值": "、".join(pp(x) for x in v),
                          "為正": f"{int((v > 0).sum())}/4"})
    out += ["### 7.7 兩個供體分開、兩個裁判分開（只描述）", "",
            md(pd.DataFrame([ciRow(f"{name}，供體 {g}", S(f"{col}_{tags[g]}")) for g in DONORS
                             for name, col in (("判定一 D_P", "D_P_M3"), ("判定二 G", "G"))]), text=True), "",
            md(pd.DataFrame(jrows), text=True), ""]

    # 7.8 殘差
    res = K3["resid"].groupby(["host", "dataset", "path"], as_index=False).residual.mean()
    rrows = []
    for path in M12:
        v = res[res.path == path]
        if len(v) == len(HOSTS) * len(DATASETS):
            rrows.append({"被換的 path": path, **ciRow("", summ(v.residual))})
        else:
            rrows.append({"被換的 path": path, "量": "", "平均": pp(v.residual.mean()), "SE": "—", "95% 區間": "—",
                          "為正的區塊": f"{int((v.residual > 0).sum())}/{len(v)}"})
    out += ["### 7.8 殘差：實際多的 − M3 預測多的（pp，評分半，對切分平均）", "",
            "每個區塊、每條 path 先對它的替換（菜單 × 供體）平均，再報 8 個區塊的平均與 95% 區間。", "",
            md(pd.DataFrame(rrows).drop(columns="量"), text=True), ""]

    # 7.9 字元數
    out += ["### 7.9 長度改用字元數（M3chars：loglen = ln(字元數)，對照）", "",
            md(pd.DataFrame([ciRow("判定一：D_P（M3chars）", S("D_P_M3chars"), True),
                             ciRow("判定二：G = T_new − M3chars 的平均絕對誤差", S("G_chars"), True)]), text=True), "",
            "註：中文、日文一個字元的資訊量和英文不同。第零階段：翻譯 path 的字元數中位數 333、英文完整 590，但 tokens 中位數是 155 對 129；"
            "所以字元數會把文字種類混進長度。這裡的狀態只是對照，不是判定。", ""]

    # 7.10 圖
    out += ["### 7.10 圖", "",
            "- `fig_a_predictions.{pdf,png}`：每個替換的預測多的對實際多的正確率（評分半，對 200 次切分平均），T_old、T_new、M3 三格並排。",
            "- `fig_b_scenarios.{pdf,png}`：第 7 節 3 的 12 個機率（M4，兩個裁判分開）。", ""]

    # 6. 結論
    out += ["## 6. 對照讀法的結論", "",
            f"- 判定一（M3 的 D_P {cellText(s1)}）：**{st1}**。{READ_J1[st1]}",
            f"- 判定二（G {cellText(sG)}）：**{st2}**。{READ_J2[st2]}",
            f"- 第 7 節 3（只描述）：{read3_text}",
            "- 第 7 節 4(a)：" + ("可以跨到 K = 2。" if k2_ok else "D_P 的區間不在 ±0.5pp 內，只列數字。"),
            "- 其餘各項只報告，不套四種狀態。", "",
            "## 輸出檔", "",
            "`rq3gjr_blocks.csv`（逐區塊）、`rq3gjr_substitutions.csv`（逐替換，K = 3 與 K = 2）、`rq3gjr_coefficients.csv`"
            f"（{len(coefs)} 次配適）、`rq3gjr_predictions_{{T_new,M0,M1,M2,M3,M4,M3chars}}.csv.gz`（sha256 在 `{MANIFEST}`）、兩張圖。"
            "每份都記錄判定標準的 sha256。", ""]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"\n判定一 D_P {cellText(s1)} → {st1}；判定二 G {cellText(sG)} → {st2}（{elapsed()}）")
    print(f"💾 {args.out_dir}/report.md 與其他輸出")


def main():
    args = parseArgs()
    started = time.time()
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    confirmed = confirmationLine(criteria)
    if confirmed is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet")
    sha = sha256(criteria)
    if sha256(os.path.join(args.gj_dir, "rq3gj_criteria.md")) != GJ_CRITERIA_SHA:
        raise SystemExit("❌ rq3gj_criteria.md does not match the sha256 recorded in rq3gjr_criteria.md §2")
    D = loadData(args)
    checks = runChecks(D, args)
    print("\n第 10 節的檢查：")
    for c in checks:
        print(f"  {'✅' if c['通過'] else '❌'} {c['項']}. {c['內容']}\n       {c['結果']}")
    print(f"（{time.time() - started:.0f}s）")
    if not all(c["通過"] for c in checks):
        raise SystemExit("❌ 開跑前的檢查沒有全部通過：停，不寫任何輸出")
    if args.checks_only:
        return
    formalRun(D, args, checks, sha, confirmed, started)


if __name__ == "__main__":
    main()
