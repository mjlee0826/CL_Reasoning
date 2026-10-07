import json
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from Analysis.alignment import loadRecords
from Analysis.crossJudge import classifyRecord
from Analysis.experimentPlan import PATHS
from Analysis.menuVote import DATASETS
from Analysis.pathImprove import M12, HOSTS, DONORS, CodedBlock
from Analysis.pathImproveGS import GROUPS, menuSets, agreementRates, menuGrouping, assignGroups
from Strategy.MenuJudge import MenuJudge
from Runner.paths import substitutionJudgePath

# ------------------------------------------------------------------
# RQ3-GJ（result/analysis/rq3gj/rq3gj_criteria.md）：讓模型當裁判時，該改哪一條 path？
#   計畫（菜單、版本、要呼叫的題目、候選順序、試跑的題目）、第零階段的費用估計、§10–§12 各步驟的評估。
#   預測（§7、§8.12、§8.13）在 Analysis/judgeSubstitutionPredict.py。
# ------------------------------------------------------------------
OUT_DIR = "result/analysis/rq3gj"
CRITERIA_FILE = "rq3gj_criteria.md"
JUDGE_DIR, JUDGE_DIR_K2, PRECHECK_DIR = "judge_outputs", "judge_outputs_k2", "precheck"
PRECHECK_FILE, PILOT_FILE, PRERUN_FILE = "precheck.json", "pilot.json", "prerun.json"
STAGE0_CALLS = "rq3gj_stage0_calls.csv"
RUNNER = "scripts/analysis_rq3gj/run_gj_judge.py"
RQ1KJ_JUDGE_DIR = "result/analysis/rq1kj/judge_outputs"
RQ3GS_BLOCKS = "result/analysis/rq3gs/rq3gs_blocks.csv"
RQ3GS_MENU_BLOCKS = "result/analysis/rq3gs/rq3gs_menu_blocks.csv"
NUMS = 2000                       # 主網格的 --nums
AGGREGATOR_SEED = 0
ORDER_SEED = 0                    # §4：RQ1-KJ §3.3 的分組旋轉，seed 0
TOL = 1e-9

# 2026-10-07 查閱的定價（美元 / 百萬 tokens：輸入、輸出）
PRICES = {"gpt4omini": (0.15, 0.60), "qwen": (0.18, 0.70)}
MAX_COST = 300.0                  # §11：依試跑推估的總費用超過就停
PILOT_MAX_INVALID = 0.01          # §11：無效（找不到選擇 + 超出範圍 + 被擋下）比例超過就停

# §10 流程核對
CHECK_CELLS = (("qwen", "truthfulqa"), ("gpt4omini", "truthfulqa"))
CHECK_MENU, CHECK_CODES = "M3S", ["EN", "S1", "S2"]
CHECK_MIN = {"qwen": 0.90, "gpt4omini": 0.80}

# §3 的四組菜單（依第 3 節表的順序）；計畫建立時重抽並核對
GROUP_ORDER = ["明顯落單", "中間", "對稱", "英文落單"]
GROUP_MENUS = {
    "明顯落單": ["GS-105", "GS-095", "GS-142", "GS-118", "GS-087", "GS-003", "GS-157", "GS-014", "GS-090", "GS-139"],
    "中間": ["GS-181", "GS-028", "GS-033", "GS-066", "GS-046", "GS-064", "GS-062", "GS-032", "GS-150", "GS-065"],
    "對稱": ["GS-019", "GS-026", "GS-020", "GS-068", "GS-135", "GS-119", "GS-085", "GS-031", "GS-109", "GS-024"],
    "英文落單": ["GS-039", "GS-022", "GS-125", "GS-038", "GS-042", "GS-021", "GS-181", "GS-124", "GS-141", "GS-041"],
}
MENU_ORDER = list(dict.fromkeys(m for g in GROUP_ORDER for m in GROUP_MENUS[g]))      # 39 份，GS-181 只一次
MIDDLE_TRANSLATED = ["GS-066", "GS-046", "GS-064", "GS-062", "GS-150", "GS-065"]       # §8.15
TRANSLATED = {"ES", "JA", "RU", "ZH"}
PAIRS = {"EN+ZH": ["EN", "ZH"], "EN+S1": ["EN", "S1"], "P1+P2": ["P1", "P2"]}          # §8.14
ORDER_LABELS = ("A", "B")         # 順序甲（配對的列出順序）、順序乙（對調）

# §11 試跑
PILOT_BLOCK, PILOT_MENU, PILOT_PAIR, PILOT_N = ("gpt4omini", "mmlu"), "GS-105", "EN+ZH", 100


def disagree(codes: np.ndarray) -> np.ndarray:
    """(K, N) 答案代碼 -> 每題答案不完全相同。"""
    return ~np.all(codes == codes[0][None, :], axis=0)


# ------------------------------------------------------------------
# 菜單（§3）
# ------------------------------------------------------------------
def checkMenus(coded: dict) -> dict:
    """重算 RQ3-GS 的分組與 §3 的抽法，四組菜單要和 GROUP_MENUS 相同；回傳 {菜單: menuGrouping 的結果 + codes}。"""
    conf, _ = menuSets()
    rates = agreementRates([coded[(h, d)] for h in HOSTS for d in DATASETS])
    gro = [menuGrouping(codes, rates) for _, codes in conf]
    groups, _, _ = assignGroups(gro)
    info = {mid: {**g, "codes": codes, "group": grp} for (mid, codes), g, grp in zip(conf, gro, groups)}
    ids = {grp: [mid for (mid, _), g in zip(conf, groups) if g == grp] for grp in GROUPS}
    old = {GROUPS[0]: GROUP_MENUS["明顯落單"][:3], GROUPS[2]: GROUP_MENUS["對稱"][:3]}
    rng0 = np.random.default_rng(0)
    drawn0 = {GROUPS[0]: [ids[GROUPS[0]][i] for i in rng0.choice(62, 3, replace=False)],
              GROUPS[2]: [ids[GROUPS[2]][i] for i in rng0.choice(62, 3, replace=False)]}
    rng1 = np.random.default_rng(1)
    pools = {g: [m for m in ids[g] if m not in old.get(g, [])] for g in GROUPS}
    new = {g: [pools[g][i] for i in rng1.choice(len(pools[g]), n, replace=False)]
           for g, n in ((GROUPS[0], 7), (GROUPS[2], 7), (GROUPS[1], 10))}
    english = sorted((mid for mid, i in info.items() if not set(i["codes"]) & TRANSLATED), key=lambda m: (-info[m]["degree"], m))
    drawn = {"明顯落單": drawn0[GROUPS[0]] + new[GROUPS[0]], "中間": new[GROUPS[1]], "對稱": drawn0[GROUPS[2]] + new[GROUPS[2]],
             "英文落單": english[:10]}
    if drawn != GROUP_MENUS or drawn0 != old:
        raise ValueError(f"§3 menus do not reproduce: {drawn} != {GROUP_MENUS}")
    return {m: info[m] for m in MENU_ORDER}


# ------------------------------------------------------------------
# 計畫（§4、§8.14）
# ------------------------------------------------------------------
@dataclass
class Version:
    K: int
    host: str
    dataset: str
    menu: str                    # GS-xxx 或配對（EN+ZH …）
    codes: list                  # 菜單的 path 代號（平手順序 / 配對的列出順序）
    version: str                 # "orig" 或 "{path}-{donor}"
    donor: str | None = None
    slot: int | None = None
    order: str | None = None     # K = 2 的 "A" / "B"
    participates: bool = True
    items: list = field(default_factory=list)       # 要呼叫 Judge 的題目（item_id 由小到大）

    @property
    def arm_ids(self) -> list:
        return [PATHS[c] for c in self.codes]

    @property
    def key(self) -> tuple:
        return (self.host, self.dataset, self.menu)

    def path(self, out_dir: str) -> str:
        return substitutionJudgePath(os.path.join(out_dir, JUDGE_DIR if self.K == 3 else JUDGE_DIR_K2), self.host, self.dataset,
                                     self.menu, self.version, self.order)


@dataclass
class Plan:
    versions: list               # 所有版本（K = 2 每個順序一個）
    orders: dict                 # (host, dataset, menu[, order]) -> {item_id: arm_ids}
    menus: dict                  # 菜單資訊（checkMenus）
    subset2: dict                # (host, donor, dataset) -> 子集二的 item_id 集合

    def ordersOf(self, v: Version) -> dict:
        return self.orders[v.key + ((v.order,) if v.K == 2 else ())]

    def find(self, K: int, host: str, dataset: str, menu: str, version: str, order: str | None = None) -> Version:
        return next(v for v in self.versions if (v.K, v.host, v.dataset, v.menu, v.version, v.order) == (K, host, dataset, menu, version, order))


def buildPlan(coded: dict, menus: dict) -> Plan:
    versions, orders, subset2 = [], {}, {}
    for h in HOSTS:
        for d in DATASETS:
            H = coded[(h, d)]
            ids = np.array([int(i) for i in H.item_ids])
            sub2 = {g: H.sub & coded[(g, d)].sub for g in DONORS}
            for g in DONORS:
                subset2[(h, g, d)] = set(ids[sub2[g]].tolist())
            union = sub2[DONORS[0]] | sub2[DONORS[1]]
            sub1_ids = ids[H.sub].tolist()
            for K, menu_list in ((3, [(m, menus[m]["codes"]) for m in MENU_ORDER]), (2, list(PAIRS.items()))):
                for menu, codes in menu_list:
                    rows = [M12.index(c) for c in codes]
                    arm_ids = [PATHS[c] for c in codes]
                    base = H.codes[rows]
                    labels = ORDER_LABELS if K == 2 else (None,)
                    if K == 3:
                        orders[(h, d, menu)] = MenuJudge.groupedOrders(arm_ids, sub1_ids, ORDER_SEED)
                    else:
                        orders[(h, d, menu, "A")] = {i: list(arm_ids) for i in sub1_ids}
                        orders[(h, d, menu, "B")] = {i: list(reversed(arm_ids)) for i in sub1_ids}
                    for order in labels:
                        versions.append(Version(K, h, d, menu, codes, "orig", order=order, items=ids[disagree(base) & union].tolist()))
                    for g in DONORS:
                        G = coded[(g, d)]
                        for j, p in enumerate(codes):
                            k = M12.index(p)
                            part = int(G.correct[k][sub2[g]].sum()) > int(H.correct[k][sub2[g]].sum())
                            cc = base.copy()
                            cc[j] = G.codes[k]
                            items = ids[disagree(cc) & sub2[g]].tolist() if part else []
                            for order in labels:
                                versions.append(Version(K, h, d, menu, codes, f"{p}-{g}", g, j, order, part, items))
    return Plan(versions, orders, menus, subset2)


def pilotItems(plan: Plan) -> dict:
    """§11：{(K, menu): 前 100 題}（原本的版本需要呼叫的題目，依 item_id 排序）。"""
    out = {}
    for K, menu in ((3, PILOT_MENU), (2, PILOT_PAIR)):
        orig = plan.find(K, *PILOT_BLOCK, menu, "orig", "A" if K == 2 else None)
        out[(K, menu)] = sorted(orig.items)[:PILOT_N]
    return out


def pilotVersions(plan: Plan) -> list:
    """§11 的試跑：[(版本, 這個版本在試跑題目中要呼叫的題目)]，K = 3 的 7 個版本與 K = 2 的 5 個版本 × 2 種順序。"""
    sample = pilotItems(plan)
    out = []
    for v in plan.versions:
        if v.key[:2] == PILOT_BLOCK and (v.K, v.menu) in sample and v.participates:
            chosen = sorted(set(v.items) & set(sample[(v.K, v.menu)]))
            out.append((v, chosen))
    return out


def checkVersions(rq1kj_dir: str = RQ1KJ_JUDGE_DIR) -> list:
    """§10：RQ1-KJ 的 qwen、gpt4omini × truthfulqa × M3S，原本的版本；題目與順序用舊紀錄。回傳 [(Version, orders)]。"""
    out = []
    for host, dataset in CHECK_CELLS:
        _, records = loadRecords(os.path.join(rq1kj_dir, host, dataset, f"{CHECK_MENU}.json"))
        orders = {int(i): r["presentation_order"] for i, r in records.items()}
        out.append((Version(3, host, dataset, CHECK_MENU, CHECK_CODES, "orig", items=sorted(orders)), orders))
    return out


def plannedCalls(plan: Plan) -> pd.DataFrame:
    """每個版本的呼叫數（K = 2 兩種順序各一列）。"""
    return pd.DataFrame([{"K": v.K, "host": v.host, "dataset": v.dataset, "menu": v.menu, "version": v.version, "order": v.order,
                          "participates": v.participates, "calls": len(v.items) if v.participates else 0} for v in plan.versions])


def checkStage0(plan: Plan, stage0_csv: str) -> dict:
    """計畫的呼叫數 = 第零階段寫好的 rq3gj_stage0_calls.csv（K = 2 的兩種順序合計）。"""
    mine = plannedCalls(plan)
    mine["version"] = mine.version.str.replace("-", "←", n=1, regex=False).where(mine.version != "orig", "原本")
    agg = mine.groupby(["K", "host", "dataset", "menu", "version"], as_index=False).calls.sum()
    ref = pd.read_csv(stage0_csv)
    ref["version"] = ref.version.astype(str)
    merged = agg.merge(ref[["K", "host", "dataset", "menu", "version", "calls"]], on=["K", "host", "dataset", "menu", "version"],
                       how="outer", suffixes=("", "_stage0"))
    bad = merged[merged.calls != merged.calls_stage0]
    return {"ok": bad.empty and len(merged) == len(ref), "n_rows": len(merged), "total": int(agg.calls.sum()),
            "mismatches": bad.head(10).to_dict("records")}


# ------------------------------------------------------------------
# 第零階段的每次呼叫費用（附錄 A.3 的算法）：K = 3 用 RQ1-KJ 三條菜單的 API 計費平均，K = 2 用主網格 Judge 檔的 metadata
# ------------------------------------------------------------------
K2_FILES = {"EN+ZH": "judge__L_en__L_zh.json", "EN+S1": "judge__L_en__S_T1.0_seed1.json", "P1+P2": "judge__P_expert__P_skeptic.json"}


def tokenMeans(aggdir: str = "result/aggregations", rq1kj_dir: str = RQ1KJ_JUDGE_DIR) -> dict:
    """{(K, host, dataset, menu or None): (input 平均, output 平均)}。"""
    out = {}
    for h in HOSTS:
        for d in DATASETS:
            ins, outs = [], []
            for menu in ("M3L", "M3S", "M3P"):
                _, recs = loadRecords(os.path.join(rq1kj_dir, h, d, f"{menu}.json"))
                ins += [r["call"]["usage_in"] for r in recs.values()]
                outs += [r["call"]["usage_out"] for r in recs.values()]
            out[(3, h, d, None)] = (float(np.mean(ins)), float(np.mean(outs)))
            for pair, name in K2_FILES.items():
                with open(os.path.join(aggdir, h, d, name), encoding="utf-8") as f:
                    usage = json.load(f)[0]["api_usage"]
                out[(2, h, d, pair)] = (usage["prompt_tokens"] / usage["calls"], usage["completion_tokens"] / usage["calls"])
    return out


def callCost(host: str, tokens_in: float, tokens_out: float) -> float:
    price_in, price_out = PRICES[host]
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


def stage0Cost(plan: Plan, means: dict) -> dict:
    calls = plannedCalls(plan)
    total = {3: 0.0, 2: 0.0}
    for r in calls.itertuples():
        i, o = means[(r.K, r.host, r.dataset, r.menu if r.K == 2 else None)]
        total[r.K] += r.calls * callCost(r.host, i, o)
    return {"K3": total[3], "K2": total[2], "total": total[3] + total[2]}


# ------------------------------------------------------------------
# §10 流程核對
# ------------------------------------------------------------------
def evaluatePrecheck(precheck_dir: str, rq1kj_dir: str = RQ1KJ_JUDGE_DIR) -> dict:
    result, complete, passed = {}, True, True
    for v, _ in checkVersions(rq1kj_dir):
        old_meta, old = loadRecords(os.path.join(rq1kj_dir, v.host, v.dataset, f"{CHECK_MENU}.json"))
        path = precheckPath(precheck_dir, v)
        new_meta, new = loadRecords(path) if os.path.exists(path) else ({}, {})
        done = sorted(set(new) & set(old))
        same_final = sum(new[i]["final_answer"] == old[i]["final_answer"] for i in done)
        same_choice = sum(new[i]["trace"].get("chosen_arm") == old[i]["trace"].get("chosen_arm") for i in done)
        rate = same_final / len(done) if done else float("nan")
        r = {"n": len(old), "n_done": len(done), "same_final": same_final, "agreement": rate,
             "same_choice": same_choice, "choice_agreement": same_choice / len(done) if done else float("nan"),
             "min_agreement": CHECK_MIN[v.host], "stop": bool(done) and rate < CHECK_MIN[v.host],
             "returned_versions": new_meta.get("model_versions", {}), "rq1kj_versions": old_meta.get("model_versions", {}),
             "same_orders": all(new[i]["presentation_order"] == old[i]["presentation_order"] for i in done)}
        r["version_matches"] = set(r["returned_versions"]) == set(r["rq1kj_versions"])
        complete &= len(done) == len(old)
        passed &= not r["stop"] and r["same_orders"]
        result[v.host] = r
    return {"cells": result, "complete": complete, "passed": complete and passed}


def precheckPath(precheck_dir: str, v: Version) -> str:
    return substitutionJudgePath(precheck_dir, v.host, v.dataset, v.menu, v.version)


# ------------------------------------------------------------------
# §11 試跑
# ------------------------------------------------------------------
def evaluatePilot(out_dir: str, plan: Plan, means: dict, stage0: dict) -> dict:
    segments = {}
    complete = True
    for v, chosen in pilotVersions(plan):
        _, recs = loadRecords(v.path(out_dir)) if os.path.exists(v.path(out_dir)) else ({}, {})
        seg = segments.setdefault(v.K, {"records": [], "missing": 0})
        missing = [i for i in chosen if i not in recs]
        seg["missing"] += len(missing)
        complete &= not missing
        for i in chosen:
            if i in recs:
                seg["records"].append((v, recs[i]))
    out = {"segments": {}, "complete": complete}
    ratios = {}
    for K, seg in segments.items():
        records = [r for _, r in seg["records"]]
        kinds = Counter()
        for r in records:
            c = classifyRecord(r)
            kinds.update({k: int(bool(c[k])) for k in ("no_choice", "out_of_range", "refused")})
        n = len(records)
        invalid = kinds["no_choice"] + kinds["out_of_range"] + kinds["refused"]
        api_in = [r["call"]["usage_in"] for r in records if r.get("call")]
        api_out = [r["call"]["usage_out"] for r in records if r.get("call")]
        positions = Counter(r["trace"].get("choice") for r in records if r["trace"].get("choice"))
        actual = float(np.mean([callCost(PILOT_BLOCK[0], i, o) for i, o in zip(api_in, api_out)])) if api_in else float("nan")
        est = callCost(PILOT_BLOCK[0], *means[(K, *PILOT_BLOCK, PILOT_PAIR if K == 2 else None)])
        ratios[K] = actual / est
        out["segments"][f"K{K}"] = {
            "calls": n, "no_choice": kinds["no_choice"], "out_of_range": kinds["out_of_range"], "refused": kinds["refused"],
            "invalid_rate": invalid / n if n else float("nan"), "out_of_range_rate": kinds["out_of_range"] / n if n else float("nan"),
            "tokens_in_api": {"mean": float(np.mean(api_in)), "max": int(max(api_in))} if api_in else {},
            "tokens_out_api": {"mean": float(np.mean(api_out)), "max": int(max(api_out))} if api_out else {},
            "tokens_in_recount": {"mean": float(np.mean([r["tokens_in"] for r in records])), "max": int(max(r["tokens_in"] for r in records))} if n else {},
            "tokens_out_recount": {"mean": float(np.mean([r["tokens_out"] for r in records])), "max": int(max(r["tokens_out"] for r in records))} if n else {},
            "position_share": {str(p): positions[p] / sum(positions.values()) for p in sorted(positions)} if positions else {},
            "cost_per_call_actual": actual, "cost_per_call_stage0": est, "ratio": ratios[K], "missing": seg["missing"]}
    projected = stage0["K3"] * ratios.get(3, float("nan")) + stage0["K2"] * ratios.get(2, float("nan"))
    out["projected_total_usd"] = projected
    out["stage0_total_usd"] = stage0["total"]
    out["max_cost_usd"] = MAX_COST
    out["stop_invalid"] = any(s["invalid_rate"] > PILOT_MAX_INVALID for s in out["segments"].values())
    out["stop_cost"] = not projected <= MAX_COST
    out["passed"] = complete and not out["stop_invalid"] and not out["stop_cost"]
    return out
