import gzip
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from Analysis.alignment import loadRecords
from Analysis.blockStats import summarize, fourState
from Analysis.crossJudge import classifyRecord
from Analysis.menuVote import DATASETS
from Analysis.pathImprove import M12, HOSTS, DONORS
from Analysis.judgeSubstitution import (GROUP_ORDER, GROUP_MENUS, MENU_ORDER, MIDDLE_TRANSLATED, PAIRS, PRICES, Plan, disagree,
                                        callCost)
from Analysis.judgeSubstitutionPredict import (NONE, UNANIMOUS, PRED_FILE, LODO_FILE, CROSS_FILE, EFFECT_FILE, MANIFEST, voteCells,
                                               positions)

# ------------------------------------------------------------------
# RQ3-GJ（result/analysis/rq3gj/rq3gj_criteria.md）的分析：逐區塊讀 Judge 輸出，算第 5、6、8 節的量
#   正確率都在子集二 (h, g) 上；原本的版本對兩個供體各算一次。K = 2 每題的得分 = 兩種順序對錯的平均。
# ------------------------------------------------------------------
SMALL_DEN = 0.10                  # 第 5 節：某區塊某一群的分母總和 < 10pp（比例 0.10）就不算
MIN_BLOCKS = 6                    # 判定少於 6 個區塊 -> 無法判定
THRESHOLD = 0.5
STATES = ["三條相同且答對", "三條相同且答錯", "票型一", "票型二", "票型三", "不完全相同且沒有一條對"]
CATEGORIES = ["a", "b", "c", "d"]


def loadPredictions(out_dir: str) -> dict:
    """讀第 7 節存好的預測；sha256 要等於 manifest。回傳 {名稱: {(host, dataset, menu, version, subset_donor): (R,) 陣列}}。"""
    with open(os.path.join(out_dir, MANIFEST), encoding="utf-8") as f:
        manifest = json.load(f)
    for name, sha in manifest["files"].items():
        with open(os.path.join(out_dir, name), "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != sha:
                raise SystemExit(f"❌ {name} does not match its sha256 in {MANIFEST}; predictions must not change after the full run")
    out = {}
    for key, name in (("pred", PRED_FILE), ("lodo", LODO_FILE), ("cross", CROSS_FILE)):
        frame = pd.read_csv(os.path.join(out_dir, name)).sort_values(["host", "dataset", "menu", "version", "subset_donor", "split"])
        out[key] = {k: g.pred_acc.to_numpy() for k, g in frame.groupby(["host", "dataset", "menu", "version", "subset_donor"], sort=False)}
    out["effects"] = pd.read_csv(os.path.join(out_dir, EFFECT_FILE))
    out["manifest"] = manifest
    return out


def stateOf(cell: np.ndarray, unan_correct: np.ndarray) -> np.ndarray:
    """§8.11 的六種狀態（索引同 STATES）。"""
    s = np.full(cell.shape, 5)
    s[(cell == UNANIMOUS) & unan_correct] = 0
    s[(cell == UNANIMOUS) & ~unan_correct] = 1
    s[(cell >= 0) & (cell < 3)] = 2
    s[(cell >= 3) & (cell < 6)] = 3
    s[cell >= 6] = 4
    return s


def voteScore(codes: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """多數決（規則 C）的每題得分。K = 2 = 兩條對錯的平均；K = 3：三條都不同時，正確答案在其中得 1/3。"""
    corr = codes == gold[None, :]
    if len(codes) == 2:
        return corr.mean(axis=0)
    alldiff = (codes[0] != codes[1]) & (codes[0] != codes[2]) & (codes[1] != codes[2])
    majority = np.where(codes[0] == codes[1], codes[0], np.where(codes[0] == codes[2], codes[0], codes[1]))
    return np.where(alldiff, corr.any(axis=0) / 3.0, (majority == gold).astype(float))


# ------------------------------------------------------------------
# 一個區塊：讀它所有版本的 Judge 檔
# ------------------------------------------------------------------
def processBlock(plan: Plan, coded: dict, gold_text: dict, h: str, d: str, out_dir: str, items_writer=None) -> dict:
    H = coded[(h, d)]
    N = len(H.item_ids)
    index = {int(i): n for n, i in enumerate(H.item_ids)}
    gold = gold_text[(h, d)]
    sub2 = {g: H.sub & coded[(g, d)].sub for g in DONORS}
    union = sub2[DONORS[0]] | sub2[DONORS[1]]
    J, codes_of, rec_rows, missing, versions = {}, {}, [], [], set()
    for v in [v for v in plan.versions if v.host == h and v.dataset == d and v.participates]:
        codes = H.codes[[M12.index(c) for c in v.codes]].copy()
        if v.donor:
            codes[v.slot] = coded[(v.donor, d)].codes[M12.index(v.codes[v.slot])]
        codes_of[(v.K, v.menu, v.version)] = codes
        domain = union if v.version == "orig" else sub2[v.donor]
        unan = domain & ~disagree(codes)
        arr = np.full(N, np.nan)
        arr[unan] = (codes[0][unan] == H.gold[unan]).astype(float)
        path = v.path(out_dir)
        meta, recs = loadRecords(path) if os.path.exists(path) else ({}, {})
        versions |= set(meta.get("model_versions", {}))
        lost = set(v.items) - set(recs)
        if lost:
            missing.append((v.menu, v.version, v.order, len(lost)))
            continue
        orders = plan.ordersOf(v)
        pos = positions(orders, v.arm_ids, H.item_ids)
        cells = voteCells(codes, H.gold, pos) if v.K == 3 else None
        for i in v.items:
            r, n = recs[i], index[i]
            final = r["final_answer"]
            correct = bool(final) and final == gold[n]
            arr[n] = float(correct)
            kinds = classifyRecord(r)
            trace, call = r.get("trace") or {}, r.get("call") or {}
            row = {"host": h, "dataset": d, "K": v.K, "menu": v.menu, "version": v.version, "order": v.order, "donor": v.donor,
                   "slot": v.slot, "item_id": i, "col": n, "choice": trace.get("choice"), "chosen_arm": trace.get("chosen_arm"),
                   "final_answer": final, "correct": correct, "no_choice": kinds["no_choice"], "out_of_range": kinds["out_of_range"],
                   "refused": kinds["refused"], "tokens_in": r.get("tokens_in"), "tokens_out": r.get("tokens_out"),
                   "usage_in": call.get("usage_in"), "usage_out": call.get("usage_out"), "pilot": bool(r.get("pilot")),
                   "donor_position": r.get("donor_position"), "cell": int(cells[n]) if cells is not None else None,
                   "alldiff": bool(len(set(codes[:, n].tolist())) == len(codes)),
                   "slot_positions": "".join(str(orders[i].index(a)) for a in v.arm_ids)}
            if v.K == 3 and row["cell"] is not None and 3 <= row["cell"] < 6:
                c = codes[:, n]
                lone = [j for j in range(3) if (c == c[j]).sum() == 1][0]
                row["lone_is_donor"] = v.slot is not None and lone == v.slot
            rec_rows.append(row)
            if items_writer is not None:
                items_writer({**row, "presentation_order": "|".join(r["presentation_order"]), "model_version": call.get("model_version"),
                              "called_at": call.get("called_at"), "prompt_sha256": trace.get("prompt_sha256")})
        J[(v.K, v.menu, v.version, v.order)] = arr
    return {"host": h, "dataset": d, "N": N, "J": J, "codes": codes_of, "records": pd.DataFrame(rec_rows), "missing": missing,
            "sub2": sub2, "union": union, "model_versions": versions}


def judgeArray(block: dict, K: int, menu: str, version: str) -> np.ndarray:
    """K = 3 的 Judge 對錯；K = 2 = 兩種順序的平均。"""
    if K == 3:
        return block["J"][(3, menu, version, None)]
    return (block["J"][(2, menu, version, "A")] + block["J"][(2, menu, version, "B")]) / 2


# ------------------------------------------------------------------
# 逐替換的量（子集二全部題目）與逐切分的量（評分半）
# ------------------------------------------------------------------
def substitutionRows(plan: Plan, coded: dict, block: dict, splits: np.ndarray, preds: dict) -> tuple[list, dict]:
    """回傳 (逐替換的列, 逐切分的陣列 {key: (R,)})。K = 2 的替換也在其中（K 欄位）。"""
    h, d = block["host"], block["dataset"]
    H = coded[(h, d)]
    rows, per_split = [], {}
    h2_all = ~splits
    for v in [v for v in plan.versions if v.host == h and v.dataset == d and v.participates and v.version != "orig"
              and (v.K == 3 or v.order == "A")]:
        g, p = v.donor, v.codes[v.slot]
        s2 = block["sub2"][g]
        n2 = int(s2.sum())
        Js, Jo = judgeArray(block, v.K, v.menu, v.version), judgeArray(block, v.K, v.menu, "orig")
        cs, co = block["codes"][(v.K, v.menu, v.version)], block["codes"][(v.K, v.menu, "orig")]
        Vs, Vo = voteScore(cs, H.gold), voteScore(co, H.gold)
        k = M12.index(p)
        host_acc, donor_acc = H.correct[k][s2].mean(), coded[(g, d)].correct[k][s2].mean()
        sb = lambda c: max(range(len(c)), key=lambda j: ((c[j] == H.gold)[s2].mean(), -j))   # 子集二上正確率最高（平手取順序在前）
        row = {"K": v.K, "host": h, "dataset": d, "menu": v.menu, "path": p, "donor": g, "version": v.version, "n2": n2,
               "role": ("lone" if p == plan.menus[v.menu]["lone"] else "pair") if v.K == 3 else None,
               "A_J_orig": np.nanmean(Jo[s2]), "A_J_sub": np.nanmean(Js[s2]), "A_V_orig": Vo[s2].mean(), "A_V_sub": Vs[s2].mean(),
               "host_acc": host_acc, "donor_acc": donor_acc,
               "SB_orig": (co[sb(co)] == H.gold)[s2].mean(), "SB_sub": (cs[sb(cs)] == H.gold)[s2].mean()}
        if np.isnan(Js[s2]).any() or np.isnan(Jo[s2]).any():
            raise ValueError(f"{h} | {d} | {v.menu} | {v.version}: Judge result missing on subset 2")
        row.update({"numJ": row["A_J_sub"] - row["A_J_orig"], "numV": row["A_V_sub"] - row["A_V_orig"], "den": donor_acc - host_acc})
        if v.K == 2:
            hosts = [(c == H.gold)[s2].mean() for c in co]
            row["higher"] = v.slot == max(range(2), key=lambda j: (hosts[j], -j))
        if v.K == 3:
            h2 = h2_all & s2[None, :]
            n = h2.sum(axis=1)
            act_s, act_o = (h2 @ np.nan_to_num(Js)) / n, (h2 @ np.nan_to_num(Jo)) / n        # 子集二以外是 NaN，評分半 ∩ 子集二不含它們
            key = (h, d, v.menu, v.version, g)
            pred_s, pred_o = preds["pred"][key], preds["pred"][(h, d, v.menu, "orig", g)]
            per_split[(v.menu, v.version)] = {
                "act_s": act_s, "act_o": act_o, "pred_s": pred_s, "pred_o": pred_o,
                "lodo_s": preds["lodo"][key], "cross_s": preds["cross"][key],
                "vote_s": (h2 @ Vs) / n, "vote_o": (h2 @ Vo) / n,
                "den": (h2.astype(int) @ coded[(g, d)].correct[k].astype(int) - h2.astype(int) @ H.correct[k].astype(int)) / n}
            dp = 100 * (act_s - pred_s)
            row.update({"D_P": dp.mean(), "MAE": np.abs(dp).mean(), "D_P_lodo": (100 * (act_s - preds["lodo"][key])).mean(),
                        "MAE_lodo": np.abs(100 * (act_s - preds["lodo"][key])).mean(),
                        "D_P_cross": (100 * (act_s - preds["cross"][key])).mean(),
                        "MAE_cross": np.abs(100 * (act_s - preds["cross"][key])).mean(),
                        "gain_actual_H2": 100 * (act_s - act_o).mean(), "gain_pred_H2": 100 * (pred_s - pred_o).mean()})
        rows.append(row)
    return rows, per_split


def originalRows(plan: Plan, coded: dict, block: dict, splits: np.ndarray, preds: dict) -> list:
    """§8.7(a)：原本的版本，每份菜單在兩個供體的子集二上各一列；§8.2 的原本的版本。"""
    h, d = block["host"], block["dataset"]
    H = coded[(h, d)]
    rows = []
    for menu in MENU_ORDER:
        Jo = judgeArray(block, 3, menu, "orig")
        co = block["codes"][(3, menu, "orig")]
        for g in DONORS:
            s2 = block["sub2"][g]
            h2 = ~splits & s2[None, :]
            act = (h2 @ np.nan_to_num(Jo)) / h2.sum(axis=1)
            dp = 100 * (act - preds["pred"][(h, d, menu, "orig", g)])
            if np.isnan(Jo[s2]).any():
                raise ValueError(f"{h} | {d} | {menu} | orig: Judge result missing on subset 2 of {g}")
            best = max(range(3), key=lambda j: ((co[j] == H.gold)[s2].mean(), -j))
            rows.append({"host": h, "dataset": d, "menu": menu, "subset_donor": g, "A_J": np.nanmean(Jo[s2]),
                         "A_V": voteScore(co, H.gold)[s2].mean(), "SB": (co[best] == H.gold)[s2].mean(),
                         "D_P": dp.mean(), "MAE": np.abs(dp).mean()})
    return rows


# ------------------------------------------------------------------
# §8.11 票型轉換
# ------------------------------------------------------------------
def transitionRows(plan: Plan, coded: dict, block: dict) -> list:
    h, d = block["host"], block["dataset"]
    H = coded[(h, d)]
    rows = []
    for v in [v for v in plan.versions if v.host == h and v.dataset == d and v.K == 3 and v.participates and v.version != "orig"]:
        s2 = block["sub2"][v.donor]
        n2 = s2.sum()
        cs, co = block["codes"][(3, v.menu, v.version)], block["codes"][(3, v.menu, "orig")]
        zero = np.zeros((3, H.codes.shape[1]), dtype=int)
        cell_s, cell_o = voteCells(cs, H.gold, zero), voteCells(co, H.gold, zero)            # 位置不影響狀態
        st_s = stateOf(cell_s, cs[0] == H.gold)
        st_o = stateOf(cell_o, co[0] == H.gold)
        lone_after = np.array([[j for j in range(3) if (cs[:, n] == cs[j, n]).sum() == 1][0] if st_s[n] == 3 else -1
                               for n in range(cs.shape[1])])
        dJ = judgeArray(block, 3, v.menu, v.version) - judgeArray(block, 3, v.menu, "orig")
        dV = voteScore(cs, H.gold) - voteScore(co, H.gold)
        cat = np.full(cs.shape[1], 3)
        a = (st_o == 2) & (st_s == 0)
        cat[(st_s == 3) & (lone_after == v.slot)] = 1
        cat[((st_s == 2) | (st_s == 0)) & ~a] = 2
        cat[a] = 0
        role = "lone" if v.codes[v.slot] == plan.menus[v.menu]["lone"] else "pair"
        base = {"host": h, "dataset": d, "menu": v.menu, "version": v.version, "role": role, "n2": int(n2)}
        for so in range(6):
            for ss in range(6):
                m = s2 & (st_o == so) & (st_s == ss)
                if m.any():
                    rows.append({**base, "kind": "cell", "before": STATES[so], "after": STATES[ss], "n": int(m.sum()),
                                 "judge_change": float(dJ[m].sum()), "vote_change": float(dV[m].sum())})
        for c in range(4):
            m = s2 & (cat == c)
            rows.append({**base, "kind": "category", "category": CATEGORIES[c], "n": int(m.sum()),
                         "judge_change": float(dJ[m].sum()), "vote_change": float(dV[m].sum()),
                         "judge_pp": 100 * float(dJ[m].sum()) / n2, "vote_pp": 100 * float(dV[m].sum()) / n2})
    return rows


# ------------------------------------------------------------------
# 區塊層級的效果（第 5 節）、判定（第 6 節）
# ------------------------------------------------------------------
def groupEffect(subs: pd.DataFrame, num: str = "numJ") -> tuple[float, float]:
    """10 × 分子總和 ÷ 分母總和；回傳 (效果, 分母總和)。分母總和 < 10pp 時效果為 NaN（第 5 節）。"""
    den = subs.den.sum()
    if len(subs) == 0:
        return float("nan"), 0.0
    return (10 * subs[num].sum() / den if den >= SMALL_DEN else float("nan")), float(den)


def ejOf(subs: pd.DataFrame, menus: list, num: str = "numJ") -> dict:
    s = subs[(subs.K == 3) & subs.menu.isin(menus)]
    pair, den_p = groupEffect(s[s.role == "pair"], num)
    lone, den_l = groupEffect(s[s.role == "lone"], num)
    if len(s[s.role == "pair"]) == 0 or len(s[s.role == "lone"]) == 0:
        raise SystemExit("❌ §5：some block has no participating substitution in a group; stop and report")
    return {"E": pair - lone, "pair": pair, "lone": lone, "den_pair": den_p, "den_lone": den_l}


def summ(values) -> dict:
    """跨區塊摘要；NaN（分母過小）的區塊不計入，另記。"""
    values = np.asarray(values, dtype=float)
    kept = values[np.isfinite(values)]
    out = summarize(kept) if len(kept) > 1 else {"n_blocks": len(kept), "mean": float(np.mean(kept)) if len(kept) else float("nan"),
                                                 "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                                                 "n_positive": int((kept > 0).sum())}
    out["n_dropped"] = int(len(values) - len(kept))
    return out


def state(s: dict, threshold: float = THRESHOLD) -> str:
    """四種狀態；分母過小而少於 MIN_BLOCKS 個區塊時 = 無法判定（第 5 節）。"""
    if s["n_blocks"] < MIN_BLOCKS or not np.isfinite(s["ci_low"]):
        return "無法判定"
    return fourState(s, threshold)


def spearmanSplits(per_split: dict) -> tuple[float, int]:
    """§8.6：每次切分，參與的替換之間「預測多的」對「實際多的」的 Spearman；無定義的切分不計。回傳 (平均, 無定義的次數)。"""
    keys = list(per_split)
    if len(keys) < 3:
        return float("nan"), -1
    pred = np.array([per_split[k]["pred_s"] - per_split[k]["pred_o"] for k in keys])        # (S, R)
    act = np.array([per_split[k]["act_s"] - per_split[k]["act_o"] for k in keys])
    vals, undefined = [], 0
    for r in range(pred.shape[1]):
        x, y = pred[:, r], act[:, r]
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            undefined += 1
            continue
        vals.append(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return float(np.mean(vals)) if vals else float("nan"), undefined


def splitEffects(plan: Plan, per_split: dict, kind: str) -> dict:
    """§8.12：每次切分在評分半上的 E_J（四組）與 D_JV；kind = "act" 用實際的 Judge 正確率，"pred" 用預測。"""
    R = len(next(iter(per_split.values()))["act_s"])
    out = {}
    for group in GROUP_ORDER:
        sums = {"pair": [np.zeros(R), np.zeros(R)], "lone": [np.zeros(R), np.zeros(R)]}
        for (menu, version), s in per_split.items():
            if menu not in GROUP_MENUS[group]:
                continue
            path = version.split("-", 1)[0]
            role = "lone" if path == plan.menus[menu]["lone"] else "pair"
            sums[role][0] += s[f"{kind}_s"] - s[f"{kind}_o"]
            sums[role][1] += s["den"]
        out[f"E_J_{group}"] = 10 * (sums["pair"][0] / sums["pair"][1] - sums["lone"][0] / sums["lone"][1])
    out["D_JV"] = np.mean([100 * ((s[f"{kind}_s"] - s[f"{kind}_o"]) - (s["vote_s"] - s["vote_o"])) for s in per_split.values()], axis=0)
    return out


# ------------------------------------------------------------------
# §8.7(c)：預測在 RQ1-KJ 紀錄上（訓練用的題目 vs 沒看過的題目）
# ------------------------------------------------------------------
def trainingErrors(train: pd.DataFrame, tables: dict, coded: dict, answered: dict, splits: dict) -> pd.DataFrame:
    """
    每個（裁判、資料集、M3 菜單），在菜單的子集（三條都有答案，RQ1-KJ 的子集_in）上：Judge 的實際正確率 − 第 7 節的預測。
    訓練用的題目 = 選擇半 H1_r，沒看過的 = 評分半 H2_r；有正負號與絕對值兩種，對切分與三份菜單平均（pp）。
    answered = {(model, dataset): {path 代號: 是否有答案（依 item_id 排序）}}（Analysis.menuVote.PathBlock.answered）。
    """
    from Analysis.judgeSubstitutionPredict import M3_MENUS
    rows = []
    for (h, d), g in train.groupby(["judge", "dataset"], sort=False):
        block = coded[(h, d)]
        N = block.codes.shape[1]
        q = tables[h]                                                     # (R, 9)
        res = {"train": [], "unseen": [], "train_abs": [], "unseen_abs": []}
        for menu, codes in M3_MENUS.items():
            c = block.codes[[M12.index(x) for x in codes]]
            subset = np.all([answered[(h, d)][x] for x in codes], axis=0)
            unan = subset & (c[0] == c[1]) & (c[1] == c[2])
            gm = g[g.menu == menu]
            cols = gm.col.to_numpy()
            if not np.array_equal(np.sort(cols), np.flatnonzero(subset & ~unan)):
                raise ValueError(f"{h} | {d} | {menu}: RQ1-KJ records are not the menu's disagreement items")
            known = (unan & (c[0] == block.gold)).astype(float)
            called_correct = np.zeros(N)
            called_correct[cols] = gm.correct.to_numpy()
            cell = np.full(N, NONE)
            cell[cols] = gm.cell.to_numpy()
            onehot = np.eye(9)[np.clip(cell, 0, 8)] * (cell >= 0)[:, None]
            for half, name in ((splits[d], "train"), (~splits[d], "unseen")):
                m = (half & subset[None, :]).astype(float)
                n = m.sum(axis=1)
                err = 100 * ((m @ (known + called_correct)) - (m @ known + ((m @ onehot) * q).sum(axis=1))) / n
                res[name].append(err.mean())
                res[f"{name}_abs"].append(np.abs(err).mean())
        rows.append({"judge": h, "dataset": d, **{k: float(np.mean(v)) for k, v in res.items()}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# §8.14 K = 2 的 (b)、(c)
# ------------------------------------------------------------------
def k2Counts(plan: Plan, coded: dict, block: dict) -> dict:
    """
    (b) 兩條答案不同、剛好一條對的題目，以每次呼叫為單位：{(類別, 對的那條的位置): [次數, 選對的次數]}；
        類別 = donor（對的是換進來的強模型 path）、host（對的是宿主自己的 path，替換的版本）、base（原本的版本）。
    (c) 兩種順序選到同一條 path 的題數：{"orig" / "sub": [題數, 相同的題數]}（兩次都沒有選擇算相同）。
    """
    from Analysis.experimentPlan import PATHS
    h, d = block["host"], block["dataset"]
    gold = coded[(h, d)].gold
    rec = block["records"]
    rec = rec[rec.K == 2]
    counts, same = defaultdict(lambda: [0, 0]), {"orig": [0, 0], "sub": [0, 0]}
    for (menu, version), g in rec.groupby(["menu", "version"], sort=False):
        cs = block["codes"][(2, menu, version)]
        corr = cs == gold[None, :]
        one = (cs[0] != cs[1]) & (corr.sum(axis=0) == 1)
        slot = None if version == "orig" else PAIRS[menu].index(version.split("-", 1)[0])
        for r in g.itertuples():
            if not one[r.col]:
                continue
            j = int(np.argmax(corr[:, r.col]))
            kind = "base" if slot is None else ("donor" if j == slot else "host")
            cell = counts[(kind, int(r.slot_positions[j]) + 1)]
            cell[0] += 1
            cell[1] += int(r.correct)
        a = g[g.order == "A"].set_index("item_id").chosen_arm
        b = g[g.order == "B"].set_index("item_id").chosen_arm
        both = a.index.intersection(b.index)
        key = "orig" if version == "orig" else "sub"
        same[key][0] += len(both)
        same[key][1] += int(sum((x == y) or (x is None and y is None) or (pd.isna(x) and pd.isna(y)) for x, y in zip(a[both], b[both])))
    return {"counts": dict(counts), "same": same}


# ------------------------------------------------------------------
# §8.8 Qwen：兩票對一票、選了落單那條的（版本、題目）
# ------------------------------------------------------------------
def qwenPopulation(plan: Plan, coded: dict, block: dict) -> list:
    from Analysis.experimentPlan import PATHS
    h, d = block["host"], block["dataset"]
    if h != "qwen":
        return []
    rec = block["records"]
    out = []
    for (menu, version), g in rec[rec.K == 3].groupby(["menu", "version"], sort=False):
        codes = plan.menus[menu]["codes"]
        arm_ids = [PATHS[c] for c in codes]
        cs = block["codes"][(3, menu, version)]
        if version == "orig":
            rank = 0
        else:
            path, donor = version.split("-", 1)
            rank = 1 + 2 * codes.index(path) + DONORS.index(donor)
        for r in g.itertuples():
            c = cs[:, r.col]
            if len(set(c.tolist())) != 2 or not isinstance(r.chosen_arm, str):
                continue
            lone = [j for j in range(3) if (c == c[j]).sum() == 1][0]
            if arm_ids.index(r.chosen_arm) == lone:
                out.append({"dataset": d, "menu": menu, "version": version, "item_id": int(r.item_id),
                            "sort": (DATASETS.index(d), MENU_ORDER.index(menu), rank, int(r.item_id))})
    return out
