import os

import numpy as np
import pandas as pd

from Analysis.alignment import loadRecords
from Analysis.experimentPlan import PATHS
from Analysis.menuVote import DATASETS
from Analysis.pathImprove import M12, HOSTS, DONORS
from Analysis.judgeSubstitution import GROUP_ORDER, GROUP_MENUS, MENU_ORDER, RQ1KJ_JUDGE_DIR, Plan

# ------------------------------------------------------------------
# RQ3-GJ §7、§8.12、§8.13 的預測（只用 RQ1-KJ 既有的三條 Judge 紀錄、答案與事先決定的順序；正式跑之前存檔，之後不可重算）
#   票型（三條候選的答案與 gold）：
#     0 = 兩票對一票、正確的是多數（位置 = 落單那條）；1 = 兩票對一票、正確的是落單那條（位置 = 它）；
#     2 = 三條都不同、其中一條對（位置 = 對的那條）。沒有一條對 = 預測 0；三條相同 = 用共同答案的對錯。
#   格子 = 票型 × 3 + 位置（0 起算），9 格；平滑 (答對 + 1) ÷ (題數 + 2)。
# ------------------------------------------------------------------
NONE, UNANIMOUS = -1, -2
M3_MENUS = {"M3L": ["EN", "ZH", "JA"], "M3S": ["EN", "S1", "S2"], "M3P": ["EN", "P1", "P2"]}
PRED_FILE, TABLE_FILE, EFFECT_FILE = "rq3gj_predictions.csv.gz", "rq3gj_prediction_tables.csv", "rq3gj_predicted_effects.csv"
LODO_FILE, CROSS_FILE = "rq3gj_predictions_lodo.csv.gz", "rq3gj_predictions_crossjudge.csv.gz"
MANIFEST = "rq3gj_predictions_manifest.json"


def voteCells(codes: np.ndarray, gold: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    codes (3, N) 答案代碼、gold (N,)、pos (3, N) 每條候選顯示的位置（0–2）。回傳每題的格子：0–8、NONE（不一致且沒有一條對）
    或 UNANIMOUS（三條相同）。
    """
    N = codes.shape[1]
    e01, e02, e12 = codes[0] == codes[1], codes[0] == codes[2], codes[1] == codes[2]
    corr = codes == gold[None, :]
    cell = np.full(N, NONE, dtype=int)
    cell[e01 & e02] = UNANIMOUS
    two = ~(e01 & e02) & (e01 | e02 | e12)
    lone = np.where(e01, 2, np.where(e02, 1, 0))
    cols = np.arange(N)
    lone_corr = corr[lone, cols]
    major_corr = corr[np.where(lone == 0, 1, 0), cols]
    lone_pos = pos[lone, cols]
    t0 = two & major_corr
    t1 = two & lone_corr
    cell[t0] = 0 * 3 + lone_pos[t0]
    cell[t1] = 1 * 3 + lone_pos[t1]
    alldiff = ~(e01 | e02 | e12)
    right = corr.argmax(axis=0)
    t2 = alldiff & corr.any(axis=0)
    cell[t2] = 2 * 3 + pos[right, cols][t2]
    return cell


def positions(order_map: dict, arm_ids: list, item_ids: np.ndarray) -> np.ndarray:
    """(3, N)：每條 path 在該題順序中的位置；沒有順序的題目為 -1。"""
    pos = np.full((len(arm_ids), len(item_ids)), -1, dtype=int)
    for col, i in enumerate(item_ids.tolist()):
        order = order_map.get(int(i))
        if order:
            for j, a in enumerate(arm_ids):
                pos[j, col] = order.index(a)
    return pos


# ------------------------------------------------------------------
# 訓練資料：RQ1-KJ 的 M3L、M3S、M3P（宿主自己裁決自己）
# ------------------------------------------------------------------
def trainingRecords(coded: dict, gold_text: dict, rq1kj_dir: str = RQ1KJ_JUDGE_DIR) -> pd.DataFrame:
    """
    每筆 RQ1-KJ 紀錄：裁判、資料集、菜單、題目、在切分陣列中的位置、格子、裁判是否選對。
    gold_text = {(model, dataset): gold 字串（依 item_id 排序，Analysis.menuVote.PathBlock.gold）}；選對 = 最終答案等於 gold
    （和 RQ1-KJ 的計分相同；agg_no_answer 算錯）。
    """
    rows = []
    for h in HOSTS:
        for d in DATASETS:
            block = coded[(h, d)]
            index = {int(i): n for n, i in enumerate(block.item_ids)}
            gold = gold_text[(h, d)]
            for menu, codes in M3_MENUS.items():
                _, recs = loadRecords(os.path.join(rq1kj_dir, h, d, f"{menu}.json"))
                arm_ids = [PATHS[c] for c in codes]
                ids = np.array(sorted(int(i) for i in recs))
                cols = np.array([index[i] for i in ids])
                pos = positions({i: recs[i]["presentation_order"] for i in ids.tolist()}, arm_ids, ids)
                cell = voteCells(block.codes[[M12.index(c) for c in codes]][:, cols], block.gold[cols], pos)
                for i, col, c in zip(ids.tolist(), cols.tolist(), cell.tolist()):
                    final = recs[i]["final_answer"]
                    rows.append({"judge": h, "dataset": d, "menu": menu, "item_id": i, "col": col, "cell": c,
                                 "correct": bool(final) and final == gold[col]})
    return pd.DataFrame(rows)


def tableBySplit(train: pd.DataFrame, judge: str, splits: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§7：每次切分，用 RQ1-KJ 紀錄中 item 在選擇半的題目估 9 格。回傳 (q, k, n)，各 (R, 9)。"""
    R = len(next(iter(splits.values())))
    k, n = np.zeros((R, 9)), np.zeros((R, 9))
    t = train[(train.judge == judge) & (train.cell >= 0)]
    for d, g in t.groupby("dataset"):
        h1 = splits[d][:, g.col.to_numpy()].astype(float)              # (R, records)
        onehot = np.eye(9)[g.cell.to_numpy()]
        n += h1 @ onehot
        k += h1 @ (onehot * g.correct.to_numpy()[:, None])
    return (k + 1) / (n + 2), k, n


def fixedTable(rows: pd.DataFrame, by_type: bool = False) -> np.ndarray:
    """全部題目（不切分）的表：9 格；by_type 時只分票型（3 格），再展開成 9 格（同票型的三個位置相同）。"""
    rows = rows[rows.cell >= 0]
    if by_type:
        types = rows.cell.to_numpy() // 3
        q3 = np.array([(rows.correct.to_numpy()[types == t].sum() + 1) / ((types == t).sum() + 2) for t in range(3)])
        return np.repeat(q3, 3)
    cells = rows.cell.to_numpy()
    return np.array([(rows.correct.to_numpy()[cells == c].sum() + 1) / ((cells == c).sum() + 2) for c in range(9)])


# ------------------------------------------------------------------
# 版本的逐題資料（K = 3）：格子、三條相同且答對、多數決（規則 C）的得分；正確率都在 H2_r ∩ 子集二 (h, g) 上
# ------------------------------------------------------------------
def versionArrays(plan: Plan, coded: dict, v) -> dict:
    H, D = coded[(v.host, v.dataset)], coded[(v.donor, v.dataset)] if v.donor else None
    codes = H.codes[[M12.index(c) for c in v.codes]].copy()
    if v.donor:
        codes[v.slot] = D.codes[M12.index(v.codes[v.slot])]
    pos = positions(plan.ordersOf(v), v.arm_ids, H.item_ids)
    cell = voteCells(codes, H.gold, pos)
    corr = codes == H.gold[None, :]
    unan_correct = (cell == UNANIMOUS) & corr[0]
    alldiff = (codes[0] != codes[1]) & (codes[0] != codes[2]) & (codes[1] != codes[2])
    majority = np.where(codes[0] == codes[1], codes[0], np.where(codes[0] == codes[2], codes[0], codes[1]))
    vote = np.where(alldiff, corr.any(axis=0) / 3.0, (majority == H.gold).astype(float))
    return {"cell": cell, "unan_correct": unan_correct, "vote": vote}


def predictHalf(arr: dict, h2: np.ndarray, q: np.ndarray) -> np.ndarray:
    """h2 (R, N) 評分半 ∩ 子集二；q (R, 9) 或 (9,)。回傳每次切分的預測正確率。"""
    h = h2.astype(float)
    known = h @ arr["unan_correct"].astype(float)
    cells = arr["cell"]
    counts = h @ (np.eye(9)[np.clip(cells, 0, 8)] * (cells >= 0)[:, None])          # (R, 9) 各格的題數
    qq = q if q.ndim == 2 else np.broadcast_to(q, counts.shape)
    return (known + (counts * qq).sum(axis=1)) / h.sum(axis=1)


def computePredictions(plan: Plan, coded: dict, splits: dict, train: pd.DataFrame) -> dict:
    """§7 的預測、§8.12 的預測 E_J 與 D_JV、§8.13 的兩種泛化預測。"""
    tables = {j: tableBySplit(train, j, splits) for j in HOSTS}
    lodo = {(j, d): fixedTable(train[(train.judge == j) & (train.dataset != d)]) for j in HOSTS for d in DATASETS}
    cross = {j: fixedTable(train[train.judge == other], by_type=True) for j, other in zip(HOSTS, reversed(HOSTS))}
    pred_rows, lodo_rows, cross_rows, effect_rows = [], [], [], []
    role = {m: plan.menus[m]["lone"] for m in MENU_ORDER}
    for h in HOSTS:
        for d in DATASETS:
            H = coded[(h, d)]
            q = tables[h][0]
            R = q.shape[0]
            subs = {}
            orig = {}
            for v in [v for v in plan.versions if v.K == 3 and v.host == h and v.dataset == d]:
                arr = versionArrays(plan, coded, v)
                for g in ([v.donor] if v.donor else DONORS):
                    if v.donor and not v.participates:
                        continue
                    s2 = H.sub & coded[(g, d)].sub
                    h2 = ~splits[d] & s2[None, :]
                    n2 = h2.sum(axis=1)
                    p, pl, pc = predictHalf(arr, h2, q), predictHalf(arr, h2, lodo[(h, d)]), predictHalf(arr, h2, cross[h])
                    vote = h2.astype(float) @ arr["vote"] / n2
                    for rows, vals in ((pred_rows, p), (lodo_rows, pl), (cross_rows, pc)):
                        rows.extend({"host": h, "dataset": d, "menu": v.menu, "version": v.version, "subset_donor": g, "split": r,
                                     "n2": int(n2[r]), "pred_acc": float(vals[r])} for r in range(R))
                    if v.donor:
                        k = M12.index(v.codes[v.slot])
                        den = (h2.astype(int) @ coded[(g, d)].correct[k].astype(int) - h2.astype(int) @ H.correct[k].astype(int)) / n2
                        subs[(v.menu, v.version)] = {"donor": g, "path": v.codes[v.slot], "pred": p, "vote": vote, "den": den}
                    else:
                        orig[(v.menu, g)] = {"pred": p, "vote": vote}
            # §8.12：每次切分的預測 E_J（四組各自）與 D_JV
            for r in range(R):
                row = {"host": h, "dataset": d, "split": r}
                for group in GROUP_ORDER:
                    sums = {"pair": [0.0, 0.0], "lone": [0.0, 0.0]}
                    for (menu, ver), s in subs.items():
                        if menu not in GROUP_MENUS[group]:
                            continue
                        key = "lone" if s["path"] == role[menu] else "pair"
                        sums[key][0] += s["pred"][r] - orig[(menu, s["donor"])]["pred"][r]
                        sums[key][1] += s["den"][r]
                    row[f"E_J_pred_{group}"] = 10 * (sums["pair"][0] / sums["pair"][1] - sums["lone"][0] / sums["lone"][1])
                diffs = [100 * ((s["pred"][r] - orig[(menu, s["donor"])]["pred"][r]) - (s["vote"][r] - orig[(menu, s["donor"])]["vote"][r]))
                         for (menu, ver), s in subs.items()]
                row["D_JV_pred"] = float(np.mean(diffs))
                effect_rows.append(row)
    table_rows = []
    for j in HOSTS:
        q, k, n = tables[j]
        for r in range(q.shape[0]):
            for c in range(9):
                table_rows.append({"judge": j, "split": r, "type": c // 3 + 1, "position": c % 3 + 1, "correct": int(k[r, c]),
                                   "n": int(n[r, c]), "value": float(q[r, c])})
    for (j, d), qv in lodo.items():
        for c in range(9):
            table_rows.append({"judge": j, "split": f"lodo-{d}", "type": c // 3 + 1, "position": c % 3 + 1, "value": float(qv[c])})
    for j, qv in cross.items():
        for t in range(3):
            table_rows.append({"judge": j, "split": "crossjudge", "type": t + 1, "position": "all", "value": float(qv[3 * t])})
    return {"pred": pd.DataFrame(pred_rows), "lodo": pd.DataFrame(lodo_rows), "cross": pd.DataFrame(cross_rows),
            "effects": pd.DataFrame(effect_rows), "tables": pd.DataFrame(table_rows)}
