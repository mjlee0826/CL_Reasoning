from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from Analysis.experimentPlan import PATHS
from Analysis.itemMatrix import Block, Cell
from Analysis.splitHalf import makeSplits

# ------------------------------------------------------------------
# RQ1：只用 H1 的資訊決定「聚合」或「用單一 path」，在 H2 上評估。
# 規格與判定標準：result/analysis/rq1/rq1_criteria.md（§2 切分與 probe、§3 方法、§4 統計、§5 判定）。
# 所有正確率以「題數」比較（平手判定不受浮點誤差影響），輸出時才換成比例。
# ------------------------------------------------------------------
KS = (25, 50, 100, 200, 500)
BASELINES = ("pair", "global")

PROBE_RANDOM, PROBE_DIS = "probe_random", "probe_dis"
TRANSFER_MODEL, TRANSFER_DATASET = "transfer_model", "transfer_dataset"
TRANSFER_SOURCE, TRANSFER_SOURCE_SAME_AXIS = "transfer_source", "transfer_source_same_axis"
PROBES = (PROBE_RANDOM, PROBE_DIS)
TRANSFERS = (TRANSFER_MODEL, TRANSFER_DATASET, TRANSFER_SOURCE)   # §5 判定用的三種遷移（同軸只當參考）
NO_K = 0   # 遷移列的 k（遷移用來源格的整個 H1，不涉及 k）

# 每次重複、每個（設定, 目標格, 來源）記下的量；acc_* 是 H2 上的正確率
METRICS = [
    "acc_D",            # 我們的決策
    "acc_A",            # 永遠聚合
    "acc_S",            # 永遠用單一 path（與 D 同一條）
    "acc_O",            # Oracle：H2 上聚合與候選 path 中最高者
    "acc_D_tieagg",     # 敏感度 (a)：Excess = 0 判為聚合
    "acc_D_truepath",   # 敏感度 (b)：不聚合時改用 H2 上真正最強的 path
    "acc_S_truepath",   # 敏感度 (b) 的「永遠用單一 path」
    "aggregate",        # D 判為聚合
    "tie",              # 來源上 Excess 剛好為 0
    "short",            # 只抽分歧題的 probe 不足 k 題（用了全部）
]


def pathKey(code: str) -> tuple:
    """平手時的固定順序：L:en（EN）優先，其餘依 path 代號字母序。"""
    return code != "EN", code


def strongest(counts: dict[str, int]) -> str:
    """答對題數最多的 path；平手依 pathKey。"""
    return min(counts, key=lambda code: (-counts[code], pathKey(code)))


@dataclass(frozen=True)
class Choice:
    """來源資訊給出的決策。"""
    aggregate: bool          # 主規則：Excess > 0
    aggregate_if_tie: bool   # 敏感度 (a)：Excess >= 0
    tie: bool                # Excess == 0
    path: str                # 不聚合時用的 path


def choose(agg_correct: int, path_correct: dict[str, int]) -> Choice:
    """Excess = acc_聚合 − acc_s（s = 較強的 path），都在同一批題目上，所以比較題數即可。"""
    path = strongest(path_correct)
    excess = int(agg_correct) - int(path_correct[path])
    return Choice(aggregate=excess > 0, aggregate_if_tie=excess >= 0, tie=excess == 0, path=path)


def score(choice: Choice, agg_correct: int, path_correct: dict[str, int], n: int, short: bool = False) -> np.ndarray:
    """choice 在 H2（n 題，各選項答對題數）上的 METRICS。"""
    true_path = strongest(path_correct)
    agg, single, true_single = agg_correct / n, path_correct[choice.path] / n, path_correct[true_path] / n
    return np.array([
        agg if choice.aggregate else single,
        agg,
        single,
        max(agg_correct, max(path_correct.values())) / n,
        agg if choice.aggregate_if_tie else single,
        agg if choice.aggregate else true_single,
        true_single,
        float(choice.aggregate),
        float(choice.tie),
        float(short),
    ])


def baselineCodes(block: Block, cell: Cell, baseline: str) -> list[str] | None:
    """配對內基準 = 配對的兩條 path；全域基準 = 14 條 path（區塊缺 path 時不做，回傳 None）。"""
    if baseline == "pair":
        return list(cell.codes)
    return sorted(PATHS) if block.has_all_paths else None


def counts(block: Block, cell: Cell, codes: list[str], index: np.ndarray) -> tuple[int, dict[str, int]]:
    return int(cell.correct_agg[index].sum()), {code: int(block.correct[code][index].sum()) for code in codes}


@dataclass
class CellRep:
    """一格在一次重複中的狀態（每個基準一份）。"""
    h2: tuple[int, dict[str, int], int]   # H2 ∩ both_answered 上：聚合答對數、各 path 答對數、題數
    h1_choice: Choice                     # 用整個 H1 ∩ both_answered 給出的決策（遷移的來源）


class Accumulator:
    """(設定, 基準, 聚合器, k, 模型, 資料集, 配對, 來源, 含 F) -> METRICS 的總和與次數；另記跳過的次數。"""
    def __init__(self):
        self.sums, self.counts, self.skipped = {}, {}, {}

    def add(self, key: tuple, values: np.ndarray):
        if key in self.sums:
            self.sums[key] += values
            self.counts[key] += 1
        else:
            self.sums[key] = values.copy()
            self.counts[key] = 1

    def skip(self, key: tuple):
        self.skipped[key] = self.skipped.get(key, 0) + 1

    KEY_COLUMNS = ["setting", "baseline", "aggregator", "k", "model", "dataset", "pair", "source", "involves_F"]

    def table(self) -> pd.DataFrame:
        rows = [dict(zip(self.KEY_COLUMNS, key), reps=self.counts[key], skipped_reps=self.skipped.get(key, 0),
                     **dict(zip(METRICS, self.sums[key] / self.counts[key]))) for key in self.sums]
        rows += [dict(zip(self.KEY_COLUMNS, key), reps=0, skipped_reps=n) for key, n in self.skipped.items() if key not in self.sums]
        return pd.DataFrame(rows, columns=self.KEY_COLUMNS + ["reps", "skipped_reps"] + METRICS)


def runProbes(blocks: dict[tuple[str, str], Block], ks=KS, reps: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    每次重複（§2）：每個資料集用 makeSplits(n, reps, seed) 的第 r 組 H1 / H2，H1 以 default_rng([seed, r]) 排序。
    回傳每個（設定, 基準, 聚合器, k, 目標格, 來源）對所有重複的平均（Accumulator.table）。
    """
    datasets = sorted({block.dataset for block in blocks.values()})
    n_items = {ds: len(next(b for b in blocks.values() if b.dataset == ds).item_ids) for ds in datasets}
    splits = {ds: makeSplits(n_items[ds], reps, seed) for ds in datasets}
    acc = Accumulator()

    for rep in range(reps):
        h1 = {ds: splits[ds][rep] for ds in datasets}
        orders = {ds: np.random.default_rng([seed, rep]).permutation(np.flatnonzero(h1[ds])) for ds in datasets}
        states: dict[tuple, dict[str, CellRep]] = {}

        # 同格 probe；同時記下每格的 H2 計數與整個 H1 的決策，給遷移用
        for block in blocks.values():
            order_all = orders[block.dataset]
            for (label, aggregator), cell in block.cells.items():
                both = block.bothAnswered(cell)
                order = order_all[both[order_all]]                       # H1 ∩ both_answered，依本次的隨機順序
                h2 = np.flatnonzero(~h1[block.dataset] & both)
                states[(block.key, label, aggregator)] = per_baseline = {}
                for baseline in BASELINES:
                    codes = baselineCodes(block, cell, baseline)
                    if codes is None:
                        continue
                    agg_h2, paths_h2 = counts(block, cell, codes, h2)
                    per_baseline[baseline] = CellRep(h2=(agg_h2, paths_h2, len(h2)),
                                                     h1_choice=choose(*counts(block, cell, codes, order)))
                    base_key = (baseline, aggregator)
                    target = (block.model, block.dataset, label)

                    # 隨機 probe：H1 ∩ both_answered 的前 k 題；不足 k 題就跳過
                    cum_agg = np.cumsum(cell.correct_agg[order])
                    cum_paths = {code: np.cumsum(block.correct[code][order]) for code in codes}
                    for k in ks:
                        key = (PROBE_RANDOM, *base_key, k, *target, "", cell.pair.axis == "F")
                        if len(order) < k:
                            acc.skip(key)
                            continue
                        choice = choose(cum_agg[k - 1], {code: c[k - 1] for code, c in cum_paths.items()})
                        acc.add(key, score(choice, agg_h2, paths_h2, len(h2)))

                    # 只抽分歧題的 probe（只做配對內基準）：同一個順序中的前 k 題分歧題，不足就用全部
                    if baseline == "pair":
                        dis_order = order[cell.dis[order]]
                        for k in ks:
                            probe = dis_order[:k]
                            choice = choose(*counts(block, cell, codes, probe))
                            acc.add((PROBE_DIS, *base_key, k, *target, "", cell.pair.axis == "F"),
                                    score(choice, agg_h2, paths_h2, len(h2), short=len(probe) < k))

        # 遷移：決策完全來自來源格的整個 H1，評估在目標格的 H2
        for block in blocks.values():
            for (label, aggregator), cell in block.cells.items():
                target_states = states[(block.key, label, aggregator)]
                for baseline, target in target_states.items():
                    agg_h2, paths_h2, n_h2 = target.h2

                    def add(setting, source_name, choice, involves_f):
                        acc.add((setting, baseline, aggregator, NO_K, block.model, block.dataset, label, source_name, involves_f),
                                score(choice, agg_h2, paths_h2, n_h2))

                    for (model, dataset), other in blocks.items():
                        source = states.get(((model, dataset), label, aggregator), {}).get(baseline)
                        if source is None or (model, dataset) == block.key:
                            continue
                        # 換模型 / 換資料集：配對相同，沿用來源較強 path 的同名 path
                        if dataset == block.dataset:
                            add(TRANSFER_MODEL, model, source.h1_choice, cell.pair.axis == "F")
                        elif model == block.model:
                            add(TRANSFER_DATASET, dataset, source.h1_choice, cell.pair.axis == "F")

                    # 換多樣性來源：同模型、同資料集、同聚合器的其他配對；path 用目標配對自己在 H1 上較強者
                    for (other_label, other_aggregator), other_cell in block.cells.items():
                        if other_aggregator != aggregator or other_label == label:
                            continue
                        source = states[(block.key, other_label, aggregator)].get(baseline)
                        if source is None:
                            continue
                        choice = Choice(source.h1_choice.aggregate, source.h1_choice.aggregate_if_tie,
                                        source.h1_choice.tie, target.h1_choice.path)
                        setting = TRANSFER_SOURCE_SAME_AXIS if other_cell.pair.axis == cell.pair.axis else TRANSFER_SOURCE
                        add(setting, other_label, choice, "F" in (cell.pair.axis, other_cell.pair.axis))

    return acc.table()


# ------------------------------------------------------------------
# §4 區塊與統計
# ------------------------------------------------------------------
BLOCK_KEYS = ["setting", "baseline", "aggregator", "k", "model", "dataset"]
# 區塊表的數值欄（百分點）
DERIVED = {
    "diff_D_A": ("acc_D", "acc_A"), "diff_D_S": ("acc_D", "acc_S"),
    "regret_D": ("acc_O", "acc_D"), "regret_A": ("acc_O", "acc_A"), "regret_S": ("acc_O", "acc_S"),
    "diff_D_A_tieagg": ("acc_D_tieagg", "acc_A"), "diff_D_S_tieagg": ("acc_D_tieagg", "acc_S"),
    "diff_D_A_truepath": ("acc_D_truepath", "acc_A"), "diff_D_S_truepath": ("acc_D_truepath", "acc_S_truepath"),
    "regret_S_truepath": ("acc_O", "acc_S_truepath"),
}
RATES = {"tie_rate": "tie", "aggregate_rate": "aggregate", "short_rate": "short"}


def blockTable(cells: pd.DataFrame) -> pd.DataFrame:
    """先在每個目標格內對來源平均，再在區塊內對目標格平均；數值換成百分點。"""
    cells = cells[cells["reps"] > 0]
    per_target = cells.groupby(BLOCK_KEYS + ["pair"], sort=True)[METRICS].mean()
    grouped = per_target.groupby(BLOCK_KEYS, sort=True)
    block = grouped.mean()
    block["n_cells"] = grouped.size()
    for name, (a, b) in DERIVED.items():
        block[name] = 100 * (block[a] - block[b])
    for name, column in RATES.items():
        block[name] = 100 * block[column]
    return block.reset_index()


def tInterval(values: np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    """跨區塊的平均、SE、t 區間（自由度 = 區塊數 − 1）。"""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan")
    se = float(values.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * se)
    return mean, se, mean - half, mean + half


def summaryTable(blocks: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """每個（設定, 基準, 聚合器, k）× 數值欄：跨區塊平均、SE、95% 區間、區塊數。"""
    rows = []
    for key, group in blocks.groupby(BLOCK_KEYS[:4], sort=True):
        for metric in list(DERIVED) + list(RATES):
            mean, se, low, high = tInterval(group[metric].to_numpy(), level)
            rows.append(dict(zip(BLOCK_KEYS[:4], key), metric=metric, mean=mean, se=se, ci_low=low, ci_high=high,
                             n_blocks=len(group)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# §5 判定
# ------------------------------------------------------------------
CRITERIA = {"k": 200, "no_room_pp": 0.2, "margin_share": 0.5, "level": 0.95}


def criteriaTable(blocks: pd.DataFrame, aggregator: str = "judge", baseline: str = "pair", criteria=CRITERIA) -> pd.DataFrame:
    """每個判定設定：空間、較好的笨方法、差距的跨區塊平均與區間、是否達標。"""
    rows = []
    for setting, k in [(PROBE_RANDOM, criteria["k"])] + [(t, NO_K) for t in TRANSFERS]:
        sub = blocks[(blocks["setting"] == setting) & (blocks["k"] == k)
                     & (blocks["aggregator"] == aggregator) & (blocks["baseline"] == baseline)]
        if sub.empty:
            rows.append(dict(setting=setting, k=k, n_blocks=0))
            continue
        regret_A, regret_S = sub["regret_A"].mean(), sub["regret_S"].mean()
        better = "A" if regret_A <= regret_S else "S"
        space = min(regret_A, regret_S)
        margin = sub["diff_D_A"] if better == "A" else sub["diff_D_S"]
        mean, se, low, high = tInterval(margin.to_numpy(), criteria["level"])
        rows.append(dict(setting=setting, k=k, n_blocks=len(sub), regret_A=regret_A, regret_S=regret_S, space=space,
                         better=better, margin=mean, margin_se=se, margin_ci_low=low, margin_ci_high=high,
                         required=criteria["margin_share"] * space,
                         met=bool(mean >= criteria["margin_share"] * space and low > 0)))
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame, criteria=CRITERIA) -> str:
    probe = table[table["setting"] == PROBE_RANDOM].iloc[0]
    if probe["n_blocks"] == 0:
        return "無法判定：沒有同格 probe 的結果"
    if probe["space"] < criteria["no_room_pp"]:
        return f"沒有空間：同格 k={criteria['k']} 的空間 {probe['space']:.3f}pp < {criteria['no_room_pp']}pp，RQ1 只當附錄"
    transfers = table[table["setting"].isin(TRANSFERS)]
    met_transfers = transfers[transfers["met"] == True]["setting"].tolist()
    if bool(probe["met"]) and met_transfers:
        return f"成功：同格 k={criteria['k']} 達標，且遷移達標：{', '.join(met_transfers)}"
    reasons = []
    if not bool(probe["met"]):
        reasons.append(f"同格 k={criteria['k']} 未達標")
    if not met_transfers:
        reasons.append("沒有任何遷移設定達標")
    return "不成功：" + "；".join(reasons)
