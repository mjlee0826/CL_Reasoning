import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Analysis.experimentPlan import PATHS, PAIRS, ARM_TO_PATH, TIE_BREAK_KEY, Pair
from Analysis.itemExport import PATHS_FILE, AGGREGATIONS_FILE
from Analysis.splitHalf import recoveryStats, splitHalfStats

# ------------------------------------------------------------------
# RQ1 的輸入：逐題匯出（Analysis/itemExport.py）-> 每個「模型 × 資料集」區塊的逐題陣列
# 一個區塊 = 一個模型在一個資料集上的所有題目（依 item_id 排序）；同一資料集的區塊題目必須相同。
# ------------------------------------------------------------------
PAIR_BY_LABEL = {pair.label: pair for pair in PAIRS}


@dataclass
class Cell:
    """一格 = 區塊 × 配對 × 聚合器；陣列與區塊的 item_ids 對齊。"""
    pair: Pair
    aggregator: str
    correct_agg: np.ndarray
    dis: np.ndarray

    @property
    def codes(self) -> tuple[str, str]:
        return ARM_TO_PATH[self.pair.arm_a], ARM_TO_PATH[self.pair.arm_b]


@dataclass
class Block:
    model: str
    dataset: str
    item_ids: np.ndarray
    correct: dict[str, np.ndarray]    # path 代號 -> 每題是否答對
    answered: dict[str, np.ndarray]   # path 代號 -> 每題是否有解析出答案
    cells: dict[tuple[str, str], Cell]  # (配對 label, 聚合器) -> Cell

    @property
    def key(self) -> tuple[str, str]:
        return self.model, self.dataset

    @property
    def has_all_paths(self) -> bool:
        return set(self.correct) == set(PATHS)

    def bothAnswered(self, cell: Cell) -> np.ndarray:
        code_a, code_b = cell.codes
        return self.answered[code_a] & self.answered[code_b]


def loadBlocks(items_dir: str, models: list[str], aggregators: list[str]) -> dict[tuple[str, str], Block]:
    """讀逐題匯出；任何題目對不齊都直接報錯。"""
    paths = pd.read_csv(os.path.join(items_dir, PATHS_FILE))
    aggregations = pd.read_csv(os.path.join(items_dir, AGGREGATIONS_FILE))
    missing = sorted(set(models) - set(paths["model"]))
    if missing:
        raise ValueError(f"{items_dir} has no path rows for models {missing}; rerun run_analysis.py")
    paths = paths[paths["model"].isin(models)]
    aggregations = aggregations[aggregations["model"].isin(models) & aggregations["aggregator"].isin(aggregators)]

    blocks = {}
    for (model, dataset), rows in paths.groupby(["model", "dataset"], sort=True):
        correct = rows.pivot(index="item_id", columns="path", values="correct")
        answered = rows.pivot(index="item_id", columns="path", values="answered")
        if correct.isna().any().any():
            raise ValueError(f"{model} | {dataset}: the paths cover different items")
        blocks[(model, dataset)] = Block(
            model=model, dataset=dataset, item_ids=correct.index.to_numpy(),
            correct={code: correct[code].to_numpy(dtype=bool) for code in correct.columns},
            answered={code: answered[code].to_numpy(dtype=bool) for code in answered.columns},
            cells={},
        )

    for (model, dataset, label, aggregator), rows in aggregations.groupby(["model", "dataset", "pair", "aggregator"], sort=True):
        block = blocks[(model, dataset)]
        rows = rows.sort_values("item_id")
        if not np.array_equal(rows["item_id"].to_numpy(), block.item_ids):
            raise ValueError(f"{model} | {dataset} | {label} | {aggregator}: items differ from the path rows")
        block.cells[(label, aggregator)] = Cell(pair=PAIR_BY_LABEL[label], aggregator=aggregator,
                                                correct_agg=rows["correct"].to_numpy(dtype=bool),
                                                dis=rows["dis"].to_numpy(dtype=bool))

    # 同一資料集的所有模型共用 H1 / H2 切分，所以題目必須相同
    by_dataset = {}
    for block in blocks.values():
        ids = by_dataset.setdefault(block.dataset, block.item_ids)
        if not np.array_equal(ids, block.item_ids):
            raise ValueError(f"{block.model} | {block.dataset}: items differ from the other models on this dataset")
    return blocks


def _same(a, b) -> bool:
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


def crossCheckCells(blocks: dict[tuple[str, str], Block], cells_csv: str, seed: int = 0, reps: int = 200) -> list[str]:
    """
    rq1_criteria.md §6：用逐題資料重算每格的 d、c、m、recovery（全樣本）與 recovery_blind_H2（split-half），
    與 aggregation_cells.csv 比對（all 與 both_answered）；回傳不一致的描述（空 = 全部一致）。
    """
    # round_trip：pandas 預設的快速 float 解析會差 1 ulp，逐位比對需要精確讀回寫入的值
    table = pd.read_csv(cells_csv, float_precision="round_trip")
    mismatches, checked = [], 0
    for block in blocks.values():
        for (label, aggregator), cell in block.cells.items():
            code_a, code_b = cell.codes
            for subset in ("all", "both_answered"):
                mask = np.ones(len(block.item_ids), dtype=bool) if subset == "all" else block.bothAnswered(cell)
                cA, cB = block.correct[code_a][mask], block.correct[code_b][mask]
                cf, dis = cell.correct_agg[mask], cell.dis[mask]
                full = recoveryStats(cA, cB, cf, dis, np.ones(int(mask.sum()), dtype=bool))
                split = splitHalfStats(cA, cB, cf, dis, cell.pair.arm_a, cell.pair.arm_b,
                                       TIE_BREAK_KEY[cell.pair.arm_a], TIE_BREAK_KEY[cell.pair.arm_b], seed, reps)
                row = table[(table["model"] == block.model) & (table["dataset"] == block.dataset) & (table["pair"] == label)
                            & (table["aggregator"] == aggregator) & (table["subset"] == subset)]
                where = f"{block.model} | {block.dataset} | {label} | {aggregator} | {subset}"
                if len(row) != 1:
                    mismatches.append(f"{where}: {len(row)} rows in {cells_csv}")
                    continue
                row = row.iloc[0]
                expected = {"d": full["d"], "c": full["c"], "m": full["m"], "recovery": full["recovery"],
                            "recovery_blind_H2": split["recovery_blind_H2"]}
                for column, value in expected.items():
                    if not _same(float(row[column]), float(value)):
                        mismatches.append(f"{where}: {column} = {row[column]!r} in the CSV, {value!r} from the items")
                checked += 1
    if checked == 0:
        mismatches.append("no cell was checked")
    return mismatches
