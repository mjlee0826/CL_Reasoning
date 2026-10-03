import os

import pandas as pd

from Arm.GenerationRecord import GenerationRecord
from Analysis.alignment import CellData, loadRecords
from Analysis.experimentPlan import PATHS, Pair

# ------------------------------------------------------------------
# 逐題匯出（paper_status 階段 1）：aggregation_cells.csv 只有彙總，RQ1 的 probe 需要逐題資料
#   paths.csv.gz         model × dataset × path × 題目：答案、有無答案、對錯、輸出 token
#   aggregations.csv.gz  model × dataset × 配對 × 聚合器 × 題目：最終答案、對錯、是否分歧 ...
# 對錯的定義與 Analysis/alignment.py 相同（dataset 的 compareTwoAnswer）。
# Blind 不匯出：它選哪一方取決於 split（H1 選、H2 量），逐題可由 paths 直接算。
# ------------------------------------------------------------------
PATH_COLUMNS = ["model", "dataset", "item_id", "path", "arm_id", "gold", "answer", "answered", "correct", "tokens_out"]
AGGREGATION_COLUMNS = ["model", "dataset", "pair", "arm_a", "arm_b", "aggregator", "item_id", "dis", "final_answer",
                       "answered", "correct", "off_menu", "n_rounds", "first_shown", "tokens_out"]
PATHS_FILE, AGGREGATIONS_FILE = "paths.csv.gz", "aggregations.csv.gz"


def pathRows(cell: CellData) -> list[dict]:
    """每條有 arm 檔的 path 的逐題列。"""
    rows = []
    for path, arm_id in PATHS.items():
        if not os.path.exists(cell.armFile(arm_id)):
            continue
        for item_id, record in sorted(cell.arm(arm_id).items()):
            rows.append({
                "model": cell.model_name, "dataset": cell.dataset_name, "item_id": item_id, "path": path, "arm_id": arm_id,
                "gold": record["gold"], "answer": record["parsed_answer"], "answered": record["parse_ok"],
                "correct": cell.compare(record["gold"], record["parsed_answer"]), "tokens_out": record["tokens_out"],
            })
    return rows


def aggregationRows(cell: CellData, pair: Pair, aggregator_id: str, aggregation_path: str) -> list[dict]:
    """一個聚合檔的逐題列；先用 alignPair 驗證過這個檔案再呼叫。"""
    _, records = loadRecords(aggregation_path)
    recordsA, recordsB = cell.arm(pair.arm_a), cell.arm(pair.arm_b)
    rows = []
    for item_id in sorted(records):
        record, a, b = records[item_id], recordsA[item_id], recordsB[item_id]
        order = record.get("presentation_order")
        rows.append({
            "model": cell.model_name, "dataset": cell.dataset_name, "pair": pair.label, "arm_a": pair.arm_a,
            "arm_b": pair.arm_b, "aggregator": aggregator_id, "item_id": item_id,
            "dis": not cell.compare(a["parsed_answer"], b["parsed_answer"]),
            "final_answer": record["final_answer"], "answered": GenerationRecord.isParseOk(record["final_answer"]),
            "correct": cell.compare(a["gold"], record["final_answer"]), "off_menu": record["off_menu"],
            "n_rounds": record.get("n_rounds"), "first_shown": order[0] if order else None,
            "tokens_out": record["tokens_out"],
        })
    return rows


def writeItems(items_dir: str, path_rows: list[dict], aggregation_rows: list[dict]) -> tuple[str, str]:
    """gzip 的 mtime 固定為 0，同樣的資料重跑會得到同樣的檔案。"""
    os.makedirs(items_dir, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0}
    paths_file, aggregations_file = os.path.join(items_dir, PATHS_FILE), os.path.join(items_dir, AGGREGATIONS_FILE)
    pd.DataFrame(path_rows, columns=PATH_COLUMNS).to_csv(paths_file, index=False, compression=compression)
    pd.DataFrame(aggregation_rows, columns=AGGREGATION_COLUMNS).to_csv(aggregations_file, index=False, compression=compression)
    return paths_file, aggregations_file
