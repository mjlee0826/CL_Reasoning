import json
import os
from dataclasses import dataclass

import numpy as np

from Arm.GenerationRecord import GenerationRecord
from Dataset.DatasetType import DatasetType, get_dataset_map
from Runner.paths import armPath
from Arm.ArmSpec import ArmSpec
from Analysis.experimentPlan import STEM_TO_ARM, FILE_AGGREGATORS, ARMS_TO_PAIR, Pair


def loadRecords(path: str, key: str = "item_id") -> tuple[dict, dict]:
    """ResultStore 寫出的結果檔 -> (metadata, {key: record})。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data[0], {record[key]: record for record in data[1:]}


class CellData:
    """一個 model × dataset：arm 檔（用到才載入）與磁碟上找到的聚合檔。"""
    def __init__(self, armdir: str, aggdir: str, model_name: str, dataset_name: str):
        self.armdir = armdir
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.compare = get_dataset_map()[DatasetType(dataset_name)].compareTwoAnswer
        self._arms: dict[str, dict] = {}
        self.aggregationFiles, self.unusedFiles = self.scanAggregations(os.path.join(aggdir, model_name, dataset_name))

    def armFile(self, arm_id: str) -> str:
        return armPath(self.armdir, self.model_name, self.dataset_name, ArmSpec.from_arm_id(arm_id))

    def hasArms(self, pair: Pair) -> bool:
        return all(os.path.exists(self.armFile(arm_id)) for arm_id in pair.arms)

    def arm(self, arm_id: str) -> dict:
        if arm_id not in self._arms:
            self._arms[arm_id] = loadRecords(self.armFile(arm_id))[1]
        return self._arms[arm_id]

    @staticmethod
    def scanAggregations(dirpath: str) -> tuple[dict, list]:
        """
        從檔名 {aggregator}__{stem}__{stem}.json 解析出 {(pair, aggregator_id): path}，
        另外回傳不屬於實驗計畫的檔案（會回報，不會靜默略過）。
        """
        found, unused = {}, []
        for name in sorted(os.listdir(dirpath)) if os.path.isdir(dirpath) else []:
            if not name.endswith(".json"):
                continue
            aggregator_id, *stems = name[: -len(".json")].split("__")
            arms = [STEM_TO_ARM.get(stem) for stem in stems]
            pair = ARMS_TO_PAIR.get(frozenset(arms)) if None not in arms and len(arms) == 2 else None
            if aggregator_id not in FILE_AGGREGATORS or pair is None or arms != pair.arms:
                unused.append(os.path.join(dirpath, name))
                continue
            found[(pair, aggregator_id)] = os.path.join(dirpath, name)
        return found, unused


@dataclass
class PairArrays:
    """
    一個配對的逐題陣列，依 item_id 排序。
        correct_* / answered_*   候選答案等於 gold / 候選有解析出答案
        dis                      兩個答案不同（聚合器只在這些題目動作）
        final_*, off_menu        聚合器的最終答案（Blind 沒有聚合檔，為 None，之後由 metrics 計算）
    """
    item_ids: np.ndarray
    correct_a: np.ndarray
    correct_b: np.ndarray
    answered_a: np.ndarray
    answered_b: np.ndarray
    dis: np.ndarray
    tokens_out_a: np.ndarray
    tokens_out_b: np.ndarray
    final_correct: np.ndarray | None = None
    final_answered: np.ndarray | None = None
    off_menu: np.ndarray | None = None
    tokens_out_agg: np.ndarray | None = None

    def subset(self, mask: np.ndarray) -> "PairArrays":
        return PairArrays(**{name: None if value is None else value[mask] for name, value in vars(self).items()})


def alignPair(cell: CellData, pair: Pair, aggregation_path: str | None = None) -> PairArrays:
    """依 item_id 對齊配對的兩個 arm 檔（與聚合檔）；任何不一致都直接報錯。"""
    where = f"{cell.model_name} | {cell.dataset_name} | {pair.label}"
    recordsA, recordsB = cell.arm(pair.arm_a), cell.arm(pair.arm_b)
    if recordsA.keys() != recordsB.keys():
        raise ValueError(f"{where}: the arm files cover different items")
    ids = sorted(recordsA)
    A, B = [recordsA[i] for i in ids], [recordsB[i] for i in ids]
    if any(a["gold"] != b["gold"] for a, b in zip(A, B)):
        raise ValueError(f"{where}: the arm files disagree on gold answers")

    compare = cell.compare
    arrays = PairArrays(
        item_ids=np.array(ids),
        correct_a=np.array([compare(a["gold"], a["parsed_answer"]) for a in A], dtype=bool),
        correct_b=np.array([compare(b["gold"], b["parsed_answer"]) for b in B], dtype=bool),
        answered_a=np.array([a["parse_ok"] for a in A], dtype=bool),
        answered_b=np.array([b["parse_ok"] for b in B], dtype=bool),
        dis=np.array([not compare(a["parsed_answer"], b["parsed_answer"]) for a, b in zip(A, B)], dtype=bool),
        tokens_out_a=np.array([a["tokens_out"] for a in A], dtype=float),
        tokens_out_b=np.array([b["tokens_out"] for b in B], dtype=float),
    )
    if aggregation_path is None:
        return arrays

    metadata, records = loadRecords(aggregation_path)
    where = f"{where} | {aggregation_path}"
    if metadata.get("candidate_arms") != pair.arms:
        raise ValueError(f"{where}: candidate_arms {metadata.get('candidate_arms')} != {pair.arms}")
    if records.keys() != recordsA.keys():
        raise ValueError(f"{where}: the aggregation file covers different items than the arm files")
    R = [records[i] for i in ids]
    finals = [r["final_answer"] for r in R]
    # 一致題上聚合器是 no-op（paper_status §3.2）：最終答案必須等於兩者共同的答案
    if any(not d and not compare(final, a["parsed_answer"]) for d, final, a in zip(arrays.dis, finals, A)):
        raise ValueError(f"{where}: a final answer differs from the candidates' shared answer on an agreement item")
    arrays.final_correct = np.array([compare(a["gold"], final) for a, final in zip(A, finals)], dtype=bool)
    arrays.final_answered = np.array([GenerationRecord.isParseOk(final) for final in finals], dtype=bool)
    arrays.off_menu = np.array([r["off_menu"] for r in R], dtype=bool)
    arrays.tokens_out_agg = np.array([r["tokens_out"] for r in R], dtype=float)
    return arrays
