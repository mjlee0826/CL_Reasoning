from Model.Model import Model
from Dataset.Dataset import Dataset
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Arm.ArmSpec import ArmSpec
from Arm.PromptBuilder import PromptBuilder
from Aggregator.Aggregator import Aggregator, AggregationItem, Candidate, AggregatorCallError
from File.File import File
from File.ResultStore import ResultStore
from Log.Log import Log

import numpy as np
from tqdm import tqdm

SCHEMA_VERSION = "aggregation/v1"

class Aggregate(Strategy):
    """
    Runs one aggregator over one candidate set (list of arms) of one model / dataset and writes
    AggregationRecords into a ResultStore (result/aggregations/{model}/{dataset}/{aggregator}__{arm}__....json).

    The candidate arm files must be complete and consistent (same item_ids, gold and nums). Only then are the
    disagreement set and the balanced presentation orders identical on every resume.
    """
    def __init__(self, config: StrategyConfig, model: Model, dataset: Dataset, log: Log, aggregator: Aggregator,
                 arms: list[ArmSpec], armFiles: list[File], armDatasets: list[Dataset], store: ResultStore):
        """
        dataset      original English dataset (the judge's question text)
        armFiles     generation result file of each arm, aligned with `arms`
        armDatasets  dataset built with ArmSpec.to_dataset_config for each arm (that arm's question text)
        """
        super().__init__(config)
        self.model: Model = model
        self.dataset: Dataset = dataset
        self.log: Log = log
        self.aggregator: Aggregator = aggregator
        self.arms: list[ArmSpec] = arms
        self.armFiles: list[File] = armFiles
        self.store: ResultStore = store

        if self.aggregator.parseAnswer is None:
            self.aggregator.parseAnswer = self.parseAnswer

        self.questions: dict = {data["id"]: data["question"] for data in dataset.getData()}
        self.armQuestions: list[dict] = [{data["id"]: data["question"] for data in d.getData()} for d in armDatasets]

        self.config.displayName = f"{aggregator.config.displayName} ({' vs '.join(arm.arm_id for arm in arms)})"
        self.itemIds: list = self.checkInputs()

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------
    def checkInputs(self) -> list:
        """Validates the candidate files and returns the sorted item_ids to aggregate."""
        self.aggregator.validateCandidates(self.arms)
        dataset_config = self.dataset.config
        item_ids = set(self.questions)

        for arm, file, arm_questions in zip(self.arms, self.armFiles, self.armQuestions):
            meta = file.metadata
            found = (meta.get("Arm", {}).get("arm_id"), meta.get("Model", {}).get("modelType"),
                     meta.get("Dataset", {}).get("datasetType"), meta.get("Dataset", {}).get("nums"))
            expected = (arm.arm_id, self.model.config.modelType, dataset_config.datasetType, dataset_config.nums)
            if found != expected:
                raise ValueError(f"{file.file_path} does not match the run: {found} != {expected}")

            missing = item_ids - set(file.records_map)
            if missing:
                raise ValueError(f"{file.file_path} is incomplete ({len(missing)} items missing); finish run_generate.py first")

            # The prompt rebuilt from this arm's question text must be the prompt that was sent
            builder = PromptBuilder(arm)
            for item_id in sorted(item_ids):
                record = file.getRecordById(item_id)
                if PromptBuilder.promptHash(builder.messages(arm_questions[item_id])) != record.get("prompt_hash"):
                    raise ValueError(f"prompt_hash mismatch in {file.file_path}, item {item_id}")

        for item_id in item_ids:
            golds = {str(file.getRecordById(item_id).get("gold")) for file in self.armFiles}
            if len(golds) != 1:
                raise ValueError(f"Candidate files disagree on gold for item {item_id}: {golds}")

        meta = self.store.metadata
        if meta:
            found = (meta.get("Aggregator", {}).get("aggregatorType"), meta.get("candidate_arms"),
                     meta.get("Model", {}).get("modelType"), meta.get("Dataset", {}).get("nums"),
                     meta.get("Aggregator", {}).get("seed"))
            expected = (self.aggregator.config.aggregatorType, [arm.arm_id for arm in self.arms],
                        self.model.config.modelType, dataset_config.nums, self.aggregator.config.seed)
            if found != expected:
                raise ValueError(f"{self.store.path} holds a different run: {found} != {expected}")

        return sorted(item_ids)

    # ------------------------------------------------------------------
    # Presentation order
    # ------------------------------------------------------------------
    @staticmethod
    def balancedOrders(arm_ids: list[str], dis_ids: list, seed: int) -> dict:
        """
        Judge position balancing (Wang et al., ACL 2024): the disagreement items are shuffled with
        default_rng(seed) and the i-th item shows the candidates rotated by i mod K.
        K=2 -> the first arm (anchor) is shown first on half of the items;
        K=5 -> every arm appears in every position equally often (±1).
        """
        k = len(arm_ids)
        permutation = np.random.default_rng(seed).permutation(len(dis_ids))
        orders = {}
        for position, index in enumerate(permutation):
            rotation = position % k
            orders[dis_ids[index]] = arm_ids[rotation:] + arm_ids[:rotation]
        return orders

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def buildItem(self, item_id, presentation_order) -> AggregationItem:
        candidates = [
            Candidate(arm=arm, record=file.getRecordById(item_id), question=arm_questions[item_id])
            for arm, file, arm_questions in zip(self.arms, self.armFiles, self.armQuestions)
        ]
        return AggregationItem(item_id=item_id, candidates=candidates, question=self.questions[item_id],
                               presentation_order=presentation_order)

    def getRes(self) -> list:
        """
        Aggregates every missing item. Returns the item_ids whose aggregator call failed (still missing).
        """
        self.log.logInfo(self, self.model, self.dataset)

        dis_ids = [
            item_id for item_id in self.itemIds
            if not self.aggregator.isUnanimous([file.getRecordById(item_id).get("parsed_answer", "") for file in self.armFiles])
        ]
        orders = {}
        if self.aggregator.needsPresentationOrder():
            orders = self.balancedOrders([arm.arm_id for arm in self.arms], dis_ids, self.aggregator.config.seed)

        if not self.store.metadata:
            self.store.metadata = {
                "Model": self.model.config.to_dict(),
                "Dataset": self.dataset.config.to_dict(),
                "Strategy": self.config.to_dict(),
                "Aggregator": self.aggregator.config.to_dict(),
                "candidate_arms": [arm.arm_id for arm in self.arms],
                "candidate_files": [file.file_path for file in self.armFiles],
                "n_items": len(self.itemIds),
                "n_disagreement": len(dis_ids),
                "schema_version": SCHEMA_VERSION,
                "source": "aggregate",
            }
        self.aggregator.onResponse = self.store.addUsage

        todo = [item_id for item_id in self.itemIds if not self.store.has(item_id)]
        self.log.logMessage(f'{self.config.displayName}: {len(todo)} items to aggregate ({len(dis_ids)} disagreements in total)')

        failed = []
        for item_id in tqdm(todo, desc=f"Aggregating {self.config.displayName}"):
            try:
                record = self.aggregator.aggregate(self.buildItem(item_id, orders.get(item_id)))
            except AggregatorCallError as e:
                failed.append(item_id)
                self.log.logMessage(f'API error on item {item_id}: {e}')
                continue

            self.store.add(record.to_dict())
            self.log.logMessage(f'Item {item_id}: final {record.final_answer} | off_menu {record.off_menu} | rounds {record.n_rounds}')

        self.store.save()
        return failed

    @staticmethod
    def getTokenLens(model: Model, data):
        """Aggregator output tokens are already stored on the record (generation cost lives in the arm files)."""
        return data.get("tokens_out") or 0
