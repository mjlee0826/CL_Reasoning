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
        self.armFileByArmId = {arm.arm_id: file for arm, file in zip(self.arms, self.armFiles)}
        self.unverifiedArms = []

        for arm, file, arm_questions, model_type in zip(self.arms, self.armFiles, self.armQuestions, self.armModelTypes()):
            meta = file.metadata
            found = (meta.get("Arm", {}).get("arm_id"), meta.get("Model", {}).get("modelType"),
                     meta.get("Dataset", {}).get("datasetType"), meta.get("Dataset", {}).get("nums"))
            expected = (arm.arm_id, model_type, dataset_config.datasetType, dataset_config.nums)
            if found != expected:
                raise ValueError(f"{file.file_path} does not match the run: {found} != {expected}")

            missing = item_ids - set(file.records_map)
            if missing:
                raise ValueError(f"{file.file_path} is incomplete ({len(missing)} items missing); finish run_generate.py first")

            # The prompt rebuilt from this arm's question text must be the prompt that was sent.
            # A derived arm (F:{lang}) also needs its base arm's output, so it can only be checked when
            # the base arm is one of the candidates; otherwise the arm is listed as unverified.
            baseFile = self.armFileByArmId.get(arm.base_arm_id) if arm.is_derived else None
            if arm.is_derived and baseFile is None:
                self.unverifiedArms.append(arm.arm_id)
                continue

            builder = PromptBuilder(arm)
            for item_id in sorted(item_ids):
                record = file.getRecordById(item_id)
                base_raw_text = baseFile.getRecordById(item_id).get("raw_text") if baseFile else None
                if PromptBuilder.promptHash(builder.messages(arm_questions[item_id], base_raw_text)) != record.get("prompt_hash"):
                    raise ValueError(f"prompt_hash mismatch in {file.file_path}, item {item_id}")

        for item_id in item_ids:
            golds = {str(file.getRecordById(item_id).get("gold")) for file in self.armFiles}
            if len(golds) != 1:
                raise ValueError(f"Candidate files disagree on gold for item {item_id}: {golds}")

        meta = self.store.metadata
        if meta:
            if meta.get("prompt_version") != self.aggregator.PROMPT_VERSION:
                raise ValueError(f"{self.store.path} was written with prompt version {meta.get('prompt_version')!r}, this run uses "
                                 f"{self.aggregator.PROMPT_VERSION!r}; move the old file to result/archive/ before rerunning")
            found = (meta.get("Aggregator", {}).get("aggregatorType"), meta.get("candidate_arms"),
                     meta.get("Model", {}).get("modelType"), meta.get("Dataset", {}).get("nums"),
                     meta.get("Aggregator", {}).get("seed"))
            expected = (self.aggregator.config.aggregatorType, [arm.arm_id for arm in self.arms],
                        self.model.config.modelType, dataset_config.nums, self.aggregator.config.seed)
            if found != expected:
                raise ValueError(f"{self.store.path} holds a different run: {found} != {expected}")

        return sorted(item_ids)

    # ------------------------------------------------------------------
    # Hooks (CrossJudge overrides them; the defaults are the main-grid behaviour)
    # ------------------------------------------------------------------
    def armModelType(self) -> str:
        """modelType the candidate arm files must belong to: the aggregating model itself."""
        return self.model.config.modelType

    def armModelTypes(self) -> list[str]:
        """modelType of each candidate arm file, aligned with `arms` (RQ3-GJ swaps one arm for another model's)."""
        return [self.armModelType()] * len(self.arms)

    def presentationOrders(self, dis_ids: list) -> dict:
        """{item_id: arm_ids in display order} for the disagreement items (only for aggregators that need one)."""
        return self.balancedOrders([arm.arm_id for arm in self.arms], dis_ids, self.aggregator.config.seed)

    def selectItems(self, dis_ids: list) -> list:
        """item_ids that get a record: every item (agreement items are no-ops)."""
        return self.itemIds

    def extraMetadata(self, dis_ids: list) -> dict:
        """Fields appended to the metadata of a new result file."""
        return {}

    def onResponse(self, response):
        """Called with every aggregator LLMResponse."""
        self.store.addUsage(response)

    def aggregateItem(self, item_id, presentation_order) -> dict:
        """The record written for one item; raises AggregatorCallError when the aggregator call failed."""
        return self.aggregator.aggregate(self.buildItem(item_id, presentation_order)).to_dict()

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
        orders = self.presentationOrders(dis_ids) if self.aggregator.needsPresentationOrder() else {}
        targets = self.selectItems(dis_ids)

        if not self.store.metadata:
            self.store.metadata = {
                "Model": self.model.config.to_dict(),
                "Dataset": self.dataset.config.to_dict(),
                "Strategy": self.config.to_dict(),
                "Aggregator": self.aggregator.config.to_dict(),
                "prompt_version": self.aggregator.PROMPT_VERSION,
                "candidate_arms": [arm.arm_id for arm in self.arms],
                "candidate_files": [file.file_path for file in self.armFiles],
                "n_items": len(self.itemIds),
                "n_disagreement": len(dis_ids),
                "prompt_hash_unverified_arms": self.unverifiedArms,
                "schema_version": SCHEMA_VERSION,
                "source": "aggregate",
                **self.extraMetadata(dis_ids),
            }
        self.aggregator.onResponse = self.onResponse

        todo = [item_id for item_id in targets if not self.store.has(item_id)]
        self.log.logMessage(f'{self.config.displayName}: {len(todo)} items to aggregate ({len(dis_ids)} disagreements in total)')

        failed = []
        for item_id in tqdm(todo, desc=f"Aggregating {self.config.displayName}"):
            try:
                record = self.aggregateItem(item_id, orders.get(item_id))
            except AggregatorCallError as e:
                failed.append(item_id)
                self.log.logMessage(f'API error on item {item_id}: {e}')
                continue

            self.store.add(record)
            self.log.logMessage(f'Item {item_id}: final {record["final_answer"]} | off_menu {record["off_menu"]} | rounds {record["n_rounds"]}')

        self.store.save()
        return failed
