from datetime import datetime, timezone
import hashlib

from Strategy.Aggregate import Aggregate
from Aggregator.JudgeAggregator import JudgeAggregator
from File.File import File

ITEM_FILTER = "both_answered_disagreement"


class CrossJudge(Aggregate):
    """
    RQ2 cross-judge (result/analysis/rq2/rq2_criteria.md): the Judge model `model` decides the candidates that
    another model, `generator`, produced. Prompt, candidate texts and decoding are the main grid's; only the
    judge model changes.

    Differences from Aggregate (all through its hooks):
      - the candidate arm files belong to `generator`
      - presentation orders are the ones recorded in the generator's own main-grid judge file (`referencePath`);
        they must cover every disagreement item and equal Aggregate.balancedOrders, otherwise the run stops
      - only disagreement items where both candidates have a parsed answer get a record (no agreement no-ops);
        `onlyItems` narrows that further (the pilot sample), and the full run later resumes the same file
      - every record also stores its call's provider model version, UTC time and API usage ("call")
    """
    def __init__(self, *args, generator: str, referencePath: str, onlyItems: set | None = None, **kwargs):
        # Set before Aggregate.__init__, whose checkInputs already needs them
        self.generator = generator
        self.referencePath = referencePath
        self.onlyItems = onlyItems
        self.lastCall = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------
    def checkInputs(self) -> list:
        if not isinstance(self.aggregator, JudgeAggregator) or self.aggregator.config.k != 2:
            raise ValueError("CrossJudge only runs the K=2 Judge")
        item_ids = super().checkInputs()

        self.reference = File(self.referencePath)
        meta = self.reference.metadata
        found = (meta.get("prompt_version"), meta.get("candidate_arms"), meta.get("Model", {}).get("modelType"),
                 meta.get("Dataset", {}).get("nums"), meta.get("Aggregator", {}).get("seed"))
        expected = (self.aggregator.PROMPT_VERSION, [arm.arm_id for arm in self.arms], self.generator,
                    self.dataset.config.nums, self.aggregator.config.seed)
        if found != expected:
            raise ValueError(f"Reference judge file {self.referencePath} does not match the run: {found} != {expected}")

        stored = self.store.metadata
        if stored and (stored.get("generator"), stored.get("judge")) != (self.generator, self.model.config.modelType):
            raise ValueError(f"{self.store.path} holds generator / judge {(stored.get('generator'), stored.get('judge'))}, "
                             f"this run is {(self.generator, self.model.config.modelType)}")
        return item_ids

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def armModelType(self) -> str:
        return self.generator

    def presentationOrders(self, dis_ids: list) -> dict:
        """The orders recorded in the main-grid judge file; never re-randomized."""
        missing = [item_id for item_id in dis_ids
                   if not (self.reference.records_map.get(item_id) or {}).get("presentation_order")]
        if missing:
            raise ValueError(f"{self.referencePath} has no recorded presentation order for {len(missing)} disagreement items "
                             f"(e.g. {missing[:5]}); stopping instead of re-randomizing")
        recorded = {item_id: self.reference.getRecordById(item_id)["presentation_order"] for item_id in dis_ids}
        if recorded != super().presentationOrders(dis_ids):
            raise ValueError(f"The orders recorded in {self.referencePath} differ from Aggregate.balancedOrders")
        return recorded

    def judgedIds(self, dis_ids: list) -> list:
        """Disagreement items on which both candidates have a parsed answer (the both_answered subset)."""
        return [item_id for item_id in dis_ids if all(file.getRecordById(item_id).get("parse_ok") for file in self.armFiles)]

    def selectItems(self, dis_ids: list) -> list:
        judged = self.judgedIds(dis_ids)
        if self.onlyItems is None:
            return judged
        outside = set(self.onlyItems) - set(judged)
        if outside:
            raise ValueError(f"Requested items are not both-answered disagreements: {sorted(outside)[:5]}")
        return [item_id for item_id in judged if item_id in self.onlyItems]

    def extraMetadata(self, dis_ids: list) -> dict:
        with open(self.referencePath, "rb") as f:
            reference_sha256 = hashlib.sha256(f.read()).hexdigest()
        return {
            "generator": self.generator,
            "judge": self.model.config.modelType,
            "reference_file": self.referencePath,
            "reference_sha256": reference_sha256,
            "item_filter": ITEM_FILTER,
            "n_judged_items": len(self.judgedIds(dis_ids)),
            "source": "cross_judge",
        }

    def onResponse(self, response):
        super().onResponse(response)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.lastCall = {"model_version": response.model_version, "called_at": now,
                         "usage_in": response.usage_in, "usage_out": response.usage_out}
        versions = self.store.metadata.setdefault("model_versions", {})
        versions[response.model_version] = versions.get(response.model_version, 0) + 1
        self.store.metadata.setdefault("calls_utc", {"first": now})["last"] = now

    def aggregateItem(self, item_id, presentation_order) -> dict:
        self.lastCall = None
        record = super().aggregateItem(item_id, presentation_order)
        record["call"] = self.lastCall
        return record
