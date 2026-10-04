import hashlib

import numpy as np

from Strategy.Aggregate import Aggregate
from Strategy.CallRecording import CallRecording
from Aggregator.JudgeAggregator import JudgeAggregator
from Aggregator.MenuJudgeAggregator import MenuJudgeAggregator
from File.File import File

ITEM_FILTER = "menu_all_answered_disagreement"
ORDER_GROUPED, ORDER_RECORDED = "grouped_rotation_seed0", "recorded_main_grid"


class MenuJudge(CallRecording, Aggregate):
    """
    RQ1-KJ (result/analysis/rq1kj/rq1kj_criteria.md): the model judges its own K candidates of one menu.

    Differences from Aggregate (all through its hooks):
      - only items where every path of the menu has a parsed answer and the answers are not all equal get a record
        (§2.4; on agreement items the analysis keeps the common answer, no call is made)
      - presentation orders = groupedOrders over those items (§3.3). With `referencePath` (the §6.3 flow check) they
        are the orders recorded in that main-grid judge file instead, and the run stops when one is missing
      - `onlyItems` narrows the items further (the §6.4 pilot sample); a later run resumes the same file
      - every record stores its call ("call", CallRecording) and the prompt sha256 (trace.prompt_sha256)
    """
    def __init__(self, *args, menu: str, referencePath: str | None = None, onlyItems: set | None = None, **kwargs):
        # Set before Aggregate.__init__, whose checkInputs already needs them
        self.menu = menu
        self.referencePath = referencePath
        self.onlyItems = onlyItems
        super().__init__(*args, **kwargs)

    @property
    def orderScheme(self) -> str:
        return ORDER_RECORDED if self.referencePath else ORDER_GROUPED

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------
    def checkInputs(self) -> list:
        if not isinstance(self.aggregator, MenuJudgeAggregator):
            raise ValueError("MenuJudge runs the K-way Judge (MenuJudgeAggregator)")
        item_ids = super().checkInputs()

        self.reference = None
        if self.referencePath:
            self.reference = File(self.referencePath)
            meta = self.reference.metadata
            found = (meta.get("prompt_version"), meta.get("candidate_arms"), meta.get("Model", {}).get("modelType"),
                     meta.get("Dataset", {}).get("nums"), meta.get("Aggregator", {}).get("seed"))
            expected = (JudgeAggregator.PROMPT_VERSION, [arm.arm_id for arm in self.arms], self.model.config.modelType,
                        self.dataset.config.nums, self.aggregator.config.seed)
            if found != expected:
                raise ValueError(f"Reference judge file {self.referencePath} does not match the run: {found} != {expected}")

        stored = self.store.metadata
        if stored and (stored.get("menu"), stored.get("order_scheme")) != (self.menu, self.orderScheme):
            raise ValueError(f"{self.store.path} holds menu / order scheme {(stored.get('menu'), stored.get('order_scheme'))}, "
                             f"this run is {(self.menu, self.orderScheme)}")
        return item_ids

    # ------------------------------------------------------------------
    # Presentation order (§3.3)
    # ------------------------------------------------------------------
    @staticmethod
    def groupedOrders(arm_ids: list[str], item_ids: list, seed: int) -> dict:
        """
        rq1kj_criteria.md §3.3: the item_ids (sorted) are shuffled with rng = default_rng(seed); the i-th shuffled item
        belongs to group g = i // K. Group by group, the same rng draws a permutation of `arm_ids` (B_g), and item i
        shows B_g rotated by i mod K. Every arm appears in every position equally often (±1), and the cyclic
        neighbours change from group to group.
        """
        k = len(arm_ids)
        item_ids = sorted(item_ids)
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(item_ids))
        orders, base = {}, None
        for position, index in enumerate(permutation):
            if position % k == 0:
                base = [arm_ids[j] for j in rng.permutation(k)]
            rotation = position % k
            orders[item_ids[index]] = base[rotation:] + base[:rotation]
        return orders

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def judgedIds(self, dis_ids: list) -> list:
        """Disagreement items on which every candidate has a parsed answer (the menu's subset, §2.4)."""
        return [item_id for item_id in dis_ids if all(file.getRecordById(item_id).get("parse_ok") for file in self.armFiles)]

    def presentationOrders(self, dis_ids: list) -> dict:
        judged = self.judgedIds(dis_ids)
        if self.reference is None:
            return self.groupedOrders([arm.arm_id for arm in self.arms], judged, self.aggregator.config.seed)
        missing = [item_id for item_id in judged
                   if not (self.reference.records_map.get(item_id) or {}).get("presentation_order")]
        if missing:
            raise ValueError(f"{self.referencePath} has no recorded presentation order for {len(missing)} judged items "
                             f"(e.g. {missing[:5]}); stopping instead of re-randomizing")
        return {item_id: self.reference.getRecordById(item_id)["presentation_order"] for item_id in judged}

    def selectItems(self, dis_ids: list) -> list:
        judged = self.judgedIds(dis_ids)
        if self.onlyItems is None:
            return judged
        outside = set(self.onlyItems) - set(judged)
        if outside:
            raise ValueError(f"Requested items are not judged items of {self.menu}: {sorted(outside)[:5]}")
        return [item_id for item_id in judged if item_id in self.onlyItems]

    def extraMetadata(self, dis_ids: list) -> dict:
        meta = {
            "menu": self.menu,
            "item_filter": ITEM_FILTER,
            "n_judged_items": len(self.judgedIds(dis_ids)),
            "order_scheme": self.orderScheme,
            "source": "menu_judge",
        }
        if self.referencePath:
            with open(self.referencePath, "rb") as f:
                meta.update(reference_file=self.referencePath, reference_sha256=hashlib.sha256(f.read()).hexdigest())
        return meta
