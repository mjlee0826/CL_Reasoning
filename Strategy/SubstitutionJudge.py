from Strategy.MenuJudge import MenuJudge

ORDER_PLANNED = "rq3gj_planned"


class SubstitutionJudge(MenuJudge):
    """
    RQ3-GJ (result/analysis/rq3gj/rq3gj_criteria.md): the host judges one version of a menu, in which at most one
    candidate (slot `donorSlot`) is the donor model's output of the same path; the other candidates are the host's.

    Differences from MenuJudge (all through its hooks):
      - the candidate arm file of `donorSlot` belongs to `donorModel` (Aggregate.armModelTypes)
      - presentation orders are given (`orders`, computed once per menu and shared by all versions; K = 2 gets one
        file per order); the run stops when an item to judge has none, instead of re-randomizing
      - `onlyItems` is required: the planned items (subset 2 of the version, answers not all equal) or a pilot sample
        of them; a later run resumes the same file
      - every record also stores the 1-based position of the donor's candidate and whether it was written by the pilot
    """
    def __init__(self, *args, version: str, donorSlot: int | None = None, donorModel: str | None = None, orders: dict,
                 orderLabel: str | None = None, pilot: bool = False, plannedItems: int | None = None, **kwargs):
        # Set before Aggregate.__init__, whose checkInputs already needs them
        self.version = version
        self.donorSlot = donorSlot
        self.donorModel = donorModel
        self.orders = orders
        self.orderLabel = orderLabel
        self.pilot = pilot
        self.plannedItems = plannedItems
        super().__init__(*args, **kwargs)

    @property
    def orderScheme(self) -> str:
        return ORDER_PLANNED

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------
    def checkInputs(self) -> list:
        if (self.donorSlot is None) != (self.donorModel is None):
            raise ValueError("SubstitutionJudge needs both donorSlot and donorModel, or neither")
        if self.donorSlot is not None and not 0 <= self.donorSlot < len(self.arms):
            raise ValueError(f"donorSlot {self.donorSlot} is outside the menu of {len(self.arms)} arms")
        if self.onlyItems is None:
            raise ValueError("SubstitutionJudge only judges planned items (onlyItems)")
        item_ids = super().checkInputs()
        stored = self.store.metadata
        found = tuple(stored.get(key) for key in ("version", "donor", "donor_slot", "order_label")) if stored else None
        expected = (self.version, self.donorModel, self.donorSlot, self.orderLabel)
        if stored and found != expected:
            raise ValueError(f"{self.store.path} holds version / donor / slot / order {found}, this run is {expected}")
        return item_ids

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def armModelTypes(self) -> list[str]:
        types = [self.model.config.modelType] * len(self.arms)
        if self.donorSlot is not None:
            types[self.donorSlot] = self.donorModel
        return types

    def presentationOrders(self, dis_ids: list) -> dict:
        targets = self.selectItems(dis_ids)
        missing = [item_id for item_id in targets if not self.orders.get(item_id)]
        if missing:
            raise ValueError(f"No planned presentation order for {len(missing)} items to judge (e.g. {missing[:5]}); "
                             "stopping instead of re-randomizing")
        arm_ids = sorted(arm.arm_id for arm in self.arms)
        wrong = [item_id for item_id in targets if sorted(self.orders[item_id]) != arm_ids]
        if wrong:
            raise ValueError(f"Planned orders of {len(wrong)} items are not a permutation of the menu (e.g. {wrong[:5]})")
        return {item_id: list(self.orders[item_id]) for item_id in targets}

    def aggregateItem(self, item_id, presentation_order) -> dict:
        record = super().aggregateItem(item_id, presentation_order)
        record["donor_position"] = (presentation_order.index(self.arms[self.donorSlot].arm_id) + 1
                                    if self.donorSlot is not None else None)
        record["pilot"] = self.pilot
        return record

    def extraMetadata(self, dis_ids: list) -> dict:
        meta = super().extraMetadata(dis_ids)
        meta.update(experiment="rq3gj", version=self.version, donor=self.donorModel, donor_slot=self.donorSlot,
                    donor_arm=self.arms[self.donorSlot].arm_id if self.donorSlot is not None else None,
                    order_label=self.orderLabel, candidate_models=self.armModelTypes(), n_planned_items=self.plannedItems)
        return meta
