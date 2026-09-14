from Aggregator.Aggregator import Aggregator, AggregationItem, Resolution
from Arm.ArmSpec import ArmSpec


class BlindAggregator(Aggregator):
    """
    Blind: always keeps the anchor's answer (deployment baseline, recovery = 2·w_A − 1).
    """
    def validateCandidates(self, arms: list[ArmSpec]):
        super().validateCandidates(arms)
        if sum(arm.is_anchor for arm in arms) != 1:
            raise ValueError(f"Blind needs exactly one anchor candidate, got {[arm.arm_id for arm in arms]}")

    def resolve(self, item: AggregationItem) -> Resolution:
        anchor = next(c for c in item.candidates if c.arm.is_anchor)
        return Resolution(final_answer=anchor.answer)
