from Aggregator.Aggregator import Aggregator, AggregationItem, Resolution
from Arm.ArmSpec import ArmSpec


class ReviseAggregator(Aggregator):
    """
    Revise: always keeps the derived candidate's answer (e.g. the self-reflection revision).

    Mirror image of Blind, which always keeps the anchor: its recovery is 2·w_B − 1 = −recovery_blind.
    It reproduces the self-reflection pipeline's own output inside the framework, which is what makes
    SR comparable with V2 / Judge / Debate on the same candidate pair.
    The reflection call itself is generation cost and is accounted in the derived arm's record, so this
    aggregator makes no model call.
    """
    def validateCandidates(self, arms: list[ArmSpec]):
        super().validateCandidates(arms)
        if sum(arm.is_derived for arm in arms) != 1:
            raise ValueError(f"Revise needs exactly one derived candidate, got {[arm.arm_id for arm in arms]}")

    def resolve(self, item: AggregationItem) -> Resolution:
        derived = next(c for c in item.candidates if c.arm.is_derived)
        return Resolution(final_answer=derived.answer)
