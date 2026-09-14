from dataclasses import dataclass, asdict

@dataclass
class AggregationRecord:
    """
    One aggregator decision on one item
    (record schema of result/aggregations/{model}/{dataset}/{aggregator}__{arm}__....json).

    off_menu            final_answer matches none of the candidates' parsed answers
    n_rounds            debate rounds (Debate only; None for other aggregators)
    presentation_order  arm_ids in the order the LLM saw them (None when no LLM call was made)
    tokens_in / out     summed over every aggregator call on this item, recounted with Model.countTokens
    trace               raw judge output / debate transcript (None when no LLM call was made)
    """
    item_id: int
    aggregator_id: str
    candidate_arms: list[str]
    final_answer: str
    off_menu: bool
    n_rounds: int | None
    presentation_order: list[str] | None
    tokens_in: int
    tokens_out: int
    trace: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)
