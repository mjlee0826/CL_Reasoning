from enum import Enum

# Aggregators of paper_status v7 §3.4 (aggregator_id values)
class AggregatorType(str, Enum):
    V2 = "v2"
    BLIND = "blind"
    JUDGE = "judge"
    DEBATE = "debate"
    VOTE3 = "vote3"
    VOTE5 = "vote5"
    JUDGE5 = "judge5"

class AggregatorDisplayNameType(str, Enum):
    V2 = "Vote@2"
    BLIND = "Blind"
    JUDGE = "Judge"
    DEBATE = "Debate"
    VOTE3 = "Vote@3"
    VOTE5 = "Vote@5"
    JUDGE5 = "Judge@5"

AGGREGATOR_TO_DISPLAYNAME = {
    member: AggregatorDisplayNameType[member.name] for member in AggregatorType
}

# Number of candidate arms each aggregator takes
AGGREGATOR_TO_K = {
    AggregatorType.V2: 2,
    AggregatorType.BLIND: 2,
    AggregatorType.JUDGE: 2,
    AggregatorType.DEBATE: 2,
    AggregatorType.VOTE3: 3,
    AggregatorType.VOTE5: 5,
    AggregatorType.JUDGE5: 5,
}

AGGREGATOR_STR_LIST = [a.value for a in AggregatorType]

def get_aggregator_map():
    """
    Returns a mapping of AggregatorTypes to their concrete classes.
    Uses lazy importing to prevent circular dependency issues during initialization.
    """
    from Aggregator.VoteAggregator import VoteAggregator
    from Aggregator.BlindAggregator import BlindAggregator
    from Aggregator.JudgeAggregator import JudgeAggregator
    from Aggregator.DebateAggregator import DebateAggregator

    return {
        AggregatorType.V2: VoteAggregator,
        AggregatorType.VOTE3: VoteAggregator,
        AggregatorType.VOTE5: VoteAggregator,
        AggregatorType.BLIND: BlindAggregator,
        AggregatorType.JUDGE: JudgeAggregator,
        AggregatorType.JUDGE5: JudgeAggregator,
        AggregatorType.DEBATE: DebateAggregator,
    }
