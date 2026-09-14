from typing import Callable

from Aggregator.Aggregator import Aggregator
from Aggregator.AggregatorConfig import AggregatorConfig
from Aggregator.AggregatorType import AggregatorType, AGGREGATOR_TO_K, get_aggregator_map
from Model.Model import Model
from Dataset.Dataset import Dataset

class AggregatorFactory():
    """
    Factory class responsible for instantiating aggregators.
    Several aggregator_ids share one class (v2 / vote3 / vote5 -> VoteAggregator), so the factory also
    fixes config.aggregatorType and config.k from the requested type.
    """
    def __init__(self):
        pass

    def buildAggregator(self, type: AggregatorType, config: AggregatorConfig, model: Model, dataset: Dataset,
                        answerParser: Callable[[str], str] | None = None) -> Aggregator:
        aggregator_type = AggregatorType(type)
        aggregator_cls = get_aggregator_map().get(aggregator_type)

        if not aggregator_cls:
            print(f"Error: Aggregator '{type}' doesn't exist!")
            return None

        config.aggregatorType = aggregator_type.value
        config.k = AGGREGATOR_TO_K[aggregator_type]
        return aggregator_cls(config, model, dataset, answerParser)
