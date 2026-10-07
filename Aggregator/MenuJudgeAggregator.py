from typing import Callable

from Aggregator.Aggregator import AggregationItem, Resolution
from Aggregator.AggregatorConfig import AggregatorConfig
from Aggregator.AggregatorType import AggregatorType
from Aggregator.JudgeAggregator import JudgeAggregator
from Arm.PromptBuilder import PromptBuilder
from Dataset.Dataset import Dataset
from Model.Model import Model


class MenuJudgeAggregator(JudgeAggregator):
    """
    RQ1-KJ (result/analysis/rq1kj/rq1kj_criteria.md §3): the Judge over all K candidates of a menu.

    Prompt text, candidate content, parsing and scoring are the main grid's choice-v1 (PromptJudgeChoiceFactory already
    takes K); only the number of candidates changes, so its files carry their own prompt_version and the main-grid
    judge files cannot be resumed with it. Built directly instead of through AggregatorFactory, which fixes the
    judge at K = 2: aggregator_id stays "judge" and config.k = K.

    The full sha256 of the messages sent goes into the trace (prompt_sha256; its first 16 hex chars are the
    PromptBuilder.promptHash of an arm record).
    """
    PROMPT_VERSION = "choice-k-v1"

    def __init__(self, k: int, model: Model, dataset: Dataset, seed: int = 0,
                 answerParser: Callable[[str], str] | None = None):
        super().__init__(AggregatorConfig(aggregatorType=AggregatorType.JUDGE.value, k=k, seed=seed), model, dataset, answerParser)
        self.lastPromptSha256 = None

    def call(self, messages: list[dict], resolution: Resolution) -> str:
        self.lastPromptSha256 = PromptBuilder.promptSha256(messages)
        return super().call(messages, resolution)

    def resolve(self, item: AggregationItem) -> Resolution:
        self.lastPromptSha256 = None
        resolution = super().resolve(item)
        resolution.trace["prompt_sha256"] = self.lastPromptSha256
        return resolution


class PairChoiceJudgeAggregator(MenuJudgeAggregator):
    """
    RQ3-GJ §8.14 (result/analysis/rq3gj/rq3gj_criteria.md): the K = 2 Judge under the main grid's prompt_version
    choice-v1. The text comes from the same PromptJudgeChoiceFactory, so at K = 2 it is the main-grid prompt word for
    word; the prompt sha256 is recorded as in MenuJudgeAggregator.
    """
    PROMPT_VERSION = JudgeAggregator.PROMPT_VERSION

    def __init__(self, model: Model, dataset: Dataset, seed: int = 0, answerParser: Callable[[str], str] | None = None):
        super().__init__(2, model, dataset, seed, answerParser)
