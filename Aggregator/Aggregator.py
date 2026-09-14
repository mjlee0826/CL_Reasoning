from dataclasses import dataclass
from typing import Callable

from Aggregator.AggregatorConfig import AggregatorConfig
from Aggregator.AggregatorType import AggregatorType, AGGREGATOR_TO_DISPLAYNAME
from Aggregator.AggregationRecord import AggregationRecord
from Arm.ArmSpec import ArmSpec
from Model.Model import Model
from Model.LLMResponse import LLMResponse
from Dataset.Dataset import Dataset


class AggregatorCallError(Exception):
    """An aggregator LLM call failed after all retries; the item is not written so a rerun retries it."""


@dataclass
class Candidate:
    """One candidate arm's generation record for an item, plus the question text that arm was asked."""
    arm: ArmSpec
    record: dict
    question: str

    @property
    def arm_id(self) -> str:
        return self.arm.arm_id

    @property
    def answer(self) -> str:
        return self.record.get("parsed_answer", "")

    @property
    def raw_text(self) -> str:
        return self.record.get("raw_text", "")


@dataclass
class AggregationItem:
    """
    Everything an aggregator needs for one item.
        question            original English question (what the judge is shown)
        presentation_order  arm_ids in display order, precomputed by the Aggregate strategy (Judge only)
    """
    item_id: int
    candidates: list[Candidate]
    question: str
    presentation_order: list[str] | None = None


@dataclass
class Resolution:
    """What a concrete aggregator decided for a disagreement item."""
    final_answer: str
    tokens_in: int = 0
    tokens_out: int = 0
    n_rounds: int | None = None
    presentation_order: list[str] | None = None
    trace: dict | None = None


class Aggregator():
    """
    Base class for all aggregators (Template Method).

    aggregate() implements the part shared by every aggregator in paper_status v7:
      1. if all candidates give the same answer, it is a no-op (the aggregator only fires on disagreement)
      2. otherwise the concrete resolve() decides the final answer
      3. off_menu = the final answer matches none of the candidates' answers
    """
    def __init__(self, config: AggregatorConfig, model: Model, dataset: Dataset,
                 answerParser: Callable[[str], str] | None = None):
        self.config: AggregatorConfig = config
        self.config.displayName = AGGREGATOR_TO_DISPLAYNAME[AggregatorType(config.aggregatorType)].value
        self.model: Model = model
        self.dataset: Dataset = dataset
        # When None, the Aggregate strategy injects Strategy.parseAnswer (the parser used for generation)
        self.parseAnswer = answerParser
        # Set by the Aggregate strategy to accumulate provider usage into the result metadata
        self.onResponse: Callable[[LLMResponse], None] | None = None

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def validateCandidates(self, arms: list[ArmSpec]):
        """Called once before a run. Subclasses add their own requirements."""
        if len(arms) != self.config.k:
            raise ValueError(f"{self.config.displayName} needs {self.config.k} candidate arms, got {len(arms)}")

    def needsPresentationOrder(self) -> bool:
        """True when the aggregator shows candidates to an LLM in an order that must be balanced across items."""
        return False

    def noOp(self, item: AggregationItem) -> Resolution:
        """Resolution when every candidate agrees."""
        return Resolution(final_answer=item.candidates[0].answer)

    def resolve(self, item: AggregationItem) -> Resolution:
        """Decides a disagreement item. Must be overridden by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------
    def isUnanimous(self, answers: list[str]) -> bool:
        return all(self.dataset.compareTwoAnswer(answers[0], answer) for answer in answers[1:])

    def aggregate(self, item: AggregationItem) -> AggregationRecord:
        answers = [c.answer for c in item.candidates]
        resolution = self.noOp(item) if self.isUnanimous(answers) else self.resolve(item)
        off_menu = not any(self.dataset.compareTwoAnswer(resolution.final_answer, answer) for answer in answers)

        return AggregationRecord(
            item_id=item.item_id,
            aggregator_id=self.config.aggregatorType,
            candidate_arms=[c.arm_id for c in item.candidates],
            final_answer=resolution.final_answer,
            off_menu=off_menu,
            n_rounds=resolution.n_rounds,
            presentation_order=resolution.presentation_order,
            tokens_in=resolution.tokens_in,
            tokens_out=resolution.tokens_out,
            trace=resolution.trace,
        )

    # ------------------------------------------------------------------
    # Shared LLM call
    # ------------------------------------------------------------------
    def call(self, messages: list[dict], resolution: Resolution) -> str:
        """
        One aggregator LLM call at T=0. Adds the recounted input / output tokens to `resolution`
        and raises AggregatorCallError when the call failed.
        """
        response = self.model.generate(messages, temperature=0.0)
        if self.onResponse:
            self.onResponse(response)
        if not response.ok:
            raise AggregatorCallError(response.error)

        resolution.tokens_in += sum(self.model.countTokens(m.get("content", "")) for m in messages)
        resolution.tokens_out += self.model.countTokens(response.text)
        return response.text
