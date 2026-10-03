import re

from Aggregator.Aggregator import Aggregator, AggregationItem, Resolution
from Strategy.PromptAbstractFactory.PromptJudgeChoiceFactory import PromptJudgeChoiceFactory
from Strategy.StrategyType import LanguageType


class JudgeAggregator(Aggregator):
    """
    Judge (K=2) / Judge@5 (K=5): a single T=0 call that sees the original English question and every
    candidate's full raw output under neutral labels "Answer 1..K", and selects one of them by number.

    Candidates are shown in item.presentation_order, which the Aggregate strategy balances across the
    disagreement items (LLM judges have position bias, Wang et al. ACL 2024). The final answer is the selected
    candidate's answer; a missing or out-of-range choice gives no answer (scored as wrong), which is then the only
    way to be off-menu. The earlier prompt asked for the option itself (PromptMultiResultCOTFactory + PromptFormatFactory);
    PROMPT_VERSION keeps files written with it from being resumed.
    """
    JUDGE_LANGUAGE = LanguageType.ENGLISH.value
    PROMPT_VERSION = "choice-v1"
    # The last {"choice": "N"} in the output; "Answer N" inside the quotes is tolerated
    CHOICE_PATTERN = re.compile(r'\{\s*"choice"\s*:\s*"?\s*(?:answer\s*)?(\d+)\s*"?\s*\}', re.IGNORECASE)

    def needsPresentationOrder(self) -> bool:
        return True

    def getJudgePrompt(self, question: str, answers: list[str]) -> str:
        return PromptJudgeChoiceFactory().getPrompt(self.JUDGE_LANGUAGE, question, answers)

    @classmethod
    def parseChoice(cls, output: str, k: int) -> int | None:
        """1-based candidate number of the last {"choice": "N"} in the output; None when missing or out of range."""
        matches = cls.CHOICE_PATTERN.findall(output or "")
        if not matches:
            return None
        choice = int(matches[-1])
        return choice if 1 <= choice <= k else None

    def resolve(self, item: AggregationItem) -> Resolution:
        if not item.presentation_order:
            raise ValueError(f"Judge needs a presentation_order (item {item.item_id})")

        by_arm = {c.arm_id: c for c in item.candidates}
        ordered = [by_arm[arm_id] for arm_id in item.presentation_order]
        prompt = self.getJudgePrompt(item.question, [c.raw_text for c in ordered])

        resolution = Resolution(final_answer="", presentation_order=list(item.presentation_order))
        output = self.call([{"role": "user", "content": prompt}], resolution)
        choice = self.parseChoice(output, len(ordered))
        chosen = ordered[choice - 1] if choice else None
        resolution.final_answer = chosen.answer if chosen else ""
        resolution.trace = {"judge_output": output, "choice": choice, "chosen_arm": chosen.arm_id if chosen else None}
        return resolution
