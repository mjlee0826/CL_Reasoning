from Aggregator.Aggregator import Aggregator, AggregationItem, Resolution
from Strategy.PromptAbstractFactory.PromptMultiResultCOTFactory import PromptMultiResultCOTFactory
from Strategy.PromptAbstractFactory.PromptFormatFactory import PromptFormatFactory
from Strategy.StrategyType import LanguageType


class JudgeAggregator(Aggregator):
    """
    Judge (K=2) / Judge@5 (K=5): a single T=0 call that sees the original English question and every
    candidate's full raw output under neutral labels "Answer 1..K".

    Candidates are shown in item.presentation_order, which the Aggregate strategy balances across the
    disagreement items (LLM judges have position bias, Wang et al. ACL 2024). The judge outputs the option
    itself, so an off-menu final answer stays possible and is measured by off_menu.
    """
    JUDGE_LANGUAGE = LanguageType.ENGLISH.value

    def needsPresentationOrder(self) -> bool:
        return True

    def getJudgePrompt(self, question: str, answers: list[str]) -> str:
        return PromptMultiResultCOTFactory().getPrompt(self.JUDGE_LANGUAGE, question, answers) \
            + PromptFormatFactory().getPrompt(self.JUDGE_LANGUAGE)

    def resolve(self, item: AggregationItem) -> Resolution:
        if not item.presentation_order:
            raise ValueError(f"Judge needs a presentation_order (item {item.item_id})")

        by_arm = {c.arm_id: c for c in item.candidates}
        ordered = [by_arm[arm_id] for arm_id in item.presentation_order]
        prompt = self.getJudgePrompt(item.question, [c.raw_text for c in ordered])

        resolution = Resolution(final_answer="", presentation_order=list(item.presentation_order))
        output = self.call([{"role": "user", "content": prompt}], resolution)
        resolution.final_answer = self.parseAnswer(output)
        resolution.trace = {"judge_output": output}
        return resolution
