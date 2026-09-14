from Aggregator.Aggregator import Aggregator, AggregationItem, Resolution
from Strategy.PromptAbstractFactory.PromptFormatFactory import PromptFormatFactory
from Strategy.PromptAbstractFactory.PromptDebateCOTFactory import PromptDebateCOTFactory
from Strategy.PromptAbstractFactory.PromptTwoResultCOTFactory import PromptTwoResultCOTFactory


class DebateAggregator(Aggregator):
    """
    Debate (IMSR). Same procedure and prompts as Strategy/Challenge.py, so new and legacy debates are comparable:
      - each agent's history starts with [its question text, its generation output]
      - up to `threshold` rounds: both agents are shown the opponent's previous output (PromptDebateCOTFactory)
      - if they still disagree, a judge in agent A's language compares the two latest outputs
        (PromptTwoResultCOTFactory)
    A is the first candidate arm; presentation_order is IMSR's fixed [A, B].
    """
    def noOp(self, item: AggregationItem) -> Resolution:
        return Resolution(final_answer=item.candidates[0].answer, n_rounds=0)

    @staticmethod
    def getDebatePrompt(target_lang: str, opponent_answer: str) -> str:
        return PromptDebateCOTFactory().getPrompt(target_lang, opponent_answer) + PromptFormatFactory().getPrompt(target_lang)

    @staticmethod
    def getJudgePrompt(target_lang: str, question: str, result1: str, result2: str, lang1: str, lang2: str) -> str:
        return PromptTwoResultCOTFactory().getPrompt(target_lang, question, result1, result2, lang1, lang2) \
            + PromptFormatFactory().getPrompt(target_lang)

    def resolve(self, item: AggregationItem) -> Resolution:
        agent1, agent2 = item.candidates
        lang1, lang2 = agent1.arm.language, agent2.arm.language
        resolution = Resolution(final_answer="", presentation_order=[agent1.arm_id, agent2.arm_id])

        result1, result2 = agent1.raw_text, agent2.raw_text
        answer1, answer2 = agent1.answer, agent2.answer
        record1 = [
            {"role": "user", "content": agent1.question},
            {"role": "assistant", "content": result1}
        ]
        record2 = [
            {"role": "user", "content": agent2.question},
            {"role": "assistant", "content": result2}
        ]
        answerRecord1, answerRecord2 = [answer1], [answer2]
        cur_turn = 0

        # Both debate prompts use the opponent's previous output (Challenge.runChallenge)
        while not self.dataset.compareTwoAnswer(answer1, answer2) and cur_turn < self.config.threshold:
            record1.append({"role": "user", "content": self.getDebatePrompt(lang1, result2)})
            record2.append({"role": "user", "content": self.getDebatePrompt(lang2, result1)})

            result1 = self.call(record1, resolution)
            result2 = self.call(record2, resolution)

            record1.append({"role": "assistant", "content": result1})
            record2.append({"role": "assistant", "content": result2})

            answer1, answer2 = self.parseAnswer(result1), self.parseAnswer(result2)
            answerRecord1.append(answer1)
            answerRecord2.append(answer2)
            cur_turn += 1

        result3 = ""
        if self.dataset.compareTwoAnswer(answer1, answer2):
            resolution.final_answer = answer1
        else:
            judge_prompt = self.getJudgePrompt(lang1, agent1.question, result1, result2, lang1, lang2)
            result3 = self.call([{"role": "user", "content": judge_prompt}], resolution)
            resolution.final_answer = self.parseAnswer(result3)

        resolution.n_rounds = cur_turn
        # The first two messages of each history are the question and the generation output, which the arm
        # files already hold; only the debate turns are kept.
        resolution.trace = {
            "Record1": record1[2:],
            "Record2": record2[2:],
            "AnswerRecord1": answerRecord1,
            "AnswerRecord2": answerRecord2,
            "Result3": result3,
        }
        return resolution
