from Strategy.PromptAbstractFactory.PromptAbstractFactory import PromptAbstractFactory


class PromptJudgeChoiceFactory(PromptAbstractFactory):
    """
    K-way judge prompt (Judge / Judge@5): the judge selects one candidate and outputs its number ({"choice":"N"}),
    so the final answer is always one of the candidates' answers.

    Candidates keep the neutral labels "Answer 1..K" of PromptMultiResultCOTFactory (the earlier judge prompt, which
    asked for the option itself and let off-menu answers through). Numbers cannot be confused with the A-E option
    letters, so a judge that outputs an option letter instead of a candidate number is detected as an invalid choice.
    The judge is always prompted in English, on every axis.
    """
    def __init__(self):
        super().__init__()

    def englishPrompt(self, question: str, answers: list[str]):
        k = len(answers)
        prompt = f'For the following question\n```\n{question}\n```\n' \
            f'There are {k} answers as follows\n'
        for i, answer in enumerate(answers, start=1):
            prompt += f'Answer {i}\n```\n{answer}\n```\n'
        prompt += f'Based on the question, select the most correct one of these {k} answers. ' \
            'You must think step by step about which parts of the reasoning in each answer are incorrect, and output your reasoning process.\n'
        prompt += 'Please strictly follow the format below for output\n' \
            'Reasoning process\n' \
            '{your reasoning process - Note: **Do not restate the original question text or add content not required by the question**}\n\n' \
            'Final Choice\n' \
            '{"choice":"answer number"}\n' \
            f'(You shouldn\'t output "answer number" directly. Replace it with the number of the answer you select, an integer from 1 to {k}. ' \
            'Do not output an option letter of the question or an answer of your own. The entire final choice block must only be ' \
            'that one line of JSON, with no extra text or explanation before or after.)\n'
        return prompt
