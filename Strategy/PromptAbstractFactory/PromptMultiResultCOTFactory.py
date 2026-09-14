from Strategy.PromptAbstractFactory.PromptAbstractFactory import PromptAbstractFactory


class PromptMultiResultCOTFactory(PromptAbstractFactory):
    """
    K-way judge prompt (Judge / Judge@5) with neutral candidate labels "Answer 1..K".

    Wording follows PromptTwoResultCOTFactory, but the judge never sees the language (or any other
    manipulation) of a candidate, so the same prompt is used on every axis. The judge is always
    prompted in English; concatenate PromptFormatFactory so it outputs the option itself
    ({"answer": "..."}), which keeps off-menu answers observable.
    """
    def __init__(self):
        super().__init__()

    def englishPrompt(self, question: str, answers: list[str]):
        prompt = f'For the following question\n```\n{question}\n```\n' \
            f'There are {len(answers)} answers as follows\n'
        for i, answer in enumerate(answers, start=1):
            prompt += f'Answer {i}\n```\n{answer}\n```\n'
        prompt += 'Based on the question, select and output the most correct answer.' \
            'You must think step by step about which parts of the reasoning in each answer are incorrect, and output your reasoning process.\n'
        return prompt
