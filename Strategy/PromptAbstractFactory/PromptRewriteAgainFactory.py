from Strategy.PromptAbstractFactory.PromptAbstractFactory import PromptAbstractFactory


class PromptRewriteAgainFactory(PromptAbstractFactory):
    """
    Prompt for a later rewrite version (run_rewrite.py --version n, n >= 2).

    The constraints are worded exactly like PromptRewriteFactory, so every version follows the same
    rules; the only addition is the list of existing rewrites, which the new paraphrase must differ
    from in wording and sentence structure. Rewrites operate on the English question only.
    """
    def __init__(self):
        super().__init__()

    def englishPrompt(self, question: str, previous: list[str]):
        prompt = (
            'Rewrite the question stem using different wording while preserving the exact '
            'meaning. Do NOT change any numbers, names, units, or answer options. Output the '
            'answer options verbatim. Keep every instruction and answer-format requirement. '
            'Do NOT solve the question, do NOT output any JSON answer, and do NOT add comments. '
            'The question has already been rewritten as shown below. Your rewrite must use clearly '
            'different wording and sentence structure from both the original question and every '
            'existing rewrite.\n'
            'Original question:\n```\n' + question + '\n```\n'
        )
        for i, text in enumerate(previous, start=1):
            prompt += f'Existing rewrite {i}:\n```\n{text}\n```\n'
        prompt += 'Output only your new rewritten question.\n'
        return prompt
