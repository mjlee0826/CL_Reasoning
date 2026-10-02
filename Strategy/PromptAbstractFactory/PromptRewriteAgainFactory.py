from Strategy.PromptAbstractFactory.PromptAbstractFactory import PromptAbstractFactory


class PromptRewriteAgainFactory(PromptAbstractFactory):
    """
    Prompt for a later rewrite version (run_rewrite.py --version n, n >= 2).

    The constraints are worded like PromptRewriteFactory. The added requirement (differ from the
    existing rewrites) is scoped to the question stem only: an earlier wording ("clearly different
    wording and sentence structure") made the model also drop the answer options and instructions
    in ~3–16% of MMLU / TruthfulQA / CSQA items. Rewrites operate on the English question only.
    """
    # Follow-up turn when a rewrite dropped answer options or instructions (see Rewrite.preservesOptions)
    REPAIR_MESSAGE = (
        'Your rewrite dropped some of the original answer options or instructions. Output the complete '
        'rewritten question again: keep your new wording of the question stem, and copy every answer '
        'option line and every instruction from the original question exactly. Output only the question.'
    )

    def __init__(self):
        super().__init__()

    def englishPrompt(self, question: str, previous: list[str]):
        prompt = (
            'Rewrite the question stem using different wording while preserving the exact '
            'meaning. Do NOT change any numbers, names, units, or answer options. Output the '
            'answer options verbatim. Keep every instruction and answer-format requirement. '
            'Do NOT solve the question, do NOT output a JSON answer of your own, and do NOT add comments. '
            'The question has already been rewritten as shown below. Word the question stem differently '
            'from both the original question and every existing rewrite. Only the question stem may '
            'change: copy every answer option line and every instruction that follows the options '
            '(including the answer-format line) exactly as they appear in the original question, and '
            'never drop or shorten them.\n'
            'Original question:\n```\n' + question + '\n```\n'
        )
        for i, text in enumerate(previous, start=1):
            prompt += f'Existing rewrite {i}:\n```\n{text}\n```\n'
        prompt += 'Output only your new rewritten question, including all of its answer options and instructions.\n'
        return prompt
