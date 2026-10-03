from Arm.ArmSpec import ArmSpec
from Strategy.PromptAbstractFactory.PromptCOTFactory import PromptCOTFactory
from Strategy.PromptAbstractFactory.PromptShortCOTFactory import PromptShortCOTFactory
from Strategy.PromptAbstractFactory.PromptDirectFactory import PromptDirectFactory
from Strategy.PromptAbstractFactory.PromptFormatFactory import PromptFormatFactory
from Strategy.PromptAbstractFactory.PromptPersonaFactory import PromptPersonaFactory
from Strategy.PromptAbstractFactory.PromptSelfReflectionCOTFactory import PromptSelfReflectionCOTFactory

import hashlib
import json

class PromptBuilder():
    """
    Builds the generation prompt of an arm by composing the existing prompt factories.
    This is the single source of the single-agent prompt; it reproduces the legacy OnlyOneLanguage prompts
    (golden hashes in check_framework check 1).
    """
    def __init__(self, arm: ArmSpec):
        self.arm = arm

    @staticmethod
    def buildText(language: str, question: str, promptStyle: str = "cot", persona: str = None) -> str:
        """
        promptStyle:
          'cot'       -> full Chain-of-Thought + Format factory (historical default)
          'short_cot' -> brief Chain-of-Thought + Format factory
          'direct'    -> self-contained direct-answer prompt (must NOT be combined with the Format factory)
        persona: optional PromptPersonaFactory key, prepended to the prompt.
        """
        if promptStyle == "direct":
            text = PromptDirectFactory().getPrompt(language, question)
        else:
            factory = PromptShortCOTFactory() if promptStyle == "short_cot" else PromptCOTFactory()
            text = factory.getPrompt(language, question) + PromptFormatFactory().getPrompt(language)

        if persona is not None:
            text = PromptPersonaFactory().getPrompt(language, persona) + text
        return text

    @staticmethod
    def buildReflectionText(language: str, question: str, previous_output: str) -> str:
        """
        Prompt of a derived (F-axis) arm: the base arm's output is part of the prompt.
        Same composition as the legacy SelfReflection strategy (golden hashes in check_framework check 2e),
        so imported self-reflection prompts rebuild exactly.
        """
        return PromptSelfReflectionCOTFactory().getPrompt(language, question, previous_output) \
            + PromptFormatFactory().getPrompt(language)

    def text(self, question: str, base_raw_text: str = None) -> str:
        if self.arm.is_derived:
            if base_raw_text is None:
                raise ValueError(f"{self.arm.arm_id} is derived; its prompt needs the base arm's output")
            return self.buildReflectionText(self.arm.language, question, base_raw_text)
        return self.buildText(self.arm.language, question, self.arm.promptStyle, self.arm.persona)

    def messages(self, question: str, base_raw_text: str = None) -> list[dict]:
        return [{"role": "user", "content": self.text(question, base_raw_text)}]

    @staticmethod
    def promptHash(messages: list[dict]) -> str:
        """sha256 of the full messages actually sent (canonical JSON), first 16 hex chars."""
        canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
