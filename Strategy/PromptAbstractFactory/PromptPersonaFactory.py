from Strategy.PromptAbstractFactory.PromptAbstractFactory import PromptAbstractFactory


class PromptPersonaFactory(PromptAbstractFactory):
    """
    Persona prefix for P-axis arms (paper_status v7 §3.3, 2 levels): PERSONAS[persona_key][language].

    The texts state an identity only, no procedure, so a persona does not overlap with the R-axis
    reasoning instructions. P-axis arms change one factor at a time and therefore only use the English
    question, so only English texts exist. An arm whose persona key / language has no text fails loudly
    instead of silently running without a persona.
    """
    PERSONAS: dict[str, dict[str, str]] = {
        "expert": {
            "english": "You are a seasoned expert with deep knowledge across mathematics, science, history, "
                       "and everyday common sense.\n\n",
        },
        "skeptic": {
            "english": "You are a careful skeptic who is hard to fool and never takes a first impression "
                       "at face value.\n\n",
        },
    }

    def __init__(self):
        super().__init__()

    def _lookup(self, language: str, persona: str) -> str:
        text = self.PERSONAS.get(persona, {}).get(language)
        if not text:
            raise ValueError(f"Persona '{persona}' has no {language} text yet; add it to PromptPersonaFactory.PERSONAS")
        return text

    def englishPrompt(self, persona: str):
        return self._lookup("english", persona)

    def chinesePrompt(self, persona: str):
        return self._lookup("chinese", persona)

    def spanishPrompt(self, persona: str):
        return self._lookup("spanish", persona)

    def japanesePrompt(self, persona: str):
        return self._lookup("japanese", persona)

    def russianPrompt(self, persona: str):
        return self._lookup("russian", persona)
