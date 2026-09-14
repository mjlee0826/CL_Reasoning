from Strategy.PromptAbstractFactory.PromptAbstractFactory import PromptAbstractFactory


class PromptPersonaFactory(PromptAbstractFactory):
    """
    Persona prefix for P-axis arms (paper_status v7 §3.3, 2 levels).

    The persona texts are not specified yet. Add them as
        PERSONAS[persona_key][language] = "..."
    An arm whose persona key / language has no text fails loudly instead of silently running
    without a persona.
    """
    PERSONAS: dict[str, dict[str, str]] = {}

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
