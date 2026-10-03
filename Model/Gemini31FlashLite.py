from Model.Gemini import Gemini


class Gemini31FlashLite(Gemini):
    """
    gemini-3.1-flash-lite through the same OpenAI-compatible endpoint as Gemini (2.5 Flash-Lite).

    - thinking_level "minimal": on Gemini 3, reasoning_effort maps 1:1 to thinking_level. "minimal" is also this
      model's default, and Google notes it does not guarantee that thinking is off.
    - max_tokens 8192, the same cap as GPT-4o mini / Qwen (Gemini 2.5 used 4096).
    - seed is rejected as on Gemini 2.5 (SUPPORTS_SEED = False) and tokens are counted with count_tokens (inherited).

    Google recommends T=1.0 for Gemini 3 (lower values "may lead to looping"); the arms keep the protocol's T=0.
    On the 150 items where Gemini 2.5 looped or was cut off at T=0, this model finished every answer (max 1,513 tokens).
    """
    MAX_TOKENS = 8192
    REASONING_EFFORT = "minimal"
