from dataclasses import dataclass, asdict

@dataclass
class GenerationRecord:
    """
    One arm's answer to one item (record schema of result/arms/{model}/{dataset}/{arm}.json).
    tokens_in / tokens_out are recounted with Model.countTokens (same definition for legacy and new data).
    """
    item_id: int
    arm_id: str
    axis: str
    is_anchor: bool
    lang: str
    raw_text: str
    parsed_answer: str
    parse_ok: bool
    tokens_in: int | None
    tokens_out: int | None
    model: str
    model_version_string: str
    temperature: float
    seed: int | None
    prompt_hash: str
    gold: str

    @staticmethod
    def isParseOk(answer) -> bool:
        """Negation of Test/TestMissingAnswer.isMissing: empty, "null" and "None" are parse failures."""
        return bool(answer) and str(answer).strip() not in ("", "null", "None")

    def to_dict(self) -> dict:
        return asdict(self)
