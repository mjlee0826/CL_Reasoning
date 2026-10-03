from dataclasses import dataclass

@dataclass
class LLMResponse:
    """
    Result of a single Model.generate() call.

    usage_in / usage_out are the provider-reported token counts. They are only summed into the
    result file metadata for reconciliation; record-level tokens are recounted with Model.countTokens
    so that legacy and new records share one definition.
    error is set (and text left empty) when the call still failed after all retries.
    refused is True when the provider blocked the content (safety filter / moderation). That is the
    model's answer, not a failure: text stays empty, the call counts as ok, and the output is scored
    as having no answer.
    """
    text: str = ""
    model_version: str = ""
    usage_in: int | None = None
    usage_out: int | None = None
    error: str | None = None
    refused: bool = False
    refusal_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
