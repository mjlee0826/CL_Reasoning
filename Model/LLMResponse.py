from dataclasses import dataclass

@dataclass
class LLMResponse:
    """
    Result of a single Model.generate() call.

    usage_in / usage_out are the provider-reported token counts. They are only summed into the
    result file metadata for reconciliation; record-level tokens are recounted with Model.countTokens
    so that legacy and new records share one definition.
    error is set (and text left empty) when the call still failed after all retries.
    """
    text: str = ""
    model_version: str = ""
    usage_in: int | None = None
    usage_out: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
