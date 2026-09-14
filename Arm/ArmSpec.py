from dataclasses import dataclass, asdict, fields
import re

from Arm.AxisType import (AxisType, LANG_CODE_TO_LANGUAGE, PROMPT_STYLE_LIST, QUESTION_SOURCE_LIST,
                          ANCHOR_ARM_ID)
from Dataset.DatasetConfig import DatasetConfig

# The single factor each axis is allowed to change relative to Agent A
AXIS_TO_FACTOR = {
    AxisType.LANGUAGE: "lang",
    AxisType.SAMPLING: "sampling",
    AxisType.REASONING: "promptStyle",
    AxisType.PERSONA: "persona",
    AxisType.REWRITE: "questionSource",
}

@dataclass(frozen=True)
class ArmSpec:
    """
    One generation arm = Agent A (the anchor) with exactly one factor changed along its axis.
    The defaults are Agent A: original English question + full CoT + T=0 (paper_status v7 §3.3).

    arm_id grammar (canonical form, one factor at a time):
        L:{lang}             lang in en / zh / ja / ru / es ("L:en" is the anchor)
        S:T{temp}:seed{n}    temperature > 0 with one decimal, integer seed
        R:{promptStyle}      short_cot / direct
        P:{persona}          persona key of PromptPersonaFactory
        W:{questionSource}   rewrite
    K-axis arms are not defined yet (the candidate source is still open).
    """
    axis: str
    lang: str = "en"
    temperature: float = 0.0
    seed: int | None = None
    promptStyle: str = "cot"
    persona: str | None = None
    questionSource: str = "original"

    def __post_init__(self):
        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def changedFactors(self) -> set:
        """Factors that differ from Agent A."""
        changed = set()
        if self.lang != "en":
            changed.add("lang")
        if self.temperature != 0.0 or self.seed is not None:
            changed.add("sampling")
        if self.promptStyle != "cot":
            changed.add("promptStyle")
        if self.persona is not None:
            changed.add("persona")
        if self.questionSource != "original":
            changed.add("questionSource")
        return changed

    def validate(self):
        if self.axis not in [a.value for a in AxisType]:
            raise ValueError(f"Unknown axis '{self.axis}'")
        if self.axis == AxisType.CANDIDATES:
            raise ValueError("K-axis arms are not defined yet")
        if self.lang not in LANG_CODE_TO_LANGUAGE:
            raise ValueError(f"Unknown language code '{self.lang}'")
        if self.promptStyle not in PROMPT_STYLE_LIST:
            raise ValueError(f"Unknown promptStyle '{self.promptStyle}'")
        if self.questionSource not in QUESTION_SOURCE_LIST:
            raise ValueError(f"Unknown questionSource '{self.questionSource}'")
        if self.persona is not None and not re.fullmatch(r"[a-z][a-z0-9_]*", self.persona):
            raise ValueError(f"Persona key must be lowercase snake_case, got '{self.persona}'")

        factor = AXIS_TO_FACTOR[AxisType(self.axis)]
        changed = self.changedFactors()
        if not changed <= {factor}:
            raise ValueError(f"Axis {self.axis} may only change '{factor}', but {sorted(changed)} differ from the anchor")
        if not changed and self.axis != AxisType.LANGUAGE:
            raise ValueError(f"Axis {self.axis} arm is identical to the anchor; use {ANCHOR_ARM_ID}")

        if self.axis == AxisType.SAMPLING:
            if self.temperature <= 0 or self.seed is None:
                raise ValueError("S-axis arms need temperature > 0 and a seed")
            if round(self.temperature, 1) != self.temperature:
                raise ValueError(f"S-axis temperature must have one decimal, got {self.temperature}")

    # ------------------------------------------------------------------
    # arm_id
    # ------------------------------------------------------------------
    @property
    def arm_id(self) -> str:
        if self.axis == AxisType.LANGUAGE:
            return f"L:{self.lang}"
        if self.axis == AxisType.SAMPLING:
            return f"S:T{self.temperature:.1f}:seed{self.seed}"
        if self.axis == AxisType.REASONING:
            return f"R:{self.promptStyle}"
        if self.axis == AxisType.PERSONA:
            return f"P:{self.persona}"
        return f"W:{self.questionSource}"

    @classmethod
    def from_arm_id(cls, arm_id: str) -> "ArmSpec":
        axis, _, rest = arm_id.partition(":")
        if axis == AxisType.LANGUAGE:
            spec = cls(axis, lang=rest)
        elif axis == AxisType.SAMPLING:
            match = re.fullmatch(r"T(\d+\.\d):seed(\d+)", rest)
            if not match:
                raise ValueError(f"S-axis arm_id must look like S:T0.7:seed3, got '{arm_id}'")
            spec = cls(axis, temperature=float(match.group(1)), seed=int(match.group(2)))
        elif axis == AxisType.REASONING:
            spec = cls(axis, promptStyle=rest)
        elif axis == AxisType.PERSONA:
            spec = cls(axis, persona=rest)
        elif axis == AxisType.REWRITE:
            spec = cls(axis, questionSource=rest)
        else:
            spec = cls(axis)  # raises for K / unknown axes

        if spec.arm_id != arm_id:
            raise ValueError(f"Non-canonical arm_id '{arm_id}', expected '{spec.arm_id}'")
        return spec

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def is_anchor(self) -> bool:
        return self.arm_id == ANCHOR_ARM_ID

    @property
    def language(self) -> str:
        """LanguageType value (e.g. 'japanese') used by Dataset and prompt factories."""
        return LANG_CODE_TO_LANGUAGE[self.lang]

    @property
    def file_stem(self) -> str:
        return self.arm_id.replace(":", "_")

    def to_dataset_config(self, dataset_type: str, nums: int) -> DatasetConfig:
        """Question text of this arm: translated (L), rewritten (W) or the original English question."""
        return DatasetConfig.from_dict({
            "datasetType": dataset_type,
            "nums": nums,
            "sample": 1,
            "language": self.language,
            "useRewrite": self.questionSource == "rewrite",
        })

    def to_dict(self) -> dict:
        return {"arm_id": self.arm_id, **asdict(self)}

    @classmethod
    def from_dict(cls, data_dict: dict) -> "ArmSpec":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data_dict.items() if k in valid_keys})
