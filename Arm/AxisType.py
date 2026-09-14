from enum import Enum
from Strategy.StrategyType import LanguageType

# Diversity axes of paper_status v7 §3.3 (arm_id prefixes)
class AxisType(str, Enum):
    LANGUAGE = "L"
    SAMPLING = "S"
    REASONING = "R"
    PERSONA = "P"
    REWRITE = "W"
    CANDIDATES = "K"

AXIS_STR_LIST = [a.value for a in AxisType]

# Short language codes used in arm_id / records <-> LanguageType values used by Dataset and prompt factories
LANG_CODE_TO_LANGUAGE = {
    "en": LanguageType.ENGLISH.value,
    "zh": LanguageType.CHINESE.value,
    "ja": LanguageType.JAPANESE.value,
    "ru": LanguageType.RUSSIAN.value,
    "es": LanguageType.SPANISH.value,
}
LANGUAGE_TO_LANG_CODE = {language: code for code, language in LANG_CODE_TO_LANGUAGE.items()}

PROMPT_STYLE_LIST = ["cot", "short_cot", "direct"]
QUESTION_SOURCE_LIST = ["original", "rewrite"]

# Agent A: original English question + full CoT + T=0, shared by every axis
ANCHOR_ARM_ID = "L:en"
