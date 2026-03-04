from enum import Enum

# Define available testing strategies for the framework
class StrategyType(str, Enum):
    ONELANGUAGE = "onelanguage"
    CHALLENGE = "challenge"
    SELFREFLECTION = "selfreflection"
    GETONEOUTPUT = 'getoneresult'
    REPAIR = 'repair'
    TRANSLATE = 'translate'

class StrategyDisplayNameType(str, Enum):
    ONELANGUAGE = "One Language"
    SELFREFLECTION = "Self Reflection"
    CHALLENGE = "Challenge"
    GETONEOUTPUT = "Get One Output"
    REPAIR = "Repair"
    TRANSLATE = 'Translate'

# Trailing commas turn the assigned value into a Tuple, breaking string comparisons.
class LanguageType(str, Enum):
    CHINESE = 'chinese'
    ENGLISH = 'english'
    SPANISH = 'spanish'
    JAPANESE = 'japanese'
    RUSSIAN = 'russian'

# Extract pure string values for quick validation
STRATEGY_STR_LIST = [s.value for s in StrategyType]

# Extract pure string values for quick validation
LANGUAGE_STR_LIST = [s.value for s in LanguageType]


# Mapping dictionary for display names. Maps StrategyType Enum to StrategyDisplayNameType Enum.
STRATEGY_TO_DISPLAYNAME = {
    member: StrategyDisplayNameType[member.name] for member in StrategyType
}

def get_strategy_map():
    """
    Returns a mapping of StrategyTypes to their respective concrete classes.
    Uses lazy importing to prevent circular dependency issues during initialization.
    """
    from Strategy.Translate import Translate
    from Strategy.OnlyOneLanguage import OnlyOneLanguage
    from Strategy.SelfReflection import SelfReflection
    from Strategy.Challenge import Challenge

    return {
        StrategyType.TRANSLATE: Translate,
        StrategyType.ONELANGUAGE: OnlyOneLanguage,
        StrategyType.SELFREFLECTION: SelfReflection,
        StrategyType.CHALLENGE: Challenge,
    }