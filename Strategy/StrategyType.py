from enum import Enum

# Define available testing strategies for the framework
class StrategyType(str, Enum):
    ONELANGUAGE = "onelanguage"
    CHALLENGE = "challenge"
    SELFREFLECTION = "selfreflection"
    GETONEOUTPUT = 'getoneresult'
    REPAIRONELANGUAGE = 'repaironelanguage'
    REPAIRCHALLENGE = 'repairchallenge'
    TRANSLATE = 'translate'
    REWRITE = 'rewrite'
    GENERATE = 'generate'
    AGGREGATE = 'aggregate'

class StrategyDisplayNameType(str, Enum):
    ONELANGUAGE = "One Language"
    SELFREFLECTION = "Self Reflection"
    CHALLENGE = "Challenge"
    GETONEOUTPUT = "Get One Output"
    REPAIRONELANGUAGE = "Repair One Language"
    REPAIRCHALLENGE = "Repair Challenge"
    TRANSLATE = 'Translate'
    REWRITE = 'Rewrite'
    GENERATE = 'Generate'
    AGGREGATE = 'Aggregate'

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
