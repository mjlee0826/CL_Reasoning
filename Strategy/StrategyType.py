from enum import Enum

class StrategyType(str, Enum):
    ONLYCHINESE = "onlyChinese"
    ONLYENGLISH = "onlyEnglish"
    ONLYSPANISH = "onlySpanish"
    ONLYJAPANESE = "onlyJapanese"
    ONLYRUSSIAN = "onlyRussian"
    CHALLENGE = "challenge"
    SELFREFLECTION = "selfreflection"
    GETONEOUTPUT = 'getoneresult'
    REPAIR = 'repair'

class StrategyNameType(str, Enum):
    ONLYCHINESE = "Only Chinese"
    ONLYENGLISH = "Only English"
    ONLYSPANISH = "Only Spanish"
    ONLYJAPANESE = "Only Japanese"
    ONLYRUSSIAN = "Only Russian"
    SELFREFLECTION = "Self Reflection"
    CHALLENGE = "Challenge"
    GETONEOUTPUT = "Get One Output"
    REPAIR = "Repair"

class LanguageType(str, Enum):
    CHINESE = 'Chinese',
    ENGLISH = 'English',
    SPANISH = 'Spanish',
    JAPANESE = 'Japanese',
    RUSSIAN = 'Russian'

# 直接取出 value，會是字串
STRATEGY_LIST = [s.value for s in StrategyType]

# 用字串當 key，比較方便查
STRATEGY_TO_NAME = {
    member: StrategyNameType[member.name] for member in StrategyType
}

NAME_TO_STRATEGY = {
    member: StrategyType[member.name] for member in StrategyNameType
}

STRATEGY_TO_LANGUAGE = {
    StrategyType.ONLYCHINESE.value: LanguageType.CHINESE.value,
    StrategyType.ONLYENGLISH.value: LanguageType.ENGLISH.value,
    StrategyType.ONLYSPANISH.value: LanguageType.SPANISH.value,
    StrategyType.ONLYJAPANESE.value: LanguageType.JAPANESE.value,
    StrategyType.ONLYRUSSIAN.value: LanguageType.RUSSIAN.value,
}

def get_strategy_map():
    # ← 只有真正用到時才 import，不會循環
    from Strategy.OnlyOneLanguage import OnlyOneLanguage
    from Strategy.SelfReflection import SelfReflection
    from Strategy.Challenge import Challenge

    return {
        StrategyNameType.ONLYCHINESE.value: OnlyOneLanguage,
        StrategyNameType.ONLYENGLISH.value: OnlyOneLanguage,
        StrategyNameType.ONLYSPANISH.value: OnlyOneLanguage,
        StrategyNameType.ONLYJAPANESE.value: OnlyOneLanguage,
        StrategyNameType.ONLYRUSSIAN.value: OnlyOneLanguage,
        StrategyNameType.SELFREFLECTION.value: SelfReflection,
        StrategyNameType.CHALLENGE.value: Challenge,
    }