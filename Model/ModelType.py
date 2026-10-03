from enum import Enum

# Define the core model identifiers used internally across the system
class ModelType(str, Enum):
    GPT41MINI = 'gpt4.1mini'
    GPT4OMINI = 'gpt4omini'
    DEEPSEEK = 'deepseek'
    GEMINI = 'gemini'
    GEMMA = 'gemma'
    QWEN = 'qwen'
    DEEPSEEK41FLASH = 'deepseek4.1flash'
    GEMINI31FLASHLITE = 'gemini3.1flashlite'

# Define the user-facing display names for each model
class ModelDisplayNameType(str, Enum):
    GPT41MINI = 'GPT4.1 mini'
    GPT4OMINI = 'GPT4o mini'
    DEEPSEEK = 'Deepseek'
    GEMINI = 'Gemini'
    GEMMA = 'Gemma'
    QWEN = 'QWEN'
    DEEPSEEK41FLASH = 'DeepSeek V4.1 Flash'
    GEMINI31FLASHLITE = 'Gemini 3.1 Flash-Lite'
    
# Define the default API model IDs required by the respective LLM providers
class DefaultModelName(str, Enum):
    GPT41MINI = 'gpt-4.1-mini-2025-04-14'
    GPT4OMINI = 'gpt-4o-mini-2024-07-18'
    DEEPSEEK = 'deepseek-ai/DeepSeek-V3.2'  # served by GMI Cloud, see Model/Deepseek.py
    GEMINI = 'gemini-2.5-flash-lite'
    GEMMA = 'models/gemma-3-27b-it'
    QWEN = 'qwen3-8b'
    DEEPSEEK41FLASH = 'deepseek-flash'  # DeepSeek's official alias, = DeepSeek-V4.1-Flash (see Model/DeepseekFlash.py)
    GEMINI31FLASHLITE = 'gemini-3.1-flash-lite'

# Mapping dictionary: ModelType -> ModelDisplayNameType
# Dynamically matches members by their variable names (e.g., DEEPSEEK -> DEEPSEEK)
MODEL_TO_DISPLAYNAME = {
    member: ModelDisplayNameType[member.name] for member in ModelType
}

# Mapping dictionary: ModelType -> DefaultModelName
# Dynamically matches members by their variable names
MODEL_TO_DEFAULT_MODELNAME = {
    member: DefaultModelName[member.name] for member in ModelType
}

# A simple list containing the string values of all supported ModelTypes
# Useful for validation, e.g., checking if a user input model name is supported
MODEL_STR_LIST = [m.value for m in ModelType]

def get_model_map():
    """
    Returns a dictionary mapping model string identifiers to their respective class definitions.
    Uses lazy importing (importing inside the function) to prevent circular import errors
    during the application initialization phase.
    """
    # Import model classes only when this function is actually called (Lazy Import)
    from Model.GPT41mini import GPT41mini
    from Model.GPT4omini import GPT4omini
    from Model.Deepseek import Deepseek
    from Model.Gemini import Gemini
    from Model.QWEN import QWEN
    from Model.Gemma import Gemma
    from Model.DeepseekFlash import DeepseekFlash
    from Model.Gemini31FlashLite import Gemini31FlashLite
    
    # Return the mapping dictionary
    return {
        ModelType.GPT41MINI: GPT41mini,
        ModelType.GPT4OMINI: GPT4omini,
        ModelType.DEEPSEEK: Deepseek,
        ModelType.GEMINI: Gemini,
        ModelType.GEMMA: Gemma,
        ModelType.QWEN: QWEN,
        ModelType.DEEPSEEK41FLASH: DeepseekFlash,
        ModelType.GEMINI31FLASHLITE: Gemini31FlashLite,
    }