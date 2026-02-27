from enum import Enum

class ModelType(str, Enum):
    GPT41MINI = 'gpt4.1mini'
    GPT4OMINI = 'gpt4omini'
    DEEPSEEK = 'deepseek'
    GEMINI = 'gemini'
    GEMMA = 'gemma'
    QWEN = 'qwen'

class ModelNameType(str, Enum):
    GPT41MINI = 'GPT4.1 mini'
    GPT4OMINI = 'GPT4o mini'
    DEEPSEEK = 'Deepseek'
    GEMINI = 'Gemini'
    GEMMA = 'Gemma'
    QWEN = 'QWEN'
    
MODEL_TO_NAME = {
    member: ModelNameType[member.name] for member in ModelType
}

MODEL_LIST = [m.value for m in ModelType]

def get_model_map():
    # ← 只有真正用到時才 import，不會循環
    from Model.GPT41mini import GPT41mini
    from Model.GPT4omini import GPT4omini
    from Model.Deepseek import Deepseek
    from Model.Gemini import Gemini
    from Model.QWEN import QWEN
    from Model.Gemma import Gemma
    return {
        ModelType.GPT41MINI.value: GPT41mini,
        ModelType.GPT4OMINI.value: GPT4omini,
        ModelType.DEEPSEEK.value: Deepseek,
        ModelType.GEMINI.value: Gemini,
        ModelType.GEMMA.value: Gemma,
        ModelType.QWEN.value: QWEN
    }
