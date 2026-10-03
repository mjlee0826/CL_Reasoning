from File.File import File
from Log.Log import Log
from Model.Model import Model
from Model.ModelType import ModelType
from Model.ModelFactory import ModelFactory
from Strategy.StrategyType import StrategyType
from Test.Test import Test


# ------------------------------------------------------------------
# 每題 output token 的定義：只算模型生成的文字，且包含 baseline 那次呼叫的輸出。
# legacy 策略類別（OnlyOneLanguage / Challenge / SelfReflection / Translate）已刪除，它們的定義原樣保留在這裡。
# ------------------------------------------------------------------
def baselineTokens(model: Model, record: dict) -> int:
    """onelanguage：baseline 的輸出 Result（題目 / prompt 是 input token，不算）。"""
    return model.getTokenLens(record.get("Result", ""))


def challengeTokens(model: Model, record: dict) -> int:
    """
    challenge：Record1 / Record2 中的 assistant 訊息（index 1 是 baseline 輸出，之後是辯論回合）加上 judge 輸出 Result3。
    user 訊息（題目、辯論 prompt）與 judge prompt 是 input token，不算。
    """
    tokens = sum(model.getTokenLens(r.get("content", ""))
                 for key in ("Record1", "Record2") for r in record.get(key, []) if r.get("role") == "assistant")
    if record.get("Result3"):
        tokens += model.getTokenLens(record["Result3"])
    return tokens


def selfReflectionTokens(model: Model, record: dict) -> int:
    """selfreflection：baseline 輸出 Response 加上反思輸出 Result（反思 prompt 是 input token，不算）。"""
    return model.getTokenLens(record.get("Response", "")) + model.getTokenLens(record.get("Result", ""))


def translateTokens(model: Model, record: dict) -> int:
    return model.getTokenLens(record.get("Translated", ""))


def rewriteTokens(model: Model, record: dict) -> int:
    return model.getTokenLens(record.get("Rewritten", ""))


def storedTokens(model: Model, record: dict) -> int:
    """generate / aggregate：紀錄裡已存了重算過的 tokens_out（聚合檔不含生成成本，那在 arm 檔）。"""
    return record.get("tokens_out") or 0


TOKEN_COUNTERS = {
    StrategyType.ONELANGUAGE: baselineTokens,
    StrategyType.REPAIRONELANGUAGE: baselineTokens,
    StrategyType.CHALLENGE: challengeTokens,
    StrategyType.REPAIRCHALLENGE: challengeTokens,
    StrategyType.SELFREFLECTION: selfReflectionTokens,
    StrategyType.TRANSLATE: translateTokens,
    StrategyType.REWRITE: rewriteTokens,
    StrategyType.GENERATE: storedTokens,
    StrategyType.AGGREGATE: storedTokens,
}


class TestTokenNums(Test):
    """
    計算每個結果檔「每題平均 output token 數」並寫回 metadata。
    只算模型生成的文字，且包含 baseline 那次呼叫的輸出（各策略的定義見 TOKEN_COUNTERS）。
    """
    METADATA_KEY = "Average Output Tokens"
    OLD_METADATA_KEY = "Average Token Nums"  # 舊定義（input + output 混算），寫入新值時一併刪除

    def __init__(self):
        super().__init__()
        self.name: str = "Test Token Nums"

    def runTest(self, fileList: list[File], log: Log):
        for file in fileList:
            # 單一檔案失敗不中斷整批；用 print 而非 log，NoLog 時也看得到哪些檔被跳過
            try:
                self.runFile(file, log)
            except Exception as e:
                print(f"❌ [TestTokenNums] 跳過 {file.file_path}: {type(e).__name__}: {e}")

    def runFile(self, file: File, log: Log):
        log.logInfo(file)

        # 適配最新的 File.py，取得 config
        model_config = file.getModelConfig()
        strategy_config = file.getStrategyConfig()
        countTokens = TOKEN_COUNTERS[strategy_config.strategyType]

        # 1. 實例化 Model (為了呼叫 model.getTokenLens 取得精準的 tokenizer 計算)
        model = ModelFactory().buildModel(ModelType(model_config.modelType), model_config)

        # 2. 從 records_map 取得所有資料
        data = list(file.records_map.values())
        total = len(data)

        if total == 0:
            print(f"⚠️ [TestTokenNums] 檔案無資料 (0 records): {file.file_path}")
            return

        cnt = 0

        for record in data:
            cnt += countTokens(model, record)

        log.logMessage(f'{self.METADATA_KEY}: {cnt / total}')

        file.metadata.pop(self.OLD_METADATA_KEY, None)
        file.updateMetadata(self.METADATA_KEY, cnt / total)
        file.save()
