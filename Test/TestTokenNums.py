from File.File import File
from Log.Log import Log
from Model.ModelType import ModelType
from Model.ModelFactory import ModelFactory
from Strategy.StrategyType import get_strategy_map
from Test.Test import Test

class TestTokenNums(Test):
    """
    計算每個結果檔「每題平均 output token 數」並寫回 metadata。
    只算模型生成的文字，且包含 baseline 那次呼叫的輸出（各策略的定義見 Strategy.getTokenLens）。
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
        strategy_type = strategy_config.strategyType
        strategy_cls = get_strategy_map()[strategy_type]

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
            cnt += strategy_cls.getTokenLens(model, record)

        log.logMessage(f'{self.METADATA_KEY}: {cnt / total}')

        file.metadata.pop(self.OLD_METADATA_KEY, None)
        file.updateMetadata(self.METADATA_KEY, cnt / total)
        file.save()
