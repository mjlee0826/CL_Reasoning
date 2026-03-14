from File.File import File
from Log.Log import Log
from Model.ModelType import ModelType
from Model.ModelFactory import ModelFactory
from Test.Test import Test

class TestTokenNums(Test):
    def __init__(self):
        super().__init__()
        self.name: str = "Test Token Nums"

    def runTest(self, fileList: list[File], log: Log):
        for file in fileList:
            log.logInfo(file)
            
            # 適配最新的 File.py，取得 config
            model_config = file.getModelConfig()
            strategy_config = file.getStrategyConfig()
            strategy_type = strategy_config.strategyType
            
            # 1. 實例化 Model (為了呼叫 model.getTokenLens 取得精準的 tokenizer 計算)
            try:
                model = ModelFactory().buildModel(ModelType(model_config.modelType), model_config)
            except Exception as e:
                log.logMessage(f"❌ 無法建立 Model: {e}")
                continue

            # 2. 從 records_map 取得所有資料
            data = list(file.records_map.values())
            total = len(data)
            
            if total == 0:
                log.logMessage("⚠️ 檔案無資料 (0 records)")
                continue

            cnt = 0
            cnt_challenge = 0
            total_challenge = 0

            for record in data:
                text_to_count = ""
                
                # 3. 根據策略類型，提取對應的完整文字來計算 Token
                if strategy_type and "challenge" in str(strategy_type).lower():
                    # Challenge 模式：包含雙方辯論紀錄 (Record1, Record2) 以及裁判結果 (Result3)
                    # 紀錄通常為 list of dict: [{"role": "user", "content": "..."}, ...]
                    for r in record.get("Record1", []):
                        text_to_count += str(r.get("content", "")) + "\n"
                    for r in record.get("Record2", []):
                        text_to_count += str(r.get("content", "")) + "\n"
                        
                    # 加入裁判的輸出
                    text_to_count += str(record.get("Result3", "")) + "\n"
                    
                    # 計算 Token
                    lens = model.getTokenLens(text_to_count)
                    cnt += lens
                    
                    # 如果實際發生了辯論 (Times > 0)，特別獨立統計這部分的平均消耗
                    if record.get("Times", 0) > 0:
                        cnt_challenge += lens
                        total_challenge += 1
                else:
                    # Baseline 模式：僅有單次 Question 與 Result
                    text_to_count += str(record.get("Question", "")) + "\n"
                    text_to_count += str(record.get("Result", "")) + "\n"
                    
                    lens = model.getTokenLens(text_to_count)
                    cnt += lens

            # 4. 輸出統計結果
            if strategy_type and "challenge" in str(strategy_type).lower():
                log.logMessage(f'Average Token (All Data): {cnt / total:.2f}')
                
                # 防止除以 0 的防呆機制
                if total_challenge > 0:
                    log.logMessage(f'Average Token (Only Debate): {cnt_challenge / total_challenge:.2f}')
                else:
                    log.logMessage('Average Token (Only Debate): N/A (無辯論發生)')
            else:
                log.logMessage(f'Average Token: {cnt / total:.2f}')
            
            file.updateMetadata("Average Token (All Data)", cnt / total)
            file.updateMetadata("Average Token (Only Debate)", cnt_challenge / total_challenge)

            file.save()