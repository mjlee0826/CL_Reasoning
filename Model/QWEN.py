from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError
from Model.Model import Model
from Model.ModelConfig import ModelConfig
from transformers import AutoTokenizer
import os
import time

class QWEN(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B",
            trust_remote_code=True
        )
        
        # 指數退避設定
        self.max_retries = 5     # 最大重試次數
        self.base_delay = 5      # 基礎延遲秒數 (5, 10, 20, 40, 80)
    
    def _execute_with_retry(self, api_call_func):
        """
        封裝了指數退避 (Exponential Backoff) 的核心邏輯，用來處理 RPM / TPM 限制。
        """
        for attempt in range(self.max_retries):
            try:
                # 嘗試執行 API 呼叫
                return api_call_func()
                
            except RateLimitError as e:
                # 遇到 RPM/TPM 限制
                if attempt == self.max_retries - 1:
                    print(f"❌ [QWEN] Rate Limit 最終重試失敗: {e}")
                    return f"Error Code: RateLimitError - {e}"
                
                delay = self.base_delay * (2 ** attempt)
                print(f"⚠️ [QWEN] 觸發 Rate Limit (TPM/RPM)，執行緒暫停 {delay} 秒後進行第 {attempt + 1} 次重試...")
                time.sleep(delay)
                
            except (APIConnectionError, InternalServerError) as e:
                # 遇到伺服器暫時性不穩或連線錯誤，也可以用退避機制處理
                if attempt == self.max_retries - 1:
                    print(f"❌ [QWEN] 伺服器連線最終失敗: {e}")
                    return f"Error Code: ServerError - {e}"
                
                delay = self.base_delay * (2 ** attempt)
                print(f"⚠️ [QWEN] 伺服器不穩，執行緒暫停 {delay} 秒後進行第 {attempt + 1} 次重試...")
                time.sleep(delay)
                
            except Exception as e:
                # 其他未預期的錯誤 (例如驗證失敗)，直接回傳 Error Code 讓 Repair 腳本未來處理
                print(f"❌ [QWEN] 發生未預期的錯誤: {e}")
                return f"Error Code: Unexpected Error - {e}"

    def getRes(self, prompt) -> str:
        def api_call():
            response = self.client.chat.completions.create(
                model=self.modelName,
                extra_body={"enable_thinking": False},
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=self.temperature
            )
            return response.choices[0].message.content
            
        return self._execute_with_retry(api_call)

    def getListRes(self, promptList):
        def api_call():
            response = self.client.chat.completions.create(
                model=self.modelName,
                extra_body={"enable_thinking": False},
                messages=promptList,
                max_tokens=8192,
                temperature=self.temperature
            )
            return response.choices[0].message.content
            
        return self._execute_with_retry(api_call)
        
    def getTokenLens(self, text: str):
        return len(self.tokenizer.encode(text))