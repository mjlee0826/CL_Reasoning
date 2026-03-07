import os
import time
import openai
from openai import OpenAI
from Model.Model import Model
from Model.ModelConfig import ModelConfig
from google.generativeai import GenerativeModel

class Gemini(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.client = OpenAI(
            api_key=os.getenv('GEMINI_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def _generate_with_retry(self, messages, max_retries=6):
        """內部方法：處理 API 呼叫並包含自動重試邏輯"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.modelName,
                    messages=messages,
                    max_tokens=4096,
                    temperature=self.temperature,
                    stream=False
                )
                if response.choices[0].message.content:
                    return response.choices[0].message.content
                return ""
            
            except openai.RateLimitError as e:
                # 遇到 429 (Rate Limit) 時觸發
                if attempt == max_retries - 1:
                    raise e  # 達到最大重試次數，將錯誤拋出給外層的 except Exception
                
                # 指數退避計算：等待 4, 5, 7, 11, 19... 秒，上限 60 秒
                wait_time = min(2 ** attempt * 5, 180)
                print(f"[Thread Wait] 觸發 429 限制，等待 {wait_time} 秒後進行第 {attempt + 1} 次重試...")
                time.sleep(wait_time)
                
            except openai.InternalServerError as e:
                # 遇到 500 等伺服器端錯誤也稍微等一下重試
                if attempt == max_retries - 1:
                    raise e
                time.sleep(5)

    def getRes(self, prompt) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            return self._generate_with_retry(messages)
        except Exception as e:
            # 如果重試到底還是失敗，或是發生其他不可預期的錯誤，才會走到這裡
            return f"Error in Gemini model: {e}"
    
    def getListRes(self, promptList):
        try:
            return self._generate_with_retry(promptList)
        except Exception as e:
            return f"Error in Gemini model: {e}"
    
    def getTokenLens(self, text: str):
        gm = GenerativeModel(self.modelName)
        ret = gm.count_tokens(text)
        # Gemini 無法取 token ID，因此回傳等長 list
        return ret.total_tokens