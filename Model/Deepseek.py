from Model.Model import Model
from openai import OpenAI
from transformers import AutoTokenizer
from Model.ModelConfig import ModelConfig
import os

class Deepseek(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V3",
            trust_remote_code=True
        )

    def getRes(self, prompt) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=self.temperature,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in Deepseek model: {e}"

    def getListRes(self, promptList):
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=promptList,
                max_tokens=8192,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in DeepSeek model: {e}"
        
    def getTokenLens(self, text: str):
        return len(self.tokenizer.encode(text))