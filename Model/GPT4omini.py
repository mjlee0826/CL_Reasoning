from openai import OpenAI
from Model.Model import Model
from Model.ModelConfig import ModelConfig
import tiktoken
import os

class GPT4omini(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    def getRes(self, prompt) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in GPT 4o model: {e}"

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
            return f"Error in GPT 4o model: {e}"
        
    def getTokenLens(self, text: str):
        return len(self.enc.encode(text))