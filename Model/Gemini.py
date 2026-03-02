from openai import OpenAI
from Model.Model import Model
from Model.ModelConfig import ModelConfig
from google.generativeai import GenerativeModel
import os

class Gemini(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.client = OpenAI(
            api_key=os.getenv('GEMINI_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
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
            if response.choices[0].message.content:
                return response.choices[0].message.content
            return ""
        except Exception as e:
            return f"Error in Gemini model: {e}"
    
    def getListRes(self, promptList):
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=promptList,
                max_tokens=8192,
                temperature=self.temperature,
                stream=False
            )

            if response.choices[0].message.content:
                return response.choices[0].message.content
            return ""
        except Exception as e:
            return f"Error in Gemini model: {e}"
    
    def getTokenLens(self, text: str):
        gm = GenerativeModel(self.modelName)
        ret = gm.count_tokens(text)
        # Gemini 無法取 token ID，因此回傳等長 list
        return ret.total_tokens