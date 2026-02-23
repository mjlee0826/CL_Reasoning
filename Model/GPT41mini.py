from openai import OpenAI
from Model.Model import Model
from Model.ModelType import ModelType, MODEL_TO_NAME
import tiktoken
import os

class GPT41mini(Model):
    NAME = MODEL_TO_NAME[ModelType.GPT41MINI]

    def __init__(self, temperture, modelName):
        if modelName == None:
            modelName = "gpt-4.1-mini-2025-04-14"
            print(f"Log: 'modelName' default to {modelName}.")

        super().__init__(temperture, modelName)
        self.name: str = GPT41mini.NAME
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    def getRes(self, prompt) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=self.temperture
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in GPT 4.1 mini model: {e}"
    
    def getListRes(self, promptList):
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=promptList,
                max_tokens=8192,
                temperature=self.temperture
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in GPT 4.1 mini model: {e}"
        
    def getTokenLens(self, text: str):
        return len(self.enc.encode(text))