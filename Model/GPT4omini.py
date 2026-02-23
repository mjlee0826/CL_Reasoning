from openai import OpenAI
from Model.Model import Model
from Model.ModelType import ModelType, MODEL_TO_NAME
import tiktoken
import os

class GPT4omini(Model):
    NAME = MODEL_TO_NAME[ModelType.GPT4OMINI]
    
    def __init__(self, tempature, modelName):
        if modelName == None:
            modelName = "gpt-4o-mini-2024-07-18"
            print(f"Log: 'modelName' default to {modelName}.")

        super().__init__(tempature, modelName)
        self.name: str = GPT4omini.NAME
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    def getRes(self, prompt) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=self.tempature
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
                temperature=self.tempature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in GPT 4o model: {e}"
        
    def getTokenLens(self, text: str):
        return len(self.enc.encode(text))