from openai import OpenAI
from Model.Model import Model
from Model.ModelType import ModelType, MODEL_TO_NAME
from google.generativeai import GenerativeModel
import os

class Gemini(Model):
    NAME = MODEL_TO_NAME[ModelType.GEMINI]

    def __init__(self, temperture, modelName):
        if modelName == None:
            modelName = "gemini-2.5-flash-lite"
            print(f"Log: 'modelName' default to {modelName}.")

        super().__init__(temperture, modelName)
        self.name: str = Gemini.NAME

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
                temperature=self.temperture,
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
                temperature=self.temperture,
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